import abc
from collections import defaultdict
from datetime import timedelta
import multiprocessing
from operator import itemgetter
import os
import sys
import time

# Cooldown, in seconds, between using abilities (spells or combat arts).
ABILITY_RECOVERY = 0.3

POISON_PILL = -1

class CastHistory(object):
    """ Represents a time sequence of spell casts against a specified number of
    targets with a given encounter duration.

    Targets are assumed to be linked, i.e. affected by green as well as blue
    aoes, and in range of all spells. Targets are assumed to have infinite
    hitpoints.
    """

    def __init__(self, spells, num_targets, time_limit):
        """ Initialize a casting history with no spells cast yet.

        Arguments:
        - `spells`: a list of all spells that can be cast
        - `num_targets`: linked encounter size
        - `time_limit`: encounter duration
        """
        self._spells = spells
        self.num_targets = num_targets
        self.time_limit = time_limit
        # The casting history is a list of SpellCast objects, which are (spell,
        # time) pairs.
        self._casts = []
        # _total_damage is the damage that will be done by this cast history
        # (without any additional casts) over the entire encounter time limit.
        # It is updated every time the cast method is called.
        self._total_damage = None
        # _upper_bound is the memoized optimistic estimate of the total damage
        # that could be done given the casting history in _casts and the
        # remaining encounter time.
        self._upper_bound = None

    def __repr__(self):
        """ Return a string representation of the casts in this history. """
        delimited_casts = ''
        for cast in self._casts:
            if delimited_casts:
                delimited_casts += ','
            delimited_casts += cast.spell.shortname
        return '<CastHistory [{}]>'.format(delimited_casts)

    def _compute_total_damage(self):
        """ Set the memoized value of the total damage that will be done by this
        cast history over the entire encounter time limit.
        """
        self._total_damage = 0.0
        for cast in self._casts:
            # Ask the spell to figure out how much damage it contributed.
            self._total_damage += cast.spell.damage(self, cast)

    def _compute_upper_bound(self):
        """ Set the memoized value of the optimistic estimate of the total
        damage that will be done by this cast history over the entire encounter
        time limit.
        """
        damage_total = self.total_damage
        # For every fully castable spell, in a fresh copy of this cast
        # history, cast it as many times as possible and record the cast time,
        # damage, and dps for each cast.
        cast_db = defaultdict(list)
        for spell in self._spells:
            isinstance(spell, Spell)
            ch_copy = self.duplicate()
            while spell.fully_castable(ch_copy):
                ch_copy.cast(spell)
                ubc = UpperBoundCast(spell, spell.damage(ch_copy),
                                     spell.cast_time(ch_copy))
                cast_db[ubc.dps].append(ubc)

        # Sort the casts by dps. Sum the damage that can be done by selecting
        # the highest dps casts until there is insufficient time to cast
        # anything fully.
        remaining_cast_time = self.time_limit - self.last_ability_recovery_finished_time
        for dps, ubc_list in sorted(cast_db.items(), key=itemgetter(0), reverse=True):
            assert remaining_cast_time >= 0.0
            if remaining_cast_time == 0.0:
                break
            for ubc in ubc_list:
                if ubc.cast_time <= remaining_cast_time:
                    remaining_cast_time -= ubc.cast_time
                    damage_total += ubc.damage
                else:
                    # compute partial damage
                    damage_total += ubc.damage * remaining_cast_time / ubc.cast_time
                    remaining_cast_time = 0.0
                    break
        self._upper_bound = damage_total
        return damage_total

    def can_cast_something(self):
        """ Return true if any spell can be (fully) cast. """
        for spell in self._spells:
            if spell.fully_castable(self):
                return True
        return False

    def cast(self, spell):
        """ Cast the specified spell. """
        # Append a new spell, time pair to the casting history. Note that when
        # it will be cast depends on what has already been cast, and some spells
        # have special rules about interrupting previous casts of themselves.
        time_of_next_cast = spell.next_casting_time(self)
        assert time_of_next_cast < self.time_limit
        self._casts.append(SpellCast(spell, time_of_next_cast))
        # Any previous damage calculations no longer apply
        self._total_damage = None
        self._upper_bound = None
        return self._casts[-1]

    def castable_spells(self):
        """ Generator on spells which could be fully cast.
        """
        for spell in self._spells:
            if spell.fully_castable(self):
                yield spell

    def castable_spell_string(self, spell_cast):
        """Return a string representing the hotbar state a user would see,
        indicating which spells are castable at the time of `spell_cast`."""
        hotbar = ""
        for spell in self._spells:
            if hotbar != "":
                hotbar += " "
            if spell.cooling_down(self, spell_cast):
                hotbar += spell.shortname.lower()
            else:
                hotbar += spell.shortname.upper()
        return hotbar

    def duplicate(self):
        """ Return a copy of this object which can be modified without affecting
        this one.
        """
        copy = CastHistory(self._spells, self.num_targets, self.time_limit)
        copy._casts = self._casts[:]
        copy._total_damage = self._total_damage
        return copy

    def find_casts(self, spell):
        """Iterator over all casts of a given spell in this history."""
        for sc in self._casts:
            if sc.spell == spell:
                yield sc

    def from_repr(self, repr_str):
        """Initialize this cast history from a string representation. """
        assert self._casts == []
        assert self._total_damage == None
        assert self._upper_bound == None
        # from <CastHistory [AD,AB,CB,E,IS,P,TD,FA,SE,VB,WV]> get
        # [AD,AB,CB,E,IS,P,TD,FA,SE,VB,WV]>
        s = repr_str.split()[1]
        # discard leading '[' and trailing ']>'
        s = s[1:-2]
        # split by commas
        spells_by_shortname = s.split(',')
        # empty cast history yields ['']
        if spells_by_shortname != ['']:
            for shortname in spells_by_shortname:
                # find the spell with this shortname. Note shortnames must be
                # unique.
                for spell in self._spells:
                    isinstance(spell, Spell)
                    if spell.shortname == shortname:
                        assert spell.fully_castable(self)
                        self.cast(spell)
                        break
        return self

    def interrupts_cast(self, spell_cast):
        """Return the SpellCast representing the cast prior to the one in
        spell_cast which is interrupted by the casting of spell_cast, or None
        if no such interruption occurs."""
        assert isinstance(spell_cast, SpellCast)
        # only dot spells can interrupt themselves
        if not isinstance(spell_cast.spell, DotSpell):
            return None
        prev_cast = self.prev_cast(spell_cast)
        # if this spell wasn't cast prior to the given instance, it couldn't
        # have interrupted itself
        if prev_cast is None:
            return None
        dot = spell_cast.spell
        assert dot == prev_cast.spell
        if (spell_cast.time_of_cast < prev_cast.time_of_cast +
            dot.cast_time(self, prev_cast) + dot.duration):
            return prev_cast
        return None

    @property
    def last_ability_recovery_finished_time(self):
        """ Return the time at which the most recently cast spell's casting time
        plus ability recovery time has passed. In other words, the time at which
        the next ability (spell cast) may be used.
        """
        last = self.last_cast()
        if not last:
            return 0.0
        return (last.time_of_cast + last.spell.cast_time(self, last) +
                ABILITY_RECOVERY)

    def last_cast(self, spell=None):
        """ Return the last cast of `spell` (or the last cast of any spell if
        `spell` is None) or None if the cast history is empty.
        """
        if not self._casts:
            return None
        if not spell:
            return self._casts[-1]
        for spell_cast in reversed(self._casts):
            if spell_cast.spell == spell:
                return spell_cast
        return None

    def next_cast(self, spell_cast):
        """ Return the first SpellCast object in the cast history which follows
        `spell_cast` and is for the same spell, or None if no such cast has
        been made.
        """
        casts_after = self._casts[self._casts.index(spell_cast) + 1:]
        for sc in casts_after:
            if sc.spell == spell_cast.spell:
                return sc
        return None

    def prev_cast(self, spell_cast):
        """ Return the first SpellCast object in the cast history which precedes
        `spell_cast` and is for the same spell, or None if no such cast has been
        made.
        """
        preceding_casts = self._casts[:self._casts.index(spell_cast)]
        for sc in reversed(preceding_casts):
            if sc.spell == spell_cast.spell:
                return sc
        return None

    def print_history(self):
        """ Write the cast history to stdout. """
        for spell_cast in self._casts:
            interrupted = self.interrupts_cast(spell_cast)
            if interrupted:
                assert isinstance(interrupted, SpellCast)
                assert interrupted.spell == spell_cast.spell
                assert interrupted.time_of_cast < spell_cast.time_of_cast
                print '{:6.2f}: {} {} (interrupts cast at {:.2f})'.format(
                    spell_cast.time_of_cast,
                    self.castable_spell_string(spell_cast),
                    spell_cast.spell.name,
                    interrupted.time_of_cast)
            else:
                print '{:6.2f}: {} {}'.format(
                    spell_cast.time_of_cast,
                    self.castable_spell_string(spell_cast),
                    spell_cast.spell.name)

    def time_of_last_cast(self, spell):
        """ Return the most recent time that the given spell was cast, or None
        if it has never been cast.
        """
        last = self.last_cast(spell)
        if not last:
            return None
        return last.time_of_cast

    @property
    def total_damage(self):
        """ Return the total damage done over the encounter time by all the
        spells cast so far.
        """
        if self._total_damage is None:
            self._compute_total_damage()
        return self._total_damage

    @property
    def upper_bound(self):
        """ Return the optimistic estimate of the amount of damage that could be
        generated from this cast history forward to the end of the encounter.
        """
        if self._upper_bound is None:
            self._compute_upper_bound()
        return self._upper_bound


class Spell(object):
    __metaclass__ = abc.ABCMeta
    """ Describes a spell: its name, casting times, and quantity and type of
    damage.
    """
    def __init__(self, name, cast_time, cooldown, initial_damage,
                 max_targets=1):
        # name of the spell, displayed to user
        self.name = name
        self._shortname = ''.join(map(itemgetter(0), name.split()))

        # time, in seconds, to cast the spell
        self._cast_time = float(cast_time)
        assert(self._cast_time >= 0.0)

        # Cooldown, in seconds, before spell can be cast again. Note some
        # spells handle cooldown specially; those should override any method
        # that references cooldown.
        self.cooldown = float(cooldown)
        assert(self.cooldown >= 0.0)

        # damage done when spell cast_time completes
        self.initial_damage = float(initial_damage)
        assert(self.initial_damage > 0.0)

        # Maximum number of targets affected. For single-target spells,
        # max_targets == 1.
        self.max_targets = int(max_targets)
        assert(self.max_targets >= 1)

    def cast_time(self, cast_history, spell_cast=None):
        """Return the time it took to cast the instance of this spell specified by
        `spell_cast` within the given cast history.

        `spell_cast`: If None, return the casting time of this spell when it
        is the next one to be cast, otherwise return the time it took to cast
        the specified instance.
        """
        # See WindsOfVelious for a spell that needs to override this method.
        return self._cast_time

    def cooling_down(self, cast_history, spell_cast):
        """Return True if this spell was cooling down at the time in
        spell_cast."""
        # Handle pathological case of asking whether this spell was in
        # cooldown when it was being cast.
        if spell_cast.spell == self:
            return False
        # Find the last instance of the cast of this spell that preceded the
        # time of the cast in spell_cast.
        most_recent_preceding = None
        for cast_of_this_spell in cast_history.find_casts(self):
            if cast_of_this_spell.time_of_cast < spell_cast.time_of_cast:
                most_recent_preceding = cast_of_this_spell
            else:
                break
        # Return True if this spell was in cooldown at that time.
        if most_recent_preceding is None:
            return False
        cooldown_ended_at = (
            most_recent_preceding.time_of_cast +
            self.cast_time(cast_history, most_recent_preceding) +
            self.cooldown)
        return cooldown_ended_at > spell_cast.time_of_cast

    @abc.abstractmethod
    def damage(self, cast_history, spell_cast):
        """ Return the amount of damage done by the invocation of this spell,
        represented by `spell_cast`, in the given cast history.
        """
        return

    def fully_castable(self, cast_history):
        """ Return True if there is enough time left to complete the cast of
        this spell (this should return True for a dot even if there is no time
        for any damage ticks).
        """
        next_time_this_can_cast = self.next_casting_time(cast_history)
        return (next_time_this_can_cast + self.cast_time(cast_history) <=
                cast_history.time_limit)

    def next_casting_time(self, cast_history):
        """ Return the soonest time at which this spell may be cast in the given
        cast history.
        """
        # Ask the casting history when this spell was most recently cast
        tolc = cast_history.time_of_last_cast(self)

        # if it has never been cast, then it can be cast as soon as it is
        # possible to cast anything.
        if tolc is None:
            return cast_history.last_ability_recovery_finished_time

        # if it has previously been cast, then it can be cast as soon as its
        # cooldown is finished.
        return max(cast_history.last_ability_recovery_finished_time,
                   tolc + self._cast_time + self.cooldown)

    @property
    def shortname(self):
        return self._shortname



class InitialDamageSpell(Spell):
    """ Represents a spell that has no damage over time component, i.e., it does
    only initial damage.
    """

    def __init__(self, name, cast_time, cooldown,
                 initial_damage, max_targets = 1):
        super(InitialDamageSpell, self).__init__(
            name, cast_time, cooldown, initial_damage, max_targets)

    def damage(self, cast_history, spell_cast=None):
        assert isinstance(cast_history, CastHistory)
        if spell_cast is None:
            spell_cast = cast_history.last_cast(self)
        # If this instance had time to fully cast, then the damage it does is
        # all of its initial damage.
        available_cast_time = cast_history.time_limit - spell_cast.time_of_cast
        if available_cast_time >= self._cast_time:
            return (min(self.max_targets, cast_history.num_targets) *
                    self.initial_damage)
        return (min(self.max_targets, cast_history.num_targets) *
                self.initial_damage * available_cast_time / self._cast_time)


class DotSpell(Spell):
    """ Represents a spell that does initial damage plus damage over time.
    """
    def __init__(self, name, cast_time, cooldown,
                 initial_damage, period, period_damage,
                 duration, max_targets = 1):
        super(DotSpell, self).__init__(
            name, cast_time, cooldown, initial_damage, max_targets)
        # time between damage ticks.
        self.period = float(period)
        assert(self.period > 0.0)

        # Damage done for every elapsed period.
        self.period_damage = float(period_damage)
        assert(self.period_damage > 0.0)

        # Duration of the spell.
        self.duration = float(duration)
        assert(self.duration > 0.0)

    def damage(self, cast_history, spell_cast=None):
        if spell_cast is None:
            spell_cast = cast_history.last_cast(self)
        # Determine how long the dot can run; the limit is either the end of the
        # encounter or the next time that this spell is recast, thus
        # interrupting itself.
        next_cast_of_this_spell = cast_history.next_cast(spell_cast)
        if next_cast_of_this_spell is None:
            last_time_for_dot = cast_history.time_limit
        else:
            last_time_for_dot = next_cast_of_this_spell.time_of_cast
        targets_hit = min(cast_history.num_targets, self.max_targets)
        cast_time = self.cast_time(cast_history, spell_cast)
        if spell_cast.time_of_cast + cast_time > cast_history.time_limit:
            # being called by optimistic estimator: return partial damage.
            # Note the first tick of dot damage happens at the time of cast.
            if cast_time > self._cast_time:
                cooldown_remaining = cast_time - self._cast_time
                partial_cast_time = (
                    cast_history.time_limit -
                    (spell_cast.time_of_cast + cooldown_remaining))
                if partial_cast_time <= 0.0:
                    return 0.0
            else:
                partial_cast_time = (
                    cast_history.time_limit - spell_cast.time_of_cast)
            return ((self.initial_damage + self.period_damage) *
                    targets_hit *
                    partial_cast_time / self._cast_time)

        # There is time to fully cast the spell, although there may be less
        # than the dot's duration remaining.
        #
        # Figure out how many times the dot got to do its periodic damage. The
        # first tick of a dot occurs at the time of the cast, so there is
        # always at least one tick.
        remaining_time = (last_time_for_dot -
                          (spell_cast.time_of_cast + cast_time))
        assert remaining_time >= 0.0
        ticks = 1 + int(min(remaining_time, self.duration) / self.period)
        return (targets_hit *
                (self.initial_damage +
                 (self.period_damage * ticks)))


class WindsOfVeliousSpell(DotSpell):
    """ Represents the conjuror spell Winds of Velious, which is a damage over
    time spell with special behavior if recast while a previous incantation is
    still active.
    """
    def __init__(self, cast_time, cooldown,
                 initial_damage, period, period_damage,
                 duration):
        super(WindsOfVeliousSpell, self).__init__(
            "Winds of Velious", cast_time, cooldown,
                 initial_damage, period, period_damage,
                 duration)
        self._shortname = 'WV'

    def cast_time(self, cast_history, this_cast=None):
        """Return the time it took to cast the specified instance of this
        spell, or if None is specified, the time it will take to cast this as
        the next spell in the cast history."""
        assert isinstance(cast_history, CastHistory)
        assert this_cast is None or isinstance(this_cast, SpellCast)
        # If winds of velious is recast while a previous invocation is still
        # active, the previous invocation is toggled off, then the spell
        # cooldown elapses, then the new instance is cast.
        #
        # If the dot is allowed to run to completion, then the spell cooldown
        # elapses after the last dot tick.
        #
        # This behavior is different from normal dots, whose cooldown timer
        # begins as soon as the spell cast is complete. Consider 4 cases where
        # there is either some cast of WoV before the current one, after it,
        # both, or neither (the time indicated by ".." may contain 0 or more
        # casts of spells other than WoV):
        #
        # i.a.  0..cur_cast..end
        # i.b.  0..cur_cast..next_cast..end
        # ii.a. 0..prev_cast..cur_cast..end
        # ii.b. 0..prev_cast..cur_cast..next_cast..end
        #
        # For case i, the cast time is the normal spell cast time. For
        # case ii, the cast time includes the cooldown time if the
        # current cast is interrupting the previous cast. In all cases, then,
        # the cast time of any invocation of this spell is affected only by a
        # previous cast, not by a future cast.

        if this_cast is None:
            prev_wov_cast = cast_history.last_cast(self)
        else:
            prev_wov_cast = cast_history.prev_cast(this_cast)

        if prev_wov_cast is None:
            return self._cast_time

        if this_cast is None:
            time_of_this_cast =\
                cast_history.last_ability_recovery_finished_time
        else:
            time_of_this_cast = this_cast.time_of_cast

        # There was a previous cast of this spell; calculate the point in time
        # its cooldown would start if its dot ran to completion.
        prev_cast_nominal_cooldown_start = (
            prev_wov_cast.time_of_cast +
            self.cast_time(cast_history, prev_wov_cast) +
            self.duration)

        # If this cast started before the previous cast's cooldown timer could
        # begin, then this cast time includes all of the cooldown.
        if time_of_this_cast <= prev_cast_nominal_cooldown_start:
            return self._cast_time + self.cooldown

        # If this cast started after the previous cast's cooldown finished,
        # then this cast time doesn't include any of the cooldown.
        prev_cast_nominal_cooldown_end = (
            prev_cast_nominal_cooldown_start + self.cooldown)
        if time_of_this_cast >= prev_cast_nominal_cooldown_end:
            return self._cast_time

        # This cast occurs after the previous cast's cooldown has started, but
        # before it has ended.
        cooldown_remaining = (
            prev_cast_nominal_cooldown_end - time_of_this_cast)
        return self._cast_time + cooldown_remaining

    def next_casting_time(self, cast_history):
        # The cast_time of any instance of WoV will include any portion of the
        # cooldown remaining from the previous cast, so the next casting time
        # of WoV is actually simpler than other spells since it ignores its
        # own cooldown.
        return cast_history.last_ability_recovery_finished_time


class SpellCast(object):
    """ Represents a cast of a given spell at a specific point in time.
    """
    def __init__(self, spell, time_of_cast):
        self.spell = spell
        self.time_of_cast = time_of_cast


class UpperBoundCast(object):
    """Represent a spell cast during upper bound computation."""
    def __init__(self, spell, damage, cast_time):
        self.spell = spell
        self.cast_time = cast_time
        self.damage = damage
        self.dps = damage / cast_time


class Stack(object):
    """Multiprocess-safe joinable stack"""
    def __init__(self, mgr):
        self._stack = mgr.list()
        # semaphore value is stack depth, so it starts at 0
        self._sem = multiprocessing.Semaphore(0)
        # lock protects against multiple writers. readers synchronize using
        # semaphore.
        self._lock = multiprocessing.Lock()
        self._task_count = multiprocessing.Value('i')
        self._task_count.value = 0
        self._event = multiprocessing.Event()

    def __len__(self):
        return len(self._stack)

    def join(self, timeout=None):
        """Block until task count is 0 or the timeout expires. Return True
        unless the timeout is not None and expires, in which case return
        False."""
        return self._event.wait(timeout)

    def push(self, obj):
        """Push a new object onto the end of the stack"""
        # take the write lock and add an item to the end of the list
        with self._lock:
            if self._event.is_set():
                assert obj == POISON_PILL
            else:
                assert isinstance(obj, CastHistory)
            self._stack.append(obj)
            self._task_count.value += 1
        # let one waiting call on pop execute
        self._sem.release()

    def pop(self, block=True, timeout=None):
        """Return item at top of stack, or None if a) the stack is empty and
        `block` is False and timeout is `None`, or b) `block` is True and the
        stack remains empty for the duration of `timeout`.
        """
        # Note there is no way to tell the difference between a stack that has
        # a None object on it vs. an empty stack with a non-blocking or
        # blocking timeout. If you need to be able to tell, have pop return a
        # (status, value) tuple instead of a single return value.

        # wait until there is at least one item on the stack, or until the
        # timeout expires.
        top = None
        if self._sem.acquire(block, timeout):
            # protect writes to the list
            with self._lock:
                top = self._stack.pop()
        return top

    def task_done(self, count=1):
        with self._lock:
            self._task_count.value -= count
            assert self._task_count.value >= 0
            if self._task_count.value == 0:
                self._event.set()


REPORT_INTERVAL = 10.0

def reporter(global_stack, time_limit, ns, io_lock):
    """Periodically write snapshot of process states and results to stdout"""
    # note we are strictly reading from the namespace and don't require
    # consistency, so we don't take the namespace lock which would slow down
    # the workers.
    while True:
        # Wait for the duration of the update interval on the work-done event.
        # If the event is gets set before the timeout, exit.
        if global_stack.join(REPORT_INTERVAL):
            break
        # The event timed out, print a report.
        with io_lock:
            print 'leaves: {:,}'.format(sum(ns.leaf_counts.itervalues()))
            print 'global  ',
            if ns.stack_depths['global']:
                print '#' * ns.stack_depths['global']
            else:
                print 'empty'
            for pid, depth in ns.stack_depths.iteritems():
                if pid == 'global':
                    continue
                print 'pid{: 5}'.format(pid), '#' * depth
            print 'damage: {:,.0f} ({:,.0f} dps)'.format(ns.best_dmg, ns.best_dmg/time_limit)
            print ns.best_ch_repr
            print
            sys.stdout.flush()


MAX_LEAVES_BETWEEN_CHECKS = 1000
MAX_DONATION = 5
MAX_GLOBAL_STACK_WAIT = 1.0

def check_global(global_stack, local, ns, ns_lock, idle_workers_sem):
    with ns_lock:
        global_best_dmg = ns.best_dmg
        if local.dmg > global_best_dmg:
            ns.best_dmg = local.dmg
            ns.best_ch_repr = local.ch_repr
        elif local.dmg < global_best_dmg:
            local.dmg = global_best_dmg
            local.ch_repr = ns.best_ch_repr
        # If there are idle workers share some of our stack. Do this under
        # protection of the namespace lock to make sure multiple workers don't
        # donate to the global stack based on the same test of the idle
        # workers semaphore.
        if idle_workers_sem.get_value():
            num_to_donate = min(MAX_DONATION, len(local.stack)/2)
            if num_to_donate:
                for i in range(num_to_donate):
                    global_stack.push(local.stack.pop(0))
                idle_workers_sem.acquire()

    # Update the status dictionaries
    sd = ns.stack_depths.copy()
    sd['global'] = len(global_stack)
    sd[os.getpid()] = len(local.stack)
    lc = ns.leaf_counts.copy()
    lc[os.getpid()] = local.leaf_count
    ns.stack_depths = sd
    ns.leaf_counts = lc


class LocalStats(object):
    def __init__(self):
        self.dmg = 0.0
        self.ch_repr = ''
        self.leaf_count = 0
        self.stack = []

def worker(global_stack, ns_lock, ns, io_lock, idle_workers_sem):
    """Worker process invoked by find_best_cast_history."""
    ch = None
    local = LocalStats()
    with ns_lock:
        local.dmg = ns.best_dmg
        local.ch_repr = ns.best_ch_repr
    global_pops = 0
    leaves_since_last_check = 0
    # Expand nodes on the local stack until we've found a leaf that has a
    # better total damage than any we've seen before. When that happens, check
    # and possibly update the global best damage.
    while True:
        if ch is None:
            # We don't have a cast history to work with; pull one off the
            # local stack if available, otherwise hit the shared stack.
            if local.stack:
                ch = local.stack.pop()
            else:
                # The local stack is empty, declare all previously acquired
                # tasks to be complete.
                if global_pops:
                    global_stack.task_done(global_pops)
                    global_pops = 0
                # Try to get the next cast history to process from the global
                # stack.
                ch = global_stack.pop(True, MAX_GLOBAL_STACK_WAIT)
                if ch is None:
                    # We timed out waiting for something to appear on the
                    # global stack. Increment the semaphore of idle workers,
                    # so that some other worker can see that this one is
                    # waiting.
                    idle_workers_sem.release()
                    # Now that this worker has announced itself as idle, it
                    # can block on the global stack until another worker sees
                    # the nonzero semaphore count and puts something on the
                    # stack, or all the work gets done and a poison pill
                    # appears, whichever comes first.
                    ch = global_stack.pop()
                if ch == POISON_PILL:
                    # we got the 'poison pill', the signal to terminate.
                    with ns_lock:
                        ns.leaves += local.leaf_count
                        if local.dmg >= ns.best_dmg:
                            ns.best_dmg = local.dmg
                            ns.best_ch_repr = local.ch_repr
                    break
                # We got a node from the global stack to process. Keep a count
                # of items popped from the global stack, because we need to
                # match global stack pushes with task_done calls.
                global_pops += 1

        # If we've reached a leaf node, check if it's the best damage so far
        # before looping around to get another non-leaf node to process.
        if not ch.can_cast_something():
            local.leaf_count += 1
            if ch.total_damage > local.dmg:
                local.dmg = ch.total_damage
                local.ch_repr = repr(ch)
                check_global(global_stack, local, ns, ns_lock, idle_workers_sem)
                leaves_since_last_check = 0
            else:
                leaves_since_last_check += 1
                if leaves_since_last_check >= MAX_LEAVES_BETWEEN_CHECKS:
                    check_global(global_stack, local, ns, ns_lock, idle_workers_sem)
                    leaves_since_last_check = 0
            # No further casts can be done on this cast history, so set ch to
            # None so the top of the loop will get the next cast history to
            # process.
            ch = None
            continue

        # If the optimistic estimate of the current node is lower than the
        # best damage we've seen so far, then don't pursue it.
        if ch.upper_bound < local.dmg:
            ch = None
            continue

        # At least one spell in the cast history can be cast.
        new_level = []
        for spell in ch.castable_spells():
            ch_copy = ch.duplicate()
            ch_copy.cast(spell)
            new_level.append(ch_copy)
        #new_level.sort(key=lambda c:c.upper_bound)
        local.stack += new_level

        # Take the last item we just pushed onto the local stack as the next
        # one to process.
        ch = local.stack.pop()


def find_best_cast_history(spells, num_targets, time_limit):
    """ Return a CastHistory object that maximizes damage on the targets within
    the specified time limit.

    Arguments:
    - `spells`: a list of all spells that can be cast
    - `num_targets`: linked encounter size
    - `time_limit`: encounter duration
    """
    mgr = multiprocessing.Manager()
    stack = Stack(mgr)
    # get multiple processes going sooner by pushing first tier instead of root
    root_ch = CastHistory(spells, num_targets, time_limit)
    for spell in root_ch.castable_spells():
        ch_copy = root_ch.duplicate()
        ch_copy.cast(spell)
        stack.push(ch_copy)
    ns = mgr.Namespace()
    lock = multiprocessing.Lock()
    idle_workers_sem = multiprocessing.Semaphore(0)
    io_lock = multiprocessing.Lock()  # for stdio writing
    ns.best_ch_repr = ''
    ns.best_dmg = 0.0
    ns.leaves = 0
    ns.stack_depths = {}
    ns.leaf_counts = {}
    start_time = time.time()
    nprocs = max(1, multiprocessing.cpu_count()-1)  # avoid 100% utilization
    procs = []
    for i in range(nprocs):
        p = multiprocessing.Process(target=worker, args=(stack, lock, ns, io_lock, idle_workers_sem))
        procs.append(p)
        p.start()
    # start the reporter process
    p = multiprocessing.Process(target=reporter, args=(stack, time_limit, ns, io_lock))
    procs.append(p)
    p.start()
    # block until there are no live nodes left
    stack.join()
    # all done, tell workers to shut down
    for i in range(nprocs):
        stack.push(POISON_PILL)
    # wait for workers and reporter to exit
    for p in procs:
        p.join()

    print 'Encounter size {} mob{}, time limit {}s'.format(
        num_targets,
        '' if num_targets==1 else 's',
        time_limit
    )
    print 'examined {:,} leaves in {}'.format(
        ns.leaves,
        timedelta(seconds=time.time() - start_time))
    best_ch = CastHistory(spells, num_targets, time_limit)
    best_ch.from_repr(ns.best_ch_repr)
    print 'best result {:,.0f} damage ({:,.0f} dps)'.format(
        best_ch.total_damage,
        best_ch.total_damage / float(time_limit),
    )
    #best_ch.print_history()
    return best_ch


def main():
    spells = [
        InitialDamageSpell('Animated Dagger', 0.29, 15, 51178),
        InitialDamageSpell('Arcane Bewilderment', 0.29, 30, 44607),
        InitialDamageSpell('Crystal Blast', 1.0, 1.5, 76498),
        InitialDamageSpell('Earthquake', 1, 10, 77448, 8),
        InitialDamageSpell('Ice Storm', 1.5, 4.5, 47762, 12),
        InitialDamageSpell('Petrify', 1.5, 22.5, 75916),
        InitialDamageSpell("Theurgist's Detonation", 0.29, 30, 106423, 8),
        DotSpell('Fiery Annihilation', 0.58, 4, 105713, 2, 7368, 12),
        DotSpell('Shattered Earth', 1.5, 20, 29348, 1.0, 18179, 5.0, 12),
        DotSpell('Vampire Bats', 1.16, 6, 76292, 4.0, 14522, 24.0),
        WindsOfVeliousSpell(0.58, 6, 82084, 2.7, 17332, 33.0),
        ]
    #InitialDamageSpell('spell 1', 0.5, 1, 10),
    #InitialDamageSpell('spell 2', 0.5, 1, 100),
    # InitialDamageSpell('Crystal Blast w/CD', 1.0, 1.5, 136032),
    #InitialDamageSpell("Master's Strike", 1.16, 60, 128622),

    find_best_cast_history(spells, 1, 9.5)

if __name__ == '__main__':
    main()