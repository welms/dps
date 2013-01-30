import abc
from operator import itemgetter

# Number of seconds that spell casting can occur.  The duration of the
# fight, in other words.
TIME_LIMIT = 8.0

# Cooldown, in seconds, between using abilities (spells or combat arts).
ABILITY_RECOVERY = 0.3

# Number of targets in encounter.
NUM_TARGETS = 1


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
        # Extend the current cast history by choosing a sequence of spells to
        # cast that result in the highest total damage for the encounter. The
        # "optimistic" part of the estimate is that we allow the last spell, the
        # one that runs into the encounter time limit, to be partially cast.
        #
        # That is, we pretend that it can do a fraction of its damage equal to
        # the fraction of casting time it has. For this special case, damage
        # over time spells are not allowed to count any of their periodic
        # damage, only the fraction of their initial damage.
        self._upper_bound = self.total_damage
        ubch = self.duplicate()
        while True:
            new_upper_bound = ubch._find_best_partial_cast()
            if new_upper_bound > self._upper_bound:
                self._upper_bound = new_upper_bound
            else:
                break

    def _find_best_partial_cast(self):
        """ Cast, or partially cast, the spell that results in the highest total
        damage for the current cast history, and return the new total damage.
        """
        # Find the spell that maximizes total damage over cumulative casting
        # time. Because spells can interact (e.g., recasting a dot before a
        # previous cast's duration is complete terminates the dot ticks of the
        # previous cast, thus lowering the damage done by the previous cast) we
        # choose the spell to cast based on the entire history's damage over the
        # entire history's cast time, rather than just choosing the highest dps
        # spell in isolation.
        #
        # Also, since we're being called by the optimistic upper bound
        # estimator, we are allowed to pretend that a spell can be partially
        # cast, such that the fraction of its cast time which is available is
        # also the fraction of the damage it does.
        best_dps = None
        best_spell = None
        for spell in self._spells:
            # Try (perhaps partially) casting this spell and see what the
            # resulting cast history dps is.
            if not spell.partially_castable(self):
                continue
            lc = self.cast(spell)
            dps = spell.damage(self, lc) / (spell.cast_time(self, lc) + ABILITY_RECOVERY)
            if best_dps is None or dps > best_dps:
                best_dps = dps
                best_spell = spell
            # Now undo that cast so we can examine the next
            self._casts.pop()
            self._total_damage = None
        if best_spell is not None:
            self.cast(best_spell)
        return self.total_damage

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
        try:
            last = (spell_cast for spell_cast in reversed(self._casts)
                    if spell_cast.spell == spell).next()
        except StopIteration:
            last = None
        return last

    def next_cast(self, spell_cast):
        """ Return the first SpellCast object in the cast history which follows
        `spell_cast` and is for the same spell, or None if no such cast has
        been made.
        """
        casts_after = self._casts[self._casts.index(spell_cast) + 1:]
        try:
            nxt = (sc for sc in casts_after
                   if sc.spell == spell_cast.spell).next()
        except StopIteration:
            nxt = None
        return nxt

    def prev_cast(self, spell_cast):
        """ Return the first SpellCast object in the cast history which precedes
        `spell_cast` and is for the same spell, or None if no such cast has been
        made.
        """
        preceding_casts = self._casts[:self._casts.index(spell_cast)]
        try:
            prev = (sc for sc in reversed(preceding_casts)
                    if sc.spell == spell_cast.spell).next()
        except StopIteration:
            prev = None
        assert prev is None or prev.spell == spell_cast.spell
        return prev

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

    def partially_castable(self, cast_history):
        """ Return True if there is enough time remaining in the specified cast
        history to start casting this spell.
        """
        return self.next_casting_time(cast_history) < cast_history.time_limit

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

    def damage(self, cast_history, spell_cast):
        assert isinstance(cast_history, CastHistory)
        assert isinstance(spell_cast, SpellCast)
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

    def damage(self, cast_history, spell_cast):
        # Determine how long the dot can run; the limit is either the end of the
        # encounter or the next time that this spell is recast, thus
        # interrupting itself.
        next_cast_of_this_spell = cast_history.next_cast(spell_cast)
        if next_cast_of_this_spell is None:
            last_time_for_dot = cast_history.time_limit
        else:
            last_time_for_dot = next_cast_of_this_spell.time_of_cast
        remaining_time = last_time_for_dot - spell_cast.time_of_cast
        # The total damage will be the sum of the initial damage and any damage
        # ticks that have time to occur.
        assert remaining_time > 0.0
        targets_hit = min(cast_history.num_targets, self.max_targets)
        cast_time = self.cast_time(cast_history, spell_cast)
        if remaining_time < cast_time:
            # being called by optimistic estimator: return partial damage.
            # Note the first tick of dot damage happens at the time of cast.
            if cast_time > self._cast_time:
                cooldown_remaining = cast_time - self._cast_time
                partial_cast_time = remaining_time - cooldown_remaining
                if partial_cast_time <= 0.0:
                    return 0.0
            else:
                partial_cast_time = cast_time
            return ((self.initial_damage + self.period_damage) *
                    targets_hit *
                    remaining_time / partial_cast_time)
        # Figure out how many times the dot got to do its periodic damage. The
        # first tick of a dot occurs at the time of the cast, so there is
        # always at least one tick.
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

    #def damage(self, cast_history, spell_cast):
        # for a normal (full) cast, can use super().damage. But for partial
        # cast, WoV must not consider any cooldown time as fractional casting
        # time, because that gives an overly generous estimate of the partial
        # cast damage.
        #raise NotImplementedError

    def next_casting_time(self, cast_history):
        # The cast_time of any instance of WoV will include any portion of the
        # cooldown remaining from the previous cast, so the next casting time
        # of WoV is actually simpler than other spells since it ignores its
        # own cooldown.
        return cast_history.last_ability_recovery_finished_time

    def partially_castable(self, cast_history):
        """ Return True if there is enough time remaining in the specified cast
        history to start casting this spell.
        """
        # WoV is only partially castable if the cooldown time from a previous
        # cast that is built into the current cast elapses before the time
        # limit. Even then, the fraction of its casting time used to compute
        # the partial damage excludes the cooldown (see self.damage method).
        casting_time = self.cast_time(cast_history)
        lar = cast_history.last_ability_recovery_finished_time
        if casting_time > self._cast_time:
            actual_cast_start_time = lar + (casting_time - self._cast_time)
            return actual_cast_start_time < cast_history.time_limit
        return lar < cast_history.time_limit


class SpellCast(object):
    """ Represents a cast of a given spell at a specific point in time.
    """
    def __init__(self, spell, time_of_cast):
        self.spell = spell
        self.time_of_cast = time_of_cast


def find_best_cast_history(spells, num_targets, time_limit):
    """ Return a CastHistory object that maximizes damage on the targets within
    the specified time limit.

    Arguments:
    - `spells`: a list of all spells that can be cast
    - `num_targets`: linked encounter size
    - `time_limit`: encounter duration
    """
    stack = [CastHistory(spells, num_targets, time_limit)]

    best_ch = None    # the highest total damage casting order
    upper_bound = None    # the highest optimistic estimate of total damage
    leaves_found = 0      # just info to display

    while len(stack) > 0:
        ch = stack.pop()
        if not ch.can_cast_something():
            # We've reached a leaf node. Check if it's the best realized (as
            # opposed to estimated, which is the bound) damage so far.
            leaves_found += 1
            if (best_ch == None or
                ch.total_damage > best_ch.total_damage):
                best_ch = ch
                # If the upper bound estimate is lower than any actual damage,
                # there's a bug in the estimator.
                assert best_ch.total_damage <= upper_bound
                print '=' * 60
                print 'Stack size:', len(stack)
                print 'Best of {} leaves so far: {:,.0f} damage, {:,.0f} dps:'.format(
                    leaves_found,
                    best_ch.total_damage,
                    best_ch.total_damage / best_ch.time_limit)
                best_ch.print_history()
            continue

        # It's possible to cast something in the current cast history, but if
        # its optimistic estimate (the upper bound) of the damage it can do is
        # lower than the current highest estimate, prune the tree at this
        # branch.
        if upper_bound != None and ch.upper_bound < upper_bound:
            pass
            continue

        # If the upper bound on damage from the current cast history is greater
        # than the highest upper bound we have, then it becomes the new upper
        # bound.
        if upper_bound == None or ch.upper_bound > upper_bound:
            upper_bound = ch.upper_bound
            print "New upper bound = {:,.0f}".format(upper_bound)

        # Expand the tree at the current node: push one new node on the stack
        # for every spell that could be cast from here. When they are next
        # examined, they'll be discarded if their optimistic estimates aren't >=
        # upper_bound.
        for spell in ch.castable_spells():
            ch_copy = ch.duplicate()
            ch_copy.cast(spell)
            stack.append(ch_copy)

    print 'examined', leaves_found, 'leaves'
    print '*** RUN COMPLETE ***'
    return best_ch


if __name__ == '__main__':
    spells = [
        InitialDamageSpell('Animated Dagger', 0.29, 15, 51178),
        InitialDamageSpell('Arcane Bewilderment', 0.29, 30, 44607),
        InitialDamageSpell('Crystal Blast', 1.0, 1.5, 76498),
        InitialDamageSpell('Earthquake', 1, 10, 77448, 8),
        InitialDamageSpell('Ice Storm', 1.5, 4.5, 47762, 12),
        InitialDamageSpell('Petrify', 1.5, 22.5, 75916),
        InitialDamageSpell("Theurgist's Detonation", 0.29, 30, 106423, 8),

        #InitialDamageSpell('Fiery Annihilation id', 0.58, 4, 105713),
        #InitialDamageSpell('Shattered Earth id', 1.5, 20, 29348),
        #InitialDamageSpell('Vampire Bats id', 1.16, 6, 76292),
        #InitialDamageSpell('Wov Standin id', 0.58, 6, 82084),

        DotSpell('Fiery Annihilation', 0.58, 4, 105713, 2, 7368, 12),
        DotSpell('Shattered Earth', 1.5, 20, 29348, 1.0, 18179, 5.0, 12),
        DotSpell('Vampire Bats', 1.16, 6, 76292, 4.0, 14522, 24.0),
        DotSpell('Wov Standin', 0.58, 6, 82084, 2.7, 17332, 33.0),
        #WindsOfVeliousSpell(0.58, 6, 82084, 2.7, 17332, 33.0),
        ]
    #InitialDamageSpell('spell 1', 0.5, 1, 10),
    #InitialDamageSpell('spell 2', 0.5, 1, 100),
    # InitialDamageSpell('Crystal Blast w/CD', 1.0, 1.5, 136032),
    #InitialDamageSpell("Master's Strike", 1.16, 60, 128622),
    find_best_cast_history(spells, NUM_TARGETS, TIME_LIMIT)
