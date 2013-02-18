import abc
from operator import itemgetter
from collections import defaultdict
import time
from datetime import timedelta

# Cooldown, in seconds, between using abilities (spells or combat arts).
ABILITY_RECOVERY = 0.3

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

    def __len__(self):
        return len(self._casts)

    def __iter__(self):
        return iter(self._casts)

    def __repr__(self):
        """ Return a string representation of the casts in this history. """
        delimited_casts = ''
        for cast in self._casts:
            if delimited_casts:
                delimited_casts += ','
            delimited_casts += cast.spell.shortname
        return '<CastHistory [{}]>'.format(delimited_casts)

    def __str__(self):
        return self.__repr__()

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


