import random
import time
from dpslib import *
from operator import itemgetter

class GaCastHistory(CastHistory):
    """ Adds methods required by a genetic algorithm to the CastHistory
    class."""

    def _cast_from_spell_list(self, sl):
        """ Cast as many spells in `sl` as can be cast, in the order they are
        presented."""
        for spell in sl:
            if spell.fully_castable(self):
                self.cast(spell)
        return self

    def _fill_from_spell_list(self, sl):
        """Cast as many spells in `sl` as possible, in the order they are
        presented, then fill in any remaining time optimally, using an
        exhaustive search."""
        self._cast_from_spell_list(sl)
        return self._optimal_fill()

    def _optimal_fill(self):
        """Fill whatever time remains in this cast history with the highest
        dps possible, using an exhaustive search."""
        choices = [self]
        for spell in self.castable_spells():
            copy = self.duplicate()
            copy.cast(spell)
            copy._optimal_fill()
            choices.append(copy)
        return max(choices, key=lambda ch:ch.total_damage)

    def breed(self, other):
        """ Return a (possibly empty) list of new cast histories that are
        created by combining self and other in some way."""
        # Self and other are combined by crossover breeding. Since they are
        # variable length, pick a length along the shortest sequence.
        min_len = min(len(self), len(other))
        if min_len < 1:
            return []
        # extract a list of the spells cast by each parent
        parent_spells = []
        parent_spells.append([sc.spell for sc in self._casts])
        parent_spells.append([sc.spell for sc in other._casts])
        # Choose a point between casts where the crossover occurs.
        xo_point = random.randint(0, min_len - 1)
        # Combine parent spell lists, producing two children
        child_spells = [
            parent_spells[0][:xo_point + 1] + parent_spells[1][xo_point + 1:],
            parent_spells[1][:xo_point + 1] + parent_spells[0][xo_point + 1:]]
        # The child spell lists as they stand may not actually be fully
        # castable, or may not fully utilize the cast history time limits.
        # Create a cast history that is as similar to the child spell list as
        # possible but still fully utilizes the encounter time limit. Because
        # of these restrictions, it is possible that the child will be a clone
        # of a parent.
        candidates = [
            GaCastHistory(self._spells, self.num_targets, self.time_limit),
            GaCastHistory(self._spells, self.num_targets, self.time_limit)]
        for i, ch in enumerate(candidates):
            ch._cast_from_spell_list(child_spells[i])
        child = max(candidates, key=lambda ch:ch.last_ability_recovery_time)
        return child._optimal_fill()

    def duplicate(self):
        """ Return a copy of this object which can be modified without affecting
        this one.
        """
        copy = GaCastHistory(self._spells, self.num_targets, self.time_limit)
        copy._casts = self._casts[:]
        copy._total_damage = self._total_damage
        return copy

    def mutate(self):
        """Mutate this cast history by some method."""
        # There are lots of ways to mutate a sequence: deletion, insertion,
        # scrambling, etc.
        my_spells = [sc.spell for sc in self._casts]

        if len(my_spells) > 2:
            original = my_spells[:]
            while original == my_spells:
                # choose a section to scramble
                slice_start = random.randint(0, len(my_spells)-2)
                slice_end = random.randint(slice_start+1, len(my_spells)-1)
                shuffled_slice = my_spells[slice_start:slice_end]
                random.shuffle(shuffled_slice)
                my_spells = (my_spells[:slice_start] +
                             shuffled_slice +
                             my_spells[slice_end:])
        else:
            random.shuffle(my_spells)
        self._casts = []
        self._fill_from_spell_list(my_spells)
        return self

    def random_fill(self):
        """Fill this cast history with randomly selected spells."""
        while self.can_cast_something():
            castable = [_ for _ in self.castable_spells()]
            next_cast = random.choice(castable)
            self.cast(next_cast)
        return self


def crossover(population_cdf):
    parent1 = None
    parent2 = None
    assert len(population_cdf) > 1
    while parent1 == parent2:
        parent1 = pick(population_cdf)
        parent2 = pick(population_cdf)
    return parent1.breed(parent2)


def make_population_cdf(population):
    damages = [ch.total_damage for ch in population]
    population_cdf = [[damages[0], population[0]]]
    for i in xrange(1, len(damages)):
        population_cdf.append(
            [population_cdf[i-1][0] + damages[i],
             population[i]])
    for pair in population_cdf:
        pair[0] /= population_cdf[-1][0]
    return population_cdf


def pick(cdf_list):
    assert cdf_list[-1][0] == 1.0
    val = random.random()
    for partition, choice in cdf_list:
        if partition >= val:
            assert choice
            return choice
    # shouldn't get here
    assert False, (
        'pick: last partition value {}'
        ' smaller than random.uniform value {}'.format(
            cdf_list[-1][0], val))
    return cdf_list[-1][1]


def find_best_cast_history(spells, num_targets, time_limit):
    start_time = time.time()
    MAX_POPULATION = 100
    MAX_GENERATIONS = 50
    MUTATION_CHANCE = 0.01
    CULL_PCT = 0.03
    ELITE_PCT = 0.03

    # create an initial population
    population = [
        GaCastHistory(spells, num_targets, time_limit).random_fill() for
        i in xrange(MAX_POPULATION)]

    # iteratively improve the population until no improvement appears in
    # successive generations or we've reached the maximum number of
    # generations.
    best_ever = None
    for generation in xrange(MAX_GENERATIONS):
        # Culling: discard the lowest-performing elements of the population
        population.sort(key=lambda ch:ch.total_damage, reverse=True)
        population = population[:-int(len(population) * CULL_PCT)]
        # Elitism: always propagate the best-performing individuals
        next_gen = population[:int(len(population) * ELITE_PCT)]
        # Use the fitness function to make a cumulative distribution function
        # for roulette-wheel selection.
        population_cdf = make_population_cdf(population)
        # Fill the next generation by breeding and mutating the current
        # population.
        while len(next_gen) < MAX_POPULATION:
            result = crossover(population_cdf)
            if random.random() < MUTATION_CHANCE:
                result.mutate()
            next_gen.append(result)
        # Report on the results
        best = max(next_gen, key=lambda x:x.total_damage)
        if best_ever is None or best.total_damage > best_ever.total_damage:
            best_ever = best.duplicate()
        print 'gen {: 3}: pop {} {:,} dmg {}'.format(
            generation, len(next_gen), best.total_damage, repr(best))
        population = next_gen
    print 'run complete in {}'.format(timedelta(seconds=time.time() - start_time))
    print 'best ever for num_targets={} time_limit={}:'.format(
        num_targets, time_limit)
    print '{:,.0f} damage, ({:,.0f} dps)'.format(
        best_ever.total_damage,
        best_ever.total_damage / time_limit)
    best_ever.print_history()
    return best_ever


def main():
    random.seed()
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
    find_best_cast_history(spells, 1, 12)


def test_pick():
    random.seed()
    cdf = [[1, 'a']]
    for i in xrange(2, 11):
        prev = cdf[i-2]
        cdf.append([i+prev[0], chr(ord(prev[1])+1)])
    for pair in cdf:
        pair[0] /= float(cdf[-1][0])
    pick_count = {}
    for i, letter in cdf:
        pick_count[letter] = 0
    for i in xrange(100):
        pick_count[pick(cdf)] += 1
    assert pick_count[cdf[0][1]] < pick_count[cdf[-1][1]]
    for i, letter in cdf:
        print letter, '#' * pick_count[letter]


if __name__ == '__main__':
    main()