from collections import defaultdict
from itertools import chain, izip_longest
from operator import itemgetter
import random
import time

from dpslib import *

standard_spells = [
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

def interleave(*iterables):
    sentinel = object()
    z = izip_longest(*iterables, fillvalue = sentinel)
    c = chain.from_iterable(z)
    f = filter(lambda x: x is not sentinel, c)
    return list(f)


class GaCastHistory(CastHistory):
    """ Adds methods required by a genetic algorithm to the CastHistory
    class."""

    def _best_candidate(self, candidate_spell_lists):
        return max([
            GaCastHistory(
                self._spells,
                self.num_targets,
                self.time_limit)._fill_from_spell_list(csl) for csl in
                    candidate_spell_lists],
                   key=lambda ch:ch.total_damage)

    def _breed_crossover(self, other):
        # Self and other are combined by crossover breeding. Since they are
        # variable length, pick a length along the shortest sequence.
        min_len = min(len(self), len(other))
        if min_len < 2:
            return self
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
        return [
            GaCastHistory(
                self._spells,
                self.num_targets,
                self.time_limit)._fill_from_spell_list(csl)
            for csl in child_spells]

    def _breed_interleave(self, other):
        my_spells = [sc.spell for sc in self._casts]
        other_spells = [sc.spell for sc in other._casts]
        child_spells = [interleave(my_spells, other_spells),
                        interleave(other_spells, my_spells)]
        return [
            GaCastHistory(
                self._spells,
                self.num_targets,
                self.time_limit)._fill_from_spell_list(csl)
            for csl in child_spells]


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
        return self.random_fill()

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

    def _mutate_delete(self):
        """Delete a spell at random, then optimally fill in the end"""
        my_spells = [sc.spell for sc in self._casts]
        delete_idx = random.randint(0, len(my_spells)-1)
        my_spells.pop(delete_idx)
        self._casts = []
        self._cast_from_spell_list(my_spells)
        self.random_fill()
        return self

    def _mutate_reverse(self):
        my_spells = [sc.spell for sc in self._casts]
        self._casts = []
        self._fill_from_spell_list(reversed(my_spells))
        return self

    def _mutate_shuffle(self):
        my_spells = [sc.spell for sc in self._casts]

        if len(my_spells) > 2:
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

    def breed(self, other):
        """ Return a new cast history that is created by combining self and
        other in some way."""
        if random.random() < 0.2:
            return self._breed_crossover(other)
        return self._breed_interleave(other)

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
        op = random.choice([self._mutate_delete, self._mutate_shuffle,
                            self._mutate_reverse])
        return op()

    def random_fill(self):
        """Fill this cast history with randomly selected spells."""
        while self.can_cast_something():
            castable = [_ for _ in self.castable_spells()]
            next_cast = random.choice(castable)
            self.cast(next_cast)
        return self


def breed(population_cdf):
    parent1 = None
    parent2 = None
    assert len(population_cdf) > 1
    while parent1 == parent2:
        parent1 = pick_proportionate(population_cdf)
        parent2 = pick_proportionate(population_cdf)
    return [parent1, parent2] + parent1.breed(parent2)


def diversity(population):
    """Return a number representing the genetic diversity of the given
    population."""
    damages = defaultdict(int)
    for ch in population:
        damages[ch.total_damage] += 1
    return len(damages)


def make_population_cdf(population):
    damages = [ch.total_damage for ch in population]
    population_cdf = [[damages[0], population[0]]]
    for i in xrange(1, len(damages)):
        population_cdf.append(
            [population_cdf[i-1][0] + damages[i],
             population[i]])
    return population_cdf


def pick_proportionate(cdf_list):
    val = random.randint(0, int(cdf_list[-1][0]))
    for partition, choice in cdf_list:
        if partition >= val:
            return choice
    # shouldn't get here
    assert False, (
        'pick: last partition value {}'
        ' smaller than random val {}'.format(
            cdf_list[-1][0], val))
    return cdf_list[-1][1]


def pick_tournament(cdf_list):
    tournament_size = 4
    best = None
    for i in range(tournament_size):
        choice = random.choice(cdf_list)[1]
        if best is None or choice.total_damage > best.total_damage:
            best = choice
    return best


def report(gen, population):
    best = max(population, key=lambda x:x.total_damage)
    damages = defaultdict(int)
    for ch in population:
        damages[ch.total_damage] += 1
    total_damage = 0.0
    total_count = 0
    for dmg, count in damages.iteritems():
        total_damage += dmg * count
        total_count += count
    mean = total_damage / total_count
    #print 'gen:{:< 4}  bins:{:< 4}  mean:{:< 8,.0f}  best: {:,.0f}'.format(
        #gen, len(damages), mean, best.total_damage)
    return len(damages), best


def find_best_cast_history(spells, num_targets, time_limit, elite=None, throttle=0):
    MAX_POPULATION = 1000
    MAX_GENERATIONS = 100
    MAX_MUTATION_CHANCE = 0.5
    MIN_MUTATION_CHANCE = 0.001
    BREED_PCT = 1.0
    MIN_BIN_PCT = 0.01

    # create an initial population
    if elite:
        population = elite
    else:
        population = []
    while len(population) < MAX_POPULATION:
        population.append(
            GaCastHistory(spells, num_targets, time_limit).random_fill())
    # report the best of the randomly generated population
    bins, best_ever = report(0, population)
    min_bins = int(MIN_BIN_PCT * MAX_POPULATION + 0.5)
    num_to_breed = int(BREED_PCT * MAX_POPULATION)
    # Iteratively improve the population
    nudge = (MAX_MUTATION_CHANCE - MIN_MUTATION_CHANCE) / MAX_GENERATIONS
    cur_mutation_chance = MAX_MUTATION_CHANCE
    for generation in xrange(1, MAX_GENERATIONS+1):
        # Use the fitness function to make a cumulative distribution function
        # for roulette-wheel selection.
        population_cdf = make_population_cdf(population)
        children = []
        cur_mutation_chance -= nudge
        for i in xrange(num_to_breed):
            result = breed(population_cdf)
            if random.random() < cur_mutation_chance:
                result[-1].mutate()
            if random.random() < cur_mutation_chance:
                result[-2].mutate()
            children += result[2:]
            if throttle:
                time.sleep(throttle)
        # init next generation to current population plus children
        next_gen = population + children
        next_gen_cdf = make_population_cdf(next_gen)
        population = [best_ever]
        # sort next generation by fitness, descending
        next_gen.sort(key=lambda ch:ch.total_damage, reverse=True)
        # take the fittest of the previous population plus the children as the
        # next generation.
        population = next_gen[:MAX_POPULATION]
        bins, best = report(generation, population)
        if best.total_damage > best_ever.total_damage:
            best_ever = best
        if bins < min_bins:
            break
    return [best_ever]

def td(s):
    time_str = str(timedelta(seconds=int(s)))
    if time_str.startswith('0:'):
        return time_str[2:]
    return time_str


def multirun(num_targets, enc_time_limit, elite=None):
    MAX_RUNS = 8
    start_time = time.time()
    for i in range(MAX_RUNS):
        elite = find_best_cast_history(standard_spells, num_targets, enc_time_limit, elite)
        cumulative_time = time.time() - start_time
        time_per_run = cumulative_time / (i + 1)
        est_total_time = time_per_run * MAX_RUNS
        etr = est_total_time - cumulative_time
        best = elite[0]
        print 'run {: 3}: {:,.0f} damage, ({:,.0f} dps) {}'.format(
            i+1,
            best.total_damage,
            best.total_damage / enc_time_limit,
            repr(best))
        print '  etr:', td(etr)
    print 'run complete in', td(time.time() - start_time)
    print 'best ever for num_targets={} time_limit={}:'.format(
        num_targets, enc_time_limit)
    print '{:,.0f} damage, ({:,.0f} dps)'.format(
        best.total_damage,
        best.total_damage / enc_time_limit)
    best.print_history()
    return best


def stmain():
    num_targets = 1
    time_limit = 9.5
    ch = GaCastHistory(standard_spells, num_targets, time_limit)
    #ch.from_repr('<CastHistory [SE,E,IS,WV,FA,CB,VB,IS,FA,P,E,TD,FA,CB,IS,CB,FA,SE,CB,IS,E,FA,AD,AB]>')
    multirun(num_targets, time_limit)

if __name__ == '__main__':
    main()