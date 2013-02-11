import abc
from operator import itemgetter
from collections import defaultdict
import time
from datetime import timedelta
from dpslib import *

# Number of seconds that spell casting can occur.  The duration of the
# fight, in other words.
TIME_LIMIT = 8.5

# Number of targets in encounter.
NUM_TARGETS = 1

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
    leaves_found = 0      # just info to display
    start_time = time.time()
    while len(stack) > 0:
        ch = stack.pop()
        if not ch.can_cast_something():
            # We've reached a leaf node. Check if it's the best damage so far.
            leaves_found += 1
            if (best_ch == None or
                ch.total_damage > best_ch.total_damage):
                best_ch = ch
                print 'best damage {:,.0f} of {:,.0f} leaves'.format(
                    best_ch.total_damage, leaves_found)
            #print '=' * 60
            #print 'Stack size:', len(stack)
            #print 'Best of {} leaves so far: {:,.0f} damage, {:,.0f} dps:'.format(
            #leaves_found,
            #best_ch.total_damage,
            #best_ch.total_damage / best_ch.time_limit)
            #best_ch.print_history()
            continue

        # If the optimistic estimate of the current node is lower than the
        # best damage we've seen so far, then don't pursue it.
        if best_ch is not None and ch.upper_bound < best_ch.total_damage:
            continue

        # Expand the tree at the current node: push one new node on the stack
        # for every spell that could be cast from here. When they are next
        # examined, they'll be discarded if their optimistic estimates aren't >=
        # upper_bound.
        for spell in ch.castable_spells():
            ch_copy = ch.duplicate()
            ch_copy.cast(spell)
            stack.append(ch_copy)

    print 'Encounter size {} mob{}, time limit {}s'.format(
        NUM_TARGETS,
        '' if NUM_TARGETS==1 else 's',
        TIME_LIMIT
    )
    print 'examined {} leaves in {}'.format(
        leaves_found,
        timedelta(seconds=time.time() - start_time))
    print 'best result {:,.0f} damage ({:,.0f} dps)'.format(
        best_ch.total_damage,
        best_ch.total_damage / float(TIME_LIMIT),
    )
    best_ch.print_history()
    print '*** RUN COMPLETE ***'
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
    find_best_cast_history(spells, NUM_TARGETS, TIME_LIMIT)


if __name__ == '__main__':
    main()