from dps import *
import re

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

#----------------------------------------------------------------------
epsilon = 1e-10
def close_enough(f1, f2):
    """Return True if two floating point numbers are almost equal."""
    return abs(f1-f2) < epsilon

#----------------------------------------------------------------------
def test_empty_cast_history():
    """
    """
    # cast time of 20 seconds, recast 10, damage 100
    spell1 = InitialDamageSpell('spell1', 20, 10, 100)
    # encounter time limit 10 seconds
    ch = CastHistory([spell1], 1, 10)
    # can't fully cast spell1
    assert not ch.can_cast_something()
    # no castable spells
    assert [_ for _ in ch.castable_spells()] == []
    # nothing cast yet
    assert ch.last_ability_recovery_finished_time == 0.0
    assert ch.last_cast() is None
    assert ch.time_of_last_cast(spell1) is None
    assert ch.total_damage == 0.0
    # damage = 100, cast time = 20, time limit is 10, so upper bound is
    # partial cast of 10/20 * 100 = 50.
    assert ch.upper_bound == 50.0

#----------------------------------------------------------------------
def ch_single_cast(ch, spell1):
    assert ch.upper_bound == 200.0
    # it should have been recorded as being cast in this cast history
    assert ch.time_of_last_cast(spell1) == 0.0
    # still time to cast again before end of encounter
    assert ch.last_ability_recovery_finished_time == 1.0
    assert spell1.next_casting_time(ch) == 1.0
    assert spell1.fully_castable
    assert ch.can_cast_something()
    # last cast should be the one we just made
    fc = ch.last_cast()
    assert isinstance(fc, SpellCast)
    assert fc.time_of_cast == 0.0
    assert fc.spell == spell1
    assert fc == ch.last_cast(spell1)
    assert ch.next_cast(fc) is None
    assert ch.prev_cast(fc) is None
    # one cast of the spell should make total damage the same as spell damage
    assert ch.total_damage == 100.0
    assert spell1.damage(ch, fc) == 100.0
    assert spell1.fully_castable(ch)
    return fc


def test_cast():
    """Test Spell, CastHistory, and SpellCast with two casts of the same
    spell which exactly fill the encounter time."""
    # cast time of 0.7 second, recast 0.3, damage 100.
    spell1 = InitialDamageSpell('spell1', 0.7, 0.3, 100)
    # Encounter time limit is 1.7 seconds, which is exactly enough to fully
    # cast spell1 twice, with no time left for a partial cast. Init cast
    # history with this one castable spell.
    ch = CastHistory([spell1], 1, 2)
    assert ch.upper_bound == 200.0
    castable = [_ for _ in ch.castable_spells()]
    assert len(castable) == 1
    assert castable[0] == spell1
    assert ch.can_cast_something()
    # now cast the spell
    ch.cast(spell1)
    fc = ch_single_cast(ch, spell1)
    # Before casting the spell again, make a copy of the cast history, which
    # will allow us to test that duplicate produces a copy which isn't changed
    # by changing the original.
    ch_copy = ch.duplicate()
    # Now cast the spell again.
    ch.cast(spell1)
    # Verify it has been recorded
    sc = ch.last_cast()
    assert isinstance(sc, SpellCast)
    assert fc.time_of_cast == 0.0 # unchanged
    assert sc.time_of_cast == 1.0
    assert sc.spell == spell1
    # Test previous and next
    assert ch.prev_cast(sc) == fc
    assert ch.next_cast(fc) == sc
    assert ch.prev_cast(fc) is None
    assert ch.next_cast(sc) is None
    # encounter is over
    assert not spell1.fully_castable(ch)
    assert not ch.can_cast_something()
    assert [_ for _ in ch.castable_spells()] == []
    # With a second cast, output of cast history total damage should grow, but
    # each (full) cast of the spell should be the same output.
    assert spell1.damage(ch, sc) == 100.0
    assert ch.upper_bound == 200.0
    assert ch.total_damage == 200.0
    # ch_copy should still look like ch did before the second cast
    ch_single_cast(ch_copy, spell1)


#----------------------------------------------------------------------
def test_last_cast():
    """get coverage on CastHistory last_cast stopiteration."""
    s1 = InitialDamageSpell('spell1', 0.7, 0.3, 100)
    s2 = InitialDamageSpell('spell2', 0.7, 0.3, 100)
    ch = CastHistory([s1, s2], 1, 2)
    ch.cast(s1)
    assert ch.last_cast(s2) is None


#----------------------------------------------------------------------
def test_print_history():
    """test it with one line of output"""
    s = InitialDamageSpell('spell1', 0.7, 0.3, 100)
    ch = CastHistory([s], 1, 2)
    ch.cast(s)
    ch.print_history()

#----------------------------------------------------------------------
def test_dot():
    """Test DotSpell class"""
    dot = DotSpell('mydot', 0.7, 0.3, 100, 1, 10, 2)
    ch = CastHistory([dot], 1, 4)
    ch.cast(dot)
    print dot.damage(ch, ch.last_cast())
    ch.cast(dot)
#    print dot.damage(ch, ch.last_cast())


#----------------------------------------------------------------------
def test_wov():
    """test WindsOfVelious"""
    wov = WindsOfVeliousSpell(0.58, 6, 82084, 2.7, 17332, 33.0)
    ch = CastHistory([wov], 1, 120)
    assert wov.cast_time(ch) == wov._cast_time
    ch.cast(wov)
    # with 60 second encouter time all ticks of wov should run
    full_wov_dmg = wov.initial_damage + wov.period_damage * 13
    assert full_wov_dmg == ch.total_damage
    # interrupt the first cast with a second cast. the first cast should get
    # off only the initial damage and one tick (since the first tick of a dot
    # is coincident with the initial damage) then there should be the
    # cooldown, then the second cast should run to completion.
    fc = ch.last_cast(wov)
    assert fc.time_of_cast == 0.0
    ch_copy = ch.duplicate()
    # haven't changed copy yet, so it should have same damage as original
    assert ch_copy.total_damage == full_wov_dmg
    # now cast wov a second time in the copy
    ch_copy.cast(wov)
    sc = ch_copy.last_cast()
    # check that the second cast of wov happened at the time that the original
    # cast history said it would.
    assert wov.next_casting_time(ch) ==  sc.time_of_cast
    # we expect the damage to be the initial damage and one tick from the
    # first cast, plus full damage from the second cast.
    assert (ch_copy.total_damage == full_wov_dmg + wov.initial_damage +
            wov.period_damage)
    assert sc.time_of_cast == wov._cast_time + ABILITY_RECOVERY
    # Now try with an intervening cast, so that the previous WoV gets in a
    # couple of damage ticks before being interrupted.

#----------------------------------------------------------------------
def test_wov_lart():
    """Check the last ability recovery time reflects the cooldown inherent in
    a self-interrupting second cast of WoV"""
    wov = WindsOfVeliousSpell(0.58, 6, 82084, 2.7, 17332, 33.0)
    ch = CastHistory([wov], 1, 120)
    # try two consecutive casts of wov
    ch.cast(wov)
    ch.cast(wov)
    assert (ch.last_ability_recovery_finished_time ==
            wov._cast_time +  # initial cast
            ABILITY_RECOVERY +  # must pass before you can cast anything else
            wov.cooldown +  # triggered by second cast
            wov._cast_time +  # second cast
            ABILITY_RECOVERY)  # after second cast
    # now try a second cast of wov that interrupts but is not consecutive
    spell1 = InitialDamageSpell('spell1', 0.7, 1, 100)
    ch = CastHistory([wov, spell1], 1, 120)
    ch.cast(wov)
    ch.cast(spell1)
    ch.cast(wov)
    assert close_enough(
        ch.last_ability_recovery_finished_time,
        wov._cast_time +  # initial cast
        ABILITY_RECOVERY +  # must pass before you can cast anything else
        spell1._cast_time +  # second spell cast
        ABILITY_RECOVERY +
        wov.cooldown +  # triggered by second cast
        wov._cast_time +  # second cast
        ABILITY_RECOVERY)  # after second cast


#----------------------------------------------------------------------
def test_find_casts():
    spell1 = InitialDamageSpell('spell1', 0.7, 1, 100)
    spell2 = InitialDamageSpell('spell2', 0.7, 1, 100)
    spell3 = InitialDamageSpell('spell3', 0.7, 1, 100)
    ch = CastHistory([spell1, spell2, spell3], 1, 1000)
    found = [_ for _ in ch.find_casts(spell1)]
    assert found == []
    ch.cast(spell1)
    found = [_ for _ in ch.find_casts(spell1)]
    assert len(found) == 1
    assert isinstance(found[0], SpellCast)
    assert found[0].time_of_cast == 0.0
    assert found[0].spell == spell1
    ch.cast(spell2)
    ch.cast(spell1)
    found = [_ for _ in ch.find_casts(spell1)]
    assert len(found) == 2
    assert found[0].spell == spell1
    assert found[1].spell == spell1
    # verify second cast happened when we thought it would, which also is
    # testing computation of cooldown time and ability recovery time.
    assert found[1].time_of_cast == 2.0

#----------------------------------------------------------------------
def test_ability_recovery():
    """Try casting spells consecutively"""
    spell1 = InitialDamageSpell('spell1', 0.7, 1, 100)
    ch = CastHistory([spell1], 1, 1000)
    ch.cast(spell1)
    assert ch.last_ability_recovery_finished_time == 1.0
    ch.cast(spell1)
    lc = ch.last_cast(spell1)
    assert lc.time_of_cast == 1.7

#----------------------------------------------------------------------
def test_cooling_down():
    """test CastHistory.cooling_down()"""
    spell1 = InitialDamageSpell('spell1', 0.7, 10, 100)
    spell2 = InitialDamageSpell('spell2', 0.7, 1, 100)
    ch = CastHistory([spell1, spell2], 1, 1000)
    # cast spell1
    ch.cast(spell1)
    lc = ch.last_cast()
    # it couldn't have been cooling down when it was cast
    assert not spell1.cooling_down(ch, lc)
    # cast a short second spell
    ch.cast(spell2)
    lc = ch.last_cast()
    # first spell should still be cooling down
    assert spell1.cooling_down(ch, lc)

#----------------------------------------------------------------------
def test_spell_string():
    """Check that the castable spell string correctly reflects cooldown
    state."""
    spell1 = InitialDamageSpell('Alpha bravo', 0.7, 10, 100)
    spell2 = InitialDamageSpell('charlie Delta', 0.7, 1, 100)
    ch = CastHistory([spell1, spell2], 1, 1000)
    ch.cast(spell1)
    lc = ch.last_cast()
    assert ch.castable_spell_string(lc) == 'AB CD'
    ch.cast(spell2)
    assert ch.castable_spell_string(ch.last_cast()) == 'ab CD'
    ch.cast(spell1)
    assert ch.castable_spell_string(ch.last_cast()) == 'AB CD'


#----------------------------------------------------------------------
def test_partial():
    """Test partial cast damage calculations."""
    spell1 = InitialDamageSpell('spell1', 10, 1, 100)
    ch = CastHistory([spell1], 1, 5)
    ch.cast(spell1)
    assert ch.total_damage == 50
    spell2 = DotSpell('spell2',
                      cast_time=10,
                      cooldown=1,
                      initial_damage=100,
                      period=1,
                      period_damage=10,
                      duration=10)
    ch = CastHistory([spell2], 1, 5)
    ch.cast(spell2)
    assert ch.total_damage == 55
    ch = CastHistory([spell2], 1, time_limit=11)
    ch.cast(spell2)
    assert ch.total_damage == 120


#----------------------------------------------------------------------
def make_leaves():
    """Generate the leaves.csv file."""
    # one target, 8 second encounter duration
    stack = [CastHistory(standard_spells, 1, 8.0)]
    leaves_found = 0      # just info to display
    start_time = time.time()
    ofile = open('leaves.csv', mode='w')
    best_ch = None
    while len(stack) > 0:
        ch = stack.pop()
        if not ch.can_cast_something():
            # We've reached a leaf node. Write it to disk.
            ofile.write('{}\t{}\n'.format(ch.total_damage, repr(ch)))
            # Check if it's the best damage so far.
            leaves_found += 1
            if (best_ch == None or
                ch.total_damage > best_ch.total_damage):
                best_ch = ch
            # nothing more to be done with a leaf node
            continue
        # not at a leaf, cast the rest
        for spell in ch.castable_spells():
            ch_copy = ch.duplicate()
            ch_copy.cast(spell)
            stack.append(ch_copy)
    ofile.close()

def test_ub():
    """test upper bound"""
    try:
        leaves = open('leaves.csv')
    except:
        make_leaves()
        leaves = open('leaves.csv')
    best_ch = CastHistory(standard_spells, 1, 8.0)
    # brute force search shows this is the maximum damage cast history
    best_ch.from_repr('<CastHistory [WV,SE,FA,TD,E,CB,AD,FA]>')
    # create a dict of upper bound values of all ancestors of this cast
    # history
    ubounds = {}
    ch = CastHistory(standard_spells, 1, 8.0)
    for spell_cast in best_ch._casts:
        ch.cast(spell_cast.spell)
        ubounds[repr(ch)] = ch.upper_bound
    # Now iterate through the leaves file. Print any leaves that have a total
    # damage greater than any upper bound in our dict.
    line_re = re.compile('([0-9]+\.[0-9]*)\t(.+)$')
    upper_bound_exceeded = False
    for line in leaves:
        dmg, leaf_repr = line_re.match(line).groups()
        dmg = float(dmg)
        for ub_repr, bound in ubounds.iteritems():
            if dmg > bound:
                print dmg, leaf_repr, 'exceeds', bound, ub_repr
                upper_bound_exceeded = True
    assert not upper_bound_exceeded


#----------------------------------------------------------------------
def test_repr():
    """Test CastHistory.__repr__"""
    # make a cast history with huge time limit so we can be sure everything
    # can be cast.
    ch = CastHistory(standard_spells, 1, 1000)
    assert repr(ch) == '<CastHistory []>'
    assert ch.total_damage == 0.0
    ch_repr = CastHistory(standard_spells, 1, 1000)
    ch_repr.from_repr(repr(ch))
    assert ch_repr.total_damage == 0.0
    ch.cast(standard_spells[0])
    assert repr(ch) == '<CastHistory [AD]>'

    ch = CastHistory(standard_spells, 1, 1000)
    # cast every spell
    for spell in standard_spells:
        ch.cast(spell)
    # expect that the resulting representation reflects this
    ch_repr = repr(ch)
    assert ch_repr == '<CastHistory [AD,AB,CB,E,IS,P,TD,FA,SE,VB,WV]>'
    # build a new cast history from the representation.
    rebuilt_ch = CastHistory(standard_spells, 1, 1000)
    rebuilt_ch.from_repr(ch_repr)
    # verify the cast history built from the repr string is the same as the
    # original
    assert repr(rebuilt_ch) == ch_repr
    assert rebuilt_ch.total_damage == ch.total_damage
    assert rebuilt_ch.upper_bound == ch.upper_bound

def show():
    ch = CastHistory(standard_spells, 1, 20)
    ch.from_repr('<CastHistory [FA,WV,VB,TD,E,FA,CB,AB,CB,FA,SE,CB,AD,FA,CB,E,CB,FA]>')
    ch.print_history()

from dps_ga import pick, GaCastHistory

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

def test_interleave():
    ch1 = GaCastHistory(standard_spells, 1, 8.4).random_fill()
    ch2 = GaCastHistory(standard_spells, 1, 8.4).random_fill()
    print repr(ch1)
    print repr(ch2)
    print repr(ch1._breed_interleave(ch2))

def test_reverse():
    ch = GaCastHistory(standard_spells, 1, 8.4).random_fill()
    print ch
    ch._mutate_reverse()
    print ch

if __name__ == '__main__':
    test_reverse()