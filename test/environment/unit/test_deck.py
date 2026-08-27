"""Unit tests for poker_ai/environment/chance.py (Deck class).

Covers deck sizing, card uniqueness, rank filtering, dealing operations,
and the remaining-card tracking property.
"""

import pytest

from environment.chance import Deck
from environment.player import Player
from environment.utils import card_rank_int


class TestDeckConstruction:
    def test_full_deck_remaining_size(self, full_deck):
        assert len(full_deck.remaining) == 52

    def test_short_deck_remaining_size(self, short_deck):
        assert len(short_deck.remaining) == 20

    def test_full_deck_all_unique(self, full_deck):
        cards = list(full_deck.remaining)
        assert len(set(cards)) == 52

    def test_short_deck_all_unique(self, short_deck):
        cards = list(short_deck.remaining)
        assert len(set(cards)) == 20

    def test_short_deck_rank_range(self, short_deck):
        for c in short_deck.remaining:
            assert card_rank_int(int(c)) >= 10

    def test_custom_range(self):
        deck = Deck(6, 14)
        assert len(deck.remaining) == 36


class TestDealPrivateCards:
    def test_each_player_gets_two_cards(self, full_deck, three_players):
        full_deck.deal_private_cards(three_players)
        for player in three_players:
            assert len(player.cards) == 2

    def test_cards_are_integers(self, full_deck, two_players):
        full_deck.deal_private_cards(two_players)
        for player in two_players:
            for card in player.cards:
                assert isinstance(card, int)

    def test_no_overlap_between_players(self, full_deck, three_players):
        full_deck.deal_private_cards(three_players)
        all_cards = [c for p in three_players for c in p.cards]
        assert len(all_cards) == len(set(all_cards))

    def test_remaining_decreases_by_n_players_times_two(self, full_deck, three_players):
        initial = len(full_deck.remaining)
        full_deck.deal_private_cards(three_players)
        assert len(full_deck.remaining) == initial - 6

    def test_deals_in_two_pass_order(self, full_deck):
        players = [Player(i, 10000) for i in range(3)]
        full_deck.deal_private_cards(players)
        # Each player must have exactly 2 hole cards
        for p in players:
            assert len(p.cards) == 2


class TestDealCommunity:
    def test_deal_flop_returns_three_tuple(self, full_deck, two_players):
        full_deck.deal_private_cards(two_players)
        flop = full_deck.deal_community(3)
        assert isinstance(flop, tuple)
        assert len(flop) == 3

    def test_deal_turn_returns_one_tuple(self, full_deck, two_players):
        full_deck.deal_private_cards(two_players)
        full_deck.deal_community(3)
        turn = full_deck.deal_community(1)
        assert isinstance(turn, tuple)
        assert len(turn) == 1

    def test_community_cards_are_integers(self, full_deck):
        flop = full_deck.deal_community(3)
        for c in flop:
            assert isinstance(c, int)

    def test_remaining_shrinks_after_community(self, full_deck, two_players):
        full_deck.deal_private_cards(two_players)
        before = len(full_deck.remaining)
        full_deck.deal_community(3)
        assert len(full_deck.remaining) == before - 3

    def test_no_overlap_with_private_cards(self, full_deck, three_players):
        full_deck.deal_private_cards(three_players)
        community = full_deck.deal_community(3)
        private = {c for p in three_players for c in p.cards}
        for c in community:
            assert c not in private

    def test_community_cards_unique(self, full_deck):
        flop = full_deck.deal_community(3)
        full_deck.deal_community(1)  # turn
        turn = full_deck.deal_community(1)  # river
        all_community = list(flop) + list(turn)
        # No duplicates within what we dealt
        assert len(all_community) == len(set(all_community))


class TestRemaining:
    def test_remaining_decreases_monotonically(self, full_deck, two_players):
        r0 = len(full_deck.remaining)
        full_deck.deal_private_cards(two_players)
        r1 = len(full_deck.remaining)
        full_deck.deal_community(3)
        r2 = len(full_deck.remaining)
        full_deck.deal_community(1)
        r3 = len(full_deck.remaining)
        assert r0 > r1 > r2 > r3

    def test_remaining_returns_numpy_array(self, full_deck):
        import numpy as np
        assert isinstance(full_deck.remaining, np.ndarray)

    def test_deal_to_single_player(self, full_deck):
        players = [Player(0, 10000)]
        full_deck.deal_private_cards(players)
        assert len(players[0].cards) == 2

    def test_deck_is_shuffled(self):
        # Two independently constructed decks should almost never be identical
        import numpy as np
        deck1 = Deck(2, 14)
        deck2 = Deck(2, 14)
        # It is astronomically unlikely both are in the same order
        assert not np.array_equal(deck1.remaining, deck2.remaining)


class TestDeckEdgeCases:
    def test_deal_community_zero_returns_empty_tuple(self):
        deck = Deck(2, 14)
        result = deck.deal_community(0)
        assert result == ()

    def test_deal_community_zero_does_not_advance_index(self):
        deck = Deck(2, 14)
        before = len(deck.remaining)
        deck.deal_community(0)
        assert len(deck.remaining) == before

    def test_remaining_accounts_for_private_and_community(self):
        deck = Deck(2, 14)
        players = [Player(i, 10000) for i in range(2)]
        deck.deal_private_cards(players)  # 4 cards dealt
        deck.deal_community(3)            # 3 more
        assert len(deck.remaining) == 52 - 4 - 3

    def test_deal_community_five_cards(self):
        deck = Deck(2, 14)
        cards = deck.deal_community(5)
        assert isinstance(cards, tuple)
        assert len(cards) == 5
        assert len(deck.remaining) == 52 - 5


class TestShuffleUndealt:
    """``Deck.shuffle_undealt`` re-randomises positions ``>= _idx``.

    Used by :meth:`PokerEnv.with_hole_cards` after
    :meth:`replace_drawn` so the next community deal is uniform over
    the undealt set rather than preferring positions that received
    displaced cards from the swap.
    """

    def test_preserves_drawn_segment(self):
        import numpy as np
        deck = Deck(2, 14)
        players = [Player(i, 10000) for i in range(3)]
        deck.deal_private_cards(players)
        deck.deal_community(3)
        drawn_before = deck._cards[: deck._idx].copy()
        deck.shuffle_undealt()
        np.testing.assert_array_equal(deck._cards[: deck._idx], drawn_before)

    def test_preserves_undealt_set(self):
        deck = Deck(2, 14)
        players = [Player(i, 10000) for i in range(3)]
        deck.deal_private_cards(players)
        deck.deal_community(3)
        undealt_before = set(int(c) for c in deck.remaining)
        deck.shuffle_undealt()
        undealt_after = set(int(c) for c in deck.remaining)
        assert undealt_before == undealt_after

    def test_preserves_idx(self):
        deck = Deck(2, 14)
        players = [Player(i, 10000) for i in range(3)]
        deck.deal_private_cards(players)
        deck.deal_community(3)
        idx_before = deck._idx
        deck.shuffle_undealt()
        assert deck._idx == idx_before

    def test_changes_order_with_high_probability(self):
        # Two consecutive shuffles of the same undealt segment should
        # almost never produce identical orderings.
        import numpy as np
        deck = Deck(2, 14)
        players = [Player(i, 10000) for i in range(3)]
        deck.deal_private_cards(players)
        deck.deal_community(3)
        before = deck.remaining.copy()
        deck.shuffle_undealt()
        after = deck.remaining.copy()
        # With ~43 undealt cards, equal-after-shuffle has probability
        # 1 / 43! - effectively zero.
        assert not np.array_equal(before, after)

    def test_no_op_on_fully_dealt_deck(self):
        # If _idx == len(_cards) the undealt view is empty; shuffle
        # is a no-op and must not raise.
        deck = Deck(2, 14)
        deck._idx = len(deck._cards)
        deck.shuffle_undealt()
        assert deck._idx == len(deck._cards)


def _global_state():
    """Hashable snapshot of the global MT19937 state.

    Both the key array *and* the position index are needed: the key only changes
    when the generator refills (every 624 draws), so comparing it alone would call
    a stream that HAS been consumed 'untouched'.
    """
    import numpy as np
    st = np.random.get_state()
    return (st[0], st[1].tobytes(), st[2], st[3], st[4])


class TestShuffleUndealtRng:
    """``shuffle_undealt(rng)`` draws from the caller's stream, not the global one.

    A hypothetical re-deal (a search leaf rollout, an AIVAT value evaluation) that
    draws from the global stream makes its randomness depend on how much unrelated
    work consumed that stream.  These pin the escape hatch that lets such callers
    own their randomness.
    """

    def _dealt_deck(self):
        deck = Deck(2, 14)
        players = [Player(i, 10000) for i in range(3)]
        deck.deal_private_cards(players)
        deck.deal_community(3)
        return deck

    def test_leaves_global_state_untouched(self):
        import numpy as np
        deck = self._dealt_deck()
        before = _global_state()
        deck.shuffle_undealt(np.random.default_rng(0))
        assert _global_state() == before

    def test_global_path_still_consumes_global(self):
        # The ``rng=None`` default must keep its old behaviour for the played deal.
        deck = self._dealt_deck()
        before = _global_state()
        deck.shuffle_undealt()
        assert _global_state() != before

    def test_same_rng_seed_gives_same_order(self):
        import numpy as np
        a, b = self._dealt_deck(), self._dealt_deck()
        b._cards = a._cards.copy()
        a.shuffle_undealt(np.random.default_rng(5))
        b.shuffle_undealt(np.random.default_rng(5))
        np.testing.assert_array_equal(a._cards, b._cards)

    def test_different_rng_seed_gives_different_order(self):
        import numpy as np
        a, b = self._dealt_deck(), self._dealt_deck()
        b._cards = a._cards.copy()
        a.shuffle_undealt(np.random.default_rng(5))
        b.shuffle_undealt(np.random.default_rng(6))
        assert not np.array_equal(a._cards, b._cards)

    def test_preserves_drawn_segment_and_undealt_set(self):
        import numpy as np
        deck = self._dealt_deck()
        drawn_before = deck._cards[: deck._idx].copy()
        undealt_before = set(int(c) for c in deck.remaining)
        deck.shuffle_undealt(np.random.default_rng(1))
        np.testing.assert_array_equal(deck._cards[: deck._idx], drawn_before)
        assert set(int(c) for c in deck.remaining) == undealt_before


class TestForceNext:
    """``force_next`` / ``unforce`` — the counterfactual-board primitive.

    It is the only deal operation that mutates ``_cards``, so the invariants that
    matter are (a) the forced cards really are what comes out of the next
    ``deal_community``, and (b) ``unforce`` restores the array **exactly**, since
    ``capture``/``restore`` (and hence ``PokerEnv.undo``) save only the cursor and
    will not clean up after it.
    """

    @staticmethod
    def _dealt_deck(n_community: int = 3) -> Deck:
        deck = Deck(2, 14)
        deck.deal_private_cards([Player(i, 100) for i in range(2)])
        if n_community:
            deck.deal_community(n_community)
        return deck

    def test_forced_card_is_dealt_next(self):
        deck = self._dealt_deck()
        target = int(deck.remaining[7])
        deck.force_next((target,))
        assert deck.deal_community(1) == (target,)

    def test_every_undealt_card_can_be_forced(self):
        # The AIVAT chance correction enumerates the whole remaining deck, so a
        # card that cannot be forced would silently drop an alternative.
        for i in range(len(self._dealt_deck().remaining)):
            deck = self._dealt_deck()
            target = int(deck.remaining[i])
            deck.force_next((target,))
            assert deck.deal_community(1) == (target,)

    def test_forcing_the_natural_next_card_is_a_no_op(self):
        deck = self._dealt_deck()
        token = deck.force_next((int(deck.remaining[0]),))
        assert token == []

    def test_multi_card_force_preserves_order(self):
        deck = self._dealt_deck()
        want = tuple(int(c) for c in deck.remaining[[5, 1, 9]])
        deck.force_next(want)
        assert deck.deal_community(3) == want

    def test_unforce_restores_the_array_exactly(self):
        import numpy as np
        deck = self._dealt_deck()
        before = deck._cards.copy()
        token = deck.force_next((int(deck.remaining[7]),))
        assert not np.array_equal(deck._cards, before)   # it really did move
        deck.unforce(token)
        np.testing.assert_array_equal(deck._cards, before)

    def test_unforce_restores_exactly_for_every_card(self):
        import numpy as np
        deck = self._dealt_deck()
        before = deck._cards.copy()
        for i in range(len(deck.remaining)):
            token = deck.force_next((int(deck.remaining[i]),))
            deck.unforce(token)
            np.testing.assert_array_equal(deck._cards, before)

    def test_unforce_restores_exactly_for_a_multi_card_force(self):
        import numpy as np
        deck = self._dealt_deck(0)
        before = deck._cards.copy()
        token = deck.force_next(tuple(int(c) for c in deck._cards[[11, 4, 30]]))
        deck.unforce(token)
        np.testing.assert_array_equal(deck._cards, before)

    def test_force_does_not_move_the_cursor(self):
        deck = self._dealt_deck()
        idx = deck._idx
        token = deck.force_next((int(deck.remaining[7]),))
        assert deck._idx == idx
        deck.unforce(token)
        assert deck._idx == idx

    def test_force_preserves_the_drawn_segment(self):
        import numpy as np
        deck = self._dealt_deck()
        drawn = deck._cards[: deck._idx].copy()
        deck.force_next((int(deck.remaining[9]),))
        np.testing.assert_array_equal(deck._cards[: deck._idx], drawn)

    def test_force_preserves_the_undealt_set(self):
        deck = self._dealt_deck()
        undealt = set(int(c) for c in deck.remaining)
        deck.force_next((int(deck.remaining[9]),))
        assert set(int(c) for c in deck.remaining) == undealt

    def test_forced_card_lands_in_the_board_region(self):
        # ``board_runout`` reads ``[_board_start : +5]``; a forced street card must
        # show up there or the compiled FastState twin would see a different board.
        deck = self._dealt_deck()
        target = int(deck.remaining[6])
        deck.force_next((target,))
        assert int(deck.board_runout(5)[3]) == target

    def test_already_dealt_card_is_rejected(self):
        deck = self._dealt_deck()
        dealt = int(deck._cards[0])
        with pytest.raises(ValueError, match="not available to deal"):
            deck.force_next((dealt,))

    def test_card_not_in_deck_is_rejected(self):
        deck = self._dealt_deck()
        with pytest.raises(ValueError, match="not available to deal"):
            deck.force_next((123456789,))

    def test_same_card_twice_is_rejected(self):
        deck = self._dealt_deck()
        c = int(deck.remaining[3])
        with pytest.raises(ValueError, match="not available to deal"):
            deck.force_next((c, c))

    def test_too_many_cards_is_rejected(self):
        deck = self._dealt_deck()
        want = [int(c) for c in deck.remaining] + [int(deck._cards[0])]
        with pytest.raises(ValueError, match="remain undealt"):
            deck.force_next(want)

    def test_a_rejected_force_leaves_the_deck_untouched(self):
        # Partial forcing would corrupt the next runout without raising anywhere
        # the caller can see it.
        import numpy as np
        deck = self._dealt_deck()
        before = deck._cards.copy()
        good = int(deck.remaining[8])
        with pytest.raises(ValueError):
            deck.force_next((good, int(deck._cards[0])))
        np.testing.assert_array_equal(deck._cards, before)
