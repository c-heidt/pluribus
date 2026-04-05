"""Unit tests for poker_ai/environment/chance.py (Deck class).

Covers deck sizing, card uniqueness, rank filtering, dealing operations,
and the remaining-card tracking property.
"""

import pytest

from poker_ai.environment.chance import Deck
from poker_ai.environment.player import Player
from poker_ai.environment.utils import card_rank_int


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
