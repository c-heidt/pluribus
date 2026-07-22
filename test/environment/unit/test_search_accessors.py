"""Tests for the env's search-facing accessors: ``public_key``,
``n_raises_this_round``, and ``cluster_for`` (§6.1)."""

import copy
import json
from collections import defaultdict

from environment.player import Player
from environment.poker_env import PokerEnv


def _env(n_players: int = 2):
    return PokerEnv(players=[Player(i, 10000) for i in range(n_players)])


def _first_raise(env) -> str:
    return next(a for a in env.legal_actions if a and a.startswith("raise"))


class TestPublicKey:

    def test_matches_current_public_state(self):
        env = _env()
        assert env.public_key == env._current_public_state()

    def test_preflop_root_value(self):
        env = _env()
        assert env.public_key == ("pre_flop", ())

    def test_reflects_history_and_is_hashable(self):
        env = _env()
        env_next = copy.deepcopy(env); env_next.step_in_place("call")
        # Full cross-street history: each touched stage carried as
        # (stage, actions), led by the current betting stage.
        assert env_next.public_key == ("pre_flop", (("pre_flop", ("call",)),))
        # Hashable → usable as a solver table key.
        _ = {env.public_key: 1, env_next.public_key: 2}

    def test_distinguishes_cross_street_lines(self):
        # Two river nodes reached via different earlier-street betting have
        # different pots/stacks and must NOT collide onto one key (regression
        # for the truncated-to-current-stage bug: every river node hashing to
        # ("river", ()) and silently sharing CFR rows / crashing widening).
        def to_river(raise_flop: bool) -> PokerEnv:
            env = _env()
            env.step_in_place("call")          # pre-flop
            env.step_in_place("call")
            if raise_flop:                     # flop: build a bigger pot on one line
                env.step_in_place(_first_raise(env))
                env.step_in_place("call")
            else:
                env.step_in_place("call")
                env.step_in_place("call")
            env.step_in_place("call")          # turn
            env.step_in_place("call")
            return env

        small, big = to_river(False), to_river(True)
        assert small.betting_round == 3 and big.betting_round == 3
        assert small.pot_size != big.pot_size          # genuinely different states
        assert small.public_key != big.public_key      # ... so distinct keys

    def test_independent_of_actor(self):
        # public_key embeds no actor cards — it's a pure function of the
        # public state, so every seat at the same public node sees the
        # same key.  A deepcopy (different object, same public state)
        # reports the same key.
        env = _env(n_players=3)
        env_copy = copy.deepcopy(env)
        assert env_copy.public_key == env.public_key

    def test_read_does_not_mutate_history_or_info_set(self):
        # ``_history`` is a defaultdict; a bare ``_history[stage]`` index
        # would insert an empty stage that then leaks into the info-set
        # key via ``_compute_info_set``'s ``items()`` iteration.  Reading
        # ``public_key`` / the overlay must be a true read: the info-set
        # string and ``_history`` are unchanged afterwards.
        env = _env()
        env.card_info_lut = defaultdict(lambda: defaultdict(lambda: 0))
        combo = (int(env.players[0].cards[0]), int(env.players[0].cards[1]))
        info_before = env._compute_info_set(combo)
        history_before = {k: list(v) for k, v in env._history.items()}

        _ = env.public_key
        _ = env.has_overlay_at_current_node

        assert env._compute_info_set(combo) == info_before
        assert {k: list(v) for k, v in env._history.items()} == history_before


class TestNRaisesThisRound:

    def test_zero_on_fresh_env(self):
        assert _env().n_raises_this_round == 0

    def test_matches_private_counter(self):
        env = _env()
        assert env.n_raises_this_round == env._n_raises

    def test_increments_after_raise(self):
        env = _env()
        env_next = copy.deepcopy(env); env_next.step_in_place(_first_raise(env))
        assert env_next.n_raises_this_round == 1

    def test_resets_at_round_boundary(self):
        # Raise then call to close pre-flop; the flop starts with the
        # counter back at zero.
        env = _env()
        env.step_in_place(_first_raise(env))
        assert env.n_raises_this_round == 1
        env.step_in_place("call")
        assert env.betting_round == 1
        assert env.n_raises_this_round == 0


class TestClusterFor:

    def test_returns_lut_cluster(self):
        env = _env()
        combo = (int(env.players[0].cards[0]), int(env.players[0].cards[1]))
        lookup = tuple(sorted(combo) + sorted(env.community_cards))
        env.card_info_lut = {env._betting_stage: {lookup: 7}}
        assert env.cluster_for(combo) == 7

    def test_agrees_with_info_set_embedded_cluster(self):
        env = _env()
        combo = (int(env.players[0].cards[0]), int(env.players[0].cards[1]))
        lookup = tuple(sorted(combo) + sorted(env.community_cards))
        env.card_info_lut = {env._betting_stage: {lookup: 13}}
        embedded, _history = env.info_set_fields(combo)
        assert env.cluster_for(combo) == embedded == 13

    def test_returns_python_int(self):
        env = _env()
        combo = (int(env.players[0].cards[0]), int(env.players[0].cards[1]))
        lookup = tuple(sorted(combo) + sorted(env.community_cards))
        env.card_info_lut = {env._betting_stage: {lookup: 4}}
        assert isinstance(env.cluster_for(combo), int)

    def test_folds_board_into_lookup_on_flop(self):
        # On the flop the LUT lookup must include the board cards.
        env = _env()
        env.step_in_place("call")
        env.step_in_place("call")
        assert env.betting_round == 1
        assert len(env.community_cards) == 3
        board = set(int(c) for c in env.community_cards)
        combo = next(
            (int(a), int(b))
            for a, b in env.combo_cards
            if int(a) not in board and int(b) not in board
        )
        lookup = tuple(sorted(combo) + sorted(env.community_cards))
        env.card_info_lut = {env._betting_stage: {lookup: 5}}
        assert env.cluster_for(combo) == 5
