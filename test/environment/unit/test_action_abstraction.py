"""The action abstraction: which actions each situation offers.

:data:`~environment.poker_env.RAISE_SIZES_BY_STAGE` defines, per stage, the
``"first_raise"`` grid (no raise in yet) and the ``"subsequent_raise"`` grid
(facing one).  These tests assert the *rules* that relate the table to
:attr:`PokerEnv.legal_actions` — never the tuning values themselves — so
re-cutting the grid does not touch this file.
"""

import pytest

from environment.player import Player
from environment.poker_env import (
    MAX_RAISES_PER_ROUND,
    PokerEnv,
    RAISE_SIZES_BY_STAGE,
    raise_level,
)
from test.abstraction_helpers import advance_to_round, passive_action

_STAGES = ("pre_flop", "flop", "turn", "river")
_CELL_KEYS = ("first_raise", "subsequent_raise")


def _env(n_players: int = 2, chips: int = 10000):
    return PokerEnv(players=[Player(i, chips) for i in range(n_players)])


def _legal(env):
    return [a for a in env.legal_actions if a is not None]


def _fractions(env):
    return sorted(
        float(a.split(":", 1)[1]) for a in _legal(env) if a.startswith("raise:")
    )


def _cell(stage, n_raises):
    """The grid cell the env draws from at ``n_raises`` raises this round."""
    return RAISE_SIZES_BY_STAGE[stage][_CELL_KEYS[raise_level(stage, n_raises)]]


class TestTableShape:
    """Structural invariants the size table must satisfy."""

    @pytest.mark.parametrize("stage", _STAGES)
    def test_every_stage_configures_both_cells(self, stage):
        cfg = RAISE_SIZES_BY_STAGE[stage]
        for key in _CELL_KEYS:
            assert cfg.get(key), f"{stage} has no {key} sizes"

    @pytest.mark.parametrize("stage", _STAGES)
    def test_fractions_are_positive_and_distinct(self, stage):
        for key in _CELL_KEYS:
            cell = RAISE_SIZES_BY_STAGE[stage][key]
            assert all(f > 0 for f in cell)
            assert len(set(cell)) == len(cell)

    @pytest.mark.parametrize("stage", _STAGES)
    def test_level_is_first_raise_only_before_a_raise(self, stage):
        assert raise_level(stage, 0) == 0
        for n_raises in range(1, MAX_RAISES_PER_ROUND + 3):
            assert raise_level(stage, n_raises) == 1


class TestCanonicalActionSet:
    """``get_canonical_actions`` is the union over both cells — the fixed
    regret-row layout — so it is a superset of any single node's legal set."""

    @pytest.mark.parametrize("betting_round", range(4))
    def test_is_the_union_of_both_cells(self, betting_round):
        stage = _STAGES[betting_round]
        cfg = RAISE_SIZES_BY_STAGE[stage]
        expected = sorted(set(cfg["first_raise"]) | set(cfg["subsequent_raise"]))
        canonical = PokerEnv.get_canonical_actions(betting_round)
        assert canonical[:3] == ["fold", "call", "all_in"]
        assert [float(a.split(":", 1)[1]) for a in canonical[3:]] == expected

    @pytest.mark.parametrize("betting_round", range(4))
    def test_is_sorted_and_deduplicated(self, betting_round):
        canonical = PokerEnv.get_canonical_actions(betting_round)
        assert len(set(canonical)) == len(canonical)

    def test_legal_actions_are_a_subset_of_the_canonical_set(self):
        # Walk a hand across every street and check the containment at each node.
        env = _env()
        guard = 0
        while not env.is_terminal and guard < 40:
            canonical = set(PokerEnv.get_canonical_actions(env.betting_round))
            assert set(_legal(env)) <= canonical
            env.step_in_place(passive_action(env))
            guard += 1


class TestRaiseCountSelectsTheGrid:
    """The offered raise sizes come from the current cell (filtered by the
    stack and the min-raise rule, which can only *remove* sizes)."""

    @pytest.mark.parametrize("betting_round", range(4))
    def test_offered_sizes_are_a_subset_of_the_cell(self, betting_round):
        env = _env()
        if betting_round:
            advance_to_round(env, betting_round)
        stage = _STAGES[betting_round]
        for _ in range(MAX_RAISES_PER_ROUND):
            if env.is_terminal or env.betting_round != betting_round:
                break
            assert set(_fractions(env)) <= set(_cell(stage, env.n_raises_this_round))
            raises = [a for a in _legal(env) if a.startswith("raise:")]
            if not raises:
                break
            env.step_in_place(raises[0])

    def test_first_in_and_facing_a_bet_use_different_cells(self):
        env = advance_to_round(_env(), 1)
        stage = "flop"
        cfg = RAISE_SIZES_BY_STAGE[stage]
        if set(cfg["first_raise"]) == set(cfg["subsequent_raise"]):
            pytest.skip("flop configures the same sizes for both cells")
        first_in = set(_fractions(env))
        env.step_in_place([a for a in _legal(env) if a.startswith("raise:")][0])
        assert set(_fractions(env)) <= set(cfg["subsequent_raise"])
        assert first_in <= set(cfg["first_raise"])


class TestPassiveActionsAlwaysAvailable:
    """The abstraction never leaves a player with chips unable to continue."""

    def test_a_free_check_is_always_offered(self):
        env = advance_to_round(_env(), 1)
        assert env.chips_to_add("call") == 0
        assert "call" in _legal(env)

    def test_call_for_less_is_offered_as_all_in(self):
        # Facing a bet bigger than the stack, the response is spelled "all_in"
        # and is a CALL — it must survive or the short stack could only fold.
        env = PokerEnv(players=[Player(0, 10000), Player(1, 300)])
        while "all_in" not in _legal(env) and not env.is_terminal:
            raises = [a for a in _legal(env) if a.startswith("raise:")]
            env.step_in_place(raises[0] if raises else passive_action(env))
        env.step_in_place("all_in")
        if not env.is_terminal:
            assert "all_in" in _legal(env) or "call" in _legal(env)

    def test_a_player_with_chips_is_never_left_only_folding(self):
        for chips in (150, 200, 260, 306, 400, 1000, 10000):
            env = PokerEnv(players=[Player(i, chips) for i in range(2)])
            guard = 0
            while not env.is_terminal and guard < 40:
                legal = _legal(env)
                assert legal != ["fold"], (
                    f"forced fold at {env.betting_stage} with {chips} chips"
                )
                env.step_in_place(passive_action(env))
                guard += 1


class TestOverlayCoversOffTreeActions:
    """An opponent is not bound by our abstraction: an observed off-tree size
    must be injectable, or it would have to be remapped before it can be played."""

    def test_off_grid_raise_can_be_injected_and_played(self):
        env = advance_to_round(_env(), 1)
        on_tree = set(_fractions(env))
        off = 0.77
        assert off not in on_tree, "pick a fraction the flop grid does not offer"
        pot_before = env.pot_size
        assert env.inject_action(f"raise:{off}") is True
        assert f"raise:{off}" in _legal(env)
        assert env.has_overlay_at_current_node
        env.step_in_place(f"raise:{off}")
        assert env.pot_size > pot_before

    def test_an_action_already_offered_is_a_no_op_injection(self):
        # Fold is offered wherever an active player acts, so injecting it is a
        # no-op returning True — and it never lands in the overlay.
        env = _env()
        assert env.inject_action("fold") is True
        assert not env.has_overlay_at_current_node

    def test_injection_is_refused_at_a_terminal(self):
        env = _env()
        env.step_in_place("fold")           # heads-up fold ends the hand
        assert env.is_terminal
        assert env.inject_action("call") is False
        assert env.inject_action("all_in") is False
