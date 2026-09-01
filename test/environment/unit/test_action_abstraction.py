"""The per-raise-level action abstraction: which actions each situation offers.

:data:`~environment.poker_env.RAISE_SIZES_BY_STAGE` and the two passive gates
beside it define, per (stage, raise level), the raise grid and whether ``call``
/ ``all_in`` are part of the abstraction.  These tests assert the *rules* that
relate the tables to :attr:`PokerEnv.legal_actions` — never the tuning values
themselves — so re-cutting the grid does not touch this file.
"""

import pytest

from environment.player import Player
from environment.poker_env import (
    ALL_IN_ALLOWED_BY_STAGE,
    CALL_ALLOWED_BY_STAGE,
    MAX_RAISES_PER_ROUND,
    PokerEnv,
    RAISE_SIZES_BY_STAGE,
    raise_level,
)
from test.abstraction_helpers import advance_to_round, passive_action

_STAGES = ("pre_flop", "flop", "turn", "river")


def _env(n_players: int = 2, chips: int = 10000):
    return PokerEnv(players=[Player(i, chips) for i in range(n_players)])


def _legal(env):
    return [a for a in env.legal_actions if a is not None]


def _fractions(env):
    return sorted(
        float(a.split(":", 1)[1]) for a in _legal(env) if a.startswith("raise:")
    )


class TestTableShape:
    """Structural invariants the three tables must satisfy together."""

    @pytest.mark.parametrize("stage", _STAGES)
    def test_every_stage_has_a_level_zero(self, stage):
        assert RAISE_SIZES_BY_STAGE[stage], f"{stage} has no raise levels"
        assert all(cell for cell in RAISE_SIZES_BY_STAGE[stage]), (
            f"{stage} has an empty level"
        )

    @pytest.mark.parametrize("stage", _STAGES)
    def test_passive_gates_align_with_the_levels(self, stage):
        n = len(RAISE_SIZES_BY_STAGE[stage])
        assert len(CALL_ALLOWED_BY_STAGE[stage]) == n
        assert len(ALL_IN_ALLOWED_BY_STAGE[stage]) == n

    @pytest.mark.parametrize("stage", _STAGES)
    def test_fractions_are_positive_and_distinct(self, stage):
        for level in RAISE_SIZES_BY_STAGE[stage]:
            assert all(f > 0 for f in level)
            assert len(set(level)) == len(level)

    @pytest.mark.parametrize("stage", _STAGES)
    def test_last_level_repeats_for_deeper_raise_counts(self, stage):
        last = len(RAISE_SIZES_BY_STAGE[stage]) - 1
        for n_raises in range(last, MAX_RAISES_PER_ROUND + 3):
            assert raise_level(stage, n_raises) == last

    @pytest.mark.parametrize("stage", _STAGES)
    def test_level_index_is_the_raise_count_while_configured(self, stage):
        for n_raises in range(len(RAISE_SIZES_BY_STAGE[stage])):
            assert raise_level(stage, n_raises) == n_raises


class TestCanonicalActionSet:
    """``get_canonical_actions`` is the union over levels — the fixed regret-row
    layout — so it is a superset of any single node's legal set."""

    @pytest.mark.parametrize("betting_round", range(4))
    def test_is_the_union_of_every_level(self, betting_round):
        stage = _STAGES[betting_round]
        expected = sorted({f for lv in RAISE_SIZES_BY_STAGE[stage] for f in lv})
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


class TestLevelSelectsTheGrid:
    """The offered raise sizes come from the current level's cell (filtered by
    the stack and the min-raise rule, which can only *remove* sizes)."""

    @pytest.mark.parametrize("betting_round", range(4))
    def test_offered_sizes_are_a_subset_of_the_level_cell(self, betting_round):
        env = _env()
        if betting_round:
            advance_to_round(env, betting_round)
        stage = _STAGES[betting_round]
        for _ in range(MAX_RAISES_PER_ROUND):
            if env.is_terminal or env.betting_round != betting_round:
                break
            cell = RAISE_SIZES_BY_STAGE[stage][raise_level(stage, env.n_raises_this_round)]
            assert set(_fractions(env)) <= set(cell)
            raises = [a for a in _legal(env) if a.startswith("raise:")]
            if not raises:
                break
            env.step_in_place(raises[0])

    def test_first_in_and_facing_a_bet_use_different_cells(self):
        # Whenever a stage configures a distinct level-1 cell, the env must
        # actually switch to it after one raise.
        env = advance_to_round(_env(), 1)
        stage = "flop"
        if len(RAISE_SIZES_BY_STAGE[stage]) < 2:
            pytest.skip("flop configures a single level")
        first_in = set(_fractions(env))
        env.step_in_place([a for a in _legal(env) if a.startswith("raise:")][0])
        assert set(_fractions(env)) <= set(RAISE_SIZES_BY_STAGE[stage][1])
        assert first_in <= set(RAISE_SIZES_BY_STAGE[stage][0])


class TestPassiveGates:

    def test_call_is_absent_exactly_where_the_gate_says_so(self):
        # Pre-flop root: the gate governs, because the actor owes the blind.
        env = _env()
        stage, level = "pre_flop", raise_level("pre_flop", 0)
        allowed = CALL_ALLOWED_BY_STAGE[stage][level]
        assert ("call" in _legal(env)) is allowed

    def test_all_in_is_absent_exactly_where_the_gate_says_so(self):
        env = _env()
        stage, level = "pre_flop", raise_level("pre_flop", 0)
        allowed = ALL_IN_ALLOWED_BY_STAGE[stage][level]
        # Deep stacks, so a shove is mechanically possible and only the gate
        # can remove it.
        assert ("all_in" in _legal(env)) is allowed

    def test_a_free_check_is_never_gated(self):
        # Nothing owed => "call" (a check) is offered whatever the gate says.
        env = advance_to_round(_env(), 1)
        assert env.chips_to_add("call") == 0
        assert "call" in _legal(env)

    def test_call_for_less_survives_a_closed_gate(self):
        # Facing a bet bigger than the stack, the response is spelled "all_in"
        # and is a CALL — the all-in gate must not remove it or the short stack
        # could only fold.
        env = PokerEnv(players=[Player(0, 10000), Player(1, 300)])
        while "all_in" not in _legal(env) and not env.is_terminal:
            raises = [a for a in _legal(env) if a.startswith("raise:")]
            env.step_in_place(raises[0] if raises else passive_action(env))
        env.step_in_place("all_in")
        if not env.is_terminal:
            assert "all_in" in _legal(env) or "call" in _legal(env)

    def test_a_player_with_chips_is_never_left_only_folding(self):
        # The gates plus the stack filters must never produce a forced fold.
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


class TestOverlayCoversTheGatedActions:
    """An opponent is not bound by our abstraction: a gated ``call`` /
    ``all_in`` must still be injectable, or an observed limp would be remapped
    to a fold."""

    def test_gated_call_can_be_injected_and_played(self):
        env = _env()
        if "call" in _legal(env):
            pytest.skip("pre-flop level 0 offers a call in this abstraction")
        pot_before = env.pot_size
        assert env.inject_action("call") is True
        assert "call" in _legal(env)
        assert env.has_overlay_at_current_node
        env.step_in_place("call")
        assert env.pot_size > pot_before

    def test_gated_all_in_can_be_injected_and_played(self):
        env = _env()
        if "all_in" in _legal(env):
            pytest.skip("pre-flop level 0 offers a shove in this abstraction")
        assert env.inject_action("all_in") is True
        assert "all_in" in _legal(env)
        env.step_in_place("all_in")
        assert env.players[0].n_chips == 0

    def test_fold_is_never_injected(self):
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
