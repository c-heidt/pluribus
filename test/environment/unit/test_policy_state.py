"""Tests for ``PokerEnv.policy_state`` and ``policy_state_for``.

:class:`PolicyState` is the env-side value object that decouples
:class:`poker_ai.search.policy.Policy.strategy` from a live env
reference.  These tests pin the two env accessors:

- ``env.policy_state`` — bundles fields for the current actor at the
  current state.
- ``env.policy_state_for(combo)`` — same bundle but ``info_set`` is
  computed under a hypothetical hole; reads *no* seat's actual
  cards (the leak-free path for ``sigma_for_combo``).
"""

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv, PolicyState


def _env(n_players: int = 2) -> PokerEnv:
    return PokerEnv(players=[Player(i, 10000) for i in range(n_players)])


def _stub_lut(env: PokerEnv) -> None:
    """Populate ``card_info_lut`` with one cluster per combo per stage.

    Maps every ``(hole + community)`` lookup an info_set query might
    perform to a distinct integer cluster, so ``info_set`` doesn't
    crash and different holes yield different clusters.
    """
    lut = {}
    for stage in ("pre_flop", "flop", "turn", "river"):
        lut[stage] = {}
    # Pre-flop: just the hole.
    for i in range(env.n_combos):
        combo = tuple(sorted(int(c) for c in env.combo_cards[i]))
        lut["pre_flop"][combo] = i + 1
    env.card_info_lut = lut


class TestPolicyState:

    def test_returns_dataclass_instance(self):
        env = _env()
        _stub_lut(env)
        ps = env.policy_state
        assert isinstance(ps, PolicyState)

    def test_fields_present_and_typed(self):
        env = _env()
        _stub_lut(env)
        ps = env.policy_state
        assert isinstance(ps.player_i, int)
        assert isinstance(ps.betting_round, int)
        assert isinstance(ps.info_set, bytes)
        assert isinstance(ps.valid_mask, np.ndarray)
        assert ps.valid_mask.dtype == bool
        assert isinstance(ps.legal_actions, tuple)
        for a in ps.legal_actions:
            assert isinstance(a, str)

    def test_player_i_matches_env(self):
        env = _env()
        _stub_lut(env)
        assert env.policy_state.player_i == env.player_i

    def test_betting_round_matches_env(self):
        env = _env()
        _stub_lut(env)
        assert env.policy_state.betting_round == env.betting_round

    def test_info_set_matches_env(self):
        env = _env()
        _stub_lut(env)
        assert env.policy_state.info_set == env.info_set

    def test_legal_actions_filters_none(self):
        env = _env()
        _stub_lut(env)
        legal = env.policy_state.legal_actions
        assert None not in legal

    def test_valid_mask_is_readonly(self):
        env = _env()
        _stub_lut(env)
        mask = env.policy_state.valid_mask
        with pytest.raises(ValueError):
            mask[0] = not mask[0]

    def test_policy_state_immutable(self):
        env = _env()
        _stub_lut(env)
        ps = env.policy_state
        import dataclasses
        with pytest.raises(dataclasses.FrozenInstanceError):
            ps.betting_round = 7  # type: ignore[misc]


class TestPolicyStateFor:

    def test_info_set_reflects_supplied_combo(self):
        env = _env()
        _stub_lut(env)
        actual = tuple(sorted(int(c) for c in env.current_player._cards))
        # Find a different combo.
        other = None
        for i in range(env.n_combos):
            combo = tuple(sorted(int(c) for c in env.combo_cards[i]))
            if combo != actual:
                other = combo
                break
        assert other is not None
        ps_actual = env.policy_state
        ps_other = env.policy_state_for(other)
        assert ps_actual.info_set != ps_other.info_set

    def test_does_not_read_other_seats_cards(self):
        # Regression for the leak motivation: policy_state_for must
        # produce identical results even if other seats' _cards are
        # mutated to sentinel values.
        env = _env(n_players=3)
        _stub_lut(env)
        combo = tuple(sorted(int(c) for c in env.combo_cards[0]))
        ps_before = env.policy_state_for(combo)
        # Corrupt seat 1 and seat 2 holes.
        env.players[1]._cards = (-1, -2)
        env.players[2]._cards = (-3, -4)
        ps_after = env.policy_state_for(combo)
        assert ps_before.info_set == ps_after.info_set
        assert ps_before.betting_round == ps_after.betting_round
        np.testing.assert_array_equal(
            ps_before.valid_mask, ps_after.valid_mask
        )
        assert ps_before.legal_actions == ps_after.legal_actions

    def test_does_not_read_current_actor_cards(self):
        # And by extension, must not read the current actor's own cards
        # either — combo is the sole source for the info_set lookup.
        env = _env()
        _stub_lut(env)
        combo = tuple(sorted(int(c) for c in env.combo_cards[0]))
        ps_before = env.policy_state_for(combo)
        env.players[env.player_i]._cards = (-9, -10)
        ps_after = env.policy_state_for(combo)
        assert ps_before.info_set == ps_after.info_set

    def test_other_fields_identical_to_policy_state(self):
        # policy_state_for differs from policy_state only in info_set
        # (when the actual cards differ from `combo`).  Other fields
        # are read from public state and must match.
        env = _env()
        _stub_lut(env)
        actual = tuple(sorted(int(c) for c in env.current_player._cards))
        ps = env.policy_state
        ps_for = env.policy_state_for(actual)
        assert ps.betting_round == ps_for.betting_round
        assert ps.legal_actions == ps_for.legal_actions
        np.testing.assert_array_equal(ps.valid_mask, ps_for.valid_mask)
        # Same combo as the actual hole → identical info_set too.
        assert ps.info_set == ps_for.info_set


class TestPolicyPublicFields:
    """The ``policy_public_fields`` / ``public=`` decomposition must be a pure
    factoring: passing the precomputed public part can never change the result."""

    def test_public_fields_are_combo_independent(self):
        # The hoist's whole premise: the public fields don't depend on any combo.
        env = _env()
        _stub_lut(env)
        pf = env.policy_public_fields()
        assert pf.player_i == env.player_i
        assert pf.betting_round == env.betting_round
        assert pf.legal_actions == tuple(
            a for a in env.legal_actions if a is not None
        )
        np.testing.assert_array_equal(pf.valid_mask, env.get_valid_mask())
        # Returned mask is immutable (shared safely across the per-combo sweep).
        assert pf.valid_mask.flags.writeable is False

    @pytest.mark.parametrize("for_blueprint", [False, True])
    def test_public_path_matches_recompute_for_every_combo(self, for_blueprint):
        # policy_state_for(combo, public=fields) must equal policy_state_for(combo)
        # field-by-field for every combo — the decomposition cannot diverge.
        env = _env()
        _stub_lut(env)
        public = env.policy_public_fields()
        for i in range(env.n_combos):
            combo = tuple(sorted(int(c) for c in env.combo_cards[i]))
            baseline = env.policy_state_for(combo, for_blueprint=for_blueprint)
            hoisted = env.policy_state_for(
                combo, for_blueprint=for_blueprint, public=public
            )
            assert hoisted.player_i == baseline.player_i
            assert hoisted.betting_round == baseline.betting_round
            assert hoisted.info_set == baseline.info_set
            assert hoisted.legal_actions == baseline.legal_actions
            np.testing.assert_array_equal(hoisted.valid_mask, baseline.valid_mask)
