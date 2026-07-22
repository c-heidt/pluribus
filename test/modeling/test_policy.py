"""Tests for ModelPolicy (poker_ai/modeling/policy.py, §4.3).

The σ̂-backed leaf-continuation Policy: ``bias="none"`` passes σ̂ through, and the
biased variants reproduce exactly the ``Policy._reweight_bias`` transform
:class:`BlueprintPolicy` applies to its own row (§8.4), so it drops into the leaf
fleet unchanged.
"""

import numpy as np

from environment.poker_env import PolicyState
from poker_ai.modeling.policy import ModelPolicy
from poker_ai.search.policy import BlueprintPolicy, Policy


_LEGAL = ("fold", "call", "raise:0.5", "all_in")


class FakeModel:
    """An OpponentModel stand-in returning a fixed σ̂ aligned to the legal set."""

    def __init__(self, row):
        self._row = np.asarray(row, dtype=np.float32)

    def strategy(self, state):
        return self._row

    def confidence(self, state):
        return 0.5


def _state():
    return PolicyState(
        player_i=0,
        betting_round=3,
        info_set=b"k",
        valid_mask=np.ones(len(_LEGAL), dtype=bool),
        legal_actions=_LEGAL,
    )


class TestModelPolicy:

    def test_is_a_policy(self):
        assert isinstance(ModelPolicy(FakeModel([1.0])), Policy)

    def test_none_bias_passes_sigma_through(self):
        row = [0.25, 0.25, 0.25, 0.25]
        out = ModelPolicy(FakeModel(row)).strategy(_state(), "none")
        assert np.allclose(out, row)

    def test_bias_matches_reweight_helper_on_the_same_row(self):
        row = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        mp = ModelPolicy(FakeModel(row), bias_multiplier=5.0)
        for bias in ("fold", "call", "raise"):
            # The reference transform BlueprintPolicy applies to a row: the shared
            # Policy helpers on the *same* legal-aligned row.
            mask = Policy._bias_mask(list(_LEGAL), bias)
            ref = Policy._reweight_bias(row, mask, 5.0)
            assert np.allclose(mp.strategy(_state(), bias), ref)

    def test_raise_bias_lifts_raise_class(self):
        # raise class = {raise:0.5, all_in}; a raise bias multiplies both by 5.
        row = np.array([0.4, 0.4, 0.1, 0.1], dtype=np.float32)
        out = ModelPolicy(FakeModel(row), bias_multiplier=5.0).strategy(_state(), "raise")
        # raise-class mass share must rise vs the base row (0.2 → higher).
        assert out[2] + out[3] > 0.2
        assert np.isclose(out.sum(), 1.0, atol=1e-6)

    def test_bias_class_identification_uses_same_prefixes_as_blueprint(self):
        # ModelPolicy and BlueprintPolicy build the same class mask from a token list.
        for bias in ("none", "fold", "call", "raise"):
            assert np.array_equal(
                ModelPolicy._bias_mask(list(_LEGAL), bias),
                BlueprintPolicy._bias_mask(list(_LEGAL), bias),
            )
