"""Shared test builders for the search suite (env / context / leaf-fleet stand-ins).

These small constructors are used across ``test_solver``, ``test_agent``,
``test_equilibrium_oracle``, the functional core/vector gates, and the evaluation
tests.  They live here (rather than in ``test_solver``, which historically owned
them) so no test module imports fixtures from another test module.  Per-module
solver configs (``_cfg``) stay in their own modules — they differ by module.
"""

import collections

import numpy as np

from environment.player import Player
from environment.poker_env import PokerEnv
from poker_ai.search.context import SubgameContext
from poker_ai.search.leaf import LeafConfig
from poker_ai.search.mccfr import _BIAS_CLASSES
from poker_ai.search.policy import Policy


class UniformPolicy(Policy):
    """Uniform over the legal actions (a self-contained leaf fleet stand-in)."""

    def strategy(self, state, bias="none"):
        n = len(state.legal_actions)
        return np.full(n, 1.0 / n, dtype=np.float32) if n else np.array([], np.float32)


def _policies():
    return {c: UniformPolicy() for c in _BIAS_CLASSES}


def _stub_lut(env: PokerEnv) -> None:
    env.card_info_lut = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0)
    )


# --------------------------------------------------------------------------- #
# Real-LUT (multi-cluster) fixture
# --------------------------------------------------------------------------- #
# ``_stub_lut`` maps every hand to cluster 0, so every future street collapses to
# ``n_rows == 1``.  That makes a whole bug class INVISIBLE: anything that depends on
# the cluster→dense-row mapping (``ClusterMapper._universe`` / ``cof``) is trivially
# correct when there is only one row.  A board-stale model-row cache, for instance,
# passed the entire stub-LUT suite unchanged.
#
# ``data/20cards_exact`` is a real 20-card LUT (ranks 10-14, 190 combos) with 25
# preflop / 50 flop / 50 turn / 45 river clusters, giving ~24-37 dense rows at future
# streets.  Use it for anything cluster-row dependent.

import functools
import os
from pathlib import Path


@functools.lru_cache(maxsize=1)
def _real_lut():
    """The real 20-card ``card_info_lut``, loaded once per session.

    Resolves the same way ``test/conftest.py`` does (``PLURIBUS_LUT_PATH``, else
    ``data/20cards_exact``).  Tests that call this must carry
    ``@pytest.mark.requires_lut`` so the root conftest skips them when the LUT is
    absent rather than erroring here.
    """
    import joblib
    import pytest
    root = Path(__file__).parent.parent.parent
    lut_dir = Path(os.environ.get("PLURIBUS_LUT_PATH", str(root / "data" / "20cards_exact")))
    path = lut_dir / "card_info_lut.joblib"
    if not path.exists():
        pytest.skip(f"LUT not found at {path}")
    return joblib.load(str(path))


def _real_lut_env(target_round: int, stacks=(10000, 10000), seed=0) -> PokerEnv:
    """Heads-up 20-card env at ``target_round`` under the REAL LUT.

    Unlike :func:`_stub_lut`, future streets have many distinct clusters, so
    ``ClusterMapper.n_rows`` is >> 1 and the cluster→row path is exercised for real
    (flop root: 24 turn / 37 river rows; turn root: 29 river rows).
    """
    from environment.poker_env import new_game
    np.random.seed(seed)
    env = new_game(2, card_info_lut=_real_lut(), initial_chips=stacks[0])
    guard = 0
    while env.betting_round < target_round and not env.is_terminal and guard < 60:
        legal = [a for a in env.legal_actions if a is not None]
        env.step_in_place("call" if "call" in legal else legal[0])
        guard += 1
    assert env.betting_round == target_round, (
        f"could not reach round {target_round} (got {env.betting_round})"
    )
    return env


def _flop_env(low=11, high=14, stacks=(200, 200), seed=0) -> PokerEnv:
    """Heads-up env advanced to the flop over a small deck (exact runouts)."""
    np.random.seed(seed)
    env = PokerEnv(
        players=[Player(i, s) for i, s in enumerate(stacks)],
        low_card_rank=low,
        high_card_rank=high,
    )
    _stub_lut(env)
    guard = 0
    while env.betting_round < 1 and not env.is_terminal and guard < 20:
        env.step_in_place("call" if "call" in env.legal_actions else "check")
        guard += 1
    return env


def _preflop_env(low=11, high=14, stacks=(200, 200), seed=0) -> PokerEnv:
    """Heads-up small-deck env at the preflop root (``street_at_root == 0``)."""
    np.random.seed(seed)
    env = PokerEnv(
        players=[Player(i, s) for i, s in enumerate(stacks)],
        low_card_rank=low,
        high_card_rank=high,
    )
    _stub_lut(env)
    return env


def _advance_to(env: PokerEnv, target_round: int) -> PokerEnv:
    """Walk a heads-up env (calls/checks only) to ``target_round``."""
    guard = 0
    while env.betting_round < target_round and not env.is_terminal and guard < 60:
        env.step_in_place("call" if "call" in env.legal_actions else "check")
        guard += 1
    return env


def _late_env(target_round, low=11, high=14, stacks=(200, 200), seed=0) -> PokerEnv:
    """Heads-up small-deck env advanced to ``target_round`` (2=turn, 3=river)."""
    np.random.seed(seed)
    env = PokerEnv(
        players=[Player(i, s) for i, s in enumerate(stacks)],
        low_card_rank=low,
        high_card_rank=high,
    )
    _stub_lut(env)
    return _advance_to(env, target_round)


def _ctx(env, *, n_rollouts=2, seed=0, ranges=None, folded=None) -> SubgameContext:
    if ranges is None:
        ranges = {s: np.ones(env.n_combos, np.float32) / env.n_combos for s in range(2)}
    leaf = LeafConfig(policies=_policies(), n_rollouts=n_rollouts)
    return SubgameContext.from_runtime(
        env=env,
        my_seat=0,
        my_hole=tuple(int(c) for c in env.players[0].cards),
        ranges=ranges,
        folded_ranges=folded or {},
        leaf=leaf,
        rng=np.random.default_rng(seed),
    )
