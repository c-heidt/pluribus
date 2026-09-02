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


# --------------------------------------------------------------------------- #
# Real-LUT (multi-cluster) fixture
# --------------------------------------------------------------------------- #
# ``install_cluster_lut`` (test.lut_helpers) is the DATA-FREE multi-cluster stand-in —
# it replaced a single-cluster stub that mapped every hand to cluster 0, collapsing every
# future street to ``n_rows == 1`` and making a whole bug class invisible.  Use the real
# LUT below when a test needs true cluster semantics rather than merely more than one row.
#
# ``data/20cards_exact`` is a real 20-card LUT (ranks 10-14, 190 combos) with 25
# preflop / 50 flop / 50 turn / 45 river clusters, giving ~24-37 dense rows at future
# streets.  Use it for anything cluster-row dependent.

import functools
import os
from pathlib import Path
from test.abstraction_helpers import passive_action
from test.lut_helpers import install_cluster_lut


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


# Default heads-up stacks for the small-deck helpers, in chips against the
# env's 50/100 blinds.  The abstraction has no pre-flop limp, so a stack must
# at least cover the ~2.6 bb open on top of the blind or every hand would end
# pre-flop as a fold-or-shove and no test would ever see a post-flop street.
# 10 bb clears that with room for post-flop betting while keeping the subgame
# trees (and so these tests) small.
_HU_STACKS = (1000, 1000)

# For fixtures whose test ENUMERATES the subgame exhaustively (the equilibrium
# oracle's best-response walks).  Their cost grows steeply with depth — every
# extra chip that lets one more raise size fit multiplies the tree — so these
# take the shallowest stack that still clears the pre-flop open: 4 bb.  Measured
# on the current grid this yields 8 turn / 12 flop betting nodes, exactly what
# the pre-abstraction fixtures had, so the oracle tests keep their old cost.
_HU_STACKS_SHALLOW = (400, 400)


def _real_lut_env(target_round: int, stacks=(10000, 10000), seed=0) -> PokerEnv:
    """Heads-up 20-card env at ``target_round`` under the REAL LUT.

    Unlike :func:`~test.lut_helpers.install_cluster_lut`'s synthetic clusters, these are
    the LUT's real ones, so the cluster→row path is exercised against true abstraction
    semantics (flop root: 24 turn / 37 river rows; turn root: 29 river rows).
    """
    from environment.poker_env import new_game
    np.random.seed(seed)
    env = new_game(2, card_info_lut=_real_lut(), initial_chips=stacks[0])
    guard = 0
    while not env.is_terminal and env.betting_round < target_round and guard < 60:
        env.step_in_place(passive_action(env))
        guard += 1
    assert env.betting_round == target_round, (
        f"could not reach round {target_round} (got {env.betting_round})"
    )
    return env


def _flop_env(low=11, high=14, stacks=_HU_STACKS, seed=0) -> PokerEnv:
    """Heads-up env advanced to the flop over a small deck (exact runouts)."""
    np.random.seed(seed)
    env = PokerEnv(
        players=[Player(i, s) for i, s in enumerate(stacks)],
        low_card_rank=low,
        high_card_rank=high,
    )
    install_cluster_lut(env)
    guard = 0
    while not env.is_terminal and env.betting_round < 1 and guard < 20:
        env.step_in_place(passive_action(env))
        guard += 1
    return env


def _preflop_env(low=11, high=14, stacks=_HU_STACKS, seed=0) -> PokerEnv:
    """Heads-up small-deck env at the preflop root (``street_at_root == 0``)."""
    np.random.seed(seed)
    env = PokerEnv(
        players=[Player(i, s) for i, s in enumerate(stacks)],
        low_card_rank=low,
        high_card_rank=high,
    )
    install_cluster_lut(env)
    return env


def _advance_to(env: PokerEnv, target_round: int) -> PokerEnv:
    """Walk a heads-up env (calls/checks only) to ``target_round``."""
    guard = 0
    while not env.is_terminal and env.betting_round < target_round and guard < 60:
        env.step_in_place(passive_action(env))
        guard += 1
    return env


def _late_env(target_round, low=11, high=14, stacks=_HU_STACKS, seed=0) -> PokerEnv:
    """Heads-up small-deck env advanced to ``target_round`` (2=turn, 3=river)."""
    np.random.seed(seed)
    env = PokerEnv(
        players=[Player(i, s) for i, s in enumerate(stacks)],
        low_card_rank=low,
        high_card_rank=high,
    )
    install_cluster_lut(env)
    return _advance_to(env, target_round)


def _ctx(env, *, seed=0, ranges=None, folded=None) -> SubgameContext:
    if ranges is None:
        ranges = {s: np.ones(env.n_combos, np.float32) / env.n_combos for s in range(2)}
    leaf = LeafConfig(policies=_policies())
    return SubgameContext.from_runtime(
        env=env,
        my_seat=0,
        my_hole=tuple(int(c) for c in env.players[0].cards),
        ranges=ranges,
        folded_ranges=folded or {},
        leaf=leaf,
        rng=np.random.default_rng(seed),
    )
