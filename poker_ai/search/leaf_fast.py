"""FastState-driven leaf continuation-value rollout (search Cython core, Phase 4b).

Ports the depth-limit leaf rollout (:func:`poker_ai.search.leaf.continuation_value`,
the ~74% cost of an MCCFR iteration — a per-leaf ``with_hole_cards`` deepcopy plus a
``PokerEnv`` make/undo/``policy_state_for`` walk per rollout) onto the compiled
:class:`poker_ai._core._state.FastState` betting engine.  The FastState is built
**once** per leaf and make/undo-walked per rollout (the per-leaf deepcopy collapses),
and terminals settle in-core via ``FastState.payout`` / ``FastState.runout_equity``
(Phase 4a).

**Equilibrium-gated, not byte-identical.**  The rollout draws a fresh 5-card board
per call (the undealt-deck runout), and the core draws its *own* sample rather than
reproducing ``PokerEnv``'s shuffle byte-stream — exactly as ``_traverse.pyx`` declined
RNG parity for the blueprint core.  A byte-identical action-line differential is
therefore infeasible: the fast path consumes ``ctx.rng`` for the per-rollout board
draw while the Python reference consumes it inside ``deck.shuffle_undealt()``, so the
two action-sampling streams diverge even under one seed.  Instead the value is gated
**statistically and componentwise**: (1) the fast rollout is an unbiased estimator of
the Python rollout's per-seat value (``test_leaf_fast::test_rollout_unbiased_vs_python``
— grand means agree within the combined standard error); (2) the two scoring building
blocks it adds are proven exactly — ``runout_equity`` is byte-identical to ``PokerEnv``
(``test_faststate_runout``), and the drawn-board ``PolicyState`` (clusters via
:meth:`FastState.refresh_clusters` + info-set/valid-mask via :func:`_policy_state`)
matches ``PokerEnv.policy_state`` at the frontier **and across turn/river rollout
nodes** (``test_leaf_fast::test_policy_state_matches_env_across_streets``).  The
**policy stays a Python callback**; the ``PolicyState`` is rebuilt from FastState per
decision with clusters recomputed for the drawn board, so a real ``BlueprintPolicy``
reads a correct info-set — only the in-core policy-fleet read is deferred (Phase 4c
crux).

Falls back to the pure-Python :func:`continuation_value` when the core is unavailable,
the frontier carries off-tree injections (its byte-code history / overlay actions are
not representable — the Phase-3 crux-#1 guard), or the frontier is already terminal.
"""

from __future__ import annotations

import logging
from typing import List, Mapping, Tuple

import numpy as np

from environment.action_space import ACTION_TO_IDX, CANONICAL_ACTIONS
from environment.poker_env import PolicyState
from poker_ai.blueprint.tree_utils import sample_index
from poker_ai.search.context import SubgameContext
from poker_ai.search.leaf import continuation_value
from poker_ai.search.policy import BiasClass, BlueprintPolicy

logger = logging.getLogger(__name__)


def _resolve_core_policy(policies, profile):
    """The shared ``BlueprintPolicy`` backing every acting seat's bias IF it exposes
    a built in-core reader — Phase 4c; else ``None`` (→ the Python callback).

    Requires ONE ``BlueprintPolicy`` instance across the biases in ``profile`` so a
    single ``CoreTables`` (queried per-decision with the seat's bias) serves them all
    — exactly how ``build_blueprint_session`` wires the fleet.  Any other fleet
    (``UniformPolicy``, a heterogeneous set, or a blueprint opened without the shm
    cache) yields ``None`` and the rollout keeps ``_policy_state`` + ``.strategy``.
    """
    try:
        used = {policies[b] for b in profile.values()}
    except (KeyError, TypeError):
        return None
    if len(used) != 1:
        return None
    p = next(iter(used))
    if not isinstance(p, BlueprintPolicy):
        return None
    return p if p._ensure_core() is not None else None


def _core_state():
    """Return a configured ``FastState`` class, or ``None`` if unavailable."""
    try:
        from poker_ai._core import CORE_AVAILABLE
        if not CORE_AVAILABLE:
            return None
        from poker_ai._core import _state as _cystate
        if not _cystate.is_configured():
            from environment.poker_env import (
                _ACTION_BYTE, _STAGE_ID, RAISE_SIZES_BY_STAGE, MAX_RAISES_PER_ROUND,
            )
            _cystate.configure(
                _STAGE_ID, _ACTION_BYTE, RAISE_SIZES_BY_STAGE, MAX_RAISES_PER_ROUND
            )
        return _cystate.FastState
    except Exception:
        return None


def _policy_state(fs, legal: Tuple[str, ...]) -> PolicyState:
    """Build a ``PolicyState`` from a FastState node (canonical-history rollout).

    ``info_set`` uses ``FastState.info_set`` with clusters recomputed for the drawn
    board, so it equals ``PokerEnv._blueprint_info_set`` on the canonical histories a
    rollout produces; ``valid_mask`` mirrors ``PokerEnv.get_valid_mask``.
    """
    r = fs.betting_round
    legal_set = set(legal)
    canonical = CANONICAL_ACTIONS[r]
    valid_mask = np.array([a in legal_set for a in canonical], dtype=bool)
    return PolicyState(
        player_i=fs.player_i,
        betting_round=r,
        info_set=fs.info_set(),
        valid_mask=valid_mask,
        legal_actions=tuple(legal),
    )


def continuation_value_fast(
    frontier_env,
    profile: Mapping[int, BiasClass],
    ctx: SubgameContext,
    runout_cache: "dict | None" = None,
) -> np.ndarray:
    """FastState leaf rollout; drop-in for :func:`continuation_value` (see module doc).

    Falls back to the pure-Python reference on any unrepresentable frontier so the
    result is always defined; the compiled path only ever *accelerates*.
    """
    n = frontier_env.n_players
    cfg = ctx.leaf
    # Fallbacks (crux-#1 overlay guard + defensive): keep the Python path.
    FastState = _core_state()
    if (
        FastState is None
        or frontier_env.is_terminal
        or cfg.n_rollouts <= 0
        or getattr(frontier_env, "_extra_legal_actions", None)
    ):
        return continuation_value(frontier_env, profile, ctx, runout_cache)

    try:
        fs = FastState.from_poker_env(frontier_env)
    except Exception:
        # Off-tree byte-code history not representable — use the Python rollout.
        return continuation_value(frontier_env, profile, ctx, runout_cache)

    rng = ctx.rng
    use_equity = cfg.use_decision_free_equity
    lut = frontier_env.card_info_lut

    prefix = [int(c) for c in frontier_env.community_cards]
    k = 5 - len(prefix)
    holes = [
        tuple(int(c) for c in frontier_env.players[i].cards) for i in range(n)
    ]
    holes_key = tuple(holes)
    used = set(prefix)
    for h in holes:
        used.update(h)
    full_deck = [int(c) for c in np.unique(frontier_env.combo_cards)]
    undealt = np.array([c for c in full_deck if c not in used], dtype=np.int64)

    # Phase 4c: serve the blueprint policy read in-core (skip the per-decision
    # PolicyState build + Python ``strategy``) when the fleet is a cache-backed
    # BlueprintPolicy; else ``None`` → the Python callback below.  ``legal`` is
    # canonical-only here (overlay frontiers already fell back above), so the
    # canonical→legal map is a clean gather.
    core_policy = _resolve_core_policy(cfg.policies, profile)

    cache = runout_cache if runout_cache is not None else {}
    accum = np.zeros(n, dtype=np.float64)
    try:
        for _r in range(cfg.n_rollouts):
            completion = (
                list(rng.choice(undealt, size=k, replace=False)) if k > 0 else []
            )
            board = prefix + [int(c) for c in completion]
            fs.set_board(board)
            fs.refresh_clusters(lut)
            tokens: List = []
            while not fs.is_terminal:
                seat = fs.player_i
                if seat not in profile:
                    raise ValueError(
                        f"continuation_value_fast: profile is missing acting seat "
                        f"{seat}; it must cover every seat that can act."
                    )
                legal = [a for a in fs.legal_actions() if a is not None]
                bias = profile[seat]
                if core_policy is not None:
                    br = fs.betting_round
                    legal_cols = np.array(
                        [ACTION_TO_IDX[br][a] for a in legal], dtype=np.int64
                    )
                    probs = core_policy.core_sigma(br, fs.info_set(), legal_cols, bias)
                else:
                    probs = cfg.policies[bias].strategy(
                        _policy_state(fs, legal), bias=bias
                    )
                idx = sample_index(rng, probs)
                tokens.append(fs.step_in_place(legal[idx]))
            if use_equity and fs.is_decision_free:
                key = (holes_key, fs.runout_key)
                eq = cache.get(key)
                if eq is None:
                    eq = fs.runout_equity(rng=rng)
                    cache[key] = eq
                for i in range(n):
                    accum[i] += eq[i]
            else:
                pay = fs.payout()
                for i in range(n):
                    accum[i] += float(pay[i])
            for tok in reversed(tokens):
                fs.undo(tok)
    except Exception:
        # Never let an unexpected core-path error break a search — fall back and
        # surface it (visible, not silent) so it can be fixed rather than masked.
        logger.warning(
            "continuation_value_fast fell back to the Python rollout after an "
            "unexpected error", exc_info=True,
        )
        return continuation_value(frontier_env, profile, ctx, runout_cache)

    return accum / cfg.n_rollouts
