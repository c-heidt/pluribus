"""Shared vector-form CFR primitives for both search regimes (§6.5).

The heads-up vector regime (:mod:`poker_ai.search.vector`) and the
traverser-vectorized MCCFR walk (:mod:`poker_ai.search.mccfr`) run the **same**
per-combo CFR arithmetic on a ``(n_rows, width)`` matrix — regret matching, a
per-combo ``sigma`` gather on future (cluster) streets, freezing the bot's
actual-hand row (§5), and the ``cv``/``v``/``delta``/``strat_delta`` update — and
differ only in how opponents are handled (range-expanded vs sampled) and how
terminals settle.  That shared arithmetic lives here so neither regime imports the
other (both depend on this neutral module and on :mod:`cluster_maps`).

``regret_match_matrix`` and its compiled-core rebind moved here verbatim from
``vector`` (this is Python module organization only — the compiled kernel is
unchanged); ``vector`` re-exports it for backward compatibility.
"""

from __future__ import annotations

import numpy as np


def regret_match_matrix(regret: np.ndarray) -> np.ndarray:
    """Row-wise regret matching over the **last** axis of a regret tensor.

    Vectorised counterpart of
    :func:`poker_ai.blueprint.tree_utils.calculate_strategy_from_row`: each row's
    strategy is proportional to its positive cumulative regret, falling back to
    uniform over the ``width`` actions when a row has no positive regret.

    Action ``width`` is always the last axis, so this serves both a root-street
    ``(n_combos, width)`` node and a future-street ``(n_clusters, width)`` node
    (§6.5) without reshaping.
    """
    pos = np.maximum(regret, 0.0)
    total = pos.sum(axis=-1, keepdims=True)
    width = regret.shape[-1]
    safe = np.where(total > 0.0, total, 1.0)
    return np.where(total > 0.0, pos / safe, 1.0 / width)


# Compiled-core wiring: when built AND enabled (``PLURIBUS_CORE_KERNELS`` includes
# ``regret_match_matrix``), swap the batched regret-matcher for its byte-identical
# Cython kernel.  Callers use the module global ``regret_match_matrix`` so the
# rebind is transparent; the pure-Python reference is kept as
# ``_regret_match_matrix_py`` (the oracle the parity tests compare against).
_regret_match_matrix_py = regret_match_matrix
try:
    from poker_ai._core import CORE_AVAILABLE as _CORE_AVAILABLE
    from poker_ai._core.flags import kernel_enabled as _kernel_enabled

    if _CORE_AVAILABLE and _kernel_enabled("regret_match_matrix"):
        from poker_ai._core._regret import (
            calculate_strategy_matrix as regret_match_matrix,
        )
except ImportError:
    pass


def node_sigma(state, pk, legal, actor, is_root, n_rows, row_space, cof):
    """Register the node and build its per-combo strategy matrix.

    Ensures the ``(n_rows, width)`` ``vregret``/``vstrat`` matrices exist, regret-
    matches the regret matrix, and gathers each combo's row: identity for a
    root-street node, a cluster gather (``cof`` maps combo→dense cluster row, ``-1``
    on infeasible combos → row 0, harmless since their reach is zeroed) otherwise.

    Returns ``(sigma, regret, strat)`` where ``sigma`` is ``(n_combos, width)`` and
    ``regret``/``strat`` are the node's stored ``(n_rows, width)`` matrices.
    """
    state.ensure_vnode(pk, legal, actor, n_rows, row_space)
    regret = state.vregret[pk]
    strat = state.vstrat[pk]
    sigma_rows = regret_match_matrix(regret)
    sigma = sigma_rows if is_root else sigma_rows[np.where(cof >= 0, cof, 0)]
    return sigma, regret, strat


def freeze_combo(state, pk, sigma, is_root, actor, my_seat, my_combo):
    """Substitute the bot's pinned actual-hand row into ``sigma`` in place (§5).

    Frozen rows are only ever set on the bot's current (root-street) decisions, so
    this fires on combo nodes only — for whichever seat is acting, so a sampled
    opponent that *is* the frozen bot also plays the pinned strategy.  Returns the
    frozen combo index if the substitution fired (so the update can skip its
    regret/strategy accrual), else ``None``.
    """
    if (
        is_root
        and actor == my_seat
        and my_combo is not None
        and (pk, my_combo) in state.frozen
    ):
        sigma[my_combo] = state.frozen[(pk, my_combo)]
        return my_combo
    return None


def traverser_update(regret, strat, sigma, child_vs, pi_p, frozen_combo, scatter):
    """The per-combo CFR update at a traverser node; returns ``v`` ``(n_combos,)``.

    ``child_vs`` is ``(width, n_combos)`` (one row per action).  Computes the
    per-combo counterfactual ``cv``, node value ``v``, regret ``delta`` and
    own-reach-weighted ``strat_delta``, then writes them into the node's matrices.
    ``scatter`` is ``None`` for a root (combo-keyed) node — rows are combos, added
    directly — or a callable ``scatter(table, per_combo)`` for a cluster-keyed node
    (segment-sums each combo's delta into its cluster row).  ``frozen_combo`` (from
    :func:`freeze_combo`), when not ``None``, zeroes that combo's accrual (the
    pinned actual hand neither regrets nor averages).
    """
    cv = np.moveaxis(child_vs, 0, -1)             # (n_combos, width)
    v = (sigma * cv).sum(axis=-1)                 # (n_combos,)
    delta = cv - v[:, None]                       # regret: v_a - v
    strat_delta = pi_p[:, None] * sigma           # strat-sum: own reach * sigma
    if frozen_combo is not None:
        delta[frozen_combo] = 0.0
        strat_delta[frozen_combo] = 0.0
    if scatter is None:
        regret += delta                           # in-place: writes the store
        strat += strat_delta
    else:
        scatter(regret, delta)
        scatter(strat, strat_delta)
    return v
