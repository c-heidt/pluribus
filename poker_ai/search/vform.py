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


def clamp_sigma(sigma, m_rows, c_rows, gof):
    """In-place ``σ ← σ + c·(σ̂ − σ)`` (the DBR mixture blend); returns ``sigma``.

    Pure-Python reference for the ``_clamp`` kernel (P2).  Uses ``c·σ̂ + (1−c)·σ``
    verbatim: the cheaper ``σ + c·(σ̂ − σ)`` is NOT exact at ``c = 1``, and
    naive best response is ``c ≡ 1``.  In-place is safe: ``sigma`` is always a fresh
    array from the caller (``regret_match_matrix`` at a root node, a fancy-index
    copy at a clustered one).
    """
    if gof is None:
        m, c = m_rows, c_rows
    else:
        m, c = m_rows[gof], c_rows[gof]
    np.multiply(sigma, 1.0 - c, out=sigma)
    sigma += c * m
    return sigma


_clamp_sigma_py = clamp_sigma
try:
    from poker_ai._core import CORE_AVAILABLE as _CORE_AVAILABLE2
    from poker_ai._core.flags import kernel_enabled as _kernel_enabled2

    if _CORE_AVAILABLE2 and _kernel_enabled2("clamp_sigma"):
        from poker_ai._core._clamp import clamp_sigma as _clamp_sigma_core

        def clamp_sigma(sigma, m_rows, c_rows, gof):          # noqa: F811
            """Compiled fused gather+blend, with a guarded fallback.

            The kernel needs C-contiguous float64 blocks and an int64 gather; any
            other layout (a test stub, an unusual dtype) falls back to the oracle
            rather than silently mis-typing.
            """
            if (sigma.dtype == np.float64 and sigma.flags.c_contiguous
                    and m_rows.dtype == np.float64 and m_rows.flags.c_contiguous
                    and c_rows.dtype == np.float64 and c_rows.flags.c_contiguous
                    and (gof is None
                         or (gof.dtype == np.int64 and gof.flags.c_contiguous))):
                return _clamp_sigma_core(sigma, m_rows, c_rows, gof)
            return _clamp_sigma_py(sigma, m_rows, c_rows, gof)
except ImportError:
    pass


def node_sigma(state, pk, legal, actor, is_root, n_rows, row_space, cof, gof=None):
    """Register the node and build its per-combo strategy matrix.

    Ensures the ``(n_rows, width)`` ``vregret``/``vstrat`` matrices exist, regret-
    matches the regret matrix, and gathers each combo's row: identity for a
    root-street node, a cluster gather (``cof`` maps combo→dense cluster row, ``-1``
    on infeasible combos → row 0, harmless since their reach is zeroed) otherwise.

    ``gof`` is the precomputed gather index (:meth:`ClusterMapper.gather_of`) — the
    same thing as ``np.where(cof >= 0, cof, 0)`` but built once per board in
    ``refresh`` rather than rebuilt at every node of every iteration.  It is passed
    by both regimes on the hot path; the ``None`` fallback recomputes it so callers
    with only ``cof`` (tests, older call sites) stay correct.

    Returns ``(sigma, regret, strat)`` where ``sigma`` is ``(n_combos, width)`` and
    ``regret``/``strat`` are the node's stored ``(n_rows, width)`` matrices.
    """
    state.ensure_vnode(pk, legal, actor, n_rows, row_space)
    regret = state.vregret[pk]
    strat = state.vstrat[pk]
    sigma_rows = regret_match_matrix(regret)
    if is_root:
        return sigma_rows, regret, strat
    if gof is None:
        gof = np.where(cof >= 0, cof, 0)
    return sigma_rows[gof], regret, strat


def _policy_state(env, cluster, combo_cards, ci, public):
    """The :class:`PolicyState` to query the model at — cluster-keyed where possible.

    The clamp needs the info-set a *hypothetical* holding would produce.  That is
    ``(cluster, history)``, so the cluster is the only card-dependent input — and
    both walk engines can build the state from it
    (:meth:`~environment.poker_env.PokerEnv.policy_state_for_cluster` /
    :meth:`FastEnvAdapter.policy_state_for_cluster`).  Keying by cluster is what
    lets a **modeled solve run on the compiled core**: the old combo-keyed
    ``policy_state_for`` is a ``PokerEnv`` method the ``FastState`` adapters have no
    way to serve, and reaching for it under the core raised ``AttributeError`` —
    which ``SearchAgent._solve_and_store`` swallowed into a silent blueprint
    fallback (zero exploitation -- DBR would read as no better than vanilla Pluribus).

    Falls back to the combo-keyed query when no cluster map is available (a
    hand-built env in a test); that path is ``PokerEnv``-only, as it always was.
    """
    if cluster is not None and cluster >= 0:
        return env.policy_state_for_cluster(int(cluster), public=public)
    return env.policy_state_for(combo_cards[ci], for_blueprint=True, public=public)


def _fill_model_rows(entry, model, env, width, combo_cards, cof, is_root,
                     row_cluster=None):
    """Lazily build any model rows the *current* board references.

    **Rows, not combos — and this is load-bearing.**  ``public_key`` encodes only
    ``(stage, action history)``; it carries **no board**.  Both regimes re-sample the
    completion every iteration (``ClusterMapper.refresh`` / ``reseat_private_cards``),
    so a cache of *combo*-space model rows keyed by ``public_key`` would be built on
    one board and silently reused on every later one.

    The row space is safe where combo space is not: ``n_rows`` comes from
    ``ClusterMapper._universe`` (built once over *all* completions) and ``refresh``
    searchsorts into it, so dense **row r ↔ a fixed LUT cluster** on every board —
    exactly why the regret tables survive across iterations.  And because
    ``info_set = (cluster, history)``, every combo in a cluster has the *same*
    info-set, hence the same ``σ̂``/``c``.  So one query per row is both exact and
    board-stable, and the caller gathers it with the current iteration's ``cof``.

    Only rows reachable on this board are filled (others are never gathered), so the
    entry fills in incrementally as later iterations expose new completions.

    ``row_cluster`` maps **row → raw LUT cluster id**: at the root street rows are
    combos, so it is ``ClusterMapper.root_cluster_of()``; at a clustered street it is
    ``ClusterMapper.universe(street)``, the inverse of ``refresh``'s ``searchsorted``.
    It is what the model is keyed by (see :func:`_policy_state`) — the dense row index
    is a local relabelling and must never be used as a cluster.

    Neither query reads any seat's actual cards, so this is leak-free.  The history is
    canonicalised exactly as the blueprint reads and the belief-likelihood swap (§6.3)
    do, so the clamp and the beliefs describe one and the same opponent at one and the
    same info-set key.
    """
    m_rows, c_rows, filled = entry
    if is_root:
        # Root street rows are lossless (row == combo), but the INFO-SET is still
        # ``(cluster, history)`` — so combos sharing a root cluster share σ̂ exactly.
        # Query once per distinct cluster and broadcast, instead of once per combo
        # (190 → ~25-45 on the 20-card LUT; ~n_combos/n_buckets at production).
        unf = np.flatnonzero(~filled)
        if unf.size == 0:
            return
        if row_cluster is None:
            need, rep, groups = unf, unf, None       # no cluster map: per-combo
        else:
            # Board-infeasible combos (cluster -1) share a card with the board, so
            # they have no info-set to query.  Drop them: their ``m_rows``/``c_rows``
            # stay zero, and the clamp skips ``c == 0`` — result-neutral anyway, since
            # their reach is zeroed upstream so their rows never reach a regret or a
            # value.  Left *unfilled* (not marked filled with a blank row) so ``filled``
            # keeps meaning "has a real model row", which the row↔cluster test relies
            # on; the re-filter each iteration is a handful of ops over the infeasible
            # tail.  (The old combo-keyed query fed the LUT a conflicting hole and used
            # whatever came back — harmless for the same reach reason, but by accident.)
            vals = row_cluster[unf]
            feasible = vals >= 0
            if not feasible.all():
                unf, vals = unf[feasible], vals[feasible]
                if unf.size == 0:
                    return
            uniq, first = np.unique(vals, return_index=True)
            need, rep = unf[first], unf[first]
            groups = [unf[vals == u] for u in uniq]  # rows sharing each info-set
        public = env.policy_public_fields()
        for k, (r, ci) in enumerate(zip(need, rep)):
            cl = None if row_cluster is None else row_cluster[r]
            st = _policy_state(env, cl, combo_cards, ci, public)
            row = np.asarray(model.strategy(st), dtype=np.float64)
            conf = min(1.0, max(0.0, float(model.confidence(st))))
            tgt = groups[k] if groups is not None else (r,)
            m_rows[np.asarray(tgt)[:, None],
                   np.arange(min(row.shape[0], width))] = row[:width]
            c_rows[np.asarray(tgt), 0] = conf
            filled[np.asarray(tgt)] = True
        return
    else:
        cand = np.flatnonzero(cof >= 0)
        r_of = cof[cand]
        fresh = ~filled[r_of]
        cand, r_of = cand[fresh], r_of[fresh]
        if cand.size == 0:
            return
        # One representative combo per not-yet-built row (any combo in the cluster
        # yields the same info-set, so the choice is immaterial).
        _, first = np.unique(r_of, return_index=True)
        need, rep = r_of[first], cand[first]
    if need.size == 0:
        return
    public = env.policy_public_fields()
    for r, ci in zip(need, rep):
        cl = None if row_cluster is None else row_cluster[r]
        st = _policy_state(env, cl, combo_cards, ci, public)
        row = np.asarray(model.strategy(st), dtype=np.float64)
        m_rows[r, : min(row.shape[0], width)] = row[:width]
        # Clamp defensively: a third-party model returning c outside [0, 1] would
        # push the mixture off the simplex (negative mass on the free component).
        c_rows[r, 0] = min(1.0, max(0.0, float(model.confidence(st))))
        filled[r] = True


def apply_model_clamp(sigma, ctx, state, env, pk, actor, width, combo_cards,
                      cof, n_rows, is_root, gof=None, cmaps=None, street=None,
                      root_cluster=None):
    """Blend a modeled seat's realized strategy toward its model — the DBR mixture, doc §5.2.

    ``σ̃ = c·σ̂ + (1 − c)·x`` per combo, where ``x`` is the seat's regret-matched free
    strategy (``sigma`` as produced by :func:`node_sigma`).  This is Data Biased
    Response (Johanson & Bowling 2009) transplanted into the depth-limited solver:
    where confidence is high the bot best-responds to the model, where it is zero the
    seat degenerates to the baseline's adversarial player.

    Applies in **both regimes** — MCCFR and vector share this seam, and both consume
    the returned matrix identically (MCCFR samples one combo's row, the vector regime
    reach-weights every combo), so no regime-specific blend logic is needed.

    Realized-vs-free semantics: the *realized* ``σ̃`` is returned, so the caller's
    child reach-weighting, ``strat_sum`` accrual and regret baseline all use the
    mixture, while the accumulated regrets it regret-matches next iteration remain
    the free component ``x``. That is exactly the restricted-response treatment.

    **Baseline equivalence:** with no models (or none for ``actor``) this returns
    ``sigma`` unchanged, before touching the cache or allocating anything — so an
    unmodeled solve is bit-for-bit the pre-change solver in both regimes.
    """
    models = getattr(ctx, "models", None)
    if not models:
        return sigma
    # The bot is never modeled.  ``SearchAgent`` already drops ``my_seat`` from its
    # per-hand snapshot, but guard here too so the invariant holds however the ctx
    # was built (a hand-assembled ctx in a test or a future caller): blending hero's
    # own rows would have it best-respond to a model *of itself*.
    if actor == getattr(ctx, "my_seat", None):
        return sigma
    model = models.get(actor)
    if model is None:
        return sigma
    key = (actor, pk)
    entry = state.model_sigma_cache.get(key)
    if entry is None:
        entry = (np.zeros((n_rows, width), dtype=np.float64),
                 np.zeros((n_rows, 1), dtype=np.float64),
                 np.zeros(n_rows, dtype=bool))
        state.model_sigma_cache[key] = entry
    # Resolve the row → LUT-cluster map LAZILY — only now, after the no-models
    # early-out.  Evaluating it eagerly at the call site made every *vanilla* node
    # pay for it (and crashed at a pre-flop root, where ``_cmaps`` is ``None``).
    #
    # Root street: rows are combos, so this is the per-combo root cluster.  The
    # caller may supply it directly (``root_cluster``) for the MCCFR pre-flop /
    # multiway-flop roots, where ``_cmaps`` is deliberately ``None`` because the
    # walk never crosses a future street.
    # Clustered street: row r *is* a cluster, recovered from the static universe.
    if is_root:
        row_cluster = root_cluster
        if row_cluster is None and cmaps is not None:
            row_cluster = cmaps.root_cluster_of()
    else:
        row_cluster = None if cmaps is None else cmaps.universe(street)
    _fill_model_rows(entry, model, env, width, combo_cards, cof, is_root,
                     row_cluster)
    m_rows, c_rows, _ = entry
    # Fused gather+blend, in place (P2).  Infeasible combos gather row 0 —
    # harmless, their reach is zeroed upstream (same convention as node_sigma).
    idx = None if is_root else (gof if gof is not None
                                else np.where(cof >= 0, cof, 0))
    return clamp_sigma(sigma, m_rows, c_rows, idx)


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
