"""Policy interface for the search package.

A :class:`Policy` is anything the depth-limited solver (§6.5) or the
continuation-value rollouts (§6.4) can query for an action distribution at a given
``PokerEnv`` state.

The four §4 continuation strategies are not separate artifacts but inference-time
reweightings of one base blueprint: σ is the blueprint's normalised *average strategy*
(with a regret-matching fallback for under-sampled rows), and a biased variant
multiplies one action class's probability by ``bias_multiplier`` and renormalizes.
All implementations share the bias-mask and reweighting helpers; only *where the base
row comes from* differs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from typing_extensions import Literal

from environment.action_space import (
    ACTION_TO_IDX,
    CANONICAL_ACTIONS,
    MAX_ACTIONS_PER_STREET,
)
from environment.poker_env import PolicyState
from poker_ai.blueprint.tree_utils import calculate_strategy_from_row
from poker_ai.search.solver_state import SolverState
from poker_ai.tables.cfr_tables import CFRTables


BiasClass = Literal["none", "fold", "call", "raise"]
"""Continuation-strategy bias modes used at depth-limit leaves."""

_BIAS_CODE = {"none": 0, "fold": 1, "call": 2, "raise": 3}
"""Bias-class → integer code for the in-core reader (matches ``CoreTables._bias_cols``)."""


class Policy(ABC):
    """Action-distribution oracle for one state.

    Subclasses implement :meth:`strategy`.  The shared helpers
    :meth:`_bias_mask` and :meth:`_reweight_bias` keep action-class
    identification and the multiplicative bias step consistent across
    implementations.
    """

    @abstractmethod
    def strategy(self, state: PolicyState, bias: BiasClass = "none") -> np.ndarray:
        """Return the action distribution at ``state``.

        ``state`` is the env's public state plus the actor's info-set key, from
        :attr:`PokerEnv.policy_state` or :meth:`PokerEnv.policy_state_for` (a
        hypothetical hole, for leak-free ``sigma_for_combo`` queries).  ``bias``
        selects the base distribution (``"none"``) or multiplies one action class by
        ``bias_multiplier`` and renormalizes (§4).

        Returns a float32 vector aligned with ``state.legal_actions``, summing to 1.
        """

    @staticmethod
    def _bias_mask(canonical_actions: List[str], bias: BiasClass) -> np.ndarray:
        """Boolean mask over ``canonical_actions`` selecting the biased class.

        Action-class identification by prefix:

        - ``"fold"``  → exactly ``"fold"``.
        - ``"call"``  → ``"call"`` or ``"check"``.
        - ``"raise"`` → any string starting with ``"raise"`` or equal
          to ``"all_in"``.
        - ``"none"``  → all-false (no bias applied).
        """
        n = len(canonical_actions)
        mask = np.zeros(n, dtype=bool)
        if bias == "none":
            return mask
        for i, a in enumerate(canonical_actions):
            if bias == "fold" and a == "fold":
                mask[i] = True
            elif bias == "call" and a in ("call", "check"):
                mask[i] = True
            elif bias == "raise" and (a.startswith("raise") or a == "all_in"):
                mask[i] = True
        return mask

    @staticmethod
    def _reweight_bias(
        sigma: np.ndarray,
        bias_mask: np.ndarray,
        multiplier: float,
    ) -> np.ndarray:
        """Multiply a target action class's probability and renormalize.

        ``σ'(a) ∝ σ(a) · (m if a ∈ biased_class else 1)`` (§4).  ``sigma`` is an
        already-regret-matched probability vector, not a regret row; a ``multiplier``
        of 1 or an all-false mask returns it unchanged.

        Returns
        -------
        numpy.ndarray
            Float32 probability vector at canonical width.
        """
        if multiplier == 1.0 or not bias_mask.any():
            return sigma
        out = sigma.astype(np.float32, copy=True)
        out[bias_mask] *= float(multiplier)
        total = out.sum()
        return out / total if total > 0 else sigma


class BlueprintPolicy(Policy):
    """Reads the blueprint strategy for ``state.info_set`` from a base blueprint.

    σ is the blueprint's **average strategy** (normalised visit counts).  Only the
    time-averaged strategy converges to equilibrium in CFR; the per-iteration
    regret-matched strategy oscillates toward near-pure play, so it serves only as a
    fallback for rows with too little visit mass.  One blueprint on disk backs all four
    §4 variants, which reweight at query time.

    Parameters
    ----------
    tables : CFRTables
        Loaded base-blueprint tables.
    bias_multiplier : float
        Factor applied to the biased action class; inert when ``bias == "none"``.
    min_strategy_mass : int
        Minimum visit mass over the *legal* actions for the average-strategy row to be
        trusted.  A row visited once is a single categorical sample, so reading it
        verbatim yields a pure action from one early training iterate — worse than
        regret matching over the much more heavily updated regret row.
    """

    def __init__(
        self,
        tables: CFRTables,
        bias_multiplier: float = 5.0,
        min_strategy_mass: int = 10,
    ) -> None:
        self._tables = tables
        self._bias_multiplier = float(bias_multiplier)
        self._min_strategy_mass = int(min_strategy_mass)
        # Phase 4c: per-process in-core reader over ``tables`` (a compiled
        # ``CoreTables``), or ``None`` when unavailable.  ``False`` = not-yet-built
        # (lazy on first use, post-fork); distinct from ``None`` = built-and-absent.
        self._core_tables = False

    def reopen_after_fork(self) -> None:
        """Reopen the backing blueprint LMDB indexes in a forked process.

        The blueprint's :class:`~poker_ai.tables.cfr_tables.CFRTables` holds
        per-street LMDB indexes that are unsafe to share across a ``fork`` (reader
        locktable slot reuse → ``MDB_BAD_RSLOT``).  A forked worker that reads this
        policy — e.g. a parallel-search replica evaluating a depth-limit leaf's
        continuation value against the blueprint — must call this once before its
        first query.  Delegates to :meth:`CFRTables.reopen_after_fork`.
        """
        # Drop this process's inherited in-core view FIRST so it is rebuilt against
        # the child's own reopened index/shm arrays on next use (Phase 4c).
        self._core_tables = False
        self._tables.reopen_after_fork()

    def _ensure_core(self):
        """Lazily build (once per process) the compiled ``CoreTables`` read view
        over ``self._tables``, or ``None`` if unavailable — Phase 4c.

        Requires the shm index cache (``enable_index_cache=True`` +
        ``prewarm_caches()``); ``CoreTables.__init__`` raises without it, so a
        blueprint opened the legacy way (no cache) transparently yields ``None`` and
        the caller keeps the pure-Python :meth:`strategy` path.  Best-effort: any
        construction failure → ``None`` (never breaks a search).
        """
        if self._core_tables is not False:
            return self._core_tables
        self._core_tables = None
        try:
            from poker_ai._core import CORE_AVAILABLE
            if not CORE_AVAILABLE:
                return None
            if getattr(self._tables, "_index_caches", None) is None:
                return None
            from poker_ai._core import _traverse as _cyt
            self._core_tables = _cyt.CoreTables(
                self._tables, CANONICAL_ACTIONS, ACTION_TO_IDX, MAX_ACTIONS_PER_STREET
            )
        except Exception:
            self._core_tables = None
        return self._core_tables

    def core_sigma(
        self, r: int, info_set: bytes, legal_cols: np.ndarray, bias: BiasClass
    ) -> np.ndarray:
        """In-core equivalent of :meth:`strategy` for a canonical-only legal set —
        Phase 4c.  ``legal_cols`` is the canonical column of each legal action (in
        legal order); returns the ``float32`` distribution over it, tolerance-equal
        (<1e-6) to ``strategy``.

        Routes through :meth:`_ensure_core` (an O(1) memo check) so it transparently
        rebuilds after a fork's ``reopen_after_fork`` nulled the reader; the caller is
        expected to have confirmed a non-``None`` reader (a ``None`` here — no shm
        cache — is a contract violation and raises loudly rather than silently
        returning a wrong distribution).
        """
        return self._ensure_core().blueprint_sigma(
            r, info_set, legal_cols, _BIAS_CODE[bias],
            self._bias_multiplier, self._min_strategy_mass,
        )

    def strategy(self, state: PolicyState, bias: BiasClass = "none") -> np.ndarray:
        r = state.betting_round
        sigma = self._average_strategy(r, state)
        if sigma is None:
            # Fallback: regret-match the cumulative regret row (the
            # pre-fix behaviour) when the average strategy has no
            # trustworthy estimate for this infoset.
            regret_row = self._tables.regret[r].get_row_if_exists(state.info_set)
            if regret_row is None:
                regret_row = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)
            sigma = calculate_strategy_from_row(regret_row, state.valid_mask)
        bias_mask = self._bias_mask(CANONICAL_ACTIONS[r], bias)
        full = self._reweight_bias(sigma, bias_mask, self._bias_multiplier)
        legal = state.legal_actions
        if not legal:
            return np.array([], dtype=np.float32)
        # Overlay-injected actions (§6.1) appear in ``legal`` but not
        # in the canonical ``ACTION_TO_IDX`` map.  The blueprint was
        # trained on the canonical abstraction and has no opinion on
        # off-tree sizes, so each such action receives zero mass; the
        # canonical probabilities are then renormalised over the
        # filtered legal set.  All-overlay (no canonical action legal)
        # falls back to uniform-over-legal so the caller never sees
        # a degenerate distribution.
        canonical_idx = ACTION_TO_IDX[r]
        probs = np.zeros(len(legal), dtype=np.float32)
        for i, a in enumerate(legal):
            if a in canonical_idx:
                probs[i] = full[canonical_idx[a]]
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs[:] = 1.0 / len(legal)
        return probs

    def _average_strategy(
        self, r: int, state: PolicyState
    ) -> "np.ndarray | None":
        """Normalised average-strategy row, or ``None`` if not trustworthy.

        Masks the visit-count row to the legal actions before both the
        mass check and the normalisation, so counts recorded for
        actions that are illegal at *this* node (the strategy tables
        are keyed by info set, which on later streets can admit
        different legal sets across stack configurations) never leak
        probability.
        """
        row = self._tables.strategy[r].get_row_if_exists(state.info_set)
        if row is None:
            return None
        masked = row.astype(np.float32)
        masked[~state.valid_mask] = 0.0
        total = float(masked.sum())
        if total < self._min_strategy_mass:
            return None
        return masked / total


class SearchPolicy:
    """Reader over a solved :class:`SolverState` (§6.5).

    A :class:`SearchResult` exposes two: the **final-iteration** policy the bot plays
    and the **average** policy that feeds the next round's belief update.  Unlike
    :class:`Policy` it is keyed by the solver's native ``(public_key, hand_row)``, which
    ``PolicyState`` does not carry; the agent holds the env, computes the key and calls
    :meth:`strategy_for`.  That keeps this regime-agnostic — the same rows whether
    MCCFR or vector wrote them.

    Parameters
    ----------
    state : SolverState
        The solved tables.
    use_average : bool
        ``True`` reads the normalised cumulative strategy for belief updates; ``False``
        the regret-matched final strategy (frozen rows pinned), which the bot plays.
    """

    def __init__(self, state: SolverState, *, use_average: bool) -> None:
        self._state = state
        self._use_average = use_average

    @property
    def use_average(self) -> bool:
        return self._use_average

    def strategy_for(
        self,
        public_key,
        hand_row: int,
        legal_actions,
        *,
        use_average: bool = None,
    ) -> np.ndarray:
        """Action distribution at ``(public_key, hand_row)`` aligned to ``legal_actions``.

        Reads the average (``strat_sum``) or the final regret-matched strategy
        (frozen rows pinned for play).  The stored row is in ``legal_at`` order;
        it is remapped onto ``legal_actions`` by string, so overlay-injected
        actions absent from the row receive zero mass and the result is
        renormalised over the legal set (mirroring :meth:`BlueprintPolicy.strategy`).
        An unseen node, or a node with no accumulated strategy, falls back to
        uniform over ``legal_actions``.
        """
        use_avg = self._use_average if use_average is None else use_average
        legal_actions = tuple(legal_actions)
        n = len(legal_actions)
        if n == 0:
            return np.array([], dtype=np.float32)

        state = self._state
        legal_at = state.legal_at.get(public_key)
        key = (public_key, hand_row)
        node = None
        if legal_at is not None:
            if use_avg:
                node = state.average_sigma(key)
            elif key in state.frozen:
                node = np.asarray(state.frozen[key], dtype=np.float32)
            else:
                node = state.sigma(key)
        if node is None or legal_at is None:
            return np.full(n, 1.0 / n, dtype=np.float32)

        idx = {a: j for j, a in enumerate(legal_at)}
        out = np.zeros(n, dtype=np.float32)
        for j, a in enumerate(legal_actions):
            if a in idx:
                out[j] = node[idx[a]]
        total = out.sum()
        if total > 0.0:
            out /= total
        else:
            out[:] = 1.0 / n
        return out
