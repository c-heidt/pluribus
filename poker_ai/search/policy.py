"""Policy interface for the search package.

A :class:`Policy` is anything the depth-limited solver (§6.5) or the
continuation-value rollouts (§6.4) can query for an action distribution
at a given ``PokerEnv`` state.  Two concrete implementations are planned
in :doc:`docs/subgame_solving`; only :class:`BlueprintPolicy` is shipped
in this module.  The other — ``SearchPolicy`` (reads in-memory subgame
regrets) — slots into the same ABC once the solver in §6.5 lands.

The four §4 continuation strategies are not separate artifacts: they are
inference-time reweightings of the single base blueprint.  The base
distribution σ is computed by regret matching; the biased variants
multiply the probability of one action class by ``bias_multiplier`` (5)
and renormalize.  All implementations share the same bias-mask and
reweighting helpers; the only per-implementation logic is *where the
regret row comes from*.
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

        Parameters
        ----------
        state : PolicyState
            Decoupled view of the env's public state plus the
            actor's info-set key; built from :attr:`PokerEnv.policy_state`
            (current actor as-is) or :meth:`PokerEnv.policy_state_for`
            (current actor under a hypothetical hole, for leak-free
            ``sigma_for_combo`` queries).
        bias : BiasClass
            ``"none"`` for the base distribution; ``"fold"`` /
            ``"call"`` / ``"raise"`` multiply that action class's
            probability by ``bias_multiplier`` and renormalize (§4).

        Returns
        -------
        numpy.ndarray
            Float32 probability vector aligned with
            ``state.legal_actions``; sums to 1 (or 0 if no actions
            are legal, which should not occur in a well-formed game
            state).
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

        Computes ``σ'(a) ∝ σ(a) · (m if a ∈ biased_class else 1)`` (§4).
        The input ``sigma`` is an already-regret-matched probability
        vector (not a regret row); a ``multiplier`` of 1 or an
        all-false ``bias_mask`` (i.e. ``bias == "none"``) returns it
        unchanged.

        Parameters
        ----------
        sigma : numpy.ndarray
            Probability vector, canonical-action width; sums to 1.
        bias_mask : numpy.ndarray
            Boolean mask of the same length marking the biased class.
        multiplier : float
            Reweighting factor ``m`` (5.0 for the §4 variants).

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
    """Reads the regret row for ``state.info_set`` from a base blueprint.

    The base distribution σ is computed by regret matching; the §4
    continuation variants multiply the requested action class's
    probability by ``bias_multiplier`` and renormalize at query time.
    A single blueprint on disk backs all four variants.

    Parameters
    ----------
    tables : CFRTables
        Loaded base-blueprint tables.
    bias_multiplier : float
        Reweighting factor applied to the biased action class; only
        takes effect when :meth:`strategy` is called with
        ``bias != "none"`` (§4 uses 5.0).
    """

    def __init__(self, tables: CFRTables, bias_multiplier: float = 5.0) -> None:
        self._tables = tables
        self._bias_multiplier = float(bias_multiplier)

    def reopen_after_fork(self) -> None:
        """Reopen the backing blueprint LMDB indexes in a forked process.

        The blueprint's :class:`~poker_ai.tables.cfr_tables.CFRTables` holds
        per-street LMDB indexes that are unsafe to share across a ``fork`` (reader
        locktable slot reuse → ``MDB_BAD_RSLOT``).  A forked worker that reads this
        policy — e.g. a parallel-search replica evaluating a depth-limit leaf's
        continuation value against the blueprint — must call this once before its
        first query.  Delegates to :meth:`CFRTables.reopen_after_fork`.
        """
        self._tables.reopen_after_fork()

    def strategy(self, state: PolicyState, bias: BiasClass = "none") -> np.ndarray:
        r = state.betting_round
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


class SearchPolicy:
    """Reader over a solved :class:`SolverState` (§6.5).

    A :class:`SearchResult` exposes two of these: the **final-iteration** policy
    the bot plays, and the **average** policy that feeds the next round's belief
    update.  Unlike :class:`Policy`, it is keyed by the solver's native
    ``(public_key, hand_row)`` — the in-memory tables are keyed that way and
    ``PolicyState`` does not carry ``public_key``/``hand_row`` (the four §4 bias
    variants are a leaf-continuation concept, not the bot's own play).  The
    consumer (the search agent) holds the env, so it computes the key and calls
    :meth:`strategy_for`; this keeps :class:`SearchPolicy` regime-agnostic — it
    reads the same rows whether they were written by the MCCFR or vector regime.

    Parameters
    ----------
    state : SolverState
        The solved tables.
    use_average : bool
        ``True`` reads the normalised cumulative strategy (``strat_sum``) — the
        weighted-average policy for belief updates; ``False`` reads the
        regret-matched final strategy (with frozen rows pinned) — the policy the
        bot plays.
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
