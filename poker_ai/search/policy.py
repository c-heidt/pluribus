"""Policy interface for the search package.

A :class:`Policy` is anything the depth-limited solver (§6.5) or the
leaf-EV rollouts (§6.4) can query for an action distribution at a given
``PokerEnv`` state.  Three concrete implementations are planned in
:doc:`docs/subgame_solving`; only :class:`BlueprintPolicy` is shipped
in this module.  The other two — ``BiasedBlueprintPolicy`` (reads one
of k precomputed biased blueprints) and ``SearchPolicy`` (reads
in-memory subgame regrets) — slot into the same ABC once their
dependencies land (biased blueprint training in §4, solver in §6.5).

All implementations share the same regret-matching helpers; the only
per-implementation logic is *where the regret row comes from*.
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
from poker_ai.tables.cfr_tables import CFRTables


BiasClass = Literal["none", "fold", "call", "raise"]
"""Continuation-strategy bias modes used at depth-limit leaves."""


class Policy(ABC):
    """Action-distribution oracle for one state.

    Subclasses implement :meth:`strategy`.  The shared helpers
    :meth:`_bias_mask` and :meth:`_regret_match_with_bias` keep
    action-class identification and the regret-matching + bias step
    consistent across implementations.
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
            ``"call"`` / ``"raise"`` add an additive bias to that
            action class before regret matching.

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
    def _regret_match_with_bias(
        regret_row: np.ndarray,
        valid_mask: np.ndarray,
        bias_mask: np.ndarray,
        bias_magnitude: float,
    ) -> np.ndarray:
        """Regret-match a row with an additive bias on a target action class.

        Computes ``σ(a) ∝ max(0, R(a) + b · 𝟙[a ∈ biased_class])`` over
        the legal subset, with uniform fallback when all positive
        biased regrets are zero.  Delegates the final normalisation to
        :func:`~poker_ai.blueprint.tree_utils.calculate_strategy_from_row`
        so behaviour matches the blueprint's regret-matching exactly
        when ``bias_magnitude == 0``.

        Parameters
        ----------
        regret_row : numpy.ndarray
            1-D regret vector, canonical-action width.
        valid_mask : numpy.ndarray
            Boolean mask of the same length marking legal actions.
        bias_mask : numpy.ndarray
            Boolean mask of the same length marking the biased class.
        bias_magnitude : float
            Bias amount ``b`` (ignored when zero).

        Returns
        -------
        numpy.ndarray
            Float32 probability vector at canonical width.
        """
        if bias_magnitude == 0.0:
            return calculate_strategy_from_row(regret_row, valid_mask)
        biased = regret_row.astype(np.float32, copy=True)
        biased[bias_mask] += float(bias_magnitude)
        return calculate_strategy_from_row(biased, valid_mask)


class BlueprintPolicy(Policy):
    """Reads the regret row for ``env.info_set`` from a base blueprint.

    Optional additive bias on a target action class supports Phase 1
    of the rollout (`docs/subgame_solving.md` §5), where biased
    continuation strategies are derived at runtime instead of from
    precomputed biased blueprints.

    Parameters
    ----------
    tables : CFRTables
        Loaded base-blueprint tables.
    bias_magnitude : float
        Bias amount ``b``; only applied when :meth:`strategy` is
        called with ``bias != "none"``.
    """

    def __init__(self, tables: CFRTables, bias_magnitude: float = 0.0) -> None:
        self._tables = tables
        self._bias_magnitude = float(bias_magnitude)

    def strategy(self, state: PolicyState, bias: BiasClass = "none") -> np.ndarray:
        r = state.betting_round
        regret_row = self._tables.regret[r].get_row_if_exists(state.info_set)
        if regret_row is None:
            regret_row = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)
        valid_mask = state.valid_mask
        bias_mask = self._bias_mask(CANONICAL_ACTIONS[r], bias)
        full = self._regret_match_with_bias(
            regret_row, valid_mask, bias_mask, self._bias_magnitude
        )
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
