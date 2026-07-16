"""``ModelPolicy`` — a σ̂-backed drop-in Policy for modeled-seat leaf continuations (§4.3).

A modeled seat's depth-limit leaf continuations are "the model, fold-/call-/raise-
biased ×5" (opponent_modeling §5.4).  :class:`ModelPolicy` is a
:class:`~poker_ai.search.policy.Policy` (same ABC as
:class:`~poker_ai.search.policy.BlueprintPolicy`) that takes ``σ̂`` from an
:class:`~poker_ai.modeling.model.OpponentModel` as the base distribution and reuses the
inherited ``_bias_mask`` / ``_reweight_bias`` for the four continuation variants — so it
slots into the leaf fleet (``LeafConfig.seat_policies``) with zero new machinery.

``σ̂`` already arrives aligned to ``state.legal_actions`` with off-tree actions zeroed
(the :class:`OpponentModel` contract), so the bias reweight is applied directly to that
row — identical to the transform :class:`BlueprintPolicy` applies to its own row.
"""

from __future__ import annotations

import numpy as np

from poker_ai.search.policy import BiasClass, Policy


class ModelPolicy(Policy):
    """A :class:`Policy` whose base row is a modeled opponent's ``σ̂``.

    Parameters
    ----------
    model
        The :class:`~poker_ai.modeling.model.OpponentModel` supplying ``σ̂(state)``.
    bias_multiplier
        The continuation-variant reweighting factor (§4 uses 5.0), applied to the
        biased action class exactly as in :class:`BlueprintPolicy`.
    """

    def __init__(self, model, bias_multiplier: float = 5.0) -> None:
        self._model = model
        self._bias_multiplier = float(bias_multiplier)

    def strategy(self, state, bias: BiasClass = "none") -> np.ndarray:
        base = np.asarray(self._model.strategy(state), dtype=np.float32)
        if bias == "none" or base.size == 0:
            return base
        # ``_bias_mask`` identifies the action class by token prefix, so the legal
        # action list works directly as the "canonical" list here.
        mask = self._bias_mask(list(state.legal_actions), bias)
        return self._reweight_bias(base, mask, self._bias_multiplier)


__all__ = ["ModelPolicy"]
