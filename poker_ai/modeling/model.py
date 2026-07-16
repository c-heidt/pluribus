"""Opponent models — the ``σ̂`` + per-infoset confidence oracle (opponent_modeling §4.1).

An :class:`OpponentModel` is one opponent seat's estimated strategy plus a
per-infoset confidence ``c``.  The solver clamp blends a modeled seat's realized
strategy toward ``σ̂`` by ``c`` (doc §5), belief tracking reads ``σ̂`` as the seat's
likelihood (§6.3), and leaf continuations roll out under it (§5.4).  Both queries
take the same :class:`PolicyState` the search's :class:`~poker_ai.search.policy.Policy`
ABC consumes, so off-tree histories canonicalize exactly as blueprint lookups do.

Two providers (doc §2 goals):

- :class:`SyntheticOpponentModel` — wraps an *exactly-known* opponent policy with a
  constant-or-scheduled ``c`` and an optional controlled ℓ1 perturbation (the design
  doc's §6.2 model-error sweep).  Needs no data; drives the headline sweeps.
- ``BayesOpponentModel`` (added in the counts/Bayes step) — the online-learned
  Dirichlet posterior over coarse behavioral buckets.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np

from environment.poker_env import PolicyState
from poker_ai.modeling.counts import (
    N_ACTION_CLASSES,
    CountsView,
    action_class,
    model_key,
)

# A confidence value, or a callable computing one from the state (a "schedule").
ConfidenceSpec = Union[float, Callable[[PolicyState], float]]


class OpponentModel(ABC):
    """One opponent's estimated strategy ``σ̂`` + per-infoset confidence ``c``."""

    @abstractmethod
    def strategy(self, state: PolicyState) -> np.ndarray:
        """``σ̂`` aligned with ``state.legal_actions``.

        Overlay-injected (off-tree) actions get zero model mass and the row is
        renormalized over the remaining legal set — mirroring
        :meth:`~poker_ai.search.policy.BlueprintPolicy.strategy`.
        """

    @abstractmethod
    def confidence(self, state: PolicyState) -> float:
        """The per-infoset confidence ``c ∈ [0, p_max]`` at ``state.info_set``."""


def _stable_hash(info_set: bytes) -> int:
    """A process-stable non-negative 64-bit hash of an info-set key.

    ``hash()`` is salted per process, which would make a "seeded" perturbation
    non-reproducible across runs; blake2b is stable, so the same ``(seed, info_set)``
    always yields the same perturbation (design-doc §6.2 requires seeded error).
    """
    return int.from_bytes(hashlib.blake2b(bytes(info_set), digest_size=8).digest(), "little")


class SyntheticOpponentModel(OpponentModel):
    """``σ̂`` = an exactly-known policy (optionally ℓ1-perturbed); ``c`` = a schedule.

    Wraps any :class:`~poker_ai.search.policy.Policy` (e.g. the evaluation's
    ``BlueprintOpponent`` bias variants), so a synthetic model of a runner opponent
    is exact by construction.  Two knobs implement the design-doc §6.2 sweeps:

    - ``confidence`` — a constant ``c`` or a callable ``c(state)`` (the schedule),
      clamped to ``[0, p_max]``.  ``c ≡ 1`` (``p_max = 1``) is condition **B1**.
    - ``error`` — a **target ℓ1 distance** for a seeded per-infoset perturbation of
      ``σ̂``.  The perturbed row is ``(1−α)·σ + α·q`` with ``q`` a seeded
      Dirichlet draw and ``α = min(1, error / ‖q−σ‖₁)``, so it stays a valid
      distribution and ``‖σ̂−σ‖₁ = min(error, ‖q−σ‖₁)`` — exact for the moderate
      targets the sweep uses.  ``error = 0`` (default) is the exact model.

    Parameters
    ----------
    policy
        The exactly-known opponent policy; queried as ``policy.strategy(state, bias)``.
    confidence
        Constant ``c`` or a ``c(state)`` schedule (default ``1.0``).
    p_max
        Confidence cap (default ``1.0``); the returned ``c`` is clamped to it.
    error
        Target ℓ1 perturbation magnitude (default ``0.0`` — exact).
    seed
        Perturbation seed; combined with the info-set hash so each infoset is
        perturbed reproducibly and independently.
    bias
        Bias class forwarded to the wrapped policy (default ``"none"``).
    """

    def __init__(
        self,
        policy,
        confidence: ConfidenceSpec = 1.0,
        *,
        p_max: float = 1.0,
        error: float = 0.0,
        seed: int = 0,
        bias: str = "none",
    ) -> None:
        self._policy = policy
        self._confidence = confidence
        self._p_max = float(p_max)
        self._error = float(error)
        self._seed = int(seed)
        self._bias = bias

    def strategy(self, state: PolicyState) -> np.ndarray:
        sigma = np.asarray(self._policy.strategy(state, self._bias), dtype=np.float32)
        if self._error > 0.0 and sigma.size > 1:
            sigma = self._perturb(sigma, state.info_set)
        return sigma

    def confidence(self, state: PolicyState) -> float:
        c = self._confidence(state) if callable(self._confidence) else self._confidence
        return float(min(self._p_max, max(0.0, float(c))))

    def _perturb(self, sigma: np.ndarray, info_set: bytes) -> np.ndarray:
        """Seeded ℓ1 perturbation toward a random point on the base's support.

        The random target ``q`` is drawn only on ``sigma``'s support, so the perturbed
        row keeps every zero of ``sigma`` at zero — in particular the overlay
        (off-tree) actions the wrapped policy already zeroed stay zero, honouring the
        :meth:`OpponentModel.strategy` contract.  It stays a valid distribution (a
        convex mix of two distributions on the same support).
        """
        support = sigma > 0.0
        if int(support.sum()) <= 1:
            return sigma                                  # a point mass — nothing to shift
        rng = np.random.default_rng([self._seed, _stable_hash(info_set)])
        q = np.zeros_like(sigma)
        q[support] = rng.dirichlet(np.ones(int(support.sum()))).astype(np.float32)
        d = float(np.abs(q - sigma).sum())
        if d <= 0.0:
            return sigma
        alpha = min(1.0, self._error / d)
        out = (1.0 - alpha) * sigma + alpha * q
        total = float(out.sum())
        return (out / total).astype(np.float32) if total > 0 else sigma


class BayesOpponentModel(OpponentModel):
    """The online-learned model: a coarse-bucket Dirichlet posterior over the blueprint.

    ``σ̂`` shrinks *this state's* blueprint row toward the observed coarse-bucket class
    frequencies (opponent_modeling §3 "Model form (learned)"):

        prior(a)      = collapse blueprint_row(state) onto the 4 classes  (per-state)
        σ̂_coarse(k,a) = (τ·prior(a) + n(k,a)) / (τ + n(k))                (k = π(state))
        σ̂(state)      = expand σ̂_coarse onto legal_actions                (blueprint sizing)

    The expand step distributes each class's coarse probability across the legal
    actions of that class in proportion to the blueprint row (so the model predicts
    the class — fold / call / raise / all-in — and the blueprint keeps the bet
    *sizing*).  ``n = 0`` recovers the exact per-state blueprint; ``n → ∞`` recovers the
    empirical class frequencies.  Confidence is ``c = min(p_max, n(k) / (n(k) + τ))``.

    Parameters
    ----------
    blueprint_policy
        The base blueprint; queried as ``strategy(state, bias)`` for the per-state
        prior (its average-strategy row, already remapped to the legal set).
    counts
        A frozen :class:`~poker_ai.modeling.counts.CountsView` (the per-hand snapshot).
    tiers
        The :class:`~poker_ai.modeling.tiers.StrengthTiers` for the model key ``π``.
    tau
        Dirichlet prior strength (default 50) — shared by the posterior and confidence.
    p_max
        Confidence cap (default 0.8).
    bias
        Bias class forwarded to the blueprint prior query (default ``"none"``).
    """

    def __init__(
        self,
        blueprint_policy,
        counts: CountsView,
        tiers,
        *,
        tau: float = 50.0,
        p_max: float = 0.8,
        bias: str = "none",
    ) -> None:
        self._bp = blueprint_policy
        self._counts = counts
        self._tiers = tiers
        self._tau = float(tau)
        self._p_max = float(p_max)
        self._bias = bias

    def strategy(self, state: PolicyState) -> np.ndarray:
        legal = state.legal_actions
        if not legal:
            return np.array([], dtype=np.float32)
        bp = np.asarray(self._bp.strategy(state, self._bias), dtype=np.float64)
        classes = np.array([action_class(a) for a in legal], dtype=np.intp)
        prior = np.zeros(N_ACTION_CLASSES, dtype=np.float64)
        np.add.at(prior, classes, bp)                       # collapse bp → 4 classes

        n = self._counts.count(model_key(state, self._tiers))
        n_total = float(n.sum())
        denom = self._tau + n_total
        # denom > 0 for any sane config (tau default 50); guard the degenerate
        # tau=0-and-no-data case so it degrades to the blueprint, never NaN.
        sigma_coarse = (self._tau * prior + n) / denom if denom > 0 else prior

        # Expand each class's mass across its legal actions by the blueprint sizing.
        probs = np.zeros(len(legal), dtype=np.float64)
        for c in range(N_ACTION_CLASSES):
            idx = np.nonzero(classes == c)[0]
            if idx.size == 0 or sigma_coarse[c] == 0.0:
                continue                                    # class not legal here → drop
            w = bp[idx]
            s = float(w.sum())
            probs[idx] = sigma_coarse[c] * (w / s if s > 0 else 1.0 / idx.size)
        total = float(probs.sum())
        if total > 0:
            probs /= total                                  # renormalise (dropped classes)
        else:
            probs[:] = 1.0 / len(legal)
        return probs.astype(np.float32)

    def confidence(self, state: PolicyState) -> float:
        n_total = self._counts.total(model_key(state, self._tiers))
        denom = n_total + self._tau
        c = (n_total / denom) if denom > 0 else 0.0
        return float(min(self._p_max, c))


__all__ = [
    "OpponentModel",
    "SyntheticOpponentModel",
    "BayesOpponentModel",
    "ConfidenceSpec",
]
