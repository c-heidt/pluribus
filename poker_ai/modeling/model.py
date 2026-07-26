"""Opponent models — the ``σ̂`` + per-infoset confidence oracle (opponent_modeling §4.1).

An :class:`OpponentModel` is one opponent seat's estimated strategy plus a
per-infoset confidence ``c``.  The solver clamp blends a modeled seat's realized
strategy toward ``σ̂`` by ``c`` (doc §5), belief tracking reads ``σ̂`` as the seat's
likelihood (§6.3), and leaf continuations roll out under it (§5.4).  Both queries
take the same :class:`PolicyState` the search's :class:`~poker_ai.search.policy.Policy`
ABC consumes, so off-tree histories canonicalize exactly as blueprint lookups do.

One provider (doc §2 goals):

- :class:`SyntheticOpponentModel` — wraps an *exactly-known* opponent policy with a
  constant-or-scheduled ``c`` and an optional controlled ℓ1 perturbation (the design
  doc's §6.2 model-error sweep).  Needs no data; it is the instrument for the
  exploitation eval, which injects model quality as a controlled variable rather than
  learning it (see :mod:`poker_ai.modeling.schedules` for the error/confidence shaping).
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np

from environment.poker_env import PolicyState

# A confidence value, or a callable computing one from the state (a "schedule").
ConfidenceSpec = Union[float, Callable[[PolicyState], float]]
# A target ℓ1 error magnitude, or a callable computing one from the state — lets the
# synthetic sweep vary model quality across infosets (street-graded / noisy) rather
# than uniformly.  See :mod:`poker_ai.modeling.schedules`.
ErrorSpec = Union[float, Callable[[PolicyState], float]]


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
      clamped to ``[0, p_max]``.  ``c ≡ 1`` (``p_max = 1``) is naive best response (the unsafe ceiling).
    - ``error`` — a **target ℓ1 distance** for a seeded per-infoset perturbation of
      ``σ̂``, either a constant or a callable ``error(state)`` (a *schedule*, e.g.
      street-graded or per-infoset-noisy — see :mod:`poker_ai.modeling.schedules`).
      The perturbed row is ``(1−α)·σ + α·q`` with ``q`` a seeded Dirichlet draw and
      ``α = min(1, error / ‖q−σ‖₁)``, so it stays a valid distribution and
      ``‖σ̂−σ‖₁ = min(error, ‖q−σ‖₁)`` — exact for the moderate targets the sweep
      uses.  ``error = 0`` (default) is the exact model.

    The pure-vs-noisy axis (whether error magnitude and confidence are uniform across
    infosets or vary, and whether ``c`` is calibrated to the local error) is expressed
    entirely through the ``error``/``confidence`` schedules; this class just consumes
    them.  Error *direction* is always per-infoset random (the seeded ``q``).

    Parameters
    ----------
    policy
        The exactly-known opponent policy; queried as ``policy.strategy(state, bias)``.
    confidence
        Constant ``c`` or a ``c(state)`` schedule (default ``1.0``).
    p_max
        Confidence cap (default ``1.0``); the returned ``c`` is clamped to it.
    error
        Target ℓ1 perturbation magnitude — constant or ``error(state)`` schedule
        (default ``0.0`` — exact).
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
        error: ErrorSpec = 0.0,
        seed: int = 0,
        bias: str = "none",
    ) -> None:
        self._policy = policy
        self._confidence = confidence
        self._p_max = float(p_max)
        self._error = error if callable(error) else float(error)
        self._seed = int(seed)
        self._bias = bias

    def _error_at(self, state: PolicyState) -> float:
        """Resolve the (possibly scheduled) target ℓ1 error at ``state``, clamped ≥ 0."""
        e = self._error(state) if callable(self._error) else self._error
        return max(0.0, float(e))

    def strategy(self, state: PolicyState) -> np.ndarray:
        sigma = np.asarray(self._policy.strategy(state, self._bias), dtype=np.float32)
        error = self._error_at(state)
        if error > 0.0 and sigma.size > 1:
            sigma = self._perturb(sigma, state.info_set, error)
        return sigma

    def confidence(self, state: PolicyState) -> float:
        c = self._confidence(state) if callable(self._confidence) else self._confidence
        return float(min(self._p_max, max(0.0, float(c))))

    def reopen_after_fork(self) -> None:
        """Reopen the wrapped policy's LMDB env after a ``fork`` (parallel replicas).

        The solver clamp queries ``σ̂`` *inside* each forked replica, and the wrapped
        policy is usually blueprint-backed.  LMDB's reader-lock table is a
        process-shared mmap, so an inherited env handle must be reopened or the first
        read trips ``MDB_BAD_RSLOT`` in child *and* parent — the recurring pitfall
        :func:`poker_ai.search.parallel._reopen_leaf_fleet_lmdb` exists to repair.
        Delegates to the wrapped policy; a no-op for in-memory policies that expose
        no such hook.
        """
        reopen = getattr(self._policy, "reopen_after_fork", None)
        if reopen is not None:
            reopen()

    def _perturb(self, sigma: np.ndarray, info_set: bytes, error: float) -> np.ndarray:
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
        alpha = min(1.0, error / d)
        out = (1.0 - alpha) * sigma + alpha * q
        total = float(out.sum())
        return (out / total).astype(np.float32) if total > 0 else sigma


__all__ = [
    "OpponentModel",
    "SyntheticOpponentModel",
    "ConfidenceSpec",
    "ErrorSpec",
]
