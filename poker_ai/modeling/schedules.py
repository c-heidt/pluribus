"""Error and confidence *schedules* for the synthetic opponent-model sweep (evaluation).

The exploitation eval injects model quality as a controlled variable (see
``docs/evaluation.md``): wrap the true opponent policy in a
:class:`~poker_ai.modeling.model.SyntheticOpponentModel` and dial its ``error`` /
``confidence``.  These builders shape how those two quantities vary **across
infosets** — the *pure vs noisy* axis — and return the plain floats / ``callable(state)``
that ``SyntheticOpponentModel`` consumes.  Error *direction* is always per-infoset
random (the model's seeded Dirichlet draw); these control error *magnitude* and the
*confidence signal*.

Three families:

- **Error magnitude** — :func:`uniform_error` (a scalar, the clean dose-response
  x-axis), :func:`street_error` (graded by betting round to mirror the measured
  saturation gradient — low preflop, high river), and :func:`with_infoset_noise`
  (per-infoset multiplicative jitter so magnitude is heterogeneous, not uniform).
- **Confidence** — :func:`confidence_from_error` is the one knob; ``c(state) =
  clip(intercept + slope·error(state) + noise)``.  ``slope < 0`` is *calibrated*
  (confident where accurate — the realistic, fair-to-DBR case); ``slope > 0`` is
  *anti-calibrated / overconfident* (confident where wrong — the adversarial case a
  strong opponent punishes); ``slope = 0`` is *flat* (the ablation that removes
  selective trust).  Named wrappers :func:`calibrated`, :func:`anti_calibrated`,
  :func:`flat` cover the three regimes.

Add per-infoset ``noise`` to the calibrated schedule to manufacture *confidently
wrong* infosets — the single most dangerous thing a real learned model does and the
sharpest test of robustness against a strong opponent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping as _Mapping
from typing import Callable, Mapping, Optional, Union

import numpy as np

from environment.poker_env import PolicyState
from poker_ai.modeling.model import _stable_hash

# A scalar or a ``callable(state) -> float`` (an error/confidence schedule).
Spec = Union[float, Callable[[PolicyState], float]]


def _resolve(spec: Spec, state: PolicyState) -> float:
    """Evaluate a scalar-or-callable spec at ``state``."""
    return float(spec(state)) if callable(spec) else float(spec)


def _infoset_normal(info_set: bytes, seed: int) -> float:
    """A reproducible standard-normal draw keyed by ``(seed, info_set)``.

    Uses the same process-stable info-set hash as the model's perturbation, so a
    schedule's jitter is deterministic across runs (paired/CRN evaluation needs the
    injected model to be identical for a given seed).
    """
    return float(np.random.default_rng([int(seed), _stable_hash(info_set)]).standard_normal())


# --------------------------------------------------------------------------- #
# Error-magnitude schedules
# --------------------------------------------------------------------------- #
def uniform_error(e: float) -> float:
    """Uniform target ℓ1 error across every infoset — the clean threshold x-axis.

    Returns the scalar unchanged; present as a named counterpart to the noisy/graded
    builders so a sweep config reads symmetrically.
    """
    return float(e)


def street_error(
    by_round: Mapping[int, float], *, default: float = 0.0
) -> Callable[[PolicyState], float]:
    """Error graded by betting round (``0=preflop … 3=river``).

    Mirrors the realistic profile the learned model actually exhibits — error
    concentrated in the deep, low-saturation streets — so the measured threshold is
    expressed against a realistic quality profile rather than a uniform fiction.
    """
    table = {int(k): float(v) for k, v in by_round.items()}
    default = float(default)

    def _sched(state: PolicyState) -> float:
        return table.get(int(state.betting_round), default)

    return _sched


def with_infoset_noise(
    base: Spec, *, sigma: float, seed: int = 0, lo: float = 0.0, hi: float = 2.0
) -> Callable[[PolicyState], float]:
    """Make an error magnitude heterogeneous across infosets (multiplicative jitter).

    ``error(state) = clip(base(state) · exp(sigma · z_infoset), lo, hi)`` with
    ``z_infoset`` a per-infoset standard normal.  Log-normal keeps magnitudes positive;
    ``sigma`` is the spread (``0`` reproduces ``base``).  Use to move off the uniform
    fiction toward the concentrated-error regime of a real model.
    """
    sigma = float(sigma)

    def _sched(state: PolicyState) -> float:
        b = _resolve(base, state)
        if b <= 0.0 or sigma <= 0.0:
            return max(lo, min(hi, b))
        z = _infoset_normal(state.info_set, seed)
        return float(min(hi, max(lo, b * math.exp(sigma * z))))

    return _sched


# --------------------------------------------------------------------------- #
# Confidence schedules
# --------------------------------------------------------------------------- #
def confidence_from_error(
    *,
    error_spec: Optional[Spec] = None,
    slope: float = 0.0,
    intercept: float = 1.0,
    noise: float = 0.0,
    seed: int = 1,
    lo: float = 0.0,
    hi: float = 1.0,
) -> Callable[[PolicyState], float]:
    """The general confidence schedule ``c(state) = clip(intercept + slope·e + noise·z)``.

    ``e`` is the local target error (``error_spec``; treated as 0 when ``None``) and
    ``z`` a per-infoset standard normal.  The named wrappers below pick the regime:

    - ``slope < 0`` — **calibrated**: ``c`` falls as error rises (confident where
      accurate).  The realistic case and the one that lets DBR's confidence weighting
      do its job.
    - ``slope > 0`` — **anti-calibrated / overconfident**: ``c`` rises with error
      (confident where wrong).  The adversarial case.
    - ``slope = 0`` — **flat**: constant ``c``, the no-selective-trust ablation.

    ``noise > 0`` jitters ``c`` per infoset; combined with a calibrated slope it
    manufactures *confidently wrong* infosets (high ``c`` where error is high by chance).
    """
    slope, intercept, noise = float(slope), float(intercept), float(noise)

    def _sched(state: PolicyState) -> float:
        e = _resolve(error_spec, state) if error_spec is not None else 0.0
        c = intercept + slope * e
        if noise > 0.0:
            c += noise * _infoset_normal(state.info_set, seed)
        return float(min(hi, max(lo, c)))

    return _sched


def calibrated(
    error_spec: Spec, *, gain: float = 1.0, noise: float = 0.0, seed: int = 1
) -> Callable[[PolicyState], float]:
    """Confident where accurate: ``c = clip(1 − gain·error(state) + noise·z)``.

    The realistic regime — a real model's confidence and accuracy are both driven by
    the same count ``n(k)``.  ``noise > 0`` adds the *confidently-wrong* tail.
    """
    return confidence_from_error(
        error_spec=error_spec, slope=-abs(gain), intercept=1.0, noise=noise, seed=seed
    )


def anti_calibrated(
    error_spec: Spec, *, base: float = 0.4, gain: float = 1.0, noise: float = 0.0, seed: int = 1
) -> Callable[[PolicyState], float]:
    """Confident where wrong: ``c = clip(base + gain·error(state) + noise·z)``.

    The adversarial regime — the model's confidence signal actively misleads, which is
    exactly where a strong opponent extracts value from bad exploitation.
    """
    return confidence_from_error(
        error_spec=error_spec, slope=abs(gain), intercept=base, noise=noise, seed=seed
    )


def flat(c: float) -> Callable[[PolicyState], float]:
    """Constant confidence — the ablation that removes selective trust.

    A *high* flat ``c`` against a heterogeneous-error model is also the simplest
    "overconfident" condition (it fails to down-weight the high-error infosets).
    """
    return confidence_from_error(intercept=float(c), slope=0.0)


# --------------------------------------------------------------------------- #
# Declarative resolvers (JSON-friendly descriptor → Spec)
# --------------------------------------------------------------------------- #
# The builders above are the programmatic API.  The eval CLI / config, though, can
# only carry data (a JSON string), not a Python callable — so these two functions map
# a small declarative descriptor onto the builders, letting a sweep specify a
# street-graded / noisy error schedule or a calibrated / anti-calibrated confidence
# schedule from the command line (see ``evaluation.opponents.ModelSpec``).  A bare
# number (or ``None``) resolves to the constant scalar, so the scalar path is just the
# degenerate descriptor.


def error_from_spec(spec: Union[float, int, Mapping[str, object], None]) -> Spec:
    """Resolve a JSON-friendly error descriptor into a scalar-or-callable ``Spec``.

    - ``None`` → ``0.0`` (exact model); a bare number → uniform constant error.
    - ``{"kind": "uniform", "e": <float>}`` — constant target ℓ1 error.
    - ``{"kind": "street", "by_round": {"0": .., "3": ..}, "default": <float>}`` —
      error graded by betting round (:func:`street_error`).

    Any descriptor may carry an optional ``"noise"`` sub-map
    ``{"sigma": .., "seed": .., "lo": .., "hi": ..}`` which wraps the resolved base in
    :func:`with_infoset_noise` (per-infoset multiplicative jitter).
    """
    if spec is None:
        return 0.0
    if not isinstance(spec, _Mapping):
        return float(spec)
    kind = str(spec.get("kind", "uniform"))
    if kind == "uniform":
        base: Spec = uniform_error(float(spec.get("e", 0.0)))
    elif kind == "street":
        by_round = dict(spec.get("by_round", {}))
        base = street_error(
            {int(k): float(v) for k, v in by_round.items()},
            default=float(spec.get("default", 0.0)),
        )
    else:
        raise ValueError(
            f"error_from_spec: unknown kind {kind!r} (expected 'uniform' | 'street')"
        )
    noise = spec.get("noise")
    if noise:
        if not isinstance(noise, _Mapping):
            raise ValueError("error_from_spec: 'noise' must be a mapping")
        base = with_infoset_noise(
            base,
            sigma=float(noise.get("sigma", 0.0)),
            seed=int(noise.get("seed", 0)),
            lo=float(noise.get("lo", 0.0)),
            hi=float(noise.get("hi", 2.0)),
        )
    return base


def confidence_from_spec(
    spec: Union[float, int, Mapping[str, object], None],
    *,
    error_spec: Optional[Spec] = None,
) -> Spec:
    """Resolve a JSON-friendly confidence descriptor into a scalar-or-callable ``Spec``.

    - ``None`` → ``1.0``; a bare number → flat constant confidence.
    - ``{"kind": "flat", "c": <float>}`` — constant (the no-selective-trust ablation).
    - ``{"kind": "calibrated", "gain": .., "noise": .., "seed": ..}`` — confident where
      accurate (needs ``error_spec``).
    - ``{"kind": "anti_calibrated", "base": .., "gain": .., "noise": .., "seed": ..}`` —
      confident where wrong (needs ``error_spec``).
    - ``{"kind": "general", "slope": .., "intercept": .., "noise": .., "seed": ..}`` —
      the raw ``c = clip(intercept + slope·error + noise·z)`` form.

    ``calibrated`` / ``anti_calibrated`` scale ``c`` by the local error, so the resolved
    ``error_spec`` (see :func:`error_from_spec`) must be supplied for them; ``flat`` /
    ``general`` ignore it.
    """
    if spec is None:
        return 1.0
    if not isinstance(spec, _Mapping):
        return float(spec)
    kind = str(spec.get("kind", "flat"))
    if kind == "flat":
        return flat(float(spec.get("c", 1.0)))
    if kind == "general":
        return confidence_from_error(
            error_spec=error_spec,
            slope=float(spec.get("slope", 0.0)),
            intercept=float(spec.get("intercept", 1.0)),
            noise=float(spec.get("noise", 0.0)),
            seed=int(spec.get("seed", 1)),
        )
    if kind in ("calibrated", "anti_calibrated"):
        if error_spec is None:
            raise ValueError(
                f"confidence_from_spec: kind {kind!r} needs an error_spec "
                "(it scales c by the local error)"
            )
        if kind == "calibrated":
            return calibrated(
                error_spec,
                gain=float(spec.get("gain", 1.0)),
                noise=float(spec.get("noise", 0.0)),
                seed=int(spec.get("seed", 1)),
            )
        return anti_calibrated(
            error_spec,
            base=float(spec.get("base", 0.4)),
            gain=float(spec.get("gain", 1.0)),
            noise=float(spec.get("noise", 0.0)),
            seed=int(spec.get("seed", 1)),
        )
    raise ValueError(
        f"confidence_from_spec: unknown kind {kind!r} "
        "(expected 'flat' | 'calibrated' | 'anti_calibrated' | 'general')"
    )


__all__ = [
    "Spec",
    "uniform_error",
    "street_error",
    "with_infoset_noise",
    "confidence_from_error",
    "calibrated",
    "anti_calibrated",
    "flat",
    "error_from_spec",
    "confidence_from_spec",
]
