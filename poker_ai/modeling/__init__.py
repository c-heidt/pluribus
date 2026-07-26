"""Opponent modeling package — DBR (Data-Biased Response; docs call it Approach A / CW-RR).

Pared down to the **synthetic-oracle** path the exploitation eval uses: model quality is
*injected* as a controlled variable, not learned (see
``docs/safe_exploitation_research_design.md`` §6). The public pieces:

- :mod:`poker_ai.modeling.model` — the :class:`OpponentModel` ABC and
  :class:`SyntheticOpponentModel` (wraps an exactly-known opponent policy with a
  scheduled confidence and a controlled ℓ1 perturbation).
- :mod:`poker_ai.modeling.schedules` — error/confidence *schedules* shaping how model
  quality varies across infosets (the pure-vs-noisy axis: uniform / street-graded /
  per-infoset-noisy error; calibrated / anti-calibrated / flat confidence).
- :mod:`poker_ai.modeling.policy` — :class:`ModelPolicy`, the σ̂-backed drop-in
  ``Policy`` for modeled-seat leaf continuations (§4.3).

The online-learned model (``BayesOpponentModel`` + the coarse-count ``tiers``/``counts``/
``store`` machinery) was removed: it underperformed the blueprint against
blueprint-derived opponents and is not needed for the controlled-injection eval.
"""
