"""Opponent modeling package (docs/opponent_modeling.md — Approach A / CW-RR).

Per-opponent behavioral models over a **coarse** abstraction, decoupled from the
blueprint so a ~10k-game evaluation budget can saturate per-infoset confidence
(doc §3, §4). The public pieces:

- :mod:`poker_ai.modeling.tiers` — the postflop strength-tier table derived from
  the clustering's own LUT centroids (§4.4); the ``s`` dimension of the model key.
- :mod:`poker_ai.modeling.counts` — the coarse-key projection ``π`` (§4.4), the
  4-class action map, and the buffer/commit ``CountsTable`` (§4.2).
- :mod:`poker_ai.modeling.model` — the :class:`OpponentModel` ABC,
  :class:`SyntheticOpponentModel` (exact-known + ℓ1 sweep), and
  :class:`BayesOpponentModel` (per-state prior + coarse-count Dirichlet blend, §3).
- :mod:`poker_ai.modeling.store` — :class:`ModelStore`: the per-opponent registry
  with hand-boundary commit and per-hand ``snapshot`` frozen views (§4.2).
- :mod:`poker_ai.modeling.policy` — :class:`ModelPolicy`, the σ̂-backed drop-in
  ``Policy`` for modeled-seat leaf continuations (§4.3).

These are Phase 1 (the standalone package).  The solver clamp, leaf wiring, and
agent integration (Phases 2–4) consume them without further model-side machinery.
"""
