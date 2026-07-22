# Opponent Modeling & Confidence-Weighted Restricted Response (CW-RR)

Implementation plan for Approach A of
[the research design](safe_exploitation_research_design.md) (§3.1, §4.1, §5):
per-opponent models with per-infoset confidence, exploited inside the existing
real-time search by clamping each modeled opponent seat to a
confidence-weighted mixture of its model and its usual adversarial
(regret-minimizing) self. The mechanism is Data Biased Response
(Johanson & Bowling 2009) transplanted into the multiplayer depth-limited
solver of [subgame_solving.md](subgame_solving.md).

> **SCOPE UPDATE (2026-07-16) — the online-learned model was removed.** The
> exploitation eval injects model quality as a *controlled variable* via
> `SyntheticOpponentModel` (wrap the true opponent policy, dial `error`/`confidence`
> — the shaping lives in `poker_ai/modeling/schedules.py`) rather than learning it; see
> [safe_exploitation_research_design.md](safe_exploitation_research_design.md) §6 and
> [[project_exploitation_eval_design]]. The learned `BayesOpponentModel` and its
> coarse-count machinery (`counts.py`, `store.py`, `tiers.py`) **underperformed the
> blueprint against blueprint-derived opponents and were deleted** — an A/B (visit-weighted
> L1 to the true opponent) showed the tier abstraction's averaging cost exceeded the
> learnable bias postflop, and the hierarchical-backoff variant made it worse. What
> survives in `poker_ai/modeling/`: `model.py` (`OpponentModel` ABC + `SyntheticOpponentModel`),
> `schedules.py`, `policy.py` (`ModelPolicy`). **The CW-RR approach below is unchanged** —
> only the *source* of `σ̂` is now the synthetic oracle. Sections describing the learned
> provider (the `Learned-model`/`Model form`/`Soft counts` rows of §3, §4.1's Bayes bullet,
> §4.2, §4.4) and the online-learner steps (§7 learning-curve bullet, §9 Phase 5) are kept
> only as a **design record** and marked *removed* inline; they are not implemented.

---

## 1. Background

The baseline search agent solves every subgame under the *unsafe-search
worst case*: all seats — the bot's and every opponent's — are
regret-minimizing players ([mccfr.py](../poker_ai/search/mccfr.py),
[vector.py](../poker_ai/search/vector.py)), and beliefs assume opponents play
the bot's own strategy. It never exploits anyone
([subgame_solving.md](subgame_solving.md) §2 non-goals). CW-RR changes exactly
one thing about the world the bot best-responds to: **an opponent seat with a
model plays the mixture**

```
σ̃_j(I) = c_j(I) · σ̂_j(I) + (1 − c_j(I)) · x_j(I)          (A-mix)
```

where `σ̂_j` is the opponent model, `x_j` is the seat's existing
regret-matched free strategy, and `c_j(I) ∈ [0, p_max]` is a per-infoset
confidence. Where we have data the bot best-responds to the model; where we
have none the seat degenerates to the baseline's adversarial player. Safety
is *graded and model-coupled* — the design doc's §4.1 discusses the failure
modes (deception via manufactured confidence, capped by `p_max`) and the
lineage caveat (behavioral per-infoset mixing is DBR, not root-level RNR).

Everything else — nesting, depth limits, leaf meta-game, freezing, budgets —
is untouched. That is what makes the baseline a controlled comparison
(design doc §5).

## 2. Goals and Non-Goals

### Goals

- Per-opponent models `σ̂_j` at blueprint granularity, with per-infoset
  confidence `c_j`. **Provider (as built):** synthetic — ground truth ± a controlled,
  optionally scheduled error, for the design doc's §6.2 sweeps. *(A second, learned-online
  provider was built then removed — see the scope note at top.)*
- The (A-mix) clamp in the **MCCFR regime**, behind a per-search mapping that
  is empty by default — an empty mapping is bit-for-bit the baseline solve
  (condition B0 needs no separate code path).
- The two shared integration points of design doc §5: belief-tracking
  likelihood `σ̂_j` for modeled seats, and model-derived leaf continuation
  policies for modeled seats.
- Conditions B0 (no models), B1 (`c ≡ 1`, naive best response), and A
  (`p_max` grid) runnable from the evaluation runner
  ([evaluation.md](evaluation.md) §10.1) at matched budgets.

### Non-Goals

- **No general multiway Approach B (PO-CES).** The multiway gadget of
  design-doc §4.2 stays a theory-chapter design. Its heads-up instantiation
  **PO-CES-HU** (design-doc §4.2b) *is* planned — as Part II of this document
  (§11), after the Part I (CW-RR) critical path. Part I's belief-likelihood
  swap is the only Part I component B-HU depends on.
- **No safety guarantees.** CW-RR has none by construction; safety is an
  empirical claim measured by the design doc's §6.1 proxy.
- **No vector-regime clamp in v1.** Modeled subgames force the MCCFR regime
  (`force_mccfr_when_modeled`, §6.6) until the matrix blend lands (§9 step 7);
  design doc §5 integration point 4 records the budget consequence.
- ~~**Learned model over a coarser abstraction, not the blueprint's.**~~ **REMOVED**
  (scope note at top). The online `BayesOpponentModel` keyed counts on coarse behavioral
  buckets (§3, §4.4) so a ~10k-game budget could saturate confidence; the blueprint's
  own cluster/infoset granularity (info_set = `(cluster, full betting history)`)
  spreads counts far too thin for `c` to ever leave 0. Synthetic models are
  unaffected. Deviations invisible in the coarse abstraction are invisible to
  the learned model — a *lower* ceiling than the blueprint's, accepted
  deliberately (design doc §8, model-class ceiling; §10).
- **No deception detection.** `p_max` is the only defense; the deception probe
  (design doc §6.2) measures the residual exposure.

## 3. Resolved Design Parameters

| Parameter | Value |
|---|---|
| Mixture | Behavioral, per infoset: `σ̃ = c·σ̂ + (1−c)·x`, `x` = the seat's regret-matched strategy (A-mix) |
| Learned-model abstraction | **(REMOVED — design record; see scope note at top.)** **Coarse behavioral buckets**, decoupled from the blueprint (§4.4). Model key `k(I) = (street, s, ctx)` with `s` = the lossless bucket preflop / a 10-tier equity bin postflop, and `ctx = (n_raises ∈ {0,1,2+}, facing_bet ∈ {0,1})`. Counts and confidence live over `k`; action space collapses to 4 classes `{fold, call/check, raise, all_in}`. The blueprint's own cluster/infoset granularity spreads a ~10k-game budget too thin for `c` to ever leave 0 — this is the fix. **Synthetic models are unaffected** — they wrap an exact policy at blueprint granularity. |
| Model form (learned) | **(REMOVED.)** Per-query shrinkage onto the state's own blueprint row. `prior(a)` = this state's blueprint row collapsed to the 4 classes; `σ̂_coarse(k,a) = (τ·prior(a) + n(k,a)) / (τ + n(k))`; then **expand** to `state.legal_actions` (raise-class mass split across the legal `raise:frac` actions by the blueprint row, renormalized — the §4.3 machinery). `n = 0` recovers the exact per-state blueprint. |
| Confidence | `c_j(I) = min(p_max, n(k) / (n(k) + τ))` over the coarse bucket `k(I)`; unvisited buckets get `c = min(p_max, prior)` with `prior = 0` (fully adversarial where dataless) |
| `p_max` grid | `{0.5, 0.8, 0.95}`; `1.0` with `c ≡ 1` is condition **B1**; default `0.8` |
| Prior strength `τ` | Default `50` observations (shared by model and confidence; separable later if calibration demands) |
| Count attribution | **(REMOVED — no counts without the learned model.)** **Soft counts** at the round-boundary belief replay: an observed action credits each coarse bucket `k(cluster, ctx)` — collapsed to the action's class — proportionally to the tracker's pre-action belief mass over clusters (§6.3) |
| Model freeze cadence | Per hand: the search and belief updates read a hand-start snapshot; observations buffer and commit at hand end (breaks the count↔belief feedback loop within a hand) |
| Belief likelihood | Modeled seats: `σ̂_j` (the model, **not** the mixture — beliefs estimate actual behavior, the mixture is the solver's hedge); bot's own range and unmodeled seats: unchanged baseline (last-search average / blueprint) |
| Leaf continuations | Modeled seat: 4 variants derived from `σ̂_j` (same ×5-reweight machinery); bot and unmodeled seats: blueprint variants, unchanged |
| Regime coverage | v1: MCCFR only, `force_mccfr_when_modeled = True`; v2: vector-regime matrix blend (§6.7) |
| Identity | Models keyed by a caller-supplied stable `opponent_id` (evaluation: the seat's `agent_label` + seat index); persistence via `ModelStore.save/load` (npz) |
| Conditions | B0 = empty model mapping (identical baseline path) · B1 = synthetic model, `c ≡ 1` · A = models + `(p_max, τ)` |

## 4. The model abstraction (`poker_ai/modeling/`)

Top-level package (as built; the learned-model files were removed — see the scope note
at the top of this doc):

```
poker_ai/modeling/
├── model.py     # OpponentModel ABC; SyntheticOpponentModel
├── schedules.py # error/confidence schedules for the synthetic sweep (pure-vs-noisy)
└── policy.py    # ModelPolicy(Policy): σ̂-backed drop-in next to BlueprintPolicy
                 # REMOVED: counts.py, tiers.py, store.py, BayesOpponentModel
```

### 4.1 `OpponentModel` (`model.py`)

```python
class OpponentModel(ABC):
    """One opponent's estimated strategy + per-infoset confidence.

    Both queries take the same PolicyState the search's Policy ABC consumes
    (built via env.policy_state_for(hole, for_blueprint=True), so off-tree
    histories canonicalize exactly as blueprint lookups do)."""

    @abstractmethod
    def strategy(self, state: PolicyState) -> np.ndarray:
        """σ̂ aligned with state.legal_actions (overlay actions get zero mass,
        renormalized — mirrors BlueprintPolicy.strategy)."""

    @abstractmethod
    def confidence(self, state: PolicyState) -> float:
        """c ∈ [0, p_max] at state.info_set."""
```

- `SyntheticOpponentModel(policy, confidence, *, error, seed, bias, p_max)` — the
  **surviving provider** and the instrument for the eval. Wraps any exactly-known
  opponent policy (the evaluation's
  [`BlueprintOpponent`](../evaluation/opponents.py) biased variants), with a
  constant-or-scheduled `c` and a **controlled perturbation** implementing the
  design doc §6.2 model-error sweep (per-infoset probability shift toward a target ℓ1
  distance, seeded). `error` and `confidence` may each be a `callable(state)` schedule;
  `poker_ai/modeling/schedules.py` builds the pure-vs-noisy shapes (uniform / street-graded
  / per-infoset-noisy error; calibrated / anti-calibrated / flat confidence).
- ~~`BayesOpponentModel(tables, counts, tiers)`~~ — **REMOVED** (see scope note at top).
  The learned coarse-bucket Dirichlet posterior underperformed the blueprint postflop
  against blueprint-derived opponents; the eval injects quality via the synthetic provider
  instead.

### 4.2 Counts and the store (`counts.py`, `store.py`) — REMOVED

*Design record only; these files were deleted (scope note at top).* The learned model
kept soft counts over a coarse model key `k = (street, s, ctx)` (4-class width
`{fold, call/check, raise, all_in}`) in a per-`opponent_id` `ModelStore` with
hand-boundary `buffer_observation`/`commit_hand` and npz `save`/`load`. Not needed by the
controlled-injection eval, which supplies `σ̂` from an exactly-known policy.

### 4.3 `ModelPolicy` (`policy.py`)

A `Policy` subclass (same ABC as
[`BlueprintPolicy`](../poker_ai/search/policy.py)) that reads
`model.strategy(state)` as the base distribution and applies the inherited
`_bias_mask` / `_reweight_bias` for the four continuation variants — so a
modeled seat's leaf continuations are "the model, fold-/call-/raise-biased
×5" with zero new machinery (design doc §5, integration point 2).

### 4.4 Strength tiers from the existing LUT (`tiers.py`) — REMOVED

*Design record only; `tiers.py` was deleted (scope note at top).* The learned model's
postflop `s` was a 10-tier equity bin per street, derived from the shipped
`centroids.joblib` (river centroid `[win, loss, tie]` → `win + tie/2`; turn/flop centroids
are histograms over the next street, rolled back to a scalar equity-of-equity and
equal-frequency binned), preflop lossless. Empirically this tiering was the source of the
learned model's postflop underperformance (the bucket-average discards per-state structure
the blueprint keeps), which is why the learned path was dropped in favor of injected
model quality.

## 5. The solver clamp (MCCFR regime)

### 5.1 Plumbing (`context.py`)

`SubgameContext` gains one frozen field, defaulted empty:

```python
models: Mapping[int, OpponentModel] = MappingProxyType({})   # seat → model
```

`SubgameContext.from_runtime(...)` threads it from the agent. **Empty mapping
⇒ no code path below activates** — B0 is the baseline solve, enforced by a
bitwise regression test (§8).

### 5.2 The blend (one seam, `mccfr.py`)

The single strategy-producing seam in the MCCFR regime is
[`_node_sigma`](../poker_ai/search/mccfr.py) (used by `_traverse` for both
opponent sampling and traverser exploration, and by `_update_strategy`).
It becomes:

```python
def _node_sigma(self, env, key, actor, holes):
    sig = self._frozen_or(self.state.sigma(key), key, actor, holes)
    model = self.ctx.models.get(actor)
    if model is None or self._is_actual_bot(actor, holes):
        return sig
    m_sig, c = self._model_row(env, key, actor, holes)   # memoized, §5.3
    return c * m_sig + (1.0 - c) * sig
```

Semantics (the standard restricted-response CFR treatment, design doc §4.1):

- A modeled seat's **realized** strategy — wherever it is sampled *or*
  averaged — is the blend. Values everywhere are computed under the blend.
- Its **regrets** accumulate exactly as before (`add_regret(key, va − ⟨σ̃,va⟩)`
  when it is the traverser): the free component `x` is the regret-match over
  those regrets, i.e. regrets accumulate *on the free component* against
  values under the mixture.
- `strat_sum` accumulates the blend (the realized strategy), so
  `average_policy` remains "what this seat actually played in the solve".
  (For modeled seats that average is no longer consumed by belief updates —
  §6.3 — but it stays correct for diagnostics.)
- The bot's own rows, frozen rows, and unmodeled seats are untouched;
  `c = 0` reproduces `sig` exactly.

`_node_sigma` needs `env` for the model query; both call sites hold it.

### 5.3 Model-row memoization (`solver_state.py`)

`model.strategy(state)` costs an `info_set` construction — the known
per-node hot spot ([subgame_solving.md](subgame_solving.md) §6.4.2 note 5).
The model is frozen for the search's lifetime, so rows are memoized in
`SolverState` exactly like the leaf caches:

```python
model_sigma_cache: _CountingCache   # (seat, public_key, hand_row) → (σ̂ row, c)
```

aligned to `legal_at[public_key]` at insert time (overlay actions zero,
renormalized). Warm-started re-searches within a hand reuse it (the model is
per-hand frozen); the cache dies with the `SolverState` at hand end. Cache
counters ride the existing `_CountingCache` stats plumbing.

### 5.4 Leaf continuations (`leaf.py`)

`LeafConfig` gains

```python
seat_policies: Mapping[int, Mapping[BiasClass, Policy]] = MappingProxyType({})
```

and `continuation_value` resolves each acting seat's policy as
`cfg.seat_policies.get(seat, cfg.policies)[profile[seat]]` — modeled seats
roll out under `ModelPolicy` variants, everyone else under the blueprint
variants, unchanged. The leaf-value cache key already contains the profile
and holes; `seat_policies` is search-static, so no key change is needed.
Without this, exploitation is silently truncated at the depth limit (design
doc §5, integration point 2).

## 6. Agent integration (`agent.py`) and regime gating

### 6.1 Construction

`SearchAgent.__init__` gains `model_store: Optional[ModelStore]` and
`opponent_ids: Mapping[int, str]` (seat → stable ID, supplied by the
runner). `on_hand_start` snapshots `models = {seat: store.snapshot(id)}`
for live opponent seats — the per-hand frozen view.

### 6.2 Solve wiring

`_solve_and_store` passes `models` into `SubgameContext.from_runtime` and
builds the per-seat `seat_policies` for the `LeafConfig` (four `ModelPolicy`
variants per modeled seat).

### 6.3 Belief likelihood (soft/hard counts REMOVED with the learned model)

[`_make_sigma_for_combo`](../poker_ai/search/agent.py) becomes per-seat (the
belief-likelihood swap is unchanged — `σ̂` is now supplied by `SyntheticOpponentModel`):

- **Modeled opponent seat:** the closure queries the hand-start model
  snapshot — `model.strategy(env_before.policy_state_for(combo, ...))` —
  vectorized by cluster exactly as the blueprint path already is. This is
  design doc §5 integration point 1: reach beliefs and behavioral models
  agree. (The baseline used the last search's average policy here; that
  path remains for the bot's own range and unmodeled seats.)
  *(The learned-model "resolution caveat" — `σ̂` constant within a tier blunting
  beliefs — no longer applies: `SyntheticOpponentModel` wraps an exact policy at
  blueprint granularity, so its `σ̂` discriminates within a tier.)*
- ~~**Soft counts, same replay**~~ and ~~**Showdown hard counts**~~ — **REMOVED** with
  the learned model (scope note at top). There is no `store.buffer_observation` /
  `commit_hand` path; the boundary belief replay still updates ranges, it just no longer
  emits counts.

### 6.4 Regime gating (`solver.py`)

`_select_regime` gains the v1 gate: `ctx.models` non-empty ⇒ MCCFR,
regardless of size/street (`force_mccfr_when_modeled`, `SolverConfig`
field). Logged per solve so matched-budget analyses can slice on it. Removed
when §6.7 lands.

### 6.5 What does not change

Freezing (bot's actual hand only), warm starts, the off-tree
inject/re-search machinery, the dual budget, final-iterate play for the bot,
`SearchPolicy` readers. CW-RR is deliberately confined to *what the opponent
seats play inside the solve* plus the two §5-doc integration points.

### 6.6 Interim cost note

Forcing MCCFR on modeled late/heads-up subgames trades the vector regime's
exactness for generality; at matched wall-clock this is a real handicap for
conditions A/B1 relative to B0 in exactly those subgames, and it is *the
baseline's* regimes that define B0. Report the fraction of solves gated, and
prioritize §6.7 if it is material. **In 4-handed play this is milder than it
reads:** the vector regime only applies once a subgame is heads-up (turn/river),
so with three opponents most subgames are already multiway/MCCFR in the
baseline — the gated fraction is small, and §6.7 is a lower priority than the
heads-up framing implies. Measure it before investing.

### 6.7 Vector-regime blend (v2)

The vector seam is one line-cluster in
[`_walk`](../poker_ai/search/vector.py): after
`sigma = _regret_match_matrix(regret)`, an acting modeled seat blends
matrices — `Σ̃ = C ⊙ Σ̂ + (1 − C) ⊙ Σ` with `Σ̂ : (n_combos, width)` the
model rows (expanded cluster→combo, cached per `public_key` on first visit,
like `legal_at`) and `C : (n_combos, 1)` the per-combo confidence. Regret
and strat-sum updates are already written against the produced `sigma`, so
they inherit the blend unchanged. The turn subgame's river-conditioned 3-D
nodes need `Σ̂` per `(combo, river)` — build lazily per sampled river slice
to avoid a `(n_combos, n_rivers, width)` precompute.

## 7. Evaluation integration ([evaluation.md](evaluation.md))

- **Conditions & budget.** Three headline approaches, **10k hands each**
  (per condition, *not* a shared 10k): **vanilla Pluribus** (blueprint-only, no
  search), **B0** (search, empty `model_store` — adversarial, no exploitation),
  **A** (search + models, over the `(p_max, τ)` grid). Optional **B1**
  (`SyntheticOpponentModel` of the true opponent, `c ≡ 1`) as the naive-BR
  ceiling. All at matched search budgets (design doc §6.3); convergence curves
  logged, not just endpoints.
  **The no-exploitation agent is *always* the baseline** — pure Pluribus (B0 =
  search + empty model store at full scale; blueprint-only in the no-search
  proxy). Every exploitation number is the **CRN-paired advantage over this
  baseline** (the incremental value of exploiting); A, B, B1 and each
  model-error sweep point are all differenced against the *same* B0 arm, never
  reported as a bare EV. (design doc §6.1/§6.3.)
- **Common random numbers across conditions (the free variance lever).** Every
  condition replays the **same per-hand deal seed** — hole cards, board, *and*
  the opponent-action RNG, keyed by hand index — so the shared-deal luck cancels
  when you difference `A − B0` (and `B0 − vanilla`) per hand. At ~10k hands in
  4-handed NLHE this beats any other single lever and costs **zero extra
  compute**: it is the same 10k deals, seeded identically, not more hands. It
  stacks with AIVAT ([evaluation.md](evaluation.md) §10.2) — report the CI on the
  **paired difference** of `aivat_value` (bootstrapped), not on each arm
  separately. The infra is specced in [evaluation.md](evaluation.md) §10.1
  ("Cross-condition pairing") + §9 step 10: arms share `run_seed`/`table_policy`
  so the existing `deck_seed` matches per hand and is the join key, plus a
  `max_hands` fixed-count mode (time-budget mode desyncs arms) and a `condition`
  label. Independent of this modeling package, so it can land first.
- ~~**Learning curve for free from the A run.**~~ **REMOVED** with the online learner
  (scope note at top): there is no `BayesOpponentModel` accumulating counts to bin by. The
  "how good must the model be" question it approximated is answered *directly* by the
  synthetic model-error sweep (design doc §6.2) — inject a known error level and read the
  paired advantage, instead of inferring quality from opaque cumulative counts.
- **Coverage-restricted reporting.** The `A − B0` signal lives only in hands
  where a modeled seat actually acted with `c > 0`; report the headline both
  overall **and restricted to modeled-decision hands** (mirrors §11.4's
  HU-coverage slicing), which concentrates the per-hand effect size and buys
  power at fixed hand count. Log per hand whether any modeled seat acted with
  `c` above a threshold.
- **True opponents.** The runner's
  [`BlueprintOpponent`](../evaluation/opponents.py) variants are exactly
  known, so synthetic models of them are exact by construction; the
  `Pr_shuffle` perturbed opponents of design doc §6.2 are a new opponent
  label (per-infoset seeded shuffle of the blueprint σ) added to
  `LABEL_TO_BIAS`'s registry pattern — runner and schema are already
  agnostic to new labels.
- **Deception probe.** A scripted opponent that plays its advertised model
  for the observation phase and then switches to a fixed exploit policy —
  an opponent-side agent, no solver involvement.
- **Schema.** One new table (`opponent_models`: run, hand, seat,
  opponent_id, mean/max `c` over visited infosets, model↔blueprint ℓ1,
  count mass) plus per-solve columns (regime, gated-by-models flag,
  `model_sigma_cache` hit rate) and per-hand columns (`condition` for the arm
  label and the CRN join — the deal join key is the existing `deck_seed` —
  `model_snapshot_id` and per-opponent cumulative-count for the learning-curve
  binning, a `modeled_decision` flag for coverage-restricted reporting). Extends
  [evaluation.md](evaluation.md) §6 additively; no existing-column changes.
- **Safety proxy.** The §6.1-design-doc unilateral BR gain is evaluation-side
  work (exact BR in the small game), independent of this plan; this plan only
  guarantees the conditions it compares are runnable and logged.

## 8. Testing

Unit-first (fast, no functional pipeline in the iteration loop):

1. **Baseline equivalence (the load-bearing test):** empty `models` ⇒
   bitwise-identical `SolverState` to the pre-change solver on a seeded
   small-deck solve, and `ctx.models` absent from every hot path branch.
2. **Blend math:** `c = 0` returns the regret-matched σ exactly; `c = 1`
   returns the model row; overlay-injected actions get zero model mass and
   renormalize; frozen bot rows never blend.
3. **Tiers & counts:** *(tiers, §4.4)* rolled-back equity is monotone across a
   hand-built toy centroid set, every cluster maps to a tier in `[0, n)`, and
   equal-frequency bins are balanced on the real `data/20cards_exact`
   centroids. *(counts)* one observation distributes exactly unit mass across
   the coarse buckets it projects to; commit-at-hand-end means mid-hand queries
   see the snapshot; the per-state posterior recovers the **exact blueprint at
   `n = 0`** and the empirical 4-class frequencies as `n → ∞`; the raise-class
   expansion preserves total mass and matches the blueprint raise split; `c`
   schedule hits `p_max` monotonically.
4. **Model policy:** bias reweighting on σ̂ matches `BlueprintPolicy`'s
   transform on the same row; `ModelPolicy` slots into `LeafConfig`
   per-seat resolution.
5. **Determinism:** seeded solve with models is reproducible; the
   `model_sigma_cache` stores first-draw rows only.
6. **Functional (slow-marked, not in the iteration loop):** 3-seat
   small-deck game, one heavily fold-biased opponent, exact synthetic model:
   condition A's search EV against that table ≥ B0's, and B1 ≥ A at zero
   model error; with a maximally wrong model, A's EV degrades gracefully
   toward B0 as `p_max` shrinks.

## 9. Implementation Steps

Each step lands green before the next starts; steps 1–5 are the v1 critical
path, 6–7 are follow-ups.

1. `poker_ai/modeling/` package: `OpponentModel`, `SyntheticOpponentModel`,
   `schedules.py` (error/confidence shaping), `ModelPolicy` + unit tests (§8.2–4).
   *(The learned pieces — `tiers.py`, `counts.py`, `BayesOpponentModel`, `ModelStore` —
   were built then **removed**; see the scope note at top.)*
2. `SubgameContext.models` + the `_node_sigma` blend + `model_sigma_cache`
   + the baseline-equivalence regression (§8.1) and blend tests.
3. `LeafConfig.seat_policies` + leaf resolution + tests.
4. Agent wiring: snapshots, per-seat belief likelihood,
   `force_mccfr_when_modeled` gate + determinism tests. *(Soft-count buffering removed
   with the learned model.)*
5. Evaluation: `Pr_shuffle` opponent label, conditions vanilla/B0/A (+optional
   B1) in the runner, **cross-condition CRN** (built per
   [evaluation.md](evaluation.md) §9 step 10 — `max_hands` + shared
   `run_seed`/`deck_seed` + `condition`; can land first, independent of steps
   1–4), paired-difference AIVAT reporting, coverage-restricted slicing, schema
   additions, the functional sanity test (§8.6).
6. ~~`BayesOpponentModel` online learner + showdown hard counts~~ — **REMOVED** (scope
   note at top); the "how good must the model be" question is answered by the §6.2
   synthetic error sweep instead of a learning curve.
7. Vector-regime blend (§6.7); retire the regime gate.

## 10. Risks and Open Questions (Part I)

*The first three risks below (model granularity ceiling, bet-sizing tells, soft-count
circularity) were **specific to the removed learned model** and no longer bind — the
synthetic provider is at blueprint granularity, sizing-aware, and count-free. Kept as
design record.*

- **Model granularity ceiling.** The learned model is *coarser* than the
  blueprint (10 equity tiers × coarse betting context, §4.4), so it can only
  express deviations visible at that resolution — a lower ceiling than the
  blueprint's (design doc §8), traded deliberately for count saturation at the
  ~10k-game budget. Do not silently change the tier count or context dimensions
  in one direction without re-checking saturation vs. expressiveness; a *finer*
  model class (e.g. back to blueprint clusters) is a design change and will
  starve `c`. Synthetic models keep blueprint granularity and are unaffected.
- **Bet-sizing tells are invisible to the learned model.** The 4-class collapse
  (§3) lumps every `raise:frac` into one "raise" class and re-splits it across
  sizes by the *blueprint's* proportions on expand, so the learned model can
  never express "this opponent sizes differently." This is exact for the
  headline: the eval opponents (`bp_fold/call/raise`) deviate in action
  *frequency*, not sizing, so 4 classes capture their whole tell. But it is a
  hard ceiling against sizing-based opponents — accepted for v1, a class change
  (per-size counts) if a later opponent needs it. Synthetic models are exempt.
- **Soft-count circularity.** Counts are attributed under beliefs that were
  themselves filtered under the model. The per-hand freeze breaks the loop
  within a hand but not across hands; miscalibration compounding across
  hands is exactly what the deliberately-miscalibrated-`c` arm of the sweep
  (design doc §6.2) measures. Showdown hard counts (§6.3) are the
  correction lever if it bites.
- **Confidence at re-visited vs fresh infosets.** `c` is count-driven and
  street-local; an opponent seen often preflop but never on rivers is
  exploited early and defended against late — intended behavior, but it
  makes headline EV sensitive to hand-count budgets; report counts.
- **Blend sampling variance.** The mixture adds no new sampling step (the
  blend is computed, then one action is sampled as before), so variance
  impact should be second-order; confirm on the §8.6 functional before
  trusting matched-budget comparisons.
- **`p_max` is an exploitation knob.** That is the RNR/SES-family property
  OX-Search criticizes and part of what the experiments test (design doc
  §4.1); it is a feature of the comparison, not a bug of this plan.
- **The behavioral-vs-realization-plan caveat.** (A-mix) is DBR-style
  behavioral mixing; its adversary is strictly weaker than RNR's root-level
  mixture (design doc §4.1). Verify the DBR formulation against
  Johanson & Bowling 2009 before citing the correspondence as an identity
  (design doc §7 item 4).

---

## 11. Part II — PO-CES-HU (Approach B, heads-up instantiation)

The paper-verbatim OX-Search gadget deployed in the baseline's **heads-up
turn/river (vector-regime) subgames**, per design-doc §4.2b: once the pot is
heads-up the subgame is constant-sum, so the 2p0s gadget, and the paper's
Theorems 4.4–4.6, apply unmodified inside it. Reach-only exploitation: the
opponent model enters solely through the tracked beliefs ($\hat p$), so Part II
consumes exactly one Part I component — the belief-likelihood swap (§6.3) — and
none of the confidence/mixture machinery.

### 11.1 Components

- **Settlement correctness** *(landed)*: `PokerEnv.vector_payout` /
  `range_showdown.showdown_cfv` settle **dead money** (folded seats'
  contributions) to the winner, `v = stake·(W−L) + dead·(W + T/2)`; fold
  terminals pay the winner `stake + dead`. Regression-tested against the
  concrete engine settlement with a folded third seat
  (`test_payout_consistency.py::test_vector_payout_includes_dead_money`).
  Precondition for the constant-sum argument and for exact references.
- **HU coverage logging** *(landed, §11.4)*: per-hand earliest street at which
  the hand was heads-up-with-hero at a round start — recorded for every
  condition so coverage-restricted comparisons are possible.
- **Reference pass** (`poker_ai/search/reference.py`, new): per-combo
  $CBV_{ref}[c]$ — the opponent's best-response CFV against the bot playing
  the *blueprint* inside the subgame. A vector-walk variant: bot rows fixed
  from `BlueprintPolicy` (queried per cluster, expanded to combos, exactly as
  the belief update vectorizes σ queries), opponent takes a per-combo max over
  actions; rivers enumerated (≤ ~46) on turn subgames. Exact — these subgames
  have terminal leaves only. The dominant new component.
- **Gadget root** (`vector.py`): a per-combo opt-out regret row (width 2:
  *enter*/*out*) for the opponent; opponent root reach becomes
  $\tfrac{1}{k\beta+1}\hat p + \tfrac{\beta}{k\beta+1}q$ with $q$ the
  regret-matched enter-probabilities; opt-out regrets updated from the root
  CFVs minus $CBV_{ref}$. Nothing inside `_walk` changes — the per-entry-combo
  shift cancels in every interior regret delta (per-combo rows) and in the
  bot's regrets (counterfactual values carry full-profile continuations).
  `β` in `SolverConfig`; per-solve logging of opt-out saturation (the paper's
  Theorem 4.5 diagnostic: enter-probability ≈ 1 anywhere ⇒ raise β).
- **Agent wiring**: gate on the existing vector-regime condition (heads-up
  turn/river) plus a config flag; the bot **plays the weighted-average
  strategy** of gadget solves (design-doc §5 — the guarantees attach to the
  average, `SearchPolicy(use_average=True)` already exists); belief-likelihood
  swap shared with Part I.

### 11.2 Non-goals (Part II)

Flop-HU subgames (MCCFR regime in the baseline) — deferred. Action
exploitation (behavioral models inside the solve) — would leave the paper's
guarantee envelope; deferred with design-doc §4.2b. The nested-anchor
convention follows design-doc §5 (blueprint anchor, flagged decision).

### 11.3 Build steps (continuing §9)

8. *(landed)* Dead-money settlement fix + regression tests.
9. *(landed)* HU coverage logging in the evaluation (§11.4).
10. Reference pass + unit tests (against a brute-force BR on a tiny deck; and
    `CBV_ref` under the blueprint must reproduce zero margins for the
    blueprint strategy itself — the `F(σ_i)` feasibility check).
11. Gadget root + tests: `β → ∞`-behavior sanity (safety branch dominates ⇒
    resolving-like), `\hat p`-only sanity (β = 0 ⇒ unsafe search), opt-out
    saturation logging, and a small-game end-to-end check that margins
    $CBV^{σ'} − CBV_{ref} ≤ Δ/β$ within solve error.
12. Agent gating + average-play flag + evaluation condition (B) wiring,
    coverage-restricted reporting in `summarize.py`.

### 11.4 HU coverage logging (evaluation)

Per hand, the earliest street (0–3, else NULL) at which a betting round began
with **exactly two active seats, the hero among them** — the earliest point a
§4.2b solve could activate. Logged unconditionally (every condition, including
blueprint-only), since coverage is a property of the play trajectory, not of
the method under test; B-HU eligibility is the derived predicate
`hu_from_street ≥ 2`. Reported by the run summary as the fraction of hands
(and, secondarily, hero decisions) covered, sliced per condition.
