# Subgame Solving & Online Play

Implementation plan for the real-time search component of the Pluribus bot.

---

## 1. Background

The repository already contains the offline half of Pluribus:

- External-sampling Linear MCCFR with CFR-P pruning ([poker_ai/blueprint/](../poker_ai/blueprint/)),
- A no-limit Texas hold'em engine with a discrete pot-relative bet-size abstraction ([environment/poker_env.py](../environment/poker_env.py)),
- Expected-hand-strength (EHS) k-means card clustering with a memmap-backed lookup ([information_abstraction/](../information_abstraction/)),
- Chunked, atomic checkpointing of per-street regret and visit tables ([poker_ai/tables/checkpoint.py](../poker_ai/tables/checkpoint.py)),
- A terminal play loop that queries the blueprint offline ([poker_ai/terminal/runner.py](../poker_ai/terminal/runner.py)).

The play loop uses the blueprint for every decision. Brown & Sandholm (2019)
report that blueprint-only play is materially weaker than the full system; the
strength of Pluribus at 6-max NLHE comes from **depth-limited subgame solving
with continuation strategies**, invoked from round 2 onwards and whenever the
game deviates from the action abstraction. Adding this online component is the
scope of this document.

## 2. Goals and Non-Goals

### Goals

- Produce a bot that, at play time, performs real-time search rooted at the
  current public state rather than reading the blueprint directly.
- Match the paper's algorithmic structure: biased continuation strategies at
  depth-limit leaves, unsafe subgame solving on first search in a hand, Linear
  MCCFR on subsequent searches in the same hand.
- Deliver in two phases so end-to-end search can be validated before the
  long-running biased-blueprint training finishes.

### Non-goals

- No changes to the action abstraction or the card clustering.
- No multi-CPU or distributed search. Search runs on one machine per decision.
- No opponent modelling or exploitation; Pluribus plays the same strategy
  regardless of opponent identity, and this project preserves that.
- No safety guarantees. Safe subgame-solving theorems do not extend beyond
  two-player zero-sum, and the paper does not claim them either.

## 3. Resolved Design Parameters

| Parameter | Value |
|---|---|
| Number of continuation strategies | k = 4 (base, fold-biased, call-biased, raise-biased) |
| Subgame depth limit | End of the current betting round |
| Opponent range representation | Dense 1326-combo per opponent, board-conflicting combos zeroed |
| Search budget | Dual cap: iteration count AND wall-clock; whichever hits first |
| Default search budget | 10 000 iterations, 15 s wall-clock (configurable via CLI) |

## 4. Biased Blueprints: Definition and Training

The k = 4 continuation strategies are **full, independent strategies over the
entire abstracted game tree**. Each is stored as its own set of regret and
visit tables, in the same joblib format as the base blueprint and queryable by
the same code.

They are produced by running MCCFR with a **modified terminal payoff**: a
bonus `b` is added, proportional to how often the biased action class
(fold / call / raise) was played along the trajectory to that terminal. The
regret updates therefore see this modified utility, so the resulting strategy
both over-uses the biased action class *and* adapts the rest of the tree to
that tendency. For example, a fold-biased blueprint folds more often and also
plays its non-folded hands differently, because it has effectively solved a
game in which continuing is costlier.

This downstream adaptation is why the biased blueprints cannot be produced by
reweighting the base blueprint at inference time.

### Training strategy

- **Warm-start** each biased variant from the finished base blueprint's
  regret/visit tables, then run biased MCCFR for a shorter schedule
  (starting target: 10–25 % of the base-blueprint iteration count per
  variant; tuned empirically).
- Reuse the existing training pipeline in [poker_ai/blueprint/](../poker_ai/blueprint/):
  `CFRTables`, checkpointing, discount/pruning schedule, multi-process server.
- Add a pluggable **bias hook** that modifies the terminal utility returned to
  the traversal. The base MCCFR path remains identical when `bias=none`.

### New CLI surface

```
poker_ai train start --bias {none,fold,call,raise} \
                     --bias-magnitude <b> \
                     --warm-start <path-to-base-blueprint>
```

### Outputs

Four directories in the same format as the base blueprint:

```
blueprints/
├── base/
├── fold_biased/
├── call_biased/
└── raise_biased/
```

### Files to add

- `poker_ai/blueprint/bias.py` — trajectory-class counter and terminal-utility
  hook.
- Extensions to [poker_ai/blueprint/runner.py](../poker_ai/blueprint/runner.py) for the new flags.
- Extensions to [poker_ai/blueprint/cfr.py](../poker_ai/blueprint/cfr.py) to call the hook at terminal evaluation.

## 5. Two-Phase Rollout

### Phase 1 — On-demand bias (no extra training)

At subgame leaves, derive biased continuation strategies at runtime by
re-running regret matching against the base blueprint's stored regrets with an
additive bias term on the target action class:

```
σ_bias(a) ∝ max(0, R(a) + b · 𝟙[a ∈ biased_class])
```

This is strictly weaker than the paper's method because it lacks the
downstream adaptation, but it is zero-training-cost and lets the entire search
pipeline (sections 6–8 below) be implemented and validated end-to-end.

### Phase 2 — Precomputed biased blueprints

Once the four biased blueprints from section 4 finish training, replace the
Phase 1 leaf-EV source with lookups into the precomputed blueprints. The
leaf-EV interface is unchanged; only the source of `σ_bias` differs. Both modes
remain selectable via CLI (`--biased-blueprints` flag present or absent).

### Rationale for order

Biased blueprint training (section 4) is long-running. It is implemented first
so those runs can start immediately and execute in the background while the
online-search components are built.

## 6. Architecture

New top-level package: `poker_ai/search/`.

```
poker_ai/search/
├── subgame.py        # Subgame construction from current public state
├── ranges.py         # Per-opponent range tracking (dense per-combo)
├── translation.py    # Off-tree size mapping + in-tree injection
├── leaf.py           # Depth-limit leaf continuation-value evaluation
├── solver.py         # Depth-limited MCCFR subgame solver
├── policy.py         # Policy Protocol, Range/BiasClass aliases, implementations
└── agent.py          # Search-aware play agent
```

### 6.0 Policy interface (`policy.py`)

Card-space dimensions depend on the deck the environment was built for
(small decks are used in tests and for sub-game LUTs), so they are not
hard-coded in the search package. The environment exposes them and the
search package consumes whatever the live `PokerEnv` reports.

**New on `PokerEnv`** (to add in [environment/poker_env.py](../environment/poker_env.py)):

```python
@property
def n_combos(self) -> int:
    """Number of distinct unordered hole-card combos: C(deck_size, 2)."""

@property
def combo_cards(self) -> np.ndarray:
    """Shape (n_combos, 2), int32 card ids; row i = (c0, c1) with c0 < c1.
    Cached on the class — same for every instance sharing a deck size."""

@property
def combo_index(self) -> Dict[Tuple[int, int], int]:
    """Inverse of `combo_cards`: (c0, c1) → row index. Cached likewise."""
```

The env already exposes [`deck_size`](../environment/poker_env.py#L736-L739),
[`low_card_rank` / `high_card_rank`](../environment/poker_env.py#L741-L749),
and [`n_players`](../environment/poker_env.py#L679-L682), which are the
inputs `n_combos` / `combo_cards` are derived from; none of the
combo-indexing helpers exist yet.

`poker_ai/search/policy.py` owns the policy abstraction and the
shared aliases the rest of the search package consumes:

```python
BiasClass = Literal["none", "fold", "call", "raise"]

class Policy(ABC):
    """Base class for anything the solver / leaf-EV can query for an
    action distribution. Subclasses implement `strategy`; shared
    regret-matching and bias-mask construction live here so the three
    tabular implementations (blueprint / biased-blueprint / search) do
    not re-derive them."""

    @abstractmethod
    def strategy(self, env: PokerEnv, bias: BiasClass = "none") -> np.ndarray:
        """Float32 vector aligned with `[a for a in env.legal_actions if a is not None]`."""

    @staticmethod
    def _bias_mask(legal_actions: List[str], bias: BiasClass) -> np.ndarray:
        """Boolean mask over `legal_actions` selecting the biased class.
        Action-class identification by prefix: `"fold"`, `"call"`/`"check"`,
        anything starting with `"raise"` or `"all_in"`."""

    @staticmethod
    def _regret_match(
        regrets: np.ndarray,
        bias_mask: np.ndarray,
        bias_magnitude: float,
    ) -> np.ndarray:
        """σ(a) ∝ max(0, R(a) + b · bias_mask). Uniform fallback when all
        biased regrets are ≤ 0."""
```

ABC over Protocol because the three implementations share real
behavior (regret matching + bias-mask construction + uniform fallback),
not just a signature; the runtime enforcement also catches half-finished
subclasses before they reach the solver. The cost is a single
inheritance hierarchy — acceptable given the closed set of tabular
implementations.

Three implementations, only the first of which is shipped today:

- `BlueprintPolicy(tables: CFRTables, bias_magnitude: float = 0.0)` —
  **implemented** in [poker_ai/search/policy.py](../poker_ai/search/policy.py).
  Reads the regret row for `env.info_set` at `env.betting_round`,
  optionally adds `b · 𝟙[a ∈ biased_class]` before regret matching;
  falls back to uniform when the row is absent (unseen info set).
  Action classes are identified via prefix: `"fold"`, `"call"` /
  `"check"`, anything starting with `"raise"` or `"all_in"`.
- `BiasedBlueprintPolicy({bias: CFRTables})` — **deferred** until the
  biased blueprint training in §4 produces the four `CFRTables`
  artifacts.  Phase 2 only; dispatches by `bias` to the matching
  precomputed tables, with no additive bias term.  Subclasses `Policy`
  and reuses the same `_bias_mask` / `_regret_match_with_bias`
  helpers; only the regret-row source differs from `BlueprintPolicy`.
- `SearchPolicy(in_memory_regret: Dict[str, np.ndarray], ...)` —
  **deferred** until the solver in §6.5 lands.  Returned by `solve()`;
  same regret-matching as `BlueprintPolicy`, but reads the subgame-local
  dict instead of a `CFRTables`.

Stubbing the two deferred classes now would add code with no callers
and no way to test against a real regret source; they slot into the
existing ABC unchanged once their inputs exist.

### 6.1 Subgame construction (`subgame.py`)

```python
@dataclass
class SubgameRoot:
    env: PokerEnv                           # deepcopy of runtime env at decision point
    ranges: Dict[int, Range]                # per seat; bot's own range = δ on real hole
    street_at_root: int                     # 0..3 — halt when env.betting_round > this
    extra_actions: Dict[str, List[str]]     # info_set → off-tree actions injected (§6.3)

def build_subgame(
    runtime_env: PokerEnv,
    ranges: Dict[int, Range],
    extra_actions: Optional[Dict[str, List[str]]] = None,
) -> SubgameRoot: ...

def legal_actions_at(root: SubgameRoot, env: PokerEnv) -> List[str]:
    """env.legal_actions (filtered, None removed) + extra_actions.get(env.info_set, [])."""
```

Traversal uses `copy.deepcopy(env).apply_action(a)` so dynamics remain
identical to blueprint training. The depth limit is checked by comparing
`env.betting_round` to `street_at_root` after each `apply_action`; crossing
the boundary hands control to `leaf.leaf_value` (§6.4) instead of recursing.

At the root, for each opponent seat the solver will sample hole cards from
that seat's range, restricted to combos compatible with the current board
(`COMBO_CARDS[i]` shares no element with `env.community_cards`) and
non-conflicting across seats. Precompute the board-conflict mask once per
root to avoid per-iteration filtering.

**Invariants** (asserted in tests): `subgame.env.pot_size`,
`subgame.env.stacks`, internal history/action log, and
`subgame.env.current_player` match the runtime values. Every leaf satisfies
`is_terminal or betting_round > street_at_root`.

### 6.2 Opponent range tracking (`ranges.py`)

```python
Range = np.ndarray                                  # float32, shape (env.n_combos,)

class RangeTracker:
    def __init__(self, n_seats: int, my_seat: int, my_hole: Tuple[int, int]): ...

    def on_board_update(self, new_cards: Sequence[int]) -> None:
        """Zero every combo sharing a card with the new board; renormalize."""

    def on_action(
        self,
        seat: int,
        env_before: PokerEnv,
        action: str,
        policy_used: Policy,
    ) -> None:
        """w(h) ← w(h) · σ(env_before | opp hole = h, policy_used)(action); renormalize."""

    def range_of(self, seat: int) -> Range: ...
    def snapshot(self) -> Dict[int, Range]: ...   # deep copy for the solver
```

Implementation notes:

- `σ(h, I)(a)` for every combo `h` in seat's range: clone `env_before`,
  patch the opponent's hole cards to `COMBO_CARDS[h]`, read `env.info_set`
  (which includes the card-cluster from the LUT), call
  `policy_used.strategy(env_patched)`, pick the index for `action`.
- Vectorize over the card cluster, not over individual combos: combos that
  map to the same cluster-id on this street have identical `env.info_set`
  and therefore identical `σ`. Group combos by cluster once per call, query
  `σ` once per distinct cluster.
- Numerical floor: if `w.sum() < 1e-12` after an update, reset to uniform
  over board-compatible combos and emit a warning (only reachable via
  zero-probability observations, e.g. off-tree bets before injection kicks
  in).
- A per-decision log `List[Tuple[seat, info_set_before, action, policy_id]]`
  is retained for debugging and Phase 1↔Phase 2 replay comparisons.

### 6.3 Action translation and off-tree action injection (`translation.py`)

```python
def canonical_raise_fractions(env: PokerEnv) -> List[float]:
    """The current street's raise-size abstraction, as fractions of pot."""

class Classification(Enum):
    ON_TREE   = auto()   # exact match with an abstraction action
    NEAR_TREE = auto()   # |Δfrac| / f_nearest ≤ tol → snap
    OFF_TREE  = auto()   # > tol → inject as extra action

def classify_observed(
    env_before: PokerEnv,
    chip_amount: int,
    tol: float = 0.15,
) -> Tuple[Classification, str]:
    """Second return is either the abstraction action string (ON/NEAR_TREE)
    or the injected action string, e.g. 'raise:0.42' (OFF_TREE)."""

def abstract_to_chips(env: PokerEnv, action: str) -> int:
    """Inverse mapping used when the bot emits an abstract action."""
```

- Tolerance is relative pot-fraction distance: `|f_obs − f_near| / f_near`.
  Default `0.15`; CLI flag `--off-tree-tol`.
- Fold / call / all-in are always on-tree — only raise sizes can deviate.
- `OFF_TREE` injection appends the action string to
  `SubgameRoot.extra_actions[env_before.info_set]`. Inside the solver,
  `legal_actions_at(root, env)` makes the injected action a first-class
  choice with its own regret slot; all other tree machinery is unchanged.

### 6.4 Leaf continuation-value evaluation (`leaf.py`)

```python
@dataclass
class LeafConfig:
    policies: Dict[BiasClass, Policy]     # one entry per k = 4 bias class
    n_rollouts: int = 20
    rng: np.random.Generator

def leaf_value(
    env: PokerEnv,                        # state AT the leaf (betting_round > street_at_root)
    ranges: Dict[int, Range],             # live ranges conditioned on path taken
    cfg: LeafConfig,
) -> np.ndarray:                          # shape (n_seats,), expected chips won/lost
```

Algorithm per call:

1. Repeat `n_rollouts` times:
   a. For each seat `i`, sample a continuation-strategy choice
      `c_i ∈ {"none","fold","call","raise"}` uniformly.
   b. Sample each opponent's hole from its range (board-compatible and
      mutually non-conflicting; rejection sample or precomputed joint
      index).
   c. Roll the hand forward: at every decision node, sample from
      `cfg.policies[c_i].strategy(env, bias=c_i)` via
      `tree_utils.sample_action`.
   d. At the terminal state, compute per-seat payoff with
      `environment.evaluator.Evaluator().evaluate(hole, board)` to determine
      the showdown ranking when reached.
2. Return the per-seat mean across rollouts.

Phase 1 vs Phase 2 differs **only** in `cfg.policies`:

- Phase 1: `policies = {c: BlueprintPolicy(base_tables, bias_magnitude=b_c)
  for c in ("none","fold","call","raise")}`.
- Phase 2: `policies = {c: BiasedBlueprintPolicy.variant(c) for c in ...}`.

Determinism: `cfg.rng` is seeded per search call; tests assert identical
output for fixed seed.

### 6.5 Depth-limited subgame solver (`solver.py`)

```python
@dataclass
class SolverConfig:
    max_iterations: int = 10_000
    max_wall_seconds: float = 15.0
    discount_interval: int = 1_000         # Linear-CFR discount cadence
    prune_threshold: int = -300_000_000    # CFR-P pruning threshold
    unsafe_init: bool = True               # True on first search of a hand
    leaf: LeafConfig

@dataclass
class SearchResult:
    policy_at_root: np.ndarray             # float32, aligned with legal actions at root
    policy_full: SearchPolicy              # queryable for range updates later in hand
    iterations_run: int
    wall_seconds: float

def solve(root: SubgameRoot, cfg: SolverConfig) -> SearchResult: ...
```

In-memory tables, keyed by the same JSON `info_set` string as blueprint
tables:

```python
subgame_regret:   Dict[str, np.ndarray]    # int32, width = len(legal_actions_at(env))
subgame_strategy: Dict[str, np.ndarray]    # int32 visit counts
```

No lmdb / no disk. Both dicts are discarded when the returned `SearchPolicy`
goes out of scope at end of hand.

Per-iteration loop (follows the structure of
[poker_ai/blueprint/cfr.py](../poker_ai/blueprint/cfr.py)):

1. Sample hole cards for all seats from `root.ranges` (board-compatible,
   non-conflicting).
2. External-sampling CFR walk rooted at `root.env`:
   - Opponent node: sample one action from `get_node_strategy` on
     `subgame_regret`.
   - Own node: expand all actions; accumulate regrets via
     [tree_utils.accumulate_regrets](../poker_ai/blueprint/tree_utils.py).
   - CFR-P pruning: skip subtrees whose regret < `prune_threshold` with
     probability 0.95 (matches the existing blueprint training schedule).
3. Leaf handling:
   - `is_terminal` → `Evaluator`-based payoff.
   - Depth-limit leaf → `leaf.leaf_value(env, live_ranges, cfg.leaf)`,
     where `live_ranges` is the solver's conditioning of the input ranges
     on cards sampled in step 1.
4. Visit-count accumulation on the acting player's strategy row.
5. Every `cfg.discount_interval` iterations, apply Linear-CFR discount to
   both subgame dicts (multiplicative factor, regret floor applied).
6. Break when `iters >= max_iterations` or
   `time.monotonic() - t0 >= max_wall_seconds`.

**Unsafe vs linear re-search.** `unsafe_init=True` starts with empty
in-memory dicts (first search of a hand). `unsafe_init=False` warm-starts
both dicts by copying rows from the previous search's `SearchPolicy` where
the info-set key matches — the paper's "linear MCCFR re-search" regime for
the 2nd+ search in a hand.

### 6.6 Search-aware agent (`agent.py`)

```python
class SearchAgent:
    def __init__(
        self,
        blueprint: BlueprintPolicy,
        biased: Optional[BiasedBlueprintPolicy],  # None → Phase 1
        card_info_lut: InfoSetLut,
        solver_cfg: SolverConfig,
        off_tree_tol: float = 0.15,
    ): ...

    def on_hand_start(self, env: PokerEnv, my_seat: int): ...
    def on_board_update(self, new_cards: Sequence[int]): ...
    def on_observed_action(self, env_before: PokerEnv, seat: int, chips: int): ...
    def act(self, env: PokerEnv) -> str:
        """Returns an abstract action string accepted by PokerEnv.apply_action."""
```

Per-hand state held on the agent:

- `tracker: RangeTracker`
- `extra_actions: Dict[str, List[str]]` — accumulated from off-tree classifications
- `last_search: Optional[SearchResult]` — warm-start source for re-search
- `search_used_this_hand: bool` — controls unsafe_init / round-1 gating

`act` decision flow:

1. If `env.betting_round == 0` **and** `not search_used_this_hand` **and**
   no `extra_actions` touch this info set → return the blueprint-sampled
   action directly via `tree_utils.sample_action`.
2. Else build `SubgameRoot` using `tracker.snapshot()` and `extra_actions`;
   call `solve(root, cfg)` with `unsafe_init = not search_used_this_hand`;
   sample the action from `policy_at_root`; store the result for re-search
   warm-start and as `policy_used` for future range updates.
3. Return the abstract action (chip translation happens only if the env
   API requires an integer; current `PokerEnv.apply_action` accepts the
   abstract string directly).

`on_observed_action` flow:

1. `classification, action_str = classify_observed(env_before, chips,
   off_tree_tol)`.
2. If `OFF_TREE`: add `action_str` to
   `extra_actions[env_before.info_set]`.
3. Pick `policy_used`: the cached `last_search.policy_full` if search ran at
   least once this hand, else the blueprint policy (wrapping the
   appropriate tables).
4. `tracker.on_action(seat, env_before, action_str, policy_used)`.

Wiring: [poker_ai/terminal/runner.py](../poker_ai/terminal/runner.py) gets a
new `--agent search` branch that instantiates `SearchAgent`, plumbs
`on_hand_start` / `on_board_update` / `on_observed_action` through the play
loop, and replaces the inline offline-lookup block with `agent.act(env)`.

## 7. Implementation Order

Biased blueprint training is implemented first so those long-running jobs can
start while the rest is being built.

| # | Component | File(s) | Status | Blocks on |
|---|---|---|---|---|
| 0 | Combo helpers + `Policy` ABC + `BlueprintPolicy` | `environment/utils.py`, `environment/poker_env.py`, `poker_ai/search/policy.py` | **done** | — |
| 1 | Biased blueprint training | `poker_ai/blueprint/bias.py`, `runner.py`, `cfr.py` | in progress | — |
| 2 | Subgame construction | `poker_ai/search/subgame.py` | todo | 0 |
| 3 | Range tracking | `poker_ai/search/ranges.py` | todo | 0 |
| 4 | Action translation | `poker_ai/search/translation.py` | todo | — |
| 5 | Leaf-EV (Phase 1 bias) | `poker_ai/search/leaf.py` | todo | 0, 2, 3 |
| 6 | Solver + `SearchPolicy` | `poker_ai/search/solver.py`, `policy.py` | todo | 0, 2, 3, 5 |
| 7 | Search-aware agent | `poker_ai/search/agent.py`, terminal wiring | todo | 3, 4, 6 |
| 8 | CLI, config, tests | existing Click runner, `test/search/` | partial | 1–7 |
| 9 | Phase 2 switch + `BiasedBlueprintPolicy` | `leaf.py`, `policy.py` | todo | 1 complete |

Step 1 can run concurrently with 2–8 on a separate machine / process.

## 8. CLI

```
# Training a biased variant (warm-started from base blueprint)
poker_ai train start \
    --bias fold \
    --bias-magnitude 0.5 \
    --warm-start blueprints/base \
    --output blueprints/fold_biased

# Play with search (Phase 1 — on-demand bias)
poker_ai play \
    --agent search \
    --blueprint blueprints/base \
    --iterations 10000 \
    --time-limit 15s

# Play with search (Phase 2 — precomputed biased blueprints)
poker_ai play \
    --agent search \
    --blueprint blueprints/base \
    --biased-blueprints blueprints/ \
    --iterations 10000 \
    --time-limit 15s
```

## 9. Testing Strategy

- **Unit tests** under `test/search/` for each module, using small
  deterministic fixtures and `@pytest.mark.requires_lut` where the card-info
  LUT is needed.
  - `ranges.py`: Bayesian-update math, board-conflict zeroing.
  - `translation.py`: nearest-size mapping within tolerance; injection of
    off-tree actions into the subgame tree when the deviation exceeds the
    tolerance; round-trip chip → abstract → chip preservation where the
    input is already in-abstraction.
  - `subgame.py`: pot / stack / history invariants after construction;
    traversal reaches only reachable states.
  - `leaf.py`: deterministic output under a fixed RNG seed; Phase 1 and
    Phase 2 sources interchange behind the same interface.
  - `solver.py`: terminates on either stopping criterion; per-hand tables
    are released.
- **Biased training regression**: with `b = 0` the biased training path is
  numerically identical to the base path; with large `b`, on a small toy
  game, the biased action class dominates.
- **Integration test**: one full hand versus a scripted opponent; assert
  search fires from round 2 and in round-1 off-tree situations; assert both
  Phase 1 and Phase 2 agents play legal hands end-to-end.
- **Empirical evaluation**: `--agent search` (Phase 1) vs. plain `--agent
  offline`, ≥ 10 000 hands heads-up, expect a material bb/100 improvement.
  Re-run at Phase 2 once biased blueprints are trained, expect a further
  improvement.
- **Profiling**: single search call within the configured time budget;
  subgame memory released between hands. Per-decision log of (searched?,
  iterations used, wall-clock, `σ_used` source at leaves).

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Biased blueprint training takes longer than expected | Phase 1 unblocks everything else; Phase 2 is a drop-in swap once training finishes. |
| Dense 1326-combo ranges are too slow under frequent updates | Update cost is O(1326) per observed action, amortized against search cost; only re-evaluate if profiling shows it as a hotspot. Fallback: cluster-bucketed ranges behind a flag. |
| Search budget not enough for a good solution on turn/river | Both iteration and wall-clock caps are CLI-configurable; tune per street if needed. |
| Off-tree translation introduces exploitability | Two-regime policy per Pluribus paper: nearest-size translation only within a pot-relative tolerance, and direct injection of the off-tree size into the subgame tree beyond it. Tolerance is CLI-configurable; tune from empirical play. |
| Warm-started biased training drifts the base strategy | Warm-started tables are a copy; base blueprint files are untouched on disk. |


