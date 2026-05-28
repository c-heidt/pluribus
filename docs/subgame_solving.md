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
├── context.py        # SubgameContext: per-search inputs static for one solve()
├── ranges.py         # Per-opponent range tracking (dense per-combo)
├── translation.py    # Off-tree size mapping + injection into env overlay
├── leaf.py           # Depth-limit leaf continuation-value evaluation
├── solver.py         # Depth-limited MCCFR subgame solver
├── policy.py         # Policy ABC, BiasClass, implementations
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

### 6.1 Subgame state and context (`context.py`, env additions)

A subgame is **not** a new object type — it is a deepcopied `PokerEnv`
at the bot's decision point. The env already owns game dynamics, the
action abstraction, history, and the LUT, and `apply_action` already
returns a new env via internal deepcopy. Wrapping it in a `SubgameRoot`
peer object would force every downstream module (solver, leaf, range
tracker) to know about the wrapper.

Two pieces are added instead:

1. **Two small env extensions** that move off-tree action injection
   into the only object that knows the public game tree.
2. **A frozen `SubgameContext` dataclass** that carries the
   *static-for-one-search* inputs the solver needs (ranges, my_seat,
   etc.). This replaces the rejected `SubgameRoot`.

#### Env extensions ([environment/poker_env.py](../environment/poker_env.py))

```python
def inject_action(self, action: str) -> None:
    """Add `action` to the legal set at the env's current public state.

    Idempotent.  The injection persists across deepcopies (the overlay
    dict is shared by reference and mutated in place), so a search
    rooted at any descendant env sees the augmented game tree at the
    matching public state."""

def reset_overlay(self) -> None:
    """Clear all injections.  Called by `SearchAgent.on_hand_start`
    so off-tree actions recorded during one hand don't leak into
    the next hand's tree."""

@property
def has_overlay_at_current_node(self) -> bool:
    """True iff at least one injected action exists at this public state."""

def with_hole_cards(self, seat: int, cards: Tuple[int, int]) -> PokerEnv:
    """Return a deepcopy with `seat`'s hole cards replaced.

    Used by opponent-response modelling (`RangeTracker`'s
    `sigma_for_combo` closure, §6.2) to evaluate "what would seat
    have done with hand X?" without reaching into player internals.
    Does NOT call `inject_action`."""
```

`legal_actions` is extended to union the overlay for the current
public state with the canonical set, deduping. `apply_action` already
accepts arbitrary `"raise:<fraction>"` strings — no other change required.

**Why "public state", not info_set, as the internal overlay key.**
The overlay describes the *game tree* (a property of public nodes),
not strategy at an information set. Two seats arriving at the same
public node — same `(betting_stage, history)` — have different
`info_set` values because `info_set` embeds the actor's card cluster.
The solver's opponent traversal and the range tracker's Bayes update
both need to see the injection regardless of which seat is "viewing"
the node, so the overlay must be keyed by what they share (public
history), not what differs (card cluster). This is an *internal*
implementation detail — callers never see the key; they call
`env.inject_action(s)` and `env.legal_actions` does the right thing.

#### `SubgameContext` ([poker_ai/search/context.py](../poker_ai/search/context.py))

```python
@dataclass(frozen=True)
class SubgameContext:
    """Inputs to one `solve()` call that do not change during the CFR walk.

    Lives for the duration of one search; carries everything that
    would otherwise be threaded through 4 arguments deep.
    """
    my_seat: int
    my_hole: Tuple[int, int]
    opponent_ranges: Dict[int, Range]     # only seats still in the hand
    board_compatible: np.ndarray          # shape (env.n_combos,), bool
    street_at_root: int                   # 0..3 — halt when env.betting_round > this
    leaf: "LeafConfig"
    rng: np.random.Generator

    @classmethod
    def from_runtime(
        cls,
        env: PokerEnv,
        my_seat: int,
        my_hole: Tuple[int, int],
        opponent_ranges: Dict[int, Range],
        leaf: "LeafConfig",
        rng: np.random.Generator,
    ) -> "SubgameContext":
        """Construct a context for a search rooted at `env`.

        Derives `board_compatible` from the env's combo table and the
        current community cards.  Sets `street_at_root` to
        `env.betting_round`.  Does not deepcopy the env — the solver's
        caller (typically `SearchAgent`) is responsible for that.
        """
```

The construction logic lives next to the dataclass (rather than on
`RangeTracker` or inline in `SearchAgent.act`) because most of the
context's fields come from neither: the board-compatible mask is a pure
function of the env, my_seat/my_hole come from per-hand state, only
`opponent_ranges` is a tracker snapshot. A classmethod keeps the dataclass
and its sole non-trivial constructor in one importable place.

Solver and leaf signatures become `solve(root_env, ctx, cfg)` and
`leaf_value(env, ctx)`. The "root" is just the env passed in; there is
no `SubgameRoot`, no `build_subgame`, no `legal_actions_at`.

#### Traversal

Inside the solver, each step is plain `env = env.apply_action(a)` —
the env's `apply_action` already deepcopies, so wrapping it in an
extra `copy.deepcopy` would copy twice. The depth limit is the same
condition as before: stop and call `leaf.leaf_value(env, ctx)` when
`env.betting_round > ctx.street_at_root`.

**Invariants** (asserted in tests): after `root_env = copy.deepcopy(runtime_env)`,
`root_env.pot_size`, player chip stacks, the per-stage history, and
`root_env.current_player` match the runtime values. Every leaf
satisfies `is_terminal or betting_round > street_at_root`.

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
        sigma_for_combo: Callable[[int], np.ndarray],
    ) -> None:
        """w(h) ← w(h) · sigma_for_combo(h)[idx_of(action)]; renormalize.

        `sigma_for_combo(h)` returns the strategy vector aligned with
        `env_before.legal_actions` for the case where seat's hole cards
        are `COMBO_CARDS[h]`.  The caller (typically `SearchAgent`)
        constructs this closure so ranges.py never imports policy.py.
        """

    def range_of(self, seat: int) -> Range: ...
    def snapshot(self) -> Dict[int, Range]: ...   # deep copy; only live seats
```

The tracker has zero search-package imports, takes only env + a
callable, and can be unit-tested with a hand-written `sigma_for_combo`.
`SubgameContext.from_runtime` (§6.1) calls `snapshot()` to populate
`opponent_ranges`; the tracker has no `build_context` method and no
dependency on `SubgameContext` / `LeafConfig` / RNG / `Policy`.

A new env helper supports the closure side cleanly:

```python
# environment/poker_env.py
def with_hole_cards(self, seat: int, cards: Tuple[int, int]) -> PokerEnv:
    """Return a deepcopy with `seat`'s hole cards replaced.

    Used by opponent-response modelling to evaluate "what would seat
    have done with hand X?" without reaching into player internals.
    """
```

The agent then builds the closure:

```python
sigma_for_combo = lambda h: opponent_response_policy.strategy(
    env_before.with_hole_cards(seat, tuple(env.combo_cards[h]))
)
```

Implementation notes:

- Vectorize over the card cluster, not over individual combos: combos that
  map to the same cluster-id on this street have identical `env.info_set`
  and therefore identical σ. Group combos by cluster once per call, query
  σ once per distinct cluster.  This optimisation lives in the agent's
  closure construction (where the policy is known), not in the tracker.
- Numerical floor: if `w.sum() < 1e-12` after an update, reset to uniform
  over board-compatible combos and emit a warning (only reachable via
  zero-probability observations, e.g. off-tree bets before injection kicks
  in).
- A per-decision log `List[Tuple[seat, info_set_before, action]]` is
  retained for debugging and Phase 1↔Phase 2 replay comparisons.

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
- `OFF_TREE` injection calls `env_before.inject_action(action_str)`
  (§6.1). The env records the injection against its own current public
  state internally; `env.legal_actions` then reports the injected
  action at every visit to that public state, regardless of which
  seat is acting. Solver, leaf, and tree-walk code see it transparently
  through `env.legal_actions` and need no special-case branch.

### 6.4 Leaf continuation-value evaluation (`leaf.py`)

```python
@dataclass
class LeafConfig:
    policies: Dict[BiasClass, Policy]     # one entry per k = 4 bias class
    n_rollouts: int = 20
    rng: np.random.Generator

def leaf_value(
    env: PokerEnv,                        # state AT the leaf (betting_round > ctx.street_at_root)
    live_ranges: Dict[int, Range],        # ranges conditioned on path taken (subset of ctx.opponent_ranges)
    ctx: SubgameContext,
) -> np.ndarray:                          # shape (n_seats,), expected chips won/lost
```

Algorithm per call:

1. Repeat `n_rollouts` times:
   a. For each seat `i`, sample a continuation-strategy choice
      `c_i ∈ {"none","fold","call","raise"}` uniformly.
   b. Sample each opponent's hole from its range (board-compatible and
      mutually non-conflicting; rejection sample or precomputed joint
      index).
   c. Roll the hand forward via `env = env.apply_action(a)` until
      `env.is_terminal`: at every decision node, sample from
      `cfg.policies[c_i].strategy(env, bias=c_i)` via
      `tree_utils.sample_action`.
   d. Read per-seat payoff from `env.payout` (terminal envs already
      have winners computed by `dynamics.compute_winners`, called
      from `apply_action`).  **Do not** re-implement payoff via
      `Evaluator().evaluate(...)` — that duplicates env logic and
      drops side-pot handling, which the env gets right.
2. Return the per-seat mean across rollouts.

Phase 1 vs Phase 2 differs **only** in `cfg.policies`:

- Phase 1: `policies = {c: BlueprintPolicy(base_tables, bias_magnitude=b_c)
  for c in ("none","fold","call","raise")}`.
- Phase 2: `policies = {c: BiasedBlueprintPolicy.variant(c) for c in ...}`.

Determinism: `cfg.rng` is seeded per search call; tests assert identical
output for fixed seed.

### 6.5 Depth-limited subgame solver (`solver.py`)

```python
@dataclass(frozen=True)
class SolverConfig:
    """Static hyperparameters; identical across every search in a session."""
    max_iterations: int = 10_000
    max_wall_seconds: float = 15.0
    discount_interval: int = 1_000         # Linear-CFR discount cadence
    prune_threshold: int = -300_000_000    # CFR-P pruning threshold
    leaf: LeafConfig

@dataclass
class SearchResult:
    policy: SearchPolicy                   # queryable at any node visited during the walk
    iterations_run: int
    wall_seconds: float

def solve(
    root_env: PokerEnv,                    # deepcopied by the caller
    ctx: SubgameContext,
    cfg: SolverConfig,
    warm_start: Optional[SearchPolicy] = None,
) -> SearchResult: ...
```

`warm_start` replaces the `unsafe_init: bool` config flag.  Passing
`None` is the unsafe-init regime (first search of a hand, empty
in-memory dicts).  Passing the previous search's
`SearchResult.policy` is the paper's linear MCCFR re-search regime
for the 2nd+ search in a hand — rows are copied lazily from the
warm-start source on first miss.  This keeps `SolverConfig` purely
static so the agent never has to `dataclasses.replace` it per call.

`SearchResult` exposes one policy field; the root distribution is
`result.policy.strategy(root_env)`.  Storing it as a separate array
introduced an invariant the caller had to maintain by hand.

In-memory tables, keyed by the JSON `info_set` string. Width is
**per-node** because injected actions (§6.1, §6.3) extend the legal
set at specific public keys, so canonical-width rows would either
waste columns or omit injected slots:

```python
subgame_regret:   Dict[str, np.ndarray]    # int32, width = len(env.legal_actions) at first visit
subgame_strategy: Dict[str, np.ndarray]    # int32 visit counts, same width
```

No lmdb / no disk. Both dicts are discarded when the returned `SearchPolicy`
goes out of scope at end of hand.

#### Reuse from the blueprint code

- **Reuse directly**:
  [`calculate_strategy_from_row`](../poker_ai/blueprint/tree_utils.py)
  at every solver node — already shape-agnostic; takes a regret row +
  valid mask of any length.
- **Do not reuse**:
  [`accumulate_regrets`](../poker_ai/blueprint/tree_utils.py),
  `get_node_strategy`, and the blueprint walker itself. They assume
  fixed canonical width via `ACTION_TO_IDX`. The subgame uses
  variable-width rows with a per-node `a_to_i` derived from
  `env.legal_actions` at first visit. Write a parallel walker in
  `poker_ai/search/solver.py` that mirrors the blueprint loop's
  *structure* rather than refactoring `blueprint/cfr.py` to be
  generic — the two have different lifecycle and persistence semantics.

Resist introducing a `RegretSource` abstraction over `CFRTables` and
the in-memory dict; the two differ in width semantics, persistence,
locking, and lifecycle, and a unifying type would leak details both
ways.

#### Per-iteration loop

1. Sample hole cards for all seats from `ctx.opponent_ranges`
   (board-compatible via `ctx.board_compatible`, non-conflicting
   across seats). The bot's hole is fixed at `ctx.my_hole`.
2. External-sampling CFR walk rooted at `root_env`. Step the env with
   plain `env = env.apply_action(a)` (no outer `deepcopy` — env's
   `apply_action` already deepcopies internally).
   - Opponent node: regret-match the row via
     `calculate_strategy_from_row`, sample one action.
   - Own node: expand all actions; accumulate regrets into the row.
   - CFR-P pruning: skip subtrees whose regret < `prune_threshold` with
     probability 0.95 (matches the existing blueprint training schedule).
3. Leaf handling:
   - `is_terminal` → read `env.payout` (computed by the env's own
     `dynamics.compute_winners`).
   - Depth-limit leaf (`env.betting_round > ctx.street_at_root`) →
     `leaf.leaf_value(env, live_ranges, ctx)`, where `live_ranges`
     is the solver's conditioning of `ctx.opponent_ranges` on cards
     sampled in step 1.
4. Visit-count accumulation on the acting player's strategy row.
5. Every `cfg.discount_interval` iterations, apply Linear-CFR discount to
   both subgame dicts (multiplicative factor, regret floor applied).
6. Break when `iters >= max_iterations` or
   `time.monotonic() - t0 >= max_wall_seconds`.

**Unsafe vs linear re-search.** Controlled by the `warm_start`
argument, not a config flag — see the `solve()` signature above.

### 6.6 Search-aware agent (`agent.py`)

```python
class SearchAgent:
    def __init__(
        self,
        leaf_policies: Dict[BiasClass, Policy],   # k=4 entries
        opponent_response_policy: Policy,         # for tracker.on_action closures
        blueprint_policy: Policy,                 # round-1 fast path (no search)
        solver_cfg: SolverConfig,
        rng: np.random.Generator,
        off_tree_tol: float = 0.15,
    ): ...

    def on_hand_start(self, env: PokerEnv, my_seat: int): ...
    def on_board_update(self, new_cards: Sequence[int]): ...
    def on_observed_action(self, env_before: PokerEnv, seat: int, chips: int): ...
    def act(self, env: PokerEnv) -> str:
        """Returns an abstract action string accepted by PokerEnv.apply_action."""
```

Phase 1 vs Phase 2 is pure dependency injection at construction time:

- **Phase 1**: `leaf_policies = {c: BlueprintPolicy(base_tables,
  bias_magnitude=b_c) for c in (...)}`.
- **Phase 2**: `leaf_policies = {c: BiasedBlueprintPolicy(c, tables_c)
  for c in (...)}`.

The agent has no `Optional[BiasedBlueprintPolicy]` branch and no
phase-awareness; the choice lives in
[poker_ai/terminal/runner.py](../poker_ai/terminal/runner.py) where the
`--biased-blueprints` flag is parsed.

Per-hand state held on the agent:

- `tracker: RangeTracker`
- `my_seat: int` (set by `on_hand_start`; `my_hole` read from
  `env.players[my_seat].cards` on demand — env owns player state)
- `last_search: Optional[SearchResult]` — warm-start source for re-search

Off-tree action overlays live on the env itself (§6.1), not on the
agent. The runtime env and any deepcopy made for search both observe
the same injections through `env.legal_actions`.

**Overlay lifecycle.** `on_hand_start` calls `env.reset_overlay()` so
injections from previous hands don't leak into this hand's tree.

`act` decision flow:

1. If `env.betting_round == 0` **and** `self.last_search is None` **and**
   `not env.has_overlay_at_current_node` → return the blueprint-sampled
   action directly via `tree_utils.sample_action` on
   `self.blueprint_policy`.
2. Else build the context and solve:
   ```python
   root_env = copy.deepcopy(env)
   my_hole = tuple(root_env.players[self.my_seat].cards)
   ctx = SubgameContext.from_runtime(
       root_env, self.my_seat, my_hole,
       self.tracker.snapshot(), self.solver_cfg.leaf, self.rng,
   )
   warm = self.last_search.policy if self.last_search is not None else None
   result = solve(root_env, ctx, self.solver_cfg, warm_start=warm)
   ```
   Sample the action from `result.policy.strategy(root_env)`; store
   `result` for re-search warm-start.
3. Return the abstract action (`PokerEnv.apply_action` accepts the
   string directly; no chip translation needed).

`on_observed_action` flow:

1. `classification, action_str = classify_observed(env_before, chips,
   off_tree_tol)`.
2. If `OFF_TREE`: `env_before.inject_action(action_str)`. The overlay
   dict is shared by reference across the deepcopy lineage and mutated
   in place, so the injection is visible to the runtime env and to
   every future search-time deepcopy at the matching public state.
3. Pick the policy used to model that seat's decision: the cached
   `last_search.policy` if search ran at least once this hand, else
   `self.opponent_response_policy`.
4. Build the per-combo σ closure (cluster-vectorised — see §6.2) and
   call `tracker.on_action(seat, env_before, action_str, sigma_for_combo)`.

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
| 2 | Env overlay (`inject_action`, `reset_overlay`, `has_overlay_at_current_node`) + `with_hole_cards` + `SubgameContext` | `environment/poker_env.py`, `poker_ai/search/context.py` | **done** | 0 |
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
  - `environment/poker_env.py` (additions): `inject_action` is
    idempotent and persists across deepcopies; `legal_actions`
    reports injected actions at the matching public state and nowhere
    else (verified by stepping the env past the injection point);
    `has_overlay_at_current_node` flips correctly across
    inject/reset; `reset_overlay` clears all injections;
    `with_hole_cards` returns a deepcopy with the named seat's cards
    replaced and leaves the original untouched.
  - `context.py`: `SubgameContext.from_runtime` produces
    `board_compatible` matching `env.community_cards`; `street_at_root`
    equals `env.betting_round`; opponent_ranges contains only live
    seats; field set is frozen.
  - `ranges.py`: `on_action` Bayes-update under a hand-written
    `sigma_for_combo` callable (no policy.py import needed in the
    test); board-conflict zeroing; uniform fallback on numerical floor.
  - `leaf.py`: deterministic output under a fixed RNG seed; payoff at
    showdown matches `env.payout` (no `Evaluator()` duplication);
    Phase 1 and Phase 2 sources interchange behind the same interface.
  - `solver.py`: terminates on either stopping criterion; per-hand tables
    are released; `warm_start=None` and `warm_start=prev_policy` produce
    consistent in-memory dicts (re-search builds on prior rows).
  - `agent.py`: no `Optional` branch on biased-vs-not (DI test:
    construct the agent with both Phase 1 and Phase 2 `leaf_policies`
    dicts and verify identical control flow); `on_hand_start` calls
    `env.reset_overlay`; round-1 fast path skipped when
    `env.has_overlay_at_current_node` is True.
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


