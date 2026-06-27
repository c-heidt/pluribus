# Subgame Solving & Online Play

Implementation plan for the real-time search component of the Pluribus bot,
following Brown & Sandholm (2019), *Superhuman AI for multiplayer poker*, and
its supplementary materials.

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
strength of Pluribus at 6-max NLHE comes from **real-time search with
continuation strategies**, run on every betting round after the first and in
rare off-tree situations on the first. Adding this online component is the
scope of this document.

## 2. Goals and Non-Goals

### Goals

- Produce a bot that, at play time, performs real-time search rooted at the
  start of the current betting round rather than reading the blueprint
  directly.
- Match the paper's algorithmic structure: nested unsafe search solving from
  the start of the current betting round; round-dependent depth limits;
  continuation strategies at depth-limit leaves that are runtime reweightings
  of the blueprint, chosen via CFR; the paper's two subgame CFR regimes —
  external-sampling Linear MCCFR for large/early subgames and vector-form
  Linear CFR for small/late ones; the final iteration's strategy played, the
  weighted average kept for belief updates.

### Non-goals

- No changes to the blueprint training pipeline (its action abstraction or the
  card clustering). Real-time search, as in the paper, uses its **own**
  abstractions: a coarser raise-size set (≤ 5–6 sizes per decision) for the
  subgame action tree, and — for cards — lossless abstraction on the root street
  with the existing 200-bucket LUT on later streets (upgraded to 500 when the new
  LUT is computed; the 200-vs-500 count is the one accepted divergence from the
  paper).
- No distributed (multi-machine) search. Search runs on one machine per
  decision; within that machine it parallelizes across cores (§6.7), matching
  the paper's per-thread public-board sampling.
- No opponent modelling or exploitation; Pluribus plays the same strategy
  regardless of opponent identity, and this project preserves that.
- No safety guarantees. Safe subgame-solving theorems do not extend beyond
  two-player zero-sum, and the paper does not claim them either.

## 3. Resolved Design Parameters

| Parameter | Value |
|---|---|
| Continuation strategies | k = 4: unaltered blueprint; fold-, call-, raise-biased = blueprint with that action class's probability ×5, renormalized at inference time |
| Subgame depth limit | Round-1 search: end of round 1. Round-2 search with > 2 players at round start: start of round 3 **or** immediately after the 2nd raise of the round, whichever is earlier. All other cases (round 2 heads-up, rounds 3–4): end of the game |
| Card abstraction in search | Lossless (per-combo) on the root street; 200-bucket LUT clusters on later streets (→ 500 with the new LUT) |
| Subgame action abstraction | Coarser, search-specific raise-size set — ≤ 5–6 pot-fractions per decision (paper); opponent raises off this set are injected and trigger re-search |
| CFR algorithm | Two regimes (paper): external-sampling Linear MCCFR for large/early subgames (round 1, round 2, large multiway); vector-form Linear CFR sampling one board runout per iteration for small/late subgames (heads-up turn/river). CFR-P pruning is **not** applied in search — the blueprint's threshold never fires at search scale (§6.5) |
| Range representation | Dense per-combo distribution for **every** player still in the hand, including the bot (observer perspective) |
| Belief updates | Bayes' rule at round boundaries under the previous search's weighted-average strategy (blueprint if no search has run yet this hand) |
| Strategy played | Final iteration of the search; the weighted average is kept only for belief updates |
| Leaf / terminal all-in runouts | **Exact** board-averaged equity at decision-free (all-in) showdowns instead of the paper's single sampled runout (§6.4.1) — an accepted, flag-toggled divergence (`use_decision_free_equity`), alongside the 200-vs-500 LUT divergence |
| Action translation | Pseudo-harmonic: randomized variant on round 1; deterministic variant for blueprint lookups on histories containing off-tree actions; rounds 2–4 always inject the off-tree action and re-search |
| Round-1 search trigger | Opponent raise more than $100 from every size in the blueprint abstraction **and** ≤ 4 players remaining in the hand; otherwise blueprint play with randomized pseudo-harmonic mapping |
| Search budget | Dual cap: 10 000 iterations AND 15 s wall-clock; whichever hits first (CLI-configurable) |

## 4. Continuation Strategies

The k = 4 continuation strategies are **inference-time modifications of the
base blueprint**. At any decision point, the base distribution σ is computed
from the blueprint's stored regrets via regret matching; the biased variants
multiply the probability of the biased action class by 5 and renormalize:

- *blueprint*: σ unaltered.
- *fold-biased*: σ(fold) ×5, renormalized.
- *call-biased*: σ(call/check) ×5, renormalized.
- *raise-biased*: σ(a) ×5 for every `a` starting with `raise` or `all_in`,
  renormalized.

No additional training artifacts are required; there is one blueprint on disk
and the four variants are derived from it at query time. The *choice* among
the four (including mixtures) is itself an action in the subgame, solved by
CFR at depth-limit leaves (§6.5).

Action classes are identified by prefix: `"fold"`, `"call"` / `"check"`,
anything starting with `"raise"` or `"all_in"`.

## 5. Search Triggering and Subgame Structure

The online lifecycle follows Algorithm 2 of the supplementary materials.

### Root placement and triggers

- The subgame root is the **public state at the start of the current betting
  round**. It does not move until a new round is reached.
- **Rounds 2–4**: search is run **at the moment the round begins** (Algorithm 2's
  `CheckNewRound`) — as soon as the new round's public state becomes the root and
  the round-boundary belief update has been applied, *before* the bot is asked to
  act. Whenever an opponent then takes an action outside the subgame's action
  abstraction, that action is added to every node of the current public state
  (`env.inject_action`, §6.1) and the subgame is **re-searched from the same
  root**.
- **Round 1**: the bot plays the blueprint. An observed off-tree raise is
  mapped onto the abstraction with **randomized pseudo-harmonic action
  translation** (§6.3) and play continues from the blueprint — *unless* the
  raise is more than $100 from every size in the blueprint abstraction and no
  more than four players remain in the hand, in which case search runs with a
  subgame extending to the end of round 1.
- **Own turn**: the bot samples its action from the current search output's
  final-iteration strategy (blueprint on round 1); no search is triggered.

### Depth limits

Per the table in §3. Subgames that extend to the end of the game have only
terminal leaves (scored at the showdown / fold-out); the continuation-strategy
machinery applies only to round-1 search and multiway round-2 search, whose
leaves are non-terminal. The CFR regime used for each subgame (MCCFR vs vector,
§6.5) is chosen by size and street, independently of the leaf type.

### Freezing on re-search

When the subgame is re-searched within a round, the bot's action
probabilities at every infoset where it already acted this round are **frozen
at the values used when it acted — for its actual hand only**. Its other
possible hands and all opponent infosets remain free to change. Moving the
root to the next round implicitly freezes everything before the new root.

### Root distribution and belief updates

The root is a probability distribution over the nodes of the root public
state: each player — including the bot — carries a dense per-combo range
(§6.2). Opponents are assumed to have played the bot's own strategy at their
past decision points (unsafe search), so hands an opponent would always have
folded carry zero probability and need no strategy.

When a betting round ends, every player's range is updated by Bayes' rule
conditioned on the actions they took during the round, evaluated under σ =
the blueprint if no search has run yet this hand, otherwise the **weighted
average strategy** of the previously-run search. The updated ranges seed the
next round's root.

## 6. Architecture

New top-level package: `poker_ai/search/`.

```
poker_ai/search/
├── context.py        # SubgameContext: per-search inputs static for one solve()
├── ranges.py         # Per-player range tracking (dense per-combo)
├── leaf.py           # Continuation-value evaluation at depth-limit leaves
├── solver_state.py   # SolverConfig, SolverState (shared CFR tables + ops), _hand_row
├── mccfr.py          # External-sampling Linear MCCFR regime (_MCCFRSolver)
├── vector.py         # Vector-form Linear CFR regime (_VectorSolver)
├── solver.py         # solve() orchestrator + SearchResult + regime selection
├── policy.py         # Policy ABC, BiasClass, BlueprintPolicy, SearchPolicy
└── agent.py          # Search-aware play agent
```

**Terminal payouts are owned by the `environment` package, not the search.** The
solver scores terminals by *calling* one of three env-owned payout evaluators, all
ranking hands with the one shared `environment.evaluator.default_evaluator` (so
they agree to the chip): `PokerEnv.payout` (a single dealt hand), `runout_equity`
(the decision-free board-average), and `PokerEnv.vector_payout` (range-vs-range,
per-combo CFV — the vector regime's terminal value). The vectorised settlement
math lives in `environment/range_showdown.py` (`rank_combos_on_board`,
`showdown_cfv`, `reach_after_removal`; board-keyed ranking cache), depending on the
evaluator only. The matched **stake** is env-owned too: `compute_winners` snapshots
the final per-seat contributions (`PokerEnv.terminal_contributions`) before it
resets the pot, so the search never reconstructs settlement.

### 6.0 Policy interface (`policy.py`)

Card-space dimensions depend on the deck the environment was built for
(small decks are used in tests and for sub-game LUTs), so they are not
hard-coded in the search package. The environment exposes them and the
search package consumes whatever the live `PokerEnv` reports:

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

`poker_ai/search/policy.py` owns the policy abstraction and the shared
aliases the rest of the search package consumes:

```python
BiasClass = Literal["none", "fold", "call", "raise"]

class Policy(ABC):
    """Base class for anything the solver / leaf evaluation can query for an
    action distribution."""

    @abstractmethod
    def strategy(self, state: PolicyState, bias: BiasClass = "none") -> np.ndarray:
        """Float32 vector aligned with `state.legal_actions`."""

    @staticmethod
    def _bias_mask(actions: List[str], bias: BiasClass) -> np.ndarray:
        """Boolean mask over `actions` selecting the biased class (§4)."""

    @staticmethod
    def _reweight_bias(
        sigma: np.ndarray,
        bias_mask: np.ndarray,
        multiplier: float,
    ) -> np.ndarray:
        """σ[biased_class] *= multiplier, renormalize. `multiplier == 1`
        returns σ unchanged."""
```

Two implementations:

- `BlueprintPolicy(tables: CFRTables, bias_multiplier: float = 5.0)` —
  reads the regret row for `state.info_set` at `state.betting_round`,
  computes σ via regret matching (`calculate_strategy_from_row`), then
  applies `_reweight_bias` for the requested bias class; falls back to
  uniform when the row is absent (unseen info set). When the queried
  history contains an off-tree action, the lookup key is canonicalized
  via deterministic pseudo-harmonic translation (§6.3) before reading the
  tables. Overlay-injected actions that have no blueprint column receive
  zero mass and the canonical probabilities are renormalized over the
  legal set (uniform fallback when only overlay actions are legal).
- `SearchPolicy(state: SolverState, use_average: bool)` — **deferred**
  until the solver in §6.5 lands. Returned by `solve()`; resolves
  `PolicyState → public_key → hand row` in the subgame-local tables and
  regret-matches the row (`use_average=False`, the strategy the bot
  plays) or normalizes the visit sums (`use_average=True`, the strategy
  used for belief updates).

ABC over Protocol because the implementations share real behavior
(regret matching, bias-mask construction, uniform fallback), not just a
signature.

### 6.1 Subgame state and context (`context.py`, env additions)

A subgame is **not** a new object type — it is a deepcopied `PokerEnv`
at the root public state. The env already owns game dynamics, the
action abstraction, history, and the LUT. The env is advanced with the
make/undo pair `step_in_place` / `undo` (the sole advance API; `deepcopy`
is used only to snapshot the root and by `with_hole_cards`).

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

def with_hole_cards(self, holes: Sequence[Tuple[int, int]]) -> PokerEnv:
    """Return a deepcopy with every seat's hole cards replaced atomically."""

@property
def public_key(self) -> Tuple:
    """Hashable identifier of the current public state:
    (betting_stage, per-stage action-history tuple).  Used to key the
    solver's in-memory tables and the overlay."""

def cluster_for(self, combo: Tuple[int, int]) -> int:
    """LUT cluster id for `combo` on the current street and board.
    Reads `card_info_lut` directly, skipping the JSON info-set build."""

def step_in_place(self, action: str) -> "UndoToken":
    """Apply `action` by **mutating this env in place**, returning an
    undo token (a snapshot of exactly the mutable per-hand state
    `__deepcopy__` copies: pot, every player's chips/bet/fold/turn
    state, history, betting-round counters, community cards and the
    deck cursor).  The sole way to advance the env."""

def undo(self, token: "UndoToken") -> None:
    """Reverse the most recent `step_in_place`, restoring the env to its
    pre-action state in O(action footprint).  Tokens are strictly LIFO."""
```

`legal_actions` unions the overlay for the current public state with the
canonical set, deduping; `step_in_place` accepts arbitrary
`"raise:<fraction>"` strings. `step_in_place` / `undo` are the make/undo pair
the CFR traversal (and the offline blueprint trainer) use to advance the env
without a per-node deepcopy (§6.7); a caller that needs the pre- and
post-action env at once `deepcopy`s first.

**Why "public state", not info_set, as the internal overlay key.**
The overlay describes the *game tree* (a property of public nodes),
not strategy at an information set. Two seats arriving at the same
public node — same `(betting_stage, history)` — have different
`info_set` values because `info_set` embeds the actor's card cluster.
The solver's traversal and the range tracker's Bayes update both need
to see the injection regardless of which seat is "viewing" the node,
so the overlay is keyed by what they share (public history), not what
differs (card cluster).

#### `SubgameContext` ([poker_ai/search/context.py](../poker_ai/search/context.py))

```python
@dataclass(frozen=True)
class SubgameContext:
    """Inputs to one `solve()` call that do not change during the CFR walk."""
    my_seat: int
    my_hole: Tuple[int, int]
    ranges: Mapping[int, Range]           # every live seat, incl. my_seat
    folded_ranges: Mapping[int, Range]    # seats that folded before the root (card removal)
    board_compatible: np.ndarray          # shape (env.n_combos,), bool
    street_at_root: int                   # 0..3
    depth_limit: DepthLimit               # see below
    leaf: "LeafConfig"
    rng: np.random.Generator

    @classmethod
    def from_runtime(cls, env, my_seat, my_hole, ranges, folded_ranges, leaf, rng): ...
```

`depth_limit` is a small descriptor derived from `(street_at_root,
players live at round start)` implementing the §3 table. It answers, for
any env reached during the walk, one of three verdicts: *internal*,
*depth-limit leaf* (continuation meta-game, §6.5), or *terminal*
(`env.payout`). The multiway round-2 case additionally cuts off
immediately after the second raise of the root round (the env tracks the
in-round raise count).

The bot's entry in `ranges` is its observer-perspective range — the
search computes a strategy for the bot's **entire range**, and the agent
plays the actual hand's row. Opponent entries exclude the bot's actual
cards (known card removal); the bot's own entry excludes only board
conflicts.

#### Traversal

The solver's inner CFR traversal descends with `token = env.step_in_place(a)`
and ascends with `env.undo(token)` (§6.7) — one mutable env per traversal, no
per-node copy. The agent's lifecycle snapshots the search root once with
`copy.deepcopy`; continuation rollouts walk a per-rollout `with_hole_cards` env
forward in place. Leaf classification uses `ctx.depth_limit`; `is_terminal` is
always checked **before** reading `betting_round` (which raises at the
`"terminal"` stage).

**Invariants** (asserted in tests): after `root_env = copy.deepcopy(runtime_env)`,
`root_env.pot_size`, player chip stacks, the per-stage history, and
`root_env.current_player` match the runtime values. Every leaf satisfies
the depth-limit verdict of the §3 table.

### 6.2 Range tracking (`ranges.py`)

```python
Range = np.ndarray                                  # float32, shape (env.n_combos,)

class RangeTracker:
    def __init__(self, env, my_seat, my_hole, live_seats): ...

    def on_board_update(self, new_cards: Sequence[int]) -> None:
        """Zero every combo sharing a card with the new board; renormalize."""

    def on_action(self, seat, env_before, action, sigma_for_combo) -> None:
        """w(h) ← w(h) · sigma_for_combo(h)[idx_of(action)]; renormalize."""

    def on_seat_folded(self, seat: int) -> None: ...
    def range_of(self, seat: int) -> Range: ...
    def snapshot(self) -> Dict[int, Range]: ...        # deep copy; live seats
    def folded_snapshot(self) -> Dict[int, Range]: ...
```

- The tracker maintains a range for **every seat in the hand, including
  `my_seat`**. The bot's own range is the observer-perspective
  distribution: it excludes board conflicts but **not** the bot's actual
  hole cards. Opponents' ranges additionally exclude the bot's actual
  cards.
- **Update timing.** Ranges are updated at **round boundaries**, not per
  action. The agent buffers `(seat, env_before, action)` tuples during
  the round; when the round ends it replays them through `on_action`
  with a `sigma_for_combo` closure built from the last search's
  **average** policy (the blueprint if no search has run yet this hand).
  The per-action `on_action` API is the replay primitive.
- `sigma_for_combo(h)` returns the strategy vector aligned with
  `env_before.legal_actions` for the case where seat's hole cards are
  `COMBO_CARDS[h]`. The agent constructs this closure so ranges.py never
  imports policy.py. Vectorize over the card cluster: combos mapping to
  the same cluster on a street have identical σ — group combos by
  cluster once per call, query σ once per distinct cluster.
- Numerical floor: if `w.sum() < 1e-12` after an update, reset to uniform
  over board-compatible combos and emit a warning.
- A per-decision log `List[Tuple[seat, info_set_before, action]]` is
  retained for debugging.

### 6.3 Action translation (env API, no search-side module)

Chip ↔ action-string conversion is a property of the environment. The
runtime translator at the chip-denominated boundary consumes the env's
public chip-math methods; the search package sees only action strings.

```python
# environment/poker_env.py — public methods
def canonical_raise_fractions(self) -> List[float]:
    """Currently-playable raise fractions, mirroring legal_actions' gating."""

def chips_to_add(self, action: str) -> int:
    """Inverse of the env's action chip math."""

def string_for_chips(self, chip_amount: int) -> str:
    """Map an observed chip raise to an action string.

    1. chip_amount == actor.n_chips  →  "all_in"
    2. exact canonical clamp         →  "raise:<f>"
    3. else                          →  off-tree "raise:<f_obs>" (4-dp)
    """
```

Off-tree raises are resolved with **pseudo-harmonic action translation**
(Ganzfried & Sandholm 2013). For an observed pot-fraction `x` between
neighbouring abstraction sizes `A < x < B`:

```
P(map to A) = ((B − x) · (1 + A)) / ((B − A) · (1 + x))
```

- **Randomized variant** (sample A or B by that probability): used on
  **round 1** to map an observed off-tree raise onto the abstraction
  before continuing from the blueprint (§5).
- **Deterministic variant** (pick the side with probability ≥ ½): used to
  canonicalize histories containing off-tree actions whenever the
  blueprint is queried — e.g. during continuation rollouts (§6.4) — so
  the `info_set` lookup resolves to the nearest node in the blueprint
  abstraction instead of missing.
- **Rounds 2–4 never snap.** Every observed off-tree raise is injected
  into the subgame (`env.inject_action`) and the subgame is re-searched
  from the same root (§5). Caller pattern:
  ```python
  action_str = env.string_for_chips(observed_chips)
  if action_str not in env.legal_actions:
      env.inject_action(action_str)
  env.step_in_place(action_str)
  agent.on_observed_action(seat, env_before, action_str)
  ```
- Fold / call / all-in are always on-tree — only raise sizes can deviate.

### 6.4 Continuation-value evaluation (`leaf.py`)

A depth-limit leaf is reached only in the **MCCFR regime** (round-1 and
multiway round-2 subgames, §6.5). At such a leaf every seat already has a
**concrete** hand — sampled once at the subgame root for the current MCCFR
traversal — and the continuation meta-game has fixed each active seat's
continuation strategy. `continuation_value` evaluates that fixed profile for
those concrete hands:

```python
@dataclass
class LeafConfig:
    policies: Dict[BiasClass, Policy]     # the four §4 variants
    n_rollouts: int = 20
    use_decision_free_equity: bool = True # §6.4.1 A/B toggle

def continuation_value(
    frontier_env: PokerEnv,               # at the leaf; every seat's hole already set
    profile: Mapping[int, BiasClass],     # one bias class per active seat (chosen by the meta-game)
    ctx: SubgameContext,
) -> np.ndarray:                          # shape (n_players,), expected chips per seat
```

Algorithm per call: repeat `n_rollouts` times — roll a per-rollout
`with_hole_cards` env forward in place via `env.step_in_place(a)` until
`env.is_terminal`, with each acting seat playing
`cfg.policies[profile[seat]].strategy(state, bias=profile[seat])`. The acting
seat is queried through `env.policy_state_for(own_hole, for_blueprint=True)` so
blueprint lookups on histories containing off-tree actions canonicalize via
deterministic pseudo-harmonic (§6.3). Future board cards are dealt by
`step_in_place` as the rollout crosses round boundaries. Read per-seat payoff
from `env.payout` (do **not** re-implement side pots) — except at a
decision-free all-in showdown, where the exact board-average is taken instead
(§6.4.1). Return the per-seat mean. `ctx.rng` drives action sampling; the board
runout uses the env's global-`np.random` deal, as elsewhere in the engine
(determinism tests pin both).

#### 6.4.1 Decision-free runout equity (over-the-paper improvement)

The paper scores an all-in showdown by sampling **one** board runout. But once
betting is closed (an all-in showdown), the rest of the hand is **pure chance**
with a **board-independent side-pot structure**, so its value can be integrated
**exactly** over the remaining board completions instead of sampled — a strict
variance reduction concentrated on the highest-variance terminals. This is an
explicit, measurable divergence from the paper.

It is a **shared env primitive**, not a leaf-only addition:

```python
# environment/poker_env.py
@property
def is_decision_free(self) -> bool: ...        # all-in showdown over an incomplete board
def runout_equity(self) -> Dict[int, float]:   # exact mean payout over all board completions
```

When a hand force-resolves to showdown over an incomplete board, the env records
a pre-runout snapshot (board prefix, frozen pot contributions, active mask)
*before* `compute_winners` resets the pot; `runout_equity` enumerates every
completion of that prefix, ranks all `(completion × active-seat)` hands in one
batched `Evaluator.evaluate_batch`, and settles the side pots — the side-pot
**structure is board-independent**, so every completion with no rank tie is
scored by a vectorised `argmin`-per-pot + `bincount` over completions, and only
the rare tie completions fall back to the exact `Pot.compute_utility` (side pots
**reused, not re-implemented**) — returning the mean minus each seat's
contribution. It is **decoupled from the F2 vectorized (range-vs-range)
showdown** — at a decision-free node the hands are concrete.
The enumeration is bounded in search (≤2 board cards → ≤~1000 unordered
completions) and capped with a Monte-Carlo fallback for the pathological deep
runout (e.g. a pre-flop all-in). Both the **leaf rollouts** and the **solver's
forced-runout terminals** (§6.5) consume the same primitive, so their estimators
stay consistent. The `LeafConfig.use_decision_free_equity` flag (read by the leaf
here and by the solver as `cfg.leaf.use_decision_free_equity`) toggles the whole
effect for A/B measurement; `False` reproduces the paper's sampled single-board
runout.

**Expected improvement** (a hypothesis to be measured, not a guarantee):
lower-variance leaf/terminal values yield a cleaner meta-game/CFR signal and
steadier convergence, amplified by the search-lifetime cache below (a value
computed once and reused across ~10⁴ iterations benefits from being exact). The
net effect is to be quantified by bb/100 and convergence-stability deltas
between flag-on and flag-off runs.

The hands are **not** resampled here — the belief is integrated by the solver's
joint root sampling across MCCFR iterations (§6.5), not inside the leaf; the
hole-sampling helper (`_sample_all_holes`, which draws one assignment directly
from the joint belief distribution, §6.5) moves to the solver's root-sampling step. The profile is **fixed** by the caller;
nothing about the continuation choice is sampled inside this function (the
random-bias draw of the original design is gone). Determinism: `ctx.rng` drives
action sampling and the board runout uses the engine's global `np.random` deal,
so tests pin both seeds for bit-exact reproducibility (RNG unification is
deferred §6.7 plumbing).

**Caching across the whole search, not just a traversal.** The four bias
policies are static blueprint reweightings, so for a fixed `(leaf public_key,
concrete-hand tuple, profile)` the continuation value is **invariant across all
of the search's CFR iterations** — only the meta-game's *weighting* over
profiles changes between iterations, never the per-profile value. The value is
therefore memoized in a search-lifetime table keyed by that triple (not
recomputed per visit), and the table is the unit of work parallelized in §6.7.
The `n_rollouts` Monte-Carlo estimate is computed **once** per key on first
demand.

#### 6.4.2 Performance notes for the solver (row 6.2)

Profiling the two implemented CFR consumers (the vector showdown and the MC leaf
`continuation_value`) established where time actually goes; the solver must be
built with these in mind. The hand *evaluator* is already batched and is **not**
the bottleneck on either path — these notes are about how the solver *uses* the
shared primitives.

1. **`runout_equity` scoring has a fast path; it fires for any completion with no
   rank tie, regardless of pot count.** The side-pot structure is fixed by the
   frozen contributions (board-independent), so no-tie completions are settled by
   a vectorised `argmin`-per-pot + `bincount`; tie completions use the exact
   scalar `Pot.compute_utility`. The MCCFR **forced-runout terminals** (the
   `env.payout` → decision-free branch, §6.5 step 3) and the leaf rollouts both
   call this same primitive, so both inherit the speedup for free. Measured: a
   heads-up flop leaf (`continuation_value`, 20 rollouts) dropped from ≈40 ms to
   ≈17 ms (2.4×), with `runout_equity`'s share falling from 76% to 40%. No
   correctness assumption is needed — the scalar path is the exact fallback.

2. **A decision-free runout is invariant in the holes + snapshot, so memoise it.**
   `continuation_value` already caches per call on the env's `_runout_info`
   snapshot `(prefix, pot_chips, active)` (holes are fixed across its rollouts),
   collapsing the ~9 runouts/leaf to 2–3 distinct integrations. The solver should
   extend this to a **search-lifetime** cache: the §6.4.1 memo on
   `(leaf public_key, concrete-hand tuple, profile)` subsumes the leaf's runout,
   but the **MCCFR forced-runout terminal is a separate call site** — key its
   runout on `(hands, prefix, pot_chips, active)` so the same all-in reached from
   many lines/iterations is integrated once, not per visit.

3. **Vector regime: rank each board once, not per iteration.** `rank_combos_on_board`
   is **reach-independent**, so it is a one-time setup cost amortised across all
   CFR iterations; the per-iteration cost is the ≈0.1 ms `showdown_cfv` settle.
   For a turn subgame (only the river varies) `solve()` should precompute the
   ≤~46 river rankings **once** and reuse them every iteration — never re-rank in
   the iteration loop. This is the right lever; a board-specialised evaluator
   would only optimise the already-amortised ranking cost and is **not** needed.

4. **The MCCFR walk must use make/undo, not deepcopy.** `step_in_place` / `undo`
   (§6.1) are the advance API and must drive the tree walk. The leaf's
   per-rollout `with_hole_cards` deepcopy is harmless at heads-up scale (~0.07 ms)
   but is linear in node count; the solver walk must not deepcopy per node. A
   rollout that never calls `undo` still pays `_capture_undo_token` it discards —
   a snapshot-free advance is an option there (negligible now, flagged for scale).

5. **info_set construction is the per-node cost that dominates at solver scale.**
   `policy_state_for` / info_set build is ≈0.047 ms/node — noise over the leaf's
   ~2.7 nodes, but linear in the full walk's node count. When wiring
   `CFRTables` / `SolverState` access, consider interned / integer-keyed infosets
   rather than string keys, and profile it on the **real** walk, not the leaf.

### 6.5 Subgame solver (`solver.py`)

```python
@dataclass(frozen=True)
class SolverConfig:
    """Static hyperparameters; identical across every search in a session."""
    max_iterations: int = 10_000
    max_wall_seconds: float = 15.0
    discount_interval: int = 1_000         # Linear-CFR discount cadence
    leaf: LeafConfig

class SolverState:
    regret:    Dict[Key, np.ndarray]       # float64, width = legal actions at the node
    strat_sum: Dict[Key, np.ndarray]       # cumulative strategy, same width
    legal_at:  Dict[PublicKey, Tuple[str, ...]]
    actor_at:  Dict[PublicKey, int]
    frozen:    Dict[Key, np.ndarray]       # pinned σ for the bot's actual-hand rows (§5)

@dataclass
class SearchResult:
    policy: SearchPolicy                   # final iteration — the bot plays this
    average_policy: SearchPolicy           # weighted average — belief updates
    state: SolverState                     # warm-start carrier / freeze store
    iterations_run: int
    wall_seconds: float

def solve(
    root_env: PokerEnv,                    # deepcopied by the caller
    ctx: SubgameContext,
    cfg: SolverConfig,
    warm_start: Optional[SolverState] = None,
) -> SearchResult: ...
```

Pluribus uses one of **two** CFR forms in the subgame depending on its size and
the part of the game (paper, p.22–23). The choice is configurable; the default
rule:

- **MCCFR regime** — *large / early* subgames: round 1, all of round 2, and any
  large multiway later subgame.
- **Vector regime** — *small / late* subgames: heads-up turn/river.

#### Shared structure (both regimes)

Tables keyed `Key = (public_key, hand_row)`. The **hand row** is per-combo on the
root street (lossless) and a 200-bucket LUT cluster (`env.cluster_for`) on later
streets (→ 500 with the new LUT). Node **width** is per-node — injected off-tree
actions extend the legal set at specific public keys. The subgame action tree is
built from the **coarse search raise-size set** (§3, §6.3), not the blueprint's;
raises off that set are injected. Per-row regret matching reuses
[`calculate_strategy_from_row`](../poker_ai/blueprint/tree_utils.py); do **not**
reuse the fixed-width `accumulate_regrets` / `get_node_strategy`. Tables are
in-memory (no lmdb / no disk), live for one hand (warm-start across re-searches),
discarded at hand end. Linear-CFR discount of `regret` and `strat_sum` by
`d = (t/Δ)/(t/Δ + 1)` every `discount_interval`. Dual stop on `max_iterations` /
`max_wall_seconds`. **Freezing** (§5): at infosets where the bot already acted
this round, the bot's **actual-hand** row returns the pinned σ from `frozen` and
skips regret updates, while all other hands and all opponent infosets stay free.
**Warm-start**: a non-`None` `SolverState` re-searches the same root after an
injection, reusing rows in place and rebuilding widened nodes on first visit; the
freeze map clears when the root advances to a new round. The bot **plays**
`policy` (final iteration) at the actual hand's row; `average_policy` (normalized
`strat_sum`) feeds the next round's belief update.

#### MCCFR regime (large / early)

External-sampling Linear MCCFR (Linear-discounted, as the blueprint), **without
CFR-P pruning** — see step 2. Per traversal:

1. **Sample one root hand assignment directly from the joint belief
   distribution.** The subgame root is a single chance node over the public state
   `G`: its outcomes are the complete, mutually card-disjoint hole assignments `h`
   (one hole pair per dealt seat — live seats from `ctx.ranges`, seats that folded
   before the root from `ctx.folded_ranges`, for card removal), and outcome `h` has
   probability `P(h) = π^σ(h) / Σ_{h'∈G} π^σ(h')` — the normalized **joint** reach
   under the belief profile σ. The joint weight of an assignment is the product of
   the seats' per-combo belief weights, zeroed whenever two seats share a card.
   Each traversal draws **one** assignment directly from this joint `P(h)`
   (`_sample_all_holes`) — **not** from independent per-seat marginals — so
   inter-seat card removal is reflected in the draw itself. Every seat is part of
   the joint draw, **including the bot**: its hole is sampled each iteration, never
   pinned to its actual hand. Hands an opponent folds with probability 1 carry zero
   joint mass and are never drawn, which is why unsafe search is cheap.
2. Pick a traversing player `i`; **explore all of `i`'s actions**, **sample one
   action** for each opponent (regret-matched σ) and **one outcome** at each chance
   node; accumulate regret on `i`'s rows. **Every traverser action is always
   explored** — the blueprint's negative-regret pruning (CFR-P) only engages after a
   long warm-up and once a regret falls below −300M, neither of which is reached in a
   short search, so it is omitted entirely here rather than carried as dead code.
3. **Terminal** node → concrete hands → `env.payout`.
4. **Depth-limit leaf** (round-1, multiway round-2) → the **continuation
   meta-game**, solved as an ordinary action. At `i`'s meta-infoset — keyed
   `(leaf_public_key, "META", i, i's hand_row)`, which carries **no other seat's
   choice**, so the simultaneous choice stays infoset-consistent — `i` explores its
   4 continuation strategies while each opponent's choice is **sampled** from its
   current meta-strategy. The value of a fully-chosen profile is
   `leaf.continuation_value(frontier_env, profile, ctx)` (§6.4). Freezing for the
   bot applies only on traversals whose sampled bot hand equals the actual hand.

This mirrors the structure of [`blueprint/cfr.py`](../poker_ai/blueprint/cfr.py)
but with variable-width rows, the belief-sampled root, the meta-game action, and
freezing.

#### Vector regime (small / late)

Vector-form Linear CFR carrying a per-combo reach vector per player
(`reach[p] = ctx.ranges[p]` at the root). **Every action is expanded at every
decision node** (no action sampling); **one board runout is sampled per iteration**
at the subgame's chance nodes, the tree then deterministic. The implementation does
**alternating updates** — one tree pass per seat, the traverser's per-combo regret
updated from opponent-reach-weighted counterfactual values (summed across the
opponent's actions, since the opponent mixes per *its* combo), the strategy sum
weighted by the traverser's own reach. The betting tree is hole- and card-independent,
so a single make/undo env walk serves every combo at once; the **river is sampled
from `ctx.rng`** over the ranges' candidate set (the cards not on the board) and the
engine's own deal is ignored — the correct chance distribution over ranges, and free
of any global-RNG dependence (the MCCFR path still deals via the engine's global
`np.random`, §6.4.1). Regret / strategy-sum updates are reach-weighted per row
(float64); opponent reach on combos conflicting with the acting combo is zeroed (card
removal, idiom from `ranges._zero_conflicting`). The per-combo rows persist as
`(n_combos, width)` matrices keyed by `public_key` in the shared `SolverState` (the
combo axis is `combo_index`, lossless at every depth), and `solve()`'s Linear-CFR
discount and `SearchPolicy` read them unchanged. These subgames extend to the **end of the game**, so
their terminals are **showdowns**, evaluated by the env-owned **vectorised payout**
`PokerEnv.vector_payout` on the sampled board (F2, settlement math in
[`environment/range_showdown.py`](../environment/range_showdown.py)):
`rank_combos_on_board` ranks every combo on the completed board once (the shared
`default_evaluator`, so it scores identically to `compute_winners` / `runout_equity`),
then `showdown_cfv` settles each acting combo against the opponent's reach-weighted
combo distribution. The regime passes only the CFR quantities — the traverser seat,
the opponent reach, and the sampled river — and the env owns the stake (from
`terminal_contributions`), showdown-vs-fold detection, board completion, card
removal, and the board-keyed ranking cache. A **fold** terminal settles on the
board the hand actually reached: the engine force-deals the community out to five
even on an early fold, so `len(community_cards)` cannot tell a turn-side fold from
a river-side one — `PokerEnv.terminal_board_len` (the board length captured before
that force-deal) does, and a pre-river fold therefore does card removal against the
shorter board it saw (not the dealt-out completion, and not the sampled river it
never reached). The value of acting combo `i` is `v_i = stake · (W_i − L_i)`,
where `W_i` (resp. `L_i`) is the opponent reach on combos `i` beats (resp. loses
to) and `stake` is each player's matched contribution — heads-up showdown is
winner-takes-pot, so ties net zero and there are **no side pots**. The win/tie/lose
aggregation is the sorted-rank trick of ref. 42, implemented **fully vectorised**
(one `argsort` into rank groups, then `bincount`/`cumsum` prefix-and-suffix sums —
no per-group Python loop, **no n² matrix**): card removal between the two ranges is
exact by subtracting, per acting combo, the opponent mass sharing either of its two
cards. A settle costs ≈0.1 ms at full-deck `n_combos = 1326`; the only heavier cost
is the board ranking (O(n·evaluator), the scalar `Evaluator`), which is computed
**once per sampled board** and reused across every showdown terminal in that
iteration. For a turn subgame (only the river varies per iteration) the solver
(6.2) can precompute the ≤~46 river rankings once rather than ranking per iteration;
a vectorised/native evaluator (perf only) stays out of scope here. Heads-up only (a single
opponent → no opponent–opponent removal term, §10). `env.payout` is **not** used
here — it scores a single concrete hand assignment, not a range; it remains the
terminal source only in the MCCFR regime and inside `continuation_value`'s
concrete-hand rollouts. The vector regime has **no** continuation meta-game
(terminal leaves only).

### 6.6 Search-aware agent (`agent.py`)

```python
class SearchAgent:
    def __init__(
        self,
        leaf_policies: Dict[BiasClass, Policy],   # the four §4 variants
        blueprint_policy: Policy,                 # round-1 play + first-round Bayes
        solver_cfg: SolverConfig,
        rng: np.random.Generator,
    ): ...

    def on_hand_start(self, env: PokerEnv, my_seat: int): ...
    def on_board_update(self, new_cards: Sequence[int]): ...
    def on_observed_action(self, env_before: PokerEnv, seat: int, action: str): ...
    def act(self, env: PokerEnv) -> str: ...
```

Per-hand state held on the agent:

- `tracker: RangeTracker` (all live seats incl. the bot, §6.2)
- `my_seat: int`; `my_hole` read from `env.players[my_seat].cards`
- `last_search: Optional[SearchResult]`
- `pending_actions: List[Tuple[seat, env_before, action]]` — the
  round's action buffer for the boundary Bayes update

Lifecycle (Algorithm 2):

1. **`on_hand_start`** — reset tracker, clear the action buffer, call
   `env.reset_overlay()`. No search (round 1 plays the blueprint).
2. **Round boundary** (`on_board_update`, fired when a new betting round begins) —
   (a) replay the buffered `(seat, env_before, action)` tuples through
   `tracker.on_action` with a `sigma_for_combo` closure from
   `last_search.average_policy` (or `blueprint_policy` if no search has run yet
   this hand), Bayes-updating every seat's range, and zero board-conflicting
   combos; (b) make the new round's public state the root and **immediately run
   `solve`** (Algorithm 2 `CheckNewRound`) — search completes *before* the bot is
   asked to act; (c) clear the buffer and the freeze map.
   ```python
   root_env = copy.deepcopy(env_at_round_start)
   ctx = SubgameContext.from_runtime(
       root_env, self.my_seat, my_hole,
       self.tracker.snapshot(), self.tracker.folded_snapshot(),
       self.solver_cfg.leaf, self.rng,
   )
   self.last_search = solve(root_env, ctx, self.solver_cfg)
   ```
3. **`act`** —
   - Round 1 (no search ran): sample from `blueprint_policy`.
   - Rounds 2–4: read the action from the already-computed
     `last_search.policy` (final iteration) at the actual hand's row — **no solve
     here**. Record the σ used into `last_search.state.frozen` for that infoset.
4. **`on_observed_action`** — append to `pending_actions`. On rounds 2–4, if the
   action was off-tree (the runtime injected it), **re-search the same root** with
   `warm_start=self.last_search.state`; the frozen rows keep the bot's
   already-taken actions fixed for its actual hand. On round 1, apply the
   randomized pseudo-harmonic mapping; or, when the > $100 / ≤ 4-players trigger
   fires (§5), `solve` the round-1 root (subgame to the end of round 1) and store
   it so `act` reads from it.

The agent never sees chips. Off-tree detection and injection are the
runtime's concern at the chip→string boundary (§6.3).

Wiring: [poker_ai/terminal/runner.py](../poker_ai/terminal/runner.py) gets a
new `--agent search` branch that instantiates `SearchAgent`, plumbs
`on_hand_start` / `on_board_update` / `on_observed_action` through the play
loop, and replaces the inline offline-lookup block with `agent.act(env)`.

### 6.7 Performance and parallelism

The dual budget (10 000 iterations **and** 15 s) is only a real-time budget if a
single search fits inside it. The dominant cost is **not** the CFR arithmetic;
it is environment cloning. A naive traversal that advanced the env with a
`copy.deepcopy` of the mutable game state (players, pot, deck, history) per move
would pay that copy at every one of the 10⁵–10⁶ nodes a search visits in the
MCCFR regime. The make/undo traversal below removes it. The work is addressed in
three tiers; the constant-factor tier comes first because no amount of
parallelism rescues a ruinous per-node copy — it only spreads it across cores.

#### Tier 1 — single-thread constant-factor (prerequisite for the budget)

- **Make/undo traversal.** The solver descends with `env.step_in_place(a)` and
  ascends with `env.undo(token)` (§6.1) — one mutable env per traversal, the
  undo token restoring exactly the mutable per-hand state `__deepcopy__` copies.
  This removes the per-node deepcopy entirely (the largest single win) and is the
  structural prerequisite for a native/`nogil` inner loop in Tier 2.
  `step_in_place`/`undo` is the **sole** way to advance the env; `deepcopy` is
  retained only to snapshot a search root and inside `with_hole_cards`. The same
  pair backs the **blueprint CFR trainer**
  ([poker_ai/blueprint/cfr.py](../poker_ai/blueprint/cfr.py)) and the
  average-strategy pass ([strategy.py](../poker_ai/blueprint/strategy.py)) — so
  the win applies to offline training as well as online search.
- **Search-lifetime leaf-value cache.** Continuation values are invariant across
  CFR iterations for a fixed `(leaf public_key, concrete-hand tuple, profile)`
  (§6.4), so the `n_rollouts` estimate is computed once per key and reused —
  collapsing the second bottleneck and yielding the embarrassingly-parallel work
  unit for Tier 2.
- **Per-row strategy memoization.** Cache the regret-matched σ per
  `(public_key, hand_row)` for the duration of an iteration sweep; invalidate on
  the `discount_interval` tick rather than recomputing
  `calculate_strategy_from_row` on every node visit.
- **Direct 7-card evaluator (optional).** *Once make/undo removes per-node
  cloning,* profiling the MC path identifies the **hand evaluator invoked via
  `runout_equity`** (and inside the leaf rollouts / the vector showdown) as the
  dominant remaining cost (~58–74% of MC time) — not env cloning. The 21-subset
  batch path (`evaluate_batch`) can be replaced by a direct 7-card evaluator
  (TwoPlusTwo table or 7-card perfect hash) on the **order-only** paths
  (showdown/runout need *ordering*, not the exact `[1,7462]` rank), validated
  order-equivalent against the proven `Evaluator`. Est. ~5–10× on the evaluator,
  ~2× overall (Amdahl), benefiting both regimes and the leaf. Optional, and
  deferred until the full pipeline can be measured (row 9.3).
- **Flat hot-loop state (longer-term).** Represent the traversal's mutable state
  as struct-of-arrays (chip/bet/fold vectors, board, deck cursor) rather than
  `Player` / `Pot` / `Deck` objects. This makes `undo` a slice restore and lets
  the inner loop drop into `numba @njit(nogil=True)` — the precondition for
  threads to scale past the GIL.

#### Tier 2 — parallelism (paper-aligned, after Tier 1)

The supplement runs search across cores, sampling **one set of public board
cards per thread**; the §2 non-goal is distribution across *machines*, not
cores.

- **Vector regime — one board per worker.** Each worker runs vector-form Linear
  CFR on its own sampled board runout; the per-board strategies/regrets are
  averaged. Embarrassingly parallel and naturally coarse-grained — the cheapest
  real win, and a direct match to the paper's per-thread scheme.
- **MCCFR regime — batched parallel traversals.** Workers run batches of
  external-sampling traversals against the shared regret tables, merging
  per-worker accumulators at the `discount_interval` boundary (summed, then
  discounted) to keep the merge lock-free.
- **Leaf-value precompute.** The Tier-1 leaf cache is a pure function of
  `(leaf, hands, profile)` with no shared mutable state — fan it out across a
  pool ahead of / alongside the solve.
- **GIL.** Pure-Python traversal does not scale on threads. Two routes:
  `multiprocessing` with coarse chunks (whole boards in the vector regime,
  iteration batches in MCCFR) where the per-task state-serialization cost is
  amortized; or the `numba nogil` native inner loop from Tier 1, which scales on
  threads with shared memory. Start with `multiprocessing` on the vector regime
  (coarse, naturally chunked by board); reserve the native route for when MCCFR
  fine-grained sharing dominates.

#### Tier 3 — lower-order

- Integer-encode the `(public_key, hand_row)` table keys once the schema is
  stable, to drop tuple-hashing from the hottest dict lookups.
- Reuse preallocated regret / strategy-sum buffers instead of allocating a fresh
  `np.ndarray` per node.
- Warm-start reuse across re-searches (§6.5) is already specified — it amortizes
  table construction across an injection's re-solve and stays.

Parallelism is **opt-in and seed-deterministic**: a single-worker run reproduces
the serial result bit-for-bit; multi-worker runs fix per-worker substreams
(`np.random.SeedSequence.spawn`) so a given `(seed, n_workers)` is reproducible.

## 7. Implementation Order

| # | Component | File(s) | Status | Blocks on |
|---|---|---|---|---|
| 0 | Combo helpers + `Policy` ABC + `BlueprintPolicy` | `environment/utils.py`, `environment/poker_env.py`, `poker_ai/search/policy.py` | **done** — ×5 probability reweighting (`_reweight_bias`, `bias_multiplier=5.0`) | — |
| 2 | Env overlay + `with_hole_cards` (done) + `SubgameContext` + new env accessors | `environment/poker_env.py`, `poker_ai/search/context.py` | **done** — all-seat `ranges` + `folded_ranges`; `DepthLimit` descriptor; `public_key`, `cluster_for`, `n_raises_this_round` | 0 |
| 3 | Range tracking | `poker_ai/search/ranges.py` | **done** — tracks the bot's own observer-perspective range (incl. `my_seat` in `snapshot`); `on_action` is the round-boundary replay primitive servicing every seat | 0 |
| 4 | Chip↔action env API + search raise-size set | `environment/poker_env.py` | **done** — pseudo-harmonic translation (`_translate_fraction`, randomized + deterministic); history canonicalization (`_canonicalize_history`/`_blueprint_info_set`/`policy_state_for(for_blueprint=…)`, no-op on on-tree histories); `SEARCH_RAISE_SIZES_BY_STAGE` + `search_raise_fractions`/`search_raise_actions`; `string_for_chips` tolerance snap removed | — |
| 5.1 | Decision-free runout evaluator (§6.4.1) | `environment/poker_env.py` | **done** — `is_decision_free` + `runout_equity` (exact board-average over completions, side-pots via `Pot.compute_utility`, cap+MC fallback); pre-runout snapshot recorded at the force-resolve (`_runout_info`, undo/deepcopy round-tripped); brute-force tested. Shared by the leaf (5.2) and the solver's forced-runout terminals (row 6) | 0 |
| 5.2 | Continuation values | `poker_ai/search/leaf.py` | **done** — `leaf_value` → `continuation_value(frontier_env, profile, ctx)`; fixed profile, per-seat scalar, rollout from concrete hands (no resampling); blueprint-canonical lookups; decision-free exact equity via 5.1 gated by `use_decision_free_equity`; obsolete hole-samplers deleted (joint sampler is the solver's, row 6) | 0, 4, 5.1 |
| 6.1 | Vectorised range-vs-range showdown (F2) | `environment/range_showdown.py` (env-owned) | **done** — `rank_combos_on_board` + `showdown_cfv`/`reach_after_removal`; O(n log n) sorted card-removal sweep, heads-up winner-takes-pot `stake·(W−L)`, no n² matrix; board-keyed ranking cache; depends on the shared `default_evaluator` only. Brute-force tested + cross-validated against concrete `payout` to the chip. Surfaced as `PokerEnv.vector_payout`, called by the vector regime (6.2) | 0, 2, 3 |
| 6.2 | Rest of solver (MCCFR + vector CFR loops) + `SearchPolicy` | `poker_ai/search/solver_state.py`, `mccfr.py`, `vector.py`, `solver.py`, `policy.py` | **done** — **both regimes + `SearchPolicy`** (composition: `SolverState` shared data+ops, `_MCCFRSolver` with joint root sampling, external-sampling traversal, meta-game-as-action leaf, freezing, warm-start widening; `solve()` orchestrator + regime selection + Linear-CFR discount + dual stop; forced-runout terminals via `runout_equity` under `use_decision_free_equity`). **Vector regime** (`_VectorSolver`, HU turn/river): alternating-updates vector-form Linear CFR carrying per-combo reach vectors, all-actions-expanded, **one river sampled per iteration from `ctx.rng`** (engine deal ignored → no global-RNG dependence); **all terminal settlement is delegated to the env-owned `PokerEnv.vector_payout`** (6.1) — the regime passes only the traverser seat, opponent reach, and sampled river, and does no stake/showdown/fold/ranking itself; per-combo rows persist as `(n_combos, width)` matrices keyed by `public_key` in the same `SolverState`, read unchanged by `SearchPolicy`. Unit + integration + fast convergence/stability tests for both. The exact-equilibrium **independent brute-force CFR cross-validation is landed** (§9; `test/search/brute_force_cfr.py` + `test_equilibrium_oracle.py`, `slow`): one oracle solves a HU-river subgame to exact Nash and both regimes match it on best-response exploitability (tight for vector) and game value (the hard gate for sampled MCCFR) | 0, 2, 3, 5.1, 5.2, 6.1 |
| 7 | Search-aware agent | `poker_ai/search/agent.py`, terminal wiring | todo | 3, 4, 6.2 |
| 8 | CLI, config, tests | existing Click runner, `test/search/` | partial | 0–7 |
| 9.1 | Make/undo traversal (`step_in_place` / `undo`) (Tier 1, §6.7) | `environment/poker_env.py` | **done** — the **sole** advance API (`apply_action` deleted); the blueprint CFR + strategy passes and the search solver all traverse via make/undo (the solver reuses one restored env across its regret and strategy passes); LIFO round-trip + no-leak tested | — |
| 9.2 | Search-lifetime caches: leaf-value / forced-runout cache + per-row σ memoization (Tier 1, §6.4.2, §6.7) | `poker_ai/search/solver.py`, `mccfr.py`, `leaf.py` | **todo (deferred)** — memoize `continuation_value` by `(leaf public_key, hand tuple, profile)` and `runout_equity` by `(holes, prefix, pot, active)`; the latter **shared across a leaf's four bias calls** is the main round-1 win (profiling shows the meta-game recomputes identical runouts once per bias — the redundancy a cache removes), plus a per-row σ cache for the hot loop. (Preflop already scores all-in terminals by `env.payout` to skip the 5-card-runout cap — landed in 6.2.) **Deferred until the full pipeline can be evaluated end-to-end** | 6.2 |
| 9.3 | **Optional** direct 7-card evaluator (Tier 1, §6.7) | `environment/evaluator.py` (consumed by `runout_equity` 5.1 + `rank_combos_on_board` 6.1) | **todo — optional** — replace the 21-subset batch path with a direct 7-card evaluator (TwoPlusTwo table or 7-card perfect hash) on the **order-only** paths (showdown/runout need *ordering*, not the exact `[1,7462]` rank), validated **order-equivalent** against the proven `Evaluator` (C(52,5) exhaustive core + large random 7-card argsort sample). Profiled as the dominant MC-path cost (~58–74%); est. **~5–10× on the evaluator, ~2× overall** (Amdahl), benefiting both CFR regimes and the leaf. ~1–2 days + a ~130 MB table artifact (or a smaller perfect-hash table). **Deferred until the complete pipeline can be evaluated**; pursue only if the evaluator is confirmed on the real-time critical path | 5.1, 6.1 |
| 10 | Flat hot-loop state + `numba nogil` inner loop (Tier 1, §6.7) | `poker_ai/search/solver.py` | todo — optional; precondition for thread scaling | 9.1 |
| 11 | Parallel search: one-board-per-worker (vector) + batched parallel traversals (MCCFR) (Tier 2, §6.7) | `poker_ai/search/solver.py` | todo — `multiprocessing` first; seed-deterministic | 9.1 |

## 8. CLI

```
# Play with search
poker_ai play \
    --agent search \
    --blueprint blueprints/base \
    --iterations 10000 \
    --time-limit 15s \
    --workers 0          # 0 = serial (default); N = parallel search across N cores (§6.7)
```

## 9. Testing Strategy

- **Unit tests** under `test/search/` for each module, using small
  deterministic fixtures and `@pytest.mark.requires_lut` where the card-info
  LUT is needed.
  - `policy.py`: ×5 reweighting — biased class mass exactly 5× its
    pre-normalization value; `bias_multiplier=1` is numerically identical
    to base blueprint regret matching; uniform fallback on absent rows;
    overlay actions get zero mass with canonical renormalization.
  - `environment/poker_env.py` chip API: `canonical_raise_fractions`
    matches `legal_actions` gating; `chips_to_add` round-trips canonical
    action strings; pseudo-harmonic golden values (e.g. `A=0.5, B=1,
    x=0.75 ⇒ P(A)=3/7`); randomized variant matches the formula in
    frequency, deterministic variant picks the ≥ ½ side; the coarse search
    raise-size set is a subset of the blueprint fractions and ≤ 5–6 per
    node; `public_key` stable across seats at the same public node;
    `cluster_for` agrees with `info_set`'s embedded cluster.
  - `environment/poker_env.py` overlay: `inject_action` idempotent,
    persists across deepcopies, visible only at the matching public
    state; `reset_overlay` clears; `with_hole_cards` leaves the original
    untouched.
  - `context.py`: `from_runtime` board mask; `street_at_root`;
    `depth_limit` verdicts for all four §3 situations, including the
    after-2nd-raise mid-round cutoff and end-of-game subgames; frozen
    field set.
  - `ranges.py`: Bayes update under a hand-written `sigma_for_combo`;
    board-conflict zeroing; uniform fallback on the numerical floor;
    the bot's own range is tracked and updated; opponents' ranges
    exclude the bot's actual cards while the bot's own does not.
  - `environment/poker_env.py` decision-free runout (§6.4.1): `runout_equity`
    matches an independent brute-force enumerate-and-score reference to exact
    integer chips (heads-up and 3-way unequal-stack side pots); `is_decision_free`
    is True only at an all-in showdown over an incomplete board (False for
    fold-terminals, complete-board river all-ins, and non-terminal states);
    zero-sum; card removal excludes every dealt hole; the exact equity equals
    the mean of the env's own sampled-board resolution; make/undo + deepcopy
    round-trip the `_runout_info` snapshot; the cap fallback samples and warns.
  - `leaf.py`: deterministic output under a fixed seed; for a fixed
    `profile`, only `policies[profile[seat]]` is consulted for that seat
    and with that bias (heterogeneous profile honoured; a missing acting seat
    raises); rolls from the env's already-set concrete hands (no resampling —
    every `with_hole_cards` call gets the frontier holes); blueprint-canonical
    lookups (`for_blueprint=True`); payoff read from `env.payout`; per-seat
    scalar return. **Decision-free flag**: with `use_decision_free_equity=True`
    an all-in line equals the env's exact `runout_equity` and is runout-RNG
    independent; with `False` it takes the sampled single-board path and can
    differ (the paper baseline).
  - `environment/range_showdown.py` (§6.5 vector regime, F2): `showdown_cfv`
    matches an independent brute-force O(n²) all-pairs reference (built from the
    same shared evaluator, with card removal) to floating tolerance across boards
    and random reach vectors; opponent reach on combos sharing a card with the
    acting combo contributes **zero** (card removal), and a no-removal reference
    differs; equal-rank matchups net zero (ties); board-incompatible acting combos
    return zero; heads-up **zero-sum** (`Σ reach_a·cfv_a + Σ reach_b·cfv_b = 0`);
    CFV is **linear in `stake`**; for a one-hot pair the CFV equals the chip delta
    from `Pot.compute_utility` (engine-consistency); deterministic (no RNG).
  - `test_payout_consistency.py` (the env's payout family is aligned):
    `PokerEnv.vector_payout` against a one-hot opponent equals the engine's concrete
    net chips when the two hands are dealt and the same betting line replayed —
    across showdown/fold/all-in terminals and equal *and* unequal stacks (the
    matched-stake / uncalled-excess case).  A **turn-side fold** in a turn subgame
    is **river-independent**: `vector_payout(..., river=r)` is identical for every
    candidate river `r` and equals the card removal on the four-card turn board
    (regression for the force-deal masking bug); `terminal_board_len` round-trips
    through make/undo.
  - `solver.py`: regime selection picks MCCFR for round-1/round-2/large
    and vector for heads-up turn/river; terminates on either stopping
    criterion; per-hand tables released; root-street rows per-combo,
    later-street rows per-cluster. **MCCFR**: root hand-sampling draws one
    assignment directly from the joint belief distribution over card-disjoint
    assignments (`ranges` ∪ `folded_ranges`), not from independent per-seat
    marginals — so the empirical sample frequencies match the normalized joint
    reach `π^σ(h)/Σπ^σ(h')`; every traverser action is always explored (no
    pruning); meta-game keys contain no other seat's choice and on a toy
    dominant-class game the meta σ converges away from uniform ¼.
    **Vector**: the vectorised showdown matches a brute-force hand-vs-hand
    reference (with card removal) to floating tolerance; one board sampled
    per iteration. Both: frozen rows return the pinned σ across a re-search
    while other rows move; final and average policies differ mid-run; same
    seed ⇒ identical output.
  - `agent.py`: search runs **at the round boundary** (`on_board_update`),
    before the first `act` of the round; round-1 fast path and the $100 /
    ≤ 4-players trigger; ranges unchanged mid-round and updated once at the
    boundary under the average policy; `on_hand_start` calls
    `env.reset_overlay`.
  - **Performance & parallelism** (§6.7): `step_in_place` followed by `undo`
    restores the env to a state equal (field-by-field) to a `deepcopy` taken
    before the action, across a randomized action sequence (LIFO property), and
    a full make/undo traversal leaves the root env unchanged (no leak). The
    search-lifetime leaf-value cache returns a
    value equal to recomputing `continuation_value` for the same
    `(leaf public_key, hand tuple, profile)`, and is computed once per key
    (call-count assertion). Determinism under parallelism: `--workers 1`
    reproduces the serial result bit-for-bit; a fixed `(seed, n_workers)` is
    reproducible across runs; the MCCFR per-worker accumulator merge equals a
    serial run over the same total iteration count.
- **Independent CFR cross-validation** (**done** — `test/search/brute_force_cfr.py`
  + `test/search/test_equilibrium_oracle.py`, marked `slow`; one oracle validates
  the MCCFR *and* vector paths). A **brute-force full-enumeration Linear CFR** —
  written as a wholly independent loop (its own regret/strategy tables and
  recursion, reusing the env only for game *rules* and terminal payoffs, **not**
  `SolverState`/`_MCCFRSolver`/`vector_payout`) — solves a tiny subgame to its exact
  equilibrium; each solver path then converges to the same fixed point.
  - **Setting**: a **heads-up river** subgame with **small-support ranges** (two
    board-compatible combos per seat, over disjoint card sets; deep stacks so the
    river root offers graded raises, not just a shove). The river is the sweet spot:
    the board is complete, so there is **no board chance, no depth-limit leaf, and
    no decision-free runout** — terminals are plain showdowns/folds via `env.payout`,
    making the full tree **deterministic and exhaustively enumerable**; and it is
    **2-player zero-sum**, so a true Nash exists for both paths to match.
  - **Reference construction**: one DFS (make/undo) captures the hole-independent
    public betting tree; a concrete **payoff tensor** `M[leaf][(a,b)]` is built by
    replaying each terminal line under every card-disjoint support pair via
    `with_hole_cards` + `env.payout` (concrete settlement — *not* `vector_payout`,
    so the vector path's exploitability also cross-checks `vector_payout`). The
    oracle CFR, the game-value walk, and an exact **best-response** (per-hole
    backward induction) all run on that static tree.
  - **Both paths on the same subgame**: the vector regime is selected for HU
    river natively; the MCCFR path is exercised by instantiating `_MCCFRSolver`
    directly on the river root (its mechanics are street-agnostic — this skips
    only the meta-game leaf, which is covered separately) and driving the same
    iterate/discount loop as `solve()`. Both expose their average via
    `SolverState.average_sigma((public_key, combo_index))`, so they compare directly.
  - **Metric**: best-response **exploitability** drives to ~0, and the unique
    zero-sum **game value** of each path matches the oracle within tolerance.
    Exploitability is the tight gate for the full-width **vector** regime. For the
    sampled **MCCFR** regime the gates are (1) **game value** matches the oracle and
    (2) the **trained** part of the average is a genuine equilibrium. The latter is
    needed because external-sampling MCCFR trains *regrets* everywhere (the regret
    pass explores all the traverser's actions) but accumulates the *average* only
    along the strategy pass's sampled trajectory (as the production blueprint
    `update_strategy` does), so off-equilibrium-path infosets keep `strat_sum == 0`
    and fall back to **uniform** — a whole-tree best response deviates into those
    branches, giving the *raw* sampled-average exploitability a residual floor that
    is roughly constant in `T` (≈15–18 chips on a mixed-equilibrium board even at
    320k iterations), even though the game value is exact and the on-path average
    matches the oracle. Filling only the untrained infosets from the oracle drives
    exploitability back to ~0, isolating that the trained strategy is correct. (In
    live play those off-path branches are never relied on — an opponent deviation
    triggers a fresh re-search, §5.) Game value is preferred over raw strategy
    equality because a zero-sum Nash is value-unique but not necessarily
    strategy-unique.
  - The MCCFR path remains additionally guarded by the **fast convergence/stability
    test** (`TestConvergence`): the root range-average drifts little between *N* and
    *2N* iterations (a genuine convergence property, no external oracle).
- **Integration test**: one full hand versus a scripted opponent; assert
  search fires at the start of round 2, an off-tree raise triggers a
  re-search from the same root with the bot's acted σ frozen, and the agent
  plays legal hands end-to-end.
- **Empirical evaluation**: `--agent search` vs. plain `--agent offline`,
  ≥ 10 000 hands heads-up, expect a material bb/100 improvement.
- **Profiling**: single search call within the configured time budget;
  subgame memory released between hands. Per-decision log of (searched?,
  iterations used, wall-clock).

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Multiway round-2 subgames too expensive (per-seat continuation choices) | Solved with MCCFR (paper regime): external sampling visits one opponent action and one continuation choice per traversal, so cost is independent of the joint profile count; the after-2nd-raise depth cutoff keeps the tree shallow; coarse search raise-size set (≤ 5–6) bounds branching. |
| Vectorised showdown is subtly wrong (card removal, ties, side pots) | Test against a brute-force hand-vs-hand reference; the vector regime is heads-up late streets, so there are no *multiway* side pots — but an **unequal all-in** still arises (one stack short), handled by taking the **matched** (smaller) final contribution as the stake (the bigger stack's uncalled excess is excluded), reconstructed at the parent as `min(contribs[other], contribs[actor] + stack[actor])` because the engine resets the pot at the terminal; MCCFR (concrete `env.payout`) covers the multiway/side-pot cases. |
| Heads-up turn/river subgames (to end of game) exceed the time budget | One sampled board runout per iteration keeps per-iteration cost linear in tree size; both caps are CLI-configurable and tunable per street. |
| Opponent–opponent card-removal in the vector reach product is O(n_combos²) exact | Vector regime is heads-up only, so there is a single opponent range — no opponent–opponent term; mask only against the acting combo. |
| Dense per-combo ranges too slow under frequent updates | Updates fire once per round boundary, O(n_combos) per observed action, amortized against search cost. Fallback: cluster-bucketed ranges behind a flag. |
| Off-tree translation introduces exploitability | Paper-matching two-regime policy: randomized pseudo-harmonic on round 1, direct injection + re-search on rounds 2–4. |
| Per-node env deepcopy dominates wall-clock; serial search misses the 15 s budget | Make/undo traversal (`step_in_place` / `undo`, §6.7 Tier 1) is the sole advance path — no per-node copy; search-lifetime leaf-value cache removes the per-iteration rollout cost; both land before parallelism. |
| Parallel search non-deterministic or races on shared regret tables | Single-worker run reproduces the serial result bit-for-bit; per-worker RNG substreams via `SeedSequence.spawn`; MCCFR merges per-worker accumulators only at the `discount_interval` boundary (lock-free); vector regime shares nothing across boards. |
| `numba`/native inner loop (Tier 1, §6.7) adds a heavy build-time dependency for uncertain gain | It is optional and gated behind Tier 1's pure-Python make/undo, which alone is expected to reach the budget; pursue only if profiling shows MCCFR fine-grained sharing dominates. |
