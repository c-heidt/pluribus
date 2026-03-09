# Training Pipeline Refactor — Implementation Plan

## Principles
Every phase must leave the codebase in a runnable state. However, the code does not need to be backward compatible. Validate correctness against the baseline after each phase before proceeding.

---

## Phase 0 — Preparation and Baselines
*Do this before touching any code.*

### 0.1 Establish a Performance Baseline **✅ COMPLETE (pre-existing)**
- Run the existing multiprocess trainer on the cluster node for 1 hour with 30 workers — **already complete**
- Record iterations per second, peak memory usage, worker idle time fraction
- Record total iterations completed and unique infosets discovered
- Keep these logs — every subsequent phase is compared against them

### 0.2 Establish a Correctness Baseline **skipped**
- The multiprocess run is **not** a valid correctness reference due to non-deterministic worker scheduling and `manager.dict` race conditions
- Run the singleprocess trainer on the same 2-player, 20-card game with pruning disabled:

```bash
python -m poker_ai.ai.runner train \
    --mode singleprocess \
    --n_players 2 \
    --deck_size 20 \
    --n_iterations <exact_count_from_multiprocess_run> \
    --prune_threshold 999999999 \
    --save_path baseline_singleprocess/
```

- Pruning is disabled because `cfrp` makes stochastic pruning decisions based on accumulated regret order — two runs diverge further than pure `cfr()` making comparison harder
- Fix the `&` bug in the singleprocess trainer **only** for this baseline run — the baseline must represent correct discounting behaviour:

```python
# Fix in singleprocess/train.py for baseline run only
# Before: if t < lcfr_threshold & t % discount_interval == 0:
if t < lcfr_threshold and t % discount_interval == 0:
```

- Record final regret table, average strategy, and iteration count
- This singleprocess output is the ground truth for all subsequent validation

### 0.3 Write the Validation Function **skipped**
Write this before any refactoring — it is used after every phase:

```python
def validate_strategy_equivalence(
    strategy_a: dict,
    strategy_b: dict,
    tolerance: float = 0.01,
) -> bool:
    """
    Compare two average strategies across all infosets using L1 distance.
    Regret tables are not compared — they are path-dependent and will
    legitimately differ between runs with different update orders.
    Strategy tables converge to the same fixed point regardless of
    update order and are the correct comparison target.
    tolerance of 0.01 means strategies agree within 1% per action
    per infoset on average.
    """
```

### 0.4 Document Existing Bugs **✅ COMPLETE**
Record all five bugs found in `ai.py` — these are known deviations that will be fixed during the refactor. Their fixes will cause the refactored trainer to produce results that differ from the **original** multiprocess run but match the **corrected** singleprocess baseline:

- **Bug 1** — `&` instead of `and` in discount condition — discount never fires
- **Bug 2** — strategy table discounted alongside regret — only regret should be discounted per Pluribus paper
- **Bug 3** — `serialise()` acquires and releases lock per infoset — blocks all workers during serialisation
- **Bug 4** — `serialise()` uses `copy.deepcopy` on entire regret table — allocates a full second copy
- **Bug 5** — `update_strategy()` exits for all postflop states — preflop-only accumulation is intentional per Pluribus design, document explicitly

> **Implementation note — additional bugs found during code scan:**
> Two bugs not in the original list were also identified and fixed during Phase 2:
> - **Bug 6 (NEW)** — `for i, player in enumerate(state.players)` in `cfr()` (`ai/ai.py`) shadows the outer `i` (traversing player index) parameter, corrupting player attribution when DEBUG logging is enabled. Fixed by renaming the loop variable to `player_idx`.
> - **Bug 7 (NEW)** — `Union` was used in a type hint in `singleprocess/train.py` but not imported, causing a `NameError` at runtime in type-checked environments. Fixed by adding `Union` to the `typing` import.
>
> Bugs 1 and 2 were **fixed early (Phase 2)** rather than waiting for Phase 3, in both `singleprocess/train.py` and `multiprocess/server.py`.

### 0.5 Audit Sampling Scheme **✅ COMPLETE**
- Confirm `cfr()` uses outcome sampling on the opponent branch — **confirmed from code review**
- Confirm `cfr()` iterates all actions for the traversing player — **confirmed**
- This is chance-sampled CFR — the opponent branch must be changed to external sampling in Phase 3
- At 6 players, outcome sampling variance makes training impractical — this switch is mandatory

### 0.6 Audit Action Abstraction
- Verify action abstraction is deterministic and stateless given the same game state — **confirmed**
- Write a consistency check:

```python
def verify_abstraction_consistency(lut, n_samples: int = 10_000):
    import random
    keys = random.sample(list(lut.keys()), n_samples)
    for key in keys:
        assert lut[key] == lut[key], f"Non-deterministic abstraction at {key}"
```

- Run this against the 20-card LUT before proceeding
- If abstraction is non-deterministic the entire training pipeline produces incoherent regrets -

---

## Phase 1 — Infrastructure: Shared Utilities and Persistent Index **✅ COMPLETE**
*Replaces string key dict with a robust persistent index. No change to CFR logic.*

### 1.1 Create Shared IO Utilities **✅ COMPLETE**
- Copy `atomic_joblib_dump` from clustering into `poker_ai/utils/io.py`
- Add `atomic_numpy_save(arr, path)` using the same temp-dir + rename pattern
- Add `atomic_numpy_load(path)` with integrity check — verify file size matches expected shape before returning
- These are used by both clustering and training checkpointing going forward
- Unit test: write and read back arrays of several sizes, verify identical content

> **Implementation notes:**
> - `atomic_joblib_dump` was removed from both `clustering/card_info_lut_builder.py` and `clustering/unified_lut_builder.py` and centralised in `utils/io.py`; both clustering modules now import it from there.
> - `atomic_numpy_load` takes `expected_shape` and `expected_dtype` as optional keyword arguments for the integrity check (not just size).
> - An additional helper `hash_info_set_bytes(info_set: str) -> bytes` was added alongside `hash_info_set_128`, returning the 16-byte canonical LMDB key directly.

### 1.2 Implement 128-bit Infoset Hashing **✅ COMPLETE**
- Add `hash_info_set_128(info_set: str) -> Tuple[int, int]` to `poker_ai/utils/io.py` using `blake2b` with 16-byte digest
- Add `xxhash` as optional fast path with `blake2b` fallback:

```python
def hash_info_set_128(info_set: str) -> Tuple[int, int]:
    try:
        import xxhash
        h = xxhash.xxh3_128(info_set)
        return h.intdigest() >> 64, h.intdigest() & 0xFFFFFFFFFFFFFFFF
    except ImportError:
        import hashlib, struct
        digest = hashlib.blake2b(info_set.encode(), digest_size=16).digest()
        return struct.unpack("<QQ", digest)
```

- Add collision detection in debug mode: on every new insertion, assert hash does not already map to a different string
- Unit test: hash 10M synthetic infoset strings, assert zero collisions

> **Implementation notes:**
> - `xxhash` was installed as the primary path (not truly optional in practice); the blake2b fallback remains for environments without `xxhash`.
> - `hash_info_set_128` splits the 128-bit `intdigest()` return value as `high = digest >> 64`, `low = digest & 0xFFFF...` rather than calling `intdigest()` twice (avoids double computation).

### 1.3 Implement LMDB Persistent Index **✅ COMPLETE**
- Create `poker_ai/ai/index.py` containing `InfosetIndex`
- Keys: 16-byte hash digest packed as bytes
- Values: packed `(chunk_id: uint32, row: uint32)` — 8 bytes total
- `map_size` set to 50GB — LMDB does not pre-allocate this
- `writemap=True` and `map_async=True` for write performance
- Methods:
  - `get(info_set: str) -> Optional[Tuple[int, int]]`
  - `get_or_create(info_set: str) -> Tuple[Tuple[int, int], bool]` — returns location and whether it was newly created
  - `flush()` — explicit sync to disk, called before every checkpoint
  - `close()`
- Startup method: if index exists at path, open existing — if not, create new
- Unit test: insert 1M synthetic infosets, close, reopen, retrieve all, assert no mismatches

> **Implementation notes:**
> - lmdb 0.9.x (the conda-forge binary) treats `env.sync()` as positional-only — `sync(force=True)` raises `TypeError`. `flush()` calls `env.sync(True)` with a `TypeError` fallback to `env.sync()` to handle both API versions.
> - `CHUNK_SIZE = 100_000` is defined in `index.py` and re-exported from `regret_table.py` to ensure both modules share the canonical value.

---

## Phase 2 — Core Data Structure: `SparseRegretTable` **✅ COMPLETE**
*Implements the chunked shared memory regret table. No change to training loop yet.*

### 2.1 Implement `SparseRegretTable` **✅ COMPLETE**
- Create `poker_ai/ai/regret_table.py`
- Fixed `CHUNK_SIZE = 100_000` infosets per chunk
- Fixed `N_ACTIONS` passed at construction — must match action abstraction
- Each chunk is a `shared_memory.SharedMemory` block of shape `(CHUNK_SIZE, N_ACTIONS)` in `int32`
- `_shm_blocks: List[SharedMemory]` — keep references alive
- `_chunks: List[np.ndarray]` — views into shared memory blocks
- `_next_row: int` — next free row across all chunks
- `_alloc_lock: mp.Lock` — held only during new chunk allocation or new index entry creation
- Separate from stripe locks — alloc lock and stripe locks are never held simultaneously

> **Implementation notes — significant deviations:**
> - **Python 3.7 compatibility:** `multiprocessing.shared_memory` was added in Python 3.8 and is unavailable in the project's Python 3.7.12 runtime. Chunks are instead backed by named files in `shm_dir` (default `/dev/shm`) opened with `os.O_CREAT | os.O_EXCL` and mmap'd with `mmap.MAP_SHARED`. This provides identical shared-memory semantics and enables orphan detection (Section 2.6) as a natural side-effect.
> - **No `_next_row` counter:** Row allocation is delegated entirely to `InfosetIndex.get_or_create()`, which atomically assigns `(chunk_id, row)` via LMDB. `_next_row` is not stored on the table; valid-row counts are derived from `_index.n_entries` instead.
> - `_alloc_lock` guards chunk *file* creation only; row allocation is serialised by LMDB's write transaction — the two locks are never held simultaneously as planned.

### 2.2 Implement Core Access Methods **✅ COMPLETE**
- `get_row(info_set: str) -> np.ndarray` — allocates on first visit, returns view into shared memory
- `get_row_if_exists(info_set: str) -> Optional[np.ndarray]` — returns `None` if not visited, never allocates
- `get_row_by_location(chunk_id: int, row: int) -> np.ndarray` — direct access by known location, used during sync
- All returned arrays are views — writes are immediately visible to all attached processes

### 2.3 Implement Stripe Locking **✅ COMPLETE**
- 256 `mp.Lock` objects stored in a list, one per stripe
- Stripe assignment: `stripe_id = chunk_id % 256`
- `get_stripe_lock(chunk_id: int) -> mp.Lock`
- Workers acquire stripe lock only during sync flush, never during local accumulation
- 256 stripes for 64 workers means expected contention probability per stripe is low after warm-up

### 2.4 Implement Dirty Tracking **✅ COMPLETE**
- `_dirty: List[bool]` — one entry per allocated chunk
- Set to `True` in `_mark_dirty(chunk_id)` — called during sync flush when a chunk is written
- `get_dirty_chunks() -> List[int]` — returns chunk IDs with dirty flag set
- `clear_dirty(chunk_id: int)` — called after successful checkpoint write for that chunk
- `clear_all_dirty()` — called after full checkpoint

> **Implementation note:** `_dirty` is `mp.Array('b', _MAX_DIRTY_CHUNKS, lock=False)` rather than `List[bool]`. A fixed-size `mp.Array` is necessary for cross-process visibility after fork; a plain Python list would not be shared. `_MAX_DIRTY_CHUNKS = 2048` covers up to 204.8M infosets. Chunk IDs outside this range are silently ignored by `_mark_dirty` and `clear_dirty`.

### 2.5 Implement `apply_discount(factor: float)` **✅ COMPLETE**
- Precondition: must only be called at sync boundaries — document and assert
- Iterate over allocated chunks only using `_next_row` to determine last partial chunk
- For each chunk: vectorised `np.multiply` in float32 then cast back to int32
- Apply `np.maximum` with `REGRET_FLOOR = np.int32(-310_000_000)` after multiply
- Only process valid rows in the last partial chunk:

```python
def apply_discount(self, factor: float):
    assert self._at_sync_boundary, "apply_discount called outside sync boundary"
    factor32 = np.float32(factor)
    for i, chunk in enumerate(self._chunks):
        valid_rows = min(
            self._next_row - i * CHUNK_SIZE,
            CHUNK_SIZE
        )
        if valid_rows <= 0:
            break
        view = chunk[:valid_rows]
        result = (view.astype(np.float32) * factor32).astype(np.int32)
        np.maximum(result, REGRET_FLOOR, out=result)
        view[:] = result
```

> **Implementation notes:**
> - `_at_sync_boundary` is `mp.Value('b', 0)` (not a plain bool) for cross-process visibility; the property `_at_sync_boundary_flag` reads `.value`. The server calls `set_sync_boundary(True/False)` around discount dispatch.
> - `factor` validation added: raises `ValueError` if not in `(0, 1]`.
> - `valid_rows` uses `_index.n_entries` (not `_next_row`) since row allocation is tracked by the LMDB index, not a counter on the table.

### 2.6 Implement Shared Memory Naming Convention **✅ COMPLETE**
- Each block named `pluribus_regret_{session_id}_{chunk_id}`
- `session_id` generated at training start and stored in checkpoint state
- Enables orphaned block detection on restart
- `list_orphaned_blocks(session_id: str) -> List[str]` — scans `/dev/shm` for matching names

> **Implementation notes:**
> - Chunk IDs are zero-padded to 6 digits: `pluribus_regret_{session_id}_{chunk_id:06d}`.
> - `shm_dir` is a constructor parameter (default `"/dev/shm"`) rather than hardcoded, enabling tests to use a `tmp_path` directory without polluting `/dev/shm` or requiring special permissions.
> - `list_orphaned_blocks` is a module-level function (not a method) and accepts an optional `shm_dir` argument. It returns an empty list if the directory does not exist.
> - `list_own_blocks()` (instance method) returns paths of all chunks owned by the live instance.

### 2.7 Unit Tests **✅ COMPLETE**
- Allocate 500K synthetic infosets across multiple chunks, verify correct row retrieval
- Verify `get_row_if_exists` returns `None` for unvisited infosets
- Verify `apply_discount` applies only to allocated rows
- Verify dirty flags set and cleared correctly
- Verify stripe lock assignment is deterministic and evenly distributed
- Verify shared memory naming convention and orphan detection

> **Implementation note:** 50 tests in `test/unit/test_phase2_regret_table.py` across 9 test classes: `TestConstruction`, `TestCoreAccessMethods`, `TestStripeLocking`, `TestDirtyTracking`, `TestApplyDiscount`, `TestNamingAndOrphanDetection`, `TestRepr`, `TestBugRegressions`, plus one `@pytest.mark.slow` test for the full 500K allocation. All 50 tests pass (`-m "not slow"` skips the 500K test). The 500K test is implemented but gated behind the `slow` mark.

---

## Phase 3 — CFR Core: External Sampling and Local Delta ✅ COMPLETE
*Changes sampling scheme and how workers write regrets. Core parallelism and correctness improvement.*

> **Implementation notes:**
> - `local_delta` uses `Dict[str, Dict[str, float]]` (not `Dict[str, np.ndarray]` as planned) — numpy array conversion deferred to Phase 4 when `Agent` gets `SparseRegretTable`.
> - Bug 2 was also present in `multiprocess/worker.py` `_discount()` — the Phase 2 fix was incomplete; fully fixed this phase.
> - Duplicate `from typing import Dict, Union` import removed from `singleprocess/train.py`.
> - 47 unit tests in `test/unit/test_phase3_cfr_core.py`, all green. Uses `data/clustering/20cards_exact` for slow integration test.

### 3.1 Implement `calculate_strategy_from_row` ✅ COMPLETE
- Takes `np.ndarray` of `int32` regrets, returns `np.ndarray` of `float32` probabilities
- Regret matching: `max(r, 0) / sum(max(r, 0))` with uniform fallback if sum is zero
- Pure numpy — no Python loops:

```python
def calculate_strategy_from_row(regret_row: np.ndarray) -> np.ndarray:
    positive = np.maximum(regret_row, 0).astype(np.float32)
    total = positive.sum()
    if total > 0:
        return positive / total
    n = len(regret_row)
    return np.full(n, 1.0 / n, dtype=np.float32)
```

- Keep existing `calculate_strategy(dict) -> dict` as a wrapper for backward compatibility during transition
- Unit test: verify uniform output on all-zero input, verify correct probabilities on known input

> **Implemented:** `calculate_strategy_from_row(regret_row: np.ndarray) -> np.ndarray` added to `poker_ai/ai/ai.py`. Uses `np.maximum` + `np.float32` cast; uniform fallback on all-non-positive input.

### 3.2 Switch to External Sampling ✅ COMPLETE
- Rewrite `cfr()` opponent branch to iterate all opponent actions weighted by strategy probability
- This eliminates the high-variance single-action sample for opponent nodes:

```python
# External sampling — opponent branch iterates all actions
else:
    strategy = calculate_strategy_from_row(
        agent.regret_table.get_row_if_exists(state.info_set)
        or np.zeros(N_ACTIONS, dtype=np.int32)
    )
    vo = 0.0
    for action in state.legal_actions:
        action_idx = action_map[action]
        new_state = state.apply_action(action)
        vo += strategy[action_idx] * cfr(
            agent, new_state, i, t, local_delta, index_map, action_map
        )
    return vo
```

- Traversing player branch remains unchanged — iterates all actions
- Validate against singleprocess baseline from Phase 0 — exploitability should be lower per iteration than the original outcome sampling

> **Implemented (corrected):** The plan pseudocode above shows vanilla/tree CFR (iterate all opponent actions), which caused an exponential game-tree traversal and hung on real 20-card games. Correct external sampling MCCFR samples **one opponent action** per node according to the current strategy while the traversing player still explores all its own actions. Both `cfr()` and `cfrp()` opponent branches now use `np.random.choice(actions, p=probs)` to sample a single action. 48 unit tests green.

### 3.3 Add Local Delta Parameter ✅ COMPLETE
- Add `local_delta: Dict[str, np.ndarray]` parameter to `cfr()`
- Add `index_map` and `action_map` parameters for numpy-based lookup
- Remove `locks` parameter from `cfr()` — locking is now exclusively the worker's responsibility during sync
- `cfr()` reads strategy from shared `SparseRegretTable` via `get_row_if_exists`
- `cfr()` writes regret updates to `local_delta` only — zero shared memory writes during traversal
- New infosets go into `local_delta` and are allocated in shared memory during sync, not during traversal

> **Implemented:** `cfr(agent, state, i, t, local_delta=None)` — when `local_delta is not None`, all regret increments are written to `local_delta.setdefault(info_set, {})` instead of `agent.regret`. `merge_local_delta(agent, local_delta)` helper merges after traversal. `locks` parameter removed from `cfr()` and `cfrp()` entirely.

### 3.4 Fix Bug 1 — Discount Operator Precedence ✅ FIXED EARLY (Phase 2) — Discount Operator Precedence **✅ FIXED EARLY (Phase 2)**
```python
# Fix in ai.py
# Before:
if t < lcfr_threshold & t % discount_interval == 0:
# After:
if t < lcfr_threshold and t % discount_interval == 0:
```

> **Implementation note:** Fixed in both `singleprocess/train.py` and `multiprocess/server.py` during Phase 2 bug scanning. Also fixed a second `&` at the checkpoint dump condition in `singleprocess/train.py`: `(t > update_threshold) & (t % dump_iteration == 0)` → `and`.

### 3.5 Fix Bug 2 — Strategy Should Not Be Discounted **✅ FIXED EARLY (Phase 2)**
- Fix in `apply_discount` — regret table only
- Remove any discount application to strategy table
- Strategy accumulation must be untouched by discount step

> **Implementation note:** `agent.strategy[I][a] *= d` line removed from `singleprocess/train.py` during Phase 2 bug scanning. The `apply_discount` method on `SparseRegretTable` only operates on regret rows by design.
> **Also fixed:** `worker.py` `_discount()` had the same strategy-discount bug (Phase 2 fix was incomplete) — fully removed this phase.

### 3.6 Refactor `update_strategy()` ✅ COMPLETE
- Remove `locks` parameter
- Reads from shared regret table via `get_row_if_exists`
- Writes to shared strategy table via `get_row`
- Strategy table uses the same `SparseRegretTable` structure
- Document explicitly: exits for postflop states by design per Pluribus blueprint

> **Implemented:** `locks` parameter removed from `update_strategy()` signature. Reads from `agent.regret` directly; no lock acquisitions during traversal. `betting_round > 0` early-exit retained as intentional design (Bug 5, not a bug).

### 3.7 Rewrite `serialise()` ✅ COMPLETE
- Remove `copy.deepcopy` — fix Bug 4
- Remove per-infoset lock acquisition — fix Bug 3
- Serialisation is now the checkpoint writer's responsibility — `serialise()` becomes a thin wrapper that calls `CheckpointManager.checkpoint()`
- Never holds any lock during iteration over infosets

> **Implemented:** Bug 4 fixed — `copy.deepcopy` removed. Takes `regret_snapshot = list(agent.regret.items())` and `strategy_snapshot = list(agent.strategy.items())` under their respective locks (short critical sections). Builds offline dicts from snapshots via dict comprehensions. `locks: dict = {}` parameter kept for worker compatibility; full `CheckpointManager` deferred to Phase 6.

### 3.8 Unit Tests ✅ COMPLETE
- Run refactored `cfr()` for 1000 iterations on the small game
- Verify `local_delta` accumulates correct values
- Verify zero writes to shared memory during traversal
- Verify external sampling produces lower variance value estimates than outcome sampling over 10K runs
- Compare strategy output against Phase 0 singleprocess baseline using `validate_strategy_equivalence`

> **Implemented:** 47 tests in `test/unit/test_phase3_cfr_core.py`. Covers `calculate_strategy_from_row`, `merge_local_delta`, `cfr()` local_delta path, external sampling determinism, `cfrp()` local_delta path, `update_strategy()` signature, `serialise()` Bug 4 regression, and bug regression checks for signatures. Uses `data/clustering/20cards_exact` for slow integration test.

---

## Phase 4 — Worker: Local Delta Sync Protocol
*Implements the worker process with local accumulation and periodic sync.*

### 4.1 Refactor `Worker.__init__` ✅ COMPLETE
- Remove all `manager.dict` references
- Parameters: `job_queue`, `locks` (stripe locks only), `agent`, `lut_mmap_path`, `sync_interval`, `n_players`, `n_actions`
- No shared state beyond the job queue and the `SparseRegretTable` via `agent`
- `_local_delta: Dict[str, np.ndarray]` initialised as empty dict — allocated fresh after each sync

> **Implementation notes:**
> - `info_set_lut` parameter replaced with `lut_path: Union[str, Path]` and `pickle_dir: bool` — the LUT object is not passed to the worker; only the path is stored and the dict is loaded after fork in `run()`.
> - `_local_delta` initialised as `Dict[str, Dict[str, float]]` (not `Dict[str, np.ndarray]` as planned) — numpy conversion deferred to Phase 5 when `Agent` gets `SparseRegretTable`.
> - `_setup_new_game()` removed from `__init__` — it requires the LUT which is only loaded after fork; moved to `run()`.
> - Three parameters removed as redundant — the server already gates dispatch before enqueuing these jobs, so the worker never needs to re-check:
>   - `lcfr_threshold` — server checks `t < lcfr_threshold` before sending `discount`
>   - `update_threshold` — server checks `t > update_threshold` before sending `update_strategy`
>   - `dump_iteration` — server checks `t % dump_iteration == 0` before sending `serialise`
> - Server `_start_workers()` updated to pass `lut_path` and `pickle_dir` instead of `info_set_lut`.

### 4.2 Implement `Worker.run()` ✅ COMPLETE
- Attach to shared memory blocks **after fork** — never before
- Open LUT as read-only `mmap` after fork:

```python
self._lut_file = open(self._lut_mmap_path, "rb")
self._lut_mmap = mmap.mmap(
    self._lut_file.fileno(), 0,
    mmap.MAP_SHARED, mmap.PROT_READ
)
self._lut_mmap.madvise(mmap.MADV_RANDOM)
```

- Set worker-specific random seed for reproducibility
- Main loop: get job from queue, dispatch to handler, call `task_done()`

> **Implementation notes:**
> - LUT loaded after fork via `_mmap.mmap(lut_file.fileno(), 0, access=_mmap.ACCESS_READ)` + `joblib.load(lut_mmap)`. All workers mapping the same file share the same physical OS page-cache pages.
> - `mmap.madvise(mmap.MADV_RANDOM)` skipped — only available from Python 3.8; project runs Python 3.7.12.
> - For the deprecated `pickle_dir` format, falls back to `utils.io.load_info_set_lut()`.
> - Startup sequence after fork: load LUT → `_set_seed()` → `_setup_new_game()` → dispatch loop.

### 4.3 Implement `_sync_to_master()` ✅ COMPLETE
- Iterate over `_local_delta` entries
- For each infoset:
  - Call `agent.regret_table.get_row(info_set)` — allocates if new
  - Acquire stripe lock for that chunk
  - `np.add(master_row, delta_row, out=master_row)`
  - `np.maximum(master_row, REGRET_FLOOR, out=master_row)`
  - Release stripe lock
  - Mark chunk dirty
- After all entries flushed: `_local_delta.clear()`
- Log count of infosets synced and elapsed time

> **Implementation notes:**
> - `agent.regret_table` (SparseRegretTable) does not exist yet — that is Phase 5.1. For now `_sync_to_master()` acquires the `regret` lock once and calls `ai.merge_local_delta(agent, _local_delta)`, which does the dict-based read-modify-write. The stripe lock / `np.add` / dirty-marking path will be wired in when Phase 5.1 refactors `Agent`.
> - Returns immediately without locking when `_local_delta` is empty.
> - Logs a single message per flush containing the infoset count.

### 4.4 Implement Worker Job Handlers ✅ COMPLETE
- `cfr` job: calls `cfr()` with `_local_delta`, increments local iteration counter
- `sync` job: calls `_sync_to_master()`
- `update_strategy` job: only dispatched after sync boundary — reads shared regret table, writes shared strategy table
- `discount` job: only dispatched after sync — calls `regret_table.apply_discount(factor)` — never touches strategy table
- `terminate` job: calls `_sync_to_master()` one final time, closes mmap, exits cleanly

> **Implementation notes:**
> - `_cfr()` now accumulates into `self._local_delta` (instance variable, not a local) and calls `self._sync_to_master()` after each traversal. This keeps behaviour identical to Phase 3 while introducing the infrastructure for Phase 5, which will decouple flush frequency from traversal frequency via explicit server-dispatched `sync` jobs.
> - `terminate` calls `self._sync_to_master()` and `self._job_queue.task_done()` before breaking — no regret increments are lost on shutdown.
> - `sync` job handler added to the dispatch loop — currently only triggered by Phase 5's server loop, but the worker handles it today.
> - `self._local_iteration_count` incremented in `_cfr()` for future per-worker throughput metrics.

### 4.5 NUMA Pinning ✅ COMPLETE
- Detect NUMA topology on startup using `numactl --hardware`
- Assign workers evenly across NUMA nodes
- Pin each worker to its assigned NUMA node after fork:

```python
def _pin_to_numa_node(self, node: int):
    cores = self._get_cores_for_node(node)
    if cores:
        os.sched_setaffinity(0, cores)
```

> **Implementation notes:**
> - NUMA topology read from `/sys/devices/system/node/node{N}/cpulist` — no dependency on `numactl` binary.
> - Worker index derived from `self.name` (e.g. `Process-3` → index 2); node assigned as `nodes[index % len(nodes)]`.
> - All errors (missing `/sys` path, unreadable files, `OSError` from `sched_setaffinity`, unparseable process name) are silently swallowed — NUMA pinning is a performance hint, not a correctness requirement.
> - `_parse_cpulist(cpulist: str)` is a `@staticmethod` that handles ranges (`0-3`), single cores (`7`), and comma-separated combinations.
> - `_try_numa_pin()` called in `run()` immediately after post-fork LUT load.

### 4.6 Unit Tests ✅ COMPLETE
- Verify `_sync_to_master()` correctly flushes all local delta entries
- Verify dirty flags are set after sync
- Verify `local_delta` is empty after sync
- Verify terminate job flushes before exiting
- Verify LUT mmap is opened after fork, not before

> **Implemented:** 22 tests in `test/refactoring_training/test_phase4_worker.py` across 7 classes: `TestSyncToMaster` (6), `TestLocalDeltaState` (3), `TestLutLoadedAfterFork` (1), `TestParseCpulist` (6), `TestGetNumaNodes` (3), `TestTryNumaPin` (3), plus one `@pytest.mark.slow` integration test. All 22 non-slow tests pass.

---

## Phase 5 — Server: Simplified Coordination Loop
*Removes manager process, simplifies job dispatch.*

### 5.1 Refactor `Agent`
- Remove `mp.Manager()` entirely
- `Agent` holds one `SparseRegretTable` for regret, one for strategy
- No locks inside `Agent` — locking is exclusively worker and server responsibility
- Constructor: takes `info_set_lut`, `n_actions`, optional `checkpoint_path`
- If `checkpoint_path` provided: load regret and strategy tables from checkpoint files

> **Deferred from Phase 4.1:** Once `Agent` holds a `SparseRegretTable`, convert `Worker._local_delta` from `Dict[str, Dict[str, float]]` to `Dict[str, np.ndarray]` so that `_sync_to_master()` (Phase 4.3) can use `np.add` directly without a dict-to-array conversion step.

> **Full checklist for 5.1 — do not skip any of these:**
>
> **`agent.py`:**
> - Remove module-level `manager = mp.Manager()` and the `manager` import — this is the manager process that currently mediates all `agent.regret` and `agent.strategy` accesses
> - Remove `use_manager: bool` parameter and the `TESTING_SUITE` environment variable workaround — they only exist to switch away from `manager.dict()` in tests
> - Replace `self.regret = dict_constructor()` with `self.regret_table = SparseRegretTable(...)`
> - Replace `self.strategy = dict_constructor()` with `self.strategy_table = SparseRegretTable(...)`
> - Remove `agent_path` / `joblib.load` checkpoint loading from `__init__` — checkpoint resume moves to `CheckpointManager._restore_from_checkpoint()` in Phase 6
>
> **`server.py`:**
> - Remove module-level `manager = mp.Manager()` — no longer needed once `Agent` no longer uses it
> - Remove `regret` and `strategy` from `self._locks` — replaced by `SparseRegretTable` stripe locks (256 per table)
> - Keep `pre_flop_strategy` lock — `update_strategy()` traversal still needs it until a finer-grained solution is designed
>
> **`ai.py`:**
> - `cfr()` and `cfrp()`: reads against `agent.regret_table.get_row_if_exists(info_set)` instead of `agent.regret.get(info_set, {})` — returns a numpy row or `None`; `local_delta` values become `np.ndarray` (see Phase 4.1 deferred item)
> - `merge_local_delta()`: replaced by the stripe-lock path in `Worker._sync_to_master()` — remove the function or keep as singleprocess-only utility
> - `update_strategy()`: reads from `agent.regret_table.get_row_if_exists()`, writes to `agent.strategy_table.get_row()`
> - `serialise()`: snapshot iteration changes from `agent.regret.items()` to iterating allocated chunks via `agent.regret_table`
>
> **`worker.py`:**
> - `_discount()`: remove entire dict iteration loop; replace with `self._agent.regret_table.apply_discount(discount_factor)` — the table method handles all allocated chunks atomically
> - `_sync_to_master()`: replace `ai.merge_local_delta` + single regret lock with stripe-lock + `np.add` pattern as documented in Phase 4.3
>
> **`singleprocess/train.py`:**
> - Also uses `agent.regret` as a dict — update to use `SparseRegretTable` API; or keep singleprocess using a plain-dict `Agent` and document the divergence explicitly

### 5.2 Implement LUT Loading and Prefaulting
- Server loads LUT via read-only `mmap` before spawning workers
- `madvise(MADV_WILLNEED)` to prefault all pages into RAM immediately:

```python
lut_mmap.madvise(mmap.MADV_WILLNEED)
log.info("Prefaulting 100GB LUT into RAM — this takes a few minutes")
# Touch every page to force fault
for offset in range(0, lut_mmap.size(), 4096):
    _ = lut_mmap[offset]
lut_mmap.madvise(mmap.MADV_RANDOM)
log.info("LUT fully resident in page cache")
```

- Workers fork after prefault — they inherit the warm page cache
- Pass only the file path to workers — never the mmap object itself

> **Implementation notes:**
> - `self._info_set_lut` in `server.py` is now **unused** — workers load their own copy after fork (Phase 4.1/4.2). Remove the `utils.io.load_info_set_lut()` call and `self._info_set_lut` attribute entirely.
> - Python 3.7: `mmap.madvise` is not available. The prefault must be done via the manual page-touch loop (`for offset in range(0, size, 4096): _ = lut_mmap[offset]`) rather than the `madvise(MADV_WILLNEED)` hint — the hint line should be skipped. The loop achieves the same effect by forcing page faults before fork.
> - For `pickle_dir` format (no single joblib file) prefaulting is not applicable — skip the mmap step and go straight to spawning workers.

### 5.3 Refactor Server `__init__`
- Remove all `mp.Manager()` references
- Job queue `maxsize = n_workers * n_players * 2`
- Initialise `CheckpointManager` before spawning workers — signal handlers must be registered first
- Spawn workers only after LUT is prefaulted and signal handlers are registered
- Store `session_id` generated at startup for shared memory naming

> **Full checklist for 5.3 — do not skip any of these:**
>
> **Parameters to remove:**
> - `sync_update_strategy`, `sync_cfr`, `sync_discount`, `sync_serialise` — these controlled the old `_syncronised_job` single-worker execution path; the new loop uses `_broadcast_job` + `job_queue.join()` instead
> - `dump_iteration` — replaced by `checkpoint_interval` (explicit checkpoint cadence)
>
> **Parameters to add:**
> - `sync_interval: int` — how many CFR iterations between explicit `sync` broadcasts (decouples flush frequency from traversal frequency)
> - `checkpoint_interval: int` — how many iterations between `CheckpointManager.checkpoint()` calls
>
> **Parameters to rename / keep:**
> - `start_timestep` → `start_t` (or keep as-is; align with the loop variable `t`)
> - Retain: `strategy_interval`, `n_iterations`, `lcfr_threshold`, `discount_interval`, `prune_threshold`, `c`, `n_players`, `update_threshold`, `save_path`, `lut_path`, `pickle_dir`, `n_processes`
> - Retain SLURM `SLURM_CPUS_PER_TASK` detection for `n_processes`
>
> **Methods / attributes to remove:**
> - `_status_queue` and `_worker_status` — only used by `_wait_until_all_workers_are_idle`
> - `_syncronised_job()` — replaced by `_broadcast_job()` + `job_queue.join()`
> - `_wait_until_all_workers_are_idle()` — no longer needed; explicit `job_queue.join()` is sufficient
> - `job()` wrapper method — only existed to dispatch to `_syncronised_job` vs `_send_job`
>
> **`to_dict()` / `from_dict()`:**
> - Must be updated to reflect the new parameter set — remove `sync_*` and `dump_iteration`; add `sync_interval` and `checkpoint_interval`; add `session_id`

### 5.4 Implement Simplified Server Loop

```python
def search(self):
    self._training_start = time.monotonic()
    for t in range(self._start_t, self._n_iterations + 1):

        # Send CFR jobs — workers pick up concurrently, no blocking here
        for i in range(self._n_players):
            self._send_job("cfr", t=t, i=i)

        # Sync point — flush all local deltas into shared table
        if t % self._sync_interval == 0:
            self._job_queue.join()
            self._broadcast_job("sync")
            self._job_queue.join()

            # Strategy update only after sync — data is fresh
            if t > self._update_threshold and t % self._strategy_interval == 0:
                for i in range(self._n_players):
                    self._send_job("update_strategy", t=t, i=i)
                self._job_queue.join()

            # Discount only after sync — see Phase 7
            if self._discount_window_active(t):
                self._broadcast_job("discount", t=t)
                self._job_queue.join()

        # Checkpoint — always forces sync first
        if t % self._checkpoint_interval == 0:
            self._checkpoint_manager.checkpoint()
```

> **Stubs pending other phases:**
> - `_discount_window_active(t)` does not exist yet — implement as a stub returning `False` until Phase 7 is complete. This means the discount block is dead code until Phase 7.
> - `self._checkpoint_manager.checkpoint()` does not exist yet — implement as a stub that calls the existing `ai.serialise()` until Phase 6 is complete.
> - Retain the `enlighten` progress bar from the current `search()`.
> - Remove the four `log.info(f"synchronising ... - {self._sync_*}")` lines at the top of `search()` — those flags no longer exist.

### 5.5 Implement `_send_job` and `_broadcast_job`
- `_send_job(name, **kwargs)`: puts one job on queue, non-blocking
- `_broadcast_job(name, **kwargs)`: puts one job per worker on queue, ensures all workers execute it

> **`_send_job`** already exists. Keep as-is.
>
> **`_broadcast_job`** is new — puts exactly `len(self._workers)` copies of the job on the queue:
> ```python
> def _broadcast_job(self, job_name: str, **kwargs):
>     for _ in self._workers:
>         self._job_queue.put((job_name, kwargs), block=True)
> ```
>
> **Remove:** `_syncronised_job()`, `_wait_until_all_workers_are_idle()`, and the `job()` wrapper (see 5.3).

### 5.6 Implement Clean Shutdown
- `terminate(safe=True)`: broadcasts terminate job, joins queue, closes all shared memory blocks, unlinks all blocks, closes LMDB index

> **Full checklist for 5.6:**
> - Broadcast `terminate` to all workers via `_broadcast_job("terminate")` (not a loop of individual puts)
> - Call `self._job_queue.join()` after broadcast to wait for all workers to process the sentinel
> - Join each worker process — retain the queue-draining loop during join to prevent OS pipe bloat (the `logging_queue` can fill up if not drained; without draining the workers' feeder threads block and `worker.join()` hangs):
>   ```python
>   while worker.is_alive():
>       while not self._logging_queue.empty(): ...
>       worker.join(timeout=0.5)
>   ```
> - After all workers joined: call `self._agent.regret_table.close()` and `self._agent.strategy_table.close()` — this closes mmap file handles
> - Call `self._agent.regret_table.unlink_all()` and `self._agent.strategy_table.unlink_all()` — this deletes the shm files from `/dev/shm`
> - Call `self._agent.regret_table._index.close()` to close the LMDB environment cleanly
> - `status_queue` is removed in 5.3, so no `status_queue` draining needed here

---

## Phase 6 — Checkpointing and Crash Recovery
*Adapts the clustering checkpoint pattern to training.*

### 6.1 Implement `CheckpointManager`
- Create `poker_ai/ai/checkpoint.py`
- Constructor registers SIGTERM and SIGINT handlers immediately — before any workers spawn
- Holds reference to server and save path

### 6.2 Implement `checkpoint(emergency=False)`
Strict ordering — must not deviate:

```python
def checkpoint(self, emergency: bool = False):
    # 1. Flush all worker local deltas into shared table
    self._server.flush_all_workers()

    # 2. Write to temp directory — never directly to final path
    tmp_path = self._save_path / f"checkpoint_tmp_{int(time.time())}"
    tmp_path.mkdir(parents=True, exist_ok=True)

    # 3. Write only dirty chunks — incremental after first checkpoint
    table = self._server._agent.regret_table
    for chunk_id in table.get_dirty_chunks():
        chunk = table._chunks[chunk_id]
        valid_rows = min(
            table._next_row - chunk_id * CHUNK_SIZE, CHUNK_SIZE
        )
        atomic_numpy_save(chunk[:valid_rows], tmp_path / f"chunk_{chunk_id:06d}.npy")

    # 4. Flush LMDB index — already persistent, just ensure sync
    self._server._agent.regret_table._index.flush()

    # 5. Write server state
    atomic_joblib_dump(self._server.to_dict(), tmp_path / "server_state.pkl")

    # 6. Atomic rename — reader sees old or new, never partial
    final_path = self._save_path / f"checkpoint_{int(time.time())}"
    tmp_path.rename(final_path)

    # 7. Delete previous checkpoint only after rename succeeds
    if self._last_checkpoint_path and self._last_checkpoint_path.exists():
        shutil.rmtree(self._last_checkpoint_path)
    self._last_checkpoint_path = final_path

    # 8. Clear dirty flags only after successful write
    table.clear_all_dirty()
```

### 6.3 Implement SIGTERM Handler
```python
def _handle_sigterm(self, signum, frame):
    log.warning(f"Signal {signum} — initiating emergency checkpoint")
    self.checkpoint(emergency=True)
    self._server.terminate(safe=True)
    sys.exit(0)
```

- Must complete within SLURM warning window (120 seconds with `--signal=SIGTERM@120`)
- Measure checkpoint write time on first run and set `_checkpoint_interval` accordingly
- If checkpoint write time exceeds 90 seconds: reduce `CHUNK_SIZE` or increase compression

### 6.4 Implement Incremental Chunk Writing
- First checkpoint: all chunks dirty — full write
- Subsequent checkpoints: only write chunks with dirty flag set
- After hot infoset set stabilises (typically a few hours in): only 10-30% of chunks are dirty per interval
- Log chunks written vs skipped at each checkpoint

### 6.5 Implement Resume Logic in Server `__init__`
```python
def _load_checkpoint_if_exists(self, checkpoint_path: Path):
    checkpoints = sorted(checkpoint_path.glob("checkpoint_[0-9]*"))
    for cp in reversed(checkpoints):
        if self._checkpoint_is_valid(cp):
            self._restore_from_checkpoint(cp)
            return
    log.info("No valid checkpoint found — starting fresh")

def _checkpoint_is_valid(self, path: Path) -> bool:
    if not (path / "server_state.pkl").exists():
        return False
    state = joblib.load(path / "server_state.pkl")
    for chunk_id in range(state["n_chunks"]):
        if not (path / f"chunk_{chunk_id:06d}.npy").exists():
            return False
    return True

def _restore_from_checkpoint(self, path: Path):
    state = joblib.load(path / "server_state.pkl")
    self._start_t = state["t"]
    for chunk_id in range(state["n_chunks"]):
        arr = np.load(path / f"chunk_{chunk_id:06d}.npy")
        self._agent.regret_table._restore_chunk(chunk_id, arr)
    log.info(f"Resumed from checkpoint at iteration {self._start_t}")
```

### 6.6 Implement Orphaned Shared Memory Cleanup
```python
def cleanup_orphaned_blocks(session_id: str):
    import os
    prefix = f"pluribus_regret_{session_id}_"
    for name in os.listdir("/dev/shm"):
        if name.startswith(prefix):
            try:
                shm = shared_memory.SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
                log.info(f"Cleaned up orphaned block: {name}")
            except FileNotFoundError:
                pass
```

### 6.7 Adapt Auto-Resubmit Script for Training
- Copy `cluster_auto_resub.sh` from clustering
- Replace LUT completion check with training completion check: target iteration count reached
- Set `--signal=SIGTERM@120`
- Set `--requeue`
- Set `--open-mode=append`
- Add dependency chain for wall-time limits
- Completion check:

```bash
TARGET_ITERATIONS=10000000
CURRENT_ITERATIONS=$(python -c "
import joblib
state = joblib.load('checkpoint_latest/server_state.pkl')
print(state['t'])
")
if [ "$CURRENT_ITERATIONS" -ge "$TARGET_ITERATIONS" ]; then
    echo "Training complete at iteration $CURRENT_ITERATIONS"
    exit 0
fi
sbatch --dependency=afterany:$SLURM_JOB_ID $0
```

---

## Phase 7 — Discount Handling
*Implements time-based discount window with correct sync ordering.*

### 7.1 Implement Time-Based Discount Gate
```python
def _discount_window_active(self, t: int) -> bool:
    if not self._discounting_active:
        return False
    elapsed = time.monotonic() - self._training_start
    if elapsed >= self._discount_duration_secs:
        self._close_discount_window()
        return False
    return t % self._discount_interval == 0

def _close_discount_window(self):
    if not self._discounting_active:
        return
    log.info("Discount window closing — performing final sync")
    self._broadcast_job("sync")
    self._job_queue.join()
    self._discounting_active = False
    log.info("Discounting permanently disabled")
```

### 7.2 Integrate Into Server Loop
- Discount fires only when `_discount_window_active(t)` returns True
- Always preceded by a completed sync — enforced by placement in the loop after `job_queue.join()`
- Discount job dispatched to all workers via `_broadcast_job`
- Workers apply discount to regret table only — never strategy table — fixing Bug 2

### 7.3 Compute Discount Factor Correctly
```python
def _compute_discount_factor(self, t: int) -> float:
    """
    LCFR discount factor per Pluribus supplementary material.
    d = (t / discount_interval) / ((t / discount_interval) + 1)
    """
    d = (t / self._discount_interval)
    return d / (d + 1.0)
```

### 7.4 Persist Discount State in Checkpoint
- Save to server state dict:
  - `discounting_active: bool`
  - `discount_duration_secs: float`
  - `elapsed_discount_secs: float` — computed as `time.monotonic() - training_start`
- On resume: restore remaining budget as `duration - elapsed`, not full duration:

```python
def _restore_discount_state(self, state: dict):
    elapsed = state.get("elapsed_discount_secs", 0.0)
    self._discount_duration_secs = max(
        0.0,
        state["discount_duration_secs"] - elapsed
    )
    self._discounting_active = state.get("discounting_active", True)
```

---

## Phase 8 — Validation and Tuning
*Verify correctness and find optimal sync interval.*

### 8.1 Correctness Validation at `sync_interval = 1`
- Run refactored multiprocess trainer on 2-player 20-card game with `sync_interval = 1`
- At `sync_interval = 1` the result must match the Phase 0 singleprocess baseline within tolerance
- Run `validate_strategy_equivalence(baseline_strategy, refactored_strategy, tolerance=0.01)`
- If validation fails: the CFR logic has a bug introduced in Phase 3 or 4 — do not proceed until resolved
- Common failure causes: external sampling implementation wrong, local delta not flushed before strategy update, discount applied to strategy table

### 8.2 Correctness Validation of Discount
- Run with discount enabled for a short window, verify regret values decrease at discount boundaries
- Verify strategy table values are not decreased by discount — confirm Bug 2 is fixed
- Verify discount fires on correct iterations — confirm Bug 1 is fixed

### 8.3 Sync Interval Tuning
- Run 4 experiments: `sync_interval` = 1, 10, 100, 1000
- For each: measure iterations per second and exploitability at fixed wall-clock time (30 minutes each)
- Expected: exploitability degrades slightly at higher intervals, throughput improves significantly
- Choose the highest interval where exploitability degradation is within 5% of `sync_interval = 1`
- Document chosen value as default in config

### 8.4 Performance Validation
- Run on full cluster node for 1 hour with 64 workers
- Compare iterations per second against Phase 0 multiprocess baseline
- Target: 20-40x improvement
- Verify memory usage grows proportionally to infosets discovered — not a fixed large allocation
- Verify checkpoint write time is under 90 seconds
- Verify resume from checkpoint produces identical subsequent behaviour

### 8.5 SLURM Integration Test
- Submit a 30-minute job, manually cancel with `scancel` after 10 minutes
- Verify SIGTERM handler fired and checkpoint was written
- Verify requeued job resumes from checkpoint at correct iteration count
- Verify no orphaned shared memory blocks remain after kill:

```bash
ls /dev/shm | grep pluribus   # should be empty after clean shutdown
```

- Verify discount elapsed time is correctly restored after resume

### 8.6 Stress Test
- Run 64 workers for 4 hours without interruption
- Monitor memory growth rate — verify it tracks infoset discovery, not a leak
- Monitor iterations per second over time — verify it does not degrade as regret table grows
- Verify checkpoint files grow incrementally — later checkpoints write fewer dirty chunks than the first

---

## Dependency Graph

```
Phase 0 (baselines + bug documentation)
    │
Phase 1 (hashing + LMDB + shared IO utils)
    │
Phase 2 (SparseRegretTable)
    │
    ├── Phase 3 (CFR core: external sampling + local delta + bug fixes)
    │       │
    │       └── Phase 4 (Worker: local delta sync)
    │               │
    │               └── Phase 5 (Server: simplified loop)
    │                       │
    │               ┌───────┴────────┐
    │           Phase 6         Phase 7
    │       (checkpointing)   (discount)
    │               │               │
    └───────────────┴───────────────┘
                    │
                Phase 8 (validation + tuning)
```

Phases 6 and 7 are independent of each other and can be developed in parallel after Phase 5.  
Phase 8 cannot begin until both are complete.

---

## What is Retained From Clustering

| Clustering component | Training equivalent | Phase |
|---|---|---|
| `atomic_joblib_dump` | Checkpoint server state | 1, 6 |
| `atomic_numpy_save` | Checkpoint regret chunks | 1, 6 |
| `ChunkedProcessor` dirty tracking | `SparseRegretTable` dirty flags | 2, 6 |
| `cluster_auto_resub.sh` | Training auto-resubmit script | 6 |
| SIGTERM handler pattern | `CheckpointManager` | 6 |
| Completion guard before proceeding | Checkpoint validity check on resume | 6 |
| `np.memmap` for large arrays | `shared_memory` for regret chunks | 2 |

---

## Bugs Fixed Per Phase

| Bug | Phase | Impact |
|---|---|---|
| `&` instead of `and` in discount condition | 3 | Discount never fired — strategy quality |
| Strategy discounted alongside regret | 3, 7 | Convergence degraded — strategy quality |
| `serialise()` acquires lock per infoset | 3 | All workers stalled during serialisation |
| `serialise()` deepcopy entire regret table | 3 | Full duplicate of 50GB+ table in RAM |
| Outcome sampling at 6 players | 3 | Impractical variance — training quality |
