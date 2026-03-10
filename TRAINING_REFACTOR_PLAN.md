# Training Pipeline Refactor — Implementation Plan

**Principle:** every phase leaves the codebase runnable. Validate against the singleprocess baseline after each phase before proceeding.

---

## Phase 0 — Preparation and Baselines ✅ COMPLETE

**Bugs found in `ai.py`:**
- **Bug 1** — `&` instead of `and` in discount condition → discount never fires
- **Bug 2** — strategy table discounted alongside regret → only regret should be discounted
- **Bug 3** — `serialise()` acquires lock per infoset → blocks all workers
- **Bug 4** — `serialise()` uses `copy.deepcopy` on entire regret table → full duplicate in RAM
- **Bug 5** — `update_strategy()` exits for all post-flop states → pre-flop-only accumulation; guard removed in Phase 5.7 to enable full-tree strategy accumulation (see Phase 5.7).
- **Bug 6** — loop variable `i` in `cfr()` shadows traversing player parameter → fixed by renaming to `player_idx`
- **Bug 7** — `Union` used but not imported in `singleprocess/train.py` → `NameError` at runtime

Bugs 1 and 2 fixed early in Phase 2 in both `singleprocess/train.py` and `multiprocess/server.py`.

**Sampling audit:** `cfr()` confirmed as chance-sampled (outcome sampling on opponent branch). Must switch to external sampling in Phase 3 — outcome sampling variance is impractical at 6 players.

**Correctness baseline:** singleprocess run on 2-player 20-card game with pruning disabled and Bug 1 fixed is the ground truth for all subsequent validation.

---

## Phase 1 — Infrastructure: Shared Utilities and Persistent Index ✅ COMPLETE

**Files created:** `poker_ai/utils/io.py`, `poker_ai/ai/index.py`

**`utils/io.py`:** `atomic_joblib_dump`, `atomic_numpy_save`, `atomic_numpy_load` (accepts `expected_shape`/`expected_dtype`), `hash_info_set_128`, `hash_info_set_bytes`.

**`InfosetIndex` (LMDB):** 16-byte blake2b/xxhash key → packed `(chunk_id: uint32, row: uint32)`. `map_size=50GB`, `writemap=True`, `map_async=True`. Methods: `get`, `get_or_create`, `flush`, `close`. `CHUNK_SIZE = 100_000` defined here and re-exported from `regret_table.py`.

**Deviations:** `hash_info_set_bytes` added as a convenience returning the raw 16-byte key. `flush()` calls `env.sync(True)` with `TypeError` fallback for lmdb 0.9.x positional-only API.

**Tests:** 1M synthetic infosets inserted, closed, reopened, all retrieved correctly.

---

## Phase 2 — Core Data Structure: `SparseRegretTable` ✅ COMPLETE

**File created:** `poker_ai/ai/regret_table.py`

**Structure:** chunks backed by files in `shm_dir` (default `/dev/shm`) opened `O_CREAT|O_EXCL` and mmap'd `MAP_SHARED` — Python 3.7 has no `multiprocessing.shared_memory`. Each chunk `(CHUNK_SIZE, N_ACTIONS)` in `int32`. Row allocation delegated to `InfosetIndex.get_or_create()` via LMDB write transaction; no `_next_row` counter on the table.

**Key methods:** `get_row`, `get_row_if_exists`, `get_row_by_location`, `apply_discount(factor)`, `get_dirty_chunks`, `clear_dirty`, `clear_all_dirty`, `close`, `unlink_all`, `list_own_blocks`.

**Stripe locking:** 256 `mp.Lock` objects; `stripe_id = chunk_id % 256`. Acquired only during sync flush.

**Dirty tracking:** `_dirty = mp.Array('b', _MAX_DIRTY_CHUNKS, lock=False)` — must be `mp.Array` for cross-process visibility. `_MAX_DIRTY_CHUNKS = 2048` → covers up to 204.8M infosets.

**`apply_discount`:** vectorised `float32` multiply → cast to `int32` → `np.maximum(..., REGRET_FLOOR)`. `_at_sync_boundary` is `mp.Value('b', 0)`. `valid_rows` computed from `_index.n_entries`.

**Naming:** `pluribus_regret_{session_id}_{chunk_id:06d}` currently. Phase 5.0 extends to `pluribus_{type}_{street}_{chunk_id:06d}`; no `session_id` needed since restarts always occur on a new node with a clean `/dev/shm`.

**Phase 5.1 amendments required (add before implementing 5.1):**
- `__init__` accepts optional `index: InfosetIndex` — if provided, use it; if `None`, create new (preserves all Phase 2 tests)
- `n_allocated: int` property — per-table row counter incremented in `get_row()`, used by checkpoint writer (shared `_index.n_entries` spans all streets and cannot be used)
- `_restore_chunk(chunk_id, arr)` — writes `arr` into shared memory, extending `_chunks` if needed

**Tests:** 50 tests in `test/unit/test_phase2_regret_table.py`. All pass; 500K allocation test gated behind `@pytest.mark.slow`.

---

## Phase 3 — CFR Core: External Sampling and Local Delta ✅ COMPLETE

**Key changes to `ai.py`:**
- `calculate_strategy_from_row(regret_row, valid_mask=None)` — pure numpy regret matching with uniform fallback. `valid_mask` support added in Phase 5.1.
- `cfr(agent, state, i, t, local_delta=None)` — regret increments written to `local_delta` only; zero shared memory writes during traversal. `locks` parameter removed.
- External sampling: opponent branch samples **one action** via `np.random.choice(actions, p=probs)` — not iterating all (that caused exponential blowup). Traversing player still explores all actions.
- `merge_local_delta(agent, local_delta)` — dict-based merge helper; replaced by stripe-lock path in Phase 5.1.
- `update_strategy()` — `locks` parameter removed; `betting_round > 0` early-exit retained (Bug 5, intentional).
- `serialise()` — `deepcopy` removed (Bug 4); takes snapshot via `list(agent.regret.items())` under short lock.

**`local_delta` type:** currently `Dict[str, Dict[str, float]]`. Converted to `Dict[Tuple[int, str], np.ndarray]` (keyed by `(betting_round, info_set)`) in Phase 5.1 when `Agent` gets `SparseRegretTable`.

**Tests:** 47 tests in `test/unit/test_phase3_cfr_core.py`, all green.

---

## Phase 4 — Worker: Local Delta Sync Protocol ✅ COMPLETE

**Key changes to `worker.py`:**
- `info_set_lut` replaced with `lut_path` + `pickle_dir` — LUT loaded after fork via `mmap` + `joblib.load`.
- `mmap.MADV_RANDOM` skipped — Python 3.7 only.
- `_local_delta` is `Dict[str, Dict[str, float]]` — numpy conversion deferred to Phase 5.1.
- `_sync_to_master()` currently calls `ai.merge_local_delta` under single regret lock; Phase 5.1 replaces with stripe-lock + `np.add` path.
- `_cfr()` syncs after every traversal in Phase 4; Phase 5 decouples this via explicit `sync` jobs.
- NUMA: topology from `/sys/devices/system/node/node{N}/cpulist`; errors silently swallowed (performance hint only).
- Three parameters removed (`lcfr_threshold`, `update_threshold`, `dump_iteration`) — server gates these before dispatch.

**Tests:** 22 tests in `test/refactoring_training/test_phase4_worker.py`, all pass.

---

## Phase 5 — Server: Simplified Coordination Loop

### 5.0 Per-Street Action Indexing Design ✅ COMPLETE

Use four `SparseRegretTable` instances per type (regret + strategy), one per betting round (0–3). Row width = `max_actions_for_street` for that street. All eight tables share one `InfosetIndex` (one LMDB environment), so `(chunk_id, row)` from a regret lookup is valid for the corresponding strategy row.

```python
# Agent:
self.regret_tables:   Dict[int, SparseRegretTable]  # keyed by betting_round 0-3
self.strategy_tables: Dict[int, SparseRegretTable]  # keyed by betting_round 0-3
```

**Canonical action ordering:** the state exposes `state.get_canonical_actions(betting_round) -> List[Action]` — full abstract action set in stable order, including actions not valid at this node. Called once at startup to build `action_to_idx`; never called on the hot path.

**Validity mask:** `state.get_valid_mask() -> np.ndarray` (bool, length `max_actions_for_street`). Table always stores full-width rows. `calculate_strategy_from_row` zeros invalid slots before regret matching (uniform fallback over valid actions only). Validity lives in the state, not the table.

**`/dev/shm` naming:** `pluribus_regret_{street}_{chunk_id:06d}` and `pluribus_strategy_{street}_{chunk_id:06d}`.

**Checklist — complete before 5.1:**
- [x] Confirm `state.legal_actions` ⊆ canonical action set for the street
- [x] Add `get_canonical_actions(betting_round)` to state/abstraction layer
- [x] Add `get_valid_mask()` to state (street is known from state; no parameter needed)
- [x] Update `calculate_strategy_from_row` with optional `valid_mask` (backward-compatible)
- [x] Determine and record `max_actions_for_street` for each of the four streets
- [x] Apply Phase 2 amendments to `SparseRegretTable` (external `index`, `n_allocated`, `_restore_chunk`)

**Implemented changes:**
- `poker_ai/games/base/state.py`: added `get_canonical_actions(betting_round)` static method — returns `["fold", "call", "all_in"] + ["raise:<f>" for f in sorted(first_raise ∪ subsequent_raise)]` for the street. Added `get_valid_mask()` instance method — boolean array of length `max_actions_for_street`, True where `legal_actions` contains the canonical action.
- `poker_ai/ai/ai.py`: `calculate_strategy_from_row(regret_row, valid_mask=None)` — zeros invalid-action slots before regret matching; uniform fallback restricted to valid actions. Module-level constants `CANONICAL_ACTIONS`, `ACTION_TO_IDX`, `MAX_ACTIONS_PER_STREET` built once at import from `get_canonical_actions`.
- `poker_ai/ai/regret_table.py`: added optional `index: InfosetIndex` constructor parameter (shared index or auto-created); added `_n_allocated: mp.Value` with `n_allocated` property; added `merge_delta_row(info_set, delta)` (stripe-lock + np.add + regret floor); added `_restore_chunk(chunk_id, arr)` for checkpoint resume.
- `test/refactoring_training/test_phase2_regret_table.py`: updated to use `table_name` kwarg and new API.

**`max_actions_for_street` recorded:**
- Street 0 (pre_flop): 12  (fold, call, all_in + 9 raise fractions)
- Street 1 (flop): 9  (fold, call, all_in + 6+4 deduplicated raise fractions)
- Street 2 (turn): 5  (fold, call, all_in + raise:0.5, raise:1.0)
- Street 3 (river): 5  (same as turn)

---

### 5.1 Refactor `Agent` ✅ COMPLETE

**`agent.py`:**
- Remove `mp.Manager()`, `use_manager` parameter, `TESTING_SUITE` workaround
- `self.regret_tables = {r: SparseRegretTable(n_actions=max_actions_per_street[r], index=shared_index) for r in range(4)}`
- `self.strategy_tables = {r: SparseRegretTable(n_actions=max_actions_per_street[r], index=shared_index) for r in range(4)}`
- Remove `agent_path` / `joblib.load` checkpoint loading → Phase 6

**`server.py`:**
- Remove `mp.Manager()`
- Remove `regret` and `strategy` from `self._locks`; keep `pre_flop_strategy` lock

**`ai.py`:**
- `cfr()`/`cfrp()`: read from `agent.regret_tables[state.betting_round].get_row_if_exists(info_set)`; write increments to `local_delta[(state.betting_round, info_set)]` as `np.ndarray` of length `max_actions_per_street[betting_round]` (zero-initialised on first visit)
- `update_strategy(agent, state, i)`: reads `regret_tables[r]`, writes `strategy_tables[r]`; one full game-tree traversal per dispatch per player
- `merge_local_delta()`: remove or keep as singleprocess utility
- `serialise()`: iterate `agent.regret_tables[r]` chunks instead of `agent.regret.items()`

**`worker.py`:**
- `_local_delta`: convert to `Dict[Tuple[int, str], np.ndarray]`
- `_sync_to_master()`: stripe-lock + `np.add` routing each delta to `agent.regret_tables[betting_round]`
- `_discount()`: `for r in range(4): self._agent.regret_tables[r].apply_discount(factor)` — never touch strategy tables

**`singleprocess/train.py`:** update to per-street `SparseRegretTable` API or keep plain-dict `Agent` and document divergence.

**Implemented changes:**
- `poker_ai/ai/agent.py`: complete rewrite — `Agent(index_path, shm_dir="/dev/shm")`. No `mp.Manager()`, no `use_manager`, no `TESTING_SUITE`. Creates one shared `InfosetIndex`, then 8 `SparseRegretTable` instances in `self.regret_tables: Dict[int, SparseRegretTable]` and `self.strategy_tables: Dict[int, SparseRegretTable]`, keyed by betting round 0–3.
- `poker_ai/ai/ai.py`: `cfr()` / `cfrp()` read from `agent.regret_tables[r].get_row_if_exists(info_set)` and accumulate increments into `local_delta: Dict[Tuple[int, str], np.ndarray]` (int64, length `MAX_ACTIONS_PER_STREET[r]`). `cfr()` / `cfrp()` auto-create+flush their own buffer when called with `local_delta=None` (singleprocess path). `_cfr_body` / `_cfrp_body` are internal helpers called recursively. `update_strategy()` reads `regret_tables[r]` and writes `strategy_tables[r]`; post-flop guard removed in Phase 5.7. `merge_local_delta(agent, local_delta)` routes each delta to the correct street table via `merge_delta_row`. `serialise()` stub logs a warning — full implementation deferred to Phase 6.
- `poker_ai/ai/multiprocess/server.py`: removed `mp.Manager()`, removed `regret`/`strategy` locks, kept `pre_flop_strategy` lock. Constructs `Agent(index_path=save_path/"lmdb_index")`. `_workers` is a plain list.
- `poker_ai/ai/multiprocess/worker.py`: `_local_delta` converted to `Dict[Tuple[int, str], np.ndarray]`; `_sync_to_master()` calls `ai.merge_local_delta()` with stripe locks; `_cfr()` no longer syncs after every traversal; `_discount()` loops over `agent.regret_tables[r].apply_discount()` — never touches strategy tables; `_serialise` and `_update_status` removed.
- `poker_ai/ai/singleprocess/train.py`: updated to `Agent(index_path=save_path/"lmdb_index")`; `local_delta: Dict[Tuple[int, str], np.ndarray] = {}`.
- Tests: `test/unit/test_phase3_cfr_core.py`, `test/refactoring_training/test_phase4_worker.py` updated to new API. All short tests pass.

---

### 5.2 LUT Loading and Prefaulting ✅ COMPLETE

Load LUT as read-only mmap in server before spawning workers. Prefault via page-touch loop (Python 3.7; `madvise` unavailable):

```python
for offset in range(0, lut_mmap.size(), 4096):
    _ = lut_mmap[offset]
```

Pass only the file path to workers; remove `self._info_set_lut` from server entirely. Skip prefault for `pickle_dir` format.

**Implemented changes:**
- `poker_ai/ai/multiprocess/server.py`: `_prefault_lut(lut_path, pickle_dir)` opens `card_info_lut.joblib` via `mmap` and touches every 4096-byte page before workers spawn. Skips prefault for `pickle_dir` format with a log message. `self._info_set_lut` removed from server entirely.
- `poker_ai/ai/multiprocess/worker.py`: `run()` loads the LUT from `self._lut_path` via `mmap` + `joblib.load` after fork (not passed from server). Workers receive only `lut_path` + `pickle_dir` strings.
- `poker_ai/ai/index.py`: added `reopen_after_fork()` — closes the inherited LMDB env handle and opens a fresh one for the child's PID (fixes `MDB_BAD_RSLOT` from forked reader lock-table slots). Called at the top of `Worker.run()` before any LMDB access.

---

### 5.3 Refactor Server `__init__`  ✅ COMPLETE

**Remove:** `mp.Manager()`, `sync_update_strategy`, `sync_cfr`, `sync_discount`, `sync_serialise`, `dump_iteration`, `_status_queue`, `_worker_status`, `_syncronised_job()`, `_wait_until_all_workers_are_idle()`, `job()` wrapper.

**Add:** `sync_interval: int`, `checkpoint_interval: int`.

**Keep:** `strategy_interval`, `n_iterations`, `lcfr_threshold`, `discount_interval`, `prune_threshold`, `c`, `n_players`, `update_threshold`, `save_path`, `lut_path`, `pickle_dir`, `n_processes`, SLURM detection.

**Startup:** call `_maybe_resume(save_path)` stub; spawn workers after LUT prefault. Signal handlers and real resume logic are Phase 6.

```python
def _maybe_resume(self, save_path: Path):
    if (save_path / "server_state.pkl").exists():
        log.info(f"Checkpoint found — resume wired in Phase 6")
    else:
        log.info("No checkpoint — starting fresh")
```

**`to_dict()`:** remove `sync_*` and `dump_iteration`; add `sync_interval`, `checkpoint_interval`.

**Implemented changes:**
- `poker_ai/ai/multiprocess/server.py`: Added `_maybe_resume(save_path)` stub — checks for `server_state.pkl`, logs intent. Called from `__init__` after agent+lock creation, before workers spawn. `to_dict()` already clean — has `sync_interval` and `checkpoint_interval`, no `sync_*` or `dump_iteration` keys.
- `poker_ai/ai/runner.py`: Added `--checkpoint_interval` CLI option (default 1000); passed to `Server(...)` constructor.

---

### 5.4 Simplified Server Loop  ✅ COMPLETE

```python
def search(self):
    self._training_start = time.monotonic()
    for t in range(self._start_t, self._n_iterations + 1):
        for i in range(self._n_players):
            self._send_job("cfr", t=t, i=i)

        if t % self._sync_interval == 0:
            self._job_queue.join()
            self._broadcast_job("sync")
            self._job_queue.join()

            if t > self._update_threshold and t % self._strategy_interval == 0:
                for i in range(self._n_players):
                    self._send_job("update_strategy", t=t, i=i)
                self._job_queue.join()

            if self._discount_window_active(t):   # stub: returns False until Phase 7
                self._broadcast_job("discount", t=t)
                self._job_queue.join()

        if t % self._checkpoint_interval == 0:
            self._checkpoint_manager.checkpoint()  # stub: log only — do NOT call serialise()
```

Retain `enlighten` progress bar. Remove the four `log.info("synchronising ...")` calls.

---

### 5.5 `_broadcast_job`  ✅ COMPLETE

```python
def _broadcast_job(self, job_name: str, **kwargs):
    for _ in self._workers:
        self._job_queue.put((job_name, kwargs), block=True)
```

`_send_job` already exists — keep as-is.

---

### 5.6 Clean Shutdown (happy path)  ✅ COMPLETE

```python
def terminate(self, safe=True):
    self._broadcast_job("terminate")
    self._job_queue.join()
    for worker in self._workers:
        worker.join()          # plain join — crash handling added in Phase 6.6
    for r in range(4):
        self._agent.regret_tables[r].close()
        self._agent.strategy_tables[r].close()
        self._agent.regret_tables[r].unlink_all()
        self._agent.strategy_tables[r].unlink_all()
    self._agent._index.close()
```

Logging queue drain and timeout added in Phase 6.6.

---

### 5.7 Sub-Game Solving Readiness  ✅ COMPLETE

**Single change required:** remove `betting_round > 0` from the guard in `update_strategy()`.

**Cascading consequences when this fires:**
- Strategy tables grow ~100–1000× — verify `_MAX_DIRTY_CHUNKS = 2048` is still sufficient
- `update_strategy` traversal covers full game tree — retune `strategy_interval` (Phase 8.3 values are pre-flop-calibrated)
- Rename `pre_flop_strategy` lock → `strategy_update_lock`
- Re-measure checkpoint write time; adjust `checkpoint_interval` for SIGTERM window
- Update Bug 5 annotation in Phase 0

**Implemented changes:**
- `poker_ai/ai/ai.py`: Removed `or state.betting_round > 0` from `update_strategy()` guard. Now traverses all four streets. Docstring updated.
- `poker_ai/ai/multiprocess/server.py`: Renamed `pre_flop_strategy` → `strategy_update_lock` in `self._locks`.
- `poker_ai/ai/multiprocess/worker.py`: Updated lock key to `strategy_update_lock` in `_update_strategy()`; docstring updated.
- `test/refactoring_training/test_phase4_worker.py`: Updated both `_make_worker` and integration test fixture to use `strategy_update_lock`.
- Phase 0 Bug 5 annotation updated.

---

## Phase 6 — Checkpointing and Resume

Restarts are always manual on a new node — no auto-requeue, no orphan cleanup needed.

### 6.1 `CheckpointManager`

Create `poker_ai/ai/checkpoint.py`. Constructor:
- Registers SIGTERM/SIGINT handlers **before workers spawn** — init `CheckpointManager` first in `Server.__init__`
- Calls `_load_checkpoint_if_exists(save_path)` — replaces Phase 5.3 stub

### 6.2 `checkpoint(emergency=False)`

```python
def checkpoint(self, emergency: bool = False):
    # 1. Flush workers. flush_all_workers() = _broadcast_job("sync") + job_queue.join().
    #    SIGTERM deadlock prevention: signal handler sets a threading.Event; the server
    #    loop checks it after each join() and breaks to call checkpoint() from the main
    #    thread — never from inside the signal handler.
    self._server.flush_all_workers()

    # 2. Write to temp dir
    tmp_path = self._save_path / f"checkpoint_tmp_{int(time.time())}"
    tmp_path.mkdir(parents=True, exist_ok=True)

    # 3. Write dirty chunks. Use table.n_allocated (per-table), NOT _index.n_entries
    #    (shared across all streets — would give wrong valid_rows).
    for r in range(4):
        for table, prefix in [
            (self._server._agent.regret_tables[r],   f"regret_{r}"),
            (self._server._agent.strategy_tables[r], f"strategy_{r}"),
        ]:
            for chunk_id in table.get_dirty_chunks():
                valid_rows = min(table.n_allocated - chunk_id * CHUNK_SIZE, CHUNK_SIZE)
                atomic_numpy_save(
                    table._chunks[chunk_id][:valid_rows],
                    tmp_path / f"{prefix}_chunk_{chunk_id:06d}.npy"
                )

    # 4. Flush LMDB
    self._server._agent._index.flush()

    # 5. Write server state (includes n_chunks_per_street for all 8 tables)
    atomic_joblib_dump(self._server.to_dict(), tmp_path / "server_state.pkl")

    # 6. Atomic rename
    final_path = self._save_path / f"checkpoint_{int(time.time())}"
    tmp_path.rename(final_path)

    # 7. Delete previous checkpoint
    if self._last_checkpoint_path and self._last_checkpoint_path.exists():
        shutil.rmtree(self._last_checkpoint_path)
    self._last_checkpoint_path = final_path

    # 8. Clear dirty flags
    for r in range(4):
        self._server._agent.regret_tables[r].clear_all_dirty()
        self._server._agent.strategy_tables[r].clear_all_dirty()
```

*Strategy tables for streets 1–3 are empty until Phase 5.7 fires — their dirty sets will be empty and no files are written.*

### 6.3 SIGTERM Handler

```python
def _handle_sigterm(self, signum, frame):
    self._sigterm_event.set()   # server loop breaks and calls checkpoint() from main thread
```

Must complete within SLURM `--signal=SIGTERM@120` window. If checkpoint write exceeds 90 s, reduce `CHUNK_SIZE` or increase `checkpoint_interval`.

### 6.4 Incremental Chunk Writing

First checkpoint: all chunks dirty → full write. Subsequent: only dirty chunks. After hot infoset set stabilises, expect 10–30% of chunks dirty per interval. Log chunks written vs skipped.

### 6.5 Resume Logic

```python
def _load_checkpoint_if_exists(self, save_path: Path):
    for cp in reversed(sorted(save_path.glob("checkpoint_[0-9]*"))):
        if self._checkpoint_is_valid(cp):
            self._restore_from_checkpoint(cp)
            return
    log.info("No valid checkpoint — starting fresh")

def _checkpoint_is_valid(self, path: Path) -> bool:
    if not (path / "server_state.pkl").exists():
        return False
    if not (path / "lmdb_index").exists():   # missing index → restore would fail mid-way
        return False
    state = joblib.load(path / "server_state.pkl")
    for r in range(4):
        for chunk_id in range(state["n_chunks_per_street"][r]):
            if not (path / f"regret_{r}_chunk_{chunk_id:06d}.npy").exists():
                return False
    return True

def _restore_from_checkpoint(self, path: Path):
    state = joblib.load(path / "server_state.pkl")
    self._server._start_t = state["t"]
    for r in range(4):
        for chunk_id in range(state["n_chunks_per_street"][r]):
            arr = np.load(path / f"regret_{r}_chunk_{chunk_id:06d}.npy")
            self._server._agent.regret_tables[r]._restore_chunk(chunk_id, arr)
        for chunk_id in range(state.get("n_strategy_chunks_per_street", {}).get(r, 0)):
            arr = np.load(path / f"strategy_{r}_chunk_{chunk_id:06d}.npy")
            self._server._agent.strategy_tables[r]._restore_chunk(chunk_id, arr)
    log.info(f"Resumed from checkpoint at t={self._server._start_t}")
```

### 6.6 Worker Join with Logging Drain

Replace `terminate()`'s plain `worker.join()` with:

```python
SHUTDOWN_TIMEOUT_SECS = 60

for worker in self._workers:
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_SECS
    while worker.is_alive():
        while not self._logging_queue.empty():
            try:
                logging.getLogger(
                    self._logging_queue.get_nowait().name
                ).handle(self._logging_queue.get_nowait())
            except Exception:
                pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            log.error(f"{worker.name} stuck — killing")
            worker.kill()
            break
        worker.join(timeout=min(0.5, remaining))
    if worker.exitcode not in (0, None):
        log.warning(f"{worker.name} exited {worker.exitcode}")
```

Worker crashes are unrecoverable — log and let `terminate()` continue so the checkpoint write completes.

---

## Phase 7 — Discount Handling

### 7.1 Time-Based Discount Gate

```python
def _discount_window_active(self, t: int) -> bool:
    if not self._discounting_active:
        return False
    elapsed = time.monotonic() - self._training_start
    if elapsed >= self._discount_duration_secs:
        log.info("Discount window closing")
        self._broadcast_job("sync")
        self._job_queue.join()
        self._discounting_active = False
        return False
    return t % self._discount_interval == 0
```

Workers apply discount **to regret tables only** — never strategy tables (fixes Bug 2).

### 7.2 Discount Factor

Per Pluribus supplementary: `d = (t / discount_interval) / ((t / discount_interval) + 1)`

### 7.3 Persist Discount State

Save `discounting_active`, `discount_duration_secs`, `elapsed_discount_secs` to `server_state.pkl`. On resume: `remaining = duration - elapsed` (not full duration).

---

## Phase 8 — Validation and Tuning

### 8.1 Correctness at `sync_interval = 1`
Multiprocess result must match Phase 0 singleprocess baseline within `tolerance=0.01` L1 per action per infoset. Failure causes: wrong external sampling, local delta not flushed before strategy update, discount on strategy table.

### 8.2 Discount Correctness
Regret values decrease at discount boundaries; strategy table unchanged; discount fires on correct iterations.

### 8.3 Sync Interval Tuning
Run `sync_interval` ∈ {1, 10, 100, 1000}. Measure iter/s and exploitability at 30 min wall-clock. Choose highest interval with exploitability degradation ≤5% vs baseline. Re-run after Phase 5.7 (full-game strategy accumulation makes `update_strategy` substantially more expensive).

### 8.4 Performance Validation
64 workers, 1 hour. Target: 20–40× over Phase 0 multiprocess baseline. Checkpoint write time < 90 s. Memory grows proportional to infosets, not a fixed allocation. Re-run after Phase 5.7.

### 8.5 SLURM Integration Test
Submit 30-min job, `scancel` at 10 min. Verify SIGTERM handler fired, checkpoint written, resumed job starts at correct `t`, discount elapsed time correctly restored.

### 8.6 Stress Test
64 workers, 4 hours. Monitor memory growth rate (tracks infoset discovery, not a leak) and iter/s stability. Verify later checkpoints write fewer dirty chunks than the first.

---

## Dependency Graph

```
Phase 0 (baselines + bugs)
    │
Phase 1 (hashing + LMDB + shared IO)
    │
Phase 2 (SparseRegretTable)
    │
Phase 3 (CFR core: external sampling + local delta)
    │
Phase 4 (Worker: local delta sync)
    │
Phase 5 (Server: simplified loop + per-street tables)
    │
    ├── Phase 6 (checkpointing)
    └── Phase 7 (discount)
            │
        Phase 8 (validation + tuning)
```

Phases 6 and 7 are independent and can be developed in parallel after Phase 5.

---

## Bugs Fixed Per Phase

| Bug | Fixed in | Impact |
|---|---|---|
| `&` instead of `and` in discount condition (Bug 1) | Phase 2 (early) | Discount never fired |
| Strategy discounted alongside regret (Bug 2) | Phase 2 (early) | Convergence degraded |
| `serialise()` acquires lock per infoset (Bug 3) | Phase 3 | Workers stalled during serialisation |
| `serialise()` deepcopy entire regret table (Bug 4) | Phase 3 | Full table duplicate in RAM |
| Loop var `i` shadows traversing player param (Bug 6) | Phase 3 | Player attribution corrupted |
| `Union` not imported in `singleprocess/train.py` (Bug 7) | Phase 3 | `NameError` at runtime |
| Outcome sampling at 6 players | Phase 3 | Impractical variance |
| Strategy discounted in `worker.py` `_discount()` | Phase 3 | Same as Bug 2, different file |
