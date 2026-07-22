# Blueprint metrics

`python -m evaluation.blueprint_metrics <blueprint_path>` is a **sanity check**
that a trained blueprint actually learned a meaningful strategy — *is it skewed
and sharpening, or still uniform noise?* — printed as a compact human-readable
block and stored next to the checkpoint as `blueprint_metrics.md` (full
per-action tables) and `blueprint_metrics.json` (programmatic cross-run
comparison).

Blueprints are large (200 GB+), so the default reads only a few evenly-spaced
chunks per street (`--sample-chunks`, default 4) — enough to distinguish a
trained strategy from an untrained one while touching a few GB instead of the
whole table. Coverage totals stay exact under sampling (see §2); play
frequencies are labelled as estimates. Regret health is a training diagnostic,
not a "did it learn" signal, so it is off by default.

```
python -m evaluation.blueprint_metrics /path/to/blueprint            # sampled sanity check
python -m evaluation.blueprint_metrics /path/to/blueprint --full     # exact, reads whole table
python -m evaluation.blueprint_metrics /path/to/blueprint \
    --checkpoint checkpoint_000000004096 --out results/ --regret --sample-chunks 8
```

## 1. What it computes (per street)

| Metric | Meaning |
|---|---|
| **Play frequencies** (visit-weighted) | The strategy tables store the average strategy as visit counts accumulated by the strategy-sampling traversal (`poker_ai/blueprint/strategy.py`), so per-action column mass ÷ total mass is the frequency the blueprint plays each abstract action when it acts on that street *under its own play*. This is the headline fold / call / raise-size / all-in profile, also aggregated into fold / call / raise / all-in buckets. |
| **Per-infoset mean strategy** (unweighted) | Each visited row normalised to a distribution and averaged with equal weight. Divergence from the visit-weighted numbers means the strategy differs sharply between hot and cold parts of the tree. |
| **Determinism** | Per-row normalised entropy (H / ln n_actions) and max-action-probability distributions: how mixed vs. near-pure the strategy is (fraction of rows with max prob > 0.5 / 0.9 / 0.99). |
| **Coverage / training mass** | Allocated infosets, visited fraction (rows with any mass), and the log10 distribution of per-row visit mass — how well trained the street is. |
| **Regret health** (opt-in with `--regret`) | Fraction of regret entries positive, below the Pluribus prune threshold (−300 M), and at the `REGRET_FLOOR` clamp (−310 M). |

## 2. Design

- **Sampled by default; exact totals regardless.** `--sample-chunks N` reads
  `N` evenly-spaced chunks per street. Because the selection always keeps the
  endpoints, and only the *final* chunk is ever partial (`ChunkStore._save`),
  the exact total infoset count is `(n_chunks − 1) × full_chunk_rows +
  last_chunk_rows`, recovered from two `.npy` headers alone (no data read). So
  the coverage/`n_infosets_total` numbers are exact even when a handful of
  chunks are scanned; only the *frequencies* are estimates over the sample.
  `--full` (or `--sample-chunks 0`) reads every chunk. Sampling assumes chunk
  membership is roughly uncorrelated with strategy — true enough for a sanity
  check since infosets are laid out in allocation order; use `--full` when you
  need publication-grade frequencies.
- **Reads chunk files directly — no LMDB, no LUT, no /dev/shm.** Saved
  checkpoint chunks are trimmed to their valid row prefix
  (`ChunkStore._save`), so `strategy_{r}_chunk_*.npy` / `regret_{r}_chunk_*.npy`
  are self-describing: row counts come from file shapes. The analysis is a
  single streaming pass over `np.load(mmap_mode="r")` views in
  `--batch-rows`-sized batches, so even a `--full` pass over a 200 GB+ blueprint
  runs with O(batch) memory on any machine that mounts the checkpoint
  directory — no staging, none of the RAM footprint the eval runner needs.
- **Column semantics.** The canonical row layout is
  `["fold", "call", "all_in", "raise:<f>", …]`
  (`PokerEnv.get_canonical_actions`). When the chunk width matches the live
  `environment.action_space.CANONICAL_ACTIONS` the real raise fractions label
  the columns; on mismatch (checkpoint trained under a different action
  config) the fold/call/all_in prefix is still positional truth and the raise
  columns get generic labels — bucket aggregates stay correct either way.
- **Percentiles from histograms.** Entropy / max-prob / mass percentiles are
  interpolated from fixed-bin histograms accumulated during the pass, so no
  per-row values are ever materialised (accurate to a bin width — plenty for a
  report).
- Checkpoint selection follows warm-start (`checkpoint_*` with the highest
  suffix); a bare checkpoint directory (containing `server_state.pkl`) is also
  accepted. Missing chunk files produce a `warnings` entry in the report, not
  a crash.

## 3. Reading the numbers

- **The core sanity signal.** An untrained (or badly collapsed) blueprint reads
  as near-uniform: normalised entropy ≈ 1.0 on every street and play
  frequencies close to `1 / n_actions` per action. A healthy blueprint is
  skewed (fold/call dominate, raises spread across sizes) and sharpens
  street-by-street (entropy falls, near-pure fractions rise toward the river).
  If preflop already looks near-pure, suspect a collapsed / under-discounted
  run.
- Visit-weighted play frequencies are conditioned on *the blueprint's own
  play reaching the node* — they are self-play action rates, not
  opponent-facing response rates. Preflop fold % here is close to (1 − VPIP)
  in spirit but aggregated over all preflop decision points, including
  re-raise spots.
- A large gap between visit-weighted and per-infoset numbers on late streets
  usually means the strategy in rarely-reached subtrees is still near its
  (untrained) uniform prior — check the visited fraction and mass p10 before
  reading anything into late-street frequencies.
- Rising near-pure fractions street-by-street are expected (fewer future
  decisions to balance); a near-pure *preflop* is a red flag for a collapsed /
  under-discounted run.

## 4. Limitations / future work

- The LMDB index stores only 128-bit infoset hashes, so metrics are
  aggregates over each street's whole table — per-context breakdowns (by
  cluster, by facing-a-raise, by position) would require enumerating infoset
  strings via a tree walk with the LUT, or a debug-mode blueprint with shadow
  string keys. Deliberately out of scope for this streaming pass.
- No exploitability proxy — this is a behavioural profile, not a strength
  measure (strength comes from the eval runner, `docs/evaluation.md`).
