# Sparse gradient accumulation feasibility benchmark

PyTorch-only CUDA benchmark of dense gradient accumulation from repeated row
indices. Vary workload size, dimension, duplicate density, concentration, locality,
and index order to investigate when grouping beats direct accumulation. No custom
CUDA/C++, compiler, downloads, or package installation is needed. Dependencies are
CUDA-enabled PyTorch, NumPy, and pandas; matplotlib is needed for plots. tqdm and
tabulate are optional. CPU benchmark execution is rejected.

Run commands from `C:\Projects\phd-segmentacija-instanci`.

## Methodology version 3: workloads and operation scopes

Version 3 defaults to finite truncated Zipf sampling, keeps index patterns fixed
across dimension sweeps, and supports independent workload draws. Operation timing
scopes remain as established in version 2. The new seed scheme changes exact
inputs even in legacy wrapped mode; wrapped preserves the distribution algorithm,
not the old run's exact sample arrays.

**Behavior change:** `index_add` and `scatter_add` now allocate and zero a fresh
dense output inside timing. Previous results for these names excluded that work
and must not be compared without accounting for the methodology change.
`sorted_segment` also has a new implementation. Metadata records
`methodology_version`, `scope_groups`, `timing_scope_by_operation`, and
`op_properties`; each raw/summary row carries `scope_group` and `timing_scope`.

| Operation | Scope group | Work inside CUDA events |
| --- | --- | --- |
| `index_add` | `fair_dense_grad` | Fresh dense allocation + zero + indexed accumulation |
| `scatter_add` | `fair_dense_grad` | Fresh dense allocation + zero + scatter accumulation |
| `embedding` | `fair_dense_grad` | Backward on retained forward graph, including dense gradient creation |
| `embedding_bag` | `fair_dense_grad` | Retained-graph backward with one index per sum bag, including dense gradient creation |
| `sorted_segment` | `fair_dense_grad` | Stable sort, source reorder, searchsorted boundaries, segmented sum directly into fresh dense output |
| `index_add_acc_only` | `accumulate_only` | Old `index_add`: accumulation only; output reset outside timing |
| `scatter_add_acc_only` | `accumulate_only` | Old `scatter_add`: accumulation only; output reset outside timing |
| `sorted_segment_sync` | `fair_dense_grad_sync` | Old sorted implementation: sort, reorder, unique_consecutive lengths, reduce, dense zeros, index_copy |
| `embedding_full_backward` | `end_to_end` | Clear gradient, construct/execute fresh forward graph, backward without retain_graph |
| `noop_kernel` | `overhead_floor` | One-element add_ launch; no reference output |

`index_add_alloc_timed` is a deprecated alias for the new `index_add`.
`index_add_zero_timed` is a deprecated legacy variant retaining allocation outside
timing and zeroing inside. Both retain their old semantics, use the
`deprecated_alias` scope group, and are excluded from headline rankings/defaults.

Default operations are the five `fair_dense_grad` operations. Embedding module
creation and forward are outside timing for retained-graph operations; gradient
references are cleared to `None` before each invocation. Index view creation and
`sorted_segment` row ID construction are also outside timing. No operation caches
sorted indices or reductions. PyTorch's warm caching allocator is used, rather
than forcing driver allocations each time. All accumulation outputs are dense;
this does not benchmark sparse gradient storage, optimizers, or end-to-end
training. EmbeddingBag uses singleton bags, not multi-index pooling.
`embedding_full_backward` measures an end-to-end operator path and is not a
kernel contention comparison. Do not rank across scope groups.

The new `sorted_segment` uses searchsorted boundaries for every dense row,
including missing rows. `segment_reduce(offsets=bounds, reduce="sum")` produces
zero rows for empty segments. The lengths form triggered CUDA synchronization
checking on the tested environment; offsets passed that check and is used here.
`--check-sync-free` checks the complete operation once per configuration before
timing and records `sync_free`; the default is `not_checked`. Runtime failures
unrelated to synchronization remain errors. The previous debug setting is
restored. PyTorch's detector is incomplete: True means no detected sync, not proof
that the operator is universally synchronization-free. Grouped reduction is
?deterministic-ish?; use `--determinism-check` to test repeatability rather than
assuming stable sorting proves deterministic execution.

## Timing, rounds, and cache control

CUDA events and synchronization surround each timing sample. Input generation,
CPU reference checks, and plotting are outside timing. `elapsed_ms` is time per
invocation (event interval divided by `inner`), not time per batch. Event timing
can include GPU idle gaps from Python/autograd dispatch and host synchronization;
it is not a separate measurement of CPU graph construction cost.

`--rounds` defaults to 3; repeats must be at least rounds. Repeats are split evenly
(e.g. 20 becomes 7, 7, 6). Each round deterministically shuffles operations using
the workload seed and round index. Factories and warmup run afresh each round;
calibration and sync-check results are reused across rounds for that operation.
Correctness and requested determinism checks run after timing in each round.
Raw `round_index` and `run_order` are zero-based; `repeat_index` increases across
rounds. Summaries concatenate all samples, report worst errors and any failure,
and record round/order history. A failed round excludes the configuration from
winners even if other rounds passed.

`--inner auto` calibrates each operation/configuration toward a 2 ms event interval
using powers of two up to 256. Explicit choices are 1, 2, 4, 8, 16, 32, 64, 128,
and 256. Batching includes host-side reference release/gradient clearing between
invocations. Operations whose prepare launches GPU work (`*_acc_only`) always
use 1; `embedding_full_backward` is conservatively unbatched as well. `inner` is
recorded in raw and summary files. `noop_kernel` provides an overhead floor, has
correctness `not_applicable`, and is not a failure or eligible winner.

`--cache-mode warm` is default. `cold` zeroes a buffer at least twice the reported
L2 size and at least 64 MiB before each sample, outside events, and forces inner=1.
When L2 size is unavailable the fallback buffer is 128 MiB. This is a best-effort
cache eviction procedure, not a guarantee of every cache line's state.

## Workloads, seeds, and correctness

- `uniform`: equal probability over rows.
- `hotspot`: the rounded requested fraction targets the first hot rows, the rest
  cold rows, before optional remapping. If all rows are hot all samples are hot.
- `zipf`: `--zipf-mode truncated` is the new default. It explicitly normalizes
  probabilities proportional to `rank**(-alpha)` over ranks 1 through rows, then
  samples row IDs 0 through rows-1. This is a proper finite Zipf distribution.
  `--zipf-mode wrapped` preserves legacy NumPy Zipf draws mapped by
  `(sample - 1) % rows`, useful as a diagnostic. Wrapping folds the unbounded tail
  back into the table and changes the distribution. Alpha must still exceed 1
  for both modes to retain the existing CLI constraint.

`--row-layout scattered` (default) applies a seeded permutation to row IDs,
spreading hot rows over the table. `adjacent` preserves the original row IDs.
`--index-order random` preserves generated order; `sorted` sorts the final index
array before timing, including for sorting-based operations. These settings are
recorded in every row and metadata. `top1_share` and `top16_share` complement
unique count, duplicate rate, maximum row count, and mean count per used row.

`index_seed` is CRC32 of `idx|base|rows|count|dist|row_layout|index_order|zipf_mode|workload_draw`.
It drives both index generation and the row permutation and does **not** depend
on dimension. Therefore each dimension uses exactly the same index array for a
fixed workload draw and settings. `src_seed` uses the distinct `src|` prefix and
includes dimension after count. `workload_seed` remains a compatibility alias
of `index_seed`, also used to shuffle operation order. Seeds are recorded in
every raw/summary row and the metadata's per-workload seed inventory. Changing
grid position, operation selection, or round count does not change these inputs.
Distribution parameters control generation and are separately recorded; changing
Zipf mode changes seeds even for non-Zipf distributions, as specified by the key.

`--workload-seeds N` (default 1, positive integer) runs draws 0 through N-1, each
with its own index and source seeds. `workload_draw` and `zipf_mode` are recorded
in raw/summary/winner rows and metadata. All operations in one draw share indices
and source gradients. Repeats and rounds measure timing variability on that
draw; multiple draws measure workload variability. Draws remain separate in
summary_results.csv and winners.csv. `winner_summary.csv` aggregates winner
frequency across draws with a denominator counting all eligible winning draws
for the base configuration, not only draws won by one operation. Draws with no
eligible winner are excluded from that denominator; operations with zero wins
are omitted. Speedup aggregates in this file describe only the draws won by that
operation, not its speedups across every draw.

An independent CPU float64 accumulation is cast to float32 for comparison.
After checking shape and finite values, each element must satisfy
`abs_error <= 1e-5 + 1e-6 * row_count`. `max_abs_error` and `max_norm_err` are saved;
max_norm_err <= 1 passes. This row-dependent tolerance is an explicit heuristic,
not a numerical error bound. Empty rows have tolerance 1e-5. Determinism is
separate: three reruns are checked for bitwise equality and diagnostic allclose
(rtol=1e-4, atol=1e-3). Global deterministic algorithms are not changed.

## Runs

First smoke test:

```powershell
python sparse-gradient/sparse_grad_benchmark.py --quick --zipf-mode truncated --workload-seeds 2 --ops index_add scatter_add embedding embedding_bag sorted_segment noop_kernel --rounds 2 --progress off
```

Legacy distribution smoke test:

```powershell
python sparse-gradient/sparse_grad_benchmark.py --quick --zipf-mode wrapped --workload-seeds 2 --ops index_add embedding sorted_segment --rounds 2 --progress off
```

Verify exact index arrays across dimensions, seed separation, Zipf modes, and
winner-summary denominators; optionally validate and plot the latest smoke results
without running any further GPU benchmark:

```powershell
python sparse-gradient/check_workload_seeds.py --results sparse-gradient/results --save-plots
```

A fuller study, only when explicitly requested:

```powershell
python sparse-gradient/sparse_grad_benchmark.py --out-dir sparse-gradient/results_study --rows 4096 16384 --indices 4096 16384 --dims 32 64 128 --distributions uniform hotspot zipf --repeats 20 --warmup 5 --rounds 3 --save-plots
```

`--quick` sets rows=1024, indices=2048, dimensions=16/64, warmup=2, repeats=5;
it preserves other options. With the default grid and operations there are 675
operation/configuration pairs with one draw; multiply by `--workload-seeds` for
multiple draws. `--max-combinations` caps operation/configuration/draw pairs, not rounds
or samples; it selects a prefix before each round's shuffle. This is not a
balanced study. `--progress auto` uses tqdm interactively, `on` always requests
it, and `off` leaves high-level messages and final tables. Use `--help` for all
flags. Only float32 is supported. `--device cuda:0` selects the GPU.

## Outputs and limitations

Default output is `sparse-gradient/results` relative to the working directory.
Use a separate `--out-dir` for each study: matching files are overwritten. Older
plots are not deleted; only plots listed in the current metadata belong to that
session.

- `raw_results.csv`: streamed operation/round timing rows, scopes, input controls,
  statistics, inner, sync checks, and validation. Failed rounds get repeat=-1 and
  blank timing; completed rounds are flushed immediately.
- `summary_results.csv`: all rounds combined per operation/configuration/draw; mean,
  median, sample standard deviation, min/max, quartiles, CV, and validation.
- `winners.csv`: fastest eligible `fair_dense_grad` operation for each matched
  workload draw (including Zipf mode), index_add baseline/speedup, runner-up, and within_noise (winner p75
  >= runner-up p25). Blank baseline means index_add was absent/ineligible. A
  single eligible operation has no runner-up/noise claim. This quartile heuristic
  is not statistical significance and a winner only refers to selected eligible
  operations.
- `winner_summary.csv`: per base configuration and winning operation, win count,
  total eligible draws, win rate, median/mean winning speedup, and median winning
  time. Always written with headers even if no rows qualify.
- `metadata.json`: versions, GPU, CLI, seed rule, scope/property mappings,
  methodology version, cache settings, tolerance, timestamps, and run status.
- `--save-plots`: median-vs-dimension and median-vs-duplicate-rate scatter plots
  split by cache mode, row layout, index order, Zipf mode, and scope group. The operation /
  distribution comparison is restricted to fair_dense_grad. Filenames carry
  those settings. Aggregated medians are descriptive, not matched speedups.
- `--save-plots` also produces `speedup_vs_top1_share_fair_dense_grad.png` and
  `speedup_vs_max_row_count_fair_dense_grad.png` (log x-axis for row counts), using
  matched per-draw winners with an eligible index_add baseline. Colors/markers
  identify winners and a horizontal line marks 1x. Titles identify cache/layout/
  order/Zipf settings; multiple settings produce separate suffixed files. Points
  span sizes/draws, so this is descriptive rather than a controlled causal curve.
  No plot is generated when no eligible baseline-backed winner exists.

Plots exclude failed correctness and runtime failures. Winners additionally
require scope_group=fair_dense_grad; diagnostics, aliases, and noop never rank.
This is a feasibility study; cache state, clocks, dispatch, WDDM, and other GPU
activity affect timing. Moderate defaults and bounded batches reduce timeout
risk but cannot guarantee safe arbitrary workload sizes. CUDA OOM is recorded
and work continues. Fatal CUDA device errors stop execution; already streamed
rows remain. A completed run with runtime/correctness failures returns nonzero.
No long study runs automatically.

GPU telemetry is deferred. There is no torch.profiler integration, detection-cost
operation, deterministic-algorithm series, fp16/bfloat16 series, custom CUDA/C++, or
end-to-end model training in this PyTorch-only feasibility benchmark.
