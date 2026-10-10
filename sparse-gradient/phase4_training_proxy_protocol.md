# Phase 4: Mini End-to-End Training-Step Proxy

## Purpose and scope

Question: Does the relative behavior observed in isolated sparse-index accumulation
remain visible when the accumulation path is embedded inside a small
backpropagation-style training step?

This is a controlled synthetic mini proxy, not full production training. Interpret
results together with Phases 1–3. No Phase 2 agreement is assumed or enforced.
There is no convergence or accuracy evaluation, distributed training, data loading,
optimizer state/update, sharding, compilation, mixed precision, or profiler in the
timing loop. Parameters and targets remain fixed across repeated steps.

## Variants and comparability

All variants use the same FP32 table, indices, mean pooling (8 references per
group), learnable linear head from `dim` to one scalar including bias, and mean
squared error against fixed synthetic targets. `autograd.grad` computes fresh dense
table, head and bias gradients. No persistent `.grad` buffer needs resetting.

| Variant | Lookup and dense table-gradient path |
| --- | --- |
| `embedding_backward_proxy` | `F.embedding`, standard dense Embedding backward, no padding or frequency scaling |
| `index_add_proxy` | Custom autograd lookup (`index_select`); backward allocates zeros and calls `index_add_` |
| `scatter_add_proxy` | Same custom lookup; backward allocates zeros and calls `scatter_add_` with a precomputed expanded index view |

The head, loss, pooling, and gradient flow are shared. Each step includes a fresh
lookup and graph. Manual variants replace only the lookup backward segment and
still traverse autograd for the surrounding computation. They do not reproduce
every detail of native Embedding dispatch: the custom Python autograd node and
native node have different overheads. The lookup APIs also differ, despite
equivalent semantics. These are API implementation comparisons, not pure kernel
speedup estimates. All dense gradients are freshly allocated within the measured
scope, avoiding an advantage from preallocating only the manual outputs.

## Workloads and reproducibility

Study grid: rows `[4096, 65536]`, index references `[8192, 65536]`, dimensions
`[32, 128]`, distributions `[uniform, zipf, hotspot]`: 24 configurations, each
with all three variants. Pilot uses rows 4096, references 8192, both dimensions
and all distributions (6 configurations). Smoke uses rows 32, references 64,
dimension 8 and all distributions (3 configurations).

Uniform samples with replacement. Zipf uses finite normalized rank probabilities
with exponent 1.2, without wrapping. Hotspot sends exactly round(0.8 * references)
draws to 16 hot rows and the rest to cold rows. A random row permutation scatters
physical row locations; reference order stays random. These match the earlier
study's distribution definitions, but **not its exact seed mapping or source
gradient distribution**. Indices are fixed across dimension sweeps; values use a
dimension-specific seed. CRC32-derived seeds and an index SHA256 are recorded.
CPU generation and transfers occur outside timing. Tables use normal values with
standard deviation 0.1; head weights use standard deviation `1/sqrt(dim)`;
targets use standard deviation 0.1 and initial bias is zero.

The default seed is 4001. Deterministic inputs do not imply bitwise deterministic
CUDA atomics; deterministic-algorithm mode is recorded but not forced. TF32 is
disabled. Record software/hardware metadata, retain every run, and use additional
seeds as independent workload draws when assessing robustness. Iteration repeats
are not independent workload draws.

## Validation

Before any timing in a configuration, independently compute the mean-pooling,
linear-head and MSE derivatives analytically in CPU float64. Accumulate the
reference dense gradient in float64. Compare every variant's table/head/bias
gradients and loss to this reference. For the separate region measurement,
generate a fixed actual FP32 upstream gradient using the shared head/loss and
validate the region result against a CPU float64 accumulation of that source.

Check shape, finite values, and elementwise
`abs(actual-reference) <= atol + rtol * abs(reference)`;
defaults are `atol=1e-8`, `rtol=1e-4`. Record maximum absolute error and maximum
relative error over reference entries with magnitude greater than `atol` (NA if
there are none). Summary error columns refer to the full-step table gradient;
JSON contains all checks. No strict bitwise comparison is used. If any variant
fails, exclude the **whole configuration** from timing and summary, preserve
validation evidence, and return a nonzero exit code. Investigate failures before
adjusting tolerances; do not tune tolerances to a desired winner.

## Timing methodology

Two separately measured scopes:

* `proxy_step_ms`: fresh lookup, pooling, head, scalar loss, and backward for all
  trainable tensors, including graph dispatch and dense gradient allocation/zeroing.
* `accumulation_region_ms`: backward through a retained lookup-only graph using
  the fixed upstream gradient. Includes autograd dispatch, dense gradient
  allocation/zeroing and row accumulation. Forward, pooling, head and loss are
  excluded. This is a separate diagnostic replay, not a nested measurement or a
  decomposition of total-step time. Do not subtract it from the total or infer
  an exact percentage of the step spent accumulating.

Use CUDA events on the current default stream with device synchronization before
and after each measured invocation. Events are reused; no event hooks are placed
inside the total step. CUDA event elapsed time includes stream execution and may
include idle gaps from Python dispatch, but is not complete host wall-clock
training latency. CPU fallback uses `perf_counter_ns` and is diagnostic only.
`--device cuda` fails explicitly if unavailable; `auto` chooses CUDA when present.

Smoke: 1 warmup and 2 timed iterations per variant/scope. Pilot: 3 and 10. Study:
10 and 40. Each scope is warmed separately. Variant order is deterministically
shuffled within each repeat to reduce fixed-order bias. No batching/calibration,
profiler or cache flush is used. Inputs and allocator caches are warm. Outputs
are released between samples. Setup, index view creation, validation, reference
calculation and transfers are excluded; forward intermediates and backward
outputs are intentionally allocated inside the total-step scope.

Report median, IQR (75th minus 25th percentile), and sample standard deviation
divided by mean (CV; NA with one repeat). Raw CSV retains all samples and execution
order. For each scope, `practical_margin_* = runner_up_median / fastest_median - 1`.
A winner is named only when this is strictly greater than 0.05; otherwise `tie`.
This matches Phase 2's 5% rule. The margin is a fraction, not a percentage or
confidence interval. It does not establish statistical significance or filter
noisy cases automatically. CPU and smoke winners have no paper-facing meaning.

## Artifacts

Default path, relative to the script directory:
`results/phase4_training_proxy/<UTC timestamp>_<profile>_<unique suffix>/`.
`--out-dir` may select a new directory; existing directories are rejected to
protect prior results.

* `phase4_raw.csv`: configuration/environment/seed identity, variant, scope,
  repeat, randomized position, individual latency in ms.
* `phase4_summary.csv`: environment, device/software versions, seed/profile,
  shape/distribution, variant, median/IQR/CV for both scopes, validation/error
  fields, both winners and observed practical margins, workload seeds/index hash.
* `phase4_validation.json`: per-configuration and per-variant checks, including
  shape/finite flags, loss/head/bias/region checks and absolute/relative errors.
* `phase4_metadata.json`: resolved CLI settings, timing scopes, hardware/software
  identity, CPU diagnostic flag, precision/determinism settings, script hash,
  timestamps and completion status. CUDA version is NA for CPU-only builds.

Unavailable CSV fields are empty; JSON uses null. Exceptions export completed
configurations and mark metadata `error`; validation failures mark
`validation_failed`. Partial artifacts must not be treated as a completed grid.
The smoke wrapper checks artifacts in a temporary directory and removes them.

## Interpretation rules

Match Phase 2 hardware/software environment, shapes, distributions, locality and
common APIs before comparing rankings. Use Phase 2's comparable dense-gradient
scope. This proxy's source gradients are structured by pooling and head/loss,
unlike arbitrary microbenchmark sources; differences are scientifically relevant.
Use the two timing scopes as complementary evidence, not interchangeable samples.

* **Supports the isolated benchmark pattern:** clear >5% rankings in the isolated
  benchmark persist in total proxy-step results and are stable across workload
  draws; region replay offers consistent supporting evidence.
* **Partially supports it:** behavior persists only in some matched configurations,
  or region ranking persists but surrounding computation reduces the total-step
  difference to a practical tie. Report exactly which subset supports it.
* **Does not support it:** adequately resolved total-step comparisons consistently
  reverse or fail to preserve the isolated pattern on the matched grid; report
  this directly, without selecting only favorable workloads.
* **Inconclusive due to overhead/noise:** practical ties, unstable ranks across
  seeds, high CV or broad IQR relative to the observed gap prevent a directional
  conclusion. CPU runs, smoke timings, validation failures and unmatched
  environments cannot supply CUDA support/non-support evidence.

The script deliberately does not assign these interpretations automatically:
they require matched Phase 2 evidence and inspection of variation. Define the
matched set before examining winners. Preserve ties and excluded configurations
in coverage accounting. Do not treat a >5% point estimate as proof of robustness.

## Commands and pre-execution review

Use a Python environment with NumPy and PyTorch (CUDA-enabled for GPU evidence).
From PowerShell:

```powershell
Set-Location C:\Projects\sparse-index-gradient\sparse-gradient
python phase4_training_proxy_smoke.py
python phase4_training_proxy.py --profile pilot
python phase4_training_proxy.py --profile study --seed 4001
```

For paper-facing runs, prefer an explicit CUDA requirement to prevent fallback:

```powershell
python phase4_training_proxy.py --profile pilot --device cuda
python phase4_training_proxy.py --profile study --seed 4001 --device cuda
```

To retain a tiny run's artifacts:

```powershell
python phase4_training_proxy.py --profile smoke
```

Review the pooling size, structured gradient distribution, tolerance/error scales,
choice of matched Phase 2 subset, and number of independent seeds before study
execution. Run pilot first to assess memory, noise and runtime on the intended
GPU. The largest configurations allocate multiple dense/intermediate tensors and
a CPU float64 reference; free memory and device contention matter. Small scopes
can be dominated by dispatch overhead. PyTorch version changes may change kernel
selection; metadata and index hashes aid auditing but do not guarantee identical
results across versions. No manuscript changes or performance conclusions follow
from implementation alone.
