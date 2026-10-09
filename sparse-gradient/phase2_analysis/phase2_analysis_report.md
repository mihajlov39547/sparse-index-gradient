# Phase 2 Analysis: Sparse-Index Gradient Accumulation in PyTorch CUDA

## 1. Scope

This is an empirical systems characterization of standard PyTorch CUDA API choices for embedding-style sparse-index gradient accumulation during backward propagation. Equivalent dense gradient outputs are compared within the `fair_dense_grad` scope. It is not an algorithmic novelty claim. This package analyzes saved results; it runs no CUDA benchmarks.

## 2. Environments

| label | gpu_name | compute_capability | pytorch_version | cuda_version | python_version | platform | methodology_version |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GTX1650Ti | NVIDIA GeForce GTX 1650 Ti | [7, 5] | 2.13.0+cu126 | 12.6 | 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)] | Windows-10-10.0.19045-SP0 | 3 |
| T4 | Tesla T4 | [7, 5] | 2.11.0+cu130 | 13.0 | 3.13.15 (main, Aug  6 2026, 11:06:22) [GCC 13.3.0] | Linux-6.6.122+-x86_64-with-glibc2.39 | 3 |
| T4_repeat | Tesla T4 | [7, 5] | 2.11.0+cu130 | 13.0 | 3.13.15 (main, Aug  6 2026, 11:06:22) [GCC 13.3.0] | Linux-6.6.122+-x86_64-with-glibc2.39 | 3 |

### Workload coverage

| label | rows | indices | dims | distributions | zipf_mode | workload_seeds | repeats | rounds | completed_combinations | failed_combinations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GTX1650Ti | [4096, 16384, 65536] | [65536, 131072, 262144] | [64, 128, 256, 512] | ["uniform", "hotspot", "zipf"] | truncated | 3 | 21 | 3 | 1944 | 0 |
| T4 | [4096, 16384, 65536] | [65536, 131072, 262144] | [64, 128, 256, 512] | ["uniform", "hotspot", "zipf"] | truncated | 3 | 21 | 3 | 1944 | 0 |
| T4_repeat | [4096, 16384, 65536] | [65536, 262144] | [64, 256, 512] | ["uniform", "hotspot", "zipf"] | truncated | 3 | 21 | 3 | 648 | 0 |

Run labels are preserved throughout. The T4 repeat is another run of that stack, not a third independent CUDA environment. Its smaller grid and reduced API set mean its overall shares are not a matched replication of the full grid. See environment_summary.csv for operation sets, hotspot parameters, Zipf alpha, cache mode, layout, and order.

## 3. Main Findings

- **GTX1650Ti** (NVIDIA GeForce GTX 1650 Ti): `index_add` 123/324 (38.0%); `embedding` 201/324 (62.0%). Measured APIs with no wins: `scatter_add`, `embedding_bag`, `sorted_segment`.
- **T4** (Tesla T4): `index_add` 21/324 (6.5%); `scatter_add` 260/324 (80.2%); `embedding` 43/324 (13.3%). Measured APIs with no wins: `embedding_bag`, `sorted_segment`. All recorded wins for `index_add` overlap the runner-up under the saved noise heuristic.
- **T4_repeat** (Tesla T4): `index_add` 7/162 (4.3%); `scatter_add` 124/162 (76.5%); `embedding` 31/162 (19.1%).  All recorded wins for `index_add` overlap the runner-up under the saved noise heuristic.

The results show that the fastest PyTorch API is not determined solely by the mathematical operation. For equivalent sparse-index accumulation, the winning operation varies by workload parameters and by GPU/software stack. These findings concern the saved workloads and environments, not universal behavior.

In the supplied confirmation runs, the GTX result supports an `embedding` versus `index_add` crossover, while the T4 stack favors `scatter_add`, particularly for hotspot/Zipf workloads. Uniform workloads and larger dimensions retain competitive embedding regimes; the dimension and distribution tables below show the boundaries. These observations motivate measuring API choices in the actual training stack.

## 4. Winner Analysis

| label | operation | per_draw_win_count | per_draw_win_rate | non_noise_win_count | non_noise_win_rate | total_draws |
| --- | --- | --- | --- | --- | --- | --- |
| GTX1650Ti | index_add | 123 | 0.3796 | 121 | 0.3735 | 324 |
| GTX1650Ti | scatter_add | 0 | 0 | 0 | 0 | 324 |
| GTX1650Ti | embedding | 201 | 0.6204 | 193 | 0.5957 | 324 |
| GTX1650Ti | embedding_bag | 0 | 0 | 0 | 0 | 324 |
| GTX1650Ti | sorted_segment | 0 | 0 | 0 | 0 | 324 |
| T4 | index_add | 21 | 0.06481 | 0 | 0 | 324 |
| T4 | scatter_add | 260 | 0.8025 | 223 | 0.6883 | 324 |
| T4 | embedding | 43 | 0.1327 | 41 | 0.1265 | 324 |
| T4 | embedding_bag | 0 | 0 | 0 | 0 | 324 |
| T4 | sorted_segment | 0 | 0 | 0 | 0 | 324 |
| T4_repeat | index_add | 7 | 0.04321 | 0 | 0 | 162 |
| T4_repeat | scatter_add | 124 | 0.7654 | 111 | 0.6852 | 162 |
| T4_repeat | embedding | 31 | 0.1914 | 27 | 0.1667 | 162 |

Per-draw rates use all retained winner rows in that run as denominator. Non-noise counts require `within_noise == False`; non-noise rates use the same all-draw denominator, not only decisive draws. Missing/unknown noise flags leave those statistics blank. Quartile overlap is a heuristic, not a significance test.

### Stability across workload draws

| label | total_base_configurations | stable_configurations | mixed_configurations | stable_percentage |
| --- | --- | --- | --- | --- |
| GTX1650Ti | 108 | 103 | 5 | 95.37 |
| T4 | 108 | 93 | 15 | 86.11 |
| T4_repeat | 54 | 47 | 7 | 87.04 |

Stability means the same operation won every eligible draw (win_rate=1); it does not imply a non-noise margin. A one-draw configuration is trivially stable. Dominant-operation ties are reported in stable_winner_details.csv and assigned alphabetically for the summary counts. Draws with no eligible winner do not enter the denominator.

### By distribution

| label | distribution | winner | win_count | win_rate |
| --- | --- | --- | --- | --- |
| GTX1650Ti | hotspot | embedding | 66 | 0.6111 |
| GTX1650Ti | hotspot | index_add | 42 | 0.3889 |
| GTX1650Ti | uniform | embedding | 75 | 0.6944 |
| GTX1650Ti | uniform | index_add | 33 | 0.3056 |
| GTX1650Ti | zipf | embedding | 60 | 0.5556 |
| GTX1650Ti | zipf | index_add | 48 | 0.4444 |
| T4 | hotspot | embedding | 1 | 0.009259 |
| T4 | hotspot | index_add | 3 | 0.02778 |
| T4 | hotspot | scatter_add | 104 | 0.963 |
| T4 | uniform | embedding | 42 | 0.3889 |
| T4 | uniform | scatter_add | 66 | 0.6111 |
| T4 | zipf | index_add | 18 | 0.1667 |
| T4 | zipf | scatter_add | 90 | 0.8333 |
| T4_repeat | hotspot | embedding | 1 | 0.01852 |
| T4_repeat | hotspot | index_add | 3 | 0.05556 |
| T4_repeat | hotspot | scatter_add | 50 | 0.9259 |
| T4_repeat | uniform | embedding | 30 | 0.5556 |
| T4_repeat | uniform | scatter_add | 24 | 0.4444 |
| T4_repeat | zipf | index_add | 4 | 0.07407 |
| T4_repeat | zipf | scatter_add | 50 | 0.9259 |

### By dimension

| label | dim | winner | win_count | win_rate |
| --- | --- | --- | --- | --- |
| GTX1650Ti | 64 | embedding | 17 | 0.2099 |
| GTX1650Ti | 64 | index_add | 64 | 0.7901 |
| GTX1650Ti | 128 | embedding | 50 | 0.6173 |
| GTX1650Ti | 128 | index_add | 31 | 0.3827 |
| GTX1650Ti | 256 | embedding | 65 | 0.8025 |
| GTX1650Ti | 256 | index_add | 16 | 0.1975 |
| GTX1650Ti | 512 | embedding | 69 | 0.8519 |
| GTX1650Ti | 512 | index_add | 12 | 0.1481 |
| T4 | 64 | index_add | 12 | 0.1481 |
| T4 | 64 | scatter_add | 69 | 0.8519 |
| T4 | 128 | embedding | 3 | 0.03704 |
| T4 | 128 | index_add | 3 | 0.03704 |
| T4 | 128 | scatter_add | 75 | 0.9259 |
| T4 | 256 | embedding | 19 | 0.2346 |
| T4 | 256 | index_add | 6 | 0.07407 |
| T4 | 256 | scatter_add | 56 | 0.6914 |
| T4 | 512 | embedding | 21 | 0.2593 |
| T4 | 512 | scatter_add | 60 | 0.7407 |
| T4_repeat | 64 | index_add | 6 | 0.1111 |
| T4_repeat | 64 | scatter_add | 48 | 0.8889 |
| T4_repeat | 256 | embedding | 15 | 0.2778 |
| T4_repeat | 256 | index_add | 1 | 0.01852 |
| T4_repeat | 256 | scatter_add | 38 | 0.7037 |
| T4_repeat | 512 | embedding | 16 | 0.2963 |
| T4_repeat | 512 | scatter_add | 38 | 0.7037 |

Additional row-count and index-count breakdowns are in winner_counts_by_rows.csv and winner_counts_by_indices.csv.

## 5. Speedup Analysis

| label | winner | count | median_speedup | mean_speedup | min_speedup | max_speedup | p25 | p75 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GTX1650Ti | embedding | 201 | 1.338 | 1.426 | 1.003 | 2.45 | 1.186 | 1.634 |
| GTX1650Ti | index_add | 123 | 1 | 1 | 1 | 1 | 1 | 1 |
| T4 | embedding | 43 | 1.311 | 1.425 | 1.112 | 1.956 | 1.188 | 1.534 |
| T4 | index_add | 21 | 1 | 1 | 1 | 1 | 1 | 1 |
| T4 | scatter_add | 260 | 1.21 | 1.261 | 1.007 | 2.068 | 1.067 | 1.377 |
| T4_repeat | embedding | 31 | 1.366 | 1.41 | 1.038 | 1.932 | 1.176 | 1.55 |
| T4_repeat | index_add | 7 | 1 | 1 | 1 | 1 | 1 | 1 |
| T4_repeat | scatter_add | 124 | 1.241 | 1.267 | 1.002 | 1.916 | 1.05 | 1.427 |

Speedups compare the winner with `index_add` in the same run, configuration, and workload draw. Missing/nonfinite baselines are excluded from numerical summaries. These are winner-conditioned speedups, not every API's average advantage across the entire grid. No absolute GTX/T4 latency comparison is made. Detailed dimension/distribution summaries are in speedup_summary_by_dim_distribution.csv.

## 6. Relevance to AI Training

Sparse-index gradient accumulation appears in embedding backward, recommender systems and DLRMs, negative sampling, GNN aggregation/backward, MoE routing or token/expert dispatch, and custom scatter/gather training kernels. These are motivating workload families; this benchmark does not measure complete models in them. The conclusions do not directly apply to dense GEMM-based layers such as ordinary linear/conv/attention matmul paths.

## 7. Interpretation

The local GTX observations suggest an embedding/index_add crossover, while the T4 observations suggest scatter_add can dominate under that stack. GPU, OS, PyTorch, CUDA, and execution conditions differ together, so the data do not isolate a hardware cause or a software-version cause. Practitioners should benchmark on their actual deployment stack. A smaller repeat supports a qualitative pattern only over its sampled subset.

## 8. Limitations

- Only two CUDA environments in the supplied study, with different PyTorch/CUDA and Python/OS versions.
- Grid coverage and candidate API sets differ for the smaller repeat; overall win shares depend on that coverage.
- No profiler attribution, deterministic-mode series, or FBGEMM/TorchRec baseline yet.
- No custom CUDA kernels or end-to-end model training.
- No generalization established for A100/H100 or AMD GPUs.
- Timing noise, caching, thermal/clock state, WDDM, and dispatch overhead can affect outcomes; winner and stability counts do not establish statistical significance.

## 9. Next Phase

Phase 3 should profile representative winner/crossover cases, run a deterministic-mode subset, consider an `index_put_(accumulate=True)` baseline, and optionally evaluate FBGEMM/TorchRec if installation is feasible. Preserve matched workloads and report versions so implementation-level attribution can be separated from end-to-end impact.

## 10. Safe Contribution Statement

This study provides a reproducible empirical characterization of standard PyTorch CUDA APIs for embedding-style sparse-index gradient accumulation. Across controlled workload parameters and two CUDA environments, we identify stable but environment-dependent winner regimes among `Embedding.backward`, `index_add`, and `scatter_add`.

## Input audit and generated plots

All requested inputs loaded; no missing files.

- [winner_share_by_run.png](winner_share_by_run.png)
- [winner_share_by_dim_by_run.png](winner_share_by_dim_by_run.png)
- [winner_share_by_distribution_by_run.png](winner_share_by_distribution_by_run.png)
- [speedup_vs_index_add_by_run.png](speedup_vs_index_add_by_run.png)
- [embedding_vs_scatter_add_cross_environment.png](embedding_vs_scatter_add_cross_environment.png)
