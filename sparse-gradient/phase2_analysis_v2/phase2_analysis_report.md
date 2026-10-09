# Phase 2 Analysis: Sparse-Index Gradient Accumulation in PyTorch CUDA

## 1. Scope and analysis rules

This is an empirical systems characterization of standard PyTorch CUDA APIs for embedding-style sparse-index gradient accumulation during backward propagation. The analysis reads saved summaries and does not execute benchmarks. Headline candidates are correctness-passed, status-ok fair_dense_grad measurements. No algorithmic novelty or universal fastest API is claimed.

Practical decisions require runner_up_ms / winner_ms > 1.05; otherwise the selection is `tie`. A solitary eligible API is also not called decisive. This margin criterion is separate from the older quartile-overlap noise heuristic. Neither establishes statistical significance.

## 2. Environments and coverage

| label | gpu_name | pytorch_version | cuda_version | platform | python_version | compute_capability | methodology_version |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GTX1650Ti | NVIDIA GeForce GTX 1650 Ti | 2.13.0+cu126 | 12.6 | Windows-10-10.0.19045-SP0 | 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)] | [7, 5] | 3 |
| T4 | Tesla T4 | 2.11.0+cu130 | 13.0 | Linux-6.6.122+-x86_64-with-glibc2.39 | 3.13.15 (main, Aug  6 2026, 11:06:22) [GCC 13.3.0] | [7, 5] | 3 |
| T4_repeat | Tesla T4 | 2.11.0+cu130 | 13.0 | Linux-6.6.122+-x86_64-with-glibc2.39 | 3.13.15 (main, Aug  6 2026, 11:06:22) [GCC 13.3.0] | [7, 5] | 3 |

| label | rows | indices | dims | distributions | zipf_mode | workload_seeds | repeats | rounds | ops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GTX1650Ti | [4096, 16384, 65536] | [65536, 131072, 262144] | [64, 128, 256, 512] | ["uniform", "hotspot", "zipf"] | truncated | 3 | 21 | 3 | ["index_add", "scatter_add", "embedding", "embedding_bag", "sorted_segment", "noop_kernel"] |
| T4 | [4096, 16384, 65536] | [65536, 131072, 262144] | [64, 128, 256, 512] | ["uniform", "hotspot", "zipf"] | truncated | 3 | 21 | 3 | ["index_add", "scatter_add", "embedding", "embedding_bag", "sorted_segment", "noop_kernel"] |
| T4_repeat | [4096, 16384, 65536] | [65536, 262144] | [64, 256, 512] | ["uniform", "hotspot", "zipf"] | truncated | 3 | 21 | 3 | ["index_add", "scatter_add", "embedding", "noop_kernel"] |

Loaded 3 runs representing 2 distinct recorded GPU/PyTorch/CUDA/OS/Python stacks. Repeated runs of a stack are not independent hardware environments. Per-run descriptive coverage can differ; matched results below restrict configurations and candidate APIs.

## 3. Descriptive within-run analysis

The legacy winner tables/plots describe raw minimum-median selections on each run's own grid and API set. They are retained for exploration, not used as the main cross-run comparison. No cross-run absolute latency ranking is made.

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

Non-noise rates in this legacy table count saved within_noise=False cases divided by all retained draws. Unknown noise flags leave these rates unavailable. This is not the practical-margin rule.

### Within-run margin-aware analysis

| label | operation | win_count | total_draws | win_rate |
| --- | --- | --- | --- | --- |
| GTX1650Ti | index_add | 112 | 324 | 0.3457 |
| GTX1650Ti | scatter_add | 0 | 324 | 0 |
| GTX1650Ti | embedding | 187 | 324 | 0.5772 |
| GTX1650Ti | embedding_bag | 0 | 324 | 0 |
| GTX1650Ti | sorted_segment | 0 | 324 | 0 |
| GTX1650Ti | tie | 25 | 324 | 0.07716 |
| T4 | index_add | 10 | 324 | 0.03086 |
| T4 | scatter_add | 208 | 324 | 0.642 |
| T4 | embedding | 38 | 324 | 0.1173 |
| T4 | embedding_bag | 0 | 324 | 0 |
| T4 | sorted_segment | 0 | 324 | 0 |
| T4 | tie | 68 | 324 | 0.2099 |
| T4_repeat | index_add | 1 | 162 | 0.006173 |
| T4_repeat | scatter_add | 93 | 162 | 0.5741 |
| T4_repeat | embedding | 27 | 162 | 0.1667 |
| T4_repeat | tie | 41 | 162 | 0.2531 |

These margin-aware counts still have unmatched coverage/API sets. They are descriptive. `within_run_margin_winners.csv` records the raw selection and runner-up for auditing.

### Stability of raw selections across workload draws

| label | total_base_configurations | stable_configurations | mixed_configurations | stable_percentage | dominant_index_add_count | dominant_scatter_add_count | dominant_embedding_count | dominant_embedding_bag_count | dominant_sorted_segment_count | dominant_tie_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GTX1650Ti | 108 | 103 | 5 | 95.37 | 40 | 0 | 68 | 0 | 0 | 0 |
| T4 | 108 | 93 | 15 | 86.11 | 3 | 91 | 14 | 0 | 0 | 0 |
| T4_repeat | 54 | 47 | 7 | 87.04 | 1 | 43 | 10 | 0 | 0 | 0 |

Stability here retains the legacy definition win_rate=1 for raw selections; it is not margin-aware stability. Equal maximum win rates are explicitly assigned `tie`, with tied APIs listed in stable_winner_details.csv. A one-draw configuration is trivially stable. Ineligible draws are absent from the denominator.

## 4. Matched cross-run analysis

Selected common APIs: index_add, scatter_add, embedding. There are **162 common draw-configurations per run**. A draw enters only if every selected API has a positive finite median, passed correctness, status=ok, and scope_group=fair_dense_grad in every run. Duplicate operation/draw records are excluded rather than cherry-picked.

Keys: rows, indices, dim, distribution, zipf_mode, row_layout, index_order, cache_mode, workload_draw. Configuration matching is not a guarantee of bitwise-identical CUDA-generated source values across software versions. Parameters outside these keys are audited below.

Raw matched selections are in matched_winners_common_ops.csv and matched_winner_counts_by_run.csv. Paper-facing results use the practical-margin counts below.

## 5. Matched margin-aware analysis

- GTX1650Ti: over 162 common matched draw-configurations, most frequent decisive API(s): embedding, with 96/162 draws each; near-ties: 6/162 (3.7%).
- T4: over 162 common matched draw-configurations, most frequent decisive API(s): scatter_add, with 92/162 draws each; near-ties: 40/162 (24.7%).
- T4_repeat: over 162 common matched draw-configurations, most frequent decisive API(s): scatter_add, with 93/162 draws each; near-ties: 41/162 (25.3%).

| label | operation | win_count | total_draws | win_rate |
| --- | --- | --- | --- | --- |
| GTX1650Ti | index_add | 60 | 162 | 0.3704 |
| GTX1650Ti | scatter_add | 0 | 162 | 0 |
| GTX1650Ti | embedding | 96 | 162 | 0.5926 |
| GTX1650Ti | tie | 6 | 162 | 0.03704 |
| T4 | index_add | 5 | 162 | 0.03086 |
| T4 | scatter_add | 92 | 162 | 0.5679 |
| T4 | embedding | 25 | 162 | 0.1543 |
| T4 | tie | 40 | 162 | 0.2469 |
| T4_repeat | index_add | 1 | 162 | 0.006173 |
| T4_repeat | scatter_add | 93 | 162 | 0.5741 |
| T4_repeat | embedding | 27 | 162 | 0.1667 |
| T4_repeat | tie | 41 | 162 | 0.2531 |

A tie is a near-tie between the fastest and runner-up; it does not imply every candidate is equally fast. Exact raw timing ties retain a reproducible operation ordering only for the audit column; they never become decisive selections.

## 6. All-operation speedup analysis

The following table includes every selected API's wins and losses on the identical matched draw-configurations. Ratios are index_add median / operation median within each run and draw; values below 1 are slowdowns.

| label | operation | count | median_speedup | geo_mean_speedup | p25 | p75 | min_speedup | max_speedup | geo_ci_low | geo_ci_high | bootstrap_clusters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GTX1650Ti | embedding | 162 | 1.205 | 1.117 | 0.7984 | 1.48 | 0.4653 | 2.45 | 1.001 | 1.235 | 54 |
| GTX1650Ti | index_add | 162 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 54 |
| GTX1650Ti | scatter_add | 162 | 0.7168 | 0.6282 | 0.4607 | 0.8679 | 0.2687 | 0.9385 | 0.5704 | 0.6876 | 54 |
| T4 | embedding | 162 | 0.7844 | 0.8172 | 0.6179 | 1.148 | 0.3406 | 1.956 | 0.7309 | 0.9105 | 54 |
| T4 | index_add | 162 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 54 |
| T4 | scatter_add | 162 | 1.137 | 1.195 | 1.051 | 1.325 | 0.8424 | 2.025 | 1.147 | 1.248 | 54 |
| T4_repeat | embedding | 162 | 0.8002 | 0.8008 | 0.614 | 1.143 | 0.3103 | 1.932 | 0.713 | 0.8917 | 54 |
| T4_repeat | index_add | 162 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 54 |
| T4_repeat | scatter_add | 162 | 1.14 | 1.203 | 1.049 | 1.345 | 0.9337 | 1.916 | 1.156 | 1.255 | 54 |

all_op_speedup_vs_index_add.csv and all_op_speedup_summary*.csv cover each run's own eligible API/baseline pairs and are descriptive. matched_all_op_speedup*.csv restrict both configurations and APIs for the main comparison. The older speedup_summary*.csv files are winner-conditioned and must not be interpreted as each API's average benefit.

Geometric means use exp(mean(log(ratio))). Optional 95% percentile intervals use 2000 bootstrap resamples with RNG seed 1234, clustering workload draws by BASE configuration; intervals are omitted with fewer than two clusters or zero samples. They describe resampling of this selected grid, not independent experiments, hardware populations, or a significance test.

## 7. GPU-bound subset (noop-screened proxy)

The screen requires operation median / same-draw noop median >= 5. **162 common draw-configurations per run** survive for all selected APIs in all runs. Missing/invalid noop values cannot pass the screen. gpu_bound_summary_results.csv contains individually eligible fair rows; the matched GPU-bound file applies the stricter all-API/all-run intersection.

- GTX1650Ti: over 162 common noop-screened draw-configurations, most frequent decisive API(s): embedding, with 96/162 draws each; near-ties: 6/162 (3.7%).
- T4: over 162 common noop-screened draw-configurations, most frequent decisive API(s): scatter_add, with 92/162 draws each; near-ties: 40/162 (24.7%).
- T4_repeat: over 162 common noop-screened draw-configurations, most frequent decisive API(s): scatter_add, with 93/162 draws each; near-ties: 41/162 (25.3%).

| label | operation | win_count | total_draws | win_rate |
| --- | --- | --- | --- | --- |
| GTX1650Ti | index_add | 60 | 162 | 0.3704 |
| GTX1650Ti | scatter_add | 0 | 162 | 0 |
| GTX1650Ti | embedding | 96 | 162 | 0.5926 |
| GTX1650Ti | tie | 6 | 162 | 0.03704 |
| T4 | index_add | 5 | 162 | 0.03086 |
| T4 | scatter_add | 92 | 162 | 0.5679 |
| T4 | embedding | 25 | 162 | 0.1543 |
| T4 | tie | 40 | 162 | 0.2469 |
| T4_repeat | index_add | 1 | 162 | 0.006173 |
| T4_repeat | scatter_add | 93 | 162 | 0.5741 |
| T4_repeat | embedding | 27 | 162 | 0.1667 |
| T4_repeat | tie | 41 | 162 | 0.2531 |

This is a launch-overhead proxy, not profiler proof of GPU-bound execution. Noop and accumulation paths may have different batching, host dispatch, allocation, or synchronization costs. Selection by this proxy changes the workload subset; report its size alongside the full matched results.

## 8. Quality diagnostics

| label | summary_rows | median_cv | p75_cv | max_cv | max_norm_err_max | rows_with_failed_correctness | sync_free_true_count | sync_free_false_count | sync_free_unchecked_count | determinism_bitwise_true_count | determinism_bitwise_false_count | determinism_bitwise_unchecked_count | determinism_allclose_true_count | determinism_allclose_false_count | determinism_allclose_unchecked_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GTX1650Ti | 1944 | 0.01185 | 0.02527 | 2.47 | 0.2301 | 0 | 0 | 0 | 1944 | 0 | 0 | 1944 | 0 | 0 | 1944 |
| T4 | 1944 | 0.0395 | 0.0657 | 1.163 | 0.283 | 0 | 0 | 0 | 1944 | 0 | 0 | 1944 | 0 | 0 | 1944 |
| T4_repeat | 648 | 0.03341 | 0.0629 | 2.406 | 0.2298 | 0 | 0 | 0 | 648 | 0 | 0 | 648 | 0 | 0 | 648 |

CV statistics cover available finite values in all saved summary rows, including diagnostics/noop; max_norm_err covers available validation values. Unchecked synchronization/determinism is not a pass. Counts and maxima describe the saved checks only; the analysis does not rerun numerical validation. Failed correctness rows are excluded from fair comparisons.

## 9. Interpretation and relevance to training

Any environment dependence refers to the combined hardware/software stack. GPU, OS, PyTorch, CUDA, and hosted runtime conditions differ together; these data do not isolate a hardware or software cause. No absolute timing across different stacks is treated as a controlled measurement. Practitioners should measure representative workloads on their deployment stack.

The workload family is relevant to embedding backward, recommender/DLRM training, negative sampling, GNN aggregation/backward, MoE routing or token/expert dispatch, and custom scatter/gather gradients. These are motivating applications, not measured complete models. Results do not directly apply to dense GEMM-based linear/conv/attention matmul paths.

## 10. Limitations and next phase

Coverage is limited to the loaded stacks and sampled grids. Practical margins and raw-draw stability are not statistical significance. There is no profiler attribution, deterministic-mode series, FBGEMM/TorchRec baseline, custom CUDA implementation, or end-to-end model training. Generalization to other accelerators is untested. Phase 3 should profile representative decisive and near-tie cases, test a deterministic-mode subset, and consider index_put_(accumulate=True) and feasible FBGEMM/TorchRec comparisons.

## 11. Safe contribution statement

This study provides a reproducible empirical characterization of standard PyTorch CUDA APIs for embedding-style sparse-index gradient accumulation. In matched workload configurations across two CUDA software/hardware environments, the fastest API varies by workload and environment; margin-aware analysis distinguishes decisive wins from near-ties.

## Input audit

No missing inputs or audit exceptions.

## Generated plots

- [winner_share_by_run.png](winner_share_by_run.png)
- [winner_share_by_dim_by_run.png](winner_share_by_dim_by_run.png)
- [winner_share_by_distribution_by_run.png](winner_share_by_distribution_by_run.png)
- [speedup_vs_index_add_by_run.png](speedup_vs_index_add_by_run.png)
- [embedding_vs_scatter_add_cross_environment.png](embedding_vs_scatter_add_cross_environment.png)
- [matched_margin_winner_share_by_run.png](matched_margin_winner_share_by_run.png)
- [gpu_bound_matched_margin_winner_share_by_run.png](gpu_bound_matched_margin_winner_share_by_run.png)
- [matched_margin_winner_share_by_dim_GTX1650Ti.png](matched_margin_winner_share_by_dim_GTX1650Ti.png)
- [matched_margin_winner_share_by_dim_T4.png](matched_margin_winner_share_by_dim_T4.png)
- [matched_margin_winner_share_by_dim_T4_repeat.png](matched_margin_winner_share_by_dim_T4_repeat.png)
- [all_op_geomean_speedup_vs_index_add_by_run.png](all_op_geomean_speedup_vs_index_add_by_run.png)
