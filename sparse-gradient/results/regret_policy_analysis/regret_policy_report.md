# Regret under saved median timings

Phase 2 and Phase 4 strict >5% baseline checks: **passed exactly**. All three candidate medians are available for every accepted observation. Phase 2 has 162 matched configuration-draw observations per run; Phase 4 has 120 seed-configuration observations per environment/scope.

This is a descriptive policy simulation, not a trained deployment router and not a statistical generalization. It requires validation on target workload and stack. No experiments, original result regeneration, manuscript edits, or LaTeX outputs were performed.

Regret = selected median / minimum candidate median - 1. CSVs report fractions and explicit percent columns; the tables below show percentages. Every observation has equal weight, regardless of absolute runtime. Near-ties keep continuous regret: the 5% rule classifies winners but does not round regret to zero. Percentiles use linear interpolation at (n−1)p. Exact API score ties are broken lexically. The oracle is only a reference and has zero regret in every case.

## Static-policy and heuristic summaries

Oracle rows are omitted from this display because all metrics are zero. Calibrated static and winner-count choices use the same evaluation group and are explicitly in-sample descriptive references.

### phase2 / local_gtx1650ti / matched grid

Best static by mean regret: **embedding**. Most frequent resolved winner: **embedding**. Choices agree.

| Policy | Mean regret % | Median % | P90 % | P95 % | Max % | Regret ≤5% | Regret ≤10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_index_add | 30.23 | 20.45 | 85.17 | 105.01 | 145.01 | 40.7% | 43.8% |
| always_scatter_add | 112.00 | 112.91 | 220.42 | 242.39 | 314.71 | 0.0% | 1.9% |
| always_embedding | 15.03 | 0.00 | 52.33 | 78.78 | 114.90 | 63.0% | 66.0% |
| best_static_in_group | 15.03 | 0.00 | 52.33 | 78.78 | 114.90 | 63.0% | 66.0% |
| most_frequent_winner_in_group | 15.03 | 0.00 | 52.33 | 78.78 | 114.90 | 63.0% | 66.0% |
| local_dim_heuristic | 6.26 | 0.00 | 26.54 | 46.82 | 70.39 | 84.6% | 84.6% |
| transfer_t4_distribution_heuristic | 112.00 | 112.91 | 220.42 | 242.39 | 314.71 | 0.0% | 1.9% |
| transfer_t4_repeat_distribution_heuristic | 112.00 | 112.91 | 220.42 | 242.39 | 314.71 | 0.0% | 1.9% |

- local_dim_heuristic: if dim == 64 choose index_add; else embedding. Compared with the group's calibrated best-static reference: mean regret changes by -8.77 percentage points; maximum regret changes by -44.50 points. Exploratory rule from the saved local grid; no held-out validation.

- transfer_t4_distribution_heuristic: if hotspot/zipf choose scatter_add; else scatter_add (fixed from t4). Compared with the group's calibrated best-static reference: mean regret changes by +96.97 percentage points; maximum regret changes by +199.81 points. Cross-stack descriptive transfer from t4; source fallback frozen before applying to local. Rule was motivated by the full saved study, so this is not a held-out generalization estimate.

- transfer_t4_repeat_distribution_heuristic: if hotspot/zipf choose scatter_add; else scatter_add (fixed from t4_repeat). Compared with the group's calibrated best-static reference: mean regret changes by +96.97 percentage points; maximum regret changes by +199.81 points. Cross-stack descriptive transfer from t4_repeat; source fallback frozen before applying to local. Rule was motivated by the full saved study, so this is not a held-out generalization estimate.
### phase2 / t4 / matched grid

Best static by mean regret: **scatter_add**. Most frequent resolved winner: **scatter_add**. Choices agree.

| Policy | Mean regret % | Median % | P90 % | P95 % | Max % | Regret ≤5% | Regret ≤10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_index_add | 27.67 | 20.24 | 64.52 | 85.10 | 102.52 | 25.3% | 33.3% |
| always_scatter_add | 5.67 | 0.00 | 18.11 | 40.97 | 70.14 | 80.9% | 84.6% |
| always_embedding | 62.48 | 50.32 | 133.06 | 164.47 | 326.49 | 20.4% | 21.0% |
| best_static_in_group | 5.67 | 0.00 | 18.11 | 40.97 | 70.14 | 80.9% | 84.6% |
| most_frequent_winner_in_group | 5.67 | 0.00 | 18.11 | 40.97 | 70.14 | 80.9% | 84.6% |
| t4_distribution_heuristic | 5.67 | 0.00 | 18.11 | 40.97 | 70.14 | 80.9% | 84.6% |
| transfer_local_dim_heuristic | 26.76 | 14.62 | 73.98 | 81.80 | 102.53 | 36.4% | 43.8% |

- t4_distribution_heuristic: if hotspot/zipf choose scatter_add; else scatter_add. Compared with the group's calibrated best-static reference: mean regret changes by +0.00 percentage points; maximum regret changes by +0.00 points. Uniform fallback is the best static API over the entire same environment, not recalibrated on uniform cases.

- transfer_local_dim_heuristic: if dim == 64 choose index_add; else embedding. Compared with the group's calibrated best-static reference: mean regret changes by +21.08 percentage points; maximum regret changes by +32.40 points. Fixed local rule applied to T4; not adjusted using target timings. Descriptive transfer, not a held-out generalization estimate.
### phase2 / t4_repeat / matched grid

Best static by mean regret: **scatter_add**. Most frequent resolved winner: **scatter_add**. Choices agree.

| Policy | Mean regret % | Median % | P90 % | P95 % | Max % | Regret ≤5% | Regret ≤10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_index_add | 28.27 | 24.06 | 59.74 | 84.23 | 93.19 | 24.1% | 32.7% |
| always_scatter_add | 5.74 | 0.00 | 18.71 | 42.37 | 72.53 | 82.7% | 86.4% |
| always_embedding | 66.79 | 54.89 | 139.87 | 190.17 | 320.21 | 19.1% | 19.1% |
| best_static_in_group | 5.74 | 0.00 | 18.71 | 42.37 | 72.53 | 82.7% | 86.4% |
| most_frequent_winner_in_group | 5.74 | 0.00 | 18.71 | 42.37 | 72.53 | 82.7% | 86.4% |
| t4_distribution_heuristic | 5.74 | 0.00 | 18.71 | 42.37 | 72.53 | 82.7% | 86.4% |
| transfer_local_dim_heuristic | 28.91 | 21.03 | 73.11 | 81.76 | 95.33 | 37.0% | 43.8% |

- t4_distribution_heuristic: if hotspot/zipf choose scatter_add; else scatter_add. Compared with the group's calibrated best-static reference: mean regret changes by +0.00 percentage points; maximum regret changes by +0.00 points. Uniform fallback is the best static API over the entire same environment, not recalibrated on uniform cases.

- transfer_local_dim_heuristic: if dim == 64 choose index_add; else embedding. Compared with the group's calibrated best-static reference: mean regret changes by +23.17 percentage points; maximum regret changes by +22.80 points. Fixed local rule applied to T4; not adjusted using target timings. Descriptive transfer, not a held-out generalization estimate.
### phase4 / local_gtx1650ti / proxy_step

Best static by mean regret: **index_add**. Most frequent resolved winner: **index_add**. Choices agree.

| Policy | Mean regret % | Median % | P90 % | P95 % | Max % | Regret ≤5% | Regret ≤10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_index_add | 0.89 | 0.00 | 1.90 | 5.81 | 13.04 | 91.7% | 95.8% |
| always_scatter_add | 18.04 | 13.96 | 39.74 | 46.70 | 64.29 | 18.3% | 39.2% |
| always_embedding | 33.41 | 30.98 | 71.26 | 76.79 | 98.28 | 12.5% | 12.5% |
| best_static_in_group | 0.89 | 0.00 | 1.90 | 5.81 | 13.04 | 91.7% | 95.8% |
| most_frequent_winner_in_group | 0.89 | 0.00 | 1.90 | 5.81 | 13.04 | 91.7% | 95.8% |
| local_index_add_heuristic | 0.89 | 0.00 | 1.90 | 5.81 | 13.04 | 91.7% | 95.8% |

- local_index_add_heuristic: choose index_add. Compared with the group's calibrated best-static reference: mean regret changes by +0.00 percentage points; maximum regret changes by +0.00 points. Requested local proxy rule; identical to always_index_add. Region scope is a separate diagnostic replay.
### phase4 / local_gtx1650ti / accumulation_region

Best static by mean regret: **index_add**. Most frequent resolved winner: **index_add**. Choices agree.

| Policy | Mean regret % | Median % | P90 % | P95 % | Max % | Regret ≤5% | Regret ≤10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_index_add | 2.16 | 0.00 | 1.18 | 12.52 | 41.27 | 91.7% | 91.7% |
| always_scatter_add | 58.93 | 40.47 | 126.81 | 137.20 | 209.18 | 5.8% | 6.7% |
| always_embedding | 169.98 | 140.59 | 458.34 | 531.01 | 694.88 | 12.5% | 12.5% |
| best_static_in_group | 2.16 | 0.00 | 1.18 | 12.52 | 41.27 | 91.7% | 91.7% |
| most_frequent_winner_in_group | 2.16 | 0.00 | 1.18 | 12.52 | 41.27 | 91.7% | 91.7% |
| local_index_add_heuristic | 2.16 | 0.00 | 1.18 | 12.52 | 41.27 | 91.7% | 91.7% |

- local_index_add_heuristic: choose index_add. Compared with the group's calibrated best-static reference: mean regret changes by +0.00 percentage points; maximum regret changes by +0.00 points. Requested local proxy rule; identical to always_index_add. Region scope is a separate diagnostic replay.
### phase4 / colab_t4 / proxy_step

Best static by mean regret: **scatter_add**. Most frequent resolved winner: **scatter_add**. Choices agree. **Tie-dominant group:** API selection ignores the tie category; this is not a no-change or simplicity-based policy.

| Policy | Mean regret % | Median % | P90 % | P95 % | Max % | Regret ≤5% | Regret ≤10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_index_add | 3.36 | 1.18 | 10.74 | 18.78 | 22.03 | 79.2% | 85.0% |
| always_scatter_add | 0.29 | 0.00 | 1.02 | 1.44 | 2.43 | 100.0% | 100.0% |
| always_embedding | 27.23 | 20.44 | 52.26 | 68.05 | 90.15 | 0.0% | 5.0% |
| best_static_in_group | 0.29 | 0.00 | 1.02 | 1.44 | 2.43 | 100.0% | 100.0% |
| most_frequent_winner_in_group | 0.29 | 0.00 | 1.02 | 1.44 | 2.43 | 100.0% | 100.0% |
### phase4 / colab_t4 / accumulation_region

Best static by mean regret: **scatter_add**. Most frequent resolved winner: **scatter_add**. Choices agree. **Tie-dominant group:** API selection ignores the tie category; this is not a no-change or simplicity-based policy.

| Policy | Mean regret % | Median % | P90 % | P95 % | Max % | Regret ≤5% | Regret ≤10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_index_add | 4.28 | 1.84 | 13.60 | 15.68 | 32.19 | 67.5% | 82.5% |
| always_scatter_add | 0.84 | 0.00 | 2.93 | 3.45 | 7.95 | 97.5% | 100.0% |
| always_embedding | 84.34 | 74.72 | 128.72 | 161.74 | 250.81 | 0.0% | 0.0% |
| best_static_in_group | 0.84 | 0.00 | 2.93 | 3.45 | 7.95 | 97.5% | 100.0% |
| most_frequent_winner_in_group | 0.84 | 0.00 | 2.93 | 3.45 | 7.95 | 97.5% | 100.0% |

## Minimax static choices on the saved grid

Minimax here means choosing the fixed API whose largest observed regret is smallest. It is retrospective and gives no assurance about unobserved workloads. A 10% worst-case ceiling is shown only as a descriptive practical-cost screen, not a new winner threshold.

| Saved observations | index_add max % | scatter_add max % | embedding max % | Lowest observed maximum | All cases within 10%? |
|---|---:|---:|---:|---|---|
| phase2 | 145.01 | 314.71 | 326.49 | index_add | no |
| phase4 | 41.27 | 209.18 | 694.88 | index_add | no |
| both_phases | 145.01 | 314.71 | 694.88 | index_add | no |

## Cross-stack transfer

The local dimension rule is applied unchanged to both T4 runs. Conversely, each T4 distribution rule is applied to local with its uniform fallback fixed from that source T4 run's best static API over the whole run. The target stack is never used to select that fallback. T4 and its repeat are not independent stacks. Rules originated from the full saved study, so these comparisons are descriptive transfer checks rather than a formal held-out learning protocol.

Cross-stack transfer is poor relative to the target's calibrated static reference: every transferred rule increases both mean and maximum regret. The local dimension rule helps on its source grid, but the same split does not carry over to the T4 measurements. Both T4 distribution heuristics collapse to always_scatter_add because their calibrated uniform fallback is also scatter_add. No target-tuned replacement or additional feature split is introduced to improve a disappointing rule.

## What practical ties imply

A practical tie only constrains the fastest and runner-up paths. The third API can still be substantially slower. Consequently, a tie-heavy group does not make arbitrary API selection inexpensive. No simplest-implementation or no-change choice is invented for the T4 proxy.

| Phase 4 T4 scope | Static API | Tie observations | Mean regret within ties % | Maximum within ties % |
|---|---|---:|---:|---:|
| proxy_step | always_index_add | 95 | 0.95 | 4.99 |
| proxy_step | always_scatter_add | 95 | 0.36 | 2.43 |
| proxy_step | always_embedding | 95 | 24.94 | 90.15 |
| accumulation_region | always_index_add | 78 | 0.91 | 4.08 |
| accumulation_region | always_scatter_add | 78 | 1.05 | 3.89 |
| accumulation_region | always_embedding | 78 | 82.94 | 242.23 |

## Assumptions and limitations

- The candidate API set is fixed to three paths. No switching overhead, implementation complexity, memory cost, optimizer behavior, or production throughput is modeled.
- Saved medians are treated as fixed. Timing variability is not propagated; extreme ratios can be sensitive to a small fastest median. The case-level CSV preserves selected/best times and full observation identity for inspection.
- The grid is controlled and synthetic, not an independent random sample. Draws/seeds sharing configurations are related. Whole-group averages can conceal heterogeneity and absolute-time differences.
- Best-static selection and the winner-count policy use the evaluated data. The fixed workload heuristics are exploratory summaries motivated by the same study, not external validation. Oracle choices require knowing each observation's timings.
- Region replay remains a diagnostic scope, not a decomposition of the proxy step. Stack differences combine hardware, software, and host conditions.
- Optional shallow tree: Skipped: scikit-learn is unavailable in the analysis interpreter; no dependency installed.

This analysis can support a narrowly framed discussion of static-choice costs and the limits of coarse workload heuristics after reviewing outlier cases. It does not establish a deployment policy. Reproduce with `python sparse-gradient/regret_policy_analysis.py`; only saved-artifact reconstruction helpers are invoked.
