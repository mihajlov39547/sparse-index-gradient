# Descriptive bootstrap intervals for aggregate winner shares

Phase 2 baseline checks: **passed** (162 matched configuration-draw observations per run).
Phase 4 baseline checks: **passed** (120 seed-configuration observations per environment/scope; seeds 4001–4005).

Bootstrap replicates: **10,000** per group and method; bootstrap seed: **20261010**.
The original strict >5% winner classification is held fixed. No benchmarks, raw timings, manuscript files, or existing result artifacts were changed.

## Resampling design

- Phase 2: resample 162 matched configuration-draw observations, with replacement, separately per run.
- Phase 4 ordinary: resample 120 seed-configuration observations, with replacement, separately per environment/scope.
- Phase 4 cluster: resample 24 configurations, with replacement; each selected cluster contributes all five saved seed observations. A repeatedly selected cluster contributes its observations repeatedly. The denominator is the number of included observations, 120 here.
- Selected shares use all observations in the denominator, including ties; they are not shares conditional on resolved cases.
- Each pair of quantities uses the same resampling indices within a replicate. A single Python random.Random stream is seeded once and consumed in the group/method order recorded in the audit.
- Endpoints are the 2.5th and 97.5th percentiles, using linear interpolation at (B−1)p in the sorted replicate shares. No rounding occurs before CSV export.

## Estimates and 95% descriptive bootstrap intervals

All entries below are percentages; CSV values are fractions. Cluster n_observations remains 120, while the number of resampling units is 24 (recorded in the audit).

| Phase | Environment / scope | Resampling unit | Quantity | Estimate | Interval |
|---|---|---|---|---:|---:|
| phase2 | local_gtx1650ti / matched grid | Configuration-draw | embedding_share | 59.3% | [51.9, 66.7]% |
| phase2 | local_gtx1650ti / matched grid | Configuration-draw | tie_share | 3.7% | [1.2, 6.8]% |
| phase2 | t4 / matched grid | Configuration-draw | scatter_add_share | 56.8% | [49.4, 64.2]% |
| phase2 | t4 / matched grid | Configuration-draw | tie_share | 24.7% | [18.5, 31.5]% |
| phase2 | t4_repeat / matched grid | Configuration-draw | scatter_add_share | 57.4% | [50.0, 64.8]% |
| phase2 | t4_repeat / matched grid | Configuration-draw | tie_share | 25.3% | [18.5, 32.1]% |
| phase4 | local_gtx1650ti / proxy_step | Observation | index_add_share | 69.2% | [60.8, 77.5]% |
| phase4 | local_gtx1650ti / proxy_step | Observation | tie_share | 22.5% | [15.0, 30.0]% |
| phase4 | local_gtx1650ti / proxy_step | Configuration cluster | index_add_share | 69.2% | [50.8, 85.8]% |
| phase4 | local_gtx1650ti / proxy_step | Configuration cluster | tie_share | 22.5% | [8.3, 39.2]% |
| phase4 | local_gtx1650ti / accumulation_region | Observation | index_add_share | 81.7% | [74.2, 88.3]% |
| phase4 | local_gtx1650ti / accumulation_region | Observation | tie_share | 10.0% | [5.0, 15.8]% |
| phase4 | local_gtx1650ti / accumulation_region | Configuration cluster | index_add_share | 81.7% | [65.8, 95.8]% |
| phase4 | local_gtx1650ti / accumulation_region | Configuration cluster | tie_share | 10.0% | [0.0, 21.7]% |
| phase4 | colab_t4 / proxy_step | Observation | tie_share | 79.2% | [71.7, 85.8]% |
| phase4 | colab_t4 / proxy_step | Observation | scatter_add_share | 20.8% | [14.2, 28.3]% |
| phase4 | colab_t4 / proxy_step | Configuration cluster | tie_share | 79.2% | [62.5, 93.3]% |
| phase4 | colab_t4 / proxy_step | Configuration cluster | scatter_add_share | 20.8% | [6.7, 37.5]% |
| phase4 | colab_t4 / accumulation_region | Observation | tie_share | 65.0% | [56.7, 73.3]% |
| phase4 | colab_t4 / accumulation_region | Observation | scatter_add_share | 32.5% | [24.2, 40.8]% |
| phase4 | colab_t4 / accumulation_region | Configuration cluster | tie_share | 65.0% | [49.2, 80.0]% |
| phase4 | colab_t4 / accumulation_region | Configuration cluster | scatter_add_share | 32.5% | [16.7, 49.2]% |

## Interpretation and limitations

The intervals summarize resampling variability of aggregate winner shares under the saved observation set. They should not be interpreted as inference to all possible workloads, GPUs, or PyTorch versions. These are descriptive uncertainty summaries and a resampling-based stability check, not hypothesis tests.

The workload grid is controlled, not an independent random sample from a population. The ordinary observation bootstrap does not preserve dependence among draws/seeds sharing a configuration. The Phase 4 cluster bootstrap preserves all five seed labels within each configuration and measures sensitivity to configuration composition; it does not model shared effects across configurations from the same seed/run. Only 24 configurations and five seeds are available. Phase 2 uses the requested configuration-draw units, not configuration clusters.

Saved medians and winner labels are treated as fixed: these intervals do not propagate within-run timing noise, threshold uncertainty, hardware variability, or software-version variability. No API difference intervals, cross-environment paired comparison, or population-sampling claim is made. The T4 repeat is a separate run on the same stack, not a third independent environment. Finite bootstrap simulation and the small discrete observation sets also limit endpoint precision.

No missing required runs or incomplete Phase 4 clusters were detected. Both 5% baselines match exactly. Any skipped runs outside the study profile/seed range are listed in the audit.

Reproduce with `python sparse-gradient/bootstrap_winner_shares.py` from the project root. Only the standard library and the read-only reconstruction helpers in `threshold_sensitivity.py` are used; its output-writing entry point is never called.
