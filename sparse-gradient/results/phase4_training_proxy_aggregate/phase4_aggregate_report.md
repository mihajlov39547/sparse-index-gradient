# Phase 4 aggregate report

Saved-artifact analysis only; no experiments executed. No benchmark or existing manuscript files modified.

Discovered **10** runs; included **10**; excluded **0**.

## Run audit

| Environment | Run | Seed | Decision |
|---|---|---:|---|
| local_gtx1650ti | `results\phase4_training_proxy\20261010T085020Z_study_0839553c` | 4001 | included |
| local_gtx1650ti | `results\phase4_training_proxy\20261010T090324Z_study_3a1b1044` | 4002 | included |
| local_gtx1650ti | `results\phase4_training_proxy\20261010T090349Z_study_e3974ed8` | 4003 | included |
| local_gtx1650ti | `results\phase4_training_proxy\20261010T090414Z_study_73b3eeae` | 4004 | included |
| local_gtx1650ti | `results\phase4_training_proxy\20261010T090438Z_study_7e9cbeff` | 4005 | included |
| colab_t4 | `results\t4_phase4_training_proxy\t4_seed4001_study` | 4001 | included |
| colab_t4 | `results\t4_phase4_training_proxy\t4_seed4002_study` | 4002 | included |
| colab_t4 | `results\t4_phase4_training_proxy\t4_seed4003_study` | 4003 | included |
| colab_t4 | `results\t4_phase4_training_proxy\t4_seed4004_study` | 4004 | included |
| colab_t4 | `results\t4_phase4_training_proxy\t4_seed4005_study` | 4005 | included |

All included runs pass metadata, 24-configuration grid, 72 summary rows, all configuration/variant/check validation flags, and empty failed-check checks. CUDA device strings `cuda:0` are accepted as CUDA devices, not treated as failures of literal equality to `cuda`. Each run has 5,760 raw samples (40 repeats × 24 configurations × three variants × two scopes). Saved winners and margins were independently recomputed from summary medians, counting each configuration once per scope, not once per variant. Raw CVs match saved summary CVs. Input file hashes are recorded in `phase4_run_audit.json`.

## Aggregate winners

| Environment | Scope | index_add | scatter_add | Embedding | Tie | Total |
|---|---|---:|---:|---:|---:|---:|
| local_gtx1650ti | proxy_step | 83 | 0 | 10 | 27 | 120 |
| local_gtx1650ti | accumulation_region | 98 | 0 | 10 | 12 | 120 |
| colab_t4 | proxy_step | 0 | 25 | 0 | 95 | 120 |
| colab_t4 | accumulation_region | 3 | 39 | 0 | 78 | 120 |

Match to the supplied expected counts: **True**. This comparison is performed after aggregation; the expected counts are not used to populate CSVs or figure data.

## Environments and validation scales

- Local GTX1650Ti: NVIDIA GeForce GTX 1650 Ti, PyTorch 2.13.0+cu126, CUDA 12.6, Windows-10-10.0.19045-SP0, 4 CPU threads. Maximum table absolute error 5.826e-08; maximum any-check absolute error 5.826e-08; maximum table relative L2 error 3.524e-06.
- Colab T4: Tesla T4, PyTorch 2.11.0+cu130, CUDA 13.0, Linux-6.6.122+-x86_64-with-glibc2.39, 1 CPU threads. Maximum table absolute error 5.64e-08; maximum any-check absolute error 5.918e-08; maximum table relative L2 error 3.693e-06.

Benchmark script hashes across environments: 1 unique. Matched seed/configuration index hashes agree across environments: True. Validation thresholds remain atol=1e-8 and rtol=1e-4. Error scales are diagnostics, not a rationale for winner-dependent tolerance changes.

## Noise and stability diagnostics

Raw data are used only for these diagnostics and sample auditing; main winner counts use summary medians. No samples or valid configurations are removed for noise. CV > 0.10 and half-run median drift > 10% are descriptive flags, not significance tests or exclusion rules.

| Environment | Scope | Median raw CV | Max raw CV | CV > 0.10 (of 360) | >10% half-run drift (of 360) | Unchanged winner across five seeds (of 24) |
|---|---|---:|---:|---:|---:|---:|
| local_gtx1650ti | proxy_step | 0.082 | 4.217 | 150 | 88 | 21 |
| local_gtx1650ti | accumulation_region | 0.010 | 0.272 | 49 | 36 | 22 |
| colab_t4 | proxy_step | 0.069 | 0.665 | 121 | 19 | 22 |
| colab_t4 | accumulation_region | 0.060 | 1.669 | 97 | 17 | 15 |

Stability includes configurations consistently classified as ties. Repeated timings within a draw are not independent workloads; 120 observations are 24 shapes/distributions × five seeds, not 120 distinct shapes. A >5% point-estimate winner is not automatically robust to noise. Inspect `phase4_noise_diagnostics.csv` and per-configuration seed counts before selecting illustrative examples.

## Interpretation and warnings

The local stack most often favors the index_add custom backward path. T4 is dominated by practical ties: 95/120 (79.2%) in the total proxy and 78/120 (65.0%) in the diagnostic replay; its resolved cases mainly favor scatter_add. No universal fastest path follows. PyTorch, CUDA, OS, host thread settings and GPU differ jointly, so hardware causality is not isolated. The region replay is not an exact decomposition and must not be subtracted from total-step time. These results do not measure production training or convergence, optimizer state, data loading, distributed execution or production-scale throughput. Custom autograd nodes introduce dispatch/graph overhead, making the comparison API/path-level rather than kernel-only.

The observed pattern establishes context dependence, not a direct matched Phase 2 replication: no Phase 2 files were joined in this aggregation. No paired bootstrap, confidence intervals, cache flushing, or new timing runs were added.

## Definitions

Aggregate CSV shares are fractions. `num_configurations` counts seed–configuration observations. Per-configuration rows summarize five seed medians with equal seed weight. `median_fastest_variant` is the minimum of the across-seed latency medians (exact equality yields tie), not a 5% winner or modal winner. `median_practical_margin` is the median of each seed's runner-up/fastest ratio minus one. CV columns are medians of saved per-seed CVs. Seed-level winner counts retain the strict >5% classification.

## Files and insertion point

New paper artifacts are stored under the root-level `paper_sparse_gradient/` directory alongside the existing manuscript assets. File paths listed below are relative to the project root. Recommended insertion: after the Phase 3 attribution subsection and before Result-Level Discussion in the actual `paper_sparse_gradient/sections/05_results_analysis.tex`. Its discussion section is currently `06_general_discussion.tex`; do not assume the alternative numbering from the task. No insertion has been performed. Tables require booktabs/tabularx and the figure requires graphicx, already used by the existing manuscript. The standalone subsection is not included by the main manuscript yet.

- `paper_sparse_gradient\figures\phase4_winner_counts_by_environment.pdf`
- `paper_sparse_gradient\figures\phase4_winner_counts_by_environment.png`
- `paper_sparse_gradient\sections\phase4_training_proxy_text.tex`
- `paper_sparse_gradient\tables\phase4_environment_comparison.tex`
- `paper_sparse_gradient\tables\phase4_winner_counts.tex`
- `sparse-gradient\aggregate_phase4.py`
- `sparse-gradient\results\phase4_training_proxy_aggregate\phase4_aggregate_report.md`
- `sparse-gradient\results\phase4_training_proxy_aggregate\phase4_aggregate_summary.csv`
- `sparse-gradient\results\phase4_training_proxy_aggregate\phase4_noise_diagnostics.csv`
- `sparse-gradient\results\phase4_training_proxy_aggregate\phase4_per_configuration_summary.csv`
- `sparse-gradient\results\phase4_training_proxy_aggregate\phase4_run_audit.json`

Reproduce aggregation only (requires matplotlib; no torch import): `python aggregate_phase4.py` from `sparse-gradient/`. The current workspace Python lacks matplotlib; generation used the existing `C:/Projects/phd-segmentacija-instanci/.venv/Scripts/python.exe` plotting environment without installing packages.
