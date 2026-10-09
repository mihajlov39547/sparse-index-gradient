# Phase 3 Comparison: Profiling Evidence for API-Dependent CUDA Paths

## 1. Purpose

Phase 2 supplies the main matched, margin-aware winner-count evidence. Phase 3 provides representative profiler attribution and within-subset event timing, not a replacement for randomized/batched confirmation. This is offline analysis only.

## 2. Inputs and environments

| environment_label | gpu_name | compute_capability | pytorch_version | cuda_version | python_version | platform | source_dir | status_summary | case_ops | failures | correctness_failures | cuda_kernel_events_available_count | likely_host_bound_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GTX1650Ti | NVIDIA GeForce GTX 1650 Ti | [7, 5] | 2.13.0+cu126 | 12.6 | 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)] | Windows-10-10.0.19045-SP0 | C:\Projects\phd-segmentacija-instanci\sparse-gradient\phase3_profile_local_v2 | {"ok": 18} | 18 | 0 | 0 | 18 | 0 |
| T4 | Tesla T4 | [7, 5] | 2.11.0+cu130 | 13.0 | 3.13.15 (main, Aug  6 2026, 11:06:22) [GCC 13.3.0] | Linux-6.6.122+-x86_64-with-glibc2.39 | C:\Projects\phd-segmentacija-instanci\sparse-gradient\phase3_profile_t4_v2 | {"ok": 18} | 18 | 0 | 0 | 18 | 1 |

- [GTX1650Ti Phase 3 report](../phase3_profile_local_v2/profile_phase3_report.md); [profile cases](../phase3_profile_local_v2/profile_cases.csv); [events](../phase3_profile_local_v2/events).
- [T4 Phase 3 report](../phase3_profile_t4_v2/profile_phase3_report.md); [profile cases](../phase3_profile_t4_v2/profile_cases.csv); [events](../phase3_profile_t4_v2/events).
- [Phase 2 matched analysis](../phase2_analysis_v2/phase2_analysis_report.md); [matched all-operation ratios](../phase2_analysis_v2/matched_all_op_speedup_summary.csv).

## 3. Case-level timing comparison

| case_id | rows | indices | dim | distribution | local_winner | local_margin_ratio | t4_winner | t4_margin_ratio | winner_relation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case01 | 4096 | 65536 | 64 | zipf | index_add | 1.67 | scatter_add | 1.022 | reversal_with_near_tie |
| case02 | 4096 | 262144 | 512 | uniform | embedding | 2.631 | embedding | 1.706 | same_decisive |
| case03 | 16384 | 65536 | 256 | zipf | embedding | 1.116 | scatter_add | 1.186 | reversal_decisive |
| case04 | 4096 | 65536 | 64 | hotspot | index_add | 1.197 | index_add | 1.058 | same_decisive |
| case05 | 65536 | 262144 | 256 | zipf | embedding | 1.257 | scatter_add | 1.285 | reversal_decisive |
| case06 | 4096 | 262144 | 64 | hotspot | embedding | 1.015 | scatter_add | 1.04 | reversal_with_near_tie |

The full [case table](phase3_case_comparison.csv) includes workload draw, layout/order/cache/Zipf settings, runner-ups, completeness, and all event medians. A decisive subset selection means runner-up/fastest >1.05. Near-ties retain their raw minimum label but are weak evidence. Failed/inconsistent operation sets are incomplete.

## 4. Winner reversals and controls

- **case01** (reversal_with_near_tie): The environments differ, but at least one side is near-tie; treat as weak evidence.
- **case02** (same_decisive): Both environments select embedding decisively for this representative case.
- **case03** (reversal_decisive): T4 selects scatter_add decisively while local selects embedding decisively; this is a representative reversal.
- **case04** (same_decisive): Both environments select index_add decisively for this representative case.
- **case05** (reversal_decisive): T4 selects scatter_add decisively while local selects embedding decisively; this is a representative reversal.
- **case06** (reversal_with_near_tie): The environments differ, but at least one side is near-tie; treat as weak evidence.

These cases are representative probes, not enough observations to statistically establish a grid-wide selection frequency. See [transition counts](phase3_winner_transition_matrix.csv).

## 5. GPU-busy proxy and host-bound warnings

| environment_label | min | median | max |
| --- | --- | --- | --- |
| GTX1650Ti | 0.6887 | 0.9966 | 1.033 |
| T4 | 0.4357 | 0.9102 | 0.9856 |

| environment_label | likely_host_bound_count |
| --- | --- |
| GTX1650Ti | 0 |
| T4 | 1 |

| environment_label | case_id | operation | gpu_busy_fraction |
| --- | --- | --- | --- |
| T4 | case04 | embedding | 0.4357 |

gpu_busy_fraction divides profiled CUDA kernel time per invocation by an unprofiled event median. It is a proxy, not utilization, can exceed 1, and can be distorted by instrumentation or changed timing conditions. Values below 0.6 flag possible host overhead only. Unflagged cases do not prove hardware saturation; flagged cases warrant caution during path attribution.

## 6. CUDA path attribution

| environment_label | operation | kernel_pattern_short |
| --- | --- | --- |
| GTX1650Ti | embedding | compute_grad_weight; sum_and_scatter; DeviceRadixSort; compute_grad_weight; sum_and_scatter; FillFunctor |
| GTX1650Ti | index_add | indexFuncLargeIndex/ReduceAdd; FillFunctor |
| GTX1650Ti | scatter_add | scatter-gather ReduceAdd; FillFunctor |
| T4 | embedding | compute_grad_weight; DeviceRadixSort; sum_and_scatter; compute_grad_weight; sum_and_scatter; DeviceRadixSort; sum_and_scatter; DeviceRadixSort; compute_grad_weight; sum_and_scatter; compute_grad_weight; DeviceRadixSort |
| T4 | index_add | indexFuncLargeIndex/ReduceAdd; FillFunctor |
| T4 | scatter_add | scatter-gather ReduceAdd; FillFunctor |

[Kernel summary](phase3_kernel_summary.csv) preserves exact top keys and profile-total times. [Category shares](phase3_kernel_category_summary.csv) apply ordered, first-match name rules to CUDA device rows only, excluding user annotations and phase3 ranges. indexFuncLargeIndex is normalized to indexFuncLargeIndex/ReduceAdd; scatter-gather/scatter_add paths to scatter-gather ReduceAdd. Embedding keys distinguish compute_grad_weight, sum_and_scatter, radix sort, unique and scan components. Names describe observed paths; they do not alone prove why an API is faster.

Full-event category shares sum to 1 per environment/case/operation. Fallback top-10 data use the separate top10_category_share column; full-profile share remains blank. kernel_count sums recorded calls, not just distinct names. Source directories above provide raw event evidence.

## 7. Connection to Phase 2

| evidence_type | environment_label | main_observation | supporting_numbers | caution |
| --- | --- | --- | --- | --- |
| Phase 2 matched margin-aware | GTX1650Ti | Most frequent decisive API(s): embedding. | index_add=60/162; scatter_add=0/162; embedding=96/162; tie=6/162; geometric mean ratios: embedding=1.117x, index_add=1.000x, scatter_add=0.628x | Main grid-level evidence; 5% practical margin is not statistical significance. Ratios include wins and losses. |
| Phase 2 matched margin-aware | T4 | Most frequent decisive API(s): scatter_add. | index_add=5/162; scatter_add=92/162; embedding=25/162; tie=40/162; geometric mean ratios: embedding=0.817x, index_add=1.000x, scatter_add=1.195x | Main grid-level evidence; 5% practical margin is not statistical significance. Ratios include wins and losses. |
| Phase 2 matched margin-aware | T4_repeat | Most frequent decisive API(s): scatter_add. | index_add=1/162; scatter_add=93/162; embedding=27/162; tie=41/162; geometric mean ratios: embedding=0.801x, index_add=1.000x, scatter_add=1.203x | Main grid-level evidence; 5% practical margin is not statistical significance. Ratios include wins and losses. |
| Phase 3 representative subset | GTX1650Ti | Within-subset event-timing minima with 5% margin caveats. | case01=index_add (decisive); case02=embedding (decisive); case03=embedding (decisive); case04=index_add (decisive); case05=embedding (decisive); case06=embedding (near-tie) | Representative probes, not a statistical replication of Phase 2 winner frequencies. |
| Phase 3 representative subset | T4 | Within-subset event-timing minima with 5% margin caveats. | case01=scatter_add (near-tie); case02=embedding (decisive); case03=scatter_add (decisive); case04=index_add (decisive); case05=scatter_add (decisive); case06=scatter_add (near-tie) | Representative probes, not a statistical replication of Phase 2 winner frequencies. |
| Phase 3 paired case | GTX1650Ti vs T4 | The environments differ, but at least one side is near-tie; treat as weak evidence. | case01: index_add vs scatter_add; margins 1.67, 1.022 | Profiler paths inform attribution; no isolated hardware-causal inference. |
| Phase 3 paired case | GTX1650Ti vs T4 | Both environments select embedding decisively for this representative case. | case02: embedding vs embedding; margins 2.631, 1.706 | Profiler paths inform attribution; no isolated hardware-causal inference. |
| Phase 3 paired case | GTX1650Ti vs T4 | T4 selects scatter_add decisively while local selects embedding decisively; this is a representative reversal. | case03: embedding vs scatter_add; margins 1.116, 1.186 | Profiler paths inform attribution; no isolated hardware-causal inference. |
| Phase 3 paired case | GTX1650Ti vs T4 | Both environments select index_add decisively for this representative case. | case04: index_add vs index_add; margins 1.197, 1.058 | Profiler paths inform attribution; no isolated hardware-causal inference. |
| Phase 3 paired case | GTX1650Ti vs T4 | T4 selects scatter_add decisively while local selects embedding decisively; this is a representative reversal. | case05: embedding vs scatter_add; margins 1.257, 1.285 | Profiler paths inform attribution; no isolated hardware-causal inference. |
| Phase 3 paired case | GTX1650Ti vs T4 | The environments differ, but at least one side is near-tie; treat as weak evidence. | case06: embedding vs scatter_add; margins 1.015, 1.04 | Profiler paths inform attribution; no isolated hardware-causal inference. |

Phase 2 characterizes matched grid-level regimes with practical margins; Phase 3 examines selected CUDA paths. Together they provide evidence that standard APIs for equivalent sparse-index accumulation can map to different execution paths whose relative costs depend on workload and the combined hardware/software stack. No isolated hardware cause is inferred.

## 8. Paper-ready claims

Phase 3 profiling confirms that the three PyTorch APIs exercise distinct CUDA execution paths. `index_add` is dominated by an index-based reduction kernel, `scatter_add` by a scatter-gather reduction kernel, while `Embedding.backward` invokes a multi-kernel embedding backward path including accumulation and sorting/segmentation-related kernels. In representative cases, the local GTX1650Ti stack and the Colab T4 stack select different fastest APIs, consistent with the matched Phase 2 grid-level results. These observations support an environment- and workload-dependent characterization, not a universal API ranking or isolated hardware-causal claim.

## 9. Limitations

- Phase 3 is a representative subset, not a full grid or statistical replication of Phase 2.
- Event timings use inner=1 and fixed operation order; Phase 2 remains the main randomized/batched winner-count evidence.
- Profiler overhead is not used as primary benchmark timing. Profiler CUDA totals and event medians are not interchangeable.
- GPU-busy fraction is a proxy, not hardware utilization; it can exceed 1. likely_host_bound is a warning signal, not causal proof.
- No Nsight Compute counters, end-to-end model training, or custom CUDA implementation.
- Absolute GTX/T4 latencies are not controlled hardware benchmarks; GPU, OS, PyTorch, CUDA, and hosted runtime differ together.
- Ordered name-based categories are approximate. Full shares exclude annotations; top-10 fallback shares describe only that truncated list.
- Casewise means weight each available case equally, not by kernel duration. Shares summarize kernel work, not profiler wall-clock utilization.
- CPU/NumPy index hash discrepancies are audit warnings; source RNG values can differ across software stacks even with equal seeds.


### Input and output audit

- case01: src_sha256 mismatch (audit warning only; source RNG may differ across stacks).
- case02: src_sha256 mismatch (audit warning only; source RNG may differ across stacks).
- case03: src_sha256 mismatch (audit warning only; source RNG may differ across stacks).
- case04: src_sha256 mismatch (audit warning only; source RNG may differ across stacks).
- case05: src_sha256 mismatch (audit warning only; source RNG may differ across stacks).
- case06: src_sha256 mismatch (audit warning only; source RNG may differ across stacks).

## 10. Generated artifacts

- [phase3_environment_summary.csv](phase3_environment_summary.csv)
- [phase3_case_comparison.csv](phase3_case_comparison.csv)
- [phase3_operation_comparison.csv](phase3_operation_comparison.csv)
- [phase3_kernel_summary.csv](phase3_kernel_summary.csv)
- [phase3_kernel_category_summary.csv](phase3_kernel_category_summary.csv)
- [phase3_winner_transition_matrix.csv](phase3_winner_transition_matrix.csv)
- [phase3_phase2_alignment.csv](phase3_phase2_alignment.csv)

- [phase3_case_winner_comparison.png](phase3_case_winner_comparison.png)
- [phase3_median_ms_by_case_operation.png](phase3_median_ms_by_case_operation.png)
- [phase3_gpu_busy_fraction_by_case_operation.png](phase3_gpu_busy_fraction_by_case_operation.png)
- [phase3_kernel_category_share.png](phase3_kernel_category_share.png)

- [Limitations](phase3_limitations.md)
