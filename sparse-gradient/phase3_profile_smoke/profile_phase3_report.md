# Phase 3: Representative Sparse-Index Gradient Profiles

## Environment

| gpu_name | compute_capability | pytorch_version | cuda_version | python_version | platform |
| --- | --- | --- | --- | --- | --- |
| NVIDIA GeForce GTX 1650 Ti | [7, 5] | 2.13.0+cu126 | 12.6 | 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)] | Windows-10-10.0.19045-SP0 |

## Cases

| case_id | rows | indices | dim | distribution | workload_draw | zipf_mode | row_layout | index_order | cache_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case01 | 4096 | 65536 | 64 | zipf | 0 | truncated | scattered | random | warm |

## Unprofiled CUDA-event timing

| case_id | winner | runner_up | margin_ratio | decisive_5pct | complete_ops | eligible_operations | index_add_ms | scatter_add_ms | embedding_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case01 | index_add | embedding | 1.3677 | True | True | 3 | 0.33728 | 1.0562 | 0.46131 |

Timings use fresh dense outputs for index_add/scatter_add and retained-graph backward for embedding. Embedding forward is outside timing; grad=None clearing is in prepare, matching v3. Samples use inner=1 and fixed operation order, so these subset decisions are not a replacement for randomized/batched Phase 2 confirmation.

## Top CUDA attribution

| case_id | operation | rank | CUDA key | total_ms | attribution |
| --- | --- | --- | --- | --- | --- |
| case01 | index_add | 1 | _ZN2at6native19indexFuncLargeIndexIfxjLi2ELi2ELin2ELb1ENS0_44_GLOBAL__N__99c14716_11_Indexing_cu_2a2c434e9ReduceAddEEEvNS_4cuda6detail10TensorInfoIT_T1_EENS6_IKS7_S8_EENS6_IKT0_S8_EEiiS8_S8_xxRKT6_S7_ | 0.32686 | cuda_kernel_events |
| case01 | index_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.005184 | cuda_kernel_events |
| case01 | scatter_add | 1 | _ZN2at6native29vectorized_scatter_add_kernelILi16EfxEEvPT0_PKS2_PKT1_ixxxxii | 1.0463 | cuda_kernel_events |
| case01 | scatter_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.005024 | cuda_kernel_events |
| case01 | embedding | 1 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261819compute_grad_weightIfxEEvPKT0_PKT_S5_xxS5_PKxPNS_14AccumulateTypeIS6_Lb1EE4typeEx | 0.14506 | cuda_kernel_events |
| case01 | embedding | 2 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261815sum_and_scatterIfxEEvPKT0_PT_xS5_PKxPKNS_14AccumulateTypeIS6_Lb1EE4typeES5_S9_xx | 0.13376 | cuda_kernel_events |
| case01 | embedding | 3 | _ZN14at_cuda_detail3cub29DeviceRadixSortOnesweepKernelINS0_21DeviceRadixSortPolicyIxN2at4cuda3cub6detail10OpaqueTypeILi8EEEyE9Policy900ELb0ExS8_yiiNS0_6detail21identity_decomposer_tEEEvPT5_SE_PT3_PKSF_PT1_PKSJ_PT2_PKSN_T4_iiT6_ | 0.05216 | cuda_kernel_events |

Times in this table total all 1 profiled invocations. Full CPU/operator/kernel key averages and top-10 tables are under events/. CUDA kernel keys are preferred; if unavailable, attributed CPU operators are used, or an explicitly labeled CPU-time fallback. Inclusive parent/child times must not be added. Aggregate CPU totals sum self times; CUDA totals sum kernel durations when available, not wall time.

## Observations

- case01: raw event-timing minimum `index_add`, runner-up `embedding`, ratio 1.37; decisive at 5%: True. Complete selected API set: True.
- `embedding`: most frequent highest-time CUDA key in this subset was `_ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261819compute_grad_weightIfxEEvPKT0_PKT_S5_xxS5_PKxPNS_14AccumulateTypeIS6_Lb1EE4typeEx` (1 cases).
- `index_add`: most frequent highest-time CUDA key in this subset was `_ZN2at6native19indexFuncLargeIndexIfxjLi2ELi2ELin2ELb1ENS0_44_GLOBAL__N__99c14716_11_Indexing_cu_2a2c434e9ReduceAddEEEvNS_4cuda6detail10TensorInfoIT_T1_EENS6_IKS7_S8_EENS6_IKT0_S8_EEiiS8_S8_xxRKT6_S7_` (1 cases).
- `scatter_add`: most frequent highest-time CUDA key in this subset was `_ZN2at6native29vectorized_scatter_add_kernelILi16EfxEEvPT0_PKS2_PKT1_ixxxxii` (1 cases).

## Warnings and failures

No data available.

## Limitations

Profiler explains operator/kernel composition, not universal causality. Results are per environment. CPU and CUDA profiler times include instrumentation and synchronization overhead and must not replace unprofiled CUDA-event timings. This is a representative subset, not a full grid or end-to-end model training. All profiled invocations synchronize. Warm cache only; record-shapes, memory, and stack capture can perturb execution. CUDA kernel attribution depends on CUPTI/Kineto support and permissions. Memory freed from allocations made before profiler start may not be attributable. Negative memory deltas are possible. There is no custom CUDA/C++ code.

For Colab, copy both profile_phase3.py and sparse_grad_benchmark.py into the same directory. Reuse the same cases and parameters. Optional --export-trace creates traces/; --record-shapes and --with-stack are off by default. CSV setting columns override CLI defaults; explicit size/draw flags expand templates and duplicates are removed before --max-cases is applied.
