# Phase 3: Representative Sparse-Index Gradient Profiles

## Environment

| gpu_name | compute_capability | pytorch_version | cuda_version | python_version | platform |
| --- | --- | --- | --- | --- | --- |
| NVIDIA GeForce GTX 1650 Ti | [7, 5] | 2.13.0+cu126 | 12.6 | 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)] | Windows-10-10.0.19045-SP0 |

## Cases

| case_id | rows | indices | dim | distribution | workload_draw | zipf_mode | row_layout | index_order | cache_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case01 | 4096 | 65536 | 64 | zipf | 0 | truncated | scattered | random | warm |
| case02 | 4096 | 262144 | 512 | uniform | 0 | truncated | scattered | random | warm |
| case03 | 16384 | 65536 | 256 | zipf | 0 | truncated | scattered | random | warm |
| case04 | 4096 | 65536 | 64 | hotspot | 0 | truncated | scattered | random | warm |
| case05 | 65536 | 262144 | 256 | zipf | 0 | truncated | scattered | random | warm |
| case06 | 4096 | 262144 | 64 | hotspot | 0 | truncated | scattered | random | warm |

## Unprofiled CUDA-event timing

| case_id | winner | runner_up | margin_ratio | decisive_5pct | complete_ops | eligible_operations | index_add_ms | scatter_add_ms | embedding_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case01 | index_add | embedding | 1.4135 | True | True | 3 | 0.33637 | 1.056 | 0.47544 |
| case02 | embedding | scatter_add | 2.6338 | True | True | 3 | 11.728 | 11.637 | 4.4184 |
| case03 | embedding | index_add | 1.1221 | True | True | 3 | 1.1144 | 1.7305 | 0.99312 |
| case04 | index_add | embedding | 1.1323 | True | True | 3 | 0.30392 | 1.128 | 0.34413 |
| case05 | embedding | index_add | 1.2581 | True | True | 3 | 4.7551 | 7.955 | 3.7796 |
| case06 | embedding | index_add | 1.0103 | False | True | 3 | 0.98531 | 3.3848 | 0.97523 |

Timings use fresh dense outputs for index_add/scatter_add and retained-graph backward for embedding. Embedding forward is outside timing; grad=None clearing is in prepare, matching v3. Samples use inner=1 and fixed operation order, so these subset decisions are not a replacement for randomized/batched Phase 2 confirmation.

## Top CUDA attribution

| case_id | operation | rank | CUDA key | total_ms | attribution |
| --- | --- | --- | --- | --- | --- |
| case01 | index_add | 1 | _ZN2at6native19indexFuncLargeIndexIfxjLi2ELi2ELin2ELb1ENS0_44_GLOBAL__N__99c14716_11_Indexing_cu_2a2c434e9ReduceAddEEEvNS_4cuda6detail10TensorInfoIT_T1_EENS6_IKS7_S8_EENS6_IKT0_S8_EEiiS8_S8_xxRKT6_S7_ | 0.9821 | cuda_kernel_events |
| case01 | index_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.015778 | cuda_kernel_events |
| case01 | scatter_add | 1 | _ZN2at6native29vectorized_scatter_add_kernelILi16EfxEEvPT0_PKS2_PKT1_ixxxxii | 3.1373 | cuda_kernel_events |
| case01 | scatter_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.015904 | cuda_kernel_events |
| case01 | embedding | 1 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261819compute_grad_weightIfxEEvPKT0_PKT_S5_xxS5_PKxPNS_14AccumulateTypeIS6_Lb1EE4typeEx | 0.43341 | cuda_kernel_events |
| case01 | embedding | 2 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261815sum_and_scatterIfxEEvPKT0_PT_xS5_PKxPKNS_14AccumulateTypeIS6_Lb1EE4typeES5_S9_xx | 0.39178 | cuda_kernel_events |
| case01 | embedding | 3 | _ZN14at_cuda_detail3cub29DeviceRadixSortOnesweepKernelINS0_21DeviceRadixSortPolicyIxN2at4cuda3cub6detail10OpaqueTypeILi8EEEyE9Policy900ELb0ExS8_yiiNS0_6detail21identity_decomposer_tEEEvPT5_SE_PT3_PKSF_PT1_PKSJ_PT2_PKSN_T4_iiT6_ | 0.15168 | cuda_kernel_events |
| case02 | index_add | 1 | _ZN2at6native19indexFuncLargeIndexIfxjLi2ELi2ELin2ELb1ENS0_44_GLOBAL__N__99c14716_11_Indexing_cu_2a2c434e9ReduceAddEEEvNS_4cuda6detail10TensorInfoIT_T1_EENS6_IKS7_S8_EENS6_IKT0_S8_EEiiS8_S8_xxRKT6_S7_ | 32.113 | cuda_kernel_events |
| case02 | index_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.13347 | cuda_kernel_events |
| case02 | scatter_add | 1 | _ZN2at6native29vectorized_scatter_add_kernelILi16EfxEEvPT0_PKS2_PKT1_ixxxxii | 34.856 | cuda_kernel_events |
| case02 | scatter_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.13501 | cuda_kernel_events |
| case02 | embedding | 1 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261819compute_grad_weightIfxEEvPKT0_PKT_S5_xxS5_PKxPNS_14AccumulateTypeIS6_Lb1EE4typeEx | 11.298 | cuda_kernel_events |
| case02 | embedding | 2 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261815sum_and_scatterIfxEEvPKT0_PT_xS5_PKxPKNS_14AccumulateTypeIS6_Lb1EE4typeES5_S9_xx | 1.2407 | cuda_kernel_events |
| case02 | embedding | 3 | _ZN14at_cuda_detail3cub29DeviceRadixSortOnesweepKernelINS0_21DeviceRadixSortPolicyIxN2at4cuda3cub6detail10OpaqueTypeILi8EEEyE9Policy900ELb0ExS8_yiiNS0_6detail21identity_decomposer_tEEEvPT5_SE_PT3_PKSF_PT1_PKSJ_PT2_PKSN_T4_iiT6_ | 0.38324 | cuda_kernel_events |
| case03 | index_add | 1 | _ZN2at6native19indexFuncLargeIndexIfxjLi2ELi2ELin2ELb1ENS0_44_GLOBAL__N__99c14716_11_Indexing_cu_2a2c434e9ReduceAddEEEvNS_4cuda6detail10TensorInfoIT_T1_EENS6_IKS7_S8_EENS6_IKT0_S8_EEiiS8_S8_xxRKT6_S7_ | 3.1338 | cuda_kernel_events |
| case03 | index_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.2681 | cuda_kernel_events |
| case03 | scatter_add | 1 | _ZN2at6native29vectorized_scatter_add_kernelILi16EfxEEvPT0_PKS2_PKT1_ixxxxii | 4.8989 | cuda_kernel_events |
| case03 | scatter_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.26609 | cuda_kernel_events |
| case03 | embedding | 1 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261819compute_grad_weightIfxEEvPKT0_PKT_S5_xxS5_PKxPNS_14AccumulateTypeIS6_Lb1EE4typeEx | 1.8198 | cuda_kernel_events |
| case03 | embedding | 2 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261815sum_and_scatterIfxEEvPKT0_PT_xS5_PKxPKNS_14AccumulateTypeIS6_Lb1EE4typeES5_S9_xx | 0.67751 | cuda_kernel_events |
| case03 | embedding | 3 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.26877 | cuda_kernel_events |
| case04 | index_add | 1 | _ZN2at6native19indexFuncLargeIndexIfxjLi2ELi2ELin2ELb1ENS0_44_GLOBAL__N__99c14716_11_Indexing_cu_2a2c434e9ReduceAddEEEvNS_4cuda6detail10TensorInfoIT_T1_EENS6_IKS7_S8_EENS6_IKT0_S8_EEiiS8_S8_xxRKT6_S7_ | 0.88858 | cuda_kernel_events |
| case04 | index_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.011936 | cuda_kernel_events |
| case04 | scatter_add | 1 | _ZN2at6native29vectorized_scatter_add_kernelILi16EfxEEvPT0_PKS2_PKT1_ixxxxii | 3.3613 | cuda_kernel_events |
| case04 | scatter_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.01232 | cuda_kernel_events |
| case04 | embedding | 1 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261819compute_grad_weightIfxEEvPKT0_PKT_S5_xxS5_PKxPNS_14AccumulateTypeIS6_Lb1EE4typeEx | 0.40468 | cuda_kernel_events |
| case04 | embedding | 2 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261815sum_and_scatterIfxEEvPKT0_PT_xS5_PKxPKNS_14AccumulateTypeIS6_Lb1EE4typeES5_S9_xx | 0.23287 | cuda_kernel_events |
| case04 | embedding | 3 | _ZN14at_cuda_detail3cub29DeviceRadixSortOnesweepKernelINS0_21DeviceRadixSortPolicyIxN2at4cuda3cub6detail10OpaqueTypeILi8EEEyE9Policy900ELb0ExS8_yiiNS0_6detail21identity_decomposer_tEEEvPT5_SE_PT3_PKSF_PT1_PKSJ_PT2_PKSN_T4_iiT6_ | 0.11536 | cuda_kernel_events |
| case05 | index_add | 1 | _ZN2at6native19indexFuncLargeIndexIfxjLi2ELi2ELin2ELb1ENS0_44_GLOBAL__N__99c14716_11_Indexing_cu_2a2c434e9ReduceAddEEEvNS_4cuda6detail10TensorInfoIT_T1_EENS6_IKS7_S8_EENS6_IKT0_S8_EEiiS8_S8_xxRKT6_S7_ | 13.282 | cuda_kernel_events |
| case05 | index_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 1.0678 | cuda_kernel_events |
| case05 | scatter_add | 1 | _ZN2at6native29vectorized_scatter_add_kernelILi16EfxEEvPT0_PKS2_PKT1_ixxxxii | 22.782 | cuda_kernel_events |
| case05 | scatter_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 1.067 | cuda_kernel_events |
| case05 | embedding | 1 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261819compute_grad_weightIfxEEvPKT0_PKT_S5_xxS5_PKxPNS_14AccumulateTypeIS6_Lb1EE4typeEx | 7.2432 | cuda_kernel_events |
| case05 | embedding | 2 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261815sum_and_scatterIfxEEvPKT0_PT_xS5_PKxPKNS_14AccumulateTypeIS6_Lb1EE4typeES5_S9_xx | 2.4914 | cuda_kernel_events |
| case05 | embedding | 3 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 1.0711 | cuda_kernel_events |
| case06 | index_add | 1 | _ZN2at6native19indexFuncLargeIndexIfxjLi2ELi2ELin2ELb1ENS0_44_GLOBAL__N__99c14716_11_Indexing_cu_2a2c434e9ReduceAddEEEvNS_4cuda6detail10TensorInfoIT_T1_EENS6_IKS7_S8_EENS6_IKT0_S8_EEiiS8_S8_xxRKT6_S7_ | 3.0023 | cuda_kernel_events |
| case06 | index_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.012641 | cuda_kernel_events |
| case06 | scatter_add | 1 | _ZN2at6native29vectorized_scatter_add_kernelILi16EfxEEvPT0_PKS2_PKT1_ixxxxii | 10.131 | cuda_kernel_events |
| case06 | scatter_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.011872 | cuda_kernel_events |
| case06 | embedding | 1 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261819compute_grad_weightIfxEEvPKT0_PKT_S5_xxS5_PKxPNS_14AccumulateTypeIS6_Lb1EE4typeEx | 1.4378 | cuda_kernel_events |
| case06 | embedding | 2 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261815sum_and_scatterIfxEEvPKT0_PT_xS5_PKxPKNS_14AccumulateTypeIS6_Lb1EE4typeES5_S9_xx | 0.75089 | cuda_kernel_events |
| case06 | embedding | 3 | _ZN14at_cuda_detail3cub29DeviceRadixSortOnesweepKernelINS0_21DeviceRadixSortPolicyIxN2at4cuda3cub6detail10OpaqueTypeILi8EEEyE9Policy900ELb0ExS8_yiiNS0_6detail21identity_decomposer_tEEEvPT5_SE_PT3_PKSF_PT1_PKSJ_PT2_PKSN_T4_iiT6_ | 0.37114 | cuda_kernel_events |

Times in this table total all 3 profiled invocations. Full CPU/operator/kernel key averages and top-10 tables are under events/. CUDA kernel keys are preferred; if unavailable, attributed CPU operators are used, or an explicitly labeled CPU-time fallback. Inclusive parent/child times must not be added. Aggregate CPU totals sum self times; CUDA totals sum kernel durations when available, not wall time.

## Observations

- case01: raw event-timing minimum `index_add`, runner-up `embedding`, ratio 1.41; decisive at 5%: True. Complete selected API set: True.
- case02: raw event-timing minimum `embedding`, runner-up `scatter_add`, ratio 2.63; decisive at 5%: True. Complete selected API set: True.
- case03: raw event-timing minimum `embedding`, runner-up `index_add`, ratio 1.12; decisive at 5%: True. Complete selected API set: True.
- case04: raw event-timing minimum `index_add`, runner-up `embedding`, ratio 1.13; decisive at 5%: True. Complete selected API set: True.
- case05: raw event-timing minimum `embedding`, runner-up `index_add`, ratio 1.26; decisive at 5%: True. Complete selected API set: True.
- case06: raw event-timing minimum `embedding`, runner-up `index_add`, ratio 1.01; decisive at 5%: False. Complete selected API set: True.
- `embedding`: most frequent highest-time CUDA key in this subset was `_ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261819compute_grad_weightIfxEEvPKT0_PKT_S5_xxS5_PKxPNS_14AccumulateTypeIS6_Lb1EE4typeEx` (6 cases).
- `index_add`: most frequent highest-time CUDA key in this subset was `_ZN2at6native19indexFuncLargeIndexIfxjLi2ELi2ELin2ELb1ENS0_44_GLOBAL__N__99c14716_11_Indexing_cu_2a2c434e9ReduceAddEEEvNS_4cuda6detail10TensorInfoIT_T1_EENS6_IKS7_S8_EENS6_IKT0_S8_EEiiS8_S8_xxRKT6_S7_` (6 cases).
- `scatter_add`: most frequent highest-time CUDA key in this subset was `_ZN2at6native29vectorized_scatter_add_kernelILi16EfxEEvPT0_PKS2_PKT1_ixxxxii` (6 cases).

## Warnings and failures

No data available.

## Limitations

Profiler explains operator/kernel composition, not universal causality. Results are per environment. CPU and CUDA profiler times include instrumentation and synchronization overhead and must not replace unprofiled CUDA-event timings. This is a representative subset, not a full grid or end-to-end model training. All profiled invocations synchronize. Warm cache only; record-shapes, memory, and stack capture can perturb execution. CUDA kernel attribution depends on CUPTI/Kineto support and permissions. Memory freed from allocations made before profiler start may not be attributable. Negative memory deltas are possible. There is no custom CUDA/C++ code.

For Colab, copy both profile_phase3.py and sparse_grad_benchmark.py into the same directory. Reuse the same cases and parameters. Optional --export-trace creates traces/; --record-shapes and --with-stack are off by default. CSV setting columns override CLI defaults; explicit size/draw flags expand templates and duplicates are removed before --max-cases is applied.
