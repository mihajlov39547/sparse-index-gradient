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

## Workload audit

| case_id | hotspot_fraction | hotspot_rows | zipf_alpha | seed | index_sha256 | src_sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| case01 | 0.95 | 8 | 1.2 | 1234 | c565c62e163c47f8a5065b03df65f9f7ed1450fc533c521c1ea16fa215ab6e96 | 447037bea889b6c9a217efc1b6f67d92bbce4d3708b94c5a7b2a07a52607edca |
| case02 | 0.95 | 8 | 1.2 | 1234 | 2d33528ba0eb923018ab3dd5a1f95a6a527952f6cf2731fa6b0b0765c8eacc28 | 634c702b8b38cb730781c3ced478b34e2b15e4dfd168a3cfa6c0e607d16e7ab6 |
| case03 | 0.95 | 8 | 1.2 | 1234 | d6c4a9be6ceb9dc995b8165359fa929e07ecf49c41f561dacdffa5ece89b14ad | 70250d4fe404434c43317e6e00f4e8dbc18a36861b0644bb893ab11d91364450 |
| case04 | 0.95 | 8 | 1.2 | 1234 | dda855fc21f3754fb375e5d5149b5afebb21529a6ce8d35d89feffbb46a19ae8 | 3578fb237c905a45391eed640e7a23d03b604d641baedb25d652785b3f572980 |
| case05 | 0.95 | 8 | 1.2 | 1234 | 5c29cf8d6c8fbab570d92ed8b47ad4ee92752459d89e126fce64c6c94974a027 | b203ba91e89752db1882c6a0ca24b8356010958f49fc9599575de5b0d643e098 |
| case06 | 0.95 | 8 | 1.2 | 1234 | 553f70996d9976bac670390b0e6bb49a6fd661ab4377749d7b5dfc7704d90af8 | 541bbaaa7c64c6e454f45d05065739729663b410794966a59fdffc8e95e7d570 |

Hashes use contiguous CPU tensor bytes (int64 indices, float32 source), computed before timing/profiling. They check exact data reproducibility; source values may differ between PyTorch/CUDA versions even with the same seed. Workload shape and settings must also match.

## Unprofiled CUDA-event timing for profiled subset

| case_id | winner | runner_up | margin_ratio | decisive_5pct | complete_ops | eligible_operations | index_add_ms | scatter_add_ms | embedding_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case01 | index_add | embedding | 1.6697 | True | True | 3 | 0.33618 | 1.0564 | 0.56133 |
| case02 | embedding | scatter_add | 2.6309 | True | True | 3 | 11.915 | 11.616 | 4.4153 |
| case03 | embedding | index_add | 1.1163 | True | True | 3 | 1.1115 | 1.7249 | 0.99576 |
| case04 | index_add | embedding | 1.1969 | True | True | 3 | 0.30387 | 1.128 | 0.36371 |
| case05 | embedding | index_add | 1.2567 | True | True | 3 | 4.7291 | 7.9558 | 3.7632 |
| case06 | embedding | index_add | 1.0148 | False | True | 3 | 0.98613 | 3.3743 | 0.97171 |

Timings use fresh dense outputs for index_add/scatter_add and retained-graph backward for embedding. Embedding forward is outside timing; grad=None clearing is in prepare, matching v3. Samples use inner=1 and fixed operation order, so these subset decisions are not a replacement for randomized/batched Phase 2 confirmation.

## Profiler-derived GPU-busy proxy

| case_id | operation | kernel_ms_per_invocation | gpu_busy_fraction | likely_host_bound |
| --- | --- | --- | --- | --- |
| case01 | index_add | 0.33335 | 0.9916 | False |
| case01 | scatter_add | 1.0524 | 0.9962 | False |
| case01 | embedding | 0.38659 | 0.6887 | False |
| case02 | index_add | 11.889 | 0.99783 | False |
| case02 | scatter_add | 11.593 | 0.99803 | False |
| case02 | embedding | 4.4202 | 1.0011 | False |
| case03 | index_add | 1.107 | 0.99593 | False |
| case03 | scatter_add | 1.72 | 0.99718 | False |
| case03 | embedding | 1.0066 | 1.0108 | False |
| case04 | index_add | 0.30005 | 0.98742 | False |
| case04 | scatter_add | 1.1245 | 0.9969 | False |
| case04 | embedding | 0.2939 | 0.80805 | False |
| case05 | index_add | 4.6864 | 0.99097 | False |
| case05 | scatter_add | 7.9498 | 0.99925 | False |
| case05 | embedding | 3.887 | 1.0329 | False |
| case06 | index_add | 0.98063 | 0.99443 | False |
| case06 | scatter_add | 3.3702 | 0.9988 | False |
| case06 | embedding | 0.94536 | 0.97288 | False |

gpu_busy_fraction = (total profiled CUDA kernel time / profile_repeat) / unprofiled median_ms. This is a profiler-derived proxy, not a hardware utilization counter. It can exceed 1 or be distorted when profiler/kernel timing differs from unprofiled event timing. Use it only as a host-overhead warning signal: likely_host_bound flags values below 0.6, not a causal conclusion. Values are unavailable when CUDA kernel events are unavailable. The fraction is not clamped.

## Top CUDA attribution

| case_id | operation | rank | CUDA key | total_ms | attribution |
| --- | --- | --- | --- | --- | --- |
| case01 | index_add | 1 | _ZN2at6native19indexFuncLargeIndexIfxjLi2ELi2ELin2ELb1ENS0_44_GLOBAL__N__99c14716_11_Indexing_cu_2a2c434e9ReduceAddEEEvNS_4cuda6detail10TensorInfoIT_T1_EENS6_IKS7_S8_EENS6_IKT0_S8_EEiiS8_S8_xxRKT6_S7_ | 0.98386 | cuda_kernel_events |
| case01 | index_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.016192 | cuda_kernel_events |
| case01 | scatter_add | 1 | _ZN2at6native29vectorized_scatter_add_kernelILi16EfxEEvPT0_PKS2_PKT1_ixxxxii | 3.1416 | cuda_kernel_events |
| case01 | scatter_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.015711 | cuda_kernel_events |
| case01 | embedding | 1 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261819compute_grad_weightIfxEEvPKT0_PKT_S5_xxS5_PKxPNS_14AccumulateTypeIS6_Lb1EE4typeEx | 0.43655 | cuda_kernel_events |
| case01 | embedding | 2 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261815sum_and_scatterIfxEEvPKT0_PT_xS5_PKxPKNS_14AccumulateTypeIS6_Lb1EE4typeES5_S9_xx | 0.39207 | cuda_kernel_events |
| case01 | embedding | 3 | _ZN14at_cuda_detail3cub29DeviceRadixSortOnesweepKernelINS0_21DeviceRadixSortPolicyIxN2at4cuda3cub6detail10OpaqueTypeILi8EEEyE9Policy900ELb0ExS8_yiiNS0_6detail21identity_decomposer_tEEEvPT5_SE_PT3_PKSF_PT1_PKSJ_PT2_PKSN_T4_iiT6_ | 0.15165 | cuda_kernel_events |
| case02 | index_add | 1 | _ZN2at6native19indexFuncLargeIndexIfxjLi2ELi2ELin2ELb1ENS0_44_GLOBAL__N__99c14716_11_Indexing_cu_2a2c434e9ReduceAddEEEvNS_4cuda6detail10TensorInfoIT_T1_EENS6_IKS7_S8_EENS6_IKT0_S8_EEiiS8_S8_xxRKT6_S7_ | 35.518 | cuda_kernel_events |
| case02 | index_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.14957 | cuda_kernel_events |
| case02 | scatter_add | 1 | _ZN2at6native29vectorized_scatter_add_kernelILi16EfxEEvPT0_PKS2_PKT1_ixxxxii | 34.646 | cuda_kernel_events |
| case02 | scatter_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.13405 | cuda_kernel_events |
| case02 | embedding | 1 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261819compute_grad_weightIfxEEvPKT0_PKT_S5_xxS5_PKxPNS_14AccumulateTypeIS6_Lb1EE4typeEx | 11.294 | cuda_kernel_events |
| case02 | embedding | 2 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261815sum_and_scatterIfxEEvPKT0_PT_xS5_PKxPKNS_14AccumulateTypeIS6_Lb1EE4typeES5_S9_xx | 1.2439 | cuda_kernel_events |
| case02 | embedding | 3 | _ZN14at_cuda_detail3cub29DeviceRadixSortOnesweepKernelINS0_21DeviceRadixSortPolicyIxN2at4cuda3cub6detail10OpaqueTypeILi8EEEyE9Policy900ELb0ExS8_yiiNS0_6detail21identity_decomposer_tEEEvPT5_SE_PT3_PKSF_PT1_PKSJ_PT2_PKSN_T4_iiT6_ | 0.38551 | cuda_kernel_events |
| case03 | index_add | 1 | _ZN2at6native19indexFuncLargeIndexIfxjLi2ELi2ELin2ELb1ENS0_44_GLOBAL__N__99c14716_11_Indexing_cu_2a2c434e9ReduceAddEEEvNS_4cuda6detail10TensorInfoIT_T1_EENS6_IKS7_S8_EENS6_IKT0_S8_EEiiS8_S8_xxRKT6_S7_ | 3.055 | cuda_kernel_events |
| case03 | index_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.26602 | cuda_kernel_events |
| case03 | scatter_add | 1 | _ZN2at6native29vectorized_scatter_add_kernelILi16EfxEEvPT0_PKS2_PKT1_ixxxxii | 4.8941 | cuda_kernel_events |
| case03 | scatter_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.26589 | cuda_kernel_events |
| case03 | embedding | 1 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261819compute_grad_weightIfxEEvPKT0_PKT_S5_xxS5_PKxPNS_14AccumulateTypeIS6_Lb1EE4typeEx | 1.8245 | cuda_kernel_events |
| case03 | embedding | 2 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261815sum_and_scatterIfxEEvPKT0_PT_xS5_PKxPKNS_14AccumulateTypeIS6_Lb1EE4typeES5_S9_xx | 0.67828 | cuda_kernel_events |
| case03 | embedding | 3 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.27005 | cuda_kernel_events |
| case04 | index_add | 1 | _ZN2at6native19indexFuncLargeIndexIfxjLi2ELi2ELin2ELb1ENS0_44_GLOBAL__N__99c14716_11_Indexing_cu_2a2c434e9ReduceAddEEEvNS_4cuda6detail10TensorInfoIT_T1_EENS6_IKS7_S8_EENS6_IKT0_S8_EEiiS8_S8_xxRKT6_S7_ | 0.88834 | cuda_kernel_events |
| case04 | index_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.011807 | cuda_kernel_events |
| case04 | scatter_add | 1 | _ZN2at6native29vectorized_scatter_add_kernelILi16EfxEEvPT0_PKS2_PKT1_ixxxxii | 3.3612 | cuda_kernel_events |
| case04 | scatter_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.012129 | cuda_kernel_events |
| case04 | embedding | 1 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261819compute_grad_weightIfxEEvPKT0_PKT_S5_xxS5_PKxPNS_14AccumulateTypeIS6_Lb1EE4typeEx | 0.40311 | cuda_kernel_events |
| case04 | embedding | 2 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261815sum_and_scatterIfxEEvPKT0_PT_xS5_PKxPKNS_14AccumulateTypeIS6_Lb1EE4typeES5_S9_xx | 0.23466 | cuda_kernel_events |
| case04 | embedding | 3 | _ZN14at_cuda_detail3cub29DeviceRadixSortOnesweepKernelINS0_21DeviceRadixSortPolicyIxN2at4cuda3cub6detail10OpaqueTypeILi8EEEyE9Policy900ELb0ExS8_yiiNS0_6detail21identity_decomposer_tEEEvPT5_SE_PT3_PKSF_PT1_PKSJ_PT2_PKSN_T4_iiT6_ | 0.11565 | cuda_kernel_events |
| case05 | index_add | 1 | _ZN2at6native19indexFuncLargeIndexIfxjLi2ELi2ELin2ELb1ENS0_44_GLOBAL__N__99c14716_11_Indexing_cu_2a2c434e9ReduceAddEEEvNS_4cuda6detail10TensorInfoIT_T1_EENS6_IKS7_S8_EENS6_IKT0_S8_EEiiS8_S8_xxRKT6_S7_ | 12.988 | cuda_kernel_events |
| case05 | index_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 1.0711 | cuda_kernel_events |
| case05 | scatter_add | 1 | _ZN2at6native29vectorized_scatter_add_kernelILi16EfxEEvPT0_PKS2_PKT1_ixxxxii | 22.782 | cuda_kernel_events |
| case05 | scatter_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 1.0673 | cuda_kernel_events |
| case05 | embedding | 1 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261819compute_grad_weightIfxEEvPKT0_PKT_S5_xxS5_PKxPNS_14AccumulateTypeIS6_Lb1EE4typeEx | 7.226 | cuda_kernel_events |
| case05 | embedding | 2 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261815sum_and_scatterIfxEEvPKT0_PT_xS5_PKxPKNS_14AccumulateTypeIS6_Lb1EE4typeES5_S9_xx | 2.476 | cuda_kernel_events |
| case05 | embedding | 3 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 1.0724 | cuda_kernel_events |
| case06 | index_add | 1 | _ZN2at6native19indexFuncLargeIndexIfxjLi2ELi2ELin2ELb1ENS0_44_GLOBAL__N__99c14716_11_Indexing_cu_2a2c434e9ReduceAddEEEvNS_4cuda6detail10TensorInfoIT_T1_EENS6_IKS7_S8_EENS6_IKT0_S8_EEiiS8_S8_xxRKT6_S7_ | 2.9301 | cuda_kernel_events |
| case06 | index_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.011841 | cuda_kernel_events |
| case06 | scatter_add | 1 | _ZN2at6native29vectorized_scatter_add_kernelILi16EfxEEvPT0_PKS2_PKT1_ixxxxii | 10.099 | cuda_kernel_events |
| case06 | scatter_add | 2 | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_ | 0.012192 | cuda_kernel_events |
| case06 | embedding | 1 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261819compute_grad_weightIfxEEvPKT0_PKT_S5_xxS5_PKxPNS_14AccumulateTypeIS6_Lb1EE4typeEx | 1.4398 | cuda_kernel_events |
| case06 | embedding | 2 | _ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261815sum_and_scatterIfxEEvPKT0_PT_xS5_PKxPKNS_14AccumulateTypeIS6_Lb1EE4typeES5_S9_xx | 0.74513 | cuda_kernel_events |
| case06 | embedding | 3 | _ZN14at_cuda_detail3cub29DeviceRadixSortOnesweepKernelINS0_21DeviceRadixSortPolicyIxN2at4cuda3cub6detail10OpaqueTypeILi8EEEyE9Policy900ELb0ExS8_yiiNS0_6detail21identity_decomposer_tEEEvPT5_SE_PT3_PKSF_PT1_PKSJ_PT2_PKSN_T4_iiT6_ | 0.36893 | cuda_kernel_events |

Times in this table total all 3 profiled invocations. Full CPU/operator/kernel key averages and top-10 tables are under events/. CUDA kernel keys are preferred; if unavailable, attributed CPU operators are used, or an explicitly labeled CPU-time fallback. Inclusive parent/child times must not be added. Aggregate CPU totals sum self times; CUDA totals sum kernel durations when available, not wall time.

## Observations

- case01: raw event-timing minimum `index_add`, runner-up `embedding`, ratio 1.67; exceeds 5% within this subset: True. Complete selected API set: True.
- case02: raw event-timing minimum `embedding`, runner-up `scatter_add`, ratio 2.63; exceeds 5% within this subset: True. Complete selected API set: True.
- case03: raw event-timing minimum `embedding`, runner-up `index_add`, ratio 1.12; exceeds 5% within this subset: True. Complete selected API set: True.
- case04: raw event-timing minimum `index_add`, runner-up `embedding`, ratio 1.2; exceeds 5% within this subset: True. Complete selected API set: True.
- case05: raw event-timing minimum `embedding`, runner-up `index_add`, ratio 1.26; exceeds 5% within this subset: True. Complete selected API set: True.
- case06: raw event-timing minimum `embedding`, runner-up `index_add`, ratio 1.01; exceeds 5% within this subset: False. Complete selected API set: True.
- `embedding`: most frequent highest-time CUDA key in this subset was `_ZN2at6native59_GLOBAL__N__37144284_26_EmbeddingBackwardKernel_cu_5202261819compute_grad_weightIfxEEvPKT0_PKT_S5_xxS5_PKxPNS_14AccumulateTypeIS6_Lb1EE4typeEx` (6 cases).
- `index_add`: most frequent highest-time CUDA key in this subset was `_ZN2at6native19indexFuncLargeIndexIfxjLi2ELi2ELin2ELb1ENS0_44_GLOBAL__N__99c14716_11_Indexing_cu_2a2c434e9ReduceAddEEEvNS_4cuda6detail10TensorInfoIT_T1_EENS6_IKS7_S8_EENS6_IKT0_S8_EEiiS8_S8_xxRKT6_S7_` (6 cases).
- `scatter_add`: most frequent highest-time CUDA key in this subset was `_ZN2at6native29vectorized_scatter_add_kernelILi16EfxEEvPT0_PKS2_PKT1_ixxxxii` (6 cases).

## Warnings and failures

No data available.

## Limitations

Profiler explains operator/kernel composition, not universal causality. Results are per environment. CPU and CUDA profiler times include instrumentation and synchronization overhead and must not replace unprofiled CUDA-event timings. This is a representative subset, not a full grid or end-to-end model training. All profiled invocations synchronize. Warm cache only; record-shapes, memory, and stack capture can perturb execution. CUDA kernel attribution depends on CUPTI/Kineto support and permissions. Memory freed from allocations made before profiler start may not be attributable. Negative memory deltas are possible. There is no custom CUDA/C++ code.

For Colab, copy both profile_phase3.py and sparse_grad_benchmark.py into the same directory. Reuse the same cases and parameters. Optional --export-trace creates traces/; --record-shapes and --with-stack are off by default. CSV setting columns override CLI defaults; explicit size/draw flags expand templates and duplicates are removed before --max-cases is applied.
