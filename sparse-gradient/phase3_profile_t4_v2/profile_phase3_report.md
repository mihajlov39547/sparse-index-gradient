# Phase 3: Representative Sparse-Index Gradient Profiles

## Environment

| gpu_name | compute_capability | pytorch_version | cuda_version | python_version | platform |
| --- | --- | --- | --- | --- | --- |
| Tesla T4 | [7, 5] | 2.11.0+cu130 | 13.0 | 3.13.15 (main, Aug  6 2026, 11:06:22) [GCC 13.3.0] | Linux-6.6.122+-x86_64-with-glibc2.39 |

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
| case01 | 0.95 | 8 | 1.2 | 1234 | c565c62e163c47f8a5065b03df65f9f7ed1450fc533c521c1ea16fa215ab6e96 | 24892552c5616a92f736fa6df945fba863c82f7ebf2e82c4ff237e7ef7fc4714 |
| case02 | 0.95 | 8 | 1.2 | 1234 | 2d33528ba0eb923018ab3dd5a1f95a6a527952f6cf2731fa6b0b0765c8eacc28 | 1c98146508d9fa316dbf830f2d70d85c93bea0fcc3187ff3ff33e05e4b91c3c3 |
| case03 | 0.95 | 8 | 1.2 | 1234 | d6c4a9be6ceb9dc995b8165359fa929e07ecf49c41f561dacdffa5ece89b14ad | 9efec563f5e7e0495b2679fa65da929c420781d6c2fe9876187fcf47b5f2e2c5 |
| case04 | 0.95 | 8 | 1.2 | 1234 | dda855fc21f3754fb375e5d5149b5afebb21529a6ce8d35d89feffbb46a19ae8 | 27c4cfe2cf6b4744c5bed64df8db81a7c6612d00185b6a657594a0c20f8fc894 |
| case05 | 0.95 | 8 | 1.2 | 1234 | 5c29cf8d6c8fbab570d92ed8b47ad4ee92752459d89e126fce64c6c94974a027 | 8004f8fb9bedd9cebfa4d928f5d8dc9a13daa23e6509aa802efbb61180813835 |
| case06 | 0.95 | 8 | 1.2 | 1234 | 553f70996d9976bac670390b0e6bb49a6fd661ab4377749d7b5dfc7704d90af8 | 038b40525eb92b33066bca74d7a75dc2f8d0ff0cb6d41e13a315d17696a9bddd |

Hashes use contiguous CPU tensor bytes (int64 indices, float32 source), computed before timing/profiling. They check exact data reproducibility; source values may differ between PyTorch/CUDA versions even with the same seed. Workload shape and settings must also match.

## Unprofiled CUDA-event timing for profiled subset

| case_id | winner | runner_up | margin_ratio | decisive_5pct | complete_ops | eligible_operations | index_add_ms | scatter_add_ms | embedding_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case01 | scatter_add | index_add | 1.0216 | False | True | 3 | 0.36382 | 0.35613 | 0.84189 |
| case02 | embedding | scatter_add | 1.7065 | True | True | 3 | 8.8677 | 5.8882 | 3.4505 |
| case03 | scatter_add | index_add | 1.1855 | True | True | 3 | 0.62624 | 0.52824 | 0.96618 |
| case04 | index_add | scatter_add | 1.0583 | True | True | 3 | 0.2663 | 0.28182 | 0.69994 |
| case05 | scatter_add | embedding | 1.2848 | True | True | 3 | 4.7283 | 2.6694 | 3.4296 |
| case06 | scatter_add | index_add | 1.0399 | False | True | 3 | 1.1026 | 1.0602 | 1.4387 |

Timings use fresh dense outputs for index_add/scatter_add and retained-graph backward for embedding. Embedding forward is outside timing; grad=None clearing is in prepare, matching v3. Samples use inner=1 and fixed operation order, so these subset decisions are not a replacement for randomized/batched Phase 2 confirmation.

## Profiler-derived GPU-busy proxy

| case_id | operation | kernel_ms_per_invocation | gpu_busy_fraction | likely_host_bound |
| --- | --- | --- | --- | --- |
| case01 | index_add | 0.33022 | 0.90765 | False |
| case01 | scatter_add | 0.30763 | 0.86383 | False |
| case01 | embedding | 0.6256 | 0.74309 | False |
| case02 | index_add | 7.6834 | 0.86645 | False |
| case02 | scatter_add | 5.8032 | 0.98557 | False |
| case02 | embedding | 3.239 | 0.93871 | False |
| case03 | index_add | 0.59574 | 0.95129 | False |
| case03 | scatter_add | 0.49004 | 0.92769 | False |
| case03 | embedding | 0.77484 | 0.80196 | False |
| case04 | index_add | 0.19237 | 0.72237 | False |
| case04 | scatter_add | 0.18474 | 0.65552 | False |
| case04 | embedding | 0.30497 | 0.43571 | True |
| case05 | index_add | 4.6579 | 0.98511 | False |
| case05 | scatter_add | 2.6303 | 0.98536 | False |
| case05 | embedding | 3.2542 | 0.94887 | False |
| case06 | index_add | 1.0066 | 0.91296 | False |
| case06 | scatter_add | 0.96778 | 0.9128 | False |
| case06 | embedding | 1.2201 | 0.84804 | False |

gpu_busy_fraction = (total profiled CUDA kernel time / profile_repeat) / unprofiled median_ms. This is a profiler-derived proxy, not a hardware utilization counter. It can exceed 1 or be distorted when profiler/kernel timing differs from unprofiled event timing. Use it only as a host-overhead warning signal: likely_host_bound flags values below 0.6, not a causal conclusion. Values are unavailable when CUDA kernel events are unavailable. The fraction is not clamped.

## Top CUDA attribution

| case_id | operation | rank | CUDA key | total_ms | attribution |
| --- | --- | --- | --- | --- | --- |
| case01 | index_add | 1 | void at::native::indexFuncLargeIndex<float, long, unsigned int, 2, 2, -2, true, at::native::(anonymous namespace)::ReduceAdd>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, long, long, at::native::(anonymous namespace)::ReduceAdd const&, float) | 0.97397 | cuda_kernel_events |
| case01 | index_add | 2 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) | 0.016704 | cuda_kernel_events |
| case01 | scatter_add | 1 | void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gather_internal_kernel<true, float, long>::operator()<at::native::ReduceAdd>(at::TensorIterator&, long, long, long, at::native::ReduceAdd const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, float, long>::operator()<at::native::ReduceAdd>(at::TensorIterator&, long, long, long, at::native::ReduceAdd const&)::{lambda(int)#1}) | 0.906 | cuda_kernel_events |
| case01 | scatter_add | 2 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) | 0.016896 | cuda_kernel_events |
| case01 | embedding | 1 | void at::native::(anonymous namespace)::sum_and_scatter<float, long>(long const*, float*, long, long const*, long const*, at::AccumulateType<float, true>::type const*, long const*, long const*, long, long) | 0.605 | cuda_kernel_events |
| case01 | embedding | 2 | void at_cuda_detail::cub::detail::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::detail::radix::policy_hub<long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long>::Policy1000, (at_cuda_detail::cub::SortOrder)0, long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long, int, int, at_cuda_detail::cub::detail::identity_decomposer_t>(int*, int*, unsigned long long*, unsigned long long const*, long*, long const*, at::cuda::cub::detail::OpaqueType<8>*, at::cuda::cub::detail::OpaqueType<8> const*, int, int, int, at_cuda_detail::cub::detail::identity_decomposer_t) | 0.52779 | cuda_kernel_events |
| case01 | embedding | 3 | void at::native::(anonymous namespace)::compute_grad_weight<float, long>(long const*, float const*, long const*, long, long, long const*, long const*, at::AccumulateType<float, true>::type*, long) | 0.33241 | cuda_kernel_events |
| case02 | index_add | 1 | void at::native::indexFuncLargeIndex<float, long, unsigned int, 2, 2, -2, true, at::native::(anonymous namespace)::ReduceAdd>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, long, long, at::native::(anonymous namespace)::ReduceAdd const&, float) | 22.948 | cuda_kernel_events |
| case02 | index_add | 2 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) | 0.10233 | cuda_kernel_events |
| case02 | scatter_add | 1 | void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gather_internal_kernel<true, float, long>::operator()<at::native::ReduceAdd>(at::TensorIterator&, long, long, long, at::native::ReduceAdd const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, float, long>::operator()<at::native::ReduceAdd>(at::TensorIterator&, long, long, long, at::native::ReduceAdd const&)::{lambda(int)#1}) | 17.308 | cuda_kernel_events |
| case02 | scatter_add | 2 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) | 0.10189 | cuda_kernel_events |
| case02 | embedding | 1 | void at::native::(anonymous namespace)::compute_grad_weight<float, long>(long const*, float const*, long const*, long, long, long const*, long const*, at::AccumulateType<float, true>::type*, long) | 7.5536 | cuda_kernel_events |
| case02 | embedding | 2 | void at_cuda_detail::cub::detail::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::detail::radix::policy_hub<long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long>::Policy1000, (at_cuda_detail::cub::SortOrder)0, long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long, int, int, at_cuda_detail::cub::detail::identity_decomposer_t>(int*, int*, unsigned long long*, unsigned long long const*, long*, long const*, at::cuda::cub::detail::OpaqueType<8>*, at::cuda::cub::detail::OpaqueType<8> const*, int, int, int, at_cuda_detail::cub::detail::identity_decomposer_t) | 1.0021 | cuda_kernel_events |
| case02 | embedding | 3 | void at::native::(anonymous namespace)::sum_and_scatter<float, long>(long const*, float*, long, long const*, long const*, at::AccumulateType<float, true>::type const*, long const*, long const*, long, long) | 0.80732 | cuda_kernel_events |
| case03 | index_add | 1 | void at::native::indexFuncLargeIndex<float, long, unsigned int, 2, 2, -2, true, at::native::(anonymous namespace)::ReduceAdd>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, long, long, at::native::(anonymous namespace)::ReduceAdd const&, float) | 1.5864 | cuda_kernel_events |
| case03 | index_add | 2 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) | 0.20086 | cuda_kernel_events |
| case03 | scatter_add | 1 | void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gather_internal_kernel<true, float, long>::operator()<at::native::ReduceAdd>(at::TensorIterator&, long, long, long, at::native::ReduceAdd const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, float, long>::operator()<at::native::ReduceAdd>(at::TensorIterator&, long, long, long, at::native::ReduceAdd const&)::{lambda(int)#1}) | 1.2671 | cuda_kernel_events |
| case03 | scatter_add | 2 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) | 0.20307 | cuda_kernel_events |
| case03 | embedding | 1 | void at::native::(anonymous namespace)::compute_grad_weight<float, long>(long const*, float const*, long const*, long, long, long const*, long const*, at::AccumulateType<float, true>::type*, long) | 1.1607 | cuda_kernel_events |
| case03 | embedding | 2 | void at::native::(anonymous namespace)::sum_and_scatter<float, long>(long const*, float*, long, long const*, long const*, at::AccumulateType<float, true>::type const*, long const*, long const*, long, long) | 0.55046 | cuda_kernel_events |
| case03 | embedding | 3 | void at_cuda_detail::cub::detail::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::detail::radix::policy_hub<long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long>::Policy1000, (at_cuda_detail::cub::SortOrder)0, long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long, int, int, at_cuda_detail::cub::detail::identity_decomposer_t>(int*, int*, unsigned long long*, unsigned long long const*, long*, long const*, at::cuda::cub::detail::OpaqueType<8>*, at::cuda::cub::detail::OpaqueType<8> const*, int, int, int, at_cuda_detail::cub::detail::identity_decomposer_t) | 0.22368 | cuda_kernel_events |
| case04 | index_add | 1 | void at::native::indexFuncLargeIndex<float, long, unsigned int, 2, 2, -2, true, at::native::(anonymous namespace)::ReduceAdd>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, long, long, at::native::(anonymous namespace)::ReduceAdd const&, float) | 0.5694 | cuda_kernel_events |
| case04 | index_add | 2 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) | 0.007712 | cuda_kernel_events |
| case04 | scatter_add | 1 | void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gather_internal_kernel<true, float, long>::operator()<at::native::ReduceAdd>(at::TensorIterator&, long, long, long, at::native::ReduceAdd const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, float, long>::operator()<at::native::ReduceAdd>(at::TensorIterator&, long, long, long, at::native::ReduceAdd const&)::{lambda(int)#1}) | 0.54645 | cuda_kernel_events |
| case04 | scatter_add | 2 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) | 0.007775 | cuda_kernel_events |
| case04 | embedding | 1 | void at::native::(anonymous namespace)::compute_grad_weight<float, long>(long const*, float const*, long const*, long, long, long const*, long const*, at::AccumulateType<float, true>::type*, long) | 0.29334 | cuda_kernel_events |
| case04 | embedding | 2 | void at::native::(anonymous namespace)::sum_and_scatter<float, long>(long const*, float*, long, long const*, long const*, at::AccumulateType<float, true>::type const*, long const*, long const*, long, long) | 0.23817 | cuda_kernel_events |
| case04 | embedding | 3 | void at_cuda_detail::cub::detail::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::detail::radix::policy_hub<long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long>::Policy1000, (at_cuda_detail::cub::SortOrder)0, long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long, int, int, at_cuda_detail::cub::detail::identity_decomposer_t>(int*, int*, unsigned long long*, unsigned long long const*, long*, long const*, at::cuda::cub::detail::OpaqueType<8>*, at::cuda::cub::detail::OpaqueType<8> const*, int, int, int, at_cuda_detail::cub::detail::identity_decomposer_t) | 0.21997 | cuda_kernel_events |
| case05 | index_add | 1 | void at::native::indexFuncLargeIndex<float, long, unsigned int, 2, 2, -2, true, at::native::(anonymous namespace)::ReduceAdd>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, long, long, at::native::(anonymous namespace)::ReduceAdd const&, float) | 13.115 | cuda_kernel_events |
| case05 | index_add | 2 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) | 0.85864 | cuda_kernel_events |
| case05 | scatter_add | 1 | void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gather_internal_kernel<true, float, long>::operator()<at::native::ReduceAdd>(at::TensorIterator&, long, long, long, at::native::ReduceAdd const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, float, long>::operator()<at::native::ReduceAdd>(at::TensorIterator&, long, long, long, at::native::ReduceAdd const&)::{lambda(int)#1}) | 7.0279 | cuda_kernel_events |
| case05 | scatter_add | 2 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) | 0.86311 | cuda_kernel_events |
| case05 | embedding | 1 | void at::native::(anonymous namespace)::compute_grad_weight<float, long>(long const*, float const*, long const*, long, long, long const*, long const*, at::AccumulateType<float, true>::type*, long) | 4.8883 | cuda_kernel_events |
| case05 | embedding | 2 | void at::native::(anonymous namespace)::sum_and_scatter<float, long>(long const*, float*, long, long const*, long const*, at::AccumulateType<float, true>::type const*, long const*, long const*, long, long) | 2.391 | cuda_kernel_events |
| case05 | embedding | 3 | void at_cuda_detail::cub::detail::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::detail::radix::policy_hub<long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long>::Policy1000, (at_cuda_detail::cub::SortOrder)0, long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long, int, int, at_cuda_detail::cub::detail::identity_decomposer_t>(int*, int*, unsigned long long*, unsigned long long const*, long*, long const*, at::cuda::cub::detail::OpaqueType<8>*, at::cuda::cub::detail::OpaqueType<8> const*, int, int, int, at_cuda_detail::cub::detail::identity_decomposer_t) | 1.0765 | cuda_kernel_events |
| case06 | index_add | 1 | void at::native::indexFuncLargeIndex<float, long, unsigned int, 2, 2, -2, true, at::native::(anonymous namespace)::ReduceAdd>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, long, long, at::native::(anonymous namespace)::ReduceAdd const&, float) | 3.0097 | cuda_kernel_events |
| case06 | index_add | 2 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) | 0.01008 | cuda_kernel_events |
| case06 | scatter_add | 1 | void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gather_internal_kernel<true, float, long>::operator()<at::native::ReduceAdd>(at::TensorIterator&, long, long, long, at::native::ReduceAdd const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, float, long>::operator()<at::native::ReduceAdd>(at::TensorIterator&, long, long, long, at::native::ReduceAdd const&)::{lambda(int)#1}) | 2.8932 | cuda_kernel_events |
| case06 | scatter_add | 2 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) | 0.010176 | cuda_kernel_events |
| case06 | embedding | 1 | void at::native::(anonymous namespace)::sum_and_scatter<float, long>(long const*, float*, long, long const*, long const*, at::AccumulateType<float, true>::type const*, long const*, long const*, long, long) | 1.0988 | cuda_kernel_events |
| case06 | embedding | 2 | void at::native::(anonymous namespace)::compute_grad_weight<float, long>(long const*, float const*, long const*, long, long, long const*, long const*, at::AccumulateType<float, true>::type*, long) | 1.0801 | cuda_kernel_events |
| case06 | embedding | 3 | void at_cuda_detail::cub::detail::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::detail::radix::policy_hub<long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long>::Policy1000, (at_cuda_detail::cub::SortOrder)0, long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long, int, int, at_cuda_detail::cub::detail::identity_decomposer_t>(int*, int*, unsigned long long*, unsigned long long const*, long*, long const*, at::cuda::cub::detail::OpaqueType<8>*, at::cuda::cub::detail::OpaqueType<8> const*, int, int, int, at_cuda_detail::cub::detail::identity_decomposer_t) | 1.0436 | cuda_kernel_events |

Times in this table total all 3 profiled invocations. Full CPU/operator/kernel key averages and top-10 tables are under events/. CUDA kernel keys are preferred; if unavailable, attributed CPU operators are used, or an explicitly labeled CPU-time fallback. Inclusive parent/child times must not be added. Aggregate CPU totals sum self times; CUDA totals sum kernel durations when available, not wall time.

## Observations

- case01: raw event-timing minimum `scatter_add`, runner-up `index_add`, ratio 1.02; exceeds 5% within this subset: False. Complete selected API set: True.
- case02: raw event-timing minimum `embedding`, runner-up `scatter_add`, ratio 1.71; exceeds 5% within this subset: True. Complete selected API set: True.
- case03: raw event-timing minimum `scatter_add`, runner-up `index_add`, ratio 1.19; exceeds 5% within this subset: True. Complete selected API set: True.
- case04: raw event-timing minimum `index_add`, runner-up `scatter_add`, ratio 1.06; exceeds 5% within this subset: True. Complete selected API set: True.
- case05: raw event-timing minimum `scatter_add`, runner-up `embedding`, ratio 1.28; exceeds 5% within this subset: True. Complete selected API set: True.
- case06: raw event-timing minimum `scatter_add`, runner-up `index_add`, ratio 1.04; exceeds 5% within this subset: False. Complete selected API set: True.
- `embedding`: most frequent highest-time CUDA key in this subset was `void at::native::(anonymous namespace)::compute_grad_weight<float, long>(long const*, float const*, long const*, long, long, long const*, long const*, at::AccumulateType<float, true>::type*, long)` (4 cases).
- `index_add`: most frequent highest-time CUDA key in this subset was `void at::native::indexFuncLargeIndex<float, long, unsigned int, 2, 2, -2, true, at::native::(anonymous namespace)::ReduceAdd>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, long, long, at::native::(anonymous namespace)::ReduceAdd const&, float)` (6 cases).
- `scatter_add`: most frequent highest-time CUDA key in this subset was `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gather_internal_kernel<true, float, long>::operator()<at::native::ReduceAdd>(at::TensorIterator&, long, long, long, at::native::ReduceAdd const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, float, long>::operator()<at::native::ReduceAdd>(at::TensorIterator&, long, long, long, at::native::ReduceAdd const&)::{lambda(int)#1})` (6 cases).

## Warnings and failures

| case_id | operation | status | notes |
| --- | --- | --- | --- |
| case01 | index_add | ok | Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True. |

## Limitations

Profiler explains operator/kernel composition, not universal causality. Results are per environment. CPU and CUDA profiler times include instrumentation and synchronization overhead and must not replace unprofiled CUDA-event timings. This is a representative subset, not a full grid or end-to-end model training. All profiled invocations synchronize. Warm cache only; record-shapes, memory, and stack capture can perturb execution. CUDA kernel attribution depends on CUPTI/Kineto support and permissions. Memory freed from allocations made before profiler start may not be attributable. Negative memory deltas are possible. There is no custom CUDA/C++ code.

For Colab, copy both profile_phase3.py and sparse_grad_benchmark.py into the same directory. Reuse the same cases and parameters. Optional --export-trace creates traces/; --record-shapes and --with-stack are off by default. CSV setting columns override CLI defaults; explicit size/draw flags expand templates and duplicates are removed before --max-cases is applied.
