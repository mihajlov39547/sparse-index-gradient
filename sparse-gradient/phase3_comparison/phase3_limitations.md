# Phase 3 Comparison Limitations

- Phase 3 is a representative subset, not a full grid or statistical replication of Phase 2.
- Event timings use inner=1 and fixed operation order; Phase 2 remains the main randomized/batched winner-count evidence.
- Profiler overhead is not used as primary benchmark timing. Profiler CUDA totals and event medians are not interchangeable.
- GPU-busy fraction is a proxy, not hardware utilization; it can exceed 1. likely_host_bound is a warning signal, not causal proof.
- No Nsight Compute counters, end-to-end model training, or custom CUDA implementation.
- Absolute GTX/T4 latencies are not controlled hardware benchmarks; GPU, OS, PyTorch, CUDA, and hosted runtime differ together.
- Ordered name-based categories are approximate. Full shares exclude annotations; top-10 fallback shares describe only that truncated list.
- Casewise means weight each available case equally, not by kernel duration. Shares summarize kernel work, not profiler wall-clock utilization.
- CPU/NumPy index hash discrepancies are audit warnings; source RNG values can differ across software stacks even with equal seeds.

- case01: src_sha256 mismatch (audit warning only; source RNG may differ across stacks).
- case02: src_sha256 mismatch (audit warning only; source RNG may differ across stacks).
- case03: src_sha256 mismatch (audit warning only; source RNG may differ across stacks).
- case04: src_sha256 mismatch (audit warning only; source RNG may differ across stacks).
- case05: src_sha256 mismatch (audit warning only; source RNG may differ across stacks).
- case06: src_sha256 mismatch (audit warning only; source RNG may differ across stacks).