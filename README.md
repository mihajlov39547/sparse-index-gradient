# Sparse-Index Gradient Accumulation in PyTorch CUDA Backpropagation

This repository contains the benchmark harness, analysis scripts, and generated artifacts supporting the manuscript:

**Tie-Aware Characterization of Sparse-Index Gradient Accumulation for PyTorch CUDA Backpropagation**

The study characterizes three standard PyTorch CUDA sparse-index accumulation paths:

- `index_add`
- `scatter_add`
- `Embedding.backward`

The experiments evaluate these paths under correctness-gated matched workloads, CUDA-event timing, profiler inspection, threshold-sensitivity checks, descriptive bootstrap intervals, regret analysis, winner-map summaries, and a controlled backward-style proxy.

## Purpose of this repository

This repository is made public to support peer review, reproducibility, and verification of the submitted manuscript.

The repository is intended to let reviewers and readers inspect:

- benchmark code;
- workload construction;
- validation logic;
- aggregation scripts;
- generated result artifacts;
- tables and figures used in the manuscript.

## Repository structure

```text
sparse-gradient/
  Benchmark scripts, analysis scripts, and generated result artifacts.

paper_sparse_gradient/
  Baseline LaTeX manuscript source and figures.

WileyDesign/cpe_sparse_gradient/
  Wiley/CPE-specific manuscript version prepared for submission.

README.md
  This file.
