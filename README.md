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
```

## Reproducibility

The reported analyses use fixed seeds, saved raw and summary CSV artifacts, metadata files recording PyTorch and CUDA versions, and validation outputs for accepted runs.

The manuscript reports results from two GPU/software stacks:

- GTX1650Ti stack;
- hosted Colab T4 stack.

The T4 repeat is a repeat run on the hosted T4 stack, not a third independent environment.

## Scope

This repository supports a controlled performance characterization of sparse-index gradient accumulation paths in PyTorch CUDA backpropagation.

The study does **not** claim:

- a universal fastest API;
- hardware-only causality;
- a validated deployment autotuner;
- full production-training throughput;
- superiority over custom CUDA kernels or specialized recommendation-system libraries.

The practical output of the study is a measurement-backed, tie-aware selection protocol for comparing candidate paths under a target sparse workload and execution stack.

## Data availability

The benchmark harness, aggregation scripts, and generated result artifacts supporting the study are available in this repository.

## Citation

If you refer to this repository or the associated manuscript, please cite the manuscript:

```text
Mihajlovic, M. Tie-Aware Characterization of Sparse-Index Gradient Accumulation for PyTorch CUDA Backpropagation. Manuscript submitted for peer review.
```

A machine-readable citation file is provided in `CITATION.cff`.

## License and reuse

This repository is publicly visible for peer review and reproducibility.

No open-source license is granted at this time. Unless otherwise stated, all rights are reserved by the author.

You may view the repository for evaluation, verification, and scholarly review purposes. Reproduction, redistribution, modification, reuse of code or artifacts, or creation of derivative works is not permitted without prior written permission from the author.

See `LICENSE` for details.

## Contact

Marko Mihajlovic  
Singidunum University, Belgrade, Serbia  
ORCID: https://orcid.org/0009-0006-7979-0341
