"""PyTorch-only CUDA feasibility study of dense sparse-index gradient accumulation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
import platform
from pathlib import Path
import random
import sys
from typing import Callable
import uuid
import zlib

import numpy as np
import pandas as pd
import torch


OPS = ("index_add", "scatter_add", "embedding", "embedding_bag", "sorted_segment",
       "index_add_zero_timed", "index_add_alloc_timed", "embedding_full_backward",
       "index_add_acc_only", "scatter_add_acc_only", "sorted_segment_sync", "noop_kernel")
SCOPE_GROUPS = {
    **dict.fromkeys(OPS[:5], "fair_dense_grad"),
    "index_add_acc_only": "accumulate_only",
    "scatter_add_acc_only": "accumulate_only",
    "sorted_segment_sync": "fair_dense_grad_sync",
    "embedding_full_backward": "end_to_end",
    "noop_kernel": "overhead_floor",
    "index_add_zero_timed": "deprecated_alias",
    "index_add_alloc_timed": "deprecated_alias",
}
OP_PROPERTIES = {
    op: {"has_output": op != "noop_kernel",
         "batchable": op not in {"index_add_acc_only", "scatter_add_acc_only", "embedding_full_backward"}}
    for op in OPS
}
TIMING_SCOPES = {
    "index_add": "fresh dense output allocation, zeroing, and indexed accumulation",
    "index_add_zero_timed": "deprecated legacy alias: output zeroing and accumulation; allocation excluded",
    "index_add_alloc_timed": "deprecated alias of index_add: fresh dense output allocation, zeroing, and accumulation",
    "scatter_add": "fresh dense output allocation, zeroing, and scatter accumulation; index view setup excluded",
    "embedding": "retained forward graph; backward including dense gradient allocation; grad clearing excluded",
    "embedding_full_backward": "grad clearing, fresh forward graph construction/execution, and backward; module creation excluded",
    "embedding_bag": "retained forward graph; backward including dense gradient allocation; grad clearing excluded",
    "sorted_segment": "stable sort, reorder, searchsorted boundaries, segment_reduce(offsets) directly to fresh dense output; row_ids setup excluded",
    "sorted_segment_sync": "legacy stable sort, reorder, unique_consecutive lengths, segment_reduce, dense zero allocation and index_copy",
    "index_add_acc_only": "legacy index_add: accumulation only; output allocation/reset excluded; inner forced to 1",
    "scatter_add_acc_only": "legacy scatter_add: accumulation only; output allocation/reset excluded; inner forced to 1",
    "noop_kernel": "one-element add_ launch overhead floor; no accumulation reference output",
}
RTOL, ATOL = 1e-4, 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("sparse-gradient/results"))
    parser.add_argument("--rows", type=int, nargs="+", default=[4096, 16384, 65536])
    parser.add_argument("--indices", type=int, nargs="+", default=[4096, 16384, 65536])
    parser.add_argument("--dims", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--distributions", nargs="+", choices=["uniform", "hotspot", "zipf"],
                        default=["uniform", "hotspot", "zipf"])
    parser.add_argument("--hotspot-fraction", type=float, default=0.8)
    parser.add_argument("--hotspot-rows", type=int, default=16)
    parser.add_argument("--zipf-alpha", type=float, default=1.2)
    parser.add_argument("--zipf-mode", choices=["truncated", "wrapped"], default="truncated")
    parser.add_argument("--workload-seeds", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32"], default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ops", nargs="+", choices=OPS, default=list(OPS[:5]))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--inner", choices=["auto"] + [str(2**i) for i in range(9)], default="auto")
    parser.add_argument("--check-sync-free", action="store_true")
    parser.add_argument("--row-layout", choices=["scattered", "adjacent"], default="scattered")
    parser.add_argument("--index-order", choices=["random", "sorted"], default="random")
    parser.add_argument("--cache-mode", choices=["warm", "cold"], default="warm")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--progress", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--determinism-check", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--max-combinations", type=int)
    args = parser.parse_args()
    # Quick mode deliberately overrides sizes and iteration counts only.
    if args.quick:
        args.rows, args.indices, args.dims = [1024], [2048], [16, 64]
        args.warmup, args.repeats = 2, 5
    for name in ("rows", "indices", "dims"):
        if any(value <= 0 for value in getattr(args, name)):
            parser.error(f"--{name} must contain positive integers")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats must be positive")
    if args.rounds < 1 or args.repeats < args.rounds:
        parser.error("rounds must be positive and repeats must be >= rounds")
    if args.workload_seeds < 1:
        parser.error("workload-seeds must be positive")
    if not 0 <= args.hotspot_fraction <= 1 or args.hotspot_rows < 1:
        parser.error("hotspot-fraction must be in [0, 1]; hotspot-rows must be positive")
    if not np.isfinite(args.zipf_alpha) or args.zipf_alpha <= 1:
        parser.error("zipf-alpha must be finite and greater than 1")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    if args.max_combinations is not None and args.max_combinations < 1:
        parser.error("max-combinations must be positive")
    for name in ("rows", "indices", "dims", "distributions", "ops"):
        setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    return args


def get_device_info(device: torch.device) -> dict:
    properties = torch.cuda.get_device_properties(device)
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "pytorch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_vram_bytes": properties.total_memory,
        "device": str(device),
        "platform": platform.platform(),
    }


def sample_truncated_zipf(num_rows: int, num_indices: int, alpha: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Finite Zipf probabilities: rank 1 maps to row 0, with no wrapped tail."""
    ranks = np.arange(1, num_rows + 1, dtype=np.float64)
    probs = np.power(ranks, -alpha)
    probs /= probs.sum()
    return rng.choice(num_rows, size=num_indices, replace=True, p=probs).astype(np.int64)


def generate_indices(num_rows: int, num_indices: int, distribution: str,
                     args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    """Generate on CPU outside timing, then transfer once to CUDA."""
    if distribution == "uniform":
        return rng.integers(0, num_rows, size=num_indices, dtype=np.int64)
    if distribution == "hotspot":
        hot_rows = min(args.hotspot_rows, num_rows)
        count = round(args.hotspot_fraction * num_indices)
        # An exact fraction targets hot rows; the remainder targets cold rows.
        hot = rng.integers(0, hot_rows, size=count, dtype=np.int64)
        low = hot_rows if hot_rows < num_rows else 0
        cold = rng.integers(low, num_rows, size=num_indices - count, dtype=np.int64)
        result = np.concatenate([hot, cold])
        rng.shuffle(result)
        return result
    if distribution == "zipf":
        if args.zipf_mode == "truncated":
            return sample_truncated_zipf(num_rows, num_indices, args.zipf_alpha, rng)
        if args.zipf_mode == "wrapped":
            return (rng.zipf(args.zipf_alpha, size=num_indices) - 1) % num_rows
    raise ValueError(f"Unsupported distribution/mode: {distribution}/{args.zipf_mode}")


def generate_workload_indices(rows, count, distribution, args, index_seed):
    """Shared by the benchmark and fixed-index regression check; no dim input."""
    rng = np.random.default_rng(index_seed)
    indices = generate_indices(rows, count, distribution, args, rng)
    if args.row_layout == "scattered":
        indices = rng.permutation(rows)[indices]
    if args.index_order == "sorted":
        indices = np.sort(indices)
    return indices


def index_stats(indices: np.ndarray) -> dict:
    _, counts = np.unique(indices, return_counts=True)
    return {
        "total_indices": int(indices.size),
        "unique_indices": int(counts.size),
        "duplicate_rate": 1 - counts.size / indices.size,
        "max_row_count": int(counts.max()),
        "mean_count_per_used_row": float(counts.mean()),
        "top1_share": float(counts.max() / indices.size),
        "top16_share": float(np.sort(counts)[-16:].sum() / indices.size),
    }


def index_seed_for(base, rows, count, dist, row_layout, index_order, zipf_mode, workload_draw):
    key = f"idx|{base}|{rows}|{count}|{dist}|{row_layout}|{index_order}|{zipf_mode}|{workload_draw}"
    return zlib.crc32(key.encode()) % 2**32


def src_seed_for(base, rows, count, dim, dist, row_layout, index_order, zipf_mode, workload_draw):
    key = f"src|{base}|{rows}|{count}|{dim}|{dist}|{row_layout}|{index_order}|{zipf_mode}|{workload_draw}"
    return zlib.crc32(key.encode()) % 2**32


def make_flush_buffer(device):
    props = torch.cuda.get_device_properties(device)
    l2 = getattr(props, "L2_cache_size", 0) or 64 * 2**20
    return torch.empty(max(2 * l2, 64 * 2**20) // 4, dtype=torch.float32, device=device)


def time_batch(prepare, run, device, inner, flush_buffer=None):
    """One sample, reported in milliseconds per invocation, not per batch."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    prepare()
    if flush_buffer is not None:
        flush_buffer.zero_()
    torch.cuda.synchronize(device)
    start.record()
    for i in range(inner):
        if i:
            prepare()  # Only host-side prepare is allowed when inner > 1.
        run()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end)) / inner


def calibrate_inner(prepare, run, device, target_ms=2.0, max_inner=256):
    inner = 1
    while inner < max_inner:
        if time_batch(prepare, run, device, inner) * inner >= target_ms:
            break
        inner *= 2
    return inner


def time_cuda_op(prepare: Callable, run: Callable, device: torch.device,
                 repeats: int, inner: int, flush_buffer=None) -> list[float]:
    return [time_batch(prepare, run, device, inner, flush_buffer) for _ in range(repeats)]


def check_sync_free(prepare, run, device):
    # PyTorch's detector is incomplete: True means no detected synchronization.
    prepare()
    run()
    torch.cuda.synchronize(device)
    prepare()
    torch.cuda.synchronize(device)
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        run()
        return True
    except RuntimeError as exc:
        if "synchroniz" not in str(exc).lower():
            raise  # Do not mislabel OOM or unsupported operators as synchronization.
        return False
    finally:
        torch.cuda.set_sync_debug_mode(previous)
        torch.cuda.synchronize(device)


def benchmark_index_add_acc_only(num_rows, indices, src):
    # Repeated feature gradients accumulate into a dense parameter gradient.
    out = torch.zeros((num_rows, src.shape[1]), device=src.device, dtype=src.dtype)
    return out.zero_, lambda: out.index_add_(0, indices, src), lambda: out


def benchmark_index_add_zero_timed(num_rows, indices, src):
    # Isolate the cost of resetting the dense gradient before accumulation.
    out = torch.zeros((num_rows, src.shape[1]), device=src.device, dtype=src.dtype)

    def run():
        out.zero_()
        out.index_add_(0, indices, src)

    return lambda: None, run, lambda: out


def benchmark_index_add(num_rows, indices, src):
    # Include fresh dense gradient creation, as in backward with grad=None.
    out = None

    def prepare():
        nonlocal out
        out = None

    def run():
        nonlocal out
        out = torch.zeros((num_rows, src.shape[1]), device=src.device, dtype=src.dtype)
        out.index_add_(0, indices, src)

    return prepare, run, lambda: out


def benchmark_scatter_add_acc_only(num_rows, indices, src):
    out = torch.zeros((num_rows, src.shape[1]), device=src.device, dtype=src.dtype)
    expanded = indices[:, None].expand(-1, src.shape[1])
    return out.zero_, lambda: out.scatter_add_(0, expanded, src), lambda: out


def benchmark_scatter_add(num_rows, indices, src):
    expanded = indices[:, None].expand(-1, src.shape[1])
    out = None

    def prepare():
        nonlocal out
        out = None

    def run():
        nonlocal out
        out = torch.zeros((num_rows, src.shape[1]), device=src.device, dtype=src.dtype)
        out.scatter_add_(0, expanded, src)

    return prepare, run, lambda: out


def benchmark_sorted_segment(num_rows: int, indices: torch.Tensor, src: torch.Tensor):
    out = None
    row_ids = torch.arange(num_rows + 1, device=src.device, dtype=indices.dtype)

    def prepare():
        nonlocal out
        out = None

    def run():
        nonlocal out
        sorted_idx, order = torch.sort(indices, stable=True)
        sorted_src = src.index_select(0, order)
        bounds = torch.searchsorted(sorted_idx, row_ids)
        # Offsets avoid the host validation sync observed with lengths= on CUDA.
        # Repeated boundaries yield zero-length segments, whose sum is zero.
        out = torch.segment_reduce(sorted_src, reduce="sum", offsets=bounds, axis=0)

    return prepare, run, lambda: out


def benchmark_sorted_segment_sync(num_rows: int, indices: torch.Tensor, src: torch.Tensor):
    """Group duplicate updates before writing each used dense row exactly once."""
    out = None

    def prepare():
        nonlocal out
        # Release the previous result, but do not cache any sorting/grouping work.
        out = None

    def run():
        nonlocal out
        # All five stages, including temporary and dense output allocation, are
        # inside the CUDA-event interval. Stable sort preserves equal-key order;
        # exact reproducibility still needs the independent determinism check.
        sorted_indices, order = torch.sort(indices, stable=True)
        sorted_src = src.index_select(0, order)
        # Consecutive equal-key runs define segment boundaries and lengths.
        unique_rows, lengths = torch.unique_consecutive(sorted_indices, return_counts=True)
        reduced = torch.segment_reduce(sorted_src, reduce="sum", lengths=lengths, axis=0)
        out = torch.zeros((num_rows, src.shape[1]), device=src.device, dtype=src.dtype)
        # unique_rows has no duplicates: no cross-update accumulation here.
        out.index_copy_(0, unique_rows, reduced)

    return prepare, run, lambda: out


def benchmark_embedding_backward(num_rows, indices, src):
    # sparse=False: the gradient is dense even though updates select sparse rows.
    embedding = torch.nn.Embedding(num_rows, src.shape[1], device=src.device,
                                   dtype=src.dtype, sparse=False)
    selected = embedding(indices)

    def prepare():
        embedding.weight.grad = None

    # Retain one fixed forward graph to exclude forward execution from timing.
    def run():
        selected.backward(src, retain_graph=True)

    return prepare, run, lambda: embedding.weight.grad


def benchmark_noop_kernel(num_rows, indices, src):
    scratch = torch.zeros(1, device=src.device)
    return lambda: None, lambda: scratch.add_(1), lambda: None


def benchmark_embedding_full_backward(num_rows, indices, src):
    # Module initialization is excluded; each invocation builds its own graph.
    embedding = torch.nn.Embedding(num_rows, src.shape[1], device=src.device,
                                   dtype=src.dtype, sparse=False)

    def run():
        embedding.weight.grad = None
        selected = embedding(indices)
        selected.backward(src)

    return lambda: None, run, lambda: embedding.weight.grad


def benchmark_embedding_bag_backward(num_rows, indices, src):
    # One index per sum bag gives exactly the same accumulation as Embedding.
    # This controlled baseline does not represent multi-index pooling workloads.
    embedding = torch.nn.EmbeddingBag(num_rows, src.shape[1], mode="sum",
                                      sparse=False, device=src.device, dtype=src.dtype)
    offsets = torch.arange(indices.numel(), device=src.device, dtype=torch.long)
    selected = embedding(indices, offsets)

    def prepare():
        embedding.weight.grad = None

    def run():
        selected.backward(src, retain_graph=True)

    return prepare, run, lambda: embedding.weight.grad


BENCHMARKS = {
    "index_add": benchmark_index_add,
    "scatter_add": benchmark_scatter_add,
    "embedding": benchmark_embedding_backward,
    "embedding_bag": benchmark_embedding_bag_backward,
    "sorted_segment": benchmark_sorted_segment,
    "index_add_zero_timed": benchmark_index_add_zero_timed,
    "index_add_alloc_timed": benchmark_index_add,
    "embedding_full_backward": benchmark_embedding_full_backward,
    "index_add_acc_only": benchmark_index_add_acc_only,
    "scatter_add_acc_only": benchmark_scatter_add_acc_only,
    "sorted_segment_sync": benchmark_sorted_segment_sync,
    "noop_kernel": benchmark_noop_kernel,
}


def run_benchmark(operation, num_rows, indices, src, reference, tolerance, args,
                  device, repeats, state, flush_buffer) -> dict:
    prepare, run, output = BENCHMARKS[operation](num_rows, indices, src)
    if args.check_sync_free and "sync_free" not in state:
        state["sync_free"] = check_sync_free(prepare, run, device)
    for _ in range(args.warmup):
        prepare()
        run()
    if "inner" not in state:
        if not OP_PROPERTIES[operation]["batchable"] or args.cache_mode == "cold":
            state["inner"] = 1
        elif args.inner == "auto":
            state["inner"] = calibrate_inner(prepare, run, device)
        else:
            state["inner"] = int(args.inner)
    times = time_cuda_op(prepare, run, device, repeats, state["inner"], flush_buffer)
    if not OP_PROPERTIES[operation]["has_output"]:
        return {"times": times, "correctness": "not_applicable"}
    result = output().detach().clone()
    valid = result.shape == reference.shape and bool(torch.isfinite(result).all())
    err = (result - reference).abs() if valid else None
    correct = valid and bool((err <= tolerance).all())
    bitwise = close = "not_checked"
    if args.determinism_check:
        bitwise, close = True, True
        # Compare three additional executions with the final timed execution.
        for _ in range(3):
            prepare()
            run()
            candidate = output().detach()
            torch.cuda.synchronize(device)
            # Integer view compares actual float32 bits, including signed zero.
            bitwise = bitwise and torch.equal(result.view(torch.int32), candidate.view(torch.int32))
            close = close and torch.allclose(result, candidate, rtol=RTOL, atol=ATOL)
    return {"times": times, "correctness": "passed" if correct else "failed",
            "max_abs_error": float(err.max()) if valid else float("inf"),
            "max_norm_err": float((err / tolerance).max()) if valid else float("inf"),
            "determinism_bitwise": bitwise, "determinism_allclose": close}


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for _, group in raw.groupby("combination_id", sort=False):
        row = group.iloc[0].drop(labels=["repeat_index", "elapsed_ms", "round_index", "run_order"]).to_dict()
        row["rounds_completed"] = group.loc[group.elapsed_ms.notna(), "round_index"].nunique()
        row["round_indices"] = json.dumps(group.round_index.unique().tolist())
        row["run_orders"] = json.dumps(group.run_order.tolist())
        # One failed round invalidates the entire operation/configuration.
        for field, success in (("status", "ok"), ("correctness", "passed")):
            bad = group.loc[~group[field].isin([success, "not_applicable"]), field]
            if not bad.empty:
                row[field] = bad.iloc[0]
        row["error"] = "; ".join(group.error.dropna().astype(str).unique())
        for field in ("max_abs_error", "max_norm_err"):
            row[field] = group[field].max()
        for field in ("determinism_bitwise", "determinism_allclose", "sync_free"):
            values = group[field].astype(str).str.lower()
            row[field] = False if values.eq("false").any() else (True if values.eq("true").any() else "not_checked")
        inners = group.inner.dropna().unique()
        row["inner"] = int(inners[0]) if len(inners) else None
        values = group["elapsed_ms"].dropna().to_numpy(dtype=float)
        row["successful_repeats"] = len(values)
        for key in ("mean_ms", "median_ms", "std_ms", "min_ms", "max_ms", "p25_ms", "p75_ms", "cv"):
            row[key] = np.nan
        if len(values):
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row.update(mean_ms=mean, median_ms=float(np.median(values)), std_ms=std,
                       min_ms=float(values.min()), max_ms=float(values.max()),
                       p25_ms=float(np.percentile(values, 25)), p75_ms=float(np.percentile(values, 75)),
                       cv=std / mean if mean else np.nan)
        summaries.append(row)
    return pd.DataFrame(summaries)


def save_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def save_plots(summary: pd.DataFrame, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved = []
    good = summary[(summary["status"] == "ok") & (summary["correctness"] == "passed")]
    if good.empty:
        print("No successful timings available for plots.")
        return saved
    facets = ["cache_mode", "row_layout", "index_order", "zipf_mode", "scope_group"]
    for facet, subset in good.groupby(facets):
        suffix = "_".join(facet)
        for x, filename, xlabel in [("dim", "median_vs_dimension", "Embedding dimension"),
                                    ("duplicate_rate", "median_vs_duplicate_rate", "Duplicate rate")]:
            fig, ax = plt.subplots(figsize=(10, 6))
            for (op, dist), group in subset.groupby(["operation", "distribution"]):
                ax.scatter(group[x], group["median_ms"], label=f"{op} / {dist}", alpha=0.7)
            ax.set(xlabel=xlabel, ylabel="Median GPU time per invocation (ms)",
                   title=f"{suffix}\nEach point is one configuration")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out_dir / f"{filename}_{suffix}.png", dpi=150)
            saved.append(f"{filename}_{suffix}.png")
            plt.close(fig)
        if facet[-1] == "fair_dense_grad":
            aggregate = subset.pivot_table(index="distribution", columns="operation", values="median_ms", aggfunc="median")
            ax = aggregate.plot.bar(figsize=(10, 6), rot=0)
            ax.set(ylabel="Median of configuration medians (ms)", title=f"{suffix}\nDescriptive aggregate; see winners.csv for matched comparisons")
            ax.figure.tight_layout()
            ax.figure.savefig(out_dir / f"operation_distribution_comparison_{suffix}.png", dpi=150)
            saved.append(f"operation_distribution_comparison_{suffix}.png")
            plt.close(ax.figure)
    return saved


def write_winners(summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    keys = ["rows", "indices", "dim", "distribution", "zipf_mode", "workload_draw",
            "row_layout", "index_order", "cache_mode"]
    ok = summary[(summary.status == "ok") & (summary.correctness == "passed")
                 & (summary.scope_group == "fair_dense_grad")]
    records = []
    for _, group in ok.groupby(keys, sort=True):
        ordered = group.sort_values(["median_ms", "operation"], kind="stable")
        first = ordered.iloc[0]
        record = first[keys + ["p25_ms", "p75_ms", "duplicate_rate", "max_row_count", "top1_share", "top16_share"]].to_dict()
        base = group.loc[group.operation == "index_add", "median_ms"]
        record.update(winner=first.operation, winner_ms=first.median_ms,
                      index_add_ms=float(base.iloc[0]) if len(base) else np.nan,
                      runner_up=None, runner_up_ms=np.nan, within_noise=None,
                      eligible_operations=len(group))
        record["speedup_vs_index_add"] = record["index_add_ms"] / first.median_ms if first.median_ms > 0 else np.nan
        if len(ordered) > 1:
            second = ordered.iloc[1]
            record.update(runner_up=second.operation, runner_up_ms=second.median_ms,
                          within_noise=bool(first.p75_ms >= second.p25_ms))
        records.append(record)
    columns = keys + ["winner", "winner_ms", "p25_ms", "p75_ms", "duplicate_rate", "max_row_count",
                      "top1_share", "top16_share", "index_add_ms", "speedup_vs_index_add",
                      "runner_up", "runner_up_ms", "within_noise", "eligible_operations"]
    winners = pd.DataFrame(records, columns=columns)
    # Always write headers, avoiding a stale winner file when no rows qualify.
    winners.to_csv(out_dir / "winners.csv", index=False)
    print("Winner counts (eligible fair_dense_grad rows only):")
    print(winners.groupby("winner").size().to_string() if len(winners) else "No eligible rows.")
    return winners


def write_winner_summary(winners: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    base_keys = ["rows", "indices", "dim", "distribution", "zipf_mode",
                 "row_layout", "index_order", "cache_mode"]
    columns = base_keys + ["operation", "win_count", "total_draws", "win_rate",
                           "median_speedup_vs_index_add", "mean_speedup_vs_index_add", "median_winner_ms"]
    if winners.empty:
        result = pd.DataFrame(columns=columns)
    else:
        # Denominator includes all eligible winning draws, not only this op's wins.
        denom = winners.groupby(base_keys)["workload_draw"].nunique().rename("total_draws")
        result = winners.groupby(base_keys + ["winner"]).agg(
            win_count=("winner", "size"),
            median_speedup_vs_index_add=("speedup_vs_index_add", "median"),
            mean_speedup_vs_index_add=("speedup_vs_index_add", "mean"),
            median_winner_ms=("winner_ms", "median"),
        ).reset_index().merge(denom.reset_index(), on=base_keys, how="left")
        result["win_rate"] = result["win_count"] / result["total_draws"]
        result = result.rename(columns={"winner": "operation"})[columns]
    result.to_csv(out_dir / "winner_summary.csv", index=False)
    return result


def save_speedup_plots(winners: pd.DataFrame, out_dir: Path) -> list[str]:
    """Use only winners produced by write_winners' fair/passed/ok filtering."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved = []
    if winners.empty:
        return saved
    good = winners[np.isfinite(winners.speedup_vs_index_add) & (winners.speedup_vs_index_add > 0)]
    facets = ["cache_mode", "row_layout", "index_order", "zipf_mode"]
    groups = list(good.groupby(facets))
    for facet, subset in groups:
        settings = ", ".join(f"{name}={value}" for name, value in zip(facets, facet))
        suffix = "_" + "_".join(facet) if len(groups) > 1 else ""
        for x, label in [("top1_share", "Share of indices targeting the hottest row"),
                         ("max_row_count", "Maximum updates to a single row")]:
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, (winner, group) in enumerate(subset.groupby("winner")):
                ax.scatter(group[x], group.speedup_vs_index_add, label=winner,
                           marker=["o", "s", "^", "D", "v"][i % 5], alpha=0.7)
            if x == "max_row_count":
                ax.set_xscale("log")
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
            ax.set(xlabel=label, ylabel="Winner speedup vs index_add",
                   title=f"fair_dense_grad: matched winners per workload draw\n{settings}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            filename = f"speedup_vs_{x}_fair_dense_grad{suffix}.png"
            fig.savefig(out_dir / filename, dpi=150)
            saved.append(filename)
            plt.close(fig)
    return saved


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable. This benchmark requires CUDA-enabled PyTorch and an NVIDIA GPU.", file=sys.stderr)
        return 1
    try:
        device = torch.device(args.device)
        if device.type != "cuda":
            raise ValueError("--device must select CUDA (e.g. cuda or cuda:0)")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        info = get_device_info(device)
    except (ValueError, RuntimeError, AssertionError) as exc:
        print(f"ERROR: Invalid CUDA device: {exc}", file=sys.stderr)
        return 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    total = len(args.rows) * len(args.indices) * len(args.dims) * len(args.distributions) * args.workload_seeds * len(args.ops)
    total = min(total, args.max_combinations) if args.max_combinations else total
    timestamp = datetime.now(timezone.utc).isoformat()
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {**info, "arguments": vars(args), "command_line": sys.argv, "seed": args.seed,
                "run_timestamp": timestamp, "session_id": session_id, "planned_combinations": total,
                "methodology_version": 3,
                "correctness_tolerance": {"formula": "abs_error <= 1e-5 + 1e-6 * row_count"},
                "determinism_allclose_tolerance": {"rtol": RTOL, "atol": ATOL},
                "scope_groups": SCOPE_GROUPS, "op_properties": OP_PROPERTIES,
                "row_layout": args.row_layout, "index_order": args.index_order, "cache_mode": args.cache_mode,
                "zipf_mode": args.zipf_mode, "workload_seeds": args.workload_seeds,
                "workload_draws": list(range(args.workload_seeds)),
                "seed_derivation": {
                    "index_seed": "CRC32(idx|base|rows|count|dist|row_layout|index_order|zipf_mode|workload_draw)",
                    "src_seed": "CRC32(src|base|rows|count|dim|dist|row_layout|index_order|zipf_mode|workload_draw)",
                    "workload_seed": "backward-compatible alias of index_seed"},
                "workloads": [],
                "inner_target_ms": 2.0, "inner_cap": 256,
                "sync_check_requested": args.check_sync_free,
                "sorted_segment_form": "offsets",
                "timing_scope": "CUDA events; input generation and module creation excluded; " + "; ".join(f"{op}: {TIMING_SCOPES[op]}" for op in args.ops),
                "timing_scope_by_operation": TIMING_SCOPES,
                "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
                "status": "running"}
    save_metadata(args.out_dir / "metadata.json", metadata)
    print(f"Device: {info['gpu_name']} ({device}) | PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    print(f"Operations: {', '.join(args.ops)}")
    print(f"Rows: {args.rows}; indices: {args.indices}; dims: {args.dims}; distributions: {args.distributions}")
    print(f"Zipf mode: {args.zipf_mode}; independent workload draws: {args.workload_seeds}")
    print(f"Benchmark combinations: {total}; warmup per round: {args.warmup}; repeats: {args.repeats}; rounds: {args.rounds}; inner: {args.inner}")
    print(f"Results: {args.out_dir.resolve()}", flush=True)
    progress = None
    if args.progress == "on" or (args.progress == "auto" and sys.stderr.isatty()):
        try:
            from tqdm import tqdm
            progress = tqdm(total=total, desc="Benchmark", unit="config")
        except ImportError:
            print("tqdm unavailable; using high-level messages only.")
    completed = 0
    flush_buffer = make_flush_buffer(device) if args.cache_mode == "cold" else None
    metadata["flush_buffer_bytes"] = flush_buffer.numel() * flush_buffer.element_size() if flush_buffer is not None else 0
    save_metadata(args.out_dir / "metadata.json", metadata)
    raw_path = args.out_dir / "raw_results.csv"
    writer = None
    round_repeats = [args.repeats // args.rounds + (r < args.repeats % args.rounds)
                     for r in range(args.rounds)]
    # Only a round's small timing list is buffered. Flush each operation/round,
    # preserving already completed samples if later work fails or is interrupted.
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        configurations = itertools.product(args.rows, args.indices, args.dims, args.distributions,
                                             range(args.workload_seeds))
        for rows, count, dim, distribution, workload_draw in configurations:
            if completed >= total:
                break
            selected_ops = args.ops[:total - completed]
            combination_ids = {op: completed + i for i, op in enumerate(selected_ops)}
            index_seed = index_seed_for(args.seed, rows, count, distribution, args.row_layout,
                                        args.index_order, args.zipf_mode, workload_draw)
            src_seed = src_seed_for(args.seed, rows, count, dim, distribution, args.row_layout,
                                    args.index_order, args.zipf_mode, workload_draw)
            workload_seed = index_seed  # Compatibility column and operation-shuffle seed.
            metadata["workloads"].append({"rows": rows, "indices": count, "dim": dim,
                                          "distribution": distribution, "zipf_mode": args.zipf_mode,
                                          "workload_draw": workload_draw, "index_seed": index_seed,
                                          "src_seed": src_seed, "workload_seed": workload_seed})
            cpu_indices = generate_workload_indices(rows, count, distribution, args, index_seed)
            stats = index_stats(cpu_indices)
            indices = src = reference = tolerance = None
            setup_error = None
            try:
                indices = torch.from_numpy(cpu_indices).to(device)
                generator = torch.Generator(device=device).manual_seed(src_seed)
                src = torch.randn((count, dim), device=device, dtype=torch.float32, generator=generator)
                cpu_reference = torch.zeros((rows, dim), dtype=torch.float64)
                cpu_reference.index_add_(0, torch.from_numpy(cpu_indices), src.cpu().double())
                reference = cpu_reference.to(device=device, dtype=src.dtype)
                del cpu_reference
                counts_row = torch.from_numpy(np.bincount(cpu_indices, minlength=rows)).to(device)
                tolerance = 1e-5 + 1e-6 * counts_row.to(torch.float32)[:, None]
                del counts_row
            except torch.cuda.OutOfMemoryError as exc:
                setup_error = str(exc)
                indices = src = reference = tolerance = None
                torch.cuda.empty_cache()
            states = {op: {} for op in selected_ops}
            for round_index, repeats in enumerate(round_repeats):
                order = list(selected_ops)
                random.Random(workload_seed * 1000003 + round_index).shuffle(order)
                for run_order, operation in enumerate(order):
                    state = states[operation]
                    base = {"timestamp": timestamp, "session_id": session_id,
                            "combination_id": combination_ids[operation],
                            "operation": operation, "rows": rows, "indices": count, "dim": dim,
                            "distribution": distribution, "dtype": args.dtype, "workload_seed": workload_seed,
                            "index_seed": index_seed, "src_seed": src_seed,
                            "zipf_mode": args.zipf_mode, "workload_draw": workload_draw,
                            "scope_group": SCOPE_GROUPS[operation], "timing_scope": TIMING_SCOPES[operation],
                            "has_output": OP_PROPERTIES[operation]["has_output"],
                            "batchable": OP_PROPERTIES[operation]["batchable"],
                            "row_layout": args.row_layout, "index_order": args.index_order,
                            "cache_mode": args.cache_mode, "round_index": round_index, "run_order": run_order,
                            "inner": state.get("inner"), "sync_free": state.get("sync_free", "not_checked"),
                            **stats, "hotspot_fraction": args.hotspot_fraction if distribution == "hotspot" else None,
                            "hotspot_rows": min(rows, args.hotspot_rows) if distribution == "hotspot" else None,
                            "zipf_alpha": args.zipf_alpha if distribution == "zipf" else None,
                            "zipf_mapping": ("finite normalized rank**(-alpha)" if args.zipf_mode == "truncated"
                                             else "(sample - 1) % rows") if distribution == "zipf" else None,
                            "status": "ok", "error": "", "correctness": "not_checked",
                            "max_abs_error": None, "max_norm_err": None,
                            "determinism_bitwise": "not_checked", "determinism_allclose": "not_checked"}
                    times = []
                    if setup_error:
                        base.update(status="oom", error=setup_error)
                    else:
                        try:
                            result = run_benchmark(operation, rows, indices, src, reference, tolerance,
                                                   args, device, repeats, state, flush_buffer)
                            times = result.pop("times")
                            base.update(result)
                        except torch.cuda.OutOfMemoryError as exc:
                            base.update(status="oom", error=str(exc))
                        except (RuntimeError, NotImplementedError) as exc:
                            message = str(exc)
                            if any(term in message.lower() for term in ("device-side assert", "illegal memory", "launch timed out")):
                                raise
                            base.update(status="error", error=message)
                    base.update(inner=state.get("inner"), sync_free=state.get("sync_free", "not_checked"))
                    if base["status"] != "ok":
                        torch.cuda.empty_cache()
                        print(f"\n{operation} {rows}/{count}/{dim}/{distribution}: {base['status']}: {base['error']}", flush=True)
                    repeat_offset = sum(round_repeats[:round_index])
                    for repeat_index, elapsed in enumerate(times or [None]):
                        row = {**base, "repeat_index": repeat_offset + repeat_index if times else -1,
                               "elapsed_ms": elapsed}
                        if writer is None:
                            writer = csv.DictWriter(handle, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                    handle.flush()
                    if round_index == args.rounds - 1 and progress:
                        progress.update(1)
            completed += len(selected_ops)
            del indices, src, reference, tolerance
    if progress:
        progress.close()
    raw = pd.read_csv(raw_path)
    summary = summarize_results(raw)
    summary.to_csv(args.out_dir / "summary_results.csv", index=False)
    winners = write_winners(summary, args.out_dir)
    write_winner_summary(winners, args.out_dir)
    failures = int(((summary.status != "ok") | summary.correctness.eq("failed")).sum())
    metadata.update(status="completed" if failures == 0 else "completed_with_failures",
                    completed_combinations=completed, failed_combinations=failures,
                    end_timestamp=datetime.now(timezone.utc).isoformat())
    save_metadata(args.out_dir / "metadata.json", metadata)
    if args.save_plots:
        metadata["plot_files"] = save_plots(summary, args.out_dir)
        metadata["plot_files"] += save_speedup_plots(winners, args.out_dir)
    else:
        metadata["plot_files"] = []
    save_metadata(args.out_dir / "metadata.json", metadata)
    columns = ["operation", "scope_group", "distribution", "workload_draw", "rows", "indices", "dim", "inner", "median_ms", "correctness"]
    try:
        from tabulate import tabulate
        print(tabulate(summary[columns], headers="keys", tablefmt="simple", showindex=False, floatfmt=".4f"))
    except ImportError:
        print(summary[columns].to_string(index=False))
    print(f"Completed {completed} combinations; failures: {failures}. Saved outputs to {args.out_dir.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
