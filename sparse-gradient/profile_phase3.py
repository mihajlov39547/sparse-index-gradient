"""Representative v3 CUDA profiling subset; event timing and profiler attribution are separate.

Requires sparse_grad_benchmark.py alongside this file (also in Colab).
CSV cache_mode must be warm: cold-cache flushing would contaminate attribution.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch

try:
    import sparse_grad_benchmark as benchmark
except ModuleNotFoundError as exc:
    if exc.name != "sparse_grad_benchmark":
        raise
    raise SystemExit("ERROR: profile_phase3.py requires the methodology-v3 sparse_grad_benchmark.py in the same directory.") from exc


OPS = ["index_add", "scatter_add", "embedding"]
CASE_KEYS = ["rows", "indices", "dim", "distribution", "workload_draw", "zipf_mode",
             "row_layout", "index_order", "cache_mode"]
REQUIRED_V3 = ["index_seed_for", "src_seed_for", "generate_workload_indices", "BENCHMARKS", "TIMING_SCOPES"]


def check_benchmark_compatibility():
    missing = [name for name in REQUIRED_V3 if not hasattr(benchmark, name)]
    if missing:
        raise ValueError("profile_phase3.py requires the methodology-v3 sparse_grad_benchmark.py in the same directory. "
                         + "Missing: " + ", ".join(missing))


def tensor_sha256(tensor):
    """Hash contiguous CPU tensor bytes, outside timing/profiling; no extra bytes copy."""
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("sparse-gradient/phase3_profile"))
    parser.add_argument("--cases", choices=["default", "csv"], default="default")
    parser.add_argument("--case-csv", type=Path)
    parser.add_argument("--ops", nargs="+", choices=OPS, default=OPS)
    for option in ["rows", "indices", "dims"]:
        parser.add_argument(f"--{option}", nargs="+", type=int)
    parser.add_argument("--distributions", nargs="+", choices=["uniform", "hotspot", "zipf"])
    parser.add_argument("--zipf-mode", choices=["truncated", "wrapped"], default="truncated")
    parser.add_argument("--zipf-alpha", type=float, default=1.2)
    parser.add_argument("--workload-draws", nargs="+", type=int, default=[0])
    parser.add_argument("--hotspot-fraction", type=float, default=.95)
    parser.add_argument("--hotspot-rows", type=int, default=8)
    parser.add_argument("--row-layout", choices=["scattered", "adjacent"], default="scattered")
    parser.add_argument("--index-order", choices=["random", "sorted"], default="random")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--profile-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--progress", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--export-trace", action="store_true")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    if args.cases == "csv" and args.case_csv is None:
        parser.error("--case-csv is required with --cases csv")
    if args.warmup < 0 or min(args.repeats, args.profile_repeat) < 1:
        parser.error("warmup must be nonnegative; repeats and profile-repeat must be positive")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("max-cases must be positive")
    if not 0 <= args.seed < 2**32 or any(draw < 0 for draw in args.workload_draws):
        parser.error("seed must be in [0, 2**32); workload draws must be nonnegative")
    if not np.isfinite(args.zipf_alpha) or args.zipf_alpha <= 1:
        parser.error("zipf-alpha must be finite and > 1")
    if not 0 <= args.hotspot_fraction <= 1 or args.hotspot_rows < 1:
        parser.error("hotspot-fraction must be in [0,1]; hotspot-rows must be positive")
    if any(v <= 0 for key in ["rows", "indices", "dims"] for v in (getattr(args, key) or [])):
        parser.error("size overrides must be positive")
    args.ops = list(dict.fromkeys(args.ops))
    # CSV draw IDs are preserved unless this flag was explicitly supplied.
    args.draws_explicit = any(v == "--workload-draws" or v.startswith("--workload-draws=") for v in sys.argv[1:])
    return args


def default_cases():
    return [dict(zip(["rows", "indices", "dim", "distribution", "workload_draw"], row)) for row in [
        (4096, 65536, 64, "zipf", 0),
        (4096, 262144, 512, "uniform", 0),
        (16384, 65536, 256, "zipf", 0),
        (4096, 65536, 64, "hotspot", 0),
        (65536, 262144, 256, "zipf", 0),
        (4096, 262144, 64, "hotspot", 0),
    ]]


def load_cases_csv(path):
    frame = pd.read_csv(path)
    required = {"rows", "indices", "dim", "distribution", "workload_draw"}
    if not required.issubset(frame):
        raise ValueError(f"Case CSV missing columns: {sorted(required - set(frame))}")
    return frame.to_dict("records")


def resolve_cases(args):
    templates = default_cases() if args.cases == "default" else load_cases_csv(args.case_csv)
    result, seen = [], set()
    for template in templates:
        case = {key: value for key, value in template.items() if key in CASE_KEYS and pd.notna(value)}
        for key in ["zipf_mode", "row_layout", "index_order"]:
            case.setdefault(key, getattr(args, key))
        case.setdefault("cache_mode", "warm")
        # Explicit size overrides expand each template, deduplicated afterwards.
        values = [args.rows or [case["rows"]], args.indices or [case["indices"]],
                  args.dims or [case["dim"]], args.distributions or [case["distribution"]],
                  args.workload_draws if args.cases == "default" or args.draws_explicit else [case["workload_draw"]]]
        for sizes in itertools.product(*values):
            item = case | dict(zip(["rows", "indices", "dim", "distribution", "workload_draw"], sizes))
            for key in ["rows", "indices", "dim", "workload_draw"]:
                value = float(item[key])
                if not np.isfinite(value) or value != int(value) or value < (0 if key == "workload_draw" else 1):
                    raise ValueError(f"Invalid case {key}: {item[key]}")
                item[key] = int(value)
            for key, choices in [("distribution", ["uniform", "hotspot", "zipf"]),
                                 ("zipf_mode", ["truncated", "wrapped"]), ("row_layout", ["scattered", "adjacent"]),
                                 ("index_order", ["random", "sorted"]), ("cache_mode", ["warm"])]:
                if item[key] not in choices:
                    raise ValueError(f"Invalid {key}={item[key]}; profiler subset supports {choices}")
            key = tuple(item[k] for k in CASE_KEYS)
            if key not in seen:
                item["case_id"] = f"case{len(result)+1:02d}"
                result.append(item)
                seen.add(key)
    if not result:
        raise ValueError("No cases selected")
    return result[:args.max_cases] if args.max_cases else result


def get_environment_metadata(args, device, cases):
    return benchmark.get_device_info(device) | {
        "timestamp": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
        "benchmark_compatibility_check": "passed", "required_v3_symbols": REQUIRED_V3,
        "benchmark_module": str(Path(benchmark.__file__).resolve()),
        "workload_hashes": "SHA256 of contiguous CPU tensor bytes (int64 indices, float32 src), outside timing/profiling. Source hashes may differ across CUDA/PyTorch stacks despite equal seeds.",
        "gpu_busy_proxy": "(profile CUDA kernel milliseconds / profile_repeat) / unprofiled median_ms; below 0.6 flags possible host overhead. Not hardware utilization; can exceed 1 or be distorted by profiler timing.",
        "command_line": sys.argv, "methodology_version": 3, "cases": cases,
        "status": "running", "dtype": "float32", "scope_group": "fair_dense_grad",
        "timing_scopes": {op: benchmark.TIMING_SCOPES[op] for op in args.ops},
        "timing": "Unprofiled CUDA events, inner=1; setup, forward and prepare outside events. Profiler overhead is not primary timing.",
        "profiler_settings": {"activities": ["CPU", "CUDA"], "profile_memory": True,
                              "record_shapes": args.record_shapes, "with_stack": args.with_stack,
                              "profile_repeat": args.profile_repeat, "export_trace": args.export_trace},
        "profiler_total_definitions": {
            "total_profile_cpu_time_ms": "Sum of CPU self times; avoids adding inclusive parent and child times.",
            "total_profile_cuda_time_ms": "Sum of CUDA kernel-event durations, or CPU operator self-device times in legacy fallback; not wall time.",
            "number_cuda_ops": "CUDA event call count; fallback counts CPU operators with attributed CUDA time, not kernels.",
            "number_cpu_ops": "CPU event call count, including runtime and profiling annotations.",
            "top_cuda": "Top 10 CUDA kernel keys when present; otherwise CUDA-attributed CPU keys, else CPU-time fallback explicitly marked."},
        "correctness": "CPU float64 reference cast to float32; abs_error <= 1e-5 + 1e-6 * row_count",
        "limitations": ["Warm-cache subset only; CSV cache_mode=cold is rejected.",
                        "CUDA-event samples are unbatched and fixed op order, unlike randomized/batched Phase 2 confirmation.",
                        "Profiler totals span all profile_repeat invocations, not one invocation.",
                        "Key-average inclusive times can overlap; do not sum them as total runtime.",
                        "No profile attribution is claimed if CUDA time is unavailable."]}


def generate_indices_v3(case, args):
    seed = benchmark.index_seed_for(args.seed, case["rows"], case["indices"], case["distribution"],
                                    case["row_layout"], case["index_order"], case["zipf_mode"], case["workload_draw"])
    settings = argparse.Namespace(**(vars(args) | case))
    return benchmark.generate_workload_indices(case["rows"], case["indices"], case["distribution"], settings, seed), seed


def generate_src_v3(case, args, device):
    seed = benchmark.src_seed_for(args.seed, case["rows"], case["indices"], case["dim"], case["distribution"],
                                  case["row_layout"], case["index_order"], case["zipf_mode"], case["workload_draw"])
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn((case["indices"], case["dim"]), generator=generator, device=device, dtype=torch.float32), seed


def build_operation(operation, rows, indices, src):
    # Fair index/scatter allocate+zero inside run. Embedding keeps one forward
    # graph; prepare clears grad=None outside the event interval, as in v3.
    return benchmark.BENCHMARKS[operation](rows, indices, src)


def time_operation_cuda_events(prepare, run, device, warmup, repeats):
    for _ in range(warmup):
        prepare()
        run()
    torch.cuda.synchronize(device)
    values = np.asarray([benchmark.time_batch(prepare, run, device, inner=1) for _ in range(repeats)])
    mean = values.mean()
    return {"median_ms": float(np.median(values)), "p25_ms": float(np.quantile(values, .25)),
            "p75_ms": float(np.quantile(values, .75)),
            "cv": float(values.std(ddof=1)/mean) if len(values) > 1 and mean else 0.,
            "event_samples_ms": json.dumps(values.tolist())}


def profiler_field(event, names):
    for name in names:
        try:
            value = getattr(event, name)
            if value is not None:
                return float(value), name
        except (AttributeError, RuntimeError):
            continue
    return np.nan, "unavailable"


def summarize_profiler_events(prof, record_shapes):
    records = []
    for event in prof.key_averages(group_by_input_shape=record_shapes):
        cuda_total, cuda_field = profiler_field(event, ["device_time_total", "cuda_time_total"])
        cuda_self, _ = profiler_field(event, ["self_device_time_total", "self_cuda_time_total"])
        cuda_avg, _ = profiler_field(event, ["device_time", "cuda_time"])
        cuda_memory, _ = profiler_field(event, ["device_memory_usage", "cuda_memory_usage"])
        records.append({"key": event.key, "device_type": str(getattr(event, "device_type", "unknown")),
                        "is_user_annotation": bool(getattr(event, "is_user_annotation", False)),
                        "self_cpu_time_total_us": float(event.self_cpu_time_total),
                        "cpu_time_total_us": float(event.cpu_time_total),
                        "cuda_time_total_us": cuda_total, "self_cuda_time_total_us": cuda_self,
                        "cuda_time_avg_us": cuda_avg, "calls": int(event.count),
                        "input_shapes": json.dumps(getattr(event, "input_shapes", None)),
                        "cpu_memory_usage_bytes": getattr(event, "cpu_memory_usage", np.nan),
                        "cuda_memory_usage_bytes": cuda_memory, "cuda_time_field": cuda_field})
    events = pd.DataFrame(records)
    if events.empty:
        raise RuntimeError("Profiler returned no events")
    # Some Kineto versions emit GPU-side user ranges as CUDA events. They span
    # their child kernels and must not be counted again or ranked as kernels.
    cuda = events[events.device_type.str.upper().str.endswith("CUDA")
                  & ~events.is_user_annotation & ~events.key.str.startswith("phase3::")]
    cpu = events[events.device_type.str.upper().str.endswith("CPU")]
    cpu_ops = cpu[~cpu.is_user_annotation & ~cpu.key.str.startswith("phase3::")]
    if not cuda.empty and cuda.cuda_time_total_us.sum() > 0:
        candidates, basis = cuda, "cuda_kernel_events"
        total_cuda = float(cuda.cuda_time_total_us.sum()/1000)
        cuda_calls = int(cuda.calls.sum())
    elif cpu_ops.cuda_time_total_us.fillna(0).gt(0).any():
        candidates, basis = cpu_ops[cpu_ops.cuda_time_total_us > 0], "cuda_attributed_cpu_operators"
        total_cuda = float(cpu_ops.self_cuda_time_total_us.sum(min_count=1)/1000)
        cuda_calls = int(candidates.calls.sum())
    else:
        candidates, basis = cpu_ops, "cpu_total_fallback"
        total_cuda, cuda_calls = np.nan, 0
    sort_column = "cpu_time_total_us" if basis == "cpu_total_fallback" else "cuda_time_total_us"
    top = candidates.sort_values(sort_column, ascending=False).head(10).copy()
    top["sort_basis"] = basis
    result = {"profile_sort_basis": basis, "total_profile_cuda_time_ms": total_cuda,
              "total_profile_cpu_time_ms": float(cpu.self_cpu_time_total_us.sum()/1000),
              "number_cuda_ops": cuda_calls, "number_cpu_ops": int(cpu.calls.sum()),
              "cuda_kernel_events_available": basis == "cuda_kernel_events"}
    for rank in range(1, 4):
        item = top.iloc[rank-1] if rank <= len(top) else None
        # CPU fallback remains visible in top CSV, never mislabeled as CUDA.
        result[f"top_cuda_operator_{rank}"] = item["key"] if item is not None and basis != "cpu_total_fallback" else None
        result[f"top_cuda_operator_{rank}_time_ms"] = float(item.cuda_time_total_us/1000) if item is not None and basis != "cpu_total_fallback" else np.nan
    return events, top, result


def profile_operation(prepare, run, device, args, case_id, operation):
    torch.cuda.synchronize(device)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=args.record_shapes, with_stack=args.with_stack, profile_memory=True) as prof:
            for _ in range(args.profile_repeat):
                prepare()
                with torch.profiler.record_function(f"phase3::{operation}"):
                    run()
                torch.cuda.synchronize(device)
        events, top, result = summarize_profiler_events(prof, args.record_shapes)
        if args.export_trace:
            prof.export_chrome_trace(str(args.out_dir / "traces" / f"trace_{case_id}_{operation}.json"))
    events.to_csv(args.out_dir / "events" / f"profile_events_{case_id}_{operation}.csv", index=False)
    top.to_csv(args.out_dir / "events" / f"profile_top_cuda_{case_id}_{operation}.csv", index=False)
    result["profiler_warnings"] = " | ".join(dict.fromkeys(str(w.message) for w in captured))
    return result


def run_one(case, operation, indices, src, reference, tolerance, device, args):
    prepare, run, output = build_operation(operation, case["rows"], indices, src)
    prepare()
    run()
    torch.cuda.synchronize(device)
    value = output().detach()
    valid = value.shape == reference.shape and bool(torch.isfinite(value).all())
    error = (value-reference).abs() if valid else None
    correct = valid and bool((error <= tolerance).all())
    record = {"correctness": "passed" if correct else "failed",
              "max_abs_error": float(error.max()) if valid else np.inf,
              "max_norm_err": float((error/tolerance).max()) if valid else np.inf}
    del value, error
    if not correct:
        return record | {"status": "failed_correctness", "notes": "Profiling and timing skipped after failed CPU reference check."}
    record.update(time_operation_cuda_events(prepare, run, device, args.warmup, args.repeats))
    try:
        record.update(profile_operation(prepare, run, device, args, case["case_id"], operation))
        total_cuda = record["total_profile_cuda_time_ms"]
        if record["profile_sort_basis"] == "cuda_kernel_events" and np.isfinite(total_cuda):
            record["kernel_ms_per_invocation"] = total_cuda / args.profile_repeat
            if np.isfinite(record["median_ms"]) and record["median_ms"] > 0:
                record["gpu_busy_fraction"] = record["kernel_ms_per_invocation"] / record["median_ms"]
                record["likely_host_bound"] = bool(record["gpu_busy_fraction"] < .6)
        record["status"] = "ok"
        record["notes"] = record.get("profiler_warnings", "")
        if record["profile_sort_basis"] != "cuda_kernel_events":
            record["notes"] += " CUDA kernel names unavailable; " + record["profile_sort_basis"] + "."
    except (RuntimeError, OSError) as exc:
        record.update(status="profile_error", notes=str(exc))
    return record


def annotate_winners(results, args):
    winners = []
    for case_id, group in results.groupby("case_id", sort=False):
        good = group[(group.correctness == "passed") & group.median_ms.notna()].sort_values(["median_ms", "operation"])
        complete = len(good) == len(args.ops)
        first = good.iloc[0] if len(good) else None
        second = good.iloc[1] if len(good) > 1 else None
        ratio = float(second.median_ms/first.median_ms) if second is not None else np.nan
        row = {"case_id": case_id, "winner": first.operation if first is not None else None,
               "runner_up": second.operation if second is not None else None, "margin_ratio": ratio,
               "decisive_5pct": bool(complete and ratio > 1.05), "complete_ops": complete,
               "eligible_operations": len(good)}
        row.update({f"{op}_ms": float(good.loc[good.operation == op, "median_ms"].iloc[0]) if op in set(good.operation) else np.nan for op in args.ops})
        winners.append(row)
        results.loc[group.index, "winner_within_case"] = row["winner"]
        results.loc[group.index, "runner_up"] = row["runner_up"]
        results.loc[group.index, "margin_ratio"] = ratio
    return results, pd.DataFrame(winners)


def markdown(frame):
    def fmt(value):
        if pd.isna(value):
            return "-"
        if isinstance(value, float):
            return f"{value:.5g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    if frame.empty:
        return "No data available."
    return "\n".join(["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"] +
                     ["| " + " | ".join(fmt(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)])


def write_case_report(results, winners, metadata, args):
    top_rows = []
    for _, row in results.iterrows():
        for rank in range(1, 4):
            key = row.get(f"top_cuda_operator_{rank}")
            if pd.notna(key):
                top_rows.append({"case_id": row.case_id, "operation": row.operation, "rank": rank,
                                 "CUDA key": key, "total_ms": row.get(f"top_cuda_operator_{rank}_time_ms"),
                                 "attribution": row.get("profile_sort_basis")})
    observations = []
    for row in winners.itertuples():
        observations.append(f"- {row.case_id}: raw event-timing minimum `{row.winner}`, runner-up `{row.runner_up}`, ratio {row.margin_ratio:.3g}; exceeds 5% within this subset: {row.decisive_5pct}. Complete selected API set: {row.complete_ops}.")
    for op, group in results.groupby("operation"):
        keys = group.get("top_cuda_operator_1", pd.Series(dtype=str)).dropna().value_counts()
        if len(keys):
            observations.append(f"- `{op}`: most frequent highest-time CUDA key in this subset was `{keys.index[0]}` ({keys.iloc[0]} cases).")
    notes = results.loc[results.notes.fillna("").ne(""), ["case_id", "operation", "status", "notes"]]
    text = ["# Phase 3: Representative Sparse-Index Gradient Profiles", "## Environment",
            markdown(pd.DataFrame([{key: metadata[key] for key in ["gpu_name", "compute_capability", "pytorch_version", "cuda_version", "python_version", "platform"]}]).assign(compute_capability=str(metadata["compute_capability"]))),
            "## Cases", markdown(results[["case_id"] + CASE_KEYS].drop_duplicates()),
            "## Workload audit", markdown(results[["case_id", "hotspot_fraction", "hotspot_rows", "zipf_alpha", "seed", "index_sha256", "src_sha256"]].drop_duplicates()),
            "Hashes use contiguous CPU tensor bytes (int64 indices, float32 source), computed before timing/profiling. They check exact data reproducibility; source values may differ between PyTorch/CUDA versions even with the same seed. Workload shape and settings must also match.",
            "## Unprofiled CUDA-event timing for profiled subset", markdown(winners),
            "Timings use fresh dense outputs for index_add/scatter_add and retained-graph backward for embedding. Embedding forward is outside timing; grad=None clearing is in prepare, matching v3. Samples use inner=1 and fixed operation order, so these subset decisions are not a replacement for randomized/batched Phase 2 confirmation.",
            "## Profiler-derived GPU-busy proxy", markdown(results[["case_id", "operation", "kernel_ms_per_invocation", "gpu_busy_fraction", "likely_host_bound"]]),
            "gpu_busy_fraction = (total profiled CUDA kernel time / profile_repeat) / unprofiled median_ms. This is a profiler-derived proxy, not a hardware utilization counter. It can exceed 1 or be distorted when profiler/kernel timing differs from unprofiled event timing. Use it only as a host-overhead warning signal: likely_host_bound flags values below 0.6, not a causal conclusion. Values are unavailable when CUDA kernel events are unavailable. The fraction is not clamped.",
            "## Top CUDA attribution", markdown(pd.DataFrame(top_rows)),
            f"Times in this table total all {args.profile_repeat} profiled invocations. Full CPU/operator/kernel key averages and top-10 tables are under events/. CUDA kernel keys are preferred; if unavailable, attributed CPU operators are used, or an explicitly labeled CPU-time fallback. Inclusive parent/child times must not be added. Aggregate CPU totals sum self times; CUDA totals sum kernel durations when available, not wall time.",
            "## Observations", "\n".join(observations), "## Warnings and failures", markdown(notes),
            "## Limitations",
            "Profiler explains operator/kernel composition, not universal causality. Results are per environment. CPU and CUDA profiler times include instrumentation and synchronization overhead and must not replace unprofiled CUDA-event timings. This is a representative subset, not a full grid or end-to-end model training. All profiled invocations synchronize. Warm cache only; record-shapes, memory, and stack capture can perturb execution. CUDA kernel attribution depends on CUPTI/Kineto support and permissions. Memory freed from allocations made before profiler start may not be attributable. Negative memory deltas are possible. There is no custom CUDA/C++ code.",
            "For Colab, copy both profile_phase3.py and sparse_grad_benchmark.py into the same directory. Reuse the same cases and parameters. Optional --export-trace creates traces/; --record-shapes and --with-stack are off by default. CSV setting columns override CLI defaults; explicit size/draw flags expand templates and duplicates are removed before --max-cases is applied."]
    (args.out_dir / "profile_phase3_report.md").write_text("\n\n".join(text)+"\n", encoding="utf-8")


def main():
    args = parse_args()
    try:
        check_benchmark_compatibility()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("Methodology-v3 benchmark compatibility check: passed", flush=True)
    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable; this profiler requires CUDA-enabled PyTorch and an NVIDIA GPU.", file=sys.stderr)
        return 1
    try:
        cases = resolve_cases(args)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    device = torch.device("cuda", torch.cuda.current_device())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "events").mkdir(exist_ok=True)
    if args.export_trace:
        (args.out_dir / "traces").mkdir(exist_ok=True)
    metadata = get_environment_metadata(args, device, cases)
    benchmark.save_metadata(args.out_dir / "metadata.json", metadata)
    print(f"GPU: {metadata['gpu_name']}; PyTorch {torch.__version__}; CUDA {torch.version.cuda}")
    print(f"Cases: {len(cases)}; operations: {args.ops}; event repeats: {args.repeats}; profiler invocations: {args.profile_repeat}", flush=True)
    progress = None
    if args.progress == "on" or (args.progress == "auto" and sys.stderr.isatty()):
        try:
            from tqdm import tqdm
            progress = tqdm(total=len(cases)*len(args.ops), desc="Profile", unit="case/op")
        except ImportError:
            print("tqdm unavailable; using high-level messages.")
    records = []
    for case in cases:
        indices = src = reference = tolerance = None
        index_seed = src_seed = None
        index_sha256 = src_sha256 = None
        setup_error = None
        try:
            cpu_indices, index_seed = generate_indices_v3(case, args)
            indices = torch.from_numpy(cpu_indices).to(device)
            src, src_seed = generate_src_v3(case, args, device)
            index_sha256 = tensor_sha256(indices)
            cpu_src = src.detach().cpu()
            src_sha256 = tensor_sha256(cpu_src)
            cpu_reference = torch.zeros((case["rows"], case["dim"]), dtype=torch.float64)
            cpu_reference.index_add_(0, torch.from_numpy(cpu_indices), cpu_src.double())
            del cpu_src
            reference = cpu_reference.to(device=device, dtype=torch.float32)
            del cpu_reference
            counts = torch.from_numpy(np.bincount(cpu_indices, minlength=case["rows"])).to(device)
            tolerance = 1e-5 + 1e-6 * counts.float()[:, None]
            del counts
        except torch.cuda.OutOfMemoryError as exc:
            setup_error = str(exc)
        for op in args.ops:
            record = case | {"operation": op, "index_seed": index_seed, "src_seed": src_seed,
                             "hotspot_fraction": args.hotspot_fraction, "hotspot_rows": args.hotspot_rows,
                             "zipf_alpha": args.zipf_alpha, "seed": args.seed,
                             "index_sha256": index_sha256, "src_sha256": src_sha256,
                             "kernel_ms_per_invocation": np.nan, "gpu_busy_fraction": np.nan,
                             "likely_host_bound": "unavailable",
                             "scope_group": "fair_dense_grad", "timing_scope": benchmark.TIMING_SCOPES[op],
                             "correctness": "not_checked", "median_ms": np.nan, "p25_ms": np.nan,
                             "p75_ms": np.nan, "cv": np.nan, "status": "error", "notes": ""}
            record.update({"profile_sort_basis": "not_profiled", "total_profile_cuda_time_ms": np.nan,
                           "total_profile_cpu_time_ms": np.nan, "number_cuda_ops": 0, "number_cpu_ops": 0,
                           "cuda_kernel_events_available": False})
            for rank in range(1, 4):
                record[f"top_cuda_operator_{rank}"] = None
                record[f"top_cuda_operator_{rank}_time_ms"] = np.nan
            try:
                if setup_error:
                    record.update(status="oom", notes=setup_error)
                else:
                    record.update(run_one(case, op, indices, src, reference, tolerance, device, args))
            except (RuntimeError, NotImplementedError) as exc:
                record.update(status="oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error", notes=str(exc))
            records.append(record)
            # Checkpoint completed operations even if a later profile fails.
            pd.DataFrame(records).to_csv(args.out_dir / "profile_cases.csv", index=False)
            if record["status"] != "ok":
                print(f"{case['case_id']} {op}: {record['status']}: {record['notes']}", flush=True)
                torch.cuda.empty_cache()
            if progress:
                progress.update(1)
        del indices, src, reference, tolerance
    if progress:
        progress.close()
    results, winners = annotate_winners(pd.DataFrame(records), args)
    results.to_csv(args.out_dir / "profile_cases.csv", index=False)
    winners.to_csv(args.out_dir / "case_winners.csv", index=False)
    failures = int(results.status.ne("ok").sum())
    metadata.update(status="completed" if failures == 0 else "completed_with_failures", completed_case_ops=len(results),
                    failed_case_ops=failures, failed_correctness=int(results.correctness.eq("failed").sum()),
                    end_timestamp=datetime.now(timezone.utc).isoformat(),
                    profiler_sort_basis_by_case_op=results[["case_id", "operation", "profile_sort_basis"]].to_dict("records") if "profile_sort_basis" in results else [],
                    profiler_warnings=results.loc[results.notes.fillna("").ne(""), ["case_id", "operation", "notes"]].to_dict("records"))
    benchmark.save_metadata(args.out_dir / "metadata.json", metadata)
    write_case_report(results, winners, metadata, args)
    print(winners.to_string(index=False))
    print(f"Completed {len(cases)} cases / {len(results)} case-ops; failures={failures}; outputs: {args.out_dir.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
