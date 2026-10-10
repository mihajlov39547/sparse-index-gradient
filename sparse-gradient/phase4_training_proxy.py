"""Phase 4: Mini End-to-End Training-Step Proxy (not production training)."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
import uuid
import zlib

import numpy as np
import torch
from torch.nn import functional as F


VARIANTS = ("embedding_backward_proxy", "index_add_proxy", "scatter_add_proxy")
VALIDATION_FIELDS = (
    "validation_passed", "max_abs_error", "max_relative_error",
    "max_abs_error_any_check", "max_relative_error_any_check",
    "max_abs_error_over_reference_max", "relative_l2_error_table", "failed_check_names",
)
PROFILES = {
    "smoke": dict(rows=[32], num_indices=[64], dim=[8], warmup=1, repeats=2),
    "pilot": dict(rows=[4096], num_indices=[8192], dim=[32, 128], warmup=3, repeats=10),
    "study": dict(rows=[4096, 65536], num_indices=[8192, 65536], dim=[32, 128],
                  warmup=10, repeats=40),
}
SCOPES = {
    "proxy_step": "fresh lookup, mean pooling, linear head, MSE, autograd gradients for table/head/bias; includes dense gradient allocation and zeroing",
    "accumulation_region": "separate retained lookup graph VJP with fixed upstream gradient; includes autograd dispatch, dense output allocation/zeroing and accumulation; excludes forward/head/loss",
}


class ManualLookup(torch.autograd.Function):
    """Same differentiable lookup, replacing only its dense row-gradient VJP."""

    @staticmethod
    def forward(ctx, table, indices, expanded_indices, method):
        ctx.save_for_backward(indices, expanded_indices)
        ctx.shape = table.shape
        ctx.method = method
        return torch.index_select(table, 0, indices)

    @staticmethod
    def backward(ctx, source):
        indices, expanded = ctx.saved_tensors
        result = source.new_zeros(ctx.shape)
        if ctx.method == "index_add_proxy":
            result.index_add_(0, indices, source)
        else:
            result.scatter_add_(0, expanded, source)
        return result, None, None, None


def lookup(w, variant):
    if variant == "embedding_backward_proxy":
        return F.embedding(w["indices"], w["table"], sparse=False, scale_grad_by_freq=False)
    return ManualLookup.apply(w["table"], w["indices"], w["expanded"], variant)


def head_loss(vectors, w):
    pooled = vectors.reshape(-1, w["pool_size"], vectors.shape[1]).mean(dim=1)
    prediction = pooled @ w["head"] + w["bias"]
    return (prediction - w["targets"]).square().mean()


def proxy_step(w, variant):
    loss = head_loss(lookup(w, variant), w)
    # autograd.grad returns fresh gradients; no .grad buffers accumulate between steps.
    grads = torch.autograd.grad(loss, (w["table"], w["head"], w["bias"]))
    return loss.detach(), grads


def seed_for(seed, *parts):
    return zlib.crc32("|".join(map(str, (seed,) + parts)).encode())


def generate_workload(config, args, device):
    rows, count, dim, dist = (config[k] for k in ("rows", "num_indices", "dim", "distribution"))
    index_seed = seed_for(args.seed, rows, count, dist, "indices")
    value_seed = seed_for(args.seed, rows, count, dim, dist, "values")
    rng = np.random.default_rng(index_seed)
    if dist == "uniform":
        indices = rng.integers(rows, size=count, dtype=np.int64)
    elif dist == "zipf":
        p = np.arange(1, rows + 1, dtype=np.float64) ** -1.2
        indices = rng.choice(rows, count, p=p / p.sum()).astype(np.int64)
    else:
        hot_rows = min(16, rows)
        hot_count = round(0.8 * count)
        indices = np.concatenate((rng.integers(hot_rows, size=hot_count),
                                  rng.integers(hot_rows if hot_rows < rows else 0,
                                               rows, size=count - hot_count))).astype(np.int64)
        rng.shuffle(indices)
    indices = rng.permutation(rows)[indices]  # scattered row layout, random reference order
    gen = torch.Generator(device="cpu").manual_seed(value_seed)
    table = torch.randn(rows, dim, generator=gen) * 0.1
    head = torch.randn(dim, generator=gen) / math.sqrt(dim)
    targets = torch.randn(count // args.pool_size, generator=gen) * 0.1
    idx = torch.from_numpy(indices).to(device)
    w = dict(table=table.to(device).requires_grad_(), head=head.to(device).requires_grad_(),
             bias=torch.zeros((), device=device, requires_grad=True), indices=idx,
             targets=targets.to(device), pool_size=args.pool_size,
             expanded=idx[:, None].expand(-1, dim))
    # Independent CPU float64 analytic reference, generated entirely outside timing.
    with torch.no_grad():
        t, h, y = table.double(), head.double(), targets.double()
        pooled = t[indices].reshape(-1, args.pool_size, dim).mean(1)
        residual = pooled @ h - y
        dp = 2 * residual / len(y)
        source = (dp[:, None] * h[None, :] / args.pool_size).repeat_interleave(args.pool_size, 0)
        reference = torch.zeros(rows, dim, dtype=torch.float64)
        reference.index_add_(0, torch.from_numpy(indices), source)
        references = (reference, pooled.T @ dp, dp.sum())
        reference_loss = residual.square().mean()
    # Region source is the actual float32 upstream gradient from the matched head.
    vectors = w["table"].detach().index_select(0, idx).requires_grad_()
    w["source"] = torch.autograd.grad(head_loss(vectors, w), vectors)[0].detach()
    with torch.no_grad():
        region_reference = torch.zeros_like(reference)
        region_reference.index_add_(0, torch.from_numpy(indices), w["source"].cpu().double())
    info = dict(index_seed=index_seed, value_seed=value_seed,
                index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                unique_rows=int(np.unique(indices).size),
                max_row_count=int(np.bincount(indices, minlength=rows).max()))
    return w, references, reference_loss, region_reference, info


def make_region(w, variant):
    vectors = lookup(w, variant)

    def region():
        return torch.autograd.grad(vectors, w["table"], w["source"], retain_graph=True)[0]

    return region


def compare(actual, reference, atol, rtol):
    actual = actual.detach().cpu().double()
    reference = reference.detach().cpu().double()
    shape_ok = actual.shape == reference.shape
    reference_finite = bool(torch.isfinite(reference).all())
    finite = bool(torch.isfinite(actual).all()) and reference_finite
    reference_max = float(reference.abs().max()) if reference_finite else None
    # Flatten so vector norms also handle scalar loss and bias tensors.
    reference_norm = float(torch.linalg.vector_norm(reference.reshape(-1))) if reference_finite else None
    diagnostics = dict(reference_max_abs=reference_max, reference_l2_norm=reference_norm,
                       error_l2_norm=None, relative_l2_error=None,
                       max_abs_error_over_reference_max=None)
    if not shape_ok or not finite:
        return dict(passed=False, shape_ok=shape_ok, finite=finite,
                    max_abs_error=None, max_relative_error=None, **diagnostics)
    error = (actual - reference).abs()
    max_error = float(error.max())
    error_norm = float(torch.linalg.vector_norm(error.reshape(-1)))
    diagnostics.update(error_l2_norm=error_norm,
                       relative_l2_error=error_norm / reference_norm if reference_norm > 0 else None,
                       max_abs_error_over_reference_max=max_error / reference_max if reference_max > 0 else None)
    mask = reference.abs() > atol
    return dict(passed=bool(torch.all(error <= atol + rtol * reference.abs())),
                shape_ok=True, finite=True, max_abs_error=max_error,
                max_relative_error=float((error[mask] / reference.abs()[mask]).max()) if mask.any() else None,
                **diagnostics)


def validate(w, variant, refs, loss_ref, region_ref, args):
    loss, grads = proxy_step(w, variant)
    checks = {name: compare(value, ref, args.atol, args.rtol)
              for name, value, ref in zip(("table", "head", "bias"), grads, refs)}
    checks["loss"] = compare(loss, loss_ref, args.atol, args.rtol)
    checks["accumulation_region"] = compare(make_region(w, variant)(), region_ref, args.atol, args.rtol)

    def max_available(field):
        return max((c[field] for c in checks.values() if c[field] is not None), default=None)

    return dict(variant=variant, validation_passed=all(c["passed"] for c in checks.values()),
                max_abs_error=checks["table"]["max_abs_error"],
                max_relative_error=checks["table"]["max_relative_error"],
                max_abs_error_any_check=max_available("max_abs_error"),
                max_relative_error_any_check=max_available("max_relative_error"),
                max_abs_error_over_reference_max=max_available("max_abs_error_over_reference_max"),
                relative_l2_error_table=checks["table"]["relative_l2_error"],
                failed_check_names=";".join(name for name, c in checks.items() if not c["passed"]),
                checks=checks)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_once(fn, device, events):
    synchronize(device)
    if events:
        start, end = events
        start.record()
        output = fn()
        end.record()
        synchronize(device)
        elapsed = start.elapsed_time(end)
    else:
        start = time.perf_counter_ns()
        output = fn()
        elapsed = (time.perf_counter_ns() - start) / 1e6
    del output  # do not retain previous-step gradient tensors in subsequent samples
    return float(elapsed)


def time_variants(w, args, device, order_rng):
    samples = []
    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if device.type == "cuda" else None
    # Separate passes: no internal events/hooks perturb total-step measurements.
    for scope in SCOPES:
        functions = {v: (lambda v=v: proxy_step(w, v)) if scope == "proxy_step"
                     else make_region(w, v) for v in VARIANTS}
        for _ in range(args.warmup):
            for v in order_rng.sample(list(VARIANTS), len(VARIANTS)):
                functions[v]()
        synchronize(device)
        for repeat in range(args.repeats):
            for order, v in enumerate(order_rng.sample(list(VARIANTS), len(VARIANTS))):
                ms = time_once(functions[v], device, events)
                if not math.isfinite(ms) or ms <= 0:
                    raise RuntimeError(f"Invalid timing sample: {v}/{scope}: {ms}")
                samples.append(dict(variant=v, scope=scope, repeat=repeat, order=order, latency_ms=ms))
    return samples


def statistics(values):
    values = np.asarray(values, dtype=float)
    return dict(median_ms=float(np.median(values)),
                iqr_ms=float(np.percentile(values, 75) - np.percentile(values, 25)),
                cv=float(values.std(ddof=1) / values.mean()) if len(values) > 1 else None)


def summarize(samples, validations):
    rows = []
    for v in VARIANTS:
        validation = next(item for item in validations if item["variant"] == v)
        row = {k: validation[k] for k in ("variant",) + VALIDATION_FIELDS}
        for scope in SCOPES:
            row.update({f"{scope}_{k}": value for k, value in statistics(
                [s["latency_ms"] for s in samples if s["variant"] == v and s["scope"] == scope]).items()})
        rows.append(row)
    for scope in SCOPES:
        ranked = sorted(rows, key=lambda r: r[f"{scope}_median_ms"])
        margin = ranked[1][f"{scope}_median_ms"] / ranked[0][f"{scope}_median_ms"] - 1
        winner = ranked[0]["variant"] if margin > 0.05 else "tie"
        for row in rows:
            row[f"winner_{scope}"] = winner
            row[f"practical_margin_{scope}"] = margin
    return rows


def write_csv(path, rows, empty_fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else empty_fields)
        writer.writeheader()
        writer.writerows(rows)


def export(out, raw, summaries, validations, metadata):
    common = ["environment", "device_name", "torch_version", "cuda_version", "seed", "profile",
              "rows", "num_indices", "dim", "distribution", "variant"]
    fields = common + [f"{scope}_{stat}" for scope in SCOPES for stat in ("median_ms", "iqr_ms", "cv")]
    fields += list(VALIDATION_FIELDS)
    fields += [f"{prefix}_{scope}" for scope in SCOPES for prefix in ("winner", "practical_margin")]
    write_csv(out / "phase4_raw.csv", raw, common + ["scope", "repeat", "order", "latency_ms"])
    write_csv(out / "phase4_summary.csv", summaries, fields)
    for name, value in (("validation", validations), ("metadata", metadata)):
        (out / f"phase4_{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=PROFILES, default="pilot")
    p.add_argument("--seed", type=int, default=4001)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--environment", help="Environment label; defaults to device name and OS")
    p.add_argument("--out-dir", type=Path, help="New directory; refuses to overwrite an existing directory")
    p.add_argument("--rows", type=int, nargs="+")
    p.add_argument("--num-indices", type=int, nargs="+")
    p.add_argument("--dim", type=int, nargs="+")
    p.add_argument("--distributions", nargs="+", choices=("uniform", "zipf", "hotspot"), default=["uniform", "zipf", "hotspot"])
    p.add_argument("--pool-size", type=int, default=8)
    p.add_argument("--warmup", type=int)
    p.add_argument("--repeats", type=int)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--rtol", type=float, default=1e-4)
    args = p.parse_args(argv)
    for key, value in PROFILES[args.profile].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.seed < 0 or args.seed >= 2**32:
        p.error("seed must be in [0, 2**32)")
    if args.pool_size < 1 or args.warmup < 0 or args.repeats < 1:
        p.error("pool-size/repeats must be positive; warmup must be nonnegative")
    for key in ("rows", "num_indices", "dim"):
        if any(v <= 0 for v in getattr(args, key)):
            p.error(f"{key} must contain positive integers")
        setattr(args, key, list(dict.fromkeys(getattr(args, key))))
    args.distributions = list(dict.fromkeys(args.distributions))
    if any(n % args.pool_size for n in args.num_indices):
        p.error("every num-indices must be divisible by pool-size")
    if any(not math.isfinite(v) or v <= 0 for v in (args.atol, args.rtol)):
        p.error("atol and rtol must be positive finite values")
    return args


def run(args):
    available = torch.cuda.is_available()
    if args.device == "cuda" and not available:
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu for diagnostics.")
    if args.device in ("auto", "cuda") and available:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    else:
        print("CPU diagnostic mode: timings are not CUDA study evidence.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Use FP32 math for both variants and validation; no autocast, TF32 or compilation.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU"
    env = dict(environment=args.environment or f"{platform.system()} / {name}", device_name=name,
               torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
               seed=args.seed, profile=args.profile)
    out = args.out_dir or Path(__file__).resolve().parent / "results" / "phase4_training_proxy" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + args.profile + "_" + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(**env, args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    created_utc=datetime.now(timezone.utc).isoformat(), device=str(device),
                    cuda_available=available, cpu_diagnostic=device.type != "cuda", python=sys.version,
                    python_executable=sys.executable, platform=platform.platform(),
                    timing_backend="cuda_events" if device.type == "cuda" else "perf_counter_ns",
                    timing_scopes=SCOPES, practical_margin_threshold=0.05,
                    dtype="float32", tf32=False, num_threads=torch.get_num_threads(),
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    zipf_alpha=1.2, zipf_mode="truncated", hotspot_fraction=0.8, hotspot_rows=16,
                    row_layout="scattered", index_order="random", status="running",
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    raw, summaries, validations = [], [], []
    try:
        grid = itertools.product(args.rows, args.num_indices, args.dim, args.distributions)
        for rows, count, dim, dist in grid:
            config = dict(rows=rows, num_indices=count, dim=dim, distribution=dist)
            print(f"{config}", flush=True)
            w, refs, loss_ref, region_ref, info = generate_workload(config, args, device)
            checks = [validate(w, v, refs, loss_ref, region_ref, args) for v in VARIANTS]
            passed = all(c["validation_passed"] for c in checks)
            validations.append(dict(**config, **info, validation_passed=passed, variants=checks))
            if passed:
                samples = time_variants(w, args, device, random.Random(seed_for(args.seed, rows, count, dim, dist, "order")))
                common = env | config | info
                raw.extend(common | s for s in samples)
                summaries.extend(common | s for s in summarize(samples, checks))
            else:
                print("Validation failed: entire configuration excluded from timing.", flush=True)
            del w, refs, loss_ref, region_ref
        metadata["status"] = "complete" if all(v["validation_passed"] for v in validations) else "validation_failed"
    except Exception as exc:
        metadata.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata.update(completed_utc=datetime.now(timezone.utc).isoformat(),
                        configurations_validated=len(validations), summary_rows=len(summaries))
        export(out, raw, summaries, validations, metadata)
        print(f"Artifacts: {out.resolve()}", flush=True)
    return out, metadata["status"] == "complete"


def main(argv=None):
    _, passed = run(parse_args(argv))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
