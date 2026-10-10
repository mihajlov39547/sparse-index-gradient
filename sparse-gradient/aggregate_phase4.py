"""Aggregate saved Phase 4 artifacts only; never imports/runs benchmark code."""
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
OUT = BASE / "results/phase4_training_proxy_aggregate"
PAPER = PROJECT_ROOT / "paper_sparse_gradient"
ROOTS = {"local_gtx1650ti": BASE / "results/phase4_training_proxy",
         "colab_t4": BASE / "results/t4_phase4_training_proxy"}
VARIANTS = ["index_add_proxy", "scatter_add_proxy", "embedding_backward_proxy"]
LABELS = ["index_add", "scatter_add", "Embedding", "Tie"]
SCOPES = ["proxy_step", "accumulation_region"]
NAMES = {"local_gtx1650ti": "Local GTX1650Ti", "colab_t4": "Colab T4"}
GRID = set(itertools.product([4096, 65536], [8192, 65536], [32, 128], ["uniform", "zipf", "hotspot"]))
FILES = [f"phase4_{name}.{ext}" for name, ext in
         [("metadata", "json"), ("validation", "json"), ("summary", "csv"), ("raw", "csv")]]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def key(row):
    return tuple(int(row[k]) for k in ("rows", "num_indices", "dim")) + (row["distribution"],)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_run(path, group):
    issues = []
    def check(condition, message):
        if not condition:
            issues.append(message)
    for filename in FILES:
        check((path / filename).is_file(), f"missing {filename}")
    if issues:
        return None, issues
    m = json.loads((path / FILES[0]).read_text(encoding="utf-8"))
    validation = json.loads((path / FILES[1]).read_text(encoding="utf-8"))
    summary, raw = read_csv(path / FILES[2]), read_csv(path / FILES[3])
    for field, expected in [("status", "complete"), ("profile", "study"),
                            ("cpu_diagnostic", False), ("configurations_validated", 24), ("summary_rows", 72)]:
        check(m.get(field) == expected, f"metadata {field} != {expected}")
    check(m.get("device", "").split(":")[0] == "cuda", "not a CUDA device")
    check(m.get("timing_backend") == "cuda_events", "not CUDA event timing")
    check(m.get("practical_margin_threshold") == 0.05, "practical margin not 5%")
    check(len(validation) == 24 and {key(v) for v in validation} == GRID, "validation grid mismatch")
    check(len(summary) == 72, "actual summary length != 72")
    check({key(r) for r in summary} == GRID, "summary grid mismatch")
    for config in validation:
        check(config.get("validation_passed") is True, f"configuration failed {key(config)}")
        check(len(config["variants"]) == 3 and {v["variant"] for v in config["variants"]} == set(VARIANTS), "validation variants mismatch")
        for v in config["variants"]:
            check(v.get("validation_passed") is True and v.get("failed_check_names") == "", "variant validation failed")
            check(set(v["checks"]) == {"table", "head", "bias", "loss", "accumulation_region"}, "missing validation checks")
            check(all(c["passed"] is True and c["shape_ok"] is True and c["finite"] is True for c in v["checks"].values()), "detailed validation failed")
    groups = defaultdict(list)
    for row in summary:
        groups[key(row)].append(row)
        check(row.get("validation_passed") == "True" and row.get("failed_check_names") == "", "summary validation failed")
        for field in ["seed", "profile", "environment", "device_name", "torch_version", "cuda_version"]:
            check(row[field] == str(m[field]), f"summary identity mismatch: {field}")
    records = []
    for config, rows in groups.items():
        check(len(rows) == 3 and {r["variant"] for r in rows} == set(VARIANTS), "duplicate/missing summary variants")
        if len(rows) != 3:
            continue
        for scope in SCOPES:
            timings = {r["variant"]: float(r[f"{scope}_median_ms"]) for r in rows}
            check(all(math.isfinite(t) and t > 0 for t in timings.values()), "invalid median")
            ranked = sorted(timings, key=timings.get)
            margin = timings[ranked[1]] / timings[ranked[0]] - 1
            winner = ranked[0] if margin > 0.05 else "tie"
            check(all(r[f"winner_{scope}"] == winner and math.isclose(float(r[f"practical_margin_{scope}"]), margin, rel_tol=1e-9, abs_tol=1e-12) for r in rows), "stored winner/margin mismatch")
            records.append(dict(group=group, seed=m["seed"], scope=scope, config=config,
                                winner=winner, margin=margin, times=timings,
                                cvs={r["variant"]: float(r[f"{scope}_cv"]) for r in rows},
                                iqrs={r["variant"]: float(r[f"{scope}_iqr_ms"]) for r in rows}))
    # Raw data is used only to audit sample completeness and noise diagnostics.
    raw_groups = defaultdict(list)
    for row in raw:
        raw_groups[(key(row), row["variant"], row["scope"])].append(row)
    repeats = m["args"]["repeats"]
    check(len(raw) == 24 * 3 * 2 * repeats and len(raw_groups) == 144, "raw coverage mismatch")
    noise = []
    for (config, variant, scope), rows in raw_groups.items():
        check(config in GRID and variant in VARIANTS and scope in SCOPES, "unknown raw identity")
        check(len(rows) == repeats and {int(r["repeat"]) for r in rows} == set(range(repeats)), "raw repeats missing/duplicated")
        times = [float(r["latency_ms"]) for r in rows]
        check(all(math.isfinite(t) and t > 0 for t in times), "invalid raw latency")
        cv = stdev(times) / (sum(times) / len(times))
        ordered = [float(r["latency_ms"]) for r in sorted(rows, key=lambda r: int(r["repeat"]))]
        half_ratio = median(ordered[len(ordered)//2:]) / median(ordered[:len(ordered)//2])
        stored = next(r for r in groups[config] if r["variant"] == variant)
        check(math.isclose(cv, float(stored[f"{scope}_cv"]), rel_tol=1e-8, abs_tol=1e-12), "raw/summary CV mismatch")
        noise.append(dict(environment_group=group, seed=m["seed"], scope=scope,
                          rows=config[0], num_indices=config[1], dim=config[2], distribution=config[3],
                          variant=variant, num_samples=len(times), cv=cv,
                          second_half_over_first_half_median=half_ratio))
    return dict(path=path, metadata=m, validation=validation, summary=summary, records=records, noise=noise), sorted(set(issues))


def counts(records):
    c = Counter(r["winner"] for r in records)
    return dict(index_add_wins=c[VARIANTS[0]], scatter_add_wins=c[VARIANTS[1]],
                embedding_wins=c[VARIANTS[2]], ties=c["tie"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ["tables", "figures", "sections"]:
        (PAPER / folder).mkdir(parents=True, exist_ok=True)
    included, audit = [], []
    seen = set()
    for group, root in ROOTS.items():
        directories = sorted({p.parent for name in FILES for p in root.rglob(name)})
        for path in directories:
            try:
                run, issues = check_run(path, group)
            except Exception as exc:
                run, issues = None, [f"{type(exc).__name__}: {exc}"]
            if run:
                identity = (group, run["metadata"]["seed"])
                if identity in seen:
                    issues.append("duplicate environment/seed")
                if not issues:
                    seen.add(identity)
                    included.append(run)
            audit.append(dict(environment_group=group, path=str(path.relative_to(BASE)),
                              seed=run["metadata"]["seed"] if run else None,
                              included=not issues, issues=issues,
                              sha256={name: hashlib.sha256((path/name).read_bytes()).hexdigest()
                                      for name in FILES if (path/name).is_file()}))
    (OUT / "phase4_run_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not included:
        raise RuntimeError("No eligible runs; inspect audit")
    records = [r for run in included for r in run["records"]]
    aggregate, per_config = [], []
    for group in ROOTS:
        runs = [run for run in included if run["records"][0]["group"] == group]
        for field in ["device_name", "torch_version", "cuda_version", "script_sha256"]:
            if len({r["metadata"][field] for r in runs}) != 1:
                raise ValueError(f"Cannot pool heterogeneous {group}: {field}")
        for scope in SCOPES:
            subset = [r for r in records if r["group"] == group and r["scope"] == scope]
            c = counts(subset)
            aggregate.append(dict(environment_group=group,
                **{f: runs[0]["metadata"][f] for f in ["device_name", "torch_version", "cuda_version"]},
                scope=scope, seeds=";".join(str(s) for s in sorted({r["seed"] for r in subset})),
                num_runs=len(runs), num_configurations=len(subset), **c,
                **{name: c[field]/len(subset) for name, field in zip(
                    ["index_add_share", "scatter_add_share", "embedding_share", "tie_share"], c)}))
            for config in sorted(GRID):
                draws = [r for r in subset if r["config"] == config]
                times = {v: median(r["times"][v] for r in draws) for v in VARIANTS}
                fastest = [v for v in VARIANTS if times[v] == min(times.values())]
                per_config.append(dict(environment_group=group, scope=scope,
                    rows=config[0], num_indices=config[1], dim=config[2], distribution=config[3],
                    num_seeds=len({r["seed"] for r in draws}), **counts(draws),
                    median_fastest_variant=fastest[0] if len(fastest) == 1 else "tie",
                    median_practical_margin=median(r["margin"] for r in draws),
                    **{f"median_{name}_ms": times[v] for name, v in zip(["index_add", "scatter_add", "embedding"], VARIANTS)},
                    **{f"median_cv_{name}": median(r["cvs"][v] for r in draws) for name, v in zip(["index_add", "scatter_add", "embedding"], VARIANTS)}))
    write_csv(OUT / "phase4_aggregate_summary.csv", aggregate)
    write_csv(OUT / "phase4_per_configuration_summary.csv", per_config)
    noise = [row for run in included for row in run["noise"]]
    write_csv(OUT / "phase4_noise_diagnostics.csv", noise)
    create_paper_outputs(aggregate)
    create_report(audit, included, records, aggregate, noise)
    print(json.dumps(aggregate, indent=2))


def create_paper_outputs(aggregate):
    lines = [r"\begin{table}[tbp]", r"\centering\small",
             r"\caption{Phase 4 winners under the strictly greater than 5\% practical-margin rule. Each environment contributes 120 seed--configuration observations (24 configurations across five seeds). Region denotes a separate accumulation diagnostic replay.}",
             r"\label{tab:phase4-winner-counts}", r"\setlength{\tabcolsep}{3pt}",
             r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Environment & Scope & \texttt{index\_add} & \texttt{scatter\_add} & Embedding & Tie & Total \\", r"\midrule"]
    for r in aggregate:
        label = "GTX1650Ti" if r["environment_group"] == "local_gtx1650ti" else "T4"
        scope = "Proxy step" if r["scope"] == "proxy_step" else "Region replay"
        lines.append(f"{label} & {scope} & " + " & ".join(str(r[k]) for k in ["index_add_wins", "scatter_add_wins", "embedding_wins", "ties", "num_configurations"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}%", r"\end{table}"]
    (PAPER / "tables/phase4_winner_counts.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(3.55, 3.0), layout="constrained")
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    left = [0] * 4
    colors = ["#0072B2", "#E69F00", "#009E73", "#DDDDDD"]
    for field, label, color, hatch in zip(["index_add_wins", "scatter_add_wins", "embedding_wins", "ties"], LABELS, colors, ["", "//", "..", ""]):
        values = [r[field] for r in aggregate]
        ax.barh(range(4), values, left=left, label=label, color=color, edgecolor="white", linewidth=.5, hatch=hatch)
        for i, value in enumerate(values):
            if value:
                ax.text(left[i]+value/2, i, str(value), ha="center", va="center", fontsize=8,
                        color="white" if field == "index_add_wins" else "black")
        left = [a+b for a, b in zip(left, values)]
    ax.set_yticks(range(4), ["GTX1650Ti\nProxy step", "GTX1650Ti\nRegion replay", "T4\nProxy step", "T4\nRegion replay"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 120)
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Seed–configuration observations", fontsize=8)
    ax.legend(loc="lower center", bbox_to_anchor=(.4, 1.01), ncol=2, frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    for ext in ["png", "pdf"]:
        fig.savefig(PAPER / f"figures/phase4_winner_counts_by_environment.{ext}", dpi=300)
    plt.close(fig)


def create_report(audit, included, records, aggregate, noise):
    lines = ["# Phase 4 aggregate report", "", "Saved-artifact analysis only; no experiments executed. No benchmark or existing manuscript files modified.", "",
             f"Discovered **{len(audit)}** runs; included **{len(included)}**; excluded **{len(audit)-len(included)}**.", "",
             "## Run audit", "", "| Environment | Run | Seed | Decision |", "|---|---|---:|---|"]
    for a in audit:
        lines.append(f"| {a['environment_group']} | `{a['path']}` | {a['seed']} | " + ("included" if a["included"] else "; ".join(a["issues"])) + " |")
    lines += ["", "All included runs pass metadata, 24-configuration grid, 72 summary rows, all configuration/variant/check validation flags, and empty failed-check checks. CUDA device strings `cuda:0` are accepted as CUDA devices, not treated as failures of literal equality to `cuda`. Each run has 5,760 raw samples (40 repeats × 24 configurations × three variants × two scopes). Saved winners and margins were independently recomputed from summary medians, counting each configuration once per scope, not once per variant. Raw CVs match saved summary CVs. Input file hashes are recorded in `phase4_run_audit.json`.", "", "## Aggregate winners", "", "| Environment | Scope | index_add | scatter_add | Embedding | Tie | Total |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in aggregate:
        lines.append(f"| {r['environment_group']} | {r['scope']} | " + " | ".join(str(r[k]) for k in ["index_add_wins", "scatter_add_wins", "embedding_wins", "ties", "num_configurations"]) + " |")
    expected = [(83, 0, 10, 27), (98, 0, 10, 12), (0, 25, 0, 95), (3, 39, 0, 78)]
    matches = [tuple(r[k] for k in ["index_add_wins", "scatter_add_wins", "embedding_wins", "ties"]) for r in aggregate] == expected
    lines += ["", f"Match to the supplied expected counts: **{matches}**. This comparison is performed after aggregation; the expected counts are not used to populate CSVs or figure data.", "", "## Environments and validation scales", ""]
    for group in ROOTS:
        runs = [r for r in included if r["records"][0]["group"] == group]
        m = runs[0]["metadata"]
        rows = [r for run in runs for r in run["summary"]]
        lines.append(f"- {NAMES[group]}: {m['device_name']}, PyTorch {m['torch_version']}, CUDA {m['cuda_version']}, {m['platform']}, {m['num_threads']} CPU threads. Maximum table absolute error {max(float(r['max_abs_error']) for r in rows):.4g}; maximum any-check absolute error {max(float(r['max_abs_error_any_check']) for r in rows):.4g}; maximum table relative L2 error {max(float(r['relative_l2_error_table']) for r in rows):.4g}.")
    hashes = {run["metadata"]["script_sha256"] for run in included}
    index_hashes = defaultdict(set)
    for run in included:
        for row in run["summary"]:
            index_hashes[(row["seed"], key(row))].add(row["index_sha256"])
    lines += ["", f"Benchmark script hashes across environments: {len(hashes)} unique. Matched seed/configuration index hashes agree across environments: {all(len(v)==1 for v in index_hashes.values())}. Validation thresholds remain atol=1e-8 and rtol=1e-4. Error scales are diagnostics, not a rationale for winner-dependent tolerance changes.", "", "## Noise and stability diagnostics", "", "Raw data are used only for these diagnostics and sample auditing; main winner counts use summary medians. No samples or valid configurations are removed for noise. CV > 0.10 and half-run median drift > 10% are descriptive flags, not significance tests or exclusion rules.", "", "| Environment | Scope | Median raw CV | Max raw CV | CV > 0.10 (of 360) | >10% half-run drift (of 360) | Unchanged winner across five seeds (of 24) |", "|---|---|---:|---:|---:|---:|---:|"]
    for group in ROOTS:
        for scope in SCOPES:
            ns = [r for r in noise if r["environment_group"] == group and r["scope"] == scope]
            rs = [r for r in records if r["group"] == group and r["scope"] == scope]
            stable = sum(len({r["winner"] for r in rs if r["config"] == c}) == 1 for c in GRID)
            lines.append(f"| {group} | {scope} | {median(r['cv'] for r in ns):.3f} | {max(r['cv'] for r in ns):.3f} | {sum(r['cv'] > .1 for r in ns)} | {sum(abs(r['second_half_over_first_half_median']-1) > .1 for r in ns)} | {stable} |")
    lines += ["", "Stability includes configurations consistently classified as ties. Repeated timings within a draw are not independent workloads; 120 observations are 24 shapes/distributions × five seeds, not 120 distinct shapes. A >5% point-estimate winner is not automatically robust to noise. Inspect `phase4_noise_diagnostics.csv` and per-configuration seed counts before selecting illustrative examples.", "", "## Interpretation and warnings", "", "The local stack most often favors the index_add custom backward path. T4 is dominated by practical ties: 95/120 (79.2%) in the total proxy and 78/120 (65.0%) in the diagnostic replay; its resolved cases mainly favor scatter_add. No universal fastest path follows. PyTorch, CUDA, OS, host thread settings and GPU differ jointly, so hardware causality is not isolated. The region replay is not an exact decomposition and must not be subtracted from total-step time. These results do not measure production training or convergence, optimizer state, data loading, distributed execution or production-scale throughput. Custom autograd nodes introduce dispatch/graph overhead, making the comparison API/path-level rather than kernel-only.", "", "The observed pattern establishes context dependence, not a direct matched Phase 2 replication: no Phase 2 files were joined in this aggregation. No paired bootstrap, confidence intervals, cache flushing, or new timing runs were added.", "", "## Definitions", "", "Aggregate CSV shares are fractions. `num_configurations` counts seed–configuration observations. Per-configuration rows summarize five seed medians with equal seed weight. `median_fastest_variant` is the minimum of the across-seed latency medians (exact equality yields tie), not a 5% winner or modal winner. `median_practical_margin` is the median of each seed's runner-up/fastest ratio minus one. CV columns are medians of saved per-seed CVs. Seed-level winner counts retain the strict >5% classification.", "", "## Files and insertion point", "", "New paper artifacts are stored under the root-level `paper_sparse_gradient/` directory alongside the existing manuscript assets. File paths listed below are relative to the project root. Recommended insertion: after the Phase 3 attribution subsection and before Result-Level Discussion in the actual `paper_sparse_gradient/sections/05_results_analysis.tex`. Its discussion section is currently `06_general_discussion.tex`; do not assume the alternative numbering from the task. No insertion has been performed. Tables require booktabs/tabularx and the figure requires graphicx, already used by the existing manuscript. The standalone subsection is not included by the main manuscript yet.", ""]
    files = [OUT / n for n in ["phase4_aggregate_summary.csv", "phase4_per_configuration_summary.csv", "phase4_run_audit.json", "phase4_noise_diagnostics.csv", "phase4_aggregate_report.md"]]
    files += list(PAPER.rglob("phase4_*")) + [BASE / "aggregate_phase4.py"]
    lines += [f"- `{p.relative_to(PROJECT_ROOT)}`" for p in sorted(files)]
    lines += ["", "Reproduce aggregation only (requires matplotlib; no torch import): `python aggregate_phase4.py` from `sparse-gradient/`. The current workspace Python lacks matplotlib; generation used the existing `C:/Projects/phd-segmentacija-instanci/.venv/Scripts/python.exe` plotting environment without installing packages."]
    (OUT / "phase4_aggregate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
