"""Descriptive threshold sensitivity from saved medians; standard library only.

Run from any directory: python sparse-gradient/threshold_sensitivity.py
Does not import benchmark code, execute experiments, or change the main 5% rule.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path, PureWindowsPath

BASE = Path(__file__).resolve().parent
OUT = BASE / "results/threshold_sensitivity"
TABLES = BASE.parent / "paper_sparse_gradient/tables"
THRESHOLDS = (0.02, 0.05, 0.10)
OPS = ("index_add", "scatter_add", "embedding")
PROXY_OPS = {"index_add_proxy": "index_add", "scatter_add_proxy": "scatter_add",
             "embedding_backward_proxy": "embedding"}
DRAW_KEYS = ("rows", "indices", "dim", "distribution", "zipf_mode", "row_layout",
             "index_order", "cache_mode", "workload_draw")
GRID4 = set(itertools.product([4096, 65536], [8192, 65536], [32, 128], ["uniform", "zipf", "hotspot"]))
COUNT_FIELDS = ("index_add_wins", "scatter_add_wins", "embedding_wins", "ties")
SHARE_FIELDS = ("index_add_share", "scatter_add_share", "embedding_share", "tie_share")
FIELDS4 = ["phase", "environment_group", "scope", "threshold", "num_runs", "num_configurations", *COUNT_FIELDS, *SHARE_FIELDS]
FIELDS2 = [f for f in FIELDS4 if f not in ("scope", "num_runs")]
SOURCES = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source(path):
    SOURCES[path.relative_to(BASE).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(source(path).read_text(encoding="utf-8"))


def read_csv(path):
    with source(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def rank(times):
    require(set(times) == set(OPS), "Missing or unknown variant")
    require(all(math.isfinite(t) and t > 0 for t in times.values()), "Invalid median latency")
    ordered = sorted(times, key=lambda op: (times[op], op))
    return ordered[0], times[ordered[1]] / times[ordered[0]] - 1


def aggregate(phase, group, scope, measurements, num_runs):
    ranked = [rank(t) for t in measurements]
    rows = []
    for threshold in THRESHOLDS:
        c = Counter(op if margin > threshold else "tie" for op, margin in ranked)
        row = dict(phase=phase, environment_group=group, scope=scope, threshold=threshold,
                   num_runs=num_runs, num_configurations=len(ranked))
        row.update(zip(COUNT_FIELDS, [c[op] for op in (*OPS, "tie")]))
        row.update(zip(SHARE_FIELDS, [row[f] / len(ranked) for f in COUNT_FIELDS]))
        rows.append(row)
    require(rows[0]["ties"] <= rows[1]["ties"] <= rows[2]["ties"], "Nonmonotonic tie counts")
    return rows


def phase2():
    manifest_path = BASE / "phase2_analysis_v2/analysis_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest["matched_ops"] == list(OPS), "Phase 2 matched API set changed")
    labels = {"GTX1650Ti": "local_gtx1650ti", "T4": "t4", "T4_repeat": "t4_repeat"}
    runs = {}
    for spec in manifest["runs"]:
        # Manifest records the previous checkout location; relocate the documented
        # run basename to this checkout, rather than choosing files by counts.
        folder = BASE / PureWindowsPath(spec["path"]).name
        summary = read_csv(folder / "summary_results.csv")
        metadata = read_json(folder / "metadata.json")
        eligible = defaultdict(list)
        for r in summary:
            if r["operation"] not in OPS:
                continue
            for k in DRAW_KEYS:
                if k not in r:
                    r[k] = str(metadata.get("arguments", {}).get(k, metadata.get(k, "unknown")))
            ms = float(r["median_ms"]) if r["median_ms"] else float("nan")
            if (r["status"] == "ok" and r["correctness"] == "passed" and
                    r["scope_group"] == "fair_dense_grad" and math.isfinite(ms) and ms > 0 and
                    all(r[k] not in ("", "unknown") for k in DRAW_KEYS)):
                eligible[(tuple(r[k] for k in DRAW_KEYS), r["operation"])].append(ms)
        configs = defaultdict(dict)
        # Match analyze_phase2.py: all duplicates excluded, never choose faster rows.
        for (config, op), values in eligible.items():
            if len(values) == 1:
                configs[config][op] = values[0]
        runs[spec["label"]] = {k: v for k, v in configs.items() if set(v) == set(OPS)}
    require(set(runs) == set(labels), "Phase 2 environment set changed")
    common = set.intersection(*(set(run) for run in runs.values()))
    require(len(common) == manifest["matched_common_draw_count"] == 162, "Phase 2 intersection mismatch")
    saved = read_csv(BASE / "phase2_analysis_v2/matched_winners_common_ops.csv")
    for label in labels:
        keys = [tuple(r[k] for k in DRAW_KEYS) for r in saved if r["label"] == label]
        require(len(keys) == 162 and set(keys) == common, f"Phase 2 saved matched keys differ: {label}")
    rows = []
    for label, group in labels.items():
        rows.extend(aggregate("phase2", group, "", [runs[label][k] for k in sorted(common)], 1))
    return rows


def config4(row):
    return tuple(int(row[k]) for k in ("rows", "num_indices", "dim")) + (row["distribution"],)


def phase4():
    rows, included, excluded = [], [], []
    roots = {"local_gtx1650ti": "phase4_training_proxy", "colab_t4": "t4_phase4_training_proxy"}
    for group, folder in roots.items():
        scopes = {s: [] for s in ("proxy_step", "accumulation_region")}
        seeds = set()
        for path in sorted((BASE / "results" / folder).rglob("phase4_metadata.json")):
            m = read_json(path)
            if m.get("profile") != "study" or m.get("seed") not in range(4001, 4006):
                excluded.append(str(path.parent.relative_to(BASE)))
                continue
            require(m["status"] == "complete" and m["device"].split(":")[0] == "cuda" and
                    m["cpu_diagnostic"] is False and m["configurations_validated"] == 24 and
                    m["summary_rows"] == 72, f"Invalid study metadata: {path}")
            require(m["seed"] not in seeds, f"Duplicate study seed: {path}")
            seeds.add(m["seed"])
            validation = read_json(path.with_name("phase4_validation.json"))
            require(len(validation) == 24 and {config4(r) for r in validation} == GRID4, "Validation grid mismatch")
            for v in validation:
                require(v["validation_passed"] is True and len(v["variants"]) == 3 and
                        {c["variant"] for c in v["variants"]} == set(PROXY_OPS), "Invalid validation configuration")
                for c in v["variants"]:
                    require(c["validation_passed"] is True and c["failed_check_names"] == "" and
                            all(x["passed"] and x["finite"] and x["shape_ok"] for x in c["checks"].values()), "Failed validation")
            summary = read_csv(path.with_name("phase4_summary.csv"))
            require(len(summary) == 72, "Invalid summary row count")
            configs = defaultdict(dict)
            for r in summary:
                require(r["validation_passed"] == "True" and r["failed_check_names"] == "", "Invalid summary validation")
                require(int(r["seed"]) == m["seed"] and r["profile"] == "study", "Summary identity mismatch")
                k, op = config4(r), PROXY_OPS[r["variant"]]
                require(op not in configs[k], "Duplicate summary variant")
                configs[k][op] = r
            require(set(configs) == GRID4, "Phase 4 summary grid mismatch")
            for config in configs.values():
                for scope in scopes:
                    times = {op: float(r[f"{scope}_median_ms"]) for op, r in config.items()}
                    fastest, margin = rank(times)
                    mapped = next(v for v, op in PROXY_OPS.items() if op == fastest)
                    winner = mapped if margin > .05 else "tie"
                    require(all(r[f"winner_{scope}"] == winner for r in config.values()), "Stored Phase 4 winner mismatch")
                    scopes[scope].append(times)
            included.append(str(path.parent.relative_to(BASE)))
        require(seeds == set(range(4001, 4006)), f"Missing Phase 4 study seeds: {group}")
        for scope, measurements in scopes.items():
            require(len(measurements) == 120, "Phase 4 coverage mismatch")
            rows.extend(aggregate("phase4", group, scope, measurements, 5))
    return rows, included, excluded


def verify_baseline(rows):
    # Sanity gates only; all values exported below originate from saved medians.
    expected = {
        ("phase2", "local_gtx1650ti", ""): (60, 0, 96, 6),
        ("phase2", "t4", ""): (5, 92, 25, 40),
        ("phase2", "t4_repeat", ""): (1, 93, 27, 41),
        ("phase4", "local_gtx1650ti", "proxy_step"): (83, 0, 10, 27),
        ("phase4", "local_gtx1650ti", "accumulation_region"): (98, 0, 10, 12),
        ("phase4", "colab_t4", "proxy_step"): (0, 25, 0, 95),
        ("phase4", "colab_t4", "accumulation_region"): (3, 39, 0, 78),
    }
    for row in rows:
        if row["threshold"] == .05:
            k = (row["phase"], row["environment_group"], row["scope"])
            actual = tuple(row[f] for f in COUNT_FIELDS)
            require(actual == expected[k], f"STOP: 5% mismatch for {k}: {actual} != {expected[k]}")


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def latex(rows, phase):
    proxy = phase == 4
    caption = "Threshold sensitivity of Phase~4 proxy winner counts." if proxy else "Threshold sensitivity of Phase~2 winner counts."
    lines = [r"\begin{table}[tbp]", r"\centering\small",
             r"\caption{" + caption + r" The main paper uses the strictly greater than 5\% practical-margin rule; 2\% and 10\% are reported only as sensitivity checks.}",
             rf"\label{{tab:threshold-sensitivity-phase{phase}}}", r"\setlength{\tabcolsep}{3pt}",
             r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{" + ("lllrrrrr" if proxy else "llrrrrr") + "}", r"\toprule",
             "Environment & " + ("Scope & " if proxy else "") + r"Threshold & \texttt{index\_add} & \texttt{scatter\_add} & Embedding & Tie & Total \\", r"\midrule"]
    names = {"local_gtx1650ti": "GTX1650Ti", "t4": "T4", "t4_repeat": "T4 repeat", "colab_t4": "T4"}
    for r in rows:
        cells = [names[r["environment_group"]]]
        if proxy:
            cells.append("Proxy step" if r["scope"] == "proxy_step" else "Region replay")
        cells += [f"{r['threshold']*100:.0f}\\%"] + [str(r[f]) for f in (*COUNT_FIELDS, "num_configurations")]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}%", r"\end{table}"]
    (TABLES / f"threshold_sensitivity_phase{phase}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    # Finish validation and both baseline gates before writing any outputs.
    p2 = phase2()
    p4, included, excluded = phase4()
    verify_baseline(p2 + p4)
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "phase2_threshold_sensitivity.csv", p2, FIELDS2)
    write_csv(OUT / "phase4_threshold_sensitivity.csv", p4, FIELDS4)
    write_csv(OUT / "threshold_sensitivity_summary.csv", p2 + p4, FIELDS4)
    latex(p2, 2)
    latex(p4, 4)
    audit = dict(thresholds=THRESHOLDS, rule="runner_up / fastest - 1 > threshold",
                 baseline_checks="passed", phase2_matched_configurations=162,
                 phase2_pairing_keys=DRAW_KEYS, phase4_included=included, phase4_excluded=excluded,
                 input_sha256=SOURCES, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 notes=["Phase 2 uses the manifest's three source summaries, relocated by basename, and their complete validated common draw-key intersection.",
                        "Phase 2 counts include workload_draw; Phase 4 counts include seed. They are not counts of distinct shapes.",
                        "T4 repeat is a run on the T4 stack, not a third independent environment.",
                        "Only saved medians are used. No experiments, raw timing reaggregation, statistical tests, or figure generation.",
                        "Run python sparse-gradient/threshold_sensitivity.py to reproduce; standard library only."])
    (OUT / "threshold_sensitivity_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for r in p2 + p4:
        print(r["phase"], r["environment_group"], r["scope"], r["threshold"],
              tuple(r[f] for f in COUNT_FIELDS))
    print("Both 5% baselines match; counts ordered index_add, scatter_add, embedding, tie.")


if __name__ == "__main__":
    main()
