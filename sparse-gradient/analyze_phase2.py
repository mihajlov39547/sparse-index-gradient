"""Analyze saved sparse-gradient runs without importing torch or running benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = ["rows", "indices", "dim", "distribution", "zipf_mode", "row_layout", "index_order", "cache_mode"]
DRAW_KEYS = BASE + ["workload_draw"]
FAIR = ["index_add", "scatter_add", "embedding", "embedding_bag", "sorted_segment"]
COLORS = dict(zip(FAIR, ["#e69f00", "#0072b2", "#009e73", "#cc79a7", "#9467bd"]))
COLORS["tie"] = "#999999"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, metavar="LABEL=PATH")
    parser.add_argument("--out-dir", type=Path, default=Path("sparse-gradient/phase2_analysis"))
    parser.add_argument("--tie-threshold", type=float, default=0.05)
    parser.add_argument("--matched-ops", nargs="+", choices=FAIR, default=["index_add", "scatter_add", "embedding"])
    parser.add_argument("--min-overhead-multiple", type=float, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if not np.isfinite(args.tie_threshold) or args.tie_threshold < 0:
        parser.error("tie-threshold must be finite and nonnegative")
    if not np.isfinite(args.min_overhead_multiple) or args.min_overhead_multiple <= 0:
        parser.error("min-overhead-multiple must be finite and positive")
    if args.bootstrap_samples < 0:
        parser.error("bootstrap-samples must be nonnegative; 0 disables intervals")
    args.matched_ops = list(dict.fromkeys(args.matched_ops))
    if len(args.matched_ops) < 2 or "index_add" not in args.matched_ops:
        parser.error("matched-ops must include index_add and at least one other operation")
    runs = []
    for spec in args.runs:
        label, sep, path = spec.partition("=")
        if not sep or not label.strip() or not path.strip():
            parser.error(f"Expected LABEL=PATH, got {spec!r}")
        if label in [item[0] for item in runs]:
            parser.error(f"Duplicate label: {label}")
        runs.append((label, Path(path)))
    args.runs = runs
    return args


def read_csv(path, notes):
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        notes.append(f"Skipped {path}: {exc}")
        return pd.DataFrame()


def environment(label, metadata):
    args = metadata.get("arguments", {})
    result = {"label": label}
    for key in ["gpu_name", "compute_capability", "pytorch_version", "cuda_version", "python_version",
                "platform", "methodology_version", "completed_combinations", "failed_combinations"]:
        value = metadata.get(key)
        result[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
    for key in ["rows", "indices", "dims", "distributions", "zipf_mode", "workload_seeds", "repeats", "rounds",
                "ops", "row_layout", "index_order", "cache_mode", "hotspot_fraction", "hotspot_rows", "zipf_alpha"]:
        value = args.get(key, metadata.get(key))
        result[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
    return result


def fill_settings(frame, metadata):
    frame = frame.copy()
    if frame.empty:
        return frame.reindex(columns=list(dict.fromkeys([*frame.columns, *BASE])))
    for key in BASE:
        if key not in frame:
            frame[key] = metadata.get("arguments", {}).get(key, metadata.get(key, "unknown"))
    return frame


def reconstruct_winner_summary(winners):
    columns = BASE + ["operation", "win_count", "total_draws", "win_rate"]
    if winners.empty:
        return pd.DataFrame(columns=columns)
    denom = winners.groupby(BASE, dropna=False).workload_draw.nunique().rename("total_draws")
    result = winners.groupby(BASE + ["winner"], dropna=False).size().rename("win_count").reset_index()
    result = result.merge(denom.reset_index(), on=BASE)
    result["win_rate"] = result.win_count / result.total_draws
    return result.rename(columns={"winner": "operation"})[columns]


def load_run(label, path, notes):
    if not path.is_dir():
        notes.append(f"Missing input folder: {label}={path}")
        return None
    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        notes.append(f"Skipped metadata for {label}: {exc}")
        metadata = {}
    summary = read_csv(path / "summary_results.csv", notes)
    winners = read_csv(path / "winners.csv", notes)
    stable = read_csv(path / "winner_summary.csv", notes)
    if not winners.empty and not {"winner", "rows", "indices", "dim", "distribution", "workload_draw"}.issubset(winners):
        notes.append(f"{label}: winners.csv lacks required keys; skipped winner analyses.")
        winners = pd.DataFrame()
    winners = fill_settings(winners, metadata)
    # Verify eligibility against summary when possible. Never rank diagnostics.
    changed = False
    if not winners.empty:
        before = len(winners)
        winners = winners[winners.winner.isin(FAIR)].copy()
        if {"operation", "scope_group", "status", "correctness", "workload_draw"}.issubset(summary):
            eligible = fill_settings(summary, metadata)
            eligible = eligible[(eligible.status == "ok") & (eligible.correctness == "passed")
                                & (eligible.scope_group == "fair_dense_grad")]
            keys = BASE + ["workload_draw", "winner"]
            eligible = eligible.rename(columns={"operation": "winner"})[keys].drop_duplicates()
            winners = winners.merge(eligible, on=keys, how="inner", validate="many_to_one")
        else:
            notes.append(f"{label}: could not cross-check winner eligibility; trusting saved winners.csv.")
        if winners.duplicated(BASE + ["workload_draw"]).any():
            notes.append(f"{label}: duplicate winner draw keys; keeping first row per draw.")
            winners = winners.drop_duplicates(BASE + ["workload_draw"])
        changed = len(winners) != before
        if changed:
            notes.append(f"{label}: retained {len(winners)}/{before} winner rows after validation.")
        if "within_noise" in winners:
            winners["noise_known"] = winners.within_noise.astype(str).str.lower().isin(["true", "false"])
            winners["non_noise"] = winners.within_noise.astype(str).str.lower().eq("false")
        else:
            notes.append(f"{label}: within_noise missing; non-noise statistics unavailable.")
            winners["noise_known"] = False
            winners["non_noise"] = False
        if "speedup_vs_index_add" not in winners:
            winners["speedup_vs_index_add"] = np.nan
        winners["speedup_vs_index_add"] = pd.to_numeric(winners.speedup_vs_index_add, errors="coerce")
        winners.loc[~np.isfinite(winners.speedup_vs_index_add) | (winners.speedup_vs_index_add <= 0), "speedup_vs_index_add"] = np.nan
        winners["label"] = label
    if changed or stable.empty or not {"operation", "win_rate", "total_draws", *BASE}.issubset(stable):
        stable = reconstruct_winner_summary(winners)
        notes.append(f"{label}: reconstructed winner summary from retained winners.")
    elif not winners.empty:
        reconstructed = reconstruct_winner_summary(winners)
        saved = stable[BASE + ["operation", "win_rate", "total_draws"]].sort_values(BASE + ["operation"]).reset_index(drop=True)
        check = reconstructed[saved.columns].sort_values(BASE + ["operation"]).reset_index(drop=True)
        if len(saved) != len(check) or not saved[BASE + ["operation"]].equals(check[BASE + ["operation"]]) or not np.allclose(saved[["win_rate", "total_draws"]], check[["win_rate", "total_draws"]]):
            stable = reconstructed
            notes.append(f"{label}: inconsistent saved winner summary; used reconstruction.")
    stable = stable[stable.operation.isin(FAIR)] if "operation" in stable else stable
    return {"label": label, "path": str(path.resolve()), "metadata": metadata, "environment": environment(label, metadata),
            "winners": winners, "stable": stable, "summary": summary}


def speedup_summary(winners, keys):
    columns = keys + ["count", "median_speedup", "mean_speedup", "min_speedup", "max_speedup", "p25", "p75"]
    if winners.empty:
        return pd.DataFrame(columns=columns)
    return winners.groupby(keys, dropna=False).speedup_vs_index_add.agg(
        count="count", median_speedup="median", mean_speedup="mean", min_speedup="min", max_speedup="max",
        p25=lambda x: x.quantile(.25), p75=lambda x: x.quantile(.75)).reset_index()[columns]


def make_tables(runs, out):
    tables = {"environment_summary": pd.DataFrame([run["environment"] for run in runs])}
    winners = pd.concat([r["winners"] for r in runs if not r["winners"].empty], ignore_index=True) if any(not r["winners"].empty for r in runs) else pd.DataFrame()
    counts, details, stable_counts = [], [], []
    for run in runs:
        w, label = run["winners"], run["label"]
        measured = set(run["summary"].get("operation", [])) | set(run["metadata"].get("arguments", {}).get("ops", []))
        operations = [op for op in FAIR if op in measured or (not w.empty and op in set(w.winner))]
        for op in operations:
            group = w[w.winner == op] if not w.empty else w
            known = not w.empty and w.noise_known.all()
            counts.append({"label": label, "operation": op, "per_draw_win_count": len(group),
                           "per_draw_win_rate": len(group)/len(w) if len(w) else np.nan,
                           "non_noise_win_count": int(group.non_noise.sum()) if known else np.nan,
                           "non_noise_win_rate": float(group.non_noise.sum()/len(w)) if known else np.nan,
                           "total_draws": len(w)})
        run_details = []
        for config, g in run["stable"].groupby(BASE, dropna=False):
            g = g.sort_values(["win_rate", "operation"], ascending=[False, True])
            best = g.iloc[0]
            tied = g.loc[g.win_rate.eq(best.win_rate), "operation"].tolist()
            row = dict(zip(BASE, config)) | {"label": label, "operation": "tie" if len(tied) > 1 else best.operation, "win_rate": best.win_rate,
                  "stable": bool(best.win_rate == 1.0), "workload_draws": int(best.total_draws),
                  "dominant_tie": len(tied) > 1, "tied_operations": ", ".join(tied)}
            details.append(row)
            run_details.append(row)
        stable_n = sum(row["stable"] for row in run_details)
        stable_counts.append({"label": label, "total_base_configurations": len(run_details),
                              "stable_configurations": stable_n, "mixed_configurations": len(run_details)-stable_n,
                              "stable_percentage": 100*stable_n/len(run_details) if run_details else np.nan,
                              **{f"dominant_{op}_count": sum(row["operation"] == op for row in run_details) for op in FAIR + ["tie"]}})
    tables["winner_counts_by_run"] = pd.DataFrame(counts, columns=["label", "operation", "per_draw_win_count", "per_draw_win_rate", "non_noise_win_count", "non_noise_win_rate", "total_draws"])
    for col, suffix in [("dim", "dimension"), ("rows", "rows"), ("indices", "indices"), ("distribution", "distribution")]:
        keys = ["label", col, "winner"]
        t = winners.groupby(keys).size().rename("win_count").reset_index() if not winners.empty else pd.DataFrame(columns=keys + ["win_count"])
        if not t.empty:
            t["win_rate"] = t.win_count / t.groupby(["label", col]).win_count.transform("sum")
        tables[f"winner_counts_by_{suffix}"] = t
    tables["stable_winner_details"] = pd.DataFrame(details, columns=BASE + ["label", "operation", "win_rate", "stable", "workload_draws", "dominant_tie", "tied_operations"])
    tables["stable_winner_summary"] = pd.DataFrame(stable_counts)
    tables["speedup_summary"] = speedup_summary(winners, ["label", "winner"])
    tables["speedup_summary_by_winner"] = tables["speedup_summary"].copy()
    tables["speedup_summary_by_dim_distribution"] = speedup_summary(winners, ["label", "winner", "dim", "distribution"])
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    return tables, winners


def make_plots(tables, winners, out):
    files = []
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=160)
        plt.close(fig)
        files.append(name)
    counts = tables["winner_counts_by_run"]
    if counts.empty or winners.empty:
        return files
    pivot = counts.pivot(index="label", columns="operation", values="per_draw_win_rate").fillna(0)*100
    ax = pivot.plot.bar(stacked=True, color=[COLORS.get(op, "gray") for op in pivot.columns], figsize=(10, 6), rot=0)
    ax.set(ylabel="Raw minimum-time selections (%)", title="Descriptive: each run's own grid and API set; no practical margin")
    save(ax.figure, "winner_share_by_run.png")
    for col, filename in [("dim", "winner_share_by_dim_by_run.png"), ("distribution", "winner_share_by_distribution_by_run.png")]:
        labels = winners.label.unique()
        fig, axes = plt.subplots(1, len(labels), figsize=(6*len(labels), 5), squeeze=False, sharey=True)
        for ax, label in zip(axes[0], labels):
            w = winners[winners.label == label]
            data = pd.crosstab(w[col], w.winner).reindex(columns=FAIR, fill_value=0)
            data = data.div(data.sum(axis=1), axis=0)*100
            data.plot.bar(stacked=True, ax=ax, color=[COLORS[op] for op in FAIR], rot=0)
            ax.set(title=f"Descriptive: {label}\nOwn grid; raw minima, no margin", xlabel=col, ylabel="Raw selections (%)", ylim=(0, 100))
            ax.legend(fontsize=7)
        save(fig, filename)
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(winners.label.unique())
    rng = np.random.default_rng(1234)
    for op in FAIR:
        w = winners[(winners.winner == op) & winners.speedup_vs_index_add.notna()]
        if len(w):
            x = [labels.index(label) for label in w.label]
            ax.scatter(np.array(x)+rng.uniform(-.18, .18, len(w)), w.speedup_vs_index_add,
                       label=op, color=COLORS[op], alpha=.5, s=20)
    ax.axhline(1, color="gray", linestyle="--")
    ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel="Winner speedup vs index_add in the same draw",
           title="Descriptive, winner-conditioned: unmatched grids; no practical margin")
    ax.legend()
    save(fig, "speedup_vs_index_add_by_run.png")
    focused = pivot.reindex(columns=["embedding", "scatter_add", "index_add"], fill_value=0)
    ax = focused.plot.bar(color=[COLORS[op] for op in focused.columns], figsize=(10, 6), rot=0)
    ax.set(ylabel="Raw minimum-time selections (%)", title="Descriptive API shares: unmatched grids; no practical margin")
    save(ax.figure, "embedding_vs_scatter_add_cross_environment.png")
    return files


WIN_COLUMNS = ["label"] + DRAW_KEYS + ["winner", "raw_winner", "winner_ms", "runner_up", "runner_up_ms",
               "margin_ratio", "speedup_vs_index_add", "eligible_operations", "margin_aware_winner", "decisive"]


def eligible_summary(run):
    """No fallback to winner exports: matching requires measured, validated medians."""
    data = run["summary"].copy()
    required = set(DRAW_KEYS + ["operation", "status", "correctness", "scope_group", "median_ms"])
    if not required.issubset(data):
        return pd.DataFrame(columns=list(required) + ["label", "overhead_multiple"])
    data["median_ms"] = pd.to_numeric(data.median_ms, errors="coerce")
    data = data[(data.status == "ok") & (data.correctness == "passed")
                & (data.scope_group == "fair_dense_grad")
                & np.isfinite(data.median_ms) & (data.median_ms > 0)].dropna(subset=DRAW_KEYS)
    # Duplicate measurements cannot safely be resolved by choosing the faster row.
    data = data[~data.duplicated(DRAW_KEYS + ["operation"], keep=False)].copy()
    data["label"] = run["label"]
    return data


def winners_from_rows(data, tie_threshold):
    records = []
    for config, group in data.groupby(["label"] + DRAW_KEYS, dropna=False):
        ranked = group.sort_values(["median_ms", "operation"])
        first = ranked.iloc[0]
        second = ranked.iloc[1] if len(ranked) > 1 else None
        ratio = second.median_ms / first.median_ms if second is not None else np.nan
        decisive = bool(np.isfinite(ratio) and ratio > 1 + tie_threshold)
        baseline = group.loc[group.operation == "index_add", "median_ms"]
        records.append(dict(zip(["label"] + DRAW_KEYS, config)) | {
            "winner": first.operation, "raw_winner": first.operation, "winner_ms": first.median_ms,
            "runner_up": second.operation if second is not None else None,
            "runner_up_ms": second.median_ms if second is not None else np.nan,
            "margin_ratio": ratio,
            "speedup_vs_index_add": baseline.iloc[0]/first.median_ms if len(baseline) else np.nan,
            "eligible_operations": len(group), "margin_aware_winner": first.operation if decisive else "tie",
            "decisive": decisive})
    return pd.DataFrame(records, columns=WIN_COLUMNS)


def matched_winners_from_summary(runs, ops=("index_add", "scatter_add", "embedding"), tie_threshold=0.05):
    """Intersect complete draw keys across every run and exactly the requested ops."""
    if len(runs) < 2:
        return pd.DataFrame(columns=WIN_COLUMNS)
    frames, common = [], None
    for run in runs:
        frame = eligible_summary(run)
        frame = frame[frame.operation.isin(ops)]
        complete = frame.groupby(DRAW_KEYS).operation.nunique()
        keys = set(complete[complete == len(ops)].index.tolist())
        common = keys if common is None else common & keys
        frames.append(frame)
    if not common:
        return pd.DataFrame(columns=WIN_COLUMNS)
    key_frame = pd.DataFrame(sorted(common), columns=DRAW_KEYS)
    data = pd.concat([frame.merge(key_frame, on=DRAW_KEYS, how="inner", validate="many_to_one")
                      for frame in frames], ignore_index=True)
    return winners_from_rows(data, tie_threshold)


def count_winners(data, labels, operations, margin=True):
    column = "margin_aware_winner" if margin else "raw_winner"
    records = []
    for label in labels:
        group = data[data.label == label]
        for op in list(operations) + (["tie"] if margin else []):
            count = int(group[column].eq(op).sum())
            records.append({"label": label, "operation": op, "win_count": count,
                            "total_draws": len(group), "win_rate": count/len(group) if len(group) else np.nan})
    return pd.DataFrame(records, columns=["label", "operation", "win_count", "total_draws", "win_rate"])


def add_overhead(run, notes):
    frame = run["summary"].copy()
    required = set(DRAW_KEYS + ["operation", "median_ms", "status"])
    if not required.issubset(frame):
        notes.append(f"{run['label']}: summary schema incomplete; matching and noop screening unavailable.")
        frame["overhead_multiple"] = np.nan
        return frame
    frame["median_ms"] = pd.to_numeric(frame.median_ms, errors="coerce")
    noop = frame[(frame.operation == "noop_kernel") & (frame.status == "ok")
                 & np.isfinite(frame.median_ms) & (frame.median_ms > 0)]
    noop = noop[~noop.duplicated(DRAW_KEYS, keep=False)][DRAW_KEYS + ["median_ms"]]
    noop = noop.rename(columns={"median_ms": "noop_ms"})
    if noop.empty:
        notes.append(f"{run['label']}: valid noop_kernel measurements missing; GPU-bound screening unavailable.")
    frame = frame.drop(columns=["noop_ms", "overhead_multiple"], errors="ignore").merge(noop, on=DRAW_KEYS, how="left", validate="many_to_one")
    frame["overhead_multiple"] = frame.median_ms / frame.noop_ms
    missing = frame.loc[frame.noop_ms.isna(), DRAW_KEYS].drop_duplicates()
    if len(missing) and not noop.empty:
        notes.append(f"{run['label']}: noop screening unavailable for {len(missing)} draw configurations.")
    return frame


def quality_summary(runs):
    records = []
    for run in runs:
        data = run["summary"]
        cv = pd.to_numeric(data.get("cv", pd.Series(dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan)
        err = pd.to_numeric(data.get("max_norm_err", pd.Series(dtype=float)), errors="coerce")
        record = {"label": run["label"], "summary_rows": len(data), "median_cv": cv.median(),
                  "p75_cv": cv.quantile(.75), "max_cv": cv.max(), "max_norm_err_max": err.max(),
                  "rows_with_failed_correctness": int(data.correctness.eq("failed").sum()) if "correctness" in data else np.nan}
        for field in ["sync_free", "determinism_bitwise", "determinism_allclose"]:
            values = data.get(field, pd.Series(index=data.index, dtype=object)).astype(str).str.lower()
            for value in ["true", "false"]:
                record[f"{field}_{value}_count"] = int(values.eq(value).sum())
            record[f"{field}_unchecked_count"] = int((~values.isin(["true", "false"])).sum())
        records.append(record)
    return pd.DataFrame(records)


def ratio_summary(data, keys, bootstrap_samples):
    columns = keys + ["count", "median_speedup", "geo_mean_speedup", "p25", "p75", "min_speedup",
                      "max_speedup", "geo_ci_low", "geo_ci_high", "bootstrap_clusters"]
    records = []
    for config, group in data.groupby(keys, dropna=False, sort=True):
        values = group.speedup_vs_index_add.to_numpy(dtype=float)
        row = dict(zip(keys, config)) | {"count": len(values), "median_speedup": np.median(values),
               "geo_mean_speedup": np.exp(np.log(values).mean()), "p25": np.quantile(values, .25),
               "p75": np.quantile(values, .75), "min_speedup": values.min(), "max_speedup": values.max(),
               "geo_ci_low": np.nan, "geo_ci_high": np.nan}
        # Resample base configurations, keeping their workload draws together.
        clusters = group.assign(log_ratio=np.log(group.speedup_vs_index_add)).groupby(BASE).log_ratio.agg(["sum", "count"])
        row["bootstrap_clusters"] = len(clusters)
        if bootstrap_samples and len(clusters) >= 2:
            rng = np.random.default_rng(1234)
            sums, counts = clusters["sum"].to_numpy(), clusters["count"].to_numpy()
            estimates = []
            for start in range(0, bootstrap_samples, 100):
                selection = rng.integers(len(clusters), size=(min(100, bootstrap_samples-start), len(clusters)))
                estimates.extend(np.exp(sums[selection].sum(axis=1)/counts[selection].sum(axis=1)))
            row["geo_ci_low"], row["geo_ci_high"] = np.quantile(estimates, [.025, .975])
        records.append(row)
    return pd.DataFrame(records, columns=columns)


def paper_tables(runs, args, notes):
    labels = [r["label"] for r in runs]
    frames = []
    for run in runs:
        run["summary"] = add_overhead(run, notes)
        frames.append(eligible_summary(run))
    all_rows = pd.concat(frames, ignore_index=True)
    own = winners_from_rows(all_rows, args.tie_threshold)
    matched = matched_winners_from_summary(runs, args.matched_ops, args.tie_threshold)
    screened = [dict(run, summary=run["summary"][run["summary"].overhead_multiple >= args.min_overhead_multiple]) for run in runs]
    gpu_matched = matched_winners_from_summary(screened, args.matched_ops, args.tie_threshold)
    tables = {
        "summary_results_with_overhead": pd.concat([run["summary"].assign(label=run["label"]) for run in runs], ignore_index=True),
        "within_run_margin_winners": own,
        "margin_winner_counts_by_run": pd.concat([
            count_winners(own, [run["label"]], [op for op in FAIR if op in set(frame.operation)])
            for run, frame in zip(runs, frames)], ignore_index=True),
        "matched_winners_common_ops": matched,
        "matched_winner_counts_by_run": count_winners(matched, labels, args.matched_ops, margin=False),
        "matched_margin_winner_counts_by_run": count_winners(matched, labels, args.matched_ops),
        "gpu_bound_summary_results": all_rows[all_rows.overhead_multiple >= args.min_overhead_multiple].copy(),
        "gpu_bound_matched_winners_common_ops": gpu_matched,
        "gpu_bound_matched_margin_winner_counts_by_run": count_winners(gpu_matched, labels, args.matched_ops),
        "quality_summary_by_run": quality_summary(runs),
    }
    ratios = []
    for frame in frames:
        baseline = frame[frame.operation == "index_add"][["label"] + DRAW_KEYS + ["median_ms"]].rename(columns={"median_ms": "index_add_ms"})
        joined = frame.merge(baseline, on=["label"] + DRAW_KEYS, how="inner", validate="many_to_one")
        joined["speedup_vs_index_add"] = joined.index_add_ms / joined.median_ms
        ratios.append(joined)
    all_ratios = pd.concat(ratios, ignore_index=True)
    all_ratios["speedup_vs_index_add"] = pd.to_numeric(all_ratios.speedup_vs_index_add, errors="coerce")
    all_ratios = all_ratios[np.isfinite(all_ratios.speedup_vs_index_add) & (all_ratios.speedup_vs_index_add > 0)]
    tables["all_op_speedup_vs_index_add"] = all_ratios
    tables["all_op_speedup_summary"] = ratio_summary(all_ratios, ["label", "operation"], args.bootstrap_samples)
    tables["all_op_speedup_summary_by_dim_distribution"] = ratio_summary(all_ratios, ["label", "operation", "dim", "distribution"], args.bootstrap_samples)
    matched_keys = matched[["label"] + DRAW_KEYS].drop_duplicates()
    matched_ratios = all_ratios[all_ratios.operation.isin(args.matched_ops)].merge(matched_keys, on=["label"] + DRAW_KEYS, how="inner", validate="many_to_one")
    tables["matched_all_op_speedup_vs_index_add"] = matched_ratios
    tables["matched_all_op_speedup_summary"] = ratio_summary(matched_ratios, ["label", "operation"], args.bootstrap_samples)
    for name, table in tables.items():
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
    return tables


def paper_plots(tables, args):
    files = []
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(args.out_dir / name, dpi=160)
        plt.close(fig)
        files.append(name)
    for key, filename, scope in [
        ("matched_margin_winner_counts_by_run", "matched_margin_winner_share_by_run.png", "Matched common configs/ops"),
        ("gpu_bound_matched_margin_winner_counts_by_run", "gpu_bound_matched_margin_winner_share_by_run.png", "Matched noop-screened subset")]:
        data = tables[key]
        if data.total_draws.max() == 0:
            continue
        pivot = data.pivot(index="label", columns="operation", values="win_rate")*100
        ax = pivot.plot.bar(stacked=True, color=[COLORS[op] for op in pivot], rot=0, figsize=(10, 6))
        n = int(data.total_draws.max())
        ax.set(ylabel="Margin-aware selections (%)", ylim=(0, 100),
               title=f"{scope}: {n} draw-configs/run\nDecisive margin > {100*args.tie_threshold:g}%; ties retained")
        ax.legend(title="Selection", bbox_to_anchor=(1.02, 1), loc="upper left")
        save(ax.figure, filename)
    matched = tables["matched_winners_common_ops"]
    for label, group in matched.groupby("label"):
        pivot = pd.crosstab(group.dim, group.margin_aware_winner)
        pivot = pivot.div(pivot.sum(axis=1), axis=0)*100
        ax = pivot.plot.bar(stacked=True, color=[COLORS[op] for op in pivot], rot=0, figsize=(9, 5))
        ax.set(ylabel="Margin-aware selections (%)", title=f"Matched common configs/ops: {label}\nDecisive margin > {100*args.tie_threshold:g}%", ylim=(0, 100))
        safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", label)
        save(ax.figure, f"matched_margin_winner_share_by_dim_{safe_label}.png")
    data = tables["matched_all_op_speedup_summary"]
    if not data.empty:
        pivot = data.pivot(index="label", columns="operation", values="geo_mean_speedup")
        ax = pivot.plot.bar(color=[COLORS[op] for op in pivot], rot=0, figsize=(10, 6))
        ax.axhline(1, color="gray", linestyle="--")
        ax.set(ylabel="Geometric mean speedup vs index_add", title="Matched common configs/ops: all wins and losses\nWithin-run ratios; not absolute cross-environment latency")
        save(ax.figure, "all_op_geomean_speedup_vs_index_add_by_run.png")
    return files


def markdown(frame):
    if frame.empty:
        return "No data available."
    def fmt(v):
        if pd.isna(v):
            return "—"
        if isinstance(v, (float, np.floating)):
            return f"{v:.4g}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
                     + ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)])


def matched_narrative(counts, scope):
    sentences = []
    for label, group in counts.groupby("label", sort=False):
        n = int(group.total_draws.max())
        if not n:
            sentences.append(f"- {label}: no eligible {scope} draw-configurations.")
            continue
        tied = int(group.loc[group.operation == "tie", "win_count"].sum())
        decisive = group[group.operation != "tie"]
        top_count = int(decisive.win_count.max())
        if top_count:
            names = ", ".join(decisive.loc[decisive.win_count == top_count, "operation"])
            claim = f"most frequent decisive API(s): {names}, with {top_count}/{n} draws each"
        else:
            claim = "no decisive API selections"
        sentences.append(f"- {label}: over {n} common {scope} draw-configurations, {claim}; near-ties: {tied}/{n} ({tied/n:.1%}).")
    return "\n".join(sentences)


def write_report(runs, tables, winners, notes, plots, out, args):
    env = tables["environment_summary"]
    matched = tables["matched_winners_common_ops"]
    common_n = len(matched[DRAW_KEYS].drop_duplicates())
    gpu_n = len(tables["gpu_bound_matched_winners_common_ops"][DRAW_KEYS].drop_duplicates())
    stack_keys = ["gpu_name", "pytorch_version", "cuda_version", "platform", "python_version"]
    stack_count = len(env[stack_keys].drop_duplicates())
    matched_counts = tables["matched_margin_winner_counts_by_run"]
    gpu_counts = tables["gpu_bound_matched_margin_winner_counts_by_run"]
    env_cols = ["label", *stack_keys, "compute_capability", "methodology_version"]
    grid_cols = ["label", "rows", "indices", "dims", "distributions", "zipf_mode", "workload_seeds", "repeats", "rounds", "ops"]
    ratios = tables["matched_all_op_speedup_summary"]
    quality = tables["quality_summary_by_run"]
    contribution = ("This study provides a reproducible empirical characterization of standard PyTorch CUDA APIs for embedding-style sparse-index gradient accumulation. In matched workload configurations across two CUDA software/hardware environments, the fastest API varies by workload and environment; margin-aware analysis distinguishes decisive wins from near-ties.")
    # Gate the requested study statement on the evidence rather than GPU names.
    different = not matched.empty and matched.groupby(DRAW_KEYS).margin_aware_winner.nunique().gt(1).any()
    if not (common_n and stack_count == 2 and different):
        contribution = "Proposed contribution statement, not fully established by the loaded inputs:\n\n" + contribution
    text = ["# Phase 2 Analysis: Sparse-Index Gradient Accumulation in PyTorch CUDA",
        "## 1. Scope and analysis rules",
        "This is an empirical systems characterization of standard PyTorch CUDA APIs for embedding-style sparse-index gradient accumulation during backward propagation. The analysis reads saved summaries and does not execute benchmarks. Headline candidates are correctness-passed, status-ok fair_dense_grad measurements. No algorithmic novelty or universal fastest API is claimed.",
        f"Practical decisions require runner_up_ms / winner_ms > {1+args.tie_threshold:g}; otherwise the selection is `tie`. A solitary eligible API is also not called decisive. This margin criterion is separate from the older quartile-overlap noise heuristic. Neither establishes statistical significance.",
        "## 2. Environments and coverage", markdown(env[env_cols]), markdown(env[grid_cols]),
        f"Loaded {len(runs)} runs representing {stack_count} distinct recorded GPU/PyTorch/CUDA/OS/Python stacks. Repeated runs of a stack are not independent hardware environments. Per-run descriptive coverage can differ; matched results below restrict configurations and candidate APIs.",
        "## 3. Descriptive within-run analysis",
        "The legacy winner tables/plots describe raw minimum-median selections on each run's own grid and API set. They are retained for exploration, not used as the main cross-run comparison. No cross-run absolute latency ranking is made.",
        markdown(tables["winner_counts_by_run"]),
        "Non-noise rates in this legacy table count saved within_noise=False cases divided by all retained draws. Unknown noise flags leave these rates unavailable. This is not the practical-margin rule.",
        "### Within-run margin-aware analysis", markdown(tables["margin_winner_counts_by_run"]),
        "These margin-aware counts still have unmatched coverage/API sets. They are descriptive. `within_run_margin_winners.csv` records the raw selection and runner-up for auditing.",
        "### Stability of raw selections across workload draws", markdown(tables["stable_winner_summary"]),
        "Stability here retains the legacy definition win_rate=1 for raw selections; it is not margin-aware stability. Equal maximum win rates are explicitly assigned `tie`, with tied APIs listed in stable_winner_details.csv. A one-draw configuration is trivially stable. Ineligible draws are absent from the denominator.",
        "## 4. Matched cross-run analysis",
        f"Selected common APIs: {', '.join(args.matched_ops)}. There are **{common_n} common draw-configurations per run**. A draw enters only if every selected API has a positive finite median, passed correctness, status=ok, and scope_group=fair_dense_grad in every run. Duplicate operation/draw records are excluded rather than cherry-picked.",
        "Keys: " + ", ".join(DRAW_KEYS) + ". Configuration matching is not a guarantee of bitwise-identical CUDA-generated source values across software versions. Parameters outside these keys are audited below.",
        "Only descriptive within-run analysis is available because the complete matched intersection is empty." if not common_n else "Raw matched selections are in matched_winners_common_ops.csv and matched_winner_counts_by_run.csv. Paper-facing results use the practical-margin counts below.",
        "## 5. Matched margin-aware analysis", matched_narrative(matched_counts, "matched"), markdown(matched_counts),
        "A tie is a near-tie between the fastest and runner-up; it does not imply every candidate is equally fast. Exact raw timing ties retain a reproducible operation ordering only for the audit column; they never become decisive selections.",
        "## 6. All-operation speedup analysis",
        "The following table includes every selected API's wins and losses on the identical matched draw-configurations. Ratios are index_add median / operation median within each run and draw; values below 1 are slowdowns.",
        markdown(ratios),
        "all_op_speedup_vs_index_add.csv and all_op_speedup_summary*.csv cover each run's own eligible API/baseline pairs and are descriptive. matched_all_op_speedup*.csv restrict both configurations and APIs for the main comparison. The older speedup_summary*.csv files are winner-conditioned and must not be interpreted as each API's average benefit.",
        f"Geometric means use exp(mean(log(ratio))). Optional 95% percentile intervals use {args.bootstrap_samples} bootstrap resamples with RNG seed 1234, clustering workload draws by BASE configuration; intervals are omitted with fewer than two clusters or zero samples. They describe resampling of this selected grid, not independent experiments, hardware populations, or a significance test.",
        "## 7. GPU-bound subset (noop-screened proxy)",
        f"The screen requires operation median / same-draw noop median >= {args.min_overhead_multiple:g}. **{gpu_n} common draw-configurations per run** survive for all selected APIs in all runs. Missing/invalid noop values cannot pass the screen. gpu_bound_summary_results.csv contains individually eligible fair rows; the matched GPU-bound file applies the stricter all-API/all-run intersection.",
        matched_narrative(gpu_counts, "noop-screened"), markdown(gpu_counts),
        "This is a launch-overhead proxy, not profiler proof of GPU-bound execution. Noop and accumulation paths may have different batching, host dispatch, allocation, or synchronization costs. Selection by this proxy changes the workload subset; report its size alongside the full matched results.",
        "## 8. Quality diagnostics", markdown(quality),
        "CV statistics cover available finite values in all saved summary rows, including diagnostics/noop; max_norm_err covers available validation values. Unchecked synchronization/determinism is not a pass. Counts and maxima describe the saved checks only; the analysis does not rerun numerical validation. Failed correctness rows are excluded from fair comparisons.",
        "## 9. Interpretation and relevance to training",
        "Any environment dependence refers to the combined hardware/software stack. GPU, OS, PyTorch, CUDA, and hosted runtime conditions differ together; these data do not isolate a hardware or software cause. No absolute timing across different stacks is treated as a controlled measurement. Practitioners should measure representative workloads on their deployment stack.",
        "The workload family is relevant to embedding backward, recommender/DLRM training, negative sampling, GNN aggregation/backward, MoE routing or token/expert dispatch, and custom scatter/gather gradients. These are motivating applications, not measured complete models. Results do not directly apply to dense GEMM-based linear/conv/attention matmul paths.",
        "## 10. Limitations and next phase",
        "Coverage is limited to the loaded stacks and sampled grids. Practical margins and raw-draw stability are not statistical significance. There is no profiler attribution, deterministic-mode series, FBGEMM/TorchRec baseline, custom CUDA implementation, or end-to-end model training. Generalization to other accelerators is untested. Phase 3 should profile representative decisive and near-tie cases, test a deterministic-mode subset, and consider index_put_(accumulate=True) and feasible FBGEMM/TorchRec comparisons.",
        "## 11. Safe contribution statement", contribution,
        "## Input audit", "\n".join(f"- {note}" for note in notes) or "No missing inputs or audit exceptions.",
        "## Generated plots", "\n".join(f"- [{name}]({name})" for name in plots)]
    (out / "phase2_analysis_report.md").write_text("\n\n".join(text)+"\n", encoding="utf-8")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    notes = []
    runs = []
    for label, path in args.runs:
        run = load_run(label, path, notes)
        if run is not None:
            runs.append(run)
    if not runs:
        print("No run directories loaded.\n" + "\n".join(notes))
        return 1
    tables, winners = make_tables(runs, args.out_dir)
    for key in ["dtype", "seed", "hotspot_fraction", "hotspot_rows", "zipf_alpha", "inner", "repeats", "rounds", "methodology_version"]:
        values = [run["metadata"].get("arguments", {}).get(key, run["metadata"].get(key)) for run in runs]
        if len({str(value) for value in values}) > 1 or any(value is None for value in values):
            notes.append(f"Matching audit: {key} differs or is unknown across runs: {values}. Keys alone do not control this factor.")
    tables.update(paper_tables(runs, args, notes))
    plots = make_plots(tables, winners, args.out_dir)
    plots += paper_plots(tables, args)
    write_report(runs, tables, winners, notes, plots, args.out_dir, args)
    manifest = {"runs": [{"label": r["label"], "path": r["path"]} for r in runs], "notes": notes,
                "plots": plots, "tables": [name + ".csv" for name in tables],
                "analysis_version": 2, "matched_ops": args.matched_ops, "tie_threshold": args.tie_threshold,
                "min_overhead_multiple": args.min_overhead_multiple, "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_seed": 1234, "bootstrap_unit": "BASE configuration (all draws retained together)",
                "matched_common_draw_count": len(tables["matched_winners_common_ops"][DRAW_KEYS].drop_duplicates()),
                "gpu_bound_common_draw_count": len(tables["gpu_bound_matched_winners_common_ops"][DRAW_KEYS].drop_duplicates()),
                "assumptions": ["Legacy/within-run tables are descriptive; matched tables intersect complete draw keys and selected APIs across all loaded runs.",
                                "GPU-bound matched subsets require every selected API in every run to pass the noop screen; this is only an overhead proxy.",
                                "Non-noise rates use all retained winner draws as denominator.",
                                "Stable means winner consistency across eligible draws, not statistical significance."]}
    (args.out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Loaded {len(runs)} runs: " + ", ".join(r["label"] for r in runs))
    print(tables["winner_counts_by_run"][["label", "operation", "per_draw_win_count", "non_noise_win_count"]].to_string(index=False))
    print(tables["stable_winner_summary"][["label", "stable_configurations", "mixed_configurations"]].to_string(index=False))
    print("Matched margin-aware counts:")
    print(tables["matched_margin_winner_counts_by_run"].to_string(index=False))
    print("GPU-bound matched margin-aware counts (proxy):")
    print(tables["gpu_bound_matched_margin_winner_counts_by_run"].to_string(index=False))
    for note in notes:
        print("NOTE:", note)
    print(f"Outputs: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
