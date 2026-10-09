"""Offline comparison of saved Phase 3 profiles and Phase 2 analysis tables.

No torch, CUDA, profiler, or benchmark imports. Matplotlib is optional.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd


OPS = ["index_add", "scatter_add", "embedding"]
PARAMS = ["rows", "indices", "dim", "distribution", "workload_draw", "zipf_mode",
          "row_layout", "index_order", "cache_mode"]
AUDIT = ["hotspot_fraction", "hotspot_rows", "zipf_alpha", "seed", "index_sha256"]
CATEGORIES = [
    ("embedding_accumulate", r"compute_grad_weight|sum_and_scatter|EmbeddingBackward"),
    ("sort_scan_unique", r"RadixSort|UniqueByKey|DeviceScan|CompactInit|ScanKernel|ScanInit"),
    ("index_add_accumulate", r"indexFuncLargeIndex"),
    ("scatter_add_accumulate", r"scatter_gather|scatter_add"),
    ("zero_fill_memset", r"FillFunctor|Memset|zero|fill"),
    ("arange_index_setup", r"arange|RangeFactories|elementwise_kernel_with_index"),
    ("other", r".*"),
]
COLORS = {"index_add": "#e69f00", "scatter_add": "#0072b2", "embedding": "#009e73"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, default=Path("sparse-gradient/phase3_profile_local_v2"))
    parser.add_argument("--t4-dir", type=Path, default=Path("sparse-gradient/phase3_profile_t4_v2"))
    parser.add_argument("--phase2-dir", type=Path, default=Path("sparse-gradient/phase2_analysis_v2"))
    parser.add_argument("--out-dir", type=Path, default=Path("sparse-gradient/phase3_comparison"))
    return parser.parse_args()


def read_csv(path, required):
    try:
        data = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Cannot read required CSV {path}: {exc}") from exc
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Required columns missing from {path}: {sorted(missing)}")
    return data


def read_metadata(path):
    try:
        return json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read metadata from {path}: {exc}") from exc


def truth(value):
    return str(value).lower() == "true"


def same_value(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    return a == b


def read_profile_cases(path, label):
    required = ["case_id", "operation", *PARAMS, "status", "correctness", "median_ms", "p25_ms", "p75_ms", "cv",
                "profile_sort_basis", "kernel_ms_per_invocation", "gpu_busy_fraction", "likely_host_bound"]
    required += [f"top_cuda_operator_{i}{suffix}" for i in range(1, 4) for suffix in ["", "_time_ms"]]
    data = read_csv(path / "profile_cases.csv", required)
    if data.duplicated(["case_id", "operation"]).any():
        raise ValueError(f"Duplicate case/operation rows in {path}/profile_cases.csv")
    data["environment_label"] = label
    return data


def read_case_winners(path, label):
    data = read_csv(path / "case_winners.csv", ["case_id", "winner", "runner_up", "margin_ratio", "decisive_5pct",
                                              "complete_ops", "eligible_operations", *[f"{op}_ms" for op in OPS]])
    if data.case_id.duplicated().any():
        raise ValueError(f"Duplicate case IDs in {path}/case_winners.csv")
    data["environment_label"] = label
    return data


def validate_cases(local, t4, notes):
    left, right = set(local.case_id), set(t4.case_id)
    if left != right:
        notes.append(f"Case-ID mismatch: local-only={sorted(left-right)}; T4-only={sorted(right-left)}.")
    common, pairing = [], {}
    for case_id in sorted(left & right):
        a, b = local[local.case_id == case_id], t4[t4.case_id == case_id]
        if len(a[PARAMS].drop_duplicates()) != 1 or len(b[PARAMS].drop_duplicates()) != 1:
            notes.append(f"{case_id}: inconsistent parameters within an environment; excluded from paired comparisons.")
            continue
        mismatch = [key for key in PARAMS if not same_value(a[key].iloc[0], b[key].iloc[0])]
        for key in AUDIT:
            if key not in a or key not in b:
                notes.append(f"{case_id}: audit field {key} unavailable.")
        for key in ["hotspot_fraction", "hotspot_rows", "zipf_alpha", "seed"]:
            if key not in a or key not in b:
                continue
            if a[key].nunique(dropna=False) != 1 or b[key].nunique(dropna=False) != 1:
                mismatch.append(key)
            elif not same_value(a[key].iloc[0], b[key].iloc[0]):
                mismatch.append(key)
        index_hash_match = False
        index_hash_available = "index_sha256" in a and "index_sha256" in b and not a.index_sha256.isna().any() and not b.index_sha256.isna().any()
        if index_hash_available:
            index_hash_match = (a.index_sha256.nunique() == 1 and b.index_sha256.nunique() == 1
                                and same_value(a.index_sha256.iloc[0], b.index_sha256.iloc[0]))
            if not index_hash_match:
                notes.append(f"{case_id}: index_sha256 mismatch; pairing downgraded without exclusion.")
        else:
            notes.append(f"{case_id}: index_sha256 audit unavailable.")
        if mismatch:
            notes.append(f"{case_id}: parameter mismatch {mismatch}; excluded from paired comparisons.")
            continue
        common.append(case_id)
        pairing[case_id] = {"index_hash_match": index_hash_match,
                    "workload_pairing_quality": ("exact_index_match" if index_hash_match
                                  else "mismatch" if index_hash_available
                                  else "parameter_match_only")}
        for field in ["src_sha256"]:
            if field not in a or field not in b or a[field].isna().any() or b[field].isna().any():
                notes.append(f"{case_id}: {field} audit unavailable.")
                continue
            if not (a[field].nunique() == 1 and b[field].nunique() == 1 and same_value(a[field].iloc[0], b[field].iloc[0])):
                notes.append(f"{case_id}: {field} mismatch (informational only; source RNG may differ across stacks).")
    return common, left == right, pairing


def validate_winner(row, cases, notes):
    good = cases[(cases.status == "ok") & (cases.correctness == "passed")
                 & cases.operation.isin(OPS) & np.isfinite(cases.median_ms) & (cases.median_ms > 0)]
    if row is None or not truth(row.complete_ops) or set(good.operation) != set(OPS):
        return False
    ranked = good.sort_values(["median_ms", "operation"])
    ratio = ranked.median_ms.iloc[1] / ranked.median_ms.iloc[0]
    consistent = (row.winner == ranked.operation.iloc[0] and row.runner_up == ranked.operation.iloc[1]
                  and np.isclose(row.margin_ratio, ratio) and truth(row.decisive_5pct) == (ratio > 1.05)
                  and row.eligible_operations == len(OPS))
    for op in OPS:
        consistent = consistent and np.isclose(row[f"{op}_ms"], good.loc[good.operation == op, "median_ms"].iloc[0])
    if not consistent:
        notes.append(f"{cases.case_id.iloc[0]} {cases.environment_label.iloc[0]}: saved winner inconsistent with valid operation timings; marked incomplete.")
    return bool(consistent)


def case_quality(cases, winner):
    good = cases[(cases.status == "ok") & (cases.correctness == "passed") & cases.operation.isin(OPS)]
    gpu_clean = len(good) == len(OPS) and np.isfinite(pd.to_numeric(good.gpu_busy_fraction, errors="coerce")).all() and not good.likely_host_bound.map(truth).any()
    if winner is None or winner.winner not in OPS or winner.runner_up not in OPS or len(good) != len(OPS):
        return gpu_clean, False, False
    rows = {row.operation: row for row in good.itertuples(index=False)}
    winner_row, runner_row = rows[winner.winner], rows[winner.runner_up]
    iqr_separated = winner_row.p75_ms < runner_row.p25_ms
    margin_gt_5pct = runner_row.median_ms / winner_row.median_ms > 1.05
    cv_guard_pass = winner.margin_ratio > 1.05 + max(winner_row.cv, runner_row.cv)
    return gpu_clean, bool(iqr_separated), bool(cv_guard_pass)


def build_case_comparison(local_winners, t4_winners, local_cases, t4_cases, common, pairing, notes):
    records = []
    for case_id in common:
        left, right = local_cases[local_cases.case_id == case_id], t4_cases[t4_cases.case_id == case_id]
        item = {"case_id": case_id} | left.iloc[0][PARAMS].to_dict()
        complete = []
        for prefix, data, cases in [("local", local_winners, left), ("t4", t4_winners, right)]:
            data = data[data.case_id == case_id]
            row = data.iloc[0] if len(data) else None
            complete.append(validate_winner(row, cases, notes))
            for key in ["winner", "runner_up", "margin_ratio", "decisive_5pct", *[f"{op}_ms" for op in OPS]]:
                item[f"{prefix}_{key}"] = row[key] if row is not None else None
            item[f"{prefix}_complete_ops"] = complete[-1]
            gpu_clean, iqr_separated, cv_guard_pass = case_quality(cases, row)
            item[f"{prefix}_gpu_clean"] = gpu_clean
            item[f"{prefix}_iqr_separated"] = iqr_separated
            item[f"{prefix}_cv_guard_pass"] = cv_guard_pass
        decisive_a, decisive_b = truth(item["local_decisive_5pct"]), truth(item["t4_decisive_5pct"])
        if not all(complete):
            relation, interpretation = "incomplete", "Incomplete, failed, or inconsistent selected API set; no winner comparison claim."
        elif item["local_winner"] == item["t4_winner"]:
            relation = "same_decisive" if decisive_a and decisive_b else "same_but_near_tie"
            interpretation = (f"Both environments select {item['local_winner']} decisively for this representative case." if relation == "same_decisive"
                              else "Both select the same raw minimum, but at least one side is a near-tie.")
        elif decisive_a and decisive_b:
            relation = "reversal_decisive"
            interpretation = f"T4 selects {item['t4_winner']} decisively while local selects {item['local_winner']} decisively; this is a representative reversal."
        else:
            relation, interpretation = "reversal_with_near_tie", "The environments differ, but at least one side is near-tie; treat as weak evidence."
        item["paired_gpu_clean"] = bool(item["local_gpu_clean"] and item["t4_gpu_clean"])
        item["paired_strong_timing_evidence"] = bool(all(complete) and
            (item["local_iqr_separated"] or item["local_cv_guard_pass"]) and
            (item["t4_iqr_separated"] or item["t4_cv_guard_pass"]))
        if not item["paired_gpu_clean"] and relation != "incomplete":
            relation += "_with_host_warning"
            interpretation += " Host-overhead warning prevents a GPU-clean claim."
        item.update(winner_relation=relation, interpretation=interpretation)
        item.update(pairing[case_id])
        records.append(item)
    columns = ["case_id", *PARAMS] + [f"{prefix}_{key}" for prefix in ["local", "t4"] for key in
                ["winner", "runner_up", "margin_ratio", "decisive_5pct", *[f"{op}_ms" for op in OPS], "complete_ops", "gpu_clean", "iqr_separated", "cv_guard_pass"]] + ["paired_gpu_clean", "paired_strong_timing_evidence", "index_hash_match", "workload_pairing_quality", "winner_relation", "interpretation"]
    return pd.DataFrame(records, columns=columns)


def build_operation_comparison(local_cases, t4_cases, common):
    metrics = ["median_ms", "gpu_busy_fraction", "likely_host_bound", "top_cuda_operator_1", "kernel_ms_per_invocation",
               "profile_sort_basis", "status", "correctness"]
    parts = []
    for prefix, data in [("local", local_cases), ("t4", t4_cases)]:
        data = data[data.case_id.isin(common) & data.operation.isin(OPS)]
        parts.append(data[["case_id", "operation", *PARAMS, *metrics]].rename(columns={key: f"{prefix}_{key}" for key in metrics}))
    return parts[0].merge(parts[1], on=["case_id", "operation", *PARAMS], how="outer", validate="one_to_one")


def normalize_kernel_name(name):
    name = str(name)
    if "indexFuncLargeIndex" in name:
        return "indexFuncLargeIndex/ReduceAdd"
    if re.search(r"scatter_gather|vectorized_scatter_add_kernel", name):
        return "scatter-gather ReduceAdd"
    for token in ["compute_grad_weight", "sum_and_scatter", "DeviceRadixSort", "DeviceUniqueByKey", "DeviceScan", "Memset", "FillFunctor"]:
        if token.lower() in name.lower():
            return token
    return "other CUDA kernel"


def categorize_kernel(name):
    return next(category for category, pattern in CATEGORIES if re.search(pattern, str(name), re.IGNORECASE))


def build_kernel_summary(cases):
    records = []
    for row in cases.itertuples(index=False):
        item = {"environment_label": row.environment_label, "case_id": row.case_id, "operation": row.operation,
                "profile_sort_basis": row.profile_sort_basis, "status": row.status, "correctness": row.correctness}
        patterns = []
        for rank in range(1, 4):
            key = getattr(row, f"top_cuda_operator_{rank}")
            item[f"top{rank}_kernel"] = key
            item[f"top{rank}_time_ms_total_profile"] = getattr(row, f"top_cuda_operator_{rank}_time_ms")
            if pd.notna(key):
                patterns.append(normalize_kernel_name(key))
        item["kernel_pattern_short"] = "; ".join(dict.fromkeys(patterns)) if row.profile_sort_basis == "cuda_kernel_events" else "CUDA kernel attribution unavailable"
        if row.operation == "embedding" and patterns and set(patterns) == {"other CUDA kernel"}:
            item["kernel_pattern_short"] = "embedding backward mixed kernels"
        records.append(item)
    return pd.DataFrame(records)


def load_top_or_event_files(path, label, cases, notes):
    records = []
    for row in cases.itertuples(index=False):
        if row.status != "ok" or row.correctness != "passed" or row.profile_sort_basis != "cuda_kernel_events":
            notes.append(f"{label}/{row.case_id}/{row.operation}: unavailable/invalid CUDA attribution; category shares skipped.")
            continue
        candidates = [(path / "events" / f"profile_events_{row.case_id}_{row.operation}.csv", "full_cuda_events"),
                      (path / "events" / f"profile_top_cuda_{row.case_id}_{row.operation}.csv", "top10_cuda_events")]
        for file, source in candidates:
            if not file.is_file():
                continue
            try:
                frame = pd.read_csv(file)
                if not {"key", "cuda_time_total_us"}.issubset(frame):
                    raise ValueError("CUDA name/time columns missing")
                if "device_type" in frame:
                    frame = frame[frame.device_type.astype(str).str.upper().str.endswith("CUDA")]
                elif source == "full_cuda_events":
                    raise ValueError("Full event file lacks device_type; cannot safely identify CUDA rows")
                frame = frame[~frame.key.astype(str).str.startswith("phase3::")]
                if "is_user_annotation" in frame:
                    frame = frame[~frame.is_user_annotation.map(truth)]
                frame["cuda_time_total_us"] = pd.to_numeric(frame.cuda_time_total_us, errors="coerce")
                frame = frame[np.isfinite(frame.cuda_time_total_us) & (frame.cuda_time_total_us > 0)]
                if frame.empty:
                    raise ValueError("No usable CUDA events")
                if source == "top10_cuda_events":
                    notes.append(f"{label}/{row.case_id}/{row.operation}: using top-10 only; shares describe the captured top-10, not full composition.")
                for event in frame.to_dict("records"):
                    records.append({"environment_label": label, "case_id": row.case_id, "operation": row.operation,
                                    "key": event["key"], "time_ms": event["cuda_time_total_us"]/1000,
                                    "calls": event.get("calls", 1), "source": source})
                break
            except (ValueError, OSError, pd.errors.ParserError) as exc:
                notes.append(f"Skipped {file}: {exc}")
        else:
            notes.append(f"{label}/{row.case_id}/{row.operation}: no usable event files; category shares skipped.")
    return pd.DataFrame(records, columns=["environment_label", "case_id", "operation", "key", "time_ms", "calls", "source"])


def build_kernel_category_summary(events):
    columns = ["environment_label", "case_id", "operation", "category", "category_time_ms_total_profile",
               "category_share_of_profile_cuda_time", "top10_category_share", "kernel_count", "source"]
    if events.empty:
        return pd.DataFrame(columns=columns)
    data = events.copy()
    data["category"] = data.key.map(categorize_kernel)
    keys = ["environment_label", "case_id", "operation", "source"]
    grouped = data.groupby(keys + ["category"]).agg(category_time_ms_total_profile=("time_ms", "sum"), kernel_count=("calls", "sum")).reset_index()
    total = grouped.groupby(keys).category_time_ms_total_profile.transform("sum")
    shares = grouped.category_time_ms_total_profile / total
    grouped["category_share_of_profile_cuda_time"] = shares.where(grouped.source == "full_cuda_events")
    grouped["top10_category_share"] = shares.where(grouped.source == "top10_cuda_events")
    return grouped[columns]


def build_phase2_case_join(cases, phase2, notes):
    join_keys = ["rows", "indices", "dim", "distribution", "zipf_mode", "row_layout", "index_order", "cache_mode", "workload_draw"]
    required = join_keys + ["label"]
    missing = set(required) - set(phase2.columns)
    if missing:
        notes.append(f"Phase 2 case-level join unavailable; missing columns: {sorted(missing)}.")
        return pd.DataFrame(columns=["case_id", "rows", "indices", "dim", "distribution", "workload_draw", "in_phase2_grid",
                                     "phase2_local_margin_aware_winner", "phase2_t4_margin_aware_winner",
                                     "phase2_t4_repeat_margin_aware_winner", "phase3_local_winner", "phase3_t4_winner",
                                     "local_phase2_phase3_agree", "t4_phase2_phase3_agree", "notes"])
    winner_column = "margin_aware_winner" if "margin_aware_winner" in phase2 else "winner"
    if winner_column == "winner":
        notes.append("Phase 2 case join uses winner fallback because margin_aware_winner is unavailable; interpret cautiously.")
    rows = []
    phase3 = cases[["case_id", *join_keys]].drop_duplicates()
    for item in phase3.to_dict("records"):
        matches = phase2
        for key in join_keys:
            matches = matches[matches[key].map(lambda value: same_value(value, item[key]))]
        values = {}
        join_notes = []
        for label, output in [("GTX1650Ti", "phase2_local_margin_aware_winner"),
                              ("T4", "phase2_t4_margin_aware_winner"),
                              ("T4_repeat", "phase2_t4_repeat_margin_aware_winner")]:
            selected = matches[matches.label == label]
            values[output] = ", ".join(sorted(set(selected[winner_column].dropna().astype(str)))) if len(selected) else None
        values["in_phase2_grid"] = bool(values["phase2_local_margin_aware_winner"] or values["phase2_t4_margin_aware_winner"] or values["phase2_t4_repeat_margin_aware_winner"])
        rows.append({"case_id": item["case_id"], "rows": item["rows"], "indices": item["indices"], "dim": item["dim"],
                     "distribution": item["distribution"], "workload_draw": item["workload_draw"], **values,
                     "phase3_local_winner": None, "phase3_t4_winner": None,
                     "local_phase2_phase3_agree": None, "t4_phase2_phase3_agree": None,
                     "notes": "; ".join(join_notes)})
    return pd.DataFrame(rows)


def complete_phase2_case_join(joined, cases):
    if joined.empty:
        return joined
    winners = cases[["case_id", "local_winner", "t4_winner"]].drop_duplicates()
    joined = joined.drop(columns=["phase3_local_winner", "phase3_t4_winner"]).merge(winners, on="case_id", how="left")
    joined = joined.rename(columns={"local_winner": "phase3_local_winner", "t4_winner": "phase3_t4_winner"})
    for side, phase2_column, phase3_column, output in [("local", "phase2_local_margin_aware_winner", "phase3_local_winner", "local_phase2_phase3_agree"),
                                                       ("t4", "phase2_t4_margin_aware_winner", "phase3_t4_winner", "t4_phase2_phase3_agree")]:
        joined[output] = joined.apply(lambda row: (row[phase2_column] is not None and not pd.isna(row[phase2_column]) and
                                                    row[phase3_column] in str(row[phase2_column]).split(", ")) if row[phase3_column] else None, axis=1)
    return joined


def build_claim_checks(categories, counts, cases):
    rows = []
    full = categories[categories.source == "full_cuda_events"].copy()
    def mean_share(label, operation, category):
        values = full[(full.environment_label == label) & (full.operation == operation) & (full.category == category)].category_share_of_profile_cuda_time
        return values.mean() if len(values) else np.nan
    def add(name, passed, numbers, caution):
        rows.append({"check_name": name, "passed": bool(passed), "supporting_numbers": numbers, "caution": caution})
    for operation, category, name, threshold, strict in [("index_add", "index_add_accumulate", "index_add_path_dominance", .5, False),
                                                          ("scatter_add", "scatter_add_accumulate", "scatter_add_path_dominance", .5, False),
                                                          ("embedding", "embedding_accumulate", "embedding_accumulation_present", 0, True),
                                                          ("embedding", "sort_scan_unique", "embedding_sort_scan_present", .05, True)]:
        values = {label: mean_share(label, operation, category) for label in ["GTX1650Ti", "T4"]}
        passed = all(np.isfinite(value) and (value > threshold if strict else value >= threshold) for value in values.values())
        add(name, passed, "; ".join(f"{label}={value:.4g}" if np.isfinite(value) else f"{label}=unavailable" for label, value in values.items()), "Full-event CUDA category shares are required; top-10 fallback data do not establish this check.")
    other = full[full.category == "other"].groupby(["environment_label", "operation"]).category_share_of_profile_cuda_time.mean()
    other_text = "; ".join(f"{label}/{operation}={value:.4g}" for (label, operation), value in other.items()) or "unavailable"
    add("other_share_controlled", len(other) > 0 and (other < .2).all(), other_text, "Unknown kernels are substantial wherever the mean other share is >= 0.2.")
    expected = {"GTX1650Ti": "embedding", "T4": "scatter_add", "T4_repeat": "scatter_add"}
    phase2_pass = True
    phase2_text = []
    for label, operation in expected.items():
        selected = counts[(counts.label == label) & (counts.operation != "tie")]
        max_count = selected.win_count.max() if len(selected) else np.nan
        passed = len(selected) > 0 and operation in set(selected[selected.win_count == max_count].operation)
        phase2_pass = phase2_pass and passed
        phase2_text.append(f"{label}={operation if passed else 'fail'}")
    add("phase2_environment_difference_present", phase2_pass, "; ".join(phase2_text), "Requires the expected most-frequent decisive API in all three Phase 2 environments.")
    reversal = cases.winner_relation.isin(["reversal_decisive", "reversal_decisive_with_host_warning"])
    add("representative_reversals_present", reversal.any(), ", ".join(cases.loc[reversal, "case_id"]) or "none", "Representative subset evidence remains illustrative, not statistical.")
    strong = cases.winner_relation.isin(["reversal_decisive", "reversal_decisive_with_host_warning"]) & cases.paired_gpu_clean & cases.paired_strong_timing_evidence & cases.workload_pairing_quality.eq("exact_index_match")
    add("strong_clean_reversals_present", strong.any(), ", ".join(cases.loc[strong, "case_id"]) or "none", "Requires clean GPU proxies, stable timing evidence, and exact index pairing.")
    return pd.DataFrame(rows)


def build_alignment(counts, speedups, cases):
    rows = []
    for label, group in counts.groupby("label", sort=False):
        decisive = group[(group.operation != "tie") & (group.win_count > 0)].sort_values("win_count", ascending=False)
        top = ", ".join(decisive.loc[decisive.win_count.eq(decisive.win_count.max()), "operation"]) if len(decisive) else "none"
        numbers = "; ".join(f"{r.operation}={int(r.win_count)}/{int(r.total_draws)}" for r in group.itertuples())
        ratios = speedups[speedups.label == label]
        numbers += "; geometric mean ratios: " + ", ".join(f"{r.operation}={r.geo_mean_speedup:.3f}x" for r in ratios.itertuples())
        rows.append({"evidence_type": "Phase 2 matched margin-aware", "environment_label": label,
                     "main_observation": f"Most frequent decisive API(s): {top}.", "supporting_numbers": numbers,
                     "caution": "Main grid-level evidence; 5% practical margin is not statistical significance. Ratios include wins and losses."})
    for prefix, label in [("local", "GTX1650Ti"), ("t4", "T4")]:
        valid = cases[cases.winner_relation != "incomplete"]
        details = "; ".join(f"{r['case_id']}={r[prefix+'_winner']} ({'decisive' if truth(r[prefix+'_decisive_5pct']) else 'near-tie'})" for _, r in valid.iterrows())
        rows.append({"evidence_type": "Phase 3 representative subset", "environment_label": label,
                     "main_observation": "Within-subset event-timing minima with 5% margin caveats.", "supporting_numbers": details,
                     "caution": "Representative probes, not a statistical replication of Phase 2 winner frequencies."})
    for row in cases.itertuples():
        rows.append({"evidence_type": "Phase 3 paired case", "environment_label": "GTX1650Ti vs T4",
                     "main_observation": row.interpretation,
                     "supporting_numbers": f"{row.case_id}: {row.local_winner} vs {row.t4_winner}; margins {row.local_margin_ratio:.4g}, {row.t4_margin_ratio:.4g}",
                     "caution": "Profiler paths inform attribution; no isolated hardware-causal inference."})
    return pd.DataFrame(rows)


def make_plots(cases, operations, categories, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=200)
        plt.close(fig)
        files.append(name)
    if not cases.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(cases))
        for prefix, offset, marker, label in [("local", -.12, "o", "GTX1650Ti"), ("t4", .12, "s", "T4")]:
            for i, (_, row) in enumerate(cases.iterrows()):
                if row.winner_relation == "incomplete":
                    continue
                decisive = truth(row[f"{prefix}_decisive_5pct"])
                op = row[f"{prefix}_winner"]
                ax.scatter(i+offset, OPS.index(op), marker=marker, s=100, facecolors=COLORS[op] if decisive else "none", edgecolors=COLORS[op])
            ax.scatter([], [], marker=marker, color="black", label=label)
        ax.set(xticks=x, xticklabels=cases.case_id, yticks=range(3), yticklabels=OPS, ylim=(-.5, 2.5),
               title="Representative subset minima: filled = >5% margin; hollow = near-tie")
        ax.legend()
        save(fig, "phase3_case_winner_comparison.png")
    if not operations.empty:
        for metric, filename, ylabel in [("median_ms", "phase3_median_ms_by_case_operation.png", "Unprofiled CUDA-event median (ms)"),
                                          ("gpu_busy_fraction", "phase3_gpu_busy_fraction_by_case_operation.png", "Profiler-derived GPU-busy proxy")]:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, prefix, label in zip(axes, ["local", "t4"], ["GTX1650Ti", "T4"]):
                valid = operations[(operations[f"{prefix}_status"] == "ok") & (operations[f"{prefix}_correctness"] == "passed")]
                data = valid.pivot(index="case_id", columns="operation", values=f"{prefix}_{metric}").reindex(columns=OPS)
                if len(data):
                    data.plot.bar(ax=ax, color=[COLORS[op] for op in OPS], rot=0)
                ax.set(title=label, ylabel=ylabel, xlabel="Case")
                if metric == "gpu_busy_fraction":
                    ax.axhline(.6, color="red", linestyle="--", label="Warning threshold 0.6")
                    ax.legend(fontsize=8)
            fig.suptitle("Diagnostic values only: stacks are not controlled hardware comparisons" if metric == "median_ms" else "Host-overhead warning proxy; not hardware utilization")
            save(fig, filename)
    if not categories.empty:
        sources = categories.source.unique()
        fig, axes = plt.subplots(1, len(sources), figsize=(11*len(sources), 6), squeeze=False)
        for ax, source in zip(axes[0], sources):
            data = categories[categories.source == source]
            share = "category_share_of_profile_cuda_time" if source == "full_cuda_events" else "top10_category_share"
            per_case = data.pivot_table(index=["environment_label", "operation", "case_id"], columns="category", values=share, fill_value=0)
            mean = per_case.groupby(level=[0, 1]).mean()
            mean.index = [f"{label}\n{op}" for label, op in mean.index]
            mean.plot.bar(stacked=True, ax=ax, rot=25, colormap="tab20")
            ax.set(ylabel="Mean per-case share (equal case weights)", ylim=(0, 1), xlabel="",
                   title="Full CUDA event composition" if source == "full_cuda_events" else "Top-10 only composition (not full profile)")
            ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
        save(fig, "phase3_kernel_category_share.png")
    return files


def markdown(frame):
    if frame.empty:
        return "No eligible data available."
    def fmt(v):
        if pd.isna(v):
            return "-"
        if isinstance(v, (float, np.floating)):
            return f"{v:.4g}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"] +
                     ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)])


LIMITATIONS = """- Phase 3 is a representative subset, not a full grid or statistical replication of Phase 2.
- Event timings use inner=1 and fixed operation order; Phase 2 remains the main randomized/batched winner-count evidence.
- Profiler overhead is not used as primary benchmark timing. Profiler CUDA totals and event medians are not interchangeable.
- GPU-busy fraction is a proxy, not hardware utilization; it can exceed 1. likely_host_bound is a warning signal, not causal proof.
- No Nsight Compute counters, end-to-end model training, or custom CUDA implementation.
- Absolute GTX/T4 latencies are not controlled hardware benchmarks; GPU, OS, PyTorch, CUDA, and hosted runtime differ together.
- Ordered name-based categories are approximate. Full shares exclude annotations; top-10 fallback shares describe only that truncated list.
- Casewise means weight each available case equally, not by kernel duration. Shares summarize kernel work, not profiler wall-clock utilization.
- CPU/NumPy index hash discrepancies are audit warnings; source RNG values can differ across software stacks even with equal seeds.
"""


def write_markdown_report(tables, combined, args, notes, plots):
    cases, env = tables["phase3_case_comparison"], tables["phase3_environment_summary"]
    sections = []
    for row in cases.itertuples():
        sections.append(f"- **{row.case_id}** ({row.winner_relation}): {row.interpretation}")
    warnings = combined[combined.likely_host_bound.map(truth)]
    busy = combined.groupby("environment_label").gpu_busy_fraction.agg(["min", "median", "max"]).reset_index()
    patterns = tables["phase3_kernel_summary"].groupby(["environment_label", "operation"]).kernel_pattern_short.agg(lambda s: "; ".join(sorted(set(s)))).reset_index()
    categories = tables["phase3_kernel_category_summary"]
    full_categories = categories[categories.source == "full_cuda_events"]
    category_means = (full_categories.groupby(["environment_label", "operation", "category"])
                      .category_share_of_profile_cuda_time.mean().reset_index())
    other_warning = category_means[(category_means.category == "other") & (category_means.category_share_of_profile_cuda_time >= .2)]
    def ref(path):
        return Path(os.path.relpath(path, args.out_dir)).as_posix()
    source_refs = "\n".join(f"- [{label} Phase 3 report]({ref(path / 'profile_phase3_report.md')}); [profile cases]({ref(path / 'profile_cases.csv')}); [events]({ref(path / 'events')})." for label, path in [("GTX1650Ti", args.local_dir), ("T4", args.t4_dir)])
    source_refs += f"\n- [Phase 2 matched analysis]({ref(args.phase2_dir / 'phase2_analysis_report.md')}); [matched all-operation ratios]({ref(args.phase2_dir / 'matched_all_op_speedup_summary.csv')})."
    checks = tables["phase3_claim_checks"].set_index("check_name")
    safe_lines = ["Phase 3 profiling provides representative execution-path evidence for the profiled subset. It does not replace the matched Phase 2 grid-level winner analysis."]
    conditional_claims = {
        "index_add_path_dominance": "`index_add` and `scatter_add` are dominated by distinct CUDA reduction kernels in both profiled environments.",
        "embedding_accumulation_present": "`Embedding.backward` follows a multi-kernel backward path that includes accumulation and sorting/segmentation-related kernels.",
        "phase2_environment_difference_present": "Phase 2 shows environment-dependent winner regimes in the matched grid: the local GTX1650Ti stack most often favors embedding, while the T4 stacks most often favor scatter_add under the practical-margin rule.",
    }
    if checks.loc["index_add_path_dominance", "passed"] and checks.loc["scatter_add_path_dominance", "passed"]:
        safe_lines.append(conditional_claims["index_add_path_dominance"])
    if checks.loc["embedding_accumulation_present", "passed"] and checks.loc["embedding_sort_scan_present", "passed"]:
        safe_lines.append(conditional_claims["embedding_accumulation_present"])
    if checks.loc["phase2_environment_difference_present", "passed"]:
        safe_lines.append(conditional_claims["phase2_environment_difference_present"])
    if checks.loc["strong_clean_reversals_present", "passed"]:
        safe_lines.append("Representative Phase 3 cases include clean decisive reversals, supporting the interpretation that the relative cost of these API paths changes with workload and software/hardware environment.")
    elif checks.loc["representative_reversals_present", "passed"]:
        safe_lines.append("Representative Phase 3 cases include decisive reversals, but at least one stricter audit gate is not satisfied; these reversals should be treated as illustrative rather than strong evidence.")
    safe_lines.append("These observations do not establish isolated hardware causality, a universal fastest API, statistical proof from six cases, or an absolute GTX versus T4 speed comparison.")
    safe = "\n\n".join(safe_lines)
    stack_values = env[["pytorch_version", "cuda_version", "platform"]].astype(str).drop_duplicates()
    stack_warning = "The environments differ in GPU, OS/runtime, PyTorch, CUDA and hosted execution conditions; observed path and timing differences are stack-level observations, not isolated GPU-causal evidence." if len(stack_values) > 1 else ""
    text = ["# Phase 3 Comparison: Profiling Evidence for API-Dependent CUDA Paths", "## 1. Purpose",
            "Phase 2 supplies the main matched, margin-aware winner-count evidence. Phase 3 provides representative profiler attribution and within-subset event timing, not a replacement for randomized/batched confirmation. This is offline analysis only.",
            "## 2. Inputs and environments", markdown(env), source_refs,
            "## 3. Case-level timing comparison", markdown(cases[["case_id", "rows", "indices", "dim", "distribution", "local_winner", "local_margin_ratio", "t4_winner", "t4_margin_ratio", "winner_relation"]]),
            "The full [case table](phase3_case_comparison.csv) includes workload draw, layout/order/cache/Zipf settings, runner-ups, completeness, and all event medians. A decisive subset selection means runner-up/fastest >1.05. Near-ties retain their raw minimum label but are weak evidence. Failed/inconsistent operation sets are incomplete.",
            "## 4. Winner reversals and controls", "\n".join(sections),
            "These cases are representative probes, not enough observations to statistically establish a grid-wide selection frequency. See [transition counts](phase3_winner_transition_matrix.csv).",
            "## 5. GPU-busy proxy and host-bound warnings", markdown(busy),
            markdown(env[["environment_label", "likely_host_bound_count"]]),
            markdown(warnings[["environment_label", "case_id", "operation", "gpu_busy_fraction"]]),
            "gpu_busy_fraction divides profiled CUDA kernel time per invocation by an unprofiled event median. It is a proxy, not utilization, can exceed 1, and can be distorted by instrumentation or changed timing conditions. Values below 0.6 flag possible host overhead only. Unflagged cases do not prove hardware saturation; flagged cases warrant caution during path attribution.",
            "## 6. CUDA path attribution", markdown(patterns),
            "### Mean kernel category share by environment and operation", markdown(category_means),
            ("WARNING: unknown kernels are substantial (mean `other` share >= 0.2):\n\n" + markdown(other_warning) if not other_warning.empty else "The mean `other` category share is below 0.2 for every available environment and operation."),
            "[Kernel summary](phase3_kernel_summary.csv) preserves exact top keys and profile-total times. [Category shares](phase3_kernel_category_summary.csv) apply ordered, first-match name rules to CUDA device rows only, excluding user annotations and phase3 ranges. indexFuncLargeIndex is normalized to indexFuncLargeIndex/ReduceAdd; scatter-gather/scatter_add paths to scatter-gather ReduceAdd. Embedding keys distinguish compute_grad_weight, sum_and_scatter, radix sort, unique and scan components. Names describe observed paths; they do not alone prove why an API is faster.",
            "Full-event category shares sum to 1 per environment/case/operation. Fallback top-10 data use the separate top10_category_share column; full-profile share remains blank. kernel_count sums recorded calls, not just distinct names. Source directories above provide raw event evidence.",
            "## 7. Connection to Phase 2", markdown(tables["phase3_phase2_alignment"]), markdown(tables["phase3_phase2_case_join"]),
            "Case-level Phase 2 alignment is claimed only for rows with `in_phase2_grid=true`.",
            "Phase 2 characterizes matched grid-level regimes with practical margins; Phase 3 examines selected CUDA paths. Together they provide evidence that standard APIs for equivalent sparse-index accumulation can map to different execution paths whose relative costs depend on workload and the combined hardware/software stack. No isolated hardware cause is inferred.",
            ("## Environment-stack warning\n\n" + stack_warning if stack_warning else ""),
            "## 8. Paper-ready claims", markdown(tables["phase3_claim_checks"]), safe,
            "## 9. Limitations", LIMITATIONS, "### Input and output audit", "\n".join(f"- {note}" for note in notes) or "No audit warnings or missing event files.",
            "## 10. Generated artifacts", "\n".join(f"- [{name}.csv]({name}.csv)" for name in tables),
            "\n".join(f"- [{name}]({name})" for name in plots) or "Plot generation unavailable; CSV and Markdown outputs are retained.",
            "- [Limitations](phase3_limitations.md)"]
    (args.out_dir / "phase3_comparison_report.md").write_text("\n\n".join(text)+"\n", encoding="utf-8")
    (args.out_dir / "phase3_limitations.md").write_text("# Phase 3 Comparison Limitations\n\n"+LIMITATIONS+"\n"+"\n".join(f"- {note}" for note in notes), encoding="utf-8")


def main():
    args = parse_args()
    notes, runs = [], []
    try:
        for label, path in [("GTX1650Ti", args.local_dir), ("T4", args.t4_dir)]:
            runs.append({"label": label, "path": path, "metadata": read_metadata(path),
                         "cases": read_profile_cases(path, label), "winners": read_case_winners(path, label)})
        phase2_counts = read_csv(args.phase2_dir / "matched_margin_winner_counts_by_run.csv", ["label", "operation", "win_count", "total_draws"])
        phase2_speedups = read_csv(args.phase2_dir / "matched_all_op_speedup_summary.csv", ["label", "operation", "geo_mean_speedup"])
        phase2_cases = read_csv(args.phase2_dir / "matched_winners_common_ops.csv", ["label", "rows", "indices", "dim", "distribution",
                          "zipf_mode", "row_layout", "index_order", "cache_mode", "workload_draw"])
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    local, t4 = runs
    common, ids_match, pairing = validate_cases(local["cases"], t4["cases"], notes)
    combined = pd.concat([run["cases"] for run in runs], ignore_index=True)
    environments, events = [], []
    for run in runs:
        data, meta, label = run["cases"], run["metadata"], run["label"]
        env = {"environment_label": label} | {key: meta.get(key) for key in ["gpu_name", "compute_capability", "pytorch_version", "cuda_version", "python_version", "platform"]}
        env["compute_capability"] = json.dumps(env["compute_capability"])
        env.update(source_dir=str(run["path"].resolve()), status_summary=json.dumps(data.status.value_counts().to_dict()), case_ops=len(data),
                   failures=int(data.status.ne("ok").sum()), correctness_failures=int(data.correctness.eq("failed").sum()),
                   cuda_kernel_events_available_count=int(data.profile_sort_basis.eq("cuda_kernel_events").sum()),
                   likely_host_bound_count=int(data.likely_host_bound.map(truth).sum()))
        environments.append(env)
        if env["failures"] or env["correctness_failures"]:
            notes.append(f"{label}: {env['failures']} status failures, {env['correctness_failures']} correctness failures; affected cases cannot support paired selection claims.")
        events.append(load_top_or_event_files(run["path"], label, data, notes))
    case_table = build_case_comparison(local["winners"], t4["winners"], local["cases"], t4["cases"], common, pairing, notes)
    op_table = build_operation_comparison(local["cases"], t4["cases"], common)
    valid = case_table[case_table.winner_relation != "incomplete"].copy()
    valid["decisive_both"] = valid.local_decisive_5pct.map(truth) & valid.t4_decisive_5pct.map(truth)
    valid["near_tie_any"] = ~valid.decisive_both
    transitions = valid.groupby(["local_winner", "t4_winner"]).agg(case_count=("case_id", "size"), decisive_both_count=("decisive_both", "sum"),
                         near_tie_in_any_count=("near_tie_any", "sum"), cases=("case_id", lambda v: ", ".join(v))).reset_index()
    phase2_join = complete_phase2_case_join(build_phase2_case_join(local["cases"], phase2_cases, notes), case_table)
    category_table = build_kernel_category_summary(pd.concat(events, ignore_index=True))
    tables = {"phase3_environment_summary": pd.DataFrame(environments), "phase3_case_comparison": case_table,
              "phase3_operation_comparison": op_table, "phase3_kernel_summary": build_kernel_summary(combined),
              "phase3_kernel_category_summary": category_table,
              "phase3_winner_transition_matrix": transitions,
              "phase3_phase2_alignment": build_alignment(phase2_counts, phase2_speedups, case_table),
              "phase3_phase2_case_join": phase2_join,
              "phase3_claim_checks": build_claim_checks(category_table, phase2_counts, case_table)}
    for name, table in tables.items():
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
    plots = []
    try:
        plots = make_plots(case_table, op_table, tables["phase3_kernel_category_summary"], args.out_dir)
    except Exception as exc:
        notes.append(f"Plot generation unavailable or incomplete: {type(exc).__name__}: {exc}. CSV/Markdown analysis retained.")
    write_markdown_report(tables, combined, args, notes, plots)
    print("Loaded Phase 3 environments: GTX1650Ti, T4")
    print(f"Compared {len(case_table)} common cases and {len(op_table)} paired operation rows")
    print(f"Correctness failures: {int(combined.correctness.eq('failed').sum())}")
    print("Likely host-bound flags: " + ", ".join(f"{row['environment_label']}={row['likely_host_bound_count']}" for row in environments))
    print(f"Index hash mismatches: {sum(not value['index_hash_match'] for value in pairing.values())}")
    print(f"Source hash mismatches: {sum('src_sha256 mismatch' in note for note in notes)}")
    print(f"Phase2 joined cases: {int(phase2_join.in_phase2_grid.sum())}/{len(case_table)}")
    print("Decisive reversals: " + ", ".join(case_table.loc[case_table.winner_relation.isin(["reversal_decisive", "reversal_decisive_with_host_warning"]), "case_id"]))
    strong = tables["phase3_claim_checks"].set_index("check_name").loc["strong_clean_reversals_present", "passed"]
    print("Clean strong reversals: " + (", ".join(case_table.loc[case_table.paired_gpu_clean & case_table.paired_strong_timing_evidence & case_table.workload_pairing_quality.eq("exact_index_match") & case_table.winner_relation.isin(["reversal_decisive", "reversal_decisive_with_host_warning"]), "case_id"]) if strong else "none"))
    passed_checks = int(tables["phase3_claim_checks"].passed.sum())
    print(f"Claim checks passed: {passed_checks}/{len(tables['phase3_claim_checks'])}")
    for note in notes:
        print("NOTE:", note)
    print(f"Generated: {(args.out_dir / 'phase3_comparison_report.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
