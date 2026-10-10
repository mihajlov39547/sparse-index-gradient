"""Exploratory descriptive rule-of-thumb summaries of saved grids (stdlib only).

Run: python sparse-gradient/decision_rule_summary.py
No experiments, benchmark imports, model fitting, or manuscript output.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import csv
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath

import bootstrap_winner_shares as reconstruction
import threshold_sensitivity as saved

BASE = Path(__file__).resolve().parent
OUT = BASE / "results/decision_rule_summary"
CAUTION = "Descriptive rule from saved grid only; validate on target workload and stack."
METRICS = ("unique_rows", "duplicate_ratio", "max_row_count", "hot_row_share")
FIELDS = ("phase", "environment_group", "scope", "grouping", "group_value", "n_observations",
          "index_add_wins", "scatter_add_wins", "embedding_wins", "ties", "dominant_category",
          "dominant_count", "dominant_share", "tie_share", "resolved_share", "n_distinct_configurations")
CANDIDATE_FIELDS = ("phase", "environment_group", "scope", "grouping", "group_value",
                    "n_observations", "dominant_category", "dominant_share", "tie_share",
                    "rule_text", "caution", "n_distinct_configurations")


def features(rows, phase):
    """Read workload metrics once per observation; check agreement across variants."""
    def common(name):
        values = [r.get(name, "") for r in rows]
        saved.require(all(v != "" for v in values), f"Missing workload feature {name}")
        nums = [float(v) for v in values]
        saved.require(all(math.isfinite(v) and math.isclose(v, nums[0], rel_tol=1e-12, abs_tol=1e-15)
                          for v in nums), f"Variant workload feature mismatch: {name}")
        return nums[0]

    nrows, dim = int(common("rows")), int(common("dim"))
    count_name = "indices" if phase == "phase2" else "num_indices"
    count = int(common(count_name))
    distributions = {r["distribution"] for r in rows}
    saved.require(len(distributions) == 1, "Distribution mismatch")
    pressure = Fraction(count, nrows)
    # Exact reduced fractions prevent floating-point differences in group identity.
    result = dict(rows=nrows, dim=dim, distribution=next(iter(distributions)),
                  **{count_name: count}, indices_to_rows_ratio=str(pressure),
                  rows_to_indices_ratio=str(1 / pressure),
                  ratio_bucket="low_reference_pressure" if pressure < 1 else
                  "balanced_reference_pressure" if pressure < 4 else "high_reference_pressure")
    unique = int(common("unique_indices" if phase == "phase2" else "unique_rows"))
    maximum = int(common("max_row_count"))
    duplicate = 1 - unique / count
    hot_share = maximum / count
    saved.require(0 < unique <= min(nrows, count) and 1 <= maximum <= count, "Invalid saved index metrics")
    if phase == "phase2":
        saved.require(math.isclose(common("duplicate_rate"), duplicate, abs_tol=1e-12), "Duplicate ratio mismatch")
        saved.require(math.isclose(common("top1_share"), hot_share, abs_tol=1e-12), "Top-row share mismatch")
    result.update(unique_rows=unique, duplicate_ratio=duplicate,
                  max_row_count=maximum, hot_row_share=hot_share)
    return result


def load_features(included):
    data = {}
    manifest = saved.read_json(BASE / "phase2_analysis_v2/analysis_manifest.json")
    labels = {"GTX1650Ti": "local_gtx1650ti", "T4": "t4", "T4_repeat": "t4_repeat"}
    for spec in manifest["runs"]:
        groups = defaultdict(list)
        for row in saved.read_csv(BASE / PureWindowsPath(spec["path"]).name / "summary_results.csv"):
            if row["operation"] in saved.OPS:
                groups[tuple(row[k] for k in saved.DRAW_KEYS)].append(row)
        for key, rows in groups.items():
            data[("phase2", labels[spec["label"]], key)] = rows
    for directory in included:
        path = BASE / directory.replace("\\", "/")
        m = saved.read_json(path / "phase4_metadata.json")
        env = "colab_t4" if "t4_phase4_training_proxy" in path.parts else "local_gtx1650ti"
        groups = defaultdict(list)
        for row in saved.read_csv(path / "phase4_summary.csv"):
            groups[saved.config4(row)].append(row)
        for config, rows in groups.items():
            data[("phase4", env, (config, m["seed"]))] = rows
    return data


def summarize(group, source_rows):
    phase, env = group["phase"], group["environment_group"]
    count_name = "indices" if phase == "phase2" else "num_indices"
    singles = ["dim", "distribution", "rows", count_name, "indices_to_rows_ratio", "ratio_bucket"]
    if phase == "phase2":
        singles.append("rows_to_indices_ratio")
    singles.extend(METRICS)
    groupings = [(f,) for f in singles] + [("distribution", "dim"),
                 ("distribution", "ratio_bucket"), ("dim", "ratio_bucket")]
    buckets = defaultdict(list)
    for key, label in group["observations"]:
        rows = source_rows[(phase, env, key)]
        saved.require(len(rows) == 3, "Expected three rows per observation")
        feature = features(rows, phase)
        # Phase 2 key ends with workload_draw; Phase 4 key is (config, seed).
        config = key[:-1] if phase == "phase2" else key[0]
        for grouping in groupings:
            value = json.dumps({f: feature[f] for f in grouping}, sort_keys=True, separators=(",", ":"))
            buckets[(" + ".join(grouping), value)].append((config, label))
    output = []
    for (grouping, value), observations in sorted(buckets.items()):
        counts = Counter(label for _, label in observations)
        n = len(observations)
        maximum = max(counts.values())
        leaders = [c for c in (*saved.OPS, "tie") if counts[c] == maximum]
        row = {k: group[k] for k in ("phase", "environment_group", "scope")}
        row.update(grouping=grouping, group_value=value, n_observations=n,
                   **dict(zip(saved.COUNT_FIELDS, [counts[c] for c in (*saved.OPS, "tie")])),
                   dominant_category=";".join(leaders), dominant_count=maximum,
                   dominant_share=maximum/n, tie_share=counts["tie"]/n,
                   resolved_share=1-counts["tie"]/n,
                   n_distinct_configurations=len({c for c, _ in observations}))
        output.append(row)
    # Each grouping partitions exactly the original saved observation set.
    for grouping in groupings:
        selected = [r for r in output if r["grouping"] == " + ".join(grouping)]
        saved.require(sum(r["n_observations"] for r in selected) == len(group["observations"]), "Partition mismatch")
        for op, field in zip((*saved.OPS, "tie"), saved.COUNT_FIELDS):
            saved.require(sum(r[field] for r in selected) == sum(label == op for _, label in group["observations"]), "Partition winner mismatch")
    return output, singles


def order(row):
    return (row["phase"], row["environment_group"], row["scope"], -row["dominant_share"],
            -row["n_observations"], row["grouping"], row["group_value"])


def candidates(rows):
    result = []
    for row in sorted(rows, key=order):
        if row["n_observations"] < 12 or row["dominant_share"] < .70:
            continue
        item = {k: row[k] for k in CANDIDATE_FIELDS if k not in ("rule_text", "caution")}
        conditions = ", ".join(f"{k}={v}" for k, v in json.loads(row["group_value"]).items())
        category = "ties" if row["dominant_category"] == "tie" else row["dominant_category"]
        scope = f" / {row['scope']}" if row["scope"] else ""
        item["rule_text"] = (f"In the saved {row['phase']} {row['environment_group']}{scope} grid, "
                             f"{conditions} is dominated by {category} "
                             f"({row['dominant_count']}/{row['n_observations']} observations).")
        item["caution"] = CAUTION
        result.append(item)
    return result


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(rules, available):
    lines = ["# Descriptive rule-of-thumb summary", "",
             "Phase 2 baseline verification: **passed** for all three runs (162 matched configuration-draw observations each).",
             "Phase 4 baseline verification: **passed** for both environments and scopes (120 seed-configuration observations each).",
             "All winner labels were reconstructed from saved per-variant medians using runner-up / fastest - 1 > 0.05; otherwise tie.", "",
             "## Available features", ""]
    for phase, features_ in available.items():
        lines.append(f"- {phase}: " + ", ".join(f"`{f}`" for f in features_) + ".")
    lines += ["", "Both phases have duplicate/skew information in saved summary rows. Phase 2 unique_indices counts distinct destination rows and is exposed as unique_rows; duplicate_rate is checked against 1 - unique_indices / indices. Phase 4 saves unique_rows and max_row_count, allowing exact duplicate_ratio = 1 - unique_rows / num_indices from summary statistics. No raw index reconstruction is used.", "",
              "hot_row_share means the single most referenced row's fraction (max_row_count / number of references), a top-one concentration descriptor, not the generator's configured hotspot fraction. Phase 2 top1_share is checked against it. Phase 2 also saves top16_share and mean_count_per_used_row; these are not additional groupings in this pass. Distribution labels remain coarse descriptors and do not isolate skew effects.", "",
              "Ratios are encoded as exact reduced fractions in group_value JSON. ratio_bucket uses indices/rows: low_reference_pressure < 1; balanced_reference_pressure from 1 inclusive to 4 exclusive; high_reference_pressure >= 4. Phase 2's matched grid covers balanced_reference_pressure and high_reference_pressure, with no low-pressure observations. Phase 4 covers all three buckets. The combinations are distribution + dim, distribution + ratio_bucket, and dim + ratio_bucket. Buckets were fixed before evaluating group winners.", "",
              "## Candidate rules", "",
              f"{len(rules)} groups meet n_observations >= 12 and dominant_share >= 0.70. These are descriptive display filters, not inferential criteria. The dominant category includes tie. If categories are equally frequent, all leaders are retained with semicolons; these groups cannot reach 70% dominance. All candidates, including overlapping and redundant groups, are saved in decision_rule_candidates.csv.", "",
              "Candidates are sorted by phase, environment/scope, dominant share descending, then observation count descending; lexical feature ordering breaks remaining ties. Below are the first ten per environment/scope in that order. A high count of candidate rows is not independent corroboration.", ""]
    sections = defaultdict(list)
    for row in rules:
        sections[(row["phase"], row["environment_group"], row["scope"])].append(row)
    for (phase, env, scope), values in sections.items():
        lines += [f"### {phase} / {env} / {scope or 'matched grid'}", "",
                  "| Condition | Category | Count | Share | Ties | Distinct configurations |",
                  "|---|---|---:|---:|---:|---:|"]
        for r in values[:10]:
            count = round(r["dominant_share"] * r["n_observations"])
            lines.append(f"| `{r['group_value']}` | {r['dominant_category']} | {count}/{r['n_observations']} | {r['dominant_share']:.1%} | {r['tie_share']:.1%} | {r['n_distinct_configurations']} |")
    lines += ["", "## Interpretation and limitations", "",
              "The grouped summaries provide practical rule-of-thumb patterns, not a predictive model and not a deployment recommendation. They indicate which workload descriptors were associated with strong dominance or tie-heavy behavior within the saved grid. The patterns should be treated as prompts for targeted benchmarking rather than direct deployment rules; each requires validation on the target workload and stack.", "",
              "Candidate selection is exploratory and uses the same observations being summarized, with no held-out validation. Groups overlap, ratios and inverse ratios are redundant, and dimensions/distributions/row counts can be confounded. Exact duplicate/skew values can fragment the grid into small groups; these rows remain in grouped CSVs but cannot become candidates without the stated support threshold. The support counts include repeated draws/seeds rather than independent workload shapes; n_distinct_configurations makes that distinction explicit. No causal attribution to a feature follows.", "",
              "The T4 repeat is a second run on the T4 stack, not a third independent environment. Phase 4 is a mini proxy, not production training, and accumulation_region is a separate diagnostic replay, not a decomposition of proxy_step. Raw timing variability and uncertainty in saved medians are not propagated in this pass. No new timing runs, bootstrap resampling, threshold changes, or statistical tests are performed.", "",
              "Simple, well-supported feature groups that recur across workload draws or the T4 repeat may merit a qualified Discussion example later. Their precise support, environment, and scope should accompany any statement; inverse-ratio or single-bucket restatements should not be presented as separate findings. No manuscript changes are made here.", "",
              "Reproduce with `python sparse-gradient/decision_rule_summary.py`. The script calls only the read-only reconstruction helpers from bootstrap_winner_shares.py and threshold_sensitivity.py, not their analysis/output entry points. Input and code hashes, feature definitions, baseline counts, and coverage are recorded in the audit."]
    return "\n".join(lines) + "\n"


def main():
    groups, checks, included, excluded = reconstruction.reconstruct()
    source_rows = load_features(included)
    all_rows, available = [], {}
    for group in groups:
        output, singles = summarize(group, source_rows)
        all_rows.extend(output)
        available[group["phase"]] = singles
    rules = candidates(all_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for phase in ("phase2", "phase4"):
        write_csv(OUT / f"{phase}_grouped_rules.csv", [r for r in all_rows if r["phase"] == phase], FIELDS)
    write_csv(OUT / "decision_rule_candidates.csv", rules, CANDIDATE_FIELDS)
    (OUT / "decision_rule_summary_report.md").write_text(report(rules, available), encoding="utf-8")
    audit = dict(baseline_checks="passed", baseline_counts=checks, phase4_included=included,
                 phase4_excluded=excluded, winner_threshold=.05, available_features=available,
                 candidate_filter=dict(min_observations=12, min_dominant_share=.70),
                 ratio_bucket_definition="indices/rows <1: low; 1<=ratio<4: balanced; >=4: high",
                 grouping_combinations=["distribution + dim", "distribution + ratio_bucket", "dim + ratio_bucket"],
                 metric_definitions=dict(unique_rows="Phase 2 unique_indices; Phase 4 unique_rows",
                                         duplicate_ratio="1 - unique_rows / reference_count; checked against Phase 2 duplicate_rate",
                                         max_row_count="Saved maximum reference count per row",
                                         hot_row_share="max_row_count / reference_count; checked against Phase 2 top1_share; not configured hotspot mass"),
                 phase2_pairing_keys=saved.DRAW_KEYS, phase4_configuration_keys=["rows", "num_indices", "dim", "distribution"],
                 phase2_units="matched configuration-draw", phase4_units="seed-configuration observation",
                 category_tie_handling="semicolon-separated leaders; no arbitrary category selection",
                 grouped_rows=dict(Counter(r["phase"] for r in all_rows)), candidate_rows=len(rules),
                 input_sha256=saved.SOURCES,
                 code_sha256={name: hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in
                              ["decision_rule_summary.py", "bootstrap_winner_shares.py", "threshold_sensitivity.py"]},
                 interpretation="Exploratory descriptive rule-of-thumb summary from saved grid; not a predictive model or deployment recommendation; validate on target workload and stack.")
    (OUT / "decision_rule_summary_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Both baselines passed. {len(all_rows)} grouped rows; {len(rules)} candidate rules.")
    for row in rules[:10]:
        print(row["rule_text"])


if __name__ == "__main__":
    main()
