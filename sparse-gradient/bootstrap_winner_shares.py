"""Analysis-only descriptive bootstrap of saved 5% winner shares (stdlib only).

Run: python sparse-gradient/bootstrap_winner_shares.py
No benchmark execution, raw timing reaggregation, or manuscript writes.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import random
import sys

import threshold_sensitivity as saved

BASE = Path(__file__).resolve().parent
OUT = BASE / "results/bootstrap_winner_shares"
BOOTSTRAP_SEED = 20261010
B = 10000
FIELDS = ("phase", "environment_group", "scope", "bootstrap_unit", "quantity",
          "estimate", "ci_lower", "ci_upper", "n_observations",
          "bootstrap_replicates", "bootstrap_seed", "method")
METHOD = "percentile_95_linear_interpolation"


def winner(times):
    fastest, margin = saved.rank(times)
    return fastest if margin > 0.05 else "tie"


def reconstruct():
    # These functions only read and validate saved inputs. Never call saved.main(),
    # which also generates manuscript tables for the separate sensitivity task.
    p2 = saved.phase2()
    p4, included, excluded = saved.phase4()
    saved.verify_baseline(p2 + p4)
    groups = []
    manifest = saved.read_json(BASE / "phase2_analysis_v2/analysis_manifest.json")
    matched = saved.read_csv(BASE / "phase2_analysis_v2/matched_winners_common_ops.csv")
    labels = {"GTX1650Ti": "local_gtx1650ti", "T4": "t4", "T4_repeat": "t4_repeat"}
    for spec in manifest["runs"]:
        label = spec["label"]
        keys = {tuple(r[k] for k in saved.DRAW_KEYS) for r in matched if r["label"] == label}
        summary = saved.read_csv(BASE / PureWindowsPath(spec["path"]).name / "summary_results.csv")
        values = defaultdict(dict)
        for row in summary:
            key = tuple(row[k] for k in saved.DRAW_KEYS)
            op = row["operation"]
            if key in keys and op in saved.OPS:
                saved.require(row["status"] == "ok" and row["correctness"] == "passed" and
                              row["scope_group"] == "fair_dense_grad", "Invalid matched row")
                saved.require(op not in values[key], "Duplicate matched row")
                values[key][op] = float(row["median_ms"])
        saved.require(set(values) == keys and len(keys) == 162, "Missing Phase 2 observations")
        observations = [(key, winner(values[key])) for key in sorted(keys)]
        quantities = ("embedding", "tie") if label == "GTX1650Ti" else ("scatter_add", "tie")
        groups.append(dict(phase="phase2", environment_group=labels[label], scope="",
                           observations=observations, quantities=quantities))
    records = defaultdict(list)
    for directory in included:
        path = BASE / Path(directory.replace("\\", "/"))
        metadata = saved.read_json(path / "phase4_metadata.json")
        env = "colab_t4" if "t4_phase4_training_proxy" in path.parts else "local_gtx1650ti"
        values = defaultdict(dict)
        for row in saved.read_csv(path / "phase4_summary.csv"):
            key = saved.config4(row)
            op = saved.PROXY_OPS[row["variant"]]
            saved.require(op not in values[key], "Duplicate proxy row")
            values[key][op] = row
        for scope in ("proxy_step", "accumulation_region"):
            for config in sorted(values):
                times = {op: float(r[f"{scope}_median_ms"]) for op, r in values[config].items()}
                records[(env, scope)].append(((config, metadata["seed"]), winner(times)))
    for env in ("local_gtx1650ti", "colab_t4"):
        for scope in ("proxy_step", "accumulation_region"):
            observations = sorted(records[(env, scope)])
            saved.require(len(observations) == 120 and len({k for k, _ in observations}) == 120,
                          "Missing/duplicate Phase 4 observations")
            groups.append(dict(phase="phase4", environment_group=env, scope=scope,
                               observations=observations,
                               quantities=("index_add", "tie") if env == "local_gtx1650ti" else ("tie", "scatter_add")))
    # Verify reconstructed labels themselves, not just the helper's aggregates.
    checks = []
    for group in groups:
        c = Counter(label for _, label in group["observations"])
        check = {k: group[k] for k in ("phase", "environment_group", "scope")}
        check.update(threshold=.05, num_configurations=len(group["observations"]))
        check.update(zip(saved.COUNT_FIELDS, [c[op] for op in (*saved.OPS, "tie")]))
        checks.append(check)
    saved.verify_baseline(checks)
    return groups, checks, included, excluded


def percentile(values, probability):
    """Linear interpolation at h=(n-1)*p, between floor(h) and ceil(h)."""
    ordered = sorted(values)
    h = (len(ordered) - 1) * probability
    low, high = math.floor(h), math.ceil(h)
    return ordered[low] + (h - low) * (ordered[high] - ordered[low])


def bootstrap(group, cluster, rng):
    observations, quantities = group["observations"], group["quantities"]
    labels_by_unit = []
    if cluster:
        configs = defaultdict(list)
        for (config, seed), label in observations:
            configs[config].append((seed, label))
        saved.require(set(configs) == saved.GRID4, "Cluster grid mismatch")
        for config in sorted(configs):
            seed_labels = configs[config]
            saved.require(len(seed_labels) == 5 and {s for s, _ in seed_labels} == set(range(4001, 4006)),
                          "A cluster must contain exactly all five seeds")
            labels_by_unit.append([label for _, label in sorted(seed_labels)])
        unit = "configuration_cluster_all_seeds"
    else:
        labels_by_unit = [[label] for _, label in observations]
        unit = "matched_configuration_draw" if group["phase"] == "phase2" else "seed_configuration_observation"
    sizes = [len(labels) for labels in labels_by_unit]
    contributions = [tuple(labels.count(q) for q in quantities) for labels in labels_by_unit]
    samples = [[] for _ in quantities]
    n_units = len(labels_by_unit)
    for _ in range(B):
        totals = [0] * len(quantities)
        denominator = 0
        # Both quantities use the same resampled observations within a replicate.
        for _ in range(n_units):
            index = rng.randrange(n_units)
            denominator += sizes[index]
            for j, value in enumerate(contributions[index]):
                totals[j] += value
        for j, total in enumerate(totals):
            samples[j].append(total / denominator)
    counts = Counter(label for _, label in observations)
    rows = []
    for q, distribution in zip(quantities, samples):
        row = {k: group[k] for k in ("phase", "environment_group", "scope")}
        row.update(bootstrap_unit=unit, quantity=f"{q}_share",
                   estimate=counts[q] / len(observations),
                   ci_lower=percentile(distribution, .025), ci_upper=percentile(distribution, .975),
                   n_observations=len(observations), bootstrap_replicates=B,
                   bootstrap_seed=BOOTSTRAP_SEED, method=METHOD)
        rows.append(row)
    design = dict(phase=group["phase"], environment_group=group["environment_group"],
                  scope=group["scope"], bootstrap_unit=unit, n_resampling_units=n_units,
                  observations_per_unit=sorted(set(sizes)), n_observations=len(observations),
                  resampled_units_per_replicate=n_units, quantities=quantities)
    return rows, design


def report(rows):
    lines = ["# Descriptive bootstrap intervals for aggregate winner shares", "",
             "Phase 2 baseline checks: **passed** (162 matched configuration-draw observations per run).",
             "Phase 4 baseline checks: **passed** (120 seed-configuration observations per environment/scope; seeds 4001–4005).",
             "", f"Bootstrap replicates: **{B:,}** per group and method; bootstrap seed: **{BOOTSTRAP_SEED}**.",
             "The original strict >5% winner classification is held fixed. No benchmarks, raw timings, manuscript files, or existing result artifacts were changed.", "",
             "## Resampling design", "",
             "- Phase 2: resample 162 matched configuration-draw observations, with replacement, separately per run.",
             "- Phase 4 ordinary: resample 120 seed-configuration observations, with replacement, separately per environment/scope.",
             "- Phase 4 cluster: resample 24 configurations, with replacement; each selected cluster contributes all five saved seed observations. A repeatedly selected cluster contributes its observations repeatedly. The denominator is the number of included observations, 120 here.",
             "- Selected shares use all observations in the denominator, including ties; they are not shares conditional on resolved cases.",
             "- Each pair of quantities uses the same resampling indices within a replicate. A single Python random.Random stream is seeded once and consumed in the group/method order recorded in the audit.",
             "- Endpoints are the 2.5th and 97.5th percentiles, using linear interpolation at (B−1)p in the sorted replicate shares. No rounding occurs before CSV export.", "",
             "## Estimates and 95% descriptive bootstrap intervals", "",
             "All entries below are percentages; CSV values are fractions. Cluster n_observations remains 120, while the number of resampling units is 24 (recorded in the audit).", "",
             "| Phase | Environment / scope | Resampling unit | Quantity | Estimate | Interval |",
             "|---|---|---|---|---:|---:|"]
    units = {"matched_configuration_draw": "Configuration-draw", "seed_configuration_observation": "Observation",
             "configuration_cluster_all_seeds": "Configuration cluster"}
    for r in rows:
        lines.append(f"| {r['phase']} | {r['environment_group']} / {r['scope'] or 'matched grid'} | {units[r['bootstrap_unit']]} | {r['quantity']} | {100*r['estimate']:.1f}% | [{100*r['ci_lower']:.1f}, {100*r['ci_upper']:.1f}]% |")
    lines += ["", "## Interpretation and limitations", "",
              "The intervals summarize resampling variability of aggregate winner shares under the saved observation set. They should not be interpreted as inference to all possible workloads, GPUs, or PyTorch versions. These are descriptive uncertainty summaries and a resampling-based stability check, not hypothesis tests.", "",
              "The workload grid is controlled, not an independent random sample from a population. The ordinary observation bootstrap does not preserve dependence among draws/seeds sharing a configuration. The Phase 4 cluster bootstrap preserves all five seed labels within each configuration and measures sensitivity to configuration composition; it does not model shared effects across configurations from the same seed/run. Only 24 configurations and five seeds are available. Phase 2 uses the requested configuration-draw units, not configuration clusters.", "",
              "Saved medians and winner labels are treated as fixed: these intervals do not propagate within-run timing noise, threshold uncertainty, hardware variability, or software-version variability. No API difference intervals, cross-environment paired comparison, or population-sampling claim is made. The T4 repeat is a separate run on the same stack, not a third independent environment. Finite bootstrap simulation and the small discrete observation sets also limit endpoint precision.", "",
              "No missing required runs or incomplete Phase 4 clusters were detected. Both 5% baselines match exactly. Any skipped runs outside the study profile/seed range are listed in the audit.", "",
              "Reproduce with `python sparse-gradient/bootstrap_winner_shares.py` from the project root. Only the standard library and the read-only reconstruction helpers in `threshold_sensitivity.py` are used; its output-writing entry point is never called."]
    return "\n".join(lines) + "\n"


def main():
    groups, checks, included, excluded = reconstruct()
    rng = random.Random(BOOTSTRAP_SEED)
    rows, designs = [], []
    for group in groups:
        for cluster in ([False, True] if group["phase"] == "phase4" else [False]):
            output, design = bootstrap(group, cluster, rng)
            rows.extend(output)
            designs.append(design)
    saved.require(len(rows) == 22, "Unexpected output coverage")
    saved.require(all(0 <= r["ci_lower"] <= r["ci_upper"] <= 1 for r in rows), "Invalid interval")
    audit = dict(phase2_baseline_checks="passed", phase4_baseline_checks="passed",
                 baseline_counts=checks, bootstrap_seed=BOOTSTRAP_SEED, bootstrap_replicates=B,
                 method=METHOD, percentile_method="Sort B replicate shares; h=(B-1)*p; linearly interpolate between floor(h) and ceil(h), p=0.025 and 0.975.",
                 rng="Python random.Random(20261010), seeded once; randrange(n_units) for each draw, with replacement; quantities share draws; group order below.",
                 python_version=sys.version, group_method_order=designs,
                 phase4_cluster_bootstrap=dict(computed=True, configuration_keys=["rows", "num_indices", "dim", "distribution"],
                                               clusters_per_group=24, seeds_per_cluster=list(range(4001, 4006))),
                 phase4_included_runs=included, phase4_excluded_runs=excluded,
                 winner_rule="runner_up_median / fastest_median - 1 > 0.05; otherwise tie",
                 input_sha256=saved.SOURCES,
                 source_code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in [Path(__file__), BASE / "threshold_sensitivity.py"]},
                 interpretation="Descriptive resampling variability conditional on saved observations; not hypothesis tests or inference to a workload/GPU/software population.")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "bootstrap_winner_shares.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "bootstrap_winner_shares_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (OUT / "bootstrap_winner_shares_report.md").write_text(report(rows), encoding="utf-8")
    print("Both baselines passed; 22 interval rows, including Phase 4 configuration clusters.")
    for r in rows:
        print(r["phase"], r["environment_group"], r["scope"], r["bootstrap_unit"], r["quantity"],
              f"{100*r['estimate']:.1f}% [{100*r['ci_lower']:.1f}, {100*r['ci_upper']:.1f}]")


if __name__ == "__main__":
    main()
