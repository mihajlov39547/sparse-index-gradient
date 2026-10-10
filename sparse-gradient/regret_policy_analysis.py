"""Descriptive policy simulation under saved median timings; stdlib only.

Run: python sparse-gradient/regret_policy_analysis.py
No experiments, benchmark imports, result regeneration, or manuscript writes.
"""
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import fmean, median
import sys

import bootstrap_winner_shares as reconstruction
import decision_rule_summary as descriptors
import threshold_sensitivity as saved

BASE = Path(__file__).resolve().parent
OUT = BASE / "results/regret_policy_analysis"
REGRET_FIELDS = ("mean_regret", "median_regret", "p90_regret", "p95_regret", "max_regret")
SUMMARY_FIELDS = ("phase", "environment_group", "scope", "policy", "selected_variant_rule", "n_observations",
                  *REGRET_FIELDS, "fraction_with_regret_le_0_05", "fraction_with_regret_le_0_10",
                  "oracle_mean_regret", "notes", *(f"{f}_percent" for f in REGRET_FIELDS),
                  "n_tie_observations", "mean_regret_on_ties", "max_regret_on_ties")
CASE_FIELDS = ("phase", "environment_group", "scope", "observation_id", "rows", "indices_or_num_indices",
               "dim", "distribution", "ratio_bucket", "policy", "selected_variant", "best_variant",
               "selected_time_ms", "best_time_ms", "regret", "winner_5pct", "is_5pct_tie", "regret_percent")


def reconstruct():
    groups, checks, included, excluded = reconstruction.reconstruct()
    sources = descriptors.load_features(included)
    for group in groups:
        observations = []
        for key, label in group["observations"]:
            records = sources[(group["phase"], group["environment_group"], key)]
            feature = descriptors.features(records, group["phase"])
            if group["phase"] == "phase2":
                times = {r["operation"]: float(r["median_ms"]) for r in records}
                identity = dict(zip(saved.DRAW_KEYS, key))
            else:
                times = {saved.PROXY_OPS[r["variant"]]: float(r[f"{group['scope']}_median_ms"]) for r in records}
                identity = dict(zip(("rows", "num_indices", "dim", "distribution"), key[0])) | {"seed": key[1]}
            saved.require(len(records) == 3 and reconstruction.winner(times) == label, "Median/winner mismatch")
            best, _ = saved.rank(times)
            observations.append(dict(features=feature, times=times, winner=label, best=best,
                                     observation_id=json.dumps(identity, sort_keys=True, separators=(",", ":"))))
        group["observations"] = observations
        mean_regrets = {op: fmean(o["times"][op]/o["times"][o["best"]]-1 for o in observations) for op in saved.OPS}
        group["best_static"] = min(saved.OPS, key=lambda op: (mean_regrets[op], op))
        winners = Counter(o["winner"] for o in observations)
        group["count_choice"] = min(saved.OPS, key=lambda op: (-winners[op], op))
        group["tie_dominant"] = winners["tie"] >= max(winners[op] for op in saved.OPS)
    return groups, checks, included, excluded


def policies(group, phase2_sources):
    policies_ = []
    for op in saved.OPS:
        policies_.append((f"always_{op}", f"choose {op}", lambda o, op=op: op, "Fixed API; no calibration."))
    policies_.append(("oracle_best_median", "choose minimum saved median per observation", lambda o: o["best"],
                      "Oracle reference only; uses the evaluation observation's timings."))
    op = group["best_static"]
    policies_.append(("best_static_in_group", f"choose {op}", lambda o, op=op: op,
                      "Static API selected by lowest mean regret on the same evaluation group; in-sample calibrated reference, not deployable without calibration."))
    op = group["count_choice"]
    warning = " Tie-dominant group: uses the most frequent resolved API; no implementation-simplicity choice is inferred." if group["tie_dominant"] else ""
    policies_.append(("most_frequent_winner_in_group", f"choose {op}", lambda o, op=op: op,
                      "API selected by highest strict >5% winner count in the same group; ties excluded from API selection." + warning))
    local_rule = lambda o: "index_add" if o["features"]["dim"] == 64 else "embedding"
    if group["phase"] == "phase2":
        if group["environment_group"] == "local_gtx1650ti":
            policies_.append(("local_dim_heuristic", "if dim == 64 choose index_add; else embedding", local_rule,
                              "Exploratory rule from the saved local grid; no held-out validation."))
            for source in ("t4", "t4_repeat"):
                fallback = phase2_sources[source]["best_static"]
                rule = lambda o, fallback=fallback: "scatter_add" if o["features"]["distribution"] in {"hotspot", "zipf"} else fallback
                policies_.append((f"transfer_{source}_distribution_heuristic", f"if hotspot/zipf choose scatter_add; else {fallback} (fixed from {source})", rule,
                                  f"Cross-stack descriptive transfer from {source}; source fallback frozen before applying to local. Rule was motivated by the full saved study, so this is not a held-out generalization estimate."))
        else:
            fallback = group["best_static"]
            rule = lambda o, fallback=fallback: "scatter_add" if o["features"]["distribution"] in {"hotspot", "zipf"} else fallback
            policies_.append(("t4_distribution_heuristic", f"if hotspot/zipf choose scatter_add; else {fallback}", rule,
                              "Uniform fallback is the best static API over the entire same environment, not recalibrated on uniform cases."))
            policies_.append(("transfer_local_dim_heuristic", "if dim == 64 choose index_add; else embedding", local_rule,
                              "Fixed local rule applied to T4; not adjusted using target timings. Descriptive transfer, not a held-out generalization estimate."))
    elif group["environment_group"] == "local_gtx1650ti":
        policies_.append(("local_index_add_heuristic", "choose index_add", lambda o: "index_add",
                          "Requested local proxy rule; identical to always_index_add. Region scope is a separate diagnostic replay."))
    return policies_


def evaluate(group, policy):
    name, description, select, note = policy
    common = {k: group[k] for k in ("phase", "environment_group", "scope")}
    cases = []
    for o in group["observations"]:
        selected = select(o)
        best_time = o["times"][o["best"]]
        selected_time = o["times"][selected]
        regret = selected_time / best_time - 1
        saved.require(regret >= 0, "Negative regret")
        if name == "oracle_best_median":
            saved.require(regret == 0, "Nonzero oracle regret")
        f = o["features"]
        cases.append(common | dict(observation_id=o["observation_id"], rows=f["rows"],
            indices_or_num_indices=f.get("indices", f.get("num_indices")), dim=f["dim"],
            distribution=f["distribution"], ratio_bucket=f["ratio_bucket"], policy=name,
            selected_variant=selected, best_variant=o["best"], selected_time_ms=selected_time,
            best_time_ms=best_time, regret=regret, winner_5pct=o["winner"], is_5pct_tie=o["winner"] == "tie",
            regret_percent=100*regret))
    values = [c["regret"] for c in cases]
    tie_values = [c["regret"] for c in cases if c["is_5pct_tie"]]
    row = common | dict(policy=name, selected_variant_rule=description, n_observations=len(values),
                        mean_regret=fmean(values), median_regret=median(values),
                        p90_regret=reconstruction.percentile(values,.90), p95_regret=reconstruction.percentile(values,.95),
                        max_regret=max(values), fraction_with_regret_le_0_05=sum(v <= .05 for v in values)/len(values),
                        fraction_with_regret_le_0_10=sum(v <= .10 for v in values)/len(values),
                        oracle_mean_regret=0., notes=note,
                        n_tie_observations=len(tie_values), mean_regret_on_ties=fmean(tie_values) if tie_values else None,
                        max_regret_on_ties=max(tie_values) if tie_values else None)
    row.update({f"{f}_percent": 100*row[f] for f in REGRET_FIELDS})
    return row, cases


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_report(groups, summaries, tree):
    lines = ["# Regret under saved median timings", "",
             "Phase 2 and Phase 4 strict >5% baseline checks: **passed exactly**. All three candidate medians are available for every accepted observation. Phase 2 has 162 matched configuration-draw observations per run; Phase 4 has 120 seed-configuration observations per environment/scope.", "",
             "This is a descriptive policy simulation, not a trained deployment router and not a statistical generalization. It requires validation on target workload and stack. No experiments, original result regeneration, manuscript edits, or LaTeX outputs were performed.", "",
             "Regret = selected median / minimum candidate median - 1. CSVs report fractions and explicit percent columns; the tables below show percentages. Every observation has equal weight, regardless of absolute runtime. Near-ties keep continuous regret: the 5% rule classifies winners but does not round regret to zero. Percentiles use linear interpolation at (n−1)p. Exact API score ties are broken lexically. The oracle is only a reference and has zero regret in every case.", "",
             "## Static-policy and heuristic summaries", "",
             "Oracle rows are omitted from this display because all metrics are zero. Calibrated static and winner-count choices use the same evaluation group and are explicitly in-sample descriptive references.", ""]
    for g in groups:
        subset = [r for r in summaries if all(r[k] == g[k] for k in ("phase", "environment_group", "scope"))]
        lines += [f"### {g['phase']} / {g['environment_group']} / {g['scope'] or 'matched grid'}", "",
                  f"Best static by mean regret: **{g['best_static']}**. Most frequent resolved winner: **{g['count_choice']}**. Choices {'agree' if g['best_static']==g['count_choice'] else 'differ'}." + (" **Tie-dominant group:** API selection ignores the tie category; this is not a no-change or simplicity-based policy." if g["tie_dominant"] else ""), "",
                  "| Policy | Mean regret % | Median % | P90 % | P95 % | Max % | Regret ≤5% | Regret ≤10% |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in subset:
            if r["policy"] == "oracle_best_median":
                continue
            lines.append(f"| {r['policy']} | " + " | ".join(f"{100*r[f]:.2f}" for f in REGRET_FIELDS) +
                         f" | {r['fraction_with_regret_le_0_05']:.1%} | {r['fraction_with_regret_le_0_10']:.1%} |")
        base = next(r for r in subset if r["policy"] == "best_static_in_group")
        for r in subset:
            if "heuristic" in r["policy"]:
                lines.append(f"\n- {r['policy']}: {r['selected_variant_rule']}. Compared with the group's calibrated best-static reference: mean regret changes by {100*(r['mean_regret']-base['mean_regret']):+.2f} percentage points; maximum regret changes by {100*(r['max_regret']-base['max_regret']):+.2f} points. {r['notes']}")
    lines += ["", "## Minimax static choices on the saved grid", "",
              "Minimax here means choosing the fixed API whose largest observed regret is smallest. It is retrospective and gives no assurance about unobserved workloads. A 10% worst-case ceiling is shown only as a descriptive practical-cost screen, not a new winner threshold.", "",
              "| Saved observations | index_add max % | scatter_add max % | embedding max % | Lowest observed maximum | All cases within 10%? |",
              "|---|---:|---:|---:|---|---|"]
    minimax = []
    for phase in ("phase2", "phase4", "both_phases"):
        rs = [r for r in summaries if phase == "both_phases" or r["phase"] == phase]
        maxima = {op:max(r["max_regret"] for r in rs if r["policy"] == f"always_{op}") for op in saved.OPS}
        chosen = min(saved.OPS,key=lambda op:(maxima[op],op))
        minimax.append(dict(scope=phase, maximum_regret_by_api=maxima, choice=chosen))
        lines.append(f"| {phase} | " + " | ".join(f"{100*maxima[op]:.2f}" for op in saved.OPS) + f" | {chosen} | {'yes' if maxima[chosen] <= .1 else 'no'} |")
    lines += ["", "## Cross-stack transfer", "",
              "The local dimension rule is applied unchanged to both T4 runs. Conversely, each T4 distribution rule is applied to local with its uniform fallback fixed from that source T4 run's best static API over the whole run. The target stack is never used to select that fallback. T4 and its repeat are not independent stacks. Rules originated from the full saved study, so these comparisons are descriptive transfer checks rather than a formal held-out learning protocol.", "",
              "Cross-stack transfer is poor relative to the target's calibrated static reference: every transferred rule increases both mean and maximum regret. The local dimension rule helps on its source grid, but the same split does not carry over to the T4 measurements. Both T4 distribution heuristics collapse to always_scatter_add because their calibrated uniform fallback is also scatter_add. No target-tuned replacement or additional feature split is introduced to improve a disappointing rule.", "",
              "## What practical ties imply", "",
              "A practical tie only constrains the fastest and runner-up paths. The third API can still be substantially slower. Consequently, a tie-heavy group does not make arbitrary API selection inexpensive. No simplest-implementation or no-change choice is invented for the T4 proxy.", "",
              "| Phase 4 T4 scope | Static API | Tie observations | Mean regret within ties % | Maximum within ties % |",
              "|---|---|---:|---:|---:|"]
    for r in summaries:
        if r["phase"]=="phase4" and r["environment_group"]=="colab_t4" and r["policy"].startswith("always_"):
            lines.append(f"| {r['scope']} | {r['policy']} | {r['n_tie_observations']} | {100*r['mean_regret_on_ties']:.2f} | {100*r['max_regret_on_ties']:.2f} |")
    lines += ["", "## Assumptions and limitations", "",
              "- The candidate API set is fixed to three paths. No switching overhead, implementation complexity, memory cost, optimizer behavior, or production throughput is modeled.",
              "- Saved medians are treated as fixed. Timing variability is not propagated; extreme ratios can be sensitive to a small fastest median. The case-level CSV preserves selected/best times and full observation identity for inspection.",
              "- The grid is controlled and synthetic, not an independent random sample. Draws/seeds sharing configurations are related. Whole-group averages can conceal heterogeneity and absolute-time differences.",
              "- Best-static selection and the winner-count policy use the evaluated data. The fixed workload heuristics are exploratory summaries motivated by the same study, not external validation. Oracle choices require knowing each observation's timings.",
              "- Region replay remains a diagnostic scope, not a decomposition of the proxy step. Stack differences combine hardware, software, and host conditions.",
              f"- Optional shallow tree: {tree['reason']}", "",
              "This analysis can support a narrowly framed discussion of static-choice costs and the limits of coarse workload heuristics after reviewing outlier cases. It does not establish a deployment policy. Reproduce with `python sparse-gradient/regret_policy_analysis.py`; only saved-artifact reconstruction helpers are invoked."]
    return "\n".join(lines)+"\n", minimax


def main():
    groups, checks, included, excluded = reconstruct()
    distilled = BASE / "results/decision_rule_summary/decision_rule_distilled_summary.csv"
    if distilled.exists():
        saved.read_csv(distilled)  # Record provenance of pre-existing rule motivation.
    sources = {g["environment_group"]:g for g in groups if g["phase"] == "phase2"}
    summaries, cases = [], []
    for group in groups:
        for policy in policies(group,sources):
            summary, rows = evaluate(group,policy)
            summaries.append(summary)
            cases.extend(rows)
    tree = dict(computed=False, sklearn_available=importlib.util.find_spec("sklearn") is not None)
    tree["reason"] = ("Skipped: scikit-learn is unavailable in the analysis interpreter; no dependency installed." if not tree["sklearn_available"]
                      else "Skipped optional model fitting to keep this pass focused on the specified fixed heuristics.")
    report, minimax = make_report(groups,summaries,tree)
    OUT.mkdir(parents=True,exist_ok=True)
    for phase in ("phase2", "phase4"):
        write_csv(OUT/f"{phase}_regret_policies.csv",[r for r in summaries if r["phase"]==phase],SUMMARY_FIELDS)
    write_csv(OUT/"regret_policy_cases.csv",cases,CASE_FIELDS)
    (OUT/"regret_policy_report.md").write_text(report,encoding="utf-8")
    audit = dict(baseline_checks="passed",baseline_counts=checks,phase4_included=included,phase4_excluded=excluded,
                 definition="regret=selected_median/minimum_of_three_medians-1; no 5% zeroing",
                 practical_winner_threshold=.05,percentage_columns="fraction*100",
                 percentiles="linear interpolation at (n-1)*p",observation_weighting="equal within each group",
                 static_calibration="lowest mean regret over same phase/environment/scope",
                 winner_count_choice="highest resolved >5% winner count; ties excluded; lexical tie break",
                 transfer="fixed source rule; source best-static uniform fallback is never recalibrated on target",
                 optional_tree=tree,python_executable=sys.executable,
                 summary_rows=len(summaries),case_rows=len(cases),minimax_static=minimax,
                 input_sha256=saved.SOURCES,
                 code_sha256={name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in
                              ["regret_policy_analysis.py","threshold_sensitivity.py","bootstrap_winner_shares.py","decision_rule_summary.py"]},
                 manuscript_edited=False,experiments_run=False)
    (OUT/"regret_policy_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print(f"Baseline checks passed. {len(summaries)} policy summaries; {len(cases)} case rows.")
    for r in summaries:
        print(r["phase"],r["environment_group"],r["scope"],r["policy"],
              f"mean={100*r['mean_regret']:.2f}%, max={100*r['max_regret']:.2f}%")


if __name__ == "__main__":
    main()
