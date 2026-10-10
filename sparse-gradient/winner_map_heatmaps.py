"""Exploratory categorical winner maps from saved medians; never runs benchmarks."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import decision_rule_summary as rules
import bootstrap_winner_shares as reconstruction
import threshold_sensitivity as saved

BASE = Path(__file__).resolve().parent
OUT = BASE / "results/winner_maps"
COLORS = {"index_add": "#0072B2", "scatter_add": "#E69F00", "embedding": "#009E73", "tie": "#D9D9D9"}
SHORT = {"index_add": "idx", "scatter_add": "scat", "embedding": "emb", "tie": "tie"}
BUCKETS = ("low_reference_pressure", "balanced_reference_pressure", "high_reference_pressure")
BUCKET_LABELS = dict(zip(BUCKETS, ("Low", "Balanced", "High")))
DISTS = ("uniform", "zipf", "hotspot")
FIGURES = {
    ("phase2", "local_gtx1650ti", ""): "phase2_winner_map_local.png",
    ("phase2", "t4", ""): "phase2_winner_map_t4.png",
    ("phase2", "t4_repeat", ""): "phase2_winner_map_t4_repeat.png",
    ("phase4", "local_gtx1650ti", "proxy_step"): "phase4_winner_map_local_proxy.png",
    ("phase4", "colab_t4", "proxy_step"): "phase4_winner_map_t4_proxy.png",
    ("phase4", "local_gtx1650ti", "accumulation_region"): "phase4_winner_map_local_region.png",
    ("phase4", "colab_t4", "accumulation_region"): "phase4_winner_map_t4_region.png",
}
FIELDS = ("phase", "environment_group", "scope", "dim", "distribution", "ratio_bucket",
          "n_observations", "index_add_wins", "scatter_add_wins", "embedding_wins", "ties",
          "dominant_category", "dominant_count", "dominant_share", "cell_label", "figure_file")


def prepare():
    groups, checks, included, excluded = reconstruction.reconstruct()
    features = rules.load_features(included)
    rows = []
    for group in groups:
        identity = tuple(group[k] for k in ("phase", "environment_group", "scope"))
        cells = defaultdict(list)
        for key, winner in group["observations"]:
            feature = rules.features(features[(group["phase"], group["environment_group"], key)], group["phase"])
            cells[(feature["dim"], feature["distribution"], feature["ratio_bucket"])].append(winner)
        group_rows = []
        for (dim, distribution, bucket), labels in sorted(cells.items()):
            c = Counter(labels)
            maximum = max(c.values())
            leaders = [cat for cat in SHORT if c[cat] == maximum]
            # A tied category count is not the same thing as practical-tie dominance.
            label = "/".join(SHORT[c] for c in leaders) + f"\n{maximum}/{len(labels)}"
            if len(leaders) > 1:
                label += " each"
            row = dict(zip(("phase", "environment_group", "scope"), identity))
            row.update(dim=dim, distribution=distribution, ratio_bucket=bucket,
                       n_observations=len(labels), **dict(zip(saved.COUNT_FIELDS, [c[op] for op in (*saved.OPS, "tie")])),
                       dominant_category=";".join(leaders), dominant_count=maximum,
                       dominant_share=maximum/len(labels), cell_label=label, figure_file=FIGURES[identity])
            group_rows.append(row)
        baseline = next(c for c in checks if all(c[k] == group[k] for k in ("phase", "environment_group", "scope")))
        for field in saved.COUNT_FIELDS:
            saved.require(sum(r[field] for r in group_rows) == baseline[field], "Cell totals do not match baseline")
        rows.extend(group_rows)
    # Independently check cell marginals against existing grouped-result artifacts.
    for phase in ("phase2", "phase4"):
        grouped = saved.read_csv(BASE / f"results/decision_rule_summary/{phase}_grouped_rules.csv")
        for r in grouped:
            if r["grouping"] != "distribution + dim":
                continue
            value = json.loads(r["group_value"])
            selected = [x for x in rows if x["phase"] == phase and x["environment_group"] == r["environment_group"]
                        and x["scope"] == r["scope"] and x["dim"] == value["dim"] and x["distribution"] == value["distribution"]]
            for field in ("n_observations", *saved.COUNT_FIELDS):
                saved.require(sum(x[field] for x in selected) == int(r[field]), f"Grouped artifact mismatch: {phase}/{field}")
    return rows, checks, included, excluded


def plot(rows, path, plt, Patch, Rectangle):
    phase, env, scope = (rows[0][k] for k in ("phase", "environment_group", "scope"))
    dims = sorted({r["dim"] for r in rows})
    buckets = [b for b in BUCKETS if any(r["ratio_bucket"] == b for r in rows)]
    columns = [(d, b) for d in DISTS for b in buckets]
    size = (8.4, 3.8) if phase == "phase2" else (10.2, 3.5)
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=.07, right=.99, top=.70, bottom=.25)
    lookup = {(r["dim"], r["distribution"], r["ratio_bucket"]): r for r in rows}
    for y, dim in enumerate(dims):
        for x, (dist, bucket) in enumerate(columns):
            r = lookup.get((dim, dist, bucket))
            cat = r["dominant_category"] if r else "missing"
            multi = r is not None and ";" in cat
            color = COLORS.get(cat, "#FFFFFF")
            ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, facecolor=color, edgecolor="white", linewidth=1.5))
            if multi:
                ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, fill=False, hatch="//", edgecolor="#AAAAAA", linewidth=.5))
            ax.text(x, y, r["cell_label"] if r else "no data", ha="center", va="center", fontsize=10,
                    color="white" if cat == "index_add" else "black", linespacing=1.35)
    ax.set_xlim(-.5, len(columns)-.5)
    ax.set_ylim(len(dims)-.5, -.5)
    ax.set_xticks(range(len(columns)), [f"{d.title()}\n{BUCKET_LABELS[b]}" for d,b in columns], fontsize=9)
    ax.set_yticks(range(len(dims)), [str(d) for d in dims], fontsize=10)
    ax.set_ylabel("Dimension", fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for boundary in range(len(buckets), len(columns), len(buckets)):
        ax.axvline(boundary-.5, color="#444444", linewidth=1)
    names = {"local_gtx1650ti": "Local GTX1650Ti", "t4": "T4", "t4_repeat": "T4 repeat", "colab_t4": "Colab T4"}
    title = f"{'Phase 2' if phase == 'phase2' else 'Phase 4'} — {names[env]}"
    if scope:
        title += " — " + ("Proxy step" if scope == "proxy_step" else "Region replay (diagnostic)")
    fig.suptitle(title, fontsize=13, y=.98)
    handles = [Patch(facecolor=color, label=f"{SHORT[c]}: {c}") for c, color in COLORS.items()]
    if any(";" in r["dominant_category"] for r in rows):
        handles.append(Patch(facecolor="white", edgecolor="#AAAAAA", hatch="//", label="equal category counts"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52,.91), ncol=len(handles), frameon=False, fontsize=9)
    fig.text(.5, .12, "Reference pressure = indices / rows: Low < 1; Balanced 1–<4; High ≥4", ha="center", fontsize=9)
    fig.text(.5, .055, "Cell: dominant category and its count / observations • saved strict >5% winners", ha="center", fontsize=9)
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)
    return dict(filename=path.name, inches=size, dpi=240, pixels=[round(v*240) for v in size],
                dimensions=dims, pressure_buckets=buckets, cells=len(rows),
                min_cell_observations=min(r["n_observations"] for r in rows),
                max_cell_observations=max(r["n_observations"] for r in rows))


def report(rows, checks, generated, skipped, plotting, dimensions):
    text = ["# Exploratory winner maps", "", "Both Phase 2 and Phase 4 5% baseline checks passed exactly. Cell marginal counts also match the saved distribution + dim grouped summaries.", "",
            "These winner maps are descriptive summaries of saved observations. They are not predictive decision rules, not statistical tests, and not deployment recommendations.", "",
            "## Construction", "",
            "Exact grouping: phase + environment + scope + dim + distribution + ratio_bucket. Phase 2 uses matched configuration-draw observations; Phase 4 uses seed-configuration observations. Winner labels are recomputed from saved per-variant medians, using runner-up / fastest - 1 > 0.05. No observation or small-support cell is discarded.", "",
            "Buckets use indices/rows: low < 1, balanced 1 <= ratio < 4, high >= 4. Phase 2 has no low-pressure observations, so its columns show balanced and high only; this is absent grid coverage, not filtering. Phase 4 shows all three buckets. Dim values are on rows; columns are distribution × pressure, ordered uniform, Zipf, hotspot. Absolute row/reference counts are pooled within buckets.", "",
            "Cells show the most frequent category (including practical ties), with its support count over all observations in that cell. A plurality below 50% is still a dominant category. Equal-frequency leaders are listed together and drawn with neutral hatching; these are not relabeled as practical ties. Category shares and all four counts are retained in the source CSV.", "",
            f"Matplotlib available: {plotting['available']}. Version: {plotting.get('version', 'unavailable')}. Python: `{sys.executable}`.", "",
            "## Figure sizes and category colors", "",
            "The same colors apply to every map: idx/index_add #0072B2 (blue); scat/scatter_add #E69F00 (orange); emb/embedding #009E73 (green); tie #D9D9D9 (gray). Every cell also has text; color is not the sole encoding.", ""]
    for d in dimensions:
        text.append(f"- `{d['filename']}`: {d['inches'][0]} × {d['inches'][1]} inches, {d['dpi']} dpi, {d['pixels'][0]} × {d['pixels'][1]} pixels; {d['cells']} cells, support {d['min_cell_observations']}–{d['max_cell_observations']} observations per cell.")
    text += ["", "## Main visual observations", "",
             "- Phase 2 local: balanced-pressure cells favor index_add (3/3 each). Under high pressure, dim=64 favors index_add for Zipf (15/15) and hotspot (12/15), but uniform has only an Embedding plurality (7/15). At dim=256/512, high-pressure cells favor Embedding (14/15 or 15/15). A general lower-dimension index_add statement needs this uniform exception.",
             "- Phase 2 T4 and repeat: larger-dimension Zipf/hotspot cells mainly favor scatter_add, while high-pressure uniform at dim=512 favors Embedding (15/15 in both runs). Low-dimensional skewed cells include practical ties. The initial T4 dim=64/high-pressure hotspot cell has equal scatter_add and tie counts (7/15 each), while the repeat has a tie majority (8/15).",
             "- Phase 4 local proxy: index_add occupies most cells, but low-pressure uniform is tied (5/5 at both dimensions). At dim=128/high pressure, uniform and hotspot favor Embedding (5/5), while Zipf is tied (5/5). These exceptions should remain visible in any rule-of-thumb prose.",
             "- Phase 4 T4 proxy: every dim=32 observation is a practical tie. At dim=128/high pressure, uniform and hotspot favor scatter_add (5/5); balanced-pressure Zipf and hotspot split evenly between scatter_add and ties (5/10 each).",
             "- Region replays are useful companion diagnostics: local index_add is more prevalent than in total proxy-step timing, and T4 shows more scatter_add cells at dim=128. These maps must not be described as a total-step decomposition.", "",
             "The source CSV retains every cell and its four category counts, including small-support cells and equal-count leaders."]
    low = [r for r in rows if r["n_observations"] < 12]
    text += ["", "## Limitations and manuscript consideration", "",
             f"{len(low)}/{len(rows)} cells contain fewer than 12 observations. This is a descriptive support flag, not an exclusion rule. A Phase 2 balanced-pressure cell contains one shape over three draws; a high-pressure cell pools five shapes over three draws. Phase 4 low/high cells each contain one shape over five seeds; balanced cells pool two shapes over five seeds. Repeated draws/seeds do not constitute independent shapes.", "",
             "Phase 2 local and T4 maps plus the two Phase 4 proxy-step maps are the primary candidates for a future four-panel, full-width figure. A single-column reduction would make counts difficult to read. The T4 repeat is useful as a companion artifact, not a separate independent environment. The optional region maps should remain diagnostic companion artifacts unless they add a specific scope comparison; the replay is not a decomposition of total proxy-step time.", "",
             "Bucket aggregation can mask changes within absolute shapes or ratios. The dominant label does not imply every observation agrees, and the support annotation must remain visible. No timing-error propagation, resampling, model fitting, or external validation is performed. Grid coverage and dimension ranges differ across phases. Software and hardware differ jointly across environments.", "",
             "No manuscript text was changed. Any later rule-of-thumb text should preserve shape/scope qualifiers and the observed support rather than generalizing a cell label to an API-wide recommendation.", "",
             "## Files", ""]
    text += [f"- `{name}`" for name in generated]
    text += ["", "Skipped/missing outputs: " + (json.dumps(skipped) if skipped else "none"), "",
             "Reproduce: `python sparse-gradient/winner_map_heatmaps.py`. Only read-only helper functions are called; their benchmark/analysis output entry points are not invoked. Matplotlib is optional: CSV, report, and audit are still produced if plotting is unavailable or fails."]
    return "\n".join(text) + "\n"


def main():
    rows, checks, included, excluded = prepare()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "winner_map_source.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    generated = ["winner_map_source.csv", "winner_map_report.md", "winner_map_audit.json"]
    skipped, dimensions = [], []
    plotting = dict(available=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch, Rectangle
        plotting.update(available=True, version=matplotlib.__version__)
    except Exception as exc:
        plotting["error"] = f"{type(exc).__name__}: {exc}"
    for key, filename in FIGURES.items():
        if not plotting["available"]:
            skipped.append(dict(filename=filename, reason=plotting["error"]))
            continue
        selected = [r for r in rows if tuple(r[k] for k in ("phase", "environment_group", "scope")) == key]
        try:
            dimensions.append(plot(selected, OUT / filename, plt, Patch, Rectangle))
            generated.append(filename)
        except Exception as exc:
            plt.close("all")
            skipped.append(dict(filename=filename, reason=f"{type(exc).__name__}: {exc}"))
    audit = dict(baseline_checks="passed", baseline_counts=checks, grouped_artifact_marginal_checks="passed",
                 grouping=["phase", "environment_group", "scope", "dim", "distribution", "ratio_bucket"],
                 bucket_definition="indices/rows: low <1; balanced >=1 and <4; high >=4",
                 category_colors=COLORS, category_short_labels=SHORT,
                 equal_category_counts="all leaders listed with shared count, neutral white hatch; distinct from practical tie category",
                 cell_count=len(rows), small_support_cells=sum(r["n_observations"] < 12 for r in rows),
                 matplotlib=plotting, figure_dimensions=dimensions, generated_files=generated,
                 skipped_outputs=skipped, phase4_included=included, phase4_excluded=excluded,
                 input_sha256=saved.SOURCES,
                 code_sha256={name: hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in
                              ["winner_map_heatmaps.py", "decision_rule_summary.py", "bootstrap_winner_shares.py", "threshold_sensitivity.py"]},
                 manuscript_files_edited=False, experiments_run=False,
                 source_note="Three-feature cells reconstructed from saved observations; checked against two-feature grouped summaries. Distilled candidate selections do not filter the maps.")
    (OUT / "winner_map_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (OUT / "winner_map_report.md").write_text(report(rows, checks, generated, skipped, plotting, dimensions), encoding="utf-8")
    print(f"Baselines and grouped marginals passed. {len(rows)} cells; {len(dimensions)} maps; {len(skipped)} skipped.")


if __name__ == "__main__":
    main()
