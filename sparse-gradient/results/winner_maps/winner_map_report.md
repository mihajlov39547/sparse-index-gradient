# Exploratory winner maps

Both Phase 2 and Phase 4 5% baseline checks passed exactly. Cell marginal counts also match the saved distribution + dim grouped summaries.

These winner maps are descriptive summaries of saved observations. They are not predictive decision rules, not statistical tests, and not deployment recommendations.

## Construction

Exact grouping: phase + environment + scope + dim + distribution + ratio_bucket. Phase 2 uses matched configuration-draw observations; Phase 4 uses seed-configuration observations. Winner labels are recomputed from saved per-variant medians, using runner-up / fastest - 1 > 0.05. No observation or small-support cell is discarded.

Buckets use indices/rows: low < 1, balanced 1 <= ratio < 4, high >= 4. Phase 2 has no low-pressure observations, so its columns show balanced and high only; this is absent grid coverage, not filtering. Phase 4 shows all three buckets. Dim values are on rows; columns are distribution × pressure, ordered uniform, Zipf, hotspot. Absolute row/reference counts are pooled within buckets.

Cells show the most frequent category (including practical ties), with its support count over all observations in that cell. A plurality below 50% is still a dominant category. Equal-frequency leaders are listed together and drawn with neutral hatching; these are not relabeled as practical ties. Category shares and all four counts are retained in the source CSV.

Matplotlib available: True. Version: 3.11.1. Python: `C:\Projects\phd-segmentacija-instanci\.venv\Scripts\python.exe`.

## Figure sizes and category colors

The same colors apply to every map: idx/index_add #0072B2 (blue); scat/scatter_add #E69F00 (orange); emb/embedding #009E73 (green); tie #D9D9D9 (gray). Every cell also has text; color is not the sole encoding.

- `phase2_winner_map_local.png`: 8.4 × 3.8 inches, 240 dpi, 2016 × 912 pixels; 18 cells, support 3–15 observations per cell.
- `phase2_winner_map_t4.png`: 8.4 × 3.8 inches, 240 dpi, 2016 × 912 pixels; 18 cells, support 3–15 observations per cell.
- `phase2_winner_map_t4_repeat.png`: 8.4 × 3.8 inches, 240 dpi, 2016 × 912 pixels; 18 cells, support 3–15 observations per cell.
- `phase4_winner_map_local_proxy.png`: 10.2 × 3.5 inches, 240 dpi, 2448 × 840 pixels; 18 cells, support 5–10 observations per cell.
- `phase4_winner_map_t4_proxy.png`: 10.2 × 3.5 inches, 240 dpi, 2448 × 840 pixels; 18 cells, support 5–10 observations per cell.
- `phase4_winner_map_local_region.png`: 10.2 × 3.5 inches, 240 dpi, 2448 × 840 pixels; 18 cells, support 5–10 observations per cell.
- `phase4_winner_map_t4_region.png`: 10.2 × 3.5 inches, 240 dpi, 2448 × 840 pixels; 18 cells, support 5–10 observations per cell.

## Main visual observations

- Phase 2 local: balanced-pressure cells favor index_add (3/3 each). Under high pressure, dim=64 favors index_add for Zipf (15/15) and hotspot (12/15), but uniform has only an Embedding plurality (7/15). At dim=256/512, high-pressure cells favor Embedding (14/15 or 15/15). A general lower-dimension index_add statement needs this uniform exception.
- Phase 2 T4 and repeat: larger-dimension Zipf/hotspot cells mainly favor scatter_add, while high-pressure uniform at dim=512 favors Embedding (15/15 in both runs). Low-dimensional skewed cells include practical ties. The initial T4 dim=64/high-pressure hotspot cell has equal scatter_add and tie counts (7/15 each), while the repeat has a tie majority (8/15).
- Phase 4 local proxy: index_add occupies most cells, but low-pressure uniform is tied (5/5 at both dimensions). At dim=128/high pressure, uniform and hotspot favor Embedding (5/5), while Zipf is tied (5/5). These exceptions should remain visible in any rule-of-thumb prose.
- Phase 4 T4 proxy: every dim=32 observation is a practical tie. At dim=128/high pressure, uniform and hotspot favor scatter_add (5/5); balanced-pressure Zipf and hotspot split evenly between scatter_add and ties (5/10 each).
- Region replays are useful companion diagnostics: local index_add is more prevalent than in total proxy-step timing, and T4 shows more scatter_add cells at dim=128. These maps must not be described as a total-step decomposition.

The source CSV retains every cell and its four category counts, including small-support cells and equal-count leaders.

## Limitations and manuscript consideration

99/126 cells contain fewer than 12 observations. This is a descriptive support flag, not an exclusion rule. A Phase 2 balanced-pressure cell contains one shape over three draws; a high-pressure cell pools five shapes over three draws. Phase 4 low/high cells each contain one shape over five seeds; balanced cells pool two shapes over five seeds. Repeated draws/seeds do not constitute independent shapes.

Phase 2 local and T4 maps plus the two Phase 4 proxy-step maps are the primary candidates for a future four-panel, full-width figure. A single-column reduction would make counts difficult to read. The T4 repeat is useful as a companion artifact, not a separate independent environment. The optional region maps should remain diagnostic companion artifacts unless they add a specific scope comparison; the replay is not a decomposition of total proxy-step time.

Bucket aggregation can mask changes within absolute shapes or ratios. The dominant label does not imply every observation agrees, and the support annotation must remain visible. No timing-error propagation, resampling, model fitting, or external validation is performed. Grid coverage and dimension ranges differ across phases. Software and hardware differ jointly across environments.

No manuscript text was changed. Any later rule-of-thumb text should preserve shape/scope qualifiers and the observed support rather than generalizing a cell label to an API-wide recommendation.

## Files

- `winner_map_source.csv`
- `winner_map_report.md`
- `winner_map_audit.json`
- `phase2_winner_map_local.png`
- `phase2_winner_map_t4.png`
- `phase2_winner_map_t4_repeat.png`
- `phase4_winner_map_local_proxy.png`
- `phase4_winner_map_t4_proxy.png`
- `phase4_winner_map_local_region.png`
- `phase4_winner_map_t4_region.png`

Skipped/missing outputs: none

Reproduce: `python sparse-gradient/winner_map_heatmaps.py`. Only read-only helper functions are called; their benchmark/analysis output entry points are not invoked. Matplotlib is optional: CSV, report, and audit are still produced if plotting is unavailable or fails.
