# Descriptive rule-of-thumb summary

Phase 2 baseline verification: **passed** for all three runs (162 matched configuration-draw observations each).
Phase 4 baseline verification: **passed** for both environments and scopes (120 seed-configuration observations each).
All winner labels were reconstructed from saved per-variant medians using runner-up / fastest - 1 > 0.05; otherwise tie.

## Available features

- phase2: `dim`, `distribution`, `rows`, `indices`, `indices_to_rows_ratio`, `ratio_bucket`, `rows_to_indices_ratio`, `unique_rows`, `duplicate_ratio`, `max_row_count`, `hot_row_share`.
- phase4: `dim`, `distribution`, `rows`, `num_indices`, `indices_to_rows_ratio`, `ratio_bucket`, `unique_rows`, `duplicate_ratio`, `max_row_count`, `hot_row_share`.

Both phases have duplicate/skew information in saved summary rows. Phase 2 unique_indices counts distinct destination rows and is exposed as unique_rows; duplicate_rate is checked against 1 - unique_indices / indices. Phase 4 saves unique_rows and max_row_count, allowing exact duplicate_ratio = 1 - unique_rows / num_indices from summary statistics. No raw index reconstruction is used.

hot_row_share means the single most referenced row's fraction (max_row_count / number of references), a top-one concentration descriptor, not the generator's configured hotspot fraction. Phase 2 top1_share is checked against it. Phase 2 also saves top16_share and mean_count_per_used_row; these are not additional groupings in this pass. Distribution labels remain coarse descriptors and do not isolate skew effects.

Ratios are encoded as exact reduced fractions in group_value JSON. ratio_bucket uses indices/rows: low_reference_pressure < 1; balanced_reference_pressure from 1 inclusive to 4 exclusive; high_reference_pressure >= 4. Phase 2's matched grid covers balanced_reference_pressure and high_reference_pressure, with no low-pressure observations. Phase 4 covers all three buckets. The combinations are distribution + dim, distribution + ratio_bucket, and dim + ratio_bucket. Buckets were fixed before evaluating group winners.

## Candidate rules

132 groups meet n_observations >= 12 and dominant_share >= 0.70. These are descriptive display filters, not inferential criteria. The dominant category includes tie. If categories are equally frequent, all leaders are retained with semicolons; these groups cannot reach 70% dominance. All candidates, including overlapping and redundant groups, are saved in decision_rule_candidates.csv.

Candidates are sorted by phase, environment/scope, dominant share descending, then observation count descending; lexical feature ordering breaks remaining ties. Below are the first ten per environment/scope in that order. A high count of candidate rows is not independent corroboration.

### phase2 / local_gtx1650ti / matched grid

| Condition | Category | Count | Share | Ties | Distinct configurations |
|---|---|---:|---:|---:|---:|
| `{"dim":512,"ratio_bucket":"high_reference_pressure"}` | embedding | 45/45 | 100.0% | 0.0% | 15 |
| `{"indices_to_rows_ratio":"1"}` | index_add | 27/27 | 100.0% | 0.0% | 9 |
| `{"ratio_bucket":"balanced_reference_pressure"}` | index_add | 27/27 | 100.0% | 0.0% | 9 |
| `{"rows_to_indices_ratio":"1"}` | index_add | 27/27 | 100.0% | 0.0% | 9 |
| `{"dim":64,"distribution":"zipf"}` | index_add | 18/18 | 100.0% | 0.0% | 6 |
| `{"dim":256,"ratio_bucket":"high_reference_pressure"}` | embedding | 44/45 | 97.8% | 2.2% | 15 |
| `{"duplicate_ratio":0.9375}` | embedding | 16/18 | 88.9% | 11.1% | 6 |
| `{"unique_rows":4096}` | embedding | 16/18 | 88.9% | 11.1% | 6 |
| `{"dim":512}` | embedding | 45/54 | 83.3% | 0.0% | 18 |
| `{"dim":256,"distribution":"hotspot"}` | embedding | 15/18 | 83.3% | 0.0% | 6 |
### phase2 / t4 / matched grid

| Condition | Category | Count | Share | Ties | Distinct configurations |
|---|---|---:|---:|---:|---:|
| `{"dim":512,"distribution":"zipf"}` | scatter_add | 18/18 | 100.0% | 0.0% | 6 |
| `{"dim":256,"distribution":"hotspot"}` | scatter_add | 17/18 | 94.4% | 0.0% | 6 |
| `{"dim":512,"distribution":"hotspot"}` | scatter_add | 17/18 | 94.4% | 5.6% | 6 |
| `{"dim":512,"distribution":"uniform"}` | embedding | 15/18 | 83.3% | 16.7% | 6 |
| `{"dim":64,"distribution":"uniform"}` | scatter_add | 15/18 | 83.3% | 16.7% | 6 |
| `{"distribution":"hotspot"}` | scatter_add | 43/54 | 79.6% | 16.7% | 18 |
| `{"distribution":"hotspot","ratio_bucket":"high_reference_pressure"}` | scatter_add | 35/45 | 77.8% | 17.8% | 15 |
### phase2 / t4_repeat / matched grid

| Condition | Category | Count | Share | Ties | Distinct configurations |
|---|---|---:|---:|---:|---:|
| `{"dim":256,"distribution":"zipf"}` | scatter_add | 18/18 | 100.0% | 0.0% | 6 |
| `{"dim":512,"distribution":"zipf"}` | scatter_add | 18/18 | 100.0% | 0.0% | 6 |
| `{"dim":512,"distribution":"hotspot"}` | scatter_add | 17/18 | 94.4% | 0.0% | 6 |
| `{"dim":256,"distribution":"hotspot"}` | scatter_add | 15/18 | 83.3% | 16.7% | 6 |
| `{"dim":512,"distribution":"uniform"}` | embedding | 15/18 | 83.3% | 16.7% | 6 |
| `{"dim":64,"distribution":"zipf"}` | tie | 14/18 | 77.8% | 77.8% | 6 |
| `{"distribution":"hotspot"}` | scatter_add | 41/54 | 75.9% | 20.4% | 18 |
| `{"distribution":"zipf"}` | scatter_add | 40/54 | 74.1% | 25.9% | 18 |
| `{"distribution":"zipf","ratio_bucket":"high_reference_pressure"}` | scatter_add | 33/45 | 73.3% | 26.7% | 15 |
| `{"distribution":"hotspot","ratio_bucket":"high_reference_pressure"}` | scatter_add | 32/45 | 71.1% | 24.4% | 15 |
### phase4 / colab_t4 / accumulation_region

| Condition | Category | Count | Share | Ties | Distinct configurations |
|---|---|---:|---:|---:|---:|
| `{"indices_to_rows_ratio":"2"}` | tie | 29/30 | 96.7% | 96.7% | 6 |
| `{"dim":32,"distribution":"zipf"}` | tie | 19/20 | 95.0% | 95.0% | 4 |
| `{"dim":32,"distribution":"hotspot"}` | tie | 18/20 | 90.0% | 90.0% | 4 |
| `{"dim":32,"ratio_bucket":"balanced_reference_pressure"}` | tie | 26/30 | 86.7% | 86.7% | 6 |
| `{"dim":128,"ratio_bucket":"high_reference_pressure"}` | scatter_add | 13/15 | 86.7% | 13.3% | 3 |
| `{"dim":32,"ratio_bucket":"low_reference_pressure"}` | tie | 13/15 | 86.7% | 86.7% | 3 |
| `{"distribution":"zipf","ratio_bucket":"balanced_reference_pressure"}` | tie | 17/20 | 85.0% | 85.0% | 4 |
| `{"dim":32}` | tie | 50/60 | 83.3% | 83.3% | 12 |
| `{"ratio_bucket":"balanced_reference_pressure"}` | tie | 48/60 | 80.0% | 80.0% | 12 |
| `{"distribution":"uniform","ratio_bucket":"balanced_reference_pressure"}` | tie | 16/20 | 80.0% | 80.0% | 4 |
### phase4 / colab_t4 / proxy_step

| Condition | Category | Count | Share | Ties | Distinct configurations |
|---|---|---:|---:|---:|---:|
| `{"dim":32}` | tie | 60/60 | 100.0% | 100.0% | 12 |
| `{"num_indices":8192}` | tie | 60/60 | 100.0% | 100.0% | 12 |
| `{"dim":32,"ratio_bucket":"balanced_reference_pressure"}` | tie | 30/30 | 100.0% | 100.0% | 6 |
| `{"indices_to_rows_ratio":"1/8"}` | tie | 30/30 | 100.0% | 100.0% | 6 |
| `{"indices_to_rows_ratio":"2"}` | tie | 30/30 | 100.0% | 100.0% | 6 |
| `{"ratio_bucket":"low_reference_pressure"}` | tie | 30/30 | 100.0% | 100.0% | 6 |
| `{"dim":32,"distribution":"hotspot"}` | tie | 20/20 | 100.0% | 100.0% | 4 |
| `{"dim":32,"distribution":"uniform"}` | tie | 20/20 | 100.0% | 100.0% | 4 |
| `{"dim":32,"distribution":"zipf"}` | tie | 20/20 | 100.0% | 100.0% | 4 |
| `{"dim":128,"ratio_bucket":"low_reference_pressure"}` | tie | 15/15 | 100.0% | 100.0% | 3 |
### phase4 / local_gtx1650ti / accumulation_region

| Condition | Category | Count | Share | Ties | Distinct configurations |
|---|---|---:|---:|---:|---:|
| `{"ratio_bucket":"balanced_reference_pressure"}` | index_add | 60/60 | 100.0% | 0.0% | 12 |
| `{"dim":128,"ratio_bucket":"balanced_reference_pressure"}` | index_add | 30/30 | 100.0% | 0.0% | 6 |
| `{"dim":32,"ratio_bucket":"balanced_reference_pressure"}` | index_add | 30/30 | 100.0% | 0.0% | 6 |
| `{"indices_to_rows_ratio":"1"}` | index_add | 30/30 | 100.0% | 0.0% | 6 |
| `{"indices_to_rows_ratio":"2"}` | index_add | 30/30 | 100.0% | 0.0% | 6 |
| `{"dim":32,"distribution":"hotspot"}` | index_add | 20/20 | 100.0% | 0.0% | 4 |
| `{"dim":32,"distribution":"zipf"}` | index_add | 20/20 | 100.0% | 0.0% | 4 |
| `{"distribution":"hotspot","ratio_bucket":"balanced_reference_pressure"}` | index_add | 20/20 | 100.0% | 0.0% | 4 |
| `{"distribution":"uniform","ratio_bucket":"balanced_reference_pressure"}` | index_add | 20/20 | 100.0% | 0.0% | 4 |
| `{"distribution":"zipf","ratio_bucket":"balanced_reference_pressure"}` | index_add | 20/20 | 100.0% | 0.0% | 4 |
### phase4 / local_gtx1650ti / proxy_step

| Condition | Category | Count | Share | Ties | Distinct configurations |
|---|---|---:|---:|---:|---:|
| `{"dim":32,"distribution":"zipf"}` | index_add | 20/20 | 100.0% | 0.0% | 4 |
| `{"distribution":"hotspot","ratio_bucket":"balanced_reference_pressure"}` | index_add | 20/20 | 100.0% | 0.0% | 4 |
| `{"distribution":"zipf","ratio_bucket":"balanced_reference_pressure"}` | index_add | 20/20 | 100.0% | 0.0% | 4 |
| `{"dim":32,"ratio_bucket":"high_reference_pressure"}` | index_add | 15/15 | 100.0% | 0.0% | 3 |
| `{"dim":128,"ratio_bucket":"balanced_reference_pressure"}` | index_add | 29/30 | 96.7% | 3.3% | 6 |
| `{"dim":32,"distribution":"hotspot"}` | index_add | 18/20 | 90.0% | 10.0% | 4 |
| `{"distribution":"zipf"}` | index_add | 35/40 | 87.5% | 12.5% | 8 |
| `{"indices_to_rows_ratio":"1"}` | index_add | 26/30 | 86.7% | 13.3% | 6 |
| `{"ratio_bucket":"balanced_reference_pressure"}` | index_add | 50/60 | 83.3% | 16.7% | 12 |
| `{"distribution":"hotspot"}` | index_add | 33/40 | 82.5% | 5.0% | 8 |

## Interpretation and limitations

The grouped summaries provide practical rule-of-thumb patterns, not a predictive model and not a deployment recommendation. They indicate which workload descriptors were associated with strong dominance or tie-heavy behavior within the saved grid. The patterns should be treated as prompts for targeted benchmarking rather than direct deployment rules; each requires validation on the target workload and stack.

Candidate selection is exploratory and uses the same observations being summarized, with no held-out validation. Groups overlap, ratios and inverse ratios are redundant, and dimensions/distributions/row counts can be confounded. Exact duplicate/skew values can fragment the grid into small groups; these rows remain in grouped CSVs but cannot become candidates without the stated support threshold. The support counts include repeated draws/seeds rather than independent workload shapes; n_distinct_configurations makes that distinction explicit. No causal attribution to a feature follows.

The T4 repeat is a second run on the T4 stack, not a third independent environment. Phase 4 is a mini proxy, not production training, and accumulation_region is a separate diagnostic replay, not a decomposition of proxy_step. Raw timing variability and uncertainty in saved medians are not propagated in this pass. No new timing runs, bootstrap resampling, threshold changes, or statistical tests are performed.

Simple, well-supported feature groups that recur across workload draws or the T4 repeat may merit a qualified Discussion example later. Their precise support, environment, and scope should accompany any statement; inverse-ratio or single-bucket restatements should not be presented as separate findings. No manuscript changes are made here.

Reproduce with `python sparse-gradient/decision_rule_summary.py`. The script calls only the read-only reconstruction helpers from bootstrap_winner_shares.py and threshold_sensitivity.py, not their analysis/output entry points. Input and code hashes, feature definitions, baseline counts, and coverage are recorded in the audit.
