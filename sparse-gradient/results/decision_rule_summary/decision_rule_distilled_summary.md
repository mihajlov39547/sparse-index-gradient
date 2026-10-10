# Distilled decision-rule summary

## Status

- Baseline checks: passed for Phase 2 and Phase 4. Phase 2 contains 162 matched configuration-draw observations per environment; Phase 4 contains 120 seed-configuration observations per environment and scope.
- Source files: `decision_rule_candidates.csv`, `phase2_grouped_rules.csv`, `phase4_grouped_rules.csv`, `decision_rule_summary_report.md`, and `decision_rule_summary_audit.json`.
- Number of original candidates: 132.
- Number of distilled patterns: 8.
- Interpretation status: descriptive rule-of-thumb summaries from the saved grid; not a predictive model, deployment recommendation, or statistically validated decision tree.

## Distilled patterns worth considering

### Pattern 1: Lower-dimensional local observations favored `index_add`

Evidence: In the local GTX1650Ti Phase 2 grouped summary, `dim=64` produced 42 `index_add` winners out of 54 observations, with 18 distinct configurations. The dominant share was 77.8% and the tie share was 9.3%.

Interpretation: Lower-dimensional cases in the saved local grid more often favored `index_add`.

Caution: This is a grouped observation from the tested grid, not a dimension threshold or deployment rule. It requires validation on the target workload and stack.

Manuscript recommendation: `yes_short_discussion`.

### Pattern 2: Higher-dimensional local observations favored `embedding`

Evidence: The disjoint `dim=256` and `dim=512` Phase 2 groups together contain 89 `embedding` winners out of 108 observations, with 36 distinct configurations. The combined dominant share is 82.4% and the combined tie share is 0.9%. Source rows are `phase2_grouped_rules.csv: dim=256` and `dim=512` for `local_gtx1650ti`.

Interpretation: The local grid shifted toward the native embedding backward path for the larger tested dimensions, contrasting with the lower-dimensional `index_add` pattern.

Caution: The combined summary aggregates two disjoint dimension groups and describes the saved grid only. It does not establish a general dimension cutoff.

Manuscript recommendation: `yes_short_discussion`.

### Pattern 3: Hotspot cases on T4 and T4 repeat more often favored `scatter_add`

Evidence: The T4 grouped row for `distribution=hotspot` contains 43 `scatter_add` winners out of 54 observations; the T4 repeat contains 41 out of 54. Combined, this is 84/108 observations, 77.8% dominant share, 18.5% tie share, and 36 distinct configurations across the two runs.

Interpretation: Hotspot-style cases in both T4 runs more often favored `scatter_add`, while ties remained substantial.

Caution: The repeat is a second run on the same T4 stack, not an independent environment. This is a recurring descriptive pattern, not a universal skew rule.

Manuscript recommendation: `yes_short_discussion`.

### Pattern 4: High-dimensional skewed T4 cases strongly favored `scatter_add`

Evidence: For `dim=512` with Zipf or hotspot distributions, the T4 and T4-repeat grouped rows contain 70 `scatter_add` winners out of 72 observations, 24 distinct configurations, a 97.2% combined dominant share, and a 1.4% combined tie share.

Interpretation: The strongest recurring T4 grouped pattern was associated with high-dimensional, skewed cases and `scatter_add`.

Caution: This is a focused subgroup and overlaps with the broader hotspot pattern. It is best used as a qualified example, not as a separate general rule.

Manuscript recommendation: `maybe`.

### Pattern 5: The local Phase 4 proxy step was mostly `index_add`-oriented

Evidence: Across the saved local Phase 4 proxy-step aggregate, `index_add` won 83/120 observations, `embedding` won 10/120, and 27/120 were ties. The dominant share was 69.2% and the tie share was 22.5%.

Interpretation: The local stack remained more often `index_add`-oriented when the accumulation path was placed inside the small synthetic training-step proxy.

Caution: The proxy is not production training. Its variants include custom autograd and graph overhead, and the aggregate is descriptive rather than predictive.

Manuscript recommendation: `yes_short_discussion`.

### Pattern 6: The local accumulation-region replay was more consistently `index_add`-oriented

Evidence: Across the local Phase 4 accumulation-region aggregate, `index_add` won 98/120 observations, `embedding` won 10/120, and 12/120 were ties. The dominant share was 81.7% and the tie share was 10.0%.

Interpretation: The diagnostic replay showed a stronger `index_add` concentration than the full proxy-step scope.

Caution: The accumulation region is a diagnostic replay. It is not an exact decomposition of the proxy step and must not be subtracted from proxy-step latency.

Manuscript recommendation: `maybe`.

### Pattern 7: The T4 Phase 4 proxy step was tie-heavy

Evidence: Across the T4 Phase 4 proxy-step aggregate, 95/120 observations were ties and 25/120 favored `scatter_add`; no `index_add` or embedding wins were recorded. The tie share was 79.2%.

Interpretation: In the T4 proxy-step scope, most observations were unresolved by the 5% practical-margin rule; the resolved observations mainly favored `scatter_add`.

Caution: Tie-heavy evidence does not support a universal ranking and remains specific to this synthetic proxy and stack.

Manuscript recommendation: `yes_short_discussion`.

### Pattern 8: The T4 accumulation-region replay remained tie-heavy

Evidence: Across the T4 Phase 4 accumulation-region aggregate, 78/120 observations were ties, 39/120 favored `scatter_add`, and 3/120 favored `index_add`. The tie share was 65.0% and `scatter_add` was the most frequent resolved winner.

Interpretation: The diagnostic T4 replay remained mostly unresolved, with `scatter_add` leading among resolved observations.

Caution: This replay is diagnostic and must not be interpreted as a production decomposition or universal API ranking.

Manuscript recommendation: `maybe`.

## Patterns not worth adding directly

The following candidate types were excluded from the distilled set:

- inverse-ratio duplicates such as `indices_to_rows_ratio=1`, `rows_to_indices_ratio=1`, and `balanced_reference_pressure` when they describe the same balanced subset;
- exact duplicate/skew values such as individual `duplicate_ratio`, `unique_rows`, `max_row_count`, or `hot_row_share` values, unless they formed a broader interpretable grouping;
- narrow two-feature restatements that merely repeated a stronger single-feature dimension or distribution observation with smaller support;
- tiny-support Phase 4 fragments with only a few distinct configurations;
- overlapping candidate rows that would make the number of rows look like independent corroboration when they reuse the same observations.

These exclusions follow the saved audit's exploratory limitations: candidates overlap, support includes repeated draws or seeds, no held-out validation was performed, and no causal attribution follows from a grouped feature association.

## Suggested manuscript use

A short Discussion note could mention the lower-dimensional versus higher-dimensional contrast on the local stack, the recurring T4/T4-repeat `scatter_add` pattern for hotspot or high-dimensional skewed cases, and the Phase 4 scope contrast: local observations were more often `index_add`-oriented, whereas T4 observations were largely tie-heavy with `scatter_add` leading among resolved cases. This should be framed as a descriptive grouped summary that requires validation on the target workload and stack, not as a predictive rule or deployment recommendation.

Exploratory grouped summaries provide practical but non-predictive rule-of-thumb context. In the saved Phase~2 grid, the local stack more often favored `index_add` for lower-dimensional cases and `embedding` for larger dimensions, whereas the T4 stack and repeat more often favored `scatter_add` in hotspot and high-dimensional skewed combinations. In the Phase~4 proxy, the local stack remained more often `index_add`-oriented, while the T4 proxy was largely tie-heavy and favored `scatter_add` mainly among resolved observations. These patterns are descriptive observations from the saved grid, not a decision tree or deployment recommendation, and should be validated on the target workload and stack.
