# Serial Dependence Averaging Levels On The SuperMemo Benchmark

Date: 2026-04-15

Analysis script: [sm17_serial_dependence_averaging_levels.py](./sm17_serial_dependence_averaging_levels.py)
Result file: [results/sm17_conditional_probability_levels.json](./results/sm17_conditional_probability_levels.json)

## Goal

This note attempts to reproduce, on the SuperMemo benchmark dataset in this repository, the main methodological point from:

- `open-spaced-repetition/Anki-button-usage/research/same_day_serial_dependence_averaging_levels/`

The target is not to prove or disprove causal serial dependence. The target is narrower:

- check whether the descriptive gap
  - `P(success | prev success) - P(success | prev fail)`
- is highly sensitive to the averaging level
  - `pooled`
  - `equal_user_mean`
  - `equal_user_day_mean`

If it is, then the older “recent outcomes are predictive” interpretation is not robust.

## SuperMemo-Specific Sequence Definitions

The SuperMemo CSVs do not expose the same fields as the Anki revlog dataset, so this reproduction uses the closest available analogs.

Shared preprocessing:

1. use the same row-level filters as the benchmark preprocessing
2. keep only `delta_t > 0`
3. compute `i` by card
4. drop the first record per card with `i == 1` to match the benchmark evaluation subset
5. sort by parsed `review_date`, then `i`, then source row order

Two sequence definitions are reported:

### `raw_long_term`

- adjacent pairs in the full benchmark-style user sequence

### `same_day_first_per_card`

- within each user-day-card, keep only the first parsed-order occurrence
- then retain same-day adjacent pairs only

This is only an analog of the Anki `same_day_first_of_day_review` target because the SuperMemo CSV lacks a `state == Review` field.

## Important Limitations

1. Some collections have weak within-day time resolution.
   Example: a subset of users have constant or near-constant times on a given day.
   So “previous” means parsed-order previous under the available timestamps and tie-breaking, not independently validated wall-clock order.
2. The `same_day_first_per_card` sequence is not identical to the Anki target because there is no `state` field.
3. As in the Anki note, `equal_user_day_mean.gap` is a separate-support gap, while `common_support_gap` is computed only on user-days where both conditionals are defined.

## Main Results

### 1. `raw_long_term`

| Level | `P(success \| prev success)` | `P(success \| prev fail)` | Gap |
| --- | ---: | ---: | ---: |
| `pooled` | `0.876065` | `0.827770` | `0.048294` |
| `equal_user_mean` | `0.859239` | `0.808034` | `0.051205` |
| `equal_user_day_mean` | `0.859724` | `0.867565` | `-0.007840` |

Common-support diagnostics:

- `equal_user_day_mean.common_support_gap = -0.029236`
- `both_defined_units = 7,617`

Restricted re-aggregations on common-support user-days:

- `pooled_on_common_support_user_days.gap = 0.045621`
- `equal_user_mean_on_common_support_user_days.gap = 0.028772`

Shrinkage summary:

- `pooled -> equal_user_mean`: `-0.002911`
- `equal_user_mean -> equal_user_day_mean`: `0.059045`

### 2. `same_day_first_per_card`

| Level | `P(success \| prev success)` | `P(success \| prev fail)` | Gap |
| --- | ---: | ---: | ---: |
| `pooled` | `0.879741` | `0.828505` | `0.051236` |
| `equal_user_mean` | `0.859291` | `0.811798` | `0.047493` |
| `equal_user_day_mean` | `0.858726` | `0.867476` | `-0.008750` |

Common-support diagnostics:

- `equal_user_day_mean.common_support_gap = -0.029523`
- `both_defined_units = 7,382`

Restricted re-aggregations on common-support user-days:

- `pooled_on_common_support_user_days.gap = 0.048648`
- `equal_user_mean_on_common_support_user_days.gap = 0.025118`

Shrinkage summary:

- `pooled -> equal_user_mean`: `0.003743`
- `equal_user_mean -> equal_user_day_mean`: `0.056243`

## Interpretation

The SuperMemo benchmark shows the same high-level methodological pattern as the Anki analysis, but even more starkly:

1. Moving from `pooled` to `equal_user_mean` changes almost nothing.
2. Moving from `equal_user_mean` to `equal_user_day_mean` changes the result dramatically.
3. Restricting to common-support user-days alone only modestly reduces the pooled or equal-user gap.
4. The large change happens when the weighting is pushed down to the `user-day` level.
5. On this dataset, that change is so strong that the day-level gap becomes slightly negative.

This implies:

- cross-user heterogeneity is not the main issue here
- day-level heterogeneity / composition is the dominant issue
- the old pooled adjacent-outcome summary is not a stable basis for claiming that recent outcomes are predictively useful in a way that should be interpreted as serial dependence

## Relation To The Older “Why recent outcomes are predictive” Section

The older section relied on statistics like:

- pooled `P(success | prev success)`
- pooled `P(success | prev fail)`
- weighted per-user summaries

On this SuperMemo dataset, those summaries are materially misleading for the same reason identified in the Anki note:

- they answer a mixed, high-activity-weighted question
- they do not answer the question at the level of a typical retained user-day

In fact, on the SuperMemo benchmark:

- the pooled gap is only about `0.05`
- the equal-user gap stays around `0.05`
- the equal-user-day gap flips sign to around `-0.008`
- the equal-user-day common-support gap is around `-0.029`

So the earlier “recent outcomes are predictive” argument does not survive the averaging-level correction.

## Most Defensible Conclusion

For the SuperMemo benchmark in this repository, the most defensible descriptive conclusion is:

1. The pooled and user-level adjacent-outcome gaps are not robust to changing the averaging level.
2. Day-level reweighting is the dominant transformation.
3. Once the analysis is pushed to equal-weight user-days, the apparent gap disappears and slightly reverses.
4. Therefore the earlier adjacent-outcome statistics should not be used as evidence that recent review outcomes carry a stable predictive signal suitable for motivating a `MOVING-AVG`-style scheduler story.

This is a descriptive correction result, not a final causal account of what drives day-level heterogeneity.
