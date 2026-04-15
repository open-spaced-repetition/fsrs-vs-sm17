# Rebuttal To 5 Internal Reviews After The SuperMemo Averaging-Level Replication

Date: 2026-04-15

This note records a second round of internal review on the original report [moving_avg_investigation.md](./moving_avg_investigation.md), after the new SuperMemo replication in [sm17_serial_dependence_averaging_levels.md](./sm17_serial_dependence_averaging_levels.md).

The reviewers were asked a narrow question:

- given the new averaging-level results, which parts of the original report are no longer publication-safe
- which parts still survive
- how the thesis should be rewritten

## New Empirical Update

The new replication materially changes the status of the old adjacent-outcome argument.

For both sequence definitions:

- `raw_long_term`
- `same_day_first_per_card`

the same pattern appears:

- `pooled` gap is about `+0.05`
- `equal_user_mean` gap is also about `+0.05`
- `equal_user_day_mean` gap flips slightly negative
- `equal_user_day_mean.common_support_gap` is about `-0.029`

Concretely:

- `raw_long_term`: `+0.048294`, `+0.051205`, `-0.007840`, `-0.029236`
- `same_day_first_per_card`: `+0.051236`, `+0.047493`, `-0.008750`, `-0.029523`

So the original “recent outcomes are predictive” subsection in [moving_avg_investigation.md](./moving_avg_investigation.md) can no longer rely on pooled or weighted-per-user adjacent conditional probabilities as robust support.

## Reviewer 1: The Adjacent-Outcome Section Must Be Withdrawn

### Concern

All 5 reviewers independently agreed that [moving_avg_investigation.md](./moving_avg_investigation.md) lines 199-209 are no longer supportable as written. The old section said the short-lag dependence was “clearly positive” and “enough to explain” why a recent-outcome tracker can be competitive.

After the new replication, that claim is not robust:

- the sign is positive under `pooled`
- still positive under `equal_user_mean`
- slightly negative under `equal_user_day_mean`
- more negative under `common_support_gap`

### Response

Accepted.

That section should no longer be treated as positive evidence for a stable one-step predictive effect. The correct replacement is an averaging-sensitivity result:

- pooled and weighted-per-user adjacent summaries are not robust
- the major shift happens when weighting is pushed to `user-day`
- the old section should be rewritten as a correction, not preserved as mechanism evidence

## Reviewer 2: The Short Answer And Bottom Line Overstate The Mechanism

### Concern

Reviewers flagged the following original claims as too strong:

- [moving_avg_investigation.md](./moving_avg_investigation.md) lines 15-22
- [moving_avg_investigation.md](./moving_avg_investigation.md) lines 247-249

In particular, these phrases now overreach:

- “recent outcomes in the same user stream contain predictive information about the next outcome”
- “`MOVING-AVG` gets to use short-term stream-level drift”
- “strong, time-varying calibration signal”

The new replication does not justify those as established facts.

### Response

Accepted.

The revised thesis should be weaker:

- `MOVING-AVG` benefits from ordered-stream, non-item information under the benchmark protocol
- the old positive adjacent-outcome story is not robust enough to identify that information
- the mechanism should now be described as unresolved stream-level calibration / composition / nonstationarity, not as proven short-lag dependence

## Reviewer 3: The Old Robustness Checks Were The Wrong Ones

### Concern

Several reviewers pointed out that the original report treated:

- `pooled`
- weighted per-user summaries

as if they were meaningful robustness checks for one another.

The new replication shows that this is the wrong comparison. `pooled` and `equal_user_mean` barely move; the decisive change only appears at:

- `equal_user_day_mean`
- `common_support_gap`

### Response

Accepted.

Any future version of the report should replace the old bullet list in the adjacent-outcome section with a table that includes:

- `pooled`
- `equal_user_mean`
- `equal_user_day_mean`
- `equal_user_day_mean.common_support_gap`

Without that table, the report foregrounds the least informative estimands and hides the estimand that actually changes the conclusion.

## Reviewer 4: The Autocorrelation Evidence Does Not Rescue The Section

### Concern

The original report also cited weighted lag autocorrelation as supporting evidence. Reviewers objected that this statistic was not subjected to the same averaging-level correction as the adjacent conditional probabilities.

So even if the autocorrelations remain numerically positive, they cannot now be used as the last unqualified support for the old mechanism story.

### Response

Accepted in substance.

The autocorrelation numbers should be downgraded to descriptive appendix-level statistics unless they are reanalyzed with comparable controls, such as:

- user-day-controlled dependence
- day-demeaned dependence
- explicitly matched common-support constructions

Until then, they should not appear in the main evidentiary chain.

## Reviewer 5: The Performance Facts Still Survive, But The Explanation Must Change

### Concern

All 5 reviewers converged on the same high-level verdict:

- the report is not invalidated wholesale
- the benchmark-performance evidence still matters
- the serial-dependence-style explanation is no longer publication-safe

The strongest surviving pieces of evidence in the original report are still:

- the near-match between `MOVING-AVG` and causal `WIN-200`
- the warm-start / prefix advantage of `MOVING-AVG`
- the complementarity between `MOVING-AVG` and `FSRS-6` in the hybrid result

### Response

Accepted.

The positive case should now rest on direct predictive benchmark behavior, not on the old adjacent-outcome bridge. The defensible thesis is:

1. `MOVING-AVG` is still well described as an intercept-only online logistic calibrator.
2. It is still genuinely strong on this benchmark.
3. Its advantage still appears to come from non-item, ordered-stream information that item-memory models do not explicitly encode.
4. But the earlier claim that this information is well captured by positive adjacent-outcome dependence is no longer supported.

## Consolidated Rebuttal Position

After the new SuperMemo averaging-level replication, I accept the central reviewer criticism:

- the original “Why recent outcomes are predictive” section should be withdrawn as mechanism evidence

I do **not** accept a stronger opposite claim such as:

- “there is no useful stream-level signal”
- “the whole MOVING-AVG story collapses”

That stronger conclusion still does not follow, because the predictive benchmark evidence remains:

- `WIN-200` nearly reproduces `MOVING-AVG`
- `MOVING-AVG` wins early and loses late
- `MOVING-AVG` and `FSRS-6` still combine well

So the updated position is narrower and more defensible:

1. The old adjacent-conditional-probability argument was statistically miscalibrated because it relied on pooled and weighted-per-user summaries.
2. On this dataset, equal-weight user-day analysis removes and slightly reverses that apparent positive gap.
3. Therefore the original report can no longer cite adjacent outcomes as robust evidence that recent outcomes are predictively useful in the required sense.
4. The surviving thesis is only that the benchmark rewards some ordered-stream, non-item signal, but the identity of that signal remains unresolved.
5. Any future revision of the report should move from “identified mechanism” language to “benchmark-consistent hypothesis” language.

## Recommended Revision Actions For The Original Report

1. Delete or fully rewrite [moving_avg_investigation.md](./moving_avg_investigation.md) lines 199-209 as an averaging-sensitivity correction result.
2. Rewrite the short answer and bottom line so they no longer assert robust positive recent-outcome predictiveness.
3. Re-anchor the main story in the surviving performance evidence: `WIN-200`, prefix behavior, and hybrid complementarity.
4. Remove weighted autocorrelation from the main causal or mechanistic argument unless it receives a matching robustness treatment.
5. Rewrite the open question about “temporal signal” to ask what ordered-stream or protocol-level non-item variation explains the surviving performance gains.

## Final Position

The first-round report was directionally right that `MOVING-AVG` is exploiting something other than item memory. The new replication shows that it was too confident about what that “something” was.

The publication-safe claim is now:

- `MOVING-AVG` is strong because the benchmark rewards low-capacity adaptation to ordered-stream, non-item variation

The publication-unsafe claim is:

- this has been demonstrated by positive adjacent conditional dependence of recent outcomes

That latter claim should be treated as retracted.
