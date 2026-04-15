# Why Does MOVING-AVG Look So Strong?

Date: 2026-04-15

This note audits the current benchmark and documents a reproducible explanation for the strong `MOVING-AVG` results in this repository. The goal is not to dismiss the result, but to explain what benchmark behavior `MOVING-AVG` is exploiting and what this benchmark is actually rewarding.

Supporting files:

- [moving_avg_analysis.py](./moving_avg_analysis.py)
- [sm17_serial_dependence_averaging_levels.md](./sm17_serial_dependence_averaging_levels.md)

## Research Question

Why does `MOVING-AVG`, a model with no item-level memory state, rank near or above much more expressive spaced-repetition models on several metrics in this benchmark?

## Short Answer

`MOVING-AVG` is not discovering a superior memory law. It is an intercept-only online logistic model that adapts to the recent calibration level of the ordered review stream. In this benchmark, that kind of low-capacity adaptation is unusually strong because:

1. The task has a very high success-rate prior.
2. Reviews are evaluated in global chronological order, so online stream-level baselines can exploit ordered-stream, non-item variation.
3. The benchmark emphasizes calibration-oriented metrics, where low-variance baselines can look very strong.
4. `MOVING-AVG` has a strong small-data and warm-start advantage.

The result is real, but it is mostly a result about benchmark-time calibration under this protocol, not a result that `MOVING-AVG` has learned card memory better than `FSRS` or `SM17`.

One important correction relative to the earlier draft: the old adjacent-outcome argument is not robust. On the same SuperMemo benchmark, pooled and weighted-per-user adjacent conditional gaps stay positive, but equal-user-day averaging removes and slightly reverses that gap. So the publication-safe claim is only that `MOVING-AVG` exploits some ordered-stream, non-item signal under this protocol; the identity of that signal is still unresolved.

## Protocol Audit

### What `MOVING-AVG` actually is

The implementation in [script.py](../script.py) is:

```python
x = 1.2
w = 0.3
y_pred = sigmoid(x)
x += w * (y - y_pred)
```

Equivalently:

- `p_t = sigmoid(x_t)`
- `x_{t+1} = x_t + 0.3 * (y_t - p_t)`

This is stochastic gradient descent on Bernoulli log loss with a single intercept parameter. It has no item representation, no interval input, and no memory-state dynamics.

### What the benchmark is actually asking

After preprocessing, the benchmark sorts each user's reviews by `review_date`, then evaluates the probability assigned to each review outcome in that global chronological order.

This means a model can benefit from ordered-stream effects such as:

- temporary fatigue
- changing deck mix
- onboarding and warm-up phases
- session composition
- short-run changes in overall recall level

`MOVING-AVG` explicitly exploits this kind of non-item variation. The present analysis does not causally disentangle fatigue, deck mix, session effects, day composition, and other time-correlated mechanisms. The strongest supported claim is narrower: the benchmark rewards online adaptation to ordered-stream variation that item-state models do not explicitly encode.

### A subtle protocol asymmetry

The repository baseline called `AVG` is not a pure online average at the current review time. It writes the probability to `next_index` when the previous review of the same card is processed. For item-only models this makes no difference, but for stream-level baselines it can.

I therefore introduced a diagnostic baseline:

- `ONLINE-AVG`: `p_t = E[y | y_{<t}]` with the same `0.9 / 1` prior as repo `AVG`, but evaluated at the current review time.

This is consistent with part of the `AVG -> MOVING-AVG` gap, although it does not by itself isolate timing from recency weighting.

## Main Empirical Findings

### 1. The benchmark is close to a high-prior problem

From the existing `raw/` files:

- total reviews: `687,662`
- overall success rate: `0.8698`
- per-user success-rate mean: `0.8502`
- per-user success-rate std: `0.0756`

As an oracle diagnostic, a per-user constant predictor computed from the full stream mean is already extremely competitive:

| Model | Weighted LogLoss | Unweighted LogLoss |
| --- | ---: | ---: |
| Per-user constant mean | 0.3812 | 0.4027 |
| MOVING-AVG | 0.3785 | 0.4010 |
| FSRS-6 | 0.3665 | 0.4014 |

Interpretation:

- this baseline is not causal and should not be read as a deployable competitor
- unweighted, both `MOVING-AVG` and `FSRS-6` are only about `0.001-0.002` better than this oracle diagnostic
- weighted, `FSRS-6` pulls ahead once large collections dominate

This shows that, after conditioning on user identity, the remaining entropy of the benchmark is already fairly low.

### 2. `MOVING-AVG` is basically a recent-recall estimator

A simple causal trailing-window baseline nearly reproduces the published `MOVING-AVG` result:

| Model | Weighted LogLoss | Weighted Avg UM | Weighted Max UM+ |
| --- | ---: | ---: | ---: |
| MOVING-AVG | 0.3785 | 0.0426 | 0.0760 |
| WIN-200 | 0.3772 | 0.0431 | 0.0694 |
| WIN-50 | 0.3915 | 0.0422 | 0.0774 |

`WIN-200` uses only the last 200 outcomes of the same user, causally. This is strong evidence that `MOVING-AVG` behaves like a recent-recall estimator on the ordered review stream. It does **not** by itself identify whether the useful signal is positive serial dependence, day-level composition, or some other form of ordered-stream variation. A user-bootstrap over collections does not cleanly separate the two: the weighted log-loss difference `MOVING-AVG - WIN-200` is `0.0013` with a 95% bootstrap CI of `[-0.0030, 0.0035]`.

### 3. Part of the `AVG -> MOVING-AVG` gain is consistent with protocol, not magic

| Model | Weighted LogLoss | Unweighted LogLoss | Weighted Avg UM |
| --- | ---: | ---: | ---: |
| Repo AVG | 0.3846 | 0.4119 | 0.0598 |
| ONLINE-AVG | 0.3813 | 0.4051 | 0.0526 |
| MOVING-AVG | 0.3785 | 0.4010 | 0.0426 |

The improvement is consistent with two effects:

1. `AVG -> ONLINE-AVG`: using the right prediction time already helps a lot.
2. `ONLINE-AVG -> MOVING-AVG`: recency weighting helps further.

This is not a fully isolated causal decomposition, because `ONLINE-AVG` and `MOVING-AVG` differ in both update rule and prediction timing. Still, the result strongly suggests that the published `AVG` baseline is disadvantaged by protocol alignment.

### 4. `MOVING-AVG` is mainly a calibration model, not a discrimination model

From the README tables:

- weighted `AUC`: `MOVING-AVG = 0.597`, `FSRS-6 = 0.662`
- unweighted `AUC`: `MOVING-AVG = 0.583`, `FSRS-6 = 0.635`

So `MOVING-AVG` is much worse at separating recalled from forgotten reviews. Its strength is calibration under a high-prior environment.

This is also why a constant predictor can look strong under UM:

| Model | Weighted Avg UM |
| --- | ---: |
| CONST-0.87 | 0.0609 |
| FSRSv4 | 0.0617 |
| FSRSv3 | 0.0681 |
| FSRS-6-default | 0.0820 |

A flat baseline with no discrimination beats several real schedulers on average UM under this exact binning and aggregation setup. That is a metric-design warning.

### 5. `MOVING-AVG` wins early; `FSRS-6` wins late

Weighted prefix log loss:

| Segment | MOVING-AVG | FSRS-6 |
| --- | ---: | ---: |
| First 50 reviews | 0.4268 | 0.4434 |
| First 200 reviews | 0.3834 | 0.4045 |
| First 1000 reviews | 0.3915 | 0.4022 |
| Last 50% of each user stream | 0.4011 | 0.3901 |

Interpretation:

- `MOVING-AVG` has a strong small-data / warm-start advantage
- `FSRS-6` needs enough history before its item-level structure pays off

This matches the README pattern where `MOVING-AVG` ties or wins on unweighted averages, but loses on repetition-weighted averages.

### 6. `MOVING-AVG` carries predictive signal complementary to `FSRS-6`

I tuned a simple logit-space mixture on the first half of each user stream and evaluated on the second half:

- best mixture: `0.65 * logit(FSRS-6) + 0.35 * logit(MOVING-AVG)`
- second-half weighted log loss:
  - `MOVING-AVG`: `0.4011`
  - `FSRS-6`: `0.3897`
  - `Hybrid`: `0.3817`

If `MOVING-AVG` were just a noisier version of `FSRS-6`, the hybrid would not improve this much. The improvement shows complementarity between the two predictors. A plausible hypothesis is that `MOVING-AVG` contributes a stream-level calibration or composition signal that `FSRS-6` does not explicitly model, but this experiment does not uniquely identify that mechanism.

### 7. Small numeric gaps should be treated as descriptive, not settled

For the three key weighted log-loss gaps, a user-bootstrap over the 19 collections gives:

| Comparison | Point Estimate | 95% CI |
| --- | ---: | ---: |
| `FSRS-6 - MOVING-AVG` | `-0.0120` | `[-0.0255, 0.0170]` |
| `MOVING-AVG - ONLINE-AVG` | `-0.0028` | `[-0.0088, 0.0023]` |
| `MOVING-AVG - WIN-200` | `0.0013` | `[-0.0030, 0.0035]` |

So the qualitative pattern is stable, but the small numeric gaps should not be over-interpreted as definitive ranking statements across only 19 users.

## Mechanistic Interpretation

### Why the update rule fits this benchmark

At `p = 0.87`, the `MOVING-AVG` update behaves like:

- after a success: `delta_x = +0.039`, `delta_p ≈ +0.004`
- after a failure: `delta_x = -0.261`, `delta_p ≈ -0.0325`

So one failure moves the predictor much more than one success. In a high-retention stream, this is sensible:

- successes are common and weakly informative
- failures are rare and informative

This asymmetric responsiveness is one reason `MOVING-AVG` can react quickly when the ordered stream shifts away from the dominant success prior.

### Adjacent-outcome summaries are not robust to averaging level

The earlier draft used pooled and weighted-per-user adjacent conditional probabilities to argue that recent outcomes were predictively useful. That argument does not survive the new averaging-level replication on the same SuperMemo benchmark.

For `raw_long_term`:

| Level | Gap |
| --- | ---: |
| `pooled` | `+0.048294` |
| `equal_user_mean` | `+0.051205` |
| `equal_user_day_mean` | `-0.007840` |
| `equal_user_day_mean.common_support_gap` | `-0.029236` |

For `same_day_first_per_card`:

| Level | Gap |
| --- | ---: |
| `pooled` | `+0.051236` |
| `equal_user_mean` | `+0.047493` |
| `equal_user_day_mean` | `-0.008750` |
| `equal_user_day_mean.common_support_gap` | `-0.029523` |

Interpretation:

- `pooled` and `equal_user_mean` answer high-activity-weighted questions
- the decisive change happens when weighting is pushed down to `user-day`
- under that correction, the apparent positive adjacent-outcome gap disappears and slightly reverses

So the old section should not be read as evidence for a stable positive one-step predictive effect. At most, it showed that pooled and activity-weighted summaries can make adjacent outcomes look informative. That is not enough to identify the mechanism behind `MOVING-AVG`.

### What the surviving evidence still supports

Even after withdrawing the old adjacent-outcome argument, three facts still survive cleanly:

1. `WIN-200` nearly reproduces `MOVING-AVG`, so recent ordered-stream history really is useful under this benchmark.
2. `MOVING-AVG` wins early and then loses late, which is exactly what a low-capacity warm-start calibrator should do.
3. `MOVING-AVG` and `FSRS-6` combine well, so the signal is at least partly complementary to item-memory modeling.

The defensible interpretation is therefore narrower:

- the benchmark rewards adaptation to ordered-stream, non-item variation
- `MOVING-AVG` is an effective low-capacity estimator of that variation
- but the identity of that variation remains unresolved

## Synthesis

`MOVING-AVG` looks excellent because the benchmark is rewarding a different capability than "modeling card memory law":

1. estimate a strong stream-level recall prior
2. adapt quickly to ordered-stream, non-item variation under the evaluation protocol
3. stay well calibrated under calibration-oriented metrics
4. avoid overfitting in the low-data regime

That combination is enough to be highly competitive on this benchmark. What has changed after the new replication is not the benchmark-performance fact; it is the level of confidence we should assign to one particular mechanism story.

## What This Does Not Mean

These results do **not** imply that `MOVING-AVG` is a better scheduler than `FSRS-6` or `SM17`.

This benchmark measures predictive accuracy of recall probability at review time. It does not directly measure the counterfactual quality of the intervals that each algorithm would choose under deployment.

These results also do **not** imply that the useful signal has been identified as positive adjacent-outcome dependence. The current evidence only supports a weaker statement: some ordered-stream, non-item information is useful under this protocol.

`MOVING-AVG` is therefore a strong benchmark participant, not a plausible standalone replacement for an item-level scheduler.

## Practical Recommendations

1. Add `ONLINE-AVG` and `WIN-k` baselines to the benchmark. Without them, `MOVING-AVG` looks more mysterious than it is.
2. Report both weighted and unweighted metrics, but interpret them differently. Unweighted metrics exaggerate small-data advantages.
3. Treat UM results with care. A constant baseline already does surprisingly well.
4. If the benchmark reports adjacent-outcome or serial-dependence summaries, report them at multiple averaging levels, including `equal_user_day_mean` and common-support diagnostics.
5. Evaluate hybrids such as `FSRS + stream-level calibration bias`. The second-half mixture result suggests clear room for improvement.
6. Separate two questions:
   - item-memory modeling quality
   - stream-level calibration / composition quality

## Open Questions

1. What ordered-stream or protocol-level non-item variation actually explains the `WIN-200` and hybrid gains?
2. Would user-day-controlled or day-demeaned dependence diagnostics recover any robust short-lag signal after the averaging correction?
3. Would the ranking change under a deployment-faithful scheduling simulation instead of pure next-outcome prediction?
4. How robust are these conclusions on additional collections outside the current 19-user sample?

## Bottom Line

`MOVING-AVG` performs so well because the benchmark rewards low-capacity adaptation to ordered-stream, non-item variation, and `MOVING-AVG` is an extremely effective estimator of that variation under the current protocol. Its success is mostly about benchmark-time stream-level calibration or composition effects, not about discovering a better spaced-repetition memory model.

The earlier claim that this had been established by positive adjacent conditional dependence of recent outcomes should be treated as withdrawn.
