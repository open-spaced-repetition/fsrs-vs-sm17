# Rebuttal To 5 Internal Reviews

Date: 2026-04-12

This note records the main criticisms raised by 5 internal reviewer agents and how the research note was revised in response.

## Reviewer 1: Implementation Correctness

### Concern

- The draft cited conditional-success numbers that were not reproduced by the script.
- The draft's failure-side `delta_p` number did not match the code-backed calculation.

### Response

- Accepted.
- I added the conditional-success and lag-autocorrelation calculations to [research/moving_avg_analysis.py](./moving_avg_analysis.py).
- I corrected the local-update example in [research/moving_avg_investigation.md](./moving_avg_investigation.md) to use `delta_p ≈ -0.0325`.

## Reviewer 2: Statistics

### Concern

- The per-user constant baseline was non-causal and should not be treated as a fair competitor.
- Several differences were very small and reported without uncertainty quantification.

### Response

- Accepted.
- The per-user constant baseline is now explicitly labeled as an oracle diagnostic.
- I added a user-bootstrap section for the main weighted log-loss differences. The note now treats small gaps as descriptive rather than settled.

## Reviewer 3: Metric Design

### Concern

- The strongest claims about UM / UM+ vulnerability were too broad relative to the exact aggregation and binning used here.

### Response

- Accepted in spirit.
- I kept the empirical warning, but narrowed the claim to "under this exact binning and aggregation setup" instead of implying a universal indictment of UM / UM+.
- The script now computes UM+ aggregation in a way aligned with the repository's README-style pairwise aggregation.

## Reviewer 4: Causal / Fairness Interpretation

### Concern

- The earlier draft spoke too confidently about a "user-global temporal signal" without causally identifying whether that signal came from fatigue, deck mix, session effects, or something else.
- The `AVG -> ONLINE-AVG -> MOVING-AVG` story was phrased too cleanly for the available ablations.

### Response

- Accepted.
- I replaced the strongest "user-global" wording with "stream-level" / "ordered review stream" language.
- I now state explicitly that the analysis does not causally disentangle the source of the time-correlated signal.
- The protocol-asymmetry section now says the decomposition is "consistent with" the observed gap, not that it fully explains it.

## Reviewer 5: Claim Scope / Writing

### Concern

- Several claims were rhetorically stronger than the supporting evidence.
- The hybrid result was presented as if it uniquely identified a missing bias term.

### Response

- Accepted.
- The hybrid section now says the result demonstrates complementarity and motivates a hypothesis about a missing stream-level calibration signal, rather than claiming unique identification.
- The note now uses more bounded language throughout and adds an explicit "What This Does Not Mean" section.

## Final Position

After revision, the note supports the following narrower thesis:

1. `MOVING-AVG` is well described as an intercept-only online logistic calibrator.
2. It is competitive because the benchmark contains exploitable short-run signal in the ordered review stream.
3. That signal is complementary to item-level memory modeling, but its causal origin is not identified here.
4. The exact numerical ranking between nearby low-capacity baselines should be treated cautiously on a 19-user sample.

The current draft is substantially stronger than the initial version, but still best understood as a careful research note rather than a definitive final paper.
