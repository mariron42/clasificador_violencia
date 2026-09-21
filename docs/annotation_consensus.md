# Annotation consensus

This new descriptive analysis, conducted in September 2026 after paper submission, reads the organizer's soft labels. It exports only counts and rates. Obtain the starting kit separately and run:

```sh
python scripts/analyze_annotations.py /path/to/startingkit.zip
```

In development, 877 of 1,212 reports (72.36%) have a divided vote on at least one of the six violence types. The corresponding training rate is 73.60% of 4,845 reports. N/A is excluded from this six-type indicator.

| Type | Divided decisions in development (%) |
|---|---:|
| Economic | 16.17 |
| Physical | 17.66 |
| Property-related | 23.76 |
| Psychological | 33.00 |
| Sexual | 6.44 |
| Vicarious | 20.71 |

The denominator is all development reports, including unanimous negatives. Rates depend on prevalence and do not directly rank annotation difficulty. The paper describes three human annotators but does not establish their profession. Their identities and individual votes are unavailable. These data cannot measure differences between specific social workers or disagreement about severity.

For each binary decision, the released proportion p is 0, 1/3, 2/3 or 1. Within-item population variance is p(1-p), which is zero for unanimity and 2/9 for divided votes. This is not a confidence interval or an estimate of between-worker variance.

Matching records by ID gives 75 training and 24 development decisions where the hard label differs from p > 0.66. The reason is unknown. Split-vote rates use soft labels directly, so these discrepancies do not change the reported rates. See [aggregate output](../results/annotation_consensus.json).

The augmented S1 recipe used auxiliary soft type labels where available. The final S2 model used hard labels. Later S2 soft-label experiments were conducted after submission. This analysis motivates future evaluation by consensus; it does not establish a causal explanation for model errors or an upper bound on F1.
