# Augmented severity model in an intermediate ensemble

Added on September 22, 2026 to support the presentation's question-and-answer slide. This comparison uses the original 3,405 S1 development reports, including 154 Severe reports. It is separate from the final official test result.

| Development system | Macro-F1 | Severe F1 | Severe recall |
|---|---:|---:|---:|
| Baseline BETO | 0.548114 | 0.417808 | 0.396104 |
| BETO with Severe report-template augmentation (`augrep08`) | 0.555198 | 0.433460 | 0.370130 |
| Intermediate ensemble including augmented BETO, May 5 | 0.58053633915804 | 0.47368421052631576 | 0.4090909090909091 |

The ensemble's Severe recall was 40.91%, compared with 39.61% for the baseline and 37.01% for the individual augmented model. On 154 Severe reports, these recalls correspond to 63, 61 and 57 correct Severe predictions, respectively. Relative to the baseline, the ensemble gains approximately 1.30 recall percentage points; relative to augmented BETO alone, it gains 3.90 points.

## Which ensemble this describes

The archived May 5 report identifies `repro_diverse10_dw0.50_equal_ordmix`, with ten equally weighted sources and cumulative thresholds [0.50, 0.50, 0.50]. Sources include `augrep08`, other BETO augmentation variants, Electricidad, XLM-R, mmBERT and a previously combined source (`mean_top3`). Ten sources therefore does not imply ten independent models or ten folds.

The report explicitly states that the materialized candidate reproduced the development search score. This is a documentary verification of that archived experiment, not a new reconstruction performed for this presentation. The candidate was selected on development data. Its result describes the full ensemble and does not isolate the causal contribution of `augrep08`.

## Correction to the earlier discussion

The May 3 search candidate `diverse10_dw0.20_equal_ordmix` had a reported recall of 41.56% and macro-F1 of 0.57737. The same report subsequently documents a failure to reproduce its score during materialization, and the May 4 report kept its promotion blocked. The earlier verbal explanation omitted this qualification. The presentation therefore uses the later May 5 candidate with documented reconstruction fidelity, rather than using 41.56% as a verified reconstruction result.

Neither intermediate result is the final submitted system's Severe recall. The official final result remains 0.60798 S1 macro-F1, as documented in [results](results.md).

## Source trail

- Baseline and individual augmentation: archived `reporte_s1_mild_report_style_followup_20260503.md`, individual-results table, and `augrep08_metrics_summary.json`.
- May 3 reconstruction issue: the same follow-up report's materialization and operational-decision sections, followed by `reporte_s1_validacion_materializacion_20260504.md`.
- May 5 ensemble, source membership and reconstruction status: `reporte_s1_reproducible_sources_search_20260505.md`, sections on the best candidate, candidate sources, materialization and decision.
- Public aggregate values: [evidence.json](../results/evidence.json).

Only aggregate findings are published here. Original narratives and private experiment artifacts are excluded.
