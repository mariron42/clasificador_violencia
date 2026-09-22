# Aggregate results and evidence

Metrics below retain their evaluation protocol. These are archived project results, not new training runs. All differences compare complete recipes and are not architecture-only causal effects.

## Classical models and Transformer encoders

| Evaluation | Classical reference | Encoder recipe | Absolute difference |
|---|---:|---:|---:|
| S1 development macro-F1 | Early ordinal TF-IDF: 0.4974 | BETO focal: 0.5429 | +0.0455 |
| S2 development macro-F1 | Character SVM: 0.7642 | BETO fixed 0.45: 0.819845 | +0.055645 |
| S2 development micro-F1 | Character SVM: 0.8449 | BETO fixed 0.45: 0.868453 | +0.023553 |

The S1 classical value 0.4974 is the **early reference discussed in paper section 3.2**, not the best later classical ensemble. Later project summaries report classical ensembles at 0.5047 and 0.5215 development macro-F1. They must not disappear from a claim about the whole classical route. Likewise, the final Top10 ensemble includes classical components; their marginal contribution needs a with/without ablation.

Source evidence: corrected system-description paper (June 2026), section Classical TF-IDF Models and representative encoder table; original S2 encoder report (2026-04-29) and project snapshot documenting fixed045. Aggregate values are preserved in `results/evidence.json`.

## Common 133-record comparison of classical and generative models

This subset caps exact combinations of severity and type labels; it is not equally distributed across every class. All systems below use the same evaluation IDs. LoRA pilots used 625 training examples and 80 steps; training budgets across all families were not equalized.

| Model | Joint exact match | S2 macro-F1 |
|---|---:|---:|
| Word-character TF-IDF + SGD | 0.2331 | 0.7602 |
| Word-character TF-IDF + logistic regression | 0.2256 | 0.7917 |
| Gemma E4B prompting | 0.1654 | 0.7318 |
| Gemma E4B LoRA | 0.1955 | 0.6633 |
| Qwen3.5-4B prompting | 0.1053 | 0.6373 |

Joint exact match requires both severity and the complete type set to be correct. These values cannot be directly compared with full-development macro-F1 values above. Source: strict balanced133 project report and final model matrix, 2026-04-19.

## Severe augmentation on the same 3,405 development reports

| Metric | Baseline | Augmented | Change × 100 |
|---|---:|---:|---:|
| Macro-F1 | 0.548114 | 0.555198 | +0.7084 |
| Severe F1 | 0.417808 | 0.433460 | +1.5652 |
| Severe recall | 0.396104 | 0.370130 | -2.5974 |

Training: 616 Severe / 13,630 became 1,132 / 14,146 after 516 report-template variants. Development was not augmented. The displayed chart deltas are calculated from the six-decimal archived summary. No confidence intervals are claimed. Source: augrep08 metrics summary and Mild report-style follow-up, 2026-05-03.

## Intermediate ensemble with augmented BETO

An intermediate development ensemble including augmented BETO reached 0.580536 macro-F1, 0.473684 Severe F1 and 0.409091 Severe recall. See [the May 5 ensemble comparison](augmentation_ensemble.md) for its source membership, reconstruction status and distinction from final official test results.

## Official test

See [final selection](final_selection.md) for the distinction between intermediate development candidates and the submitted router, including newly recomputed final-development recall and direct verification that the final test ZIP retains every anchor Severe prediction.

| Measure | Previous anchor | Final |
|---|---:|---:|
| S1 macro-F1 | 0.6067770721254157 | 0.607976891066 |
| S2 macro-F1 | 0.808022474168 | 0.808022474168 |
| S2 micro-F1 | 0.870164998648 | 0.870164998648 |

Final rankings: S1 third of 16, S2 sixth of 15. The official ranking used macro-F1 for S1 and micro-F1 for S2. Router changes: 177 Medium→Mild, 101 Mild→Medium, 12 High→Severe; 290 of 4,219 in total. The test scores informed selection. Source: corrected paper's official results and delta tables, based on archived [Codabench 14614](https://www.codabench.org/competitions/14614/) results.

## New annotation analysis

See [annotation_consensus.md](annotation_consensus.md). This descriptive analysis was performed after paper submission and does not establish that annotation disagreement caused any model error.
