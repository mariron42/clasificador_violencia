# How the final system was selected

The project compared classical TF-IDF models, generative LLM pilots and contextual encoders, then adapted the prediction outputs and training objectives to severity and violence types. The two tasks followed different experimental paths.

For S2, replacing the earlier component with dedicated BETO at a fixed 0.45 threshold improved official micro-F1 from 0.849374 to 0.870165 while leaving S1 unchanged. That became the retained S2 component in submission 703502 and in the final package.

For S1, several locally promising alternatives underperformed the Top10 anchor on official evaluation. The ordinal `random10_0008` candidate reached 0.591950 official macro-F1; the later `highpen` candidate reached 0.601431, versus 0.606777 for the anchor. `highpen` is different from the intermediate May 5 candidate discussed in [the ensemble comparison](augmentation_ensemble.md); no official result for that exact intermediate candidate was identified in the recovered records.

Severe report-template augmentation changed the individual BETO error profile: macro-F1 and Severe F1 improved, while Severe recall decreased. The augmented recipe was retained as a candidate source. A subsequent route trained five augmented BETO folds and five Electricidad folds. The final system accepted only selected transitions from this candidate and otherwise retained the anchor. The router was selected on development data, and official scores informed submission selection.

## Recomputed development results of the final system

The following values were recomputed on September 22 from the final package's saved `devel_verbose.csv` and checked against `packaging_summary.json`. They describe 3,405 original S1 development reports, including 154 Severe reports. See [aggregate values and confusion matrices](../results/final_development.json).

| System | Macro-F1 | Severe precision | Severe recall | Severe F1 |
|---|---:|---:|---:|---:|
| Top10 anchor | 0.558879 | 0.588785 | 0.409091 | 0.482759 |
| Augmented BETO + Electricidad foldbag | 0.564563 | 0.584906 | 0.402597 | 0.476923 |
| Anchor + final router | 0.566899 | 0.558333 | 0.435065 | 0.489051 |

On development, the router retained the anchor's 63 correct Severe predictions and added four. It also added nine Severe false positives. The increase in recall therefore comes with a precision decrease. These results do not isolate the causal contribution of augmentation and are not an independent evaluation of the router-selection procedure.

The saved predictions and packaging JSON agree on router development micro-F1 of 0.5882525697503671. An archived narrative report lists 0.588840; use the recomputed value. This correction does not change the development macro-F1 or the official test results.

## What the final test package establishes

The recovered final ZIP matches the archived SHA-256 `cea327e1adc00370a17c7d2444cdd94b39b05844bb4a14c880810552cf693a78`. Direct comparison with the anchor package gives:

- 177 Medium to Mild changes;
- 101 Mild to Medium changes;
- 12 High to Severe changes;
- no other S1 changes, and byte-identical S2 predictions.

All 116 anchor Severe predictions are retained, and twelve are added. For the same fixed gold labels and row alignment, the final Severe recall cannot be lower than the anchor's recall: the predicted Severe set only grows. Absolute official-test recall and the correctness of the twelve additions remain unknown because test gold is unavailable. This property does not guarantee precision, F1 or improvement in every ordinal error cost.

The official final scores remain S1 macro-F1 0.607976891066 and S2 micro-F1 0.870164998648. The modest final S1 gain evaluates the complete system, not augmentation alone.

## Historical distinctions

- The early failed LoRA adapter and later working LoRA pilots are different experiments. Later pilots produced answers but did not beat the selected classical reference on the common 133-report joint comparison.
- A search score, a successfully reconstructed package and an officially evaluated submission are three different evidence states.
- Some historical reports prioritized macro-F1 for both tasks. The final official S2 ranking used micro-F1; do not retroactively claim that all S2 experiment selection optimized the official ranking criterion.
- Soft-label S2 experiments occurred after the final submission. The annotation-consensus analysis was added for the September presentation. Neither was part of final S2 selection.
- Cross-institution evaluation and operational validation remain future work.

## Sources

Archived reports: `resultado_externo_envio_703502_20260429.md`, `reporte_zips_ordinales_s1_20260430.md`, `reporte_s1_mild_report_style_followup_20260503.md`, `reporte_s1_reproducible_sources_search_20260505.md`, `reporte_s1_drift_penalty_validacion_dura_20260506.md`, `reporte_hailmary_final_20260507.md`, and `leaderboard_final_wh2026.md`. Final artifact verification used the two original prediction ZIPs and the saved final development predictions. Only aggregate results are included in this public release.
