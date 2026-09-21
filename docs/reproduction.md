# Reproduction guide and boundaries

## What can run without competition data

```bash
python -m pip install -e ".[audit]"
python -m unittest discover -s tests -v
python scripts/demo_public.py
python scripts/check_public_tree.py
```

These commands check loaders, label conventions, parsing, aggregate calculations and augmentation invariants using artificial examples. They do not remeasure competition accuracy.

## Obtain and keep data locally

Request data through the [WomenHelp competition](https://www.codabench.org/competitions/14614/) under the organizers' applicable terms. Do not upload narratives, record IDs, example-level predictions or generated derivatives. Keep datasets under ignored `data/`, checkpoints under `models/` or `outputs/`, and logs/results from runs under ignored directories.

The readers expect:

```text
data/extracted/
  subtask1/train.csv           # ID,TEXT,CLASS
  subtask1/devel.csv
  subtask2/train.csv           # ID,Text,L0,...,L6
  subtask2/devel.csv
  subtask2/SoftLabels/trainSoft.csv
  subtask2/SoftLabels/develSoft.csv
```

Pass `--data-dir` explicitly, or set `WOMENHELP_DATA_DIR`. Released files use 0–3 for severity and L0–L6 for the ordered type list in `labels.py`. The subtasks do not have exactly the same records; auxiliary supervision only uses matching labeled rows.

## Representative entry points

Install the relevant optional dependencies first. GPU libraries are hardware-specific. The project does not have a recovered lockfile that guarantees the historical training stack. The public source retains original script interfaces; inspect `--help` before launching a training job.

```bash
python -m pip install -e ".[classical]"
python scripts/run_classical_baseline_suite.py --data-dir data/extracted --experiment-name word_char_sgd --output-root outputs/classical
```

For the highlighted Severe augmentation recipe:

```bash
python -m pip install -e ".[encoders]"
python scripts/run_competitive_encoder_severity.py --data-dir data/extracted --model-name-or-path dccuchile/bert-base-spanish-wwm-uncased --output-dir outputs/augrep08 --max-seq-length 512 --loss-type focal --severity-head softmax --ordinal-distance-weight 0.05 --severity-class-weight balanced --type-label-source soft --type-loss-weight 0.3 --augment-mode report_style --augment-target-label Severe --augment-target-ratio 0.08 --augment-max-copies 2 --learning-rate 1e-5 --num-train-epochs 6 --optim adamw_torch --max-grad-norm 1.0 --precision fp32 --seed 3407
```

For S2, `run_subtask2_encoder_20260429.py` trains the dedicated encoder. Its default evaluation searches thresholds. The **submitted component** used fixed 0.45: distinguish fixed-threshold metrics from the tuned-threshold metrics. `materialize_subtask2_encoder_test_20260429.py` handles inference from a saved checkpoint. Inspect its explicit paths and threshold options.

`run_subtask1_encoder_oof_20260506.py` and `aggregate_subtask1_encoder_oof_20260506.py` implement fold training and aggregation. `materialize_hailmary_s1_foldbag_router_20260507.py` requires the anchor predictions, fold checkpoints and candidate scores. Its historical defaults name ignored experiment outputs; those artifacts are not included in this source release.

The generative route is preserved in `prompt_baseline_gemma4.py`, `finetune_gemma4_multitask.py`, and the SFT preparation scripts. Optional dependencies do not grant access or a license to gated models. Do not send report narratives to an external API without separate authorization.

## Exact official reproduction is not claimed

The source and aggregate evidence can be inspected, and the recipes can be rerun when data, models and compatible hardware are available. Exact reproduction of official submissions additionally needs the original checkpoints, ensemble probability sources and organizer scoring. Public source tests do not validate numerical equivalence to those missing artifacts.

Source release changes are limited to portable private-path defaults, restored standard TLS verification, documentation and data-free examples/tests. Scientific training and augmentation logic comes from the camera-ready project snapshot of June 22, 2026, retrieved from the original research workspace on September 21. The selected scientific source files also match the earlier GitHub audit branch before portability edits.
