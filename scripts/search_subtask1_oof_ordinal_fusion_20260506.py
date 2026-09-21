from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_competitive_severity_ensemble import (  # noqa: E402
    BaseSpec,
    average_probabilities,
    build_base_specs,
    classwise_average_probabilities,
    ensure_probabilities,
    fit_full_base_probabilities,
    fit_meta_and_predict,
    fit_oof_base_probabilities,
    make_aux_records,
    meta_oof_probabilities,
    metrics_for_indices,
    metrics_for_probabilities,
    probabilities_to_indices,
    resolve_cv_folds,
    stack_feature_matrix,
)
from womenhelp_competition.data import load_subtask1, load_subtask2  # noqa: E402
from womenhelp_competition.labels import SUBTASK1_LABELS  # noqa: E402

DEFAULT_DATA_DIR = "data/official_dev"
SEED = 3407
NUM_CLASSES = len(SUBTASK1_LABELS)
THRESHOLD_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S1 OOF ordinal fusion search with devel audit")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "subtask1_oof_fusion" / "20260506_classic_ordinal"))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=SEED)
    parser.add_argument("--aux-type-model", choices=["logreg_word_char", "char_svm"], default="logreg_word_char")
    parser.add_argument("--max-weighted-combos", type=int, default=1200)
    parser.add_argument(
        "--spec-names",
        default="",
        help="Lista separada por comas para filtrar modelos base; vacio usa todos.",
    )
    return parser.parse_args()


def extra_metrics(gold: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    distances = np.abs(gold.astype(np.int64) - pred.astype(np.int64))
    severe_mask = gold == 3
    severe_recall = float(np.mean(pred[severe_mask] == 3)) if np.any(severe_mask) else 0.0
    mild_severe = float(np.mean(distances == 3))
    return {
        "ordinal_mae": float(distances.mean()),
        "catastrophic_rate": float(np.mean(distances >= 2)),
        "mild_severe_rate": mild_severe,
        "severe_recall": severe_recall,
    }


def pred_distribution(pred: np.ndarray) -> dict[str, int]:
    counter = Counter(int(value) for value in pred.tolist())
    return {SUBTASK1_LABELS[index]: int(counter.get(index, 0)) for index in range(NUM_CLASSES)}


def metric_bundle(gold: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    metrics = metrics_for_indices(gold, pred)
    severe = metrics["per_label"]["Severe"]
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "micro_f1": float(metrics["micro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "accuracy": float(metrics["accuracy"]),
        "severe_f1": float(severe["f1"]),
        "severe_recall": float(severe["recall"]),
        "per_label": metrics["per_label"],
        "pred_distribution": pred_distribution(pred),
        **extra_metrics(gold, pred),
    }


def tail_scores(probabilities: np.ndarray) -> np.ndarray:
    values = ensure_probabilities(probabilities)
    return np.column_stack(
        [
            values[:, 1:].sum(axis=1),
            values[:, 2:].sum(axis=1),
            values[:, 3],
        ]
    )


def class_from_tail(scores: np.ndarray, thresholds: tuple[float, float, float]) -> np.ndarray:
    threshold_array = np.asarray(thresholds, dtype=np.float64).reshape(1, 3)
    return (scores >= threshold_array).sum(axis=1).astype(np.int64)


def evaluate_probabilities(gold: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    pred = probabilities_to_indices(probabilities)
    return metric_bundle(gold, pred)


def search_ordinal_thresholds(gold: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    scores = tail_scores(probabilities)
    best: dict[str, Any] | None = None
    for thresholds in product(THRESHOLD_GRID, repeat=3):
        cast_thresholds = (float(thresholds[0]), float(thresholds[1]), float(thresholds[2]))
        pred = class_from_tail(scores, cast_thresholds)
        row = metric_bundle(gold, pred)
        row["thresholds"] = list(cast_thresholds)
        key = (
            row["macro_f1"],
            -row["catastrophic_rate"],
            row["severe_recall"],
            -row["ordinal_mae"],
            row["micro_f1"],
        )
        if best is None or key > best["_key"]:
            row["_key"] = key
            row["predictions"] = pred
            best = row
    assert best is not None
    return best


def apply_thresholds(gold: np.ndarray, probabilities: np.ndarray, thresholds: list[float]) -> dict[str, Any]:
    pred = class_from_tail(tail_scores(probabilities), (float(thresholds[0]), float(thresholds[1]), float(thresholds[2])))
    row = metric_bundle(gold, pred)
    row["thresholds"] = [float(value) for value in thresholds]
    return row


def compact_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro_f1": row["macro_f1"],
        "micro_f1": row["micro_f1"],
        "weighted_f1": row["weighted_f1"],
        "accuracy": row["accuracy"],
        "severe_f1": row["severe_f1"],
        "severe_recall": row["severe_recall"],
        "ordinal_mae": row["ordinal_mae"],
        "catastrophic_rate": row["catastrophic_rate"],
        "mild_severe_rate": row["mild_severe_rate"],
        "pred_distribution": row["pred_distribution"],
        "per_label": row["per_label"],
    }


def candidate_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    oof = row["oof_metrics"]
    return (
        float(oof["macro_f1"]),
        -float(oof["catastrophic_rate"]),
        float(oof["severe_recall"]),
        -float(oof["ordinal_mae"]),
        float(oof["micro_f1"]),
        float(row["devel_metrics"]["macro_f1"]),
    )


def add_candidate(
    rows: list[dict[str, Any]],
    *,
    name: str,
    family: str,
    components: list[str],
    train_gold: np.ndarray,
    devel_gold: np.ndarray,
    train_probabilities: np.ndarray,
    devel_probabilities: np.ndarray,
    weights: list[float] | None = None,
) -> None:
    raw_oof = evaluate_probabilities(train_gold, train_probabilities)
    raw_devel = evaluate_probabilities(devel_gold, devel_probabilities)
    rows.append(
        {
            "name": name,
            "family": family,
            "calibration": "argmax",
            "components": components,
            "weights": weights,
            "oof_metrics": raw_oof,
            "devel_metrics": raw_devel,
        }
    )

    tuned_oof = search_ordinal_thresholds(train_gold, train_probabilities)
    thresholds = tuned_oof["thresholds"]
    tuned_devel = apply_thresholds(devel_gold, devel_probabilities, thresholds)
    rows.append(
        {
            "name": f"{name}_ordcal",
            "family": family,
            "calibration": "oof_ordinal_thresholds",
            "components": components,
            "weights": weights,
            "thresholds": thresholds,
            "oof_metrics": compact_metrics(tuned_oof),
            "devel_metrics": tuned_devel,
        }
    )


def disagreement_score(probabilities: np.ndarray, selected: list[np.ndarray]) -> float:
    if not selected:
        return 0.0
    pred = probabilities_to_indices(probabilities)
    return float(np.mean([np.mean(pred != probabilities_to_indices(item)) for item in selected]))


def greedy_diverse_names(names: list[str], probability_map: dict[str, np.ndarray], oof_metrics: dict[str, Any], size: int) -> list[str]:
    selected: list[str] = []
    selected_probs: list[np.ndarray] = []
    pool = list(names)
    while pool and len(selected) < size:
        best_name = pool[0]
        best_score = -1e9
        for name in pool:
            metrics = oof_metrics[name]
            score = (
                float(metrics["macro_f1"])
                + 0.08 * float(metrics["per_label"]["Severe"]["recall"])
                - 0.12 * float(extra_metrics_cache(metrics).get("catastrophic_rate", 0.0))
                + 0.30 * disagreement_score(probability_map[name], selected_probs)
            )
            if score > best_score:
                best_score = score
                best_name = name
        selected.append(best_name)
        selected_probs.append(probability_map[best_name])
        pool.remove(best_name)
    return selected


def extra_metrics_cache(metrics: dict[str, Any]) -> dict[str, float]:
    # Base OOF metrics from run_competitive_severity_ensemble do not include ordinal extras.
    # The caller only uses this function as a harmless default in greedy selection.
    return metrics.get("_ordinal_extra", {})


def weight_grid(num_items: int) -> list[np.ndarray]:
    if num_items == 2:
        return [np.asarray([w, 1.0 - w], dtype=np.float64) for w in np.linspace(0.1, 0.9, 9)]
    if num_items == 3:
        weights = []
        for a in np.linspace(0.1, 0.8, 8):
            for b in np.linspace(0.1, 0.8, 8):
                c = 1.0 - a - b
                if c >= 0.1:
                    weights.append(np.asarray([a, b, c], dtype=np.float64))
        return weights
    if num_items == 4:
        weights = []
        values = np.linspace(0.1, 0.7, 7)
        for a in values:
            for b in values:
                for c in values:
                    d = 1.0 - a - b - c
                    if d >= 0.1:
                        weights.append(np.asarray([a, b, c, d], dtype=np.float64))
        return weights
    return [np.ones(num_items, dtype=np.float64) / num_items]


def build_report(output_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# S1 OOF ordinal fusion search",
        "",
        "Los candidatos se ordenan por train-OOF. Devel se reporta como auditoria y no como criterio de seleccion.",
        "",
        "## Baselines",
        "",
        f"- mejor envio externo conocido S1 macro: `{payload['known_external_best']['s1_macro_f1']}`",
        f"- optimized_top10 local devel macro: `{payload['known_external_best']['optimized_top10_local_devel_macro']}`",
        "",
        "## Top candidatos por OOF",
        "",
        "| rank | name | calibration | OOF macro | devel macro | OOF Severe F1 | devel Severe F1 | OOF catastrophic | devel catastrophic | thresholds | components |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for index, row in enumerate(payload["candidates"][:40], start=1):
        lines.append(
            f"| {index} | `{row['name']}` | `{row['calibration']}` | "
            f"{row['oof_metrics']['macro_f1']:.6f} | {row['devel_metrics']['macro_f1']:.6f} | "
            f"{row['oof_metrics']['severe_f1']:.6f} | {row['devel_metrics']['severe_f1']:.6f} | "
            f"{row['oof_metrics']['catastrophic_rate']:.6f} | {row['devel_metrics']['catastrophic_rate']:.6f} | "
            f"`{row.get('thresholds', '')}` | `{','.join(row['components'])}` |"
        )

    lines.extend(
        [
            "",
            "## Modelos base",
            "",
            "| modelo | OOF macro | devel macro | OOF Severe F1 | devel Severe F1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["base_rows"]:
        lines.append(
            f"| `{row['name']}` | {row['oof_metrics']['macro_f1']:.6f} | {row['devel_metrics']['macro_f1']:.6f} | "
            f"{row['oof_extra']['severe_f1']:.6f} | {row['devel_extra']['severe_f1']:.6f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_subtask1("train", args.data_dir)
    devel_records = load_subtask1("devel", args.data_dir)
    subtask2_train = load_subtask2("train", args.data_dir)
    train_ids = [record.record_id for record in train_records]
    train_texts = [record.text for record in train_records]
    devel_texts = [record.text for record in devel_records]
    train_gold = np.asarray([int(record.severity_id) for record in train_records], dtype=np.int64)
    devel_gold = np.asarray([int(record.severity_id) for record in devel_records], dtype=np.int64)

    specs: list[BaseSpec] = build_base_specs()
    if args.spec_names.strip():
        requested = {name.strip() for name in args.spec_names.split(",") if name.strip()}
        specs = [spec for spec in specs if spec.name in requested]
        missing = sorted(requested - {spec.name for spec in specs})
        if missing:
            raise ValueError(f"Specs no encontradas: {missing}")
    if len(specs) < 2:
        raise ValueError("Se requieren al menos dos specs para una fusion OOF.")
    cv_folds = resolve_cv_folds(train_gold, args.cv_folds)
    oof_probabilities = fit_oof_base_probabilities(
        specs,
        train_ids,
        train_texts,
        train_gold,
        subtask2_train,
        cv_folds,
        args.random_state,
        args.aux_type_model,
    )
    devel_probabilities = fit_full_base_probabilities(
        specs,
        train_texts,
        train_gold,
        devel_texts,
        subtask2_train,
        args.aux_type_model,
    )

    model_order = [spec.name for spec in specs]
    base_rows = []
    base_oof_metrics: dict[str, Any] = {}
    for name in model_order:
        oof_raw = evaluate_probabilities(train_gold, oof_probabilities[name])
        devel_raw = evaluate_probabilities(devel_gold, devel_probabilities[name])
        oof_for_weight = metrics_for_probabilities(train_gold, oof_probabilities[name])
        base_oof_metrics[name] = oof_for_weight
        base_oof_metrics[name]["_ordinal_extra"] = extra_metrics(train_gold, probabilities_to_indices(oof_probabilities[name]))
        base_rows.append(
            {
                "name": name,
                "oof_metrics": oof_for_weight,
                "devel_metrics": metrics_for_probabilities(devel_gold, devel_probabilities[name]),
                "oof_extra": oof_raw,
                "devel_extra": devel_raw,
            }
        )

    rows: list[dict[str, Any]] = []
    for name in model_order:
        add_candidate(
            rows,
            name=name,
            family="base",
            components=[name],
            train_gold=train_gold,
            devel_gold=devel_gold,
            train_probabilities=oof_probabilities[name],
            devel_probabilities=devel_probabilities[name],
        )

    def add_blend(name: str, component_names: list[str], weights: np.ndarray | None = None, family: str = "blend") -> None:
        train_probs = average_probabilities([oof_probabilities[item] for item in component_names], weights=weights)
        devel_probs = average_probabilities([devel_probabilities[item] for item in component_names], weights=weights)
        add_candidate(
            rows,
            name=name,
            family=family,
            components=component_names,
            weights=weights.tolist() if weights is not None else None,
            train_gold=train_gold,
            devel_gold=devel_gold,
            train_probabilities=train_probs,
            devel_probabilities=devel_probs,
        )

    add_blend("mean_all", model_order)
    macro_weights = np.asarray([base_oof_metrics[name]["macro_f1"] for name in model_order], dtype=np.float64)
    add_blend("macro_weighted_all", model_order, macro_weights)
    top_by_oof = [row["name"] for row in sorted(base_rows, key=lambda item: (-item["oof_extra"]["macro_f1"], item["name"]))]
    for size in [2, 3, 4, 5, 6]:
        add_blend(f"mean_top{size}_oof", top_by_oof[:size])
    diverse = greedy_diverse_names(model_order, oof_probabilities, base_oof_metrics, min(6, len(model_order)))
    for size in [3, 4, 5, 6]:
        if len(diverse) >= size:
            add_blend(f"mean_diverse{size}_oof", diverse[:size])

    classwise_weights = np.asarray(
        [
            [base_oof_metrics[name]["per_label"][label]["f1"] + 1e-6 for label in SUBTASK1_LABELS]
            for name in model_order
        ],
        dtype=np.float64,
    )
    train_classwise = classwise_average_probabilities([oof_probabilities[name] for name in model_order], classwise_weights)
    devel_classwise = classwise_average_probabilities([devel_probabilities[name] for name in model_order], classwise_weights)
    add_candidate(
        rows,
        name="classwise_f1_all",
        family="blend_classwise",
        components=model_order,
        weights=None,
        train_gold=train_gold,
        devel_gold=devel_gold,
        train_probabilities=train_classwise,
        devel_probabilities=devel_classwise,
    )

    weighted_count = 0
    for size in [2, 3, 4]:
        for combo in combinations(top_by_oof[:7], size):
            for weights in weight_grid(size):
                add_blend(
                    "grid_" + "_".join(combo),
                    list(combo),
                    weights,
                    family="grid_weighted",
                )
                weighted_count += 1
                if weighted_count >= args.max_weighted_combos:
                    break
            if weighted_count >= args.max_weighted_combos:
                break
        if weighted_count >= args.max_weighted_combos:
            break

    stack_train_features = stack_feature_matrix(oof_probabilities, model_order)
    stack_devel_features = stack_feature_matrix(devel_probabilities, model_order)
    for meta_name, kind, c_value, class_weight in [
        ("stack_logreg_balanced_c1", "logreg", 1.0, "balanced"),
        ("stack_logreg_balanced_c4", "logreg", 4.0, "balanced"),
        ("stack_ordinal_balanced_c1", "ordinal", 1.0, "balanced"),
        ("stack_ordinal_balanced_c4", "ordinal", 4.0, "balanced"),
    ]:
        train_probs = meta_oof_probabilities(
            stack_train_features,
            train_gold,
            kind=kind,
            c_value=c_value,
            class_weight=class_weight,
            cv_folds=cv_folds,
            random_state=args.random_state + 97,
        )
        devel_probs = fit_meta_and_predict(
            stack_train_features,
            train_gold,
            stack_devel_features,
            kind=kind,
            c_value=c_value,
            class_weight=class_weight,
        )
        add_candidate(
            rows,
            name=meta_name,
            family="stacking",
            components=model_order,
            weights=None,
            train_gold=train_gold,
            devel_gold=devel_gold,
            train_probabilities=train_probs,
            devel_probabilities=devel_probs,
        )

    rows.sort(key=candidate_key, reverse=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": perf_counter() - start,
        "data_dir": args.data_dir,
        "cv_folds": cv_folds,
        "random_state": args.random_state,
        "aux_type_model": args.aux_type_model,
        "spec_names": [spec.name for spec in specs],
        "train_rows": len(train_records),
        "devel_rows": len(devel_records),
        "known_external_best": {
            "submission": "703502 / outputs/official_submission/20260429_public_top10_s1_beto_encoder_s2_fixed045/predictions.zip",
            "s1_macro_f1": 0.6067770721254157,
            "s1_micro_f1": 0.6022754207158094,
            "optimized_top10_local_devel_macro": 0.5588786159649086,
        },
        "base_rows": base_rows,
        "candidates": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    build_report(output_dir, payload)
    print(json.dumps({"output_dir": str(output_dir), "top": rows[:8]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
