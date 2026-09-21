from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for candidate in (SRC_DIR, SCRIPTS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from transformers import AutoTokenizer

from run_competitive_encoder_severity import MultitaskSeverityEncoder, maybe_pad_token, sanitize_text
from run_mega_ensemble import (
    optimize_weights as optimize_ensemble_weights,
    ordinal_threshold_ids as mega_ensemble_ordinal_threshold_ids,
)
from run_severity_ensemble_v2 import (
    build_all_feature_bundles,
    build_base_specs,
    build_estimator,
    build_type_scores,
    compose_features,
    fit_auxiliary_type_model,
    predict_probs,
)
from womenhelp_competition.data import load_subtask1, load_subtask2
from womenhelp_competition.hf_utils import configure_hf_backend
from womenhelp_competition.labels import SUBTASK2_COLUMNS, SUBTASK2_LABELS
from womenhelp_competition.metrics import compute_multiclass_metrics, compute_multilabel_metrics

SEED = 3407
SUBTASK1_LABEL_IDS = ["0", "1", "2", "3"]
NON_NA_LABELS = [label for label in SUBTASK2_LABELS if label != "N/A"]
NON_NA_COLUMNS = SUBTASK2_COLUMNS[: len(NON_NA_LABELS)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua en desarrollo y prepara submission oficial WomenHelp")
    parser.add_argument(
        "--dev-data-dir",
        default=str(Path("data/official_dev")),
    )
    parser.add_argument(
        "--test-data-dir",
        default=str(Path("data/official_test")),
    )
    parser.add_argument(
        "--severity-ensemble-dir",
        default=str(REPO_ROOT / "outputs" / "mega_ensemble" / "final_with_v2"),
    )
    parser.add_argument(
        "--severity-v2-dir",
        default=str(REPO_ROOT / "outputs" / "severity_ensemble_v2"),
    )
    parser.add_argument("--severity-variant", default="optimized_top10")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "official_submission" / "20260425_optimized_top10_charsvm"),
    )
    parser.add_argument("--subtask1-batch-size", type=int, default=8)
    parser.add_argument("--subtask2-ovr-jobs", type=int, default=1)
    parser.add_argument(
        "--subtask2-external-csv",
        default=None,
        help="CSV externo para subtask2 listo para empaquetar en submission",
    )
    parser.add_argument(
        "--subtask2-external-summary-json",
        default=None,
        help="Resumen JSON opcional del artefacto externo para arrastrar variante e indicadores de devel",
    )
    parser.add_argument(
        "--subtask2-variant-name",
        default=None,
        help="Nombre de variante para subtask2 cuando se usa --subtask2-external-csv",
    )
    parser.add_argument("--severity-optimizer-restarts", type=int, default=10)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--limit-subtask1-devel", type=int, default=0)
    parser.add_argument("--limit-subtask1-test", type=int, default=0)
    parser.add_argument("--limit-subtask2-devel", type=int, default=0)
    parser.add_argument("--limit-subtask2-test", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_output_component_path(relative_component: str) -> Path:
    candidates = [
        REPO_ROOT / "outputs" / relative_component,
        REPO_ROOT / "outputs" / "severity_ensemble_v2" / relative_component,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No se encontro el componente {relative_component} en outputs/")


def slice_items(items: list[Any], limit: int) -> list[Any]:
    if limit <= 0:
        return list(items)
    return list(items)[:limit]


def load_unlabeled_text_rows(path: Path, limit: int = 0) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            rows.append(row[0])
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def load_prediction_vectors_csv(path: Path) -> list[list[int]]:
    vectors: list[list[int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            vectors.append([int(value) for value in row])
    return vectors


def resolve_external_subtask2_metrics(summary_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary_payload is None:
        return None
    if "devel_metrics" in summary_payload:
        return summary_payload["devel_metrics"]
    best_result = summary_payload.get("best_result")
    if isinstance(best_result, dict) and "metrics" in best_result:
        return best_result["metrics"]
    if "ensemble_metrics" in summary_payload:
        return summary_payload["ensemble_metrics"]
    if "subtask2_metrics" in summary_payload:
        return summary_payload["subtask2_metrics"]
    if "subtask2_hybrid" in summary_payload:
        return summary_payload["subtask2_hybrid"].get("metrics")
    return None


def resolve_external_subtask2_variant(summary_payload: dict[str, Any] | None, variant_name: str | None) -> str:
    if variant_name:
        return variant_name
    if summary_payload is None:
        return "external_subtask2_csv"
    for key in ("name", "variant", "display_name"):
        value = summary_payload.get(key)
        if value:
            return str(value)
    return "external_subtask2_csv"


def build_external_subtask2_result(
    external_csv_path: Path,
    external_summary_path: Path | None,
    variant_name: str | None,
) -> dict[str, Any]:
    summary_payload = load_json(external_summary_path) if external_summary_path is not None else None
    predictions = load_prediction_vectors_csv(external_csv_path)
    return {
        "variant": resolve_external_subtask2_variant(summary_payload, variant_name),
        "variant_source": str(external_csv_path),
        "selected_trial": summary_payload.get("selected_trial") if summary_payload else None,
        "train_rows": summary_payload.get("train_rows") if summary_payload else None,
        "devel_rows": summary_payload.get("devel_rows") if summary_payload else None,
        "test_rows": len(predictions),
        "devel_metrics": resolve_external_subtask2_metrics(summary_payload),
        "test_predictions": predictions,
    }


def resolve_variant_components(
    ensemble_dir: Path,
    severity_v2_dir: Path,
    variant_name: str,
) -> tuple[list[Path], float, str, dict[str, Any] | None]:
    payload = load_json(ensemble_dir / "mega_ensemble_results.json")
    for row in payload["all_results"]:
        if row["name"] != variant_name:
            continue
        return (
            [resolve_output_component_path(component) for component in row["components"]],
            float(row["macro_f1"]),
            "mega_ensemble",
            row.get("calibration"),
        )

    matches = sorted(severity_v2_dir.glob(f"*/{variant_name}/metrics_summary.json"))
    if matches:
        best_match = max(
            matches,
            key=lambda path: float(load_json(path).get("subtask1_metrics", {}).get("macro_f1", -1.0)),
        )
        metrics_summary = load_json(best_match)
        selected_ensemble = metrics_summary.get("selected_ensemble") or {}
        return (
            [best_match.parent],
            float(metrics_summary["subtask1_metrics"]["macro_f1"]),
            "severity_ensemble_v2",
            selected_ensemble.get("calibration"),
        )

    raise RuntimeError(
        f"No se encontro la variante {variant_name} ni en {ensemble_dir / 'mega_ensemble_results.json'} "
        f"ni en {severity_v2_dir}"
    )


def severity_gold_ids(records: list[Any]) -> list[str]:
    return [str(int(record.severity_id)) for record in records]


def prediction_ids_to_strings(prediction_ids: np.ndarray) -> list[str]:
    return [str(int(value)) for value in prediction_ids]


def prediction_ids_from_probs(
    probabilities: np.ndarray,
    calibration: dict[str, Any] | None,
) -> np.ndarray:
    if calibration and calibration.get("type") == "ordinal_thresholds":
        thresholds = calibration.get("thresholds") or []
        if len(thresholds) != 3:
            raise ValueError(f"Calibracion ordinal invalida: {calibration!r}")
        return mega_ensemble_ordinal_threshold_ids(probabilities, thresholds)
    return probabilities.argmax(axis=1)


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-9, None)
    return clipped / clipped.sum(axis=1, keepdims=True).clip(1e-9)


def average_probabilities(
    probability_list: list[np.ndarray],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    stacked = np.stack(probability_list, axis=0)
    if weights is None:
        return normalize_probabilities(stacked.mean(axis=0))
    normalized_weights = np.asarray(weights, dtype=np.float64)
    normalized_weights = normalized_weights / normalized_weights.sum().clip(1e-9)
    return normalize_probabilities(np.tensordot(normalized_weights, stacked, axes=(0, 0)))


def load_component_summary(component_dir: Path) -> dict[str, Any]:
    return load_json(component_dir / "metrics_summary.json")


def is_v2_component_dir(component_dir: Path) -> bool:
    return component_dir.parent.parent.name == "severity_ensemble_v2"


def describe_v2_component(component_dir: Path) -> dict[str, Any]:
    metrics_summary = load_component_summary(component_dir)
    if not is_v2_component_dir(component_dir) and metrics_summary.get("family") != "severity_ensemble_v2":
        raise RuntimeError(f"{component_dir} no es un artefacto severity_ensemble_v2")

    selected_ensemble = metrics_summary.get("selected_ensemble")
    if selected_ensemble:
        ensemble_type = str(selected_ensemble.get("ensemble_type", ""))
        if ensemble_type not in {"soft_vote", "anchor_pair", "anchor_triple"}:
            raise RuntimeError(
                f"La reconstruccion directa de {component_dir.name} no soporta ensemble_type={ensemble_type}"
            )
        base_components = list(selected_ensemble.get("components") or [])
    elif component_dir.name.startswith("anchor_pair_"):
        ensemble_type = "anchor_pair"
        partner_component = component_dir.name.removeprefix("anchor_pair_")
        base_components = ["wc_ordinal_c8_types", partner_component]
    else:
        raise RuntimeError(f"{component_dir} no incluye metadata suficiente para reconstruccion")

    if not base_components:
        raise RuntimeError(f"{component_dir} no declara componentes base")

    saved_metrics = metrics_summary.get("subtask1_metrics") or metrics_summary.get("devel_metrics")
    if not saved_metrics:
        raise RuntimeError(f"{component_dir} no incluye metricas guardadas de devel")

    return {
        "component_dir": str(component_dir),
        "family": "severity_ensemble_v2",
        "ensemble_type": ensemble_type,
        "base_components": base_components,
        "saved_eval_macro_f1": float(saved_metrics["macro_f1"]),
    }


def fit_selected_v2_base_models(
    spec_names: list[str],
    dev_data_dir: Path,
    devel_records: list[Any],
    test_texts: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    spec_index = {spec.name: spec for spec in build_base_specs()}
    missing_specs = sorted(name for name in spec_names if name not in spec_index)
    if missing_specs:
        raise RuntimeError(f"No se encontraron specs base de severity_ensemble_v2: {missing_specs}")

    selected_specs = [spec_index[name] for name in spec_names]
    train_records = load_subtask1("train", dev_data_dir)
    subtask2_train_records = load_subtask2("train", dev_data_dir)

    train_texts = [record.text for record in train_records]
    train_labels = np.asarray([int(record.severity_id) for record in train_records], dtype=np.int32)
    devel_texts = [record.text for record in devel_records]
    devel_gold = severity_gold_ids(devel_records)

    feature_bundles = build_all_feature_bundles(train_texts, devel_texts, test_texts)
    use_type_scores = any(spec.use_type_scores for spec in selected_specs)

    train_type_scores = None
    devel_type_scores = None
    test_type_scores = None
    if use_type_scores:
        aux_texts = [record.text for record in subtask2_train_records]
        aux_labels = np.asarray(
            [[int(record.label_vector[column]) for column in NON_NA_COLUMNS] for record in subtask2_train_records],
            dtype=np.int32,
        )
        aux_vectorizer, aux_model = fit_auxiliary_type_model(aux_texts, aux_labels)
        train_type_scores = build_type_scores(train_texts, aux_vectorizer, aux_model)
        devel_type_scores = build_type_scores(devel_texts, aux_vectorizer, aux_model)
        test_type_scores = build_type_scores(test_texts, aux_vectorizer, aux_model)

    devel_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    base_summaries: dict[str, dict[str, Any]] = {}
    for spec in selected_specs:
        train_features = feature_bundles[spec.feature_family]["train"]
        devel_features = feature_bundles[spec.feature_family]["valid"]
        test_features = feature_bundles[spec.feature_family]["eval"]

        if spec.use_type_scores and train_type_scores is not None:
            scaler = StandardScaler()
            scaler.fit(train_type_scores)
            train_features = compose_features(train_features, type_scores=train_type_scores, type_scaler=scaler)
            devel_features = compose_features(devel_features, type_scores=devel_type_scores, type_scaler=scaler)
            test_features = compose_features(test_features, type_scores=test_type_scores, type_scaler=scaler)

        model = build_estimator(spec)
        model.fit(train_features, train_labels)

        devel_probs = predict_probs(model, spec, devel_features)
        test_probs = (
            predict_probs(model, spec, test_features)
            if test_features is not None
            else np.zeros((0, len(SUBTASK1_LABEL_IDS)), dtype=np.float64)
        )

        devel_probabilities[spec.name] = devel_probs
        test_probabilities[spec.name] = test_probs

        devel_metrics = compute_multiclass_metrics(
            devel_gold,
            prediction_ids_to_strings(devel_probs.argmax(axis=1)),
            SUBTASK1_LABEL_IDS,
        )
        base_summaries[spec.name] = {
            "family": "severity_ensemble_v2_base",
            "base_component": spec.name,
            "use_type_scores": bool(spec.use_type_scores),
            "recomputed_devel_macro_f1": float(devel_metrics["macro_f1"]),
        }

    return devel_probabilities, test_probabilities, base_summaries


def rebuild_v2_component_probabilities(
    component_dirs: list[Path],
    dev_data_dir: Path,
    devel_records: list[Any],
    test_texts: list[str],
) -> dict[Path, dict[str, Any]]:
    component_descriptions = {component_dir: describe_v2_component(component_dir) for component_dir in component_dirs}
    required_base_specs = sorted(
        {
            base_name
            for description in component_descriptions.values()
            for base_name in description["base_components"]
        }
    )

    devel_base_probs, test_base_probs, _ = fit_selected_v2_base_models(
        required_base_specs,
        dev_data_dir,
        devel_records,
        test_texts,
    )
    devel_gold = severity_gold_ids(devel_records)

    rebuilt_components: dict[Path, dict[str, Any]] = {}
    for component_dir, description in component_descriptions.items():
        devel_prob_list = [devel_base_probs[name] for name in description["base_components"]]
        test_prob_list = [test_base_probs[name] for name in description["base_components"]]

        devel_probs = average_probabilities(devel_prob_list)
        test_probs = average_probabilities(test_prob_list)
        devel_metrics = compute_multiclass_metrics(
            devel_gold,
            prediction_ids_to_strings(devel_probs.argmax(axis=1)),
            SUBTASK1_LABEL_IDS,
        )

        rebuilt_components[component_dir] = {
            "devel_probs": devel_probs,
            "test_probs": test_probs,
            "metadata": {
                **description,
                "recomputed_devel_macro_f1": float(devel_metrics["macro_f1"]),
            },
        }

    return rebuilt_components


def choose_severity_combination(
    variant_name: str,
    devel_component_probs: list[np.ndarray],
    test_component_probs: list[np.ndarray],
    component_summaries: list[dict[str, Any]],
    devel_gold: list[str],
    optimizer_restarts: int,
) -> tuple[np.ndarray, np.ndarray, str, list[dict[str, Any]]]:
    if variant_name.startswith("optimized_") and len(devel_component_probs) > 1:
        optimized_weights, _ = optimize_ensemble_weights(
            devel_component_probs,
            np.asarray([int(value) for value in devel_gold], dtype=np.int32),
            n_restarts=optimizer_restarts,
        )
        return (
            average_probabilities(devel_component_probs, optimized_weights),
            average_probabilities(test_component_probs, optimized_weights),
            "optimized_recomputed",
            [
                {"component_dir": summary["component_dir"], "weight": float(weight)}
                for summary, weight in zip(component_summaries, optimized_weights)
            ],
        )

    if (variant_name.startswith("macro_weighted") or variant_name.startswith("weighted_")) and len(component_summaries) > 1:
        saved_weights = np.asarray(
            [float(summary["saved_eval_macro_f1"]) for summary in component_summaries],
            dtype=np.float64,
        )
        saved_weights = saved_weights / saved_weights.sum().clip(1e-9)
        return (
            average_probabilities(devel_component_probs, saved_weights),
            average_probabilities(test_component_probs, saved_weights),
            "macro_weighted_saved_f1",
            [
                {"component_dir": summary["component_dir"], "weight": float(weight)}
                for summary, weight in zip(component_summaries, saved_weights)
            ],
        )

    return (
        average_probabilities(devel_component_probs),
        average_probabilities(test_component_probs),
        "mean",
        [],
    )


def build_severity_note(severity_summary: dict[str, Any]) -> str:
    if severity_summary["combination_method"] == "optimized_recomputed":
        return (
            "Se reconstruyeron al vuelo los componentes faltantes de severity_ensemble_v2 y se reoptimizó "
            "la mezcla en devel para habilitar la variante optimized_top10 sin depender de artefactos "
            "clasicos serializados."
        )
    if severity_summary["variant_source"] == "severity_ensemble_v2":
        return (
            "Se reconstruyó directamente una variante de severity_ensemble_v2 al vuelo, evitando la "
            "dependencia en modelos clasicos no serializados."
        )
    return (
        "La variante de severidad se reconstruyó combinando los componentes disponibles desde checkpoints "
        "guardados y/o reglas de reconstrucción clasica."
    )


def find_model_weights(checkpoint_dir: Path) -> Path:
    for candidate in ("model.safetensors", "pytorch_model.bin", "model.bin"):
        path = checkpoint_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"No encontre pesos de modelo en {checkpoint_dir}")


def load_checkpoint_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    weights_path = find_model_weights(checkpoint_dir)
    if weights_path.suffix == ".safetensors":
        return load_safetensors(str(weights_path))
    state_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        return state_dict["state_dict"]
    return state_dict


def component_metadata(component_dir: Path) -> dict[str, Any]:
    metrics_summary = load_json(component_dir / "metrics_summary.json")
    train_config = load_json(component_dir / "train_config.json")
    return {
        "component_dir": str(component_dir),
        "model_name_or_path": metrics_summary["model_name_or_path"],
        "attn_implementation": metrics_summary.get("attn_implementation", "eager"),
        "severity_head": metrics_summary.get("severity_head", train_config.get("severity_head", "softmax")),
        "loss_type": metrics_summary.get("loss_type", train_config.get("loss_type", "cross_entropy")),
        "focal_gamma": float(train_config.get("focal_gamma", 2.0)),
        "ordinal_distance_weight": float(
            metrics_summary.get("ordinal_distance_weight", train_config.get("ordinal_distance_weight", 0.0))
        ),
        "severity_class_weights": metrics_summary.get("severity_class_weights"),
        "type_loss_weight": float(metrics_summary.get("type_loss_weight", train_config.get("type_loss_weight", 0.0))),
        "type_pos_weight": metrics_summary.get("type_pos_weight"),
        "max_seq_length": int(train_config.get("max_seq_length", 384)),
        "best_checkpoint": metrics_summary["best_checkpoint"],
        "tokenizer_path": str(component_dir / "tokenizer"),
        "saved_eval_macro_f1": float(metrics_summary["eval_metrics"]["macro_f1"]),
    }


def load_component_model(component_dir: Path, device: torch.device) -> tuple[MultitaskSeverityEncoder, Any, dict[str, Any]]:
    metadata = component_metadata(component_dir)
    tokenizer = AutoTokenizer.from_pretrained(metadata["tokenizer_path"], trust_remote_code=True)
    maybe_pad_token(tokenizer)
    model = MultitaskSeverityEncoder(
        metadata["model_name_or_path"],
        attn_implementation=metadata["attn_implementation"],
        severity_head=metadata["severity_head"],
        loss_type=metadata["loss_type"],
        focal_gamma=metadata["focal_gamma"],
        ordinal_distance_weight=metadata["ordinal_distance_weight"],
        severity_class_weights=metadata["severity_class_weights"],
        type_loss_weight=metadata["type_loss_weight"],
        type_pos_weight=metadata["type_pos_weight"],
    )
    state_dict = load_checkpoint_state_dict(Path(metadata["best_checkpoint"]))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer, metadata


def predict_probs_with_loaded_model(
    model: MultitaskSeverityEncoder,
    tokenizer,
    texts: list[str],
    *,
    batch_size: int,
    max_seq_length: int,
    device: torch.device,
) -> np.ndarray:
    all_probs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = [sanitize_text(text) for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_seq_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            probs = torch.softmax(outputs["severity_logits"], dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 4), dtype=np.float64)


def predict_component_probs(
    component_dir: Path,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    model, tokenizer, metadata = load_component_model(component_dir, device)
    probabilities = predict_probs_with_loaded_model(
        model,
        tokenizer,
        texts,
        batch_size=batch_size,
        max_seq_length=int(metadata["max_seq_length"]),
        device=device,
    )
    return probabilities, metadata


def scores_to_full_vectors(scores: np.ndarray, threshold: float) -> tuple[np.ndarray, list[list[int]]]:
    score_array = np.asarray(scores)
    if score_array.ndim == 1:
        score_array = score_array[:, np.newaxis]

    full_vectors: list[list[int]] = []
    for row in score_array:
        positive = [1 if float(value) >= threshold else 0 for value in row.tolist()]
        full_vector = positive + [0]
        if sum(positive) == 0:
            full_vector[-1] = 1
        full_vectors.append(full_vector)
    return score_array, full_vectors


def score_dicts_from_scores(score_array: np.ndarray, prediction_vectors: list[list[int]]) -> list[dict[str, float]]:
    score_dicts: list[dict[str, float]] = []
    for row, vector in zip(score_array, prediction_vectors):
        payload = {label: float(value) for label, value in zip(NON_NA_LABELS, row.tolist())}
        payload["N/A"] = 1.0 if vector[-1] == 1 else 0.0
        score_dicts.append(payload)
    return score_dicts


def write_subtask1_csv(path: Path, prediction_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for prediction_id in prediction_ids:
            writer.writerow([prediction_id])


def write_subtask2_csv(path: Path, prediction_vectors: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for vector in prediction_vectors:
            writer.writerow(vector)


def write_subtask2_test_score_outputs(
    output_dir: Path,
    prediction_vectors: list[list[int]],
    score_dicts: list[dict[str, float]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "subtask2_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for index, (vector, score_dict) in enumerate(zip(prediction_vectors, score_dicts)):
            payload = {
                "record_id": str(index),
                "pred_class": None,
                "pred_types": [label for label, value in zip(SUBTASK2_LABELS, vector) if int(value) == 1],
                "positive_margins": score_dict,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    verbose_path = output_dir / "subtask2_verbose.csv"
    fieldnames = ["record_id", *SUBTASK2_COLUMNS, "pred_types", "positive_margins"]
    with verbose_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (vector, score_dict) in enumerate(zip(prediction_vectors, score_dicts)):
            payload: dict[str, Any] = {
                "record_id": str(index),
                "pred_types": ",".join([label for label, value in zip(SUBTASK2_LABELS, vector) if int(value) == 1]),
                "positive_margins": json.dumps(score_dict, ensure_ascii=False, sort_keys=True),
            }
            for column, value in zip(SUBTASK2_COLUMNS, vector):
                payload[column] = value
            writer.writerow(payload)

    return {
        "predictions_path": str(predictions_path),
        "verbose_path": str(verbose_path),
    }


def package_zip(output_path: Path, subtask1_path: Path, subtask2_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(subtask1_path, arcname="subtask1.csv")
        archive.write(subtask2_path, arcname="subtask2.csv")


def train_and_predict_char_svm(
    dev_data_dir: Path,
    test_data_dir: Path,
    *,
    devel_limit: int,
    test_limit: int,
    ovr_jobs: int,
) -> dict[str, Any]:
    train_records = load_subtask2("train", dev_data_dir)
    devel_records = slice_items(load_subtask2("devel", dev_data_dir), devel_limit)
    test_texts = load_unlabeled_text_rows(test_data_dir / "subtask2" / "test.csv", limit=test_limit)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
    )
    train_texts = [record.text for record in train_records]
    devel_texts = [record.text for record in devel_records]
    train_labels = [[record.label_vector[column] for column in NON_NA_COLUMNS] for record in train_records]
    devel_gold = [[record.label_vector[column] for column in SUBTASK2_COLUMNS] for record in devel_records]

    estimator = OneVsRestClassifier(
        LinearSVC(C=0.5, class_weight=None, random_state=SEED),
        n_jobs=ovr_jobs,
    )
    train_features = vectorizer.fit_transform(train_texts)
    estimator.fit(train_features, train_labels)

    devel_scores = estimator.decision_function(vectorizer.transform(devel_texts))
    _, devel_pred_vectors = scores_to_full_vectors(devel_scores, threshold=-0.25)
    devel_metrics = compute_multilabel_metrics(devel_gold, devel_pred_vectors, SUBTASK2_LABELS)

    test_scores = estimator.decision_function(vectorizer.transform(test_texts))
    test_score_array, test_pred_vectors = scores_to_full_vectors(test_scores, threshold=-0.25)
    test_score_dicts = score_dicts_from_scores(test_score_array, test_pred_vectors)

    return {
        "variant": "char_svm",
        "variant_source": "internal_char_svm",
        "selected_trial": {
            "classifier": "linear_svc",
            "score_mode": "decision",
            "C": 0.5,
            "class_weight": None,
            "threshold": -0.25,
        },
        "train_rows": len(train_records),
        "devel_rows": len(devel_records),
        "test_rows": len(test_texts),
        "devel_metrics": devel_metrics,
        "test_predictions": test_pred_vectors,
        "test_score_dicts": test_score_dicts,
    }


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    subtask2_devel_metrics = summary["subtask2"].get("devel_metrics")
    subtask2_macro_f1 = (
        f"{subtask2_devel_metrics['macro_f1']:.4f}"
        if subtask2_devel_metrics is not None and "macro_f1" in subtask2_devel_metrics
        else "n/a"
    )

    lines = [
        "# Preparacion de submission oficial WomenHelp",
        "",
        f"- severity_variant: {summary['severity']['variant']}",
        f"- severity_variant_source: {summary['severity']['variant_source']}",
        f"- severity_combination_method: {summary['severity']['combination_method']}",
        f"- severity_saved_devel_macro_f1: {summary['severity']['saved_devel_macro_f1']:.4f}",
        f"- severity_recomputed_devel_macro_f1: {summary['severity']['devel_metrics']['macro_f1']:.4f}",
        f"- severity_test_rows: {summary['severity']['test_rows']}",
        f"- subtask2_variant: {summary['subtask2']['variant']}",
        f"- subtask2_variant_source: {summary['subtask2'].get('variant_source', 'internal')}",
        f"- subtask2_recomputed_devel_macro_f1: {subtask2_macro_f1}",
        f"- subtask2_test_rows: {summary['subtask2']['test_rows']}",
        "",
        "## Severity components",
        "",
    ]
    for component in summary["severity"]["components"]:
        lines.append(
            "- "
            f"{component['component_dir']}: family={component.get('family', 'unknown')}, "
            f"saved_macro_f1={component['saved_eval_macro_f1']:.4f}, "
            f"recomputed_macro_f1={component['recomputed_devel_macro_f1']:.4f}"
        )
    if summary["severity"]["component_weights"]:
        lines.extend(["", "## Severity weights", ""])
        for row in summary["severity"]["component_weights"]:
            lines.append(f"- {row['component_dir']}: weight={row['weight']:.6f}")
    lines.extend(
        [
            "",
            f"Zip final: {summary['submission_zip']}",
            "",
            "## Nota",
            "",
            build_severity_note(summary["severity"]),
        ]
    )
    (output_dir / "evaluation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_hf_backend()

    dev_data_dir = Path(args.dev_data_dir)
    test_data_dir = Path(args.test_data_dir)
    ensemble_dir = Path(args.severity_ensemble_dir)
    severity_v2_dir = Path(args.severity_v2_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    severity_component_dirs, saved_variant_macro_f1, variant_source, saved_variant_calibration = resolve_variant_components(
        ensemble_dir,
        severity_v2_dir,
        args.severity_variant,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")

    devel_records = slice_items(load_subtask1("devel", dev_data_dir), args.limit_subtask1_devel)
    test_texts = load_unlabeled_text_rows(
        test_data_dir / "subtask1" / "test.csv",
        limit=args.limit_subtask1_test,
    )
    devel_texts = [record.text for record in devel_records]
    devel_gold = severity_gold_ids(devel_records)

    v2_component_dirs = [component_dir for component_dir in severity_component_dirs if is_v2_component_dir(component_dir)]
    rebuilt_v2_components = {}
    if v2_component_dirs:
        rebuilt_v2_components = rebuild_v2_component_probabilities(
            v2_component_dirs,
            dev_data_dir,
            devel_records,
            test_texts,
        )

    devel_component_probs: list[np.ndarray] = []
    test_component_probs: list[np.ndarray] = []
    severity_components_summary: list[dict[str, Any]] = []
    for component_dir in severity_component_dirs:
        if component_dir in rebuilt_v2_components:
            rebuilt = rebuilt_v2_components[component_dir]
            devel_probs = rebuilt["devel_probs"]
            test_probs = rebuilt["test_probs"]
            metadata = rebuilt["metadata"]
        else:
            model, tokenizer, metadata = load_component_model(component_dir, device)
            devel_probs = predict_probs_with_loaded_model(
                model,
                tokenizer,
                devel_texts,
                batch_size=args.subtask1_batch_size,
                max_seq_length=int(metadata["max_seq_length"]),
                device=device,
            )
            test_probs = predict_probs_with_loaded_model(
                model,
                tokenizer,
                test_texts,
                batch_size=args.subtask1_batch_size,
                max_seq_length=int(metadata["max_seq_length"]),
                device=device,
            )
            metadata = {
                **metadata,
                "family": load_component_summary(component_dir).get("family", "competitive_encoder_severity"),
            }

        devel_component_probs.append(devel_probs)
        test_component_probs.append(test_probs)

        recomputed_pred_ids = devel_probs.argmax(axis=1)
        recomputed_metrics = compute_multiclass_metrics(
            devel_gold,
            prediction_ids_to_strings(recomputed_pred_ids),
            SUBTASK1_LABEL_IDS,
        )
        severity_components_summary.append(
            {
                **metadata,
                "recomputed_devel_macro_f1": float(recomputed_metrics["macro_f1"]),
            }
        )

    severity_devel_probs, severity_test_probs, severity_combination_method, severity_component_weights = choose_severity_combination(
        args.severity_variant,
        devel_component_probs,
        test_component_probs,
        severity_components_summary,
        devel_gold,
        args.severity_optimizer_restarts,
    )
    severity_devel_pred_ids = prediction_ids_from_probs(severity_devel_probs, saved_variant_calibration)
    severity_test_pred_ids = prediction_ids_from_probs(severity_test_probs, saved_variant_calibration)
    severity_devel_metrics = compute_multiclass_metrics(
        devel_gold,
        prediction_ids_to_strings(severity_devel_pred_ids),
        SUBTASK1_LABEL_IDS,
    )

    if args.subtask2_external_csv:
        subtask2_result = build_external_subtask2_result(
            Path(args.subtask2_external_csv),
            Path(args.subtask2_external_summary_json) if args.subtask2_external_summary_json else None,
            args.subtask2_variant_name,
        )
    else:
        subtask2_result = train_and_predict_char_svm(
            dev_data_dir,
            test_data_dir,
            devel_limit=args.limit_subtask2_devel,
            test_limit=args.limit_subtask2_test,
            ovr_jobs=args.subtask2_ovr_jobs,
        )

    submission_dir = output_dir / "submission"
    subtask1_path = submission_dir / "subtask1.csv"
    subtask2_path = submission_dir / "subtask2.csv"
    write_subtask1_csv(subtask1_path, severity_test_pred_ids.tolist())
    write_subtask2_csv(subtask2_path, subtask2_result["test_predictions"])
    if subtask2_result.get("test_score_dicts") is not None:
        subtask2_result["test_outputs"] = write_subtask2_test_score_outputs(
            output_dir / "subtask2_test_scores",
            subtask2_result["test_predictions"],
            subtask2_result["test_score_dicts"],
        )

    submission_zip = output_dir / "predictions.zip"
    package_zip(submission_zip, subtask1_path, subtask2_path)

    summary = {
        "device": str(device),
        "severity": {
            "variant": args.severity_variant,
            "variant_source": variant_source,
            "combination_method": severity_combination_method,
            "saved_devel_macro_f1": saved_variant_macro_f1,
            "saved_calibration": saved_variant_calibration,
            "devel_rows": len(devel_records),
            "test_rows": len(test_texts),
            "devel_metrics": severity_devel_metrics,
            "components": severity_components_summary,
            "component_weights": severity_component_weights,
        },
        "subtask2": subtask2_result,
        "submission_zip": str(submission_zip),
    }
    write_summary(output_dir, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
