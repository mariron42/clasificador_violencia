from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression, SGDClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402
from sklearn.naive_bayes import ComplementNB  # noqa: E402
from sklearn.pipeline import FeatureUnion  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from womenhelp_competition.data import load_subtask1, load_subtask2, slice_records  # noqa: E402
from womenhelp_competition.labels import (  # noqa: E402
    SUBTASK1_ID_TO_NAME,
    SUBTASK1_LABELS,
    SUBTASK1_NAME_TO_ID,
    SUBTASK2_COLUMNS,
    SUBTASK2_LABELS,
)
from womenhelp_competition.metrics import compute_multiclass_metrics  # noqa: E402

SEED = 3407
NUM_CLASSES = len(SUBTASK1_LABELS)
NON_NA_LABELS = [label for label in SUBTASK2_LABELS if label != "N/A"]
NON_NA_COLUMNS = SUBTASK2_COLUMNS[: len(NON_NA_LABELS)]


@dataclass(frozen=True)
class BaseSpec:
    name: str
    feature_family: str
    classifier: str
    params: dict[str, Any]
    use_type_scores: bool


@dataclass
class MatrixBundle:
    train: Any
    valid: Any | None
    eval: Any | None


class OrdinalSeverityModel:
    def __init__(self, *, c_value: float, class_weight: str | None, max_iter: int = 1200):
        self.models = [
            LogisticRegression(
                C=c_value,
                class_weight=class_weight,
                max_iter=max_iter,
                solver="lbfgs",
                random_state=SEED,
            )
            for _ in range(NUM_CLASSES - 1)
        ]

    def fit(self, features, labels: np.ndarray) -> "OrdinalSeverityModel":
        for threshold, model in enumerate(self.models):
            binary_labels = (labels > threshold).astype(np.int32)
            model.fit(features, binary_labels)
        return self

    def predict_proba(self, features) -> np.ndarray:
        ge_probabilities = np.column_stack([model.predict_proba(features)[:, 1] for model in self.models])
        monotonic_ge = np.minimum.accumulate(ge_probabilities, axis=1)
        probabilities = np.zeros((features.shape[0], NUM_CLASSES), dtype=np.float64)
        probabilities[:, 0] = 1.0 - monotonic_ge[:, 0]
        probabilities[:, 1] = monotonic_ge[:, 0] - monotonic_ge[:, 1]
        probabilities[:, 2] = monotonic_ge[:, 1] - monotonic_ge[:, 2]
        probabilities[:, 3] = monotonic_ge[:, 2]
        np.clip(probabilities, 0.0, 1.0, out=probabilities)
        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return probabilities / row_sums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye un ensamble competitivo de severidad con stacking clásico")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--subtask1-train-split", default="train")
    parser.add_argument("--subtask1-eval-split", default="devel")
    parser.add_argument("--subtask2-train-split", default="train")
    parser.add_argument("--limit-subtask1-train", type=int, default=0)
    parser.add_argument("--limit-subtask1-eval", type=int, default=0)
    parser.add_argument("--limit-subtask2-train", type=int, default=0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=SEED)
    parser.add_argument(
        "--aux-type-model",
        choices=["logreg_word_char", "char_svm"],
        default="logreg_word_char",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "competitive_severity_ensemble" / "20260419"),
    )
    parser.add_argument(
        "--report-markdown",
        default=str(REPO_ROOT / "reports" / "baselines" / "competitive_severity_ensemble_20260419.md"),
    )
    parser.add_argument(
        "--report-json",
        default=str(REPO_ROOT / "reports" / "baselines" / "competitive_severity_ensemble_20260419.json"),
    )
    return parser.parse_args()


def maybe_sample_records(records: list[Any], limit: int, random_state: int) -> list[Any]:
    items = list(records)
    if limit <= 0 or limit >= len(items):
        return items
    rng = np.random.default_rng(random_state)
    selected_indices = np.sort(rng.choice(len(items), size=limit, replace=False))
    return [items[int(index)] for index in selected_indices]


def build_base_specs() -> list[BaseSpec]:
    return [
        BaseSpec(
            name="word_logreg",
            feature_family="word",
            classifier="logreg",
            params={"C": 2.0, "class_weight": "balanced", "solver": "saga"},
            use_type_scores=False,
        ),
        BaseSpec(
            name="word_svm",
            feature_family="word",
            classifier="linear_svc",
            params={"C": 0.5, "class_weight": "balanced"},
            use_type_scores=False,
        ),
        BaseSpec(
            name="char_svm",
            feature_family="char",
            classifier="linear_svc",
            params={"C": 0.5, "class_weight": "balanced"},
            use_type_scores=False,
        ),
        BaseSpec(
            name="word_cnb",
            feature_family="word",
            classifier="complement_nb",
            params={"alpha": 0.2},
            use_type_scores=False,
        ),
        BaseSpec(
            name="word_char_sgd",
            feature_family="word_char",
            classifier="sgd",
            params={"alpha": 3e-6, "class_weight": "balanced", "loss": "modified_huber"},
            use_type_scores=False,
        ),
        BaseSpec(
            name="word_char_logreg",
            feature_family="word_char",
            classifier="logreg",
            params={"C": 1.0, "class_weight": "balanced", "solver": "saga"},
            use_type_scores=False,
        ),
        BaseSpec(
            name="word_char_logreg_types",
            feature_family="word_char",
            classifier="logreg",
            params={"C": 4.0, "class_weight": "balanced", "solver": "lbfgs"},
            use_type_scores=True,
        ),
        BaseSpec(
            name="word_char_ordinal_logreg_types",
            feature_family="word_char",
            classifier="ordinal_logreg",
            params={"C": 4.0, "class_weight": "balanced"},
            use_type_scores=True,
        ),
    ]


def build_word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        norm="l2",
    )


def build_char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )


def build_union_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", build_word_vectorizer()),
            ("char", build_char_vectorizer()),
        ]
    )


def sigmoid_scores(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def writable_csr(matrix) -> Any:
    csr = matrix.tocsr(copy=True)
    csr.sort_indices()
    return csr


def build_feature_bundle(
    feature_family: str,
    train_texts: list[str],
    valid_texts: list[str] | None,
    eval_texts: list[str] | None,
) -> MatrixBundle:
    if feature_family == "word":
        vectorizer = build_word_vectorizer()
        train_matrix = writable_csr(vectorizer.fit_transform(train_texts))
        valid_matrix = writable_csr(vectorizer.transform(valid_texts)) if valid_texts is not None else None
        eval_matrix = writable_csr(vectorizer.transform(eval_texts)) if eval_texts is not None else None
        return MatrixBundle(train=train_matrix, valid=valid_matrix, eval=eval_matrix)
    if feature_family == "char":
        vectorizer = build_char_vectorizer()
        train_matrix = writable_csr(vectorizer.fit_transform(train_texts))
        valid_matrix = writable_csr(vectorizer.transform(valid_texts)) if valid_texts is not None else None
        eval_matrix = writable_csr(vectorizer.transform(eval_texts)) if eval_texts is not None else None
        return MatrixBundle(train=train_matrix, valid=valid_matrix, eval=eval_matrix)
    if feature_family == "word_char":
        word_vectorizer = build_word_vectorizer()
        char_vectorizer = build_char_vectorizer()
        train_word = word_vectorizer.fit_transform(train_texts)
        train_char = char_vectorizer.fit_transform(train_texts)
        train_matrix = writable_csr(sparse.hstack([train_word, train_char], format="csr"))
        valid_matrix = None
        if valid_texts is not None:
            valid_word = word_vectorizer.transform(valid_texts)
            valid_char = char_vectorizer.transform(valid_texts)
            valid_matrix = writable_csr(sparse.hstack([valid_word, valid_char], format="csr"))
        eval_matrix = None
        if eval_texts is not None:
            eval_word = word_vectorizer.transform(eval_texts)
            eval_char = char_vectorizer.transform(eval_texts)
            eval_matrix = writable_csr(sparse.hstack([eval_word, eval_char], format="csr"))
        return MatrixBundle(train=train_matrix, valid=valid_matrix, eval=eval_matrix)
    raise ValueError(f"feature_family no soportada: {feature_family}")


def build_feature_bundles(
    train_texts: list[str],
    valid_texts: list[str] | None,
    eval_texts: list[str] | None,
) -> dict[str, MatrixBundle]:
    families = sorted({spec.feature_family for spec in build_base_specs()})
    return {
        family: build_feature_bundle(family, train_texts, valid_texts, eval_texts)
        for family in families
    }


def fit_auxiliary_type_model(
    texts: list[str],
    label_matrix: np.ndarray,
    aux_type_model: str,
) -> tuple[Any, OneVsRestClassifier]:
    if aux_type_model == "char_svm":
        vectorizer = build_char_vectorizer()
        classifier = OneVsRestClassifier(
            LinearSVC(
                C=0.5,
                class_weight=None,
                dual="auto",
                random_state=SEED,
            )
        )
    else:
        vectorizer = build_union_vectorizer()
        classifier = OneVsRestClassifier(
            LogisticRegression(
                C=2.0,
                class_weight="balanced",
                max_iter=1200,
                solver="liblinear",
                random_state=SEED,
            )
        )
    features = vectorizer.fit_transform(texts)
    classifier.fit(features, label_matrix)
    return vectorizer, classifier


def build_type_score_features(
    texts: list[str],
    vectorizer,
    classifier: OneVsRestClassifier,
) -> np.ndarray:
    features = vectorizer.transform(texts)
    if hasattr(classifier, "predict_proba"):
        positive_probabilities = classifier.predict_proba(features)
    else:
        decision_scores = np.asarray(classifier.decision_function(features), dtype=np.float64)
        if decision_scores.ndim == 1:
            decision_scores = decision_scores[:, np.newaxis]
        positive_probabilities = sigmoid_scores(decision_scores)
    max_probability = positive_probabilities.max(axis=1, keepdims=True)
    mean_probability = positive_probabilities.mean(axis=1, keepdims=True)
    count_over_half = (positive_probabilities > 0.5).sum(axis=1, keepdims=True).astype(np.float64)
    na_proxy = 1.0 - max_probability
    return np.hstack([positive_probabilities, max_probability, mean_probability, count_over_half, na_proxy])


def compose_features(
    text_features,
    *,
    type_scores: np.ndarray | None,
    type_scaler: StandardScaler | None,
):
    blocks = [text_features]
    if type_scores is not None and type_scaler is not None:
        scaled_type_scores = type_scaler.transform(type_scores)
        blocks.append(sparse.csr_matrix(scaled_type_scores))
    return sparse.hstack(blocks, format="csr")


def build_estimator(spec: BaseSpec):
    if spec.classifier == "complement_nb":
        return ComplementNB(alpha=float(spec.params["alpha"]))
    if spec.classifier == "logreg":
        return LogisticRegression(
            C=float(spec.params["C"]),
            class_weight=spec.params["class_weight"],
            max_iter=1500,
            solver=str(spec.params["solver"]),
            random_state=SEED,
        )
    if spec.classifier == "linear_svc":
        return LinearSVC(
            C=float(spec.params["C"]),
            class_weight=spec.params["class_weight"],
            dual="auto",
            random_state=SEED,
        )
    if spec.classifier == "sgd":
        return SGDClassifier(
            loss=str(spec.params["loss"]),
            alpha=float(spec.params["alpha"]),
            class_weight=spec.params["class_weight"],
            max_iter=2000,
            tol=1e-3,
            random_state=SEED,
        )
    if spec.classifier == "ordinal_logreg":
        return OrdinalSeverityModel(
            c_value=float(spec.params["C"]),
            class_weight=spec.params["class_weight"],
        )
    raise ValueError(f"Clasificador no soportado: {spec.classifier}")


def softmax_rows(values: np.ndarray) -> np.ndarray:
    row_max = values.max(axis=1, keepdims=True)
    shifted = values - row_max
    exp_values = np.exp(shifted)
    row_sums = exp_values.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return exp_values / row_sums


def ensure_probabilities(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-9, None)
    row_sums = clipped.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return clipped / row_sums


def predict_probabilities(model, spec: BaseSpec, features) -> np.ndarray:
    if spec.classifier == "linear_svc":
        decision_scores = np.asarray(model.decision_function(features), dtype=np.float64)
        if decision_scores.ndim == 1:
            decision_scores = decision_scores[:, np.newaxis]
        return ensure_probabilities(softmax_rows(decision_scores))
    if hasattr(model, "predict_proba"):
        return ensure_probabilities(np.asarray(model.predict_proba(features), dtype=np.float64))
    raise ValueError(f"El modelo {spec.name} no expone predict_proba")


def probabilities_to_indices(probabilities: np.ndarray) -> np.ndarray:
    return np.asarray(probabilities).argmax(axis=1).astype(np.int32)


def metrics_for_indices(gold_indices: np.ndarray, pred_indices: np.ndarray) -> dict[str, Any]:
    gold_labels = [SUBTASK1_LABELS[int(index)] for index in gold_indices]
    pred_labels = [SUBTASK1_LABELS[int(index)] for index in pred_indices]
    return compute_multiclass_metrics(gold_labels, pred_labels, SUBTASK1_LABELS)


def metrics_for_probabilities(gold_indices: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    pred_indices = probabilities_to_indices(probabilities)
    return metrics_for_indices(gold_indices, pred_indices)


def ordinal_tail_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    return np.column_stack(
        [
            values[:, 1:].sum(axis=1),
            values[:, 2:].sum(axis=1),
            values[:, 3],
        ]
    )


def ordinal_threshold_predictions(probabilities: np.ndarray, thresholds: list[float]) -> np.ndarray:
    tail_probabilities = ordinal_tail_probabilities(probabilities)
    predictions = np.zeros(tail_probabilities.shape[0], dtype=np.int32)
    for threshold_index, threshold_value in enumerate(thresholds):
        predictions += (tail_probabilities[:, threshold_index] >= float(threshold_value)).astype(np.int32)
    return predictions


def metric_priority_tuple(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    severe_scores = metrics["per_label"].get("Severe", {})
    return (
        float(metrics["macro_f1"]),
        float(severe_scores.get("recall", 0.0)),
        float(metrics["accuracy"]),
        float(metrics["weighted_f1"]),
    )


def search_ordinal_thresholds(gold_indices: np.ndarray, probabilities: np.ndarray) -> tuple[list[float], np.ndarray, dict[str, Any]]:
    candidate_values = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    best_predictions = probabilities_to_indices(probabilities)
    best_metrics = metrics_for_indices(gold_indices, best_predictions)
    best_thresholds = [0.50, 0.50, 0.50]
    best_key = metric_priority_tuple(best_metrics)

    for threshold_0 in candidate_values:
        for threshold_1 in candidate_values:
            for threshold_2 in candidate_values:
                thresholds = [threshold_0, threshold_1, threshold_2]
                predictions = ordinal_threshold_predictions(probabilities, thresholds)
                metrics = metrics_for_indices(gold_indices, predictions)
                metric_key = metric_priority_tuple(metrics)
                if metric_key > best_key:
                    best_key = metric_key
                    best_thresholds = thresholds
                    best_predictions = predictions
                    best_metrics = metrics

    return best_thresholds, best_predictions, best_metrics


def resolve_cv_folds(labels: np.ndarray, requested_folds: int) -> int:
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    positive_counts = class_counts[class_counts > 0]
    if positive_counts.size == 0:
        raise RuntimeError("No hay etiquetas disponibles para construir folds estratificados")
    return max(2, min(requested_folds, int(positive_counts.min())))


def make_aux_records(records: list[Any], excluded_ids: set[str]) -> list[Any]:
    return [record for record in records if record.record_id not in excluded_ids]


def fit_oof_base_probabilities(
    specs: list[BaseSpec],
    train_ids: list[str],
    train_texts: list[str],
    train_labels: np.ndarray,
    subtask2_train: list[Any],
    cv_folds: int,
    random_state: int,
    aux_type_model: str,
) -> dict[str, np.ndarray]:
    oof_probabilities = {
        spec.name: np.zeros((len(train_texts), NUM_CLASSES), dtype=np.float64)
        for spec in specs
    }
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    use_type_scores = any(spec.use_type_scores for spec in specs)

    for fold_index, (fit_indices, valid_indices) in enumerate(splitter.split(train_texts, train_labels), start=1):
        fit_texts = [train_texts[index] for index in fit_indices]
        valid_texts = [train_texts[index] for index in valid_indices]
        fold_feature_bundles = build_feature_bundles(fit_texts, valid_texts, None)

        fit_type_scores = None
        valid_type_scores = None
        if use_type_scores:
            excluded_ids = {train_ids[index] for index in valid_indices}
            fold_aux_records = make_aux_records(subtask2_train, excluded_ids)
            if not fold_aux_records:
                raise RuntimeError(f"Fold {fold_index} se quedó sin datos auxiliares para subtask2")
            aux_texts = [record.text for record in fold_aux_records]
            aux_matrix = np.asarray(
                [[int(record.label_vector[column]) for column in NON_NA_COLUMNS] for record in fold_aux_records],
                dtype=np.int32,
            )
            aux_vectorizer, aux_classifier = fit_auxiliary_type_model(aux_texts, aux_matrix, aux_type_model)
            fit_type_scores = build_type_score_features(fit_texts, aux_vectorizer, aux_classifier)
            valid_type_scores = build_type_score_features(valid_texts, aux_vectorizer, aux_classifier)

        for spec in specs:
            fit_features = fold_feature_bundles[spec.feature_family].train
            valid_features = fold_feature_bundles[spec.feature_family].valid
            if spec.use_type_scores:
                type_scaler = StandardScaler()
                type_scaler.fit(fit_type_scores)
                fit_features = compose_features(
                    fit_features,
                    type_scores=fit_type_scores,
                    type_scaler=type_scaler,
                )
                valid_features = compose_features(
                    valid_features,
                    type_scores=valid_type_scores,
                    type_scaler=type_scaler,
                )
            model = build_estimator(spec)
            model.fit(fit_features, train_labels[fit_indices])
            oof_probabilities[spec.name][valid_indices] = predict_probabilities(model, spec, valid_features)

    return oof_probabilities


def fit_full_base_probabilities(
    specs: list[BaseSpec],
    train_texts: list[str],
    train_labels: np.ndarray,
    eval_texts: list[str],
    subtask2_train: list[Any],
    aux_type_model: str,
) -> dict[str, np.ndarray]:
    eval_probabilities: dict[str, np.ndarray] = {}
    feature_bundles = build_feature_bundles(train_texts, None, eval_texts)
    use_type_scores = any(spec.use_type_scores for spec in specs)

    train_type_scores = None
    eval_type_scores = None
    if use_type_scores:
        aux_texts = [record.text for record in subtask2_train]
        aux_matrix = np.asarray(
            [[int(record.label_vector[column]) for column in NON_NA_COLUMNS] for record in subtask2_train],
            dtype=np.int32,
        )
        aux_vectorizer, aux_classifier = fit_auxiliary_type_model(aux_texts, aux_matrix, aux_type_model)
        train_type_scores = build_type_score_features(train_texts, aux_vectorizer, aux_classifier)
        eval_type_scores = build_type_score_features(eval_texts, aux_vectorizer, aux_classifier)

    for spec in specs:
        train_features = feature_bundles[spec.feature_family].train
        eval_features = feature_bundles[spec.feature_family].eval
        if spec.use_type_scores:
            type_scaler = StandardScaler()
            type_scaler.fit(train_type_scores)
            train_features = compose_features(
                train_features,
                type_scores=train_type_scores,
                type_scaler=type_scaler,
            )
            eval_features = compose_features(
                eval_features,
                type_scores=eval_type_scores,
                type_scaler=type_scaler,
            )
        model = build_estimator(spec)
        model.fit(train_features, train_labels)
        eval_probabilities[spec.name] = predict_probabilities(model, spec, eval_features)

    return eval_probabilities


def stack_feature_matrix(probability_map: dict[str, np.ndarray], model_order: list[str]) -> np.ndarray:
    return np.hstack([probability_map[name] for name in model_order])


def average_probabilities(probability_list: list[np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    stacked = np.stack(probability_list, axis=0)
    if weights is None:
        return ensure_probabilities(stacked.mean(axis=0))
    normalized_weights = np.asarray(weights, dtype=np.float64)
    weight_sum = normalized_weights.sum()
    if weight_sum <= 0.0:
        normalized_weights = np.ones_like(normalized_weights) / float(len(normalized_weights))
    else:
        normalized_weights = normalized_weights / weight_sum
    combined = np.tensordot(normalized_weights, stacked, axes=(0, 0))
    return ensure_probabilities(combined)


def classwise_average_probabilities(
    probability_list: list[np.ndarray],
    class_weights: np.ndarray,
) -> np.ndarray:
    stacked = np.stack(probability_list, axis=0)
    weighted = stacked * class_weights[:, np.newaxis, :]
    denominators = class_weights.sum(axis=0, keepdims=True)
    denominators[denominators == 0.0] = 1.0
    combined = weighted.sum(axis=0) / denominators
    return ensure_probabilities(combined)


def build_meta_estimator(kind: str, *, c_value: float, class_weight: str | None):
    if kind == "logreg":
        return LogisticRegression(
            C=c_value,
            class_weight=class_weight,
            max_iter=2000,
            solver="lbfgs",
            random_state=SEED,
        )
    if kind == "ordinal":
        return OrdinalSeverityModel(c_value=c_value, class_weight=class_weight)
    raise ValueError(f"Meta-modelo no soportado: {kind}")


def meta_oof_probabilities(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    kind: str,
    c_value: float,
    class_weight: str | None,
    cv_folds: int,
    random_state: int,
) -> np.ndarray:
    oof_probabilities = np.zeros((train_features.shape[0], NUM_CLASSES), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    for fit_indices, valid_indices in splitter.split(train_features, train_labels):
        model = build_meta_estimator(kind, c_value=c_value, class_weight=class_weight)
        model.fit(train_features[fit_indices], train_labels[fit_indices])
        if kind == "logreg":
            valid_probabilities = model.predict_proba(train_features[valid_indices])
        else:
            valid_probabilities = model.predict_proba(train_features[valid_indices])
        oof_probabilities[valid_indices] = ensure_probabilities(valid_probabilities)
    return oof_probabilities


def fit_meta_and_predict(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    *,
    kind: str,
    c_value: float,
    class_weight: str | None,
) -> np.ndarray:
    model = build_meta_estimator(kind, c_value=c_value, class_weight=class_weight)
    model.fit(train_features, train_labels)
    return ensure_probabilities(model.predict_proba(eval_features))


def write_submission(path: Path, prediction_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for prediction_id in prediction_ids:
            writer.writerow([prediction_id])


def write_verbose_csv(
    path: Path,
    eval_records: list[Any],
    eval_probabilities: np.ndarray,
    prediction_indices: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "gold_class_id",
        "gold_class",
        "pred_class_id",
        "pred_class",
        "confidence",
        "margin",
        *[f"prob_{label.lower()}" for label in SUBTASK1_LABELS],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record, probabilities, pred_index in zip(eval_records, eval_probabilities, prediction_indices):
            sorted_probabilities = np.sort(probabilities)
            margin = float(sorted_probabilities[-1] - sorted_probabilities[-2]) if len(sorted_probabilities) > 1 else 0.0
            row: dict[str, Any] = {
                "record_id": record.record_id,
                "gold_class_id": record.severity_id,
                "gold_class": SUBTASK1_ID_TO_NAME[record.severity_id],
                "pred_class_id": str(pred_index),
                "pred_class": SUBTASK1_LABELS[int(pred_index)],
                "confidence": float(probabilities[int(pred_index)]),
                "margin": margin,
            }
            for label_index, label_name in enumerate(SUBTASK1_LABELS):
                row[f"prob_{label_name.lower()}"] = float(probabilities[label_index])
            writer.writerow(row)


def report_lines(payload: dict[str, Any]) -> list[str]:
    lines = [
        "# Ensamble competitivo de severidad",
        "",
        f"Generado: {payload['generated_at']}",
        "",
        "## Metodología",
        "",
        "- Se construyó un conjunto diverso de modelos clásicos fuertes de severidad con TF-IDF palabra, carácter y palabra+carácter.",
        f"- La señal auxiliar de tipos se generó con: {payload['aux_type_model']}.",
        "- Las variantes con señal auxiliar de tipos usan la subtarea 2 como fuente de puntajes densos, sin usar etiquetas duras de severidad.",
        "- El ensamble se entrena con predicciones fuera de pliegue para evitar mezclar entrenamiento y meta-modelado en la misma pasada.",
        "",
        "## Modelos base",
        "",
        "| modelo | usa tipos | oof_macro_f1 | devel_macro_f1 | devel_accuracy |",
        "|---|---|---:|---:|---:|",
    ]
    for row in payload["base_rows"]:
        lines.append(
            f"| {row['name']} | {'si' if row['use_type_scores'] else 'no'} | {row['oof_macro_f1']:.4f} | {row['devel_macro_f1']:.4f} | {row['devel_accuracy']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Variantes de ensamble",
            "",
            "| variante | tipo | oof_macro_f1 | devel_macro_f1 | devel_accuracy | devel_weighted_f1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["ensemble_rows"]:
        lines.append(
            f"| {row['name']} | {row['ensemble_type']} | {row['oof_macro_f1']:.4f} | {row['devel_macro_f1']:.4f} | {row['devel_accuracy']:.4f} | {row['devel_weighted_f1']:.4f} |"
        )

    best_base = payload["best_base"]
    best_ensemble = payload["best_ensemble"]
    lines.extend(
        [
            "",
            "## Lectura rápida",
            "",
            f"- Mejor modelo base: {best_base['name']} con macro-F1 devel {best_base['devel_metrics']['macro_f1']:.4f}",
            f"- Mejor ensamble: {best_ensemble['name']} con macro-F1 devel {best_ensemble['devel_metrics']['macro_f1']:.4f}",
            f"- Delta ensamble vs mejor base: {best_ensemble['devel_metrics']['macro_f1'] - best_base['devel_metrics']['macro_f1']:+.4f}",
            "",
            "## Mejor ensamble por etiqueta en devel",
            "",
            "| etiqueta | precision | recall | f1 | support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label_name in SUBTASK1_LABELS:
        scores = best_ensemble["devel_metrics"]["per_label"][label_name]
        lines.append(
            f"| {label_name} | {scores['precision']:.4f} | {scores['recall']:.4f} | {scores['f1']:.4f} | {scores['support']} |"
        )
    return lines


def main() -> None:
    args = parse_args()
    start_time = perf_counter()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = maybe_sample_records(
        load_subtask1(args.subtask1_train_split, args.data_dir),
        args.limit_subtask1_train,
        args.random_state,
    )
    eval_records = maybe_sample_records(
        load_subtask1(args.subtask1_eval_split, args.data_dir),
        args.limit_subtask1_eval,
        args.random_state + 1,
    )
    subtask2_train = maybe_sample_records(
        load_subtask2(args.subtask2_train_split, args.data_dir),
        args.limit_subtask2_train,
        args.random_state + 2,
    )

    train_ids = [record.record_id for record in train_records]
    train_texts = [record.text for record in train_records]
    train_labels = np.asarray([int(record.severity_id) for record in train_records], dtype=np.int32)
    eval_texts = [record.text for record in eval_records]
    eval_labels = np.asarray([int(record.severity_id) for record in eval_records], dtype=np.int32)

    specs = build_base_specs()
    cv_folds = resolve_cv_folds(train_labels, args.cv_folds)
    oof_probabilities = fit_oof_base_probabilities(
        specs,
        train_ids,
        train_texts,
        train_labels,
        subtask2_train,
        cv_folds,
        args.random_state,
        args.aux_type_model,
    )
    eval_probabilities = fit_full_base_probabilities(
        specs,
        train_texts,
        train_labels,
        eval_texts,
        subtask2_train,
        args.aux_type_model,
    )

    base_rows: list[dict[str, Any]] = []
    base_oof_metrics: dict[str, dict[str, Any]] = {}
    for spec in specs:
        oof_metrics = metrics_for_probabilities(train_labels, oof_probabilities[spec.name])
        devel_metrics = metrics_for_probabilities(eval_labels, eval_probabilities[spec.name])
        base_oof_metrics[spec.name] = oof_metrics
        base_rows.append(
            {
                "name": spec.name,
                "use_type_scores": spec.use_type_scores,
                "oof_metrics": oof_metrics,
                "devel_metrics": devel_metrics,
                "oof_macro_f1": oof_metrics["macro_f1"],
                "devel_macro_f1": devel_metrics["macro_f1"],
                "devel_accuracy": devel_metrics["accuracy"],
                "devel_weighted_f1": devel_metrics["weighted_f1"],
            }
        )

    model_order = [spec.name for spec in specs]
    probability_list_oof = [oof_probabilities[name] for name in model_order]
    probability_list_eval = [eval_probabilities[name] for name in model_order]
    stack_train_features = stack_feature_matrix(oof_probabilities, model_order)
    stack_eval_features = stack_feature_matrix(eval_probabilities, model_order)

    ensemble_rows: list[dict[str, Any]] = []

    def add_ensemble_row(
        name: str,
        ensemble_type: str,
        train_probs: np.ndarray,
        eval_probs: np.ndarray,
        components: list[str],
        calibration: dict[str, Any] | None = None,
    ) -> None:
        train_metrics = metrics_for_probabilities(train_labels, train_probs)
        devel_metrics = metrics_for_probabilities(eval_labels, eval_probs)
        eval_prediction_indices = probabilities_to_indices(eval_probs)
        ensemble_rows.append(
            {
                "name": name,
                "ensemble_type": ensemble_type,
                "components": components,
                "calibration": calibration,
                "oof_metrics": train_metrics,
                "devel_metrics": devel_metrics,
                "oof_macro_f1": train_metrics["macro_f1"],
                "devel_macro_f1": devel_metrics["macro_f1"],
                "devel_accuracy": devel_metrics["accuracy"],
                "devel_weighted_f1": devel_metrics["weighted_f1"],
                "eval_probabilities": eval_probs,
                "eval_prediction_indices": eval_prediction_indices,
            }
        )

        calibrated_thresholds, calibrated_train_predictions, calibrated_train_metrics = search_ordinal_thresholds(
            train_labels,
            train_probs,
        )
        calibrated_eval_predictions = ordinal_threshold_predictions(eval_probs, calibrated_thresholds)
        calibrated_devel_metrics = metrics_for_indices(eval_labels, calibrated_eval_predictions)
        ensemble_rows.append(
            {
                "name": f"{name}_ordcal",
                "ensemble_type": f"{ensemble_type}_ordinal_thresholds",
                "components": components,
                "calibration": {
                    "type": "ordinal_thresholds",
                    "thresholds": calibrated_thresholds,
                },
                "oof_metrics": calibrated_train_metrics,
                "devel_metrics": calibrated_devel_metrics,
                "oof_macro_f1": calibrated_train_metrics["macro_f1"],
                "devel_macro_f1": calibrated_devel_metrics["macro_f1"],
                "devel_accuracy": calibrated_devel_metrics["accuracy"],
                "devel_weighted_f1": calibrated_devel_metrics["weighted_f1"],
                "eval_probabilities": eval_probs,
                "eval_prediction_indices": calibrated_eval_predictions,
            }
        )

    add_ensemble_row(
        "mean_vote_all",
        "soft_vote",
        average_probabilities(probability_list_oof),
        average_probabilities(probability_list_eval),
        model_order,
    )

    macro_weights = np.asarray([base_oof_metrics[name]["macro_f1"] for name in model_order], dtype=np.float64)
    add_ensemble_row(
        "macro_weighted_vote",
        "soft_vote_weighted",
        average_probabilities(probability_list_oof, weights=macro_weights),
        average_probabilities(probability_list_eval, weights=macro_weights),
        model_order,
    )

    classwise_weights = np.asarray(
        [
            [base_oof_metrics[name]["per_label"][label_name]["f1"] + 1e-6 for label_name in SUBTASK1_LABELS]
            for name in model_order
        ],
        dtype=np.float64,
    )
    add_ensemble_row(
        "classwise_f1_vote",
        "soft_vote_classwise",
        classwise_average_probabilities(probability_list_oof, classwise_weights),
        classwise_average_probabilities(probability_list_eval, classwise_weights),
        model_order,
    )

    top_model_order = [
        row["name"]
        for row in sorted(base_rows, key=lambda row: (-row["oof_macro_f1"], row["name"]))[:4]
    ]
    add_ensemble_row(
        "mean_vote_top4",
        "soft_vote",
        average_probabilities([oof_probabilities[name] for name in top_model_order]),
        average_probabilities([eval_probabilities[name] for name in top_model_order]),
        top_model_order,
    )

    anchor_name = "word_char_ordinal_logreg_types"
    anchor_partners = [name for name in model_order if name != anchor_name]
    for partner_name in anchor_partners:
        pair_names = [anchor_name, partner_name]
        add_ensemble_row(
            f"anchor_pair_{partner_name}",
            "soft_vote_anchor_pair",
            average_probabilities([oof_probabilities[name] for name in pair_names]),
            average_probabilities([eval_probabilities[name] for name in pair_names]),
            pair_names,
        )

    anchor_support_names = [
        name
        for name in top_model_order
        if name != anchor_name
    ][:2]
    anchor_top3_names = [anchor_name, *anchor_support_names]
    add_ensemble_row(
        "anchor_mean_top3",
        "soft_vote_anchor",
        average_probabilities([oof_probabilities[name] for name in anchor_top3_names]),
        average_probabilities([eval_probabilities[name] for name in anchor_top3_names]),
        anchor_top3_names,
    )

    anchor_top3_weights = np.asarray(
        [base_oof_metrics[name]["macro_f1"] for name in anchor_top3_names],
        dtype=np.float64,
    )
    add_ensemble_row(
        "anchor_macro_top3",
        "soft_vote_anchor_weighted",
        average_probabilities(
            [oof_probabilities[name] for name in anchor_top3_names],
            weights=anchor_top3_weights,
        ),
        average_probabilities(
            [eval_probabilities[name] for name in anchor_top3_names],
            weights=anchor_top3_weights,
        ),
        anchor_top3_names,
    )

    meta_folds = resolve_cv_folds(train_labels, cv_folds)
    meta_configs = [
        {"name": "stack_logreg_balanced_c1", "kind": "logreg", "C": 1.0, "class_weight": "balanced"},
        {"name": "stack_logreg_balanced_c4", "kind": "logreg", "C": 4.0, "class_weight": "balanced"},
        {"name": "stack_ordinal_balanced_c1", "kind": "ordinal", "C": 1.0, "class_weight": "balanced"},
    ]
    for meta in meta_configs:
        train_probs = meta_oof_probabilities(
            stack_train_features,
            train_labels,
            kind=meta["kind"],
            c_value=float(meta["C"]),
            class_weight=meta["class_weight"],
            cv_folds=meta_folds,
            random_state=args.random_state + 97,
        )
        eval_probs = fit_meta_and_predict(
            stack_train_features,
            train_labels,
            stack_eval_features,
            kind=meta["kind"],
            c_value=float(meta["C"]),
            class_weight=meta["class_weight"],
        )
        add_ensemble_row(
            meta["name"],
            "stacking",
            train_probs,
            eval_probs,
            model_order,
        )

    ensemble_eval_probabilities = {
        row["name"]: row.pop("eval_probabilities")
        for row in ensemble_rows
    }
    ensemble_eval_prediction_indices = {
        row["name"]: row.pop("eval_prediction_indices")
        for row in ensemble_rows
    }

    sorted_base_rows = sorted(
        base_rows,
        key=lambda row: (-row["devel_macro_f1"], -row["devel_accuracy"], -row["devel_weighted_f1"]),
    )
    sorted_ensemble_rows = sorted(
        ensemble_rows,
        key=lambda row: (-row["devel_macro_f1"], -row["devel_accuracy"], -row["devel_weighted_f1"]),
    )
    best_base = sorted_base_rows[0]
    best_ensemble = sorted_ensemble_rows[0]

    best_dir = output_dir / best_ensemble["name"]
    best_dir.mkdir(parents=True, exist_ok=True)
    best_probabilities = ensemble_eval_probabilities[best_ensemble["name"]]
    prediction_indices = ensemble_eval_prediction_indices[best_ensemble["name"]]
    prediction_ids = [str(pred_index) for pred_index in prediction_indices.tolist()]
    write_submission(best_dir / "submission" / "subtask1.csv", prediction_ids)
    write_verbose_csv(best_dir / "subtask1_verbose.csv", eval_records, best_probabilities, prediction_indices)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": perf_counter() - start_time,
        "cv_folds": cv_folds,
        "aux_type_model": args.aux_type_model,
        "subtask1_train_rows": len(train_records),
        "subtask1_eval_rows": len(eval_records),
        "subtask2_aux_rows": len(subtask2_train),
        "base_rows": sorted_base_rows,
        "ensemble_rows": sorted_ensemble_rows,
        "best_base": best_base,
        "best_ensemble": {
            **best_ensemble,
            "output_dir": str(best_dir),
        },
    }

    report_json = Path(args.report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report_markdown = Path(args.report_markdown)
    report_markdown.parent.mkdir(parents=True, exist_ok=True)
    report_markdown.write_text("\n".join(report_lines(payload)) + "\n", encoding="utf-8")

    metrics_summary = {
        "model_name_or_path": best_ensemble["name"],
        "family": "competitive_severity_ensemble",
        "description": "Ensamble clásico con stacking y soft-voting alrededor de los mejores modelos de severidad.",
        "subtask1_split": args.subtask1_eval_split,
        "subtask1_records": len(eval_records),
        "subtask1_metrics": best_ensemble["devel_metrics"],
        "best_base": best_base,
        "selected_ensemble": payload["best_ensemble"],
        "runtime_seconds": payload["runtime_seconds"],
    }
    (best_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()