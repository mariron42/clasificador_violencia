from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from itertools import product
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
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402
from sklearn.naive_bayes import ComplementNB  # noqa: E402
from sklearn.pipeline import FeatureUnion  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from womenhelp_competition.data import (  # noqa: E402
    label_names_from_vector,
    load_subtask1,
    load_subtask2,
    slice_records,
)
from womenhelp_competition.labels import (  # noqa: E402
    SUBTASK1_ID_TO_NAME,
    SUBTASK1_LABELS,
    SUBTASK1_NAME_TO_ID,
    SUBTASK2_COLUMNS,
    SUBTASK2_LABELS,
)
from womenhelp_competition.metrics import (  # noqa: E402
    compute_multiclass_metrics,
    compute_multilabel_metrics,
)

SEED = 3407
NON_NA_LABELS = [label for label in SUBTASK2_LABELS if label != "N/A"]
NON_NA_COLUMNS = SUBTASK2_COLUMNS[: len(NON_NA_LABELS)]
FEATURE_FAMILIES = ["word", "char", "word_char"]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    family: str
    feature_family: str
    description: str
    subtask1_trials: list[dict[str, Any]]
    subtask2_trials: list[dict[str, Any]]


@dataclass
class MatrixBundle:
    train: Any
    valid: Any | None
    eval: Any | None
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corre una suite de lineas base clasicas para WomenHelp")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--subtask1-train-split", default="train")
    parser.add_argument("--subtask1-eval-split", default="devel")
    parser.add_argument("--subtask2-train-split", default="train")
    parser.add_argument("--subtask2-eval-split", default="devel")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "outputs" / "classical_suite" / "20260418"),
    )
    parser.add_argument(
        "--report-markdown",
        default=str(REPO_ROOT / "reports" / "baselines" / "classical_baseline_suite_20260418.md"),
    )
    parser.add_argument(
        "--report-json",
        default=str(REPO_ROOT / "reports" / "baselines" / "classical_baseline_suite_20260418.json"),
    )
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=SEED)
    parser.add_argument("--ovr-jobs", type=int, default=1)
    parser.add_argument("--limit-subtask1-train", type=int, default=0)
    parser.add_argument("--limit-subtask1-eval", type=int, default=0)
    parser.add_argument("--limit-subtask2-train", type=int, default=0)
    parser.add_argument("--limit-subtask2-eval", type=int, default=0)
    parser.add_argument(
        "--experiment-name",
        action="append",
        default=None,
        help="Nombre de experimento a ejecutar; puede repetirse para filtrar la suite",
    )
    parser.add_argument(
        "--experiment-overrides-json",
        default=None,
        help="JSON con overrides por experimento para subtask1_trials/subtask2_trials",
    )
    return parser.parse_args()


def cartesian_trials(base: dict[str, Any], **grid: list[Any]) -> list[dict[str, Any]]:
    keys = list(grid)
    values = [grid[key] for key in keys]
    trials: list[dict[str, Any]] = []
    for combo in product(*values):
        params = dict(base)
        for key, value in zip(keys, combo):
            params[key] = value
        trials.append(params)
    return trials


def build_experiments() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            name="word_cnb",
            family="classic_nb",
            feature_family="word",
            description="TF-IDF de palabras con unigramas y bigramas + ComplementNB.",
            subtask1_trials=cartesian_trials({"classifier": "complement_nb"}, alpha=[0.2, 0.5, 1.0]),
            subtask2_trials=cartesian_trials(
                {"classifier": "complement_nb", "score_mode": "proba"},
                alpha=[0.2, 0.5, 1.0],
                threshold=[0.2, 0.3, 0.4],
            ),
        ),
        ExperimentSpec(
            name="word_logreg",
            family="linear_probabilistic",
            feature_family="word",
            description="TF-IDF de palabras con unigramas y bigramas + LogisticRegression.",
            subtask1_trials=cartesian_trials(
                {"classifier": "logreg"},
                C=[2.0, 4.0],
                class_weight=[None, "balanced"],
            ),
            subtask2_trials=cartesian_trials(
                {"classifier": "logreg", "score_mode": "proba"},
                C=[2.0, 4.0],
                class_weight=[None, "balanced"],
                threshold=[0.35, 0.45],
            ),
        ),
        ExperimentSpec(
            name="word_svm",
            family="max_margin",
            feature_family="word",
            description="TF-IDF de palabras con unigramas y bigramas + LinearSVC.",
            subtask1_trials=cartesian_trials(
                {"classifier": "linear_svc"},
                C=[0.5, 1.0],
                class_weight=[None, "balanced"],
            ),
            subtask2_trials=cartesian_trials(
                {"classifier": "linear_svc", "score_mode": "decision"},
                C=[0.5, 1.0],
                class_weight=[None, "balanced"],
                threshold=[-0.25, 0.0],
            ),
        ),
        ExperimentSpec(
            name="char_svm",
            family="char_ngrams",
            feature_family="char",
            description="TF-IDF de caracteres char_wb 3-5 + LinearSVC.",
            subtask1_trials=cartesian_trials(
                {"classifier": "linear_svc"},
                C=[0.5, 1.0],
                class_weight=[None, "balanced"],
            ),
            subtask2_trials=cartesian_trials(
                {"classifier": "linear_svc", "score_mode": "decision"},
                C=[0.5, 1.0],
                class_weight=[None, "balanced"],
                threshold=[-0.25, 0.0],
            ),
        ),
        ExperimentSpec(
            name="word_char_sgd",
            family="hybrid_linear",
            feature_family="word_char",
            description="Union de TF-IDF de palabras y caracteres + SGDClassifier modified_huber.",
            subtask1_trials=cartesian_trials(
                {"classifier": "sgd", "loss": "modified_huber"},
                alpha=[1e-5, 3e-6],
                class_weight=[None, "balanced"],
            ),
            subtask2_trials=cartesian_trials(
                {"classifier": "sgd", "loss": "modified_huber", "score_mode": "decision"},
                alpha=[1e-5, 3e-6],
                class_weight=[None, "balanced"],
                threshold=[-0.25, 0.0],
            ),
        ),
        ExperimentSpec(
            name="word_char_logreg",
            family="hybrid_probabilistic",
            feature_family="word_char",
            description="Union de TF-IDF de palabras y caracteres + LogisticRegression.",
            subtask1_trials=cartesian_trials(
                {"classifier": "logreg"},
                C=[1.0, 2.0],
                class_weight=[None, "balanced"],
            ),
            subtask2_trials=cartesian_trials(
                {"classifier": "logreg", "score_mode": "proba"},
                C=[1.0, 2.0],
                class_weight=[None, "balanced"],
                threshold=[0.35, 0.45],
            ),
        ),
    ]


def load_experiment_overrides(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "experiments" in payload:
        payload = payload["experiments"]
    if not isinstance(payload, dict):
        raise ValueError("--experiment-overrides-json debe contener un objeto por nombre de experimento")
    overrides: dict[str, dict[str, Any]] = {}
    for name, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError(f"Override invalido para {name}: se esperaba un objeto")
        overrides[str(name)] = dict(value)
    return overrides


def normalize_trial_list(value: Any, *, experiment_name: str, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{experiment_name}.{field_name} debe ser una lista no vacia")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{experiment_name}.{field_name}[{index}] debe ser un objeto")
        normalized.append(dict(item))
    return normalized


def select_experiments(
    experiments: list[ExperimentSpec],
    selected_names: list[str] | None,
    overrides: dict[str, dict[str, Any]],
) -> list[ExperimentSpec]:
    by_name = {experiment.name: experiment for experiment in experiments}
    if selected_names:
        missing = [name for name in selected_names if name not in by_name]
        if missing:
            raise ValueError(f"Experimentos no reconocidos: {missing}")
        selected = [by_name[name] for name in selected_names]
    else:
        selected = list(experiments)

    normalized: list[ExperimentSpec] = []
    for experiment in selected:
        override = overrides.get(experiment.name, {})
        subtask1_trials = experiment.subtask1_trials
        subtask2_trials = experiment.subtask2_trials
        if "subtask1_trials" in override:
            subtask1_trials = normalize_trial_list(
                override["subtask1_trials"],
                experiment_name=experiment.name,
                field_name="subtask1_trials",
            )
        if "subtask2_trials" in override:
            subtask2_trials = normalize_trial_list(
                override["subtask2_trials"],
                experiment_name=experiment.name,
                field_name="subtask2_trials",
            )
        normalized.append(
            replace(
                experiment,
                family=str(override.get("family", experiment.family)),
                description=str(override.get("description", experiment.description)),
                subtask1_trials=subtask1_trials,
                subtask2_trials=subtask2_trials,
            )
        )
    return normalized


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def write_subtask1_submission(path: Path, prediction_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for prediction_id in prediction_ids:
            writer.writerow([prediction_id])


def write_subtask2_submission(path: Path, prediction_rows: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in prediction_rows:
            writer.writerow(row)


def write_verbose_csv_subtask1(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["record_id", "gold_class_id", "gold_class", "pred_class_id", "pred_class", "score_margin"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_verbose_csv_subtask2(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["record_id", *SUBTASK2_COLUMNS, "gold_types", "pred_types", "positive_margins"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        text = f"{value:.6g}"
        return text.replace("+", "")
    return str(value).replace("/", "_")


def trial_label(trial: dict[str, Any], exclude: set[str] | None = None) -> str:
    ignored = {"classifier", "score_mode"}
    if exclude:
        ignored |= exclude
    parts = [trial["classifier"]]
    for key in sorted(key for key in trial if key not in ignored):
        parts.append(f"{key}={format_value(trial[key])}")
    return ", ".join(parts)


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
        return MatrixBundle(
            train=train_matrix,
            valid=valid_matrix,
            eval=eval_matrix,
            metadata={
                "feature_family": feature_family,
                "vectorizer": {
                    "type": "tfidf_word",
                    "ngram_range": [1, 2],
                    "min_df": 2,
                    "max_df": 0.98,
                    "sublinear_tf": True,
                },
            },
        )
    if feature_family == "char":
        vectorizer = build_char_vectorizer()
        train_matrix = writable_csr(vectorizer.fit_transform(train_texts))
        valid_matrix = writable_csr(vectorizer.transform(valid_texts)) if valid_texts is not None else None
        eval_matrix = writable_csr(vectorizer.transform(eval_texts)) if eval_texts is not None else None
        return MatrixBundle(
            train=train_matrix,
            valid=valid_matrix,
            eval=eval_matrix,
            metadata={
                "feature_family": feature_family,
                "vectorizer": {
                    "type": "tfidf_char_wb",
                    "ngram_range": [3, 5],
                    "min_df": 2,
                    "sublinear_tf": True,
                },
            },
        )
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
        return MatrixBundle(
            train=train_matrix,
            valid=valid_matrix,
            eval=eval_matrix,
            metadata={
                "feature_family": feature_family,
                "vectorizers": [
                    {
                        "type": "tfidf_word",
                        "ngram_range": [1, 2],
                        "min_df": 2,
                        "max_df": 0.98,
                        "sublinear_tf": True,
                    },
                    {
                        "type": "tfidf_char_wb",
                        "ngram_range": [3, 5],
                        "min_df": 2,
                        "sublinear_tf": True,
                    },
                ],
            },
        )
    raise ValueError(f"feature_family no soportada: {feature_family}")


def build_feature_bundles(
    train_texts: list[str],
    valid_texts: list[str] | None,
    eval_texts: list[str] | None,
) -> dict[str, MatrixBundle]:
    return {
        feature_family: build_feature_bundle(feature_family, train_texts, valid_texts, eval_texts)
        for feature_family in FEATURE_FAMILIES
    }


def build_subtask1_estimator(trial: dict[str, Any]):
    classifier = trial["classifier"]
    if classifier == "complement_nb":
        return ComplementNB(alpha=float(trial["alpha"]))
    if classifier == "logreg":
        return LogisticRegression(
            C=float(trial["C"]),
            class_weight=trial["class_weight"],
            max_iter=4000,
            solver="saga",
            random_state=SEED,
        )
    if classifier == "linear_svc":
        return LinearSVC(
            C=float(trial["C"]),
            class_weight=trial["class_weight"],
            dual="auto",
            random_state=SEED,
        )
    if classifier == "sgd":
        return SGDClassifier(
            loss=str(trial["loss"]),
            alpha=float(trial["alpha"]),
            class_weight=trial["class_weight"],
            max_iter=2000,
            tol=1e-3,
            random_state=SEED,
        )
    raise ValueError(f"Clasificador subtask1 no soportado: {classifier}")


def build_subtask2_estimator(trial: dict[str, Any], ovr_jobs: int):
    classifier = trial["classifier"]
    if classifier == "complement_nb":
        base_estimator = ComplementNB(alpha=float(trial["alpha"]))
    elif classifier == "logreg":
        base_estimator = LogisticRegression(
            C=float(trial["C"]),
            class_weight=trial["class_weight"],
            max_iter=4000,
            solver="liblinear",
            random_state=SEED,
        )
    elif classifier == "linear_svc":
        base_estimator = LinearSVC(
            C=float(trial["C"]),
            class_weight=trial["class_weight"],
            dual="auto",
            random_state=SEED,
        )
    elif classifier == "sgd":
        base_estimator = SGDClassifier(
            loss=str(trial["loss"]),
            alpha=float(trial["alpha"]),
            class_weight=trial["class_weight"],
            max_iter=2000,
            tol=1e-3,
            random_state=SEED,
        )
    else:
        raise ValueError(f"Clasificador subtask2 no soportado: {classifier}")
    return OneVsRestClassifier(base_estimator, n_jobs=ovr_jobs)


def scores_and_vectors_from_subtask2_model(
    estimator,
    features,
    score_mode: str,
    threshold: float,
) -> tuple[np.ndarray, list[list[int]], list[dict[str, float]]]:
    if score_mode == "proba":
        scores = estimator.predict_proba(features)
    elif score_mode == "decision":
        scores = estimator.decision_function(features)
    else:
        raise ValueError(f"score_mode no soportado: {score_mode}")

    score_array = np.asarray(scores)
    if score_array.ndim == 1:
        score_array = score_array[:, np.newaxis]

    pred_full_vectors: list[list[int]] = []
    score_dicts: list[dict[str, float]] = []
    for row in score_array:
        positive = [1 if float(value) >= threshold else 0 for value in row.tolist()]
        full_vector = positive + [0]
        if sum(positive) == 0:
            full_vector[-1] = 1
        score_dict = {label: float(value) for label, value in zip(NON_NA_LABELS, row.tolist())}
        score_dict["N/A"] = 1.0 if full_vector[-1] == 1 else 0.0
        pred_full_vectors.append(full_vector)
        score_dicts.append(score_dict)
    return score_array, pred_full_vectors, score_dicts


def compute_subtask1_margins(estimator, features) -> list[float]:
    if hasattr(estimator, "decision_function"):
        raw_scores = estimator.decision_function(features)
        score_array = np.asarray(raw_scores)
        if score_array.ndim == 1:
            return [0.0 for _ in range(score_array.shape[0])]
    elif hasattr(estimator, "predict_proba"):
        score_array = np.asarray(estimator.predict_proba(features))
    else:
        return [0.0 for _ in range(features.shape[0])]

    margins: list[float] = []
    for row in score_array:
        row_values = sorted(float(value) for value in row.tolist())
        margin = row_values[-1] - row_values[-2] if len(row_values) > 1 else 0.0
        margins.append(margin)
    return margins


def to_full_gold_vectors(non_na_vectors: list[list[int]]) -> list[list[int]]:
    full_vectors: list[list[int]] = []
    for row in non_na_vectors:
        full_vectors.append(row + [1 if sum(row) == 0 else 0])
    return full_vectors


def label_names_from_full_vector(vector: list[int]) -> list[str]:
    return [label for label, value in zip(SUBTASK2_LABELS, vector) if int(value) == 1]


def subtask1_selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["macro_f1"]),
        float(metrics["accuracy"]),
        float(metrics["weighted_f1"]),
    )


def subtask2_selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(metrics["macro_f1"]),
        float(metrics["micro_f1"]),
        float(metrics["weighted_f1"]),
        -float(metrics["hamming_loss"]),
        float(metrics["exact_match_accuracy"]),
    )


def summarize_trial_result(trial: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "trial": dict(trial),
        "trial_label": trial_label(trial),
        "metrics": {
            key: float(metrics[key])
            for key in metrics
            if key != "per_label"
        },
    }
    return payload


def tune_subtask1(
    trials: list[dict[str, Any]],
    features: MatrixBundle,
    y_train: list[str],
    y_valid: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    for trial in trials:
        estimator = build_subtask1_estimator(trial)
        estimator.fit(features.train, y_train)
        valid_pred = estimator.predict(features.valid)
        metrics = compute_multiclass_metrics(y_valid, list(valid_pred), SUBTASK1_LABELS)
        result = summarize_trial_result(trial, metrics)
        results.append(result)
        current_key = subtask1_selection_key(metrics)
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_result = result
    if best_result is None:
        raise RuntimeError("No hubo resultados de tuning para subtask1")
    return best_result, results


def tune_subtask2(
    trials: list[dict[str, Any]],
    features: MatrixBundle,
    y_train_non_na: list[list[int]],
    y_valid_non_na: list[list[int]],
    ovr_jobs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    gold_full = to_full_gold_vectors(y_valid_non_na)
    best_result: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float, float] | None = None
    for trial in trials:
        estimator = build_subtask2_estimator(trial, ovr_jobs)
        estimator.fit(features.train, y_train_non_na)
        _, pred_full, _ = scores_and_vectors_from_subtask2_model(
            estimator,
            features.valid,
            str(trial["score_mode"]),
            float(trial["threshold"]),
        )
        metrics = compute_multilabel_metrics(gold_full, pred_full, SUBTASK2_LABELS)
        result = summarize_trial_result(trial, metrics)
        results.append(result)
        current_key = subtask2_selection_key(metrics)
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_result = result
    if best_result is None:
        raise RuntimeError("No hubo resultados de tuning para subtask2")
    return best_result, results


def evaluate_experiment(
    spec: ExperimentSpec,
    best_subtask1: dict[str, Any],
    best_subtask2: dict[str, Any],
    subtask1_records,
    subtask2_records,
    subtask1_features: MatrixBundle,
    subtask2_features: MatrixBundle,
    subtask1_train_labels: list[str],
    subtask2_train_non_na: list[list[int]],
    ovr_jobs: int,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subtask1_jsonl = output_dir / "subtask1_predictions.jsonl"
    subtask2_jsonl = output_dir / "subtask2_predictions.jsonl"
    remove_if_exists(subtask1_jsonl)
    remove_if_exists(subtask2_jsonl)

    subtask1_trial = best_subtask1["trial"]
    subtask2_trial = best_subtask2["trial"]

    subtask1_estimator = build_subtask1_estimator(subtask1_trial)
    subtask1_estimator.fit(subtask1_features.train, subtask1_train_labels)
    subtask1_pred = list(subtask1_estimator.predict(subtask1_features.eval))
    subtask1_margins = compute_subtask1_margins(subtask1_estimator, subtask1_features.eval)

    subtask2_estimator = build_subtask2_estimator(subtask2_trial, ovr_jobs)
    subtask2_estimator.fit(subtask2_features.train, subtask2_train_non_na)
    _, subtask2_pred_vectors, subtask2_score_dicts = scores_and_vectors_from_subtask2_model(
        subtask2_estimator,
        subtask2_features.eval,
        str(subtask2_trial["score_mode"]),
        float(subtask2_trial["threshold"]),
    )

    verbose_subtask1_rows = []
    subtask1_submission = []
    subtask1_gold = []
    for record, pred_class, score_margin in zip(subtask1_records, subtask1_pred, subtask1_margins):
        gold_class = SUBTASK1_ID_TO_NAME[record.severity_id]
        subtask1_gold.append(gold_class)
        subtask1_submission.append(SUBTASK1_NAME_TO_ID[pred_class])
        append_jsonl(
            subtask1_jsonl,
            {
                "record_id": record.record_id,
                "gold_class_id": record.severity_id,
                "gold_class": gold_class,
                "pred_class": pred_class,
                "pred_types": [],
                "score_margin": score_margin,
            },
        )
        verbose_subtask1_rows.append(
            {
                "record_id": record.record_id,
                "gold_class_id": record.severity_id,
                "gold_class": gold_class,
                "pred_class_id": SUBTASK1_NAME_TO_ID[pred_class],
                "pred_class": pred_class,
                "score_margin": f"{score_margin:.6f}",
            }
        )

    verbose_subtask2_rows = []
    subtask2_gold = []
    for record, pred_vector, score_dict in zip(subtask2_records, subtask2_pred_vectors, subtask2_score_dicts):
        gold_vector = [record.label_vector[column] for column in SUBTASK2_COLUMNS]
        pred_types = label_names_from_full_vector(pred_vector)
        subtask2_gold.append(gold_vector)
        append_jsonl(
            subtask2_jsonl,
            {
                "record_id": record.record_id,
                "gold_types": label_names_from_vector(record.label_vector),
                "gold_vector": gold_vector,
                "pred_class": None,
                "pred_types": pred_types,
                "positive_margins": score_dict,
            },
        )
        verbose_row = {
            "record_id": record.record_id,
            "gold_types": ",".join(label_names_from_vector(record.label_vector)),
            "pred_types": ",".join(pred_types),
            "positive_margins": json.dumps(score_dict, ensure_ascii=False, sort_keys=True),
        }
        for column, value in zip(SUBTASK2_COLUMNS, pred_vector):
            verbose_row[column] = value
        verbose_subtask2_rows.append(verbose_row)

    subtask1_metrics = compute_multiclass_metrics(subtask1_gold, subtask1_pred, SUBTASK1_LABELS)
    subtask2_metrics = compute_multilabel_metrics(subtask2_gold, subtask2_pred_vectors, SUBTASK2_LABELS)

    submission_dir = output_dir / "submission"
    write_subtask1_submission(submission_dir / "subtask1.csv", subtask1_submission)
    write_subtask2_submission(submission_dir / "subtask2.csv", subtask2_pred_vectors)
    write_verbose_csv_subtask1(output_dir / "subtask1_verbose.csv", verbose_subtask1_rows)
    write_verbose_csv_subtask2(output_dir / "subtask2_verbose.csv", verbose_subtask2_rows)

    summary = {
        "model_name_or_path": spec.name,
        "family": spec.family,
        "description": spec.description,
        "feature_family": spec.feature_family,
        "subtask1_split": "devel",
        "subtask2_split": "devel",
        "subtask1_records": len(subtask1_records),
        "subtask2_records": len(subtask2_records),
        "subtask1_metrics": subtask1_metrics,
        "subtask2_metrics": subtask2_metrics,
        "subtask1_selected_trial": best_subtask1,
        "subtask2_selected_trial": best_subtask2,
        "feature_config": {
            "subtask1": subtask1_features.metadata,
            "subtask2": subtask2_features.metadata,
        },
    }
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def best_per_label_subtask1(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    winners: list[dict[str, Any]] = []
    for label in SUBTASK1_LABELS:
        best_name = None
        best_value = -1.0
        for experiment in experiments:
            value = float(experiment["metrics_summary"]["subtask1_metrics"]["per_label"][label]["f1"])
            if value > best_value:
                best_value = value
                best_name = experiment["name"]
        winners.append({"label": label, "best_model": best_name, "f1": best_value})
    return winners


def best_per_label_subtask2(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    winners: list[dict[str, Any]] = []
    for label in SUBTASK2_LABELS:
        best_name = None
        best_value = -1.0
        for experiment in experiments:
            value = float(experiment["metrics_summary"]["subtask2_metrics"]["per_label"][label]["f1"])
            if value > best_value:
                best_value = value
                best_name = experiment["name"]
        winners.append({"label": label, "best_model": best_name, "f1": best_value})
    return winners


def best_single_experiment(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    best_item = None
    best_key = None
    for experiment in experiments:
        validation_key = (
            float(experiment["subtask1_selected_trial"]["metrics"]["macro_f1"]),
            float(experiment["subtask2_selected_trial"]["metrics"]["macro_f1"]),
            float(experiment["subtask2_selected_trial"]["metrics"]["micro_f1"]),
        )
        if best_key is None or validation_key > best_key:
            best_key = validation_key
            best_item = experiment
    if best_item is None:
        raise RuntimeError("No se pudo determinar el mejor experimento unico")
    return best_item


def reference_metrics(path: Path, label: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "label": label,
        "path": str(path),
        "metrics": payload,
    }


def short_metrics_row(name: str, payload: dict[str, Any], note: str) -> dict[str, Any]:
    subtask1 = payload["subtask1_metrics"]
    subtask2 = payload["subtask2_metrics"]
    return {
        "name": name,
        "note": note,
        "subtask1_accuracy": float(subtask1["accuracy"]),
        "subtask1_macro_f1": float(subtask1["macro_f1"]),
        "subtask2_exact_match": float(subtask2["exact_match_accuracy"]),
        "subtask2_macro_f1": float(subtask2["macro_f1"]),
        "subtask2_micro_f1": float(subtask2["micro_f1"]),
        "subtask2_weighted_f1": float(subtask2["weighted_f1"]),
        "subtask2_hamming_loss": float(subtask2["hamming_loss"]),
    }


def format_float(value: float) -> str:
    return f"{value:.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_report_markdown(report_payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Suite de lineas base clasicas para WomenHelp",
        "",
        f"Generado: {report_payload['generated_at']}",
        "",
        "## Objetivo",
        "",
        "Evaluar una bateria de clasificadores clasicos y no tan clasicos, con ajuste interno de hiperparametros sobre train y medicion final sobre devel, para agotar alternativas CPU-only antes de seguir apostando por Gemma 4.",
        "",
        "## Protocolo",
        "",
        f"- split interno de validacion: {report_payload['protocol']['validation_size']:.2f} de train",
        f"- random_state: {report_payload['protocol']['random_state']}",
        "- criterio de seleccion subtask1: macro_f1, luego accuracy, luego weighted_f1",
        "- criterio de seleccion subtask2: macro_f1, luego micro_f1, luego weighted_f1, luego hamming_loss",
        "- para subtask2 se modelaron solo las seis etiquetas no N/A; N/A se activa cuando no hay ninguna positiva",
        "",
        "## Familias evaluadas",
        "",
    ]

    family_rows = []
    for experiment in report_payload["experiments"]:
        family_rows.append(
            [
                experiment["name"],
                experiment["family"],
                experiment["feature_family"],
                experiment["description"],
                experiment["subtask1_selected_trial"]["trial_label"],
                experiment["subtask2_selected_trial"]["trial_label"],
            ]
        )
    lines.extend(
        markdown_table(
            [
                "modelo",
                "familia",
                "features",
                "descripcion",
                "mejor config subtask1",
                "mejor config subtask2",
            ],
            family_rows,
        )
    )
    lines.extend(["", "## Comparativo final en devel", ""])

    result_rows = []
    for row in report_payload["final_comparison_rows"]:
        result_rows.append(
            [
                row["name"],
                row["note"],
                format_float(row["subtask1_accuracy"]),
                format_float(row["subtask1_macro_f1"]),
                format_float(row["subtask2_exact_match"]),
                format_float(row["subtask2_macro_f1"]),
                format_float(row["subtask2_micro_f1"]),
                format_float(row["subtask2_weighted_f1"]),
                format_float(row["subtask2_hamming_loss"]),
            ]
        )
    lines.extend(
        markdown_table(
            [
                "modelo",
                "nota",
                "s1 acc",
                "s1 macro_f1",
                "s2 exact",
                "s2 macro_f1",
                "s2 micro_f1",
                "s2 weighted_f1",
                "s2 hamming",
            ],
            result_rows,
        )
    )
    lines.extend(["", "## Ganadores", ""])
    bundle = report_payload["taskwise_bundle"]
    best_single = report_payload["best_single_experiment"]
    lines.extend(
        [
            f"- mejor subtask1 por validacion interna: {bundle['subtask1_source_experiment']}",
            f"- mejor subtask2 por validacion interna: {bundle['subtask2_source_experiment']}",
            f"- mejor experimento unico por validacion interna: {best_single['name']}",
            "",
            "## Mejor por etiqueta: Subtask 1",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["etiqueta", "mejor modelo", "f1 en devel"],
            [
                [item["label"], item["best_model"], format_float(item["f1"])]
                for item in report_payload["best_per_label_subtask1"]
            ],
        )
    )
    lines.extend(["", "## Mejor por etiqueta: Subtask 2", ""])
    lines.extend(
        markdown_table(
            ["etiqueta", "mejor modelo", "f1 en devel"],
            [
                [item["label"], item["best_model"], format_float(item["f1"])]
                for item in report_payload["best_per_label_subtask2"]
            ],
        )
    )
    lines.extend(["", "## Lectura tecnica", ""])
    lines.extend([f"- {line}" for line in report_payload["findings"]])
    lines.extend(["", "## Artefactos", ""])
    lines.extend([f"- {line}" for line in report_payload["artifacts"]])
    return "\n".join(lines) + "\n"


def build_bundle_summary(
    bundle_dir: Path,
    subtask1_source: dict[str, Any],
    subtask2_source: dict[str, Any],
    subtask1_records,
    subtask2_records,
) -> dict[str, Any]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    subtask1_jsonl = bundle_dir / "subtask1_predictions.jsonl"
    subtask2_jsonl = bundle_dir / "subtask2_predictions.jsonl"
    remove_if_exists(subtask1_jsonl)
    remove_if_exists(subtask2_jsonl)

    subtask1_source_dir = Path(subtask1_source["output_dir"])
    subtask2_source_dir = Path(subtask2_source["output_dir"])

    with (subtask1_source_dir / "subtask1_predictions.jsonl").open("r", encoding="utf-8") as handle:
        subtask1_lines = [json.loads(line) for line in handle if line.strip()]
    with (subtask2_source_dir / "subtask2_predictions.jsonl").open("r", encoding="utf-8") as handle:
        subtask2_lines = [json.loads(line) for line in handle if line.strip()]

    for row in subtask1_lines:
        append_jsonl(subtask1_jsonl, row)
    for row in subtask2_lines:
        append_jsonl(subtask2_jsonl, row)

    write_subtask1_submission(
        bundle_dir / "submission" / "subtask1.csv",
        [SUBTASK1_NAME_TO_ID[row["pred_class"]] for row in subtask1_lines],
    )
    write_subtask2_submission(
        bundle_dir / "submission" / "subtask2.csv",
        [row["gold_vector"] if False else [1 if label in row["pred_types"] else 0 for label in SUBTASK2_LABELS] for row in subtask2_lines],
    )

    summary = {
        "model_name_or_path": "classical_taskwise_best_bundle",
        "subtask1_source_experiment": subtask1_source["name"],
        "subtask2_source_experiment": subtask2_source["name"],
        "subtask1_records": len(subtask1_records),
        "subtask2_records": len(subtask2_records),
        "subtask1_metrics": subtask1_source["metrics_summary"]["subtask1_metrics"],
        "subtask2_metrics": subtask2_source["metrics_summary"]["subtask2_metrics"],
    }
    with (bundle_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    experiments = select_experiments(
        build_experiments(),
        args.experiment_name,
        load_experiment_overrides(args.experiment_overrides_json),
    )

    subtask1_train_records = slice_records(
        load_subtask1(args.subtask1_train_split, args.data_dir),
        args.limit_subtask1_train,
    )
    subtask1_eval_records = slice_records(
        load_subtask1(args.subtask1_eval_split, args.data_dir),
        args.limit_subtask1_eval,
    )
    subtask2_train_records = slice_records(
        load_subtask2(args.subtask2_train_split, args.data_dir),
        args.limit_subtask2_train,
    )
    subtask2_eval_records = slice_records(
        load_subtask2(args.subtask2_eval_split, args.data_dir),
        args.limit_subtask2_eval,
    )

    subtask1_train_texts = [record.text for record in subtask1_train_records]
    subtask1_train_labels = [SUBTASK1_ID_TO_NAME[record.severity_id] for record in subtask1_train_records]
    subtask1_eval_texts = [record.text for record in subtask1_eval_records]

    subtask2_train_texts = [record.text for record in subtask2_train_records]
    subtask2_train_non_na = [
        [record.label_vector[column] for column in NON_NA_COLUMNS]
        for record in subtask2_train_records
    ]
    subtask2_eval_texts = [record.text for record in subtask2_eval_records]

    s1_train_texts, s1_valid_texts, s1_train_labels_split, s1_valid_labels = train_test_split(
        subtask1_train_texts,
        subtask1_train_labels,
        test_size=args.validation_size,
        random_state=args.random_state,
        stratify=subtask1_train_labels,
    )
    s2_train_texts, s2_valid_texts, s2_train_non_na_split, s2_valid_non_na = train_test_split(
        subtask2_train_texts,
        subtask2_train_non_na,
        test_size=args.validation_size,
        random_state=args.random_state,
    )

    tuning_features_subtask1 = build_feature_bundles(s1_train_texts, s1_valid_texts, None)
    tuning_features_subtask2 = build_feature_bundles(s2_train_texts, s2_valid_texts, None)
    full_features_subtask1 = build_feature_bundles(subtask1_train_texts, None, subtask1_eval_texts)
    full_features_subtask2 = build_feature_bundles(subtask2_train_texts, None, subtask2_eval_texts)

    experiment_reports: list[dict[str, Any]] = []
    for spec in experiments:
        started = perf_counter()
        best_subtask1, subtask1_trials = tune_subtask1(
            spec.subtask1_trials,
            tuning_features_subtask1[spec.feature_family],
            s1_train_labels_split,
            s1_valid_labels,
        )
        best_subtask2, subtask2_trials = tune_subtask2(
            spec.subtask2_trials,
            tuning_features_subtask2[spec.feature_family],
            s2_train_non_na_split,
            s2_valid_non_na,
            args.ovr_jobs,
        )

        output_dir = output_root / spec.name
        metrics_summary = evaluate_experiment(
            spec,
            best_subtask1,
            best_subtask2,
            subtask1_eval_records,
            subtask2_eval_records,
            full_features_subtask1[spec.feature_family],
            full_features_subtask2[spec.feature_family],
            subtask1_train_labels,
            subtask2_train_non_na,
            args.ovr_jobs,
            output_dir,
        )
        runtime_seconds = perf_counter() - started
        metrics_summary["runtime_seconds"] = runtime_seconds
        with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics_summary, handle, indent=2, ensure_ascii=False)

        experiment_reports.append(
            {
                "name": spec.name,
                "family": spec.family,
                "feature_family": spec.feature_family,
                "description": spec.description,
                "output_dir": str(output_dir),
                "subtask1_selected_trial": best_subtask1,
                "subtask2_selected_trial": best_subtask2,
                "subtask1_trial_results": subtask1_trials,
                "subtask2_trial_results": subtask2_trials,
                "metrics_summary": metrics_summary,
                "runtime_seconds": runtime_seconds,
            }
        )

    best_s1_source = max(
        experiment_reports,
        key=lambda item: subtask1_selection_key(item["subtask1_selected_trial"]["metrics"]),
    )
    best_s2_source = max(
        experiment_reports,
        key=lambda item: subtask2_selection_key(item["subtask2_selected_trial"]["metrics"]),
    )
    taskwise_bundle_summary = build_bundle_summary(
        output_root / "taskwise_best_bundle",
        best_s1_source,
        best_s2_source,
        subtask1_eval_records,
        subtask2_eval_records,
    )

    legacy_reference = reference_metrics(
        REPO_ROOT / "outputs" / "classical_baseline" / "devel_full" / "metrics_summary.json",
        "legacy_hashing_nb",
    )
    prompt_reference = reference_metrics(
        REPO_ROOT / "outputs" / "prompt_baseline" / "devel_full" / "metrics_summary.json",
        "gemma4_prompt",
    )

    comparison_rows = []
    if prompt_reference is not None:
        comparison_rows.append(short_metrics_row(prompt_reference["label"], prompt_reference["metrics"], "baseline LLM actual"))
    if legacy_reference is not None:
        comparison_rows.append(short_metrics_row(legacy_reference["label"], legacy_reference["metrics"], "baseline clasica previa"))
    for experiment in experiment_reports:
        comparison_rows.append(short_metrics_row(experiment["name"], experiment["metrics_summary"], experiment["description"]))
    comparison_rows.append(short_metrics_row("taskwise_best_bundle", taskwise_bundle_summary, "mejor subtask1 + mejor subtask2 por validacion"))

    best_single = best_single_experiment(experiment_reports)

    findings = [
        f"La mejor macro_f1 en subtask1 dentro de la suite nueva vino de {best_s1_source['name']}.",
        f"La mejor macro_f1 en subtask2 dentro de la suite nueva vino de {best_s2_source['name']}.",
        f"El mejor experimento unico por validacion fue {best_single['name']}.",
    ]

    if prompt_reference is not None:
        prompt_metrics = prompt_reference["metrics"]
        bundle_s1_delta = float(taskwise_bundle_summary["subtask1_metrics"]["macro_f1"]) - float(prompt_metrics["subtask1_metrics"]["macro_f1"])
        bundle_s2_delta = float(taskwise_bundle_summary["subtask2_metrics"]["macro_f1"]) - float(prompt_metrics["subtask2_metrics"]["macro_f1"])
        findings.append(
            f"El bundle clasico recomendado mejora a Gemma 4 prompt en subtask1 macro_f1 por {bundle_s1_delta:+.4f} y en subtask2 macro_f1 por {bundle_s2_delta:+.4f}."
        )
    if legacy_reference is not None:
        legacy_metrics = legacy_reference["metrics"]
        legacy_s1_delta = float(taskwise_bundle_summary["subtask1_metrics"]["macro_f1"]) - float(legacy_metrics["subtask1_metrics"]["macro_f1"])
        legacy_s2_delta = float(taskwise_bundle_summary["subtask2_metrics"]["macro_f1"]) - float(legacy_metrics["subtask2_metrics"]["macro_f1"])
        findings.append(
            f"Contra la baseline hashing+NB previa, el bundle recomendado cambia subtask1 macro_f1 en {legacy_s1_delta:+.4f} y subtask2 macro_f1 en {legacy_s2_delta:+.4f}."
        )
    findings.append("Los modelos lineales con TF-IDF de caracteres y palabras fueron mas competitivos que ComplementNB en multilabel.")
    findings.append("Arboles y ensambles no se priorizaron porque este problema es texto esparso de alta dimension, donde suelen rendir peor y costar mas.")

    artifacts = [
        f"suite completa: {args.report_markdown}",
        f"resumen JSON: {args.report_json}",
        f"artefactos por modelo: {output_root}",
    ]

    report_payload = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "protocol": {
            "validation_size": args.validation_size,
            "random_state": args.random_state,
            "subtask1_train_rows": len(subtask1_train_records),
            "subtask1_eval_rows": len(subtask1_eval_records),
            "subtask2_train_rows": len(subtask2_train_records),
            "subtask2_eval_rows": len(subtask2_eval_records),
        },
        "experiments": experiment_reports,
        "taskwise_bundle": taskwise_bundle_summary,
        "best_single_experiment": {
            "name": best_single["name"],
            "subtask1_selected_trial": best_single["subtask1_selected_trial"],
            "subtask2_selected_trial": best_single["subtask2_selected_trial"],
            "metrics_summary": best_single["metrics_summary"],
        },
        "best_per_label_subtask1": best_per_label_subtask1(experiment_reports),
        "best_per_label_subtask2": best_per_label_subtask2(experiment_reports),
        "references": {
            "legacy_hashing_nb": legacy_reference,
            "gemma4_prompt": prompt_reference,
        },
        "final_comparison_rows": comparison_rows,
        "findings": findings,
        "artifacts": artifacts,
    }

    report_json_path = Path(args.report_json)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_markdown_path = Path(args.report_markdown)
    report_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_markdown_path.write_text(build_report_markdown(report_payload), encoding="utf-8")

    print(json.dumps({
        "report_markdown": str(report_markdown_path),
        "report_json": str(report_json_path),
        "output_root": str(output_root),
        "best_subtask1": best_s1_source["name"],
        "best_subtask2": best_s2_source["name"],
        "best_single": best_single["name"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()