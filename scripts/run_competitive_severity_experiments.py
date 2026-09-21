from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402
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
NON_NA_LABELS = [label for label in SUBTASK2_LABELS if label != "N/A"]
NON_NA_COLUMNS = SUBTASK2_COLUMNS[: len(NON_NA_LABELS)]
SEVERITY_TO_INDEX = {label: index for index, label in enumerate(SUBTASK1_LABELS)}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    classifier: str
    params: dict[str, Any]
    use_type_scores: bool
    use_text_stats: bool


class OrdinalSeverityModel:
    def __init__(self, *, c_value: float, class_weight: str | None, max_iter: int = 1200):
        self.models = [
            LogisticRegression(
                C=c_value,
                class_weight=class_weight,
                max_iter=max_iter,
                solver="lbfgs",
            )
            for _ in range(len(SUBTASK1_LABELS) - 1)
        ]

    def fit(self, features, labels: np.ndarray) -> "OrdinalSeverityModel":
        for threshold, model in enumerate(self.models):
            binary_labels = (labels > threshold).astype(np.int32)
            model.fit(features, binary_labels)
        return self

    def predict_proba(self, features) -> np.ndarray:
        ge_probabilities = np.column_stack([model.predict_proba(features)[:, 1] for model in self.models])
        monotonic_ge = np.minimum.accumulate(ge_probabilities, axis=1)
        probabilities = np.zeros((features.shape[0], len(SUBTASK1_LABELS)), dtype=np.float64)
        probabilities[:, 0] = 1.0 - monotonic_ge[:, 0]
        probabilities[:, 1] = monotonic_ge[:, 0] - monotonic_ge[:, 1]
        probabilities[:, 2] = monotonic_ge[:, 1] - monotonic_ge[:, 2]
        probabilities[:, 3] = monotonic_ge[:, 2]
        np.clip(probabilities, 0.0, 1.0, out=probabilities)
        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return probabilities / row_sums

    def predict(self, features) -> np.ndarray:
        return np.argmax(self.predict_proba(features), axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corre experimentos competitivos de severidad")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--subtask1-train-split", default="train")
    parser.add_argument("--subtask1-eval-split", default="devel")
    parser.add_argument("--subtask2-train-split", default="train")
    parser.add_argument("--limit-subtask1-train", type=int, default=0)
    parser.add_argument("--limit-subtask1-eval", type=int, default=0)
    parser.add_argument("--limit-subtask2-train", type=int, default=0)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=SEED)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "competitive_severity" / "20260419"),
    )
    parser.add_argument(
        "--report-markdown",
        default=str(REPO_ROOT / "reports" / "baselines" / "competitive_severity_experiments_20260419.md"),
    )
    parser.add_argument(
        "--report-json",
        default=str(REPO_ROOT / "reports" / "baselines" / "competitive_severity_experiments_20260419.json"),
    )
    return parser.parse_args()


def build_word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )


def build_char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
    )


def build_union_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", build_word_vectorizer()),
            ("char", build_char_vectorizer()),
        ]
    )


def text_stats_matrix(texts: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for text in texts:
        char_count = float(len(text))
        token_count = float(len(text.split()))
        upper_count = float(sum(character.isupper() for character in text))
        digit_count = float(sum(character.isdigit() for character in text))
        exclamation_count = float(text.count("!"))
        question_count = float(text.count("?"))
        comma_count = float(text.count(","))
        newline_count = float(text.count("\n"))
        upper_ratio = upper_count / char_count if char_count else 0.0
        rows.append(
            [
                char_count,
                token_count,
                upper_ratio,
                digit_count,
                exclamation_count,
                question_count,
                comma_count,
                newline_count,
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def fit_auxiliary_type_model(texts: list[str], label_matrix: np.ndarray) -> tuple[FeatureUnion, OneVsRestClassifier]:
    vectorizer = build_union_vectorizer()
    features = vectorizer.fit_transform(texts)
    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=1200,
            solver="liblinear",
        )
    )
    classifier.fit(features, label_matrix)
    return vectorizer, classifier


def build_type_score_features(
    texts: list[str],
    vectorizer: FeatureUnion,
    classifier: OneVsRestClassifier,
) -> np.ndarray:
    features = vectorizer.transform(texts)
    positive_probabilities = classifier.predict_proba(features)
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
    text_stats: np.ndarray | None,
    stats_scaler: StandardScaler | None,
) -> sparse.csr_matrix:
    blocks = [text_features]
    if type_scores is not None and type_scaler is not None:
        scaled_type_scores = type_scaler.transform(type_scores)
        blocks.append(sparse.csr_matrix(scaled_type_scores))
    if text_stats is not None and stats_scaler is not None:
        scaled_text_stats = stats_scaler.transform(text_stats)
        blocks.append(sparse.csr_matrix(scaled_text_stats))
    return sparse.hstack(blocks, format="csr")


def candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            name="word_char_logreg_text_only",
            classifier="logreg",
            params={"C": 2.0, "class_weight": "balanced"},
            use_type_scores=False,
            use_text_stats=False,
        ),
        CandidateSpec(
            name="word_char_logreg_types",
            classifier="logreg",
            params={"C": 4.0, "class_weight": "balanced"},
            use_type_scores=True,
            use_text_stats=False,
        ),
        CandidateSpec(
            name="word_char_logreg_types_stats",
            classifier="logreg",
            params={"C": 4.0, "class_weight": "balanced"},
            use_type_scores=True,
            use_text_stats=True,
        ),
        CandidateSpec(
            name="word_char_linear_svc_types_stats",
            classifier="linear_svc",
            params={"C": 0.75, "class_weight": "balanced"},
            use_type_scores=True,
            use_text_stats=True,
        ),
        CandidateSpec(
            name="word_char_ordinal_logreg_types_stats",
            classifier="ordinal_logreg",
            params={"C": 2.0, "class_weight": "balanced"},
            use_type_scores=True,
            use_text_stats=True,
        ),
        CandidateSpec(
            name="word_char_ordinal_logreg_types",
            classifier="ordinal_logreg",
            params={"C": 4.0, "class_weight": "balanced"},
            use_type_scores=True,
            use_text_stats=False,
        ),
    ]


def fit_candidate(spec: CandidateSpec, features, labels: np.ndarray):
    if spec.classifier == "logreg":
        model = LogisticRegression(
            C=spec.params["C"],
            class_weight=spec.params["class_weight"],
            max_iter=1500,
            solver="lbfgs",
        )
    elif spec.classifier == "linear_svc":
        model = LinearSVC(
            C=spec.params["C"],
            class_weight=spec.params["class_weight"],
        )
    elif spec.classifier == "ordinal_logreg":
        model = OrdinalSeverityModel(
            c_value=spec.params["C"],
            class_weight=spec.params["class_weight"],
        )
    else:
        raise ValueError(f"Clasificador no soportado: {spec.classifier}")
    model.fit(features, labels)
    return model


def predict_candidate(model, features) -> np.ndarray:
    predictions = model.predict(features)
    if predictions.ndim == 1:
        return predictions.astype(np.int32)
    return np.asarray(predictions).astype(np.int32)


def metrics_for_indices(gold_indices: np.ndarray, pred_indices: np.ndarray) -> dict[str, Any]:
    gold_labels = [SUBTASK1_LABELS[index] for index in gold_indices]
    pred_labels = [SUBTASK1_LABELS[index] for index in pred_indices]
    return compute_multiclass_metrics(gold_labels, pred_labels, SUBTASK1_LABELS)


def write_verbose_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["record_id", "gold_class_id", "gold_class", "pred_class_id", "pred_class"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_submission(path: Path, prediction_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for prediction_id in prediction_ids:
            writer.writerow([prediction_id])


def report_lines(payload: dict[str, Any]) -> list[str]:
    lines = [
        "# Experimentos competitivos de severidad",
        "",
        f"Generado: {payload['generated_at']}",
        "",
        "## Hipótesis",
        "",
        "- El techo de ensamble con la suite clásica previa era bajo: el oráculo por etiqueta sobre devel solo llegaba a macro-F1 0.5061.",
        "- La señal de tipos de violencia sí correlaciona con severidad, así que vale la pena usar la subtarea 2 como señal auxiliar para subtask 1.",
        "- La ordinalidad Mild < Medium < High < Severe también puede explotarse explícitamente.",
        "",
        "## Resultados en validación interna",
        "",
        "| experimento | usa tipos | usa stats | macro_f1 | accuracy | weighted_f1 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in payload["validation_rows"]:
        lines.append(
            f"| {row['name']} | {'si' if row['use_type_scores'] else 'no'} | {'si' if row['use_text_stats'] else 'no'} | {row['macro_f1']:.4f} | {row['accuracy']:.4f} | {row['weighted_f1']:.4f} |"
        )

    best = payload["best_experiment"]
    lines.extend(
        [
            "",
            "## Mejor experimento",
            "",
            f"- nombre: {best['name']}",
            f"- clasificador: {best['classifier']}",
            f"- usa tipos auxiliares: {'si' if best['use_type_scores'] else 'no'}",
            f"- usa estadisticas de texto: {'si' if best['use_text_stats'] else 'no'}",
            f"- macro_f1 de validacion: {best['validation_metrics']['macro_f1']:.4f}",
            "",
            "## Evaluación final en devel",
            "",
            f"- accuracy: {payload['devel_metrics']['accuracy']:.4f}",
            f"- macro_f1: {payload['devel_metrics']['macro_f1']:.4f}",
            f"- micro_f1: {payload['devel_metrics']['micro_f1']:.4f}",
            f"- weighted_f1: {payload['devel_metrics']['weighted_f1']:.4f}",
            "",
            "## F1 por etiqueta en devel",
            "",
            "| etiqueta | precision | recall | f1 | support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in SUBTASK1_LABELS:
        scores = payload["devel_metrics"]["per_label"][label]
        lines.append(
            f"| {label} | {scores['precision']:.4f} | {scores['recall']:.4f} | {scores['f1']:.4f} | {scores['support']} |"
        )
    return lines


def main() -> None:
    args = parse_args()
    start_time = perf_counter()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subtask1_train = slice_records(load_subtask1(args.subtask1_train_split, args.data_dir), args.limit_subtask1_train)
    subtask1_eval = slice_records(load_subtask1(args.subtask1_eval_split, args.data_dir), args.limit_subtask1_eval)
    subtask2_train = slice_records(load_subtask2(args.subtask2_train_split, args.data_dir), args.limit_subtask2_train)

    train_texts = [record.text for record in subtask1_train]
    train_labels = np.asarray([int(record.severity_id) for record in subtask1_train], dtype=np.int32)
    eval_texts = [record.text for record in subtask1_eval]
    eval_labels = np.asarray([int(record.severity_id) for record in subtask1_eval], dtype=np.int32)

    aux_type_texts = [record.text for record in subtask2_train]
    aux_type_matrix = np.asarray(
        [[int(record.label_vector[column]) for column in NON_NA_COLUMNS] for record in subtask2_train],
        dtype=np.int32,
    )
    aux_vectorizer, aux_classifier = fit_auxiliary_type_model(aux_type_texts, aux_type_matrix)

    train_type_scores = build_type_score_features(train_texts, aux_vectorizer, aux_classifier)
    eval_type_scores = build_type_score_features(eval_texts, aux_vectorizer, aux_classifier)
    train_stats = text_stats_matrix(train_texts)
    eval_stats = text_stats_matrix(eval_texts)

    fit_texts, valid_texts, fit_labels, valid_labels, fit_type_scores, valid_type_scores, fit_stats, valid_stats = train_test_split(
        train_texts,
        train_labels,
        train_type_scores,
        train_stats,
        test_size=args.validation_size,
        random_state=args.random_state,
        stratify=train_labels,
    )

    text_vectorizer = build_union_vectorizer()
    fit_text_features = text_vectorizer.fit_transform(fit_texts)
    valid_text_features = text_vectorizer.transform(valid_texts)
    full_train_text_features = text_vectorizer.fit_transform(train_texts)
    full_eval_text_features = text_vectorizer.transform(eval_texts)

    validation_rows: list[dict[str, Any]] = []
    best_candidate: CandidateSpec | None = None
    best_validation_metrics: dict[str, Any] | None = None

    for spec in candidate_specs():
        type_scaler = None
        stats_scaler = None
        if spec.use_type_scores:
            type_scaler = StandardScaler()
            type_scaler.fit(fit_type_scores)
        if spec.use_text_stats:
            stats_scaler = StandardScaler()
            stats_scaler.fit(fit_stats)

        fit_features = compose_features(
            fit_text_features,
            type_scores=fit_type_scores if spec.use_type_scores else None,
            type_scaler=type_scaler,
            text_stats=fit_stats if spec.use_text_stats else None,
            stats_scaler=stats_scaler,
        )
        valid_features = compose_features(
            valid_text_features,
            type_scores=valid_type_scores if spec.use_type_scores else None,
            type_scaler=type_scaler,
            text_stats=valid_stats if spec.use_text_stats else None,
            stats_scaler=stats_scaler,
        )

        model = fit_candidate(spec, fit_features, fit_labels)
        valid_predictions = predict_candidate(model, valid_features)
        valid_metrics = metrics_for_indices(valid_labels, valid_predictions)
        validation_rows.append(
            {
                "name": spec.name,
                "classifier": spec.classifier,
                "params": spec.params,
                "use_type_scores": spec.use_type_scores,
                "use_text_stats": spec.use_text_stats,
                "accuracy": valid_metrics["accuracy"],
                "macro_f1": valid_metrics["macro_f1"],
                "weighted_f1": valid_metrics["weighted_f1"],
            }
        )
        if best_validation_metrics is None or (
            valid_metrics["macro_f1"],
            valid_metrics["accuracy"],
            valid_metrics["weighted_f1"],
        ) > (
            best_validation_metrics["macro_f1"],
            best_validation_metrics["accuracy"],
            best_validation_metrics["weighted_f1"],
        ):
            best_candidate = spec
            best_validation_metrics = valid_metrics

    if best_candidate is None or best_validation_metrics is None:
        raise RuntimeError("No se pudo seleccionar un experimento ganador")

    final_type_scaler = None
    final_stats_scaler = None
    if best_candidate.use_type_scores:
        final_type_scaler = StandardScaler()
        final_type_scaler.fit(train_type_scores)
    if best_candidate.use_text_stats:
        final_stats_scaler = StandardScaler()
        final_stats_scaler.fit(train_stats)

    full_train_features = compose_features(
        full_train_text_features,
        type_scores=train_type_scores if best_candidate.use_type_scores else None,
        type_scaler=final_type_scaler,
        text_stats=train_stats if best_candidate.use_text_stats else None,
        stats_scaler=final_stats_scaler,
    )
    full_eval_features = compose_features(
        full_eval_text_features,
        type_scores=eval_type_scores if best_candidate.use_type_scores else None,
        type_scaler=final_type_scaler,
        text_stats=eval_stats if best_candidate.use_text_stats else None,
        stats_scaler=final_stats_scaler,
    )

    final_model = fit_candidate(best_candidate, full_train_features, train_labels)
    eval_predictions = predict_candidate(final_model, full_eval_features)
    devel_metrics = metrics_for_indices(eval_labels, eval_predictions)

    verbose_rows = []
    prediction_ids = []
    for record, pred_index in zip(subtask1_eval, eval_predictions):
        pred_label = SUBTASK1_LABELS[int(pred_index)]
        prediction_ids.append(SUBTASK1_NAME_TO_ID[pred_label])
        verbose_rows.append(
            {
                "record_id": record.record_id,
                "gold_class_id": record.severity_id,
                "gold_class": SUBTASK1_ID_TO_NAME[record.severity_id],
                "pred_class_id": SUBTASK1_NAME_TO_ID[pred_label],
                "pred_class": pred_label,
            }
        )

    experiment_dir = output_dir / best_candidate.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_verbose_csv(experiment_dir / "subtask1_verbose.csv", verbose_rows)
    write_submission(experiment_dir / "submission" / "subtask1.csv", prediction_ids)

    payload = {
        "generated_at": json.loads(json.dumps(__import__("datetime").datetime.utcnow().isoformat() + "Z")),
        "runtime_seconds": perf_counter() - start_time,
        "subtask1_train_rows": len(subtask1_train),
        "subtask1_eval_rows": len(subtask1_eval),
        "subtask2_aux_rows": len(subtask2_train),
        "validation_rows": sorted(validation_rows, key=lambda row: (-row["macro_f1"], -row["accuracy"], -row["weighted_f1"])),
        "best_experiment": {
            "name": best_candidate.name,
            "classifier": best_candidate.classifier,
            "params": best_candidate.params,
            "use_type_scores": best_candidate.use_type_scores,
            "use_text_stats": best_candidate.use_text_stats,
            "validation_metrics": best_validation_metrics,
            "output_dir": str(experiment_dir),
        },
        "devel_metrics": devel_metrics,
    }

    report_json = Path(args.report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report_markdown = Path(args.report_markdown)
    report_markdown.parent.mkdir(parents=True, exist_ok=True)
    report_markdown.write_text("\n".join(report_lines(payload)) + "\n", encoding="utf-8")

    metrics_summary = {
        "model_name_or_path": best_candidate.name,
        "family": "competitive_severity",
        "description": "Severidad con TF-IDF palabra+char, ordinalidad y señal auxiliar de tipos.",
        "subtask1_split": args.subtask1_eval_split,
        "subtask1_records": len(subtask1_eval),
        "subtask1_metrics": devel_metrics,
        "selected_trial": payload["best_experiment"],
        "runtime_seconds": payload["runtime_seconds"],
    }
    (experiment_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()