"""Ensamble v2 de severidad: exploración masiva de combinaciones clásicas + transformers.

Mejoras sobre v1:
- Más clasificadores base (GradientBoosting sobre TF-IDF, RidgeClassifier, ExtraTrees)
- Más rangos de n-grams (char 2-4, char 4-6, word 1-3)
- Calibración de probabilidades (CalibratedClassifierCV)
- Búsqueda de pesos óptimos de votación con scipy.optimize
- Temperature scaling
- Stacking con XGBoost/LightGBM si disponibles
- Integración opcional de probabilidades de encoders (transformers)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from womenhelp_competition.data import load_subtask1, load_subtask2
from womenhelp_competition.labels import (
    SUBTASK1_ID_TO_NAME,
    SUBTASK1_LABELS,
    SUBTASK1_NAME_TO_ID,
    SUBTASK2_COLUMNS,
    SUBTASK2_LABELS,
)
from womenhelp_competition.metrics import compute_multiclass_metrics

SEED = 3407
NUM_CLASSES = len(SUBTASK1_LABELS)
NON_NA_LABELS = [label for label in SUBTASK2_LABELS if label != "N/A"]
NON_NA_COLUMNS = SUBTASK2_COLUMNS[: len(NON_NA_LABELS)]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaseSpec:
    name: str
    feature_family: str
    classifier: str
    params: dict[str, Any]
    use_type_scores: bool
    calibrate: bool = False


# ---------------------------------------------------------------------------
# Ordinal Model
# ---------------------------------------------------------------------------

class OrdinalSeverityModel:
    def __init__(self, *, c_value: float, class_weight: str | None, max_iter: int = 1200):
        self.models = [
            LogisticRegression(
                C=c_value, class_weight=class_weight, max_iter=max_iter,
                solver="lbfgs", random_state=SEED,
            )
            for _ in range(NUM_CLASSES - 1)
        ]

    def fit(self, features, labels: np.ndarray) -> "OrdinalSeverityModel":
        for threshold, model in enumerate(self.models):
            binary_labels = (labels > threshold).astype(np.int32)
            model.fit(features, binary_labels)
        return self

    def predict_proba(self, features) -> np.ndarray:
        ge_probs = np.column_stack([m.predict_proba(features)[:, 1] for m in self.models])
        monotonic = np.minimum.accumulate(ge_probs, axis=1)
        probs = np.zeros((features.shape[0], NUM_CLASSES), dtype=np.float64)
        probs[:, 0] = 1.0 - monotonic[:, 0]
        probs[:, 1] = monotonic[:, 0] - monotonic[:, 1]
        probs[:, 2] = monotonic[:, 1] - monotonic[:, 2]
        probs[:, 3] = monotonic[:, 2]
        np.clip(probs, 0.0, 1.0, out=probs)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return probs / row_sums

    def predict(self, features) -> np.ndarray:
        return self.predict_proba(features).argmax(axis=1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensamble v2 competitivo de severidad")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--subtask1-train-split", default="train")
    p.add_argument("--subtask1-eval-split", default="devel")
    p.add_argument("--subtask2-train-split", default="train")
    p.add_argument("--limit-subtask1-train", type=int, default=0)
    p.add_argument("--limit-subtask1-eval", type=int, default=0)
    p.add_argument("--limit-subtask2-train", type=int, default=0)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=SEED)
    p.add_argument("--encoder-probs-dir", default=None,
                    help="Dir with encoder verbose CSVs to fold into ensemble")
    p.add_argument("--output-dir",
                    default=str(REPO_ROOT / "outputs" / "severity_ensemble_v2" / datetime.now().strftime("%Y%m%d_%H%M")))
    p.add_argument("--report-json",
                    default=str(REPO_ROOT / "reports" / "baselines" / "severity_ensemble_v2.json"))
    p.add_argument("--report-markdown",
                    default=str(REPO_ROOT / "reports" / "baselines" / "severity_ensemble_v2.md"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def maybe_sample_records(records: list[Any], limit: int, random_state: int) -> list[Any]:
    items = list(records)
    if limit <= 0 or limit >= len(items):
        return items
    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(len(items), size=limit, replace=False))
    return [items[int(i)] for i in indices]


# ---------------------------------------------------------------------------
# Model specs — MASSIVELY expanded
# ---------------------------------------------------------------------------

def build_base_specs() -> list[BaseSpec]:
    specs = []

    # --- Word TF-IDF (1,2) ---
    specs.append(BaseSpec("word12_logreg_c2", "word12", "logreg",
                          {"C": 2.0, "class_weight": "balanced", "solver": "saga"}, False))
    specs.append(BaseSpec("word12_logreg_c5", "word12", "logreg",
                          {"C": 5.0, "class_weight": "balanced", "solver": "saga"}, False))
    specs.append(BaseSpec("word12_logreg_c1", "word12", "logreg",
                          {"C": 1.0, "class_weight": "balanced", "solver": "saga"}, False))
    specs.append(BaseSpec("word12_svm_c05", "word12", "linear_svc",
                          {"C": 0.5, "class_weight": "balanced"}, False))
    specs.append(BaseSpec("word12_svm_c1", "word12", "linear_svc",
                          {"C": 1.0, "class_weight": "balanced"}, False))
    specs.append(BaseSpec("word12_cnb_a02", "word12", "complement_nb",
                          {"alpha": 0.2}, False))
    specs.append(BaseSpec("word12_cnb_a05", "word12", "complement_nb",
                          {"alpha": 0.5}, False))
    specs.append(BaseSpec("word12_mnb_a03", "word12", "multinomial_nb",
                          {"alpha": 0.3}, False))
    specs.append(BaseSpec("word12_ridge", "word12", "ridge",
                          {"alpha": 1.0, "class_weight": "balanced"}, False))
    specs.append(BaseSpec("word12_sgd_mh", "word12", "sgd",
                          {"alpha": 3e-6, "class_weight": "balanced", "loss": "modified_huber"}, False))

    # --- Word TF-IDF (1,3) ---
    specs.append(BaseSpec("word13_logreg_c2", "word13", "logreg",
                          {"C": 2.0, "class_weight": "balanced", "solver": "saga"}, False))
    specs.append(BaseSpec("word13_svm_c05", "word13", "linear_svc",
                          {"C": 0.5, "class_weight": "balanced"}, False))
    specs.append(BaseSpec("word13_cnb_a02", "word13", "complement_nb",
                          {"alpha": 0.2}, False))

    # --- Char TF-IDF (3,5) ---
    specs.append(BaseSpec("char35_svm_c05", "char35", "linear_svc",
                          {"C": 0.5, "class_weight": "balanced"}, False))
    specs.append(BaseSpec("char35_svm_c1", "char35", "linear_svc",
                          {"C": 1.0, "class_weight": "balanced"}, False))
    specs.append(BaseSpec("char35_logreg_c2", "char35", "logreg",
                          {"C": 2.0, "class_weight": "balanced", "solver": "saga"}, False))

    # --- Char TF-IDF (2,4) ---
    specs.append(BaseSpec("char24_svm_c05", "char24", "linear_svc",
                          {"C": 0.5, "class_weight": "balanced"}, False))
    specs.append(BaseSpec("char24_logreg_c2", "char24", "logreg",
                          {"C": 2.0, "class_weight": "balanced", "solver": "saga"}, False))

    # --- Char TF-IDF (4,6) ---
    specs.append(BaseSpec("char46_svm_c05", "char46", "linear_svc",
                          {"C": 0.5, "class_weight": "balanced"}, False))
    specs.append(BaseSpec("char46_logreg_c2", "char46", "logreg",
                          {"C": 2.0, "class_weight": "balanced", "solver": "saga"}, False))

    # --- Word+Char union ---
    specs.append(BaseSpec("wc_logreg_c1", "word_char", "logreg",
                          {"C": 1.0, "class_weight": "balanced", "solver": "saga"}, False))
    specs.append(BaseSpec("wc_logreg_c4", "word_char", "logreg",
                          {"C": 4.0, "class_weight": "balanced", "solver": "saga"}, False))
    specs.append(BaseSpec("wc_svm_c05", "word_char", "linear_svc",
                          {"C": 0.5, "class_weight": "balanced"}, False))
    specs.append(BaseSpec("wc_svm_c1", "word_char", "linear_svc",
                          {"C": 1.0, "class_weight": "balanced"}, False))
    specs.append(BaseSpec("wc_sgd_mh", "word_char", "sgd",
                          {"alpha": 3e-6, "class_weight": "balanced", "loss": "modified_huber"}, False))
    specs.append(BaseSpec("wc_sgd_log", "word_char", "sgd",
                          {"alpha": 1e-6, "class_weight": "balanced", "loss": "log_loss"}, False))

    # --- Word+Char with type scores ---
    specs.append(BaseSpec("wc_logreg_c4_types", "word_char", "logreg",
                          {"C": 4.0, "class_weight": "balanced", "solver": "lbfgs"}, True))
    specs.append(BaseSpec("wc_logreg_c2_types", "word_char", "logreg",
                          {"C": 2.0, "class_weight": "balanced", "solver": "lbfgs"}, True))
    specs.append(BaseSpec("wc_logreg_c8_types", "word_char", "logreg",
                          {"C": 8.0, "class_weight": "balanced", "solver": "lbfgs"}, True))
    specs.append(BaseSpec("wc_ordinal_c4_types", "word_char", "ordinal_logreg",
                          {"C": 4.0, "class_weight": "balanced"}, True))
    specs.append(BaseSpec("wc_ordinal_c2_types", "word_char", "ordinal_logreg",
                          {"C": 2.0, "class_weight": "balanced"}, True))
    specs.append(BaseSpec("wc_ordinal_c8_types", "word_char", "ordinal_logreg",
                          {"C": 8.0, "class_weight": "balanced"}, True))
    specs.append(BaseSpec("wc_svm_c1_types", "word_char", "linear_svc",
                          {"C": 1.0, "class_weight": "balanced"}, True))

    # --- Random Forest / ExtraTrees on dense TF-IDF (truncated) ---
    specs.append(BaseSpec("wc_rf_200", "word_char_dense", "random_forest",
                          {"n_estimators": 200, "class_weight": "balanced_subsample", "max_features": "sqrt"}, False))
    specs.append(BaseSpec("wc_et_200", "word_char_dense", "extra_trees",
                          {"n_estimators": 200, "class_weight": "balanced_subsample", "max_features": "sqrt"}, False))

    # --- Gradient Boosting on stacked type scores + TF-IDF (dense) ---
    specs.append(BaseSpec("wc_gb_200_types", "word_char_dense", "gradient_boosting",
                          {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "subsample": 0.8}, True))

    # --- KNN on TF-IDF (dense) ---
    specs.append(BaseSpec("wc_knn_15", "word_char_dense", "knn",
                          {"n_neighbors": 15, "weights": "distance"}, False))

    # --- Calibrated SVC ---
    specs.append(BaseSpec("wc_cal_svm_c05", "word_char", "calibrated_svc",
                          {"C": 0.5, "class_weight": "balanced"}, False))

    return specs


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_vectorizer(family: str) -> Any:
    if family == "word12":
        return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.98,
                               sublinear_tf=True, norm="l2")
    if family == "word13":
        return TfidfVectorizer(lowercase=True, ngram_range=(1, 3), min_df=2, max_df=0.98,
                               sublinear_tf=True, norm="l2")
    if family == "char35":
        return TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
                               min_df=2, sublinear_tf=True, norm="l2")
    if family == "char24":
        return TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 4),
                               min_df=2, sublinear_tf=True, norm="l2")
    if family == "char46":
        return TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(4, 6),
                               min_df=2, sublinear_tf=True, norm="l2")
    if family in ("word_char", "word_char_dense"):
        return FeatureUnion([
            ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                     max_df=0.98, sublinear_tf=True, norm="l2")),
            ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=2, sublinear_tf=True, norm="l2")),
        ])
    raise ValueError(f"Unknown feature family: {family}")


def writable_csr(m):
    c = m.tocsr(copy=True)
    c.sort_indices()
    return c


def build_feature_bundle(family: str, train_texts, valid_texts, eval_texts):
    vec = build_vectorizer(family)
    train_m = vec.fit_transform(train_texts)
    valid_m = vec.transform(valid_texts) if valid_texts is not None else None
    eval_m = vec.transform(eval_texts) if eval_texts is not None else None
    if family.endswith("_dense"):
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=300, random_state=SEED)
        train_m = svd.fit_transform(train_m)
        valid_m = svd.transform(valid_m) if valid_m is not None else None
        eval_m = svd.transform(eval_m) if eval_m is not None else None
        return {"train": train_m, "valid": valid_m, "eval": eval_m}
    return {
        "train": writable_csr(train_m),
        "valid": writable_csr(valid_m) if valid_m is not None else None,
        "eval": writable_csr(eval_m) if eval_m is not None else None,
    }


def build_all_feature_bundles(train_texts, valid_texts, eval_texts):
    families = sorted({s.feature_family for s in build_base_specs()})
    return {f: build_feature_bundle(f, train_texts, valid_texts, eval_texts) for f in families}


# ---------------------------------------------------------------------------
# Type aux model
# ---------------------------------------------------------------------------

def fit_auxiliary_type_model(texts, label_matrix):
    vec = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                 max_df=0.98, sublinear_tf=True, norm="l2")),
        ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=2, sublinear_tf=True, norm="l2")),
    ])
    X = vec.fit_transform(texts)
    clf = OneVsRestClassifier(
        LogisticRegression(C=2.0, class_weight="balanced", max_iter=1200,
                           solver="liblinear", random_state=SEED)
    )
    clf.fit(X, label_matrix)
    return vec, clf


def build_type_scores(texts, vec, clf):
    X = vec.transform(texts)
    probs = clf.predict_proba(X)
    max_p = probs.max(axis=1, keepdims=True)
    mean_p = probs.mean(axis=1, keepdims=True)
    count_over = (probs > 0.5).sum(axis=1, keepdims=True).astype(np.float64)
    na_proxy = 1.0 - max_p
    return np.hstack([probs, max_p, mean_p, count_over, na_proxy])


def compose_features(text_features, *, type_scores=None, type_scaler=None):
    blocks = [text_features]
    if type_scores is not None and type_scaler is not None:
        scaled = type_scaler.transform(type_scores)
        if sparse.issparse(text_features):
            blocks.append(sparse.csr_matrix(scaled))
            return sparse.hstack(blocks, format="csr")
        else:
            return np.hstack([text_features, scaled])
    if sparse.issparse(text_features):
        return sparse.hstack(blocks, format="csr")
    return text_features


# ---------------------------------------------------------------------------
# Estimator factory
# ---------------------------------------------------------------------------

def build_estimator(spec: BaseSpec):
    c = spec.classifier
    p = spec.params
    if c == "complement_nb":
        return ComplementNB(alpha=float(p["alpha"]))
    if c == "multinomial_nb":
        return MultinomialNB(alpha=float(p["alpha"]))
    if c == "logreg":
        return LogisticRegression(C=float(p["C"]), class_weight=p.get("class_weight"),
                                  max_iter=1500, solver=p.get("solver", "lbfgs"), random_state=SEED)
    if c == "linear_svc":
        return LinearSVC(C=float(p["C"]), class_weight=p.get("class_weight"),
                         dual="auto", max_iter=2000, random_state=SEED)
    if c == "calibrated_svc":
        base = LinearSVC(C=float(p["C"]), class_weight=p.get("class_weight"),
                         dual="auto", max_iter=2000, random_state=SEED)
        return CalibratedClassifierCV(base, cv=3, method="sigmoid")
    if c == "sgd":
        return SGDClassifier(loss=p["loss"], alpha=float(p["alpha"]),
                             class_weight=p.get("class_weight"), max_iter=2000,
                             tol=1e-3, random_state=SEED)
    if c == "ridge":
        return RidgeClassifier(alpha=float(p["alpha"]), class_weight=p.get("class_weight"))
    if c == "ordinal_logreg":
        return OrdinalSeverityModel(c_value=float(p["C"]), class_weight=p.get("class_weight"))
    if c == "random_forest":
        return RandomForestClassifier(n_estimators=int(p["n_estimators"]),
                                      class_weight=p.get("class_weight"),
                                      max_features=p.get("max_features", "sqrt"),
                                      random_state=SEED, n_jobs=-1)
    if c == "extra_trees":
        return ExtraTreesClassifier(n_estimators=int(p["n_estimators"]),
                                    class_weight=p.get("class_weight"),
                                    max_features=p.get("max_features", "sqrt"),
                                    random_state=SEED, n_jobs=-1)
    if c == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=int(p["n_estimators"]),
                                          max_depth=int(p.get("max_depth", 5)),
                                          learning_rate=float(p.get("learning_rate", 0.1)),
                                          subsample=float(p.get("subsample", 0.8)),
                                          random_state=SEED)
    if c == "knn":
        return KNeighborsClassifier(n_neighbors=int(p["n_neighbors"]),
                                    weights=p.get("weights", "distance"), n_jobs=-1)
    raise ValueError(f"Unknown classifier: {c}")


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------

def softmax_rows(v: np.ndarray) -> np.ndarray:
    shifted = v - v.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True).clip(1e-9)


def ensure_probs(p: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(p, dtype=np.float64), 1e-9, None)
    return c / c.sum(axis=1, keepdims=True).clip(1e-9)


def predict_probs(model, spec, X):
    if spec.classifier in ("linear_svc", "ridge"):
        d = np.asarray(model.decision_function(X), dtype=np.float64)
        if d.ndim == 1:
            d = d[:, np.newaxis]
        return ensure_probs(softmax_rows(d))
    if hasattr(model, "predict_proba"):
        return ensure_probs(np.asarray(model.predict_proba(X), dtype=np.float64))
    raise ValueError(f"No predict_proba for {spec.name}")


def probs_to_ids(p):
    return np.asarray(p).argmax(axis=1).astype(np.int32)


def calc_metrics_from_ids(gold, pred_ids):
    gold_names = [SUBTASK1_LABELS[int(i)] for i in gold]
    pred_names = [SUBTASK1_LABELS[int(i)] for i in pred_ids]
    return compute_multiclass_metrics(gold_names, pred_names, SUBTASK1_LABELS)


def calc_metrics(gold, probs):
    pred_ids = probs_to_ids(probs)
    return calc_metrics_from_ids(gold, pred_ids)


def ordinal_tail_probs(probs: np.ndarray) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float64)
    return np.column_stack([
        values[:, 1:].sum(axis=1),
        values[:, 2:].sum(axis=1),
        values[:, 3],
    ])


def ordinal_threshold_ids(probs: np.ndarray, thresholds: list[float]) -> np.ndarray:
    tails = ordinal_tail_probs(probs)
    pred_ids = np.zeros(tails.shape[0], dtype=np.int32)
    for threshold_index, threshold_value in enumerate(thresholds):
        pred_ids += (tails[:, threshold_index] >= float(threshold_value)).astype(np.int32)
    return pred_ids


def metric_priority_tuple(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    severe_scores = metrics["per_label"].get("Severe", {})
    return (
        float(metrics["macro_f1"]),
        float(severe_scores.get("recall", 0.0)),
        float(metrics["accuracy"]),
        float(metrics["weighted_f1"]),
    )


def search_ordinal_thresholds(gold, probs: np.ndarray) -> tuple[list[float], np.ndarray, dict[str, Any]]:
    candidate_values = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    best_pred_ids = probs_to_ids(probs)
    best_metrics = calc_metrics_from_ids(gold, best_pred_ids)
    best_thresholds = [0.50, 0.50, 0.50]
    best_key = metric_priority_tuple(best_metrics)

    for threshold_0 in candidate_values:
        for threshold_1 in candidate_values:
            for threshold_2 in candidate_values:
                thresholds = [threshold_0, threshold_1, threshold_2]
                pred_ids = ordinal_threshold_ids(probs, thresholds)
                metrics = calc_metrics_from_ids(gold, pred_ids)
                metric_key = metric_priority_tuple(metrics)
                if metric_key > best_key:
                    best_key = metric_key
                    best_thresholds = thresholds
                    best_pred_ids = pred_ids
                    best_metrics = metrics

    return best_thresholds, best_pred_ids, best_metrics


def resolve_cv_folds(labels, requested):
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    pos = counts[counts > 0]
    return max(2, min(requested, int(pos.min())))


# ---------------------------------------------------------------------------
# OOF probability generation
# ---------------------------------------------------------------------------

def fit_oof_probabilities(specs, train_ids, train_texts, train_labels,
                          subtask2_train, cv_folds, random_state):
    n = len(train_texts)
    oof = {s.name: np.zeros((n, NUM_CLASSES), dtype=np.float64) for s in specs}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    use_types = any(s.use_type_scores for s in specs)

    for fold_i, (fit_idx, val_idx) in enumerate(skf.split(train_texts, train_labels), 1):
        print(f"  Fold {fold_i}/{cv_folds} ...", flush=True)
        fit_texts = [train_texts[i] for i in fit_idx]
        val_texts = [train_texts[i] for i in val_idx]
        bundles = build_all_feature_bundles(fit_texts, val_texts, None)

        fit_type_scores = val_type_scores = None
        if use_types:
            excluded = {train_ids[i] for i in val_idx}
            aux_recs = [r for r in subtask2_train if r.record_id not in excluded]
            if aux_recs:
                aux_texts = [r.text for r in aux_recs]
                aux_mat = np.asarray(
                    [[int(r.label_vector[c]) for c in NON_NA_COLUMNS] for r in aux_recs],
                    dtype=np.int32,
                )
                aux_vec, aux_clf = fit_auxiliary_type_model(aux_texts, aux_mat)
                fit_type_scores = build_type_scores(fit_texts, aux_vec, aux_clf)
                val_type_scores = build_type_scores(val_texts, aux_vec, aux_clf)

        for spec in specs:
            try:
                fit_X = bundles[spec.feature_family]["train"]
                val_X = bundles[spec.feature_family]["valid"]
                if spec.use_type_scores and fit_type_scores is not None:
                    scaler = StandardScaler()
                    scaler.fit(fit_type_scores)
                    fit_X = compose_features(fit_X, type_scores=fit_type_scores, type_scaler=scaler)
                    val_X = compose_features(val_X, type_scores=val_type_scores, type_scaler=scaler)
                model = build_estimator(spec)
                model.fit(fit_X, train_labels[fit_idx])
                oof[spec.name][val_idx] = predict_probs(model, spec, val_X)
            except Exception as e:
                print(f"    WARN: {spec.name} fold {fold_i} failed: {e}", flush=True)
    return oof


def fit_full_probabilities(specs, train_texts, train_labels, eval_texts, subtask2_train):
    bundles = build_all_feature_bundles(train_texts, None, eval_texts)
    use_types = any(s.use_type_scores for s in specs)

    train_type_scores = eval_type_scores = None
    if use_types:
        aux_texts = [r.text for r in subtask2_train]
        aux_mat = np.asarray(
            [[int(r.label_vector[c]) for c in NON_NA_COLUMNS] for r in subtask2_train],
            dtype=np.int32,
        )
        aux_vec, aux_clf = fit_auxiliary_type_model(aux_texts, aux_mat)
        train_type_scores = build_type_scores(train_texts, aux_vec, aux_clf)
        eval_type_scores = build_type_scores(eval_texts, aux_vec, aux_clf)

    eval_probs = {}
    for spec in specs:
        try:
            train_X = bundles[spec.feature_family]["train"]
            eval_X = bundles[spec.feature_family]["eval"]
            if spec.use_type_scores and train_type_scores is not None:
                scaler = StandardScaler()
                scaler.fit(train_type_scores)
                train_X = compose_features(train_X, type_scores=train_type_scores, type_scaler=scaler)
                eval_X = compose_features(eval_X, type_scores=eval_type_scores, type_scaler=scaler)
            model = build_estimator(spec)
            model.fit(train_X, train_labels)
            eval_probs[spec.name] = predict_probs(model, spec, eval_X)
        except Exception as e:
            print(f"  WARN: {spec.name} full fit failed: {e}", flush=True)
    return eval_probs


# ---------------------------------------------------------------------------
# Ensemble combiners
# ---------------------------------------------------------------------------

def avg_probs(prob_list, weights=None):
    stacked = np.stack(prob_list, axis=0)
    if weights is None:
        return ensure_probs(stacked.mean(axis=0))
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum().clip(1e-9)
    return ensure_probs(np.tensordot(w, stacked, axes=(0, 0)))


def classwise_avg(prob_list, cw):
    stacked = np.stack(prob_list, axis=0)
    weighted = stacked * cw[:, np.newaxis, :]
    denom = cw.sum(axis=0, keepdims=True).clip(1e-9)
    return ensure_probs(weighted.sum(axis=0) / denom)


def temperature_scale(probs, T):
    log_p = np.log(probs.clip(1e-9))
    scaled = log_p / T
    return ensure_probs(softmax_rows(scaled))


def optimize_weights(oof_list, gold_labels, n_restarts=5):
    """Find optimal weights for averaging via Nelder-Mead on OOF."""
    n = len(oof_list)
    best_score = -1.0
    best_w = np.ones(n) / n

    def neg_f1(w):
        w_pos = np.abs(w)
        w_norm = w_pos / w_pos.sum().clip(1e-9)
        combined = avg_probs(oof_list, weights=w_norm)
        m = calc_metrics(gold_labels, combined)
        return -m["macro_f1"]

    rng = np.random.default_rng(SEED)
    for _ in range(n_restarts):
        w0 = rng.dirichlet(np.ones(n))
        result = minimize(neg_f1, w0, method="Nelder-Mead",
                          options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-5})
        if -result.fun > best_score:
            best_score = -result.fun
            w_abs = np.abs(result.x)
            best_w = w_abs / w_abs.sum().clip(1e-9)
    return best_w, best_score


def optimize_temperature(probs, gold_labels):
    """Find optimal temperature for a single model's probabilities."""
    best_T = 1.0
    best_f1 = -1.0
    for T in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0]:
        scaled = temperature_scale(probs, T)
        m = calc_metrics(gold_labels, scaled)
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_T = T
    return best_T, best_f1


# ---------------------------------------------------------------------------
# Load encoder probabilities (optional)
# ---------------------------------------------------------------------------

def load_encoder_probs(encoder_dir: str, eval_ids: list[str]) -> dict[str, np.ndarray]:
    """Load probability columns from encoder verbose CSVs."""
    result = {}
    p = Path(encoder_dir)
    for csv_file in sorted(p.glob("*/subtask1_verbose.csv")):
        name = csv_file.parent.name
        id_to_probs = {}
        with csv_file.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row["record_id"]
                probs = [float(row[f"prob_{l.lower()}"]) for l in SUBTASK1_LABELS]
                id_to_probs[rid] = probs

        if not id_to_probs:
            continue
        prob_matrix = np.zeros((len(eval_ids), NUM_CLASSES), dtype=np.float64)
        valid = True
        for i, rid in enumerate(eval_ids):
            if rid in id_to_probs:
                prob_matrix[i] = id_to_probs[rid]
            else:
                valid = False
                break
        if valid:
            result[f"enc_{name}"] = ensure_probs(prob_matrix)
            print(f"  Loaded encoder probs: {name} ({len(id_to_probs)} records)")
    return result


# ---------------------------------------------------------------------------
# Stacking meta-learners
# ---------------------------------------------------------------------------

def stack_features(prob_map, model_order):
    return np.hstack([prob_map[n] for n in model_order])


def meta_oof(train_X, train_y, kind, c_val, cw, cv_folds, rs):
    oof = np.zeros((train_X.shape[0], NUM_CLASSES), dtype=np.float64)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rs)
    for fit_idx, val_idx in skf.split(train_X, train_y):
        m = _build_meta(kind, c_val, cw)
        m.fit(train_X[fit_idx], train_y[fit_idx])
        oof[val_idx] = ensure_probs(m.predict_proba(train_X[val_idx]))
    return oof


def meta_predict(train_X, train_y, eval_X, kind, c_val, cw):
    m = _build_meta(kind, c_val, cw)
    m.fit(train_X, train_y)
    return ensure_probs(m.predict_proba(eval_X))


def _build_meta(kind, c_val, cw):
    if kind == "logreg":
        return LogisticRegression(C=c_val, class_weight=cw, max_iter=2000,
                                  solver="lbfgs", random_state=SEED)
    if kind == "ordinal":
        return OrdinalSeverityModel(c_value=c_val, class_weight=cw)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_submission(path, pred_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        for pid in pred_ids:
            w.writerow([pid])


def write_verbose_csv(path, records, probs, pred_ids=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["record_id", "gold_class_id", "gold_class", "pred_class_id", "pred_class",
              "confidence", "margin"] + [f"prob_{l.lower()}" for l in SUBTASK1_LABELS]
    if pred_ids is None:
        pred_ids = probs_to_ids(probs)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec, p, pid in zip(records, probs, pred_ids):
            sp = np.sort(p)
            margin = float(sp[-1] - sp[-2]) if len(sp) > 1 else 0.0
            row = {
                "record_id": rec.record_id,
                "gold_class_id": rec.severity_id,
                "gold_class": SUBTASK1_ID_TO_NAME[rec.severity_id],
                "pred_class_id": str(pid),
                "pred_class": SUBTASK1_LABELS[int(pid)],
                "confidence": float(p[int(pid)]),
                "margin": margin,
            }
            for li, ln in enumerate(SUBTASK1_LABELS):
                row[f"prob_{ln.lower()}"] = float(p[li])
            w.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    t0 = perf_counter()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...", flush=True)
    train_recs = maybe_sample_records(
        load_subtask1(args.subtask1_train_split, args.data_dir),
        args.limit_subtask1_train, args.random_state)
    eval_recs = maybe_sample_records(
        load_subtask1(args.subtask1_eval_split, args.data_dir),
        args.limit_subtask1_eval, args.random_state + 1)
    st2_train = maybe_sample_records(
        load_subtask2(args.subtask2_train_split, args.data_dir),
        args.limit_subtask2_train, args.random_state + 2)

    train_ids = [r.record_id for r in train_recs]
    train_texts = [r.text for r in train_recs]
    train_labels = np.asarray([int(r.severity_id) for r in train_recs], dtype=np.int32)
    eval_ids = [r.record_id for r in eval_recs]
    eval_texts = [r.text for r in eval_recs]
    eval_labels = np.asarray([int(r.severity_id) for r in eval_recs], dtype=np.int32)

    print(f"Train: {len(train_recs)}, Eval: {len(eval_recs)}, ST2: {len(st2_train)}", flush=True)

    specs = build_base_specs()
    cv_folds = resolve_cv_folds(train_labels, args.cv_folds)

    print(f"Fitting {len(specs)} base models with {cv_folds}-fold CV ...", flush=True)
    oof = fit_oof_probabilities(specs, train_ids, train_texts, train_labels,
                                st2_train, cv_folds, args.random_state)
    print("Fitting full base models on all train ...", flush=True)
    eval_probs = fit_full_probabilities(specs, train_texts, train_labels,
                                        eval_texts, st2_train)

    # Load encoder probs if available
    enc_probs = {}
    if args.encoder_probs_dir and Path(args.encoder_probs_dir).is_dir():
        print(f"Loading encoder probabilities from {args.encoder_probs_dir} ...", flush=True)
        enc_probs = load_encoder_probs(args.encoder_probs_dir, eval_ids)

    # --- Evaluate base models ---
    base_rows = []
    base_oof_metrics = {}
    for spec in specs:
        if spec.name not in oof or spec.name not in eval_probs:
            continue
        om = calc_metrics(train_labels, oof[spec.name])
        dm = calc_metrics(eval_labels, eval_probs[spec.name])
        base_oof_metrics[spec.name] = om
        base_rows.append({
            "name": spec.name, "use_type_scores": spec.use_type_scores,
            "oof_metrics": om, "devel_metrics": dm,
            "oof_macro_f1": om["macro_f1"], "devel_macro_f1": dm["macro_f1"],
            "devel_accuracy": dm["accuracy"], "devel_weighted_f1": dm["weighted_f1"],
        })

    model_names = [r["name"] for r in base_rows]
    print(f"\n=== {len(model_names)} base models evaluated ===", flush=True)
    for r in sorted(base_rows, key=lambda x: -x["devel_macro_f1"])[:10]:
        print(f"  {r['name']:40s} oof={r['oof_macro_f1']:.4f}  devel={r['devel_macro_f1']:.4f}", flush=True)

    # --- Build ensemble variants ---
    oof_list_all = [oof[n] for n in model_names]
    eval_list_all = [eval_probs[n] for n in model_names]
    ensemble_rows = []
    ensemble_eval_probs = {}
    ensemble_eval_pred_ids = {}

    def add_ens(name, etype, train_p, eval_p, comps):
        tm = calc_metrics(train_labels, train_p)
        dm = calc_metrics(eval_labels, eval_p)
        eval_pred_ids = probs_to_ids(eval_p)
        ensemble_rows.append({
            "name": name, "ensemble_type": etype, "components": comps,
            "oof_metrics": tm, "devel_metrics": dm,
            "oof_macro_f1": tm["macro_f1"], "devel_macro_f1": dm["macro_f1"],
            "devel_accuracy": dm["accuracy"], "devel_weighted_f1": dm["weighted_f1"],
            "calibration": None,
        })
        ensemble_eval_probs[name] = eval_p
        ensemble_eval_pred_ids[name] = eval_pred_ids

        cal_thresholds, cal_train_ids, cal_train_metrics = search_ordinal_thresholds(train_labels, train_p)
        cal_eval_ids = ordinal_threshold_ids(eval_p, cal_thresholds)
        cal_eval_metrics = calc_metrics_from_ids(eval_labels, cal_eval_ids)
        cal_name = f"{name}_ordcal"
        ensemble_rows.append({
            "name": cal_name,
            "ensemble_type": f"{etype}_ordinal_thresholds",
            "components": comps,
            "oof_metrics": cal_train_metrics,
            "devel_metrics": cal_eval_metrics,
            "oof_macro_f1": cal_train_metrics["macro_f1"],
            "devel_macro_f1": cal_eval_metrics["macro_f1"],
            "devel_accuracy": cal_eval_metrics["accuracy"],
            "devel_weighted_f1": cal_eval_metrics["weighted_f1"],
            "calibration": {"type": "ordinal_thresholds", "thresholds": cal_thresholds},
        })
        ensemble_eval_probs[cal_name] = eval_p
        ensemble_eval_pred_ids[cal_name] = cal_eval_ids

    # 1. Simple mean all
    add_ens("mean_all", "soft_vote",
            avg_probs(oof_list_all), avg_probs(eval_list_all), model_names)

    # 2. Macro-weighted all
    macro_w = np.array([base_oof_metrics[n]["macro_f1"] for n in model_names])
    add_ens("macro_weighted_all", "weighted_vote",
            avg_probs(oof_list_all, macro_w), avg_probs(eval_list_all, macro_w), model_names)

    # 3. Classwise F1 weighted
    cw = np.array([
        [base_oof_metrics[n]["per_label"][l]["f1"] + 1e-6 for l in SUBTASK1_LABELS]
        for n in model_names
    ])
    add_ens("classwise_f1_all", "classwise_vote",
            classwise_avg(oof_list_all, cw), classwise_avg(eval_list_all, cw), model_names)

    # 4. Top-K subsets
    sorted_by_oof = sorted(base_rows, key=lambda x: -x["oof_macro_f1"])
    for k in [3, 4, 5, 6, 8]:
        top_names = [r["name"] for r in sorted_by_oof[:k]]
        top_oof = [oof[n] for n in top_names]
        top_eval = [eval_probs[n] for n in top_names]
        add_ens(f"mean_top{k}", "soft_vote",
                avg_probs(top_oof), avg_probs(top_eval), top_names)

        # Weighted top-K
        tw = np.array([base_oof_metrics[n]["macro_f1"] for n in top_names])
        add_ens(f"weighted_top{k}", "weighted_vote",
                avg_probs(top_oof, tw), avg_probs(top_eval, tw), top_names)

    # 5. Optimized weights on top-K
    for k in [4, 6, 8]:
        top_names = [r["name"] for r in sorted_by_oof[:k]]
        top_oof = [oof[n] for n in top_names]
        top_eval = [eval_probs[n] for n in top_names]
        opt_w, opt_score = optimize_weights(top_oof, train_labels)
        print(f"  Optimized top{k}: oof_macro_f1={opt_score:.4f} weights={opt_w.round(3)}", flush=True)
        add_ens(f"optimized_top{k}", "optimized_vote",
                avg_probs(top_oof, opt_w), avg_probs(top_eval, opt_w), top_names)

    # 6. Anchor-based pairs and triples
    anchor = sorted_by_oof[0]["name"]
    partners = [n for n in model_names if n != anchor]
    for partner in partners:
        pair = [anchor, partner]
        add_ens(f"anchor_pair_{partner}", "anchor_pair",
                avg_probs([oof[n] for n in pair]),
                avg_probs([eval_probs[n] for n in pair]), pair)

    # Best anchor triples
    anchor_support = [r["name"] for r in sorted_by_oof[1:] if r["name"] != anchor][:4]
    for i in range(len(anchor_support)):
        for j in range(i + 1, len(anchor_support)):
            triple = [anchor, anchor_support[i], anchor_support[j]]
            add_ens(f"anchor_triple_{anchor_support[i]}_{anchor_support[j]}", "anchor_triple",
                    avg_probs([oof[n] for n in triple]),
                    avg_probs([eval_probs[n] for n in triple]), triple)

    # 7. Temperature scaling on best model
    best_base_name = sorted_by_oof[0]["name"]
    best_T, _ = optimize_temperature(oof[best_base_name], train_labels)
    if best_T != 1.0:
        add_ens(f"temp_{best_base_name}_T{best_T}", "temperature",
                temperature_scale(oof[best_base_name], best_T),
                temperature_scale(eval_probs[best_base_name], best_T),
                [best_base_name])

    # 8. Stacking meta-learners
    stack_train_X = stack_features(oof, model_names)
    stack_eval_X = stack_features(eval_probs, model_names)
    meta_folds = resolve_cv_folds(train_labels, cv_folds)
    meta_configs = [
        ("stack_logreg_c1", "logreg", 1.0, "balanced"),
        ("stack_logreg_c4", "logreg", 4.0, "balanced"),
        ("stack_logreg_c10", "logreg", 10.0, "balanced"),
        ("stack_logreg_c05", "logreg", 0.5, "balanced"),
        ("stack_ordinal_c1", "ordinal", 1.0, "balanced"),
        ("stack_ordinal_c4", "ordinal", 4.0, "balanced"),
    ]
    for mname, mkind, mc, mcw in meta_configs:
        try:
            tp = meta_oof(stack_train_X, train_labels, mkind, mc, mcw, meta_folds, args.random_state + 97)
            ep = meta_predict(stack_train_X, train_labels, stack_eval_X, mkind, mc, mcw)
            add_ens(mname, "stacking", tp, ep, model_names)
        except Exception as e:
            print(f"  WARN: {mname} failed: {e}", flush=True)

    # 9. Stacking on top-K only
    for k in [5, 8]:
        top_names = [r["name"] for r in sorted_by_oof[:k]]
        st_train = stack_features(oof, top_names)
        st_eval = stack_features(eval_probs, top_names)
        for mname_suffix, mkind, mc in [("logreg_c4", "logreg", 4.0), ("ordinal_c4", "ordinal", 4.0)]:
            try:
                tp = meta_oof(st_train, train_labels, mkind, mc, "balanced", meta_folds, args.random_state + 97)
                ep = meta_predict(st_train, train_labels, st_eval, mkind, mc, "balanced")
                add_ens(f"stack_top{k}_{mname_suffix}", "stacking", tp, ep, top_names)
            except Exception as e:
                print(f"  WARN: stack_top{k}_{mname_suffix} failed: {e}", flush=True)

    # 10. Include encoder probs in ensemble if available
    if enc_probs:
        all_names_with_enc = model_names + list(enc_probs.keys())
        all_eval_with_enc = eval_list_all + [enc_probs[n] for n in enc_probs]
        add_ens("mean_all_with_enc", "soft_vote_enc",
                avg_probs(oof_list_all),  # no OOF for encoders, use classical only for train estimate
                avg_probs(all_eval_with_enc),
                all_names_with_enc)
        # Weighted with encoders weighted higher
        enc_w = np.ones(len(all_eval_with_enc))
        enc_w[len(model_names):] = 2.0  # boost encoder weight
        add_ens("weighted_with_enc", "weighted_vote_enc",
                avg_probs(oof_list_all),
                avg_probs(all_eval_with_enc, enc_w),
                all_names_with_enc)

    # --- Sort and select best ---
    sorted_base = sorted(base_rows, key=lambda x: (-x["devel_macro_f1"], -x["devel_accuracy"]))
    sorted_ens = sorted(ensemble_rows, key=lambda x: (-x["devel_macro_f1"], -x["devel_accuracy"]))

    best_base = sorted_base[0]
    best_ens = sorted_ens[0]

    print(f"\n=== RESULTS ===", flush=True)
    print(f"Best base:     {best_base['name']:40s} devel_macro_f1={best_base['devel_macro_f1']:.4f}", flush=True)
    print(f"Best ensemble: {best_ens['name']:40s} devel_macro_f1={best_ens['devel_macro_f1']:.4f}", flush=True)
    print(f"Delta: {best_ens['devel_macro_f1'] - best_base['devel_macro_f1']:+.4f}", flush=True)

    print(f"\nTop 15 ensembles:", flush=True)
    for r in sorted_ens[:15]:
        print(f"  {r['name']:45s} oof={r['oof_macro_f1']:.4f}  devel={r['devel_macro_f1']:.4f}", flush=True)

    # --- Write outputs ---
    best_dir = out_dir / best_ens["name"]
    best_dir.mkdir(parents=True, exist_ok=True)
    best_p = ensemble_eval_probs[best_ens["name"]]
    pred_ids = ensemble_eval_pred_ids[best_ens["name"]]
    write_submission(best_dir / "submission" / "subtask1.csv",
                     [str(int(i)) for i in pred_ids])
    write_verbose_csv(best_dir / "subtask1_verbose.csv", eval_recs, best_p, pred_ids)

    # Also write top 3 ensembles
    for r in sorted_ens[:3]:
        edir = out_dir / r["name"]
        edir.mkdir(parents=True, exist_ok=True)
        ep = ensemble_eval_probs[r["name"]]
        eids = ensemble_eval_pred_ids[r["name"]]
        write_verbose_csv(edir / "subtask1_verbose.csv", eval_recs, ep, eids)
        write_submission(edir / "submission" / "subtask1.csv",
                         [str(int(i)) for i in eids])
        (edir / "metrics_summary.json").write_text(
            json.dumps({
                "name": r["name"], "devel_metrics": r["devel_metrics"],
                "oof_metrics": r["oof_metrics"],
                "calibration": r["calibration"],
            }, indent=2, ensure_ascii=False))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": perf_counter() - t0,
        "cv_folds": cv_folds,
        "train_rows": len(train_recs),
        "eval_rows": len(eval_recs),
        "subtask2_aux_rows": len(st2_train),
        "num_base_models": len(base_rows),
        "num_ensemble_variants": len(ensemble_rows),
        "base_rows": sorted_base,
        "ensemble_rows": sorted_ens,
        "best_base": best_base,
        "best_ensemble": {**best_ens, "output_dir": str(best_dir)},
    }

    rj = Path(args.report_json)
    rj.parent.mkdir(parents=True, exist_ok=True)
    rj.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    rm = Path(args.report_markdown)
    rm.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Ensamble v2 competitivo de severidad",
        "",
        f"Generado: {payload['generated_at']}",
        f"Runtime: {payload['runtime_seconds']:.1f}s",
        f"Base models: {payload['num_base_models']}, Ensemble variants: {payload['num_ensemble_variants']}",
        "",
        "## Top 10 modelos base (devel macro F1)",
        "",
        "| modelo | tipos | oof_f1 | devel_f1 | devel_acc |",
        "|---|---|---:|---:|---:|",
    ]
    for r in sorted_base[:10]:
        lines.append(f"| {r['name']} | {'si' if r['use_type_scores'] else 'no'} | "
                     f"{r['oof_macro_f1']:.4f} | {r['devel_macro_f1']:.4f} | {r['devel_accuracy']:.4f} |")

    lines.extend([
        "",
        "## Top 15 ensambles (devel macro F1)",
        "",
        "| variante | tipo | oof_f1 | devel_f1 | devel_acc | devel_wf1 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for r in sorted_ens[:15]:
        lines.append(f"| {r['name']} | {r['ensemble_type']} | "
                     f"{r['oof_macro_f1']:.4f} | {r['devel_macro_f1']:.4f} | "
                     f"{r['devel_accuracy']:.4f} | {r['devel_weighted_f1']:.4f} |")

    lines.extend([
        "",
        f"**Mejor base**: {best_base['name']} → macro F1 = {best_base['devel_macro_f1']:.4f}",
        f"**Mejor ensamble**: {best_ens['name']} → macro F1 = {best_ens['devel_macro_f1']:.4f}",
        f"**Delta**: {best_ens['devel_macro_f1'] - best_base['devel_macro_f1']:+.4f}",
    ])
    rm.write_text("\n".join(lines) + "\n")

    summary = {
        "model_name_or_path": best_ens["name"],
        "family": "severity_ensemble_v2",
        "subtask1_split": args.subtask1_eval_split,
        "subtask1_records": len(eval_recs),
        "subtask1_metrics": best_ens["devel_metrics"],
        "best_base": best_base,
        "selected_ensemble": payload["best_ensemble"],
        "runtime_seconds": payload["runtime_seconds"],
    }
    (best_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nDone in {perf_counter() - t0:.1f}s. Output: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
