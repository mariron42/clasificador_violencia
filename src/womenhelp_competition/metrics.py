from __future__ import annotations

from typing import Iterable


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_multiclass_metrics(
    golds: Iterable[str],
    preds: Iterable[str],
    labels: list[str],
) -> dict[str, object]:
    gold_list = list(golds)
    pred_list = list(preds)
    total = len(gold_list)
    correct = sum(gold == pred for gold, pred in zip(gold_list, pred_list))

    per_label = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_f1 = 0.0
    total_support = 0

    for label in labels:
        tp = sum(gold == label and pred == label for gold, pred in zip(gold_list, pred_list))
        fp = sum(gold != label and pred == label for gold, pred in zip(gold_list, pred_list))
        fn = sum(gold == label and pred != label for gold, pred in zip(gold_list, pred_list))
        support = sum(gold == label for gold in gold_list)
        scores = _prf(tp, fp, fn)
        scores["support"] = support
        per_label[label] = scores
        macro_precision += scores["precision"]
        macro_recall += scores["recall"]
        macro_f1 += scores["f1"]
        weighted_f1 += scores["f1"] * support
        total_support += support

    accuracy = _safe_div(correct, total)
    macro_precision = _safe_div(macro_precision, len(labels))
    macro_recall = _safe_div(macro_recall, len(labels))
    macro_f1 = _safe_div(macro_f1, len(labels))
    weighted_f1 = _safe_div(weighted_f1, total_support)

    tp_total = correct
    fp_total = total - correct
    fn_total = total - correct
    micro_scores = _prf(tp_total, fp_total, fn_total)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_scores["precision"],
        "micro_recall": micro_scores["recall"],
        "micro_f1": micro_scores["f1"],
        "weighted_f1": weighted_f1,
        "per_label": per_label,
    }


def compute_multilabel_metrics(
    golds: Iterable[list[int]],
    preds: Iterable[list[int]],
    label_names: list[str],
) -> dict[str, object]:
    gold_list = list(golds)
    pred_list = list(preds)
    n_rows = len(gold_list)
    n_labels = len(label_names)

    per_label = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_f1 = 0.0
    total_support = 0
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    mismatches = 0
    exact_match = 0

    for label_index, label_name in enumerate(label_names):
        tp = 0
        fp = 0
        fn = 0
        support = 0
        for gold_vector, pred_vector in zip(gold_list, pred_list):
            gold_value = int(gold_vector[label_index])
            pred_value = int(pred_vector[label_index])
            if gold_value == 1:
                support += 1
            if gold_value == 1 and pred_value == 1:
                tp += 1
            elif gold_value == 0 and pred_value == 1:
                fp += 1
            elif gold_value == 1 and pred_value == 0:
                fn += 1

        scores = _prf(tp, fp, fn)
        scores["support"] = support
        per_label[label_name] = scores
        macro_precision += scores["precision"]
        macro_recall += scores["recall"]
        macro_f1 += scores["f1"]
        weighted_f1 += scores["f1"] * support
        total_support += support
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    for gold_vector, pred_vector in zip(gold_list, pred_list):
        mismatches += sum(gold != pred for gold, pred in zip(gold_vector, pred_vector))
        if list(gold_vector) == list(pred_vector):
            exact_match += 1

    macro_precision = _safe_div(macro_precision, n_labels)
    macro_recall = _safe_div(macro_recall, n_labels)
    macro_f1 = _safe_div(macro_f1, n_labels)
    weighted_f1 = _safe_div(weighted_f1, total_support)
    micro_scores = _prf(micro_tp, micro_fp, micro_fn)

    return {
        "exact_match_accuracy": _safe_div(exact_match, n_rows),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_scores["precision"],
        "micro_recall": micro_scores["recall"],
        "micro_f1": micro_scores["f1"],
        "weighted_f1": weighted_f1,
        "hamming_loss": _safe_div(mismatches, n_rows * n_labels),
        "per_label": per_label,
    }
