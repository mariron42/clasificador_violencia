from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from prepare_official_submission import load_checkpoint_state_dict, load_unlabeled_text_rows  # noqa: E402
from run_competitive_encoder_severity import MultitaskSeverityEncoder, maybe_pad_token, sanitize_text  # noqa: E402
from womenhelp_competition.data import load_subtask1  # noqa: E402
from womenhelp_competition.hf_utils import configure_hf_backend  # noqa: E402
from womenhelp_competition.labels import SUBTASK1_LABELS  # noqa: E402
from womenhelp_competition.metrics import compute_multiclass_metrics  # noqa: E402


LABEL_IDS = [0, 1, 2, 3]
PROB_COLUMNS = ["prob_mild", "prob_medium", "prob_high", "prob_severe"]
TAIL_COLUMNS = ["tail_gt_mild", "tail_gt_medium", "tail_gt_high"]

SEVERE_PATTERNS = [
    r"\b(matar|mat[oó]|maten|matarla|matarlo|matarme|asesin|homicid|feminicid)\b",
    r"\b(muerte|muerta|muerto|cad[aá]ver)\b",
    r"\b(arma|pistola|revolver|rifle|escopeta|cuchillo|navaja|machete|dispar|balazo|bala)\b",
    r"\b(viola|violaci[oó]n|abuso sexual|agresi[oó]n sexual|tocamientos|manose)\b",
    r"\b(estrangul|ahorc|asfixi|sofoc)\b",
    r"\b(secuestr|privaci[oó]n de la libertad|encerr[oó]|encerrada|encerrado)\b",
    r"\b(hospital|urgencias|fractur|sangr|inconsciente|lesi[oó]n grave|quemad)\b",
    r"\b(gasolina|quemar|incendiar)\b",
    r"\b(amenaz[ao].{0,35}muerte|amenaza.{0,35}matar)\b",
]
HIGH_PATTERNS = [
    r"\b(golp|peg[oó]|patad|puñet|cachetad|empuj|arrastr|jal[oó]|forceje)\b",
    r"\b(amenaz|persegu|acos|hostig|vigil|intimid)\b",
    r"\b(denuncia|polic[ií]a|orden de protecci[oó]n|ministerio p[uú]blico)\b",
    r"\b(dañ[oó]|rompi[oó]|destruy|avent[oó]|quit[oó]|rob[oó])\b",
]
SEVERE_REGEXES = [re.compile(pattern, re.IGNORECASE) for pattern in SEVERE_PATTERNS]
HIGH_REGEXES = [re.compile(pattern, re.IGNORECASE) for pattern in HIGH_PATTERNS]


@dataclass(frozen=True)
class RuleSpec:
    name: str
    transitions: tuple[str, ...]
    min_confidence: float
    min_margin: float
    evidence_mode: str
    length_scope: str
    pure_candidate: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materializa el ultimo Hail Mary S1 + S2 seguro")
    parser.add_argument("--dev-data-dir", default="data/official_dev")
    parser.add_argument("--test-data-dir", default="data/official_test")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--beto-fold-root",
        default="outputs/subtask1_encoder_oof/20260506_initial_adamw/beto_augrep08",
        type=Path,
    )
    parser.add_argument(
        "--electricidad-fold-root",
        default="outputs/subtask1_encoder_oof/20260506_initial_adamw/electricidad_focal",
        type=Path,
    )
    parser.add_argument(
        "--beto-devel-jsonl",
        default="outputs/subtask1_encoder_oof/20260506_initial_adamw/beto_augrep08_aggregate/devel_predictions.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--electricidad-devel-jsonl",
        default="outputs/subtask1_encoder_oof/20260506_initial_adamw/electricidad_focal_aggregate/devel_predictions.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--base-devel-verbose",
        default="outputs/mega_ensemble/final_with_v2/optimized_top10/subtask1_verbose.csv",
        type=Path,
    )
    parser.add_argument(
        "--base-test-s1",
        default="outputs/official_submission/20260429_public_top10_s1_beto_encoder_s2_fixed045/submission/subtask1.csv",
        type=Path,
    )
    parser.add_argument(
        "--safe-subtask2",
        default="outputs/official_submission/20260429_public_top10_s1_beto_encoder_s2_fixed045/submission/subtask2.csv",
        type=Path,
    )
    parser.add_argument("--thresholds", nargs=3, type=float, default=[0.6, 0.55, 0.55])
    parser.add_argument("--weights", nargs=2, type=float, default=[0.5, 0.5])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--reuse-test-probs", action="store_true")
    parser.add_argument("--max-catastrophic-delta", type=float, default=0.006)
    parser.add_argument("--min-severe-f1-delta", type=float, default=-0.04)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_single_column(path: Path) -> np.ndarray:
    values: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if row:
                values.append(int(row[0]))
    return np.asarray(values, dtype=np.int64)


def write_single_column(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for value in values.astype(int).tolist():
            writer.writerow([value])


def package_zip(zip_path: Path, subtask1: Path, subtask2: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(subtask1, arcname="subtask1.csv")
        archive.write(subtask2, arcname="subtask2.csv")


def class_names(values: np.ndarray) -> list[str]:
    return [str(int(value)) for value in values.tolist()]


def metric_payload(gold: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    metrics = compute_multiclass_metrics(class_names(gold), class_names(pred), ["0", "1", "2", "3"])
    distance = np.abs(gold.astype(np.int64) - pred.astype(np.int64))
    severe_gold = gold == 3
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "micro_f1": float(metrics["micro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "severe_f1": float(metrics["per_label"]["3"]["f1"]),
        "severe_recall": float(metrics["per_label"]["3"]["recall"]),
        "catastrophic_rate": float(np.mean(distance >= 2)),
        "mild_severe_rate": float(np.mean(distance == 3)),
        "severe_to_mild_or_medium": int(np.sum(severe_gold & (pred <= 1))),
        "pred_distribution": {str(label): int(np.sum(pred == label)) for label in LABEL_IDS},
        "per_label": metrics["per_label"],
    }


def fast_metric_payload(gold: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    distance = np.abs(gold.astype(np.int64) - pred.astype(np.int64))
    severe_gold = gold == 3
    severe_pred = pred == 3
    severe_tp = int(np.sum(severe_gold & severe_pred))
    severe_precision = severe_tp / max(int(np.sum(severe_pred)), 1)
    severe_recall = severe_tp / max(int(np.sum(severe_gold)), 1)
    severe_f1 = 0.0 if severe_precision + severe_recall == 0.0 else (
        2.0 * severe_precision * severe_recall / (severe_precision + severe_recall)
    )
    return {
        "macro_f1": float(f1_score(gold, pred, labels=LABEL_IDS, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(gold, pred, labels=LABEL_IDS, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(gold, pred, labels=LABEL_IDS, average="weighted", zero_division=0)),
        "severe_f1": float(severe_f1),
        "severe_recall": float(severe_recall),
        "catastrophic_rate": float(np.mean(distance >= 2)),
        "mild_severe_rate": float(np.mean(distance == 3)),
        "severe_to_mild_or_medium": int(np.sum(severe_gold & (pred <= 1))),
    }


def tail_scores(probabilities: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            probabilities[:, 1:].sum(axis=1),
            probabilities[:, 2:].sum(axis=1),
            probabilities[:, 3],
        ]
    )


def ordinal_predict(probabilities: np.ndarray, thresholds: list[float]) -> np.ndarray:
    threshold_array = np.asarray(thresholds, dtype=np.float64).reshape(1, 3)
    return (tail_scores(probabilities) >= threshold_array).sum(axis=1).astype(np.int64)


def read_probability_jsonl(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]
    ids = [str(row["record_id"]) for row in rows]
    probabilities = np.asarray(
        [[float(row["probabilities"][label]) for label in SUBTASK1_LABELS] for row in rows],
        dtype=np.float64,
    )
    gold = np.asarray([int(row["gold_class_id"]) for row in rows], dtype=np.int64)
    return ids, probabilities, gold


def align_probability_source(path: Path, target_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ids, probabilities, gold = read_probability_jsonl(path)
    order = {record_id: index for index, record_id in enumerate(ids)}
    missing = [record_id for record_id in target_ids if record_id not in order]
    if missing:
        raise ValueError(f"Faltan ids en {path}: {missing[:5]}")
    remap = [order[record_id] for record_id in target_ids]
    return probabilities[remap], gold[remap]


def read_base_devel(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    ids: list[str] = []
    gold: list[int] = []
    pred: list[int] = []
    probs: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ids.append(str(row["record_id"]))
            gold.append(int(row["gold_class_id"]))
            pred.append(int(row["pred_class_id"]))
            probs.append([float(row[column]) for column in PROB_COLUMNS])
    return ids, np.asarray(gold, dtype=np.int64), np.asarray(pred, dtype=np.int64), np.asarray(probs, dtype=np.float64)


def text_features(texts: list[str], q1: float, q2: float) -> dict[str, np.ndarray]:
    severe_hits: list[int] = []
    high_hits: list[int] = []
    lengths: list[int] = []
    for text in texts:
        clean = sanitize_text(text)
        severe_hits.append(sum(len(regex.findall(clean)) for regex in SEVERE_REGEXES))
        high_hits.append(sum(len(regex.findall(clean)) for regex in HIGH_REGEXES))
        lengths.append(len(clean.split()))
    length_arr = np.asarray(lengths, dtype=np.float64)
    return {
        "severe_hits": np.asarray(severe_hits, dtype=np.int64),
        "high_hits": np.asarray(high_hits, dtype=np.int64),
        "any_hits": ((np.asarray(severe_hits) + np.asarray(high_hits)) > 0),
        "length": length_arr,
        "short": length_arr <= q1,
        "long": length_arr > q2,
    }


def fold_dirs(root: Path) -> list[Path]:
    dirs = [root / f"fold_{index}" for index in range(5)]
    missing = [path for path in dirs if not (path / "metrics_summary.json").exists()]
    if missing:
        raise FileNotFoundError(f"Folds incompletos en {root}: {missing}")
    return dirs


def load_fold_model(fold_dir: Path, device: torch.device) -> tuple[MultitaskSeverityEncoder, Any, dict[str, Any]]:
    metrics = read_json(fold_dir / "metrics_summary.json")
    train_config = read_json(fold_dir / "train_config.json")
    tokenizer = AutoTokenizer.from_pretrained(str(fold_dir / "tokenizer"), trust_remote_code=True)
    maybe_pad_token(tokenizer)
    model = MultitaskSeverityEncoder(
        metrics["model_name_or_path"],
        attn_implementation=metrics.get("attn_implementation", train_config.get("attn_implementation", "eager")),
        severity_head=metrics.get("severity_head", train_config.get("severity_head", "softmax")),
        loss_type=metrics.get("loss_type", train_config.get("loss_type", "cross_entropy")),
        focal_gamma=float(train_config.get("focal_gamma", 2.0)),
        ordinal_distance_weight=float(metrics.get("ordinal_distance_weight", train_config.get("ordinal_distance_weight", 0.0))),
        severity_class_weights=metrics.get("severity_class_weights"),
        type_loss_weight=float(metrics.get("type_loss_weight", train_config.get("type_loss_weight", 0.0))),
        type_pos_weight=metrics.get("type_pos_weight"),
    )
    state_dict = load_checkpoint_state_dict(Path(metrics["best_checkpoint"]))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer, {"max_seq_length": int(train_config.get("max_seq_length", 512)), **metrics}


def predict_with_model(
    model: MultitaskSeverityEncoder,
    tokenizer,
    texts: list[str],
    *,
    batch_size: int,
    max_seq_length: int,
    device: torch.device,
) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = [sanitize_text(text) for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=max_seq_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            arrays.append(torch.softmax(outputs["severity_logits"], dim=1).detach().cpu().numpy())
    return np.concatenate(arrays, axis=0)


def infer_family(
    root: Path,
    texts: list[str],
    *,
    cache_path: Path,
    batch_size: int,
    device: torch.device,
    reuse: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if reuse and cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)
        return payload["probabilities"], json.loads(str(payload["metadata"].item()))
    probs: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for fold_dir in fold_dirs(root):
        print(f"[infer] loading {fold_dir}", flush=True)
        model, tokenizer, meta = load_fold_model(fold_dir, device)
        print(f"[infer] predicting {fold_dir}", flush=True)
        probs.append(
            predict_with_model(
                model,
                tokenizer,
                texts,
                batch_size=batch_size,
                max_seq_length=int(meta["max_seq_length"]),
                device=device,
            )
        )
        metadata.append(
            {
                "fold_dir": str(fold_dir),
                "best_checkpoint": meta["best_checkpoint"],
                "model_name_or_path": meta["model_name_or_path"],
                "max_seq_length": int(meta["max_seq_length"]),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[infer] done {fold_dir}", flush=True)
    averaged = np.mean(np.stack(probs, axis=0), axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, probabilities=averaged, metadata=json.dumps(metadata, ensure_ascii=False))
    return averaged, metadata


def candidate_confidence(probabilities: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_probs = np.sort(probabilities, axis=1)
    confidence = probabilities[np.arange(len(pred)), pred]
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return confidence, margin


def length_mask(scope: str, features: dict[str, np.ndarray]) -> np.ndarray:
    if scope == "all":
        return np.ones(len(features["length"]), dtype=bool)
    if scope == "long":
        return features["long"].astype(bool)
    if scope == "not_long":
        return ~features["long"].astype(bool)
    if scope == "short":
        return features["short"].astype(bool)
    if scope == "not_short":
        return ~features["short"].astype(bool)
    raise ValueError(f"length_scope desconocido: {scope}")


def evidence_mask(mode: str, pred: np.ndarray, features: dict[str, np.ndarray]) -> np.ndarray:
    if mode == "none":
        return np.ones(len(pred), dtype=bool)
    if mode == "any_hit":
        return features["any_hits"].astype(bool)
    if mode == "severe_if_target_severe":
        return (pred < 3) | (features["severe_hits"] > 0)
    if mode == "any_if_upward":
        return features["any_hits"].astype(bool)
    if mode == "severe_or_high_if_target_severe":
        return (pred < 3) | ((features["severe_hits"] + features["high_hits"]) > 0)
    raise ValueError(f"evidence_mode desconocido: {mode}")


def apply_rule(
    rule: RuleSpec,
    base: np.ndarray,
    cand: np.ndarray,
    cand_probs: np.ndarray,
    features: dict[str, np.ndarray],
) -> np.ndarray:
    if rule.pure_candidate:
        return cand.copy()
    confidence, margin = candidate_confidence(cand_probs, cand)
    transitions = np.asarray([f"{int(old)}->{int(new)}" for old, new in zip(base, cand)])
    mask = cand != base
    mask &= np.isin(transitions, np.asarray(rule.transitions))
    mask &= confidence >= rule.min_confidence
    mask &= margin >= rule.min_margin
    mask &= length_mask(rule.length_scope, features)
    mask &= evidence_mask(rule.evidence_mode, cand, features)
    out = base.copy()
    out[mask] = cand[mask]
    return out


def build_rule_space(base: np.ndarray, cand: np.ndarray, gold: np.ndarray, features: dict[str, np.ndarray], cand_probs: np.ndarray) -> list[RuleSpec]:
    observed = sorted({f"{int(old)}->{int(new)}" for old, new in zip(base, cand) if int(old) != int(new)})
    transition_scores: list[tuple[float, str]] = []
    base_macro = float(f1_score(gold, base, labels=LABEL_IDS, average="macro"))
    for transition in observed:
        rule = RuleSpec(
            name=f"transition_{transition}",
            transitions=(transition,),
            min_confidence=0.0,
            min_margin=0.0,
            evidence_mode="none",
            length_scope="all",
        )
        pred = apply_rule(rule, base, cand, cand_probs, features)
        transition_scores.append((float(f1_score(gold, pred, labels=LABEL_IDS, average="macro")) - base_macro, transition))
    top_transitions = [transition for _delta, transition in sorted(transition_scores, reverse=True)[:6]]
    transition_sets: list[tuple[str, ...]] = []
    transition_sets.extend((transition,) for transition in top_transitions)
    for size in [2, 3]:
        transition_sets.extend(tuple(combo) for combo in combinations(top_transitions, size))
    transition_sets.extend(
        [
            tuple(top_transitions),
            tuple(t for t in observed if int(t.split("->")[1]) > int(t.split("->")[0])),
            tuple(t for t in observed if int(t.split("->")[1]) < int(t.split("->")[0])),
            tuple(observed),
        ]
    )
    deduped = sorted({tuple(item) for item in transition_sets if item})
    rules = [
        RuleSpec(
            name="pure_candidate",
            transitions=tuple(),
            min_confidence=0.0,
            min_margin=0.0,
            evidence_mode="none",
            length_scope="all",
            pure_candidate=True,
        )
    ]
    for transitions in deduped:
        for confidence in [0.0, 0.50, 0.55, 0.60]:
            for margin in [0.0, 0.10, 0.20]:
                for evidence in ["none", "severe_if_target_severe"]:
                    for scope in ["all", "long", "not_short"]:
                        rules.append(
                            RuleSpec(
                                name="router",
                                transitions=transitions,
                                min_confidence=confidence,
                                min_margin=margin,
                                evidence_mode=evidence,
                                length_scope=scope,
                            )
                        )
    return rules


def evaluate_rules(
    rules: list[RuleSpec],
    base: np.ndarray,
    cand: np.ndarray,
    cand_probs: np.ndarray,
    gold: np.ndarray,
    features: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, rule in enumerate(rules):
        pred = apply_rule(rule, base, cand, cand_probs, features)
        metrics = fast_metric_payload(gold, pred)
        changed = pred != base
        rows.append(
            {
                "rank_input": index,
                "rule": {
                    "name": rule.name,
                    "transitions": list(rule.transitions),
                    "min_confidence": rule.min_confidence,
                    "min_margin": rule.min_margin,
                    "evidence_mode": rule.evidence_mode,
                    "length_scope": rule.length_scope,
                    "pure_candidate": rule.pure_candidate,
                },
                "metrics": metrics,
                "changed_rows": int(np.sum(changed)),
            }
        )
    rows.sort(
        key=lambda row: (
            row["metrics"]["macro_f1"],
            row["metrics"]["severe_f1"],
            -row["metrics"]["catastrophic_rate"],
            -row["changed_rows"],
        ),
        reverse=True,
    )
    return rows


def select_rule(rows: list[dict[str, Any]], base_metrics: dict[str, Any], max_cat_delta: float, min_sev_delta: float) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["metrics"]["catastrophic_rate"] <= base_metrics["catastrophic_rate"] + max_cat_delta
        and row["metrics"]["severe_f1"] >= base_metrics["severe_f1"] + min_sev_delta
    ]
    return eligible[0] if eligible else rows[0]


def rule_from_payload(payload: dict[str, Any]) -> RuleSpec:
    rule = payload["rule"]
    return RuleSpec(
        name=str(rule["name"]),
        transitions=tuple(rule["transitions"]),
        min_confidence=float(rule["min_confidence"]),
        min_margin=float(rule["min_margin"]),
        evidence_mode=str(rule["evidence_mode"]),
        length_scope=str(rule["length_scope"]),
        pure_candidate=bool(rule["pure_candidate"]),
    )


def write_verbose(
    path: Path,
    *,
    ids: list[str],
    gold: np.ndarray | None,
    base: np.ndarray,
    cand: np.ndarray,
    final: np.ndarray,
    cand_probs: np.ndarray,
    features: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "record_id",
        "gold_class_id",
        "base_pred",
        "candidate_pred",
        "final_pred",
        "changed_vs_base",
        "candidate_confidence",
        "candidate_margin",
        "candidate_tail_gt_mild",
        "candidate_tail_gt_medium",
        "candidate_tail_gt_high",
        "severe_hits",
        "high_hits",
        "length_tokens",
        *PROB_COLUMNS,
    ]
    confidence, margin = candidate_confidence(cand_probs, cand)
    tails = tail_scores(cand_probs)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, record_id in enumerate(ids):
            row = {
                "record_id": record_id,
                "gold_class_id": "" if gold is None else int(gold[index]),
                "base_pred": int(base[index]),
                "candidate_pred": int(cand[index]),
                "final_pred": int(final[index]),
                "changed_vs_base": int(final[index] != base[index]),
                "candidate_confidence": float(confidence[index]),
                "candidate_margin": float(margin[index]),
                "candidate_tail_gt_mild": float(tails[index, 0]),
                "candidate_tail_gt_medium": float(tails[index, 1]),
                "candidate_tail_gt_high": float(tails[index, 2]),
                "severe_hits": int(features["severe_hits"][index]),
                "high_hits": int(features["high_hits"][index]),
                "length_tokens": float(features["length"][index]),
            }
            for col_index, column in enumerate(PROB_COLUMNS):
                row[column] = float(cand_probs[index, col_index])
            writer.writerow(row)


def build_report(summary: dict[str, Any]) -> str:
    selected = summary["selected"]
    lines = [
        "# Hail Mary S1 foldbag router 2026-05-07",
        "",
        "## Decision",
        "",
        f"- selected_rule: `{selected['rule']}`",
        f"- selected_devel_macro: `{selected['metrics']['macro_f1']:.6f}`",
        f"- base_devel_macro: `{summary['base_metrics']['macro_f1']:.6f}`",
        f"- candidate_full_devel_macro: `{summary['candidate_full_metrics']['macro_f1']:.6f}`",
        f"- selected_devel_severe_f1: `{selected['metrics']['severe_f1']:.6f}`",
        f"- selected_changed_rows_devel: `{selected['changed_rows']}`",
        f"- selected_changed_rows_test: `{summary['test_changed_rows']}`",
        "",
        "## Top devel candidates",
        "",
        "| rank | macro | severe_f1 | catastrophic | changed | rule |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for index, row in enumerate(summary["top_candidates"], start=1):
        lines.append(
            f"| {index} | {row['metrics']['macro_f1']:.6f} | {row['metrics']['severe_f1']:.6f} | "
            f"{row['metrics']['catastrophic_rate']:.6f} | {row['changed_rows']} | `{row['rule']}` |"
        )
    lines.extend(
        [
            "",
            "## Package",
            "",
            f"- zip: `{summary['zip']}`",
            f"- zip_sha256: `{summary['zip_sha256']}`",
            f"- subtask1_rows: `{summary['subtask1_rows']}`",
            f"- subtask2_rows: `{summary['subtask2_rows']}`",
            f"- subtask2_source: `{summary['safe_subtask2']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    configure_hf_backend()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    weights = np.asarray(args.weights, dtype=np.float64)
    weights = weights / weights.sum()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")

    base_ids, gold, base_pred, _base_probs = read_base_devel(args.base_devel_verbose)
    beto_dev, beto_gold = align_probability_source(args.beto_devel_jsonl, base_ids)
    elec_dev, elec_gold = align_probability_source(args.electricidad_devel_jsonl, base_ids)
    if not np.array_equal(gold, beto_gold) or not np.array_equal(gold, elec_gold):
        raise ValueError("Gold mismatch entre base y sources OOF")
    devel_candidate_probs = weights[0] * beto_dev + weights[1] * elec_dev
    devel_candidate_pred = ordinal_predict(devel_candidate_probs, args.thresholds)

    devel_records = load_subtask1("devel", str(args.dev_data_dir))
    text_by_id = {str(record.record_id): record.text for record in devel_records}
    devel_texts = [text_by_id[record_id] for record_id in base_ids]
    devel_lengths = np.asarray([len(sanitize_text(text).split()) for text in devel_texts], dtype=np.float64)
    q1, q2 = np.quantile(devel_lengths, [1 / 3, 2 / 3])
    devel_features = text_features(devel_texts, float(q1), float(q2))

    rules = build_rule_space(base_pred, devel_candidate_pred, gold, devel_features, devel_candidate_probs)
    print(f"[router] evaluating {len(rules)} rules", flush=True)
    evaluated = evaluate_rules(rules, base_pred, devel_candidate_pred, devel_candidate_probs, gold, devel_features)
    print("[router] search done", flush=True)
    base_metrics = metric_payload(gold, base_pred)
    candidate_full_metrics = metric_payload(gold, devel_candidate_pred)
    selected = select_rule(evaluated, base_metrics, args.max_catastrophic_delta, args.min_severe_f1_delta)
    selected_rule = rule_from_payload(selected)
    selected_devel_pred = apply_rule(selected_rule, base_pred, devel_candidate_pred, devel_candidate_probs, devel_features)
    selected = {
        **selected,
        "metrics": metric_payload(gold, selected_devel_pred),
    }

    test_texts = load_unlabeled_text_rows(Path(args.test_data_dir) / "subtask1" / "test.csv")
    prob_dir = args.output_dir / "probabilities"
    print("[test] infer beto_augrep08 foldbag", flush=True)
    beto_test, beto_meta = infer_family(
        args.beto_fold_root,
        test_texts,
        cache_path=prob_dir / "beto_augrep08_test_probs.npz",
        batch_size=args.batch_size,
        device=device,
        reuse=args.reuse_test_probs,
    )
    print("[test] infer electricidad_focal foldbag", flush=True)
    elec_test, elec_meta = infer_family(
        args.electricidad_fold_root,
        test_texts,
        cache_path=prob_dir / "electricidad_focal_test_probs.npz",
        batch_size=args.batch_size,
        device=device,
        reuse=args.reuse_test_probs,
    )
    print("[test] applying selected rule and packaging", flush=True)
    test_candidate_probs = weights[0] * beto_test + weights[1] * elec_test
    test_candidate_pred = ordinal_predict(test_candidate_probs, args.thresholds)
    base_test_pred = read_single_column(args.base_test_s1)
    test_features = text_features(test_texts, float(q1), float(q2))
    test_final_pred = apply_rule(selected_rule, base_test_pred, test_candidate_pred, test_candidate_probs, test_features)

    submission_dir = args.output_dir / "submission"
    s1_path = submission_dir / "subtask1.csv"
    s2_path = submission_dir / "subtask2.csv"
    write_single_column(s1_path, test_final_pred)
    s2_path.write_bytes(args.safe_subtask2.read_bytes())
    zip_path = args.output_dir / "predictions.zip"
    package_zip(zip_path, s1_path, s2_path)

    write_verbose(
        args.output_dir / "devel_verbose.csv",
        ids=base_ids,
        gold=gold,
        base=base_pred,
        cand=devel_candidate_pred,
        final=selected_devel_pred,
        cand_probs=devel_candidate_probs,
        features=devel_features,
    )
    write_verbose(
        args.output_dir / "test_verbose.csv",
        ids=[str(index) for index in range(len(test_texts))],
        gold=None,
        base=base_test_pred,
        cand=test_candidate_pred,
        final=test_final_pred,
        cand_probs=test_candidate_probs,
        features=test_features,
    )

    selected_compact = {
        "rule": selected["rule"],
        "metrics": selected["metrics"],
        "changed_rows": selected["changed_rows"],
    }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "family": "hailmary_s1_foldbag_router_s2_best",
        "device": str(device),
        "thresholds": [float(value) for value in args.thresholds],
        "weights": weights.tolist(),
        "devel_length_quantiles": {"q1": float(q1), "q2": float(q2)},
        "base_metrics": base_metrics,
        "candidate_full_metrics": candidate_full_metrics,
        "selected": selected_compact,
        "top_candidates": [
            {"rule": row["rule"], "metrics": row["metrics"], "changed_rows": row["changed_rows"]}
            for row in evaluated[:20]
        ],
        "test_changed_rows": int(np.sum(test_final_pred != base_test_pred)),
        "test_candidate_changed_rows": int(np.sum(test_candidate_pred != base_test_pred)),
        "test_distribution": {str(label): int(np.sum(test_final_pred == label)) for label in LABEL_IDS},
        "test_candidate_distribution": {str(label): int(np.sum(test_candidate_pred == label)) for label in LABEL_IDS},
        "test_base_distribution": {str(label): int(np.sum(base_test_pred == label)) for label in LABEL_IDS},
        "beto_test_metadata": beto_meta,
        "electricidad_test_metadata": elec_meta,
        "safe_subtask2": str(args.safe_subtask2),
        "subtask1_rows": int(len(test_final_pred)),
        "subtask2_rows": int(len(read_single_column(args.safe_subtask2))),
        "zip": str(zip_path),
        "zip_sha256": sha256(zip_path),
        "subtask1_sha256": sha256(s1_path),
        "subtask2_sha256": sha256(s2_path),
    }
    (args.output_dir / "packaging_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "report.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
