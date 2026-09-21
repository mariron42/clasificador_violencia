from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from datasets import Dataset  # noqa: E402
from torch import nn  # noqa: E402
from transformers import (  # noqa: E402
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from womenhelp_competition.data import (  # noqa: E402
    load_subtask1,
    load_subtask2,
    load_subtask2_soft,
    slice_records,
)
from womenhelp_competition.hf_utils import configure_hf_backend  # noqa: E402
from womenhelp_competition.labels import (  # noqa: E402
    SUBTASK1_ID_TO_NAME,
    SUBTASK1_LABELS,
    SUBTASK2_COLUMNS,
    SUBTASK2_LABELS,
)
from womenhelp_competition.metrics import (  # noqa: E402
    compute_multiclass_metrics,
    compute_multilabel_metrics,
)
from womenhelp_competition.text_augmentation import augment_rows_for_label  # noqa: E402

SEED = 3407
NUM_SUBTASK1_LABELS = len(SUBTASK1_LABELS)
NUM_SUBTASK2_LABELS = len(SUBTASK2_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corre un encoder competitivo de severidad con multitarea opcional")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="devel")
    parser.add_argument("--model-name-or-path", default="dccuchile/bert-base-spanish-wwm-cased")
    parser.add_argument("--attn-implementation", choices=["auto", "eager", "sdpa"], default="eager")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "competitive_encoder_severity" / "beto_multitask"),
    )
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--optim", default="stable_adamw")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="fp32")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--loss-type", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--severity-head", choices=["softmax", "coral", "corn"], default="softmax")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--ordinal-distance-weight",
        type=float,
        default=0.0,
        help="Peso de una penalizacion ordinal suave sobre la clase esperada; aplica principalmente a softmax.",
    )
    parser.add_argument(
        "--severity-class-weight",
        choices=["none", "balanced", "sqrt_balanced"],
        default="balanced",
    )
    parser.add_argument("--type-loss-weight", type=float, default=0.3)
    parser.add_argument("--type-label-source", choices=["none", "hard", "soft"], default="soft")
    parser.add_argument("--type-use-pos-weight", action="store_true")
    parser.add_argument("--type-threshold", type=float, default=0.5)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument(
        "--augment-mode",
        "--severe-augment-mode",
        dest="augment_mode",
        choices=["none", "synonym", "report_style"],
        default="none",
    )
    parser.add_argument(
        "--augment-target-label",
        choices=SUBTASK1_LABELS,
        default="Severe",
    )
    parser.add_argument(
        "--augment-target-ratio",
        "--severe-augment-target-ratio",
        dest="augment_target_ratio",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--augment-max-copies",
        "--severe-augment-max-copies",
        dest="augment_max_copies",
        type=int,
        default=2,
    )
    return parser.parse_args()


def resolve_precision_flags(precision: str) -> tuple[bool, bool]:
    return precision == "bf16", precision == "fp16"


def maybe_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return
    tokenizer.pad_token = tokenizer.unk_token


def resolve_strategy(max_steps: int) -> tuple[str, str]:
    if max_steps > 0:
        return "steps", "steps"
    return "epoch", "epoch"


def severity_id_to_name(label_id: int) -> str:
    return SUBTASK1_ID_TO_NAME[str(int(label_id))]


def severity_name_to_id(label_name: str) -> int:
    return SUBTASK1_LABELS.index(str(label_name))


def sanitize_text(text: str) -> str:
    return " ".join(text.split())


def soft_label_map(split: str, data_dir: str | None) -> dict[str, list[float]]:
    rows = load_subtask2_soft(split, data_dir)
    label_map: dict[str, list[float]] = {}
    for row in rows:
        label_map[row["ID"]] = [float(row[column]) for column in SUBTASK2_COLUMNS]
    return label_map


def hard_label_map(split: str, data_dir: str | None) -> dict[str, list[float]]:
    rows = load_subtask2(split, data_dir)
    label_map: dict[str, list[float]] = {}
    for row in rows:
        label_map[row.record_id] = [float(int(row.label_vector[column])) for column in SUBTASK2_COLUMNS]
    return label_map


def build_type_label_map(split: str, data_dir: str | None, source: str) -> dict[str, list[float]]:
    if source == "none":
        return {}
    if source == "soft":
        return soft_label_map(split, data_dir)
    return hard_label_map(split, data_dir)


def build_rows(
    split: str,
    data_dir: str | None,
    *,
    type_label_map: dict[str, list[float]],
    limit: int,
) -> list[dict[str, Any]]:
    records = slice_records(load_subtask1(split, data_dir), limit)
    rows: list[dict[str, Any]] = []
    for record in records:
        type_labels = type_label_map.get(record.record_id)
        types_mask = 1.0 if type_labels is not None else 0.0
        rows.append(
            {
                "record_id": record.record_id,
                "text": sanitize_text(record.text),
                "labels": int(record.severity_id),
                "types_labels": type_labels if type_labels is not None else [0.0] * NUM_SUBTASK2_LABELS,
                "types_mask": types_mask,
            }
        )
    return rows


def tokenize_rows(rows: list[dict[str, Any]], tokenizer, max_seq_length: int) -> Dataset:
    dataset = Dataset.from_list(rows)

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    return dataset.map(tokenize_batch, batched=True, remove_columns=["record_id", "text"])


def compute_class_weights(rows: list[dict[str, Any]], mode: str) -> list[float] | None:
    if mode == "none":
        return None
    labels = np.asarray([row["labels"] for row in rows], dtype=np.int64)
    counts = np.bincount(labels, minlength=NUM_SUBTASK1_LABELS).astype(np.float64)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (NUM_SUBTASK1_LABELS * counts)
    if mode == "sqrt_balanced":
        weights = np.sqrt(weights)
    return weights.tolist()


def compute_type_pos_weight(rows: list[dict[str, Any]]) -> list[float] | None:
    masked_rows = [row for row in rows if row["types_mask"] > 0.0]
    if not masked_rows:
        return None
    label_matrix = np.asarray([row["types_labels"] for row in masked_rows], dtype=np.float32)
    positives = label_matrix.sum(axis=0)
    negatives = label_matrix.shape[0] - positives
    positives = np.clip(positives, 1e-6, None)
    weights = negatives / positives
    return weights.tolist()


class MultitaskSeverityEncoder(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        attn_implementation: str,
        severity_head: str,
        loss_type: str,
        focal_gamma: float,
        ordinal_distance_weight: float,
        severity_class_weights: list[float] | None,
        type_loss_weight: float,
        type_pos_weight: list[float] | None,
    ) -> None:
        super().__init__()
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if attn_implementation != "auto":
            model_kwargs["attn_implementation"] = attn_implementation
        # Force safetensors to bypass torch CVE-2025-32434 on torch<2.6
        model_kwargs["use_safetensors"] = True
        self.encoder = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
        self.config = self.encoder.config
        hidden_size = getattr(self.encoder.config, "hidden_size")
        dropout_prob = getattr(self.encoder.config, "classifier_dropout", None)
        if dropout_prob is None:
            dropout_prob = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.severity_head = severity_head
        if severity_head == "softmax":
            self.severity_classifier = nn.Linear(hidden_size, NUM_SUBTASK1_LABELS)
            self.ordinal_ranker = None
            self.ordinal_classifier = None
            self.ordinal_cutpoint_start = None
            self.ordinal_cutpoint_deltas = None
        elif severity_head == "coral":
            self.severity_classifier = None
            self.ordinal_ranker = nn.Linear(hidden_size, 1, bias=False)
            self.ordinal_classifier = None
            self.ordinal_cutpoint_start = nn.Parameter(torch.zeros(1))
            self.ordinal_cutpoint_deltas = nn.Parameter(torch.zeros(NUM_SUBTASK1_LABELS - 1))
        elif severity_head == "corn":
            self.severity_classifier = None
            self.ordinal_ranker = None
            self.ordinal_classifier = nn.Linear(hidden_size, NUM_SUBTASK1_LABELS - 1)
            self.ordinal_cutpoint_start = None
            self.ordinal_cutpoint_deltas = None
        else:
            raise ValueError(f"severity_head no soportado: {severity_head}")
        self.type_classifier = nn.Linear(hidden_size, NUM_SUBTASK2_LABELS)
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.ordinal_distance_weight = ordinal_distance_weight
        self.type_loss_weight = type_loss_weight
        if severity_class_weights is not None:
            self.register_buffer(
                "severity_class_weights",
                torch.tensor(severity_class_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.severity_class_weights = None
        if type_pos_weight is not None:
            self.register_buffer(
                "type_pos_weight",
                torch.tensor(type_pos_weight, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.type_pos_weight = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                self.encoder.gradient_checkpointing_enable()
            else:
                self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()

    def pooled_output(self, encoder_outputs) -> torch.Tensor:
        return encoder_outputs.last_hidden_state[:, 0]

    def compute_severity_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.severity_head == "coral":
            loss = self.compute_coral_loss(logits, labels)
            probabilities = self.ordinal_logits_to_class_probabilities(logits)
            return loss + self.compute_ordinal_distance_penalty(probabilities, labels)
        if self.severity_head == "corn":
            loss = self.compute_corn_loss(logits, labels)
            probabilities = self.ordinal_logits_to_class_probabilities(logits)
            return loss + self.compute_ordinal_distance_penalty(probabilities, labels)

        if self.loss_type == "cross_entropy":
            loss = F.cross_entropy(logits, labels, weight=self.severity_class_weights)
            probabilities = F.softmax(logits, dim=-1)
            return loss + self.compute_ordinal_distance_penalty(probabilities, labels)

        log_probabilities = F.log_softmax(logits, dim=-1)
        probabilities = log_probabilities.exp()
        target_indices = labels.unsqueeze(1)
        target_log_probabilities = log_probabilities.gather(1, target_indices).squeeze(1)
        target_probabilities = probabilities.gather(1, target_indices).squeeze(1)
        focal_factor = (1.0 - target_probabilities).pow(self.focal_gamma)
        if self.severity_class_weights is None:
            alpha = torch.ones_like(target_probabilities)
        else:
            alpha = self.severity_class_weights[labels]
        loss = (-alpha * focal_factor * target_log_probabilities).mean()
        return loss + self.compute_ordinal_distance_penalty(probabilities, labels)

    def compute_ordinal_distance_penalty(self, probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.ordinal_distance_weight <= 0.0:
            return probabilities.new_tensor(0.0)
        class_positions = torch.arange(NUM_SUBTASK1_LABELS, device=probabilities.device, dtype=probabilities.dtype)
        expected_class = (probabilities * class_positions.view(1, -1)).sum(dim=1)
        target_class = labels.to(probabilities.dtype)
        per_row_penalty = (expected_class - target_class).pow(2)
        if self.severity_class_weights is not None:
            per_row_penalty = per_row_penalty * self.severity_class_weights[labels]
        return self.ordinal_distance_weight * per_row_penalty.mean()

    def ordinal_threshold_logits(self, pooled_output: torch.Tensor) -> torch.Tensor:
        if self.severity_head == "corn":
            if self.ordinal_classifier is None:
                raise RuntimeError("ordinal_threshold_logits requiere ordinal_classifier para CORN")
            return self.ordinal_classifier(pooled_output)
        if self.ordinal_ranker is None or self.ordinal_cutpoint_start is None or self.ordinal_cutpoint_deltas is None:
            raise RuntimeError("ordinal_threshold_logits requiere severity_head ordinal")
        rank_score = self.ordinal_ranker(pooled_output)
        cutpoints = self.ordinal_cutpoint_start + torch.cumsum(F.softplus(self.ordinal_cutpoint_deltas), dim=0)
        return rank_score - cutpoints.view(1, -1)

    def ordinal_logits_to_class_probabilities(self, ordinal_logits: torch.Tensor) -> torch.Tensor:
        threshold_probabilities = torch.sigmoid(ordinal_logits)
        if self.severity_head == "corn":
            threshold_probabilities = torch.cumprod(threshold_probabilities, dim=1)
        p0 = 1.0 - threshold_probabilities[:, 0]
        p1 = threshold_probabilities[:, 0] - threshold_probabilities[:, 1]
        p2 = threshold_probabilities[:, 1] - threshold_probabilities[:, 2]
        p3 = threshold_probabilities[:, 2]
        probabilities = torch.stack([p0, p1, p2, p3], dim=1)
        return probabilities.clamp_min(1e-7)

    def compute_coral_loss(self, ordinal_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        thresholds = torch.arange(NUM_SUBTASK1_LABELS - 1, device=labels.device).view(1, -1)
        targets = (labels.view(-1, 1) > thresholds).float()
        per_row_loss = F.binary_cross_entropy_with_logits(ordinal_logits, targets, reduction="none").mean(dim=1)
        if self.severity_class_weights is not None:
            per_row_loss = per_row_loss * self.severity_class_weights[labels]
        return per_row_loss.mean()

    def compute_corn_loss(self, ordinal_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        thresholds = torch.arange(NUM_SUBTASK1_LABELS - 1, device=labels.device).view(1, -1)
        targets = (labels.view(-1, 1) > thresholds).float()
        mask = (labels.view(-1, 1) >= thresholds).float()
        per_cell_loss = F.binary_cross_entropy_with_logits(ordinal_logits, targets, reduction="none") * mask
        per_row_loss = per_cell_loss.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        if self.severity_class_weights is not None:
            per_row_loss = per_row_loss * self.severity_class_weights[labels]
        return per_row_loss.mean()

    def compute_type_loss(
        self,
        type_logits: torch.Tensor,
        types_labels: torch.Tensor,
        types_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.type_loss_weight <= 0.0:
            return None
        per_row_loss = F.binary_cross_entropy_with_logits(
            type_logits,
            types_labels,
            pos_weight=self.type_pos_weight,
            reduction="none",
        ).mean(dim=1)
        if types_mask is None:
            return per_row_loss.mean()
        masked_weight = types_mask.float().view(-1)
        if not torch.any(masked_weight > 0.0):
            return None
        return (per_row_loss * masked_weight).sum() / masked_weight.sum().clamp_min(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        types_labels: torch.Tensor | None = None,
        types_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        encoder_outputs = self.encoder(**encoder_kwargs)
        pooled_output = self.dropout(self.pooled_output(encoder_outputs))
        if self.severity_head == "softmax":
            classifier_dtype = self.severity_classifier.weight.dtype
        elif self.severity_head == "corn":
            classifier_dtype = self.ordinal_classifier.weight.dtype
        else:
            classifier_dtype = self.ordinal_ranker.weight.dtype
        if pooled_output.dtype != classifier_dtype:
            pooled_output = pooled_output.to(classifier_dtype)
        if self.severity_head == "softmax":
            severity_loss_logits = self.severity_classifier(pooled_output)
            severity_logits = severity_loss_logits
        else:
            severity_loss_logits = self.ordinal_threshold_logits(pooled_output)
            severity_probabilities = self.ordinal_logits_to_class_probabilities(severity_loss_logits)
            severity_logits = torch.log(severity_probabilities)
        type_logits = self.type_classifier(pooled_output)

        loss = None
        if labels is not None:
            severity_loss = self.compute_severity_loss(severity_loss_logits, labels.long())
            loss = severity_loss
            if types_labels is not None:
                auxiliary_loss = self.compute_type_loss(type_logits, types_labels.float(), types_mask)
                if auxiliary_loss is not None:
                    loss = loss + (self.type_loss_weight * auxiliary_loss)

        return {
            "loss": loss,
            "logits": severity_logits,
            "severity_logits": severity_logits,
            "type_logits": type_logits,
        }


def build_compute_metrics():
    def compute_metrics(eval_prediction) -> dict[str, float]:
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        label_ids = eval_prediction.label_ids
        if isinstance(label_ids, tuple):
            label_ids = label_ids[0]
        predicted_ids = np.asarray(logits).argmax(axis=1)
        gold_names = [severity_id_to_name(label_id) for label_id in label_ids]
        predicted_names = [severity_id_to_name(label_id) for label_id in predicted_ids]
        metrics = compute_multiclass_metrics(gold_names, predicted_names, SUBTASK1_LABELS)
        return {
            "accuracy": float(metrics["accuracy"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
            "micro_f1": float(metrics["micro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
        }

    return compute_metrics


def write_submission(path: Path, prediction_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for prediction_id in prediction_ids:
            writer.writerow([prediction_id])


def write_verbose_csv(
    path: Path,
    eval_rows: list[dict[str, Any]],
    severity_probabilities: np.ndarray,
    predicted_ids: np.ndarray,
) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, probabilities, predicted_id in zip(eval_rows, severity_probabilities, predicted_ids):
            sorted_probabilities = np.sort(probabilities)
            margin = float(sorted_probabilities[-1] - sorted_probabilities[-2]) if len(sorted_probabilities) > 1 else 0.0
            payload: dict[str, Any] = {
                "record_id": row["record_id"],
                "gold_class_id": row["labels"],
                "gold_class": severity_id_to_name(row["labels"]),
                "pred_class_id": int(predicted_id),
                "pred_class": severity_id_to_name(int(predicted_id)),
                "confidence": float(probabilities[int(predicted_id)]),
                "margin": margin,
            }
            for label_index, label_name in enumerate(SUBTASK1_LABELS):
                payload[f"prob_{label_name.lower()}"] = float(probabilities[label_index])
            writer.writerow(payload)


def markdown_lines(payload: dict[str, Any]) -> list[str]:
    devel_metrics = payload["eval_metrics"]
    label_augmentation = payload.get("label_augmentation") or payload.get("severe_augmentation") or {}
    lines = [
        "# Experimento competitivo de severidad con encoder",
        "",
        f"- generado: {payload['generated_at']}",
        f"- modelo: {payload['model_name_or_path']}",
        f"- split train: {payload['train_split']}",
        f"- split eval: {payload['eval_split']}",
        f"- runtime_seconds: {payload['runtime_seconds']:.2f}",
        f"- train_rows: {payload['train_rows']}",
        f"- eval_rows: {payload['eval_rows']}",
        f"- train_rows_with_type_labels: {payload['train_rows_with_type_labels']}",
        f"- eval_rows_with_type_labels: {payload['eval_rows_with_type_labels']}",
        f"- loss_type: {payload['loss_type']}",
        f"- severity_head: {payload['severity_head']}",
        f"- ordinal_distance_weight: {payload['ordinal_distance_weight']:.4f}",
        f"- severity_class_weight: {payload['severity_class_weight']}",
        f"- type_label_source: {payload['type_label_source']}",
        f"- type_loss_weight: {payload['type_loss_weight']:.4f}",
    ]
    if label_augmentation.get("mode") != "none":
        target_label_name = label_augmentation.get("target_label_name")
        if not target_label_name and "target_label_id" in label_augmentation:
            target_label_name = severity_id_to_name(int(label_augmentation["target_label_id"]))
        lines.extend(
            [
                f"- augment_mode: {label_augmentation['mode']}",
                f"- augment_target_label: {target_label_name}",
                f"- augment_target_ratio: {label_augmentation['target_ratio']:.4f}",
                f"- augment_added_rows: {label_augmentation['added_rows']}",
                f"- augment_final_ratio: {label_augmentation['final_target_ratio']:.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Subtask 1",
            f"- accuracy: {devel_metrics['accuracy']:.4f}",
            f"- macro_precision: {devel_metrics['macro_precision']:.4f}",
            f"- macro_recall: {devel_metrics['macro_recall']:.4f}",
            f"- macro_f1: {devel_metrics['macro_f1']:.4f}",
            f"- micro_f1: {devel_metrics['micro_f1']:.4f}",
            f"- weighted_f1: {devel_metrics['weighted_f1']:.4f}",
            "",
            "## Per-label",
        ]
    )
    for label_name in SUBTASK1_LABELS:
        label_metrics = devel_metrics["per_label"][label_name]
        lines.append(
            "- "
            f"{label_name}: precision={label_metrics['precision']:.4f}, "
            f"recall={label_metrics['recall']:.4f}, f1={label_metrics['f1']:.4f}, "
            f"support={label_metrics['support']}"
        )
    type_metrics = payload.get("eval_type_metrics")
    if type_metrics is not None:
        lines.extend(
            [
                "",
                "## Subtask 2 Auxiliar",
                f"- exact_match_accuracy: {type_metrics['exact_match_accuracy']:.4f}",
                f"- macro_f1: {type_metrics['macro_f1']:.4f}",
                f"- micro_f1: {type_metrics['micro_f1']:.4f}",
                f"- weighted_f1: {type_metrics['weighted_f1']:.4f}",
                f"- hamming_loss: {type_metrics['hamming_loss']:.4f}",
            ]
        )
    return lines


def main() -> None:
    args = parse_args()
    configure_hf_backend()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    maybe_pad_token(tokenizer)

    train_type_map = build_type_label_map(args.train_split, args.data_dir, args.type_label_source)
    eval_type_map = hard_label_map(args.eval_split, args.data_dir)
    train_rows = build_rows(
        args.train_split,
        args.data_dir,
        type_label_map=train_type_map,
        limit=args.limit_train,
    )
    augment_target_label_id = severity_name_to_id(args.augment_target_label)
    train_rows, label_augmentation = augment_rows_for_label(
        train_rows,
        target_label_id=augment_target_label_id,
        mode=args.augment_mode,
        target_ratio=args.augment_target_ratio,
        max_copies_per_row=args.augment_max_copies,
        seed=args.seed,
    )
    label_augmentation = dict(label_augmentation)
    label_augmentation["target_label_name"] = args.augment_target_label
    eval_rows = build_rows(
        args.eval_split,
        args.data_dir,
        type_label_map=eval_type_map,
        limit=args.limit_eval,
    )

    train_dataset = tokenize_rows(train_rows, tokenizer, args.max_seq_length)
    eval_dataset = tokenize_rows(eval_rows, tokenizer, args.max_seq_length)

    severity_class_weights = compute_class_weights(train_rows, args.severity_class_weight)
    type_pos_weight = compute_type_pos_weight(train_rows) if args.type_use_pos_weight else None

    model = MultitaskSeverityEncoder(
        args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        severity_head=args.severity_head,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        ordinal_distance_weight=args.ordinal_distance_weight,
        severity_class_weights=severity_class_weights,
        type_loss_weight=args.type_loss_weight,
        type_pos_weight=type_pos_weight,
    )
    if getattr(model.encoder.config, "use_cache", None) is not None:
        model.encoder.config.use_cache = False

    bf16, fp16 = resolve_precision_flags(args.precision)
    eval_strategy, save_strategy = resolve_strategy(args.max_steps)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
        args=TrainingArguments(
            output_dir=str(output_dir),
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            optim=args.optim,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            eval_strategy=eval_strategy,
            eval_steps=args.eval_steps if eval_strategy == "steps" else None,
            save_strategy=save_strategy,
            save_steps=args.save_steps if save_strategy == "steps" else None,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_macro_f1",
            greater_is_better=True,
            report_to="none",
            bf16=bf16,
            fp16=fp16,
            use_cpu=args.use_cpu,
            seed=args.seed,
            gradient_checkpointing=args.gradient_checkpointing,
            dataloader_num_workers=args.dataloader_num_workers,
            remove_unused_columns=True,
            save_only_model=False,
        ),
    )

    start_time = perf_counter()
    train_result = trainer.train()
    eval_prediction = trainer.predict(eval_dataset)
    runtime_seconds = perf_counter() - start_time

    prediction_logits = eval_prediction.predictions
    if isinstance(prediction_logits, tuple):
        prediction_logits = prediction_logits[0]
    prediction_logits = np.asarray(prediction_logits)
    severity_probabilities = torch.softmax(torch.tensor(prediction_logits), dim=1).numpy()
    predicted_ids = prediction_logits.argmax(axis=1)
    gold_names = [severity_id_to_name(row["labels"]) for row in eval_rows]
    predicted_names = [severity_id_to_name(label_id) for label_id in predicted_ids]
    eval_metrics = compute_multiclass_metrics(gold_names, predicted_names, SUBTASK1_LABELS)

    eval_type_metrics = None
    eval_mask = np.asarray([row["types_mask"] for row in eval_rows], dtype=np.float32) > 0.0
    if np.any(eval_mask):
        with torch.no_grad():
            type_logits = []
            dataloader = trainer.get_eval_dataloader(eval_dataset)
            model_for_eval = trainer.model
            model_for_eval.eval()
            for batch in dataloader:
                batch = trainer._prepare_inputs(batch)
                outputs = model_for_eval(**batch)
                type_logits.append(outputs["type_logits"].detach().cpu())
        all_type_logits = torch.cat(type_logits, dim=0)
        type_probabilities = torch.sigmoid(all_type_logits).numpy()
        type_predictions = (type_probabilities >= args.type_threshold).astype(int)
        gold_type_labels = np.asarray([row["types_labels"] for row in eval_rows], dtype=np.int64)
        eval_type_metrics = compute_multilabel_metrics(
            gold_type_labels[eval_mask].tolist(),
            type_predictions[eval_mask].tolist(),
            SUBTASK2_LABELS,
        )

    write_submission(output_dir / "submission" / "subtask1.csv", predicted_ids.tolist())
    write_verbose_csv(output_dir / "subtask1_verbose.csv", eval_rows, severity_probabilities, predicted_ids)
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    with (output_dir / "train_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": runtime_seconds,
        "model_name_or_path": args.model_name_or_path,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "attn_implementation": args.attn_implementation,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_rows_with_type_labels": int(sum(row["types_mask"] for row in train_rows)),
        "eval_rows_with_type_labels": int(sum(row["types_mask"] for row in eval_rows)),
        "loss_type": args.loss_type,
        "severity_head": args.severity_head,
        "ordinal_distance_weight": args.ordinal_distance_weight,
        "severity_class_weight": args.severity_class_weight,
        "severity_class_weights": severity_class_weights,
        "optim": args.optim,
        "max_grad_norm": args.max_grad_norm,
        "label_augmentation": label_augmentation,
        "severe_augmentation": label_augmentation,
        "type_label_source": args.type_label_source,
        "type_loss_weight": args.type_loss_weight,
        "type_pos_weight": type_pos_weight,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "eval_type_metrics": eval_type_metrics,
    }
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (output_dir / "report.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown_lines(summary)))

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
