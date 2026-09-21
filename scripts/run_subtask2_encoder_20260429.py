from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from womenhelp_competition.data import label_names_from_vector, load_subtask2, slice_records
from womenhelp_competition.hf_utils import configure_hf_backend
from womenhelp_competition.labels import SUBTASK2_COLUMNS, SUBTASK2_LABELS
from womenhelp_competition.metrics import compute_multilabel_metrics

SEED = 3407
NON_NA_LABELS = [label for label in SUBTASK2_LABELS if label != "N/A"]
NON_NA_COLUMNS = SUBTASK2_COLUMNS[: len(NON_NA_LABELS)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning dedicado de encoder para Subtask 2")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="devel")
    parser.add_argument("--model-name-or-path", default="dccuchile/bert-base-spanish-wwm-uncased")
    parser.add_argument("--attn-implementation", choices=["auto", "eager", "sdpa"], default="eager")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "subtask2_encoder" / "20260429_beto_unc"))
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float, default=4.0)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="fp32")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pos-weight", choices=["none", "balanced", "sqrt_balanced"], default="sqrt_balanced")
    parser.add_argument("--threshold-grid", default="0.25,0.275,0.30,0.325,0.35,0.375,0.40,0.425,0.45,0.475,0.50,0.525,0.55,0.575,0.60,0.625,0.65")
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
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


def sanitize_text(text: str) -> str:
    return " ".join(text.split())


def build_rows(split: str, data_dir: str | None, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in slice_records(load_subtask2(split, data_dir), limit):
        rows.append(
            {
                "record_id": record.record_id,
                "text": sanitize_text(record.text),
                "labels": [float(int(record.label_vector[column])) for column in NON_NA_COLUMNS],
                "gold_vector": [int(record.label_vector[column]) for column in SUBTASK2_COLUMNS],
                "gold_types": label_names_from_vector(record.label_vector),
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

    return dataset.map(tokenize_batch, batched=True, remove_columns=["record_id", "text", "gold_vector", "gold_types"])


def compute_pos_weight(rows: list[dict[str, Any]], mode: str) -> list[float] | None:
    if mode == "none":
        return None
    labels = np.asarray([row["labels"] for row in rows], dtype=np.float64)
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    positives = np.clip(positives, 1.0, None)
    weights = negatives / positives
    if mode == "sqrt_balanced":
        weights = np.sqrt(weights)
    return weights.tolist()


class Subtask2Encoder(nn.Module):
    def __init__(self, model_name_or_path: str, *, attn_implementation: str, pos_weight: list[float] | None) -> None:
        super().__init__()
        model_kwargs: dict[str, Any] = {"trust_remote_code": True, "use_safetensors": True}
        if attn_implementation != "auto":
            model_kwargs["attn_implementation"] = attn_implementation
        self.encoder = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
        hidden_size = getattr(self.encoder.config, "hidden_size")
        dropout_prob = getattr(self.encoder.config, "classifier_dropout", None)
        if dropout_prob is None:
            dropout_prob = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, len(NON_NA_LABELS))
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32), persistent=False)
        else:
            self.pos_weight = None

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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        encoder_outputs = self.encoder(**encoder_kwargs)
        pooled = self.dropout(self.pooled_output(encoder_outputs))
        if pooled.dtype != self.classifier.weight.dtype:
            pooled = pooled.to(self.classifier.weight.dtype)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=self.pos_weight)
        return {"loss": loss, "logits": logits}


def vectors_from_probabilities(probabilities: np.ndarray, thresholds: dict[str, float]) -> list[list[int]]:
    threshold_array = np.asarray([thresholds[label] for label in NON_NA_LABELS], dtype=np.float64)
    vectors: list[list[int]] = []
    for row in probabilities:
        active_indices = np.flatnonzero(row >= threshold_array).tolist()
        vector = [0 for _ in SUBTASK2_LABELS]
        for index in active_indices:
            vector[index] = 1
        if not active_indices:
            vector[-1] = 1
        vectors.append(vector)
    return vectors


def metric_key(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(metrics["macro_f1"]),
        float(metrics["micro_f1"]),
        float(metrics["exact_match_accuracy"]),
        float(metrics["weighted_f1"]),
        -float(metrics["hamming_loss"]),
    )


def optimize_thresholds(
    probabilities: np.ndarray,
    gold_vectors: list[list[int]],
    threshold_grid: list[float],
    max_passes: int,
) -> tuple[dict[str, float], list[list[int]], dict[str, Any], list[dict[str, Any]]]:
    thresholds = {label: 0.45 for label in NON_NA_LABELS}
    vectors = vectors_from_probabilities(probabilities, thresholds)
    current_metrics = compute_multilabel_metrics(gold_vectors, vectors, SUBTASK2_LABELS)
    history: list[dict[str, Any]] = []
    for pass_index in range(1, max_passes + 1):
        improved = False
        for label in NON_NA_LABELS:
            best_thresholds = dict(thresholds)
            best_vectors = vectors
            best_metrics = current_metrics
            best_key = metric_key(current_metrics)
            for candidate in threshold_grid:
                trial_thresholds = dict(thresholds)
                trial_thresholds[label] = float(candidate)
                trial_vectors = vectors_from_probabilities(probabilities, trial_thresholds)
                trial_metrics = compute_multilabel_metrics(gold_vectors, trial_vectors, SUBTASK2_LABELS)
                trial_key = metric_key(trial_metrics)
                if trial_key > best_key:
                    best_thresholds = trial_thresholds
                    best_vectors = trial_vectors
                    best_metrics = trial_metrics
                    best_key = trial_key
            if best_key > metric_key(current_metrics):
                improved = True
                thresholds = best_thresholds
                vectors = best_vectors
                current_metrics = best_metrics
                history.append({"pass": pass_index, "label": label, "threshold": thresholds[label], "metrics": current_metrics})
        if not improved:
            break
    return thresholds, vectors, current_metrics, history


def build_compute_metrics():
    default_thresholds = {label: 0.45 for label in NON_NA_LABELS}

    def compute_metrics(eval_prediction) -> dict[str, float]:
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        labels = eval_prediction.label_ids
        probabilities = torch.sigmoid(torch.tensor(np.asarray(logits))).numpy()
        non_na_vectors = np.asarray(labels, dtype=np.int64)
        gold_vectors = []
        for row in non_na_vectors:
            vector = row.astype(int).tolist() + [0]
            if not any(vector[:-1]):
                vector[-1] = 1
            gold_vectors.append(vector)
        pred_vectors = vectors_from_probabilities(probabilities, default_thresholds)
        metrics = compute_multilabel_metrics(gold_vectors, pred_vectors, SUBTASK2_LABELS)
        return {
            "macro_f1": float(metrics["macro_f1"]),
            "micro_f1": float(metrics["micro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
            "hamming_loss": float(metrics["hamming_loss"]),
        }

    return compute_metrics


def write_outputs(
    output_dir: Path,
    eval_rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    vectors: list[list[int]],
) -> None:
    with (output_dir / "subtask2_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, probability_row, vector in zip(eval_rows, probabilities, vectors):
            pred_types = [label for label, value in zip(SUBTASK2_LABELS, vector) if int(value) == 1]
            margins = {label: float(probability_row[index]) for index, label in enumerate(NON_NA_LABELS)}
            margins["N/A"] = 1.0 if pred_types == ["N/A"] else 0.0
            handle.write(
                json.dumps(
                    {
                        "record_id": row["record_id"],
                        "gold_types": row["gold_types"],
                        "gold_vector": row["gold_vector"],
                        "pred_class": None,
                        "pred_types": pred_types,
                        "positive_margins": margins,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (output_dir / "subtask2_verbose.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["record_id", *SUBTASK2_COLUMNS, "gold_types", "pred_types", "positive_margins"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, probability_row, vector in zip(eval_rows, probabilities, vectors):
            payload = {"record_id": row["record_id"]}
            for column, value in zip(SUBTASK2_COLUMNS, vector):
                payload[column] = int(value)
            payload["gold_types"] = "|".join(row["gold_types"])
            payload["pred_types"] = "|".join(label for label, value in zip(SUBTASK2_LABELS, vector) if int(value) == 1)
            payload["positive_margins"] = json.dumps(
                {label: float(probability_row[index]) for index, label in enumerate(NON_NA_LABELS)},
                ensure_ascii=False,
                sort_keys=True,
            )
            writer.writerow(payload)
    submission_dir = output_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    with (submission_dir / "subtask2.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(vectors)


def build_report(summary: dict[str, Any]) -> str:
    metrics = summary["eval_metrics"]
    lines = [
        "# Encoder dedicado Subtask 2",
        "",
        f"- generado: {summary['generated_at']}",
        f"- modelo: {summary['model_name_or_path']}",
        f"- runtime_seconds: {summary['runtime_seconds']:.2f}",
        f"- train_rows: {summary['train_rows']}",
        f"- eval_rows: {summary['eval_rows']}",
        f"- pos_weight: {summary['pos_weight_mode']}",
        "",
        "## Metricas",
        f"- macro_f1: {metrics['macro_f1']:.6f}",
        f"- micro_f1: {metrics['micro_f1']:.6f}",
        f"- weighted_f1: {metrics['weighted_f1']:.6f}",
        f"- hamming_loss: {metrics['hamming_loss']:.6f}",
        "",
        "## Thresholds",
    ]
    for label, value in summary["thresholds"].items():
        lines.append(f"- {label}: {value:.3f}")
    lines.extend(["", "## Per-label", "", "| label | precision | recall | f1 | support |", "| --- | ---: | ---: | ---: | ---: |"])
    for label in SUBTASK2_LABELS:
        row = metrics["per_label"][label]
        lines.append(f"| {label} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['support']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    configure_hf_backend()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    maybe_pad_token(tokenizer)
    train_rows = build_rows(args.train_split, args.data_dir, args.limit_train)
    eval_rows = build_rows(args.eval_split, args.data_dir, args.limit_eval)
    train_dataset = tokenize_rows(train_rows, tokenizer, args.max_seq_length)
    eval_dataset = tokenize_rows(eval_rows, tokenizer, args.max_seq_length)
    pos_weight = compute_pos_weight(train_rows, args.pos_weight)
    model = Subtask2Encoder(args.model_name_or_path, attn_implementation=args.attn_implementation, pos_weight=pos_weight)
    if getattr(model.encoder.config, "use_cache", None) is not None:
        model.encoder.config.use_cache = False
    bf16, fp16 = resolve_precision_flags(args.precision)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
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
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
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
    prediction = trainer.predict(eval_dataset)
    runtime_seconds = perf_counter() - start_time
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    probabilities = torch.sigmoid(torch.tensor(np.asarray(logits))).numpy()
    threshold_grid = [float(value) for value in args.threshold_grid.split(",") if value.strip()]
    gold_vectors = [row["gold_vector"] for row in eval_rows]
    thresholds, vectors, eval_metrics, history = optimize_thresholds(probabilities, gold_vectors, threshold_grid, args.max_passes)
    write_outputs(output_dir, eval_rows, probabilities, vectors)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": runtime_seconds,
        "model_name_or_path": args.model_name_or_path,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "max_seq_length": args.max_seq_length,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "pos_weight_mode": args.pos_weight,
        "pos_weight": pos_weight,
        "thresholds": thresholds,
        "threshold_history": history,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
