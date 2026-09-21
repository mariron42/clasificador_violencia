from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_competitive_encoder_severity import (  # noqa: E402
    MultitaskSeverityEncoder,
    build_compute_metrics,
    build_rows,
    build_type_label_map,
    compute_class_weights,
    compute_type_pos_weight,
    hard_label_map,
    maybe_pad_token,
    resolve_precision_flags,
    resolve_strategy,
    severity_id_to_name,
    severity_name_to_id,
    tokenize_rows,
)
from womenhelp_competition.hf_utils import configure_hf_backend  # noqa: E402
from womenhelp_competition.labels import SUBTASK1_LABELS  # noqa: E402
from womenhelp_competition.metrics import compute_multiclass_metrics  # noqa: E402
from womenhelp_competition.text_augmentation import augment_rows_for_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena un fold OOF de encoder S1 usando solo train")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--optim", default="stable_adamw")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="fp32")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--loss-type", choices=["cross_entropy", "focal"], default="focal")
    parser.add_argument("--severity-head", choices=["softmax", "coral", "corn"], default="softmax")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--ordinal-distance-weight", type=float, default=0.05)
    parser.add_argument("--severity-class-weight", choices=["none", "balanced", "sqrt_balanced"], default="balanced")
    parser.add_argument("--type-loss-weight", type=float, default=0.3)
    parser.add_argument("--type-label-source", choices=["none", "hard", "soft"], default="soft")
    parser.add_argument("--type-use-pos-weight", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-devel", type=int, default=0)
    parser.add_argument("--augment-mode", choices=["none", "synonym", "report_style"], default="none")
    parser.add_argument("--augment-target-label", choices=SUBTASK1_LABELS, default="Severe")
    parser.add_argument("--augment-target-ratio", type=float, default=0.0)
    parser.add_argument("--augment-max-copies", type=int, default=2)
    parser.add_argument("--attn-implementation", choices=["auto", "eager", "sdpa"], default="eager")
    return parser.parse_args()


def split_fold(
    rows: list[dict[str, Any]],
    *,
    fold_index: int,
    num_folds: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    labels = np.asarray([int(row["labels"]) for row in rows], dtype=np.int64)
    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.arange(len(rows)), labels))
    train_idx, eval_idx = splits[fold_index]
    return (
        [rows[int(index)] for index in train_idx],
        [rows[int(index)] for index in eval_idx],
        {
            "strategy": "stratified_severity",
            "fold_index": int(fold_index),
            "num_folds": int(num_folds),
            "train_rows_before_augmentation": int(len(train_idx)),
            "eval_rows": int(len(eval_idx)),
            "train_label_counts": {
                label: int(np.sum(labels[train_idx] == index)) for index, label in enumerate(SUBTASK1_LABELS)
            },
            "eval_label_counts": {
                label: int(np.sum(labels[eval_idx] == index)) for index, label in enumerate(SUBTASK1_LABELS)
            },
        },
    )


def prediction_payload(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    *,
    source: str,
) -> list[dict[str, Any]]:
    predicted_ids = probabilities.argmax(axis=1).astype(np.int64)
    payload: list[dict[str, Any]] = []
    for row, probability_row, predicted_id in zip(rows, probabilities, predicted_ids):
        payload.append(
            {
                "record_id": str(row["record_id"]),
                "source": source,
                "gold_class_id": int(row["labels"]),
                "gold_class": severity_id_to_name(int(row["labels"])),
                "pred_class_id": int(predicted_id),
                "pred_class": severity_id_to_name(int(predicted_id)),
                "probabilities": {
                    label: float(probability_row[index]) for index, label in enumerate(SUBTASK1_LABELS)
                },
                "text_length_chars": int(len(str(row.get("text", "")))),
            }
        )
    return payload


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def metrics_from_probabilities(rows: list[dict[str, Any]], probabilities: np.ndarray) -> dict[str, Any]:
    predicted_ids = probabilities.argmax(axis=1).astype(np.int64)
    gold_names = [severity_id_to_name(int(row["labels"])) for row in rows]
    predicted_names = [severity_id_to_name(int(label_id)) for label_id in predicted_ids]
    return compute_multiclass_metrics(gold_names, predicted_names, SUBTASK1_LABELS)


def predict_probabilities(trainer: Trainer, dataset: Dataset) -> np.ndarray:
    prediction = trainer.predict(dataset)
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    return torch.softmax(torch.tensor(logits), dim=1).numpy()


def main() -> None:
    args = parse_args()
    if args.fold_index < 0 or args.fold_index >= args.num_folds:
        raise ValueError("--fold-index must be in [0, num_folds)")

    configure_hf_backend()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_type_map = build_type_label_map("train", args.data_dir, args.type_label_source)
    train_all_rows = build_rows(
        "train",
        args.data_dir,
        type_label_map=train_type_map,
        limit=args.limit_train,
    )
    train_rows, oof_rows, split_info = split_fold(
        train_all_rows,
        fold_index=args.fold_index,
        num_folds=args.num_folds,
        seed=args.seed,
    )

    augment_target_label_id = severity_name_to_id(args.augment_target_label)
    train_rows, label_augmentation = augment_rows_for_label(
        train_rows,
        target_label_id=augment_target_label_id,
        mode=args.augment_mode,
        target_ratio=args.augment_target_ratio,
        max_copies_per_row=args.augment_max_copies,
        seed=args.seed + args.fold_index,
    )
    label_augmentation = dict(label_augmentation)
    label_augmentation["target_label_name"] = args.augment_target_label
    split_info["train_rows_after_augmentation"] = len(train_rows)

    devel_type_map = hard_label_map("devel", args.data_dir)
    devel_rows = build_rows(
        "devel",
        args.data_dir,
        type_label_map=devel_type_map,
        limit=args.limit_devel,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    maybe_pad_token(tokenizer)
    train_dataset = tokenize_rows(train_rows, tokenizer, args.max_seq_length)
    oof_dataset = tokenize_rows(oof_rows, tokenizer, args.max_seq_length)
    devel_dataset = tokenize_rows(devel_rows, tokenizer, args.max_seq_length)

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
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=oof_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(),
        args=TrainingArguments(
            output_dir=str(args.output_dir),
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
    oof_probabilities = predict_probabilities(trainer, oof_dataset)
    devel_probabilities = predict_probabilities(trainer, devel_dataset)
    runtime_seconds = perf_counter() - start_time

    write_jsonl(
        args.output_dir / "subtask1_oof_predictions.jsonl",
        prediction_payload(oof_rows, oof_probabilities, source="train_oof"),
    )
    write_jsonl(
        args.output_dir / "subtask1_devel_predictions.jsonl",
        prediction_payload(devel_rows, devel_probabilities, source="devel_fold_model"),
    )
    tokenizer.save_pretrained(str(args.output_dir / "tokenizer"))
    with (args.output_dir / "train_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False, default=str)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "family": "subtask1_encoder_oof_fold",
        "model_key": args.model_key,
        "model_name_or_path": args.model_name_or_path,
        "runtime_seconds": float(runtime_seconds),
        "split": split_info,
        "max_seq_length": args.max_seq_length,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "loss_type": args.loss_type,
        "severity_head": args.severity_head,
        "ordinal_distance_weight": args.ordinal_distance_weight,
        "severity_class_weight": args.severity_class_weight,
        "severity_class_weights": severity_class_weights,
        "type_label_source": args.type_label_source,
        "type_loss_weight": args.type_loss_weight,
        "type_pos_weight": type_pos_weight,
        "label_augmentation": label_augmentation,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_metrics": train_result.metrics,
        "oof_fold_metrics": metrics_from_probabilities(oof_rows, oof_probabilities),
        "devel_fold_model_metrics": metrics_from_probabilities(devel_rows, devel_probabilities),
    }
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "fold_assignment.json").write_text(
        json.dumps(split_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
