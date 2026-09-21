from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
for candidate in (SCRIPTS_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from run_subtask2_encoder_20260429 import (  # noqa: E402
    NON_NA_LABELS,
    Subtask2Encoder,
    maybe_pad_token,
    sanitize_text,
    vectors_from_probabilities,
)
from womenhelp_competition.hf_utils import configure_hf_backend  # noqa: E402
from womenhelp_competition.labels import SUBTASK2_COLUMNS, SUBTASK2_LABELS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materializa predicciones test desde un encoder S2 fine-tuned")
    parser.add_argument(
        "--summary-json",
        default=str(REPO_ROOT / "outputs" / "subtask2_encoder" / "20260429" / "beto_unc_sqrt_e4" / "metrics_summary.json"),
    )
    parser.add_argument(
        "--test-csv",
        default="data/official_test/subtask2/test.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "subtask2_encoder" / "20260429" / "beto_unc_sqrt_e4_test"),
    )
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.0,
        help="Si es >0, reemplaza los thresholds del summary por este valor global.",
    )
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_test_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            if not row:
                continue
            rows.append({"record_id": str(index), "text": sanitize_text(row[0])})
    return rows


def tokenize_rows(rows: list[dict[str, str]], tokenizer, max_seq_length: int) -> Dataset:
    dataset = Dataset.from_list(rows)

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_length, padding=False)

    return dataset.map(tokenize_batch, batched=True, remove_columns=["record_id", "text"])


def write_outputs(output_dir: Path, rows: list[dict[str, str]], probabilities: np.ndarray, vectors: list[list[int]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "subtask2_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, probability_row, vector in zip(rows, probabilities, vectors):
            pred_types = [label for label, value in zip(SUBTASK2_LABELS, vector) if int(value) == 1]
            margins = {label: float(probability_row[index]) for index, label in enumerate(NON_NA_LABELS)}
            margins["N/A"] = 1.0 if pred_types == ["N/A"] else 0.0
            handle.write(
                json.dumps(
                    {
                        "record_id": row["record_id"],
                        "pred_class": None,
                        "pred_types": pred_types,
                        "positive_margins": margins,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (output_dir / "subtask2_verbose.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["record_id", *SUBTASK2_COLUMNS, "pred_types", "positive_margins"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, probability_row, vector in zip(rows, probabilities, vectors):
            payload = {"record_id": row["record_id"]}
            for column, value in zip(SUBTASK2_COLUMNS, vector):
                payload[column] = int(value)
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


def main() -> None:
    args = parse_args()
    configure_hf_backend()
    summary = read_summary(Path(args.summary_json))
    checkpoint = Path(summary["best_checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
    maybe_pad_token(tokenizer)
    model = Subtask2Encoder(
        summary["model_name_or_path"],
        attn_implementation="eager",
        pos_weight=summary.get("pos_weight"),
    )
    state_dict = load_file(str(checkpoint / "model.safetensors"))
    model.load_state_dict(state_dict)
    rows = read_test_rows(Path(args.test_csv))
    dataset = tokenize_rows(rows, tokenizer, int(summary["max_seq_length"]))
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    logits_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            logits_parts.append(outputs["logits"].detach().cpu())
    logits = torch.cat(logits_parts, dim=0).numpy()
    probabilities = torch.sigmoid(torch.tensor(np.asarray(logits))).numpy()
    thresholds = summary["thresholds"]
    if args.fixed_threshold > 0.0:
        thresholds = {label: float(args.fixed_threshold) for label in NON_NA_LABELS}
    vectors = vectors_from_probabilities(probabilities, thresholds)
    output_dir = Path(args.output_dir)
    write_outputs(output_dir, rows, probabilities, vectors)
    payload = {
        "summary_json": args.summary_json,
        "checkpoint": str(checkpoint),
        "test_csv": args.test_csv,
        "rows": len(rows),
        "thresholds": thresholds,
        "output_dir": str(output_dir),
    }
    (output_dir / "materialization_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
