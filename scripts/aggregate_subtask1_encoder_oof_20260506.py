from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_competitive_severity_ensemble import probabilities_to_indices  # noqa: E402
from search_subtask1_oof_ordinal_fusion_20260506 import (  # noqa: E402
    apply_thresholds,
    compact_metrics,
    metric_bundle,
    search_ordinal_thresholds,
)
from womenhelp_competition.labels import SUBTASK1_LABELS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agrega folds OOF de encoder S1 y audita devel por promedio de folds")
    parser.add_argument("--fold-dir", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def probability_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(row["probabilities"][label]) for label in SUBTASK1_LABELS] for row in rows],
        dtype=np.float64,
    )


def gold_vector(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["gold_class_id"]) for row in rows], dtype=np.int64)


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: str(row["record_id"]))


def aggregate_devel(fold_dirs: list[Path]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fold_dir in fold_dirs:
        for row in read_jsonl(fold_dir / "subtask1_devel_predictions.jsonl"):
            grouped[str(row["record_id"])].append(row)

    rows: list[dict[str, Any]] = []
    for record_id, fold_rows in grouped.items():
        gold_values = {int(row["gold_class_id"]) for row in fold_rows}
        if len(gold_values) != 1:
            raise ValueError(f"gold mismatch for devel record {record_id}")
        probabilities = np.asarray(
            [[float(row["probabilities"][label]) for label in SUBTASK1_LABELS] for row in fold_rows],
            dtype=np.float64,
        ).mean(axis=0)
        predicted_id = int(probabilities.argmax())
        first = fold_rows[0]
        rows.append(
            {
                "record_id": record_id,
                "source": "devel_fold_average",
                "gold_class_id": int(first["gold_class_id"]),
                "gold_class": first["gold_class"],
                "pred_class_id": predicted_id,
                "pred_class": SUBTASK1_LABELS[predicted_id],
                "probabilities": {
                    label: float(probabilities[index]) for index, label in enumerate(SUBTASK1_LABELS)
                },
                "fold_count": len(fold_rows),
                "text_length_chars": int(first.get("text_length_chars", 0)),
            }
        )
    return sort_rows(rows)


def aggregate_oof(fold_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold_dir in fold_dirs:
        for row in read_jsonl(fold_dir / "subtask1_oof_predictions.jsonl"):
            record_id = str(row["record_id"])
            if record_id in seen:
                raise ValueError(f"duplicate OOF record_id: {record_id}")
            seen.add(record_id)
            rows.append(row)
    return sort_rows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = gold_vector(rows)
    probabilities = probability_matrix(rows)
    return metric_bundle(gold, probabilities_to_indices(probabilities))


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Subtask 1 encoder OOF aggregation",
        "",
        f"- OOF rows: {summary['oof_rows']}",
        f"- devel rows: {summary['devel_rows']}",
        f"- fold_dirs: {len(summary['fold_dirs'])}",
        "",
        "## Metricas",
        "",
        "| modo | OOF macro | devel macro | OOF Severe F1 | devel Severe F1 | OOF catastrophic | devel catastrophic | thresholds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in ["argmax", "oof_ordinal_thresholds"]:
        oof = summary[key]["oof_metrics"]
        devel = summary[key]["devel_metrics"]
        lines.append(
            f"| {key} | {oof['macro_f1']:.6f} | {devel['macro_f1']:.6f} | "
            f"{oof['severe_f1']:.6f} | {devel['severe_f1']:.6f} | "
            f"{oof['catastrophic_rate']:.6f} | {devel['catastrophic_rate']:.6f} | "
            f"`{summary[key].get('thresholds', '')}` |"
        )
    lines.extend(["", "## Devel per-label con thresholds OOF", ""])
    devel_per_label = summary["oof_ordinal_thresholds"]["devel_metrics"]["per_label"]
    lines.extend(["| label | precision | recall | f1 | support |", "| --- | ---: | ---: | ---: | ---: |"])
    for label in SUBTASK1_LABELS:
        row = devel_per_label[label]
        lines.append(
            f"| {label} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['support']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    oof_rows = aggregate_oof(args.fold_dir)
    devel_rows = aggregate_devel(args.fold_dir)

    oof_gold = gold_vector(oof_rows)
    devel_gold = gold_vector(devel_rows)
    oof_probabilities = probability_matrix(oof_rows)
    devel_probabilities = probability_matrix(devel_rows)

    argmax_oof = evaluate_rows(oof_rows)
    argmax_devel = evaluate_rows(devel_rows)
    tuned_oof = search_ordinal_thresholds(oof_gold, oof_probabilities)
    thresholds = tuned_oof["thresholds"]
    tuned_devel = apply_thresholds(devel_gold, devel_probabilities, thresholds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "oof_predictions.jsonl", oof_rows)
    write_jsonl(args.output_dir / "devel_predictions.jsonl", devel_rows)
    summary = {
        "fold_dirs": [str(path) for path in args.fold_dir],
        "oof_rows": len(oof_rows),
        "devel_rows": len(devel_rows),
        "argmax": {
            "oof_metrics": argmax_oof,
            "devel_metrics": argmax_devel,
        },
        "oof_ordinal_thresholds": {
            "thresholds": thresholds,
            "oof_metrics": compact_metrics(tuned_oof),
            "devel_metrics": tuned_devel,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "report.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
