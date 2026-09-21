from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from womenhelp_competition.labels import SUBTASK1_LABELS, SUBTASK2_LABELS
from womenhelp_competition.parsing import extract_json_candidate, parse_joint_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida formato y cobertura de salidas conjuntas de Gemma 4")
    parser.add_argument("--subtask1-jsonl", required=True)
    parser.add_argument("--subtask2-jsonl", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def extract_payload(raw_text: str) -> dict[str, Any] | None:
    candidate = extract_json_candidate(raw_text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def ensure_sequence(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def build_summary(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    valid_json_count = 0
    required_keys_count = 0
    exact_keyset_count = 0
    missing_severity_count = 0
    missing_types_count = 0
    illegal_severity_count = 0
    illegal_type_count = 0
    na_mixed_count = 0
    empty_types_count = 0
    severity_distribution: Counter[str] = Counter()
    type_distribution: Counter[str] = Counter()

    for row in rows:
        raw_output = row.get("raw_output", "")
        payload = extract_payload(raw_output)
        parsed = parse_joint_prediction(raw_output)

        if payload is not None:
            valid_json_count += 1
            keyset = set(payload)
            if {"severity", "violence_types"}.issubset(keyset):
                required_keys_count += 1
            if keyset == {"severity", "violence_types"}:
                exact_keyset_count += 1

            if "severity" not in payload:
                missing_severity_count += 1
            if "violence_types" not in payload:
                missing_types_count += 1

            raw_severity = payload.get("severity")
            if raw_severity is not None and parsed["severity"] is None:
                illegal_severity_count += 1

            raw_types = ensure_sequence(payload.get("violence_types"))
            invalid_raw_types = 0
            normalized_types = set(parsed["violence_types"])
            for item in raw_types:
                if isinstance(item, str) and item.strip() == "":
                    invalid_raw_types += 1
                    continue
                if item not in normalized_types and str(item) not in normalized_types:
                    candidate_type = parse_joint_prediction(json.dumps({"violence_types": [item]}))["violence_types"]
                    if not candidate_type:
                        invalid_raw_types += 1
            if invalid_raw_types > 0:
                illegal_type_count += 1
            if "N/A" in raw_types and len(raw_types) > 1:
                na_mixed_count += 1
        else:
            missing_severity_count += 1
            missing_types_count += 1

        severity = parsed["severity"]
        if severity is None:
            missing_severity_count += 0 if payload is None else 0
        else:
            severity_distribution[severity] += 1

        if not parsed["violence_types"]:
            empty_types_count += 1
        for label in parsed["violence_types"]:
            type_distribution[label] += 1

    total = len(rows)
    return {
        "name": name,
        "rows": total,
        "valid_json_rate": safe_div(valid_json_count, total),
        "required_keys_rate": safe_div(required_keys_count, total),
        "exact_keyset_rate": safe_div(exact_keyset_count, total),
        "missing_severity_rate": safe_div(missing_severity_count, total),
        "missing_types_rate": safe_div(missing_types_count, total),
        "illegal_severity_rate": safe_div(illegal_severity_count, total),
        "illegal_type_rate": safe_div(illegal_type_count, total),
        "na_mixed_rate": safe_div(na_mixed_count, total),
        "empty_types_rate": safe_div(empty_types_count, total),
        "severity_distribution": {label: severity_distribution.get(label, 0) for label in SUBTASK1_LABELS},
        "type_distribution": {label: type_distribution.get(label, 0) for label in SUBTASK2_LABELS},
    }


def safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    total_rows = sum(summary["rows"] for summary in summaries)
    severity_distribution: Counter[str] = Counter()
    type_distribution: Counter[str] = Counter()
    for summary in summaries:
        severity_distribution.update(summary["severity_distribution"])
        type_distribution.update(summary["type_distribution"])

    weighted = {}
    rate_keys = [
        "valid_json_rate",
        "required_keys_rate",
        "exact_keyset_rate",
        "missing_severity_rate",
        "missing_types_rate",
        "illegal_severity_rate",
        "illegal_type_rate",
        "na_mixed_rate",
        "empty_types_rate",
    ]
    for key in rate_keys:
        weighted[key] = safe_div(
            sum(summary[key] * summary["rows"] for summary in summaries),
            total_rows,
        )

    weighted["rows"] = total_rows
    weighted["severity_distribution"] = {label: severity_distribution.get(label, 0) for label in SUBTASK1_LABELS}
    weighted["type_distribution"] = {label: type_distribution.get(label, 0) for label in SUBTASK2_LABELS}
    return weighted


def format_distribution(distribution: dict[str, int]) -> str:
    parts = [f"{label}={value}" for label, value in distribution.items()]
    return ", ".join(parts)


def summary_lines(title: str, summary: dict[str, Any]) -> list[str]:
    return [
        f"## {title}",
        f"- filas: {summary['rows']}",
        f"- valid_json_rate: {summary['valid_json_rate']:.4f}",
        f"- required_keys_rate: {summary['required_keys_rate']:.4f}",
        f"- exact_keyset_rate: {summary['exact_keyset_rate']:.4f}",
        f"- missing_severity_rate: {summary['missing_severity_rate']:.4f}",
        f"- missing_types_rate: {summary['missing_types_rate']:.4f}",
        f"- illegal_severity_rate: {summary['illegal_severity_rate']:.4f}",
        f"- illegal_type_rate: {summary['illegal_type_rate']:.4f}",
        f"- na_mixed_rate: {summary['na_mixed_rate']:.4f}",
        f"- empty_types_rate: {summary['empty_types_rate']:.4f}",
        f"- severity_distribution: {format_distribution(summary['severity_distribution'])}",
        f"- type_distribution: {format_distribution(summary['type_distribution'])}",
        "",
    ]


def main() -> None:
    args = parse_args()
    subtask1_rows = read_jsonl(args.subtask1_jsonl)
    subtask2_rows = read_jsonl(args.subtask2_jsonl)
    subtask1_summary = build_summary(subtask1_rows, "subtask1")
    subtask2_summary = build_summary(subtask2_rows, "subtask2")
    aggregate = aggregate_summaries([subtask1_summary, subtask2_summary])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subtask1": subtask1_summary,
        "subtask2": subtask2_summary,
        "aggregate": aggregate,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Validación de formato de salidas conjuntas",
        "",
        f"Generado: {payload['generated_at']}",
        "",
        *summary_lines("Subtask 1", subtask1_summary),
        *summary_lines("Subtask 2", subtask2_summary),
        *summary_lines("Agregado", aggregate),
    ]
    output_markdown = Path(args.output_markdown)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()