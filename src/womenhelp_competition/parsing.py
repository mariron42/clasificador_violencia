from __future__ import annotations

import json
from typing import Any

from .labels import SEVERITY_ALIASES, SUBTASK2_LABELS, TYPE_ALIASES


def extract_json_candidate(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text.strip()
    return text[start : end + 1].strip()


def _normalize_token(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def normalize_severity(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = str(int(value))
    normalized = _normalize_token(str(value))
    return SEVERITY_ALIASES.get(normalized)


def normalize_types(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            maybe_json = json.loads(stripped)
            if isinstance(maybe_json, list):
                raw_items = maybe_json
            else:
                raw_items = [stripped]
        except json.JSONDecodeError:
            raw_items = [part.strip() for part in stripped.replace(";", ",").split(",") if part.strip()]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = [str(value)]

    resolved: list[str] = []
    for item in raw_items:
        normalized = TYPE_ALIASES.get(_normalize_token(str(item)))
        if normalized is not None and normalized not in resolved:
            resolved.append(normalized)

    if "N/A" in resolved and len(resolved) > 1:
        resolved = [label for label in resolved if label != "N/A"]

    return [label for label in SUBTASK2_LABELS if label in resolved]


def parse_joint_prediction(raw_text: str) -> dict[str, Any]:
    candidate = extract_json_candidate(raw_text)
    parsed: dict[str, Any] = {}
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            parsed = payload
    except json.JSONDecodeError:
        parsed = {}

    severity = normalize_severity(
        parsed.get("severity")
        or parsed.get("class")
        or parsed.get("violence_degree")
        or parsed.get("degree")
    )
    violence_types = normalize_types(
        parsed.get("violence_types")
        or parsed.get("types")
        or parsed.get("labels")
        or parsed.get("violence_labels")
    )

    return {
        "severity": severity,
        "violence_types": violence_types,
    }
