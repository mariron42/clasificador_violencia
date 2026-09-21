from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .labels import SUBTASK2_COLUMNS, SUBTASK2_COLUMN_TO_NAME

DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "WOMENHELP_DATA_DIR",
        "data/extracted",
    )
)


@dataclass(frozen=True)
class Subtask1Record:
    record_id: str
    text: str
    severity_id: str


@dataclass(frozen=True)
class Subtask2Record:
    record_id: str
    text: str
    label_vector: dict[str, int]


@dataclass(frozen=True)
class JointRecord:
    record_id: str
    text: str
    severity_id: str
    label_vector: dict[str, int]


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    return Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_subtask1(split: str, data_dir: str | Path | None = None) -> list[Subtask1Record]:
    path = resolve_data_dir(data_dir) / "subtask1" / f"{split}.csv"
    rows = _read_csv(path)
    return [
        Subtask1Record(
            record_id=row["ID"],
            text=row["TEXT"],
            severity_id=row["CLASS"],
        )
        for row in rows
    ]


def load_subtask2(split: str, data_dir: str | Path | None = None) -> list[Subtask2Record]:
    path = resolve_data_dir(data_dir) / "subtask2" / f"{split}.csv"
    rows = _read_csv(path)
    return [
        Subtask2Record(
            record_id=row["ID"],
            text=row["Text"],
            label_vector={column: int(row[column]) for column in SUBTASK2_COLUMNS},
        )
        for row in rows
    ]


def load_subtask2_soft(split: str, data_dir: str | Path | None = None) -> list[dict[str, str]]:
    path = resolve_data_dir(data_dir) / "subtask2" / "SoftLabels" / f"{split}Soft.csv"
    return _read_csv(path)


def load_joint_records(split: str, data_dir: str | Path | None = None) -> list[JointRecord]:
    subtask1 = {record.record_id: record for record in load_subtask1(split, data_dir)}
    subtask2 = {record.record_id: record for record in load_subtask2(split, data_dir)}
    shared_ids = sorted(set(subtask1) & set(subtask2), key=lambda value: int(value))
    return [
        JointRecord(
            record_id=record_id,
            text=subtask1[record_id].text,
            severity_id=subtask1[record_id].severity_id,
            label_vector=subtask2[record_id].label_vector,
        )
        for record_id in shared_ids
    ]


def label_names_from_vector(label_vector: dict[str, int]) -> list[str]:
    return [
        SUBTASK2_COLUMN_TO_NAME[column]
        for column in SUBTASK2_COLUMNS
        if int(label_vector[column]) == 1
    ]


def slice_records(records: Iterable, limit: int | None = None) -> list:
    items = list(records)
    if limit is None or limit <= 0:
        return items
    return items[:limit]
