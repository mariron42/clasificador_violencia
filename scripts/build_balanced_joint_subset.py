from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from womenhelp_competition.data import label_names_from_vector, load_joint_records  # noqa: E402
from womenhelp_competition.labels import SUBTASK1_ID_TO_NAME, SUBTASK2_COLUMNS  # noqa: E402
from womenhelp_competition.prompts import BASE_SYSTEM_PROMPT, build_joint_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye subconjuntos joint balanceados para fine-tuning y evaluación")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "sft_data" / "balanced_joint_cap_20260419"),
    )
    parser.add_argument("--train-cap", type=int, default=6)
    parser.add_argument("--devel-cap", type=int, default=4)
    parser.add_argument("--min-train-support", type=int, default=1)
    parser.add_argument("--min-devel-support", type=int, default=1)
    return parser.parse_args()


def combo_key(record) -> tuple[str, tuple[int, ...]]:
    return (
        record.severity_id,
        tuple(int(record.label_vector[column]) for column in SUBTASK2_COLUMNS),
    )


def build_messages(text: str, severity_id: str, label_vector: dict[str, int]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": build_joint_prompt(text)},
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "severity": SUBTASK1_ID_TO_NAME[severity_id],
                    "violence_types": label_names_from_vector(label_vector),
                },
                ensure_ascii=False,
            ),
        },
    ]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def capped_selection(records: list[object], cap: int, min_support: int) -> tuple[list[object], list[dict[str, object]], Counter]:
    grouped: dict[tuple[str, tuple[int, ...]], list[object]] = defaultdict(list)
    for record in sorted(records, key=lambda item: int(item.record_id)):
        grouped[combo_key(record)].append(record)

    selected: list[object] = []
    combo_rows: list[dict[str, object]] = []
    support_counter = Counter({key: len(rows) for key, rows in grouped.items()})
    ordered_keys = sorted(
        grouped,
        key=lambda key: (
            support_counter[key],
            int(grouped[key][0].record_id),
            int(key[0]),
            key[1],
        ),
    )

    for key in ordered_keys:
        support = support_counter[key]
        if support < min_support:
            continue
        chosen = grouped[key][: min(cap, support)]
        selected.extend(chosen)
        combo_rows.append(
            {
                "severity_id": key[0],
                "severity": SUBTASK1_ID_TO_NAME[key[0]],
                "label_vector": list(key[1]),
                "support": support,
                "selected": len(chosen),
                "selected_record_ids": [record.record_id for record in chosen],
            }
        )

    return selected, combo_rows, support_counter


def build_rows(split: str, selected_records: list[object], support_counter: Counter) -> list[dict[str, object]]:
    rows = []
    for record in selected_records:
        key = combo_key(record)
        rows.append(
            {
                "record_id": record.record_id,
                "split": split,
                "task": "joint_balanced_cap",
                "severity": SUBTASK1_ID_TO_NAME[record.severity_id],
                "violence_types": label_names_from_vector(record.label_vector),
                "combo_support": support_counter[key],
                "combo_vector": list(key[1]),
                "messages": build_messages(record.text, record.severity_id, record.label_vector),
            }
        )
    return rows


def support_summary(counter: Counter) -> dict[str, int]:
    values = sorted(counter.values())
    if not values:
        return {"unique_combos": 0, "min": 0, "median": 0, "max": 0}
    return {
        "unique_combos": len(values),
        "min": values[0],
        "median": values[len(values) // 2],
        "max": values[-1],
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "selection_strategy": "capped_per_exact_joint_combination",
        "selection_notes": (
            "Se preservan todas las combinaciones observadas con soporte suficiente y se limita el número de ejemplos por combinación."
        ),
    }

    split_configs = {
        "train": {"cap": args.train_cap, "min_support": args.min_train_support},
        "devel": {"cap": args.devel_cap, "min_support": args.min_devel_support},
    }

    for split, config in split_configs.items():
        records = load_joint_records(split, args.data_dir)
        selected_records, combo_rows, support_counter = capped_selection(
            records,
            config["cap"],
            config["min_support"],
        )
        rows = build_rows(split, selected_records, support_counter)
        record_ids = [record.record_id for record in selected_records]

        write_jsonl(output_dir / f"joint_balanced_{split}.jsonl", rows)
        write_jsonl(output_dir / f"joint_balanced_{split}_combos.jsonl", combo_rows)
        write_lines(output_dir / f"joint_balanced_{split}_record_ids.txt", record_ids)

        manifest[split] = {
            "raw_joint_records": len(records),
            "selected_records": len(selected_records),
            "cap": config["cap"],
            "min_support": config["min_support"],
            "support_summary": support_summary(support_counter),
            "selected_unique_combos": len(combo_rows),
        }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()