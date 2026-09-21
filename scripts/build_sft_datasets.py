from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from womenhelp_competition.data import (  # noqa: E402
    label_names_from_vector,
    load_joint_records,
    load_subtask1,
    load_subtask2,
)
from womenhelp_competition.labels import SUBTASK1_ID_TO_NAME  # noqa: E402
from womenhelp_competition.prompts import (  # noqa: E402
    BASE_SYSTEM_PROMPT,
    build_joint_prompt,
    build_severity_prompt,
    build_types_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye datasets SFT para WomenHelp 2026")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "sft_data"),
    )
    return parser.parse_args()


def build_messages(prompt: str, response_payload: dict[str, object]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": json.dumps(response_payload, ensure_ascii=False),
        },
    ]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_severity_rows(split: str, data_dir: str | None) -> list[dict[str, object]]:
    rows = []
    for record in load_subtask1(split, data_dir):
        rows.append(
            {
                "record_id": record.record_id,
                "task": "severity_only",
                "messages": build_messages(
                    build_severity_prompt(record.text),
                    {"severity": SUBTASK1_ID_TO_NAME[record.severity_id]},
                ),
            }
        )
    return rows


def build_types_rows(split: str, data_dir: str | None) -> list[dict[str, object]]:
    rows = []
    for record in load_subtask2(split, data_dir):
        rows.append(
            {
                "record_id": record.record_id,
                "task": "types_only",
                "messages": build_messages(
                    build_types_prompt(record.text),
                    {"violence_types": label_names_from_vector(record.label_vector)},
                ),
            }
        )
    return rows


def build_joint_rows(split: str, data_dir: str | None) -> list[dict[str, object]]:
    rows = []
    for record in load_joint_records(split, data_dir):
        rows.append(
            {
                "record_id": record.record_id,
                "task": "joint",
                "messages": build_messages(
                    build_joint_prompt(record.text),
                    {
                        "severity": SUBTASK1_ID_TO_NAME[record.severity_id],
                        "violence_types": label_names_from_vector(record.label_vector),
                    },
                ),
            }
        )
    return rows


def build_joint_aligned_rows(split: str, data_dir: str | None) -> list[dict[str, object]]:
    rows = []
    for record in load_joint_records(split, data_dir):
        rows.append(
            {
                "record_id": record.record_id,
                "task": "joint_aligned",
                "messages": build_messages(
                    build_joint_prompt(record.text),
                    {
                        "severity": SUBTASK1_ID_TO_NAME[record.severity_id],
                        "violence_types": label_names_from_vector(record.label_vector),
                    },
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for split in ["train", "devel"]:
        severity_rows = build_severity_rows(split, args.data_dir)
        types_rows = build_types_rows(split, args.data_dir)
        joint_rows = build_joint_rows(split, args.data_dir)
        joint_aligned_rows = build_joint_aligned_rows(split, args.data_dir)
        mixture_rows = severity_rows + types_rows + joint_rows

        write_jsonl(output_dir / f"severity_{split}.jsonl", severity_rows)
        write_jsonl(output_dir / f"types_{split}.jsonl", types_rows)
        write_jsonl(output_dir / f"joint_{split}.jsonl", joint_rows)
        write_jsonl(output_dir / f"joint_aligned_{split}.jsonl", joint_aligned_rows)
        write_jsonl(output_dir / f"mixture_{split}.jsonl", mixture_rows)

        manifest[split] = {
            "severity_only": len(severity_rows),
            "types_only": len(types_rows),
            "joint": len(joint_rows),
            "joint_aligned": len(joint_aligned_rows),
            "mixture": len(mixture_rows),
        }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
