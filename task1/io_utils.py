from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten_result(record: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "model": record.get("model"),
        "recbole_model": record.get("recbole_model"),
        "parameters": json.dumps(record.get("parameters", {}), sort_keys=True),
        "best_valid_score": record.get("best_valid_score"),
        "duration_seconds": record.get("duration_seconds"),
        "seed": record.get("seed"),
        "device": record.get("device"),
        "checkpoint": record.get("checkpoint"),
        "candidate_key": record.get("candidate_key"),
        "status": record.get("status", "ok"),
        "error": record.get("error"),
    }
    for prefix in ("best_valid_result", "test_result"):
        for metric, value in (record.get(prefix) or {}).items():
            flat[f"{prefix}.{metric}"] = value
    return flat

