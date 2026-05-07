from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def dump_json(payload: Any, path: str | Path) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def dump_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser()
    if source.suffix == ".jsonl":
        return load_jsonl(source)
    payload = load_json(source)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"unsupported record file: {source}")


def compact_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def stable_key(value: str) -> str:
    import hashlib

    return hashlib.md5(value.encode("utf-8")).hexdigest()
