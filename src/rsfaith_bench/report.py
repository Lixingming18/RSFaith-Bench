from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from .utils import load_jsonl


GROUP_FIELDS = ("model_name", "level", "subcategory")
GROUP_FIELDS_BY_NAME = {
    "overall": ("model_name",),
    "level": ("model_name", "level"),
    "subcategory": GROUP_FIELDS,
}
METRIC_FIELDS = ("aa", "cp", "fa", "C-CUR", "M-CUR")


def summarize_records(
    records: list[dict[str, Any]],
    group_fields: tuple[str, ...] = GROUP_FIELDS,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = tuple(str(row.get(field) or "") for field in group_fields)
        groups[key].append(row)

    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        out = {field: value for field, value in zip(group_fields, key)}
        out["n"] = len(rows)
        for metric in METRIC_FIELDS:
            if metric == "cp":
                vals = [float(row["metrics"][metric]) for row in rows if row.get("metrics", {}).get(metric) is not None]
            else:
                vals = [float(row.get("metrics", {}).get(metric, 0.0)) for row in rows]
            out[metric] = round(sum(vals) / len(vals), 6) if vals else None
        summaries.append(out)
    return summaries


def write_summary_csv(
    records: list[dict[str, Any]],
    path: str | Path,
    group_fields: tuple[str, ...] = GROUP_FIELDS,
) -> None:
    rows = summarize_records(records, group_fields=group_fields)
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [*group_fields, "n", *METRIC_FIELDS]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group_fields_for_name(name: str) -> tuple[str, ...]:
    try:
        return GROUP_FIELDS_BY_NAME[name]
    except KeyError as exc:
        choices = ", ".join(sorted(GROUP_FIELDS_BY_NAME))
        raise ValueError(f"unknown summary group {name!r}; choose one of: {choices}") from exc


def summarize_jsonl(eval_jsonl: str | Path, group_fields: tuple[str, ...] = GROUP_FIELDS) -> list[dict[str, Any]]:
    return summarize_records(load_jsonl(eval_jsonl), group_fields=group_fields)
