from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .utils import load_json, read_records


LEVEL_DIRS = {
    "perception": "Perception",
    "relational_reasoning": "Relational reasoning",
    "temporal_reasoning": "Temporal reasoning",
}


@dataclass(frozen=True)
class ItemPaths:
    item_dir: Path
    image_t1: Path
    image_t2: Path | None
    scene_graph: Path


def iter_category_files(data_root: str | Path) -> Iterable[Path]:
    root = Path(data_root).expanduser()
    for level_dir in LEVEL_DIRS:
        level_path = root / level_dir
        if not level_path.is_dir():
            continue
        for subdir in sorted(path for path in level_path.iterdir() if path.is_dir()):
            json_path = subdir / f"{subdir.name}.json"
            if json_path.is_file():
                yield json_path


def load_items(data_root: str | Path) -> list[dict[str, Any]]:
    root = Path(data_root).expanduser().resolve()
    items: list[dict[str, Any]] = []
    for json_path in iter_category_files(root):
        rows = read_records(json_path)
        for row in rows:
            items.append(_attach_item_context(row, root, json_path.parent))
    if items:
        return items

    metadata = root / "metadata.jsonl"
    if metadata.is_file():
        rows = read_records(metadata)
        return [_attach_item_context(row, root, metadata.parent) for row in rows]
    return items


def _attach_item_context(row: dict[str, Any], data_root: Path, item_dir: Path) -> dict[str, Any]:
    item = dict(row)
    level_slug = _infer_level_slug(item, item_dir)
    if "level" not in item:
        item["level"] = LEVEL_DIRS.get(level_slug, level_slug)
    if "subcategory" not in item:
        item["subcategory"] = item_dir.name
    item["_data_root"] = str(data_root)
    item["_item_dir"] = str(item_dir)
    item["_level_slug"] = level_slug
    item["_question_key"] = str(item.get("question_id") or item.get("item_id") or "")
    return item


def _infer_level_slug(item: dict[str, Any], item_dir: Path) -> str:
    raw = str(item.get("level") or "").strip().lower().replace(" ", "_")
    if raw.startswith("relational"):
        return "relational_reasoning"
    if raw.startswith("temporal"):
        return "temporal_reasoning"
    if raw.startswith("perception"):
        return "perception"
    try:
        return item_dir.parent.name
    except IndexError:
        return raw


def item_paths(item: dict[str, Any]) -> ItemPaths:
    item_dir = Path(str(item["_item_dir"]))
    images = item.get("images", {})
    if not isinstance(images, dict):
        raise ValueError(f"missing images object for {item_id(item)}")
    t1 = item_dir / str(images.get("t1") or "")
    t2_value = images.get("t2")
    scene_graph = item_dir / str(item.get("scene_graph") or "")
    return ItemPaths(
        item_dir=item_dir,
        image_t1=t1,
        image_t2=(item_dir / str(t2_value)) if t2_value else None,
        scene_graph=scene_graph,
    )


def load_scene_graph(item: dict[str, Any]) -> dict[str, Any]:
    return load_json(item_paths(item).scene_graph)


def item_id(item: dict[str, Any]) -> str:
    return str(item.get("question_id") or item.get("item_id") or item.get("_question_key") or "")


def index_by_id(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("question_id") or row.get("item_id") or row.get("id") or "")
        if key:
            indexed[key] = row
    return indexed
