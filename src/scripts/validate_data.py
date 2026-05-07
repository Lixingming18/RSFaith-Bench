#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

from rsfaith_bench.answer import gold_answer
from rsfaith_bench.data import item_id, item_paths, load_items, load_scene_graph
from rsfaith_bench.verifier import validate_refs


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an RSFaith-Bench dataset directory.")
    parser.add_argument("--data", default="RSFaith-Bench_subset", help="Dataset root.")
    parser.add_argument("--expect-per-category", type=int, default=None)
    args = parser.parse_args()

    items = load_items(args.data)
    counts: Counter[tuple[str, str]] = Counter()
    errors: list[str] = []
    for item in items:
        counts[(str(item.get("level")), str(item.get("subcategory")))] += 1
        qid = item_id(item)
        try:
            paths = item_paths(item)
            check_image(paths.image_t1)
            if paths.image_t2 is not None:
                check_image(paths.image_t2)
            scene_graph = load_scene_graph(item)
            errors.extend(f"{qid}: {err}" for err in validate_refs(item, scene_graph))
            choices = item.get("choices")
            if not isinstance(choices, list) or not choices:
                errors.append(f"{qid}: missing choices")
            elif gold_answer(item) not in [str(choice) for choice in choices]:
                errors.append(f"{qid}: answer is not in choices")
            question = str(item.get("question") or "").strip()
            if not (question.endswith("?") or question.lower().startswith(("count ", "choose "))):
                errors.append(f"{qid}: unsupported question form")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{qid}: {exc}")

    if args.expect_per_category is not None:
        for key, count in counts.items():
            if count != args.expect_per_category:
                errors.append(f"{key}: expected {args.expect_per_category}, got {count}")

    print(f"items: {len(items)}")
    for (level, subcategory), count in sorted(counts.items()):
        print(f"{level}/{subcategory}: {count}")
    if errors:
        print("\nValidation errors:")
        for err in errors[:200]:
            print(f"- {err}")
        raise SystemExit(1)
    print("validation passed")


def check_image(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        if width < 32 or height < 32:
            raise ValueError(f"image too small: {path} {width}x{height}")


if __name__ == "__main__":
    main()
