#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tqdm.auto import tqdm

from rsfaith_bench.data import index_by_id, item_id, load_items
from rsfaith_bench.evaluate import evaluate_item
from rsfaith_bench.utils import dump_jsonl, read_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RSFaith-Bench predictions.")
    parser.add_argument("--data", default="RSFaith-Bench_subset", help="Dataset root.")
    parser.add_argument("--pred", required=True, help="Prediction JSON/JSONL.")
    parser.add_argument("--claims", default=None, help="Extracted claims JSON/JSONL; optional if predictions already include claims.")
    parser.add_argument("--output", required=True, help="Item-level evaluation JSONL.")
    args = parser.parse_args()

    items = load_items(args.data)
    predictions = index_by_id(read_records(args.pred))
    claims = index_by_id(read_records(args.claims)) if args.claims else {}

    rows = []
    missing = []
    for item in tqdm(items, desc="evaluate"):
        key = item_id(item)
        prediction = predictions.get(key)
        if prediction is None:
            missing.append(key)
            continue
        rows.append(evaluate_item(item, prediction, claim_record=claims.get(key)))
    if missing:
        print(f"warning: {len(missing)} items missing predictions")
    dump_jsonl(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
