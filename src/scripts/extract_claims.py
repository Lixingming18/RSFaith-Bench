#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from tqdm.auto import tqdm

from rsfaith_bench.answer import prediction_text
from rsfaith_bench.claims import ExtractionConfig, extract_claims_openai
from rsfaith_bench.data import index_by_id, item_id, load_items
from rsfaith_bench.utils import dump_jsonl, read_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract visual spans and schema claims from model responses.")
    parser.add_argument("--data", default="RSFaith-Bench_subset")
    parser.add_argument("--responses", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("missing --api-key or OPENAI_API_KEY")
    if not args.model:
        raise SystemExit("missing --model or OPENAI_MODEL")

    items = load_items(args.data)
    if args.limit:
        items = items[: args.limit]
    responses = index_by_id(read_records(args.responses))
    config = ExtractionConfig(base_url=args.base_url, model=args.model, api_key=args.api_key)

    rows = []
    for item in tqdm(items, desc="extract"):
        key = item_id(item)
        response = responses.get(key)
        if response is None:
            continue
        rows.append(extract_claims_openai(item, prediction_text(response), config))
    dump_jsonl(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
