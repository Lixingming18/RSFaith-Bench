#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rsfaith_bench.report import group_fields_for_name, write_summary_csv
from rsfaith_bench.utils import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RSFaith-Bench item-level results.")
    parser.add_argument("--eval", required=True, help="Evaluation JSONL.")
    parser.add_argument("--output", required=True, help="Summary CSV path.")
    parser.add_argument(
        "--group",
        choices=("overall", "level", "subcategory"),
        default="subcategory",
        help="Aggregation level.",
    )
    args = parser.parse_args()
    write_summary_csv(load_jsonl(args.eval), args.output, group_fields=group_fields_for_name(args.group))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
