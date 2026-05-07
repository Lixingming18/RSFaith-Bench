#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import requests
from tqdm.auto import tqdm

from rsfaith_bench.data import item_id, item_paths, load_items
from rsfaith_bench.prompts import build_prompt
from rsfaith_bench.utils import dump_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RSFaith-Bench with an OpenAI-compatible vision API.")
    parser.add_argument("--data", default="RSFaith-Bench_subset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("missing --api-key or OPENAI_API_KEY")
    if not args.model:
        raise SystemExit("missing --model or OPENAI_MODEL")

    rows = []
    items = load_items(args.data)
    if args.limit:
        items = items[: args.limit]
    for item in tqdm(items, desc="infer"):
        response = call_api(item, args)
        rows.append(
            {
                "question_id": item_id(item),
                "model_name": args.model,
                "response": response,
            }
        )
    dump_jsonl(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


def call_api(item: dict[str, Any], args: argparse.Namespace) -> str:
    paths = item_paths(item)
    content: list[dict[str, Any]] = []
    content.append(image_block(paths.image_t1))
    if paths.image_t2 is not None:
        content.append(image_block(paths.image_t2))
    content.append({"type": "text", "text": build_prompt(item)})
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
    }
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.post(
                f"{args.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=args.timeout,
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"])
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"API inference failed: {last_error}") from last_error


def image_block(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


if __name__ == "__main__":
    main()
