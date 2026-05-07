#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

DATA_DIR="${DATA_DIR:-RSFaith-Bench_subset}"
OUT_DIR="${OUT_DIR:-outputs/smoke}"
PRED_PATH="${OUT_DIR}/predictions.jsonl"
CLAIMS_PATH="${OUT_DIR}/claims.jsonl"
EVAL_PATH="${OUT_DIR}/eval_faithfulness.jsonl"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

python src/scripts/validate_data.py \
  --data "${DATA_DIR}" \
  --expect-per-category 50

python - <<'PY'
import os
from pathlib import Path

from rsfaith_bench.answer import gold_answer
from rsfaith_bench.data import item_id, load_items
from rsfaith_bench.utils import dump_jsonl

data_dir = os.environ.get("DATA_DIR", "RSFaith-Bench_subset")
out_dir = Path(os.environ.get("OUT_DIR", "outputs/smoke"))
items = load_items(data_dir)

selected = items

predictions = []
claims = []
for item in selected:
    qid = item_id(item)
    answer = gold_answer(item)
    predictions.append(
        {
            "question_id": qid,
            "model_name": "smoke-gold-support",
            "response": f"Evidence: Smoke-test response generated from released support claims.\nAnswer: {answer}",
        }
    )
    claims.append(
        {
            "question_id": qid,
            "model_name": "smoke-gold-support",
            "claims": item.get("support", []),
        }
    )

dump_jsonl(predictions, out_dir / "predictions.jsonl")
dump_jsonl(claims, out_dir / "claims.jsonl")
print(f"wrote {len(selected)} smoke predictions and claim rows")
PY

python src/scripts/evaluate.py \
  --data "${DATA_DIR}" \
  --pred "${PRED_PATH}" \
  --claims "${CLAIMS_PATH}" \
  --output "${EVAL_PATH}"

python src/scripts/summarize.py --eval "${EVAL_PATH}" --group overall --output "${OUT_DIR}/summary_overall.csv"
python src/scripts/summarize.py --eval "${EVAL_PATH}" --group level --output "${OUT_DIR}/summary_level.csv"
python src/scripts/summarize.py --eval "${EVAL_PATH}" --group subcategory --output "${OUT_DIR}/summary_subcategory.csv"

python -m unittest discover -s tests

echo "Smoke test completed: ${OUT_DIR}"
