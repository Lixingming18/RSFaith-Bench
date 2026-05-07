# Reproduction

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
```

## Smoke Test

Run the no-network smoke test:

```bash
bash scripts/smoke_test.sh
```

The smoke test validates the released subset, creates a small synthetic
prediction/claim fixture from released support claims, runs evaluation, writes
summary CSV files, and runs the lightweight unit tests.

## Data Validation

```bash
python src/scripts/validate_data.py \
  --data RSFaith-Bench_subset \
  --expect-per-category 50
```

## Inference

```bash
python src/scripts/infer_api.py \
  --data RSFaith-Bench_subset \
  --model "$OPENAI_MODEL" \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --output outputs/responses.jsonl
```

## Claim Extraction

```bash
python src/scripts/extract_claims.py \
  --data RSFaith-Bench_subset \
  --responses outputs/responses.jsonl \
  --model "$OPENAI_MODEL" \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --output outputs/claims.jsonl
```

## Evaluation

```bash
python src/scripts/evaluate.py \
  --data RSFaith-Bench_subset \
  --pred outputs/responses.jsonl \
  --claims outputs/claims.jsonl \
  --output outputs/eval_faithfulness.jsonl
```

## Summaries

```bash
python src/scripts/summarize.py --eval outputs/eval_faithfulness.jsonl --group overall --output outputs/summary_overall.csv
python src/scripts/summarize.py --eval outputs/eval_faithfulness.jsonl --group level --output outputs/summary_level.csv
python src/scripts/summarize.py --eval outputs/eval_faithfulness.jsonl --group subcategory --output outputs/summary_subcategory.csv
```
