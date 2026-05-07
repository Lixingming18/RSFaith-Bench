# Evaluation

## Environment

```bash
conda create -n rsfaith python=3.10 -y
conda activate rsfaith
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
```

## Validate Data

```bash
python src/scripts/validate_data.py --data RSFaith-Bench_subset --expect-per-category 50
```

## Prediction Format

Predictions should be JSON or JSONL. Each row must contain `question_id` and a text response:

```json
{"question_id": "RSF-Q00000001", "model_name": "my-model", "response": "Evidence: ...\nAnswer: C"}
```

## Evaluation

Evaluation uses extracted visual claims. The included extractor first segments each response sentence into `visual_perception` or `analytical` spans, then normalizes visual spans into atomic schema claims with an OpenAI-compatible text model.
Only mapped schema claims enter verification. The verifier outputs `support`, `contradict`, or `uncertain`; normalization or label-space failures are logged separately as `unmapped_claims`.
The evaluator derives closed scopes from `program.slots`. Direct positive fact matches can be supported across scopes. Absence, missing-relation evidence, and exact-count verification are used only when the corresponding slot has `complete_scope: true`.

```bash
python src/scripts/extract_claims.py \
  --data RSFaith-Bench_subset \
  --responses predictions.jsonl \
  --model "$OPENAI_MODEL" \
  --output claims.jsonl

python src/scripts/evaluate.py \
  --data RSFaith-Bench_subset \
  --pred predictions.jsonl \
  --claims claims.jsonl \
  --output eval_faithfulness.jsonl
```

## Summary

```bash
python src/scripts/summarize.py --eval eval_faithfulness.jsonl --group overall --output summary_overall.csv
python src/scripts/summarize.py --eval eval_faithfulness.jsonl --group level --output summary_level.csv
python src/scripts/summarize.py --eval eval_faithfulness.jsonl --group subcategory --output summary_subcategory.csv
```
