# Prediction And Evaluation Formats

## Model Predictions

Predictions may be JSON or JSONL. Each row should contain a question identifier
and a response string:

```json
{"question_id": "RSF-Q00000001", "model_name": "my-model", "response": "Evidence: ...\nAnswer: C"}
```

The evaluator also accepts `raw_response`, `prediction`, `output`, `text`, or
`answer` as response text fields. If a final answer line is present, the parser
uses the last line beginning with `Answer:` or `Final answer:`.

## Extracted Claims

Claim extraction output is JSONL keyed by `question_id`:

```json
{
  "question_id": "RSF-Q00000001",
  "model_name": "my-model",
  "claims": [
    {"claim_type": "Existence", "time": "t1", "subject": "airplane", "value": true}
  ]
}
```

Rows may use either `claims` or `normalized_claims`. Claims that cannot be mapped
to the released schema or label space are reported separately by the evaluator
as `unmapped_claims`.

## Evaluation Output

`src/scripts/evaluate.py` writes one JSONL row per evaluated question. Important
fields include:

- `question_id`, `scene_id`, `level`, `subcategory`, `model_name`
- `gold_answer`, `predicted_answer`, `answer_correct`
- `claims`: input claims used for verification.
- `verification`: mapped claims with `support`, `contradict`, or `uncertain`
  labels.
- `unmapped_claims`: extraction or normalization failures outside the verifier
  label distribution.
- `metrics`: item-level AA, CP, FA, C-CUR, M-CUR, and claim counts.
