# RSFaith-Bench

<p align="center">
  <strong>Evaluating Answer-Evidence Faithfulness in Remote Sensing MLLMs</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> |
  <a href="#released-subset">Released Subset</a> |
  <a href="#evaluation-workflow">Evaluation</a> |
  <a href="#documentation">Documentation</a> |
  <a href="#metrics">Metrics</a> |
  <a href="#repository-layout">Repository Layout</a>
</p>

<p align="center">
  <img src="assets/figures/teaser.png" alt="RSFaith-Bench teaser" width="88%">
</p>

RSFaith-Bench evaluates remote-sensing VQA models at two levels: whether the
final answer is correct, and whether the model's stated visual evidence
faithfully supports that answer. Each example pairs a question with answer
choices, image path(s), a compact scene graph, gold evidence claims, and an
executable question program used by the evaluator.

## At a Glance

| Item | Description |
|---|---|
| Benchmark scale | 13,511 questions, 16,288 images, 12 task categories |
| Released subset | 600 examples, 50 per subcategory |
| Evaluation target | Final-answer accuracy and answer-evidence faithfulness |
| Core metrics | AA, CP, FA, C-CUR, M-CUR |
| Included | Evaluation code, schema docs, smoke test, released subset, paper figure previews |
| Not included | Full benchmark data, model outputs, claim extraction logs, figure source data |
| License scope | Code is MIT-licensed; released data and assets follow `DATA_LICENSE.md` |

## Quick Start

Run the no-network smoke test:

```bash
bash scripts/smoke_test.sh
```

The smoke test validates the released subset, creates a runtime fixture from
released support claims, runs evaluation, writes summaries under
`outputs/smoke/`, and executes the lightweight unit tests.

## Released Subset

Validate the 600-example released subset:

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src

python src/scripts/validate_data.py \
  --data RSFaith-Bench_subset \
  --expect-per-category 50
```

Prediction files use JSONL records with one model response per question:

```json
{"question_id": "RSF-Q00000001", "model_name": "my-model", "response": "Evidence: ...\nAnswer: C"}
```

## Evaluation Workflow

Extract visual claims from model reasoning with an OpenAI-compatible text model:

```bash
python src/scripts/extract_claims.py \
  --data RSFaith-Bench_subset \
  --responses predictions.jsonl \
  --model "$OPENAI_MODEL" \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --output outputs/claims.jsonl
```

Run answer and faithfulness evaluation:

```bash
python src/scripts/evaluate.py \
  --data RSFaith-Bench_subset \
  --pred predictions.jsonl \
  --claims outputs/claims.jsonl \
  --output outputs/eval_faithfulness.jsonl
```

Summarize item-level results:

```bash
python src/scripts/summarize.py --eval outputs/eval_faithfulness.jsonl --group overall --output outputs/summary_overall.csv
python src/scripts/summarize.py --eval outputs/eval_faithfulness.jsonl --group level --output outputs/summary_level.csv
python src/scripts/summarize.py --eval outputs/eval_faithfulness.jsonl --group subcategory --output outputs/summary_subcategory.csv
```

## Documentation

| Document | Contents |
|---|---|
| `docs/dataset_schema.md` | Released subset fields, claim records, and question programs |
| `docs/prediction_format.md` | Prediction, extracted-claim, and evaluation-output formats |
| `docs/evaluation_protocol.md` | Operational definitions for verification and metrics |
| `docs/reproduction.md` | End-to-end reproduction commands |
| `DATA_LICENSE.md` | License scope for code, released subset data, and paper assets |

## OpenAI-Compatible Inference

The inference script uses the evidence-first response prompt described in the
paper and in `docs/reproduction.md`.

```bash
python src/scripts/infer_api.py \
  --data RSFaith-Bench_subset \
  --model "$OPENAI_MODEL" \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --output outputs/responses.jsonl
```

## Metrics

| Metric | Meaning |
|---|---|
| AA | Final-answer accuracy |
| CP | Claim precision over questions with at least one supported or contradicted claim |
| FA | Correct answer with supported answer-critical evidence and no contradicted claim |
| C-CUR | Correct answer with at least one contradicted visual claim |
| M-CUR | Correct answer with missing required visual evidence and no contradicted claim |

For faithfulness evaluation, each correct answer falls into exactly one of
`FA`, `C-CUR`, or `M-CUR`.

## Paper Figures

<details>
<summary>Show selected paper figures</summary>

| Benchmark Overview | Evidence Decomposition |
|---|---|
| <img src="assets/figures/overview.png" alt="Benchmark overview" width="420"> | <img src="assets/figures/evidence_decomposition.png" alt="Answer-evidence decomposition" width="420"> |

| Model Diagnostics | Subtype Diagnostics |
|---|---|
| <img src="assets/figures/model_evidence_diagnostics.png" alt="Model evidence diagnostics" width="420"> | <img src="assets/figures/subtype_diagnostic.png" alt="Subtype diagnostics" width="420"> |

| Visual Control Analysis |
|---|
| <img src="assets/figures/visual_control.png" alt="Visual control analysis" width="520"> |

</details>

## Repository Layout

```text
RSFaith-Bench/
|-- RSFaith-Bench_subset/     # 600-example released subset
|-- DATA_LICENSE.md           # data and asset license notes
|-- assets/figures/           # selected paper figure previews
|-- configs/                  # public taxonomy and label ontology
|-- docs/                     # schema, prediction format, evaluation protocol
|-- scripts/smoke_test.sh     # no-network smoke test
|-- src/rsfaith_bench/        # core evaluator and metric implementation
|-- src/scripts/              # command-line entry points
`-- tests/                    # lightweight runtime tests
```

## Citation

Please cite the RSFaith-Bench paper when using this benchmark:

```bibtex
@inproceedings{anonymous2026rsfaithbench,
  title = {RSFaith-Bench: Evaluating Answer-Evidence Faithfulness in Remote Sensing MLLMs},
  author = {Anonymous RSFaith-Bench Authors},
  booktitle = {Under review},
  year = {2026}
}
```
