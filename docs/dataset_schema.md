# Dataset Schema

RSFaith-Bench examples are stored as JSON records grouped by reasoning level and
subcategory. The released subset follows this layout:

```text
RSFaith-Bench_subset/
  perception/
    object_presence/
    object_counting/
    object_localization/
    fine_grained_recognition/
  relational_reasoning/
    directional/
    topological/
    proximity/
    projective_ordering/
    aggregate_distribution/
  temporal_reasoning/
    category_turnover/
    semantic_transition/
    net_change/
```

Each subcategory directory contains one JSON file with a list of examples,
images, and scene graphs.

## Item Fields

- `question_id`: stable example identifier.
- `scene_id`: scene identifier.
- `level`: reasoning level.
- `subcategory`: question subtype.
- `question`: natural-language VQA prompt.
- `choices`: answer options.
- `answer`: gold answer text or option.
- `answer_type`: answer format metadata.
- `images`: relative image paths. `t1` is always present; `t2` is present for
  temporal questions.
- `scene_graph`: relative path to the compact fact layer.
- `support`: gold evidence claims for the answer.
- `program`: executable evidence contract used by the evaluator.

## Claim Records

Claims use a typed schema. Common fields are:

- `claim_type`: one of `Existence`, `Counting`, `Attribute`, `Location`, or
  `Relation`.
- `time`: temporal scope, usually `t1`, `t2`, or `pair`.
- `subject`: object or scoped object name.
- `value`: claim value when applicable.
- `quantity`: count value for `Counting` claims.
- `name`: attribute name for `Attribute` claims.
- `predicate` / `relation`: relation predicate for `Relation` claims.
- `object`: relation object for `Relation` claims.
- `refs`: node and edge references into the scene graph when available.

## Program Fields

`program.slots` specifies the evidence contract for a question. A slot contains:

- `slot_id`: local slot identifier.
- `claim_type`: claim type expected for the slot.
- `scope`: time or pair scope.
- `constraints`: claim fields that must match.
- `outputs`: fields that are bound into an answer-function assignment.
- `complete_scope`: whether the corresponding fact inventory is closed.
- `refs`: expected scene-graph denotation.
- `sr_required`: whether the slot is required for answer support.

`program.answer_fn` defines the deterministic answer function evaluated over a
complete slot assignment.
