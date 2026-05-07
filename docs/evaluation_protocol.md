# Evaluation Protocol

RSFaith-Bench evaluates both answer accuracy and whether the model's stated
visual evidence supports the answer.

## Pipeline

1. Parse the model response into an evidence trace and final answer.
2. Extract normalized visual claims from the response.
3. Verify each mappable claim against the released scene graph and question
   program.
4. Fill the question program with supported claims.
5. Execute the answer function over complete slot assignments.
6. Report item-level and aggregate metrics.

## Claim Labels

Mappable claims receive one of three labels:

- `support`: the claim is verified by the scene graph or derived fact layer.
- `contradict`: the claim conflicts with a closed verification scope.
- `uncertain`: the released fact layer does not decide the claim.

Claims that cannot be stably normalized into the schema are logged as
`unmapped_claims` and do not enter claim precision.

## Metrics

- `AA`: answer accuracy.
- `CP`: claim precision over questions with at least one supported or
  contradicted claim.
- `FA`: correct answer with supported answer-critical evidence and no
  contradicted claim.
- `C-CUR`: correct answer with at least one contradicted visual claim.
- `M-CUR`: correct answer with missing required visual evidence and no
  contradicted claim.

For faithfulness evaluation, each correct answer falls into exactly one of
`FA`, `C-CUR`, or `M-CUR`.

## Closed Scopes

Closed scopes are attached to evidence slots whose corresponding fact inventory
is complete. Direct positive matches can support claims across the available
fact layer. Absence, missing-relation evidence, and exact-count disagreement are
used as contradiction evidence only within complete scopes.

## Answer Support

Supported claims fill typed slots when claim type, scope, predicate/value, and
denotation are compatible with the slot. The evaluator enumerates complete slot
assignments and executes the question's `answer_fn`. A response is
answer-supported when at least one complete assignment derives the gold answer.
