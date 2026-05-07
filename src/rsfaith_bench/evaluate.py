from __future__ import annotations

from typing import Any

from .answer import answer_correct, gold_answer, normalize_answer, prediction_text, split_response
from .data import item_id, load_scene_graph
from .metrics import compute_metrics
from .verifier import (
    ClosedScopeIndex,
    SceneGraphIndex,
    build_item_label_space,
    contract_supports_gold_answer,
    verify_claims_with_mapping,
)


def evaluate_item(
    item: dict[str, Any],
    prediction: dict[str, Any],
    *,
    claim_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response_text = prediction_text(prediction)
    parsed = split_response(response_text)
    correct = answer_correct(item, {**prediction, "response": response_text})
    claims = _claims_from_records(prediction, claim_record)

    scene_graph = load_scene_graph(item)
    index = SceneGraphIndex(scene_graph, label_space=build_item_label_space(item))
    closed_scopes = ClosedScopeIndex.from_item(item)
    verification, unmapped = verify_claims_with_mapping(index, claims, closed_scopes=closed_scopes)
    support_rel = contract_supports_gold_answer(item, verification, index)
    verification_rows = [row.to_dict() for row in verification]
    unmapped_rows = [row.to_dict() for row in unmapped]
    metrics = compute_metrics(
        answer_correct=correct,
        num_supported=sum(row.label == "support" for row in verification),
        num_contradicted=sum(row.label == "contradict" for row in verification),
        num_uncertain=sum(row.label == "uncertain" for row in verification),
        num_unmapped=len(unmapped),
        support_relevant=support_rel,
    )

    return {
        "question_id": item_id(item),
        "scene_id": item.get("scene_id"),
        "level": item.get("level"),
        "subcategory": item.get("subcategory"),
        "model_name": prediction.get("model_name") or (claim_record or {}).get("model_name") or "",
        "question": item.get("question"),
        "gold_answer": gold_answer(item),
        "predicted_answer": normalize_answer(item, prediction.get("final_answer") or parsed.get("final_answer")),
        "answer_correct": correct,
        "metrics": metrics,
        "parsed_response": parsed,
        "claims": claims,
        "verification": verification_rows,
        "unmapped_claims": unmapped_rows,
    }


def _claims_from_records(prediction: dict[str, Any], claim_record: dict[str, Any] | None) -> list[dict[str, Any]]:
    for source in (claim_record or {}, prediction):
        for key in ("claims", "normalized_claims"):
            claims = source.get(key)
            if isinstance(claims, list):
                return [claim for claim in claims if isinstance(claim, dict)]
    return []
