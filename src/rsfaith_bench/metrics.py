from __future__ import annotations


def compute_metrics(
    *,
    answer_correct: bool,
    num_supported: int = 0,
    num_contradicted: int = 0,
    num_uncertain: int = 0,
    num_unmapped: int = 0,
    support_relevant: bool = False,
) -> dict[str, float | bool | int | None]:
    num_verified = num_supported + num_contradicted + num_uncertain
    num_total = num_verified + num_unmapped
    num_scorable = num_supported + num_contradicted
    faithful_answer = bool(answer_correct and support_relevant and num_contradicted == 0)
    contradiction_cur = bool(answer_correct and num_contradicted > 0)
    missing_cur = bool(answer_correct and not faithful_answer and not contradiction_cur)
    return {
        "aa": bool(answer_correct),
        "cp": float(num_supported) / float(num_scorable) if num_scorable > 0 else None,
        "fa": faithful_answer,
        "C-CUR": contradiction_cur,
        "M-CUR": missing_cur,
        "num_claims": num_total,
        "num_verified_claims": num_verified,
        "num_scorable": num_scorable,
        "num_supported": num_supported,
        "num_contradicted": num_contradicted,
        "num_uncertain": num_uncertain,
        "num_unmapped_claims": num_unmapped,
    }
