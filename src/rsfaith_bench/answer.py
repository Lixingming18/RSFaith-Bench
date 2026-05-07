from __future__ import annotations

import re
from typing import Any

from .prompts import labeled_choices
from .utils import compact_text


FINAL_ANSWER_RE = re.compile(
    r"(?im)^\s*(?:final\s*answer|answer)\s*[:：]?\s*(?P<answer>.+?)\s*$"
)
CHOICE_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")
EVIDENCE_PREFIX_RE = re.compile(r"(?is)^\s*evidence\s*[:：]\s*")


def split_response(text: str) -> dict[str, str]:
    raw = str(text or "").strip()
    matches = list(FINAL_ANSWER_RE.finditer(raw))
    if matches:
        match = matches[-1]
        answer = match.group("answer").strip()
        trace = raw[: match.start()].strip()
        return {"reasoning_trace": _strip_evidence_prefix(trace), "final_answer": answer, "raw_response": raw}

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) == 1:
        return {"reasoning_trace": "", "final_answer": lines[0], "raw_response": raw}
    return {"reasoning_trace": raw, "final_answer": "", "raw_response": raw}


def _strip_evidence_prefix(text: str) -> str:
    return EVIDENCE_PREFIX_RE.sub("", str(text or "").strip()).strip()


def prediction_text(prediction: dict[str, Any]) -> str:
    for key in ("response", "raw_response", "prediction", "output", "text", "answer"):
        value = prediction.get(key)
        if value is not None:
            return str(value)
    return ""


def normalize_answer(item: dict[str, Any], answer: Any) -> str:
    text = compact_text(answer)
    if not text:
        return ""
    choices = labeled_choices(item)
    upper = text.strip().upper()
    if len(upper) == 1 and any(label == upper for label, _choice in choices):
        return choice_text_for_label(item, upper)
    if len(upper) >= 2 and upper[0] in {label for label, _choice in choices} and upper[1] in {".", ")", ":", " "}:
        return choice_text_for_label(item, upper[0])
    for label, choice in choices:
        if normalize_text_key(text) == normalize_text_key(choice):
            return choice
        if normalize_text_key(text).startswith(normalize_text_key(f"{label}. {choice}")):
            return choice
    match = CHOICE_RE.search(upper)
    if match and any(label == match.group(1) for label, _choice in choices):
        return choice_text_for_label(item, match.group(1))
    return text


def choice_text_for_label(item: dict[str, Any], label: str) -> str:
    for current, choice in labeled_choices(item):
        if current == label:
            return choice
    return ""


def gold_answer(item: dict[str, Any]) -> str:
    return normalize_answer(item, item.get("answer"))


def answer_correct(item: dict[str, Any], prediction: dict[str, Any]) -> bool:
    parsed = split_response(prediction_text(prediction))
    final = prediction.get("final_answer") or parsed.get("final_answer")
    return normalize_text_key(normalize_answer(item, final)) == normalize_text_key(gold_answer(item))


def normalize_text_key(value: Any) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(value or "")).split())
