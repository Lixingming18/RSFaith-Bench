from __future__ import annotations

from typing import Any


RESPONSE_INSTRUCTION = (
    'Look at the image carefully. First describe the visual evidence you observe in the image that is relevant '
    'to this question. Based on this evidence, then state your answer. Begin with "Evidence:" and end with '
    '"Answer:".'
)


def labeled_choices(item: dict[str, Any]) -> list[tuple[str, str]]:
    choices = item.get("choices") if isinstance(item.get("choices"), list) else []
    return [(chr(ord("A") + index), str(choice)) for index, choice in enumerate(choices)]


def build_question_text(item: dict[str, Any]) -> str:
    lines = [str(item["question"]).strip()]
    for label, choice in labeled_choices(item):
        lines.append(f"{label}. {choice}")
    return "\n".join(lines)


def build_prompt(item: dict[str, Any]) -> str:
    question = build_question_text(item)
    return f"Question: {question}\nImage: [image]\n{RESPONSE_INSTRUCTION}"
