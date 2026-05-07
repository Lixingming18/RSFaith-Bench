from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from .answer import split_response
from .data import item_id, load_scene_graph


SPAN_IDENTIFICATION_PROMPT = """You are separating visual perception from analytical reasoning for schema-constrained claim extraction.

Task:
Given a model rationale, process one focused sentence at a time. The input contains the previous sentence, the focused sentence, and the next sentence. Segment only the text inside <FOCUS>...</FOCUS> into short meaningful spans and label each span as visual_perception or analytical.

Label definitions:
- visual_perception: a statement about what is directly visible in the image, or directly comparable between earlier and later images. This includes object presence or absence, count, attribute, location, spatial relation, subtype/category classification grounded in visible appearance, and temporal visual change.
- analytical: interpretation, explanation, causal reasoning, answer selection, uncertainty, meta reasoning, or content that is not itself a direct visual statement.

Instructions:
- Only label and extract spans from text inside <FOCUS>...</FOCUS>. Use the surrounding context only to resolve pronouns, ellipsis, and time anchors.
- If a pronoun or phrase such as "it", "they", "one", "the other", or "this area" has a clear referent, rewrite it in resolved_span. Otherwise keep the original text and label it as analytical.
- Split mixed content whenever possible, but keep each span schema-complete. Preserve time anchors, negation, and scope-limiting clauses with the visual fact they modify.
- If visual cues identify an object's visible subtype or category, extract the subtype/category statement as visual_perception even when introduced by words such as "indicates", "suggests", "appears to be", "characteristic of", or "therefore".
- Do not include answer-selection phrases such as "therefore the answer is B" as visual evidence.

Output format:
Return JSON only:
[{"span":"...", "resolved_span":"...", "label":"visual_perception|analytical"}]

Example:
Input:
  Context before: There is a ship next to the harbor.
  <FOCUS> It is located on the left side of the image, so option B is correct. </FOCUS>
  Context after:
Output:
[
  {"span":"It is located on the left side of the image",
   "resolved_span":"The ship is located on the left side of the image",
   "label":"visual_perception"},
  {"span":"so option B is correct",
   "resolved_span":"so option B is correct",
   "label":"analytical"}
]

Example:
Input:
  Context before: The deck is covered with pipes and manifolds.
  <FOCUS>This deck structure is characteristic of a liquid cargo ship, so the answer is A.</FOCUS>
  Context after:
Output:
[
  {"span":"This deck structure is characteristic of a liquid cargo ship",
   "resolved_span":"The ship is a liquid cargo ship",
   "label":"visual_perception"},
  {"span":"so the answer is A",
   "resolved_span":"so the answer is A",
   "label":"analytical"}
]"""


CLAIM_NORMALIZATION_PROMPT = """You are extracting atomic visual claims under a strict target schema.

Task:
Convert each visual span into normalized atomic visual claims of types: Existence, Counting, Attribute, Relation, Location.

Rules:
- Each claim contains exactly one directly visually verifiable fact. Split multiple facts into separate claims.
- Exclude reasoning, interpretation, uncertainty, and subjective language.
- Use only labels from the provided label space. Subject and object must be singular canonical nouns.
- Match each schema field to its own label-space list: subjects/objects from objects, relations from relations, locations from locations, attributes from attributes, and Attribute values from values.
- For Attribute claims with attribute="subtype", copy the value exactly from values; if the named subtype/category is not in values, omit that Attribute claim.
- Use t1 for the single or earlier image and t2 for the later image. Decompose paired-image statements into separate time-anchored claims.
- Do not invent entities, times, relations, attributes, or locations. Return [] if a statement cannot be normalized.
- Negative Existence applies only to image-global absence. Encode regional absence as a negative Location claim.
- Do not convert vague quantities such as "several", "many", or "multiple" into exact counts.
- Encode relative spatial phrases such as "to the right of X" as Relation claims, not Location.
- Encode subtype statements as Attribute claims with attribute="subtype".

Schema:
Return a JSON array only.
Existence:        {"type":"Existence","time":"t1|t2","subject":"...","value":true|false}
Counting (exact): {"type":"Counting","time":"t1|t2","subject":"...","quantity_kind":"exact","quantity":number}
Counting (cmp):   {"type":"Counting","time":"t1|t2","subject":"...","quantity_kind":"bound","quantity_operator":"gt|lt","object":"..."}
Attribute:        {"type":"Attribute","time":"t1|t2","subject":"...","attribute":"...","value":"..."}
Relation:         {"type":"Relation","time":"t1|t2","subject":"...","relation":"...","object":"...","value":true}
Pair transfer:    {"type":"Relation","time":"pair","subject":"...","relation":"transferred_to","object":"...","value":true}
Location:         {"type":"Location","time":"t1|t2","subject":"...","location":"...","value":true|false}

Example:
Input: The ship in the center-left is a dry cargo ship.
Output: [
  {"type":"Existence","time":"t1","subject":"ship","value":true},
  {"type":"Location","time":"t1","subject":"ship","location":"center-left","value":true},
  {"type":"Attribute","time":"t1","subject":"ship","attribute":"subtype","value":"dry_cargo_ship"}
]"""


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(frozen=True)
class ExtractionConfig:
    base_url: str
    model: str
    api_key: str
    timeout: int = 120
    max_tokens: int = 1024
    temperature: float = 0.0
    max_retries: int = 3


def extract_claims_openai(item: dict[str, Any], response: str, config: ExtractionConfig) -> dict[str, Any]:
    parsed = split_response(response)
    sentences = split_sentences(parsed.get("reasoning_trace", ""))
    visual_spans: list[dict[str, Any]] = []
    raw_span_outputs: list[dict[str, Any]] = []

    for index, focus in enumerate(sentences):
        prompt = build_span_prompt(
            previous_sentence=sentences[index - 1] if index > 0 else "",
            focused_sentence=focus,
            next_sentence=sentences[index + 1] if index + 1 < len(sentences) else "",
        )
        raw = _chat_completion(config, prompt)
        spans = parse_span_json(raw)
        raw_span_outputs.append({"sentence_index": index, "focus": focus, "raw": raw, "spans": spans})
        for span in spans:
            if span.get("label") != "visual_perception":
                continue
            resolved = str(span.get("resolved_span") or span.get("span") or "").strip()
            if resolved:
                visual_spans.append({"sentence_index": index, **span, "resolved_span": resolved})

    claims: list[dict[str, Any]] = []
    raw_claim_outputs: list[dict[str, Any]] = []
    for span in visual_spans:
        prompt = build_claim_prompt(item, str(span["resolved_span"]))
        raw = _chat_completion(config, prompt)
        span_claims = parse_claim_json(raw)
        raw_claim_outputs.append({"span": span["resolved_span"], "raw": raw, "claims": span_claims})
        claims.extend(span_claims)

    return {
        "question_id": item_id(item),
        "model_name": config.model,
        "parsed_response": parsed,
        "visual_spans": visual_spans,
        "raw_span_outputs": raw_span_outputs,
        "raw_claim_outputs": raw_claim_outputs,
        "claims": claims,
    }


def build_span_prompt(*, previous_sentence: str, focused_sentence: str, next_sentence: str) -> str:
    return (
        f"{SPAN_IDENTIFICATION_PROMPT}\n\n"
        "Input:\n"
        f"  Context before: {previous_sentence}\n"
        f"  <FOCUS>{focused_sentence}</FOCUS>\n"
        f"  Context after: {next_sentence}\n"
        "Output:"
    )


def build_claim_prompt(item: dict[str, Any], visual_span: str) -> str:
    label_space = json.dumps(build_label_space(item), ensure_ascii=False, indent=2)
    return (
        f"{CLAIM_NORMALIZATION_PROMPT}\n\n"
        "Provided label space:\n"
        f"{label_space}\n\n"
        f"Input: {visual_span}\n"
        "Output:"
    )


def split_sentences(text: str) -> list[str]:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if not value:
        return []
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(value) if part.strip()]


def build_label_space(item: dict[str, Any]) -> dict[str, list[str]]:
    objects: set[str] = set()
    relations: set[str] = {"transferred_to"}
    locations: set[str] = set()
    attributes: set[str] = {"subtype", "aggregate_area"}
    values: set[str] = set()

    for choice in item.get("choices", []):
        _add_label(values, choice)

    for claim in item.get("support", []):
        if not isinstance(claim, dict):
            continue
        _add_label(objects, claim.get("subject"))
        _add_label(objects, claim.get("object"))
        _add_label(relations, claim.get("predicate") or claim.get("relation"))
        _add_label(attributes, claim.get("name") or claim.get("attribute"))
        if claim.get("claim_type") == "Location":
            _add_label(locations, claim.get("value") or claim.get("location"))
        else:
            _add_label(values, claim.get("value"))

    scene_graph = _safe_scene_graph(item)
    for node in scene_graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        _add_label(objects, node.get("canonical_leaf"))
        _add_label(objects, node.get("canonical_root"))
        geometry = node.get("geometry", {}) if isinstance(node.get("geometry"), dict) else {}
        _add_label(locations, geometry.get("region_bin_3x3"))
        attributes_map = node.get("attributes", {}) if isinstance(node.get("attributes"), dict) else {}
        for name, value in attributes_map.items():
            _add_label(attributes, name)
            _add_label(values, value)

    for edge in scene_graph.get("edges", []):
        if isinstance(edge, dict):
            _add_label(relations, edge.get("relation_type"))

    inventory = scene_graph.get("global_inventory", {})
    semantic_inventory = inventory.get("semantic_inventory", {}) if isinstance(inventory, dict) else {}
    if isinstance(semantic_inventory, dict):
        _add_semantic_inventory_labels(semantic_inventory, objects, relations, attributes, values)

    return {
        "objects": sorted(objects),
        "relations": sorted(relations),
        "locations": sorted(locations),
        "attributes": sorted(attributes),
        "values": sorted(values),
    }


def _safe_scene_graph(item: dict[str, Any]) -> dict[str, Any]:
    try:
        return load_scene_graph(item)
    except Exception:  # noqa: BLE001 - extraction can still run from support labels.
        return {}


def _add_semantic_inventory_labels(
    inventory: dict[str, Any],
    objects: set[str],
    relations: set[str],
    attributes: set[str],
    values: set[str],
) -> None:
    semantic_areas = inventory.get("semantic_areas", {})
    if isinstance(semantic_areas, dict):
        _add_label(attributes, "aggregate_area")
        for area_map in semantic_areas.values():
            if not isinstance(area_map, dict):
                continue
            for label, entry in area_map.items():
                _add_label(objects, label)
                if isinstance(entry, dict):
                    for name, value in entry.items():
                        _add_label(attributes, name)
                        _add_label(values, value)

    transfers = inventory.get("transfers", {})
    if isinstance(transfers, dict):
        _add_label(relations, "transferred_to")
        for source, targets in transfers.items():
            _add_label(objects, source)
            if isinstance(targets, dict):
                for target in targets:
                    _add_label(objects, target)


def _add_label(labels: set[str], value: Any) -> None:
    if isinstance(value, bool) or value is None:
        return
    text = str(value).strip()
    if text:
        labels.add(text)


def _chat_completion(config: ExtractionConfig, prompt: str) -> str:
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    last_error: Exception | None = None
    for attempt in range(1, config.max_retries + 1):
        try:
            return _post_chat_completion(config, payload)
        except Exception as exc:  # noqa: BLE001 - keep retry path simple for API compatibility.
            last_error = exc
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"claim extraction failed: {last_error}") from last_error


def _post_chat_completion(config: ExtractionConfig, payload: dict[str, Any]) -> str:
    response = requests.post(
        f"{config.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=config.timeout,
    )
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"])


def parse_span_json(text: str) -> list[dict[str, Any]]:
    payload = parse_json_payload(text)
    spans = payload.get("spans", payload.get("segments", [])) if isinstance(payload, dict) else payload
    if not isinstance(spans, list):
        return []
    return [normalize_span(span) for span in spans if isinstance(span, dict)]


def parse_claim_json(text: str) -> list[dict[str, Any]]:
    payload = parse_json_payload(text)
    claims = payload.get("claims", []) if isinstance(payload, dict) else payload
    if not isinstance(claims, list):
        return []
    return [normalize_claim(claim) for claim in claims if isinstance(claim, dict)]


def parse_json_payload(text: str) -> Any:
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for pattern in (r"\[.*\]", r"\{.*\}"):
        match = re.search(pattern, cleaned, flags=re.S)
        if not match:
            continue
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return []


def strip_code_fence(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def normalize_span(span: dict[str, Any]) -> dict[str, str]:
    raw_span = str(span.get("span") or "").strip()
    resolved = str(span.get("resolved_span") or raw_span).strip()
    label = str(span.get("label") or "").strip().lower().replace("-", "_").replace(" ", "_")
    if label not in {"visual_perception", "analytical"}:
        label = "analytical"
    return {"span": raw_span, "resolved_span": resolved, "label": label}


def normalize_claim(claim: dict[str, Any]) -> dict[str, Any]:
    out = dict(claim)
    if "type" in out and "claim_type" not in out:
        out["claim_type"] = out.pop("type")
    if "time" not in out:
        out["time"] = "t1"

    claim_type = str(out.get("claim_type") or "")
    if claim_type == "Attribute" and "attribute" in out and "name" not in out:
        out["name"] = out.pop("attribute")
    if claim_type == "Relation" and "relation" in out and "predicate" not in out:
        out["predicate"] = out.pop("relation")
    if claim_type == "Location" and "location" in out:
        location = out.pop("location")
        polarity = out.get("value", True)
        out["value"] = location
        out["polarity"] = _bool_value(polarity)
    if claim_type == "Counting" and "quantity" not in out and "value" in out:
        out["quantity"] = out["value"]
    return out


def _bool_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return value
