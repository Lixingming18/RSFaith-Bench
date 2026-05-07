from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


CLAIM_FIELDS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "Existence": ("subject", "value"),
    "Counting": ("subject", "quantity"),
    "Attribute": ("subject", "name", "value"),
    "Location": ("subject", "value", "polarity"),
    "Relation": ("subject", "predicate", "object", "value"),
}

GRID_REGIONS: tuple[str, ...] = (
    "top-left",
    "top-center",
    "top-right",
    "center-left",
    "center",
    "center-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
)
REGION_COORDS: dict[str, tuple[int, int]] = {region: divmod(index, 3) for index, region in enumerate(GRID_REGIONS)}
BROAD_REGION_GROUPS: dict[str, tuple[str, ...]] = {
    "broad:left_side": ("top-left", "center-left", "bottom-left"),
    "broad:right_side": ("top-right", "center-right", "bottom-right"),
    "broad:upper_part": ("top-left", "top-center", "top-right"),
    "broad:lower_part": ("bottom-left", "bottom-center", "bottom-right"),
    "broad:central_area": ("top-center", "center-left", "center", "center-right", "bottom-center"),
}


def build_program_spec(
    *,
    template_id: str,
    template_variant: str,
    level: str,
    family: str,
    render_mode: str,
    answer_type: str,
    bindings: dict[str, Any],
    support: list[dict[str, Any]],
    answer: Any,
    choices: list[str] | None,
) -> dict[str, Any]:
    slots = [_slot_from_support(index, row) for index, row in enumerate(support, start=1)]
    answer_fn = _infer_answer_fn(
        template_variant=template_variant,
        bindings=bindings,
        answer_type=answer_type,
    )
    _attach_outputs(slots, answer_fn)
    _attach_sr_roles(slots)
    program = {
        "program_id": template_id,
        "template_variant": template_variant,
        "level": level,
        "family": family,
        "render_mode": render_mode,
        "answer_type": answer_type,
        "answer_space": list(choices or []),
        "slots": slots,
        "answer_fn": answer_fn,
        "gold_answer": answer,
    }
    closure = _build_closure_spec(template_variant=template_variant, bindings=bindings, slots=slots)
    if closure:
        program["closure"] = closure
    return program


def execute_answer_fn(program: dict[str, Any], assignment: dict[str, Any]) -> Any:
    answer_fn = program.get("answer_fn", {})
    if not isinstance(answer_fn, dict):
        return None
    op = str(answer_fn.get("op") or "")

    if op == "slot_value":
        return assignment.get(str(answer_fn.get("var") or ""))

    if op == "slot_value_text":
        value = assignment.get(str(answer_fn.get("var") or ""))
        return None if value is None else str(value)

    if op == "bool_label":
        value = _to_bool(assignment.get(str(answer_fn.get("var") or "")))
        return _yes_no(value)

    if op == "equals":
        return _same_scalar(assignment.get(str(answer_fn.get("var") or "")), answer_fn.get("target"))

    if op == "equals_label":
        value = _same_scalar(assignment.get(str(answer_fn.get("var") or "")), answer_fn.get("target"))
        return _yes_no(value)

    if op == "relation_field":
        value = assignment.get(str(answer_fn.get("var") or ""))
        node_to_value = answer_fn.get("node_to_value")
        if isinstance(node_to_value, dict) and str(value) in node_to_value:
            return node_to_value[str(value)]
        return value

    if op == "aggregate_pattern":
        answer_key = _aggregate_pattern_key(answer_fn, assignment)
        return _choice_value(answer_fn, answer_key)

    if op == "temporal_turnover":
        mode = str(answer_fn.get("mode") or "")
        rows: list[str] = []
        for category in answer_fn.get("candidates", []):
            category = str(category)
            t1 = _to_bool(assignment.get(f"exists:{category}:t1"))
            t2 = _to_bool(assignment.get(f"exists:{category}:t2"))
            if t1 is None or t2 is None:
                continue
            if mode == "appearance" and t1 is False and t2 is True:
                rows.append(category)
            if mode == "disappearance" and t1 is True and t2 is False:
                rows.append(category)
        return rows[0] if len(rows) == 1 else None

    if op == "temporal_delta_argmax":
        direction = str(answer_fn.get("direction") or "")
        rows: list[tuple[str, float]] = []
        for category in answer_fn.get("candidates", []):
            category = str(category)
            t1 = _temporal_area_value(assignment, category, "t1")
            t2 = _temporal_area_value(assignment, category, "t2")
            if t1 is None or t2 is None:
                return None
            score = t2 - t1 if direction == "increase" else t1 - t2
            rows.append((category, score))
        return _argmax_unique(rows)

    if op == "projective_relation":
        ordered = _projective_ordered_values(answer_fn, assignment)
        if ordered is None:
            return None
        task = str(answer_fn.get("task") or "")
        if task == "sequence_order":
            return ", ".join(value for _node_id, value in ordered)
        if task == "ordinal_position":
            try:
                rank_index = int(answer_fn.get("rank_index"))
            except (TypeError, ValueError):
                return None
            if rank_index < 0 or rank_index >= len(ordered):
                return None
            return ordered[rank_index][1]
        if task == "adjacent_order":
            anchor = str(answer_fn.get("anchor_node") or "")
            try:
                step = int(answer_fn.get("adjacent_step"))
            except (TypeError, ValueError):
                return None
            indices = {node_id: index for index, (node_id, _value) in enumerate(ordered)}
            if anchor not in indices:
                return None
            answer_index = indices[anchor] + step
            if answer_index < 0 or answer_index >= len(ordered):
                return None
            return ordered[answer_index][1]
        if task == "between_order":
            anchors = [str(node_id) for node_id in answer_fn.get("anchor_node_ids", [])]
            if len(anchors) != 2:
                return None
            indices = {node_id: index for index, (node_id, _value) in enumerate(ordered)}
            if anchors[0] not in indices or anchors[1] not in indices:
                return None
            left, right = sorted((indices[anchors[0]], indices[anchors[1]]))
            between = ordered[left + 1 : right]
            return between[0][1] if len(between) == 1 else None

    if op == "proximity":
        anchor = str(answer_fn.get("anchor") or "")
        anchor_center = assignment.get(f"center:{anchor}")
        if not _is_point(anchor_center):
            return None
        reverse = str(answer_fn.get("direction") or "") == "farthest"
        rows: list[tuple[str, float]] = []
        for node_id, choice in zip(answer_fn.get("choice_node_ids", []), answer_fn.get("choice_values", []), strict=False):
            center = assignment.get(f"center:{node_id}")
            if not _is_point(center):
                return None
            distance = math.hypot(float(center[0]) - float(anchor_center[0]), float(center[1]) - float(anchor_center[1]))
            rows.append((str(choice), distance if reverse else -distance))
        return _argmax_unique(rows)

    return None


def _projective_ordered_values(answer_fn: dict[str, Any], assignment: dict[str, Any]) -> list[tuple[str, str]] | None:
    axis_index = 0 if str(answer_fn.get("axis") or "") == "x" else 1
    reverse = bool(answer_fn.get("reverse", False))
    rows: list[tuple[float, str, str]] = []
    for node_id, value in zip(answer_fn.get("ordering_node_ids", []), answer_fn.get("ordering_values", []), strict=False):
        node_id = str(node_id)
        center = assignment.get(f"center:{node_id}")
        if not _is_point(center):
            return None
        rows.append((float(center[axis_index]), node_id, str(value)))
    rows.sort(key=lambda row: (row[0], row[2]), reverse=reverse)
    return [(node_id, value) for _coordinate, node_id, value in rows]


def _aggregate_pattern_key(answer_fn: dict[str, Any], assignment: dict[str, Any]) -> str | None:
    choice_keys = [str(key) for key in answer_fn.get("choice_keys", [])]
    counts = _aggregate_grid_counts(answer_fn, assignment)
    if counts is None:
        return None

    if any(key.startswith("specific:") for key in choice_keys):
        return _argmax_unique(
            [
                (key, counts.get(key.split(":", 1)[1], 0.0))
                for key in choice_keys
                if key.startswith("specific:")
            ]
        )
    if any(key.startswith("broad:") for key in choice_keys):
        return _argmax_unique(
            [
                (key, sum(counts[region] for region in BROAD_REGION_GROUPS[key]))
                for key in choice_keys
                if key in BROAD_REGION_GROUPS
            ]
        )
    return _argmax_unique(
        [
            (key, score)
            for key in choice_keys
            if (score := _aggregate_pattern_score(key, counts)) is not None
        ]
    )


def _aggregate_grid_counts(answer_fn: dict[str, Any], assignment: dict[str, Any]) -> dict[str, float] | None:
    category = str(answer_fn.get("category") or "")
    if not category:
        return None
    counts: dict[str, float] = {}
    for region in GRID_REGIONS:
        value = _to_float(assignment.get(f"count:{category}::{region}"))
        if value is None:
            return None
        counts[region] = value
    return counts


def _choice_value(answer_fn: dict[str, Any], answer_key: str | None) -> Any:
    if answer_key is None:
        return None
    for key, value in zip(answer_fn.get("choice_keys", []), answer_fn.get("choice_values", []), strict=False):
        if str(key) == answer_key:
            return value
    return None


def _aggregate_pattern_score(key: str, counts: dict[str, float]) -> float | None:
    total = sum(counts.values())
    if total <= 0:
        return None
    row_sums = [sum(counts[GRID_REGIONS[row * 3 + col]] for col in range(3)) for row in range(3)]
    col_sums = [sum(counts[GRID_REGIONS[row * 3 + col]] for row in range(3)) for col in range(3)]
    active = sum(1 for value in counts.values() if value > 0)
    height, width = _active_bbox(counts)

    if key == "pattern:horizontal_band":
        return max(row_sums) / total + 0.25 * (width / 3) - 0.15 * (height / 3)
    if key == "pattern:vertical_band":
        return max(col_sums) / total + 0.25 * (height / 3) - 0.15 * (width / 3)
    if key == "pattern:widely_spread":
        row_coverage = sum(1 for value in row_sums if value > 0) / 3
        col_coverage = sum(1 for value in col_sums if value > 0) / 3
        max_share = max(counts.values()) / total
        return (
            0.35 * (active / 9)
            + 0.25 * row_coverage
            + 0.25 * col_coverage
            + 0.15 * _weighted_dispersion(counts)
            - 0.25 * max_share
        )
    if key == "pattern:two_area_split":
        return _two_area_split_score(counts)
    if key == "pattern:local_cluster":
        return _local_cluster_score(counts) - 0.06 * max(0, active - 3)
    if key == "pattern:central_concentration":
        weights = {
            "center": 1.0,
            "top-center": 0.55,
            "center-left": 0.55,
            "center-right": 0.55,
            "bottom-center": 0.55,
            "top-left": 0.15,
            "top-right": 0.15,
            "bottom-left": 0.15,
            "bottom-right": 0.15,
        }
        return sum(counts[region] * weights[region] for region in GRID_REGIONS) / total
    if key == "pattern:peripheral_spread":
        edge_regions = [region for region in GRID_REGIONS if region != "center"]
        edge_mass = sum(counts[region] for region in edge_regions) / total
        edge_coverage = sum(1 for region in edge_regions if counts[region] > 0) / len(edge_regions)
        return 0.35 * edge_mass + 0.65 * edge_coverage - 0.25 * (counts["center"] / total)
    return None


def _active_bbox(counts: dict[str, float]) -> tuple[int, int]:
    coords = [REGION_COORDS[region] for region, value in counts.items() if value > 0]
    if not coords:
        return 0, 0
    rows = [row for row, _col in coords]
    cols = [col for _row, col in coords]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _weighted_dispersion(counts: dict[str, float]) -> float:
    rows: list[tuple[tuple[int, int], float]] = [
        (REGION_COORDS[region], value) for region, value in counts.items() if value > 0
    ]
    if len(rows) < 2:
        return 0.0
    weighted_distance = 0.0
    total_weight = 0.0
    for index, (left_coord, left_weight) in enumerate(rows):
        for right_coord, right_weight in rows[index + 1 :]:
            weight = left_weight * right_weight
            weighted_distance += weight * math.hypot(left_coord[0] - right_coord[0], left_coord[1] - right_coord[1])
            total_weight += weight
    if total_weight <= 0:
        return 0.0
    return weighted_distance / total_weight / math.hypot(2, 2)


def _two_area_split_score(counts: dict[str, float]) -> float:
    total = sum(counts.values())
    active = [(region, value) for region, value in counts.items() if value > 0]
    if total <= 0 or len(active) < 2:
        return 0.0
    best = 0.0
    for index, (left_region, left_value) in enumerate(active):
        for right_region, right_value in active[index + 1 :]:
            left_coord = REGION_COORDS[left_region]
            right_coord = REGION_COORDS[right_region]
            distance = math.hypot(left_coord[0] - right_coord[0], left_coord[1] - right_coord[1]) / math.hypot(2, 2)
            mass = (left_value + right_value) / total
            balance = min(left_value, right_value) / max(left_value, right_value)
            best = max(best, 0.55 * mass + 0.35 * distance + 0.20 * balance)
    return best


def _local_cluster_score(counts: dict[str, float]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    windows = []
    for row in (0, 1):
        for col in (0, 1):
            windows.append(
                sum(
                    value
                    for region, value in counts.items()
                    if row <= REGION_COORDS[region][0] <= row + 1 and col <= REGION_COORDS[region][1] <= col + 1
                )
            )
    return max(windows) / total


def _slot_from_support(index: int, row: dict[str, Any]) -> dict[str, Any]:
    claim_type = str(row.get("claim_type") or "")
    constraints: dict[str, Any] = {}
    for field in CLAIM_FIELDS_BY_TYPE.get(claim_type, ()):
        if field == "quantity":
            value = row.get("quantity", row.get("value"))
        else:
            value = row.get(field)
        if value is not None:
            constraints[field] = deepcopy(value)

    return {
        "slot_id": f"s_{index:03d}",
        "claim_type": claim_type,
        "scope": {"time": str(row.get("time") or "t1")},
        "constraints": constraints,
        "outputs": {},
        "required": True,
        "complete_scope": True,
        "refs": deepcopy(row.get("refs", {})) if isinstance(row.get("refs"), dict) else {},
    }


def _infer_answer_fn(*, template_variant: str, bindings: dict[str, Any], answer_type: str) -> dict[str, Any]:
    if template_variant == "perception.object_presence.verify":
        return {"op": "bool_label", "claim_type": "Existence", "field": "value", "var": "answer"}
    if template_variant == "perception.object_count.open":
        return {"op": "slot_value_text", "claim_type": "Counting", "field": "quantity", "var": "answer"}
    if template_variant == "perception.object_attribute.verify":
        return {
            "op": "equals_label",
            "claim_type": "Attribute",
            "field": "value",
            "var": "answer_value",
            "target": bindings.get("attr_value"),
        }
    if template_variant == "perception.object_attribute.select":
        return {"op": "slot_value", "claim_type": "Attribute", "field": "value", "var": "answer"}
    if template_variant == "perception.object_location.verify":
        return {
            "op": "equals_label",
            "claim_type": "Location",
            "field": "value",
            "var": "answer_value",
            "target": bindings.get("region"),
        }
    if template_variant == "perception.object_location.select":
        return {"op": "slot_value", "claim_type": "Location", "field": "value", "var": "answer"}

    if template_variant == "relational.pair_relation.verify":
        return {"op": "bool_label", "claim_type": "Relation", "field": "value", "var": "answer"}
    if template_variant == "relational.pair_relation.select":
        return {"op": "relation_field", "claim_type": "Relation", "field": "predicate", "var": "answer"}
    if template_variant == "relational.subject_identification.select":
        return {
            "op": "relation_field",
            "claim_type": "Relation",
            "field": "subject_node",
            "var": "answer",
            "node_to_value": dict(zip(bindings.get("choice_node_ids", []), bindings.get("choice_values", []), strict=False)),
        }
    if template_variant == "relational.object_identification.select":
        return {
            "op": "relation_field",
            "claim_type": "Relation",
            "field": "object_node",
            "var": "answer",
            "node_to_value": dict(zip(bindings.get("choice_node_ids", []), bindings.get("choice_values", []), strict=False)),
        }

    if template_variant == "spatial.proximity.select":
        return {
            "op": "proximity",
            "direction": bindings.get("proximity_direction"),
            "anchor": bindings.get("anchor"),
            "choice_node_ids": list(bindings.get("choice_node_ids", [])),
            "choice_values": list(bindings.get("choice_values", [])),
        }
    if template_variant == "spatial.projective_ordering.select":
        direction_key = str(bindings.get("ordering_direction") or "")
        reverse = direction_key in {"right_to_left", "bottom_to_top"}
        return {
            "op": "projective_relation",
            "task": bindings.get("projective_task"),
            "direction": direction_key,
            "axis": bindings.get("ordering_axis"),
            "reverse": reverse,
            "ordering_node_ids": list(bindings.get("ordering_node_ids", [])),
            "ordering_values": list(bindings.get("ordering_values", [])),
            "rank_index": bindings.get("rank_index"),
            "anchor_node": bindings.get("anchor_node"),
            "adjacent_step": bindings.get("adjacent_step"),
            "anchor_node_ids": list(bindings.get("anchor_node_ids", [])),
        }
    if template_variant == "spatial.aggregate_distribution.select":
        return {
            "op": "aggregate_pattern",
            "task": bindings.get("aggregate_task"),
            "category": bindings.get("category"),
            "answer_key": bindings.get("answer_key"),
            "choice_keys": list(bindings.get("choice_keys", [])),
            "choice_values": list(bindings.get("choice_values", [])),
        }

    if template_variant == "temporal.category_appearance.select":
        return {"op": "temporal_turnover", "mode": "appearance", "candidates": list(bindings.get("choice_values", []))}
    if template_variant == "temporal.category_disappearance.select":
        return {"op": "temporal_turnover", "mode": "disappearance", "candidates": list(bindings.get("choice_values", []))}
    if template_variant == "temporal.max_area_increase.select":
        return {"op": "temporal_delta_argmax", "direction": "increase", "candidates": list(bindings.get("choice_values", []))}
    if template_variant == "temporal.max_area_decrease.select":
        return {"op": "temporal_delta_argmax", "direction": "decrease", "candidates": list(bindings.get("choice_values", []))}
    if template_variant == "temporal.building_damage.select":
        return {"op": "slot_value", "claim_type": "Attribute", "field": "value", "var": "answer", "name": "damage_level"}
    if template_variant == "temporal.semantic_transition_target.select":
        return {"op": "relation_field", "claim_type": "Relation", "field": "object", "var": "answer"}
    if template_variant == "temporal.semantic_transition_source.select":
        return {"op": "relation_field", "claim_type": "Relation", "field": "subject", "var": "answer"}

    raise ValueError(f"unsupported answer function for template variant: {template_variant!r} ({answer_type=})")


def _attach_outputs(slots: list[dict[str, Any]], answer_fn: dict[str, Any]) -> None:
    op = str(answer_fn.get("op") or "")
    claim_type = str(answer_fn.get("claim_type") or "")
    field = str(answer_fn.get("field") or "")
    if op in {"slot_value", "slot_value_text", "bool_label", "equals", "equals_label", "relation_field"}:
        slot = _find_answer_slot(slots, claim_type=claim_type, answer_fn=answer_fn)
        _make_slot_output(slot, field=field, var=str(answer_fn.get("var") or "answer"))
        return

    if op == "temporal_turnover":
        for slot in slots:
            if slot.get("claim_type") != "Existence":
                continue
            subject = str(slot.get("constraints", {}).get("subject") or "")
            time = str(slot.get("scope", {}).get("time") or "")
            _make_slot_output(slot, field="value", var=f"exists:{subject}:{time}")
        return

    if op == "temporal_delta_argmax":
        for slot in slots:
            constraints = slot.get("constraints", {})
            subject = str(constraints.get("subject") or "")
            time = str(slot.get("scope", {}).get("time") or "")
            if slot.get("claim_type") == "Existence":
                _make_slot_output(slot, field="value", var=f"exists:{subject}:{time}")
                continue
            if slot.get("claim_type") != "Attribute" or str(constraints.get("name") or "") != "aggregate_area":
                continue
            _make_slot_output(slot, field="value", var=f"area:{subject}:{time}")
        return

    if op in {"projective_relation", "proximity"}:
        for slot in slots:
            constraints = slot.get("constraints", {})
            if slot.get("claim_type") != "Attribute" or str(constraints.get("name") or "") != "center":
                continue
            node_ids = slot.get("refs", {}).get("node_ids", [])
            if not node_ids:
                continue
            _make_slot_output(slot, field="value", var=f"center:{node_ids[0]}")


def _build_closure_spec(
    *,
    template_variant: str,
    bindings: dict[str, Any],
    slots: list[dict[str, Any]],
) -> dict[str, Any]:
    if template_variant not in {
        "temporal.semantic_transition_target.select",
        "temporal.semantic_transition_source.select",
    }:
        return {}

    source = str(bindings.get("source") or "")
    target = str(bindings.get("target") or "")
    if not source or not target:
        return {}

    relation_slot_ids: set[str] = set()
    state_slot_ids: list[str] = []
    for slot in slots:
        slot_id = str(slot.get("slot_id") or "")
        if not slot_id:
            continue
        if (
            slot.get("claim_type") == "Relation"
            and slot.get("scope", {}).get("time") == "pair"
            and slot.get("constraints", {}).get("predicate") == "transferred_to"
        ):
            relation_slot_ids.add(slot_id)
            continue
        constraints = slot.get("constraints", {})
        subject = str(constraints.get("subject") or "")
        if subject in {source, target} and slot.get("claim_type") in {"Existence", "Attribute"}:
            state_slot_ids.append(slot_id)

    if not state_slot_ids:
        return {}
    return {
        "temporal_compositions": [
            {
                "source": source,
                "target": target,
                "predicate": "transferred_to",
                "value": True,
                "required_slot_ids": state_slot_ids,
                "target_slot_ids": sorted(relation_slot_ids),
            }
        ]
    }


def _find_answer_slot(slots: list[dict[str, Any]], *, claim_type: str, answer_fn: dict[str, Any]) -> dict[str, Any]:
    name = answer_fn.get("name")
    for slot in reversed(slots):
        if slot.get("claim_type") != claim_type:
            continue
        constraints = slot.get("constraints", {})
        if name is not None and constraints.get("name") != name:
            continue
        return slot
    raise ValueError(f"program has no slot for answer function: {answer_fn}")


def _make_slot_output(slot: dict[str, Any], *, field: str, var: str) -> None:
    constraints = slot.setdefault("constraints", {})
    constraints.pop(field, None)
    if field == "subject_node":
        constraints.pop("subject", None)
    if field == "object_node":
        constraints.pop("object", None)
    if field == "quantity":
        constraints.pop("value", None)
    slot.setdefault("outputs", {})[var] = field


def _attach_sr_roles(slots: list[dict[str, Any]]) -> None:
    for slot in slots:
        outputs = slot.get("outputs", {})
        constraints = slot.get("constraints", {})
        if (
            slot.get("claim_type") == "Attribute"
            and isinstance(constraints, dict)
            and str(constraints.get("name") or "") == "aggregate_area"
            and not (isinstance(outputs, dict) and outputs)
        ):
            slot["slot_role"] = "internal_measure"
            slot["sr_required"] = False
            continue
        slot["slot_role"] = "evidence"
        slot["sr_required"] = bool(slot.get("required", True))


def _same_scalar(left: Any, right: Any) -> bool:
    return _normalize_scalar(left) == _normalize_scalar(right)


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower().replace(" ", "_").replace("-", "_")
    if isinstance(value, list):
        return tuple(_normalize_scalar(item) for item in value)
    return value


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes"}:
            return True
        if normalized in {"false", "no"}:
            return False
    return None


def _yes_no(value: bool | None) -> str | None:
    if value is None:
        return None
    return "Yes" if value else "No"


def _temporal_area_value(assignment: dict[str, Any], category: str, time_key: str) -> float | None:
    area = _to_float(assignment.get(f"area:{category}:{time_key}"))
    if area is not None:
        return area
    exists = _to_bool(assignment.get(f"exists:{category}:{time_key}"))
    if exists is False:
        return 0.0
    return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _argmax_unique(rows: list[tuple[str, float]]) -> str | None:
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: (-row[1], row[0]))
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None
    return ordered[0][0]


def _is_point(value: Any) -> bool:
    return isinstance(value, list | tuple) and len(value) == 2 and _to_float(value[0]) is not None and _to_float(value[1]) is not None
