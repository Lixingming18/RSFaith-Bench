from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .answer import gold_answer, normalize_answer, normalize_text_key
from .program import execute_answer_fn


STANDARD_LOCATIONS = (
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


@dataclass(frozen=True)
class VerificationResult:
    claim: dict[str, Any]
    label: str
    reason: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "label": self.label,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class UnmappedClaim:
    claim: Any
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "reason": self.reason}


class SceneGraphIndex:
    def __init__(self, scene_graph: dict[str, Any], label_space: dict[str, set[str]] | None = None) -> None:
        self.scene_graph = scene_graph
        self.nodes = {str(node.get("node_id")): node for node in scene_graph.get("nodes", []) if isinstance(node, dict)}
        self.edges = {str(edge.get("edge_id")): edge for edge in scene_graph.get("edges", []) if isinstance(edge, dict)}
        self.nodes_by_time: dict[str, list[dict[str, Any]]] = {}
        self.object_labels: set[str] = set()
        self.relation_labels: set[str] = {normalize_label("transferred_to")}
        self.location_labels: set[str] = {normalize_label(value) for value in STANDARD_LOCATIONS}
        self.attribute_labels: set[str] = {normalize_label("subtype"), normalize_label("aggregate_area")}
        for node in self.nodes.values():
            self.nodes_by_time.setdefault(str(node.get("time") or "t1"), []).append(node)
            self._add_object_label(node.get("canonical_leaf"))
            self._add_object_label(node.get("canonical_root"))
            geom = node.get("geometry", {}) if isinstance(node.get("geometry"), dict) else {}
            self._add_location_label(geom.get("region_bin_3x3"))
            attrs = node.get("attributes", {}) if isinstance(node.get("attributes"), dict) else {}
            for name in attrs:
                self._add_attribute_label(name)
        for edge in self.edges.values():
            self._add_relation_label(edge.get("relation_type"))
        inventory = scene_graph.get("global_inventory", {})
        self.semantic_inventory = inventory.get("semantic_inventory", {}) if isinstance(inventory, dict) else {}
        self._add_semantic_inventory_labels()
        self._add_label_space(label_space or {})

    def nodes_of_category(self, time: str, category: str) -> list[dict[str, Any]]:
        key = normalize_label(category)
        return [
            node
            for node in self.nodes_by_time.get(time or "t1", [])
            if key in {
                normalize_label(node.get("canonical_leaf")),
                normalize_label(node.get("canonical_root")),
            }
        ]

    def relation_edges(self, predicate: str, subject: str, object_: str, time: str) -> list[dict[str, Any]]:
        subject_ids = {str(node.get("node_id")) for node in self.nodes_of_category(time, subject)}
        object_ids = {str(node.get("node_id")) for node in self.nodes_of_category(time, object_)}
        pred_key = normalize_label(predicate)
        return [
            edge
            for edge in self.edges.values()
            if normalize_label(edge.get("relation_type")) == pred_key
            and str(edge.get("source")) in subject_ids
            and str(edge.get("target")) in object_ids
        ]

    def object_known(self, label: Any) -> bool:
        return normalize_label(label) in self.object_labels

    def relation_known(self, label: Any) -> bool:
        return normalize_label(label) in self.relation_labels

    def location_known(self, label: Any) -> bool:
        return normalize_label(label) in self.location_labels

    def attribute_known(self, label: Any) -> bool:
        return normalize_label(label) in self.attribute_labels

    def transfer_exists(self, source: str, target: str) -> bool:
        transfers = self.semantic_inventory.get("transfers", {})
        if not isinstance(transfers, dict):
            return False
        source_key = normalize_label(source)
        target_key = normalize_label(target)
        for raw_source, target_map in transfers.items():
            if normalize_label(raw_source) != source_key or not isinstance(target_map, dict):
                continue
            for raw_target, value in target_map.items():
                if normalize_label(raw_target) == target_key:
                    try:
                        return float(value) > 0
                    except (TypeError, ValueError):
                        return bool(value)
        return False

    def _add_label_space(self, label_space: dict[str, set[str]]) -> None:
        for value in label_space.get("objects", set()):
            self._add_object_label(value)
        for value in label_space.get("relations", set()):
            self._add_relation_label(value)
        for value in label_space.get("locations", set()):
            self._add_location_label(value)
        for value in label_space.get("attributes", set()):
            self._add_attribute_label(value)

    def _add_semantic_inventory_labels(self) -> None:
        areas = self.semantic_inventory.get("semantic_areas", {})
        if isinstance(areas, dict):
            self._add_attribute_label("aggregate_area")
            for table in areas.values():
                if isinstance(table, dict):
                    for label in table:
                        self._add_object_label(label)
        transfers = self.semantic_inventory.get("transfers", {})
        if isinstance(transfers, dict):
            self._add_relation_label("transferred_to")
            for source, targets in transfers.items():
                self._add_object_label(source)
                if isinstance(targets, dict):
                    for target in targets:
                        self._add_object_label(target)

    def _add_object_label(self, value: Any) -> None:
        _add_normalized_label(self.object_labels, value)

    def _add_relation_label(self, value: Any) -> None:
        _add_normalized_label(self.relation_labels, value)

    def _add_location_label(self, value: Any) -> None:
        _add_normalized_label(self.location_labels, value)

    def _add_attribute_label(self, value: Any) -> None:
        _add_normalized_label(self.attribute_labels, value)


@dataclass(frozen=True)
class ClosedScopeIndex:
    slots: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_item(cls, item: dict[str, Any]) -> ClosedScopeIndex:
        program = item.get("program", {})
        if not isinstance(program, dict):
            return cls()
        slots = tuple(
            slot
            for slot in program.get("slots", [])
            if isinstance(slot, dict) and _slot_has_complete_scope(slot)
        )
        return cls(slots)

    def claim_is_closed(self, claim: dict[str, Any]) -> bool:
        return any(slot_closes_claim(slot, claim) for slot in self.slots)

    def count_is_closed(self, time: str, subject: str) -> bool:
        return self.claim_is_closed({"claim_type": "Counting", "time": time, "subject": subject})


def verify_claims(
    index: SceneGraphIndex,
    claims: list[dict[str, Any]],
    closed_scopes: ClosedScopeIndex | None = None,
) -> list[VerificationResult]:
    verification, _ = verify_claims_with_mapping(index, claims, closed_scopes=closed_scopes)
    return verification


def verify_claims_with_mapping(
    index: SceneGraphIndex,
    claims: list[dict[str, Any]],
    *,
    closed_scopes: ClosedScopeIndex | None = None,
) -> tuple[list[VerificationResult], list[UnmappedClaim]]:
    mapped_claims, unmapped = partition_mappable_claims(index, claims)
    scope_index = closed_scopes or ClosedScopeIndex()
    return [verify_claim(index, claim, scope_index) for claim in mapped_claims], unmapped


def partition_mappable_claims(
    index: SceneGraphIndex,
    claims: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[UnmappedClaim]]:
    mapped: list[dict[str, Any]] = []
    unmapped: list[UnmappedClaim] = []
    for claim in claims:
        if not isinstance(claim, dict):
            unmapped.append(UnmappedClaim(claim, "claim is not a JSON object"))
            continue
        reason = mapping_failure_reason(index, claim)
        if reason:
            unmapped.append(UnmappedClaim(claim, reason))
        else:
            mapped.append(claim)
    return mapped, unmapped


def verify_claim(
    index: SceneGraphIndex,
    claim: dict[str, Any],
    closed_scopes: ClosedScopeIndex | None = None,
) -> VerificationResult:
    scope_index = closed_scopes or ClosedScopeIndex()
    claim_type = claim_type_name(claim)
    if claim_type == "Existence":
        return _verify_existence(index, claim, scope_index)
    if claim_type == "Counting":
        return _verify_counting(index, claim, scope_index)
    if claim_type == "Attribute":
        return _verify_attribute(index, claim, scope_index)
    if claim_type == "Location":
        return _verify_location(index, claim, scope_index)
    if claim_type == "Relation":
        return _verify_relation(index, claim, scope_index)
    raise ValueError(f"claim type did not pass schema mapping: {claim_type}")


def mapping_failure_reason(index: SceneGraphIndex, claim: dict[str, Any]) -> str | None:
    claim_type = claim_type_name(claim)
    if not claim_type:
        return "missing claim_type"
    if claim_type == "Existence":
        return _known_object_failure(index, claim.get("subject"), "subject")
    if claim_type == "Counting":
        return _counting_mapping_failure(index, claim)
    if claim_type == "Attribute":
        return _attribute_mapping_failure(index, claim)
    if claim_type == "Location":
        return _location_mapping_failure(index, claim)
    if claim_type == "Relation":
        return _relation_mapping_failure(index, claim)
    return f"claim type is outside the verifier schema: {claim_type}"


def _counting_mapping_failure(index: SceneGraphIndex, claim: dict[str, Any]) -> str | None:
    subject, region = split_scoped_subject(str(claim.get("subject") or ""))
    failure = _known_object_failure(index, subject, "subject")
    if failure:
        return failure
    if region and not index.location_known(region):
        return f"unknown location label: {region}"

    quantity_kind = normalize_label(claim.get("quantity_kind") or "exact")
    if quantity_kind == "exact":
        expected = claim.get("quantity", claim.get("value"))
        try:
            int(expected)
        except (TypeError, ValueError):
            return f"non-integer exact count: {expected!r}"
        return None
    if quantity_kind == "bound":
        operator = str(claim.get("quantity_operator") or "").strip().lower()
        if operator not in {"gt", "lt"}:
            return f"unknown count comparison operator: {operator or '<missing>'}"
        return _known_object_failure(index, claim.get("object"), "object")
    return f"counting quantity_kind is outside the verifier schema: {claim.get('quantity_kind')!r}"


def _attribute_mapping_failure(index: SceneGraphIndex, claim: dict[str, Any]) -> str | None:
    subject = str(claim.get("subject") or "")
    name = str(claim.get("name") or claim.get("attribute") or "")
    failure = _known_object_failure(index, subject, "subject")
    if failure:
        return failure
    if not index.attribute_known(name):
        return f"unknown attribute label: {name}"
    if normalize_label(name) == "subtype" and not index.object_known(claim.get("value")):
        return f"unknown subtype value label: {claim.get('value')}"
    return None


def _location_mapping_failure(index: SceneGraphIndex, claim: dict[str, Any]) -> str | None:
    subject = str(claim.get("subject") or "")
    location = claim.get("location", claim.get("value"))
    failure = _known_object_failure(index, subject, "subject")
    if failure:
        return failure
    if not index.location_known(location):
        return f"unknown location label: {location}"
    return None


def _relation_mapping_failure(index: SceneGraphIndex, claim: dict[str, Any]) -> str | None:
    subject = str(claim.get("subject") or "")
    object_ = str(claim.get("object") or "")
    predicate = relation_name(claim)
    failure = _known_object_failure(index, subject, "subject")
    if failure:
        return failure
    failure = _known_object_failure(index, object_, "object")
    if failure:
        return failure
    if not index.relation_known(predicate):
        return f"unknown relation label: {predicate}"
    return None


def _known_object_failure(index: SceneGraphIndex, value: Any, field: str) -> str | None:
    if index.object_known(value):
        return None
    return f"unknown {field} label: {value}"


def evidence_label(*, witness: bool, conflict: bool, closed: bool) -> str:
    if witness:
        return "support"
    if conflict and closed:
        return "contradict"
    return "uncertain"


def _verify_existence(
    index: SceneGraphIndex,
    claim: dict[str, Any],
    closed_scopes: ClosedScopeIndex,
) -> VerificationResult:
    time = str(claim.get("time") or "t1")
    subject = str(claim.get("subject") or "")
    nodes = index.nodes_of_category(time, subject)
    expected = to_bool(claim.get("value", True))
    actual = bool(nodes)
    evidence = {"node_ids": [str(node.get("node_id")) for node in nodes]}
    closed = closed_scopes.claim_is_closed(claim)
    if actual:
        label = evidence_label(witness=expected is True, conflict=expected is False, closed=closed)
        return VerificationResult(claim, label, f"existence expected={expected} actual={actual}", evidence)
    label = evidence_label(witness=expected is False and closed, conflict=expected is True, closed=closed)
    if label == "uncertain":
        return VerificationResult(
            claim,
            "uncertain",
            f"existence absence outside closed scope expected={expected}",
            evidence,
        )
    return VerificationResult(claim, label, f"existence expected={expected} actual={actual}", evidence)


def _verify_counting(
    index: SceneGraphIndex,
    claim: dict[str, Any],
    closed_scopes: ClosedScopeIndex,
) -> VerificationResult:
    time = str(claim.get("time") or "t1")
    raw_subject = str(claim.get("subject") or "")
    subject, _region = split_scoped_subject(raw_subject)
    nodes = scoped_nodes(index, time, raw_subject)
    if normalize_label(claim.get("quantity_kind") or "exact") == "bound":
        raw_object = str(claim.get("object") or "")
        object_, _object_region = split_scoped_subject(raw_object)
        other_nodes = scoped_nodes(index, time, raw_object)
        operator = str(claim.get("quantity_operator") or "").strip().lower()
        if operator == "gt":
            holds = len(nodes) > len(other_nodes)
        elif operator == "lt":
            holds = len(nodes) < len(other_nodes)
        else:
            raise ValueError(f"count comparison operator did not pass schema mapping: {operator}")
        evidence = {
            "subject_count": len(nodes),
            "object_count": len(other_nodes),
            "subject_node_ids": [str(node.get("node_id")) for node in nodes],
            "object_node_ids": [str(node.get("node_id")) for node in other_nodes],
        }
        closed = closed_scopes.count_is_closed(time, raw_subject) and closed_scopes.count_is_closed(time, raw_object)
        label = evidence_label(witness=holds and closed, conflict=not holds, closed=closed)
        if label == "uncertain":
            return VerificationResult(
                claim,
                "uncertain",
                f"count comparison outside closed scope: {subject} {operator} {object_}",
                evidence,
            )
        return VerificationResult(
            claim,
            label,
            f"count comparison {subject}={len(nodes)} {operator} {object_}={len(other_nodes)}",
            evidence,
        )
    expected = claim.get("quantity", claim.get("value"))
    try:
        expected_int = int(expected)
    except (TypeError, ValueError):
        raise ValueError(f"exact count did not pass schema mapping: {expected!r}")
    evidence = {"node_ids": [str(node.get("node_id")) for node in nodes], "actual_count": len(nodes)}
    closed = closed_scopes.count_is_closed(time, raw_subject)
    label = evidence_label(witness=len(nodes) == expected_int and closed, conflict=len(nodes) != expected_int, closed=closed)
    if label == "uncertain":
        return VerificationResult(claim, "uncertain", f"exact count outside closed scope: {subject}", evidence)
    return VerificationResult(claim, label, f"count expected={expected_int} actual={len(nodes)}", evidence)


def _verify_attribute(
    index: SceneGraphIndex,
    claim: dict[str, Any],
    closed_scopes: ClosedScopeIndex,
) -> VerificationResult:
    time = str(claim.get("time") or "t1")
    subject = str(claim.get("subject") or "")
    name = str(claim.get("name") or claim.get("attribute") or "")
    name_key = normalize_label(name)
    expected = claim.get("value")
    nodes = index.nodes_of_category(time, subject)
    closed = closed_scopes.claim_is_closed(claim)
    if name_key == "aggregate area":
        actual = semantic_area(index, time, subject)
        area_evidence = {"node_ids": [str(node.get("node_id")) for node in nodes]}
        if actual is None:
            label = evidence_label(witness=False, conflict=True, closed=closed)
            reason = "aggregate_area not present" if closed else "aggregate_area outside closed scope"
            return VerificationResult(claim, label, reason, area_evidence)
        area_evidence["actual"] = actual
        if scalar_equal(actual, expected):
            return VerificationResult(claim, "support", f"aggregate_area expected={expected} actual={actual}", area_evidence)
        label = evidence_label(witness=False, conflict=True, closed=closed)
        if label == "uncertain":
            return VerificationResult(
                claim,
                "uncertain",
                f"aggregate_area mismatch outside closed scope expected={expected} actual={actual}",
                area_evidence,
            )
        return VerificationResult(
            claim,
            label,
            f"aggregate_area expected={expected} actual={actual}",
            area_evidence,
        )
    matched_node_ids: list[str] = []
    for node in nodes:
        attrs = node.get("attributes", {}) if isinstance(node.get("attributes"), dict) else {}
        geom = node.get("geometry", {}) if isinstance(node.get("geometry"), dict) else {}
        if name_key == "subtype":
            actual_values = [node.get("canonical_leaf"), node.get("canonical_root"), attrs.get("subtype")]
            if any(scalar_equal(actual, expected) for actual in actual_values if actual is not None):
                matched_node_ids.append(str(node.get("node_id")))
            continue
        actual = attrs.get(name, geom.get(name))
        if actual is not None and scalar_equal(actual, expected):
            matched_node_ids.append(str(node.get("node_id")))
    if matched_node_ids:
        return VerificationResult(claim, "support", f"attribute {name} matched", {"node_ids": matched_node_ids})
    label = evidence_label(witness=False, conflict=True, closed=closed)
    if label == "uncertain":
        return VerificationResult(claim, "uncertain", f"attribute {name} outside closed scope", {})
    if nodes:
        return VerificationResult(claim, label, f"no {subject} node matched attribute {name}", {})
    return VerificationResult(claim, label, f"no {subject} node found inside closed scope", {})


def _verify_location(
    index: SceneGraphIndex,
    claim: dict[str, Any],
    closed_scopes: ClosedScopeIndex,
) -> VerificationResult:
    time = str(claim.get("time") or "t1")
    subject = str(claim.get("subject") or "")
    location = claim.get("location", claim.get("value"))
    expected = normalize_label(location)
    matched: list[str] = []
    candidates = index.nodes_of_category(time, subject)
    candidate_ids = [str(node.get("node_id")) for node in candidates]
    for node in candidates:
        geom = node.get("geometry", {}) if isinstance(node.get("geometry"), dict) else {}
        if normalize_label(geom.get("region_bin_3x3")) == expected:
            matched.append(str(node.get("node_id")))
    polarity = location_polarity(claim)
    holds = bool(matched)
    evidence = {"node_ids": matched, "candidate_node_ids": candidate_ids}
    closed = closed_scopes.claim_is_closed(claim)
    if polarity is False:
        label = evidence_label(witness=not holds and closed, conflict=holds, closed=closed)
    else:
        label = evidence_label(witness=holds, conflict=not holds, closed=closed)
    if label == "uncertain":
        return VerificationResult(claim, "uncertain", "location mismatch outside closed scope", evidence)
    return VerificationResult(
        claim,
        label,
        f"location matched={bool(matched)}",
        evidence,
    )


def _verify_relation(
    index: SceneGraphIndex,
    claim: dict[str, Any],
    closed_scopes: ClosedScopeIndex,
) -> VerificationResult:
    predicate = relation_name(claim)
    subject = str(claim.get("subject") or "")
    object_ = str(claim.get("object") or "")
    if normalize_label(predicate) == "transferred to":
        actual = index.transfer_exists(subject, object_)
        expected = to_bool(claim.get("value", True))
        closed = closed_scopes.claim_is_closed(claim)
        if actual:
            label = evidence_label(witness=expected is True, conflict=expected is False, closed=closed)
            return VerificationResult(
                claim,
                label,
                f"transfer expected={expected} actual={actual}",
                {},
            )
        label = evidence_label(witness=expected is False and closed, conflict=expected is True, closed=closed)
        if label == "uncertain":
            return VerificationResult(claim, "uncertain", "transfer absence outside closed scope", {})
        return VerificationResult(
            claim,
            label,
            f"transfer expected={expected} actual={actual}",
            {},
        )
    time = str(claim.get("time") or "t1")
    edges = index.relation_edges(predicate, subject, object_, time)
    expected = to_bool(claim.get("value", True))
    actual = bool(edges)
    evidence = {
        "edge_ids": [str(edge.get("edge_id")) for edge in edges],
        "subject_node_ids": [str(edge.get("source")) for edge in edges],
        "object_node_ids": [str(edge.get("target")) for edge in edges],
    }
    closed = closed_scopes.claim_is_closed(claim)
    if actual:
        label = evidence_label(witness=expected is True, conflict=expected is False, closed=closed)
        return VerificationResult(
            claim,
            label,
            f"relation expected={expected} actual={actual}",
            evidence,
        )
    label = evidence_label(witness=expected is False and closed, conflict=expected is True, closed=closed)
    if label == "uncertain":
        return VerificationResult(claim, "uncertain", "relation absence outside closed scope", evidence)
    return VerificationResult(
        claim,
        label,
        f"relation expected={expected} actual={actual}",
        evidence,
    )


def semantic_area(index: SceneGraphIndex, time: str, category: str) -> Any:
    areas = index.semantic_inventory.get("semantic_areas", {})
    if not isinstance(areas, dict):
        return None
    table = areas.get(time, {})
    if not isinstance(table, dict):
        return None
    key = normalize_label(category)
    for raw_label, entry in table.items():
        if normalize_label(raw_label) != key:
            continue
        return entry.get("aggregate_area") if isinstance(entry, dict) else entry
    return None


_MISSING = object()


def contract_supports_gold_answer(
    item: dict[str, Any],
    verification: list[VerificationResult],
    index: SceneGraphIndex,
) -> bool:
    program = item.get("program", {})
    if not isinstance(program, dict):
        return False
    supported = [result for result in verification if result.label == "support"]
    required_slots = [
        slot
        for slot in program.get("slots", [])
        if isinstance(slot, dict) and bool(slot.get("sr_required", True))
    ]
    if not required_slots:
        return False
    for assignment in _complete_slot_assignments(index, supported, required_slots, program):
        derived = execute_answer_fn(program, assignment)
        if _answer_equal(item, derived, gold_answer(item)):
            return True
    return False


def result_matches_slot(result: VerificationResult, slot: dict[str, Any]) -> bool:
    return claim_matches_slot(result.claim, slot) and _refs_compatible(slot, result)


def _complete_slot_assignments(
    index: SceneGraphIndex,
    supported: list[VerificationResult],
    slots: list[dict[str, Any]],
    program: dict[str, Any],
    *,
    slot_index: int = 0,
    assignment: dict[str, Any] | None = None,
):
    current = {} if assignment is None else assignment
    if slot_index >= len(slots):
        yield dict(current)
        return
    slot = slots[slot_index]
    for result in supported:
        if not result_matches_slot(result, slot):
            continue
        candidate = dict(current)
        if not _assign_slot_outputs(index, candidate, result, slot, program):
            continue
        yield from _complete_slot_assignments(
            index,
            supported,
            slots,
            program,
            slot_index=slot_index + 1,
            assignment=candidate,
        )


def _refs_compatible(slot: dict[str, Any], result: VerificationResult) -> bool:
    refs = slot.get("refs", {}) if isinstance(slot.get("refs"), dict) else {}
    node_refs = {str(node_id) for node_id in refs.get("node_ids", [])}
    edge_refs = {str(edge_id) for edge_id in refs.get("edge_ids", [])}
    if node_refs:
        if not node_refs.issubset(result_node_denotation(result)):
            return False
    if edge_refs:
        if not edge_refs.issubset(result_edge_denotation(result)):
            return False
    return True


def result_node_denotation(result: VerificationResult) -> set[str]:
    claim_refs = result.claim.get("refs", {}) if isinstance(result.claim.get("refs"), dict) else {}
    nodes = {str(node_id) for node_id in claim_refs.get("node_ids", [])}
    for key in ("node_ids", "candidate_node_ids", "subject_node_ids", "object_node_ids"):
        nodes.update(str(node_id) for node_id in result.evidence.get(key, []))
    return nodes


def result_edge_denotation(result: VerificationResult) -> set[str]:
    claim_refs = result.claim.get("refs", {}) if isinstance(result.claim.get("refs"), dict) else {}
    edges = {str(edge_id) for edge_id in claim_refs.get("edge_ids", [])}
    edges.update(str(edge_id) for edge_id in result.evidence.get("edge_ids", []))
    return edges


def _assign_slot_outputs(
    index: SceneGraphIndex,
    assignment: dict[str, Any],
    result: VerificationResult,
    slot: dict[str, Any],
    program: dict[str, Any],
) -> bool:
    for var, field in _slot_output_bindings(slot, program):
        value = _slot_output_value(index, result, slot, var, field, program)
        if value is _MISSING:
            return False
        if var in assignment and not scalar_equal(assignment[var], value):
            return False
        assignment[var] = value
    return True


def _slot_output_bindings(slot: dict[str, Any], program: dict[str, Any]) -> list[tuple[str, str]]:
    outputs = slot.get("outputs", {}) if isinstance(slot.get("outputs"), dict) else {}
    bindings = [(str(var), str(field)) for var, field in outputs.items()]
    count_var = _aggregate_count_var(slot, program)
    if count_var is not None and all(var != count_var for var, _field in bindings):
        bindings.append((count_var, "quantity"))
    return bindings


def _aggregate_count_var(slot: dict[str, Any], program: dict[str, Any]) -> str | None:
    answer_fn = program.get("answer_fn", {}) if isinstance(program.get("answer_fn"), dict) else {}
    if answer_fn.get("op") != "aggregate_pattern" or slot.get("claim_type") != "Counting":
        return None
    constraints = slot.get("constraints", {}) if isinstance(slot.get("constraints"), dict) else {}
    subject = str(constraints.get("subject") or "")
    if "::" not in subject:
        return None
    category = str(answer_fn.get("category") or "")
    slot_category, region = subject.rsplit("::", 1)
    if category and normalize_label(slot_category) != normalize_label(category):
        return None
    return f"count:{category or slot_category}::{region}"


def _slot_output_value(
    index: SceneGraphIndex,
    result: VerificationResult,
    slot: dict[str, Any],
    var: str,
    field: str,
    program: dict[str, Any],
) -> Any:
    claim = result.claim
    if field == "value":
        if claim_type_name(claim) == "Location":
            location = claim.get("location", claim.get("value"))
            polarity = location_polarity(claim)
            if polarity is False:
                target = answer_target_for_output(program, slot, var, field)
                if target is not _MISSING and scalar_equal(location, target):
                    return False
                return _MISSING
            return location
        return claim.get("value")
    if field == "quantity":
        return claim.get("quantity", claim.get("value"))
    if field == "predicate":
        return relation_name(claim)
    if field in {"subject", "object"}:
        return claim.get(field)
    if field in {"subject_node", "object_node"}:
        edge = _output_edge(index, result, slot)
        if edge is None:
            return _MISSING
        return edge.get("source" if field == "subject_node" else "target")
    return claim.get(field, _MISSING)


def answer_target_for_output(program: dict[str, Any], slot: dict[str, Any], var: str, field: str) -> Any:
    answer_fn = program.get("answer_fn", {}) if isinstance(program.get("answer_fn"), dict) else {}
    if answer_fn.get("op") not in {"equals", "equals_label"}:
        return _MISSING
    if str(answer_fn.get("claim_type") or "") != str(slot.get("claim_type") or ""):
        return _MISSING
    if str(answer_fn.get("var") or "") != var or str(answer_fn.get("field") or "") != field:
        return _MISSING
    return answer_fn.get("target", _MISSING)


def _output_edge(index: SceneGraphIndex, result: VerificationResult, slot: dict[str, Any]) -> dict[str, Any] | None:
    refs = slot.get("refs", {}) if isinstance(slot.get("refs"), dict) else {}
    slot_edge_ids = [str(edge_id) for edge_id in refs.get("edge_ids", [])]
    result_edge_ids = [str(edge_id) for edge_id in result.evidence.get("edge_ids", [])]
    for edge_id in [*slot_edge_ids, *result_edge_ids]:
        edge = index.edges.get(edge_id)
        if edge is not None:
            return edge
    return None


def _answer_equal(item: dict[str, Any], left: Any, right: Any) -> bool:
    left_norm = normalize_text_key(normalize_answer(item, left))
    right_norm = normalize_text_key(normalize_answer(item, right))
    return left_norm == right_norm or _compact_answer_key(left_norm) == _compact_answer_key(right_norm)


def _compact_answer_key(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def build_item_label_space(item: dict[str, Any]) -> dict[str, set[str]]:
    label_space = {"objects": set(), "relations": set(), "locations": set(), "attributes": set()}
    for claim in item.get("support", []):
        if isinstance(claim, dict):
            _add_claim_labels(label_space, claim.get("claim_type"), claim)
    for slot in item.get("program", {}).get("slots", []):
        if not isinstance(slot, dict):
            continue
        constraints = slot.get("constraints", {}) if isinstance(slot.get("constraints"), dict) else {}
        _add_claim_labels(label_space, slot.get("claim_type"), constraints)
    return label_space


def _slot_has_complete_scope(slot: dict[str, Any]) -> bool:
    value = slot.get("complete_scope", True)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "no"}


def slot_closes_claim(slot: dict[str, Any], claim: dict[str, Any]) -> bool:
    claim_type = claim_type_name(claim)
    if claim_type != str(slot.get("claim_type") or ""):
        return False
    scope = slot.get("scope", {}) if isinstance(slot.get("scope"), dict) else {}
    if scope.get("time") and str(claim.get("time") or "t1") != str(scope.get("time")):
        return False
    constraints = slot.get("constraints", {}) if isinstance(slot.get("constraints"), dict) else {}
    for key in closed_scope_fields(claim_type):
        if key not in constraints:
            continue
        if not scalar_equal(claim_scope_value(claim, key), constraints[key]):
            return False
    return True


def closed_scope_fields(claim_type: str) -> tuple[str, ...]:
    if claim_type == "Existence":
        return ("subject",)
    if claim_type == "Counting":
        return ("subject", "object")
    if claim_type == "Attribute":
        return ("subject", "name")
    if claim_type == "Location":
        return ("subject",)
    if claim_type == "Relation":
        return ("subject", "predicate", "object")
    return ()


def claim_scope_value(claim: dict[str, Any], key: str) -> Any:
    if key == "name":
        return claim.get("name", claim.get("attribute"))
    if key == "predicate":
        return relation_name(claim)
    return claim.get(key)


def _add_claim_labels(label_space: dict[str, set[str]], claim_type: Any, claim: dict[str, Any]) -> None:
    subject, region = split_scoped_subject(str(claim.get("subject") or ""))
    _add_raw_label(label_space["objects"], subject)
    _add_raw_label(label_space["locations"], region)
    _add_raw_label(label_space["objects"], claim.get("object"))
    _add_raw_label(label_space["relations"], claim.get("predicate") or claim.get("relation"))
    _add_raw_label(label_space["attributes"], claim.get("name") or claim.get("attribute"))
    if str(claim_type or "") == "Location":
        _add_raw_label(label_space["locations"], claim.get("location", claim.get("value")))
    if str(claim_type or "") == "Attribute" and normalize_label(claim.get("name") or claim.get("attribute")) == "subtype":
        _add_raw_label(label_space["objects"], claim.get("value"))


def claim_matches_slot(claim: dict[str, Any], slot: dict[str, Any]) -> bool:
    claim_type = claim_type_name(claim)
    if claim_type != str(slot.get("claim_type") or ""):
        return False
    scope = slot.get("scope", {}) if isinstance(slot.get("scope"), dict) else {}
    if scope.get("time") and str(claim.get("time") or "") != str(scope.get("time")):
        return False
    constraints = slot.get("constraints", {}) if isinstance(slot.get("constraints"), dict) else {}
    for key, expected in constraints.items():
        actual = claim.get(key)
        if key == "value" and claim_type == "Location" and "location" in claim:
            actual = claim.get("location")
        if key == "quantity":
            actual = claim.get("quantity", claim.get("value"))
        if key == "name":
            actual = claim.get("name", claim.get("attribute"))
        if key == "predicate":
            actual = claim.get("predicate", claim.get("relation"))
        if not scalar_equal(actual, expected):
            return False
    return True


def split_scoped_subject(subject: str) -> tuple[str, str | None]:
    if "::" not in subject:
        return subject, None
    base, region = subject.split("::", 1)
    return base, region


def scoped_nodes(index: SceneGraphIndex, time: str, scoped_subject: str) -> list[dict[str, Any]]:
    subject, region = split_scoped_subject(scoped_subject)
    nodes = index.nodes_of_category(time, subject)
    if region:
        return [node for node in nodes if node_region(node) == normalize_label(region)]
    return nodes


def location_polarity(claim: dict[str, Any]) -> bool:
    if "polarity" in claim:
        return to_bool(claim.get("polarity"))
    if "location" in claim:
        return to_bool(claim.get("value", True))
    return True


def claim_type_name(claim: dict[str, Any]) -> str:
    return str(claim.get("claim_type") or claim.get("type") or "")


def relation_name(claim: dict[str, Any]) -> str:
    return str(claim.get("predicate") or claim.get("relation") or "")


def _add_normalized_label(labels: set[str], value: Any) -> None:
    normalized = normalize_label(value)
    if normalized:
        labels.add(normalized)


def _add_raw_label(labels: set[str], value: Any) -> None:
    if isinstance(value, bool) or value is None:
        return
    text = str(value).strip()
    if text:
        labels.add(text)


def node_region(node: dict[str, Any]) -> str:
    geom = node.get("geometry", {}) if isinstance(node.get("geometry"), dict) else {}
    return normalize_label(geom.get("region_bin_3x3"))


def validate_refs(item: dict[str, Any], scene_graph: dict[str, Any]) -> list[str]:
    node_ids = {str(node.get("node_id")) for node in scene_graph.get("nodes", []) if isinstance(node, dict)}
    edge_ids = {str(edge.get("edge_id")) for edge in scene_graph.get("edges", []) if isinstance(edge, dict)}
    errors: list[str] = []
    for source, refs in iter_ref_sets(item):
        for node_id in refs.get("node_ids", []):
            if str(node_id) not in node_ids:
                errors.append(f"{source}: missing node {node_id}")
        for edge_id in refs.get("edge_ids", []):
            if str(edge_id) not in edge_ids:
                errors.append(f"{source}: missing edge {edge_id}")
    return errors


def iter_ref_sets(item: dict[str, Any]):
    for index, support in enumerate(item.get("support", []), start=1):
        refs = support.get("refs", {}) if isinstance(support, dict) else {}
        if isinstance(refs, dict):
            yield f"support[{index}]", refs
    for index, slot in enumerate(item.get("program", {}).get("slots", []), start=1):
        refs = slot.get("refs", {}) if isinstance(slot, dict) else {}
        if isinstance(refs, dict):
            yield f"program.slots[{index}]", refs


def normalize_label(value: Any) -> str:
    return normalize_text_key(str(value or "").replace("_", " "))


def scalar_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        try:
            return abs(float(left) - float(right)) <= 1e-6
        except (TypeError, ValueError):
            return False
    return normalize_label(left) == normalize_label(right)


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return bool(value)
