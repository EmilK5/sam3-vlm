"""Candidate graph with legacy scores and serializable ASHT belief state."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from agent.asht.belief import PatchBelief


GRAPH_SCHEMA_VERSION = "2.0"


class OrchardNode:
    """One detected candidate tracked across passes.

    ``classification`` and ``scores`` remain the legacy verifier state.  ASHT
    state is stored separately in ``belief`` so an intermediate MAP class never
    silently becomes a final declaration.
    """

    def __init__(
        self,
        box: list,
        score: float,
        found_in_pass: int,
        *,
        node_id: str | None = None,
        source_detection_id: str | None = None,
    ):
        self.id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        self.box = [float(coord) for coord in box]
        self.found_in_pass = int(found_in_pass)

        self.scores = {
            "detection_confidence": float(score),
            "fruit_verification": 0.0,
            "leaf_verification": 0.0,
        }
        self.classification = "unresolved"
        self.tree_roi = None
        self.cached_leaf_boxes = None

        self.vip_chain = None
        self.vip_posterior = None

        self.support = 1
        self.signatures = set()
        self.jitter = 0.0
        self.area = self._box_area()
        self.mask = None

        # Phase 5: probabilistic state and complete node lineage.
        self.belief: "PatchBelief | None" = None
        self.source_detection_ids: list[str] = []
        if source_detection_id is not None:
            self.source_detection_ids.append(str(source_detection_id))
        self.found_in_passes: list[int] = [int(found_in_pass)]
        self.dedup_decision_ids: list[str] = []
        self.registration_ids: list[str] = []
        self.verification_action_ids: list[str] = []
        self.observation_ids: list[str] = []
        self.belief_update_ids: list[str] = []
        self.exemplar_uses: list[dict[str, Any]] = []
        self.used_semantic_keys: set[str] = set()

    def _box_area(self) -> float:
        return float((self.box[2] - self.box[0]) * (self.box[3] - self.box[1]))

    @staticmethod
    def _center(box) -> tuple:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @property
    def prior(self) -> dict[str, float] | None:
        return None if self.belief is None else self.belief.prior_mapping()

    @property
    def posterior(self) -> dict[str, float] | None:
        return None if self.belief is None else self.belief.as_mapping()

    @property
    def temporary_map_class(self) -> str | None:
        return None if self.belief is None else self.belief.map_class

    @property
    def final_declaration(self) -> str | None:
        return None if self.belief is None else self.belief.decision

    @property
    def query_count(self) -> int:
        return 0 if self.belief is None else self.belief.query_count

    def initialize_belief(
        self,
        probabilities: Mapping[str, float],
        *,
        overwrite: bool = False,
    ) -> "PatchBelief":
        from agent.asht.belief import PatchBelief

        if self.belief is not None and not overwrite:
            return self.belief
        self.belief = PatchBelief.from_mapping(self.id, probabilities)
        return self.belief

    def reinforce(
        self,
        box: list,
        signature: str | None,
        *,
        source_detection_id: str | None = None,
        found_in_pass: int | None = None,
        dedup_decision_id: str | None = None,
        registration_id: str | None = None,
    ) -> None:
        """Record a cross-pass re-detection and its provenance."""

        cx, cy = self._center(box)
        rx, ry = self._center(self.box)
        displacement = float(((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5)

        self.support += 1
        if signature is not None:
            self.signatures.add(signature)
        n_redetections = self.support - 1
        self.jitter += (displacement - self.jitter) / n_redetections

        if source_detection_id is not None and source_detection_id not in self.source_detection_ids:
            self.source_detection_ids.append(source_detection_id)
        if found_in_pass is not None and int(found_in_pass) not in self.found_in_passes:
            self.found_in_passes.append(int(found_in_pass))
            self.found_in_passes.sort()
        if dedup_decision_id is not None:
            self.dedup_decision_ids.append(dedup_decision_id)
        if registration_id is not None:
            self.registration_ids.append(registration_id)

    def record_verification(
        self,
        *,
        action_id: str,
        semantic_key: str,
        observation_id: str,
        belief_update_id: str,
        positive_exemplar_node_ids: tuple[str, ...] = (),
        negative_exemplar_node_ids: tuple[str, ...] = (),
    ) -> None:
        self.verification_action_ids.append(action_id)
        self.observation_ids.append(observation_id)
        self.belief_update_ids.append(belief_update_id)
        self.used_semantic_keys.add(semantic_key)
        for exemplar_id in positive_exemplar_node_ids:
            self.exemplar_uses.append(
                {"action_id": action_id, "role": "positive", "node_id": exemplar_id}
            )
        for exemplar_id in negative_exemplar_node_ids:
            self.exemplar_uses.append(
                {"action_id": action_id, "role": "negative", "node_id": exemplar_id}
            )

    def exemplar_eligible(
        self,
        class_name: str,
        *,
        threshold: float,
    ) -> bool:
        if self.belief is None or self.belief.status != "stopped":
            return False
        posterior = self.belief.as_mapping()
        return (
            self.belief.decision == class_name
            and posterior.get(class_name, 0.0) >= float(threshold)
        )

    def to_dict(
        self,
        *,
        include_mask: bool = True,
        mask_artifact_path: str | None = None,
    ) -> dict:
        d: dict[str, Any] = {
            "id": self.id,
            "box": list(self.box),
            "found_in_pass": self.found_in_pass,
            "scores": dict(self.scores),
            "classification": self.classification,
            "support": self.support,
            "jitter": self.jitter,
            "area": self.area,
            "signatures": sorted(self.signatures),
            "belief": None if self.belief is None else self.belief.to_dict(),
            "lineage": {
                "source_detection_ids": list(self.source_detection_ids),
                "found_in_passes": list(self.found_in_passes),
                "dedup_decision_ids": list(self.dedup_decision_ids),
                "registration_ids": list(self.registration_ids),
                "verification_action_ids": list(self.verification_action_ids),
                "observation_ids": list(self.observation_ids),
                "belief_update_ids": list(self.belief_update_ids),
                "exemplar_uses": [dict(item) for item in self.exemplar_uses],
                "used_semantic_keys": sorted(self.used_semantic_keys),
            },
        }
        if include_mask:
            d["mask"] = _encode_mask(self.mask)
        elif mask_artifact_path is not None:
            d["mask_artifact_path"] = mask_artifact_path
        if self.vip_chain is not None:
            d["vip_chain"] = self.vip_chain
        if self.vip_posterior is not None:
            d["vip_posterior"] = self.vip_posterior
        return d

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OrchardNode":
        """Load both Phase 5 and legacy node dictionaries."""

        node = cls(
            list(payload["box"]),
            float(payload.get("scores", {}).get("detection_confidence", 0.0)),
            int(payload.get("found_in_pass", 0)),
            node_id=str(payload["id"]),
        )
        node.scores.update(
            {str(key): float(value) for key, value in payload.get("scores", {}).items()}
        )
        node.classification = str(payload.get("classification", "unresolved"))
        node.support = int(payload.get("support", 1))
        node.jitter = float(payload.get("jitter", 0.0))
        node.area = float(payload.get("area", node._box_area()))
        node.signatures = {str(value) for value in payload.get("signatures", [])}
        node.vip_chain = payload.get("vip_chain")
        node.vip_posterior = payload.get("vip_posterior")
        node.mask = _decode_mask(payload.get("mask"))

        belief_payload = payload.get("belief")
        if isinstance(belief_payload, Mapping):
            from agent.asht.belief import PatchBelief

            node.belief = PatchBelief.from_dict(belief_payload)
            if node.belief.node_id != node.id:
                raise ValueError("serialized belief node_id does not match graph node id")

        lineage = payload.get("lineage", {})
        if not isinstance(lineage, Mapping):
            lineage = {}
        node.source_detection_ids = [
            str(value) for value in lineage.get("source_detection_ids", payload.get("source_detection_ids", []))
        ]
        node.found_in_passes = sorted(
            {int(value) for value in lineage.get("found_in_passes", [node.found_in_pass])}
        )
        node.dedup_decision_ids = [str(value) for value in lineage.get("dedup_decision_ids", [])]
        node.registration_ids = [str(value) for value in lineage.get("registration_ids", [])]
        node.verification_action_ids = [
            str(value) for value in lineage.get("verification_action_ids", [])
        ]
        node.observation_ids = [str(value) for value in lineage.get("observation_ids", [])]
        node.belief_update_ids = [str(value) for value in lineage.get("belief_update_ids", [])]
        node.exemplar_uses = [dict(value) for value in lineage.get("exemplar_uses", [])]
        node.used_semantic_keys = {
            str(value) for value in lineage.get("used_semantic_keys", [])
        }
        return node


class OrchardGraph:
    """Tracks candidates, posterior beliefs, and complete object lineage."""

    def __init__(self):
        self.nodes: dict[str, OrchardNode] = {}
        self.tree_roi = None
        self.cached_leaf_boxes = None
        self.cached_leaf_roi = None

    def clear(self):
        self.nodes.clear()
        self.tree_roi = None
        self.cached_leaf_boxes = None
        self.cached_leaf_roi = None

    def add_candidate(
        self,
        box: list,
        score: float,
        found_in_pass: int,
        *,
        node_id: str | None = None,
        source_detection_id: str | None = None,
    ) -> str:
        node = OrchardNode(
            box,
            score,
            found_in_pass,
            node_id=node_id,
            source_detection_id=source_detection_id,
        )
        if node.id in self.nodes:
            raise ValueError(f"duplicate graph node id: {node.id}")
        self.nodes[node.id] = node
        return node.id

    def initialize_beliefs(
        self,
        probabilities: Mapping[str, float],
        *,
        overwrite: bool = False,
    ) -> tuple[str, ...]:
        initialized = []
        for node in self.nodes.values():
            if node.belief is None or overwrite:
                node.initialize_belief(probabilities, overwrite=overwrite)
                initialized.append(node.id)
        return tuple(initialized)

    def unresolved_nodes(self) -> tuple[OrchardNode, ...]:
        return tuple(
            node
            for node in self.nodes.values()
            if node.belief is not None and node.belief.status == "unresolved"
        )

    def update_verdict(self, node_id: str, fruit_score: float, leaf_score: float):
        """Legacy verifier state; intentionally separate from ASHT decisions."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.scores["fruit_verification"] = float(fruit_score)
            node.scores["leaf_verification"] = float(leaf_score)
            if fruit_score > 0.2 and leaf_score < 0.75:
                node.classification = "fruit"
            else:
                node.classification = "leaf"

    def get_exemplars(
        self,
        *,
        positive_class: str = "fruit",
        negative_class: str = "leaf",
        positive_threshold: float = 0.90,
        negative_threshold: float = 0.90,
    ) -> tuple[np.ndarray, np.ndarray]:
        pos_boxes = []
        neg_boxes = []
        for node in self.nodes.values():
            if node.belief is not None:
                if node.exemplar_eligible(positive_class, threshold=positive_threshold):
                    pos_boxes.append(node.box)
                elif node.exemplar_eligible(negative_class, threshold=negative_threshold):
                    neg_boxes.append(node.box)
            elif node.classification == positive_class:
                pos_boxes.append(node.box)
            elif node.classification == negative_class:
                neg_boxes.append(node.box)
        return _box_array(pos_boxes), _box_array(neg_boxes)

    def snapshot_records(
        self,
        *,
        pass_id: str,
        class_names: tuple[str, ...],
        positive_class: str | None = None,
        negative_class: str | None = None,
        positive_threshold: float = 0.90,
        negative_threshold: float = 0.90,
    ) -> tuple[Any, ...]:
        return tuple(
            _node_snapshot_record(
                node,
                pass_id=pass_id,
                class_names=class_names,
                positive_class=positive_class,
                negative_class=negative_class,
                positive_threshold=positive_threshold,
                negative_threshold=negative_threshold,
            )
            for node in sorted(self.nodes.values(), key=lambda value: value.id)
        )

    def to_dict(
        self,
        *,
        include_masks: bool = True,
        mask_artifact_paths: Mapping[str, str] | None = None,
    ) -> dict:
        paths = dict(mask_artifact_paths or {})
        return {
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "nodes": [
                node.to_dict(
                    include_mask=include_masks,
                    mask_artifact_path=paths.get(node.id),
                )
                for node in sorted(self.nodes.values(), key=lambda value: value.id)
            ],
            "metadata": {
                "tree_roi": self.tree_roi,
                "cached_leaf_roi": self.cached_leaf_roi,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OrchardGraph":
        graph = cls()
        # Legacy payloads contain only {"nodes": [...]} and are accepted.
        for node_payload in payload.get("nodes", []):
            node = OrchardNode.from_dict(node_payload)
            if node.id in graph.nodes:
                raise ValueError(f"duplicate graph node id: {node.id}")
            graph.nodes[node.id] = node
        metadata = payload.get("metadata", {})
        if isinstance(metadata, Mapping):
            graph.tree_roi = metadata.get("tree_roi")
            graph.cached_leaf_roi = metadata.get("cached_leaf_roi")
        return graph


def _node_snapshot_record(
    node: OrchardNode,
    *,
    pass_id: str,
    class_names: tuple[str, ...],
    positive_class: str | None,
    negative_class: str | None,
    positive_threshold: float,
    negative_threshold: float,
):
    from provenance.contracts import GraphNodeSnapshotRecord
    from provenance.schema import NodeStatus

    if node.belief is None:
        uniform = 1.0 / len(class_names)
        prior = {name: uniform for name in class_names}
        posterior = dict(prior)
        status = NodeStatus.UNRESOLVED
        temporary = max(posterior, key=posterior.get)
        final = None
        query_count = 0
    else:
        prior = node.belief.prior_mapping()
        posterior = node.belief.as_mapping()
        status = NodeStatus(node.belief.status)
        temporary = node.belief.map_class
        final = node.belief.decision
        query_count = node.belief.query_count

    return GraphNodeSnapshotRecord(
        graph_node_id=node.id,
        pass_id=pass_id,
        box=tuple(float(value) for value in node.box),
        status=status,
        source_detection_ids=tuple(node.source_detection_ids),
        found_in_passes=tuple(node.found_in_passes),
        prior=prior,
        posterior=posterior,
        temporary_map_class=temporary,
        final_declaration=final,
        query_count=query_count,
        exemplar_eligible_positive=(
            positive_class is not None
            and node.exemplar_eligible(positive_class, threshold=positive_threshold)
        ),
        exemplar_eligible_negative=(
            negative_class is not None
            and node.exemplar_eligible(negative_class, threshold=negative_threshold)
        ),
        legacy_scores={str(key): float(value) for key, value in node.scores.items()},
        metadata={
            "legacy_classification": node.classification,
            "lifecycle": {
                "created_in_pass": node.found_in_pass,
                "created_from_detection_id": (
                    node.source_detection_ids[0] if node.source_detection_ids else None
                ),
                "updates": [
                    {
                        "pass": pass_number,
                        "detection_id": (
                            node.source_detection_ids[index]
                            if index < len(node.source_detection_ids)
                            else None
                        ),
                    }
                    for index, pass_number in enumerate(node.found_in_passes[1:], start=1)
                ],
            },
            "dedup_decision_ids": list(node.dedup_decision_ids),
            "registration_ids": list(node.registration_ids),
            "verification_action_ids": list(node.verification_action_ids),
            "observation_ids": list(node.observation_ids),
            "belief_update_ids": list(node.belief_update_ids),
            "exemplar_uses": [dict(item) for item in node.exemplar_uses],
            "used_semantic_keys": sorted(node.used_semantic_keys),
        },
    )


def _box_array(boxes: list[list[float]]) -> np.ndarray:
    if not boxes:
        return np.empty((0, 4), dtype=float)
    return np.asarray(boxes, dtype=float).reshape((-1, 4))


def _encode_mask(mask: np.ndarray | None) -> dict[str, Any] | None:
    if mask is None:
        return None
    flat = np.asarray(mask, dtype=bool).reshape((-1,))
    runs: list[int] = []
    current = False
    count = 0
    for value in flat:
        bit = bool(value)
        if bit == current:
            count += 1
        else:
            runs.append(count)
            count = 1
            current = bit
    runs.append(count)
    return {
        "encoding": "bool_rle_v1",
        "shape": list(mask.shape),
        "starts_with": False,
        "runs": runs,
    }


def _decode_mask(payload: Any) -> np.ndarray | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping) or payload.get("encoding") != "bool_rle_v1":
        raise ValueError("unsupported serialized mask encoding")
    shape = tuple(int(value) for value in payload["shape"])
    values: list[bool] = []
    bit = bool(payload.get("starts_with", False))
    for count in payload["runs"]:
        values.extend([bit] * int(count))
        bit = not bit
    expected = int(np.prod(shape, dtype=int))
    if len(values) != expected:
        raise ValueError("serialized mask RLE length does not match shape")
    return np.asarray(values, dtype=bool).reshape(shape)


__all__ = ["GRAPH_SCHEMA_VERSION", "OrchardGraph", "OrchardNode"]
