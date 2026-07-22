"""Qwen-backed candidate sensing-action generation for ASHT.

The generator asks Qwen for a finite, structured action set.  It validates every
field, records the exact request/response, materializes surrogate kernels, and
falls back to a static action bank when generation fails.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from agent.asht.action_bank import MaterializedStaticAction, StaticActionBank
from agent.asht.kernels import SensorProfile, build_surrogate_kernel
from provenance.base import ContractError
from provenance.contracts import (
    CandidateActionSetRecord,
    QwenCallRecord,
    SensingActionRecord,
)
from provenance.ids import EntityKind
from provenance.schema import ActionFamily, ActionKind, TilingMode


class QwenActionClient(Protocol):
    """Minimal adapter expected by :class:`QwenCandidateActionGenerator`."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image: Any,
        sampling_parameters: Mapping[str, Any],
    ) -> Any:
        ...


@dataclass(frozen=True)
class QwenActionGeneratorConfig:
    model_id: str = "qwen"
    max_actions: int = 8
    min_actions: int = 1
    max_prompt_chars: int = 320
    max_retries: int = 1
    default_threshold: float = 0.50
    default_roi_padding_fraction: float = 0.25
    allow_qwen_regions: bool = True
    allow_tiling: bool = True
    sampling_parameters: Mapping[str, Any] = field(
        default_factory=lambda: {"temperature": 0.0}
    )
    image_artifact_ids: tuple[str, ...] = ()
    system_prompt: str = (
        "You generate candidate sensing actions for a SAM3 active visual "
        "verification system. Return strict JSON only. Do not claim that a box "
        "or prompt is a detection. Each action is only a proposal for SAM3. "
        "Use concise rationales, not hidden chain-of-thought."
    )

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if self.min_actions < 1 or self.max_actions < self.min_actions:
            raise ValueError("invalid action-count bounds")
        if self.max_prompt_chars < 1 or self.max_retries < 0:
            raise ValueError("invalid prompt/retry limits")
        if not 0.0 <= self.default_threshold <= 1.0:
            raise ValueError("default_threshold must lie in [0, 1]")
        if self.default_roi_padding_fraction < 0.0:
            raise ValueError("default_roi_padding_fraction must be non-negative")


@dataclass(frozen=True)
class QwenGenerationResult:
    actions: tuple[MaterializedStaticAction, ...]
    candidate_set: CandidateActionSetRecord
    qwen_call: QwenCallRecord


class QwenCandidateActionGenerator:
    """Generate, validate, and materialize image-dependent action candidates."""

    def __init__(
        self,
        *,
        client: QwenActionClient,
        profile: SensorProfile,
        fallback_bank: StaticActionBank,
        config: QwenActionGeneratorConfig | None = None,
    ) -> None:
        self.client = client
        self.profile = profile
        self.fallback_bank = fallback_bank
        self.config = config or QwenActionGeneratorConfig()

    def resolved_mapping(self) -> Mapping[str, Any]:
        """Return every non-secret setting that can change generated actions."""

        return {
            "generator_type": f"{type(self).__module__}.{type(self).__qualname__}",
            "client_type": f"{type(self.client).__module__}.{type(self.client).__qualname__}",
            "config": {
                "model_id": self.config.model_id,
                "max_actions": self.config.max_actions,
                "min_actions": self.config.min_actions,
                "max_prompt_chars": self.config.max_prompt_chars,
                "max_retries": self.config.max_retries,
                "default_threshold": self.config.default_threshold,
                "default_roi_padding_fraction": self.config.default_roi_padding_fraction,
                "allow_qwen_regions": self.config.allow_qwen_regions,
                "allow_tiling": self.config.allow_tiling,
                "sampling_parameters": dict(self.config.sampling_parameters),
                "image_artifact_ids": list(self.config.image_artifact_ids),
                "system_prompt": self.config.system_prompt,
            },
            "sensor_profile": {
                "observation_labels": list(self.profile.observation_labels),
                "present": list(self.profile.present),
                "absent": list(self.profile.absent),
                "source": self.profile.source,
            },
            "fallback_bank": {
                "allow_semantic_reuse": self.fallback_bank.allow_semantic_reuse,
                "positive_class": self.fallback_bank.positive_class,
                "negative_class": self.fallback_bank.negative_class,
                "positive_exemplar_threshold": self.fallback_bank.positive_exemplar_threshold,
                "negative_exemplar_threshold": self.fallback_bank.negative_exemplar_threshold,
                "templates": [
                    {
                        "name": item.name,
                        "family": item.family.value,
                        "prompt": item.prompt,
                        "semantic_key": item.semantic_key,
                        "beta_by_class": dict(item.beta_by_class),
                        "threshold": item.threshold,
                        "tiling_mode": item.tiling_mode.value,
                        "tile_scale": item.tile_scale,
                        "roi_padding_fraction": item.roi_padding_fraction,
                        "expected_cost": item.expected_cost,
                        "use_positive_exemplars": item.use_positive_exemplars,
                        "use_negative_exemplars": item.use_negative_exemplars,
                        "rationale": item.rationale,
                        "expected_visual_distinction": item.expected_visual_distinction,
                    }
                    for item in self.fallback_bank.templates
                ],
            },
        }

    def generate(
        self,
        *,
        graph,
        node,
        class_names: Sequence[str],
        image: Any,
        image_width: int,
        image_height: int,
        pass_id: str,
        id_source,
    ) -> QwenGenerationResult:
        classes = tuple(str(value) for value in class_names)
        system_prompt = self.config.system_prompt
        user_prompt = self._build_user_prompt(
            graph=graph,
            node=node,
            class_names=classes,
            image_width=image_width,
            image_height=image_height,
        )
        attempt_log: list[Mapping[str, Any]] = []
        request_payload = {
            "class_names": list(classes),
            "target_node_id": node.id,
            "image_width": int(image_width),
            "image_height": int(image_height),
            "sampling_parameters": dict(self.config.sampling_parameters),
            "response_schema": self._response_schema(classes),
        }

        started = time.perf_counter()
        raw_response: Any = None
        parsed_response: Mapping[str, Any] | None = None
        validation_errors: list[str] = []
        rejected_payloads: list[Mapping[str, Any]] = []
        generated: tuple[MaterializedStaticAction, ...] = ()
        retries_used = 0
        input_tokens: int | None = None
        output_tokens: int | None = None

        for attempt in range(self.config.max_retries + 1):
            retries_used = attempt
            try:
                response = self.client.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image=image,
                    sampling_parameters=self.config.sampling_parameters,
                )
                raw_response, usage = _unwrap_client_response(response)
                input_tokens = usage.get("input_tokens")
                output_tokens = usage.get("output_tokens")
                parsed_response = _parse_json_response(raw_response)
                attempt_log.append(
                    {
                        "attempt": attempt + 1,
                        "raw_response": _json_safe_raw(raw_response),
                        "parsed": True,
                    }
                )
                generated, errors, rejected_payloads = self._materialize_response(
                    parsed_response,
                    graph=graph,
                    node=node,
                    class_names=classes,
                    image_width=image_width,
                    image_height=image_height,
                    id_source=id_source,
                )
                validation_errors.extend(errors)
                if len(generated) >= self.config.min_actions:
                    break
                validation_errors.append(
                    f"Qwen produced {len(generated)} valid actions; "
                    f"minimum is {self.config.min_actions}"
                )
            except Exception as exc:  # Preserve all failures for provenance.
                message = f"attempt {attempt + 1}: {type(exc).__name__}: {exc}"
                validation_errors.append(message)
                attempt_log.append(
                    {
                        "attempt": attempt + 1,
                        "raw_response": _json_safe_raw(raw_response),
                        "parsed": False,
                        "error": message,
                    }
                )
                generated = ()

        request_payload["attempts"] = attempt_log
        fallback_used = len(generated) < self.config.min_actions
        if fallback_used:
            generated = self.fallback_bank.materialize(
                graph=graph,
                node=node,
                class_names=classes,
                image_width=image_width,
                image_height=image_height,
                id_source=id_source,
            )

        reasoning_summary = None
        if isinstance(parsed_response, Mapping):
            value = parsed_response.get("reasoning_summary")
            if isinstance(value, str) and value.strip():
                reasoning_summary = value.strip()

        qwen_call_id = id_source.new_id(EntityKind.QWEN_CALL)
        qwen_call = QwenCallRecord(
            qwen_call_id=qwen_call_id,
            pass_id=pass_id,
            model_id=self.config.model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_artifact_ids=self.config.image_artifact_ids,
            request_payload=request_payload,
            raw_response=_json_safe_raw(raw_response),
            parsed_response=dict(parsed_response) if isinstance(parsed_response, Mapping) else None,
            reasoning_summary=reasoning_summary,
            validation_errors=tuple(validation_errors),
            retries=retries_used,
            fallback_used=fallback_used,
            latency_seconds=float(time.perf_counter() - started),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            sampling_parameters=dict(self.config.sampling_parameters),
        )
        candidate_set = CandidateActionSetRecord(
            candidate_set_id=id_source.new_id(EntityKind.CANDIDATE),
            pass_id=pass_id,
            target_node_id=node.id,
            actions=tuple(value.record for value in generated),
            source_qwen_call_id=qwen_call_id,
            generation_policy=(
                "qwen_candidate_generator_fallback"
                if fallback_used
                else "qwen_candidate_generator"
            ),
            validation_errors=tuple(validation_errors),
            rejected_action_payloads=tuple(dict(value) for value in rejected_payloads),
            fallback_used=fallback_used,
            metadata={
                "valid_action_count": len(generated),
                "requested_max_actions": self.config.max_actions,
            },
        )
        return QwenGenerationResult(
            actions=generated,
            candidate_set=candidate_set,
            qwen_call=qwen_call,
        )

    def _build_user_prompt(
        self,
        *,
        graph,
        node,
        class_names: tuple[str, ...],
        image_width: int,
        image_height: int,
    ) -> str:
        trusted_nodes = []
        for candidate in sorted(graph.nodes.values(), key=lambda value: value.id):
            if candidate.id == node.id or candidate.belief is None:
                continue
            trusted_nodes.append(
                {
                    "node_id": candidate.id,
                    "box": [float(value) for value in candidate.box],
                    "posterior": candidate.belief.as_mapping(),
                    "status": candidate.belief.status,
                }
            )
        payload = {
            "task": "Generate diverse candidate SAM3 sensing actions for one unresolved node.",
            "image": {"width": image_width, "height": image_height},
            "classes": list(class_names),
            "target_node": {
                "node_id": node.id,
                "box": [float(value) for value in node.box],
                "posterior": node.belief.as_mapping(),
                "query_count": node.belief.query_count,
                "used_semantic_keys": sorted(node.used_semantic_keys),
            },
            "trusted_context_nodes": trusted_nodes,
            "requirements": {
                "max_actions": self.config.max_actions,
                "families": ["target", "negative", "explore"],
                "beta_classes": list(class_names),
                "region_coordinates": "global xyxy pixels",
                "tiling_modes": (
                    ["off", "always", "agent_controlled"]
                    if self.config.allow_tiling
                    else ["off"]
                ),
                "return": self._response_schema(class_names),
            },
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    def _response_schema(self, class_names: Sequence[str]) -> Mapping[str, Any]:
        return {
            "reasoning_summary": "short explicit summary",
            "actions": [
                {
                    "family": "target|negative|explore",
                    "prompt": "SAM3 concept prompt",
                    "semantic_key": "canonical semantic coordinate",
                    "beta_by_class": {name: "probability in [0,1]" for name in class_names},
                    "threshold": "optional probability",
                    "region": "optional [x1,y1,x2,y2] global pixels",
                    "positive_exemplar_node_ids": [],
                    "negative_exemplar_node_ids": [],
                    "tiling_mode": "off|always|agent_controlled",
                    "tile_scale": "optional positive number",
                    "expected_cost": "optional positive number",
                    "rationale": "short explicit reason",
                    "expected_visual_distinction": "what the result should distinguish",
                }
            ],
        }

    def _materialize_response(
        self,
        response: Mapping[str, Any],
        *,
        graph,
        node,
        class_names: tuple[str, ...],
        image_width: int,
        image_height: int,
        id_source,
    ) -> tuple[
        tuple[MaterializedStaticAction, ...],
        list[str],
        list[Mapping[str, Any]],
    ]:
        payloads = response.get("actions")
        if not isinstance(payloads, list):
            raise ContractError("Qwen response must contain an actions list")
        errors: list[str] = []
        rejected: list[Mapping[str, Any]] = []
        actions: list[MaterializedStaticAction] = []
        semantic_keys: set[str] = set()
        known_nodes = set(graph.nodes)

        for index, value in enumerate(payloads[: self.config.max_actions]):
            if not isinstance(value, Mapping):
                errors.append(f"actions[{index}] must be an object")
                rejected.append({"raw": value})
                continue
            payload = dict(value)
            try:
                prompt = _required_string(payload, "prompt")
                if len(prompt) > self.config.max_prompt_chars:
                    raise ContractError("prompt exceeds max_prompt_chars")
                semantic_key = _required_string(payload, "semantic_key")
                if semantic_key in semantic_keys:
                    raise ContractError("duplicate semantic_key in generated set")
                if semantic_key in node.used_semantic_keys:
                    raise ContractError("semantic_key was already queried for this node")
                family = ActionFamily(_required_string(payload, "family"))
                beta = payload.get("beta_by_class")
                if not isinstance(beta, Mapping) or set(beta) != set(class_names):
                    raise ContractError("beta_by_class keys must exactly match class_names")
                beta_by_class = {
                    name: _probability(beta[name], f"beta_by_class[{name!r}]")
                    for name in class_names
                }
                threshold = _probability(
                    payload.get("threshold", self.config.default_threshold),
                    "threshold",
                )
                region = self._resolve_region(
                    payload.get("region"),
                    node.box,
                    image_width=image_width,
                    image_height=image_height,
                )
                positive_ids = _node_ids(
                    payload.get("positive_exemplar_node_ids", []),
                    known_nodes,
                    "positive_exemplar_node_ids",
                )
                negative_ids = _node_ids(
                    payload.get("negative_exemplar_node_ids", []),
                    known_nodes,
                    "negative_exemplar_node_ids",
                )
                for exemplar_id in positive_ids:
                    if not graph.nodes[exemplar_id].exemplar_eligible(
                        self.fallback_bank.positive_class,
                        threshold=self.fallback_bank.positive_exemplar_threshold,
                    ):
                        raise ContractError(
                            f"positive exemplar {exemplar_id!r} is not trusted"
                        )
                for exemplar_id in negative_ids:
                    if not graph.nodes[exemplar_id].exemplar_eligible(
                        self.fallback_bank.negative_class,
                        threshold=self.fallback_bank.negative_exemplar_threshold,
                    ):
                        raise ContractError(
                            f"negative exemplar {exemplar_id!r} is not trusted"
                        )
                tiling_mode = TilingMode(str(payload.get("tiling_mode", "off")))
                if not self.config.allow_tiling and tiling_mode is not TilingMode.OFF:
                    raise ContractError("tiling is disabled for Qwen-generated actions")
                if tiling_mode is TilingMode.DENSITY_ADAPTIVE:
                    raise ContractError("verification actions may not request density_adaptive")
                tile_scale_value = payload.get("tile_scale")
                tile_scale = None
                if tile_scale_value is not None:
                    tile_scale = float(tile_scale_value)
                    if not math.isfinite(tile_scale) or tile_scale <= 0:
                        raise ContractError("tile_scale must be finite and positive")
                expected_cost = float(payload.get("expected_cost", 1.0))
                if not math.isfinite(expected_cost) or expected_cost <= 0:
                    raise ContractError("expected_cost must be finite and positive")
                action = SensingActionRecord(
                    action_id=id_source.new_id(EntityKind.ACTION),
                    action_kind=ActionKind.VERIFY,
                    family=family,
                    prompt=prompt,
                    region=region,
                    threshold=threshold,
                    semantic_key=semantic_key,
                    target_node_id=node.id,
                    positive_exemplar_node_ids=positive_ids,
                    negative_exemplar_node_ids=negative_ids,
                    tiling_mode=tiling_mode,
                    tile_scale=tile_scale,
                    beta_by_class=beta_by_class,
                    rationale=_optional_string(payload.get("rationale")),
                    expected_visual_distinction=_optional_string(
                        payload.get("expected_visual_distinction")
                    ),
                    generator="qwen_candidate_generator",
                    metadata={
                        "expected_cost": expected_cost,
                        "source_action_index": index,
                    },
                )
                actions.append(
                    MaterializedStaticAction(
                        record=action,
                        kernel=build_surrogate_kernel(
                            class_names,
                            beta_by_class,
                            self.profile,
                        ),
                        expected_cost=expected_cost,
                    )
                )
                semantic_keys.add(semantic_key)
            except Exception as exc:
                errors.append(f"actions[{index}]: {type(exc).__name__}: {exc}")
                rejected.append(payload)
        return tuple(actions), errors, rejected

    def _resolve_region(
        self,
        supplied: Any,
        node_box: Sequence[float],
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float]:
        if supplied is None or not self.config.allow_qwen_regions:
            return _padded_region(
                node_box,
                image_width=image_width,
                image_height=image_height,
                padding_fraction=self.config.default_roi_padding_fraction,
            )
        if not isinstance(supplied, (list, tuple)) or len(supplied) != 4:
            raise ContractError("region must contain four coordinates")
        values = tuple(float(value) for value in supplied)
        if not all(math.isfinite(value) for value in values):
            raise ContractError("region coordinates must be finite")
        x1, y1, x2, y2 = values
        x1 = max(0.0, min(x1, float(image_width)))
        y1 = max(0.0, min(y1, float(image_height)))
        x2 = max(0.0, min(x2, float(image_width)))
        y2 = max(0.0, min(y2, float(image_height)))
        if x2 <= x1 or y2 <= y1:
            raise ContractError("region becomes empty after clipping")
        nx1, ny1, nx2, ny2 = (float(value) for value in node_box)
        intersection = max(0.0, min(x2, nx2) - max(x1, nx1)) * max(
            0.0, min(y2, ny2) - max(y1, ny1)
        )
        if intersection <= 0.0:
            raise ContractError("verification region must intersect the target node")
        return (x1, y1, x2, y2)


def _unwrap_client_response(response: Any) -> tuple[Any, dict[str, int | None]]:
    if isinstance(response, Mapping):
        raw = response.get("content", response.get("raw_response", response))
        usage = response.get("usage", {})
        if not isinstance(usage, Mapping):
            usage = {}
        return raw, {
            "input_tokens": _optional_int(usage.get("input_tokens", usage.get("prompt_tokens"))),
            "output_tokens": _optional_int(usage.get("output_tokens", usage.get("completion_tokens"))),
        }
    content = getattr(response, "content", response)
    usage_obj = getattr(response, "usage", None)
    usage = usage_obj if isinstance(usage_obj, Mapping) else {}
    return content, {
        "input_tokens": _optional_int(usage.get("input_tokens", usage.get("prompt_tokens"))),
        "output_tokens": _optional_int(usage.get("output_tokens", usage.get("completion_tokens"))),
    }


def _parse_json_response(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        raise ContractError("Qwen response must be JSON text or an object")
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ContractError("Qwen JSON response must be an object")
    return dict(payload)


def _json_safe_raw(raw: Any) -> Mapping[str, Any] | str | None:
    if raw is None or isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        return dict(raw)
    return str(raw)


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractError("optional text fields must be strings")
    value = value.strip()
    return value or None


def _probability(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ContractError(f"{name} must lie in [0, 1]")
    return result


def _node_ids(value: Any, known: set[str], name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ContractError(f"{name} must be a list")
    ids = tuple(str(item) for item in value)
    unknown = [item for item in ids if item not in known]
    if unknown:
        raise ContractError(f"{name} contains unknown nodes: {unknown}")
    if len(ids) != len(set(ids)):
        raise ContractError(f"{name} contains duplicates")
    return ids


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    return result if result >= 0 else None


def _padded_region(
    box: Sequence[float],
    *,
    image_width: int,
    image_height: int,
    padding_fraction: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return (
        max(0.0, x1 - width * padding_fraction),
        max(0.0, y1 - height * padding_fraction),
        min(float(image_width), x2 + width * padding_fraction),
        min(float(image_height), y2 + height * padding_fraction),
    )


__all__ = [
    "QwenActionClient",
    "QwenActionGeneratorConfig",
    "QwenCandidateActionGenerator",
    "QwenGenerationResult",
]
