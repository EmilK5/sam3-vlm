"""Static sensing-action banks for the first end-to-end ASHT runner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from agent.asht.kernels import SensorProfile, SurrogateKernel, build_surrogate_kernel
from provenance.contracts import SensingActionRecord
from provenance.ids import EntityKind
from provenance.schema import ActionFamily, ActionKind, TilingMode


@dataclass(frozen=True)
class StaticActionTemplate:
    """Dataset-configurable action before it is bound to one graph node."""

    name: str
    family: ActionFamily
    prompt: str
    semantic_key: str
    beta_by_class: Mapping[str, float]
    threshold: float = 0.50
    roi_padding_fraction: float = 0.25
    expected_cost: float = 1.0
    tiling_mode: TilingMode = TilingMode.OFF
    tile_scale: float | None = None
    use_positive_exemplars: bool = False
    use_negative_exemplars: bool = False
    rationale: str | None = None
    expected_visual_distinction: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.prompt.strip() or not self.semantic_key.strip():
            raise ValueError("action name, prompt, and semantic_key must be non-empty")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        if not math.isfinite(self.roi_padding_fraction) or self.roi_padding_fraction < 0.0:
            raise ValueError("roi_padding_fraction must be finite and non-negative")
        if not math.isfinite(self.expected_cost) or self.expected_cost <= 0.0:
            raise ValueError("expected_cost must be finite and positive")
        if self.tile_scale is not None and (
            not math.isfinite(self.tile_scale) or self.tile_scale <= 0.0
        ):
            raise ValueError("tile_scale must be finite and positive")


@dataclass(frozen=True)
class MaterializedStaticAction:
    record: SensingActionRecord
    kernel: SurrogateKernel
    expected_cost: float


@dataclass(frozen=True)
class StaticActionBank:
    templates: tuple[StaticActionTemplate, ...]
    profile: SensorProfile
    allow_semantic_reuse: bool = False
    positive_class: str = "fruit"
    negative_class: str = "leaf"
    positive_exemplar_threshold: float = 0.90
    negative_exemplar_threshold: float = 0.90

    def __post_init__(self) -> None:
        if not self.templates:
            raise ValueError("a static action bank must contain at least one template")
        keys = [template.semantic_key for template in self.templates]
        if len(keys) != len(set(keys)):
            raise ValueError("static action semantic keys must be unique")

    def materialize(
        self,
        *,
        graph,
        node,
        class_names: Sequence[str],
        image_width: int,
        image_height: int,
        id_source,
    ) -> tuple[MaterializedStaticAction, ...]:
        classes = tuple(str(value) for value in class_names)
        positive_ids = tuple(
            candidate.id
            for candidate in sorted(graph.nodes.values(), key=lambda value: value.id)
            if candidate.id != node.id
            and candidate.exemplar_eligible(
                self.positive_class,
                threshold=self.positive_exemplar_threshold,
            )
        )
        negative_ids = tuple(
            candidate.id
            for candidate in sorted(graph.nodes.values(), key=lambda value: value.id)
            if candidate.id != node.id
            and candidate.exemplar_eligible(
                self.negative_class,
                threshold=self.negative_exemplar_threshold,
            )
        )

        materialized = []
        for template in self.templates:
            if not self.allow_semantic_reuse and template.semantic_key in node.used_semantic_keys:
                continue
            if set(template.beta_by_class) != set(classes):
                raise ValueError(
                    f"action {template.name!r} beta classes do not match {classes!r}"
                )
            action_id = id_source.new_id(EntityKind.ACTION)
            region = _padded_region(
                node.box,
                image_width=image_width,
                image_height=image_height,
                padding_fraction=template.roi_padding_fraction,
            )
            action = SensingActionRecord(
                action_id=action_id,
                action_kind=ActionKind.VERIFY,
                family=template.family,
                prompt=template.prompt,
                region=region,
                threshold=float(template.threshold),
                semantic_key=template.semantic_key,
                target_node_id=node.id,
                positive_exemplar_node_ids=(
                    positive_ids if template.use_positive_exemplars else ()
                ),
                negative_exemplar_node_ids=(
                    negative_ids if template.use_negative_exemplars else ()
                ),
                tiling_mode=template.tiling_mode,
                tile_scale=template.tile_scale,
                beta_by_class={
                    name: float(template.beta_by_class[name]) for name in classes
                },
                rationale=template.rationale,
                expected_visual_distinction=template.expected_visual_distinction,
                generator="static_action_bank",
                metadata={
                    "template_name": template.name,
                    "expected_cost": float(template.expected_cost),
                    "roi_padding_fraction": float(template.roi_padding_fraction),
                },
            )
            materialized.append(
                MaterializedStaticAction(
                    record=action,
                    kernel=build_surrogate_kernel(
                        classes,
                        action.beta_by_class,
                        self.profile,
                    ),
                    expected_cost=float(template.expected_cost),
                )
            )
        return tuple(materialized)


def green_citrus_static_action_bank() -> StaticActionBank:
    """Small initial bank used to validate the controller, not a final policy."""

    profile = SensorProfile(
        observation_labels=("not_found", "weak_match", "strong_match"),
        present=(0.05, 0.20, 0.75),
        absent=(0.75, 0.20, 0.05),
        source="configured-static-v1",
    )
    return StaticActionBank(
        templates=(
            StaticActionTemplate(
                name="target_round_green_fruit",
                family=ActionFamily.TARGET,
                prompt="round green citrus fruit",
                semantic_key="round+green+citrus",
                beta_by_class={"fruit": 0.92, "leaf": 0.12, "background": 0.03},
                use_positive_exemplars=True,
                rationale="Confirm fruit-like round green appearance.",
            ),
            StaticActionTemplate(
                name="negative_flat_veined_leaf",
                family=ActionFamily.NEGATIVE,
                prompt="flat veined green leaf",
                semantic_key="flat+veined+leaf",
                beta_by_class={"fruit": 0.04, "leaf": 0.90, "background": 0.08},
                use_negative_exemplars=True,
                rationale="Test the strongest foliage alternative.",
            ),
            StaticActionTemplate(
                name="target_glossy_spherical_fruit",
                family=ActionFamily.TARGET,
                prompt="glossy spherical green fruit",
                semantic_key="glossy+spherical+fruit",
                beta_by_class={"fruit": 0.86, "leaf": 0.09, "background": 0.03},
                use_positive_exemplars=True,
                rationale="Use a distinct target conjunction rather than a paraphrase.",
            ),
            StaticActionTemplate(
                name="background_clutter",
                family=ActionFamily.NEGATIVE,
                prompt="background clutter or branch texture",
                semantic_key="background+branch+clutter",
                beta_by_class={"fruit": 0.04, "leaf": 0.18, "background": 0.82},
                rationale="Separate object candidates from clutter.",
            ),
        ),
        profile=profile,
    )


def _padded_region(
    box,
    *,
    image_width: int,
    image_height: int,
    padding_fraction: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = width * padding_fraction
    pad_y = height * padding_fraction
    return (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(image_width), x2 + pad_x),
        min(float(image_height), y2 + pad_y),
    )


__all__ = [
    "MaterializedStaticAction",
    "StaticActionBank",
    "StaticActionTemplate",
    "green_citrus_static_action_bank",
]
