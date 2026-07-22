"""
project_dashboard_factory.py

Connects the experiment dashboard to:

1. Real SAM3 model
2. Real Qwen endpoint
3. Local datasets
4. Experiment presets
5. Output directory

Start the dashboard with:

python experiment_dashboard.py \
    --factory project_dashboard_factory:create_dependencies \
    --host 127.0.0.1 \
    --port 7860
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from openai import OpenAI
from PIL import Image

import inference

from agent.asht import (
    AdaptiveTilingConfig,
    QwenActionGeneratorConfig,
    QwenCandidateActionGenerator,
    SensorProfile,
    StaticActionBank,
    StaticActionTemplate,
    StaticAshtConfig,
)
from dashboard.service import (
    DashboardDependencies,
    DashboardRunRequest,
)
from eval.dataset_adapters import (
    CanonicalSample,
    CarpkAdapter,
    DatasetRegistry,
    FourDSemanticMappingAdapter,
    Fscd147Adapter,
    GenericFolderAdapter,
    GreenCitrusAdapter,
    OmniCountAdapter,
)
from experiments.config import (
    DiscoveryPassSpec,
    ExperimentMode,
    UnifiedExperimentConfig,
)
from experiments.runner import UnifiedRuntime
from pipeline_stages import Sam3QuerySpec
from provenance.run_store import ReportingLevel
from provenance.schema import ActionFamily, TilingMode


# =============================================================================
# 1. CHANGE THESE PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Where experiment results will be saved.
OUTPUT_ROOT = Path(
    os.environ.get(
        "SAM3_VLM_OUTPUT_ROOT",
        PROJECT_ROOT / "out" / "experiments",
    )
).expanduser()


# ---------------------------------------------------------------------------
# SAM3
# ---------------------------------------------------------------------------

SAM3_REPO_ROOT = Path(
    os.environ.get(
        "SAM3_REPO_ROOT",
        "/home/ekielar/sam3",
    )
).expanduser()

SAM3_BPE_PATH = Path(
    os.environ.get(
        "SAM3_BPE_PATH",
        SAM3_REPO_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz",
    )
).expanduser()


# ---------------------------------------------------------------------------
# Datasets
#
# A dataset is registered only when its directory exists.
# You can therefore start with TEST_IMAGES_ROOT and add the others later.
# ---------------------------------------------------------------------------

TEST_IMAGES_ROOT = Path(
    os.environ.get(
        "TEST_IMAGES_ROOT",
        PROJECT_ROOT / "datasets/test_dataset",
    )
).expanduser()

GREEN_CITRUS_ROOT = Path(
    os.environ.get(
        "GREEN_CITRUS_ROOT",
        PROJECT_ROOT / "datasets/citruses_dataset",
    )
).expanduser()

CARPK_ROOT = Path(
    os.environ.get(
        "CARPK_ROOT",
        PROJECT_ROOT / "datasets/CARPK_devkit/data",
    )
).expanduser()

FSCD147_ROOT = Path(
    os.environ.get(
        "FSCD147_ROOT",
        "/path/to/FSC147",
    )
).expanduser()

OMNICOUNT_ROOT = Path(
    os.environ.get(
        "OMNICOUNT_ROOT",
        "/path/to/OmniCount",
    )
).expanduser()

OMNICOUNT_ANNOTATIONS = Path(
    os.environ.get(
        "OMNICOUNT_ANNOTATIONS",
        OMNICOUNT_ROOT / "annotations.json",
    )
).expanduser()

FOUR_D_ROOT = Path(
    os.environ.get(
        "FOUR_D_ROOT",
        "/path/to/4d_semantic_mapping",
    )
).expanduser()

FOUR_D_MANIFEST = os.environ.get("FOUR_D_MANIFEST")


# =============================================================================
# 2. QWEN CLIENT
# =============================================================================

class OpenAICompatibleQwenClient:
    """
    Small adapter between the new ASHT action generator and your
    OpenAI-compatible Qwen endpoint.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str,
    ) -> None:
        if not base_url.strip():
            raise ValueError("QWEN_BASE_URL is empty")

        if not model_name.strip():
            raise ValueError("QWEN_MODEL is empty")

        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image: Any,
        sampling_parameters: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """
        Return the format expected by QwenCandidateActionGenerator:

        {
            "content": "...JSON...",
            "usage": {
                "input_tokens": ...,
                "output_tokens": ...
            }
        }
        """

        image_pil = self._to_pil_image(image)
        image_url = self._to_data_url(image_pil)

        parameters = dict(sampling_parameters)
        temperature = float(parameters.pop("temperature", 0.0))

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                },
            ],
            **parameters,
        )

        content = response.choices[0].message.content

        usage = getattr(response, "usage", None)

        input_tokens = (
            getattr(usage, "prompt_tokens", None)
            if usage is not None
            else None
        )

        output_tokens = (
            getattr(usage, "completion_tokens", None)
            if usage is not None
            else None
        )

        return {
            "content": content,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }

    @staticmethod
    def _to_pil_image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        # Supports numpy arrays.
        return Image.fromarray(image).convert("RGB")

    @staticmethod
    def _to_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        return f"data:image/png;base64,{encoded}"


# =============================================================================
# 3. STATIC ACTION BANK
# =============================================================================

def build_static_action_bank(
    *,
    class_names: Sequence[str],
    target_prompt: str,
) -> StaticActionBank:
    """
    Build a small generic action bank.

    These probabilities are only initial test assumptions.
    They are not calibrated experimental values.
    """

    classes = tuple(class_names)

    if len(classes) != 3:
        raise ValueError(
            "This starter configuration expects exactly three classes"
        )

    target_class = classes[0]
    confounder_class = classes[1]
    background_class = classes[2]

    sensor_profile = SensorProfile(
        observation_labels=(
            "not_found",
            "weak_match",
            "strong_match",
        ),
        present=(
            0.05,
            0.20,
            0.75,
        ),
        absent=(
            0.75,
            0.20,
            0.05,
        ),
        source="project-dashboard-initial-profile",
    )

    return StaticActionBank(
        templates=(
            StaticActionTemplate(
                name="target_direct",
                family=ActionFamily.TARGET,
                prompt=target_prompt,
                semantic_key=f"target:{target_prompt.lower()}",
                beta_by_class={
                    target_class: 0.90,
                    confounder_class: 0.15,
                    background_class: 0.05,
                },
                threshold=0.50,
                use_positive_exemplars=True,
                rationale="Directly test the requested target concept.",
            ),
            StaticActionTemplate(
                name="target_precise",
                family=ActionFamily.TARGET,
                prompt=f"clear {target_prompt}",
                semantic_key=f"clear:{target_prompt.lower()}",
                beta_by_class={
                    target_class: 0.85,
                    confounder_class: 0.10,
                    background_class: 0.03,
                },
                threshold=0.55,
                use_positive_exemplars=True,
                rationale="Use a stricter target description.",
            ),
            StaticActionTemplate(
                name="confounder",
                family=ActionFamily.NEGATIVE,
                prompt=confounder_class.replace("_", " "),
                semantic_key=f"confounder:{confounder_class}",
                beta_by_class={
                    target_class: 0.08,
                    confounder_class: 0.88,
                    background_class: 0.10,
                },
                threshold=0.50,
                use_negative_exemplars=True,
                rationale="Test the strongest alternative class.",
            ),
            StaticActionTemplate(
                name="background",
                family=ActionFamily.NEGATIVE,
                prompt="background clutter",
                semantic_key="background:clutter",
                beta_by_class={
                    target_class: 0.03,
                    confounder_class: 0.12,
                    background_class: 0.85,
                },
                threshold=0.50,
                rationale="Test whether the candidate is background clutter.",
            ),
        ),
        profile=sensor_profile,
        positive_class=target_class,
        negative_class=confounder_class,
    )


# =============================================================================
# 4. DATASET-AWARE QWEN GENERATOR
# =============================================================================

class DatasetAwareQwenGenerator:
    """
    Creates a matching fallback action bank for the classes used by the
    current dataset.

    This avoids using fruit/leaf fallback actions while testing CARPK,
    FSCD-147, or another dataset.
    """

    def __init__(
        self,
        *,
        client: OpenAICompatibleQwenClient,
    ) -> None:
        self.client = client

        self.generator_config = QwenActionGeneratorConfig(
            model_id=os.environ.get("QWEN_MODEL", "qwen"),
            min_actions=2,
            max_actions=8,
            max_retries=1,
            allow_qwen_regions=True,
            allow_tiling=True,
            sampling_parameters={
                "temperature": 0.0,
            },
        )

    def generate(self, **kwargs: Any) -> Any:
        class_names = tuple(kwargs["class_names"])

        fallback_bank = build_static_action_bank(
            class_names=class_names,
            target_prompt=class_names[0].replace("_", " "),
        )

        generator = QwenCandidateActionGenerator(
            client=self.client,
            profile=fallback_bank.profile,
            fallback_bank=fallback_bank,
            config=self.generator_config,
        )

        return generator.generate(**kwargs)


# =============================================================================
# 5. DATASET REGISTRY
# =============================================================================

def build_dataset_registry() -> DatasetRegistry:
    adapters = []

    # -----------------------------------------------------------------------
    # Easiest first test:
    #
    # Put one or more images in:
    #     test_images/
    #
    # Optional sidecar:
    #     image1.jpg
    #     image1.json
    #
    # Example image1.json:
    #
    # {
    #   "target_concept": "green citrus fruit",
    #   "target_class": "fruit",
    #   "count": 4,
    #   "boxes": [
    #       [10, 20, 50, 70],
    #       [80, 30, 120, 75]
    #   ]
    # }
    # -----------------------------------------------------------------------

    if TEST_IMAGES_ROOT.is_dir():
        adapters.append(
            GenericFolderAdapter(
                TEST_IMAGES_ROOT,
                name="test_images",
                target_concept="green citrus fruit",
                target_class="fruit",
            )
        )

    if GREEN_CITRUS_ROOT.is_dir():
        adapters.append(
            GreenCitrusAdapter(
                GREEN_CITRUS_ROOT,
            )
        )

    if CARPK_ROOT.is_dir():
        adapters.append(
            CarpkAdapter(
                CARPK_ROOT,
            )
        )

    if FSCD147_ROOT.is_dir():
        adapters.append(
            Fscd147Adapter(
                FSCD147_ROOT,
            )
        )

    if (
        OMNICOUNT_ROOT.is_dir()
        and OMNICOUNT_ANNOTATIONS.is_file()
    ):
        adapters.append(
            OmniCountAdapter(
                root=OMNICOUNT_ROOT,
                annotation_file=OMNICOUNT_ANNOTATIONS,
                image_directory="images",

                # Initially restrict this to categories you have checked.
                # Example:
                # category_names=("apple", "orange", "banana"),
                category_names=None,
            )
        )

    if FOUR_D_ROOT.is_dir():
        adapters.append(
            FourDSemanticMappingAdapter(
                FOUR_D_ROOT,
                manifest=FOUR_D_MANIFEST,
                target_concept="fruit",
                target_class="fruit",
            )
        )

    if not adapters:
        raise RuntimeError(
            "No dataset directories were found.\n"
            "Create test_images/ or set one of the dataset environment "
            "variables."
        )

    return DatasetRegistry(adapters)


# =============================================================================
# 6. SAM3 AND QWEN RUNTIME
# =============================================================================

def build_runtime() -> UnifiedRuntime:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Loading SAM3 on {device}...")

    _model, processor = inference.load_sam3_model(
        str(SAM3_BPE_PATH),
        confidence=0.35,
        device=device,
    )

    qwen_generator = None

    qwen_base_url = os.environ.get("QWEN_BASE_URL", "").strip()
    qwen_model = os.environ.get("QWEN_MODEL", "").strip()
    qwen_api_key = os.environ.get("QWEN_API_KEY", "EMPTY")

    if qwen_base_url and qwen_model:
        qwen_client = OpenAICompatibleQwenClient(
            base_url=qwen_base_url,
            model_name=qwen_model,
            api_key=qwen_api_key,
        )

        qwen_generator = DatasetAwareQwenGenerator(
            client=qwen_client,
        )

        print(f"Qwen enabled: {qwen_model}")
    else:
        print(
            "Qwen is not configured. "
            "SAM3 and static-ASHT modes will still work."
        )

    return UnifiedRuntime(
        # The existing inference.py module supplies:
        # - run_raw_inference
        # - apply_nms
        # - apply_nms_dualgate
        backend=inference,

        # load_sam3_model attaches the model to processor.model.
        processor=processor,

        qwen_action_generator=qwen_generator,

        # Add your old legacy VLM runner here later if required.
        legacy_executor=None,
    )


# =============================================================================
# 7. DATASET-SPECIFIC CLASS SCHEMA
# =============================================================================

def resolve_classes(
    sample: CanonicalSample,
) -> tuple[tuple[str, str, str], str]:
    """
    Return:

        (class_names, target_class)

    The first class is always the counted target.
    """

    dataset_name = sample.dataset_name.lower()

    if dataset_name == "green_citrus":
        return (
            ("fruit", "leaf", "background"),
            "fruit",
        )

    if dataset_name == "carpk":
        return (
            ("car", "non_car", "background"),
            "car",
        )

    target_class = sample.target_class.strip() or "target"

    confounder_class = next(
        (
            item
            for item in sample.confounder_classes
            if item not in {target_class, "background"}
        ),
        "non_target",
    )

    if confounder_class == target_class:
        confounder_class = "non_target"

    return (
        (
            target_class,
            confounder_class,
            "background",
        ),
        target_class,
    )


# =============================================================================
# 8. DATASET-SPECIFIC DEFAULTS
# =============================================================================

def default_detection_threshold(
    sample: CanonicalSample,
) -> float:
    if sample.dataset_name.lower() == "carpk":
        return 0.45

    return 0.35


def default_deduplication(
    sample: CanonicalSample,
) -> tuple[str, float]:
    """
    Return:

        metric, threshold
    """

    if sample.dataset_name.lower() == "carpk":
        return "iom", 0.85

    return "iou", 0.40


# =============================================================================
# 9. CONFIGURATION FACTORY
# =============================================================================

def build_experiment_config(
    request: DashboardRunRequest,
    sample: CanonicalSample,
) -> UnifiedExperimentConfig:
    """
    Convert the dashboard choices into one immutable experiment config.
    """

    class_names, target_class = resolve_classes(sample)

    threshold = default_detection_threshold(sample)

    dedup_metric, dedup_threshold = default_deduplication(sample)

    reporting_level = ReportingLevel(
        request.reporting_level or "full"
    )

    requested_tiling_mode = (
        TilingMode(request.tiling_mode)
        if request.tiling_mode
        else None
    )

    max_passes = request.max_passes or 3
    max_sam3_calls = request.max_sam3_calls or 12

    output_root = OUTPUT_ROOT

    target_prompt = sample.target_concept

    adaptive_tiling = AdaptiveTilingConfig(
        max_tiles_per_action=32,
    )

    discovery_passes: tuple[DiscoveryPassSpec, ...] = ()
    asht_config: StaticAshtConfig | None = None

    # -----------------------------------------------------------------------
    # SAM3 single-pass
    # -----------------------------------------------------------------------

    if request.mode is ExperimentMode.SAM3_SINGLE_PASS:
        discovery_passes = (
            DiscoveryPassSpec(
                prompt=target_prompt,
                threshold=threshold,
                tiling_mode=TilingMode.OFF,
                return_masks=True,
            ),
        )

    # -----------------------------------------------------------------------
    # Fixed multipass
    # -----------------------------------------------------------------------

    elif request.mode is ExperimentMode.SAM3_FIXED_MULTIPASS:
        candidate_prompts = [
            target_prompt,
            f"clear {target_prompt}",
            f"small {target_prompt}",
        ]

        discovery_passes = tuple(
            DiscoveryPassSpec(
                prompt=candidate_prompts[index % len(candidate_prompts)],
                threshold=min(
                    0.85,
                    threshold + 0.05 * index,
                ),
                tiling_mode=TilingMode.OFF,
                return_masks=True,
            )
            for index in range(max_passes)
        )

    # -----------------------------------------------------------------------
    # SAM3 adaptive tiling
    # -----------------------------------------------------------------------

    elif request.mode is ExperimentMode.SAM3_ADAPTIVE_TILING:
        tiling_mode = (
            requested_tiling_mode
            if requested_tiling_mode not in {None, TilingMode.OFF}
            else TilingMode.DENSITY_ADAPTIVE
        )

        discovery_passes = (
            DiscoveryPassSpec(
                prompt=target_prompt,
                threshold=threshold,
                tiling_mode=tiling_mode,
                return_masks=True,
            ),
        )

    # -----------------------------------------------------------------------
    # ASHT modes
    # -----------------------------------------------------------------------

    elif request.mode in {
        ExperimentMode.ASHT_STATIC,
        ExperimentMode.ASHT_QWEN,
        ExperimentMode.ASHT_ADAPTIVE_TILING,
        ExperimentMode.ASHT_AGENT_TILING,
    }:
        width, height = sample.image_size

        action_bank = build_static_action_bank(
            class_names=class_names,
            target_prompt=target_prompt,
        )

        if request.mode is ExperimentMode.ASHT_ADAPTIVE_TILING:
            bootstrap_tiling_mode = TilingMode.DENSITY_ADAPTIVE
        elif request.mode is ExperimentMode.ASHT_AGENT_TILING:
            bootstrap_tiling_mode = TilingMode.OFF
        else:
            bootstrap_tiling_mode = (
                requested_tiling_mode or TilingMode.OFF
            )

        stopping_threshold = (
            request.stopping_confidence
            if request.stopping_confidence is not None
            else 0.90
        )

        # Use the smallest dashboard budget as the effective limit.
        possible_limits = [
            value
            for value in (
                request.max_passes,
                request.max_sam3_calls,
                request.max_qwen_calls,
            )
            if value is not None and value > 0
        ]

        max_total_queries = (
            min(possible_limits)
            if possible_limits
            else max_sam3_calls
        )

        uniform_prior = 1.0 / len(class_names)

        asht_config = StaticAshtConfig(
            class_names=class_names,
            target_class=target_class,
            prior={
                class_name: uniform_prior
                for class_name in class_names
            },
            action_bank=action_bank,
            stopping_threshold=stopping_threshold,
            max_total_queries=max_total_queries,
            max_queries_per_node=min(
                4,
                max_total_queries,
            ),
            bootstrap_query=Sam3QuerySpec(
                prompt=target_prompt,
                threshold=threshold,
                region=(
                    0.0,
                    0.0,
                    float(width),
                    float(height),
                ),
                return_masks=True,
            ),
            bootstrap_dedup_metric=dedup_metric,
            bootstrap_dedup_threshold=dedup_threshold,
            bootstrap_tiling_mode=bootstrap_tiling_mode,
            adaptive_tiling=adaptive_tiling,
        )

    # -----------------------------------------------------------------------
    # Legacy mode
    # -----------------------------------------------------------------------

    elif request.mode is ExperimentMode.LEGACY_VLM:
        raise RuntimeError(
            "LEGACY_VLM is not connected in this starter factory. "
            "Connect your old episode runner as UnifiedRuntime.legacy_executor."
        )

    else:
        raise ValueError(
            f"Unsupported experiment mode: {request.mode}"
        )

    resolved_overrides = {
        "dashboard": {
            "dataset": request.dataset_name,
            "split": request.split,
            "sample_index": request.sample_index,
            "requested_max_passes": request.max_passes,
            "requested_max_sam3_calls": request.max_sam3_calls,
            "requested_max_qwen_calls": request.max_qwen_calls,
            "requested_tiling_mode": request.tiling_mode,
            "expert_overrides": dict(request.expert_overrides),
        }
    }

    return UnifiedExperimentConfig(
        mode=request.mode,
        output_root=output_root,
        reporting_level=reporting_level,
        random_seed=0,
        overwrite=False,
        raise_on_error=True,
        class_names=class_names,
        target_class=target_class,
        discovery_passes=discovery_passes,
        asht_config=asht_config,
        adaptive_tiling=adaptive_tiling,
        suppression_confidence=threshold,
        cross_pass_dedup_metric=dedup_metric,
        cross_pass_dedup_threshold=dedup_threshold,
        model_id="facebook/sam3",
        run_name=request.run_name,
        resolved_overrides=resolved_overrides,
    )


# =============================================================================
# 10. DASHBOARD ENTRY POINT
# =============================================================================

def create_dependencies() -> DashboardDependencies:
    """
    This is the function passed to:

        --factory project_dashboard_factory:create_dependencies
    """

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    datasets = build_dataset_registry()
    runtime = build_runtime()

    print("Registered datasets:")
    for name in datasets.names():
        adapter = datasets.adapter(name)
        print(
            f"  - {name}: "
            f"splits={adapter.splits()}"
        )

    print(f"Experiment output: {OUTPUT_ROOT}")

    return DashboardDependencies(
        datasets=datasets,
        runtime=runtime,
        config_factory=build_experiment_config,
        output_root=OUTPUT_ROOT,
    )