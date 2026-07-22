"""Active sequential hypothesis testing implementation primitives."""

from agent.asht.action_bank import (
    MaterializedStaticAction,
    StaticActionBank,
    StaticActionTemplate,
    green_citrus_static_action_bank,
)
from agent.asht.belief import BeliefError, PatchBelief, bayes_update, normalize_distribution
from agent.asht.counting import CountEstimate, estimate_count
from agent.asht.information import (
    RankedAction,
    entropy,
    expected_information_gain,
    hypothetical_posterior,
    kl_divergence,
    predictive_distribution,
    rank_actions,
    realized_entropy_reduction,
)
from agent.asht.kernels import KernelError, SensorProfile, SurrogateKernel, build_surrogate_kernel
from agent.asht.observations import EncodedObservation, ObservationEncoderConfig, encode_target_observation
from agent.asht.stopping import StopDecision, posterior_stop
from agent.asht.qwen_generator import (
    QwenActionClient,
    QwenActionGeneratorConfig,
    QwenCandidateActionGenerator,
    QwenGenerationResult,
)
from agent.asht.runner_qwen import QwenAshtRunner
from agent.asht.runner_static import StaticAshtConfig, StaticAshtRunResult, StaticAshtRunner
from agent.asht.tiling import (
    AdaptiveTilingConfig,
    DensityAssessment,
    TileRule,
    TileSpec,
    TiledExecutionResult,
    assess_density,
    calculate_tile_parameters,
    execute_adaptive_tiling,
    generate_tiles,
)
from agent.asht.update import MathematicalUpdate, UpdateIds, allocate_update_ids, perform_update

__all__ = [
    "AdaptiveTilingConfig",
    "BeliefError",
    "DensityAssessment",
    "CountEstimate",
    "EncodedObservation",
    "KernelError",
    "MaterializedStaticAction",
    "MathematicalUpdate",
    "ObservationEncoderConfig",
    "PatchBelief",
    "QwenActionClient",
    "QwenActionGeneratorConfig",
    "QwenAshtRunner",
    "QwenCandidateActionGenerator",
    "QwenGenerationResult",
    "RankedAction",
    "SensorProfile",
    "StaticActionBank",
    "StaticActionTemplate",
    "StaticAshtConfig",
    "StaticAshtRunResult",
    "StaticAshtRunner",
    "StopDecision",
    "SurrogateKernel",
    "TileRule",
    "TileSpec",
    "TiledExecutionResult",
    "UpdateIds",
    "allocate_update_ids",
    "assess_density",
    "bayes_update",
    "build_surrogate_kernel",
    "calculate_tile_parameters",
    "encode_target_observation",
    "entropy",
    "estimate_count",
    "execute_adaptive_tiling",
    "expected_information_gain",
    "generate_tiles",
    "green_citrus_static_action_bank",
    "hypothetical_posterior",
    "kl_divergence",
    "normalize_distribution",
    "perform_update",
    "posterior_stop",
    "predictive_distribution",
    "rank_actions",
    "realized_entropy_reduction",
]
