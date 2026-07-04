"""
agent/actions.py

The sensing action layer (proposal §"Action Space"): plain dataclasses for the
allowed actions and a single execute() dispatcher.

    QueryA(region, prompt, conf)   - SAM3 on one region (roi_override)
    TileQueryA(prompt, conf)       - tiled SAM3 over the (canopy) image
    SubdivideA(region)             - split a region 2x2 in the partition (no model)
    VerifyA(node_ids)              - FM+V-IP verify named candidates, update them
    StopA(estimate_name)           - terminate; names the count estimator to report

execute(action, ctx) -> int  (number of new candidate tracks the action created;
0 for Subdivide / Verify / Stop). ctx (ActionContext) bundles the sensing state.

The VLM never injects boxes: no action accepts detection coordinates. Regions come
from the image partition; candidates come only from SAM3 via execute_pass.
"""

import dataclasses

import numpy as np

from verifier.verify import verify_candidate  # torch-free

# target/distractor/spurious -> candidate-graph classification tags (mirrors pipeline).
_VIP_TAG = {"target": "fruit", "distractor": "leaf", "spurious": "spurious"}


# ----------------------- action dataclasses -----------------------

@dataclasses.dataclass
class QueryA:
    region: tuple            # xyxy pixel region to query
    prompt: str
    conf: float


@dataclasses.dataclass
class TileQueryA:
    prompt: str
    conf: float


@dataclasses.dataclass
class SubdivideA:
    region: tuple            # xyxy region to split 2x2


@dataclasses.dataclass
class VerifyA:
    node_ids: list           # candidate node ids to (re)verify


@dataclasses.dataclass
class StopA:
    estimate_name: str       # which count estimator to report (e.g. "N_obs")


@dataclasses.dataclass
class ActionContext:
    """Bundle passed to execute(); the mutable episode sensing state."""
    processor: object = None
    image_pil: object = None
    graph: object = None
    cfg: object = None
    oracle: object = None
    query_set: object = None
    discovery: object = None
    partition: list = dataclasses.field(default_factory=list)


# ----------------------- helpers -----------------------

def subdivide_region(region) -> list:
    """Split an xyxy region into its four quadrants (top-left, top-right,
    bottom-left, bottom-right)."""
    x1, y1, x2, y2 = region
    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0
    return [
        (x1, y1, mx, my),
        (mx, y1, x2, my),
        (x1, my, mx, y2),
        (mx, my, x2, y2),
    ]


def _next_pass_number(ctx) -> int:
    """execute_pass pass_number for the next sensing action in the episode."""
    n_prior = len(ctx.discovery.counts) if ctx.discovery is not None else 0
    return n_prior + 1


def _execute_query(action, ctx, tiling, roi_override) -> int:
    from pipeline import execute_pass  # lazy: pipeline imports torch/inference
    stats = execute_pass(
        processor=ctx.processor, image_pil=ctx.image_pil, graph=ctx.graph,
        conf=action.conf, clahe=False, tiling=tiling,
        pass_number=_next_pass_number(ctx), prompt=action.prompt,
        cfg=ctx.cfg, oracle=ctx.oracle, query_set=ctx.query_set,
        roi_override=roi_override,
    )
    return int(stats)


def _execute_subdivide(action, ctx) -> int:
    quadrants = subdivide_region(action.region)
    if action.region in ctx.partition:
        idx = ctx.partition.index(action.region)
        ctx.partition[idx:idx + 1] = quadrants
    else:
        raise ValueError(f"SubdivideA region {action.region} is not in the current partition.")
    return 0  # no model call, no new candidates


def _execute_verify(action, ctx) -> int:
    image_np = np.array(ctx.image_pil)
    classes = ctx.query_set.classes
    dist_idx = classes.index("distractor") if "distractor" in classes else 0
    for node_id in action.node_ids:
        node = ctx.graph.nodes.get(node_id)
        if node is None:
            continue
        result = verify_candidate(image_np, node.box, ctx.oracle, ctx.query_set, ctx.cfg)
        node.scores["fruit_verification"] = float(result["p_target"])
        node.scores["leaf_verification"] = float(result["posterior"][dist_idx])
        node.classification = _VIP_TAG.get(result["verdict"], "unresolved")
        node.vip_chain = result["chain"]
        node.vip_posterior = result["posterior"]
    return 0  # verification doesn't create new tracks


def execute(action, ctx) -> int:
    """Dispatch an action, returning the number of new candidate tracks created."""
    if isinstance(action, QueryA):
        return _execute_query(action, ctx, tiling=False, roi_override=action.region)
    if isinstance(action, TileQueryA):
        return _execute_query(action, ctx, tiling=True, roi_override=None)
    if isinstance(action, SubdivideA):
        return _execute_subdivide(action, ctx)
    if isinstance(action, VerifyA):
        return _execute_verify(action, ctx)
    if isinstance(action, StopA):
        return 0
    raise TypeError(f"Unknown action type: {type(action).__name__}")
