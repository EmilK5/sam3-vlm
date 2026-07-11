"""
agent/runner.py

Episode loop for a sensing policy: while not stopped, choose an action, execute
it, and update the belief summary. Logs one JSON line per step and returns the
final graph plus the three zero-shot count estimates.
"""

import json
import logging

from agent import belief
from agent.actions import QueryA, LookROIA, StopA
from agent.actions import execute as default_execute
from agent import policy_vlm, policy_vlm_v3
from agent.history import EpisodeHistory, action_params, node_summary

logger = logging.getLogger(__name__)


def make_vlm_policy(ctx, vlm_client=None):
    """Adapt the v2 full-history VLM orchestrator into a runner policy
    `callable(phi, partition, cfg)`.

    The policy is stateless per step: it shows the VLM the whole episode history
    (ctx.history) plus the raw + overlay frames and asks for the next region to
    sense. policy_vlm.choose hard-validates and falls back to the heuristic on any
    violation. There is no separate scene-inspection (z) call in v2 -- the full
    history subsumes it. Client injectable for offline testing.
    """
    def policy(phi, partition, cfg):
        return policy_vlm.choose(
            phi, getattr(ctx, "history", None), ctx.graph, cfg,
            client=vlm_client, image=ctx.image_pil,
            sensed_rois=getattr(ctx, "sensed_rois", None),
        )

    return policy


def make_refine_policy(ctx, vlm_client=None):
    """Adapt the v3 prompt-refinement policy (phase 9) into a runner policy
    `callable(phi, partition, cfg)`.

    Same episode wiring as make_vlm_policy -- the VLM is shown the whole history
    plus the raw + overlay frames -- but it refines the SAM3 TEXT PROMPT (1-2
    adjectives + noun) + threshold over a FIXED region (the tree ROI), so there is
    no sensed_rois to pass (no sub-ROI zoom). policy_vlm_v3.choose hard-validates
    and falls back to the heuristic on any violation. Client injectable for offline
    testing. make_vlm_policy is left unchanged (v2 look/stop stays the default)."""
    def policy(phi, partition, cfg):
        return policy_vlm_v3.choose(
            phi, getattr(ctx, "history", None), ctx.graph, cfg,
            client=vlm_client, image=ctx.image_pil,
        )

    return policy


# Sensing actions observe new candidates and so feed the discovery curve; the
# non-sensing StopA does not (recording its 0 would fake saturation and could
# end the episode before it has ever sensed).
_SENSING = (QueryA, LookROIA)


def _totals(ctx) -> dict:
    """A compact snapshot of the current count state for a history record."""
    return {"K": len(ctx.graph.nodes),
            "N_obs": int(belief.count_estimates(ctx.graph, ctx.cfg)["N_obs"])}


def _obs(ctx, n_new, new_nodes, totals=None) -> dict:
    """Build an observation record y. Beyond n_new (previously-unseen tracks) and the
    new-node summaries, it carries the last sensing pass's detection feedback:
    n_detections (total distinct detections the prompt produced) and n_redetected
    (detections that matched already-known tracks -- dedup re-detections). n_redetected
    is the signal that lets the policy judge a prompt by how much of the known target
    set it re-finds, not only by what it adds. Zeros when no sensing pass ran (the
    stats are stashed on ctx by actions._execute_query)."""
    stats = getattr(ctx, "last_pass_stats", None) or {}
    return {
        "n_new": n_new,
        "n_redetected": int(stats.get("n_redetected", 0)),
        "n_detections": int(stats.get("n_detections", 0)),
        "new_nodes": new_nodes,
        "totals": totals if totals is not None else _totals(ctx),
    }


def run_episode(image, ctx, policy, max_actions, execute_fn=None,
                bootstrap_global_pass=False, bootstrap_tiled_pass=False,
                auto_stop=False) -> dict:
    """Run a heuristic (or any phi->action) policy to termination.

    image      : PIL image (stored on ctx.image_pil).
    ctx        : ActionContext (graph, cfg, oracle, query_set, discovery,
                 partition, cost). An EpisodeHistory is attached as ctx.history.
    policy     : callable(phi, partition, cfg) -> action.
    max_actions: hard budget cap on the number of SENSING actions.
    execute_fn : injectable executor (defaults to agent.actions.execute).
    bootstrap_global_pass: when True, run one mandatory global (non-tiled) QueryA
        over the full-frame region (partition[0], the canopy tree_roi) BEFORE the
        policy loop, seeding candidates + pseudo-exemplars. This ONE sensing pass
        is decomposed into three explicit history records the policy will see --
        canopy_roi, leaf_map, global_pass -- and is billed as ONE sensing action
        against max_actions (the canopy/leaf-map records document the pass, they
        are not separately billed). Default off so existing (heuristic-baseline)
        episodes are unchanged.
    bootstrap_tiled_pass: when True (only meaningful with bootstrap_global_pass), run
        ONE additional TILED pass over the tree ROI with the same seed prompt at
        cfg.refine_conf_default right after the global bootstrap -- the generic
        cascade's recall floor -- recorded as a "tiled_seed_pass" and billed as one
        more sensing action. Default off.
    auto_stop: when True, end the episode once discovery saturates on a non-empty
        graph (DiscoveryCurve.saturated over cfg.window_m / cfg.delta_disc), even
        if the policy never returns StopA. Budget exhaustion is the other backstop
        (the loop cap); "all proposed ROIs sensed" is subsumed by saturation.
        Default off.

    Returns {"graph", "counts": {N_obs,N_supp,N_cons}, "log": [...],
             "history": [...], "cost"}. "log" and "history" have equal length
    (one entry per executed action, with the bootstrap contributing three).
    """
    execute_fn = execute_fn or default_execute
    if image is not None:
        ctx.image_pil = image
    ctx.history = EpisodeHistory()

    log = []

    def _record(x, y):
        """Append one history record + its mirrored one-line log entry."""
        ctx.history.append(x, y)
        u = belief.uncertainty(ctx.graph, ctx.discovery, ctx.cfg)
        cost_so_far = ctx.cost.total(ctx.cfg) if ctx.cost is not None else 0.0
        log.append({
            "t": len(log) + 1,
            "action": x["action"],
            "n_new": y["n_new"],
            "U": round(u, 3),
            "cost_so_far": round(cost_so_far, 3),
        })
        logger.info(json.dumps(log[-1]))

    def _sense(action):
        """Execute a sensing/stop action; return (n_new, [new node summaries])."""
        pre_ids = set(ctx.graph.nodes)
        n_new = int(execute_fn(action, ctx))
        if isinstance(action, _SENSING):
            ctx.discovery.append(n_new)
        new_nodes = [node_summary(ctx.graph.nodes[nid])
                     for nid in ctx.graph.nodes if nid not in pre_ids]
        return n_new, new_nodes

    sensing_used = 0  # sensing actions consumed (the budget counter, != len(log))

    # Mandatory global bootstrap pass (docs/active_perception_formulation.md §
    # "bootstrap"): one non-tiled QueryA over partition[0] (the canopy tree_roi),
    # decomposed into three explicit records so the policy sees what was done.
    if bootstrap_global_pass:
        region = tuple(ctx.partition[0]) if ctx.partition else (0, 0, 0, 0)
        prompt = getattr(ctx.cfg, "target_prompt", "green fruit")
        conf = getattr(ctx.cfg, "conf", 0.3)
        boot = QueryA(region=region, prompt=prompt, conf=conf)
        n_new, new_nodes = _sense(boot)
        sensing_used += 1
        global_y = _obs(ctx, n_new, new_nodes)   # reads the bootstrap pass's stats
        totals = global_y["totals"]
        empty_y = {"n_new": 0, "n_redetected": 0, "n_detections": 0,
                   "new_nodes": [], "totals": totals}

        # (1) canopy_roi: the tree ROI box actually used (graph.tree_roi or, when
        #     unset -- offline/stub runs -- the region the pass ran over).
        tree_roi = getattr(ctx.graph, "tree_roi", None) or list(region)
        _record({"action": "canopy_roi", "roi": list(tree_roi)}, dict(empty_y))
        # (2) leaf_map: the number of background-leaf boxes the pass generated.
        leaf_boxes = getattr(ctx.graph, "cached_leaf_boxes", None)
        n_leaves = int(len(leaf_boxes)) if leaf_boxes is not None else 0
        _record({"action": "leaf_map", "n_leaves": n_leaves}, dict(empty_y))
        # (3) global_pass: the standard observation record for the QueryA.
        x_global = action_params(boot)
        x_global["action"] = "global_pass"
        _record(x_global, global_y)

        # (4) optional tiled_seed_pass: a TILED pass over the same tree ROI with the
        #     SAME seed prompt at refine_conf_default -- the generic cascade's recall
        #     floor, so the active arm never starts below the fixed cascade's coverage
        #     before the VLM's refines begin. Billed as one more sensing action.
        if bootstrap_tiled_pass and sensing_used < max_actions:
            tconf = getattr(ctx.cfg, "refine_conf_default", conf)
            tiled = QueryA(region=region, prompt=prompt, conf=tconf, tiling=True)
            n_new_t, new_nodes_t = _sense(tiled)
            sensing_used += 1
            x_tiled = action_params(tiled)
            x_tiled["action"] = "tiled_seed_pass"
            _record(x_tiled, _obs(ctx, n_new_t, new_nodes_t))

    while sensing_used < max_actions:
        # Auto-stop backstop: saturated discovery on a non-empty graph ends the
        # episode before spending another action, even if the policy never stops.
        if (auto_stop and len(ctx.graph.nodes) > 0
                and ctx.discovery.saturated(ctx.cfg.window_m, ctx.cfg.delta_disc)):
            break
        remaining = max_actions - sensing_used
        phi = belief.summarize(ctx.graph, ctx.discovery, remaining, ctx.cfg)
        action = policy(phi, ctx.partition, ctx.cfg)
        n_new, new_nodes = _sense(action)
        if isinstance(action, _SENSING):
            sensing_used += 1
        _record(action_params(action), _obs(ctx, n_new, new_nodes))
        if isinstance(action, StopA):
            break

    return {
        "graph": ctx.graph,
        "counts": belief.count_estimates(ctx.graph, ctx.cfg),
        "log": log,
        "history": ctx.history.as_list(),
        "cost": ctx.cost.total(ctx.cfg) if ctx.cost is not None else 0.0,
    }
