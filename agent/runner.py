"""
agent/runner.py

Episode loop for a sensing policy: while not stopped, choose an action, execute
it, and update the belief summary. Logs one JSON line per step and returns the
final graph plus the three zero-shot count estimates.
"""

import json
import logging

from agent import belief
from agent.actions import QueryA, TileQueryA, LookROIA, StopA
from agent.actions import execute as default_execute
from agent import policy_vlm
from agent.inspect import inspect_scene, should_inspect

logger = logging.getLogger(__name__)


def make_vlm_policy(ctx, inspect_client=None, vlm_client=None):
    """Adapt the VLM orchestrator into a runner policy `callable(phi, partition, cfg)`.

    Maintains the episode step counter and last scene assessment z. Runs
    inspect_scene only when the fixed protocol (should_inspect) fires, then asks
    policy_vlm.choose to pick an action (which hard-validates and falls back to
    the heuristic on any violation). Clients are injectable for offline testing.
    """
    state = {"t": 0, "last_z": None}

    def policy(phi, partition, cfg):
        state["t"] += 1
        if should_inspect(state["t"], phi, state["last_z"]):
            state["last_z"] = inspect_scene(ctx.image_pil, phi, cfg, client=inspect_client)
        # The VLM sees the frame so it can place ROI ("look") boxes.
        return policy_vlm.choose(phi, state["last_z"], partition, ctx.graph, cfg,
                                 client=vlm_client, image=ctx.image_pil)

    return policy


# Sensing actions observe new candidates and so feed the discovery curve; the
# non-sensing ones (subdivide/verify/stop) do not (recording their 0 would fake
# saturation and could end the episode before it has ever sensed).
_SENSING = (QueryA, TileQueryA, LookROIA)


def run_episode(image, ctx, policy, max_actions, execute_fn=None,
                bootstrap_global_pass=False, auto_stop=False) -> dict:
    """Run a heuristic (or any phi->action) policy to termination.

    image      : PIL image (stored on ctx.image_pil).
    ctx        : ActionContext (graph, cfg, oracle, query_set, discovery,
                 partition, cost).
    policy     : callable(phi, partition, cfg) -> action.
    max_actions: hard budget cap on the number of actions.
    execute_fn : injectable executor (defaults to agent.actions.execute).
    bootstrap_global_pass: when True, run one mandatory global (non-tiled) QueryA
        over the full-frame region (partition[0], the canopy tree_roi) BEFORE the
        policy loop, seeding candidates + pseudo-exemplars. Counts against
        max_actions. Default off so existing (heuristic-baseline) episodes are
        unchanged.
    auto_stop: when True, end the episode once discovery saturates on a non-empty
        graph (DiscoveryCurve.saturated over cfg.window_m / cfg.delta_disc), even
        if the policy never returns StopA. Budget exhaustion is the other backstop
        (the loop cap); "all proposed ROIs sensed" is subsumed by saturation.
        Default off.

    Returns {"graph", "counts": {N_obs,N_supp,N_cons}, "log": [...], "cost"}.
    """
    execute_fn = execute_fn or default_execute
    if image is not None:
        ctx.image_pil = image

    log = []

    def _step(t, action):
        n_new = int(execute_fn(action, ctx))
        if isinstance(action, _SENSING):
            ctx.discovery.append(n_new)
        u = belief.uncertainty(ctx.graph, ctx.discovery, ctx.cfg)
        cost_so_far = ctx.cost.total(ctx.cfg) if ctx.cost is not None else 0.0
        log.append({
            "t": t,
            "action": type(action).__name__,
            "n_new": n_new,
            "U": round(u, 3),
            "cost_so_far": round(cost_so_far, 3),
        })
        logger.info(json.dumps(log[-1]))
        return n_new

    t = 0
    # Mandatory global bootstrap pass (proposal §"Guided-ROI policy"): a non-tiled
    # QueryA over the episode's full-frame region seeds candidates + pseudo-exemplars
    # before the policy ever acts, so later tiled/Look passes are exemplar-primed.
    if bootstrap_global_pass:
        t += 1
        region = tuple(ctx.partition[0]) if ctx.partition else (0, 0, 0, 0)
        prompt = getattr(ctx.cfg, "target_prompt", "green fruit")
        conf = getattr(ctx.cfg, "conf", 0.3)
        _step(t, QueryA(region=region, prompt=prompt, conf=conf))

    while t < max_actions:
        # Auto-stop backstop: saturated discovery on a non-empty graph ends the
        # episode before spending another action, even if the policy never stops.
        if (auto_stop and len(ctx.graph.nodes) > 0
                and ctx.discovery.saturated(ctx.cfg.window_m, ctx.cfg.delta_disc)):
            break
        t += 1
        remaining = max_actions - (t - 1)
        phi = belief.summarize(ctx.graph, ctx.discovery, remaining, ctx.cfg)
        action = policy(phi, ctx.partition, ctx.cfg)
        _step(t, action)
        if isinstance(action, StopA):
            break

    return {
        "graph": ctx.graph,
        "counts": belief.count_estimates(ctx.graph, ctx.cfg),
        "log": log,
        "cost": ctx.cost.total(ctx.cfg) if ctx.cost is not None else 0.0,
    }
