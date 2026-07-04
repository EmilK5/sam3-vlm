"""
agent/runner.py

Episode loop for a sensing policy: while not stopped, choose an action, execute
it, and update the belief summary. Logs one JSON line per step and returns the
final graph plus the three zero-shot count estimates.
"""

import json
import logging

from agent import belief
from agent.actions import StopA
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
        return policy_vlm.choose(phi, state["last_z"], partition, ctx.graph, cfg, client=vlm_client)

    return policy


def run_episode(image, ctx, policy, max_actions, execute_fn=None) -> dict:
    """Run a heuristic (or any phi->action) policy to termination.

    image      : PIL image (stored on ctx.image_pil).
    ctx        : ActionContext (graph, cfg, oracle, query_set, discovery,
                 partition, cost).
    policy     : callable(phi, partition, cfg) -> action.
    max_actions: hard budget cap on the number of actions.
    execute_fn : injectable executor (defaults to agent.actions.execute).

    Returns {"graph", "counts": {N_obs,N_supp,N_cons}, "log": [...], "cost"}.
    """
    execute_fn = execute_fn or default_execute
    if image is not None:
        ctx.image_pil = image

    log = []
    for t in range(1, max_actions + 1):
        remaining = max_actions - (t - 1)
        phi = belief.summarize(ctx.graph, ctx.discovery, remaining, ctx.cfg)
        action = policy(phi, ctx.partition, ctx.cfg)

        n_new = int(execute_fn(action, ctx))
        ctx.discovery.append(n_new)

        u = belief.uncertainty(ctx.graph, ctx.discovery, ctx.cfg)
        cost_so_far = ctx.cost.total(ctx.cfg) if ctx.cost is not None else 0.0
        entry = {
            "t": t,
            "action": type(action).__name__,
            "n_new": n_new,
            "U": round(u, 3),
            "cost_so_far": round(cost_so_far, 3),
        }
        log.append(entry)
        logger.info(json.dumps(entry))

        if isinstance(action, StopA):
            break

    return {
        "graph": ctx.graph,
        "counts": belief.count_estimates(ctx.graph, ctx.cfg),
        "log": log,
        "cost": ctx.cost.total(ctx.cfg) if ctx.cost is not None else 0.0,
    }
