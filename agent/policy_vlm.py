"""
agent/policy_vlm.py

VLM orchestration policy. choose() asks the VLM (one text call, no image) to pick
the next sensing action from a menu of legal, id-referenced actions, then HARD-
VALIDATES the response. Any parse failure or validation violation falls back to
the non-visual heuristic policy, so the VLM can never drive an illegal action.

Grounding guarantee: the VLM selects regions by id and nodes by id -- it never
supplies box coordinates. No action here accepts coordinates from the model.
"""

import json
import logging

from agent.actions import QueryA, TileQueryA, SubdivideA, VerifyA, StopA
from agent.belief import support_score
from agent import policy_heuristic

logger = logging.getLogger(__name__)

_ESTIMATORS = {"N_obs", "N_supp", "N_cons"}

SYSTEM_PROMPT = (
    "You are the orchestrator of a zero-shot object-counting system. Choose the "
    "single best next sensing action from the provided menu. You select regions and "
    "candidate nodes by id only; you never provide pixel coordinates. Reply with "
    "ONLY a JSON object {\"action\": <name>, \"args\": {...}} and nothing else."
)


def choose(phi, z, partition, graph, cfg, client=None):
    """Return the VLM-chosen action if it validates, else the heuristic action."""
    if client is None:
        client = _build_client(cfg)

    messages = _build_messages(phi, z, partition, graph, cfg)
    logger.info("policy_vlm prompt: %s", messages[-1]["content"])

    try:
        content = _request(client, cfg, messages)
        logger.info("policy_vlm response: %s", content)
    except Exception as exc:  # network / client error -> fall back
        logger.warning("policy_vlm: request failed (%s); fallback to heuristic.", exc)
        return policy_heuristic.choose(phi, partition, cfg)

    action = _parse_and_validate(content, partition, graph, cfg)
    if action is None:
        logger.warning("policy_vlm: invalid/illegal response; fallback to heuristic.")
        return policy_heuristic.choose(phi, partition, cfg)
    return action


# ----------------------- request plumbing -----------------------

def _build_client(cfg):
    import os
    from openai import OpenAI  # lazy
    return OpenAI(base_url=cfg.oracle_base_url, api_key=os.environ.get("QWEN_API_KEY", "EMPTY"))


def _request(client, cfg, messages) -> str:
    response = client.chat.completions.create(
        model=cfg.oracle_model_name, temperature=cfg.oracle_temperature, messages=messages,
    )
    return response.choices[0].message.content


def _verifiable_ids(phi):
    """Node ids that are unresolved (VerifyA targets); paired with low-w in _validate."""
    return [phi["ids"][i] for i in range(phi["K"]) if phi["classification"][i] == "unresolved"]


def _compact_phi(phi):
    """Compact belief summary: keep the 15 most-uncertain nodes' per-node fields."""
    K = phi["K"]
    def instability(i):
        return 1.0 / (1 + phi["k"][i]) + phi["delta"][i] + (1.0 - phi["s"][i])
    top = sorted(range(K), key=instability, reverse=True)[:15]
    nodes = [{"id": phi["ids"][i], "w": phi["w"][i], "s": phi["s"][i],
              "k": phi["k"][i], "delta": phi["delta"][i], "class": phi["classification"][i]}
             for i in top]
    return {
        "K": K, "n_t": phi["n_t"], "D": phi["D"], "U": phi["U"],
        "tiling_status": phi["tiling_status"], "remaining_budget": phi["remaining_budget"],
        "top_nodes": nodes,
    }


def _build_messages(phi, z, partition, graph, cfg):
    target = getattr(cfg, "target_prompt", "green fruit")
    regions = [{"region_id": i, "xyxy": list(r)} for i, r in enumerate(partition)]
    # Angle-bracket placeholders (not bare literals): weak models copy a literal
    # value like "0.1-0.9" straight into args, which then fails validation. The
    # brackets signal "substitute a value here" instead.
    menu = {
        "query": {"region_id": "<int>", "conf": "<float 0.1-0.9>"},
        "tile": {"conf": "<float 0.1-0.9>"},
        "subdivide": {"region_id": "<int>"},
        "stop": {"estimate_name": "<N_obs|N_supp|N_cons>"},
    }
    # verify needs the FM+V-IP oracle; only offer it when the vip verifier is on.
    verify_available = getattr(cfg, "verifier_mode", "ioc") == "vip"
    if verify_available:
        menu["verify"] = {"node_ids": ["<verifiable id>"]}
    body = {
        "target_concept": target,
        "phi": _compact_phi(phi),
        "z": z,
        "regions": regions,
        "verifiable_node_ids": _verifiable_ids(phi) if verify_available else [],
        "remaining_budget": phi.get("remaining_budget"),
        "action_menu": menu,
    }
    text = "Choose the next action.\n" + json.dumps(body) + "\nReply with ONLY {\"action\":..., \"args\":...}."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


# ----------------------- strict validation -----------------------

def _valid_region_id(rid, partition):
    return isinstance(rid, int) and not isinstance(rid, bool) and 0 <= rid < len(partition)


def _valid_conf(c):
    return isinstance(c, (int, float)) and not isinstance(c, bool) and 0.1 <= c <= 0.9


def _valid_prompt(p, target):
    return p is None or p == target


def _valid_node_ids(ids, graph, cfg):
    if not isinstance(ids, list) or not ids:
        return False
    tau_w = getattr(cfg, "tau_w", 0.5)
    for nid in ids:
        node = graph.nodes.get(nid)
        if node is None:
            return False
        low_w = support_score(node, cfg) < tau_w
        if not (node.classification == "unresolved" or low_w):
            return False
    return True


def _parse_and_validate(content, partition, graph, cfg):
    """Return a validated action dataclass, or None on any parse/validation failure."""
    try:
        data = json.loads(content)
        name = data["action"]
        args = data.get("args", {})
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(args, dict):
        return None

    target = getattr(cfg, "target_prompt", "green fruit")

    if name == "query":
        rid = args.get("region_id")
        if not _valid_region_id(rid, partition):
            return None
        if not _valid_conf(args.get("conf")):
            return None
        if not _valid_prompt(args.get("prompt"), target):
            return None
        return QueryA(region=tuple(partition[rid]), prompt=target, conf=float(args["conf"]))

    if name == "tile":
        if not _valid_conf(args.get("conf")):
            return None
        if not _valid_prompt(args.get("prompt"), target):
            return None
        return TileQueryA(prompt=target, conf=float(args["conf"]))

    if name == "subdivide":
        rid = args.get("region_id")
        if not _valid_region_id(rid, partition):
            return None
        return SubdivideA(region=tuple(partition[rid]))

    if name == "verify":
        if getattr(cfg, "verifier_mode", "ioc") != "vip":
            return None  # verify is only legal with the FM+V-IP oracle configured
        node_ids = args.get("node_ids")
        if not _valid_node_ids(node_ids, graph, cfg):
            return None
        return VerifyA(node_ids=list(node_ids))

    if name == "stop":
        # Never terminate before any candidate has been registered: a stop on an
        # empty graph reports zero having sensed nothing. Fall back so the
        # heuristic picks a sensing action instead.
        if len(graph.nodes) == 0:
            return None
        est = args.get("estimate_name", "N_cons")
        if est not in _ESTIMATORS:
            return None
        return StopA(estimate_name=est)

    return None
