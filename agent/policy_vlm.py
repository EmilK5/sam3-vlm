"""
agent/policy_vlm.py

VLM orchestration policy (guided-ROI). choose() shows the VLM the image and asks
it to pick the next sensing action from a menu, then HARD-VALIDATES the response.
Any parse failure or validation violation falls back to the non-visual heuristic
policy, so the VLM can never drive an illegal action.

Grounding guarantee: the VLM only steers the sensor. For "look" it proposes an ROI
box [x1,y1,x2,y2] that says WHERE to run SAM3; that box is a sensing target only
and is never added to the graph as a candidate (enforced in actions.LookROIA /
execute). For "verify" it names existing node ids. No VLM box ever becomes a
detection; candidates originate only from SAM3.
"""

import base64
import io
import json
import logging

from agent.actions import TileQueryA, LookROIA, VerifyA, StopA
from agent.belief import support_score
from agent import policy_heuristic

logger = logging.getLogger(__name__)

_ESTIMATORS = {"N_obs", "N_supp", "N_cons"}

SYSTEM_PROMPT = (
    "You are the orchestrator of a zero-shot object-counting system. You SEE the "
    "image. Choose the single best next sensing action from the provided menu. For "
    "\"look\" you give an ROI box [x1,y1,x2,y2] in image pixels around an area with "
    "many target objects -- this only tells the sensor WHERE to look and never adds "
    "objects to the count. For \"verify\" you name candidate node ids. You never "
    "label or add objects yourself. Reply with ONLY a JSON object "
    "{\"action\": <name>, \"args\": {...}} and nothing else."
)


def choose(phi, z, partition, graph, cfg, client=None, image=None):
    """Return the VLM-chosen action if it validates, else the heuristic action.

    image: the current PIL frame, shown to the VLM so it can place ROI boxes. When
    None (offline / no image), "look" is not offered and any look response falls
    back to the heuristic.
    """
    if client is None:
        client = _build_client(cfg)

    messages = _build_messages(phi, z, graph, cfg, image)
    logger.info("policy_vlm prompt: %s", _log_text(messages))

    try:
        content = _request(client, cfg, messages)
        logger.info("policy_vlm response: %s", content)
    except Exception as exc:  # network / client error -> fall back
        logger.warning("policy_vlm: request failed (%s); fallback to heuristic.", exc)
        return policy_heuristic.choose(phi, partition, cfg)

    action = _parse_and_validate(content, graph, cfg, image)
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
    # Disable qwen3-vl "thinking" for the (structured) policy-loop decision by
    # default -- same model, per-call toggle (config.thinking_call_kwargs).
    from config import thinking_call_kwargs
    extra = thinking_call_kwargs(getattr(cfg, "policy_enable_thinking", False))
    response = client.chat.completions.create(
        model=cfg.oracle_model_name, temperature=cfg.oracle_temperature,
        messages=messages, **extra,
    )
    return response.choices[0].message.content


def _log_text(messages) -> str:
    """The user message's text part (skip the base64 image) for readable logs."""
    content = messages[-1]["content"]
    if isinstance(content, list):
        return next((p.get("text") for p in content if p.get("type") == "text"), "")
    return content


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


def _build_body(phi, z, graph, cfg, image):
    """The JSON state + action menu shown to the VLM (image sent separately)."""
    target = getattr(cfg, "target_prompt", "green fruit")
    verify_available = getattr(cfg, "verifier_mode", "ioc") == "vip"

    # Menu. "look" is only offered when the image is present (the VLM needs to see
    # the frame to place a box). Angle-bracket placeholders signal "substitute a
    # value" so a weak model doesn't echo a literal like "0.1-0.9".
    menu = {
        "tile": {"conf": "<float 0.1-0.9>"},
        "stop": {"estimate_name": "<N_obs|N_supp|N_cons>"},
    }
    if image is not None:
        menu = {"look": {"region": "[x1,y1,x2,y2]"}, **menu}
    if verify_available:
        menu["verify"] = {"node_ids": ["<verifiable id>"]}

    return {
        "target_concept": target,
        "image_size": list(image.size) if image is not None else None,  # [w, h] px
        "phi": _compact_phi(phi),
        "z": z,
        "verifiable_node_ids": _verifiable_ids(phi) if verify_available else [],
        "remaining_budget": phi.get("remaining_budget"),
        "action_menu": menu,
    }


def _build_messages(phi, z, graph, cfg, image=None):
    body = _build_body(phi, z, graph, cfg, image)
    text = ("Choose the next action.\n" + json.dumps(body)
            + "\nReply with ONLY {\"action\":..., \"args\":...}.")
    if image is not None:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
    else:
        content = text
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


# ----------------------- strict validation -----------------------

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


def _validate_look(region, image):
    """A "look" ROI is legal iff we have an image and region is an in-bounds xyxy
    box (four numbers, x1<x2<=W, y1<y2<=H). Returns a LookROIA (sensing target
    only) or None. The 10% margin / min-size / depth / dedup guards live in
    actions.execute; here we only reject out-of-frame or malformed boxes."""
    if image is None:
        return None
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in region):
        return None
    x1, y1, x2, y2 = (float(v) for v in region)
    w, h = image.size
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        return None
    return LookROIA(region=(x1, y1, x2, y2))


def _parse_and_validate(content, graph, cfg, image):
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

    if name == "look":
        return _validate_look(args.get("region"), image)

    if name == "tile":
        if not _valid_conf(args.get("conf")):
            return None
        if not _valid_prompt(args.get("prompt"), target):
            return None
        return TileQueryA(prompt=target, conf=float(args["conf"]))

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
