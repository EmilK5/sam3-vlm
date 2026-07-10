"""
agent/policy_vlm.py

v2 full-history VLM orchestration policy (docs/active_perception_formulation.md).
choose() shows the VLM the whole episode history (x_1^t, y_1^t) together with the
raw frame and an annotated overlay, and asks it to pick the next REGION x_t to
sense. The response is HARD-VALIDATED; any parse or validation failure falls back
to the non-visual heuristic policy, so the VLM can never drive an illegal action.

Grounding guarantee: the VLM only steers the sensor. For "look" it proposes an ROI
box [x1,y1,x2,y2] that says WHERE to run SAM3; that box is a sensing target only
and is never added to the graph as a candidate (enforced in actions.LookROIA /
execute). No VLM box ever becomes a detection; candidates originate only from SAM3.

Menu: {"look", "stop"} only. A look region must lie INSIDE the tree ROI (not
merely the frame). Verification runs automatically inside the pipeline
(cfg.verifier_mode); it is not a policy action.
"""

import base64
import io
import json
import logging

from agent.actions import LookROIA, StopA
from agent import policy_heuristic

logger = logging.getLogger(__name__)

_ESTIMATORS = {"N_obs", "N_supp", "N_cons"}

SYSTEM_PROMPT = (
    "You are the sensing controller of a zero-shot object-counting system, framed "
    "as active perception. At each step you choose the next REGION x_t to point the "
    "SAM3 sensor at, so as to reveal the most previously-unseen target objects, "
    "guided by the FULL history of past actions and observations you are given. You "
    "see two images: the raw frame, and an overlay showing already-found objects "
    "(colored by class), already-sensed regions, and the tree region of interest. "
    "For \"look\" you give an ROI box [x1,y1,x2,y2] in image pixels INSIDE the tree "
    "ROI -- this only tells the sensor WHERE to look and never adds objects to the "
    "count. You never label or add objects yourself. Reply with ONLY a JSON object "
    "{\"action\": <name>, \"args\": {...}} and nothing else."
)


def choose(phi, history, graph, cfg, client=None, image=None, sensed_rois=None):
    """Return the VLM-chosen action if it validates, else the heuristic action.

    phi         : compact belief summary (agent.belief.summarize).
    history     : the episode history (EpisodeHistory or a list of records) shown
                  to the VLM verbatim as the past (x_1^t, y_1^t).
    graph       : OrchardGraph (source of tree_roi + candidate boxes for the overlay).
    image       : the current PIL frame. When None (offline / no image), "look" is
                  not offered and any look response falls back to the heuristic.
    sensed_rois : xyxy ROIs already sensed, drawn on the overlay. None -> none.
    """
    if client is None:
        client = _build_client(cfg)

    tree_roi = _resolve_tree_roi(graph, image)
    messages = _build_messages(phi, history, graph, cfg, image, tree_roi, sensed_rois)
    logger.info("policy_vlm prompt: %s", _log_text(messages))

    try:
        content = _request(client, cfg, messages)
        logger.info("policy_vlm response: %s", content)
    except Exception as exc:  # network / client error -> fall back
        logger.warning("policy_vlm: request failed (%s); fallback to heuristic.", exc)
        return _fallback(phi, cfg, tree_roi)

    action = _parse_and_validate(content, graph, cfg, image, tree_roi)
    if action is None:
        logger.warning("policy_vlm: invalid/illegal response; fallback to heuristic.")
        return _fallback(phi, cfg, tree_roi)
    return action


def _fallback(phi, cfg, tree_roi):
    """The safety net: the non-visual heuristic, anchored to the tree ROI (which is
    the v2 partition -- a single region). tree_roi None (offline, no image) yields a
    degenerate anchor, which only occurs in contrived offline calls."""
    partition = [tuple(tree_roi)] if tree_roi is not None else [(0.0, 0.0, 0.0, 0.0)]
    return policy_heuristic.choose(phi, partition, cfg)


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
    """The user message's text part (skip the base64 images) for readable logs."""
    content = messages[-1]["content"]
    if isinstance(content, list):
        return next((p.get("text") for p in content if p.get("type") == "text"), "")
    return content


def _resolve_tree_roi(graph, image):
    """The xyxy tree ROI: graph.tree_roi if set, else the full frame (when an image
    is present), else None (offline, no image)."""
    roi = getattr(graph, "tree_roi", None)
    if roi is not None:
        return [float(v) for v in roi]
    if image is not None:
        w, h = image.size
        return [0.0, 0.0, float(w), float(h)]
    return None


def _history_records(history):
    """Normalize an EpisodeHistory / list / None into a plain JSON-ready list."""
    if history is None:
        return []
    if hasattr(history, "as_list"):
        return history.as_list()
    return list(history)


def _data_url(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


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


def _build_body(phi, history, graph, cfg, image, tree_roi):
    """The JSON state + action menu shown to the VLM (images sent separately)."""
    target = getattr(cfg, "target_prompt", "green fruit")

    # Menu (v2): look/stop only. "look" is only offered when the image is present
    # (the VLM needs to see the frame to place a box). Angle-bracket placeholders
    # signal "substitute a value" so a weak model doesn't echo a literal.
    menu = {"stop": {"estimate_name": "<N_obs|N_supp|N_cons>"}}
    if image is not None:
        menu = {"look": {"region": "[x1,y1,x2,y2]"}, **menu}

    return {
        "target_concept": target,
        "image_size": list(image.size) if image is not None else None,  # [w, h] px
        "tree_roi": [float(v) for v in tree_roi] if tree_roi is not None else None,
        "history": _history_records(history),   # the complete past (x_1^t, y_1^t)
        "phi": _compact_phi(phi),
        "action_menu": menu,
    }


def _build_messages(phi, history, graph, cfg, image=None, tree_roi=None, sensed_rois=None):
    body = _build_body(phi, history, graph, cfg, image, tree_roi)
    text = ("Choose the next region to sense (or stop).\n" + json.dumps(body)
            + "\nReply with ONLY {\"action\":..., \"args\":...}.")
    if image is not None:
        from agent.overlay import render_overlay  # lazy: only the image path needs PIL draw
        overlay = render_overlay(image, graph, sensed_rois=sensed_rois, tree_roi=tree_roi)
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": _data_url(image)}},
            {"type": "image_url", "image_url": {"url": _data_url(overlay)}},
        ]
    else:
        content = text
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


# ----------------------- strict validation -----------------------

def _coerce_json_string(value):
    """Some local VLMs (observed with Qwen3-VL via Ollama) stringify nested JSON
    values, e.g. "region": "[x1,y1,x2,y2]" instead of emitting a real array. If
    `value` is a string, try to parse it as JSON and return the parsed value;
    otherwise (or on parse failure) return `value` unchanged so downstream type
    checks still fail closed."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _validate_look(region, image, tree_roi):
    """A "look" ROI is legal iff we have an image + tree ROI and region is an xyxy
    box (four numbers, x1<x2, y1<y2) lying INSIDE the tree ROI. Returns a LookROIA
    (sensing target only) or None. The 10% margin / min-size / depth / dedup guards
    live in actions.execute; here we only reject malformed or out-of-tree-ROI boxes."""
    if image is None or tree_roi is None:
        return None
    region = _coerce_json_string(region)
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in region):
        return None
    x1, y1, x2, y2 = (float(v) for v in region)
    tx1, ty1, tx2, ty2 = (float(v) for v in tree_roi)
    if not (tx1 <= x1 < x2 <= tx2 and ty1 <= y1 < y2 <= ty2):
        return None
    return LookROIA(region=(x1, y1, x2, y2))


def _parse_and_validate(content, graph, cfg, image, tree_roi):
    """Return a validated action dataclass, or None on any parse/validation failure."""
    try:
        data = json.loads(content)
        name = data["action"]
        args = data.get("args", {})
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(args, dict):
        return None

    if name == "look":
        return _validate_look(args.get("region"), image, tree_roi)

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
