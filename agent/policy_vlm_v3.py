"""
agent/policy_vlm_v3.py

Phase 9 prompt-refinement policy. Where the v2 policy (policy_vlm.py) locks the
SAM3 text prompt to a system-level constant and lets the VLM choose only WHERE to
look, this policy does the opposite: the region is FIXED to the canopy tree ROI
(one global pass per step) and the VLM chooses WHAT to ask SAM3 -- a refined text
prompt (1-2 adjectives + a noun) plus the detection threshold.

Why global-tree-ROI only:
  * A global pass over the tree ROI keeps every already-found positive exemplar in
    frame, so exemplar priming is never lost. SAM3 visual exemplars are welded to
    the query frame (roi_align'd against that image's own features); there is no
    cross-image exemplar conditioning, so a sub-ROI crop would silently drop the
    out-of-crop exemplars.
  * Fixing the region isolates the causal effect of prompt refinement -- no tiling
    or ensembling confound -- which is exactly the hypothesis under test (does the
    VLM proposing better descriptive terms improve SAM3's precision/recall?).

Grounding guarantee (CLAUDE.md #3): the VLM emits TEXT, never a box. SAM3 grounds
the prompt into detections; no VLM-supplied box ever becomes a candidate. The
prompt only parameterizes a SAM3 query (region + threshold).

Menu: {"refine", "stop"}. A refine maps to a global QueryA over the tree ROI with
the validated prompt + clamped threshold. Any parse/validation failure falls back
to the non-visual heuristic, so the VLM can never drive an illegal action.

Shares the request plumbing (client build, request, image encode, belief compaction,
tree-ROI resolution, JSON coercion) with policy_vlm to stay behaviorally identical
where the two overlap; only the action menu, validation, and body differ.
"""

import json
import logging

from agent.actions import QueryA, StopA
from agent import policy_heuristic
from agent.policy_vlm import (
    _ESTIMATORS,
    _build_client,
    _coerce_json_string,
    _compact_phi,
    _data_url,
    _history_records,
    _log_text,
    _request,
    _resolve_tree_roi,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the sensing controller of a zero-shot object-counting system, framed "
    "as active perception. The SAM3 sensor always scans the SAME region -- the tree "
    "region of interest -- so your job each step is to choose WHAT to ask it: a short "
    "text prompt of one or two adjectives plus a noun (e.g. \"green fruit\", "
    "\"round fruit\", \"yellow citrus\"), and a detection threshold. You are shown the "
    "raw frame, an overlay of the objects already found (colored by class), and the "
    "FULL history of the prompts already tried and what each one yielded. Look at the "
    "boxes: refine the wording to reveal previously-missed target objects and to avoid "
    "the distractors the last prompt picked up. You never label or add objects "
    "yourself; SAM3 grounds your text into boxes. Reply with ONLY a JSON object "
    "{\"action\": <name>, \"args\": {...}} and nothing else."
)


def choose(phi, history, graph, cfg, client=None, image=None):
    """Return the VLM-chosen action if it validates, else the heuristic action.

    phi     : compact belief summary (agent.belief.summarize).
    history : the episode history (EpisodeHistory or a list) shown verbatim as the
              past (x_1^t, y_1^t) -- this carries every past prompt and its yield.
    graph   : OrchardGraph (source of tree_roi + candidate boxes for the overlay).
    image   : the current PIL frame. When None (offline / no image), "refine" is not
              offered (the VLM needs to see the boxes) and any refine response falls
              back to the heuristic.
    """
    if client is None:
        client = _build_client(cfg)

    tree_roi = _resolve_tree_roi(graph, image)
    messages = _build_messages(phi, history, graph, cfg, image, tree_roi)
    logger.info("policy_vlm_v3 prompt: %s", _log_text(messages))

    try:
        content = _request(client, cfg, messages)
        logger.info("policy_vlm_v3 response: %s", content)
    except Exception as exc:  # network / client error -> fall back
        logger.warning("policy_vlm_v3: request failed (%s); fallback to heuristic.", exc)
        return _fallback(phi, cfg, tree_roi)

    action = _parse_and_validate(content, graph, cfg, image, tree_roi)
    if action is None:
        logger.warning("policy_vlm_v3: invalid/illegal response; fallback to heuristic.")
        return _fallback(phi, cfg, tree_roi)
    return action


def _fallback(phi, cfg, tree_roi):
    """The safety net: the non-visual heuristic, anchored to the tree ROI (the single
    v2/v3 partition region). tree_roi None (offline, no image) yields a degenerate
    anchor, which only occurs in contrived offline calls."""
    partition = [tuple(tree_roi)] if tree_roi is not None else [(0.0, 0.0, 0.0, 0.0)]
    return policy_heuristic.choose(phi, partition, cfg)


# ----------------------- prompt body -----------------------

def _prompts_tried(history):
    """Distill the history into the ordered list of prompts already tried and what
    each yielded, so the refinement trajectory is legible at a glance (in addition to
    the full history, which is also included)."""
    tried = []
    for rec in _history_records(history):
        x = rec.get("x", {})
        if "prompt" in x:
            tried.append({
                "prompt": x["prompt"],
                "conf": x.get("conf"),
                "n_new": rec.get("y", {}).get("n_new"),
            })
    return tried


def _build_body(phi, history, graph, cfg, image, tree_roi):
    """The JSON state + action menu shown to the VLM (images sent separately)."""
    target = getattr(cfg, "target_prompt", "green fruit")

    # Menu: refine/stop. "refine" is offered only when the image is present (the VLM
    # needs to see the current boxes to judge the last prompt and revise it). Angle
    # brackets signal "substitute a value" so a weak model doesn't echo the literal.
    menu = {"stop": {"estimate_name": "<N_obs|N_supp|N_cons>"}}
    if image is not None:
        lo = getattr(cfg, "refine_conf_min", 0.30)
        hi = getattr(cfg, "refine_conf_max", 0.70)
        menu = {
            "refine": {"prompt": "<1-2 adjectives + noun>",
                       "threshold": "<%.2f-%.2f>" % (lo, hi)},
            **menu,
        }

    return {
        "initial_concept": target,                       # the seed concept t_0
        "image_size": list(image.size) if image is not None else None,  # [w, h] px
        "tree_roi": [float(v) for v in tree_roi] if tree_roi is not None else None,
        "prompts_tried": _prompts_tried(history),        # the refinement trajectory
        "history": _history_records(history),            # the complete past (x_1^t, y_1^t)
        "phi": _compact_phi(phi),
        "action_menu": menu,
    }


def _build_messages(phi, history, graph, cfg, image=None, tree_roi=None):
    body = _build_body(phi, history, graph, cfg, image, tree_roi)
    text = ("Choose the next SAM3 text prompt to try (or stop).\n" + json.dumps(body)
            + "\nReply with ONLY {\"action\":..., \"args\":...}.")
    if image is not None:
        from agent.overlay import render_overlay  # lazy: only the image path needs PIL draw
        overlay = render_overlay(image, graph, sensed_rois=None, tree_roi=tree_roi)
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

def _validate_prompt(prompt, cfg):
    """A refine prompt is legal iff it is a string of refine_min_words..refine_max_words
    words, each a plain word (letters, with an optional internal hyphen). Returns the
    normalized "w1 w2 ..." string (single-spaced) or None. This caps the prompt to a
    short noun phrase because SAM3 mean-pools text tokens into one query vector, so a
    long descriptive sentence dilutes the query (see inference.verify_box_semantics)."""
    prompt = _coerce_json_string(prompt)
    if not isinstance(prompt, str):
        return None
    words = prompt.strip().split()
    lo = getattr(cfg, "refine_min_words", 1)
    hi = getattr(cfg, "refine_max_words", 3)
    if not (lo <= len(words) <= hi):
        return None
    for w in words:
        if not w.replace("-", "").isalpha():
            return None
    return " ".join(words)


def _validate_threshold(threshold, cfg):
    """Clamp the VLM threshold to [refine_conf_min, refine_conf_max]; fall back to
    refine_conf_default when it is missing or non-numeric. Always returns a float."""
    lo = getattr(cfg, "refine_conf_min", 0.30)
    hi = getattr(cfg, "refine_conf_max", 0.70)
    default = getattr(cfg, "refine_conf_default", 0.40)
    threshold = _coerce_json_string(threshold)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        return float(default)
    return float(min(hi, max(lo, threshold)))


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

    if name == "refine":
        # A refine needs the image (the VLM must see the boxes) and the tree ROI (the
        # fixed sensing region). The refined text is grounded by SAM3; the region and
        # threshold are the only sensor knobs the prompt sets -- never a box.
        if image is None or tree_roi is None:
            return None
        prompt = _validate_prompt(args.get("prompt"), cfg)
        if prompt is None:
            return None
        conf = _validate_threshold(args.get("threshold"), cfg)
        region = tuple(float(v) for v in tree_roi)
        return QueryA(region=region, prompt=prompt, conf=conf)

    if name == "stop":
        # Never terminate before any candidate has been registered: a stop on an
        # empty graph reports zero having sensed nothing. Fall back so the heuristic
        # picks a sensing action instead.
        if len(graph.nodes) == 0:
            return None
        est = args.get("estimate_name", "N_cons")
        if est not in _ESTIMATORS:
            return None
        return StopA(estimate_name=est)

    return None
