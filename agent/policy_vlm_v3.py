"""
agent/policy_vlm_v3.py

Phase 9 prompt-refinement policy. Where the v2 policy (policy_vlm.py) locks the
SAM3 text prompt to a system-level constant and lets the VLM choose only WHERE to
look, this policy does the opposite: the region is FIXED to the canopy tree ROI
(one global pass per step) and the VLM chooses WHAT to ask SAM3 -- a refined text
prompt (1-2 adjectives + a noun) plus the detection threshold.

Why the region is FIXED to the tree ROI (and LookROIA is banned here):
  * A pass over the whole tree ROI keeps every already-found positive exemplar in
    frame, so exemplar priming is never lost. SAM3 visual exemplars are welded to
    the query frame (roi_align'd against that image's own features); there is no
    cross-image exemplar conditioning, so a sub-ROI crop (LookROIA) would silently
    drop the out-of-crop exemplars and is intentionally NOT available in this arm.
  * Fixing the region isolates the causal effect of prompt refinement -- the only
    thing that varies step to step is the wording, which is exactly the hypothesis
    under test (does the VLM proposing better descriptive terms improve SAM3's r/p?).

Recall parity: the region is fixed but a pass may be TILED (cfg.refine_tiling) over
the tree ROI, which is the recall mechanism the generic cascade uses -- tiling is
NOT a sub-ROI zoom (the region is unchanged), so it does not reintroduce LookROIA.

Grounding guarantee (CLAUDE.md #3): the VLM emits TEXT, never a box. SAM3 grounds
the prompt into detections; no VLM-supplied box ever becomes a candidate. The
prompt only parameterizes a SAM3 query (region + threshold + tiled-or-not).

Menu: {"refine", "stop"}. A refine maps to a QueryA over the tree ROI (tiled per
cfg.refine_tiling) with the validated NEW prompt + clamped threshold. Any parse/
validation failure -- including a repeated prompt -- makes the arm STOP; it never
falls back to a sensing action the VLM did not choose (in particular never LookROIA).

Shares the request plumbing (client build, request, image encode, belief compaction,
tree-ROI resolution, JSON coercion) with policy_vlm to stay behaviorally identical
where the two overlap; only the action menu, validation, and body differ.
"""

import json
import logging

from agent.actions import QueryA, StopA
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
    "\"round fruit\", \"yellow citrus\"), and a detection threshold. Your prompt MUST "
    "be different from every prompt already tried. You are shown the raw frame, an "
    "overlay of the objects already found (colored by class), and, for each prompt "
    "already tried, how it performed: n_detections (total objects it found), n_new "
    "(previously-unseen objects it added), and n_redetected (objects it re-found that "
    "were ALREADY known -- matched by the dedup step). Judge a prompt by n_redetected, "
    "not just n_new: a wording that covers the target concept re-finds MOST of the "
    "already-known target objects AND adds new ones; a wording that re-detects few of "
    "the known objects is off-concept even if it added a couple. Look at the boxes, "
    "then pick a NEW wording that covers the concept better. Use a HIGHER detection "
    "threshold on later prompts (the system enforces a rising minimum) so a "
    "well-covered scene does not re-admit clutter. Keep proposing new "
    "wordings that might reveal target objects the earlier prompts missed; choose "
    "\"stop\" only when you believe no different wording would find more. The only "
    "actions are \"refine\" and \"stop\" -- never anything else. You never label or "
    "add objects yourself; SAM3 grounds your text into boxes. Reply with ONLY a JSON "
    "object {\"action\": <name>, \"args\": {...}} and nothing else."
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
    except Exception as exc:  # network / client error -> stop
        logger.warning("policy_vlm_v3: request failed (%s); stopping (no LookROIA fallback).", exc)
        return _fallback(cfg)

    tried = _tried_prompt_set(history)
    n_prior_refines = _n_prior_refines(history)
    action = _parse_and_validate(content, graph, cfg, image, tree_roi, tried, n_prior_refines)
    if action is None:
        logger.warning("policy_vlm_v3: invalid/repeated/illegal response; stopping.")
        return _fallback(cfg)
    return action


def _fallback(cfg):
    """LookROIA is banned in this arm. When the VLM returns no valid NEW prompt (a
    parse error, a repeat, or an out-of-menu action), the arm STOPS rather than emit
    any sensing action the VLM did not choose -- in particular it never falls back to
    the v2 look/stop heuristic, which would produce a sub-ROI LookROIA. Termination is
    then governed by the VLM, discovery saturation, or budget."""
    return StopA(estimate_name="N_obs")


# ----------------------- prompt body -----------------------

def _prompts_tried(history):
    """Distill the history into the ordered list of prompts already tried and how each
    performed -- n_detections (total found), n_new (previously-unseen added), and
    n_redetected (already-known objects re-found, the prompt-quality signal). This is
    the feedback that lets the VLM truly evaluate a wording, not just enumerate past
    ones; the full history is also included in the body."""
    tried = []
    for rec in _history_records(history):
        x = rec.get("x", {})
        if "prompt" in x:
            y = rec.get("y", {})
            tried.append({
                "prompt": x["prompt"],
                "conf": x.get("conf"),
                "n_detections": y.get("n_detections"),
                "n_new": y.get("n_new"),
                "n_redetected": y.get("n_redetected"),
            })
    return tried


def _tried_prompt_set(history):
    """The normalized set of prompts already tried (incl. the seed), so a refine that
    merely repeats an earlier wording is rejected -- each step must try something new."""
    return {" ".join(t["prompt"].split()).lower()
            for t in _prompts_tried(history) if t.get("prompt")}


def _n_prior_refines(history):
    """How many refine passes (VLM QueryA actions) have already run this episode. Used
    to raise the threshold floor per new prompt (the bootstrap global_pass/
    tiled_seed_pass records are relabeled, so only refines carry action == 'QueryA')."""
    return sum(1 for rec in _history_records(history)
               if rec.get("x", {}).get("action") == "QueryA")


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
            "refine": {"prompt": "<NEW 1-2 adjectives + noun, not already tried>",
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


def _validate_threshold(threshold, cfg, n_prior_refines=0):
    """Resolve the effective refine threshold.

    The floor RISES with each new prompt: floor = refine_conf_default +
    n_prior_refines * refine_conf_step, bounded to [refine_conf_min, refine_conf_max].
    The VLM may choose a stricter (higher) value but never a looser one -- so later
    prompts, which scan an already-well-covered scene, do not re-admit clutter at a low
    threshold. Missing/non-numeric -> the floor. Always returns a float."""
    lo = getattr(cfg, "refine_conf_min", 0.45)
    hi = getattr(cfg, "refine_conf_max", 0.85)
    default = getattr(cfg, "refine_conf_default", 0.50)
    step = getattr(cfg, "refine_conf_step", 0.05)
    floor = max(lo, min(hi, default + n_prior_refines * step))
    threshold = _coerce_json_string(threshold)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        return float(floor)
    return float(min(hi, max(floor, float(threshold))))


def _parse_and_validate(content, graph, cfg, image, tree_roi, tried_prompts=frozenset(),
                        n_prior_refines=0):
    """Return a validated action dataclass, or None on any parse/validation failure.

    tried_prompts: normalized prompts already tried this episode; a refine that
    repeats one is rejected so every step tries a genuinely new wording.
    n_prior_refines: number of refines already run -- raises the threshold floor."""
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
        if " ".join(prompt.split()).lower() in tried_prompts:
            return None                                  # must try a NEW wording each step
        conf = _validate_threshold(args.get("threshold"), cfg, n_prior_refines)
        region = tuple(float(v) for v in tree_roi)
        # Tiled over the tree ROI for recall parity with the generic cascade; still the
        # whole region (not a sub-ROI zoom). The prompt sets only text/threshold/tiling.
        tiling = bool(getattr(cfg, "refine_tiling", True))
        return QueryA(region=region, prompt=prompt, conf=conf, tiling=tiling)

    if name == "stop":
        # The bootstrap always senses before the policy loop, so an empty graph here
        # means the seed genuinely found nothing (empty orchard) -- stopping with 0 is
        # correct, and there is no LookROIA to fall back to.
        est = args.get("estimate_name", "N_cons")
        if est not in _ESTIMATORS:
            return None
        return StopA(estimate_name=est)

    return None
