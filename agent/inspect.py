"""
agent/inspect.py

VLM scene inspection z_t: a pre-decision, non-graph-updating assessment of the
image (or a crop) used by the orchestrator to bias its next sensing action.
Inspection never adds candidates -- it only produces the structured summary z.

inspect_scene() mirrors QwenOracle's strict-parse / retry / neutral-fallback
pattern. should_inspect() implements the fixed inspection protocol (proposal
§"Approximate VLM Orchestration Policy").
"""

import base64
import io
import json
import logging

logger = logging.getLogger(__name__)

_DENSITY = {"sparse", "medium", "dense"}
_SCALE = {"large", "medium", "small"}
_OCCLUSION = {"low", "medium", "high"}
_RECOMMEND = {"query", "tile", "subdivide", "verify", "stop"}

# Neutral fallback assessment (used on unparseable responses); recommend="query"
# keeps the episode moving without a confident, possibly-wrong assessment.
_NEUTRAL_Z = {
    "target_present": True,
    "density": "medium",
    "object_scale": "medium",
    "occlusion": "medium",
    "recommend": "query",
    "notes": "fallback",
}

# Window (in passes) over which "no new discoveries" counts as saturation for the
# inspection protocol. Mirrors the default cfg.window_m.
_SATURATION_WINDOW = 3

SYSTEM_PROMPT = (
    "You assess images for an object-counting system. Look at the image and judge "
    "the scene at a high level. Reply with ONLY a JSON object of the form "
    "{\"target_present\": true|false, \"density\": \"sparse|medium|dense\", "
    "\"object_scale\": \"large|medium|small\", \"occlusion\": \"low|medium|high\", "
    "\"recommend\": \"query|tile|subdivide|verify|stop\", \"notes\": \"<=200 chars\"} "
    "and nothing else."
)


def inspect_scene(image_or_crop, phi, oracle_cfg, client=None) -> dict:
    """One structured VLM call assessing the scene. Returns the z dict; on repeated
    parse failure returns a neutral z with recommend='query'.

    `client` may be injected for offline testing; otherwise an openai client is
    built lazily from oracle_cfg (keyless-friendly).
    """
    if client is None:
        client = _build_client(oracle_cfg)
    messages = _build_messages(image_or_crop, phi)

    for attempt in range(oracle_cfg.oracle_max_retries + 1):
        content = _request(client, oracle_cfg, messages)
        z = _parse_z(content)
        if z is not None:
            return z
        logger.warning("inspect_scene: unparseable response (attempt %d).", attempt + 1)

    logger.warning("inspect_scene: giving up after %d attempts; returning neutral z.",
                   oracle_cfg.oracle_max_retries + 1)
    return dict(_NEUTRAL_Z)


def _build_client(cfg):
    import os
    from openai import OpenAI  # lazy: not needed for tests
    return OpenAI(base_url=cfg.oracle_base_url, api_key=os.environ.get("QWEN_API_KEY", "EMPTY"))


def _build_messages(image, phi):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    image_url = f"data:image/png;base64,{b64}"

    context = {k: phi.get(k) for k in ("K", "n_t", "tiling_status", "remaining_budget")}
    text = (
        "Assess this scene for counting the target objects.\n"
        f"Current counting state: {json.dumps(context)}\n"
        "Reply with ONLY the JSON schema."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]},
    ]


def _request(client, cfg, messages) -> str:
    response = client.chat.completions.create(
        model=cfg.oracle_model_name, temperature=cfg.oracle_temperature, messages=messages,
    )
    return response.choices[0].message.content


def _parse_z(content):
    """Parse + strictly validate a response into a z dict, or None if invalid."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    if not isinstance(data.get("target_present"), bool):
        return None
    if data.get("density") not in _DENSITY:
        return None
    if data.get("object_scale") not in _SCALE:
        return None
    if data.get("occlusion") not in _OCCLUSION:
        return None
    if data.get("recommend") not in _RECOMMEND:
        return None

    return {
        "target_present": data["target_present"],
        "density": data["density"],
        "object_scale": data["object_scale"],
        "occlusion": data["occlusion"],
        "recommend": data["recommend"],
        "notes": str(data.get("notes", ""))[:200],
    }


def _saturated(phi) -> bool:
    """Discovery-saturation proxy: no new candidates in the last few passes."""
    D = phi.get("D", [])
    return len(D) >= _SATURATION_WINDOW and sum(D[-_SATURATION_WINDOW:]) == 0


def should_inspect(t, phi, last_z) -> bool:
    """Fixed inspection protocol. Inspect when:
      1. t == 1 (once at the start of the episode);
      2. a tiling/subdivide decision is pending -- proxied by the last assessment
         recommending "tile" or "subdivide" (should_inspect has no action scores);
      3. discovery has saturated while the scene still looks dense.
    """
    if t == 1:
        return True
    if last_z is not None and last_z.get("recommend") in ("tile", "subdivide"):
        return True
    if _saturated(phi) and last_z is not None and last_z.get("density") == "dense":
        return True
    return False
