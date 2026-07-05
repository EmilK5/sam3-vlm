"""
verifier/verify.py

Crop extraction and the FM+V-IP verify API. Given a candidate box, an oracle,
and a query set, this classifies the crop into target / distractor / spurious
and returns a short, human-readable query->answer chain that justifies the
verdict (interpretable-by-design verification).

Coordinate frame: boxes are global-frame xyxy pixels; image_np is an RGB
HxWx3 uint8 array (as produced by np.array(image_pil) elsewhere).
"""

import dataclasses
import logging

import cv2
import numpy as np
from PIL import Image

from verifier import vip
from verifier.queries import QuerySet

logger = logging.getLogger(__name__)

_ANSWER_WORD = {1: "yes", -1: "no", 0: "unsure"}


def extract_crop(image_np: np.ndarray, box, scale: float, size: int) -> Image.Image:
    """Center-scale a box by `scale`, clamp to the image, and resample to
    (size, size) with cv2 INTER_CUBIC. Mirrors the geometry in
    inference.verify_box_semantics. Returns an RGB PIL image.
    """
    xmin_orig, ymin_orig, xmax_orig, ymax_orig = box
    w = xmax_orig - xmin_orig
    h = ymax_orig - ymin_orig
    cx = xmin_orig + (w / 2.0)
    cy = ymin_orig + (h / 2.0)

    new_w = w * scale
    new_h = h * scale

    img_h, img_w = image_np.shape[:2]
    xmin = int(max(0, cx - (new_w / 2.0)))
    ymin = int(max(0, cy - (new_h / 2.0)))
    xmax = int(min(img_w, cx + (new_w / 2.0)))
    ymax = int(min(img_h, cy + (new_h / 2.0)))

    # Guard degenerate / zero-area regions so the resize never fails.
    if xmax <= xmin:
        xmin = min(xmin, max(0, img_w - 1))
        xmax = min(img_w, xmin + 1)
    if ymax <= ymin:
        ymin = min(ymin, max(0, img_h - 1))
        ymax = min(img_h, ymin + 1)

    crop_np = image_np[ymin:ymax, xmin:xmax]
    if crop_np.size == 0:
        crop_np = image_np[:1, :1]

    crop_resized = cv2.resize(crop_np, (size, size), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(crop_resized)


def _single_query_view(query_set: QuerySet, m: int) -> QuerySet:
    """A QuerySet containing only query m, preserving classes/epsilon/names."""
    return QuerySet(
        concept=query_set.concept,
        classes=query_set.classes,
        epsilon=query_set.epsilon,
        queries=[query_set.queries[m]],
        class_names=query_set.class_names,
    )


def _sequential_ip(crop, oracle, query_set, table, prior, stop_eps, max_q):
    """Greedy IP that queries the oracle one selected query at a time.

    Identical selection logic to vip.run_ip, but each chosen query is answered
    by a separate oracle call on a single-query view (ablation / expensive mode).
    Returns (posterior, chain, answers, n_oracle_calls).
    """
    M = len(query_set.queries)
    answers = np.zeros(M, dtype=np.int8)
    S = []
    chain = []
    n_calls = 0

    while len(S) < max_q:
        post = vip.posterior(answers, S, table, prior)
        if post.max() >= 1.0 - stop_eps:
            break

        best_m, best_gain = None, 0.0
        for m in range(M):
            if m in S:
                continue
            g = vip.info_gain(m, S, answers, table, prior)
            if g > best_gain:
                best_gain, best_m = g, m

        if best_m is None or best_gain <= 1e-12:
            break

        a = int(oracle.answer_batch(crop, _single_query_view(query_set, best_m))[0])
        n_calls += 1
        answers[best_m] = a
        S.append(best_m)
        chain.append((best_m, a))

    post = vip.posterior(answers, S, table, prior)
    return post, chain, answers, n_calls


def verify_candidate(image_np, box, oracle, query_set, cfg) -> dict:
    """Classify a candidate crop into target/distractor/spurious via FM+V-IP.

    Returns a dict:
        verdict        : class name at the MAP index (e.g. "target").
        p_target       : P(v = target | answers).
        posterior      : full posterior over classes, as a list.
        chain          : [{"q": query text, "a": "yes"|"no"|"unsure"}, ...].
        n_oracle_calls : 1 in batched mode, chain length in sequential mode.
    """
    crop = extract_crop(image_np, box, cfg.crop_scale, cfg.crop_size)
    # cfg.vip_epsilon = None defers to the query set's epsilon; a float overrides it
    # (the "raise epsilon" knob from the plan's risk mitigations).
    eps_override = getattr(cfg, "vip_epsilon", None)
    if eps_override is not None:
        query_set = dataclasses.replace(query_set, epsilon=float(eps_override))
    table = vip.likelihood_table(query_set)
    K = len(query_set.classes)
    prior = np.full(K, 1.0 / K)

    if cfg.answer_mode == "sequential":
        posterior, chain_raw, _answers, n_calls = _sequential_ip(
            crop, oracle, query_set, table, prior, cfg.vip_stop, cfg.vip_max_queries
        )
        verdict_idx = int(posterior.argmax())
    else:  # batched (default)
        answers = oracle.answer_batch(crop, query_set)
        result = vip.run_ip(answers, table, prior,
                            stop_eps=cfg.vip_stop, max_q=cfg.vip_max_queries)
        posterior = result["posterior"]
        chain_raw = result["chain"]
        verdict_idx = result["verdict_idx"]
        n_calls = 1

    classes = query_set.classes
    target_idx = classes.index("target") if "target" in classes else 0
    chain = [
        {"q": query_set.queries[qi].text, "a": _ANSWER_WORD[av]}
        for qi, av in chain_raw
    ]

    return {
        "verdict": classes[verdict_idx],
        "p_target": float(posterior[target_idx]),
        "posterior": [float(x) for x in posterior],
        "chain": chain,
        "n_oracle_calls": int(n_calls),
    }
