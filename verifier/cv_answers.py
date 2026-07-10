"""
verifier/cv_answers.py

Deterministic classical-CV answers for FM+V-IP queries that ask about low-level
appearance (color / shape / texture). These need no model call at all, so routing
such queries here (via a query's `cv_check`, see queries.py + oracle.RouterOracle)
keeps the far more expensive VLM oracle for the genuinely semantic questions.

Features on the 256px crop:
  - hue_fraction(lo, hi) : fraction of colored pixels whose HSV hue is in [lo, hi]
                           (color). OpenCV hue range is 0-179.
  - circularity          : 4*pi*area / perimeter^2 of the largest Otsu-thresholded
                           contour (shape); ~1 for a disc, small for a thin leaf.
  - edge_density         : fraction of pixels with a strong Laplacian response
                           (texture); ~0 for smooth sky, higher for foliage/soil.

answer(crop_pil, cv_check) -> {-1, 0, +1} via a yes-above / no-below /
unsure-between threshold pair. cv_check may set "direction": "low" to flip the
sense (a "yes" at the LOW end of the feature), which is exactly the default
high-sense with +1/-1 swapped. cv2 / numpy / PIL only -- no model, no torch.
"""

import cv2
import numpy as np

# Feature names a cv-routed query may name in its cv_check.
CV_FEATURES = {"hue_fraction", "circularity", "edge_density"}


def validate_cv_check(cv_check):
    """Raise ValueError unless cv_check is a well-formed feature+threshold spec."""
    if not isinstance(cv_check, dict):
        raise ValueError(f"cv_check must be a dict; got {type(cv_check).__name__}.")
    feature = cv_check.get("feature")
    if feature not in CV_FEATURES:
        raise ValueError(f"cv_check 'feature' must be one of {sorted(CV_FEATURES)}; got {feature!r}.")
    direction = cv_check.get("direction", "high")
    if direction not in ("high", "low"):
        raise ValueError(f"cv_check 'direction' must be 'high' or 'low'; got {direction!r}.")
    for key in ("yes_above", "no_below"):
        v = cv_check.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"cv_check '{key}' must be a number; got {v!r}.")
    if cv_check["no_below"] > cv_check["yes_above"]:
        raise ValueError(
            f"cv_check needs no_below <= yes_above; got "
            f"no_below={cv_check['no_below']} > yes_above={cv_check['yes_above']}."
        )
    if feature == "hue_fraction":
        for key in ("hue_lo", "hue_hi"):
            v = cv_check.get(key)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0 <= v <= 179):
                raise ValueError(f"cv_check '{key}' must be a number in [0, 179]; got {v!r}.")
        if cv_check["hue_lo"] > cv_check["hue_hi"]:
            raise ValueError("cv_check needs hue_lo <= hue_hi for hue_fraction.")


# ----------------------- features -----------------------

def hue_fraction(rgb: np.ndarray, lo, hi, sat_min=40, val_min=40) -> float:
    """Fraction of sufficiently-colored pixels whose hue lies in [lo, hi]
    (OpenCV HSV, hue 0-179). The saturation/value floors ignore near-gray
    pixels whose hue is meaningless."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (h >= lo) & (h <= hi) & (s >= sat_min) & (v >= val_min)
    return float(mask.mean())


def circularity(rgb: np.ndarray) -> float:
    """4*pi*area / perimeter^2 of the largest Otsu-thresholded contour, in [0, 1]
    (1 = perfect circle). Returns 0 when no contour is found."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    if area <= 0.0 or perim <= 0.0:
        return 0.0
    return float(min(1.0, 4.0 * np.pi * area / (perim * perim)))


def edge_density(rgb: np.ndarray, edge_thresh=20.0) -> float:
    """Fraction of pixels whose absolute Laplacian response exceeds edge_thresh
    (a texture proxy), in [0, 1]."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float((np.abs(lap) > edge_thresh).mean())


def feature_value(crop_pil, cv_check) -> float:
    """Compute the cv_check's feature on the crop (RGB). Assumes a validated cv_check."""
    rgb = np.asarray(crop_pil.convert("RGB"))
    feature = cv_check["feature"]
    if feature == "hue_fraction":
        return hue_fraction(rgb, cv_check["hue_lo"], cv_check["hue_hi"],
                            cv_check.get("sat_min", 40), cv_check.get("val_min", 40))
    if feature == "circularity":
        return circularity(rgb)
    if feature == "edge_density":
        return edge_density(rgb, cv_check.get("edge_thresh", 20.0))
    raise ValueError(f"unknown cv_check feature {feature!r}")  # unreachable if validated


# ----------------------- answer -----------------------

def answer(crop_pil, cv_check) -> int:
    """Answer a cv-routed query on a crop: +1 (yes) / -1 (no) / 0 (unsure).

    direction "high" (default): value >= yes_above -> +1; value <= no_below -> -1.
    direction "low": the same thresholds with the answer sign flipped, so a "yes"
    is at the LOW end (e.g. "is it leaf-like?" -> low circularity)."""
    value = feature_value(crop_pil, cv_check)
    yes_above = float(cv_check["yes_above"])
    no_below = float(cv_check["no_below"])
    if cv_check.get("direction", "high") == "low":
        if value <= no_below:
            return 1
        if value >= yes_above:
            return -1
        return 0
    if value >= yes_above:
        return 1
    if value <= no_below:
        return -1
    return 0
