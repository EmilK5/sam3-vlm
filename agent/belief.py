"""
agent/belief.py

Zero-shot belief summary over the candidate graph. Pure functions (no models),
implementing the formulas from the proposal:

  - support_score  (w_i)  -- policy-facing candidate reliability statistic
  - uncertainty    (U)    -- scalar zero-shot uncertainty over the belief summary
  - DiscoveryCurve (D_t)  -- per-step new-candidate counts + saturation test
  - summarize      (phi)  -- json-serializable dict the policies consume

All quantities read the belief fields set in step 2.1 (support k, jitter Delta,
area A) plus the detection confidence used as the aggregate SAM3 score s_bar.
Weights come from cfg.lambdas and cfg.window_m.
"""

import math


def support_score(node, cfg) -> float:
    """w_i = s_bar + lambda_k*log(1+k) - lambda_Delta*Delta - lambda_A*1[A outside plausible].

    The plausible-area range is read from cfg.area_min / cfg.area_max if present;
    with the current config those are absent, so the indicator is 0 (the lambda_A
    term is inert until area bounds are added to config).
    """
    lam = cfg.lambdas
    s_bar = node.scores["detection_confidence"]
    k = node.support
    delta = node.jitter

    area_min = getattr(cfg, "area_min", 0.0)
    area_max = getattr(cfg, "area_max", float("inf"))
    implausible = 0.0 if (area_min <= node.area <= area_max) else 1.0

    return (
        s_bar
        + lam["lambda_k"] * math.log(1 + k)
        - lam["lambda_delta"] * delta
        - lam["lambda_A"] * implausible
    )


def uncertainty(graph, discovery, cfg) -> float:
    """U(b~) = lambda_D * mean_recent(m) + lambda_S * sum_{non-spurious}(
        1/(1+k) + alpha_Delta*Delta + alpha_s*(1 - s_bar) ).
    """
    lam = cfg.lambdas
    coverage = lam["lambda_D"] * discovery.mean_recent(cfg.window_m)

    instability = 0.0
    for node in graph.nodes.values():
        if node.classification == "spurious":
            continue
        k = node.support
        delta = node.jitter
        s_bar = node.scores["detection_confidence"]
        instability += 1.0 / (1 + k) + lam["alpha_delta"] * delta + lam["alpha_s"] * (1.0 - s_bar)

    return coverage + lam["lambda_S"] * instability


class DiscoveryCurve:
    """The discovery curve D_t = (n_1, ..., n_t): new candidates registered per step."""

    def __init__(self):
        self.counts = []

    def append(self, n_new: int):
        self.counts.append(int(n_new))

    def mean_recent(self, m: int) -> float:
        """Mean of the last m entries (or all if fewer). 0.0 when empty."""
        if not self.counts:
            return 0.0
        window = self.counts[-m:]
        return sum(window) / len(window)

    def saturated(self, m: int, delta: float) -> bool:
        """True once a full window of m steps averages at or below delta."""
        return len(self.counts) >= m and self.mean_recent(m) <= delta

    def as_list(self) -> list:
        return list(self.counts)


def summarize(graph, discovery, budget, cfg) -> dict:
    """Build the policy input phi (json-serializable).

    Lists w/s/k/delta cover every node (K_t) in graph order. tiling_status is
    derived from candidate signatures (any tiled-mode detection so far).
    """
    nodes = list(graph.nodes.values())
    tiling_status = any(":tiled:" in sig for n in nodes for sig in n.signatures)

    return {
        "K": len(nodes),
        "n_t": discovery.counts[-1] if discovery.counts else 0,
        "D": discovery.as_list(),
        "w": [round(support_score(n, cfg), 3) for n in nodes],
        "s": [round(n.scores["detection_confidence"], 3) for n in nodes],
        "k": [n.support for n in nodes],
        "delta": [round(n.jitter, 3) for n in nodes],
        "tiling_status": tiling_status,
        "remaining_budget": budget,
    }
