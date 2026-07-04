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

    def center(box):
        return [round((box[0] + box[2]) / 2.0, 1), round((box[1] + box[3]) / 2.0, 1)]

    return {
        "K": len(nodes),
        "n_t": discovery.counts[-1] if discovery.counts else 0,
        "D": discovery.as_list(),
        "U": round(uncertainty(graph, discovery, cfg), 3),
        "ids": [n.id for n in nodes],
        "centers": [center(n.box) for n in nodes],
        "area": [round(n.area, 1) for n in nodes],
        "classification": [n.classification for n in nodes],
        "w": [round(support_score(n, cfg), 3) for n in nodes],
        "s": [round(n.scores["detection_confidence"], 3) for n in nodes],
        "k": [n.support for n in nodes],
        "delta": [round(n.jitter, 3) for n in nodes],
        "tiling_status": tiling_status,
        "remaining_budget": budget,
    }


# ----------------------- zero-shot count estimators -----------------------

def _countable_nodes(graph):
    """Nodes that count toward the target: verified fruit + still-unresolved
    candidates (leaf / spurious are excluded from all count estimators)."""
    return [n for n in graph.nodes.values() if n.classification in ("fruit", "unresolved")]


def count_estimates(graph, cfg) -> dict:
    """The three zero-shot count estimators (proposal §"Zero-Shot Count Estimates").

        N_obs  = |countable candidates|
        N_supp = #{ w_i >= tau_w }
        N_cons = #{ k_i >= k_min OR s_bar_i >= tau_high }

    Thresholds are read from cfg via getattr with documented defaults (config has
    no such fields yet; a later step should promote them): tau_w=0.5, k_min=2,
    tau_high=0.5.
    """
    tau_w = getattr(cfg, "tau_w", 0.5)
    k_min = getattr(cfg, "k_min", 2)
    tau_high = getattr(cfg, "tau_high", 0.5)

    nodes = _countable_nodes(graph)
    n_supp = sum(1 for n in nodes if support_score(n, cfg) >= tau_w)
    n_cons = sum(1 for n in nodes
                 if n.support >= k_min or n.scores["detection_confidence"] >= tau_high)
    return {"N_obs": len(nodes), "N_supp": n_supp, "N_cons": n_cons}
