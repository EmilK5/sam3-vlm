"""
agent/policy_heuristic.py

Non-visual value-of-information heuristic policy (the baseline controller).
choose() ranks every concrete candidate action by an approximate VoI-per-cost
ratio using cheap predicted-Delta-U proxies, then returns the argmax. It stops
when the discovery curve has saturated and uncertainty is low.

It consumes only the belief summary phi (from belief.summarize) and the current
image partition -- no image/crop observations (that is the VLM policy's job).

Config knobs read via getattr (documented defaults; not yet in config):
    c0=1.0          base orchestration cost keeping the ratio finite
    target_prompt="green fruit"   the concept string for Query/TileQuery
    small_area=1024 median-area threshold below which tiling is boosted
    tau_w=0.5       low-support threshold for choosing verification
"""

from agent.actions import QueryA, TileQueryA, SubdivideA, VerifyA, StopA


def _in_region(center, region) -> bool:
    cx, cy = center
    x1, y1, x2, y2 = region
    return x1 <= cx < x2 and y1 <= cy < y2


def _region_area(region) -> float:
    x1, y1, x2, y2 = region
    return max(1.0, (x2 - x1) * (y2 - y1))


def _median(values):
    if not values:
        return 0.0
    return sorted(values)[len(values) // 2]


def choose(phi, partition, cfg):
    """Pick the next action by VoI-per-cost, or StopA when saturated and low-U.

    Per-action value proxies (all cheap, no model calls):
      QueryA(r)    = lambda_D * (#fresh k==1 candidates in r)          # recent local discovery
                   + lambda_U * (region instability sum)              # U(b~; r)
                   + lambda_C * (|G(r)| / area(r))                     # crowding C_t(r)
      TileQueryA   = the global version of the above, doubled when the median
                     candidate area is small (tiling helps small objects)
      SubdivideA(r)= fraction of r's candidates with k==1              # region still churning
      VerifyA      = sum of instability over unresolved / low-w nodes
    Each value is divided by (c0 + cost_sense(action)).
    """
    lam = cfg.lambdas
    m = cfg.window_m
    c0 = getattr(cfg, "c0", 1.0)
    prompt = getattr(cfg, "target_prompt", "green fruit")
    small_area = getattr(cfg, "small_area", 1024.0)
    tau_w = getattr(cfg, "tau_w", 0.5)
    conf = cfg.conf

    # --- stopping rule: saturated discovery AND low uncertainty ---
    D = phi["D"]
    recent = (sum(D[-m:]) / min(len(D), m)) if D else 0.0
    saturated = len(D) >= m and recent <= cfg.delta_disc
    if saturated and phi["U"] <= cfg.delta_U:
        return StopA(estimate_name="N_cons")

    K = phi["K"]
    centers, k, delta, s, w = phi["centers"], phi["k"], phi["delta"], phi["s"], phi["w"]
    cls, ids, area = phi["classification"], phi["ids"], phi["area"]

    def instability(i):
        return 1.0 / (1 + k[i]) + lam["alpha_delta"] * delta[i] + lam["alpha_s"] * (1.0 - s[i])

    scored = []  # (voi_per_cost, action)

    # --- region-restricted Query and Subdivide ---
    for r in partition:
        region = tuple(r)
        idxs = [i for i in range(K) if _in_region(centers[i], region)]
        local_disc = sum(1 for i in idxs if k[i] == 1)
        region_inst = sum(instability(i) for i in idxs)
        crowding = len(idxs) / _region_area(region)
        v_query = lam["lambda_D"] * local_disc + lam["lambda_U"] * region_inst + lam["lambda_C"] * crowding
        scored.append((v_query / (c0 + cfg.c_sam), QueryA(region=region, prompt=prompt, conf=conf)))

        frac_fresh = (sum(1 for i in idxs if k[i] == 1) / len(idxs)) if idxs else 0.0
        scored.append((frac_fresh / (c0 + 0.0), SubdivideA(region=region)))  # no model call

    # --- global TileQuery, boosted for small objects ---
    global_disc = sum(1 for i in range(K) if k[i] == 1)
    global_inst = sum(instability(i) for i in range(K))
    total_area = sum(_region_area(tuple(r)) for r in partition) or 1.0
    v_tile = lam["lambda_D"] * global_disc + lam["lambda_U"] * global_inst + lam["lambda_C"] * (K / total_area)
    if area and _median(area) < small_area:
        v_tile *= 2.0
    tile_cost = cfg.c_tile * 4  # ~4 tiles (proxy pending real tile count from execution)
    scored.append((v_tile / (c0 + tile_cost), TileQueryA(prompt=prompt, conf=conf)))

    # --- VerifyA over unresolved / low-support-score nodes ---
    verify_idxs = [i for i in range(K) if cls[i] == "unresolved" or w[i] < tau_w]
    if verify_idxs:
        v_verify = sum(instability(i) for i in verify_idxs)
        verify_cost = cfg.c_verify * len(verify_idxs)
        node_ids = [ids[i] for i in verify_idxs]
        scored.append((v_verify / (c0 + verify_cost), VerifyA(node_ids=node_ids)))

    if not scored:
        return StopA(estimate_name="N_cons")
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[0][1]
