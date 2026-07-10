"""
agent/policy_heuristic.py

Minimal non-visual look/stop fallback policy (v2). choose() stops when the
discovery curve has saturated and uncertainty is low; otherwise it proposes a
LookROIA on one cell of a fixed 2x2 grid over the episode's anchor region
(partition[0], the tree ROI or full frame).

Cell choice: cells are ranked by how few registered candidates they contain
(fewest first -- the least-sensed area is where predictive information is
highest), and the pick rotates through that ranking with the number of sensing
passes taken so far (len(phi["D"])), so consecutive calls never re-propose the
same cell and trip the duplicate-ROI guard in actions._execute_look.

It consumes only the belief summary phi (from belief.summarize) and the current
partition -- no image/crop observations (that is the VLM policy's job). All
boxes are global-frame xyxy pixels.
"""

from agent.actions import LookROIA, StopA


def _in_region(center, region) -> bool:
    cx, cy = center
    x1, y1, x2, y2 = region
    return x1 <= cx < x2 and y1 <= cy < y2


def _grid_cells(region) -> list:
    """The four 2x2 grid cells of an xyxy region (TL, TR, BL, BR)."""
    x1, y1, x2, y2 = region
    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0
    return [
        (x1, y1, mx, my),
        (mx, y1, x2, my),
        (x1, my, mx, y2),
        (mx, my, x2, y2),
    ]


def choose(phi, partition, cfg):
    """Pick LookROIA on the emptiest grid cell (rotating), or StopA when done.

    Stopping rule (unchanged from the validated baseline): discovery saturated
    over cfg.window_m at cfg.delta_disc AND uncertainty U <= cfg.delta_U, and
    never on an empty graph (an empty graph passes both tests vacuously, which
    would end the episode having detected nothing).
    """
    m = cfg.window_m
    D = phi["D"]
    recent = (sum(D[-m:]) / min(len(D), m)) if D else 0.0
    saturated = len(D) >= m and recent <= cfg.delta_disc
    if phi["K"] > 0 and saturated and phi["U"] <= cfg.delta_U:
        return StopA(estimate_name="N_cons")

    anchor = tuple(partition[0]) if partition else (0.0, 0.0, 0.0, 0.0)
    cells = _grid_cells(anchor)
    counts = [sum(1 for c in phi["centers"] if _in_region(c, cell)) for cell in cells]
    # Rank cells emptiest-first (index breaks ties deterministically), then
    # rotate the pick with the sensing-pass count so repeats cycle the grid.
    order = sorted(range(len(cells)), key=lambda j: (counts[j], j))
    pick = order[len(D) % len(cells)]
    return LookROIA(region=cells[pick])
