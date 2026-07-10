"""
agent/history.py

EpisodeHistory: the running record of an active-perception episode -- the full
past (x_1^t, y_1^t) that the v2 policy is shown at each step
(docs/active_perception_formulation.md §6: the amortized policy is conditioned on
the whole history). One plain dict per executed action, JSON-ready, appended in
execution order.

Record shape:
    {
      "t":  int,                       # 1-based record index (one per log line)
      "x":  {"action": str, ...},      # the sensing action + its params
      "y":  {                          # the observation the action produced
        "n_new":     int,
        "new_nodes": [{"id", "box", "conf", "class"}, ...],
        "totals":    {"K": int, "N_obs": int},
      },
    }

All boxes are global-frame xyxy pixels. This module has no torch/model
dependency; it consumes only OrchardNode-shaped objects and plain dataclasses.
"""

import dataclasses


def action_params(action) -> dict:
    """Serialize a sensing-action dataclass to {"action": <type>, ...fields}.

    Tuples/lists (e.g. an xyxy region) are copied to plain lists so the result is
    JSON-ready. Used for the QueryA/LookROIA/StopA records; the synthetic
    bootstrap records (canopy_roi / leaf_map / global_pass) are built by hand.
    """
    d = {"action": type(action).__name__}
    for f in dataclasses.fields(action):
        v = getattr(action, f.name)
        d[f.name] = list(v) if isinstance(v, (tuple, list)) else v
    return d


def node_summary(node) -> dict:
    """The minimal per-node view recorded inside an observation (JSON-ready)."""
    return {
        "id": node.id,
        "box": [float(c) for c in node.box],
        "conf": float(node.scores.get("detection_confidence", 0.0)),
        "class": node.classification,
    }


class EpisodeHistory:
    """An append-only list of JSON-ready {t, x, y} action/observation records."""

    def __init__(self):
        self.records = []

    def append(self, x: dict, y: dict) -> dict:
        """Append one record (stamped with the next 1-based t) and return it."""
        rec = {"t": len(self.records) + 1, "x": x, "y": y}
        self.records.append(rec)
        return rec

    def as_list(self) -> list:
        """The records as a plain list (a shallow copy of the backing list)."""
        return list(self.records)

    def __len__(self):
        return len(self.records)
