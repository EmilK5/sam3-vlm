"""
agent/budget.py

Sensing cost meter. Counts model/orchestration calls during an episode and
reports the normalized cost (relative to one global SAM3 call) from the proposal's
compute model, using the cfg.costs ratios. This mirrors eval.metrics.normalized_cost
but as a live, incrementable counter that agent.actions.execute updates.
"""

import dataclasses


@dataclasses.dataclass
class CostMeter:
    n_sam: int = 0        # global SAM3 detection calls
    n_tile: int = 0       # tile-level SAM3 calls
    n_verify: int = 0     # FM+V-IP oracle calls (|C| batched, chain length sequential)
    n_inspect: int = 0    # VLM scene-inspection calls
    n_orch: int = 0       # lightweight orchestration decisions

    def total(self, cfg) -> float:
        """Normalized cost = n_sam + (c_tile/c_sam)*n_tile + (c_verify/c_sam)*n_verify
        + (c_inspect/c_sam)*n_inspect + (c_orch/c_sam)*n_orch."""
        c_sam = cfg.c_sam
        return (
            self.n_sam
            + (cfg.c_tile / c_sam) * self.n_tile
            + (cfg.c_verify / c_sam) * self.n_verify
            + (cfg.c_inspect / c_sam) * self.n_inspect
            + (cfg.c_orch / c_sam) * self.n_orch
        )

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)
