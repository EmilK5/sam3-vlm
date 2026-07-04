"""
eval/run_eval.py

Policy x verifier sweep over a dataset split, writing one CSV row per image.

Policies (this reduced version, per the plan's suggested-order "6.2 with
{oneshot, cascade} only" milestone):
    oneshot     - 1 global pass
    cascade     - 4 fixed global passes
    tiled       - 4 fixed tiled passes
    convergence - global passes until a pass discovers 0 new candidates
    heuristic / vlm - require the agent runner (Phase 4/5); not built -> raise.

Count estimators: N_obs = predicted (fruit / unresolved) node count. N_supp and
N_cons are placeholders equal to N_obs until belief.py (step 2.2) and support
tracking (step 2.1) land; run_eval will then call the real estimators. Cost is a
pass-count approximation (n_global / n_tile / n_verify) pending CostMeter (3.2).

The sweep is resume-safe (skips image/policy/verifier rows already in the CSV)
and uses a fixed numpy seed per image for reproducibility.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses

import numpy as np

from config import Config
from graph import OrchardGraph
from eval.datasets import load_split
from eval.matching import match, pool_recall, per_pass_pool_recall
from eval.metrics import mae, rmse, exact, normalized_cost

logger = logging.getLogger(__name__)

FIXED_PASS_POLICIES = {"cascade": 4, "tiled": 4}
AGENT_POLICIES = {"heuristic", "vlm"}
CONVERGENCE_MAX_PASSES = 8

CSV_FIELDS = [
    "image", "policy", "verifier", "N_gt", "N_obs", "N_supp", "N_cons",
    "precision", "recall", "f1", "pool_recall", "per_pass_pool_recall",
    "cost", "n_actions", "seconds",
]

# A candidate counts toward the target if it's verified fruit or still unresolved
# (unverified). Leaf/spurious are excluded from the count.
_TARGET_CLASSES = ("fruit", "unresolved")


def predicted_nodes(graph):
    """Nodes counted as target objects (fruit or unresolved)."""
    return [n for n in graph.nodes.values() if n.classification in _TARGET_CLASSES]


def estimate_counts(graph) -> dict:
    """Zero-shot count estimates. N_supp/N_cons are placeholders (== N_obs) until
    Phase 2 (support tracking + belief.py estimators)."""
    n_obs = len(predicted_nodes(graph))
    return {"N_obs": n_obs, "N_supp": n_obs, "N_cons": n_obs}


def build_verifier(verifier: str, base_cfg=None, query_file=None, oracle=None):
    """Return (cfg, oracle, query_set) for a verifier mode. Oracle/query_set are
    None unless verifier == 'vip'. A prebuilt oracle may be injected (tests)."""
    cfg = dataclasses.replace(base_cfg or Config(), verifier_mode=verifier)
    if verifier != "vip":
        return cfg, None, None
    from verifier.queries import load_query_set
    query_set = load_query_set(query_file or cfg.vip_query_file)
    if oracle is None:
        from verifier.oracle import QwenOracle
        oracle = QwenOracle(cfg)
    return cfg, oracle, query_set


def run_policy(policy, processor, image_pil, cfg, oracle, query_set, prompt, conf,
               execute_pass_fn=None):
    """Execute a fixed policy, returning (graph, counts, n_actions).

    execute_pass_fn is injectable for testing; defaults to pipeline.execute_pass.
    """
    if execute_pass_fn is None:
        from pipeline import execute_pass as execute_pass_fn

    graph = OrchardGraph()
    counts = {"n_global": 0, "n_tile": 0, "n_verify": 0}

    def do_pass(pass_number, tiling):
        stats = execute_pass_fn(
            processor=processor, image_pil=image_pil, graph=graph, conf=conf,
            clahe=False, tiling=tiling, pass_number=pass_number, prompt=prompt,
            cfg=cfg, oracle=oracle, query_set=query_set,
        )
        counts["n_tile" if tiling else "n_global"] += 1
        if cfg.verifier_mode == "vip":
            counts["n_verify"] += int(getattr(stats, "post_verify", 0))
        return int(stats)

    if policy == "oneshot":
        do_pass(1, tiling=False)
    elif policy in FIXED_PASS_POLICIES:
        tiling = policy == "tiled"
        for p in range(1, FIXED_PASS_POLICIES[policy] + 1):
            do_pass(p, tiling=tiling)
    elif policy == "convergence":
        for p in range(1, CONVERGENCE_MAX_PASSES + 1):
            if do_pass(p, tiling=False) == 0:
                break
    elif policy in AGENT_POLICIES:
        raise NotImplementedError(
            f"policy '{policy}' requires the agent runner (Phase 4/5), not yet built."
        )
    else:
        raise ValueError(f"Unknown policy {policy!r}.")

    n_actions = counts["n_global"] + counts["n_tile"]
    return graph, counts, n_actions


def evaluate_image(sample, policy, cfg, oracle, query_set, processor, prompt, conf,
                   execute_pass_fn=None, seed=0) -> dict:
    """Run one image through one policy+verifier and build its CSV row."""
    from PIL import Image

    np.random.seed(seed)  # deterministic leaf sampling in execute_pass
    image_pil = Image.open(sample["image_path"]).convert("RGB")

    t0 = time.time()
    graph, counts, n_actions = run_policy(
        policy, processor, image_pil, cfg, oracle, query_set, prompt, conf, execute_pass_fn
    )
    seconds = time.time() - t0

    gt = sample["gt_boxes"]
    pred_boxes = [n.box for n in predicted_nodes(graph)]
    all_boxes = [n.box for n in graph.nodes.values()]
    m = match(pred_boxes, gt)
    est = estimate_counts(graph)

    return {
        "image": os.path.basename(sample["image_path"]),
        "policy": policy,
        "verifier": cfg.verifier_mode,
        "N_gt": sample["count"],
        "N_obs": est["N_obs"],
        "N_supp": est["N_supp"],
        "N_cons": est["N_cons"],
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "pool_recall": round(pool_recall(all_boxes, gt), 4),
        "per_pass_pool_recall": json.dumps(per_pass_pool_recall(graph, gt)),
        "cost": round(normalized_cost(counts, cfg), 4),
        "n_actions": n_actions,
        "seconds": round(seconds, 3),
    }


# ----------------------- CSV / resume helpers -----------------------

def existing_keys(csv_path: str) -> set:
    """Set of (image, policy, verifier) already present in the CSV (for resume)."""
    if not os.path.exists(csv_path):
        return set()
    keys = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            keys.add((row["image"], row["policy"], row["verifier"]))
    return keys


def append_row(csv_path: str, row: dict):
    """Append one row, writing the header first if the file is new."""
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def print_aggregate(csv_path: str, policy: str, verifier: str):
    """Print mean metrics over all rows matching this policy+verifier."""
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["policy"] == policy and r["verifier"] == verifier]
    if not rows:
        return
    n_obs = [float(r["N_obs"]) for r in rows]
    n_gt = [float(r["N_gt"]) for r in rows]
    print(f"\n=== Aggregate: policy={policy} verifier={verifier} (n={len(rows)}) ===")
    print(f"  MAE(N_obs)   = {mae(n_obs, n_gt):.3f}")
    print(f"  RMSE(N_obs)  = {rmse(n_obs, n_gt):.3f}")
    print(f"  Exact(N_obs) = {exact(n_obs, n_gt):.3f}")
    for col in ("precision", "recall", "f1", "pool_recall", "cost"):
        vals = [float(r[col]) for r in rows]
        print(f"  mean {col:11s}= {sum(vals) / len(vals):.3f}")


# ----------------------- CLI -----------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Policy x verifier evaluation sweep.")
    parser.add_argument("--root", required=True, help="Dataset root (images/<split>, labels/<split>).")
    parser.add_argument("--fmt", choices=["yolo", "minneapple"], required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--policy", required=True,
                        choices=["oneshot", "cascade", "tiled", "convergence", "heuristic", "vlm"])
    parser.add_argument("--verifier", choices=["ioc", "vip", "off"], default="ioc")
    parser.add_argument("--prompt", default="green fruit")
    parser.add_argument("--conf", type=float, default=None, help="Detection conf (default cfg.conf).")
    parser.add_argument("--query-file", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max images to evaluate.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    base_cfg = Config()
    conf = args.conf if args.conf is not None else base_cfg.conf
    cfg, oracle, query_set = build_verifier(args.verifier, base_cfg, args.query_file)

    samples = load_split(args.root, args.fmt, args.split)
    if args.limit is not None:
        samples = samples[:args.limit]

    # Load SAM3 once (real run). Imported lazily so this module stays torch-free.
    import torch
    import inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bpe_path = f"{cfg.sam3_repo}/assets/bpe_simple_vocab_16e6.txt.gz"
    _, processor = inference.load_sam3_model(bpe_path, conf, device=device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = existing_keys(args.out)

    for i, sample in enumerate(samples):
        key = (os.path.basename(sample["image_path"]), args.policy, args.verifier)
        if key in done:
            logger.info("skip (already done): %s", key)
            continue
        row = evaluate_image(sample, args.policy, cfg, oracle, query_set, processor,
                             args.prompt, conf, seed=i)
        append_row(args.out, row)
        logger.info("row: %s", {k: row[k] for k in ("image", "N_gt", "N_obs", "f1", "pool_recall", "cost")})

    print_aggregate(args.out, args.policy, args.verifier)


if __name__ == "__main__":
    main()
