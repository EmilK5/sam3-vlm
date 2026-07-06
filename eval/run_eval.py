"""
eval/run_eval.py

Policy x verifier sweep over a dataset split, writing one CSV row per image.

Policies:
    oneshot     - 1 global pass
    cascade     - 4 fixed global passes
    tiled       - 4 fixed tiled passes
    convergence - global passes until a pass discovers 0 new candidates
    heuristic / vlm - agent episodes via agent.runner.run_episode.

Count estimators come from agent.belief.count_estimates (N_obs / N_supp /
N_cons). Cost comes from a CostMeter fed by the actual per-pass call counts on
PassStats (n_sam_calls / n_tiles / n_verify_calls), so fixed policies and agent
episodes are metered on the same scale.

The sweep is resume-safe (skips image/policy/verifier rows already in the CSV)
and uses a fixed numpy seed per image for reproducibility.

--gate-mode picks which apply_nms_dualgate suppression criterion applies (dual
default | iou_only | iom_only); --force-tile makes the first sensing pass tiled
regardless of policy. Both are folded into the CSV "verifier" tag (via
verifier_label) so different settings never collide on resume. Every image
where N_obs != N_gt gets an annotated mismatch overlay saved under
--mismatch-dir (default: <out dir>/mismatches/<policy>_<verifier tag>/).
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
from agent.belief import count_estimates
from agent.budget import CostMeter
from eval.datasets import load_split
from eval.matching import match, pool_recall, per_pass_pool_recall
from eval.metrics import mae, rmse, exact

logger = logging.getLogger(__name__)

FIXED_PASS_POLICIES = {"cascade": 4, "tiled": 4}
AGENT_POLICIES = {"heuristic", "vlm"}
CONVERGENCE_MAX_PASSES = 8

CSV_FIELDS = [
    "image", "dataset", "policy", "verifier", "prompt", "N_gt", "N_obs", "N_supp", "N_cons",
    "precision", "recall", "f1", "pool_recall", "per_pass_pool_recall",
    "cost", "n_actions", "seconds",
]

# pixmo/countbench/carpk have no box-level ground truth (see eval.datasets'
# count-only section), only a scalar count -- rows for these leave the
# box-metric columns blank rather than crashing on a missing gt_boxes array.
COUNT_ONLY_DATASETS = ("pixmo", "countbench", "carpk")

# A candidate counts toward the target if it's verified fruit or still unresolved
# (unverified). Leaf/spurious are excluded from the count.
_TARGET_CLASSES = ("fruit", "unresolved")


def predicted_nodes(graph):
    """Nodes counted as target objects (fruit or unresolved)."""
    return [n for n in graph.nodes.values() if n.classification in _TARGET_CLASSES]


def verifier_label(cfg, force_tile=False) -> str:
    """CSV 'verifier' value: the verifier mode, tagged with any non-default
    overlap/gate/tiling setting, so different sweeps of the same policy never
    collide on resume."""
    tag = ""
    if getattr(cfg, "overlap_mode", "box") == "mask":
        tag += "+mask"
    gate = getattr(cfg, "gate_mode", "dual")
    if gate != "dual":
        tag += f"+{gate}"
    if force_tile:
        tag += "+ftile"
    return cfg.verifier_mode + tag


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


def _run_fixed_policy(policy, processor, image_pil, cfg, oracle, query_set, prompt, conf,
                      graph, execute_pass_fn, force_tile=False):
    """oneshot / cascade / tiled / convergence. Returns (CostMeter, n_actions).

    force_tile: pass 1 is tiled regardless of the policy's normal tiling choice
    (subsequent passes keep their usual behavior); a no-op for "tiled", which is
    already fully tiled.
    """
    if execute_pass_fn is None:
        from pipeline import execute_pass as execute_pass_fn

    cost = CostMeter()
    gate_mode = getattr(cfg, "gate_mode", "dual")

    def do_pass(pass_number, tiling):
        stats = execute_pass_fn(
            processor=processor, image_pil=image_pil, graph=graph, conf=conf,
            clahe=False, tiling=tiling, pass_number=pass_number, prompt=prompt,
            cfg=cfg, oracle=oracle, query_set=query_set, gate_mode=gate_mode,
        )
        # Meter from the pass's actual call counts (canopy + leaf map + proposal
        # in n_sam_calls; tiles; real vip oracle calls) — same scale as episodes.
        cost.n_sam += int(getattr(stats, "n_sam_calls", 0))
        cost.n_tile += int(getattr(stats, "n_tiles", 0))
        cost.n_verify += int(getattr(stats, "n_verify_calls", 0))
        do_pass.n_passes += 1
        return int(stats)

    do_pass.n_passes = 0

    if policy == "oneshot":
        do_pass(1, tiling=force_tile)
    elif policy in FIXED_PASS_POLICIES:
        base_tiling = policy == "tiled"
        for p in range(1, FIXED_PASS_POLICIES[policy] + 1):
            do_pass(p, tiling=base_tiling or (force_tile and p == 1))
    elif policy == "convergence":
        for p in range(1, CONVERGENCE_MAX_PASSES + 1):
            if do_pass(p, tiling=(force_tile and p == 1)) == 0:
                break
    else:
        raise ValueError(f"Unknown fixed policy {policy!r}.")

    return cost, do_pass.n_passes


def _run_agent_policy(policy, processor, image_pil, cfg, oracle, query_set,
                      graph, episode_execute_fn, force_tile=False):
    """heuristic / vlm episodes. Returns (CostMeter, n_actions).

    force_tile: run one unconditional TileQueryA before the policy loop starts
    (consuming 1 unit of the action budget), so every episode gets at least one
    genuine tiled pass regardless of what the policy would have picked.
    """
    from agent.actions import ActionContext, TileQueryA, execute as default_execute
    from agent.belief import DiscoveryCurve
    from agent import runner, policy_heuristic

    cost = CostMeter()

    # Partition starts as [tree_roi] (plan 4.1): anchor the episode's regions to
    # the canopy so region queries don't waste budget on soil/sky. Needs SAM3, so
    # with no processor (offline tests / stubbed executors) fall back to the full
    # frame. The canopy sweep is one real global SAM3 call -> metered.
    if processor is not None:
        from pipeline import initialize_canopy_roi  # lazy: pipeline imports torch
        roi = initialize_canopy_roi(processor, np.array(image_pil), graph)
        cost.n_sam += 1
        partition = [tuple(int(v) for v in roi)]
    else:
        w, h = image_pil.size
        partition = [(0, 0, w, h)]

    ctx = ActionContext(
        processor=processor, image_pil=image_pil, graph=graph, cfg=cfg,
        oracle=oracle, query_set=query_set, discovery=DiscoveryCurve(),
        partition=partition, cost=cost,
    )
    forced_actions = 0
    if force_tile:
        exec_fn = episode_execute_fn or default_execute
        tile_prompt = getattr(cfg, "target_prompt", "green fruit")
        tile_conf = getattr(cfg, "conf", 0.3)
        n_new = int(exec_fn(TileQueryA(prompt=tile_prompt, conf=tile_conf), ctx))
        ctx.discovery.append(n_new)
        forced_actions = 1

    pol = policy_heuristic.choose if policy == "heuristic" else runner.make_vlm_policy(ctx)
    # The guided-ROI VLM episode opens with a mandatory global bootstrap pass
    # (seeds candidates + exemplars) and auto-stops on discovery saturation; the
    # heuristic baseline keeps its original loop (flags default off).
    is_vlm = policy == "vlm"
    remaining_budget = max(0, cfg.budget_max_actions - forced_actions)
    result = runner.run_episode(image_pil, ctx, pol, max_actions=remaining_budget,
                                execute_fn=episode_execute_fn,
                                bootstrap_global_pass=is_vlm, auto_stop=is_vlm)
    return ctx.cost, len(result["log"]) + forced_actions


def run_policy(policy, processor, image_pil, cfg, oracle, query_set, prompt, conf,
               execute_pass_fn=None, episode_execute_fn=None, force_tile=False):
    """Run a policy on one image. Returns (graph, CostMeter, n_actions).

    Fixed policies call execute_pass directly; agent policies (heuristic/vlm) run
    an episode via agent.runner. Executors are injectable for testing.
    """
    graph = OrchardGraph()
    if policy in AGENT_POLICIES:
        cost, n_actions = _run_agent_policy(
            policy, processor, image_pil, cfg, oracle, query_set, graph, episode_execute_fn,
            force_tile=force_tile)
    else:
        cost, n_actions = _run_fixed_policy(
            policy, processor, image_pil, cfg, oracle, query_set, prompt, conf, graph, execute_pass_fn,
            force_tile=force_tile)
    return graph, cost, n_actions


def evaluate_image(sample, policy, cfg, oracle, query_set, processor, prompt, conf,
                   execute_pass_fn=None, episode_execute_fn=None, seed=0, force_tile=False,
                   mismatch_dir=None, dataset="local") -> dict:
    """Run one image through one policy+verifier and build its CSV row.

    sample: either the box-annotated shape from eval.datasets.load_split
    ({"image_path", "gt_boxes", "count"}) or the count-only shape from
    eval.datasets.get_count_sample ({"image_pil", "image_name", "count",
    "gt_boxes": None, "prompt": per-sample concept override}) used for
    pixmo/countbench/carpk. When gt_boxes is None the box-metric columns
    (precision/recall/f1/pool_recall/per_pass_pool_recall) are left blank --
    those three datasets have no box ground truth, only a scalar count.

    mismatch_dir: when given and N_obs != N_gt, save an annotated overlay there
    (see export_mismatch_overlay).
    """
    from PIL import Image

    np.random.seed(seed)  # deterministic leaf sampling in execute_pass
    if "image_pil" in sample:
        image_pil = sample["image_pil"]
        image_name = sample.get("image_name", f"{dataset}_{seed:05d}")
    else:
        image_pil = Image.open(sample["image_path"]).convert("RGB")
        image_name = os.path.basename(sample["image_path"])

    # Count-only datasets carry a per-sample resolved concept (their generic
    # prompt map); agent policies read the concept from cfg.target_prompt, so
    # override it per call rather than sharing one cfg across the whole sweep.
    effective_prompt = sample.get("prompt") or prompt
    call_cfg = dataclasses.replace(cfg, target_prompt=effective_prompt)

    t0 = time.time()
    graph, cost_meter, n_actions = run_policy(
        policy, processor, image_pil, call_cfg, oracle, query_set, effective_prompt, conf,
        execute_pass_fn=execute_pass_fn, episode_execute_fn=episode_execute_fn,
        force_tile=force_tile,
    )
    seconds = time.time() - t0

    gt_boxes = sample.get("gt_boxes")
    est = count_estimates(graph, call_cfg)  # real Phase-2 estimators

    if gt_boxes is not None:
        pred_boxes = [n.box for n in predicted_nodes(graph)]
        all_boxes = [n.box for n in graph.nodes.values()]
        m = match(pred_boxes, gt_boxes)
        precision, recall, f1 = round(m["precision"], 4), round(m["recall"], 4), round(m["f1"], 4)
        pool_rec = round(pool_recall(all_boxes, gt_boxes), 4)
        pppr = json.dumps(per_pass_pool_recall(graph, gt_boxes))
    else:
        precision = recall = f1 = pool_rec = ""
        pppr = ""

    row = {
        "image": image_name,
        "dataset": dataset,
        "policy": policy,
        "verifier": verifier_label(call_cfg, force_tile),
        "prompt": effective_prompt,
        "N_gt": sample["count"],
        "N_obs": est["N_obs"],
        "N_supp": est["N_supp"],
        "N_cons": est["N_cons"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pool_recall": pool_rec,
        "per_pass_pool_recall": pppr,
        "cost": round(cost_meter.total(call_cfg), 4),
        "n_actions": n_actions,
        "seconds": round(seconds, 3),
    }

    if mismatch_dir is not None and row["N_obs"] != row["N_gt"]:
        export_mismatch_overlay(image_pil, graph, image_name, effective_prompt,
                                row["N_gt"], row["N_obs"], mismatch_dir)

    return row


# ----------------------- visual diagnostics -----------------------

_MISMATCH_COLORS = {
    "fruit": (57, 255, 20), "leaf": (255, 59, 59),
    "spurious": (154, 160, 166), "unresolved": (0, 229, 255),
}


def export_mismatch_overlay(image_pil, graph, image_name, prompt, gt_count, pred_count, out_dir):
    """Save an annotated overlay for one mismatched prediction: every candidate
    box colored by classification, with a GT-vs-pred banner. Colors mirror
    citrus_orchestration_app.py so eyeballing failures matches the dashboard."""
    from PIL import ImageDraw, ImageFont

    os.makedirs(out_dir, exist_ok=True)
    canvas = image_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for node in graph.nodes.values():
        x1, y1, x2, y2 = node.box
        color = _MISMATCH_COLORS.get(node.classification, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, max(0, y1 - 12)),
                  f"P{node.found_in_pass} {node.classification[:1]}:{node.scores['detection_confidence']:.2f}",
                  fill=color, font=font)

    direction = "under" if pred_count < gt_count else "over"
    banner = f"'{prompt}' | GT={gt_count} PRED={pred_count} ({direction})"
    draw.text((8, 8), banner, fill=(231, 76, 60), font=font)

    stem = os.path.splitext(image_name)[0]
    clean_prompt = "".join(c if c.isalnum() else "_" for c in str(prompt)[:25]).strip("_")
    canvas.save(os.path.join(out_dir, f"mismatch_{stem}_{direction}_{clean_prompt}.png"))


# ----------------------- CSV / resume helpers -----------------------

def existing_keys(csv_path: str) -> set:
    """Set of (image, dataset, policy, verifier) already present in the CSV (for
    resume). row.get("dataset", "local") so CSVs written before the "dataset"
    column existed (always a "local" run) still resume correctly."""
    if not os.path.exists(csv_path):
        return set()
    keys = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            keys.add((row["image"], row.get("dataset", "local"), row["policy"], row["verifier"]))
    return keys


def append_row(csv_path: str, row: dict):
    """Append one row, writing the header first if the file is new."""
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def print_aggregate(csv_path: str, dataset: str, policy: str, verifier: str):
    """Print mean metrics over all rows matching this dataset+policy+verifier.

    Box-metric columns (precision/recall/f1/pool_recall) are blank for
    count-only datasets (pixmo/countbench/carpk) -- skipped rather than
    crashing on float("").
    """
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("dataset", "local") == dataset and r["policy"] == policy
                and r["verifier"] == verifier]
    if not rows:
        return
    n_obs = [float(r["N_obs"]) for r in rows]
    n_gt = [float(r["N_gt"]) for r in rows]
    print(f"\n=== Aggregate: dataset={dataset} policy={policy} verifier={verifier} (n={len(rows)}) ===")
    print(f"  MAE(N_obs)   = {mae(n_obs, n_gt):.3f}")
    print(f"  RMSE(N_obs)  = {rmse(n_obs, n_gt):.3f}")
    print(f"  Exact(N_obs) = {exact(n_obs, n_gt):.3f}")
    for col in ("precision", "recall", "f1", "pool_recall", "cost"):
        vals = [float(r[col]) for r in rows if r[col] not in ("", None)]
        if vals:
            print(f"  mean {col:11s}= {sum(vals) / len(vals):.3f}")


# ----------------------- CLI -----------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Policy x verifier evaluation sweep.")
    parser.add_argument("--dataset", choices=["local", "pixmo", "countbench", "carpk"],
                        default="local",
                        help="'local' (default): your own box-annotated split via --root/--fmt/"
                             "--split (has real GT boxes -> full metrics). 'pixmo'/'countbench'/"
                             "'carpk': the count-only benchmark datasets (eval.datasets' "
                             "get_count_sample) -- no box GT, so precision/recall/f1/pool_recall "
                             "are blank; --root/--fmt/--split are ignored. Each carries its own "
                             "generic prompt map/purged-id list/GT overrides (see eval/datasets.py); "
                             "pass --prompt to override the per-sample resolved concept for all of "
                             "them (CARPK's concept is always 'car' regardless).")
    parser.add_argument("--root", default=None,
                        help="Dataset root (images/<split>, labels/<split>). Required for "
                             "--dataset local.")
    parser.add_argument("--fmt", choices=["yolo", "minneapple"], default=None,
                        help="Local split format. Required for --dataset local.")
    parser.add_argument("--split", default="val", help="--dataset local only.")
    parser.add_argument("--policy", required=True,
                        choices=["oneshot", "cascade", "tiled", "convergence", "heuristic", "vlm"])
    parser.add_argument("--verifier", choices=["ioc", "vip", "off"], default="ioc")
    parser.add_argument("--overlap-mode", choices=["box", "mask"], default="box",
                        help="Overlap geometry: NMS IoU/IoM + cross-pass dedup on boxes "
                             "(default) or SAM3 instance masks.")
    parser.add_argument("--gate-mode", choices=["dual", "iou_only", "iom_only"], default="dual",
                        help="Which apply_nms_dualgate suppression gate(s) apply (when "
                             "nms_mode=='dualgate', the pipeline default). 'dual' (default, "
                             "unchanged) ORs Gate A (IoU) and Gate B (IoM containment). "
                             "'iou_only'/'iom_only' pick a single criterion -- e.g. iou_only "
                             "for countbench-style scenes, iom_only for CARPK-style dense "
                             "grids of uniform-size objects.")
    parser.add_argument("--prompt", default=None,
                        help="Target concept. --dataset local: the fixed concept for every "
                             "image (default 'green fruit'). Count-only datasets: overrides "
                             "the per-sample generic-map concept for every image (default: "
                             "resolve per-sample via eval.datasets.default_prompt_for).")
    parser.add_argument("--conf", type=float, default=None, help="Detection conf (default cfg.conf).")
    parser.add_argument("--force-tile", action="store_true",
                        help="Force the first sensing pass to be tiled, regardless of policy "
                             "(oneshot/cascade/convergence pass 1 becomes tiled; heuristic/vlm "
                             "run one extra tiled pass before the policy loop, consuming 1 unit "
                             "of the action budget).")
    parser.add_argument("--query-file", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max images to evaluate.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--mismatch-dir", default=None,
                        help="Save an annotated overlay for every image where N_obs != N_gt "
                             "here (default: <out dir>/mismatches/<policy>_<verifier tag>/). "
                             "Pass an empty string to disable.")
    return parser.parse_args(argv)


def _iter_local_rows(args, cfg, oracle, query_set, processor, conf, done, mismatch_dir, label):
    """--dataset local: box-annotated split via eval.datasets.load_split. Every
    sample uses the same fixed --prompt concept (default 'green fruit'). Image
    paths are cheap to check against `done`, so this mirrors the original
    (pre-count-dataset) resume flow exactly."""
    if not args.root or not args.fmt:
        raise SystemExit("--dataset local requires --root and --fmt.")
    target = args.prompt or "green fruit"
    samples = load_split(args.root, args.fmt, args.split)
    if args.limit is not None:
        samples = samples[:args.limit]

    for i, sample in enumerate(samples):
        image_name = os.path.basename(sample["image_path"])
        key = (image_name, "local", args.policy, label)
        if key in done:
            logger.info("skip (already done): %s", key)
            continue
        yield evaluate_image(sample, args.policy, cfg, oracle, query_set, processor,
                             target, conf, seed=i, force_tile=args.force_tile,
                             mismatch_dir=mismatch_dir, dataset="local")


def _iter_count_rows(args, cfg, oracle, query_set, processor, conf, done, mismatch_dir, label):
    """--dataset pixmo/countbench/carpk: eval.datasets' count-only accessor.

    The resume check happens BEFORE fetching the image (cheap: only needs the
    index), not after -- pixmo/countbench fetch over the network, so eagerly
    pulling every image up front would both waste bandwidth on rows already in
    the CSV and delay the first row being written until the whole dataset
    finished downloading. Purged/errored indices are skipped and don't count
    against --limit; each row gets its own resolved concept (the generic
    prompt map) unless --prompt overrides it for all.
    """
    from eval.datasets import get_count_sample, count_dataset_size, default_prompt_for

    name = args.dataset
    n = count_dataset_size(name)
    limit = args.limit if args.limit is not None else n

    evaluated = 0
    idx = 0
    while evaluated < limit and idx < n:
        image_name = f"{name}_{idx:05d}"
        key = (image_name, name, args.policy, label)
        if key in done:
            logger.info("skip (already done): %s", key)
            idx += 1
            continue

        image_pil, raw_prompt, gt, err = get_count_sample(name, idx)
        idx += 1
        if err is not None:
            logger.info("skip idx %d (%s): %s", idx - 1, name, err)
            continue

        resolved_prompt = args.prompt if args.prompt is not None else default_prompt_for(name, raw_prompt)
        sample = {"image_pil": image_pil, "image_name": image_name,
                  "gt_boxes": None, "count": gt, "prompt": resolved_prompt}
        yield evaluate_image(sample, args.policy, cfg, oracle, query_set, processor,
                             resolved_prompt, conf, seed=idx - 1, force_tile=args.force_tile,
                             mismatch_dir=mismatch_dir, dataset=name)
        evaluated += 1


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    if args.dataset == "local" and (not args.root or not args.fmt):
        raise SystemExit("--dataset local requires --root and --fmt.")
    if args.dataset != "local" and (args.root or args.fmt):
        logger.warning("--root/--fmt are ignored for --dataset %s.", args.dataset)

    base_cfg = Config()
    conf = args.conf if args.conf is not None else base_cfg.conf
    cfg, oracle, query_set = build_verifier(args.verifier, base_cfg, args.query_file)
    # Agent policies read the concept/confidence from cfg (fixed policies get them
    # as arguments); evaluate_image overrides target_prompt per-sample, so this is
    # just the shared baseline. --overlap-mode/--gate-mode apply everywhere.
    cfg = dataclasses.replace(cfg, conf=conf, overlap_mode=args.overlap_mode, gate_mode=args.gate_mode)

    # Load SAM3 once (real run). Imported lazily so this module stays torch-free.
    import torch
    import inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bpe_path = f"{cfg.sam3_repo}/assets/bpe_simple_vocab_16e6.txt.gz"
    _, processor = inference.load_sam3_model(bpe_path, conf, device=device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = existing_keys(args.out)

    label = verifier_label(cfg, args.force_tile)
    if args.mismatch_dir == "":
        mismatch_dir = None
    elif args.mismatch_dir is not None:
        mismatch_dir = args.mismatch_dir
    else:
        mismatch_dir = os.path.join(os.path.dirname(args.out) or ".",
                                    "mismatches", f"{args.dataset}_{args.policy}_{label}")

    row_iter = (_iter_local_rows(args, cfg, oracle, query_set, processor, conf, done, mismatch_dir, label)
               if args.dataset == "local" else
               _iter_count_rows(args, cfg, oracle, query_set, processor, conf, done, mismatch_dir, label))
    for row in row_iter:
        append_row(args.out, row)
        logger.info("row: %s", {k: row[k] for k in ("image", "N_gt", "N_obs", "f1", "pool_recall", "cost")})

    print_aggregate(args.out, args.dataset, args.policy, label)


if __name__ == "__main__":
    main()
