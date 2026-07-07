import csv
import dataclasses
import os

import numpy as np
import pytest
from PIL import Image

from config import Config
from graph import OrchardGraph
from agent.actions import StopA
from agent.budget import CostMeter
from eval import metrics, run_eval

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GREEN_CITRUS = os.path.join(REPO_ROOT, "queries", "green_citrus.json")


# ----------------------- metrics.py -----------------------

def test_mae_rmse_exact():
    preds = [3, 5, 2]
    gts = [2, 5, 4]
    assert metrics.mae(preds, gts) == pytest.approx((1 + 0 + 2) / 3)
    assert metrics.rmse(preds, gts) == pytest.approx(((1 + 0 + 4) / 3) ** 0.5)
    assert metrics.exact(preds, gts) == pytest.approx(1 / 3)


def test_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.mae([1, 2], [1])


def test_normalized_cost_uses_cfg_weights():
    cfg = Config()  # c_sam=1, c_tile=.25, c_verify=.5, c_inspect=.5, c_orch=.05
    cost = metrics.normalized_cost({"n_global": 2, "n_tile": 4, "n_verify": 3}, cfg)
    assert cost == pytest.approx(2 + 0.25 * 4 + 0.5 * 3)  # 4.5


# ----------------------- estimators -----------------------

def test_evaluate_image_uses_real_belief_estimators():
    from agent.belief import count_estimates
    g = OrchardGraph()
    for cls in ("fruit", "fruit", "leaf", "spurious", "unresolved"):
        nid = g.add_candidate([0, 0, 5, 5], 0.9, found_in_pass=1)
        g.nodes[nid].classification = cls
    est = count_estimates(g, Config())
    assert est["N_obs"] == 3          # 2 fruit + 1 unresolved; leaf/spurious excluded


# ----------------------- run_policy with a stub execute_pass -----------------------

class _FakeStats(int):
    """Mirrors PassStats' call-count fields (the sweep meters from these)."""
    def __new__(cls, val, n_sam_calls=0, n_tiles=0, n_verify_calls=0):
        obj = super().__new__(cls, val)
        obj.n_sam_calls = n_sam_calls
        obj.n_tiles = n_tiles
        obj.n_verify_calls = n_verify_calls
        return obj


def _stub(per_pass):
    """execute_pass stand-in: returns per_pass[i] and adds that many fruit nodes.
    Reports one global SAM3 call per global pass, one tile per tiled pass, and
    one vip oracle call per accepted node."""
    state = {"i": 0}

    def fn(**kw):
        graph, pass_number = kw["graph"], kw["pass_number"]
        n = per_pass[min(state["i"], len(per_pass) - 1)]
        for j in range(n):
            nid = graph.add_candidate([j * 10, 0, j * 10 + 8, 8], 0.9, found_in_pass=pass_number)
            graph.nodes[nid].classification = "fruit"
        state["i"] += 1
        tiling = kw["tiling"]
        return _FakeStats(n, n_sam_calls=0 if tiling else 1, n_tiles=1 if tiling else 0,
                          n_verify_calls=n)

    fn.state = state
    return fn


def _run(policy, cfg, per_pass):
    return run_eval.run_policy(policy, processor=None, image_pil=None, cfg=cfg,
                               oracle=None, query_set=None, prompt="green fruit",
                               conf=0.35, execute_pass_fn=_stub(per_pass))


def test_oneshot_runs_one_global_pass():
    graph, cost, n_actions = _run("oneshot", Config(), [3])
    assert cost.n_sam == 1 and cost.n_tile == 0
    assert n_actions == 1
    assert len(graph.nodes) == 3


def test_cascade_runs_four_global_passes():
    _, cost, n_actions = _run("cascade", Config(), [1, 1, 1, 1])
    assert cost.n_sam == 4 and n_actions == 4


def test_tiled_runs_four_tiled_passes():
    _, cost, n_actions = _run("tiled", Config(), [1, 1, 1, 1])
    assert cost.n_tile == 4 and cost.n_sam == 0 and n_actions == 4


def test_convergence_stops_when_no_new_candidates():
    _, _cost, n_actions = _run("convergence", Config(), [2, 1, 0, 5])
    assert n_actions == 3          # pass 3 returns 0 -> stop (the 5 is never reached)


def test_vip_counts_verify_calls():
    cfg = dataclasses.replace(Config(), verifier_mode="vip")
    _, cost, _ = _run("oneshot", cfg, [4])
    assert cost.n_verify == 4  # sum of the pass's actual n_verify_calls


def _episode_stub(script):
    """Action-level executor stub for agent episodes: adds `script[i]` fruit nodes."""
    state = {"i": 0}

    def fn(action, ctx):
        if isinstance(action, StopA):
            return 0
        n = script[min(state["i"], len(script) - 1)]
        for j in range(n):
            nid = ctx.graph.add_candidate([j, 0, j + 5, 5], 0.9, len(ctx.discovery.counts) + 1)
            ctx.graph.nodes[nid].classification = "fruit"
        state["i"] += 1
        return n

    return fn


def test_heuristic_policy_runs_an_episode():
    cfg = dataclasses.replace(Config(), delta_U=1e9, delta_disc=1.0)  # stop on saturation
    graph, cost, n_actions = run_eval.run_policy(
        "heuristic", processor=None, image_pil=Image.new("RGB", (64, 64)), cfg=cfg,
        oracle=None, query_set=None, prompt="green fruit", conf=0.35,
        episode_execute_fn=_episode_stub([3, 0, 0, 0]),
    )
    assert isinstance(cost, CostMeter)
    assert 1 <= n_actions <= 6
    assert len(graph.nodes) >= 1


def test_verifier_label_tags_mask_mode():
    assert run_eval.verifier_label(Config()) == "ioc"
    cfg = dataclasses.replace(Config(), verifier_mode="ioc", overlap_mode="mask")
    assert run_eval.verifier_label(cfg) == "ioc+mask"


def test_verifier_label_tags_gate_mode_and_force_tile():
    cfg = dataclasses.replace(Config(), gate_mode="iou_only")
    assert run_eval.verifier_label(cfg) == "ioc+iou_only"
    assert run_eval.verifier_label(Config(), force_tile=True) == "ioc+ftile"


def test_verifier_label_tags_no_canopy():
    cfg = dataclasses.replace(Config(), use_canopy_roi=False)
    assert run_eval.verifier_label(cfg) == "ioc+nocanopy"
    assert run_eval.verifier_label(Config()) == "ioc"  # default True -> no tag


def test_verifier_label_tags_nms_thresholds():
    cfg = dataclasses.replace(Config(), nms_iou_threshold=0.3, nms_iom_threshold=0.8)
    assert run_eval.verifier_label(cfg) == "ioc+iou0.3+iom0.8"
    assert run_eval.verifier_label(Config()) == "ioc"  # defaults (0.40/0.90) -> no tag


def test_nms_thresholds_forwarded_to_execute_pass():
    captured = {}

    def fn(**kw):
        captured["nms_iou_threshold"] = kw.get("nms_iou_threshold")
        captured["nms_iom_threshold"] = kw.get("nms_iom_threshold")
        graph, pass_number = kw["graph"], kw["pass_number"]
        nid = graph.add_candidate([0, 0, 5, 5], 0.9, found_in_pass=pass_number)
        graph.nodes[nid].classification = "fruit"
        return _FakeStats(1, n_sam_calls=1)

    cfg = dataclasses.replace(Config(), nms_iou_threshold=0.25, nms_iom_threshold=0.75)
    run_eval.run_policy("oneshot", processor=None, image_pil=None, cfg=cfg,
                        oracle=None, query_set=None, prompt="car", conf=0.45,
                        execute_pass_fn=fn)
    assert captured["nms_iou_threshold"] == 0.25
    assert captured["nms_iom_threshold"] == 0.75


def test_verifier_label_tags_cross_pass_dedup():
    cfg = dataclasses.replace(Config(), cross_pass_dedup_metric="iom",
                             cross_pass_dedup_threshold=0.85)
    assert run_eval.verifier_label(cfg) == "ioc+dedup-iom0.85"
    assert run_eval.verifier_label(Config()) == "ioc"  # defaults (iou/0.40) -> no tag


def test_cross_pass_dedup_forwarded_to_execute_pass():
    captured = {}

    def fn(**kw):
        captured["cross_pass_dedup_metric"] = kw.get("cross_pass_dedup_metric")
        captured["cross_pass_dedup_threshold"] = kw.get("cross_pass_dedup_threshold")
        graph, pass_number = kw["graph"], kw["pass_number"]
        nid = graph.add_candidate([0, 0, 5, 5], 0.9, found_in_pass=pass_number)
        graph.nodes[nid].classification = "fruit"
        return _FakeStats(1, n_sam_calls=1)

    cfg = dataclasses.replace(Config(), cross_pass_dedup_metric="iom",
                             cross_pass_dedup_threshold=0.85)
    run_eval.run_policy("oneshot", processor=None, image_pil=None, cfg=cfg,
                        oracle=None, query_set=None, prompt="car", conf=0.45,
                        execute_pass_fn=fn)
    assert captured["cross_pass_dedup_metric"] == "iom"
    assert captured["cross_pass_dedup_threshold"] == 0.85


def test_dedup_defaults_match_validated_reference_values():
    """Sanity-lock DEDUP_DEFAULTS against the validated reference sandbox.py's
    DATASET_UI_DEFAULTS (dedup_iou/use_iom per dataset) -- a typo here silently
    changes what --dedup-metric=auto resolves to."""
    assert run_eval.DEDUP_DEFAULTS["local"] == {"metric": "iou", "threshold": 0.40}
    assert run_eval.DEDUP_DEFAULTS["pixmo"] == {"metric": "iou", "threshold": 0.65}
    assert run_eval.DEDUP_DEFAULTS["countbench"] == {"metric": "iou", "threshold": 0.60}
    assert run_eval.DEDUP_DEFAULTS["carpk"] == {"metric": "iom", "threshold": 0.85}


def test_parse_args_dedup_flags_default_auto_and_none():
    args = run_eval.parse_args(["--root", "r", "--fmt", "yolo", "--policy", "oneshot", "--out", "x.csv"])
    assert args.dedup_metric == "auto" and args.dedup_threshold is None


def test_parse_args_dedup_flags_explicit():
    args = run_eval.parse_args(["--dataset", "carpk", "--policy", "oneshot",
                                "--dedup-metric", "iom", "--dedup-threshold", "0.85", "--out", "x.csv"])
    assert args.dedup_metric == "iom" and args.dedup_threshold == 0.85


def test_parse_args_iou_iom_threshold_default_none():
    args = run_eval.parse_args(["--root", "r", "--fmt", "yolo", "--policy", "oneshot", "--out", "x.csv"])
    assert args.iou_threshold is None and args.iom_threshold is None


def test_parse_args_iou_iom_threshold_explicit():
    args = run_eval.parse_args(["--root", "r", "--fmt", "yolo", "--policy", "oneshot",
                                "--iou-threshold", "0.3", "--iom-threshold", "0.85", "--out", "x.csv"])
    assert args.iou_threshold == 0.3 and args.iom_threshold == 0.85


def test_use_canopy_roi_forwarded_to_execute_pass():
    captured = {}

    def fn(**kw):
        captured["use_canopy_roi"] = kw.get("use_canopy_roi")
        graph, pass_number = kw["graph"], kw["pass_number"]
        nid = graph.add_candidate([0, 0, 5, 5], 0.9, found_in_pass=pass_number)
        graph.nodes[nid].classification = "fruit"
        return _FakeStats(1, n_sam_calls=1)

    cfg = dataclasses.replace(Config(), use_canopy_roi=False)
    run_eval.run_policy("oneshot", processor=None, image_pil=None, cfg=cfg,
                        oracle=None, query_set=None, prompt="car", conf=0.45,
                        execute_pass_fn=fn)
    assert captured["use_canopy_roi"] is False


def test_parse_args_canopy_roi_default_auto():
    args = run_eval.parse_args(["--root", "r", "--fmt", "yolo", "--policy", "oneshot", "--out", "x.csv"])
    assert args.canopy_roi == "auto"


def test_parse_args_canopy_roi_explicit_off():
    args = run_eval.parse_args(["--dataset", "carpk", "--policy", "oneshot",
                                "--canopy-roi", "off", "--out", "x.csv"])
    assert args.canopy_roi == "off"


def test_gate_mode_forwarded_to_execute_pass():
    captured = {}

    def fn(**kw):
        captured["gate_mode"] = kw.get("gate_mode")
        graph, pass_number = kw["graph"], kw["pass_number"]
        nid = graph.add_candidate([0, 0, 5, 5], 0.9, found_in_pass=pass_number)
        graph.nodes[nid].classification = "fruit"
        return _FakeStats(1, n_sam_calls=1)

    cfg = dataclasses.replace(Config(), gate_mode="iom_only")
    run_eval.run_policy("oneshot", processor=None, image_pil=None, cfg=cfg,
                        oracle=None, query_set=None, prompt="green fruit", conf=0.35,
                        execute_pass_fn=fn)
    assert captured["gate_mode"] == "iom_only"


def test_force_tile_makes_oneshot_pass1_tiled():
    _, cost, n_actions = run_eval.run_policy(
        "oneshot", processor=None, image_pil=None, cfg=Config(),
        oracle=None, query_set=None, prompt="green fruit", conf=0.35,
        execute_pass_fn=_stub([3]), force_tile=True)
    assert cost.n_tile == 1 and cost.n_sam == 0
    assert n_actions == 1


def test_force_tile_leaves_oneshot_untouched_when_off():
    _, cost, n_actions = run_eval.run_policy(
        "oneshot", processor=None, image_pil=None, cfg=Config(),
        oracle=None, query_set=None, prompt="green fruit", conf=0.35,
        execute_pass_fn=_stub([3]), force_tile=False)
    assert cost.n_sam == 1 and cost.n_tile == 0
    assert n_actions == 1


def test_force_tile_cascade_only_tiles_pass1():
    _, cost, n_actions = run_eval.run_policy(
        "cascade", processor=None, image_pil=None, cfg=Config(),
        oracle=None, query_set=None, prompt="green fruit", conf=0.35,
        execute_pass_fn=_stub([1, 1, 1, 1]), force_tile=True)
    assert cost.n_tile == 1 and cost.n_sam == 3 and n_actions == 4


def test_force_tile_agent_episode_runs_tile_first_and_counts_one_extra_action():
    seen_actions = []

    def episode_fn(action, ctx):
        seen_actions.append(type(action).__name__)
        return 0  # never discovers anything -> exercises the "never stop empty" guard

    cfg = dataclasses.replace(Config(), delta_U=1e9, delta_disc=1.0, budget_max_actions=3)
    _, cost, n_actions = run_eval.run_policy(
        "heuristic", processor=None, image_pil=Image.new("RGB", (64, 64)), cfg=cfg,
        oracle=None, query_set=None, prompt="green fruit", conf=0.35,
        episode_execute_fn=episode_fn, force_tile=True)
    assert seen_actions[0] == "TileQueryA"
    assert isinstance(cost, CostMeter)
    assert n_actions == 3  # forced pass (1) + remaining budget (2), never over budget_max_actions


def test_evaluate_image_row_uses_mask_tagged_verifier(tmp_path):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64)).save(img_path)
    sample = {"image_path": str(img_path),
              "gt_boxes": np.array([[0, 0, 8, 8]], dtype=float), "count": 1}
    cfg = dataclasses.replace(Config(), overlap_mode="mask")
    row = run_eval.evaluate_image(sample, "oneshot", cfg, None, None,
                                  processor=None, prompt="green fruit", conf=0.35,
                                  execute_pass_fn=_stub([2]))
    assert row["verifier"] == "ioc+mask"


# ----------------------- build_verifier -----------------------

def test_build_verifier_ioc_has_no_oracle():
    cfg, oracle, qs = run_eval.build_verifier("ioc")
    assert cfg.verifier_mode == "ioc" and oracle is None and qs is None


def test_build_verifier_vip_loads_queries_and_uses_injected_oracle():
    sentinel = object()
    cfg, oracle, qs = run_eval.build_verifier("vip", query_file=GREEN_CITRUS, oracle=sentinel)
    assert cfg.verifier_mode == "vip"
    assert oracle is sentinel
    assert qs is not None and len(qs.queries) > 0


# ----------------------- evaluate_image + CSV resume -----------------------

def test_evaluate_image_builds_full_row(tmp_path):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64)).save(img_path)
    sample = {"image_path": str(img_path),
              "gt_boxes": np.array([[0, 0, 8, 8]], dtype=float), "count": 1}

    row = run_eval.evaluate_image(sample, "oneshot", Config(), None, None,
                                  processor=None, prompt="green fruit", conf=0.35,
                                  execute_pass_fn=_stub([3]))

    assert set(row.keys()) == set(run_eval.CSV_FIELDS)
    assert row["image"] == "img.png"
    assert row["N_gt"] == 1
    assert row["N_obs"] == 3
    assert 0.0 <= row["precision"] <= 1.0
    assert row["recall"] == 1.0        # the GT box matches one of the fruit boxes


def test_csv_resume_roundtrip(tmp_path):
    csv_path = str(tmp_path / "sweep.csv")
    row = {k: 0 for k in run_eval.CSV_FIELDS}
    row.update({"image": "a.png", "dataset": "local", "policy": "oneshot", "verifier": "ioc"})
    run_eval.append_row(csv_path, row)

    keys = run_eval.existing_keys(csv_path)
    assert ("a.png", "local", "oneshot", "ioc") in keys
    assert ("a.png", "local", "cascade", "ioc") not in keys


def test_csv_resume_roundtrip_old_schema_without_dataset_column(tmp_path):
    """A CSV written before the 'dataset' column existed must still resume
    (every such row is implicitly a 'local' run)."""
    csv_path = str(tmp_path / "sweep.csv")
    old_fields = [f for f in run_eval.CSV_FIELDS if f != "dataset"]
    old_row = {k: 0 for k in old_fields}
    old_row.update({"image": "a.png", "policy": "oneshot", "verifier": "ioc"})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow(old_row)

    keys = run_eval.existing_keys(csv_path)
    assert ("a.png", "local", "oneshot", "ioc") in keys


# ----------------------- count-only datasets (pixmo/countbench/carpk) -----------------------

def test_evaluate_image_blanks_box_metrics_for_count_only_sample():
    sample = {"image_pil": Image.new("RGB", (64, 64)), "image_name": "carpk_00000",
              "gt_boxes": None, "count": 5, "prompt": "car"}
    row = run_eval.evaluate_image(sample, "oneshot", Config(), None, None,
                                  processor=None, prompt="green fruit", conf=0.35,
                                  execute_pass_fn=_stub([5]), dataset="carpk")
    assert row["image"] == "carpk_00000"
    assert row["dataset"] == "carpk"
    assert row["prompt"] == "car"
    assert row["N_gt"] == 5 and row["N_obs"] == 5
    assert row["precision"] == "" and row["recall"] == "" and row["f1"] == ""
    assert row["pool_recall"] == "" and row["per_pass_pool_recall"] == ""


def test_evaluate_image_keeps_box_metrics_when_gt_boxes_present(tmp_path):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64)).save(img_path)
    sample = {"image_path": str(img_path),
              "gt_boxes": np.array([[0, 0, 8, 8]], dtype=float), "count": 1}
    row = run_eval.evaluate_image(sample, "oneshot", Config(), None, None,
                                  processor=None, prompt="green fruit", conf=0.35,
                                  execute_pass_fn=_stub([1]))
    assert row["precision"] != "" and row["pool_recall"] != ""
    assert row["dataset"] == "local"


def test_count_only_sample_prompt_overrides_cfg_target_prompt():
    """Each count-only sample carries its own resolved concept (the generic
    prompt map); agent policies must see it via cfg.target_prompt, not the
    sweep-wide --prompt fallback."""
    seen = {}

    def episode_fn(action, ctx):
        seen["target_prompt"] = ctx.cfg.target_prompt
        return 0

    sample = {"image_pil": Image.new("RGB", (64, 64)), "image_name": "pixmo_00001",
              "gt_boxes": None, "count": 2, "prompt": "orange"}
    cfg = dataclasses.replace(Config(), budget_max_actions=1)
    row = run_eval.evaluate_image(sample, "heuristic", cfg, None, None,
                                  processor=None, prompt="green fruit", conf=0.35,
                                  episode_execute_fn=episode_fn, dataset="pixmo")
    assert seen["target_prompt"] == "orange"
    assert row["prompt"] == "orange"
    assert row["dataset"] == "pixmo"
    assert row["precision"] == ""


def test_print_aggregate_skips_blank_box_metrics(tmp_path, capsys):
    csv_path = str(tmp_path / "sweep.csv")
    row = {k: "" for k in run_eval.CSV_FIELDS}
    row.update({"image": "carpk_00000", "dataset": "carpk", "policy": "oneshot",
               "verifier": "ioc", "N_gt": 5, "N_obs": 5, "N_supp": 5, "N_cons": 5,
               "cost": 1.0, "n_actions": 1, "seconds": 0.1})
    run_eval.append_row(csv_path, row)

    run_eval.print_aggregate(csv_path, "carpk", "oneshot", "ioc")
    out = capsys.readouterr().out
    assert "MAE(N_obs)" in out
    assert "mean precision" not in out  # blank column -> skipped, no crash


# ----------------------- CLI: --dataset local vs count-only -----------------------

def test_parse_args_root_fmt_optional_for_count_datasets():
    args = run_eval.parse_args(["--dataset", "carpk", "--policy", "oneshot", "--out", "x.csv"])
    assert args.dataset == "carpk"
    assert args.root is None and args.fmt is None


def test_parse_args_defaults_to_local_dataset():
    args = run_eval.parse_args(["--root", "r", "--fmt", "yolo", "--policy", "oneshot", "--out", "x.csv"])
    assert args.dataset == "local"
