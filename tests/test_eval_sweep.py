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
    def __new__(cls, val, post_verify=0):
        obj = super().__new__(cls, val)
        obj.post_verify = post_verify
        return obj


def _stub(per_pass):
    """execute_pass stand-in: returns per_pass[i] and adds that many fruit nodes."""
    state = {"i": 0}

    def fn(**kw):
        graph, pass_number = kw["graph"], kw["pass_number"]
        n = per_pass[min(state["i"], len(per_pass) - 1)]
        for j in range(n):
            nid = graph.add_candidate([j * 10, 0, j * 10 + 8, 8], 0.9, found_in_pass=pass_number)
            graph.nodes[nid].classification = "fruit"
        state["i"] += 1
        return _FakeStats(n, post_verify=n)

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
    assert cost.n_verify == 4  # sum of post_verify


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
    row.update({"image": "a.png", "policy": "oneshot", "verifier": "ioc"})
    run_eval.append_row(csv_path, row)

    keys = run_eval.existing_keys(csv_path)
    assert ("a.png", "oneshot", "ioc") in keys
    assert ("a.png", "cascade", "ioc") not in keys
