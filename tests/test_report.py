import csv

from eval import report


def _write_csv(path, rows):
    fields = ["image", "policy", "verifier", "N_gt", "N_obs", "f1", "pool_recall", "cost"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _rows():
    return [
        {"image": "a", "policy": "oneshot", "verifier": "ioc", "N_gt": 10, "N_obs": 8,
         "f1": 0.7, "pool_recall": 0.6, "cost": 1.0},
        {"image": "b", "policy": "oneshot", "verifier": "ioc", "N_gt": 10, "N_obs": 12,
         "f1": 0.9, "pool_recall": 0.8, "cost": 1.0},
        {"image": "a", "policy": "cascade", "verifier": "vip", "N_gt": 10, "N_obs": 10,
         "f1": 0.95, "pool_recall": 0.9, "cost": 4.0},
    ]


def test_load_and_aggregate(tmp_path):
    p = tmp_path / "sweep.csv"
    _write_csv(p, _rows())
    agg = report.aggregate(report.load_rows([str(p)]))

    by_key = {(a["policy"], a["verifier"]): a for a in agg}
    oneshot = by_key[("oneshot", "ioc")]
    assert oneshot["n"] == 2
    # |8-10| and |12-10| -> MAE 2.0
    assert oneshot["MAE"] == 2.0
    assert oneshot["cost"] == 1.0
    assert round(oneshot["F1"], 2) == 0.80

    cascade = by_key[("cascade", "vip")]
    assert cascade["MAE"] == 0.0        # exact count
    assert cascade["Exact"] == 1.0


def test_markdown_table_contains_rows(tmp_path):
    p = tmp_path / "sweep.csv"
    _write_csv(p, _rows())
    table = report.markdown_table(report.aggregate(report.load_rows([str(p)])))
    assert "| policy | verifier | MAE" in table
    assert "oneshot" in table and "cascade" in table
    # two data rows + header + separator
    assert len(table.splitlines()) == 4


def test_plot_is_saved(tmp_path):
    p = tmp_path / "sweep.csv"
    _write_csv(p, _rows())
    agg = report.aggregate(report.load_rows([str(p)]))
    out = tmp_path / "acc_vs_cost.png"
    report.accuracy_vs_cost_plot(agg, str(out))
    assert out.exists() and out.stat().st_size > 0
