"""
eval/report.py

Summarize one or more run_eval sweep CSVs: print a markdown results table
(rows = policy x verifier) and save the accuracy-vs-cost figure for the paper.
Matplotlib only (no seaborn); the plot uses a headless Agg figure.
"""

import argparse
import csv
import os

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from eval.metrics import mae, rmse, exact

# Marker style per verifier for the accuracy-vs-cost scatter.
_MARKERS = {"ioc": "o", "vip": "s", "off": "^"}


def load_rows(csv_paths) -> list:
    rows = []
    for path in csv_paths:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def aggregate(rows) -> list:
    """Group rows by (policy, verifier) and compute the summary metrics.

    MAE/RMSE/Exact are over N_obs vs N_gt; F1/pool_recall/cost are row means.
    """
    groups = {}
    for r in rows:
        groups.setdefault((r["policy"], r["verifier"]), []).append(r)

    out = []
    for (policy, verifier), rs in sorted(groups.items()):
        n_obs = [float(r["N_obs"]) for r in rs]
        n_gt = [float(r["N_gt"]) for r in rs]
        f1 = [float(r["f1"]) for r in rs]
        pr = [float(r["pool_recall"]) for r in rs]
        cost = [float(r["cost"]) for r in rs]
        out.append({
            "policy": policy,
            "verifier": verifier,
            "n": len(rs),
            "MAE": mae(n_obs, n_gt),
            "RMSE": rmse(n_obs, n_gt),
            "Exact": exact(n_obs, n_gt),
            "F1": sum(f1) / len(f1),
            "pool_recall": sum(pr) / len(pr),
            "cost": sum(cost) / len(cost),
        })
    return out


def markdown_table(agg) -> str:
    cols = ["policy", "verifier", "MAE", "RMSE", "Exact", "F1", "pool_recall", "cost"]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for a in agg:
        lines.append("| " + " | ".join([
            a["policy"], a["verifier"],
            f"{a['MAE']:.3f}", f"{a['RMSE']:.3f}", f"{a['Exact']:.3f}",
            f"{a['F1']:.3f}", f"{a['pool_recall']:.3f}", f"{a['cost']:.3f}",
        ]) + " |")
    return "\n".join(lines)


def accuracy_vs_cost_plot(agg, out_path):
    """Scatter: MAE (y) vs normalized cost (x), one point per policy x verifier,
    verifier encoded by marker style. Saved headlessly."""
    fig = Figure(figsize=(8, 6))
    FigureCanvasAgg(fig)
    ax = fig.subplots()

    labeled = set()
    for a in agg:
        marker = _MARKERS.get(a["verifier"], "x")
        label = a["verifier"] if a["verifier"] not in labeled else None
        labeled.add(a["verifier"])
        ax.scatter(a["cost"], a["MAE"], marker=marker, s=80, label=label)
        ax.annotate(a["policy"], (a["cost"], a["MAE"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("normalized cost")
    ax.set_ylabel("MAE (count error)")
    ax.set_title("Accuracy vs cost")
    if labeled:
        ax.legend(title="verifier")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Report sweep results as a table + plot.")
    parser.add_argument("--csv", nargs="+", required=True, help="One or more sweep CSVs.")
    parser.add_argument("--out", default="out/accuracy_vs_cost.png", help="Output plot path.")
    args = parser.parse_args()

    agg = aggregate(load_rows(args.csv))
    print(markdown_table(agg))
    accuracy_vs_cost_plot(agg, args.out)
    print(f"\nSaved plot to {args.out}")


if __name__ == "__main__":
    main()
