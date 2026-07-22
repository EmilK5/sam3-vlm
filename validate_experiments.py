"""Command-line validation and reproducibility checks for saved experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validation import compare_run_semantics, validate_run_directory, validate_run_tree


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate one or more run directories")
    validate.add_argument("paths", nargs="+", type=Path)
    validate.add_argument("--no-artifacts", action="store_true")
    validate.add_argument("--no-event-parity", action="store_true")
    validate.add_argument("--no-graph-replay", action="store_true")
    validate.add_argument("--json", action="store_true", dest="as_json")
    scan = sub.add_parser("scan", help="validate every run.json below a root")
    scan.add_argument("root", type=Path)
    scan.add_argument("--json", action="store_true", dest="as_json")
    compare = sub.add_parser("compare", help="compare the semantic trace of two runs")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate":
        reports = [
            validate_run_directory(
                path,
                verify_artifacts=not args.no_artifacts,
                verify_event_parity=not args.no_event_parity,
                verify_graph_replay=not args.no_graph_replay,
            )
            for path in args.paths
        ]
        return _print_reports(reports, args.as_json)
    if args.command == "scan":
        reports = validate_run_tree(args.root)
        return _print_reports(reports, args.as_json)
    comparison = compare_run_semantics(args.left, args.right)
    payload = {
        "equal": comparison.equal,
        "left_fingerprint": comparison.left_fingerprint,
        "right_fingerprint": comparison.right_fingerprint,
        "differences": list(comparison.differences),
    }
    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("MATCH" if comparison.equal else "DIFFERENT")
        print(f"left:  {comparison.left_fingerprint}")
        print(f"right: {comparison.right_fingerprint}")
        for difference in comparison.differences:
            print(f"- {difference}")
    return 0 if comparison.equal else 1


def _print_reports(reports, as_json: bool) -> int:
    if as_json:
        print(json.dumps([report.to_dict() for report in reports], indent=2, sort_keys=True))
    else:
        for report in reports:
            print(f"{'PASS' if report.valid else 'FAIL'} {report.run_directory}")
            if report.semantic_fingerprint:
                print(f"  fingerprint: {report.semantic_fingerprint}")
            for issue in report.issues:
                print(f"  {issue.severity.upper()} [{issue.code}] {issue.message}")
    return 0 if all(report.valid for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
