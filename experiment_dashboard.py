"""CLI entry point for the focused experiment dashboard.

A project-specific factory is required because model loading and ASHT presets are
machine dependent. The factory must return ``dashboard.DashboardDependencies``.
"""

from __future__ import annotations

import argparse
import importlib

from dashboard.app import build_app


def _load_factory(path: str):
    module_name, separator, attribute = path.partition(":")
    if not separator:
        raise ValueError("factory must use 'module:function' syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    return factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", required=True, help="module:function returning DashboardDependencies")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    dependencies = _load_factory(args.factory)()
    app = build_app(__import__("dashboard").DashboardService(dependencies))
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
