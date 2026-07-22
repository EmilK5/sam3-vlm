"""Resume-safe sweeps using the unified runner and canonical run index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from eval.run_index import RunIndex
from experiments.runner import UnifiedExperimentResult, UnifiedExperimentRunner
from provenance.io import sha256_json


@dataclass(frozen=True)
class SweepResult:
    completed: tuple[UnifiedExperimentResult, ...]
    skipped_run_directories: tuple[str, ...]
    failed: tuple[tuple[int, str], ...]


def run_sweep(
    runner: UnifiedExperimentRunner,
    *,
    dataset_name: str,
    split: str,
    indices: Iterable[int],
    resume: bool = True,
) -> SweepResult:
    index = RunIndex(runner.config.output_root)
    completed, skipped, failed = [], [], []
    for sample_index in indices:
        sample = runner.datasets.get(dataset_name, split, int(sample_index))
        resolved = dict(runner.resolved_config_for_sample(sample))
        if resume:
            existing = index.find_complete(
                dataset_name=sample.dataset_name,
                split=sample.split,
                sample_key=sample.sample_key,
                image_sha256=sample.image_sha256,
                config_sha256=sha256_json(resolved),
            )
            if existing is not None:
                skipped.append(str(existing.directory))
                continue
        try:
            completed.append(runner.run_sample(sample))
        except Exception as exc:
            failed.append((int(sample_index), f"{type(exc).__name__}: {exc}"))
    return SweepResult(tuple(completed), tuple(skipped), tuple(failed))


__all__ = ["SweepResult", "run_sweep"]
