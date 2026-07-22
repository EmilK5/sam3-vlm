"""Run discovery, compatibility checks, replay indexing, and resume decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from provenance.contracts import RunRecord
from provenance.io import strict_json_load
from provenance.schema import RunStatus


@dataclass(frozen=True)
class IndexedRun:
    directory: Path
    run: RunRecord


class RunIndex:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def scan(self, *, successful_only: bool = False) -> tuple[IndexedRun, ...]:
        records = []
        if not self.root.exists():
            return ()
        for path in sorted(self.root.rglob("run.json")):
            try:
                run = RunRecord.from_dict(strict_json_load(path))
            except Exception:
                continue
            if successful_only and run.status is not RunStatus.SUCCEEDED:
                continue
            records.append(IndexedRun(path.parent, run))
        records.sort(key=lambda item: (item.run.created_at, item.run.run_id), reverse=True)
        return tuple(records)

    def find_complete(
        self,
        *,
        dataset_name: str,
        split: str,
        sample_key: str,
        image_sha256: str,
        config_sha256: str,
    ) -> IndexedRun | None:
        for indexed in self.scan(successful_only=True):
            run = indexed.run
            if (
                run.dataset_sample.dataset_name == dataset_name
                and run.dataset_sample.split == split
                and run.dataset_sample.metadata.get("sample_key") == sample_key
                and run.dataset_sample.image_sha256 == image_sha256
                and run.config_sha256 == config_sha256
            ):
                return indexed
        return None

    def directories(self, *, successful_only: bool = False) -> tuple[Path, ...]:
        return tuple(item.directory for item in self.scan(successful_only=successful_only))


__all__ = ["IndexedRun", "RunIndex"]
