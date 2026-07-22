"""Unified, box-aware dataset adapters for experiment execution.

The adapters in this module are evaluation-only.  They never expose labels to
SAM3, Qwen, or the ASHT controller.  Their sole responsibility is to normalize
heterogeneous datasets into one immutable :class:`CanonicalSample` contract.

Supported adapters:

* ``GreenCitrusAdapter`` / ``YoloDatasetAdapter``
* ``CarpkAdapter``
* ``Fscd147Adapter``
* ``OmniCountAdapter``
* ``FourDSemanticMappingAdapter``
* ``GenericFolderAdapter``

All coordinates are global-frame ``xyxy`` pixels.  Optional dependencies stay
outside this module; every adapter works with the Python standard library,
NumPy, and Pillow only.
"""

from __future__ import annotations

import abc
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
Box = tuple[float, float, float, float]


class DatasetAdapterError(ValueError):
    """Raised when a dataset cannot be normalized safely."""


@dataclass(frozen=True)
class DatasetCapabilities:
    """Which annotations and navigation features a sample supports."""

    count: bool = False
    boxes: bool = False
    masks: bool = False
    exemplars: bool = False
    points: bool = False
    sequences: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {
            "count": self.count,
            "boxes": self.boxes,
            "masks": self.masks,
            "exemplars": self.exemplars,
            "points": self.points,
            "sequences": self.sequences,
        }


@dataclass(frozen=True)
class CanonicalSample:
    """One dataset sample in the pipeline's canonical coordinate frame."""

    dataset_name: str
    dataset_version: str
    split: str
    sample_key: str
    image_path: Path
    target_concept: str
    target_class: str
    sample_index: int | None = None
    sequence_id: str | None = None
    confounder_classes: tuple[str, ...] = ()
    ground_truth_count: int | None = None
    ground_truth_boxes: tuple[Box, ...] = ()
    ground_truth_mask_paths: tuple[Path, ...] = ()
    exemplar_boxes: tuple[Box, ...] = ()
    point_annotations: tuple[tuple[float, float], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    capabilities: DatasetCapabilities = field(default_factory=DatasetCapabilities)

    def __post_init__(self) -> None:
        if not self.dataset_name.strip():
            raise DatasetAdapterError("dataset_name must be non-empty")
        if not self.dataset_version.strip():
            raise DatasetAdapterError("dataset_version must be non-empty")
        if not self.split.strip():
            raise DatasetAdapterError("split must be non-empty")
        if not self.sample_key.strip():
            raise DatasetAdapterError("sample_key must be non-empty")
        if not self.target_concept.strip() or not self.target_class.strip():
            raise DatasetAdapterError("target concept and class must be non-empty")
        path = Path(self.image_path)
        if not path.is_file():
            raise DatasetAdapterError(f"image file does not exist: {path}")
        if self.sample_index is not None and self.sample_index < 0:
            raise DatasetAdapterError("sample_index must be non-negative")
        if self.ground_truth_count is not None and self.ground_truth_count < 0:
            raise DatasetAdapterError("ground_truth_count must be non-negative")
        for index, box in enumerate(self.ground_truth_boxes):
            _validate_box(box, f"ground_truth_boxes[{index}]")
        for index, box in enumerate(self.exemplar_boxes):
            _validate_box(box, f"exemplar_boxes[{index}]")
        for index, point in enumerate(self.point_annotations):
            if len(point) != 2 or not all(math.isfinite(float(value)) for value in point):
                raise DatasetAdapterError(f"point_annotations[{index}] is invalid")
        for path_value in self.ground_truth_mask_paths:
            if not Path(path_value).is_file():
                raise DatasetAdapterError(f"mask file does not exist: {path_value}")

    @property
    def image_size(self) -> tuple[int, int]:
        with Image.open(self.image_path) as image:
            return image.size

    @property
    def image_sha256(self) -> str:
        return _sha256_file(self.image_path)

    def load_image(self) -> Image.Image:
        with Image.open(self.image_path) as image:
            return image.convert("RGB")


class DatasetAdapter(abc.ABC):
    """Read-only adapter interface used by the unified experiment runner."""

    name: str
    version: str

    @abc.abstractmethod
    def splits(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def size(self, split: str) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, split: str, index: int) -> CanonicalSample:
        raise NotImplementedError

    def iter_samples(self, split: str) -> Iterator[CanonicalSample]:
        for index in range(self.size(split)):
            yield self.get(split, index)

    def summary(self, split: str) -> Mapping[str, Any]:
        samples = tuple(self.iter_samples(split))
        counts = [sample.ground_truth_count for sample in samples if sample.ground_truth_count is not None]
        return {
            "dataset_name": self.name,
            "dataset_version": self.version,
            "split": split,
            "sample_count": len(samples),
            "annotated_count_samples": len(counts),
            "total_ground_truth_count": int(sum(counts)) if counts else None,
            "box_annotated_samples": sum(bool(sample.ground_truth_boxes) for sample in samples),
            "sequence_count": len({sample.sequence_id for sample in samples if sample.sequence_id}),
        }


class DatasetRegistry:
    """Explicit registry; no dataset-specific branching in experiment code."""

    def __init__(self, adapters: Iterable[DatasetAdapter] = ()) -> None:
        self._adapters: dict[str, DatasetAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: DatasetAdapter, *, replace: bool = False) -> None:
        key = _normalize_name(adapter.name)
        if key in self._adapters and not replace:
            raise DatasetAdapterError(f"dataset adapter already registered: {adapter.name}")
        self._adapters[key] = adapter

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def adapter(self, name: str) -> DatasetAdapter:
        key = _normalize_name(name)
        try:
            return self._adapters[key]
        except KeyError as exc:
            raise DatasetAdapterError(
                f"unknown dataset {name!r}; registered datasets: {self.names()}"
            ) from exc

    def get(self, name: str, split: str, index: int) -> CanonicalSample:
        return self.adapter(name).get(split, index)


class GenericFolderAdapter(DatasetAdapter):
    """Recursive image folder with optional JSON sidecar annotations."""

    def __init__(
        self,
        root: str | Path,
        *,
        name: str = "generic",
        version: str = "local",
        target_concept: str = "object",
        target_class: str = "target",
        split: str = "all",
        recursive: bool = True,
        sidecar_suffix: str = ".json",
    ) -> None:
        self.root = Path(root)
        self.name = name
        self.version = version
        self.target_concept = target_concept
        self.target_class = target_class
        self.split_name = split
        self.recursive = recursive
        self.sidecar_suffix = sidecar_suffix
        if not self.root.is_dir():
            raise DatasetAdapterError(f"generic image root does not exist: {self.root}")
        iterator = self.root.rglob("*") if recursive else self.root.glob("*")
        self._images = tuple(
            sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        )

    def splits(self) -> tuple[str, ...]:
        return (self.split_name,)

    def size(self, split: str) -> int:
        self._check_split(split)
        return len(self._images)

    def get(self, split: str, index: int) -> CanonicalSample:
        self._check_split(split)
        path = _index(self._images, index)
        sidecar = path.with_suffix(self.sidecar_suffix)
        payload = _load_json(sidecar) if sidecar.is_file() else {}
        boxes = tuple(_coerce_box(value) for value in payload.get("boxes", ()))
        exemplar_boxes = tuple(_coerce_box(value) for value in payload.get("exemplar_boxes", ()))
        mask_paths = tuple(
            _resolve_optional_path(path.parent, value)
            for value in payload.get(
                "ground_truth_mask_paths", payload.get("mask_paths", ())
            )
        )
        count_value = payload.get("count")
        count = int(count_value) if count_value is not None else (len(boxes) if boxes else None)
        target_concept = str(payload.get("target_concept", self.target_concept))
        target_class = str(payload.get("target_class", self.target_class))
        confounders = tuple(str(value) for value in payload.get("confounder_classes", ()))
        sequence_id = payload.get("sequence_id")
        return CanonicalSample(
            dataset_name=self.name,
            dataset_version=self.version,
            split=split,
            sample_key=str(path.relative_to(self.root)),
            sample_index=index,
            image_path=path,
            target_concept=target_concept,
            target_class=target_class,
            sequence_id=str(sequence_id) if sequence_id is not None else None,
            confounder_classes=confounders,
            ground_truth_count=count,
            ground_truth_boxes=boxes,
            ground_truth_mask_paths=mask_paths,
            exemplar_boxes=exemplar_boxes,
            metadata={
                "relative_path": str(path.relative_to(self.root)),
                "sidecar_path": str(sidecar) if sidecar.is_file() else None,
                **dict(payload.get("metadata", {})),
            },
            capabilities=DatasetCapabilities(
                count=count is not None,
                boxes=bool(boxes),
                masks=bool(mask_paths),
                exemplars=bool(exemplar_boxes),
                sequences=sequence_id is not None,
            ),
        )

    def _check_split(self, split: str) -> None:
        if split != self.split_name:
            raise DatasetAdapterError(f"unknown split {split!r}; expected {self.split_name!r}")


class YoloDatasetAdapter(DatasetAdapter):
    """YOLO normalized-box dataset, including the green-citrus layout."""

    def __init__(
        self,
        root: str | Path,
        *,
        name: str,
        version: str = "local",
        target_concept: str,
        target_class: str,
        confounder_classes: Sequence[str] = (),
        class_ids: Sequence[int] | None = None,
        available_splits: Sequence[str] = ("train", "val", "test"),
    ) -> None:
        self.root = Path(root)
        self.name = name
        self.version = version
        self.target_concept = target_concept
        self.target_class = target_class
        self.confounder_classes = tuple(confounder_classes)
        self.class_ids = None if class_ids is None else frozenset(int(value) for value in class_ids)
        self._split_images: dict[str, tuple[Path, ...]] = {}
        for split in available_splits:
            image_dir = _resolve_kind_dir(self.root, "images", split)
            if image_dir is None:
                continue
            self._split_images[split] = tuple(_list_images(image_dir))
        if not self._split_images:
            image_dir = _resolve_kind_dir(self.root, "images", "all")
            if image_dir is None:
                raise DatasetAdapterError(f"could not locate images below {self.root}")
            self._split_images["all"] = tuple(_list_images(image_dir))

    def splits(self) -> tuple[str, ...]:
        return tuple(self._split_images)

    def size(self, split: str) -> int:
        return len(self._images(split))

    def get(self, split: str, index: int) -> CanonicalSample:
        path = _index(self._images(split), index)
        with Image.open(path) as image:
            width, height = image.size
        label_dir = _resolve_kind_dir(self.root, "labels", split)
        label_path = (label_dir / f"{path.stem}.txt") if label_dir is not None else None
        boxes, class_values = _parse_yolo_file(
            label_path,
            width=width,
            height=height,
            class_ids=self.class_ids,
        )
        return CanonicalSample(
            dataset_name=self.name,
            dataset_version=self.version,
            split=split,
            sample_key=path.stem,
            sample_index=index,
            image_path=path,
            target_concept=self.target_concept,
            target_class=self.target_class,
            confounder_classes=self.confounder_classes,
            ground_truth_count=len(boxes),
            ground_truth_boxes=boxes,
            metadata={
                "label_path": str(label_path) if label_path is not None else None,
                "class_ids": class_values,
            },
            capabilities=DatasetCapabilities(count=True, boxes=True),
        )

    def _images(self, split: str) -> tuple[Path, ...]:
        try:
            return self._split_images[split]
        except KeyError as exc:
            raise DatasetAdapterError(f"unknown split {split!r}; available: {self.splits()}") from exc


class GreenCitrusAdapter(YoloDatasetAdapter):
    def __init__(
        self,
        root: str | Path,
        *,
        version: str = "local",
        class_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            root,
            name="green_citrus",
            version=version,
            target_concept="green citrus fruit",
            target_class="fruit",
            confounder_classes=("leaf", "background"),
            class_ids=class_ids,
        )


class CarpkAdapter(DatasetAdapter):
    """Official CARPK-style ``Images/Annotations/ImageSets`` layout."""

    def __init__(
        self,
        root: str | Path,
        *,
        version: str = "official",
        split_names: Sequence[str] = ("train", "test"),
    ) -> None:
        self.root = Path(root)
        self.name = "carpk"
        self.version = version
        self.image_dir = _first_existing_dir(self.root / "Images", self.root / "images")
        self.annotation_dir = _first_existing_dir(
            self.root / "Annotations", self.root / "annotations"
        )
        if self.image_dir is None or self.annotation_dir is None:
            raise DatasetAdapterError("CARPK requires Images and Annotations directories")
        self._split_stems: dict[str, tuple[str, ...]] = {}
        for split in split_names:
            split_file = _first_existing_file(
                self.root / "ImageSets" / f"{split}.txt",
                self.root / "ImageSets" / "Main" / f"{split}.txt",
            )
            if split_file is not None:
                stems = tuple(line.strip().split()[0] for line in split_file.read_text().splitlines() if line.strip())
                self._split_stems[split] = stems
        if not self._split_stems:
            self._split_stems["all"] = tuple(path.stem for path in _list_images(self.image_dir))

    def splits(self) -> tuple[str, ...]:
        return tuple(self._split_stems)

    def size(self, split: str) -> int:
        return len(self._stems(split))

    def get(self, split: str, index: int) -> CanonicalSample:
        stem = _index(self._stems(split), index)
        image_path = _find_image_by_stem(self.image_dir, stem)
        annotation_path = self.annotation_dir / f"{stem}.txt"
        boxes = _parse_xyxy_text(annotation_path)
        return CanonicalSample(
            dataset_name=self.name,
            dataset_version=self.version,
            split=split,
            sample_key=stem,
            sample_index=index,
            image_path=image_path,
            target_concept="car",
            target_class="car",
            confounder_classes=("non_car", "background"),
            ground_truth_count=len(boxes),
            ground_truth_boxes=boxes,
            metadata={"annotation_path": str(annotation_path)},
            capabilities=DatasetCapabilities(count=True, boxes=True),
        )

    def _stems(self, split: str) -> tuple[str, ...]:
        try:
            return self._split_stems[split]
        except KeyError as exc:
            raise DatasetAdapterError(f"unknown split {split!r}; available: {self.splits()}") from exc


class Fscd147Adapter(DatasetAdapter):
    """FSCD/FSC-147 adapter for the standard annotation and split JSON files."""

    def __init__(
        self,
        root: str | Path,
        *,
        version: str = "official",
        annotation_file: str | Path = "annotation_FSC147_384.json",
        split_file: str | Path = "Train_Test_Val_FSC_147.json",
        image_directory: str | Path = "images_384_VarV2",
    ) -> None:
        self.root = Path(root)
        self.name = "fscd147"
        self.version = version
        self.annotation_path = _resolve_file(self.root, annotation_file)
        self.split_path = _resolve_file(self.root, split_file)
        self.image_dir = _resolve_dir(self.root, image_directory)
        self._annotations = _load_json(self.annotation_path)
        split_payload = _load_json(self.split_path)
        self._splits = _normalize_split_mapping(split_payload)

    def splits(self) -> tuple[str, ...]:
        return tuple(self._splits)

    def size(self, split: str) -> int:
        return len(self._files(split))

    def get(self, split: str, index: int) -> CanonicalSample:
        filename = str(_index(self._files(split), index))
        entry = self._annotations.get(filename)
        if entry is None:
            entry = self._annotations.get(Path(filename).name, {})
        if not isinstance(entry, Mapping):
            raise DatasetAdapterError(f"invalid FSCD annotation for {filename}")
        image_path = self.image_dir / Path(filename).name
        if not image_path.is_file():
            candidate = self.root / filename
            image_path = candidate if candidate.is_file() else image_path
        points = tuple(_coerce_point(value) for value in entry.get("points", ()))
        exemplars = tuple(
            _quadrilateral_or_box_to_xyxy(value)
            for value in entry.get("box_examples_coordinates", entry.get("exemplar_boxes", ()))
        )
        explicit_boxes = tuple(
            _quadrilateral_or_box_to_xyxy(value)
            for value in entry.get("boxes", entry.get("gt_boxes", ()))
        )
        count = len(points) if points else len(explicit_boxes)
        concept = str(
            entry.get("category", entry.get("class", entry.get("label", "target object")))
        )
        return CanonicalSample(
            dataset_name=self.name,
            dataset_version=self.version,
            split=split,
            sample_key=Path(filename).stem,
            sample_index=index,
            image_path=image_path,
            target_concept=concept,
            target_class="target",
            confounder_classes=("non_target", "background"),
            ground_truth_count=count,
            ground_truth_boxes=explicit_boxes,
            exemplar_boxes=exemplars,
            point_annotations=points,
            metadata={
                "annotation_file": str(self.annotation_path),
                "split_file": str(self.split_path),
                "source_filename": filename,
                "raw_annotation_keys": sorted(str(key) for key in entry),
            },
            capabilities=DatasetCapabilities(
                count=True,
                boxes=bool(explicit_boxes),
                exemplars=bool(exemplars),
                points=bool(points),
            ),
        )

    def _files(self, split: str) -> tuple[str, ...]:
        try:
            return self._splits[split]
        except KeyError as exc:
            raise DatasetAdapterError(f"unknown split {split!r}; available: {self.splits()}") from exc


class OmniCountAdapter(DatasetAdapter):
    """COCO-style OmniCount adapter, indexed by image and target category."""

    def __init__(
        self,
        root: str | Path,
        annotation_file: str | Path,
        *,
        version: str = "official",
        image_directory: str | Path = "images",
        category_names: Sequence[str] | None = None,
        split: str = "test",
    ) -> None:
        self.root = Path(root)
        self.name = "omnicount"
        self.version = version
        self.split_name = split
        self.annotation_path = _resolve_file(self.root, annotation_file)
        self.image_dir = _resolve_dir(self.root, image_directory)
        payload = _load_json(self.annotation_path)
        categories = {int(item["id"]): str(item["name"]) for item in payload.get("categories", ())}
        wanted = None if category_names is None else {value.lower() for value in category_names}
        images = {int(item["id"]): item for item in payload.get("images", ())}
        grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
        for annotation in payload.get("annotations", ()):
            category_id = int(annotation["category_id"])
            category_name = categories.get(category_id, str(category_id))
            if wanted is not None and category_name.lower() not in wanted:
                continue
            key = (int(annotation["image_id"]), category_id)
            grouped.setdefault(key, []).append(annotation)
        self._entries = tuple(
            (image_id, category_id, tuple(annotations))
            for (image_id, category_id), annotations in sorted(grouped.items())
            if image_id in images
        )
        self._images = images
        self._categories = categories

    def splits(self) -> tuple[str, ...]:
        return (self.split_name,)

    def size(self, split: str) -> int:
        self._check_split(split)
        return len(self._entries)

    def get(self, split: str, index: int) -> CanonicalSample:
        self._check_split(split)
        image_id, category_id, annotations = _index(self._entries, index)
        image_info = self._images[image_id]
        image_path = self.image_dir / str(image_info["file_name"])
        boxes = tuple(_coco_bbox_to_xyxy(annotation["bbox"]) for annotation in annotations)
        category = self._categories.get(category_id, str(category_id))
        segmentation_count = sum(bool(annotation.get("segmentation")) for annotation in annotations)
        return CanonicalSample(
            dataset_name=self.name,
            dataset_version=self.version,
            split=split,
            sample_key=f"{image_id}:{category_id}",
            sample_index=index,
            image_path=image_path,
            target_concept=category,
            target_class=category,
            confounder_classes=("other_object", "background"),
            ground_truth_count=len(boxes),
            ground_truth_boxes=boxes,
            metadata={
                "coco_image_id": image_id,
                "coco_category_id": category_id,
                "annotation_ids": [annotation.get("id") for annotation in annotations],
                "segmentation_annotation_count": segmentation_count,
                "annotation_file": str(self.annotation_path),
            },
            capabilities=DatasetCapabilities(
                count=True,
                boxes=True,
                masks=segmentation_count == len(annotations) and bool(annotations),
            ),
        )

    def _check_split(self, split: str) -> None:
        if split != self.split_name:
            raise DatasetAdapterError(f"unknown split {split!r}; expected {self.split_name!r}")


class FourDSemanticMappingAdapter(DatasetAdapter):
    """RGB-frame adapter for the 4D semantic-mapping orchard data.

    A manifest is preferred because it preserves sequence, species, date, and
    annotation metadata.  JSON, JSONL, and CSV manifests are supported.  When no
    manifest is supplied, the adapter recursively indexes images and derives a
    conservative sequence ID from the parent directory.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        manifest: str | Path | None = None,
        version: str = "local",
        split: str = "all",
        target_concept: str = "fruit",
        target_class: str = "fruit",
    ) -> None:
        self.root = Path(root)
        self.name = "4d_semantic_mapping"
        self.version = version
        self.split_name = split
        self.default_target_concept = target_concept
        self.default_target_class = target_class
        if not self.root.is_dir():
            raise DatasetAdapterError(f"4D dataset root does not exist: {self.root}")
        if manifest is not None:
            manifest_path = _resolve_file(self.root, manifest)
            rows = _load_manifest(manifest_path)
            self._rows = tuple(row for row in rows if str(row.get("split", split)) == split)
            self.manifest_path = manifest_path
        else:
            self.manifest_path = None
            self._rows = tuple(
                {
                    "image_path": str(path.relative_to(self.root)),
                    "sequence_id": str(path.parent.relative_to(self.root)),
                    "split": split,
                }
                for path in sorted(self.root.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

    def splits(self) -> tuple[str, ...]:
        return (self.split_name,)

    def size(self, split: str) -> int:
        self._check_split(split)
        return len(self._rows)

    def get(self, split: str, index: int) -> CanonicalSample:
        self._check_split(split)
        row = dict(_index(self._rows, index))
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = self.root / image_path
        boxes = tuple(_coerce_box(value) for value in _coerce_nested_json(row.get("boxes", ())))
        exemplar_boxes = tuple(
            _coerce_box(value) for value in _coerce_nested_json(row.get("exemplar_boxes", ()))
        )
        mask_paths = tuple(
            _resolve_optional_path(self.root, value)
            for value in _coerce_nested_json(
                row.get("ground_truth_mask_paths", row.get("mask_paths", ()))
            )
        )
        count_value = row.get("ground_truth_count", row.get("count"))
        count = int(count_value) if count_value not in (None, "") else (len(boxes) if boxes else None)
        sequence_id = row.get("sequence_id", row.get("sequence"))
        target_concept = str(row.get("target_concept", row.get("species", self.default_target_concept)))
        target_class = str(row.get("target_class", self.default_target_class))
        reserved = {
            "image_path", "split", "boxes", "exemplar_boxes", "mask_paths",
            "ground_truth_mask_paths", "count", "ground_truth_count",
            "sequence", "sequence_id", "target_concept", "target_class", "confounder_classes",
        }
        metadata = {key: value for key, value in row.items() if key not in reserved}
        return CanonicalSample(
            dataset_name=self.name,
            dataset_version=self.version,
            split=split,
            sample_key=str(row.get("sample_key", image_path.stem)),
            sample_index=index,
            image_path=image_path,
            target_concept=target_concept,
            target_class=target_class,
            sequence_id=str(sequence_id) if sequence_id not in (None, "") else None,
            confounder_classes=tuple(
                str(value) for value in _coerce_nested_json(row.get("confounder_classes", ()))
            ),
            ground_truth_count=count,
            ground_truth_boxes=boxes,
            ground_truth_mask_paths=mask_paths,
            exemplar_boxes=exemplar_boxes,
            metadata={
                "manifest_path": str(self.manifest_path) if self.manifest_path else None,
                **metadata,
            },
            capabilities=DatasetCapabilities(
                count=count is not None,
                boxes=bool(boxes),
                masks=bool(mask_paths),
                exemplars=bool(exemplar_boxes),
                sequences=sequence_id not in (None, ""),
            ),
        )

    def _check_split(self, split: str) -> None:
        if split != self.split_name:
            raise DatasetAdapterError(f"unknown split {split!r}; expected {self.split_name!r}")


def validate_adapter(adapter: DatasetAdapter, split: str) -> Mapping[str, Any]:
    """Eagerly validate all samples and report integrity diagnostics."""

    errors: list[str] = []
    keys: set[str] = set()
    sample_count = adapter.size(split)
    for index in range(sample_count):
        try:
            sample = adapter.get(split, index)
            if sample.sample_key in keys:
                errors.append(f"duplicate sample_key {sample.sample_key!r}")
            keys.add(sample.sample_key)
            width, height = sample.image_size
            for box_index, box in enumerate(sample.ground_truth_boxes):
                if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                    errors.append(
                        f"{sample.sample_key}: box {box_index} outside image bounds {width}x{height}"
                    )
        except Exception as exc:  # validation should collect all failures
            errors.append(f"index {index}: {type(exc).__name__}: {exc}")
    return {
        "dataset_name": adapter.name,
        "dataset_version": adapter.version,
        "split": split,
        "sample_count": sample_count,
        "valid": not errors,
        "errors": errors,
    }


def _normalize_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        raise DatasetAdapterError("dataset name must be non-empty")
    return normalized


def _resolve_kind_dir(root: Path, kind: str, split: str) -> Path | None:
    candidates = (
        root / kind / split,
        root / split / kind,
        root / kind,
    )
    return _first_existing_dir(*candidates)


def _list_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _parse_yolo_file(
    path: Path | None,
    *,
    width: int,
    height: int,
    class_ids: frozenset[int] | None,
) -> tuple[tuple[Box, ...], tuple[int, ...]]:
    boxes: list[Box] = []
    observed_classes: list[int] = []
    if path is None or not path.is_file():
        return (), ()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            xc, yc, box_width, box_height = (float(value) for value in parts[1:5])
        except ValueError as exc:
            raise DatasetAdapterError(f"invalid YOLO row at {path}:{line_number}") from exc
        if class_ids is not None and class_id not in class_ids:
            continue
        x1 = (xc - box_width / 2.0) * width
        y1 = (yc - box_height / 2.0) * height
        x2 = (xc + box_width / 2.0) * width
        y2 = (yc + box_height / 2.0) * height
        box = _clamp_box((x1, y1, x2, y2), width, height)
        boxes.append(box)
        observed_classes.append(class_id)
    return tuple(boxes), tuple(observed_classes)


def _parse_xyxy_text(path: Path) -> tuple[Box, ...]:
    if not path.is_file():
        return ()
    boxes: list[Box] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 4:
            continue
        try:
            values = tuple(float(value) for value in parts[:4])
        except ValueError as exc:
            raise DatasetAdapterError(f"invalid box at {path}:{line_number}") from exc
        boxes.append(_coerce_box(values))
    return tuple(boxes)


def _normalize_split_mapping(payload: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(payload, Mapping):
        raise DatasetAdapterError("split JSON must be an object")
    aliases = {"training": "train", "validation": "val", "testing": "test"}
    result: dict[str, tuple[str, ...]] = {}
    for key, value in payload.items():
        split = aliases.get(str(key).lower(), str(key).lower())
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            continue
        result[split] = tuple(str(item) for item in value)
    if not result:
        raise DatasetAdapterError("split JSON contains no usable split lists")
    return result


def _quadrilateral_or_box_to_xyxy(value: Any) -> Box:
    if isinstance(value, Mapping):
        if "bbox" in value:
            return _quadrilateral_or_box_to_xyxy(value["bbox"])
        keys = ("x1", "y1", "x2", "y2")
        if all(key in value for key in keys):
            return _coerce_box(tuple(value[key] for key in keys))
    arr = np.asarray(value, dtype=float)
    if arr.shape == (4,):
        return _coerce_box(tuple(float(item) for item in arr))
    if arr.ndim == 2 and arr.shape[1] >= 2:
        xs = arr[:, 0]
        ys = arr[:, 1]
        return _coerce_box((float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())))
    raise DatasetAdapterError(f"cannot interpret exemplar/box coordinates: {value!r}")


def _coco_bbox_to_xyxy(value: Sequence[float]) -> Box:
    if len(value) < 4:
        raise DatasetAdapterError("COCO bbox must contain x, y, width, height")
    x, y, width, height = (float(item) for item in value[:4])
    return _coerce_box((x, y, x + width, y + height))


def _coerce_box(value: Any) -> Box:
    if isinstance(value, Mapping):
        if "bbox" in value:
            value = value["bbox"]
        else:
            value = (value["x1"], value["y1"], value["x2"], value["y2"])
    if len(value) != 4:
        raise DatasetAdapterError(f"box must have four coordinates: {value!r}")
    box = tuple(float(item) for item in value)
    _validate_box(box, "box")
    return box  # type: ignore[return-value]


def _coerce_point(value: Any) -> tuple[float, float]:
    if isinstance(value, Mapping):
        value = (value["x"], value["y"])
    if len(value) < 2:
        raise DatasetAdapterError(f"point must have two coordinates: {value!r}")
    point = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in point):
        raise DatasetAdapterError("point coordinates must be finite")
    return point


def _validate_box(box: Sequence[float], name: str) -> None:
    if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
        raise DatasetAdapterError(f"{name} must contain four finite values")
    if float(box[2]) < float(box[0]) or float(box[3]) < float(box[1]):
        raise DatasetAdapterError(f"{name} must satisfy x2 >= x1 and y2 >= y1")


def _clamp_box(box: Box, width: int, height: int) -> Box:
    return (
        max(0.0, min(float(width), box[0])),
        max(0.0, min(float(height), box[1])),
        max(0.0, min(float(width), box[2])),
        max(0.0, min(float(height), box[3])),
    )


def _find_image_by_stem(directory: Path, stem: str) -> Path:
    for extension in sorted(IMAGE_EXTENSIONS):
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
        upper = directory / f"{stem}{extension.upper()}"
        if upper.is_file():
            return upper
    matches = tuple(path for path in directory.glob(f"{stem}.*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not matches:
        raise DatasetAdapterError(f"image for stem {stem!r} not found in {directory}")
    return sorted(matches)[0]


def _first_existing_dir(*paths: Path) -> Path | None:
    return next((path for path in paths if path.is_dir()), None)


def _first_existing_file(*paths: Path) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _resolve_file(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise DatasetAdapterError(f"file does not exist: {path}")
    return path


def _resolve_dir(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if not path.is_dir():
        raise DatasetAdapterError(f"directory does not exist: {path}")
    return path


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetAdapterError(f"failed to read JSON {path}: {exc}") from exc


def _load_manifest(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = _load_json(path)
        if isinstance(payload, Mapping):
            payload = payload.get("samples", payload.get("frames", payload.get("images", ())))
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise DatasetAdapterError("JSON manifest must contain a list of samples")
        return [dict(item) for item in payload]
    if suffix == ".jsonl":
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetAdapterError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(item, Mapping):
                raise DatasetAdapterError(f"JSONL row {line_number} is not an object")
            rows.append(dict(item))
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise DatasetAdapterError(f"unsupported manifest format: {path.suffix}")


def _coerce_nested_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        if stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetAdapterError(f"invalid JSON-encoded manifest value: {value}") from exc
        return tuple(item.strip() for item in stripped.split(";") if item.strip())
    return value


def _resolve_optional_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index(values: Sequence[Any], index: int) -> Any:
    if index < 0 or index >= len(values):
        raise IndexError(f"sample index {index} outside [0, {len(values)})")
    return values[index]


__all__ = [
    "CanonicalSample",
    "CarpkAdapter",
    "DatasetAdapter",
    "DatasetAdapterError",
    "DatasetCapabilities",
    "DatasetRegistry",
    "FourDSemanticMappingAdapter",
    "Fscd147Adapter",
    "GenericFolderAdapter",
    "GreenCitrusAdapter",
    "OmniCountAdapter",
    "YoloDatasetAdapter",
    "validate_adapter",
]
