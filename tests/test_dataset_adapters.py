import json
from pathlib import Path

from PIL import Image

from eval.dataset_adapters import (
    CarpkAdapter,
    DatasetRegistry,
    FourDSemanticMappingAdapter,
    Fscd147Adapter,
    GenericFolderAdapter,
    GreenCitrusAdapter,
    OmniCountAdapter,
    validate_adapter,
)


def _image(path: Path, size=(100, 80)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (20, 40, 60)).save(path)


def test_generic_folder_sidecar_and_registry(tmp_path):
    image_path = tmp_path / "images" / "a.png"
    _image(image_path)
    image_path.with_suffix(".json").write_text(
        json.dumps({"count": 2, "boxes": [[1, 2, 10, 20], [20, 10, 30, 25]]})
    )
    adapter = GenericFolderAdapter(tmp_path / "images", target_concept="car")
    registry = DatasetRegistry([adapter])
    sample = registry.get("generic", "all", 0)
    assert sample.ground_truth_count == 2
    assert len(sample.ground_truth_boxes) == 2
    assert sample.image_size == (100, 80)
    assert validate_adapter(adapter, "all")["valid"] is True


def test_green_citrus_yolo_layout(tmp_path):
    _image(tmp_path / "images" / "test" / "tree.jpg", size=(200, 100))
    label = tmp_path / "labels" / "test" / "tree.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.1 0.1\n")
    adapter = GreenCitrusAdapter(tmp_path, class_ids=(0,))
    sample = adapter.get("test", 0)
    assert sample.ground_truth_count == 1
    assert sample.ground_truth_boxes[0] == (80.0, 30.0, 120.0, 70.0)
    assert sample.confounder_classes == ("leaf", "background")


def test_carpk_boxes_and_split(tmp_path):
    _image(tmp_path / "Images" / "0001.png")
    (tmp_path / "Annotations").mkdir()
    (tmp_path / "Annotations" / "0001.txt").write_text("1 2 11 12\n20 21 30 31 0\n")
    (tmp_path / "ImageSets").mkdir()
    (tmp_path / "ImageSets" / "test.txt").write_text("0001\n")
    adapter = CarpkAdapter(tmp_path, split_names=("test",))
    sample = adapter.get("test", 0)
    assert sample.target_concept == "car"
    assert sample.ground_truth_count == 2
    assert sample.ground_truth_boxes[1] == (20.0, 21.0, 30.0, 31.0)


def test_fscd147_points_and_exemplars(tmp_path):
    _image(tmp_path / "images_384_VarV2" / "sample.jpg")
    (tmp_path / "annotation_FSC147_384.json").write_text(
        json.dumps(
            {
                "sample.jpg": {
                    "points": [[5, 6], [10, 11], [20, 21]],
                    "box_examples_coordinates": [
                        [[1, 2], [8, 2], [8, 9], [1, 9]]
                    ],
                    "category": "apples",
                }
            }
        )
    )
    (tmp_path / "Train_Test_Val_FSC_147.json").write_text(
        json.dumps({"train": ["sample.jpg"], "val": [], "test": []})
    )
    adapter = Fscd147Adapter(tmp_path)
    sample = adapter.get("train", 0)
    assert sample.ground_truth_count == 3
    assert sample.exemplar_boxes == ((1.0, 2.0, 8.0, 9.0),)
    assert sample.capabilities.points is True
    assert sample.target_concept == "apples"


def test_omnicount_coco_categories(tmp_path):
    _image(tmp_path / "images" / "scene.jpg")
    annotations = {
        "images": [{"id": 1, "file_name": "scene.jpg"}],
        "categories": [{"id": 3, "name": "orange"}, {"id": 4, "name": "apple"}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 3, "bbox": [1, 2, 10, 20]},
            {"id": 11, "image_id": 1, "category_id": 3, "bbox": [20, 2, 8, 9]},
            {"id": 12, "image_id": 1, "category_id": 4, "bbox": [3, 4, 5, 6]},
        ],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(annotations))
    adapter = OmniCountAdapter(tmp_path, "annotations.json", category_names=("orange",))
    sample = adapter.get("test", 0)
    assert sample.target_concept == "orange"
    assert sample.ground_truth_count == 2
    assert sample.ground_truth_boxes[0] == (1.0, 2.0, 11.0, 22.0)


def test_4d_manifest_metadata_and_annotations(tmp_path):
    _image(tmp_path / "frames" / "row1" / "0001.jpg")
    manifest = [
        {
            "image_path": "frames/row1/0001.jpg",
            "split": "test",
            "sequence_id": "row1",
            "species": "orange",
            "date": "2026-05-01",
            "count": 1,
            "boxes": [[1, 2, 10, 12]],
        }
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    adapter = FourDSemanticMappingAdapter(
        tmp_path, manifest="manifest.json", split="test"
    )
    sample = adapter.get("test", 0)
    assert sample.sequence_id == "row1"
    assert sample.target_concept == "orange"
    assert sample.metadata["date"] == "2026-05-01"
    assert sample.ground_truth_boxes == ((1.0, 2.0, 10.0, 12.0),)


def test_adapter_factory_and_summary_export(tmp_path):
    from eval.adapter_factory import (
        DatasetAdapterSpec,
        build_registry,
        export_dataset_summary,
    )

    image_root = tmp_path / "folder"
    _image(image_root / "one.png")
    registry = build_registry(
        [
            DatasetAdapterSpec(
                kind="generic",
                root=image_root,
                options={"name": "demo", "target_concept": "object"},
            )
        ]
    )
    adapter = registry.adapter("demo")
    output = export_dataset_summary(adapter, "all", tmp_path / "summary.json")
    assert json.loads(output.read_text())["sample_count"] == 1
