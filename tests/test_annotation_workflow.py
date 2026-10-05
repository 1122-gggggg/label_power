import json

from PIL import Image
import pytest

from annotation_session import DatasetConfig, Session
from annotation_store import Box, LabelError, ReviewStore, collect_images, read_yolo_boxes, write_yolo_boxes
from annotation_workflow import export_verified, reference_crops


@pytest.fixture
def dataset(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for name in ("flight_000001_000100ms.png", "flight_000002_000200ms.png"):
        Image.new("RGB", (100, 80), "navy").save(images / name)
    classes = tmp_path / "classes.txt"
    classes.write_text("insulator\npole\n", encoding="utf-8")
    return DatasetConfig(images, labels, classes, labels / ".annotation" / "review.json")


def confirmed_dataset(config):
    images = collect_images(config.images_dir)
    store = ReviewStore(config.pass_state, images, ["insulator", "pole"], labels_dir=config.labels_dir)
    write_yolo_boxes(config.labels_dir / f"{images[0].stem}.txt", [Box(0, 10, 10, 30, 30)], 100, 80, 2)
    write_yolo_boxes(config.labels_dir / f"{images[1].stem}.txt", [], 100, 80, 2)
    for image in images:
        for cls in range(2):
            store.confirm(image.name, cls, image == images[0] and cls == 0)
    return store


def test_profile_relative_paths_resolve_from_profile_not_working_directory(dataset, tmp_path, monkeypatch):
    profile = tmp_path / "dataset.json"
    profile.write_text(
        json.dumps({"version": 1, "images_dir": "images", "labels_dir": "labels", "classes_file": "classes.txt"})
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = DatasetConfig.from_profile(profile)
    assert loaded == dataset


def test_profile_roundtrip_preserves_explicit_external_state_path(dataset, tmp_path):
    config = DatasetConfig(
        dataset.images_dir, dataset.labels_dir, dataset.classes_file, tmp_path / "state" / "reviews.json"
    )
    profile = tmp_path / "profiles" / "dataset.json"
    config.save_profile(profile)
    assert DatasetConfig.from_profile(profile) == config


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"version": 9},
        {"version": 1, "images_dir": "images"},
        {"version": 1, "images_dir": "images", "labels_dir": "labels", "classes_file": "classes.txt", "pass_state": 42},
    ],
)
def test_invalid_profile_refuses_to_select_a_partial_dataset(tmp_path, payload):
    profile = tmp_path / "invalid.json"
    profile.write_text(json.dumps(payload))
    with pytest.raises(LabelError):
        DatasetConfig.from_profile(profile)


def test_session_restores_navigation_and_view_without_leaking_between_datasets(dataset, tmp_path):
    session = Session(dataset, ["insulator", "pole"])
    view = {"zoom": 2.5, "center_x": 21.25, "center_y": 42.5}
    session.save(
        image="flight_000002_000200ms.png",
        class_id=1,
        image_mode=True,
        filter="Edited",
        lens=False,
        geometry="1100x720",
        view=view,
    )
    restored = session.load()
    assert restored["image"] == "flight_000002_000200ms.png"
    assert restored["class_id"] == 1
    assert restored["view"] == view
    assert restored["image_mode"] is True and restored["lens"] is False
    before = session.path.read_bytes()
    moved = DatasetConfig(tmp_path / "other-images", dataset.labels_dir, dataset.classes_file, dataset.pass_state)
    assert Session(moved, ["insulator", "pole"]).load() == {}
    assert Session(dataset, ["pole", "insulator"]).load() == {}
    assert session.path.read_bytes() == before


def test_corrupt_session_is_reported_and_preserved(dataset):
    session = Session(dataset, ["insulator", "pole"])
    session.path.parent.mkdir(parents=True)
    session.path.write_text('{"version":')
    before = session.path.read_bytes()
    assert session.load() == {}
    assert "retained" in session.notice
    assert session.path.read_bytes() == before


def test_export_blocks_unreviewed_empty_labels_without_creating_destination(dataset, tmp_path):
    for image in collect_images(dataset.images_dir):
        write_yolo_boxes(dataset.labels_dir / f"{image.stem}.txt", [], 100, 80, 2)
    destination = tmp_path / "export"
    with pytest.raises(LabelError, match="not explicitly confirmed"):
        export_verified(dataset, destination)
    assert not destination.exists()


def test_verified_export_preserves_images_labels_and_confirmation_states(dataset, tmp_path):
    confirmed_dataset(dataset)
    destination = tmp_path / "export"
    result = export_verified(dataset, destination)
    assert result == {"images": 2, "boxes": 1, "classes": 2}
    first, second = collect_images(destination / "images" / "all")
    with Image.open(first) as exported_image:
        boxes = read_yolo_boxes(destination / "labels" / "all" / f"{first.stem}.txt", *exported_image.size, 2)
    assert [(box.cls, box.x1, box.y1, box.x2, box.y2) for box in boxes] == [(0, 10, 10, 30, 30)]
    with Image.open(second) as exported_image:
        assert read_yolo_boxes(destination / "labels" / "all" / f"{second.stem}.txt", *exported_image.size, 2) == []
    manifest = json.loads((destination / "review.json").read_text())
    assert manifest["review"]["flight_000001_000100ms.png"] == {"0": "positive", "1": "negative"}
    assert manifest["review"]["flight_000002_000200ms.png"] == {"0": "negative", "1": "negative"}


def test_external_annotation_change_invalidates_verified_export(dataset, tmp_path):
    confirmed_dataset(dataset)
    first = collect_images(dataset.images_dir)[0]
    write_yolo_boxes(dataset.labels_dir / f"{first.stem}.txt", [Box(0, 20, 20, 50, 50)], 100, 80, 2)
    with pytest.raises(LabelError, match="not explicitly confirmed"):
        export_verified(dataset, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_export_never_overwrites_existing_destination(dataset, tmp_path):
    confirmed_dataset(dataset)
    destination = tmp_path / "export"
    destination.mkdir()
    existing = destination / "user.txt"
    existing.write_text("keep me")
    with pytest.raises(LabelError, match="already exists"):
        export_verified(dataset, destination)
    assert existing.read_text() == "keep me"


def test_references_use_only_confirmed_annotations(dataset):
    first = collect_images(dataset.images_dir)[0]
    write_yolo_boxes(dataset.labels_dir / f"{first.stem}.txt", [Box(0, 10, 10, 30, 30)], 100, 80, 2)
    assert reference_crops(dataset, 0)[0] == []
    confirmed_dataset(dataset)
    crops, errors = reference_crops(dataset, 0)
    assert errors == []
    assert crops[0][0] == first.name
    assert crops[0][1].getpixel((0, 0)) == (0, 0, 128)
