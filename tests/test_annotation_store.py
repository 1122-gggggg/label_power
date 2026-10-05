from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from hypothesis import example, given, settings, strategies as st
from PIL import Image

import annotation_store
import verify_yolo_labels
from annotation_store import (
    HISTORY_LIMIT,
    Box,
    LabelError,
    ReviewStore,
    atomic_write_json,
    collect_images,
    copy_boxes,
    label_path_for,
    read_classes,
    read_yolo_boxes,
    write_yolo_boxes,
)


def make_images(root: Path, names: tuple[str, ...] = ("frame.png",)) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    images = []
    for name in names:
        path = root / name
        Image.new("RGB", (100, 80), "white").save(path)
        images.append(path)
    return images


def backups(path: Path) -> list[Path]:
    return sorted((path.parent / ".history" / path.name).glob("version-*.bak"))


def label_fixture(tmp_path: Path) -> tuple[list[Path], Path, Path]:
    images = make_images(tmp_path / "images")
    labels_dir = tmp_path / "labels"
    label = label_path_for(images[0], labels_dir)
    write_yolo_boxes(label, [Box(0, 0, 0, 100, 80), Box(1, 10, 10, 20, 20)], 100, 80, 2)
    return images, labels_dir, label


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        ("0 0.5 0.5 0.2", "5 fields"),
        ("0 0.5 0.5 0.2 0.2 extra", "5 fields"),
        ("0 text 0.5 0.2 0.2", "numeric"),
        ("-1 0.5 0.5 0.2 0.2", "class id"),
        ("1.0 0.5 0.5 0.2 0.2", "class id"),
        ("true 0.5 0.5 0.2 0.2", "class id"),
        ("2 0.5 0.5 0.2 0.2", "catalog"),
        ("0 -0.1 0.5 0.2 0.2", r"\[0, 1\]"),
        ("0 1.1 0.5 0.2 0.2", r"\[0, 1\]"),
        ("0 0.5 0.5 1.1 0.2", r"\[0, 1\]"),
        ("0 0.5 0.5 0 0.2", "positive"),
        ("0 0.5 0.5 0.2 0", "positive"),
        ("0 0.5 0.5 -0.2 0.2", r"\[0, 1\]"),
        ("0 0.1 0.5 0.4 0.2", "extents"),
        ("0 0.9 0.5 0.4 0.2", "extents"),
        ("0 0.5 0.1 0.2 0.4", "extents"),
        ("0 0.5 0.9 0.2 0.4", "extents"),
        ("0 0.1 0.5 0.20000000000000001 0.2", "extents"),
        ("0 0.5 0.5 1e-999 0.2", "too small"),
    ],
)
def test_malformed_rows_raise_with_source_without_overwrite(tmp_path, row, reason):
    label = tmp_path / "frame.txt"
    original = f"0 0.5 0.5 1 1\n\n{row}\n".encode()
    label.write_bytes(original)
    with pytest.raises(LabelError, match=reason) as error:
        loaded = read_yolo_boxes(label, 100, 80, 2)
        write_yolo_boxes(label, loaded, 100, 80, 2)
    assert error.value.path == label
    assert error.value.line == 3
    assert f"{label}:3:" in str(error.value)
    assert label.read_bytes() == original
    assert not backups(label)


@pytest.mark.parametrize("token", ["nan", "NaN", "inf", "+inf", "-inf", "1e999"])
@pytest.mark.parametrize("field", range(4))
def test_nonfinite_input_is_never_a_label(tmp_path, token, field):
    values = ["0.5", "0.5", "0.2", "0.2"]
    values[field] = token
    label = tmp_path / "frame.txt"
    label.write_text("0 " + " ".join(values), encoding="utf-8")
    with pytest.raises(LabelError, match="finite"):
        read_yolo_boxes(label, 100, 80, 1)


def test_missing_label_is_not_a_verified_empty_label(tmp_path):
    label = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError):
        read_yolo_boxes(label, 100, 80, 1)
    label.write_text("\n \n", encoding="utf-8")
    assert read_yolo_boxes(label, 100, 80, 1) == []


def test_continuous_full_bounds_roundtrip_and_explicit_clamp(tmp_path):
    box = Box(0, 0, 0, 100, 80)
    assert Box(0, -4, -5, 104, 85).clamp(100, 80) == box
    assert box.normalized(100, 80) == (0.5, 0.5, 1.0, 1.0)
    label = tmp_path / "frame.txt"
    write_yolo_boxes(label, [box], 100, 80, 1)
    assert read_yolo_boxes(label, 100, 80, 1) == [box]
    assert copy_boxes([box]) == [box]
    assert copy_boxes([box])[0] is not box


@pytest.mark.parametrize(
    "box",
    [
        Box(True, 0, 0, 20, 20),
        Box(1.0, 0, 0, 20, 20),
        Box("0", 0, 0, 20, 20),
        Box(-1, 0, 0, 20, 20),
        Box(2, 0, 0, 20, 20),
        Box(0, math.nan, 0, 20, 20),
        Box(0, 0, math.inf, 20, 20),
        Box(0, 0, 0, math.inf, 20),
        Box(0, 0, 0, 20, -math.inf),
        Box(0, -1, 0, 20, 20),
        Box(0, 0, -1, 20, 20),
        Box(0, 0, 0, 101, 20),
        Box(0, 0, 0, 20, 81),
        Box(0, 10, 0, 10, 20),
        Box(0, 0, 10, 20, 10),
        Box(0, 20, 0, 10, 20),
        Box(0, 0, 20, 20, 10),
        Box(0, False, 0, 20, 20),
    ],
)
def test_writer_validates_entire_stream_before_touching_disk(tmp_path, box):
    label = tmp_path / "frame.txt"
    original = b"0 0.5 0.5 1 1\n"
    label.write_bytes(original)
    with pytest.raises(LabelError):
        write_yolo_boxes(label, iter([Box(0, 0, 0, 20, 20), box]), 100, 80, 2)
    assert label.read_bytes() == original
    assert not backups(label)
    absent = tmp_path / "uncreated" / "frame.txt"
    with pytest.raises(LabelError):
        write_yolo_boxes(absent, [box], 100, 80, 2)
    assert not absent.parent.exists()


@pytest.mark.parametrize("dimensions", [(0, 80), (100, -1), (math.nan, 80), (100, math.inf), (True, 80)])
def test_invalid_dimensions_are_rejected_before_io(tmp_path, dimensions):
    label = tmp_path / "frame.txt"
    with pytest.raises(LabelError, match="width and height"):
        write_yolo_boxes(label, [], *dimensions, 1)
    with pytest.raises(LabelError, match="width and height"):
        read_yolo_boxes(label, *dimensions, 1)
    assert not label.exists()


@pytest.mark.parametrize("class_count", [True, 0, -1, 1.5, "1"])
def test_class_count_is_strict(tmp_path, class_count):
    with pytest.raises(LabelError, match="class_count"):
        write_yolo_boxes(tmp_path / "frame.txt", [], 100, 80, class_count)


@st.composite
def valid_box_sets(draw):
    width = draw(st.integers(min_value=1, max_value=4096))
    height = draw(st.integers(min_value=1, max_value=4096))
    count = draw(st.integers(min_value=0, max_value=12))
    boxes = []
    for _ in range(count):
        x1 = draw(st.integers(min_value=0, max_value=width * 16 - 1))
        x2 = draw(st.integers(min_value=x1 + 1, max_value=width * 16))
        y1 = draw(st.integers(min_value=0, max_value=height * 16 - 1))
        y2 = draw(st.integers(min_value=y1 + 1, max_value=height * 16))
        cls = draw(st.integers(min_value=0, max_value=2))
        boxes.append(Box(cls, x1 / 16, y1 / 16, x2 / 16, y2 / 16))
    return width, height, boxes


@given(valid_box_sets())
@example((7, 3, [Box(0, 0, 0, 1, 1)]))
@example((100, 80, [Box(2, 0, 0, 100, 80)]))
@example((1, 1, []))
@settings(max_examples=100, deadline=None)
def test_valid_geometry_roundtrips_without_clipping(case):
    """All valid continuous edges survive writing and strict parsing, including borders."""
    width, height, boxes = case
    with tempfile.TemporaryDirectory() as directory:
        label = Path(directory) / "frame.txt"
        write_yolo_boxes(label, boxes, width, height, 3)
        restored = read_yolo_boxes(label, width, height, 3)
    assert len(restored) == len(boxes)
    for expected, actual in zip(boxes, restored):
        assert actual.cls == expected.cls
        assert (actual.x1, actual.y1, actual.x2, actual.y2) == pytest.approx(
            (expected.x1, expected.y1, expected.x2, expected.y2), rel=1e-12, abs=1e-10
        )
        assert 0 <= actual.x1 < actual.x2 <= width
        assert 0 <= actual.y1 < actual.y2 <= height


@pytest.mark.parametrize("kind", ["labels", "json"])
def test_atomic_replace_failure_preserves_original_and_backup(tmp_path, monkeypatch, kind):
    path = tmp_path / ("frame.txt" if kind == "labels" else "review.json")
    original = b"prior bytes\n"
    path.write_bytes(original)

    def fail_replace(source, destination):
        assert Path(source).parent == path.parent
        assert Path(destination) == path
        raise OSError("replacement denied")

    monkeypatch.setattr(annotation_store.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement denied"):
        if kind == "labels":
            write_yolo_boxes(path, [], 100, 80, 2)
        else:
            atomic_write_json(path, {"version": 2})
    assert path.read_bytes() == original
    assert any(backup.read_bytes() == original for backup in backups(path))
    assert not list(tmp_path.glob(".*.tmp"))


def test_same_directory_temporaries_are_unique(tmp_path, monkeypatch):
    path = tmp_path / "frame.txt"
    observed = []
    real_replace = annotation_store.os.replace

    def observe_replace(source, destination):
        observed.append(Path(source))
        real_replace(source, destination)

    monkeypatch.setattr(annotation_store.os, "replace", observe_replace)
    write_yolo_boxes(path, [], 100, 80, 1)
    write_yolo_boxes(path, [Box(0, 0, 0, 100, 80)], 100, 80, 1)
    assert len(set(observed)) == 2
    assert all(temporary.parent == tmp_path for temporary in observed)
    assert all(not temporary.exists() for temporary in observed)


@pytest.mark.parametrize("kind", ["labels", "json"])
def test_version_backups_are_bounded_and_preserve_previous_contents(tmp_path, kind):
    path = tmp_path / ("frame.txt" if kind == "labels" else "review.json")
    versions = []
    for index in range(HISTORY_LIMIT + 5):
        if kind == "labels":
            write_yolo_boxes(path, [Box(0, index, 0, index + 1, 1)], 100, 80, 1)
        else:
            atomic_write_json(path, {"revision": index})
        versions.append(path.read_bytes())
    assert len(backups(path)) == HISTORY_LIMIT
    assert {p.read_bytes() for p in backups(path)} == set(versions[-HISTORY_LIMIT - 1 : -1])
    assert path.read_bytes() == versions[-1]


@pytest.mark.parametrize("payload", [{"confidence": math.nan}, {"value": math.inf}, {"invalid": object()}])
def test_invalid_json_never_overwrites_or_creates_backup(tmp_path, payload):
    path = tmp_path / "review.json"
    original = b'{"previous": true}\n'
    path.write_bytes(original)
    with pytest.raises(LabelError, match="serialize"):
        atomic_write_json(path, payload)
    assert path.read_bytes() == original
    assert not backups(path)


def test_catalog_preserves_order_and_rejects_duplicates_and_empty(tmp_path):
    catalog = tmp_path / "classes.txt"
    catalog.write_text(" pole \n\nwire\n", encoding="utf-8")
    assert read_classes(catalog) == ["pole", "wire"]
    catalog.write_text("pole\nwire\npole\n", encoding="utf-8")
    with pytest.raises(LabelError, match="duplicate") as error:
        read_classes(catalog)
    assert error.value.line == 3
    catalog.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(LabelError, match="empty"):
        read_classes(catalog)


def test_collect_images_uses_extensions_files_and_rejects_stem_collisions(tmp_path):
    images = make_images(tmp_path, ("a.PNG", "b.jpeg", "c.tiff", "d.webp"))
    (tmp_path / "not_an_image.txt").write_text("text", encoding="utf-8")
    (tmp_path / "directory.jpg").mkdir()
    assert collect_images(tmp_path) == sorted(images)
    Image.new("RGB", (100, 80)).save(tmp_path / "a.jpg")
    with pytest.raises(LabelError, match="collision"):
        collect_images(tmp_path)
    with pytest.raises(LabelError, match="collision"):
        ReviewStore(tmp_path / "review.json", [images[0], tmp_path / "a.jpg"], ["pole"])


def test_only_explicit_confirmation_is_reviewed_and_edit_invalidates_it(tmp_path):
    images = make_images(tmp_path / "images")
    path = tmp_path / "review.json"
    review = ReviewStore(path, images, ["pole", "wire"])
    assert review.reviewed == {}
    assert review.status("frame.png", 0) == "pending"
    review.mark_edited("frame.png", [0])
    assert review.status("frame.png", 0) == "edited"
    assert review.reviewed == {}
    review.confirm("frame.png", 0, True)
    review.confirm("frame.png", 1, False)
    assert review.status("frame.png", 0) == "positive"
    assert review.status("frame.png", 1) == "negative"
    assert review.reviewed == {"frame.png": {0, 1}}
    snapshot = review.reviewed
    snapshot["frame.png"].clear()
    assert review.reviewed == {"frame.png": {0, 1}}
    review.mark_edited("frame.png", [0])
    reopened = ReviewStore(path, images, ["pole", "wire"])
    assert reopened.status("frame.png", 0) == "edited"
    assert reopened.reviewed == {"frame.png": {1}}


@pytest.mark.parametrize("cls", [True, False, -1, 2, 0.0, "0", None])
def test_review_class_ids_are_strict_and_invalid_edits_are_atomic(tmp_path, cls):
    images = make_images(tmp_path / "images")
    review = ReviewStore(tmp_path / "review.json", images, ["pole", "wire"])
    previous = review.path.read_bytes()
    with pytest.raises(LabelError):
        review.status("frame.png", cls)
    with pytest.raises(LabelError):
        review.confirm("frame.png", cls, False)
    with pytest.raises(LabelError):
        review.mark_edited("frame.png", [0, cls])
    assert review.status("frame.png", 0) == "pending"
    assert review.path.read_bytes() == previous


def test_review_rejects_unknown_images_and_nonbool_confirmation(tmp_path):
    images = make_images(tmp_path / "images")
    review = ReviewStore(tmp_path / "review.json", images, ["pole"])
    for name in ("absent.png", "../frame.png", True):
        with pytest.raises(LabelError, match="unknown"):
            review.status(name, 0)
        with pytest.raises(LabelError, match="unknown"):
            review.mark_edited(name, [0])
        with pytest.raises(LabelError, match="unknown"):
            review.confirm(name, 0, False)
    for value in (0, 1, "false", None):
        with pytest.raises(LabelError, match="has_boxes"):
            review.confirm("frame.png", 0, value)
    assert review.reviewed == {}


def test_legacy_auto_review_migrates_to_edited_with_original_backup(tmp_path):
    images = make_images(tmp_path / "images")
    path = tmp_path / "review.json"
    original = b'{"version":1,"reviewed":{"frame.png":[0,1],"stale.jpg":[0]}}\n'
    path.write_bytes(original)
    review = ReviewStore(path, images, ["pole", "wire"])
    assert review.status("frame.png", 0) == review.status("frame.png", 1) == "edited"
    assert review.reviewed == {}
    assert any("untrusted" in notice for notice in review.notices)
    assert any("stale.jpg" in notice for notice in review.notices)
    assert original in [backup.read_bytes() for backup in backups(path)]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["dataset"]["image_root"] == str(images[0].parent.resolve())
    assert payload["dataset"]["class_names"] == ["pole", "wire"]
    record = payload["dataset"]["manifest"][0]
    assert record == {"name": "frame.png", "size": images[0].stat().st_size, "mtime_ns": images[0].stat().st_mtime_ns}


@pytest.mark.parametrize(
    "contents",
    [
        b"{broken",
        b"[]",
        b"null",
        b'{"version":true,"reviewed":{}}',
        b'{"version":1,"reviewed":{"frame.png":[true]}}',
        b'{"version":1,"reviewed":{"frame.png":[-1]}}',
        b'{"version":1,"reviewed":{"frame.png":[2]}}',
        b'{"version":1,"reviewed":{"frame.png":["0"]}}',
        b'{"version":1,"reviewed":{"frame.png":0}}',
        b'{"version":1,"reviewed":{},"invalid":NaN}',
        b'{"version":1,"reviewed":{},"reviewed":{"frame.png":[0]}}',
        b"\xff",
    ],
)
def test_malformed_progress_never_resets_or_overwrites_state(tmp_path, contents):
    images = make_images(tmp_path / "images")
    path = tmp_path / "review.json"
    path.write_bytes(contents)
    with pytest.raises(LabelError) as error:
        ReviewStore(path, images, ["pole", "wire"])
    assert str(path) in str(error.value)
    assert path.read_bytes() == contents
    assert not backups(path)


@pytest.mark.parametrize("status", [True, False, 1, None, [], "reviewed", "Positive"])
def test_v2_review_status_types_are_strict(tmp_path, status):
    images = make_images(tmp_path / "images")
    path = tmp_path / "review.json"
    ReviewStore(path, images, ["pole"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["statuses"] = {"frame.png": {"0": status}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(LabelError, match="status"):
        ReviewStore(path, images, ["pole"])
    assert path.read_bytes() == original


@pytest.mark.parametrize("token", ["-1", "1", "1.0", "true", "00"])
def test_v2_review_class_keys_are_strict(tmp_path, token):
    images = make_images(tmp_path / "images")
    path = tmp_path / "review.json"
    ReviewStore(path, images, ["pole"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["statuses"] = {"frame.png": {token: "positive"}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LabelError, match="class id"):
        ReviewStore(path, images, ["pole"])


def test_v2_unknown_images_and_malformed_manifest_are_not_silently_dropped(tmp_path):
    images = make_images(tmp_path / "images")
    path = tmp_path / "review.json"
    ReviewStore(path, images, ["pole"])
    baseline = json.loads(path.read_text(encoding="utf-8"))
    payload = {**baseline, "statuses": {"absent.png": {"0": "negative"}}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LabelError, match="absent from its manifest"):
        ReviewStore(path, images, ["pole"])
    baseline["dataset"]["manifest"][0]["size"] = True
    path.write_text(json.dumps(baseline), encoding="utf-8")
    with pytest.raises(LabelError, match="size"):
        ReviewStore(path, images, ["pole"])


def test_changed_image_resets_only_its_statuses_and_preserves_backup(tmp_path):
    images = make_images(tmp_path / "images", ("a.png", "b.png"))
    path = tmp_path / "review.json"
    review = ReviewStore(path, images, ["pole"])
    for image in images:
        review.confirm(image.name, 0, False)
    original = path.read_bytes()
    info = images[0].stat()
    os.utime(images[0], ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
    reopened = ReviewStore(path, images, ["pole"])
    assert reopened.status("a.png", 0) == "pending"
    assert reopened.status("b.png", 0) == "negative"
    assert any("a.png" in notice and "changed" in notice for notice in reopened.notices)
    assert original in [backup.read_bytes() for backup in backups(path)]


def test_stale_images_are_removed_new_images_pending_and_old_state_backed_up(tmp_path):
    images = make_images(tmp_path / "images", ("a.png", "b.png"))
    path = tmp_path / "review.json"
    review = ReviewStore(path, images, ["pole"])
    review.confirm("a.png", 0, True)
    review.confirm("b.png", 0, False)
    original = path.read_bytes()
    images[0].unlink()
    make_images(tmp_path / "images", ("c.png",))
    reopened = ReviewStore(path, collect_images(tmp_path / "images"), ["pole"])
    assert reopened.reviewed == {"b.png": {0}}
    assert reopened.status("c.png", 0) == "pending"
    assert any("Stale" in notice and "a.png" in notice for notice in reopened.notices)
    assert any("New" in notice and "c.png" in notice for notice in reopened.notices)
    assert original in [backup.read_bytes() for backup in backups(path)]


def test_class_order_change_invalidates_all_confirmations(tmp_path):
    images = make_images(tmp_path / "images")
    path = tmp_path / "review.json"
    review = ReviewStore(path, images, ["pole", "wire"])
    review.confirm("frame.png", 0, True)
    review.confirm("frame.png", 1, False)
    original = path.read_bytes()
    reopened = ReviewStore(path, images, ["wire", "pole"])
    assert reopened.reviewed == {}
    assert reopened.status("frame.png", 0) == reopened.status("frame.png", 1) == "pending"
    assert any("Class catalog/order changed" in notice for notice in reopened.notices)
    assert original in [backup.read_bytes() for backup in backups(path)]


def test_identical_filenames_in_another_image_root_never_reuse_completion(tmp_path):
    images = make_images(tmp_path / "first")
    other = tmp_path / "second" / images[0].name
    other.parent.mkdir()
    shutil.copy2(images[0], other)
    path = tmp_path / "review.json"
    review = ReviewStore(path, images, ["pole"])
    review.confirm("frame.png", 0, False)
    original = path.read_bytes()
    reopened = ReviewStore(path, [other], ["pole"])
    assert reopened.reviewed == {}
    assert reopened.status("frame.png", 0) == "pending"
    assert any("unrelated dataset" in notice for notice in reopened.notices)
    assert original in [backup.read_bytes() for backup in backups(path)]


def test_read_only_migration_and_missing_progress_have_no_disk_side_effects(tmp_path):
    images = make_images(tmp_path / "images")
    path = tmp_path / "review.json"
    original = b'{"version":1,"reviewed":{"frame.png":[0]}}'
    path.write_bytes(original)
    review = ReviewStore(path, images, ["pole"], read_only=True)
    assert review.status("frame.png", 0) == "edited"
    assert review.reviewed == {}
    assert path.read_bytes() == original
    assert not backups(path)
    with pytest.raises(LabelError, match="read-only"):
        review.confirm("frame.png", 0, False)
    absent = tmp_path / "uncreated" / "review.json"
    pending = ReviewStore(absent, images, ["pole"], read_only=True)
    assert pending.status("frame.png", 0) == "pending"
    assert not absent.parent.exists()


def test_conflicting_progress_write_refuses_to_lose_other_session_work(tmp_path):
    images = make_images(tmp_path / "images")
    path = tmp_path / "review.json"
    first = ReviewStore(path, images, ["pole", "wire"])
    second = ReviewStore(path, images, ["pole", "wire"])
    first.confirm("frame.png", 0, False)
    saved = path.read_bytes()
    with pytest.raises(LabelError, match="changed on disk"):
        second.confirm("frame.png", 1, False)
    assert path.read_bytes() == saved
    assert second.status("frame.png", 1) == "pending"
    assert ReviewStore(path, images, ["pole", "wire"]).reviewed == {"frame.png": {0}}


def test_failed_review_save_does_not_publish_confirmation_in_memory(tmp_path, monkeypatch):
    images = make_images(tmp_path / "images")
    path = tmp_path / "review.json"
    review = ReviewStore(path, images, ["pole"])
    original = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("disk failure")

    monkeypatch.setattr(annotation_store.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        review.confirm("frame.png", 0, False)
    assert review.reviewed == {}
    assert review.status("frame.png", 0) == "pending"
    assert path.read_bytes() == original


def test_confirmation_requires_saved_label_and_matches_class_presence(tmp_path):
    images = make_images(tmp_path / "images")
    labels_dir = tmp_path / "labels"
    path = tmp_path / "review.json"
    review = ReviewStore(path, images, ["pole"], labels_dir=labels_dir)
    with pytest.raises(LabelError, match="save a valid label"):
        review.confirm("frame.png", 0, False)
    label = label_path_for(images[0], labels_dir)
    write_yolo_boxes(label, [], 100, 80, 1)
    with pytest.raises(LabelError, match="does not match saved boxes"):
        review.confirm("frame.png", 0, True)
    review.confirm("frame.png", 0, False)
    assert review.status("frame.png", 0) == "negative"
    assert ReviewStore(path, images, ["pole"], labels_dir=labels_dir).reviewed == {"frame.png": {0}}


def test_per_class_signatures_preserve_unaffected_confirmation(tmp_path):
    images, labels_dir, label = label_fixture(tmp_path)
    path = tmp_path / "review.json"
    review = ReviewStore(path, images, ["pole", "wire"], labels_dir=labels_dir)
    review.confirm("frame.png", 0, True)
    review.confirm("frame.png", 1, True)
    original = path.read_bytes()
    write_yolo_boxes(label, [Box(0, 0, 0, 100, 80), Box(1, 10, 10, 30, 20)], 100, 80, 2)
    reopened = ReviewStore(path, images, ["pole", "wire"], labels_dir=labels_dir)
    assert reopened.status("frame.png", 0) == "positive"
    assert reopened.status("frame.png", 1) == "edited"
    assert reopened.reviewed == {"frame.png": {0}}
    assert any("class 1" in notice and "invalidated" in notice for notice in reopened.notices)
    assert original in [backup.read_bytes() for backup in backups(path)]


def test_editing_other_class_and_row_reordering_do_not_invalidate_unchanged_class(tmp_path):
    images, labels_dir, label = label_fixture(tmp_path)
    path = tmp_path / "review.json"
    review = ReviewStore(path, images, ["pole", "wire"], labels_dir=labels_dir)
    review.confirm("frame.png", 0, True)
    review.confirm("frame.png", 1, True)
    review.mark_edited("frame.png", [1])
    write_yolo_boxes(label, [Box(1, 10, 10, 30, 20), Box(0, 0, 0, 100, 80)], 100, 80, 2)
    reopened = ReviewStore(path, images, ["pole", "wire"], labels_dir=labels_dir)
    assert reopened.reviewed == {"frame.png": {0}}
    assert reopened.status("frame.png", 1) == "edited"
    reopened.confirm("frame.png", 1, True)
    lines = label.read_text(encoding="utf-8").splitlines()
    label.write_text("\n" + "\n".join("  " + row + "  " for row in reversed(lines)) + "\n", encoding="utf-8")
    assert ReviewStore(path, images, ["pole", "wire"], labels_dir=labels_dir).reviewed == {"frame.png": {0, 1}}


@pytest.mark.parametrize("external_change", ["missing", "malformed"])
def test_missing_or_invalid_confirmed_labels_invalidate_without_mutating_on_read_only(tmp_path, external_change):
    images, labels_dir, label = label_fixture(tmp_path)
    path = tmp_path / "review.json"
    review = ReviewStore(path, images, ["pole", "wire"], labels_dir=labels_dir)
    review.confirm("frame.png", 0, True)
    review.confirm("frame.png", 1, True)
    original = path.read_bytes()
    history_before = backups(path)
    if external_change == "missing":
        label.unlink()
    else:
        label.write_text("0 nan 0.5 1 1\n", encoding="utf-8")
    reopened = ReviewStore(path, images, ["pole", "wire"], labels_dir=labels_dir, read_only=True)
    assert reopened.reviewed == {}
    assert reopened.status("frame.png", 0) == reopened.status("frame.png", 1) == "edited"
    assert any("missing or invalid" in notice for notice in reopened.notices)
    assert path.read_bytes() == original
    assert backups(path) == history_before


def test_negative_confirmation_is_invalidated_when_boxes_appear(tmp_path):
    images = make_images(tmp_path / "images")
    labels_dir = tmp_path / "labels"
    label = label_path_for(images[0], labels_dir)
    write_yolo_boxes(label, [], 100, 80, 1)
    path = tmp_path / "review.json"
    review = ReviewStore(path, images, ["pole"], labels_dir=labels_dir)
    review.confirm("frame.png", 0, False)
    write_yolo_boxes(label, [Box(0, 0, 0, 100, 80)], 100, 80, 1)
    reopened = ReviewStore(path, images, ["pole"], labels_dir=labels_dir)
    assert reopened.status("frame.png", 0) == "edited"
    assert reopened.reviewed == {}


def test_distinct_labels_root_never_reuses_review_completion(tmp_path):
    images, labels_dir, label = label_fixture(tmp_path)
    path = tmp_path / "review.json"
    review = ReviewStore(path, images, ["pole", "wire"], labels_dir=labels_dir)
    review.confirm("frame.png", 0, True)
    another = tmp_path / "other-labels"
    another.mkdir()
    shutil.copy2(label, another / label.name)
    reopened = ReviewStore(path, images, ["pole", "wire"], labels_dir=another)
    assert reopened.status("frame.png", 0) == "pending"
    assert reopened.reviewed == {}
    assert any("Labels root changed" in notice for notice in reopened.notices)


def cli_args(images_dir: Path, labels_dir: Path, classes_file: Path) -> list[str]:
    return ["--images-dir", str(images_dir), "--labels-dir", str(labels_dir), "--classes-file", str(classes_file)]


def test_verifier_supports_all_image_extensions_and_catalog_count(tmp_path, capsys):
    images = make_images(tmp_path / "images", ("a.PNG", "b.jpeg", "c.tiff"))
    labels_dir = tmp_path / "labels"
    for image in images:
        write_yolo_boxes(label_path_for(image, labels_dir), [Box(0, 0, 0, 100, 80)], 100, 80, 1)
    classes = tmp_path / "classes.txt"
    classes.write_text("pole\n", encoding="utf-8")
    verify_yolo_labels.main(cli_args(images[0].parent, labels_dir, classes))
    output = capsys.readouterr().out
    assert "images=3\n" in output
    assert "missing_labels=0\n" in output
    assert "bad_labels=0\n" in output
    assert "class_counts={0: 3}\n" in output


def test_verifier_distinguishes_missing_empty_and_malformed(tmp_path, capsys):
    images = make_images(tmp_path / "images", ("a.png", "b.png", "c.png"))
    labels_dir = tmp_path / "labels"
    write_yolo_boxes(label_path_for(images[0], labels_dir), [], 100, 80, 1)
    label_path_for(images[2], labels_dir).write_text("0 inf 0.5 0.2 0.2\n", encoding="utf-8")
    classes = tmp_path / "classes.txt"
    classes.write_text("pole\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        verify_yolo_labels.main(cli_args(images[0].parent, labels_dir, classes))
    assert error.value.code == 1
    output = capsys.readouterr().out
    assert "images=3\n" in output
    assert "missing_labels=1\n" in output
    assert "empty_labels=1\n" in output
    assert "bad_labels=1\n" in output
    assert "finite" in output
    assert "class_counts={0: 0}\n" in output


def test_verifier_review_requirement_never_migrates_on_disk(tmp_path, capsys):
    images = make_images(tmp_path / "images")
    labels_dir = tmp_path / "labels"
    write_yolo_boxes(label_path_for(images[0], labels_dir), [], 100, 80, 1)
    classes = tmp_path / "classes.txt"
    classes.write_text("pole\n", encoding="utf-8")
    path = tmp_path / "legacy.json"
    original = b'{"version":1,"reviewed":{"frame.png":[0]}}'
    path.write_bytes(original)
    args = cli_args(images[0].parent, labels_dir, classes) + ["--require-reviewed", "--pass-state", str(path)]
    with pytest.raises(SystemExit) as error:
        verify_yolo_labels.main(args)
    assert error.value.code == 1
    assert "unreviewed=1\n" in capsys.readouterr().out
    assert path.read_bytes() == original
    assert not backups(path)


def test_verifier_accepts_explicit_positive_and_negative_without_writing(tmp_path, capsys):
    images = make_images(tmp_path / "images")
    labels_dir = tmp_path / "labels"
    write_yolo_boxes(label_path_for(images[0], labels_dir), [Box(0, 0, 0, 100, 80)], 100, 80, 2)
    classes = tmp_path / "classes.txt"
    classes.write_text("pole\nwire\n", encoding="utf-8")
    path = tmp_path / "review.json"
    review = ReviewStore(path, images, ["pole", "wire"], labels_dir=labels_dir)
    review.confirm("frame.png", 0, True)
    review.confirm("frame.png", 1, False)
    original = path.read_bytes()
    history_before = backups(path)
    args = cli_args(images[0].parent, labels_dir, classes) + ["--require-reviewed", "--pass-state", str(path)]
    verify_yolo_labels.main(args)
    assert "unreviewed=0\n" in capsys.readouterr().out
    assert path.read_bytes() == original
    assert backups(path) == history_before


def test_verifier_malformed_progress_is_actionable_and_preserved(tmp_path, capsys):
    images = make_images(tmp_path / "images")
    labels_dir = tmp_path / "labels"
    write_yolo_boxes(label_path_for(images[0], labels_dir), [], 100, 80, 1)
    classes = tmp_path / "classes.txt"
    classes.write_text("pole\n", encoding="utf-8")
    path = tmp_path / "review.json"
    original = b"{broken"
    path.write_bytes(original)
    with pytest.raises(SystemExit) as error:
        verify_yolo_labels.main(
            cli_args(images[0].parent, labels_dir, classes) + ["--require-reviewed", "--pass-state", str(path)]
        )
    assert error.value.code == 2
    assert "restore a .history backup" in capsys.readouterr().err
    assert path.read_bytes() == original
    assert not backups(path)


def test_expected_signature_refuses_lost_label_update_without_new_backup(tmp_path):
    label = tmp_path / "frame.txt"
    write_yolo_boxes(label, [Box(0, 0, 0, 100, 80)], 100, 80, 1)
    loaded_signature = hashlib.sha256(label.read_bytes()).hexdigest()
    write_yolo_boxes(label, [Box(0, 10, 10, 30, 30)], 100, 80, 1)
    other_session_bytes = label.read_bytes()
    history_before = backups(label)
    with pytest.raises(LabelError, match="changed on disk"):
        write_yolo_boxes(label, [], 100, 80, 1, expected_signature=loaded_signature)
    assert label.read_bytes() == other_session_bytes
    assert backups(label) == history_before


def test_expected_missing_distinguishes_absence_from_an_empty_file(tmp_path):
    label = tmp_path / "frame.txt"
    write_yolo_boxes(label, [], 100, 80, 1, expected_signature=None)
    assert label.read_bytes() == b""
    with pytest.raises(LabelError, match="changed on disk"):
        write_yolo_boxes(label, [Box(0, 0, 0, 100, 80)], 100, 80, 1, expected_signature=None)
    assert label.read_bytes() == b""
    assert not backups(label)


def test_matching_expected_signature_allows_atomic_save_with_prior_backup(tmp_path):
    label = tmp_path / "frame.txt"
    write_yolo_boxes(label, [Box(0, 0, 0, 100, 80)], 100, 80, 1)
    previous = label.read_bytes()
    signature = hashlib.sha256(previous).hexdigest()
    write_yolo_boxes(label, [], 100, 80, 1, expected_signature=signature)
    assert label.read_bytes() == b""
    assert previous in [backup.read_bytes() for backup in backups(label)]


@pytest.mark.parametrize("signature", [True, 1, "not a digest", b"0" * 64])
def test_invalid_expected_signature_is_rejected_before_io(tmp_path, signature):
    label = tmp_path / "uncreated" / "frame.txt"
    with pytest.raises(LabelError, match="expected_signature"):
        write_yolo_boxes(label, [], 100, 80, 1, expected_signature=signature)
    assert not label.parent.exists()


def test_progress_lock_closes_compare_and_replace_race(tmp_path, monkeypatch):
    images = make_images(tmp_path / "images")
    path = tmp_path / "review.json"
    first = ReviewStore(path, images, ["pole", "wire"])
    second = ReviewStore(path, images, ["pole", "wire"])
    replacing = Event()
    release = Event()
    real_replace = annotation_store.os.replace

    def paused_replace(source, destination):
        if Path(destination) == path:
            replacing.set()
            assert release.wait(5)
        real_replace(source, destination)

    monkeypatch.setattr(annotation_store.os, "replace", paused_replace)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(first.confirm, "frame.png", 0, False)
        try:
            assert replacing.wait(5)
            with pytest.raises(LabelError, match="another session"):
                second.confirm("frame.png", 1, False)
        finally:
            release.set()
        pending.result(timeout=5)
    assert second.status("frame.png", 1) == "pending"
    assert ReviewStore(path, images, ["pole", "wire"]).reviewed == {"frame.png": {0}}


def test_label_lock_closes_compare_and_replace_race(tmp_path, monkeypatch):
    label = tmp_path / "frame.txt"
    write_yolo_boxes(label, [Box(0, 0, 0, 100, 80)], 100, 80, 1)
    loaded_signature = hashlib.sha256(label.read_bytes()).hexdigest()
    replacing = Event()
    release = Event()
    real_replace = annotation_store.os.replace

    def paused_replace(source, destination):
        if Path(destination) == label:
            replacing.set()
            assert release.wait(5)
        real_replace(source, destination)

    monkeypatch.setattr(annotation_store.os, "replace", paused_replace)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            write_yolo_boxes,
            label,
            [Box(0, 0, 0, 40, 40)],
            100,
            80,
            1,
            expected_signature=loaded_signature,
        )
        try:
            assert replacing.wait(5)
            with pytest.raises(LabelError, match="another session"):
                write_yolo_boxes(label, [], 100, 80, 1, expected_signature=loaded_signature)
        finally:
            release.set()
        pending.result(timeout=5)
    first_session_bytes = label.read_bytes()
    with pytest.raises(LabelError, match="changed on disk"):
        write_yolo_boxes(label, [], 100, 80, 1, expected_signature=loaded_signature)
    assert label.read_bytes() == first_session_bytes
    assert read_yolo_boxes(label, 100, 80, 1) == [Box(0, 0, 0, 40, 40)]
