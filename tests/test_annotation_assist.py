"""Consumer coverage for image delivery and real optical-flow assistance.

YOLO inference is deliberately not replaced by echoing mocked model outputs. Real
weights are exercised separately; these tests cover its independent mapping rules.
"""

from __future__ import annotations

import math
import os
import string
import threading
import time
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st
from PIL import Image

import annotation_assist as assist
from annotation_store import Box


def _drain_until(loader, predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        loader.drain()
        if predicate():
            return
        if time.monotonic() >= deadline:
            pytest.fail("ImageLoader did not deliver within the deadline")
        time.sleep(0.005)


def _request(loader, path):
    records = []
    loader.request(path, lambda requested, image, error: records.append((requested, image, error)))
    _drain_until(loader, lambda: bool(records))
    return records[0]


def _save(path, color, size=(20, 15), mode="RGB"):
    with Image.new(mode, size, color) as image:
        image.save(path)
    return path


@pytest.fixture
def make_loader():
    loaders = []

    def create(max_bytes=128 * 1024 * 1024):
        loader = assist.ImageLoader(max_bytes=max_bytes)
        loaders.append(loader)
        return loader

    yield create
    for loader in loaders:
        loader.close()


def test_rgb_decode_retains_original_detail_and_defers_callbacks(tmp_path, make_loader):
    path = _save(tmp_path / "gray.png", 83, size=(39, 27), mode="L")
    loader = make_loader()
    records = []
    owner = threading.get_ident()
    loader.request(
        path,
        lambda requested, image, error: records.append((requested, image, error, threading.get_ident())),
    )
    assert records == []
    _drain_until(loader, lambda: bool(records))
    requested, image, error, callback_thread = records[0]
    assert requested == path
    assert error is None
    assert image.mode == "RGB"
    assert image.size == (39, 27)
    assert image.getpixel((38, 26)) == (83, 83, 83)
    assert callback_thread == owner
    assert loader.cache_bytes == 39 * 27 * 3

    cached = []
    loader.request(path, lambda requested, loaded, error: cached.append((loaded, error)))
    assert cached == []
    loader.drain()
    assert cached == [(image, None)]
    assert cached[0][0] is image
    assert loader.cache_hits == 1


@pytest.mark.parametrize("change", ["mtime", "size"])
def test_cache_invalidates_changed_image_files(tmp_path, make_loader, change):
    path = _save(tmp_path / "frame.bmp", (10, 20, 30))
    loader = make_loader()
    _, old_image, error = _request(loader, path)
    assert error is None
    original_stat = path.stat()
    new_size = (20, 15) if change == "mtime" else (26, 17)
    _save(path, (200, 90, 40), new_size)
    mtime = original_stat.st_mtime_ns + 1_000_000_000 if change == "mtime" else original_stat.st_mtime_ns
    os.utime(path, ns=(original_stat.st_atime_ns, mtime))
    _, new_image, error = _request(loader, path)
    assert error is None
    assert new_image is not old_image
    assert new_image.size == new_size
    assert new_image.getpixel((0, 0)) == (200, 90, 40)
    assert old_image.getpixel((0, 0)) == (10, 20, 30)
    assert loader.cache_bytes == new_size[0] * new_size[1] * 3
    assert loader.cache_hits == 0


def test_cache_hit_rechecks_file_before_delivery(tmp_path, make_loader):
    path = _save(tmp_path / "frame.bmp", (1, 2, 3))
    loader = make_loader()
    _, original, error = _request(loader, path)
    assert error is None
    records = []
    loader.request(path, lambda requested, image, error: records.append((image, error)))
    stat = path.stat()
    _save(path, (11, 22, 33))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    _drain_until(loader, lambda: bool(records))
    image, error = records[0]
    assert error is None
    assert image is not original
    assert image.getpixel((0, 0)) == (11, 22, 33)
    assert loader.cache_hits == 0


def test_lru_is_bounded_and_does_not_close_consumer_images(tmp_path, make_loader):
    paths = [_save(tmp_path / f"{index}.png", (index, 50, 90), (10, 10)) for index in range(3)]
    loader = make_loader(max_bytes=600)
    _, first, _ = _request(loader, paths[0])
    _, second, _ = _request(loader, paths[1])
    assert loader.cache_bytes == 600
    assert _request(loader, paths[0])[1] is first  # Make frame zero most recent.
    _request(loader, paths[2])
    assert loader.cache_bytes == 600
    assert _request(loader, paths[0])[1] is first
    assert loader.cache_hits == 2
    _, reloaded_second, error = _request(loader, paths[1])
    assert error is None
    assert reloaded_second is not second
    assert loader.cache_hits == 2
    assert loader.cache_bytes == 600
    assert second.getpixel((0, 0)) == (1, 50, 90)
    assert first.getpixel((0, 0)) == (0, 50, 90)


def test_oversized_image_is_delivered_but_never_cached(tmp_path, make_loader):
    path = _save(tmp_path / "oversize.png", (20, 40, 60), (30, 20))
    loader = make_loader(max_bytes=50)
    _, first, error = _request(loader, path)
    assert error is None
    assert first.size == (30, 20)
    assert loader.cache_bytes == 0
    _, second, error = _request(loader, path)
    assert error is None
    assert second is not first
    assert loader.cache_bytes == loader.cache_hits == 0


def test_pending_requests_share_actual_worker_decode(tmp_path, make_loader, monkeypatch):
    path = _save(tmp_path / "shared.png", (6, 7, 8))
    loader = make_loader()
    real_open = Image.open
    started = threading.Event()
    release = threading.Event()
    decode_threads = []

    def blocked_open(*args, **kwargs):
        decode_threads.append(threading.get_ident())
        started.set()
        assert release.wait(10), "Test decode gate was not released"
        return real_open(*args, **kwargs)

    monkeypatch.setattr(assist.Image, "open", blocked_open)
    records = []

    def callback(requested, image, error):
        records.append((requested, image, error, threading.get_ident()))

    try:
        loader.request(path, callback)
        assert started.wait(10)
        for _ in range(8):
            loader.request(path, callback)
        loader.prefetch([path, path])
        assert records == []
        release.set()
        _drain_until(loader, lambda: len(records) == 9)
    finally:
        release.set()
    assert len(decode_threads) == 1
    assert decode_threads[0] != threading.get_ident()
    first = records[0][1]
    assert all(record[0] == path and record[1] is first and record[2] is None for record in records)
    assert all(record[3] == threading.get_ident() for record in records)


def test_prefetch_populates_cache_without_callbacks(tmp_path, make_loader):
    paths = [_save(tmp_path / f"prefetch-{index}.png", (10, 20, index)) for index in range(2)]
    loader = make_loader()
    loader.prefetch(paths)
    _drain_until(loader, lambda: loader.cache_bytes == 2 * 20 * 15 * 3)
    for path in paths:
        assert _request(loader, path)[2] is None
    assert loader.cache_hits == 2


def test_undrained_prefetch_does_not_keep_decoding(tmp_path, make_loader, monkeypatch):
    paths = [_save(tmp_path / f"{index}.png", (index, 30, 40)) for index in range(12)]
    loader = make_loader()
    real_open = Image.open
    lock = threading.Lock()
    two_started = threading.Event()
    opens = []

    def tracked_open(*args, **kwargs):
        with lock:
            opens.append(Path(args[0]))
            if len(opens) == 2:
                two_started.set()
        return real_open(*args, **kwargs)

    monkeypatch.setattr(assist.Image, "open", tracked_open)
    loader.prefetch(paths)
    assert two_started.wait(10)
    time.sleep(0.04)
    assert len(opens) == 2
    assert loader.cache_bytes == 0  # Only drain, never a worker, changes the cache.
    _drain_until(loader, lambda: loader.cache_bytes > 0)
    loader.close()


@pytest.mark.parametrize("kind", ["missing", "corrupt"])
def test_decode_errors_are_delivered_not_silently_ignored(tmp_path, make_loader, kind):
    path = tmp_path / f"{kind}.png"
    if kind == "corrupt":
        path.write_bytes(b"not an image")
    loader = make_loader()
    requested, image, error = _request(loader, path)
    assert requested == path
    assert image is None
    assert isinstance(error, Exception)
    assert loader.cache_bytes == 0


def test_requests_from_worker_still_deliver_only_on_owner_thread(tmp_path, make_loader):
    path = _save(tmp_path / "threaded.png", (10, 20, 30))
    loader = make_loader()
    callback_threads = []
    drain_errors = []

    def worker_request():
        loader.request(
            path,
            lambda requested, image, error: callback_threads.append(threading.get_ident()),
        )
        try:
            loader.drain()
        except RuntimeError as exc:
            drain_errors.append(exc)

    worker = threading.Thread(target=worker_request)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert len(drain_errors) == 1
    assert callback_threads == []
    _drain_until(loader, lambda: bool(callback_threads))
    assert callback_threads == [threading.get_ident()]


def test_close_cancels_queued_delivery_and_is_idempotent(tmp_path, make_loader, monkeypatch):
    paths = [_save(tmp_path / f"closing-{index}.png", (10, index, 30)) for index in range(7)]
    loader = make_loader()
    _request(loader, paths[0])
    assert loader.cache_bytes > 0
    real_open = Image.open
    started = threading.Event()
    release = threading.Event()

    def blocked_open(*args, **kwargs):
        started.set()
        assert release.wait(10), "Test decode gate was not released"
        return real_open(*args, **kwargs)

    monkeypatch.setattr(assist.Image, "open", blocked_open)
    records = []
    try:
        for path in paths[1:]:
            loader.request(path, lambda requested, image, error: records.append((image, error)))
        assert started.wait(10)
        loader.close()  # Must return while the actual decode is still blocked.
        loader.close()
        assert loader.cache_bytes == 0
        assert loader.drain() == 0
        assert records == []
        with pytest.raises(RuntimeError, match="closed"):
            loader.request(paths[0], lambda requested, image, error: None)
        with pytest.raises(RuntimeError, match="closed"):
            loader.prefetch(paths)
    finally:
        release.set()
    assert loader.drain() == 0
    assert records == []


def test_callback_exception_does_not_discard_other_callbacks(tmp_path, make_loader):
    path = _save(tmp_path / "callbacks.png", (10, 20, 30))
    loader = make_loader()
    records = []

    def failing_callback(requested, image, error):
        raise ValueError("consumer callback failed")

    loader.request(path, failing_callback)
    loader.request(path, lambda requested, image, error: records.append((image, error)))
    with pytest.raises(ValueError, match="consumer callback failed"):
        _drain_until(loader, lambda: bool(records))
    assert len(records) == 1
    assert records[0][1] is None


@pytest.mark.parametrize("max_bytes", [-1, 1.5, True])
def test_cache_bound_requires_nonnegative_integer(max_bytes):
    with pytest.raises(ValueError, match="nonnegative integer"):
        assist.ImageLoader(max_bytes=max_bytes)


@settings(max_examples=75)
@given(
    st.lists(
        st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=12),
        min_size=1,
        max_size=4,
    )
)
def test_normalized_exact_mapping_does_not_use_model_class_positions(words):
    dataset_name = " ".join(words)
    model_name = " __ ".join(word.upper() for word in words)
    resolver = assist._ClassResolver(["unrelated!", dataset_name], None)
    assert resolver.resolve(model_name) == 1
    normalized = assist._normalize_name(model_name)
    assert assist._normalize_name(normalized) == normalized


def test_mapping_rejects_ambiguous_or_missing_names():
    resolver = assist._ClassResolver(["Power_Line", " power line "], None)
    with pytest.raises(ValueError, match="Ambiguous"):
        resolver.resolve("POWER_LINE")
    with pytest.raises(ValueError, match="Missing"):
        assist._ClassResolver(["person", "dog"], None).resolve("cat")
    explicit = assist._ClassResolver(["person", "dog"], {"kitty": "dog"})
    with pytest.raises(ValueError, match="Missing explicit"):
        explicit.resolve("person")
    ambiguous = assist._ClassResolver(["person", "dog"], {"cat": 0, " CAT ": 1})
    with pytest.raises(ValueError, match="Ambiguous explicit"):
        ambiguous.resolve("cat")


def test_explicit_model_names_can_target_dataset_names_or_ids():
    resolver = assist._ClassResolver(["Human", "domestic cat"], {"Kitten": "Domestic_Cat", "person": 0})
    assert resolver.resolve(" kitten ") == 1
    assert resolver.resolve("PERSON") == 0


@pytest.mark.parametrize("target", [-1, 2, True, 1.5, None, "unlisted"])
def test_explicit_mapping_rejects_invalid_targets(target):
    resolver = assist._ClassResolver(["first", "second"], {"model class": target})
    with pytest.raises(ValueError):
        resolver.resolve("model_class")


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf")])
def test_prediction_rejects_invalid_confidence_before_loading_models(tmp_path, confidence):
    with pytest.raises(ValueError, match="confidence"):
        assist.predict_boxes(tmp_path / "frame.png", tmp_path / "weights.pt", ["object"], confidence)


def test_prediction_requires_a_local_weights_file(tmp_path):
    path = _save(tmp_path / "frame.png", (10, 20, 30))
    with pytest.raises(FileNotFoundError):
        assist.predict_boxes(path, tmp_path / "missing-weights.pt", ["object"])


@pytest.fixture
def vision():
    cv2 = pytest.importorskip("cv2", reason="Install the vision extra for real propagation tests")
    np = pytest.importorskip("numpy")
    return cv2, np


def _texture(vision, seed=2026, size=(320, 240)):
    cv2, np = vision
    width, height = size
    gray = np.random.default_rng(seed).integers(0, 256, size=(height, width), dtype=np.uint8)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _save_gray(path, gray):
    with Image.fromarray(gray) as image:
        image.save(path)
    return path


@pytest.mark.parametrize("destination_size", [(320, 240), (640, 480), (480, 300)])
def test_real_optical_flow_tracks_translation_across_resolutions(tmp_path, vision, destination_size):
    cv2, np = vision
    gray = _texture(vision)
    dx, dy = 6, 4
    translated = cv2.warpAffine(
        gray,
        np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32),
        (320, 240),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    destination = cv2.resize(translated, destination_size, interpolation=cv2.INTER_LINEAR)
    previous_path = _save_gray(tmp_path / "previous.png", gray)
    current_path = _save_gray(tmp_path / "current.png", destination)
    boxes = [Box(0, 50, 55, 140, 150), Box(4, 175, 70, 265, 170)]
    original_coordinates = [(box.x1, box.y1, box.x2, box.y2) for box in boxes]
    candidates = assist.propagate_boxes(previous_path, current_path, boxes)
    assert len(candidates) == 2
    assert len({candidate.source for candidate in candidates}) == 2
    scale_x, scale_y = destination_size[0] / 320, destination_size[1] / 240
    for old_box, candidate in zip(boxes, candidates, strict=True):
        box = candidate.box
        assert box.cls == old_box.cls
        assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx(
            (
                (old_box.x1 + dx) * scale_x,
                (old_box.y1 + dy) * scale_y,
                (old_box.x2 + dx) * scale_x,
                (old_box.y2 + dy) * scale_y,
            ),
            abs=2.0,
        )
        assert candidate.source.startswith("flow:previous.png:box-")
        assert math.isfinite(candidate.confidence)
        assert 0 < candidate.confidence <= 1
        assert 0 <= box.x1 < box.x2 <= destination_size[0]
        assert 0 <= box.y1 < box.y2 <= destination_size[1]
    assert [(box.x1, box.y1, box.x2, box.y2) for box in boxes] == original_coordinates


def test_real_optical_flow_rejects_untextured_frames(tmp_path, vision):
    _, np = vision
    gray = np.full((240, 320), 128, dtype=np.uint8)
    previous = _save_gray(tmp_path / "blank-previous.png", gray)
    current = _save_gray(tmp_path / "blank-current.png", gray)
    assert assist.propagate_boxes(previous, current, [Box(0, 50, 50, 180, 180)]) == []


def test_real_optical_flow_rejects_unrelated_frames(tmp_path, vision):
    previous = _save_gray(tmp_path / "unrelated-previous.png", _texture(vision, seed=881))
    current = _save_gray(tmp_path / "unrelated-current.png", _texture(vision, seed=338))
    assert assist.propagate_boxes(previous, current, [Box(0, 50, 40, 230, 200)]) == []


def test_real_optical_flow_rejects_spatially_unstable_tracks(tmp_path, vision):
    _, np = vision
    gray = np.zeros((240, 320), dtype=np.uint8)
    textured = _texture(vision, seed=144)
    gray[50:190, 114:121] = textured[50:190, 114:121]
    previous = _save_gray(tmp_path / "narrow-previous.png", gray)
    current = _save_gray(tmp_path / "narrow-current.png", gray)
    # Real, stable flow on a narrow stripe is insufficient to transform a broad box.
    assert assist.propagate_boxes(previous, current, [Box(0, 90, 40, 180, 200)]) == []


def test_empty_propagation_does_not_open_frames(tmp_path):
    assert assist.propagate_boxes(tmp_path / "missing-previous.png", tmp_path / "missing-current.png", []) == []
