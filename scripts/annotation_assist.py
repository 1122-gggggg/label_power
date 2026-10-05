"""Read-only, optional annotation assistance and asynchronous RGB image loading.

ImageLoader is constructed and drained on the UI thread. Its RGB images are shared:
consumers must not mutate or close them. Crop/resize creates a separate display image;
this module never creates a Tk object and never writes annotation labels.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from stat import S_ISREG
from typing import Any

from PIL import Image

from annotation_store import Box


@dataclass(frozen=True)
class Candidate:
    box: Box
    source: str
    confidence: float | None = None


@dataclass(frozen=True)
class _FileKey:
    path: Path
    size: int
    mtime_ns: int


def _file_key(path: Path) -> _FileKey:
    resolved = path.expanduser().resolve(strict=True)
    stat = resolved.stat()
    if not S_ISREG(stat.st_mode):
        raise OSError(f"Not a regular file: {resolved}")
    return _FileKey(resolved, stat.st_size, stat.st_mtime_ns)


ImageCallback = Callable[[Path, Image.Image | None, Exception | None], None]


@dataclass
class _DecodeJob:
    callbacks: list[tuple[Path, ImageCallback]] = field(default_factory=list)
    future: Future[Image.Image] | None = None


class ImageLoader:
    """Decode off-thread; dispatch callbacks exclusively from the creating thread.

    Pixel cache accounting is width * height * 3, not compressed file size. The cap
    excludes images retained by consumers and the two active decode results. Workers
    do not start another decode until drain consumes a result, bounding undrained RGB
    buffers as well as concurrent work. Prefetch retains at most four files;
    demand requests take priority and duplicate file versions share one decode.
    """

    _WORKERS = 2
    _PREFETCH_LIMIT = 4

    def __init__(self, max_bytes: int = 128 * 1024 * 1024) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        self.max_bytes = max_bytes
        self._owner = threading.get_ident()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=self._WORKERS, thread_name_prefix="annotation-image")
        self._cache: OrderedDict[_FileKey, Image.Image] = OrderedDict()
        self._cache_bytes = 0
        self._cache_hits = 0
        self._waiting: OrderedDict[_FileKey, _DecodeJob] = OrderedDict()
        self._active: dict[_FileKey, _DecodeJob] = {}
        self._completed: deque[_FileKey] = deque()
        self._hits: deque[tuple[Path, _FileKey, ImageCallback]] = deque()
        self._errors: deque[tuple[Path, ImageCallback, Exception]] = deque()
        self._closed = False

    @property
    def cache_bytes(self) -> int:
        with self._lock:
            return self._cache_bytes

    @property
    def cache_hits(self) -> int:
        """Number of callbacks actually served from the pixel cache."""
        with self._lock:
            return self._cache_hits

    @staticmethod
    def _decode(key: _FileKey) -> Image.Image:
        if _file_key(key.path) != key:
            raise OSError(f"Image changed before decoding: {key.path}")
        # No draft/downsampling: annotation coordinates need the original RGB detail.
        with Image.open(key.path) as source:
            image = source.convert("RGB")
            image.load()
        if _file_key(key.path) != key:
            image.close()
            raise OSError(f"Image changed while decoding: {key.path}")
        return image

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("ImageLoader is closed")

    def _invalidate_path(self, key: _FileKey) -> None:
        obsolete = [cached for cached in self._cache if cached.path == key.path and cached != key]
        for cached in obsolete:
            image = self._cache.pop(cached)
            self._cache_bytes -= image.width * image.height * 3
            # Consumers may still display this shared image; eviction never closes it.

    def _start_jobs(self) -> None:
        while self._waiting and len(self._active) < self._WORKERS and not self._closed:
            key = next(
                (key for key, job in self._waiting.items() if job.callbacks),
                next(iter(self._waiting)),
            )
            job = self._waiting.pop(key)
            job.future = self._executor.submit(self._decode, key)
            self._active[key] = job
            job.future.add_done_callback(lambda _future, key=key: self._decode_done(key))

    def _decode_done(self, key: _FileKey) -> None:
        # A worker records completion only. Cache updates and callbacks happen in drain.
        with self._lock:
            if not self._closed:
                self._completed.append(key)

    def request(self, path: Path | str, callback: ImageCallback) -> None:
        """Queue an image or error callback without decoding/blocking the caller."""
        if not callable(callback):
            raise TypeError("callback must be callable")
        requested = Path(path)
        with self._lock:
            self._require_open()
        try:
            key = _file_key(requested)
        except (OSError, ValueError, RuntimeError) as exc:
            with self._lock:
                self._require_open()
                self._errors.append((requested, callback, exc))
            return
        with self._lock:
            self._require_open()
            self._invalidate_path(key)
            if key in self._cache:
                self._cache.move_to_end(key)
                # Store a key, not another RGB reference; queued hits cannot pin evictions.
                self._hits.append((requested, key, callback))
                return
            job = self._active.get(key)
            if job is None:
                job = self._waiting.setdefault(key, _DecodeJob())
            job.callbacks.append((requested, callback))
            self._start_jobs()

    def prefetch(self, paths: Iterable[Path | str]) -> None:
        """Queue bounded low-priority work; errors are reported by a later request."""
        with self._lock:
            self._require_open()
        for path in islice(paths, self._PREFETCH_LIMIT):
            with self._lock:
                self._require_open()
                queued = sum(not job.callbacks for job in self._waiting.values())
                queued += sum(not job.callbacks for job in self._active.values())
                if queued >= self._PREFETCH_LIMIT:
                    break
            try:
                key = _file_key(Path(path))
            except (OSError, ValueError, RuntimeError):
                continue
            with self._lock:
                self._require_open()
                self._invalidate_path(key)
                if key in self._cache or key in self._active or key in self._waiting:
                    continue
                self._waiting[key] = _DecodeJob()
                self._start_jobs()

    def _cache_image(self, key: _FileKey, image: Image.Image) -> None:
        size = image.width * image.height * 3
        if size > self.max_bytes:
            return
        old = self._cache.pop(key, None)
        if old is not None:
            self._cache_bytes -= old.width * old.height * 3
        while self._cache and self._cache_bytes + size > self.max_bytes:
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= evicted.width * evicted.height * 3
        self._invalidate_path(key)
        self._cache[key] = image
        self._cache_bytes += size

    def drain(self) -> int:
        """Dispatch ready callbacks on the creating/Tk thread; return delivery count.

        A callback exception does not discard other ready callbacks; the first such
        exception is re-raised after delivery. No callback runs after close().
        """
        if threading.get_ident() != self._owner:
            raise RuntimeError("ImageLoader.drain() must run on its creating/Tk thread")
        deliveries: list[tuple[Path, ImageCallback, Image.Image | None, Exception | None]] = []
        with self._lock:
            if self._closed:
                return 0
            errors = list(self._errors)
            self._errors.clear()
            hits = list(self._hits)
            self._hits.clear()
            completed = list(self._completed)
            self._completed.clear()
        deliveries.extend((path, callback, None, error) for path, callback, error in errors)
        for path, key, callback in hits:
            try:
                current = _file_key(path)
            except (OSError, ValueError, RuntimeError) as exc:
                deliveries.append((path, callback, None, exc))
                continue
            with self._lock:
                if self._closed:
                    return 0
                image = self._cache.get(key) if current == key else None
                if image is not None:
                    self._cache.move_to_end(key)
                    self._cache_hits += 1
            if image is None:
                self.request(path, callback)
            else:
                deliveries.append((path, callback, image, None))
        for key in completed:
            with self._lock:
                if self._closed:
                    return 0
                job = self._active.get(key)
            if job is None or job.future is None:
                continue
            image = None
            error = None
            try:
                image = job.future.result()
                if _file_key(key.path) != key:
                    raise OSError(f"Image changed before delivery: {key.path}")
            except Exception as exc:
                image = None
                error = exc
            with self._lock:
                if self._closed:
                    return 0
                self._active.pop(key, None)
                if image is not None:
                    self._cache_image(key, image)
                deliveries.extend((path, callback, image, error) for path, callback in job.callbacks)
        with self._lock:
            self._start_jobs()
        count = 0
        first_exception = None
        for path, callback, image, error in deliveries:
            with self._lock:
                if self._closed:
                    break
                try:
                    callback(path, image, error)
                except Exception as exc:
                    if first_exception is None:
                        first_exception = exc
                count += 1
        if first_exception is not None:
            raise first_exception
        return count

    def close(self) -> None:
        """Cancel queued work and discard results; do not wait for running file I/O."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for job in self._active.values():
                if job.future is not None:
                    job.future.cancel()
            self._waiting.clear()
            self._active.clear()
            self._completed.clear()
            self._hits.clear()
            self._errors.clear()
            self._cache.clear()
            self._cache_bytes = 0
            self._executor.shutdown(wait=False, cancel_futures=True)


def _normalize_name(name: str) -> str:
    return " ".join(name.replace("_", " ").split()).casefold()


class _ClassResolver:
    def __init__(self, class_names: Sequence[str], mapping: Mapping[str, str | int] | None) -> None:
        if not class_names or any(not isinstance(name, str) or not _normalize_name(name) for name in class_names):
            raise ValueError("Dataset classes must be nonempty names")
        self.class_count = len(class_names)
        self.dataset: dict[str, list[int]] = {}
        for index, name in enumerate(class_names):
            self.dataset.setdefault(_normalize_name(name), []).append(index)
        self.explicit: dict[str, list[str | int]] | None = None
        if mapping is not None:
            self.explicit = {}
            for name, target in mapping.items():
                if not isinstance(name, str) or not _normalize_name(name):
                    raise ValueError("Mapping keys must be nonempty model class names")
                self.explicit.setdefault(_normalize_name(name), []).append(target)

    def _dataset_id(self, name: str) -> int:
        matches = self.dataset.get(_normalize_name(name), [])
        if len(matches) != 1:
            problem = "ambiguous" if matches else "missing"
            raise ValueError(f"{problem.capitalize()} dataset class mapping for {name!r}")
        return matches[0]

    def resolve(self, model_name: str) -> int:
        if self.explicit is None:
            return self._dataset_id(model_name)
        targets = self.explicit.get(_normalize_name(model_name), [])
        if len(targets) != 1:
            problem = "ambiguous" if targets else "missing"
            raise ValueError(f"{problem.capitalize()} explicit mapping for model class {model_name!r}")
        target = targets[0]
        if isinstance(target, str):
            return self._dataset_id(target)
        if isinstance(target, int) and not isinstance(target, bool) and 0 <= target < self.class_count:
            return target
        raise ValueError(f"Invalid dataset class target {target!r} for model class {model_name!r}")


@dataclass
class _ModelEntry:
    model: Any
    lock: threading.Lock = field(default_factory=threading.Lock)


_MODEL_CACHE: OrderedDict[_FileKey, _ModelEntry] = OrderedDict()
_MODEL_LOCK = threading.Lock()
_MODEL_CACHE_LIMIT = 2


def _get_model(key: _FileKey) -> _ModelEntry:
    # Construction is serialized, and each predictor has its own inference lock:
    # Ultralytics mutates predictor state and is not safe to call concurrently.
    with _MODEL_LOCK:
        entry = _MODEL_CACHE.get(key)
        if entry is not None:
            _MODEL_CACHE.move_to_end(key)
            return entry
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO assistance requires ultralytics. Install the vision dependencies with `uv sync --extra vision`."
            ) from exc
        try:
            model = YOLO(str(key.path))
        except Exception as exc:
            raise RuntimeError(f"Cannot load YOLO weights {key.path}: {exc}") from exc
        if _file_key(key.path) != key:
            raise OSError(f"YOLO weights changed while loading: {key.path}")
        for stale in [cached for cached in _MODEL_CACHE if cached.path == key.path]:
            del _MODEL_CACHE[stale]
        while len(_MODEL_CACHE) >= _MODEL_CACHE_LIMIT:
            _MODEL_CACHE.popitem(last=False)
        entry = _ModelEntry(model)
        _MODEL_CACHE[key] = entry
        return entry


def predict_boxes(
    image_path: Path | str,
    weights: Path | str,
    class_names: Sequence[str],
    confidence: float = 0.25,
    mapping: Mapping[str, str | int] | None = None,
) -> list[Candidate]:
    """Run real local YOLO weights and return explicitly reviewable candidates.

    Names match exactly after case folding and underscore/whitespace normalization.
    If mapping is supplied, each emitted model name needs an explicit entry whose
    value is a dataset class name or id. Model ids are never used as dataset ids.
    Missing/ambiguous emitted class mappings reject the entire result, not just an
    unmapped subset. Calling code must perform inference in its background executor.
    """
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be finite and between 0 and 1")
    resolver = _ClassResolver(class_names, mapping)
    weights_key = _file_key(Path(weights))
    image_key = _file_key(Path(image_path))
    entry = _get_model(weights_key)
    candidates = []
    # PIL input uses the same unrotated pixel coordinate system as ImageLoader.
    with Image.open(image_key.path) as original:
        with original.convert("RGB") as rgb:
            width, height = rgb.size
            with entry.lock:
                results = entry.model.predict(source=rgb, conf=confidence, verbose=False, save=False, stream=False)
                for result in results:
                    if result.boxes is None:
                        raise ValueError("These YOLO weights do not produce axis-aligned detections")
                    detections = result.boxes.cpu()
                    for coordinates, model_id, score in zip(
                        detections.xyxy.tolist(),
                        detections.cls.tolist(),
                        detections.conf.tolist(),
                        strict=True,
                    ):
                        if not math.isfinite(score) or not confidence <= score <= 1:
                            continue
                        if len(coordinates) != 4 or not all(math.isfinite(value) for value in coordinates):
                            continue
                        x1, y1, x2, y2 = coordinates
                        x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
                        y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        if not math.isfinite(model_id) or model_id < 0 or not float(model_id).is_integer():
                            raise ValueError(f"Invalid YOLO model class id: {model_id!r}")
                        try:
                            model_name = result.names[int(model_id)]
                        except (KeyError, IndexError, TypeError) as exc:
                            raise ValueError(f"YOLO has no class name for id {model_id!r}") from exc
                        if not isinstance(model_name, str) or not _normalize_name(model_name):
                            raise ValueError(f"Invalid YOLO model class name: {model_name!r}")
                        cls = resolver.resolve(model_name)
                        candidates.append(Candidate(Box(cls, x1, y1, x2, y2), f"yolo:{weights_key.path}", float(score)))
    if _file_key(image_key.path) != image_key:
        raise OSError(f"Image changed during YOLO inference: {image_key.path}")
    return candidates


def propagate_boxes(previous_path: Path | str, current_path: Path | str, boxes: Sequence[Box]) -> list[Candidate]:
    """Track per-box texture with pyramidal LK, FB checks, and robust partial affine.

    Both frames are brought to a common, at-most-1280-pixel working resolution.
    Coordinates are mapped back to the destination's original resolution. Weak,
    collinear, inconsistent, excessive-motion, and mostly off-frame tracks produce
    no candidate; there is intentionally no unchanged-box fallback.
    """
    if not boxes:
        return []
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Frame propagation requires OpenCV and NumPy. Install the vision dependencies "
            "with `uv sync --extra vision`."
        ) from exc
    previous_key = _file_key(Path(previous_path))
    current_key = _file_key(Path(current_path))

    def read_gray(key: _FileKey) -> Any:
        # Pillow's coordinate convention matches the loader, including EXIF handling.
        with Image.open(key.path) as image:
            with image.convert("L") as gray:
                return np.asarray(gray)

    previous = read_gray(previous_key)
    source_height, source_width = previous.shape
    scale = min(1.0, 1280.0 / max(source_width, source_height))
    work_width = max(1, round(source_width * scale))
    work_height = max(1, round(source_height * scale))
    if (work_width, work_height) != (source_width, source_height):
        previous = cv2.resize(previous, (work_width, work_height), interpolation=cv2.INTER_AREA)
    current = read_gray(current_key)
    destination_height, destination_width = current.shape
    if (destination_width, destination_height) != (work_width, work_height):
        interpolation = (
            cv2.INTER_AREA
            if destination_width >= work_width and destination_height >= work_height
            else cv2.INTER_LINEAR
        )
        current = cv2.resize(current, (work_width, work_height), interpolation=interpolation)
    source_scale_x, source_scale_y = work_width / source_width, work_height / source_height
    destination_scale_x = destination_width / work_width
    destination_scale_y = destination_height / work_height
    lk_options = {
        "winSize": (21, 21),
        # Keep at least two tracking windows across the coarsest image axis.
        # Over-deep pyramids on thumbnails lose otherwise valid ROI correspondences.
        "maxLevel": min(3, max(0, int(math.log2(max(1, min(work_width, work_height) / 42))))),
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        "minEigThreshold": 1e-4,
    }
    prepared = []
    all_features = []
    point_offset = 0
    for index, box in enumerate(boxes):
        coordinates = (box.x1, box.y1, box.x2, box.y2)
        if (
            not isinstance(box.cls, int)
            or isinstance(box.cls, bool)
            or box.cls < 0
            or not all(math.isfinite(value) for value in coordinates)
        ):
            continue
        left = max(0.0, min(source_width, box.x1)) * source_scale_x
        top = max(0.0, min(source_height, box.y1)) * source_scale_y
        right = max(0.0, min(source_width, box.x2)) * source_scale_x
        bottom = max(0.0, min(source_height, box.y2)) * source_scale_y
        if right - left < 6 or bottom - top < 6:
            continue
        roi_left, roi_top = math.floor(left), math.floor(top)
        roi_right, roi_bottom = math.ceil(right), math.ceil(bottom)
        features = cv2.goodFeaturesToTrack(
            previous[roi_top:roi_bottom, roi_left:roi_right],
            maxCorners=120,
            qualityLevel=0.01,
            minDistance=3,
            blockSize=3,
        )
        if features is None or len(features) < 12:
            continue
        features += np.array([roi_left, roi_top], dtype=np.float32)
        end = point_offset + len(features)
        prepared.append((index, box, (left, top, right, bottom), point_offset, end))
        all_features.append(features)
        point_offset = end
    if not prepared:
        return []
    # Batch every ROI's tracks so LK builds image pyramids once per direction,
    # rather than reallocating/recomputing full-frame pyramids for every box.
    points = np.concatenate(all_features)
    points_flat = points.reshape(-1, 2)
    forward, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **lk_options)
    if forward is None or forward_status is None or forward_error is None:
        return []
    forward_points = forward.reshape(-1, 2)
    good = forward_status.reshape(-1).astype(bool)
    good &= np.isfinite(forward_points).all(axis=1)
    good &= np.isfinite(forward_error.reshape(-1)) & (forward_error.reshape(-1) <= 30)
    good &= (forward_points[:, 0] >= 0) & (forward_points[:, 0] < work_width)
    good &= (forward_points[:, 1] >= 0) & (forward_points[:, 1] < work_height)
    if np.count_nonzero(good) < 8:
        return []
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, forward_points[good].reshape(-1, 1, 2), None, **lk_options
    )
    if backward is None or backward_status is None:
        return []
    backward_points = backward.reshape(-1, 2)
    reverse_error = np.linalg.norm(backward_points - points_flat[good], axis=1)
    backward_good = backward_status.reshape(-1).astype(bool)
    backward_good &= np.isfinite(backward_points).all(axis=1) & (reverse_error <= 1.5)
    consistent = np.zeros(point_offset, dtype=bool)
    consistent[good] = backward_good
    fb_errors = np.full(point_offset, np.inf, dtype=np.float32)
    fb_errors[good] = reverse_error
    candidates = []
    for index, box, rectangle, start, end in prepared:
        left, top, right, bottom = rectangle
        valid = consistent[start:end]
        count = int(np.count_nonzero(valid))
        feature_count = end - start
        if count < 8 or count / feature_count < 0.4:
            continue
        source_points = points_flat[start:end][valid]
        destination_points = forward_points[start:end][valid]
        fb_error = fb_errors[start:end][valid]
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            source_points,
            destination_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
        if affine is None or inlier_mask is None or not np.isfinite(affine).all():
            continue
        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = inlier_count / count
        if inlier_count < 8 or inlier_ratio < 0.65:
            continue
        support = source_points[inliers]
        span = np.ptp(support, axis=0)
        box_area = (right - left) * (bottom - top)
        hull_area = cv2.contourArea(cv2.convexHull(support))
        if span[0] < 0.25 * (right - left) or span[1] < 0.25 * (bottom - top) or hull_area < 0.08 * box_area:
            continue
        motion_scale = math.hypot(float(affine[0, 0]), float(affine[1, 0]))
        rotation = abs(math.degrees(math.atan2(float(affine[1, 0]), float(affine[0, 0]))))
        displacement = math.hypot(float(affine[0, 2]), float(affine[1, 2]))
        if not 0.7 <= motion_scale <= 1.4 or rotation > 25 or displacement > 0.4 * math.hypot(work_width, work_height):
            continue
        projected = support @ affine[:, :2].T + affine[:, 2]
        residual = np.linalg.norm(projected - destination_points[inliers], axis=1)
        median_residual = float(np.median(residual))
        median_fb = float(np.median(fb_error[inliers]))
        if median_residual > 1.25 or float(np.quantile(residual, 0.9)) > 2 or median_fb > 0.75:
            continue
        corners = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
        transformed = corners @ affine[:, :2].T + affine[:, 2]
        low, high = transformed.min(axis=0), transformed.max(axis=0)
        raw_area = float((high[0] - low[0]) * (high[1] - low[1]))
        low = np.maximum(low, [0, 0])
        high = np.minimum(high, [work_width, work_height])
        if raw_area <= 0 or np.any(high <= low):
            continue
        visible_fraction = float((high[0] - low[0]) * (high[1] - low[1]) / raw_area)
        if visible_fraction < 0.6:
            continue
        quality = (
            inlier_ratio
            * (count / feature_count)
            * min(1.0, hull_area / (0.35 * box_area))
            * math.exp(-median_fb / 1.5)
            * math.exp(-median_residual / 2)
            * visible_fraction
        )
        candidates.append(
            Candidate(
                Box(
                    box.cls,
                    float(low[0] * destination_scale_x),
                    float(low[1] * destination_scale_y),
                    float(high[0] * destination_scale_x),
                    float(high[1] * destination_scale_y),
                ),
                f"flow:{previous_key.path.name}:box-{index + 1}",
                min(1.0, max(0.0, quality)),
            )
        )
    if _file_key(previous_key.path) != previous_key or _file_key(current_key.path) != current_key:
        raise OSError("A frame changed during propagation; reload the source frames")
    return candidates
