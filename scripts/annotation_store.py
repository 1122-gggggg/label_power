"""Strict YOLO labels, atomic versioned writes, and explicit dataset-bound review state."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"})
HISTORY_LIMIT = 10
_STATUSES = frozenset({"pending", "edited", "positive", "negative"})
_CLASS_TOKEN = re.compile(r"[0-9]+\Z")


class _Unset:
    __slots__ = ()


_UNSET = _Unset()


class LabelError(ValueError):
    """Invalid label/catalog/progress data, with its source and actionable reason."""

    def __init__(self, path: Path | str, reason: str, line: int | None = None) -> None:
        self.path = Path(path)
        self.line = line
        self.reason = reason
        location = f"{self.path}:{line}" if line is not None else str(self.path)
        super().__init__(f"{location}: {reason}")


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    def normalized(self, width: int, height: int) -> tuple[float, float, float, float]:
        return (
            (self.x1 / width + self.x2 / width) / 2.0,
            (self.y1 / height + self.y2 / height) / 2.0,
            (self.x2 - self.x1) / width,
            (self.y2 - self.y1) / height,
        )

    def clamp(self, width: int, height: int) -> Box:
        """Explicit UI operation, never applied to labels during parsing or saving."""
        _dimensions(Path("<box>"), width, height)
        values = (self.x1, self.y1, self.x2, self.y2)
        if any(not _finite_number(value) for value in values):
            raise LabelError("<box>", "coordinates must be finite numbers")
        x1, x2 = sorted((clamp(self.x1, 0.0, float(width)), clamp(self.x2, 0.0, float(width))))
        y1, y2 = sorted((clamp(self.y1, 0.0, float(height)), clamp(self.y2, 0.0, float(height))))
        return Box(self.cls, x1, y1, x2, y2)

    @property
    def width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)


def copy_boxes(boxes: Iterable[Box]) -> list[Box]:
    return [Box(box.cls, box.x1, box.y1, box.x2, box.y2) for box in boxes]


def _finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _dimensions(path: Path, width: int, height: int) -> None:
    if not _finite_number(width) or not _finite_number(height) or width <= 0 or height <= 0:
        raise LabelError(path, "image width and height must be finite, positive numbers")


def _class_count(path: Path, count: int | None) -> None:
    if count is not None and (type(count) is not int or count <= 0):
        raise LabelError(path, "class_count must be a positive integer")


def _class_id(path: Path, cls: Any, count: int | None, line: int | None = None) -> None:
    if type(cls) is not int or cls < 0:
        raise LabelError(path, "class id must be a non-negative integer (not bool)", line)
    if count is not None and cls >= count:
        raise LabelError(path, f"class id {cls} is outside the catalog [0, {count - 1}]", line)


def _parse_class_token(path: Path, token: Any, line: int | None = None) -> int:
    if type(token) is not str or _CLASS_TOKEN.fullmatch(token) is None:
        raise LabelError(path, "class id must be a non-negative integer token", line)
    try:
        return int(token)
    except ValueError as exc:
        raise LabelError(path, "class id token is too long to parse", line) from exc


def _image_name(path: Path, name: Any) -> None:
    if type(name) is not str or not name or Path(name).name != name or "\\" in name:
        raise LabelError(path, "image names must be non-empty filenames without directories")
    if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
        raise LabelError(path, f"unsupported image name {name!r}")


def _unique_images(path: Path, names: Iterable[str]) -> None:
    stems: dict[str, str] = {}
    for name in names:
        _image_name(path, name)
        stem = Path(name).stem.casefold()
        if stem in stems:
            raise LabelError(path, f"image label-stem collision: {stems[stem]!r} and {name!r}")
        stems[stem] = name


def collect_images(path: Path) -> list[Path]:
    """Collect one image directory, rejecting ambiguous shared label filenames."""
    path = Path(path)
    images = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    _unique_images(path, (p.name for p in images))
    return images


def read_classes(path: Path) -> list[str]:
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise LabelError(path, f"cannot read class catalog: {exc}") from exc
    classes: list[str] = []
    seen: set[str] = set()
    for line_no, line in enumerate(lines, 1):
        name = line.strip()
        if not name:
            continue
        if name in seen:
            raise LabelError(path, f"duplicate class name {name!r}", line_no)
        seen.add(name)
        classes.append(name)
    if not classes:
        raise LabelError(path, "class catalog is empty; provide one class name per line")
    return classes


def label_path_for(image_path: Path, labels_dir: Path) -> Path:
    return Path(labels_dir) / f"{Path(image_path).stem}.txt"


def read_yolo_boxes(label_path: Path, width: int, height: int, class_count: int | None = None) -> list[Box]:
    """Read complete valid labels; missing files raise FileNotFoundError, not an empty result."""
    path = Path(label_path)
    _dimensions(path, width, height)
    _class_count(path, class_count)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError) as exc:
        raise LabelError(path, f"cannot read labels: {exc}") from exc
    boxes: list[Box] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise LabelError(path, f"expected 5 fields, found {len(parts)}", line_no)
        cls = _parse_class_token(path, parts[0], line_no)
        _class_id(path, cls, class_count, line_no)
        try:
            values = [float(token) for token in parts[1:]]
        except ValueError as exc:
            raise LabelError(path, "box coordinates must be numeric", line_no) from exc
        if not all(math.isfinite(value) for value in values):
            raise LabelError(path, "box coordinates must be finite (NaN/inf are not labels)", line_no)
        # Exact decimal ratios distinguish real out-of-bounds input from binary rounding.
        try:
            xc, yc, bw, bh = (Fraction(token) for token in parts[1:])
        except (ValueError, ZeroDivisionError) as exc:
            raise LabelError(path, "box coordinates must be decimal numbers", line_no) from exc
        if not all(0 <= value <= 1 for value in (xc, yc, bw, bh)):
            raise LabelError(path, "normalized center and size must be within [0, 1]", line_no)
        if bw <= 0 or bh <= 0:
            raise LabelError(path, "box width and height must be positive", line_no)
        x1, x2 = xc - bw / 2, xc + bw / 2
        y1, y2 = yc - bh / 2, yc + bh / 2
        if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
            raise LabelError(path, "normalized box extents must be within [0, 1]", line_no)
        box = Box(cls, float(x1 * width), float(y1 * height), float(x2 * width), float(y2 * height))
        if box.x2 <= box.x1 or box.y2 <= box.y1:
            raise LabelError(path, "box size is too small to represent in image coordinates", line_no)
        boxes.append(box)
    return boxes


def _normalized_axis(start: int | float, end: int | float, dimension: int) -> tuple[Decimal, Decimal]:
    """Serialize valid edges inward at high precision, then derive exact center/extent.

    Independently rounding center and size can make a boundary box invalid on reload.
    Deriving both from the same decimal edges avoids that without accepting invalid input.
    """
    with localcontext() as context:
        context.prec = 20
        context.rounding = ROUND_CEILING
        low_ratio = Fraction(start) / Fraction(dimension)
        low = Decimal(low_ratio.numerator) / Decimal(low_ratio.denominator)
        context.rounding = ROUND_FLOOR
        high_ratio = Fraction(end) / Fraction(dimension)
        high = Decimal(high_ratio.numerator) / Decimal(high_ratio.denominator)
        context.prec = max(
            50,
            len(low.as_tuple().digits) + len(high.as_tuple().digits) + abs(low.adjusted() - high.adjusted()) + 5,
        )
        return (low + high) / 2, high - low


def write_yolo_boxes(
    label_path: Path,
    boxes: Iterable[Box],
    width: int,
    height: int,
    class_count: int | None = None,
    *,
    expected_signature: str | None | _Unset = _UNSET,
) -> None:
    """Validate and atomically save; an expected SHA256 guards against lost edits.

    ``expected_signature=None`` requires a missing file. Omitting the keyword
    permits an explicit programmatic overwrite of the current version.
    """
    path = Path(label_path)
    _dimensions(path, width, height)
    _class_count(path, class_count)
    lines: list[str] = []
    for line_no, box in enumerate(boxes, 1):
        if not isinstance(box, Box):
            raise LabelError(path, "expected a Box", line_no)
        _class_id(path, box.cls, class_count, line_no)
        if any(not _finite_number(value) for value in (box.x1, box.y1, box.x2, box.y2)):
            raise LabelError(path, "box coordinates must be finite numbers", line_no)
        if box.x1 < 0 or box.y1 < 0 or box.x2 > width or box.y2 > height:
            raise LabelError(path, "box extents must be within the image's continuous bounds", line_no)
        if box.x2 <= box.x1 or box.y2 <= box.y1:
            raise LabelError(path, "box width and height must be positive; coordinates must be ordered", line_no)
        xc, bw = _normalized_axis(box.x1, box.x2, width)
        yc, bh = _normalized_axis(box.y1, box.y2, height)
        if bw <= 0 or bh <= 0:
            raise LabelError(path, "box size is too small to serialize", line_no)
        lines.append(f"{box.cls} {xc} {yc} {bw} {bh}\n")
    _atomic_write_bytes(path, "".join(lines).encode("utf-8"), expected_signature=expected_signature)


def _backup(path: Path, contents: bytes) -> None:
    history = path.parent / ".history" / path.name
    history.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f"version-{time.time_ns()}-", suffix=".bak", dir=history, delete=False
    ) as backup:
        backup_path = Path(backup.name)
        try:
            backup.write(contents)
            backup.flush()
            os.fsync(backup.fileno())
        except BaseException:
            backup_path.unlink(missing_ok=True)
            raise
    versions = sorted(history.glob("version-*.bak"), key=lambda p: p.name, reverse=True)
    for old in versions[HISTORY_LIMIT:]:
        old.unlink(missing_ok=True)


@contextmanager
def _path_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"
    # Keep this stable inode: unlinking it could let a new writer bypass an old lock.
    with lock_path.open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt

                if os.fstat(lock.fileno()).st_size == 0:
                    lock.write(b"\0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise LabelError(path, "another session is writing this file; retry after it finishes") from exc
        yield


def _atomic_write_bytes(path: Path, contents: bytes, *, expected_signature: str | None | _Unset = _UNSET) -> None:
    if expected_signature is not _UNSET and expected_signature is not None:
        if type(expected_signature) is not str or re.fullmatch(r"[0-9a-fA-F]{64}", expected_signature) is None:
            raise LabelError(path, "expected_signature must be a SHA256 hex digest, or None for a missing file")
        expected_signature = expected_signature.lower()
    with _path_lock(path):
        try:
            previous = path.read_bytes()
        except FileNotFoundError:
            previous = None
        if expected_signature is not _UNSET:
            current_signature = hashlib.sha256(previous).hexdigest() if previous is not None else None
            if current_signature != expected_signature:
                raise LabelError(
                    path,
                    "label changed on disk since loading; reload it before saving to avoid losing another session's work",
                )
        _replace_bytes(path, contents, previous)


def _replace_bytes(path: Path, contents: bytes, previous: bytes | None) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        if previous is not None:
            _backup(path, previous)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    _atomic_write_bytes(path, _json_bytes(path, payload))


def _json_bytes(path: Path, payload: Any) -> bytes:
    try:
        return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise LabelError(path, f"cannot serialize JSON: {exc}") from exc


def _classes(path: Path, names: Any) -> list[str]:
    if type(names) is not list or not names:
        raise LabelError(path, "class_names must be a non-empty ordered list")
    if any(type(name) is not str or not name.strip() or name != name.strip() for name in names):
        raise LabelError(path, "class names must be non-empty, trimmed strings")
    if len(set(names)) != len(names):
        raise LabelError(path, "class names must be unique")
    return names.copy()


def _identity(dataset: dict[str, Any]) -> str:
    fields = {key: dataset[key] for key in ("image_root", "labels_root", "class_names", "manifest")}
    encoded = json.dumps(fields, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _current_dataset(path: Path, images: list[Path], classes: list[str], labels_dir: Path | None) -> dict[str, Any]:
    if not images:
        raise LabelError(path, "review state requires a non-empty image dataset")
    root = Path(images[0]).absolute().parent.resolve()
    _unique_images(path, (Path(image).name for image in images))
    manifest = []
    for image in sorted((Path(image) for image in images), key=lambda p: p.name):
        if image.absolute().parent.resolve() != root:
            raise LabelError(path, "all review images must belong to the same image directory")
        try:
            info = image.stat()
        except OSError as exc:
            raise LabelError(path, f"cannot stat dataset image {image}: {exc}") from exc
        if not stat.S_ISREG(info.st_mode):
            raise LabelError(path, f"dataset image is not a file: {image}")
        manifest.append({"name": image.name, "size": info.st_size, "mtime_ns": info.st_mtime_ns})
    dataset: dict[str, Any] = {
        "image_root": str(root),
        "labels_root": str(labels_dir.resolve()) if labels_dir is not None else None,
        "class_names": classes,
        "manifest": manifest,
    }
    dataset["identity"] = _identity(dataset)
    return dataset


def _stored_dataset(path: Path, raw: Any) -> dict[str, Any]:
    if type(raw) is not dict:
        raise LabelError(path, "v2 progress must include a dataset object")
    root = raw.get("image_root")
    if type(root) is not str or not Path(root).is_absolute():
        raise LabelError(path, "dataset image_root must be an absolute path")
    labels_root = raw.get("labels_root")
    if labels_root is not None and (type(labels_root) is not str or not Path(labels_root).is_absolute()):
        raise LabelError(path, "dataset labels_root must be an absolute path or null")
    classes = _classes(path, raw.get("class_names"))
    manifest = raw.get("manifest")
    if type(manifest) is not list:
        raise LabelError(path, "dataset manifest must be a list of image name/size/mtime_ns records")
    records = []
    for record in manifest:
        if type(record) is not dict:
            raise LabelError(path, "each manifest entry must be an object")
        name = record.get("name")
        _image_name(path, name)
        for field in ("size", "mtime_ns"):
            if type(record.get(field)) is not int or record[field] < 0:
                raise LabelError(path, f"manifest {name!r} {field} must be a non-negative integer")
        records.append({"name": name, "size": record["size"], "mtime_ns": record["mtime_ns"]})
    _unique_images(path, (record["name"] for record in records))
    dataset: dict[str, Any] = {
        "image_root": root,
        "labels_root": labels_root,
        "class_names": classes,
        "manifest": records,
    }
    identity = raw.get("identity")
    if type(identity) is not str or identity != _identity(dataset):
        raise LabelError(path, "dataset identity does not match its manifest; restore or repair the progress file")
    dataset["identity"] = identity
    return dataset


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value}")


def _unique_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_progress(path: Path, contents: bytes) -> dict[str, Any]:
    try:
        raw = json.loads(
            contents.decode("utf-8-sig"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_json_keys,
        )
    except json.JSONDecodeError as exc:
        raise LabelError(
            path,
            f"malformed progress JSON ({exc.msg}, column {exc.colno}); fix it or restore a .history backup",
            exc.lineno,
        ) from exc
    except (OSError, UnicodeError, ValueError) as exc:
        raise LabelError(path, f"cannot read progress JSON: {exc}; fix it or restore a .history backup") from exc
    if type(raw) is not dict:
        raise LabelError(path, "progress JSON must contain an object, not an array or scalar")
    return raw


class ReviewStore:
    """Explicit confirmations, bound to an image root, class order, and file manifest.

    Normal construction migrates/reconciles with a version backup. ``read_only=True``
    does the same safety reconciliation only in memory, for verification/export.
    """

    def __init__(
        self,
        path: Path,
        images: list[Path],
        class_names: list[str],
        *,
        read_only: bool = False,
        labels_dir: Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.notices: list[str] = []
        self._read_only = read_only
        self._classes = _classes(self.path, class_names)
        self._labels_dir = Path(labels_dir) if labels_dir is not None else None
        self._dataset = _current_dataset(self.path, images, self._classes, self._labels_dir)
        self._images = {Path(image).name: Path(image) for image in images}
        self._names = {record["name"] for record in self._dataset["manifest"]}
        self._statuses: dict[str, dict[int, str]] = {}
        self._signatures: dict[str, dict[int, str]] = {}
        try:
            self._expected_progress = self.path.read_bytes()
        except FileNotFoundError:
            self._expected_progress = None
        if self._expected_progress is not None:
            raw = _read_progress(self.path, self._expected_progress)
            version = raw.get("version", 1)
            if type(version) is not int or version not in (1, 2):
                raise LabelError(self.path, "unsupported progress version; expected integer 1 or 2")
            if version == 1:
                self._migrate_legacy(raw)
                changed = True
            else:
                changed = self._reconcile(raw)
        else:
            changed = True
        if changed and not self._read_only:
            self._persist(self._statuses, self._signatures)

    def _migrate_legacy(self, raw: dict[str, Any]) -> None:
        reviewed = raw.get("reviewed")
        if type(reviewed) is not dict:
            raise LabelError(self.path, "legacy progress must have a reviewed object")
        self.notices.append("Legacy v1 auto-review is untrusted: matching entries migrated to edited, never confirmed.")
        for image_name, class_ids in reviewed.items():
            _image_name(self.path, image_name)
            if type(class_ids) is not list:
                raise LabelError(self.path, f"legacy reviewed entry for {image_name!r} must be a list of class ids")
            for cls in class_ids:
                _class_id(self.path, cls, len(self._classes))
            if image_name not in self._names:
                self.notices.append(f"Stale image removed from active review state: {image_name}")
                continue
            if class_ids:
                self._statuses[image_name] = dict.fromkeys(class_ids, "edited")

    def _reconcile(self, raw: dict[str, Any]) -> bool:
        old = _stored_dataset(self.path, raw.get("dataset"))
        old_manifest = {record["name"]: record for record in old["manifest"]}
        new_manifest = {record["name"]: record for record in self._dataset["manifest"]}
        statuses = raw.get("statuses")
        if type(statuses) is not dict:
            raise LabelError(self.path, "v2 progress must have a statuses object")
        validated: dict[str, dict[int, str]] = {}
        for name, class_statuses in statuses.items():
            _image_name(self.path, name)
            if name not in old_manifest:
                raise LabelError(self.path, f"review status references an image absent from its manifest: {name!r}")
            if type(class_statuses) is not dict:
                raise LabelError(self.path, f"statuses for {name!r} must be a class-id/status object")
            validated[name] = {}
            for token, status in class_statuses.items():
                cls = _parse_class_token(self.path, token)
                if str(cls) != token:
                    raise LabelError(self.path, f"invalid class id key {token!r} for {name!r}")
                _class_id(self.path, cls, len(old["class_names"]))
                if type(status) is not str or status not in _STATUSES:
                    raise LabelError(self.path, f"invalid review status {status!r} for {name!r}, class {cls}")
                validated[name][cls] = status
        saved_signatures = raw.get("label_signatures", {})
        if type(saved_signatures) is not dict:
            raise LabelError(self.path, "label_signatures must be an image/class-id/signature object")
        signatures: dict[str, dict[int, str]] = {}
        for name, entries in saved_signatures.items():
            if name not in validated or type(entries) is not dict:
                raise LabelError(self.path, f"invalid label signature image entry {name!r}")
            signatures[name] = {}
            for token, signature in entries.items():
                cls = _parse_class_token(self.path, token)
                if str(cls) != token:
                    raise LabelError(self.path, f"invalid signature class id {token!r}")
                if validated[name].get(cls) not in ("positive", "negative"):
                    raise LabelError(self.path, f"signature must refer to a confirmed class: {name!r}, {cls}")
                if type(signature) is not str or re.fullmatch(r"[0-9a-f]{64}", signature) is None:
                    raise LabelError(self.path, f"invalid label SHA256 signature for {name!r}, class {cls}")
                signatures[name][cls] = signature
        same_root = old["image_root"] == self._dataset["image_root"]
        same_classes = old["class_names"] == self._classes
        same_labels = old["labels_root"] == self._dataset["labels_root"]
        if not same_root:
            self.notices.append(
                "Image root changed: unrelated dataset review state was reset; no confirmations reused."
            )
        if not same_classes:
            self.notices.append("Class catalog/order changed: all review statuses were reset safely.")
        if not same_labels:
            self.notices.append("Labels root changed: all review statuses were reset for the selected label set.")
        for name, record in old_manifest.items():
            if name not in new_manifest:
                self.notices.append(f"Stale image removed from active review state: {name}")
            elif record != new_manifest[name]:
                self.notices.append(f"Image changed (size or mtime): review statuses reset for {name}")
            elif same_root and same_classes and same_labels and validated.get(name):
                self._statuses[name] = validated[name].copy()
                if signatures.get(name):
                    self._signatures[name] = signatures[name].copy()
        for name in sorted(new_manifest.keys() - old_manifest.keys()):
            self.notices.append(f"New dataset image starts pending: {name}")
        changed = old != self._dataset
        if self._labels_dir is not None:
            changed = self._refresh_confirmations() or changed
        return changed

    def _payload(self, statuses: dict[str, dict[int, str]], signatures: dict[str, dict[int, str]]) -> dict[str, Any]:
        return {
            "version": 2,
            "dataset": self._dataset,
            "statuses": {
                name: {str(cls): status for cls, status in sorted(entries.items())}
                for name, entries in sorted(statuses.items())
                if entries
            },
            "label_signatures": {
                name: {str(cls): signature for cls, signature in sorted(entries.items())}
                for name, entries in sorted(signatures.items())
                if entries
            },
        }

    def _label_signatures(self, image_name: str) -> tuple[dict[int, str], set[int]]:
        from PIL import Image

        if self._labels_dir is None:
            raise LabelError(self.path, "no labels directory was configured")
        image_path = self._images[image_name]
        with Image.open(image_path) as image:
            width, height = image.size
        label = label_path_for(image_path, self._labels_dir)
        boxes = read_yolo_boxes(label, width, height, len(self._classes))
        rows: dict[int, list[list[str]]] = {cls: [] for cls in range(len(self._classes))}
        present: set[int] = set()
        for box in boxes:
            present.add(box.cls)
            # Exact normalized ratios of the parsed geometry ignore row order/formatting.
            coordinates = (box.x1, box.y1, box.x2, box.y2)
            dimensions = (width, height, width, height)
            rows[box.cls].append(
                [str(Fraction(value) / dimension) for value, dimension in zip(coordinates, dimensions)]
            )
        signatures = {
            cls: hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode("ascii")).hexdigest()
            for cls, values in rows.items()
        }
        return signatures, present

    def _refresh_confirmations(self) -> bool:
        changed = False
        for name, entries in self._statuses.items():
            confirmed = {cls for cls, status in entries.items() if status in ("positive", "negative")}
            if not confirmed:
                continue
            try:
                current, present = self._label_signatures(name)
                reason = "class label contents changed or a prior signature is missing"
            except (LabelError, OSError, ValueError) as exc:
                current, present = {}, set()
                reason = f"confirmed label is missing or invalid ({exc})"
            old = self._signatures.get(name, {})
            for cls in sorted(confirmed):
                expected_presence = entries[cls] == "positive"
                if current.get(cls) != old.get(cls) or cls not in current or (cls in present) != expected_presence:
                    entries[cls] = "edited"
                    old.pop(cls, None)
                    self.notices.append(f"{name}, class {cls}: {reason}; explicit confirmation invalidated to edited.")
                    changed = True
        return changed

    def _persist(self, statuses: dict[str, dict[int, str]], signatures: dict[str, dict[int, str]]) -> None:
        contents = _json_bytes(self.path, self._payload(statuses, signatures))
        with _path_lock(self.path):
            try:
                current = self.path.read_bytes()
            except FileNotFoundError:
                current = None
            if current != self._expected_progress:
                raise LabelError(
                    self.path,
                    "progress changed on disk since loading; reopen it before saving to avoid losing another session's work",
                )
            _replace_bytes(self.path, contents, current)
            self._expected_progress = contents

    def _validate_target(self, image_name: str, cls: int) -> None:
        if type(image_name) is not str or image_name not in self._names:
            raise LabelError(self.path, f"unknown dataset image {image_name!r}")
        _class_id(self.path, cls, len(self._classes))

    def status(self, image_name: str, cls: int) -> str:
        self._validate_target(image_name, cls)
        return self._statuses.get(image_name, {}).get(cls, "pending")

    @property
    def reviewed(self) -> dict[str, set[int]]:
        return {
            name: confirmed
            for name, entries in self._statuses.items()
            if (confirmed := {cls for cls, status in entries.items() if status in ("positive", "negative")})
        }

    def _commit(self, image_name: str, entries: dict[int, str], signatures: dict[int, str]) -> None:
        if self._read_only:
            raise LabelError(self.path, "read-only review state cannot be modified")
        if entries == self._statuses.get(image_name, {}) and signatures == self._signatures.get(image_name, {}):
            return
        changed = {**self._statuses, image_name: entries}
        changed_signatures = {**self._signatures, image_name: signatures}
        self._persist(changed, changed_signatures)
        self._statuses = changed
        self._signatures = changed_signatures

    def mark_edited(self, image_name: str, class_ids: Iterable[int]) -> None:
        if type(image_name) is not str or image_name not in self._names:
            raise LabelError(self.path, f"unknown dataset image {image_name!r}")
        entries = self._statuses.get(image_name, {}).copy()
        signatures = self._signatures.get(image_name, {}).copy()
        for cls in class_ids:
            self._validate_target(image_name, cls)
            entries[cls] = "edited"
            signatures.pop(cls, None)
        self._commit(image_name, entries, signatures)

    def confirm(self, image_name: str, cls: int, has_boxes: bool) -> None:
        self._validate_target(image_name, cls)
        if type(has_boxes) is not bool:
            raise LabelError(self.path, "has_boxes must be a bool from an explicit confirmation")
        entries = self._statuses.get(image_name, {}).copy()
        entries[cls] = "positive" if has_boxes else "negative"
        signatures = self._signatures.get(image_name, {}).copy()
        if self._labels_dir is not None:
            try:
                current, present = self._label_signatures(image_name)
            except (LabelError, OSError, ValueError) as exc:
                raise LabelError(
                    self.path,
                    f"save a valid label before confirming {image_name!r} (including an empty label for a negative): {exc}",
                ) from exc
            if (cls in present) != has_boxes:
                raise LabelError(
                    self.path,
                    f"confirmation does not match saved boxes for {image_name!r}, class {cls}; save labels first",
                )
            signatures[cls] = current[cls]
        self._commit(image_name, entries, signatures)
