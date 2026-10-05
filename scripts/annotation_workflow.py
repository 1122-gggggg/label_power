"""Read-only dataset inspection and an all-or-nothing verified YOLO export."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from annotation_session import DatasetConfig
from annotation_store import LabelError, ReviewStore, collect_images, label_path_for, read_classes, read_yolo_boxes


def reference_crops(
    config: DatasetConfig, class_id: int, limit: int = 6
) -> tuple[list[tuple[str, Image.Image]], list[str]]:
    classes = read_classes(config.classes_file)
    images = collect_images(config.images_dir)
    store = ReviewStore(config.pass_state, images, classes, read_only=True, labels_dir=config.labels_dir)
    crops = []
    errors = []
    for path in images:
        if store.status(path.name, class_id) != "positive":
            continue
        try:
            with Image.open(path) as source:
                boxes = read_yolo_boxes(label_path_for(path, config.labels_dir), *source.size, len(classes))
                for box in boxes:
                    if box.cls != class_id:
                        continue
                    pad = max(8, int(max(box.width, box.height) * 0.15))
                    crop = source.crop(
                        (
                            max(0, int(box.x1) - pad),
                            max(0, int(box.y1) - pad),
                            min(source.width, int(box.x2) + pad),
                            min(source.height, int(box.y2) + pad),
                        )
                    ).convert("RGB")
                    crop.thumbnail((280, 220), Image.Resampling.LANCZOS)
                    crops.append((path.name, crop))
                    if len(crops) >= limit:
                        return crops, errors
        except (OSError, LabelError) as exc:
            errors.append(str(exc))
    return crops, errors


def export_verified(config: DatasetConfig, destination: Path) -> dict[str, int]:
    """Export only when every image/class has explicit, content-consistent review."""
    destination = destination.resolve()
    if destination.exists():
        raise LabelError(destination, "Export destination already exists; choose a new directory.")
    if (
        destination == config.images_dir
        or config.images_dir in destination.parents
        or destination == config.labels_dir
        or config.labels_dir in destination.parents
    ):
        raise LabelError(destination, "Export must be outside the source image and label directories.")
    classes = read_classes(config.classes_file)
    images = collect_images(config.images_dir)
    if not images:
        raise LabelError(config.images_dir, "Cannot export an empty image dataset.")
    store = ReviewStore(config.pass_state, images, classes, read_only=True, labels_dir=config.labels_dir)
    snapshots = []
    box_count = 0
    for path in images:
        label = label_path_for(path, config.labels_dir)
        with Image.open(path) as image:
            boxes = read_yolo_boxes(label, *image.size, len(classes))
            image.verify()
        present = {box.cls for box in boxes}
        for cls in range(len(classes)):
            state = store.status(path.name, cls)
            if state not in {"positive", "negative"}:
                raise LabelError(path, f"class {cls} ({classes[cls]}) is not explicitly confirmed ({state}).")
            if (state == "positive") != (cls in present):
                raise LabelError(path, f"class {cls} review contradicts the label; review again.")
        snapshots.append((path, label, path.stat(), label.stat()))
        box_count += len(boxes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        (stage / "images" / "all").mkdir(parents=True)
        (stage / "labels" / "all").mkdir(parents=True)
        for path, label, image_stat, label_stat in snapshots:
            shutil.copy2(path, stage / "images" / "all" / path.name)
            shutil.copy2(label, stage / "labels" / "all" / label.name)
            for source, before in ((path, image_stat), (label, label_stat)):
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise LabelError(source, "Source changed during export")
        (stage / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
        (stage / "data.yaml").write_text(
            "# Verified annotation export; create disjoint train/val splits before training.\n"
            "train: images/all\n" + "names: " + json.dumps(classes, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (stage / "review.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "images": len(images),
                    "boxes": box_count,
                    "classes": classes,
                    "source": str(config.images_dir),
                    "review": {
                        image.name: {str(cls): store.status(image.name, cls) for cls in range(len(classes))}
                        for image in images
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            raise LabelError(destination, "Export destination appeared during export")
        os.rename(stage, destination)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return {"images": len(images), "boxes": box_count, "classes": len(classes)}
