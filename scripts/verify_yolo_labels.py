#!/usr/bin/env python3
"""Read-only verification using the annotation platform's strict label parser."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from annotation_store import (
    LabelError,
    ReviewStore,
    collect_images,
    label_path_for,
    read_classes,
    read_yolo_boxes,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES_DIR = ROOT / "images_adaptive_master" / "all"
DEFAULT_LABELS_DIR = ROOT / "labels_adaptive" / "all"
DEFAULT_CLASSES_FILE = ROOT / "classes.txt"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify finite, in-bounds YOLO bounding-box labels.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--classes-file", type=Path, default=DEFAULT_CLASSES_FILE)
    parser.add_argument("--pass-state", type=Path, help="Review JSON (default: LABELS_DIR/.annotation/review.json)")
    parser.add_argument("--require-reviewed", action="store_true", help="Require explicit review of every image/class")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        classes = read_classes(args.classes_file)
        images = collect_images(args.images_dir)
        if not images:
            raise LabelError(args.images_dir, "no supported image files found; select a non-empty image directory")
    except (LabelError, OSError) as exc:
        print(f"error={exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    missing: list[str] = []
    bad: list[tuple[str, int | None, str]] = []
    empty: list[str] = []
    counts = dict.fromkeys(range(len(classes)), 0)
    for image_path in images:
        label = label_path_for(image_path, args.labels_dir)
        if not label.exists():
            missing.append(image_path.name)
            continue
        try:
            with Image.open(image_path) as image:
                width, height = image.size
            boxes = read_yolo_boxes(label, width, height, len(classes))
        except LabelError as exc:
            bad.append((label.name, exc.line, exc.reason))
            continue
        except (OSError, ValueError) as exc:
            bad.append((image_path.name, None, f"cannot read image/label: {exc}"))
            continue
        if not boxes:
            empty.append(image_path.name)
        for box in boxes:
            counts[box.cls] += 1

    print(f"images={len(images)}")
    print(f"images_dir={args.images_dir}")
    print(f"labels_dir={args.labels_dir}")
    print(f"missing_labels={len(missing)}")
    print(f"empty_labels={len(empty)}")
    print(f"bad_labels={len(bad)}")
    print("class_counts=" + repr(counts))
    if missing:
        print("missing_examples=" + repr(missing[:10]))
    if bad:
        print("bad_examples=" + repr(bad[:10]))

    unreviewed: list[tuple[str, int]] = []
    if (args.require_reviewed or args.pass_state is not None) and images:
        state_path = args.pass_state or args.labels_dir / ".annotation" / "review.json"
        try:
            review = ReviewStore(state_path, images, classes, read_only=True, labels_dir=args.labels_dir)
        except (LabelError, OSError) as exc:
            print(f"error={exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        for image in images:
            for cls in range(len(classes)):
                if review.status(image.name, cls) not in ("positive", "negative"):
                    unreviewed.append((image.name, cls))
        print(f"unreviewed={len(unreviewed)}")
        if unreviewed:
            print("unreviewed_examples=" + repr(unreviewed[:10]))
        if review.notices:
            print("review_notices=" + repr(review.notices))
    if missing or bad or (args.require_reviewed and unreviewed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
