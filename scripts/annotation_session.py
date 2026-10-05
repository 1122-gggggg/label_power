"""Dataset-specific navigation state and portable dataset profiles."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from annotation_store import LabelError, atomic_write_json


@dataclass(frozen=True)
class DatasetConfig:
    images_dir: Path
    labels_dir: Path
    classes_file: Path
    pass_state: Path

    @classmethod
    def from_profile(cls, path: Path) -> "DatasetConfig":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LabelError(path, f"Cannot read dataset profile: {exc}") from exc
        if not isinstance(payload, dict) or type(payload.get("version")) is not int or payload["version"] != 1:
            raise LabelError(path, "Unsupported dataset profile")
        resolved = {}
        for key in ("images_dir", "labels_dir", "classes_file"):
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                raise LabelError(path, f"Missing {key} in dataset profile")
            item = Path(value).expanduser()
            resolved[key] = (path.parent / item).resolve() if not item.is_absolute() else item.resolve()
        state = payload.get("pass_state")
        if state is not None and (not isinstance(state, str) or not state.strip()):
            raise LabelError(path, "Invalid pass_state in dataset profile")
        state_path = Path(state).expanduser() if state else resolved["labels_dir"] / ".annotation" / "review.json"
        if not state_path.is_absolute():
            state_path = path.parent / state_path
        return cls(**resolved, pass_state=state_path.resolve())

    def save_profile(self, path: Path) -> None:
        base = path.resolve().parent

        def relative(value: Path) -> str:
            try:
                return os.path.relpath(value.resolve(), base)
            except ValueError:
                return str(value.resolve())

        atomic_write_json(
            path,
            {
                "version": 1,
                "images_dir": relative(self.images_dir),
                "labels_dir": relative(self.labels_dir),
                "classes_file": relative(self.classes_file),
                "pass_state": relative(self.pass_state),
            },
        )


class Session:
    def __init__(self, config: DatasetConfig, class_names: list[str]) -> None:
        self.config = config
        self.class_names = class_names
        self.path = config.pass_state.with_name(config.pass_state.stem + ".session.json")
        self.notice = ""

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.notice = f"Session could not be restored; original file retained: {exc}"
            return {}
        if not isinstance(payload, dict) or type(payload.get("version")) is not int or payload["version"] != 1:
            self.notice = "Unsupported session; original file retained."
            return {}
        if (
            payload.get("images_dir") != str(self.config.images_dir)
            or payload.get("labels_dir") != str(self.config.labels_dir)
            or payload.get("class_names") != self.class_names
        ):
            self.notice = "Dataset or class order changed; navigation session was not reused."
            return {}
        if (
            not isinstance(payload.get("image"), str)
            or type(payload.get("class_id")) is not int
            or not 0 <= payload["class_id"] < len(self.class_names)
        ):
            self.notice = "Invalid navigation state; original file retained."
            return {}
        if any(key in payload and type(payload[key]) is not bool for key in ("image_mode", "lens")):
            self.notice = "Invalid session mode flags; original file retained."
            return {}
        view = payload.get("view")
        if view is not None and (
            not isinstance(view, dict)
            or not all(
                type(view.get(key)) in (int, float) and math.isfinite(view[key])
                for key in ("zoom", "center_x", "center_y")
            )
        ):
            self.notice = "Invalid session view; original file retained."
            return {}
        return payload

    def save(self, **state) -> None:
        atomic_write_json(
            self.path,
            {
                "version": 1,
                "images_dir": str(self.config.images_dir),
                "labels_dir": str(self.config.labels_dir),
                "class_names": self.class_names,
                **state,
            },
        )
