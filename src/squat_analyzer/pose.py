"""Pose estimation wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

POSE_MODEL_PATH = Path("./runs/pose/pose19_experiment/weights/best.pt")
CONF_THRESHOLD = 0.723


def load_pose_model(model_path: str | Path | None = None) -> YOLO:
    if model_path is None:
        model_path = POSE_MODEL_PATH
    return YOLO(str(model_path))


def best_person_keypoints(result: Any) -> Any | None:
    """Return keypoints tensor for the highest-confidence person, or None."""
    if not result.boxes:
        return None
    best_idx = max(
        range(len(result.boxes)), key=lambda i: result.boxes[i].conf.item()
    )
    if result.keypoints and len(result.keypoints.data) > best_idx:
        return result.keypoints.data[best_idx]
    return None
