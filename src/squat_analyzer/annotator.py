"""Video / image annotation: skeleton + feedback overlay."""
from __future__ import annotations

import os
from typing import Sequence

import cv2
import numpy as np

from squat_analyzer.features import (
    SKELETON_CONNECTIONS,
    get_kpt_coords,
    CONF_THRESHOLD,
)

KEYPOINT_COLOR = (0, 255, 0)
LINE_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0)
RADIUS = 5
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2


def draw_skeleton(frame: np.ndarray, kpts_xyc: np.ndarray) -> None:
    """Draw keypoints and skeleton connections on the frame in-place."""
    for x, y, conf in kpts_xyc:
        if conf > CONF_THRESHOLD:
            cv2.circle(frame, (int(x), int(y)), RADIUS, KEYPOINT_COLOR, -1)
    for p1, p2 in SKELETON_CONNECTIONS:
        a = get_kpt_coords(p1, kpts_xyc, CONF_THRESHOLD)
        b = get_kpt_coords(p2, kpts_xyc, CONF_THRESHOLD)
        if a is not None and b is not None:
            cv2.line(
                frame,
                (int(a[0]), int(a[1])),
                (int(b[0]), int(b[1])),
                LINE_COLOR,
                THICKNESS,
            )


def overlay_tags(frame: np.ndarray, tags: Sequence[str]) -> None:
    """Overlay predicted tags in the top-left corner."""
    y_offset = 30
    line_h = 30
    for j, tag in enumerate(sorted(tags)):
        label = f"Pred: {tag}"
        (tw, th), baseline = cv2.getTextSize(
            label, FONT, FONT_SCALE, FONT_THICKNESS
        )
        cv2.rectangle(
            frame,
            (10, y_offset + j * line_h),
            (10 + tw, y_offset + j * line_h - th - baseline),
            TEXT_BG,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (10, y_offset + j * line_h),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )


def save_frame(frame: np.ndarray, path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    cv2.imwrite(path, frame)
