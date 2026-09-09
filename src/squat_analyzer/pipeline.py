"""End-to-end squat analysis pipeline."""
from __future__ import annotations

import os
from typing import Sequence

import cv2
import numpy as np

from squat_analyzer.features import (
    NUM_FEATURES_EXPECTED,
    extract_keypoint_features,
)
from squat_analyzer.pose import load_pose_model, best_person_keypoints, CONF_THRESHOLD
from squat_analyzer.classifier import (
    load_classifier,
    load_class_names,
    load_scaler,
)
from squat_analyzer.annotator import draw_skeleton, overlay_tags, save_frame


def analyze_image(
    image_path: str,
    output_path: str,
    models_dir: str = "models",
) -> list[str]:
    """Analyze a single image and write the annotated output."""
    pose = load_pose_model()
    clf = load_classifier(models_dir)
    names = load_class_names(models_dir)
    scaler = load_scaler(models_dir)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    result = pose(img, conf=CONF_THRESHOLD, verbose=False)[0]
    kpts = best_person_keypoints(result)
    tags: list[str] = []

    if kpts is not None:
        feats = extract_keypoint_features(
            kpts.cpu().numpy(), img.shape[1], img.shape[0]
        )
        scaled = scaler.transform(feats.reshape(1, -1))
        preds = clf.predict(scaled)[0]
        tags = [names[i] for i, v in enumerate(preds) if v == 1]
        draw_skeleton(img, kpts.cpu().numpy())

    overlay_tags(img, tags or ["No issues detected"])
    save_frame(img, output_path)
    return tags


def analyze_video(
    video_path: str,
    output_path: str,
    models_dir: str = "models",
) -> list[str]:
    """Analyze every frame of a video and write the annotated output."""
    pose = load_pose_model()
    clf = load_classifier(models_dir)
    names = load_class_names(models_dir)
    scaler = load_scaler(models_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    d = os.path.dirname(output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    last_tags: list[str] = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        result = pose(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        kpts = best_person_keypoints(result)
        tags: list[str] = []

        if kpts is not None:
            feats = extract_keypoint_features(kpts.cpu().numpy(), w, h)
            scaled = scaler.transform(feats.reshape(1, -1))
            preds = clf.predict(scaled)[0]
            tags = [names[i] for i, v in enumerate(preds) if v == 1]
            draw_skeleton(frame, kpts.cpu().numpy())
            last_tags = tags

        overlay_tags(frame, last_tags or ["No issues detected"])
        out.write(frame)

    cap.release()
    out.release()
    return last_tags
