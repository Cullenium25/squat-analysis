"""Train the multi-label squat-form classifier from extracted pose features."""
from __future__ import annotations

import argparse
import os

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from squat_analyzer.features import NUM_FEATURES_EXPECTED, CONF_THRESHOLD
from squat_analyzer.pose import load_pose_model, best_person_keypoints

GROUND_TRUTH_BASE_PATH = os.path.join("configs", "coco_annotations")


def _load_split(
    split_name: str, pose_model, all_tags: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    csv_path = os.path.join(GROUND_TRUTH_BASE_PATH, split_name, "image_tags_ground_truth.csv")
    image_dir = os.path.join(split_name, "images")
    if not os.path.exists(csv_path) or not os.path.exists(image_dir):
        return np.empty((0, NUM_FEATURES_EXPECTED)), np.empty((0, len(all_tags))), []

    df = pd.read_csv(csv_path)
    tag_cols = [c for c in df.columns if c != "image_filename"]
    global_map = {t: i for i, t in enumerate(sorted(all_tags))}

    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    fnames: list[str] = []

    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, str(row["image_filename"]))
        if not os.path.exists(img_path):
            continue
        result = pose_model(img_path, conf=CONF_THRESHOLD, verbose=False)
        kpts = best_person_keypoints(result[0]) if result else None
        if kpts is None:
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        from squat_analyzer.features import extract_keypoint_features
        feats = extract_keypoint_features(kpts.cpu().numpy(), img.shape[1], img.shape[0])
        if feats is None or len(feats) != NUM_FEATURES_EXPECTED:
            continue
        rows.append(feats)
        vec = np.zeros(len(all_tags), dtype=int)
        for tag in tag_cols:
            if row[tag] == 1 and tag in global_map:
                vec[global_map[tag]] = 1
        labels.append(vec)
        fnames.append(str(row["image_filename"]))

    if not rows:
        return np.empty((0, NUM_FEATURES_EXPECTED)), np.empty((0, len(all_tags))), []
    return np.vstack(rows), np.vstack(labels), fnames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args(argv)

    pose = load_pose_model()

    all_tags: set[str] = set()
    for split in ("train", "valid", "test"):
        csv = os.path.join(GROUND_TRUTH_BASE_PATH, split, "image_tags_ground_truth.csv")
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            all_tags.update(c for c in df.columns if c != "image_filename")
    tag_list = sorted(all_tags)
    if not tag_list:
        print("ERROR: no tags found; check coco_annotations path")
        return 1

    X_tr, y_tr, _ = _load_split("train", pose, tag_list)
    X_val, y_val, _ = _load_split("valid", pose, tag_list)
    if X_tr.shape[0] == 0 or X_val.shape[0] == 0:
        print("ERROR: need train + valid splits with images and annotations")
        return 1

    scaler = StandardScaler().fit(np.vstack([X_tr, X_val]))
    X_s = scaler.transform(np.vstack([X_tr, X_val]))
    y_all = np.vstack([y_tr, y_val])

    clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    clf.fit(X_s, y_all)

    os.makedirs(args.models_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(args.models_dir, "squat_classifier_balanced_rfc.joblib"))
    joblib.dump(tag_list, os.path.join(args.models_dir, "squat_classifier_balanced_rfc_class_names.joblib"))
    joblib.dump(scaler, os.path.join(args.models_dir, "squat_classifier_scaler.joblib"))

    y_pred = clf.predict(X_s)
    print(f"Accuracy: {(y_all == y_pred).mean():.3f}")
    print(f"F1 (micro): {f1_score(y_all, y_pred, average='micro', zero_division=0):.3f}")
    print(f"Saved artifacts to {args.models_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
