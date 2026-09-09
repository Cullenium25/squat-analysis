"""Model loading helpers."""
from __future__ import annotations

import os
from typing import Any

import joblib

DEFAULT_MODELS_DIR = "models"
CLASSIFIER_TYPE = "balanced_rfc"


def _path(models_dir: str, filename: str) -> str:
    return os.path.join(models_dir, filename)


def load_classifier(
    models_dir: str = DEFAULT_MODELS_DIR,
    classifier_type: str = CLASSIFIER_TYPE,
) -> Any:
    return joblib.load(
        _path(models_dir, f"squat_classifier_{classifier_type}.joblib")
    )


def load_class_names(
    models_dir: str = DEFAULT_MODELS_DIR,
    classifier_type: str = CLASSIFIER_TYPE,
) -> list[str]:
    return joblib.load(
        _path(models_dir, f"squat_classifier_{classifier_type}_class_names.joblib")
    )


def load_scaler(models_dir: str = DEFAULT_MODELS_DIR) -> Any:
    return joblib.load(_path(models_dir, "squat_classifier_scaler.joblib"))
