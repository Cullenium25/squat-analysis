"""Tests for squat_analyzer."""
from __future__ import annotations

import numpy as np

from squat_analyzer.features import (
    NUM_FEATURES_EXPECTED,
    calculate_angle,
    extract_keypoint_features,
)


def test_feature_vector_length():
    fake_kpts = np.zeros((19, 3), dtype=np.float64)
    features = extract_keypoint_features(fake_kpts, 640, 480)
    assert features.shape == (NUM_FEATURES_EXPECTED,)


def test_angle_basics():
    p1 = np.array([0.0, 1.0])
    p2 = np.array([0.0, 0.0])
    p3 = np.array([1.0, 0.0])
    angle = calculate_angle(p1, p2, p3)
    assert abs(angle - 90.0) < 1e-6


def test_angle_returns_zero_when_any_none():
    assert calculate_angle(None, np.array([0, 0]), np.array([1, 0])) == 0.0
    assert calculate_angle(np.array([0, 1]), None, np.array([1, 0])) == 0.0
    assert calculate_angle(np.array([0, 1]), np.array([0, 0]), None) == 0.0
