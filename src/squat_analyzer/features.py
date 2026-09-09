"""Biomechanical feature engineering from pose keypoints."""
from __future__ import annotations

import numpy as np

NUM_FEATURES_EXPECTED = 19
CONF_THRESHOLD = 0.723

(
    NOSE,
    L_EYE,
    R_EYE,
    L_EAR,
    R_EAR,
    L_SHOULDER,
    R_SHOULDER,
    L_ELBOW,
    R_ELBOW,
    L_WRIST,
    R_WRIST,
    L_HIP,
    R_HIP,
    L_KNEE,
    R_KNEE,
    L_ANKLE,
    R_ANKLE,
    L_FOOT,
    R_FOOT,
) = range(19)

KEYPOINT_NAMES = {
    0: "nose",
    1: "l_eye",
    2: "r_eye",
    3: "l_ear",
    4: "r_ear",
    5: "l_shoulder",
    6: "r_shoulder",
    7: "l_elbow",
    8: "r_elbow",
    9: "l_wrist",
    10: "r_wrist",
    11: "l_hip",
    12: "r_hip",
    13: "l_knee",
    14: "r_knee",
    15: "l_ankle",
    16: "r_ankle",
    17: "l_foot",
    18: "r_foot",
}

SKELETON_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (15, 17),
    (16, 18),
    (11, 12),
]


def _midpoint(
    p1: np.ndarray | None, p2: np.ndarray | None
) -> np.ndarray | None:
    if p1 is None and p2 is None:
        return None
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    return (p1 + p2) / 2.0


def get_kpt_coords(
    idx: int, kpts_array: np.ndarray, confidence_threshold: float = CONF_THRESHOLD
):
    """Return (x, y) if confidence is above threshold, else None."""
    if kpts_array is None or idx >= kpts_array.shape[0]:
        return None
    if kpts_array[idx, 2] > confidence_threshold:
        return kpts_array[idx, :2]
    return None


def calculate_angle(
    p1: np.ndarray | None,
    p2: np.ndarray | None,
    p3: np.ndarray | None,
) -> float:
    """Angle P1-P2-P3 in degrees. Returns 0.0 if any point is missing."""
    if p1 is None or p2 is None or p3 is None:
        return 0.0
    v1 = p1 - p2
    v2 = p3 - p2
    mag1 = float(np.linalg.norm(v1))
    mag2 = float(np.linalg.norm(v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = float(np.dot(v1, v2) / (mag1 * mag2))
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def build_keypoint_dict(
    kpts_xyc: np.ndarray, confidence_threshold: float = CONF_THRESHOLD
) -> dict[str, np.ndarray | None]:
    """Map named keypoints to (x, y) coordinates or None."""
    return {
        name: get_kpt_coords(idx, kpts_xyc, confidence_threshold)
        for idx, name in KEYPOINT_NAMES.items()
    }


def extract_keypoint_features(
    keypoints_xyc: np.ndarray, img_width: int, img_height: int
) -> np.ndarray:
    """Extract 19 biomechanical features from a single person's keypoints."""
    if keypoints_xyc is None or keypoints_xyc.shape[0] < 19:
        return np.zeros(NUM_FEATURES_EXPECTED)

    k = build_keypoint_dict(keypoints_xyc)
    features: list[float] = []

    # 1-6: Lower-body joint angles
    features.append(calculate_angle(k["l_hip"], k["l_knee"], k["l_ankle"]))
    features.append(calculate_angle(k["r_hip"], k["r_knee"], k["r_ankle"]))
    features.append(calculate_angle(k["l_shoulder"], k["l_hip"], k["l_knee"]))
    features.append(calculate_angle(k["r_shoulder"], k["r_hip"], k["r_knee"]))
    features.append(calculate_angle(k["l_knee"], k["l_ankle"], k["l_foot"]))
    features.append(calculate_angle(k["r_knee"], k["r_ankle"], k["r_foot"]))

    # 7: Trunk angle relative to vertical
    mid_shoulder = _midpoint(k["l_shoulder"], k["r_shoulder"])
    mid_hip = _midpoint(k["l_hip"], k["r_hip"])
    mid_ankle = _midpoint(k["l_ankle"], k["r_ankle"])
    mid_knee = _midpoint(k["l_knee"], k["r_knee"])

    if mid_shoulder is not None and mid_hip is not None:
        trunk_vec = mid_shoulder - mid_hip
        features.append(float(np.degrees(np.arctan2(trunk_vec[0], trunk_vec[1]))))
    else:
        features.append(0.0)

    # 8-9: Femur coronal proxies (knee valgus)
    l_knee_vref = (
        np.array([k["l_knee"][0], k["l_knee"][1] + 100])
        if k["l_knee"] is not None
        else None
    )
    r_knee_vref = (
        np.array([k["r_knee"][0], k["r_knee"][1] + 100])
        if k["r_knee"] is not None
        else None
    )
    features.append(calculate_angle(k["l_hip"], k["l_knee"], l_knee_vref))
    features.append(calculate_angle(k["r_hip"], k["r_knee"], r_knee_vref))

    # 10: Torso horizontal lean ratio
    if mid_shoulder is not None and mid_hip is not None:
        horiz = mid_shoulder[0] - mid_hip[0]
        vert = abs(mid_hip[1] - mid_shoulder[1])
        features.append(horiz / vert if vert > 10 else 0.0)
    else:
        features.append(0.0)

    # 11: Hip-ankle horizontal offset ratio
    if mid_hip is not None and mid_ankle is not None:
        horiz = mid_hip[0] - mid_ankle[0]
        vert = abs(mid_hip[1] - mid_ankle[1])
        features.append(horiz / vert if vert > 10 else 0.0)
    else:
        features.append(0.0)

    # 12: Ankle-foot vertical distance ratio
    if all(
        k[n] is not None
        for n in ["l_ankle", "r_ankle", "l_foot", "r_foot", "l_hip", "r_hip"]
    ):
        avg_ankle_y = (k["l_ankle"][1] + k["r_ankle"][1]) / 2
        avg_foot_y = (k["l_foot"][1] + k["r_foot"][1]) / 2
        vert_norm = abs(mid_hip[1] - mid_ankle[1])
        features.append(
            (avg_ankle_y - avg_foot_y) / vert_norm if vert_norm > 10 else 0.0
        )
    else:
        features.append(0.0)

    # 13: Foot stability (spread vs hip width)
    if (
        k["l_foot"] is not None
        and k["r_foot"] is not None
        and k["l_hip"] is not None
        and k["r_hip"] is not None
    ):
        foot_dist = abs(k["l_foot"][0] - k["r_foot"][0])
        hip_dist = abs(k["l_hip"][0] - k["r_hip"][0])
        features.append(foot_dist / hip_dist if hip_dist > 10 else 0.0)
    else:
        features.append(0.0)

    # 14-15: Foot orientation (vertical reference)
    vref_l = (
        np.array([k["l_ankle"][0], k["l_ankle"][1] + 100])
        if k["l_ankle"] is not None
        else None
    )
    vref_r = (
        np.array([k["r_ankle"][0], k["r_ankle"][1] + 100])
        if k["r_ankle"] is not None
        else None
    )
    features.append(calculate_angle(vref_l, k["l_ankle"], k["l_foot"]))
    features.append(calculate_angle(vref_r, k["r_ankle"], k["r_foot"]))

    # 16: Hip-knee horizontal alignment ratio
    if mid_hip is not None and mid_knee is not None:
        horiz = mid_hip[0] - mid_knee[0]
        vert = abs(mid_hip[1] - mid_knee[1])
        features.append(horiz / vert if vert > 10 else 0.0)
    else:
        features.append(0.0)

    # 17-18: Foot inclination (horizontal reference)
    href_l = (
        np.array([k["l_ankle"][0] + 100, k["l_ankle"][1]])
        if k["l_ankle"] is not None
        else None
    )
    href_r = (
        np.array([k["r_ankle"][0] + 100, k["r_ankle"][1]])
        if k["r_ankle"] is not None
        else None
    )
    features.append(calculate_angle(href_l, k["l_ankle"], k["l_foot"]))
    features.append(calculate_angle(href_r, k["r_ankle"], k["r_foot"]))

    # 19: Knee-ankle horizontal offset ratio
    if mid_knee is not None and mid_ankle is not None:
        horiz = mid_knee[0] - mid_ankle[0]
        vert = abs(mid_knee[1] - mid_ankle[1])
        features.append(horiz / vert if vert > 10 else 0.0)
    else:
        features.append(0.0)

    # Pad / truncate
    while len(features) < NUM_FEATURES_EXPECTED:
        features.append(0.0)
    return np.array(features[:NUM_FEATURES_EXPECTED], dtype=np.float64)
