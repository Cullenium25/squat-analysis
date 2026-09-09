import cv2
import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for the web client

# --- Configuration ---
# Path to your YOLOv8 pose model checkpoint
POSE_MODEL_PATH = "./runs/pose/pose19_experiment/weights/best.pt"
# Confidence threshold for pose detection (keypoints and bounding boxes)
CONF_THRESHOLD = 0.723
# Number of features expected from extract_keypoint_features
NUM_FEATURES_EXPECTED = 19
# Base directory for all models
MODELS_DIR = "models"
# Specific classifier type to load (e.g., 'balanced_rfc', 'rfc', 'logistic')
LOAD_CLASSIFIER_TYPE = "balanced_rfc"

# Construct full paths for loading
TRAINED_CLASSIFIER_LOAD_PATH = os.path.join(MODELS_DIR, f"squat_classifier_{LOAD_CLASSIFIER_TYPE}.joblib")
TRAINED_CLASS_NAMES_LOAD_PATH = os.path.join(MODELS_DIR, f"squat_classifier_{LOAD_CLASSIFIER_TYPE}_class_names.joblib")
TRAINED_SCALER_LOAD_PATH = os.path.join(MODELS_DIR, "squat_classifier_scaler.joblib")

# --- Global Variables for Models ---
pose_model = None
classifier = None
class_names = None
scaler = None

# --- Helper Function for Keypoint Coordinate Retrieval ---
def get_kpt_coords(idx, kpts_array, confidence_threshold):
    """Safely gets keypoint coordinates (x, y) if the confidence score for that keypoint is above the specified threshold."""
    if kpts_array is None or idx >= len(kpts_array): return None
    if kpts_array[idx, 2] > confidence_threshold: return kpts_array[idx, :2]
    return None

# --- Feature Engineering Functions ---
def calculate_angle(p1, p2, p3):
    """Calculates angle between three points (P1-P2-P3)."""
    if p1 is None or p2 is None or p3 is None: return 0.0
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    if magnitude_v1 == 0 or magnitude_v2 == 0: return 0.0
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle_rad)

def extract_keypoint_features(keypoints_xyc, img_width, img_height):
    """Extracts numerical features from YOLOv8 keypoint predictions."""
    if keypoints_xyc is None or len(keypoints_xyc) < 19:
        return np.zeros(NUM_FEATURES_EXPECTED)

    NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
    L_FOOT, R_FOOT = 17, 18

    kpts_dict = {
        'nose': get_kpt_coords(NOSE, keypoints_xyc, CONF_THRESHOLD),
        'l_shoulder': get_kpt_coords(L_SHOULDER, keypoints_xyc, CONF_THRESHOLD), 'r_shoulder': get_kpt_coords(R_SHOULDER, keypoints_xyc, CONF_THRESHOLD),
        'l_elbow': get_kpt_coords(L_ELBOW, keypoints_xyc, CONF_THRESHOLD), 'r_elbow': get_kpt_coords(R_ELBOW, keypoints_xyc, CONF_THRESHOLD),
        'l_wrist': get_kpt_coords(L_WRIST, keypoints_xyc, CONF_THRESHOLD), 'r_wrist': get_kpt_coords(R_WRIST, keypoints_xyc, CONF_THRESHOLD),
        'l_hip': get_kpt_coords(L_HIP, keypoints_xyc, CONF_THRESHOLD), 'r_hip': get_kpt_coords(R_HIP, keypoints_xyc, CONF_THRESHOLD),
        'l_knee': get_kpt_coords(L_KNEE, keypoints_xyc, CONF_THRESHOLD), 'r_knee': get_kpt_coords(R_KNEE, keypoints_xyc, CONF_THRESHOLD),
        'l_ankle': get_kpt_coords(L_ANKLE, keypoints_xyc, CONF_THRESHOLD), 'r_ankle': get_kpt_coords(R_ANKLE, keypoints_xyc, CONF_THRESHOLD),
        'l_foot': get_kpt_coords(L_FOOT, keypoints_xyc, CONF_THRESHOLD), 'r_foot': get_kpt_coords(R_FOOT, keypoints_xyc, CONF_THRESHOLD)
    }

    features = []
    # 1. Left knee angle
    features.append(calculate_angle(kpts_dict['l_hip'], kpts_dict['l_knee'], kpts_dict['l_ankle']))
    # 2. Right knee angle
    features.append(calculate_angle(kpts_dict['r_hip'], kpts_dict['r_knee'], kpts_dict['r_ankle']))
    # 3. Left hip angle
    features.append(calculate_angle(kpts_dict['l_shoulder'], kpts_dict['l_hip'], kpts_dict['l_knee']))
    # 4. Right hip angle
    features.append(calculate_angle(kpts_dict['r_shoulder'], kpts_dict['r_hip'], kpts_dict['r_knee']))
    # 5. Left ankle angle
    features.append(calculate_angle(kpts_dict['l_knee'], kpts_dict['l_ankle'], kpts_dict['l_foot']))
    # 6. Right ankle angle
    features.append(calculate_angle(kpts_dict['r_knee'], kpts_dict['r_ankle'], kpts_dict['r_foot']))

    # 7. Torso angle
    mid_shoulder = None
    mid_hip = None
    if kpts_dict['l_shoulder'] is not None and kpts_dict['r_shoulder'] is not None:
        mid_shoulder = (kpts_dict['l_shoulder'] + kpts_dict['r_shoulder']) / 2
    elif kpts_dict['l_shoulder'] is not None: mid_shoulder = kpts_dict['l_shoulder']
    elif kpts_dict['r_shoulder'] is not None: mid_shoulder = kpts_dict['r_shoulder']
    if kpts_dict['l_hip'] is not None and kpts_dict['r_hip'] is not None:
        mid_hip = (kpts_dict['l_hip'] + kpts_dict['r_hip']) / 2
    elif kpts_dict['l_hip'] is not None: mid_hip = kpts_dict['l_hip']
    elif kpts_dict['r_hip'] is not None: mid_hip = kpts_dict['r_hip']

    if mid_shoulder is not None and mid_hip is not None:
        trunk_vector = mid_shoulder - mid_hip
        trunk_angle = np.degrees(np.arctan2(trunk_vector[0], trunk_vector[1]))
        features.append(trunk_angle)
    else:
        features.append(0.0)

    # 8. Left knee-vertical angle
    if kpts_dict['l_hip'] is not None and kpts_dict['l_knee'] is not None:
        l_knee_vertical_ref = np.array([kpts_dict['l_knee'][0], kpts_dict['l_knee'][1] + 100])
        features.append(calculate_angle(kpts_dict['l_hip'], kpts_dict['l_knee'], l_knee_vertical_ref))
    else:
        features.append(0.0)

    # 9. Right knee-vertical angle
    if kpts_dict['r_hip'] is not None and kpts_dict['r_knee'] is not None:
        r_knee_vertical_ref = np.array([kpts_dict['r_knee'][0], kpts_dict['r_knee'][1] + 100])
        features.append(calculate_angle(kpts_dict['r_hip'], kpts_dict['r_knee'], r_knee_vertical_ref))
    else:
        features.append(0.0)

    # 10. Torso lean ratio
    if mid_shoulder is not None and mid_hip is not None:
        horizontal_offset = mid_shoulder[0] - mid_hip[0]
        vertical_height = abs(mid_hip[1] - mid_shoulder[1])
        if vertical_height > 10:
            torso_lean_ratio = horizontal_offset / vertical_height
            features.append(torso_lean_ratio)
        else:
            features.append(0.0)
    else:
        features.append(0.0)
    
    # 11. Hip-to-ankle offset ratio
    mid_ankle = None
    if kpts_dict['l_ankle'] is not None and kpts_dict['r_ankle'] is not None:
        mid_ankle = (kpts_dict['l_ankle'] + kpts_dict['r_ankle']) / 2
    elif kpts_dict['l_ankle'] is not None:
        mid_ankle = kpts_dict['l_ankle']
    elif kpts_dict['r_ankle'] is not None:
        mid_ankle = kpts_dict['r_ankle']

    if mid_hip is not None and mid_ankle is not None:
        hip_ankle_horizontal_offset = mid_hip[0] - mid_ankle[0]
        vertical_lower_body_height = abs(mid_hip[1] - mid_ankle[1])
        if vertical_lower_body_height > 10:
            hip_ankle_offset_ratio = hip_ankle_horizontal_offset / vertical_lower_body_height
            features.append(hip_ankle_offset_ratio)
        else:
            features.append(0.0)
    else:
        features.append(0.0)

    # 12. Ankle-foot vertical ratio
    if kpts_dict['l_ankle'] is not None and kpts_dict['r_ankle'] is not None and \
       kpts_dict['l_foot'] is not None and kpts_dict['r_foot'] is not None and \
       mid_hip is not None and mid_ankle is not None:
        avg_ankle_y = (kpts_dict['l_ankle'][1] + kpts_dict['r_ankle'][1]) / 2
        avg_foot_y = (kpts_dict['l_foot'][1] + kpts_dict['r_foot'][1]) / 2
        vertical_ankle_foot_dist = avg_ankle_y - avg_foot_y
        vertical_lower_body_height_norm = abs(mid_hip[1] - mid_ankle[1])
        if vertical_lower_body_height_norm > 10:
            ankle_foot_ratio = vertical_ankle_foot_dist / vertical_lower_body_height_norm
            features.append(ankle_foot_ratio)
        else:
            features.append(0.0)
    else:
        features.append(0.0)

    # 13. Foot spread ratio
    if kpts_dict['l_foot'] is not None and kpts_dict['r_foot'] is not None and \
       kpts_dict['l_hip'] is not None and kpts_dict['r_hip'] is not None:
        horizontal_foot_dist = abs(kpts_dict['l_foot'][0] - kpts_dict['r_foot'][0])
        horizontal_hip_dist = abs(kpts_dict['l_hip'][0] - kpts_dict['r_hip'][0])
        if horizontal_hip_dist > 10:
            foot_spread_ratio = horizontal_foot_dist / horizontal_hip_dist
            features.append(foot_spread_ratio)
        else:
            features.append(0.0)
    else:
        features.append(0.0)
        
    # 14. Left ankle-foot vertical angle
    if kpts_dict['l_ankle'] is not None and kpts_dict['l_foot'] is not None:
        vertical_ref_l = np.array([kpts_dict['l_ankle'][0], kpts_dict['l_ankle'][1] + 100])
        features.append(calculate_angle(vertical_ref_l, kpts_dict['l_ankle'], kpts_dict['l_foot']))
    else:
        features.append(0.0)

    # 15. Right ankle-foot vertical angle
    if kpts_dict['r_ankle'] is not None and kpts_dict['r_foot'] is not None:
        vertical_ref_r = np.array([kpts_dict['r_ankle'][0], kpts_dict['r_ankle'][1] + 100])
        features.append(calculate_angle(vertical_ref_r, kpts_dict['r_ankle'], kpts_dict['r_foot']))
    else:
        features.append(0.0)

    # 16. Left elbow angle
    features.append(calculate_angle(kpts_dict['l_shoulder'], kpts_dict['l_elbow'], kpts_dict['l_wrist']))
    # 17. Right elbow angle
    features.append(calculate_angle(kpts_dict['r_shoulder'], kpts_dict['r_elbow'], kpts_dict['r_wrist']))
    # 18. Left shoulder angle
    features.append(calculate_angle(kpts_dict['l_elbow'], kpts_dict['l_shoulder'], kpts_dict['l_hip']))
    # 19. Right shoulder angle
    features.append(calculate_angle(kpts_dict['r_elbow'], kpts_dict['r_shoulder'], kpts_dict['r_hip']))
    
    # Ensure the feature vector is the correct size before returning
    if len(features) != NUM_FEATURES_EXPECTED:
        print(f"Warning: Extracted {len(features)} features, but expected {NUM_FEATURES_EXPECTED}.")
        # Pad with zeros or handle the error appropriately
        features.extend([0.0] * (NUM_FEATURES_EXPECTED - len(features)))
    
    return np.array(features)

# --- Model Loading on Server Startup ---
def load_models():
    """Loads all models into memory once at the start of the server."""
    global pose_model, classifier, class_names, scaler
    try:
        pose_model = YOLO(POSE_MODEL_PATH)
        print(f"Server: Successfully loaded pose estimation model.")
    except Exception as e:
        print(f"Server ERROR: Could not load pose model: {e}")
        return False
    try:
        classifier = joblib.load(TRAINED_CLASSIFIER_LOAD_PATH)
        class_names = joblib.load(TRAINED_CLASS_NAMES_LOAD_PATH)
        scaler = joblib.load(TRAINED_SCALER_LOAD_PATH)
        print(f"Server: Successfully loaded classifier ({LOAD_CLASSIFIER_TYPE}), class names, and scaler.")
    except Exception as e:
        print(f"Server ERROR: Could not load classifier, class names, or scaler: {e}")
        return False
    return True

# --- API Endpoint for Analysis ---
@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    """Receives a frame, analyzes it, and returns the predictions."""
    if pose_model is None or classifier is None or scaler is None:
        return jsonify({"error": "Models are not loaded. Server may have failed to start correctly."}), 500

    if 'frame' not in request.files:
        return jsonify({"error": "No frame part in the request"}), 400

    # Read the image from the request
    frame_file = request.files['frame']
    img_array = np.frombuffer(frame_file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    # 1. Pose Estimation
    results = pose_model(img, conf=CONF_THRESHOLD, verbose=False)
    predicted_tags = []
    keypoints_xyc_data_serializable = None

    if results and len(results[0].boxes) > 0:
        best_box_idx = -1
        max_box_conf = -1.0
        for b_idx, box in enumerate(results[0].boxes):
            if box.conf.item() > max_box_conf:
                max_box_conf = box.conf.item()
                best_box_idx = b_idx
        
        if best_box_idx != -1 and results[0].keypoints is not None and len(results[0].keypoints.data) > best_box_idx:
            keypoints_xyc_data = results[0].keypoints.data[best_box_idx].cpu().numpy()
            
            # 2. Extract Features
            features_for_classifier = extract_keypoint_features(keypoints_xyc_data, img.shape[1], img.shape[0])
            
            # 3. Classify (only if features are valid length)
            if features_for_classifier is not None and len(features_for_classifier) == NUM_FEATURES_EXPECTED:
                try:
                    features_scaled = scaler.transform(features_for_classifier.reshape(1, -1))
                    predictions = classifier.predict(features_scaled)[0]
                    predicted_tags = [class_names[i] for i, pred in enumerate(predictions) if pred == 1]
                except Exception as e:
                    print(f"Error during classification: {e}")
                    predicted_tags = ["CLASSIF. ERROR"]
            
            # Prepare keypoints for JSON response (convert numpy array to list)
            keypoints_xyc_data_serializable = keypoints_xyc_data.tolist()
    
    # 4. Return results as JSON
    response_data = {
        "predicted_tags": predicted_tags,
        "keypoints": keypoints_xyc_data_serializable
    }
    return jsonify(response_data)


if __name__ == '__main__':
    # It's better to load the models once at startup rather than on every request.
    if load_models():
        print("Server is ready to receive requests.")
        # Make sure to run the server on a port that is accessible on your local network.
        # '0.0.0.0' makes it accessible from any device on your network.
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
