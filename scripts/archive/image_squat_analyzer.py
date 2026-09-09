import os
import cv2
import numpy as np
import joblib
from ultralytics import YOLO
import matplotlib.pyplot as plt # For displaying the image

image = '****' # input image path to analyze

# --- Configuration (Consistent with classify_squat_form.py) ---
GROUND_TRUTH_BASE_PATH = os.path.join("configs", "coco_annotations") # Needed for load_features_and_labels_for_split if used, but not directly for single image analysis
model_checkpoint_path = "./runs/pose/pose19_experiment/weights/best.pt"
CONF_THRESHOLD = 0.723 # Adjusted based on evaluation results
NUM_FEATURES_EXPECTED = 19 

# --- Model Artifact Paths (Loading the balanced_rfc by default) ---
MODELS_DIR = "models"
CLASSIFIER_FILENAME_PREFIX = "squat_classifier"
CLASS_NAMES_FILENAME_SUFFIX = "_class_names"
SCALER_FILENAME = "squat_classifier_scaler.joblib"

# Specific paths for the classifier and class names to load
# We'll load the balanced_rfc model as it's often the best performing
LOAD_CLASSIFIER_TYPE = "balanced_rfc" 
TRAINED_CLASSIFIER_LOAD_PATH = os.path.join(MODELS_DIR, f"{CLASSIFIER_FILENAME_PREFIX}_{LOAD_CLASSIFIER_TYPE}.joblib")
TRAINED_CLASS_NAMES_LOAD_PATH = os.path.join(MODELS_DIR, f"{CLASSIFIER_FILENAME_PREFIX}_{LOAD_CLASSIFIER_TYPE}{CLASS_NAMES_FILENAME_SUFFIX}.joblib")
TRAINED_SCALER_LOAD_PATH = os.path.join(MODELS_DIR, SCALER_FILENAME)


# --- Helper Function for Keypoint Coordinate Retrieval (Copied from classify_squat_form.py) ---
def get_kpt_coords(idx, kpts_array, confidence_threshold=CONF_THRESHOLD):
    """Safely gets keypoint coordinates if confident."""
    if kpts_array is None or idx >= kpts_array.shape[0]:
        return None
    if kpts_array[idx, 2] > confidence_threshold:
        return kpts_array[idx, :2]
    return None

# --- Feature Engineering Function (Copied from classify_squat_form.py) ---
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
    if keypoints_xyc is None or keypoints_xyc.shape[0] < 19: 
        return np.zeros(NUM_FEATURES_EXPECTED) 

    # Define keypoint indices based on Person-19 schema
    NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
    L_FOOT, R_FOOT = 17, 18 

    kpts_dict = {
        'nose': get_kpt_coords(NOSE, keypoints_xyc),
        'l_shoulder': get_kpt_coords(L_SHOULDER, keypoints_xyc), 'r_shoulder': get_kpt_coords(R_SHOULDER, keypoints_xyc),
        'l_elbow': get_kpt_coords(L_ELBOW, keypoints_xyc), 'r_elbow': get_kpt_coords(R_ELBOW, keypoints_xyc),
        'l_wrist': get_kpt_coords(L_WRIST, keypoints_xyc), 'r_wrist': get_kpt_coords(R_WRIST, keypoints_xyc),
        'l_hip': get_kpt_coords(L_HIP, keypoints_xyc), 'r_hip': get_kpt_coords(R_HIP, keypoints_xyc),
        'l_knee': get_kpt_coords(L_KNEE, keypoints_xyc), 'r_knee': get_kpt_coords(R_KNEE, keypoints_xyc),
        'l_ankle': get_kpt_coords(L_ANKLE, keypoints_xyc), 'r_ankle': get_kpt_coords(R_ANKLE, keypoints_xyc),
        'l_foot': get_kpt_coords(L_FOOT, keypoints_xyc), 'r_foot': get_kpt_coords(R_FOOT, keypoints_xyc)
    }

    features = []
    # Lower Body Angles (Knees, Hips, Ankles)
    features.append(calculate_angle(kpts_dict['l_hip'], kpts_dict['l_knee'], kpts_dict['l_ankle']))
    features.append(calculate_angle(kpts_dict['r_hip'], kpts_dict['r_knee'], kpts_dict['r_ankle']))
    features.append(calculate_angle(kpts_dict['l_shoulder'], kpts_dict['l_hip'], kpts_dict['l_knee']))
    features.append(calculate_angle(kpts_dict['r_shoulder'], kpts_dict['r_hip'], kpts_dict['r_knee']))
    features.append(calculate_angle(kpts_dict['l_knee'], kpts_dict['l_ankle'], kpts_dict['l_foot']))
    features.append(calculate_angle(kpts_dict['r_knee'], kpts_dict['r_ankle'], kpts_dict['r_foot']))

    # Trunk Angle (Absolute angle relative to vertical)
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

    # --- Genu Valgus Proxies (2D Coronal Angles) ---
    # Left Femur Coronal Angle Proxy
    if kpts_dict['l_hip'] is not None and kpts_dict['l_knee'] is not None:
        l_knee_vertical_ref = np.array([kpts_dict['l_knee'][0], kpts_dict['l_knee'][1] + 100])
        features.append(calculate_angle(kpts_dict['l_hip'], kpts_dict['l_knee'], l_knee_vertical_ref))
    else:
        features.append(0.0)

    # Right Femur Coronal Angle Proxy
    if kpts_dict['r_hip'] is not None and kpts_dict['r_knee'] is not None:
        r_knee_vertical_ref = np.array([kpts_dict['r_knee'][0], kpts_dict['r_knee'][1] + 100])
        features.append(calculate_angle(kpts_dict['r_hip'], kpts_dict['r_knee'], r_knee_vertical_ref))
    else:
        features.append(0.0)

    # --- Torso_Horizontal_Lean_Ratio ---
    if mid_shoulder is not None and mid_hip is not None:
        horizontal_offset = mid_shoulder[0] - mid_hip[0]
        vertical_height = abs(mid_hip[1] - mid_shoulder[1])

        if vertical_height > 10: # Avoid division by zero
            torso_lean_ratio = horizontal_offset / vertical_height
            features.append(torso_lean_ratio)
        else:
            features.append(0.0)
    else:
        features.append(0.0)
    
    # --- Hip_Ankle_Horizontal_Offset_Ratio ---
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

    # --- Ankle_Foot_Vertical_Distance_Ratio (for Heels-Up) ---
    if kpts_dict['l_ankle'] is not None and kpts_dict['r_ankle'] is not None and \
       kpts_dict['l_foot'] is not None and kpts_dict['r_foot'] is not None and \
       mid_hip is not None and mid_ankle is not None: # Re-use mid_hip/mid_ankle for normalization
        
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

    # --- Foot_Stability_Horizontal_Spread_Ratio ---
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

    # --- Left_Foot_Orientation_Angle & Right_Foot_Orientation_Angle ---
    if kpts_dict['l_ankle'] is not None and kpts_dict['l_foot'] is not None:
        vertical_ref_l = np.array([kpts_dict['l_ankle'][0], kpts_dict['l_ankle'][1] + 100])
        features.append(calculate_angle(vertical_ref_l, kpts_dict['l_ankle'], kpts_dict['l_foot']))
    else:
        features.append(0.0)

    if kpts_dict['r_ankle'] is not None and kpts_dict['r_foot'] is not None:
        vertical_ref_r = np.array([kpts_dict['r_ankle'][0], kpts_dict['r_ankle'][1] + 100])
        features.append(calculate_angle(vertical_ref_r, kpts_dict['r_ankle'], kpts_dict['r_foot']))
    else:
        features.append(0.0)

    # --- Hip_Knee_Horizontal_Alignment_Ratio ---
    mid_knee = None
    if kpts_dict['l_knee'] is not None and kpts_dict['r_knee'] is not None:
        mid_knee = (kpts_dict['l_knee'] + kpts_dict['r_knee']) / 2
    elif kpts_dict['l_knee'] is not None:
        mid_knee = kpts_dict['l_knee']
    elif kpts_dict['r_knee'] is not None:
        mid_knee = kpts_dict['r_knee']

    if mid_hip is not None and mid_knee is not None:
        hip_knee_horizontal_offset = mid_hip[0] - mid_knee[0]
        vertical_hip_knee_distance = abs(mid_hip[1] - mid_knee[1])

        if vertical_hip_knee_distance > 10:
            hip_knee_alignment_ratio = hip_knee_horizontal_offset / vertical_hip_knee_distance
            features.append(hip_knee_alignment_ratio)
        else:
            features.append(0.0)
    else:
        features.append(0.0)

    # --- Left_Foot_Inclination_Angle (for Heels-Up) ---
    if kpts_dict['l_ankle'] is not None and kpts_dict['l_foot'] is not None:
        horizontal_ref_l = np.array([kpts_dict['l_ankle'][0] + 100, kpts_dict['l_ankle'][1]])
        features.append(calculate_angle(horizontal_ref_l, kpts_dict['l_ankle'], kpts_dict['l_foot']))
    else:
        features.append(0.0)

    # --- Right_Foot_Inclination_Angle (for Heels-Up) ---
    if kpts_dict['r_ankle'] is not None and kpts_dict['r_foot'] is not None:
        horizontal_ref_r = np.array([kpts_dict['r_ankle'][0] + 100, kpts_dict['r_ankle'][1]])
        features.append(calculate_angle(horizontal_ref_r, kpts_dict['r_ankle'], kpts_dict['r_foot']))
    else:
        features.append(0.0)

    # --- Knee_Ankle_Horizontal_Offset_Ratio (for Knee Travel) ---
    if mid_knee is not None and mid_ankle is not None:
        knee_ankle_horizontal_offset = mid_knee[0] - mid_ankle[0]
        vertical_knee_ankle_distance = abs(mid_knee[1] - mid_ankle[1])

        if vertical_knee_ankle_distance > 10:
            knee_ankle_offset_ratio = knee_ankle_horizontal_offset / vertical_knee_ankle_distance
            features.append(knee_ankle_offset_ratio)
        else:
            features.append(0.0)
    else:
        features.append(0.0)


    # Pad or truncate features to ensure consistent length.
    while len(features) < NUM_FEATURES_EXPECTED:
        features.append(0.0)
    
    return np.array(features[:NUM_FEATURES_EXPECTED])


# --- Main Image Analysis Function ---
def analyze_image_squat_form(image_path, output_image_name="output_analyzed_image.jpg"):
    print(f"--- Starting Image Analysis for: {image_path} ---")

    # Load models
    try:
        pose_model = YOLO(model_checkpoint_path)
        print(f"Successfully loaded pose estimation model from: {model_checkpoint_path}")
    except Exception as e:
        print(f"ERROR: Could not load pose estimation model. Please check POSE_MODEL_PATH: {e}")
        return

    try:
        classifier = joblib.load(TRAINED_CLASSIFIER_LOAD_PATH)
        class_names = joblib.load(TRAINED_CLASS_NAMES_LOAD_PATH)
        scaler = joblib.load(TRAINED_SCALER_LOAD_PATH)
        print(f"Successfully loaded classifier ({LOAD_CLASSIFIER_TYPE}), class names, and scaler.")
        print(f"Loaded class names: {class_names}")
    except Exception as e:
        print(f"ERROR: Could not load classifier, class names, or scaler. Please check paths: {e}")
        return

    # Read the image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"ERROR: Could not read image file: {image_path}. Please check path.")
        return

    img_height, img_width, _ = img_bgr.shape
    img_copy = img_bgr.copy() # Create a copy to draw on

    # 1. Pose Estimation
    results = pose_model(img_copy, conf=CONF_THRESHOLD, verbose=False)
    keypoints_xyc_to_draw = None
    features_for_classifier = None
    predicted_tags = []

    if results and len(results[0].boxes) > 0:
        best_box_idx = -1
        max_box_conf = -1.0
        # Find the person with the highest detection confidence
        for b_idx, box in enumerate(results[0].boxes):
            if box.conf.item() > max_box_conf:
                max_box_conf = box.conf.item()
                best_box_idx = b_idx
        
        if best_box_idx != -1 and results[0].keypoints is not None and len(results[0].keypoints.data) > best_box_idx:
            keypoints_xyc_data = results[0].keypoints.data[best_box_idx].cpu().numpy()

            # 2. Extract Features
            features_for_classifier = extract_keypoint_features(keypoints_xyc_data, img_width, img_height)

            # 3. Classify (only if features are valid length)
            if features_for_classifier is not None and len(features_for_classifier) == NUM_FEATURES_EXPECTED:
                try:
                    # Reshape for single prediction and APPLY SCALER
                    features_scaled = scaler.transform(features_for_classifier.reshape(1, -1))
                    predictions = classifier.predict(features_scaled)[0]
                    # Get active tags
                    predicted_tags = [class_names[i] for i, pred in enumerate(predictions) if pred == 1]
                except Exception as e:
                    print(f"Error during classification: {e}")
                    predicted_tags = ["CLASSIF. ERROR"]
            else:
                print("WARNING: No valid pose features extracted for classification.")

            # 4. Draw Keypoints and Skeleton (using CONF_THRESHOLD for drawing)
            SKELETON_CONNECTIONS = [
                (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (7, 9),
                (6, 8), (8, 10), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14),
                (14, 16), (15, 17), (16, 18), (11, 12)
            ]
            KEYPOINT_COLOR = (0, 255, 0) # Green
            LINE_COLOR = (255, 0, 0) # Red
            RADIUS = 5
            THICKNESS = 2

            for kpt_idx, (x, y, conf) in enumerate(keypoints_xyc_data):
                if conf > CONF_THRESHOLD:
                    cv2.circle(img_copy, (int(x), int(y)), RADIUS, KEYPOINT_COLOR, -1)
            
            for connection in SKELETON_CONNECTIONS:
                p1_idx, p2_idx = connection
                p1_coords = get_kpt_coords(p1_idx, keypoints_xyc_data, CONF_THRESHOLD)
                p2_coords = get_kpt_coords(p2_idx, keypoints_xyc_data, CONF_THRESHOLD)
                if p1_coords is not None and p2_coords is not None:
                    cv2.line(img_copy, (int(p1_coords[0]), int(p1_coords[1])),
                             (int(p2_coords[0]), int(p2_coords[1])), LINE_COLOR, THICKNESS)
    else:
        print("No person detected in the image.")

    # 5. Overlay Predicted Tags
    TEXT_COLOR = (255, 255, 255) # White
    TEXT_BACKGROUND_COLOR = (0, 0, 0) # Black
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    text_line_height = 30
    y_offset = 30

    if predicted_tags:
        predicted_tags_sorted = sorted(predicted_tags) 
        for j, tag in enumerate(predicted_tags_sorted):
            text_to_display = f"Pred: {tag}"
            (text_width, text_height), baseline = cv2.getTextSize(text_to_display, FONT, FONT_SCALE, FONT_THICKNESS)
            cv2.rectangle(img_copy, (10, y_offset + j * text_line_height), 
                          (10 + text_width, y_offset + j * text_line_height - text_height - baseline), 
                          TEXT_BACKGROUND_COLOR, -1)
            cv2.putText(img_copy, text_to_display, (10, y_offset + j * text_line_height),
                        FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    else:
        text_to_display = "Pred: No tags detected"
        (text_width, text_height), baseline = cv2.getTextSize(text_to_display, FONT, FONT_SCALE, FONT_THICKNESS)
        cv2.rectangle(img_copy, (10, 30), (10 + text_width, 30 - text_height - baseline), TEXT_BACKGROUND_COLOR, -1)
        cv2.putText(img_copy, text_to_display, (10, 30),
                    FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    # Save the output image
    output_dir = os.path.dirname(output_image_name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cv2.imwrite(output_image_name, img_copy)
    print(f"Analyzed image saved to: {os.path.abspath(output_image_name)}")

    # Display the image using Matplotlib (optional, for interactive environments)
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title(f"Analyzed Squat Form: {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()

    print("--- Image Analysis Complete ---")


if __name__ == "__main__":
    # Example Usage:
    # Ensure you have a sample image in a known path.
    # For demonstration, I'll use a placeholder. Replace with your actual image.
    sample_input_image = image
    output_image_file = "runs/analyzed_images/analyzed_sample_squat.jpg"

    # Ensure the output directory exists
    if not os.path.exists(os.path.dirname(output_image_file)):
        os.makedirs(os.path.dirname(output_image_file))
    
    analyze_image_squat_form(sample_input_image, output_image_file)
