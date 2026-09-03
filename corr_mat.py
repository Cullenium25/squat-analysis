import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
from ultralytics import YOLO
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm 

model_checkpoint_path = "./runs/pose/pose19_experiment/weights/best.pt"
CONF_THRESHOLD = 0.723 # Adjusted based on evaluation results
GROUND_TRUTH_BASE_PATH = os.path.join("configs", "coco_annotations")
NUM_FEATURES_EXPECTED = 19

def get_kpt_coords(idx, kpts_array, confidence_threshold):
    """Safely gets keypoint coordinates if confident."""
    if kpts_array is None or idx >= kpts_array.shape[0]:
        return None
    if kpts_array[idx, 2] > confidence_threshold:
        return kpts_array[idx, :2]
    return None

# --- Feature Engineering Function ---
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

    # --- Forward Trunk: Torso_Horizontal_Lean_Ratio ---
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
    
    # --- Forward Trunk: Hip_Ankle_Horizontal_Offset_Ratio ---
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

    # Ankle_Foot_Vertical_Distance_Ratio (for Heels-Up) ---
    if kpts_dict['l_ankle'] is not None and kpts_dict['r_ankle'] is not None and \
       kpts_dict['l_foot'] is not None and kpts_dict['r_foot'] is not None and \
       mid_hip is not None and mid_ankle is not None: # Re-use mid_hip/mid_ankle for normalization
        
        avg_ankle_y = (kpts_dict['l_ankle'][1] + kpts_dict['r_ankle'][1]) / 2
        avg_foot_y = (kpts_dict['l_foot'][1] + kpts_dict['r_foot'][1]) / 2
        
        vertical_ankle_foot_dist = avg_ankle_y - avg_foot_y # Ankle is usually above foot (smaller y)
        
        # Normalize by the vertical height of the lower body to make it scale-invariant
        vertical_lower_body_height_norm = abs(mid_hip[1] - mid_ankle[1]) # From hip to ankle
        
        if vertical_lower_body_height_norm > 10: # Avoid division by zero
            ankle_foot_ratio = vertical_ankle_foot_dist / vertical_lower_body_height_norm
            features.append(ankle_foot_ratio)
        else:
            features.append(0.0)
    else:
        features.append(0.0)

    # Foot_Stability_Horizontal_Spread_Ratio ---
    if kpts_dict['l_foot'] is not None and kpts_dict['r_foot'] is not None and \
       kpts_dict['l_hip'] is not None and kpts_dict['r_hip'] is not None:
        
        horizontal_foot_dist = abs(kpts_dict['l_foot'][0] - kpts_dict['r_foot'][0])
        horizontal_hip_dist = abs(kpts_dict['l_hip'][0] - kpts_dict['r_hip'][0])
        
        if horizontal_hip_dist > 10: # Avoid division by zero
            foot_spread_ratio = horizontal_foot_dist / horizontal_hip_dist
            features.append(foot_spread_ratio)
        else:
            features.append(0.0)
    else:
        features.append(0.0)

    # Left Foot Orientation Angle
    if kpts_dict['l_ankle'] is not None and kpts_dict['l_foot'] is not None:
        vertical_ref_l = np.array([kpts_dict['l_ankle'][0], kpts_dict['l_ankle'][1] + 100])
        features.append(calculate_angle(vertical_ref_l, kpts_dict['l_ankle'], kpts_dict['l_foot']))
    else:
        features.append(0.0)

    # Right Foot Orientation Angle
    if kpts_dict['r_ankle'] is not None and kpts_dict['r_foot'] is not None:
        vertical_ref_r = np.array([kpts_dict['r_ankle'][0], kpts_dict['r_ankle'][1] + 100])
        features.append(calculate_angle(vertical_ref_r, kpts_dict['r_ankle'], kpts_dict['r_foot']))
    else:
        features.append(0.0)

    # Hip_Knee_Horizontal_Alignment_Ratio
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

    # Left_Foot_Inclination_Angle (for Heels-Up) ---
    if kpts_dict['l_ankle'] is not None and kpts_dict['l_foot'] is not None:
        horizontal_ref_l = np.array([kpts_dict['l_ankle'][0] + 100, kpts_dict['l_ankle'][1]])
        features.append(calculate_angle(horizontal_ref_l, kpts_dict['l_ankle'], kpts_dict['l_foot']))
    else:
        features.append(0.0)

    # Right_Foot_Inclination_Angle (for Heels-Up) ---
    if kpts_dict['r_ankle'] is not None and kpts_dict['r_foot'] is not None:
        horizontal_ref_r = np.array([kpts_dict['r_ankle'][0] + 100, kpts_dict['r_ankle'][1]])
        features.append(calculate_angle(horizontal_ref_r, kpts_dict['r_ankle'], kpts_dict['r_foot']))
    else:
        features.append(0.0)

    # Knee_Ankle_Horizontal_Offset_Ratio (for Knee Travel) ---
    # Measures how far horizontally the knees are from the ankles, normalized by shin length.
    if mid_knee is not None and mid_ankle is not None:
        knee_ankle_horizontal_offset = mid_knee[0] - mid_ankle[0] # X-coordinate difference
        vertical_knee_ankle_distance = abs(mid_knee[1] - mid_ankle[1]) # Absolute Y-difference for length

        if vertical_knee_ankle_distance > 10: # Avoid division by zero or very small numbers
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

def load_features_and_labels_for_split(split_name, pose_model, all_unique_tags):
    """
    Loads image data and ground truth labels for a specified split,
    extracting pose features.
    """
    split_image_data = []
    split_ground_truth_labels = []
    split_image_filenames = []

    csv_path = os.path.join(GROUND_TRUTH_BASE_PATH, split_name, "image_tags_ground_truth.csv")
    image_folder_path = os.path.join(split_name, "images") 

    if not os.path.exists(csv_path):
        print(f"WARNING: CSV file not found for {split_name} split at: {csv_path}. Skipping.")
        return split_image_data, split_ground_truth_labels, split_image_filenames

    if not os.path.exists(image_folder_path):
        print(f"WARNING: Image folder not found for {split_name} split at: {image_folder_path}. Skipping.")
        return split_image_data, split_ground_truth_labels, split_image_filenames

    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} entries from {csv_path}")

        # Filter out 'image_filename' column only
        current_tags_in_csv = [col for col in df.columns if col != 'image_filename']
        
        # Ensure all unique tags (including those from other splits) are considered when building labels
        global_tag_mapping = {tag: i for i, tag in enumerate(sorted(list(all_unique_tags)))}
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name} images for pose features"):
            image_filename = row['image_filename']
            full_image_path = os.path.join(image_folder_path, image_filename)
            
            if not os.path.exists(full_image_path):
                continue
            
            # Extract features
            results = pose_model(full_image_path, conf=CONF_THRESHOLD, verbose=False) 
            features_for_image = None
            
            if results and len(results[0].keypoints) > 0: 
                best_box_idx = -1
                max_box_conf = -1.0
                if results[0].boxes:
                    for b_idx, box in enumerate(results[0].boxes):
                        if box.conf.item() > max_box_conf:
                            max_box_conf = box.conf.item()
                            best_box_idx = b_idx
                
                if best_box_idx != -1:
                    keypoints_xyc_data = results[0].keypoints.data[best_box_idx].cpu().numpy() 
                    img = cv2.imread(full_image_path)
                    if img is None:
                        continue
                    img_height, img_width, _ = img.shape
                    features_for_image = extract_keypoint_features(keypoints_xyc_data, img_width, img_height)

            if features_for_image is not None and len(features_for_image) == NUM_FEATURES_EXPECTED:
                # Create a binary label vector for this image based on ALL unique tags
                binary_label_vector = np.zeros(len(all_unique_tags))
                for tag_col in current_tags_in_csv:
                    if row[tag_col] == 1:
                        if tag_col in global_tag_mapping:
                            binary_label_vector[global_tag_mapping[tag_col]] = 1

                split_image_data.append(features_for_image)
                split_ground_truth_labels.append(binary_label_vector) 
                split_image_filenames.append(image_filename)

    except pd.errors.EmptyDataError:
        print(f"WARNING: CSV file for {split_name} is empty: {csv_path}. Skipping.")
    except Exception as e:
        print(f"ERROR processing {split_name} split or its images/CSVs: {e}")

    return split_image_data, split_ground_truth_labels, split_image_filenames

def display_test_examples(pose_model, test_image_filenames, y_test, y_pred, class_names, image_base_paths_for_display, num_examples=5):
    print(f"\n--- Displaying {num_examples} Random Test Examples ---")
    if len(test_image_filenames) == 0:
        print("No test images available to display.")
        return
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (7, 9),
        (6, 8), (8, 10), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14),
        (14, 16), (15, 17), (16, 18), (11, 12)
    ]
    KEYPOINT_COLOR = (0, 0, 255) # Blue
    LINE_COLOR = (255, 0, 0) # Red
    RADIUS = 5
    THICKNESS = 2
    
    num_examples = min(num_examples, len(test_image_filenames))
    if num_examples == 0:
        print("Not enough test images to display examples.")
        return

    random_indices = random.sample(range(len(test_image_filenames)), num_examples)
    plt.figure(figsize=(15, 6 * num_examples))

    for i, idx in enumerate(random_indices):
        img_filename = test_image_filenames[idx]
        true_labels_binary = y_test[idx]
        pred_labels_binary = y_pred[idx]
        
        true_tags = [class_names[j] for j, val in enumerate(true_labels_binary) if val == 1]
        pred_tags = [class_names[j] for j, val in enumerate(pred_labels_binary) if val == 1]
        
        image_path = os.path.join(image_base_paths_for_display['test'], img_filename)
        
        if not os.path.exists(image_path):
            continue
        
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                continue
            
            results = pose_model(img_bgr, conf=CONF_THRESHOLD, verbose=False)
            
            keypoints_xyc_to_draw = None
            best_box_idx = -1
            max_box_conf = -1.0
            if results and results[0].boxes:
                for b_idx, box in enumerate(results[0].boxes):
                    if box.conf.item() > max_box_conf:
                        max_box_conf = box.conf.item()
                        best_box_idx = b_idx
            
            if best_box_idx != -1 and results[0].keypoints is not None and len(results[0].keypoints.data) > best_box_idx:
                keypoints_xyc_to_draw = results[0].keypoints.data[best_box_idx].cpu().numpy()
            
            display_conf_threshold = CONF_THRESHOLD

            if keypoints_xyc_to_draw is not None:
                for kpt_idx, (x, y, conf) in enumerate(keypoints_xyc_to_draw):
                    if conf > display_conf_threshold:
                        cv2.circle(img_bgr, (int(x), int(y)), RADIUS, KEYPOINT_COLOR, -1)
                for connection in SKELETON_CONNECTIONS:
                    p1_idx, p2_idx = connection
                    p1_coords = get_kpt_coords(p1_idx, keypoints_xyc_to_draw, display_conf_threshold) 
                    p2_coords = get_kpt_coords(p2_idx, keypoints_xyc_to_draw, display_conf_threshold)
                    if p1_coords is not None and p2_coords is not None:
                        cv2.line(img_bgr, (int(p1_coords[0]), int(p1_coords[1])), 
                                 (int(p2_coords[0]), int(p2_coords[1])), LINE_COLOR, THICKNESS)
            else:
                pass 

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            base_filename = img_filename.split('.rf.')[0] + os.path.splitext(img_filename)[1]
            title_text = f"Image: {base_filename}\nGT: {', '.join(true_tags) if true_tags else 'None'}\nPred: {', '.join(pred_tags) if pred_tags else 'None'}"
            
            plt.subplot(num_examples, 1, i + 1)
            plt.imshow(img_rgb)
            plt.title(title_text, fontsize=10)
            plt.axis('off')
        except Exception as e:
            print(f"Error displaying image {img_filename} or drawing keypoints: {e}")
            import traceback
            traceback.print_exc()
    plt.tight_layout()
    plt.show()

def plot_feature_correlation_matrix(X_data, feature_names):
    """
    Generates and displays a correlation matrix heatmap for the given features.

    Args:
        X_data (np.array or pd.DataFrame): The feature data.
        feature_names (list): A list of strings, the names of the features,
                              in the same order as columns in X_data.
    """
    if X_data.shape[1] != len(feature_names):
        print("WARNING: Number of features in data does not match the number of provided feature names.")
        print("Expected:", len(feature_names), "Got:", X_data.shape[1])
        print("Correlation matrix labels might be incorrect.")

    df_features = pd.DataFrame(X_data, columns=feature_names)
    correlation_matrix = df_features.corr()

    plt.figure(figsize=(18, 16)) 
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8})
    
    plt.title('Feature Correlation Matrix', fontsize=20)
    plt.xticks(rotation=90, ha='right', fontsize=12) 
    plt.yticks(rotation=0, va='center', fontsize=12) 
    plt.tight_layout() 
    plt.show()
    print("Feature correlation matrix displayed.")



if __name__ == "__main__":
    print("--- Starting Feature Correlation Matrix Generation with REAL DATA ---")

    # --- Load Pose Model ---
    try:
        pose_model = YOLO(model_checkpoint_path)
        print(f"Successfully loaded pose estimation model from: {model_checkpoint_path}")
    except Exception as e:
        print(f"ERROR: Could not load pose estimation model. Please check model_checkpoint_path: {e}")
        exit() 

    # --- Collect ALL unique tags from all CSVs first (to ensure consistent label mapping) ---
    all_unique_tags = set()
    for split_name in ['train', 'valid', 'test']: # Check all splits to get a comprehensive list of tags
        csv_path = os.path.join(GROUND_TRUTH_BASE_PATH, split_name, "image_tags_ground_truth.csv")
        if os.path.exists(csv_path):
            try:
                df_temp = pd.read_csv(csv_path)
                current_tags_in_csv = [col for col in df_temp.columns if col != 'image_filename']
                all_unique_tags.update(current_tags_in_csv)
            except pd.errors.EmptyDataError:
                print(f"WARNING: CSV file for {split_name} is empty: {csv_path}. Skipping tag collection for this split.")
            except Exception as e:
                print(f"ERROR collecting tags from {split_name} CSV: {e}")
    
    if not all_unique_tags:
        print("ERROR: No unique tags found across any CSV files. Cannot proceed with feature loading.")
        exit() 

    # --- Load Features from Training and Validation Data ---
    print("\n--- Loading Features from Train and Valid Splits ---")
    X_train_list, _, _ = [], [], []
    for split_name in ['train', 'valid']:
        x_s, _, _ = load_features_and_labels_for_split(split_name, pose_model, all_unique_tags)
        X_train_list.extend(x_s)

    X_data_for_plotting = np.array(X_train_list)
    print(f"Loaded {len(X_data_for_plotting)} samples for correlation analysis.")

    if X_data_for_plotting.shape[0] == 0:
        print("ERROR: No valid data samples loaded for correlation matrix. Cannot plot.")
        exit()

    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_data_for_plotting_scaled = scaler.fit_transform(X_data_for_plotting)
    print("Features scaled using StandardScaler.")

    # --- Feature Names ---
    feature_names = [
        "Left_Knee_Angle", "Right_Knee_Angle",
        "Left_Hip_Angle", "Right_Hip_Angle",
        "Left_Ankle_Angle", "Right_Ankle_Angle",
        "Trunk_Angle",
        "Left_Femur_Q_Angle", "Right_Femur_Q_Angle",
        "Torso_Horizontal_Lean_Ratio", "Hip_Ankle_Horizontal_Offset_Ratio",
        "Ankle_Foot_Vertical_Distance_Ratio", "Foot_Stability_Horizontal_Spread_Ratio",
        "Left_Foot_Orientation_Angle", "Right_Foot_Orientation_Angle",
        "Hip_Knee_Horizontal_Alignment_Ratio",
        "Left_Foot_Inclination_Angle", "Right_Foot_Inclination_Angle",
        "Knee_Ankle_Horizontal_Offset_Ratio"
    ]

    plot_feature_correlation_matrix(X_data_for_plotting_scaled, feature_names)

    print("\n--- Correlation Matrix Generation Complete ---")
