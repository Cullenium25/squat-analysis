import cv2
import os
import random
import matplotlib.pyplot as plt
from ultralytics import YOLO

CONF_THRESHOLD = 0.723 # Adjusted based on evaluation results
model_checkpoint_path = "./runs/pose/pose19_experiment/weights/best.pt"
TEST_IMAGES_BASE_PATH = "test/images" 

PERSON_19_KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_foot", "right_foot"
]

PERSON_19_SKELETON_CONNECTIONS = [
    # Head and Shoulders
    [0, 1], [0, 2], [1, 3], [2, 4], # Nose to eyes, eyes to ears
    [5, 6], # Shoulders
    # Arms
    [5, 7], [7, 9], # Left arm
    [6, 8], [8, 10], # Right arm
    # Torso
    [5, 11], [6, 12], # Shoulder to hip
    [11, 12], # Hips
    # Legs
    [11, 13], [13, 15], # Left leg
    [12, 14], [14, 16], # Right leg
    # Feet
    [15, 17], # Left ankle to left foot
    [16, 18] # Right ankle to right foot
]

# --- Drawing Constants for Skeleton ---
KEYPOINT_COLOR = (0, 0, 255) # Red for keypoints (easily visible)
LINE_COLOR = (255, 0, 0) # Blue for skeleton lines (contrasting)
RADIUS = 5 # Radius for keypoint circles
THICKNESS = 2 # Thickness for lines and keypoint outlines

# --- Drawing Constants for Keypoint Labels ---
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.5 # Adjust for readability
LABEL_FONT_THICKNESS = 1
LABEL_COLOR = (0, 255, 0) # Green for labels (contrasting)
LABEL_OFFSET_X = 10 # Offset label from keypoint
LABEL_OFFSET_Y = -5 # Offset label from keypoint

def get_kpt_coords(idx, kpts_array, confidence_threshold):
    """
    Safely gets keypoint coordinates if confident.
    Returns (x,y) if confident, None otherwise.
    """
    if kpts_array is None or idx >= kpts_array.shape[0]:
        return None
    if kpts_array[idx, 2] > confidence_threshold:
        return kpts_array[idx, :2].astype(int) # Ensure integer coordinates for drawing
    return None

def test_custom_model_on_random_image():
    try:
        # 1. Load the trained YOLOv8 pose model
        model = YOLO(model_checkpoint_path)
        print("Custom 19-keypoint model loaded successfully!")

        print("Model metadata (keypoint names and skeleton connections) are defined for manual plotting.")

        # 2. Get a list of all image files in the test directory
        if not os.path.exists(TEST_IMAGES_BASE_PATH):
            print(f"Error: Test image folder not found at {TEST_IMAGES_BASE_PATH}")
            print("Please ensure your Roboflow dataset is downloaded and organized correctly.")
            return

        image_files = [f for f in os.listdir(TEST_IMAGES_BASE_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
        
        if not image_files:
            print(f"Error: No image files found in {TEST_IMAGES_BASE_PATH}. Please check the folder content.")
            return

        # 3. Select a random image file
        random_image_filename = random.choice(image_files)
        image_path = os.path.join(TEST_IMAGES_BASE_PATH, random_image_filename)

        print(f"\nTesting on random image: {image_path}")

        # Check if the image can be read by OpenCV
        img_bgr_original = cv2.imread(image_path)
        if img_bgr_original is None:
            print(f"Error: OpenCV could not read image from {image_path}. It might be corrupted or not a valid image file.")
            return
        print(f"Debug: Image read successfully by OpenCV. Shape: {img_bgr_original.shape}, Type: {img_bgr_original.dtype}")


        # 4. Run inference on the selected image
        # Using the original image array directly for inference
        results = model(img_bgr_original, verbose=False) 

        # 5. Process and display results for each detected object (person)
        for result in results: # 'results' is a list, but for single image, it's usually one result object
            num_persons_detected = len(result.boxes)
            best_person_confidence = 0.0
            num_keypoints_best_person = 0
            
            keypoints_data_for_metrics = None # For displaying confidences in text
            keypoints_xyc_to_draw = None # For actual drawing

            # Create a copy of the original image to draw on
            image_to_display_bgr = img_bgr_original.copy()

            if num_persons_detected > 0:
                best_box_idx = -1
                max_box_conf = -1.0
                for b_idx, box in enumerate(result.boxes):
                    if box.conf.item() > max_box_conf:
                        max_box_conf = box.conf.item()
                        best_box_idx = b_idx
                
                best_person_confidence = max_box_conf 

                if best_box_idx != -1 and result.keypoints is not None and len(result.keypoints.data) > best_box_idx:
                    keypoints_data_for_metrics = result.keypoints.data[best_box_idx].cpu().numpy()
                    keypoints_xyc_to_draw = keypoints_data_for_metrics # Use the same keypoints for drawing
                    num_keypoints_best_person = keypoints_data_for_metrics.shape[0]
                    
                    # --- Drawing Bounding Box ---
                    box_coords = result.boxes.xyxy[best_box_idx].cpu().numpy().astype(int)
                    cv2.rectangle(image_to_display_bgr, (box_coords[0], box_coords[1]), 
                                  (box_coords[2], box_coords[3]), (0, 255, 255), 2) # Yellow box for visibility
                    
                    # === MANUAL KEYPOINT, SKELETON, AND LABEL DRAWING ===
                    if keypoints_xyc_to_draw is not None:
                        # Draw keypoints (circles) and labels
                        for kpt_idx, (x, y, conf) in enumerate(keypoints_xyc_to_draw):
                            # Only draw keypoints if their confidence is above DRAWING_CONF_THRESHOLD
                            if conf > CONF_THRESHOLD: 
                                cv2.circle(image_to_display_bgr, (int(x), int(y)), RADIUS, KEYPOINT_COLOR, -1)
                                
                                # Draw keypoint label
                                kpt_name = PERSON_19_KPT_NAMES[kpt_idx] # Get the name
                                text_pos = (int(x) + LABEL_OFFSET_X, int(y) + LABEL_OFFSET_Y)
                                cv2.putText(image_to_display_bgr, kpt_name, text_pos, 
                                            LABEL_FONT, LABEL_FONT_SCALE, LABEL_COLOR, LABEL_FONT_THICKNESS, cv2.LINE_AA)
                        
                        # Draw skeleton lines
                        for connection in PERSON_19_SKELETON_CONNECTIONS: # Use your custom defined connections
                            p1_idx, p2_idx = connection
                            
                            # Get coordinates, but this time check against DRAWING_CONF_THRESHOLD
                            # to determine if the points are confident enough for drawing a line.
                            # IMPORTANT: get_kpt_coords uses the DRAWING_CONF_THRESHOLD from its parameter
                            p1_coords_draw = get_kpt_coords(p1_idx, keypoints_xyc_to_draw, CONF_THRESHOLD) 
                            p2_coords_draw = get_kpt_coords(p2_idx, keypoints_xyc_to_draw, CONF_THRESHOLD)
                            
                            if p1_coords_draw is not None and p2_coords_draw is not None:
                                cv2.line(image_to_display_bgr, (int(p1_coords_draw[0]), int(p1_coords_draw[1])), 
                                         (int(p2_coords_draw[0]), int(p2_coords_draw[1])), LINE_COLOR, THICKNESS)
                    else:
                        print("Debug: No confident keypoints found for the best person to draw skeleton.")
                    # ============================================

            # --- Prepare text for metrics display ---
            metrics_text = f"Detection Details for:\n{random_image_filename}\n\n"
            metrics_text += f"Persons Detected: {num_persons_detected}\n"
            if num_persons_detected > 0:
                metrics_text += f"Best Person Conf: {best_person_confidence:.2f}\n"
                metrics_text += f"Keypoints Found (Best Person): {num_keypoints_best_person}\n"
                
                metrics_text += "\nRelevant Keypoint Confidences:\n"
                # Updated to iterate PERSON_19_KPT_NAMES to match actual drawing
                for kpt_idx, kpt_name in enumerate(PERSON_19_KPT_NAMES):
                    if kpt_idx < num_keypoints_best_person:
                        confidence = keypoints_data_for_metrics[kpt_idx, 2]
                        # Only show if confidence is above the DRAWING threshold, for consistency
                        if confidence > CONF_THRESHOLD:
                            metrics_text += f"  {kpt_name}: {confidence:.2f}\n"
            else:
                metrics_text += "No confident person detection.\n"

            # Convert BGR image to RGB format for correct display with Matplotlib
            annotated_image_rgb = cv2.cvtColor(image_to_display_bgr, cv2.COLOR_BGR2RGB)
            
            print(f"Debug: Shape of annotated_image_rgb (before imshow): {annotated_image_rgb.shape}, Type: {annotated_image_rgb.dtype}")

            # Create a figure with two subplots (1 row, 2 columns) for side-by-side display
            fig, axes = plt.subplots(1, 2, figsize=(15, 8)) 

            # Plot the image in the first subplot
            axes[0].imshow(annotated_image_rgb)
            axes[0].set_title(f"Image: {os.path.basename(random_image_filename)}")
            axes[0].axis('off') 

            # Display metrics text in the second subplot
            axes[1].text(0.05, 0.95, metrics_text, 
                         verticalalignment='top', 
                         fontsize=10, 
                         fontfamily='monospace', 
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=1, alpha=0.9)) 
            axes[1].set_title("Detection Metrics")
            axes[1].axis('off') 

            plt.tight_layout() 
            plt.show() 

    except FileNotFoundError:
        print("Error: Trained model or test image folder not found. Please double check paths.")
        print(f"Model expected at: {os.path.abspath(model_checkpoint_path)}")
        print(f"Test image folder expected at: {os.path.abspath(TEST_IMAGES_BASE_PATH)}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_custom_model_on_random_image()