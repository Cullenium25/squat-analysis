import os
import yaml
import torch 
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from PIL import Image   

def evaluate_custom_pose():
    model_checkpoint_path = "./runs/pose/pose19_experiment/weights/best.pt"
    # Path to your dataset configuration YAML file
    data_yaml_path = './configs/data.yaml'

    # Define the output directory for evaluation results
    EVAL_PROJECT = 'runs/pose'
    EVAL_NAME = 'pose19_experiment_evaluation' # Choose a clear name for your evaluation run

    # --- Pre-checks ---
    if not os.path.exists(model_checkpoint_path):
        print(f"Error: Model checkpoint not found at {model_checkpoint_path}")
        print("Please ensure you have trained your model and the path is correct.")
        return

    if not os.path.exists(data_yaml_path):
        print(f"Error: Dataset YAML not found at {data_yaml_path}")
        print("Please ensure your data is organized and data.yaml is in 'configs/'")
        return

    # --- Load Model ---
    try:
        model = YOLO(model_checkpoint_path)
        print(f"Loaded model from: {model_checkpoint_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Load Dataset Config and Print Info ---
    with open(data_yaml_path, 'r') as file:
        data_config = yaml.safe_load(file)

    print(f"Evaluating model on dataset configured by: {data_yaml_path}")
    print(f"Dataset root path: {data_config.get('path', 'Not specified in YAML')}")
    if 'test' in data_config:
        print(f"Using test set: {data_config['test']}")
    elif 'val' in data_config:
        print(f"Using validation set: {data_config['val']} (No 'test' split found)")
    else:
        print("Warning: Neither 'test' nor 'val' split found in data.yaml. Evaluation might fail.")

    # --- Determine Device ---
    evaluation_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluation will use the {evaluation_device.upper()}.")

    # --- Evaluate the Model ---
    try:
        results = model.val(
            data=data_yaml_path,
            imgsz=640, # Use the same image size as training
            batch=16,  # Adjust batch size based on your GPU/CPU memory
            device=evaluation_device,
            verbose=True,
            project=EVAL_PROJECT,
            name=EVAL_NAME,
            save_json=True # Needed if you want to also programmatically extract F1 threshold
        )

        print("\n--- Evaluation Metrics ---")
        print("Object Detection (Bounding Box) Metrics:")
        print(f"  mAP50-95: {results.box.map:.4f}")
        print(f"  mAP50:    {results.box.map50:.4f}")
        print(f"  mAP75:    {results.box.map75:.4f}")

        print("\nPose Estimation (Keypoint) Metrics:")
        print(f"  kpt_mAP50-95 (mAP^pose 50-95): {results.pose.map:.4f}")
        print(f"  kpt_mAP50 (mAP^pose 50):    {results.pose.map50:.4f}")

        # The results will be saved to the specified project/name directory
        results_save_dir = Path(results.save_dir)
        print(f"Evaluation results saved to: {results_save_dir}")

        # --- Display the Pose F1 Curve Graph ---
        # Corrected filename here:
        f1_curve_path = results_save_dir / 'PoseF1_curve.png' # <--- CHANGED THIS LINE

        if f1_curve_path.exists():
            try:
                img = Image.open(f1_curve_path)
                plt.figure(figsize=(10, 6)) # Optional: Adjust figure size
                plt.imshow(img)
                plt.axis('off') # Hide axes for cleaner image display
                plt.title("Pose Estimation F1 Curve (Confidence vs F1-score)")
                plt.show() # Display the plot window
                print(f"Displayed Pose F1 curve from: {f1_curve_path}")
            except Exception as plot_e:
                print(f"Error displaying Pose F1 curve: {plot_e}")
                print(f"Please check if {f1_curve_path} is a valid image or if matplotlib/Pillow are correctly installed.")
        else:
            print(f"Warning: PoseF1_curve.png not found at {f1_curve_path}.")
            print("Ensure evaluation completed successfully and the file was generated.")


    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        print("Please ensure your dataset configuration in data.yaml is correct and accessible.")


if __name__ == "__main__":
    evaluate_custom_pose()