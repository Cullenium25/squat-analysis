import os
import shutil
from roboflow import Roboflow
from dotenv import load_dotenv


load_dotenv()
ROBOFLOW_API_KEY = os.getenv("API_KEY")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("squat-form-capstone").project("squat-detection-h1dwo")
version = project.version(19)

def download_and_organize_dataset():
    # Step 1: Download dataset using Roboflow API
    dataset = version.download("yolov8")

    # Step 2: Create necessary directories
    folders = [
        "train/images", "valid/images", "test/images",
        "train/labels", "valid/labels", "test/labels",
        "configs", "models"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # Step 3: Move image/label files for each split
    base_path = dataset.location  # e.g., 'squat-detection-4'
    splits = {'train': 'train', 'valid': 'valid', 'test': 'test'}  # Map Roboflow -> your structure

    for split_src, split_dst in splits.items():
        image_src = os.path.join(base_path, split_src, 'images')
        label_src = os.path.join(base_path, split_src, 'labels')

        image_dst = os.path.join(split_dst, 'images')
        label_dst = os.path.join(split_dst, 'labels')

        # Move image files
        if os.path.exists(image_src):
            for file in os.listdir(image_src):
                shutil.move(os.path.join(image_src, file), os.path.join(image_dst, file))

        # Move label files
        if os.path.exists(label_src):
            for file in os.listdir(label_src):
                shutil.move(os.path.join(label_src, file), os.path.join(label_dst, file))

    # Step 4: Move data.yaml to configs/
    data_yaml_src = os.path.join(base_path, "data.yaml")
    data_yaml_dst = "configs/data.yaml"
    if os.path.exists(data_yaml_src):
        shutil.move(data_yaml_src, data_yaml_dst)
        print(f"Moved data.yaml to {data_yaml_dst}")

    shutil.rmtree(base_path)  # Clean up the original dataset folder

    print("Dataset fully organized and files moved successfully!")

if __name__ == "__main__":
    download_and_organize_dataset()