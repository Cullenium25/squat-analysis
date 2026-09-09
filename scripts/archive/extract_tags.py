import csv
import json
import os
import shutil
from collections import defaultdict
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()
ROBOFLOW_API_KEY = os.getenv("API_KEY")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("squat-form-capstone").project("squat-detection-h1dwo")
version = project.version(19)

def extract_coco_tags_and_organize():
    TARGET_COCO_JSON_BASE_PATH = os.path.join("configs", "coco_annotations")

    # --- Step 1: Download COCO Keypoints Format ---
    coco_download_format = "coco"
    try:
        print(f"Attempting to download COCO format '{coco_download_format}' (version {version})...")
        coco_dataset = version.download(coco_download_format)
        coco_base_path = coco_dataset.location 
        print(f"COCO data successfully downloaded to temporary path: {coco_base_path}")
    except Exception as e:
        print(f"Error downloading COCO format '{coco_download_format}': {e}")
        print("Could not download COCO data. Please verify the project name, version, and available export formats on Roboflow website.")
        return

    # --- Step 2: Organize COCO JSON files ---
    print("\n--- Organizing COCO JSON files ---")
    splits_map = {'train': 'train', 'valid': 'val', 'test': 'test'}

    for split_src in splits_map.keys():
        src_json_path = os.path.join(coco_base_path, split_src, "_annotations.coco.json")
        dst_split_dir = os.path.join(TARGET_COCO_JSON_BASE_PATH, split_src)
        dst_json_path = os.path.join(dst_split_dir, "_annotations.coco.json")

        os.makedirs(dst_split_dir, exist_ok=True) # Create target directory for this split

        if os.path.exists(src_json_path):
            shutil.move(src_json_path, dst_json_path)
            print(f"Moved {split_src} JSON to: {dst_json_path}")
        else:
            print(f"Warning: Source JSON not found for {split_src} at: {src_json_path}. Skipping move.")

    # --- Step 3: Clean up the redundant Roboflow download folder ---
    print("\n--- Cleaning up temporary download folder ---")
    if coco_dataset and os.path.exists(coco_base_path) and os.path.isdir(coco_base_path):
        try:
            shutil.rmtree(coco_base_path)
            print(f"Cleaned up original COCO dataset folder: {coco_base_path}")
        except Exception as e:
            print(f"ERROR: Could not clean up {coco_base_path}: {e}")
    else:
        print("No COCO dataset folder to clean up (perhaps download failed or was manually cleaned).")

    # --- Step 4: Extract Tags from the NOW ORGANIZED COCO JSONs ---
    print("\n--- Starting Tag Extraction Per Split from organized JSONs ---")

    split_image_tags_ground_truth = defaultdict(dict)
    all_possible_tags_found = set()

    for split_src in splits_map.keys():
        current_json_path = os.path.join(TARGET_COCO_JSON_BASE_PATH, split_src, "_annotations.coco.json")

        print(f"Processing organized JSON file for {split_src} split: {current_json_path}")
        if os.path.exists(current_json_path):
            try:
                with open(current_json_path, 'r') as f:
                    coco_data = json.load(f)
                print(f"Successfully loaded JSON for {split_src} split.")
            except json.JSONDecodeError as e:
                print(f"ERROR: Could not decode JSON from {current_json_path}: {e}")
                continue

            if 'images' in coco_data:
                print(f"Extracting tags from {len(coco_data['images'])} images in {split_src} split.")
                for i, image_info in enumerate(coco_data['images']):
                    image_filename = image_info.get('file_name')

                    if not image_filename:
                        print(f"WARNING: Image {i} in {split_src} has no 'file_name'. Skipping.")
                        continue

                    # The keypoint to extract 'user_tags' from the 'extra' field
                    if 'extra' in image_info and 'user_tags' in image_info['extra']:
                        current_image_tags = image_info['extra']['user_tags']
                        if current_image_tags:
                            if i < 5: # Debug for first few images
                                print(f"  [{split_src}] Image '{image_filename}': Found user_tags: {current_image_tags}")

                            all_possible_tags_found.update(current_image_tags)

                            if image_filename not in split_image_tags_ground_truth[split_src]:
                                split_image_tags_ground_truth[split_src][image_filename] = []
                            split_image_tags_ground_truth[split_src][image_filename].extend(current_image_tags)
                            split_image_tags_ground_truth[split_src][image_filename] = list(set(split_image_tags_ground_truth[split_src][image_filename]))
                    else:
                        if i < 5: print(f"  [{split_src}] Image '{image_filename}': No 'extra' or 'user_tags' found.")
            else:
                print(f"No 'images' key found in {current_json_path} for {split_src}. Skipping tag extraction.")
        else:
            print(f"Organized JSON file not found for {split_src} split at: {current_json_path}")

    if not all_possible_tags_found:
        print("\n--- WARNING: No tags were extracted from any split. ---")
        print("Please verify your dataset's annotations and the COCO JSON structure.")
    else:
        print(f"\nSuccessfully extracted unique tags: {sorted(list(all_possible_tags_found))}")
        print("Collected tags per split.")


    # --- Step 5: Save Extracted Tags to CSV for each split ---
    print("\n--- Saving Tags to CSVs Per Split alongside JSONs ---")

    csv_fieldnames_base = ['image_filename']
    sorted_all_possible_tags = sorted(list(all_possible_tags_found))
    csv_fieldnames = csv_fieldnames_base + sorted_all_possible_tags

    for split_src, image_tags_data in split_image_tags_ground_truth.items():
        if image_tags_data:
            output_folder_path = os.path.join(TARGET_COCO_JSON_BASE_PATH, split_src)
            output_csv_path = os.path.join(output_folder_path, "image_tags_ground_truth.csv")

            # Directory already created in Step 2
            with open(output_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
                writer.writeheader()
                for filename, tags_list_for_image in image_tags_data.items():
                    row = {'image_filename': filename}
                    for tag in sorted_all_possible_tags:
                        row[tag] = 1 if tag in tags_list_for_image else 0
                    writer.writerow(row)

            print(f"Tags for '{split_src}' split saved to: {output_csv_path}")
        else:
            print(f"No tags found for '{split_src}' split. Skipping CSV creation for this split.")

    print("\nCOCO tag extraction process finished.")

if __name__ == "__main__":
    extract_coco_tags_and_organize()