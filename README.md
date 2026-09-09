# AI Squat Analyzer

[![Live Demo](https://img.shields.io/badge/HuggingFace-Demo-ffcc00)](https://huggingface.co/spaces/Cullenium/squat-analysis-demo)

This repository explores whether a computer-vision and machine-learning pipeline can identify selected squat-form patterns from pre-recorded video. The purpose is to explore the feasibility and limitations of applying pose estimation and multi-label classification to movement analysis, without wearable sensors or lab equipment.
Among some of the more common squat faults are: knees caving in, excessive forward lean, insufficient depth. This AI Squat Analyzer is a **proof of concept** designed to identify common squatting faults and subject point of view, not to designed to make authoritative judgments about exercise safety, clinical status, or technique quality. Although built by a physical therapist learning applied ML, this tool is not a validated clinical assessment and should not be used as replacement for proper diagnostic clinical assessment. Possible uses may be for recreational use by fitness enthusiasts who lack the means to observe themselves properly, coaches, and clinical professionals as a supplementary educational tool.

--- 

## Live Demo

![Demo GIF](docs/demo.gif)

Upload a squat video or image and get real-time annotated feedback.  
**[Try it live →](https://huggingface.co/spaces/Cullenium/squat-analysis-demo)**

---

## What It Does
The project demonstrates an end-to-end workflow for:

- Reading a pre-recorded squat video
- Estimating body pose on each video frame
- Deriving biomechanical-style features from detected joint positions
- Applying a machine-learning classifier to flag selected movement patterns
- Producing an annotated output video with visual feedback and scoring



This project uses computer vision to give objective, automated feedback on squat form from a plain video file.  

The project demonstrates an end-to-end workflow for:

```
Input Video
    ↓
YOLOv11 Pose Estimation      19 keypoints per frame (Retrained)
    ↓
Keypoint Extraction          Best-confidence person selected
    ↓
Biomechanical Features       interpretable features:
  • Knee/hip/ankle angles     joint angles, trunk lean,
  • Trunk lean ratios         femur coronal proxies,
  • Femur coronal proxies     foot orientation
    ↓
Multi-Label Classifier       Trained scikit-learn model
  Predicts simultaneous faults
    ↓
Annotated Video + Feedback
```

Note: YOLOv11 was used as baseline model and retrained for 100 epochs to include 19 keypoints (instead of default 17), for additional points for the feet, enable feature engineering of ankle angles. See notebooks folder for walkthrough process

---

## Tech Stack

| Component | Tool |
|---|---|
|| Pose estimation | Ultralytics YOLOv12 |
| Video I/O & annotation | OpenCV |
| Classification model | scikit-learn (multi-label) |
| Model persistence | joblib |
| Feature computation | NumPy, pandas |
| Prototyping | Jupyter Notebook |
| Web demo | Gradio (Hugging Face Spaces) |

---

## Project Structure

```
squat-analysis/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── src/squat_analyzer/          # Core package
│   ├── features.py              # 19 biomechanical features from keypoints
│   ├── pose.py                  # YOLO pose model wrapper
│   ├── classifier.py            # joblib artifact loading helpers
│   ├── annotator.py             # Skeleton + tag overlay for OpenCV frames
│   ├── pipeline.py              # analyze_image() and analyze_video()
│   └── __main__.py              # CLI entry point
│
├── scripts/                     # Training and evaluation utilities
│   ├── train_pose_model.py      # Fine-tune YOLO pose on squat dataset
│   ├── train_classifier.py      # Train multi-label classifier on features
│   └── evaluate_pose.py         # Evaluate pose model (mAP, F1)
│
├── app/
│   └── app.py                   # Gradio web demo → deploys to HF Spaces
│
├── notebooks/                   # Exploratory analysis (gitignored)
├── tests/                       # pytest suite
├── configs/                     # Dataset YAML, annotation configs
├── models/                      # Trained joblib artifacts (gitignored)
└── runs/                        # YOLO training outputs (gitignored)
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/Cullenium25/squat-analysis.git
cd squat-analysis

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install package in editable mode
pip install -e .

# 4. Place trained model artifacts in models/
#    - models/squat_classifier_balanced_rfc.joblib
#    - models/squat_classifier_balanced_rfc_class_names.joblib
#    - models/squat_classifier_scaler.joblib
```

Note: Model artifacts are not included in this repo. Generate them by running `scripts/train_classifier.py` after extracting pose features.

---

## How to Use

### CLI

```bash
# Analyze a video
python -m squat_analyzer squat_clip.mp4 output.mp4

# Analyze an image
python -m squat_analyzer squat_photo.jpg output.jpg

# Use models from a custom directory
python -m squat_analyzer squat_clip.mp4 output.mp4 --models-dir path/to/models
```

### Python API

```python
from squat_analyzer.pipeline import analyze_image, analyze_video

# Single image
tags = analyze_image("photo.jpg", "result.jpg")
print("Detected:", tags)

# Full video
tags = analyze_video("squat_clip.mp4", "annotated.mp4")
print("Detected:", tags)
```

### Script reference

| File | What it does |
|---|---|
| `src/squat_analyzer/pipeline.py` | Core inference — used by CLI, app, and tests |
| `scripts/train_pose_model.py` | Fine-tune YOLO pose on your squat dataset |
| `scripts/train_classifier.py` | Train multi-label classifier on extracted features |
| `scripts/evaluate_pose.py` | Evaluate pose model (mAP, precision/recall) |
| `app/app.py` | Gradio web demo — upload video, get annotated output |
| `notebooks/` | Exploratory analysis, feature correlations, tag distributions |

---

## Model Artifacts

The `models/` directory should contain three joblib files:

- `squat_classifier_balanced_rfc.joblib` — the trained classifier
- `squat_classifier_balanced_rfc_class_names.joblib` — ordered class labels
- `squat_classifier_scaler.joblib` — StandardScaler fitted on training features

Generate them by running `scripts/train_classifier.py` after you have pose features extracted.

---

## Limitations

This is a proof-of-concept, not a clinical tool.

- Trained on a limited, self-collected video set — accuracy has not been validated against a labeled clinical benchmark
- Single-person, single-camera-angle assumption; no multi-person or occlusion handling
- Classifier thresholds are not yet calibrated per body type or camera angle

**Planned improvements:** expanded training data, rep-by-rep summary report, web-based batch upload, inter-rater reliability validation with licensed PTs, and hopefully implementing more complex movement faults based on more temporal data in the future.

---

## About the Author

Built by a physical therapist learning applied ML. Clinical background in movement assessment + modern ML tooling (pose estimation, feature engineering, supervised classification).

