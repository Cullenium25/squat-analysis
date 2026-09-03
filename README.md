# AI Squat Analyzer
A computer vision system that analyzes squat form from video, combining pose estimation with a custom-trained multi-label classifier to detect biomechanical errors — built by a physical therapist learning applied ML.






[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Cullenium/squat-analysis-demo)


# Overview
Poor squat mechanics, such knees caving in, excessive forward lean, and insufficient depth, are among the most common squatting faults people have when performing a squat. Judging these faults can be an even harder task. This project uses computer vision to give objective, automated feedback on squat form from a plain video file, no wearable sensors or lab equipment required.

As a former physical therapist, I built this project to bridge clinical movement-assessment knowledge with modern ML tooling: pose estimation, feature engineering from joint kinematics, and a supervised classifier trained to recognize specific form faults.

## Project Goals
Process a video of a person performing squats

Detect and track the person via pose estimation

Compute biomechanical features (joint angles, torso inclination) from raw keypoints

Classify common form errors (e.g. knee valgus, forward trunk lean, shallow depth)

Score each rep and overlay real-time feedback on an annotated output video

## How It Works

```text
Input Video
    ↓
YOLOv11 Pose Estimation
    ↓
Keypoint Extraction
    ↓
Biomechanical Feature Engineering
    ↓
Multi-Label Form Classifier
    ↓
Annotated Video + Squat Feedback
```


Video ingestion – OpenCV reads the input file frame by frame.

Pose estimation – Ultralytics YOLOv11 detects 17 keypoints (shoulders, hips, knees, ankles, etc.) per frame.

Feature engineering – Raw coordinates are converted into 19 interpretable biomechanical features, including knee flexion angle, hip angle, and torso inclination.

Form classification – A pre-trained scikit-learn multi-label classifier flags multiple simultaneous errors per rep (a squat can have more than one fault at once).

Feedback rendering – OpenCV overlays the skeleton, a form score, and text feedback onto each frame, then compiles the annotated video.

###  Tech Stack
Component	Tool
Pose estimation	Ultralytics YOLOv11
Video I/O & annotation	OpenCV
Classification model	scikit-learn (multi-label)
Model persistence	joblib
Feature computation	NumPy, pandas
Prototyping / API key demo	Jupyter Notebook, Roboflow API
Language	Python 3

### Getting Started
Prerequisites
bash
python -m pip install -r requirements.txt
Run the notebook demo
The included Jupyter notebook (squat_analysis.ipynb) walks through the full pipeline end-to-end using a Roboflow API key for pose data. Add your own key as an environment variable before running:

bash
export ROBOFLOW_API_KEY="your_key_here"
jupyter notebook squat_analysis.ipynb
Run on your own video
bash
python analyze_squat.py --input path/to/video.mp4 --output path/to/annotated_output.mp4
### Example Output
The system outputs an annotated video showing:

Real-time skeletal overlay

A per-rep squat score

Text feedback flagging specific faults (e.g. "Knees caving in", "Excessive forward lean")

### Limitations & Next Steps
This is a proof-of-concept, not a clinical tool. Known limitations and planned improvements:

Trained on a limited, self-collected video set — accuracy has not been validated against a labeled clinical benchmark

Single-person, single-camera-angle assumption; no multi-person or occlusion handling

Classifier thresholds are not yet calibrated per body type or camera angle

Planned: web-based interface (Gradio/Streamlit), batch video upload, rep-by-rep summary report, expanded training data with inter-rater reliability from licensed PTs
