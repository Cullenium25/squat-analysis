# AI Squat Analyzer: A Computer Vision Capstone Project
This repository contains the code for a proof-of-concept computer vision-based application that analyzes squat form from pre-recorded video files. The project processes each frame of a video to provide actionable feedback on squat technique. This project demonstrates the integration of computer vision and machine learning for fitness analysis. My objective was to combined my growing programming skills with my experience as a physical therapist

Provided is a jupyter notebook (Squat_jupyter (pre-run).ipynb) demonstrating how the model works using my Roboflow API key.

[Video Presentation- DEMO](https://youtu.be/sGshgkvAnV0?si=sNZspG7i29eCTpSh&t=98) 

## Project Goals
The primary goal of this project is to build an intelligent squat analyzer that can:

Process a video file containing a person performing squats.

Detect and track the person throughout the video using pose estimation.

Analyze key biomechanical metrics of the squat form.

Identify common form errors (e.g., knees caving in, leaning forward).

Provide a score and feedback on the squat performance, saved as a new video.

### How It Works: The System Architecture
The application runs as a single, self-contained Python script that takes a video file as input.

The script uses OpenCV to read and process a video file, frame by frame.

For each frame, it employs Ultralytics YOLOv11 to perform real-time pose estimation. This process accurately identifies the location of key joints on the person, such as the shoulders, hips, and knees.

From these joint coordinates, the script calculates a set of 19 biomechanical features, including joint angles and torso inclination, which are crucial for evaluating form.

A pre-trained scikit-learn multi-label classifier then takes these features as input to detect multiple form errors simultaneously.

Finally, the script uses OpenCV to annotate each frame with a pose overlay, a score, and real-time feedback. These annotated frames are then compiled and saved as a new video file.

Key Technologies
Python 3: The core language for the project.

OpenCV: For reading video files, processing frames, and writing the output video.

Ultralytics YOLOv11: The state-of-the-art model for real-time pose estimation.

scikit-learn: Used to train and deploy the custom multi-label classifier for form analysis.

joblib: For saving and loading the trained machine learning model.

numpy and pandas: For efficient data manipulation and feature calculation.
