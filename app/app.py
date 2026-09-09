"""Gradio web demo — deployable to Hugging Face Spaces (ZeroGPU compatible)."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

import gradio as gr

try:
    import spaces  # available on HF Spaces runtime
except ImportError:  # local development fallback
    spaces = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Make src/ importable in the Space. The Space clones the repo into /home/user,
# so we try a few likely locations before failing.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_SRC = [
    os.path.join(_HERE, "..", "src"),       # /home/user/src  (repo root layout)
    os.path.join(_HERE, "src"),             # /home/user/app/src
    "/home/user/src",                       # absolute fallback
    os.path.join(os.getcwd(), "src"),       # cwd-relative
]

for _candidate in _CANDIDATE_SRC:
    _abs = os.path.abspath(_candidate)
    if os.path.isdir(_abs) and _abs not in sys.path:
        sys.path.insert(0, _abs)
        print(f"[app] Added to sys.path: {_abs}")

print(f"[app] sys.path = {sys.path}")
print(f"[app] cwd = {os.getcwd()}")
if os.path.isdir("/home/user"):
    print(f"[app] listing /home/user: {sorted(os.listdir('/home/user'))}")

from squat_analyzer.pipeline import analyze_image, analyze_video
from squat_analyzer.pose import POSE_MODEL_PATH
from squat_analyzer.classifier import (
    DEFAULT_MODELS_DIR,
    load_classifier,
    load_class_names,
    load_scaler,
)


def _run_with_gpu(fn):
    """Decorate a function to request ZeroGPU if available, else run on CPU."""
    if spaces is not None and hasattr(spaces, "GPU"):
        return spaces.GPU(fn)
    return fn


# --- Configuration via environment variables (set these in Space Settings → Variables & secrets) ---
MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "")  # e.g. "Cullenium/squat-analysis-models"
MODEL_REPO_TYPE = os.environ.get("MODEL_REPO_TYPE", "model")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")

# Cache downloaded models here within the Space
_SPACE_CACHE = "/tmp/squat_models"


def _download_from_hf() -> bool:
    """Download model artifacts from a HuggingFace Hub repo if configured."""
    if not MODEL_REPO_ID:
        return False
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False

    os.makedirs(_SPACE_CACHE, exist_ok=True)
    local = snapshot_download(
        repo_id=MODEL_REPO_ID,
        repo_type=MODEL_REPO_TYPE,
        local_dir=_SPACE_CACHE,
    )
    print(f"Downloaded models from HF Hub to: {local}")
    return True


def _ensure_models() -> str:
    """Return path to a directory containing the three .joblib artifacts."""
    # If the user uploaded them directly into the Space repo
    if os.path.isdir(MODELS_DIR) and all(
        os.path.exists(os.path.join(MODELS_DIR, f))
        for f in [
            "squat_classifier_balanced_rfc.joblib",
            "squat_classifier_balanced_rfc_class_names.joblib",
            "squat_classifier_scaler.joblib",
        ]
    ):
        print(f"[app] Using local models dir: {os.path.abspath(MODELS_DIR)}")
        return os.path.abspath(MODELS_DIR)

    # Try downloading from HF Hub
    if _download_from_hf():
        if all(
            os.path.exists(os.path.join(_SPACE_CACHE, f))
            for f in [
                "squat_classifier_balanced_rfc.joblib",
                "squat_classifier_balanced_rfc_class_names.joblib",
                "squat_classifier_scaler.joblib",
            ]
        ):
            print(f"[app] Using downloaded models from HF Hub: {_SPACE_CACHE}")
            return _SPACE_CACHE

    raise FileNotFoundError(
        "Model artifacts not found. Either:\n"
        "1. Upload the 3 .joblib files to a 'models/' folder in this Space, or\n"
        "2. Set MODEL_REPO_ID to a HF Hub repo containing them."
    )


def _pose_model_path() -> str:
    """Resolve pose model path — prefer env override, then HF cache, then repo path."""
    env_path = os.environ.get("POSE_MODEL_PATH", "")
    if env_path and os.path.exists(env_path):
        print(f"[app] Using pose model from env: {env_path}")
        return env_path
    for candidate in [
        os.path.join(_SPACE_CACHE, "best.pt"),
        os.path.join("runs", "pose", "pose19_experiment", "weights", "best.pt"),
        str(POSE_MODEL_PATH),
    ]:
        if os.path.exists(candidate):
            print(f"[app] Using pose model: {candidate}")
            return candidate
    # Fall back to ultralytics auto-download of the base model
    print("[app] Falling back to ultralytics auto-download: yolov12n-pose.pt")
    return "yolov12n-pose.pt"


@_run_with_gpu
def analyze_file(filepath: str, progress=gr.Progress()) -> tuple[str, str]:
    """Run analysis on an uploaded image or video, return (output_path, feedback)."""
    print(f"[app] analyze_file called with: {filepath!r}")
    print(f"[app] file exists: {os.path.exists(filepath) if filepath else 'N/A (empty path)'}")
    if not filepath:
        return "", "Please upload a file first."
    if not os.path.exists(filepath):
        return "", f"Uploaded file not found: {filepath}"

    try:
        models_dir = _ensure_models()
    except FileNotFoundError as exc:
        return "", f"Model error: {exc}"

    pose_path = _pose_model_path()

    # Patch module-level constants so the pipeline uses our resolved paths
    import squat_analyzer.pose as pose_mod
    import squat_analyzer.classifier as clf_mod
    pose_mod.POSE_MODEL_PATH.__class__.__new__(pose_mod.POSE_MODEL_PATH.__class__)
    pose_mod.POSE_MODEL_PATH = pose_path  # type: ignore[assignment]
    clf_mod.DEFAULT_MODELS_DIR = models_dir

    lower = filepath.lower()
    is_image = lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

    # Use a real temp dir that persists until Gradio finishes serving the file
    tmp = tempfile.mkdtemp(prefix="squat_")
    ext = ".mp4" if not is_image else ".jpg"
    out_path = os.path.join(tmp, f"result{ext}")

    try:
        if is_image:
            progress(0.1, "Loading models...")
            progress(0.3, "Analyzing image...")
            tags = analyze_image(filepath, out_path, models_dir)
        else:
            progress(0.1, "Loading models...")
            progress(0.2, "Processing video frames (this takes ~1-2 min)...")
            tags = analyze_video(filepath, out_path, models_dir)

        progress(1.0, "Done!")
        feedback = ", ".join(tags) if tags else "No issues detected."
        return out_path, feedback
    except Exception as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        return "", f"Analysis error: {exc}"


with gr.Blocks(title="AI Squat Analyzer") as demo:
    gr.Markdown(
        "# Squat Analyzer\n"
        "Upload a squat **video** or **image** and get real-time form feedback.\n\n"
        "> Uses pose estimation + a trained multi-label classifier to detect "
        "knee valgus, forward trunk lean, shallow depth, and more."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.File(label="Video or image", type="filepath")
            run_btn = gr.Button("Analyze", variant="primary", size="lg")
        with gr.Column():
            out = gr.Video(label="Annotated result")
    feedback = gr.Textbox(label="Detected form tags", interactive=False)
    status = gr.Markdown("Ready — upload a file and click **Analyze**.")

    run_btn.click(
        analyze_file,
        inputs=inp,
        outputs=[out, feedback],
    ).then(
        lambda: "Analysis complete!", outputs=status
    ).then(
        # Clean up temp dir after a delay (Gradio has already copied the file)
        lambda: None,
        None,
        None,
    )

if __name__ == "__main__":
    demo.launch()
