"""CLI entry point: python -m squat_analyzer."""
from __future__ import annotations

import argparse
import sys

from squat_analyzer.pipeline import analyze_image, analyze_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze squat form from an image or video."
    )
    parser.add_argument("input", help="Path to input image or video file")
    parser.add_argument("output", help="Path for annotated output file")
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory with .joblib model artifacts",
    )
    args = parser.parse_args(argv)

    lower = args.input.lower()
    try:
        if lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            tags = analyze_image(args.input, args.output, args.models_dir)
        else:
            tags = analyze_video(args.input, args.output, args.models_dir)
        print("Detected tags:", ", ".join(tags) if tags else "None")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
