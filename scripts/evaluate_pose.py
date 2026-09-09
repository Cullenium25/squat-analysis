"""Evaluate a trained YOLO pose model."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from PIL import Image
from ultralytics import YOLO


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./runs/pose/pose19_experiment/weights/best.pt")
    parser.add_argument("--data-yaml", default="./configs/data.yaml")
    parser.add_argument("--project", default="runs/pose")
    parser.add_argument("--name", default="pose19_evaluation")
    args = parser.parse_args(argv)

    if not os.path.exists(args.model):
        print(f"ERROR: model not found at {args.model}")
        return 1
    if not os.path.exists(args.data_yaml):
        print(f"ERROR: data.yaml not found at {args.data_yaml}")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)

    with open(args.data_yaml) as f:
        cfg = yaml.safe_load(f)

    results = model.val(
        data=args.data_yaml,
        imgsz=640,
        batch=16,
        device=device,
        verbose=True,
        project=args.project,
        name=args.name,
        save_json=True,
    )

    print(f"  box mAP50-95: {results.box.map:.4f}")
    print(f"  box mAP50:    {results.box.map50:.4f}")
    print(f"  pose mAP50-95: {results.pose.map:.4f}")
    print(f"  pose mAP50:    {results.pose.map50:.4f}")

    save_dir = Path(results.save_dir)
    f1_path = save_dir / "PoseF1_curve.png"
    if f1_path.exists():
        img = Image.open(f1_path)
        img.show()
        print(f"F1 curve: {f1_path}")
    else:
        print("PoseF1_curve.png not found; check eval output directory.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
