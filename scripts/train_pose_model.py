"""Train / fine-tune YOLO pose model on squat dataset."""
from __future__ import annotations

import argparse
import os

from ultralytics import YOLO
import yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", default="configs/data.yaml")
    parser.add_argument("--base-model", default="yolov12n-pose.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="runs/pose")
    parser.add_argument("--name", default="pose_experiment")
    args = parser.parse_args(argv)

    if not os.path.exists(args.data_yaml):
        print(f"ERROR: data.yaml not found at {args.data_yaml}")
        return 1

    model = YOLO(args.base_model)
    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
