"""Fine-tune YOLO11n on the steel defect images, on CPU.

    python -m defects.train                 # the settings used for the published model
    python -m defects.train --epochs 3      # a quick smoke run

Needs requirements-train.txt (PyTorch and Ultralytics). Serving does not.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import DATASET_DIR, REPORTS_DIR, ROOT

RUNS_DIR = ROOT / "runs"


def train(epochs: int, imgsz: int, batch: int, name: str, seed: int = 0) -> Path:
    from ultralytics import YOLO

    # Start from weights pretrained on COCO: 1,257 images are far too few to
    # learn edges and textures from nothing, but enough to re-aim a model
    # that already knows them.
    model = YOLO("yolo11n.pt")
    started = time.time()
    model.train(
        data=str(DATASET_DIR / "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        workers=4,
        seed=seed,
        deterministic=True,
        project=str(RUNS_DIR),
        name=name,
        exist_ok=True,
        patience=epochs,  # no early stop: the run length is the budget
        # A steel surface has no "up": vertical flips are as valid as
        # horizontal ones and double the variety for free.
        flipud=0.5,
        fliplr=0.5,
        # The images are grey, so hue and saturation jitter would only add noise.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        plots=False,
        verbose=False,
    )
    minutes = (time.time() - started) / 60
    run_dir = RUNS_DIR / name
    summary = {
        "base_weights": "yolo11n.pt (COCO pretrained)",
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "seed": seed,
        "device": "cpu",
        "train_minutes": round(minutes, 1),
        "history": _history(run_dir / "results.csv"),
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"trained {epochs} epochs in {minutes:.1f} min -> {run_dir / 'weights' / 'best.pt'}")
    return run_dir / "weights" / "best.pt"


def _history(csv_path: Path) -> list[dict]:
    """Per-epoch validation scores as logged by the trainer, for the training curve."""
    import csv

    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v for k, v in row.items()}
            rows.append(
                {
                    "epoch": int(float(row["epoch"])),
                    "val_map50": round(float(row["metrics/mAP50(B)"]), 4),
                    "val_map50_95": round(float(row["metrics/mAP50-95(B)"]), 4),
                    "train_box_loss": round(float(row["train/box_loss"]), 4),
                }
            )
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--name", default="yolo11n-neu")
    args = ap.parse_args()
    train(args.epochs, args.imgsz, args.batch, args.name)
