"""The inspector's feedback, kept as future training data.

Every saved review stores the image itself next to the model's answer and the
person's judgement. The images the model got wrong on the real line are the
most valuable ones to label and add to the next training run.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .config import REVIEW_DIR
from .service import Inspection

JUDGEMENTS = ("correct", "missed a defect", "false alarm", "wrong defect type")
FIELDS = ("time_utc", "image", "model_verdict", "judgement", "note", "detections")


def save_review(
    image: np.ndarray, inspection: Inspection, judgement: str, note: str = "", review_dir: Path | None = None
) -> Path:
    """Append one row to reviews.csv and store the image under its content hash."""
    if judgement not in JUDGEMENTS:
        raise ValueError(f"judgement must be one of {JUDGEMENTS}")
    root = review_dir or REVIEW_DIR
    (root / "images").mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("image could not be encoded")
    # Naming by content hash means reviewing the same image twice does not store it twice.
    name = hashlib.sha256(encoded.tobytes()).hexdigest()[:16] + ".png"
    (root / "images" / name).write_bytes(encoded.tobytes())

    log = root / "reviews.csv"
    is_new = not log.exists()
    with log.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "image": name,
                "model_verdict": inspection.verdict,
                "judgement": judgement,
                "note": note.strip()[:500],
                "detections": json.dumps([d.to_dict() for d in inspection.detections]),
            }
        )
    return log


def summary(review_dir: Path | None = None) -> dict:
    """Counts per judgement: a running agreement rate between the model and the inspector."""
    log = (review_dir or REVIEW_DIR) / "reviews.csv"
    counts = {j: 0 for j in JUDGEMENTS}
    if log.exists():
        with log.open(newline="") as f:
            for row in csv.DictReader(f):
                if row["judgement"] in counts:
                    counts[row["judgement"]] += 1
    total = sum(counts.values())
    return {"reviews": total, "by_judgement": counts, "agreement": round(counts["correct"] / total, 4) if total else None}
