"""A classical OpenCV baseline: find defects without any training.

    python -m defects.baseline

The idea is the one a vision engineer would try first: a defect is a region
that differs from the local background. Subtract a heavily blurred copy of the
image, threshold what is left, and box the blobs. It cannot name the defect
type, so it is scored class-agnostically (was a defect found, whatever it is).
Its three settings are tuned on the validation images and the reported number
comes from the test images, exactly like the trained model.
"""

from __future__ import annotations

import itertools
import json

import cv2
import numpy as np

from . import metrics as M
from .config import REPORTS_DIR
from .data import load_split

GRID = {"sigma": (5, 10, 20), "kernel": (5, 9, 15), "min_area_frac": (0.005, 0.02, 0.05)}


def detect_classical(gray: np.ndarray, sigma: float, kernel: int, min_area_frac: float) -> np.ndarray:
    """Boxes [N, 6] with class 0. The score is the blob's mean contrast, scaled to 0..1."""
    background = cv2.GaussianBlur(gray, (0, 0), sigma)
    contrast = cv2.GaussianBlur(cv2.absdiff(gray, background), (5, 5), 0)
    _, mask = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_frac * gray.shape[0] * gray.shape[1]
    peak = max(float(contrast.max()), 1.0)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        boxes.append([x, y, x + w, y + h, float(contrast[y : y + h, x : x + w].mean()) / peak, 0])
    return np.array(boxes, dtype=np.float64).reshape(-1, 6)


def score(split: str, params: dict) -> dict:
    items = load_split(split)
    preds = [detect_classical(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), **params) for p, _ in items]
    gts = [g for _, g in items]
    records = M.collect(preds, gts, 1, class_agnostic=True)
    out = M.summarise(records)
    thr = M.best_f1_threshold(records[0])
    return {"map50": out["map50"], "map50_95": out["map50_95"], "best_f1_threshold": thr, "records": records, "n": len(items)}


def main() -> None:
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    val = [(score("val", p), p) for p in combos]
    best_val, best = max(val, key=lambda t: t[0]["map50"])
    test = score("test", best)
    # The operating threshold was chosen on validation data; apply it unchanged.
    at = M.at_threshold(test["records"][0], best_val["best_f1_threshold"])
    lo, hi = M.bootstrap_interval(test["records"], test["n"])
    report = {
        "method": "background subtraction + Otsu threshold + contours (OpenCV, no training)",
        "scoring": "class-agnostic, IoU 0.5",
        "settings_tried_on_validation": len(combos),
        "chosen_settings": best,
        "validation_ap50": round(best_val["map50"], 4),
        "test_ap50": round(test["map50"], 4),
        "test_ap50_interval95": [round(lo, 4), round(hi, 4)],
        "test_ap50_95": round(test["map50_95"], 4),
        "confidence_threshold_from_validation": best_val["best_f1_threshold"],
        "test_precision": round(at["precision"], 4),
        "test_recall": round(at["recall"], 4),
        "test_images": test["n"],
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
