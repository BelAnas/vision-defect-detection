"""Export the trained weights to ONNX and check the export against PyTorch.

    python -m defects.export                       # export runs/yolo11n-neu/weights/best.pt
    python -m defects.export --weights other.pt

Writes model/defects.onnx, model/model_card.json and reports/parity.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .config import MODEL_PATH, REPORTS_DIR, ROOT
from .data import load_split
from .detector import Detector
from .metrics import iou_matrix


def export(weights: Path, imgsz: int) -> Path:
    from ultralytics import YOLO

    # A fixed input size and a plain opset keep the file readable by older
    # runtimes on factory PCs; `simplify` folds constants so it loads faster.
    exported = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    shutil.move(exported, MODEL_PATH)
    return MODEL_PATH


def parity(weights: Path, n_images: int = 60, confidence: float = 0.25) -> dict:
    """Compare boxes from PyTorch and from the ONNX detector on the same validation images.

    An export can go wrong quietly (a different resize, a swapped channel)
    and still produce plausible boxes. This is the check that it did not.
    """
    from ultralytics import YOLO

    torch_model = YOLO(str(weights))
    detector = Detector(MODEL_PATH)
    items = load_split("val")[:: max(1, 270 // n_images)][:n_images]
    n_torch = n_onnx = matched = 0
    max_score_diff, ious = 0.0, []
    for path, _ in items:
        image = cv2.imread(str(path))
        r = torch_model.predict(image, imgsz=detector.size, conf=confidence, iou=0.5, device="cpu", verbose=False)[0]
        a = np.column_stack([r.boxes.xyxy.numpy(), r.boxes.conf.numpy(), r.boxes.cls.numpy()]).reshape(-1, 6)
        b = detector.predict(image, confidence)
        n_torch, n_onnx = n_torch + len(a), n_onnx + len(b)
        if len(a) and len(b):
            iou = iou_matrix(a[:, :4], b[:, :4])
            for i in range(len(a)):
                j = int(iou[i].argmax())
                if iou[i, j] > 0.9 and a[i, 5] == b[j, 5]:
                    matched += 1
                    ious.append(float(iou[i, j]))
                    max_score_diff = max(max_score_diff, abs(float(a[i, 4] - b[j, 4])))
    return {
        "images": len(items),
        "confidence": confidence,
        "boxes_pytorch": n_torch,
        "boxes_onnx": n_onnx,
        "boxes_matched": matched,
        "mean_iou_of_matched": round(float(np.mean(ious)), 4) if ious else None,
        "max_confidence_difference": round(max_score_diff, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "yolo11n-neu" / "weights" / "best.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    args = ap.parse_args()

    path = export(args.weights, args.imgsz)
    detector = Detector(path)
    card = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "architecture": "YOLO11n (Ultralytics), fine-tuned from COCO weights",
        "input": f"1 x 3 x {detector.size} x {detector.size}, RGB, 0..1, letterboxed with grey 114",
        "output": "1 x (4 + classes) x anchors: cx, cy, w, h in input pixels, then one score per class",
        "classes": list(detector.class_names),
        "opset": 17,
        "licence": "AGPL-3.0 (inherited from Ultralytics YOLO); see README",
    }
    (MODEL_PATH.parent / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")
    report = parity(args.weights)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"model": card, "parity": report}, indent=2))


if __name__ == "__main__":
    main()
