"""Measure the exported ONNX model on the held-out test images.

    python -m defects.evaluate

Order matters and is the point of this file:
  1. Validation images choose the confidence thresholds (written to model/thresholds.json).
  2. Test images, never used for any choice, give the reported numbers.

Writes reports/metrics.json, reports/evaluation.md and the figures in docs/.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from . import metrics as M
from .config import DOCS_DIR, MODEL_PATH, REPORTS_DIR, SAMPLES_DIR, THRESHOLDS_PATH
from .data import load_split
from .detector import Detector
from .service import PASS, REJECT, REVIEW, Inspector, Thresholds, draw, verdict_for

# Scoring needs the whole precision-recall curve, so boxes are collected far
# below any threshold a person would use.
SCORING_CONFIDENCE = 0.001
# At most this share of defective validation images may pass with no box at all.
MAX_SILENT_MISS = 0.01


def predict_split(detector: Detector, split: str) -> tuple[list[Path], list[np.ndarray], list[np.ndarray]]:
    items = load_split(split)
    preds = [detector.predict(cv2.imread(str(p)), SCORING_CONFIDENCE) for p, _ in items]
    return [p for p, _ in items], preds, [g for _, g in items]


def choose_thresholds(preds: list[np.ndarray], gts: list[np.ndarray], names: tuple[str, ...]) -> Thresholds:
    """Per class: the confidence with the best F1. Review floor: the highest value
    that still leaves a box on all but 1% of the (all defective) validation images."""
    records = M.collect(preds, gts, len(names))
    per_class = tuple(M.best_f1_threshold(r) for r in records)
    top_score = np.array([p[:, 4].max() if len(p) else 0.0 for p in preds])
    floor = float(np.floor(np.quantile(top_score, MAX_SILENT_MISS) * 100) / 100)
    floor = float(min(max(floor, 0.05), min(per_class)))
    return Thresholds(per_class, floor)


def latency(detector: Detector, paths: list[Path], confidence: float, repeats: int = 3) -> dict:
    """Wall time per image for the full call: resize, model, suppression."""
    images = [cv2.imread(str(p)) for p in paths]
    for image in images[:10]:  # warm-up: the first calls allocate buffers
        detector.predict(image, confidence)
    times = []
    for _ in range(repeats):
        for image in images:
            t = time.perf_counter()
            detector.predict(image, confidence)
            times.append((time.perf_counter() - t) * 1000)
    times = np.array(times)
    return {
        "images_timed": len(times),
        "mean_ms": round(float(times.mean()), 1),
        "p50_ms": round(float(np.percentile(times, 50)), 1),
        "p95_ms": round(float(np.percentile(times, 95)), 1),
        "images_per_second": round(1000 / float(times.mean()), 1),
        "cpu": _cpu_name(),
    }


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def evaluate() -> dict:
    detector = Detector(MODEL_PATH)
    names = detector.class_names
    n = len(names)

    _, val_preds, val_gts = predict_split(detector, "val")
    thresholds = choose_thresholds(val_preds, val_gts, names)
    THRESHOLDS_PATH.write_text(
        json.dumps(
            {
                "chosen_on": "validation split (270 images), never the test split",
                "rule": "per class: confidence with the best F1 at IoU 0.5; review floor: "
                f"highest value leaving a box on at least {1 - MAX_SILENT_MISS:.0%} of validation images",
                "per_class": dict(zip(names, thresholds.per_class)),
                "review_floor": thresholds.review_floor,
            },
            indent=2,
        )
        + "\n"
    )
    val_summary = M.summarise(M.collect(val_preds, val_gts, n))

    paths, preds, gts = predict_split(detector, "test")
    records = M.collect(preds, gts, n)
    summary = M.summarise(records)
    ci50 = M.bootstrap_interval(records, len(paths), "map50")
    ci = M.bootstrap_interval(records, len(paths), "map50_95")

    per_class, tp, fp, fn = [], 0, 0, 0
    for i, name in enumerate(names):
        at = M.at_threshold(records[i], thresholds.per_class[i])
        tp, fp, fn = tp + at["tp"], fp + at["fp"], fn + at["fn"]
        per_class.append(
            {
                "class": name,
                "labelled_boxes": records[i].n_gt,
                "ap50": round(summary["ap50_per_class"][i], 4),
                "ap50_95": round(summary["ap50_95_per_class"][i], 4),
                "threshold": thresholds.per_class[i],
                "precision": round(at["precision"], 4),
                "recall": round(at["recall"], 4),
                "f1": round(at["f1"], 4),
                "found": at["tp"],
                "missed": at["fn"],
                "false_alarms": at["fp"],
            }
        )

    agnostic = M.collect(preds, gts, n, class_agnostic=True)
    agnostic_ap50 = M.summarise(agnostic)["map50"]
    agnostic_ci = M.bootstrap_interval(agnostic, len(paths), "map50")

    # Every test image contains a defect, so any "pass" here is a defective part let through.
    verdicts = [verdict_for(p, thresholds)[0] for p in preds]
    image_level = {v: verdicts.count(v) for v in (REJECT, REVIEW, PASS)}

    confusion = M.confusion_matrix(preds, gts, n, np.array(thresholds.per_class))
    speed = latency(detector, paths, thresholds.review_floor)

    baseline_path = REPORTS_DIR / "baseline.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None

    report = {
        "model": {"file": MODEL_PATH.name, "size_mb": round(MODEL_PATH.stat().st_size / 1e6, 2), "input": detector.size},
        "runtime": "ONNX Runtime, CPUExecutionProvider",
        "validation": {"images": len(val_preds), "map50": round(val_summary["map50"], 4), "map50_95": round(val_summary["map50_95"], 4)},
        "test": {
            "images": len(paths),
            "labelled_boxes": int(sum(len(g) for g in gts)),
            "map50": round(summary["map50"], 4),
            "map50_interval95": [round(ci50[0], 4), round(ci50[1], 4)],
            "map50_95": round(summary["map50_95"], 4),
            "map50_95_interval95": [round(ci[0], 4), round(ci[1], 4)],
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
            "found": tp,
            "missed": fn,
            "false_alarms": fp,
            "class_agnostic_ap50": round(agnostic_ap50, 4),
            "class_agnostic_ap50_interval95": [round(agnostic_ci[0], 4), round(agnostic_ci[1], 4)],
        },
        "per_class": per_class,
        "thresholds": {"per_class": dict(zip(names, thresholds.per_class)), "review_floor": thresholds.review_floor},
        "image_level_on_defective_test_images": image_level,
        "confusion_matrix": {"labels": [*names, "nothing"], "rows_labelled_columns_predicted": confusion.tolist()},
        "latency": speed,
        "baseline": baseline,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (REPORTS_DIR / "evaluation.md").write_text(render_markdown(report))
    figures(report, detector, thresholds)
    return report


def render_markdown(r: dict) -> str:
    t, lat = r["test"], r["latency"]
    lines = [
        "# Evaluation report",
        "",
        "Produced by `python -m defects.evaluate`. The model is the exported ONNX file, run with ONNX Runtime on CPU.",
        f"Thresholds were chosen on the {r['validation']['images']} validation images. "
        f"All numbers below are from the {t['images']} test images ({t['labelled_boxes']} labelled defects), which were used for nothing else.",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| mAP@0.5 | {t['map50']:.3f} (95% interval {t['map50_interval95'][0]:.3f} to {t['map50_interval95'][1]:.3f}) |",
        f"| mAP@0.5:0.95 | {t['map50_95']:.3f} (95% interval {t['map50_95_interval95'][0]:.3f} to {t['map50_95_interval95'][1]:.3f}) |",
        f"| Precision at the chosen thresholds | {t['precision']:.3f} |",
        f"| Recall at the chosen thresholds | {t['recall']:.3f} |",
        f"| Defects found / missed / false alarms | {t['found']} / {t['missed']} / {t['false_alarms']} |",
        f"| Latency per image | {lat['p50_ms']} ms median, {lat['p95_ms']} ms 95th percentile ({lat['cpu']}) |",
        "",
        "## Per defect type",
        "",
        "| Defect type | Labelled | AP@0.5 | Threshold | Precision | Recall | Found | Missed | False alarms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in r["per_class"]:
        lines.append(
            f"| {c['class']} | {c['labelled_boxes']} | {c['ap50']:.3f} | {c['threshold']:.2f} | {c['precision']:.3f} | "
            f"{c['recall']:.3f} | {c['found']} | {c['missed']} | {c['false_alarms']} |"
        )
    il = r["image_level_on_defective_test_images"]
    lines += [
        "",
        "## Image-level outcome",
        "",
        f"Every test image contains at least one defect. Of {t['images']} images: "
        f"**{il['reject']} rejected**, **{il['review']} sent to review**, **{il['pass']} passed** (a pass here is a defective part let through).",
        "The dataset has no defect-free images, so the false reject rate on good parts is not measured.",
        "",
    ]
    if r.get("baseline"):
        b = r["baseline"]
        lines += [
            "## Against the classical baseline",
            "",
            "Finding a defect regardless of its type (class-agnostic AP@0.5):",
            "",
            "| Method | AP@0.5 | 95% interval |",
            "|---|---|---|",
            f"| OpenCV background subtraction, no training | {b['test_ap50']:.3f} | {b['test_ap50_interval95'][0]:.3f} to {b['test_ap50_interval95'][1]:.3f} |",
            f"| Fine-tuned YOLO11n (ONNX) | {t['class_agnostic_ap50']:.3f} | {t['class_agnostic_ap50_interval95'][0]:.3f} to {t['class_agnostic_ap50_interval95'][1]:.3f} |",
            "",
        ]
    worst = sorted(r["per_class"], key=lambda c: c["ap50"])[:2]
    lines += [
        "## Weak spots",
        "",
        f"The two weakest defect types are **{worst[0]['class']}** (AP@0.5 {worst[0]['ap50']:.3f}) and "
        f"**{worst[1]['class']}** (AP@0.5 {worst[1]['ap50']:.3f}). See `docs/confusion.png` for where the errors go.",
        "",
    ]
    return "\n".join(lines)


def figures(report: dict, detector: Detector, thresholds: Thresholds) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DOCS_DIR.mkdir(exist_ok=True)
    names = [c["class"] for c in report["per_class"]]

    # Per-class AP with the overall mAP as a reference line.
    fig, ax = plt.subplots(figsize=(7, 3.4))
    values = [c["ap50"] for c in report["per_class"]]
    bars = ax.barh(names[::-1], values[::-1], color="#3b6ea5")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.axvline(report["test"]["map50"], color="#c0392b", linestyle="--", linewidth=1, label=f"mAP@0.5 = {report['test']['map50']:.2f}")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("AP@0.5 on the test images")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "per_class.png", dpi=130)
    plt.close(fig)

    # Confusion matrix at the chosen thresholds.
    cm = np.array(report["confusion_matrix"]["rows_labelled_columns_predicted"])
    labels = report["confusion_matrix"]["labels"]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Labelled")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "confusion.png", dpi=130)
    plt.close(fig)

    # Training curve, if the training summary is present.
    training = REPORTS_DIR / "training.json"
    if training.exists():
        hist = json.loads(training.read_text())["history"]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot([h["epoch"] for h in hist], [h["val_map50"] for h in hist], label="validation mAP@0.5", color="#3b6ea5")
        ax.plot([h["epoch"] for h in hist], [h["val_map50_95"] for h in hist], label="validation mAP@0.5:0.95", color="#e08e0b")
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(DOCS_DIR / "training_curve.png", dpi=130)
        plt.close(fig)

    # One sample per class: labelled boxes on top, model output below.
    inspector = Inspector(detector, thresholds)
    labels_json = json.loads((SAMPLES_DIR / "labels.json").read_text())
    stems = sorted(labels_json)[::2]
    tiles_gt, tiles_pred = [], []
    for stem in stems:
        image = cv2.imread(str(SAMPLES_DIR / f"{stem}.jpg"))
        truth = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        for x1, y1, x2, y2, _ in labels_json[stem]:
            cv2.rectangle(truth, (round(x1 * 2), round(y1 * 2)), (round(x2 * 2), round(y2 * 2)), (80, 200, 80), 2)
        tiles_gt.append(truth)
        tiles_pred.append(cv2.resize(draw(image, inspector.inspect(image), min_side=400), (400, 400)))
    gap = np.full((400, 6, 3), 255, np.uint8)

    def row(tiles: list[np.ndarray]) -> np.ndarray:
        parts = []
        for t in tiles:
            parts += [t, gap]
        return np.hstack(parts[:-1])

    grid = np.vstack([row(tiles_gt), np.full((6, row(tiles_gt).shape[1], 3), 255, np.uint8), row(tiles_pred)])
    # JPEG keeps this photo-like figure small enough for a README.
    cv2.imwrite(str(DOCS_DIR / "predictions.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    out = evaluate()
    print(json.dumps({k: out[k] for k in ("validation", "test", "thresholds", "image_level_on_defective_test_images", "latency")}, indent=2))
    print(f"\nreport: {REPORTS_DIR / 'evaluation.md'}")
