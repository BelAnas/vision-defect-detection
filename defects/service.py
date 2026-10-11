"""The decision layer: from boxes to something an inspector can act on.

A detector returns boxes with confidences. A production line needs a verdict.
The rule is deliberately simple and lives in one function so a plant can
replace it with its own acceptance standard without touching the model.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import THRESHOLDS_PATH
from .detector import Detection, Detector

DEFAULT_CONFIDENCE = 0.25
DEFAULT_REVIEW_FLOOR = 0.10

PASS, REVIEW, REJECT = "pass", "review", "reject"

# One fixed colour per class (BGR), chosen to stay readable on grey steel.
COLOURS = [(0, 165, 255), (60, 60, 230), (60, 180, 75), (200, 130, 0), (180, 60, 180), (0, 200, 220)]


@dataclass(frozen=True)
class Thresholds:
    """Confidence needed to call a box a defect, per class, and the floor below
    which a box is not shown at all. Boxes in between go to a person."""

    per_class: tuple[float, ...]
    review_floor: float

    @classmethod
    def load(cls, class_names: tuple[str, ...], path: Path = THRESHOLDS_PATH) -> "Thresholds":
        override = os.environ.get("DEFECTS_CONFIDENCE")
        if override:
            value = float(override)
            return cls((value,) * len(class_names), min(DEFAULT_REVIEW_FLOOR, value))
        if path.exists():
            data = json.loads(path.read_text())
            return cls(
                tuple(float(data["per_class"].get(n, DEFAULT_CONFIDENCE)) for n in class_names),
                float(data.get("review_floor", DEFAULT_REVIEW_FLOOR)),
            )
        return cls((DEFAULT_CONFIDENCE,) * len(class_names), DEFAULT_REVIEW_FLOOR)

    def scaled(self, factor: float) -> "Thresholds":
        """Stricter (>1) or more sensitive (<1) thresholds, for the slider in the review screen."""
        per_class = tuple(min(0.99, max(0.01, t * factor)) for t in self.per_class)
        return Thresholds(per_class, min(self.review_floor, min(per_class)))


def verdict_for(boxes: np.ndarray, thresholds: Thresholds) -> tuple[str, np.ndarray]:
    """Verdict for one image and a flag per box saying whether it is confident.

    reject: at least one box at or above its class threshold.
    review: only boxes between the review floor and their class threshold.
    pass:   nothing at or above the review floor.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 6)
    boxes = boxes[boxes[:, 4] >= thresholds.review_floor]
    if len(boxes) == 0:
        return PASS, np.zeros(0, dtype=bool)
    confident = boxes[:, 4] >= np.asarray(thresholds.per_class)[boxes[:, 5].astype(int)]
    return (REJECT if confident.any() else REVIEW), confident


@dataclass
class Inspection:
    verdict: str
    reason: str
    detections: list[Detection]
    confident: list[bool]
    latency_ms: float
    image_size: tuple[int, int]  # width, height
    thresholds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "detections": [{**d.to_dict(), "confident": c} for d, c in zip(self.detections, self.confident)],
            "latency_ms": round(self.latency_ms, 1),
            "image": {"width": self.image_size[0], "height": self.image_size[1]},
            "thresholds": self.thresholds,
        }


class Inspector:
    """Detector plus thresholds: the object the API, the screen and the CLI all share."""

    def __init__(self, detector: Detector | None = None, thresholds: Thresholds | None = None) -> None:
        self.detector = detector or Detector()
        self.thresholds = thresholds or Thresholds.load(self.detector.class_names)

    def inspect(self, image: np.ndarray, sensitivity: float = 1.0) -> Inspection:
        thresholds = self.thresholds if sensitivity == 1.0 else self.thresholds.scaled(sensitivity)
        started = time.perf_counter()
        boxes = self.detector.predict(image, thresholds.review_floor)
        latency = (time.perf_counter() - started) * 1000
        verdict, confident = verdict_for(boxes, thresholds)
        names = self.detector.class_names
        detections = [
            Detection(names[int(c)], int(c), float(s), float(x1), float(y1), float(x2), float(y2))
            for x1, y1, x2, y2, s, c in boxes
        ]
        return Inspection(
            verdict=verdict,
            reason=_reason(verdict, detections, confident),
            detections=detections,
            confident=[bool(c) for c in confident],
            latency_ms=latency,
            image_size=(image.shape[1], image.shape[0]),
            thresholds={**dict(zip(names, thresholds.per_class)), "review_floor": thresholds.review_floor},
        )


def _reason(verdict: str, detections: list[Detection], confident: np.ndarray) -> str:
    """One plain sentence for the inspector, naming what was found."""
    if verdict == PASS:
        return "No defect found."
    chosen = [d for d, c in zip(detections, confident) if c] if verdict == REJECT else detections
    counts: dict[str, int] = {}
    for d in chosen:
        counts[d.label] = counts.get(d.label, 0) + 1
    found = ", ".join(f"{n} x {label}" for label, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    if verdict == REJECT:
        return f"Defect found: {found}."
    return f"Possible defect, low confidence: {found}. A person should look."


def draw(image: np.ndarray, inspection: Inspection, min_side: int = 600) -> np.ndarray:
    """The image with boxes and labels drawn on it, enlarged so small images stay readable."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    k = max(1.0, min_side / min(image.shape[:2]))
    canvas = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST) if k > 1 else image.copy()
    for det, sure in zip(inspection.detections, inspection.confident):
        colour = COLOURS[det.class_id % len(COLOURS)]
        p1, p2 = (round(det.x1 * k), round(det.y1 * k)), (round(det.x2 * k), round(det.y2 * k))
        # Uncertain boxes are thin, so the eye goes to the confident ones first.
        cv2.rectangle(canvas, p1, p2, colour, 3 if sure else 1)
        text = f"{det.label} {det.confidence:.2f}" + ("" if sure else " ?")
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x = min(p1[0], canvas.shape[1] - tw - 4)
        y = p1[1] + th + base + 2 if p1[1] < th + base + 4 else p1[1]
        cv2.rectangle(canvas, (x, y - th - base - 2), (x + tw + 4, y), colour, -1)
        cv2.putText(canvas, text, (x + 2, y - base), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
