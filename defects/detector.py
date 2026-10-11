"""Run the exported model with ONNX Runtime and OpenCV. No PyTorch here.

The export contains the network only. Everything around it (resizing the
image, turning raw output into boxes, removing duplicates) is written out
below, which is what makes the model portable to any runtime that reads ONNX.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .config import CLASS_NAMES, MODEL_PATH, NMS_IOU
from .metrics import iou_matrix

PAD_VALUE = 114  # the grey Ultralytics pads with during training; must match at inference
MAX_DETECTIONS = 100


@dataclass(frozen=True)
class Detection:
    """One defect found in an image, in the pixel coordinates of that image."""

    label: str
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 4 if k == "confidence" else 1) if isinstance(v, float) else v) for k, v in d.items()}


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize to `size` x `size` keeping the aspect ratio, padding the rest.

    Stretching instead would change the shape of a scratch on a non-square
    image and the model has never seen stretched defects. Returns the padded
    image, the scale applied, and the (left, top) padding, which are needed
    to map boxes back.
    """
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR) if (new_w, new_h) != (w, h) else image
    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2
    # The -0.1/+0.1 rounding puts an odd leftover pixel on the bottom/right, as the trainer does.
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    return padded, scale, (float(left), float(top))


def unletterbox(boxes: np.ndarray, scale: float, pad: tuple[float, float], shape: tuple[int, int]) -> np.ndarray:
    """Map [N,4] boxes from model input pixels back to the original image and clip them."""
    out = boxes.astype(np.float64).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / scale
    h, w = shape
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Non-maximum suppression: keep the best box, drop the ones that overlap it, repeat."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while len(order):
        best = order[0]
        keep.append(int(best))
        if len(order) == 1:
            break
        overlap = iou_matrix(boxes[best : best + 1], boxes[order[1:]])[0]
        order = order[1:][overlap <= iou_threshold]
    return np.array(keep, dtype=int)


def decode(output: np.ndarray, confidence: float, iou_threshold: float = NMS_IOU) -> np.ndarray:
    """Raw YOLO output [4 + classes, anchors] -> [N, 6] boxes in model input pixels.

    Each column is one candidate: centre x, centre y, width, height, then one
    score per class. A candidate keeps its best class only, so the inspector
    never sees two names on the same box. Suppression then runs separately
    per class: an inclusion lying inside a patch must not erase it.
    """
    pred = output.T  # [anchors, 4 + classes]
    cls_all = pred[:, 4:].argmax(axis=1)
    conf_all = pred[np.arange(len(pred)), 4 + cls_all]
    anchor = np.where(conf_all >= confidence)[0]
    if len(anchor) == 0:
        return np.zeros((0, 6))
    cls, conf = cls_all[anchor], conf_all[anchor]
    cx, cy, w, h = pred[anchor, 0], pred[anchor, 1], pred[anchor, 2], pred[anchor, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    kept = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        kept.append(idx[nms(boxes[idx], conf[idx], iou_threshold)])
    kept = np.concatenate(kept)
    kept = kept[np.argsort(-conf[kept], kind="stable")][:MAX_DETECTIONS]
    return np.column_stack([boxes[kept], conf[kept], cls[kept]]).astype(np.float64)


class Detector:
    """Loads the ONNX file once and detects defects in images."""

    def __init__(self, model_path: Path | str | None = None, threads: int | None = None) -> None:
        path = Path(model_path or MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"model not found: {path}")
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])
        self.class_names = self._names_from_model() or CLASS_NAMES
        self.path = path

    def _names_from_model(self) -> tuple[str, ...] | None:
        # The exporter writes the class names into the file's metadata. Reading
        # them from there means a retrained model with other classes just works.
        raw = self.session.get_modelmeta().custom_metadata_map.get("names")
        if not raw:
            return None
        names = ast.literal_eval(raw)
        return tuple(names[i] for i in sorted(names))

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        padded, scale, pad = letterbox(image, self.size)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(blob), scale, pad

    def predict(self, image: np.ndarray, confidence: float = 0.25, iou_threshold: float = NMS_IOU) -> np.ndarray:
        """Detections as an [N, 6] array (x1, y1, x2, y2, score, class) in image pixels."""
        if image is None or image.size == 0:
            raise ValueError("empty image")
        blob, scale, pad = self.preprocess(image)
        output = self.session.run(None, {self.input_name: blob})[0][0]
        boxes = decode(output, confidence, iou_threshold)
        if len(boxes):
            boxes[:, :4] = unletterbox(boxes[:, :4], scale, pad, image.shape[:2])
        return boxes

    def detect(self, image: np.ndarray, confidence: float = 0.25) -> list[Detection]:
        return [
            Detection(self.class_names[int(c)], int(c), float(s), float(x1), float(y1), float(x2), float(y2))
            for x1, y1, x2, y2, s, c in self.predict(image, confidence)
        ]
