"""Detection metrics, written out in NumPy so the scoring is not a black box.

Conventions used everywhere in this project:
    predictions per image: float array [N, 6] = x1, y1, x2, y2, score, class
    ground truth per image: float array [M, 5] = x1, y1, x2, y2, class

Average precision follows the COCO recipe (101 recall points, each detection
matched to at most one labelled box, highest score first).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over union of every box in `a` [N,4] with every box in `b` [M,4]."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match_image(pred: np.ndarray, gt: np.ndarray, iou_thresholds: np.ndarray = IOU_THRESHOLDS) -> np.ndarray:
    """True-positive flags [N, T] for the predictions of one image and one class.

    Predictions are taken in order of falling score and each claims the
    best still-unclaimed labelled box. A second box on the same defect is
    therefore a false alarm, which is what an inspector would call it too.
    """
    tp = np.zeros((len(pred), len(iou_thresholds)), dtype=bool)
    if len(pred) == 0 or len(gt) == 0:
        return tp
    ious = iou_matrix(pred[:, :4], gt[:, :4])
    order = np.argsort(-pred[:, 4], kind="stable")
    for t, thr in enumerate(iou_thresholds):
        taken = np.zeros(len(gt), dtype=bool)
        for i in order:
            candidates = np.where(~taken, ious[i], -1.0)
            j = int(candidates.argmax())
            if candidates[j] >= thr:
                taken[j] = True
                tp[i, t] = True
    return tp


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """Area under the precision-recall curve, sampled at 101 recall points."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    hits = tp[order].astype(np.float64)
    cum_tp = np.cumsum(hits)
    cum_fp = np.cumsum(1.0 - hits)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    # Make precision non-increasing: at a given recall, the achievable
    # precision is the best one at that recall or any higher one.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    sampled = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(sampled.mean())


@dataclass
class ClassRecords:
    """Everything needed to recompute the metrics of one class at any threshold."""

    scores: np.ndarray  # [N]
    tp: np.ndarray  # [N, T]
    image: np.ndarray  # [N] index of the image each prediction came from
    gt_per_image: np.ndarray  # [images] number of labelled boxes of this class

    @property
    def n_gt(self) -> int:
        return int(self.gt_per_image.sum())


def collect(
    preds: list[np.ndarray], gts: list[np.ndarray], n_classes: int, class_agnostic: bool = False
) -> list[ClassRecords]:
    """Match every image once; all later numbers are computed from these records."""
    if class_agnostic:
        preds = [np.column_stack([p[:, :5], np.zeros(len(p))]) if len(p) else p.reshape(0, 6) for p in preds]
        gts = [np.column_stack([g[:, :4], np.zeros(len(g))]) if len(g) else g.reshape(0, 5) for g in gts]
        n_classes = 1
    records = []
    for c in range(n_classes):
        scores, tps, image_ids = [], [], []
        gt_counts = np.zeros(len(gts), dtype=int)
        for i, (p, g) in enumerate(zip(preds, gts)):
            p = np.asarray(p, dtype=np.float64).reshape(-1, 6)
            g = np.asarray(g, dtype=np.float64).reshape(-1, 5)
            pc, gc = p[p[:, 5] == c], g[g[:, 4] == c]
            gt_counts[i] = len(gc)
            if len(pc):
                scores.append(pc[:, 4])
                tps.append(match_image(pc, gc))
                image_ids.append(np.full(len(pc), i))
        records.append(
            ClassRecords(
                np.concatenate(scores) if scores else np.zeros(0),
                np.concatenate(tps) if tps else np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool),
                np.concatenate(image_ids) if image_ids else np.zeros(0, dtype=int),
                gt_counts,
            )
        )
    return records


def summarise(records: list[ClassRecords], image_mask: np.ndarray | None = None) -> dict:
    """mAP@0.5 and mAP@0.5:0.95 plus the per-class AP. `image_mask` gives per-image
    weights (0, 1, 2, ...) so a bootstrap resample can reuse the same matches."""
    ap50, ap = [], []
    for r in records:
        if image_mask is None:
            scores, tp, n_gt = r.scores, r.tp, r.n_gt
        else:
            reps = image_mask[r.image]
            scores, tp = np.repeat(r.scores, reps), np.repeat(r.tp, reps, axis=0)
            n_gt = int((r.gt_per_image * image_mask).sum())
        per_thr = [average_precision(scores, tp[:, t], n_gt) for t in range(tp.shape[1])]
        ap50.append(per_thr[0])
        ap.append(float(np.mean(per_thr)))
    return {
        "map50": float(np.nanmean(ap50)),
        "map50_95": float(np.nanmean(ap)),
        "ap50_per_class": ap50,
        "ap50_95_per_class": ap,
    }


def at_threshold(record: ClassRecords, confidence: float) -> dict:
    """Precision, recall and F1 at IoU 0.5 when only boxes above `confidence` are kept."""
    keep = record.scores >= confidence
    tp = int(record.tp[keep, 0].sum())
    fp = int(keep.sum()) - tp
    fn = record.n_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def best_f1_threshold(record: ClassRecords, grid: np.ndarray | None = None) -> float:
    """The confidence that maximises F1. Called on validation data only."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    f1 = [at_threshold(record, float(c))["f1"] for c in grid]
    return float(grid[int(np.argmax(f1))])


def bootstrap_interval(
    records: list[ClassRecords], n_images: int, key: str = "map50", n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% interval of a summary metric, from resampling the images with replacement."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_images, n_images), minlength=n_images)
        values.append(summarise(records, counts)[key])
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def confusion_matrix(
    preds: list[np.ndarray], gts: list[np.ndarray], n_classes: int, confidence: float | np.ndarray, iou: float = 0.5
) -> np.ndarray:
    """[n+1, n+1] counts, rows = labelled class, columns = predicted class.

    The extra last row is "nothing labelled here" (false alarms) and the extra
    last column is "nothing predicted here" (missed defects). Matching ignores
    the class, so a defect found with the wrong name lands off the diagonal.
    """
    conf = np.broadcast_to(np.asarray(confidence, dtype=np.float64), (n_classes,))
    m = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    for p, g in zip(preds, gts):
        p = np.asarray(p, dtype=np.float64).reshape(-1, 6)
        g = np.asarray(g, dtype=np.float64).reshape(-1, 5)
        p = p[p[:, 4] >= conf[p[:, 5].astype(int)]] if len(p) else p
        ious = iou_matrix(p[:, :4], g[:, :4])
        gt_taken = np.zeros(len(g), dtype=bool)
        for i in np.argsort(-p[:, 4], kind="stable"):
            cand = np.where(~gt_taken, ious[i], -1.0) if len(g) else np.array([-1.0])
            j = int(cand.argmax())
            if cand[j] >= iou:
                gt_taken[j] = True
                m[int(g[j, 4]), int(p[i, 5])] += 1
            else:
                m[n_classes, int(p[i, 5])] += 1
        for j in np.where(~gt_taken)[0]:
            m[int(g[j, 4]), n_classes] += 1
    return m
