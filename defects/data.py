"""Dataset preparation: Pascal VOC to YOLO labels, and a reproducible split.

The public NEU-DET release stores one XML file per image with pixel boxes.
YOLO training wants one text file per image with normalised centre boxes.
The conversion is kept here, tested, and independent of the training library
so the same labels can feed any detector.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import CLASS_NAMES, SPLIT_FRACTIONS, SPLIT_SEED


@dataclass(frozen=True)
class Box:
    """One labelled defect in pixel coordinates (left, top, right, bottom)."""

    cls: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Annotation:
    """All labelled defects of one image."""

    stem: str
    width: int
    height: int
    boxes: tuple[Box, ...]


def parse_voc(xml_path: Path, class_names: tuple[str, ...] = CLASS_NAMES) -> Annotation:
    """Read one Pascal VOC file. Boxes are clipped to the image and must have area."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path.name}: no <size> element")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"{xml_path.name}: invalid image size {width}x{height}")

    boxes: list[Box] = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in class_names:
            # An unknown label is a data error. Dropping it quietly would
            # teach the model that the defect is background.
            raise ValueError(f"{xml_path.name}: unknown class {name!r}")
        bnd = obj.find("bndbox")
        if bnd is None:
            raise ValueError(f"{xml_path.name}: object without <bndbox>")
        x1, y1, x2, y2 = (float(bnd.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, min(x1, x2)), min(float(width), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(height), max(y1, y2))
        if not (x2 > x1 and y2 > y1):
            raise ValueError(f"{xml_path.name}: box with no area")
        boxes.append(Box(class_names.index(name), x1, y1, x2, y2))
    return Annotation(xml_path.stem, width, height, tuple(boxes))


def to_yolo_lines(ann: Annotation) -> list[str]:
    """YOLO label lines: `class cx cy w h`, all normalised to the image size."""
    lines = []
    for b in ann.boxes:
        cx = (b.x1 + b.x2) / 2 / ann.width
        cy = (b.y1 + b.y2) / 2 / ann.height
        w = (b.x2 - b.x1) / ann.width
        h = (b.y2 - b.y1) / ann.height
        lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def read_yolo_labels(path: Path, width: int, height: int) -> list[Box]:
    """Inverse of `to_yolo_lines`: back to pixel boxes for scoring and drawing."""
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = (float(p) for p in parts[1:])
        boxes.append(
            Box(cls, (cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height)
        )
    return boxes


def group_of(stem: str) -> str:
    """NEU file names are `<defect type>_<number>`; the type is the split stratum."""
    return stem.rsplit("_", 1)[0]


def stratified_split(
    stems: list[str],
    fractions: dict[str, float] = SPLIT_FRACTIONS,
    seed: int = SPLIT_SEED,
) -> dict[str, list[str]]:
    """Split image names into train/val/test with the same class mix in each part.

    Sorting before shuffling makes the result depend only on the seed and the
    set of names, not on the order the file system lists them in.
    """
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    by_group: dict[str, list[str]] = defaultdict(list)
    for stem in sorted(stems):
        by_group[group_of(stem)].append(stem)

    rng = random.Random(seed)
    names = list(fractions)
    out: dict[str, list[str]] = {name: [] for name in names}
    for group in sorted(by_group):
        members = by_group[group]
        rng.shuffle(members)
        start = 0
        for i, name in enumerate(names):
            # The last part takes the remainder so no image is lost to rounding.
            end = len(members) if i == len(names) - 1 else start + round(fractions[name] * len(members))
            out[name].extend(members[start:end])
            start = end
    return {name: sorted(v) for name, v in out.items()}


def load_split(split: str, dataset_dir: Path | None = None) -> list[tuple[Path, "np.ndarray"]]:
    """Image paths of one split with their labelled boxes as [M, 5] pixel arrays."""
    import cv2
    import numpy as np

    from .config import DATASET_DIR

    root = dataset_dir or DATASET_DIR
    items = []
    for img_path in sorted((root / "images" / split).glob("*.jpg")):
        h, w = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).shape[:2]
        boxes = read_yolo_labels(root / "labels" / split / f"{img_path.stem}.txt", w, h)
        gt = np.array([[b.x1, b.y1, b.x2, b.y2, b.cls] for b in boxes], dtype=np.float64).reshape(-1, 5)
        items.append((img_path, gt))
    if not items:
        raise FileNotFoundError(f"no images in {root / 'images' / split}; run scripts/download_data.py first")
    return items
