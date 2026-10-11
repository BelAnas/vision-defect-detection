"""Download the NEU steel surface defect dataset and prepare it for training.

    python scripts/download_data.py

Result, under data/neu/ (not committed):
    raw/                      the files as published
    images/{train,val,test}/  200x200 grey JPG
    labels/{train,val,test}/  YOLO labels converted from the Pascal VOC XML
    data.yaml                 the dataset description Ultralytics reads

The split is written to data/splits/*.txt, which IS committed, so every
number in the README refers to a known list of images.
"""

from __future__ import annotations

import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from defects.config import (  # noqa: E402
    CLASS_NAMES,
    DATASET_DIR,
    HF_DATASET,
    HF_REVISION,
    RAW_DIR,
    SPLITS_DIR,
)
from defects.data import group_of, parse_voc, stratified_split, to_yolo_lines  # noqa: E402


def download() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        HF_DATASET,
        repo_type="dataset",
        revision=HF_REVISION,
        local_dir=RAW_DIR,
        allow_patterns=["images/**", "annotations/**", "LICENSE", "README.md"],
    )


def prepare() -> dict[str, int]:
    images = {p.stem: p for p in RAW_DIR.glob("images/*/*.jpg")}
    annotations = {p.stem: parse_voc(p) for p in sorted(RAW_DIR.glob("annotations/*.xml"))}
    if set(images) != set(annotations):
        raise SystemExit("images and annotations do not pair up; the download is incomplete")

    split = stratified_split(list(images))
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, stems in split.items():
        (SPLITS_DIR / f"{name}.txt").write_text("\n".join(stems) + "\n")
        img_dir, lbl_dir = DATASET_DIR / "images" / name, DATASET_DIR / "labels" / name
        for d in (img_dir, lbl_dir):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
        for stem in stems:
            shutil.copyfile(images[stem], img_dir / f"{stem}.jpg")
            (lbl_dir / f"{stem}.txt").write_text("\n".join(to_yolo_lines(annotations[stem])) + "\n")
        counts[name] = len(stems)
        per_class = Counter(group_of(s) for s in stems)
        boxes = sum(len(annotations[s].boxes) for s in stems)
        print(f"{name:5s} {len(stems):5d} images {boxes:5d} boxes  " + " ".join(f"{k}={v}" for k, v in sorted(per_class.items())))

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (DATASET_DIR / "data.yaml").write_text(
        f"path: {DATASET_DIR}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n"
    )
    return counts


if __name__ == "__main__":
    download()
    prepare()
