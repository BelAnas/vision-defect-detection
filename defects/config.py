"""Paths and constants shared by every module.

Everything that a deployment may want to change is read from the environment,
so the same code runs in tests, on a laptop and behind a line camera.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The class order is the label order inside the trained model. Changing it
# silently renames every detection, so it lives in exactly one place.
CLASS_NAMES: tuple[str, ...] = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)

DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "neu"
RAW_DIR = DATASET_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
SAMPLES_DIR = DATA_DIR / "samples"
REPORTS_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"

# The dataset mirror is pinned to one commit so a later upload cannot change
# the images under a published result.
HF_DATASET = "KeenForgeAI/NEU-DET-corrected"
HF_REVISION = "cbb799e5e5b8198d6228d94778f835082e42883c"

SPLIT_SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

MODEL_PATH = Path(os.environ.get("DEFECTS_MODEL") or ROOT / "model" / "defects.onnx")
THRESHOLDS_PATH = ROOT / "model" / "thresholds.json"
REVIEW_DIR = Path(os.environ.get("DEFECTS_REVIEW_DIR") or ROOT / "reviews")

# Boxes of the same class that overlap more than this are treated as one defect.
NMS_IOU = 0.5
