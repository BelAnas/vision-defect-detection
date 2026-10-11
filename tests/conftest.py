import json

import cv2
import pytest

from defects.config import SAMPLES_DIR
from defects.detector import Detector
from defects.service import Inspector


@pytest.fixture(scope="session")
def detector() -> Detector:
    return Detector()


@pytest.fixture(scope="session")
def inspector(detector) -> Inspector:
    return Inspector(detector)


@pytest.fixture(scope="session")
def samples() -> dict:
    """The twelve committed test images with their labelled boxes."""
    labels = json.loads((SAMPLES_DIR / "labels.json").read_text())
    return {stem: (cv2.imread(str(SAMPLES_DIR / f"{stem}.jpg")), boxes) for stem, boxes in labels.items()}
