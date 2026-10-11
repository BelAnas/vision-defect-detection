import csv

import numpy as np
import pytest

from defects.detector import Detection
from defects.reviews import save_review, summary
from defects.service import REJECT, Inspection

IMAGE = np.full((50, 60, 3), 90, np.uint8)
INSPECTION = Inspection(REJECT, "Defect found: 1 x scratches.", [Detection("scratches", 5, 0.91, 1, 2, 30, 40)], [True], 12.0, (60, 50))


def test_review_is_logged_with_its_image(tmp_path):
    log = save_review(IMAGE, INSPECTION, "false alarm", "  oil stain  ", tmp_path)
    rows = list(csv.DictReader(log.open()))
    assert len(rows) == 1
    assert rows[0]["model_verdict"] == "reject" and rows[0]["judgement"] == "false alarm" and rows[0]["note"] == "oil stain"
    assert '"label": "scratches"' in rows[0]["detections"]
    assert (tmp_path / "images" / rows[0]["image"]).exists()


def test_same_image_is_stored_once(tmp_path):
    save_review(IMAGE, INSPECTION, "correct", review_dir=tmp_path)
    save_review(IMAGE, INSPECTION, "correct", review_dir=tmp_path)
    assert len(list((tmp_path / "images").iterdir())) == 1
    assert summary(tmp_path)["reviews"] == 2


def test_unknown_judgement_is_refused(tmp_path):
    with pytest.raises(ValueError):
        save_review(IMAGE, INSPECTION, "maybe", review_dir=tmp_path)
    assert not (tmp_path / "reviews.csv").exists()


def test_summary_counts_agreement(tmp_path):
    for j in ("correct", "correct", "correct", "missed a defect"):
        save_review(IMAGE, INSPECTION, j, review_dir=tmp_path)
    s = summary(tmp_path)
    assert s["by_judgement"]["correct"] == 3 and s["by_judgement"]["missed a defect"] == 1
    assert s["agreement"] == 0.75
