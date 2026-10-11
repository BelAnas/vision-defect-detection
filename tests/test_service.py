import json

import numpy as np
import pytest

from defects.detector import Detection
from defects.service import PASS, REJECT, REVIEW, Inspection, Thresholds, draw, verdict_for

T = Thresholds((0.5, 0.6), 0.1)


def boxes(*rows):
    return np.array([[0, 0, 10, 10, s, c] for s, c in rows], dtype=float).reshape(-1, 6)


def test_no_boxes_passes():
    assert verdict_for(boxes(), T)[0] == PASS


def test_boxes_below_the_review_floor_are_ignored():
    assert verdict_for(boxes((0.05, 0)), T)[0] == PASS


def test_low_confidence_box_goes_to_review():
    verdict, confident = verdict_for(boxes((0.3, 0)), T)
    assert verdict == REVIEW and confident.tolist() == [False]


def test_confident_box_rejects():
    verdict, confident = verdict_for(boxes((0.3, 0), (0.7, 1)), T)
    assert verdict == REJECT and confident.tolist() == [False, True]


def test_threshold_is_per_class():
    # 0.55 is enough for class 0 (0.5) but not for class 1 (0.6).
    assert verdict_for(boxes((0.55, 0)), T)[0] == REJECT
    assert verdict_for(boxes((0.55, 1)), T)[0] == REVIEW


def test_thresholds_default_when_no_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DEFECTS_CONFIDENCE", raising=False)
    t = Thresholds.load(("a", "b"), tmp_path / "missing.json")
    assert t.per_class == (0.25, 0.25) and t.review_floor == 0.10


def test_thresholds_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DEFECTS_CONFIDENCE", raising=False)
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"per_class": {"a": 0.4}, "review_floor": 0.2}))
    t = Thresholds.load(("a", "b"), p)
    assert t.per_class == (0.4, 0.25) and t.review_floor == 0.2


def test_environment_overrides_thresholds(tmp_path, monkeypatch):
    monkeypatch.setenv("DEFECTS_CONFIDENCE", "0.7")
    assert Thresholds.load(("a", "b"), tmp_path / "missing.json").per_class == (0.7, 0.7)


def test_scaling_makes_the_rule_stricter_or_more_sensitive():
    assert T.scaled(1.5).per_class == pytest.approx((0.75, 0.9))
    sensitive = T.scaled(0.1)
    assert sensitive.per_class == pytest.approx((0.05, 0.06))
    assert sensitive.review_floor <= min(sensitive.per_class)


def test_inspection_on_a_sample(inspector, samples):
    image, _ = samples["scratches_101"]
    result = inspector.inspect(image)
    assert result.verdict in (PASS, REVIEW, REJECT)
    assert len(result.detections) == len(result.confident)
    payload = result.to_dict()
    json.dumps(payload)  # must be serialisable as is
    assert payload["image"] == {"width": 200, "height": 200}
    assert "review_floor" in payload["thresholds"]


def test_lower_multiplier_never_flags_less(inspector, samples):
    order = {PASS: 0, REVIEW: 1, REJECT: 2}
    for image, _ in samples.values():
        strict = inspector.inspect(image, 1.5)
        sensitive = inspector.inspect(image, 0.5)
        assert order[sensitive.verdict] >= order[strict.verdict]


def test_reason_names_what_was_found():
    from defects.service import _reason

    dets = [Detection("scratches", 5, 0.9, 0, 0, 1, 1), Detection("scratches", 5, 0.8, 0, 0, 1, 1), Detection("patches", 2, 0.2, 0, 0, 1, 1)]
    assert _reason(REJECT, dets, np.array([True, True, False])) == "Defect found: 2 x scratches."
    assert _reason(PASS, [], np.zeros(0, bool)) == "No defect found."
    assert "A person should look" in _reason(REVIEW, dets[2:], np.array([False]))


def test_draw_enlarges_small_images_and_changes_pixels():
    image = np.full((200, 200), 128, np.uint8)
    insp = Inspection(REJECT, "", [Detection("scratches", 5, 0.9, 20, 20, 120, 80)], [True], 1.0, (200, 200))
    out = draw(image, insp)
    assert out.shape == (600, 600, 3)
    assert (out != 128).any()
