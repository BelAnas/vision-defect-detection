import cv2
import numpy as np
import pytest

from defects.config import CLASS_NAMES
from defects.data import group_of
from defects.detector import Detector, decode, letterbox, nms, unletterbox
from defects.metrics import iou_matrix


def test_letterbox_square_image_is_only_resized():
    padded, scale, pad = letterbox(np.zeros((200, 200, 3), np.uint8), 320)
    assert padded.shape == (320, 320, 3)
    assert scale == pytest.approx(1.6) and pad == (0.0, 0.0)


def test_letterbox_wide_image_is_padded_top_and_bottom():
    padded, scale, pad = letterbox(np.zeros((100, 400, 3), np.uint8), 320)
    assert padded.shape == (320, 320, 3)
    assert scale == pytest.approx(0.8) and pad == (0.0, 120.0)
    assert (padded[0, 0] == 114).all() and (padded[160, 160] == 0).all()


def test_unletterbox_inverts_letterbox():
    _, scale, pad = letterbox(np.zeros((100, 400, 3), np.uint8), 320)
    original = np.array([[40.0, 10.0, 200.0, 90.0]])
    in_model = original * scale + np.array([pad[0], pad[1], pad[0], pad[1]])
    np.testing.assert_allclose(unletterbox(in_model, scale, pad, (100, 400)), original)


def test_unletterbox_clips_to_image():
    out = unletterbox(np.array([[-20.0, -20.0, 500.0, 500.0]]), 1.0, (0.0, 0.0), (100, 200))
    assert out.tolist() == [[0, 0, 200, 100]]


def test_nms_keeps_best_of_overlapping_and_all_separate():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    keep = nms(boxes, np.array([0.6, 0.9, 0.5]), 0.5)
    assert keep.tolist() == [1, 2]


def _raw(candidates: list[tuple], n_classes: int = 6) -> np.ndarray:
    """Build a fake model output [4 + classes, anchors] from (cx, cy, w, h, class, score)."""
    out = np.zeros((4 + n_classes, len(candidates)), np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(candidates):
        out[:4, i] = (cx, cy, w, h)
        out[4 + c, i] = s
    return out


def test_decode_converts_centre_boxes_and_filters_by_confidence():
    boxes = decode(_raw([(100, 100, 40, 20, 5, 0.9), (200, 200, 10, 10, 1, 0.05)]), 0.25)
    assert boxes.shape == (1, 6)
    np.testing.assert_allclose(boxes[0], [80, 90, 120, 110, 0.9, 5], rtol=1e-5)


def test_decode_suppresses_within_a_class_but_not_across_classes():
    raw = _raw([(100, 100, 40, 40, 2, 0.9), (102, 100, 40, 40, 2, 0.8), (100, 100, 40, 40, 1, 0.7)])
    boxes = decode(raw, 0.25)
    assert sorted(boxes[:, 5].tolist()) == [1, 2]
    assert boxes[0, 4] == pytest.approx(0.9)  # sorted by falling confidence


def test_decode_empty():
    assert decode(_raw([(10, 10, 5, 5, 0, 0.01)]), 0.25).shape == (0, 6)


def test_model_carries_the_class_names(detector):
    assert detector.class_names == CLASS_NAMES
    assert detector.size == 320


def test_predictions_stay_inside_the_image(detector, samples):
    for image, _ in samples.values():
        boxes = detector.predict(image, 0.05)
        h, w = image.shape[:2]
        assert boxes.shape[1] == 6
        assert (boxes[:, [0, 2]] >= 0).all() and (boxes[:, [0, 2]] <= w).all()
        assert (boxes[:, [1, 3]] >= 0).all() and (boxes[:, [1, 3]] <= h).all()
        assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()


def test_grey_and_bgra_inputs_give_the_same_boxes(detector, samples):
    image = next(iter(samples.values()))[0]
    ref = detector.predict(image, 0.25)
    grey = detector.predict(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0.25)
    bgra = detector.predict(cv2.cvtColor(image, cv2.COLOR_BGR2BGRA), 0.25)
    np.testing.assert_allclose(grey, ref, atol=1e-3)
    np.testing.assert_allclose(bgra, ref, atol=1e-3)


def test_non_square_image_is_handled(detector, samples):
    image = next(iter(samples.values()))[0]
    wide = np.hstack([image, image])  # 400 x 200
    boxes = detector.predict(wide, 0.25)
    assert (boxes[:, 2] <= 400).all() and (boxes[:, 3] <= 200).all()


def test_empty_image_is_rejected(detector):
    with pytest.raises(ValueError):
        detector.predict(np.zeros((0, 0, 3), np.uint8))


def test_missing_model_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Detector(tmp_path / "nope.onnx")


def test_model_finds_the_labelled_defect_on_most_samples(detector, samples):
    """A regression guard on the shipped model, not a benchmark: on the twelve
    sample images it must find a labelled box, with the right name, on most."""
    hits = 0
    for stem, (image, labelled) in samples.items():
        boxes = detector.predict(image, 0.25)
        gt = np.array(labelled, dtype=float)
        wanted = CLASS_NAMES.index(group_of(stem))
        same_class = boxes[boxes[:, 5] == wanted]
        if len(same_class) and iou_matrix(same_class[:, :4], gt[:, :4]).max() >= 0.5:
            hits += 1
    assert hits >= 9, f"only {hits} of 12 samples had a correct detection"
