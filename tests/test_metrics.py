import numpy as np
import pytest

from defects import metrics as M


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(M.iou_matrix(a, b)[0], [1.0, 1 / 3, 0.0])


def test_iou_empty_inputs():
    assert M.iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


def test_duplicate_detection_is_a_false_alarm():
    gt = np.array([[0, 0, 10, 10, 0]])
    pred = np.array([[0, 0, 10, 10, 0.9, 0], [1, 0, 10, 10, 0.8, 0]])
    tp = M.match_image(pred, gt)
    assert tp[0, 0] and not tp[1, 0]


def test_perfect_detector_scores_one():
    gts = [np.array([[0, 0, 10, 10, 0], [20, 20, 40, 40, 1]]), np.array([[5, 5, 15, 15, 1]])]
    preds = [np.array([[0, 0, 10, 10, 0.9, 0], [20, 20, 40, 40, 0.8, 1]]), np.array([[5, 5, 15, 15, 0.7, 1]])]
    s = M.summarise(M.collect(preds, gts, 2))
    assert s["map50"] == pytest.approx(1.0)
    assert s["map50_95"] == pytest.approx(1.0)


def test_average_precision_hand_computed():
    # Ranked hits: TP, FP, TP with 2 labelled boxes.
    # Recall reaches 0.5 at precision 1, and 1.0 at precision 2/3.
    ap = M.average_precision(np.array([0.9, 0.8, 0.7]), np.array([True, False, True]), 2)
    expected = (51 * 1.0 + 50 * (2 / 3)) / 101
    assert ap == pytest.approx(expected)


def test_missed_defect_caps_recall():
    ap = M.average_precision(np.array([0.9]), np.array([True]), 2)
    assert ap == pytest.approx(51 / 101)


def test_no_predictions_scores_zero_and_no_labels_is_nan():
    assert M.average_precision(np.zeros(0), np.zeros(0, dtype=bool), 3) == 0.0
    assert np.isnan(M.average_precision(np.array([0.5]), np.array([False]), 0))


def test_loose_box_counts_at_low_iou_only():
    gt = [np.array([[0, 0, 10, 10, 0]])]
    pred = [np.array([[0, 0, 10, 7, 0.9, 0]])]  # IoU 0.7
    s = M.summarise(M.collect(pred, gt, 1))
    assert s["map50"] == pytest.approx(1.0)
    assert s["map50_95"] == pytest.approx(0.5)  # hits at 0.50..0.70, misses at 0.75..0.95


def test_at_threshold_counts():
    gts = [np.array([[0, 0, 10, 10, 0], [20, 20, 30, 30, 0]])]
    preds = [np.array([[0, 0, 10, 10, 0.9, 0], [50, 50, 60, 60, 0.6, 0], [20, 20, 30, 30, 0.3, 0]])]
    rec = M.collect(preds, gts, 1)[0]
    assert M.at_threshold(rec, 0.5) == {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5}
    low = M.at_threshold(rec, 0.1)
    assert (low["tp"], low["fp"], low["fn"]) == (2, 1, 0)


def test_best_f1_threshold_separates_hits_from_false_alarms():
    gts = [np.array([[0, 0, 10, 10, 0]])]
    preds = [np.array([[0, 0, 10, 10, 0.8, 0], [50, 50, 60, 60, 0.3, 0]])]
    thr = M.best_f1_threshold(M.collect(preds, gts, 1)[0])
    assert 0.3 < thr <= 0.8


def test_class_agnostic_ignores_wrong_name():
    gts = [np.array([[0, 0, 10, 10, 0]])]
    preds = [np.array([[0, 0, 10, 10, 0.9, 1]])]
    assert M.summarise(M.collect(preds, gts, 2))["map50"] == 0.0
    assert M.summarise(M.collect(preds, gts, 2, class_agnostic=True))["map50"] == pytest.approx(1.0)


def test_bootstrap_weights_match_explicit_duplication():
    gts = [np.array([[0, 0, 10, 10, 0]]), np.array([[0, 0, 10, 10, 0]])]
    preds = [np.array([[0, 0, 10, 10, 0.9, 0]]), np.array([[50, 50, 60, 60, 0.95, 0]])]
    weighted = M.summarise(M.collect(preds, gts, 1), np.array([2, 1]))
    explicit = M.summarise(M.collect([preds[0], preds[0], preds[1]], [gts[0], gts[0], gts[1]], 1))
    assert weighted["map50"] == pytest.approx(explicit["map50"])


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    gts, preds = [], []
    for _ in range(40):
        gts.append(np.array([[0, 0, 10, 10, 0]]))
        hit = rng.random() < 0.7
        preds.append(np.array([[0, 0, 10, 10, rng.random(), 0]]) if hit else np.array([[50, 50, 60, 60, rng.random(), 0]]))
    rec = M.collect(preds, gts, 1)
    lo, hi = M.bootstrap_interval(rec, 40, n_boot=200)
    assert lo <= M.summarise(rec)["map50"] <= hi


def test_confusion_matrix_places_errors():
    gts = [np.array([[0, 0, 10, 10, 0], [20, 20, 30, 30, 1], [40, 40, 50, 50, 1]])]
    preds = [np.array([[0, 0, 10, 10, 0.9, 1], [20, 20, 30, 30, 0.9, 1], [70, 70, 80, 80, 0.9, 0]])]
    m = M.confusion_matrix(preds, gts, 2, 0.5)
    assert m[0, 1] == 1  # found, wrong name
    assert m[1, 1] == 1  # correct
    assert m[1, 2] == 1  # missed
    assert m[2, 0] == 1  # false alarm
    assert m.sum() == 4
