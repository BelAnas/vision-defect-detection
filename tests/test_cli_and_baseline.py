import json

import cv2
import numpy as np

from defects.baseline import detect_classical
from defects.cli import main
from defects.config import SAMPLES_DIR
from defects.metrics import iou_matrix


def test_cli_inspects_a_folder(capsys):
    assert main([str(SAMPLES_DIR)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 12
    assert all(any(v in line for v in ("PASS", "REVIEW", "REJECT")) for line in lines)


def test_cli_json_and_saved_images(capsys, tmp_path):
    assert main([str(SAMPLES_DIR / "inclusion_102.jpg"), "--json", "--save", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["file"] == "inclusion_102.jpg" and "verdict" in payload
    assert cv2.imread(str(tmp_path / "inclusion_102.png")).shape == (600, 600, 3)


def test_cli_missing_path_fails(capsys, tmp_path):
    assert main([str(tmp_path / "nothing.jpg")]) == 2


def test_cli_reports_an_unreadable_file(capsys, tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not a jpeg")
    assert main([str(bad)]) == 1
    assert "not a readable image" in capsys.readouterr().err


def test_classical_baseline_finds_an_obvious_blob():
    image = np.full((200, 200), 150, np.uint8)
    cv2.rectangle(image, (60, 80), (120, 130), 40, -1)
    boxes = detect_classical(image, sigma=20, kernel=5, min_area_frac=0.02)
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0, :4]
    # The blur leaves a halo, so the box is looser than the blob: it must
    # contain it and still overlap it substantially.
    assert x1 <= 60 and y1 <= 80 and x2 >= 120 and y2 >= 130
    assert iou_matrix(boxes[:, :4], np.array([[60, 80, 120, 130]])).max() > 0.4


def test_classical_baseline_is_quiet_on_a_flat_image():
    assert len(detect_classical(np.full((200, 200), 150, np.uint8), sigma=20, kernel=5, min_area_frac=0.02)) == 0
