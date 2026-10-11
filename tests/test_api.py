import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from defects import app as app_module
from defects.config import CLASS_NAMES, SAMPLES_DIR


@pytest.fixture(scope="module")
def client():
    return TestClient(app_module.create_api())


def test_health_reports_model_and_thresholds(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["classes"] == list(CLASS_NAMES)
    assert body["runtime"] == "onnxruntime-cpu"
    assert set(CLASS_NAMES) < set(body["thresholds"])


def test_detect_returns_verdict_and_boxes(client):
    with (SAMPLES_DIR / "patches_109.jpg").open("rb") as f:
        r = client.post("/api/detect", files={"file": ("patches_109.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["file"] == "patches_109.jpg"
    assert body["verdict"] in ("pass", "review", "reject")
    for d in body["detections"]:
        assert d["label"] in CLASS_NAMES
        assert 0 <= d["x1"] < d["x2"] <= 200 and 0 <= d["y1"] < d["y2"] <= 200
        assert isinstance(d["confident"], bool)


def test_png_upload_works(client):
    ok, png = cv2.imencode(".png", np.full((120, 160, 3), 127, np.uint8))
    r = client.post("/api/detect", files={"file": ("blank.png", png.tobytes(), "image/png")})
    assert r.status_code == 200
    assert r.json()["image"] == {"width": 160, "height": 120}


def test_non_image_is_rejected_with_a_clear_message(client):
    r = client.post("/api/detect", files={"file": ("notes.txt", b"not an image", "text/plain")})
    assert r.status_code == 415
    assert "not an image" in r.json()["detail"]


def test_empty_upload_is_rejected(client):
    assert client.post("/api/detect", files={"file": ("empty.jpg", b"", "image/jpeg")}).status_code == 400


def test_oversized_upload_is_rejected(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 10)
    r = client.post("/api/detect", files={"file": ("big.jpg", b"x" * 11, "image/jpeg")})
    assert r.status_code == 413


def test_missing_file_field_is_a_validation_error(client):
    assert client.post("/api/detect").status_code == 422


def test_sensitivity_out_of_range_is_rejected(client):
    with (SAMPLES_DIR / "patches_109.jpg").open("rb") as f:
        r = client.post("/api/detect", files={"file": ("a.jpg", f, "image/jpeg")}, data={"sensitivity": "9"})
    assert r.status_code == 422


def test_review_summary_endpoint(client, tmp_path, monkeypatch):
    monkeypatch.setattr("defects.reviews.REVIEW_DIR", tmp_path)
    assert client.get("/api/reviews/summary").json() == {
        "reviews": 0,
        "by_judgement": {"correct": 0, "missed a defect": 0, "false alarm": 0, "wrong defect type": 0},
        "agreement": None,
    }


def test_review_screen_builds_and_mounts():
    app = app_module.create_app()
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/api/detect" in paths
