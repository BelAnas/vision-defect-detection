from pathlib import Path

import pytest

from defects.config import CLASS_NAMES, SPLITS_DIR
from defects.data import Annotation, Box, parse_voc, read_yolo_labels, stratified_split, to_yolo_lines

VOC = """<annotation><filename>scratches_1.jpg</filename>
<size><width>200</width><height>200</height><depth>1</depth></size>
{objects}</annotation>"""
OBJ = "<object><name>{name}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"


def write(tmp_path: Path, *objects: str) -> Path:
    p = tmp_path / "scratches_1.xml"
    p.write_text(VOC.format(objects="".join(objects)))
    return p


def test_parse_voc_reads_boxes(tmp_path):
    ann = parse_voc(write(tmp_path, OBJ.format(name="scratches", x1=10, y1=20, x2=110, y2=60)))
    assert (ann.width, ann.height) == (200, 200)
    assert ann.boxes == (Box(CLASS_NAMES.index("scratches"), 10, 20, 110, 60),)


def test_parse_voc_clips_to_image(tmp_path):
    ann = parse_voc(write(tmp_path, OBJ.format(name="patches", x1=-5, y1=0, x2=250, y2=100)))
    assert (ann.boxes[0].x1, ann.boxes[0].x2) == (0, 200)


def test_parse_voc_rejects_unknown_class(tmp_path):
    with pytest.raises(ValueError, match="unknown class"):
        parse_voc(write(tmp_path, OBJ.format(name="dent", x1=1, y1=1, x2=5, y2=5)))


def test_parse_voc_rejects_empty_box(tmp_path):
    with pytest.raises(ValueError, match="no area"):
        parse_voc(write(tmp_path, OBJ.format(name="patches", x1=5, y1=5, x2=5, y2=50)))


def test_yolo_round_trip(tmp_path):
    ann = Annotation("a_1", 200, 100, (Box(2, 10, 20, 110, 60), Box(5, 0, 0, 200, 100)))
    lines = to_yolo_lines(ann)
    assert lines[0] == "2 0.300000 0.400000 0.500000 0.400000"
    p = tmp_path / "a_1.txt"
    p.write_text("\n".join(lines))
    back = read_yolo_labels(p, 200, 100)
    for got, want in zip(back, ann.boxes):
        assert got.cls == want.cls
        assert (got.x1, got.y1, got.x2, got.y2) == pytest.approx((want.x1, want.y1, want.x2, want.y2), abs=1e-3)


def test_split_is_deterministic_stratified_and_disjoint():
    stems = [f"{c}_{i}" for c in ("crazing", "pitted_surface") for i in range(1, 101)]
    a = stratified_split(stems)
    b = stratified_split(list(reversed(stems)))
    assert a == b
    assert sum(len(v) for v in a.values()) == 200
    assert not set(a["train"]) & set(a["val"]) and not set(a["train"]) & set(a["test"]) and not set(a["val"]) & set(a["test"])
    for part, n in (("train", 70), ("val", 15), ("test", 15)):
        assert sum(s.startswith("crazing") for s in a[part]) == n
        assert sum(s.startswith("pitted_surface") for s in a[part]) == n


def test_committed_split_has_no_leakage():
    parts = {n: set((SPLITS_DIR / f"{n}.txt").read_text().split()) for n in ("train", "val", "test")}
    assert len(parts["train"]) == 1257 and len(parts["val"]) == 270 and len(parts["test"]) == 270
    assert not parts["train"] & parts["val"]
    assert not parts["train"] & parts["test"]
    assert not parts["val"] & parts["test"]
