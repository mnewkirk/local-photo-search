"""Tests for manual face-box editing (features 4-6): IoU dedup on re-detect,
add-box with and without a recoverable encoding, and delete.

detect_faces is monkeypatched everywhere so these run without InsightFace.
"""
import pytest
from PIL import Image

from photosearch import face_edit


# ---- pure helpers -----------------------------------------------------------

def test_iou_identical_boxes_is_one():
    b = (10, 20, 30, 5)  # top, right, bottom, left
    assert face_edit._iou(b, b) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert face_edit._iou((0, 10, 10, 0), (100, 110, 110, 100)) == 0.0


def test_iou_partial_overlap():
    # Two 10x10 boxes overlapping in a 5x5 corner → inter 25, union 175.
    a = (0, 10, 10, 0)
    b = (5, 15, 15, 5)
    assert face_edit._iou(a, b) == pytest.approx(25 / 175)


def test_norm_bbox_orientation_and_clamp():
    # Drag from bottom-right to top-left + out-of-range values get normalized.
    top, right, bottom, left = face_edit._norm_bbox(
        {"top": 0.9, "right": 0.1, "bottom": 0.2, "left": 0.8})
    assert (top, bottom) == (0.2, 0.9)
    assert (left, right) == (0.1, 0.8)
    top, right, bottom, left = face_edit._norm_bbox([-0.5, 1.5, 2.0, -1.0])
    assert top == 0.0 and left == 0.0 and right == 1.0 and bottom == 1.0


# ---- re-detect --------------------------------------------------------------

def _mk_photo(db, faces=()):
    pid = db.add_photo(filepath="2026/x.jpg", filename="x.jpg",
                       image_width=1000, image_height=800)
    for (bbox, enc) in faces:
        db.add_face(pid, bbox, enc, det_score=0.9)
    return pid


def test_redetect_augment_skips_overlapping_keeps_existing(db, monkeypatch):
    # Existing face at ~same spot as one detection; a second detection is new.
    pid = _mk_photo(db, faces=[((100, 200, 300, 100), [0.1] * 512)])
    monkeypatch.setattr(
        "photosearch.faces.detect_faces",
        lambda p: [
            {"bbox": (102, 205, 305, 98), "encoding": [0.2] * 512, "det_score": 0.8},  # dup
            {"bbox": (500, 600, 700, 500), "encoding": [0.3] * 512, "det_score": 0.7},  # new
        ])
    res = face_edit.redetect_photo_faces(db, pid, "/nope.jpg")
    assert res["mode"] == "augment"
    assert res["added"] == 1 and res["kept"] == 1
    rows = db.conn.execute("SELECT COUNT(*) FROM faces WHERE photo_id=?", (pid,)).fetchone()[0]
    assert rows == 2


def test_redetect_replace_wipes_then_adds(db, monkeypatch):
    pid = _mk_photo(db, faces=[((100, 200, 300, 100), [0.1] * 512)])
    monkeypatch.setattr(
        "photosearch.faces.detect_faces",
        lambda p: [{"bbox": (10, 20, 30, 10), "encoding": [0.4] * 512, "det_score": 0.9}])
    res = face_edit.redetect_photo_faces(db, pid, "/nope.jpg", replace=True)
    assert res["mode"] == "replace" and res["removed"] == 1 and res["added"] == 1
    ids = [r[0] for r in db.conn.execute("SELECT id FROM faces WHERE photo_id=?", (pid,)).fetchall()]
    assert len(ids) == 1


# ---- add manual box ---------------------------------------------------------

def _write_jpeg(tmp_path):
    p = tmp_path / "img.jpg"
    Image.new("RGB", (400, 300), (128, 128, 128)).save(p, "JPEG")
    return str(p)


def test_add_box_stores_even_without_detection(db, monkeypatch, tmp_path):
    pid = _mk_photo(db)
    path = _write_jpeg(tmp_path)
    monkeypatch.setattr("photosearch.faces.detect_faces", lambda p: [])  # nothing found
    photo = db.get_photo(pid)
    res = face_edit.add_manual_face(db, pid, path, photo,
                                    {"top": 0.2, "right": 0.6, "bottom": 0.7, "left": 0.3})
    assert res["ok"] and res["encoded"] is False
    row = db.conn.execute(
        "SELECT match_source FROM faces WHERE id=?", (res["face_id"],)).fetchone()
    assert row["match_source"] == "manual_box"


def test_add_box_with_detection_and_person(db, monkeypatch, tmp_path):
    pid = _mk_photo(db)
    path = _write_jpeg(tmp_path)
    monkeypatch.setattr(
        "photosearch.faces.detect_faces",
        lambda p: [{"bbox": (5, 50, 55, 5), "encoding": [0.5] * 512, "det_score": 0.88}])
    photo = db.get_photo(pid)
    res = face_edit.add_manual_face(db, pid, path, photo,
                                    {"top": 0.1, "right": 0.5, "bottom": 0.6, "left": 0.2},
                                    person_name="Test Person")
    assert res["ok"] and res["encoded"] is True and res["person_id"]
    row = db.conn.execute(
        "SELECT person_id, match_source FROM faces WHERE id=?", (res["face_id"],)).fetchone()
    assert row["person_id"] == res["person_id"] and row["match_source"] == "manual"


def test_delete_face_removes_row_and_encoding(db):
    pid = _mk_photo(db, faces=[((1, 2, 3, 0), [0.1] * 512)])
    fid = db.conn.execute("SELECT id FROM faces WHERE photo_id=?", (pid,)).fetchone()[0]
    assert db.delete_face(fid) is True
    assert db.conn.execute("SELECT COUNT(*) FROM faces WHERE id=?", (fid,)).fetchone()[0] == 0
    assert db.delete_face(fid) is False  # already gone


# ---- web endpoints (non-replica: run locally against the test DB) -----------

def test_delete_face_endpoint(client, db):
    pid = db.add_photo(filepath="w/f.jpg", filename="f.jpg",
                       image_width=100, image_height=100)
    fid = db.add_face(pid, (1, 2, 3, 0), [0.1] * 512, det_score=0.5)
    db.conn.commit()
    assert client.delete("/api/faces/%d" % fid).json()["ok"] is True
    assert db.conn.execute("SELECT COUNT(*) FROM faces WHERE id=?", (fid,)).fetchone()[0] == 0
    assert client.delete("/api/faces/%d" % fid).status_code == 404


def test_detail_includes_det_score(client, db):
    pid = db.add_photo(filepath="w/g.jpg", filename="g.jpg",
                       image_width=100, image_height=100)
    db.add_face(pid, (1, 2, 3, 0), [0.1] * 512, det_score=0.42)
    db.conn.commit()
    faces = client.get("/api/photos/%d" % pid).json()["faces"]
    assert faces[0]["det_score"] == pytest.approx(0.42)


def test_detect_faces_endpoint(client, db, monkeypatch, tmp_path):
    img = tmp_path / "h.jpg"
    Image.new("RGB", (80, 60), (100, 100, 100)).save(img, "JPEG")
    # Absolute filepath → _abs_photo_path returns it verbatim, os.path.exists real.
    pid = db.add_photo(filepath=str(img), filename="h.jpg",
                       image_width=80, image_height=60)
    db.conn.commit()
    monkeypatch.setattr(
        "photosearch.faces.detect_faces",
        lambda p: [{"bbox": (5, 40, 45, 5), "encoding": [0.5] * 512, "det_score": 0.9}])
    r = client.post("/api/photos/%d/detect-faces" % pid, json={"replace": False})
    assert r.status_code == 200 and r.json()["added"] == 1


def test_add_face_box_endpoint(client, db, monkeypatch, tmp_path):
    img = tmp_path / "i.jpg"
    Image.new("RGB", (80, 60), (100, 100, 100)).save(img, "JPEG")
    pid = db.add_photo(filepath=str(img), filename="i.jpg",
                       image_width=80, image_height=60)
    db.conn.commit()
    monkeypatch.setattr("photosearch.faces.detect_faces", lambda p: [])
    r = client.post("/api/photos/%d/add-face-box" % pid,
                    json={"bbox": {"top": 0.2, "right": 0.6, "bottom": 0.7, "left": 0.3}})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert db.conn.execute("SELECT COUNT(*) FROM faces WHERE photo_id=?", (pid,)).fetchone()[0] == 1
