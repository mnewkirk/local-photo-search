"""The process-wide face-encoding cache behind "More of this kid" and
verify-labels (db._FaceEncodingCache)."""

import numpy as np
import pytest

from photosearch import db as dbmod
from photosearch.db import PhotoDB, _FACE_ENCODING_CACHE

pytestmark = pytest.mark.skipif(not dbmod.HAS_SQLITE_VEC,
                                reason="needs sqlite-vec")


def _enc(seed):
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture(autouse=True)
def _fresh_cache():
    _FACE_ENCODING_CACHE.clear()
    yield
    _FACE_ENCODING_CACHE.clear()


def _db_with_faces(path, seeds):
    db = PhotoDB(str(path))
    pid = db.add_photo(filepath="a.jpg", filename="a.jpg")
    ids = [db.add_face(pid, (0, 1, 1, 0), _enc(s)) for s in seeds]
    db.conn.commit()
    return db, ids


def test_cached_encodings_equal_the_uncached_ones(tmp_path):
    db, ids = _db_with_faces(tmp_path / "a.db", [1, 2, 3])
    bulk = db.get_face_encodings_bulk(ids)
    for _ in range(2):  # cold, then warm
        cached = db.get_face_encodings_cached(ids)
        assert set(cached) == set(ids)
        for fid in ids:
            assert cached[fid].dtype == np.float32
            np.testing.assert_array_equal(cached[fid], np.float32(bulk[fid]))


def test_warm_calls_do_not_touch_the_vector_table(tmp_path):
    db, ids = _db_with_faces(tmp_path / "a.db", [1, 2])
    db.get_face_encodings_cached(ids)
    seen = []
    db.conn.set_trace_callback(seen.append)
    db.get_face_encodings_cached(ids)
    db.conn.set_trace_callback(None)
    assert not [s for s in seen if "face_encodings" in s]


def test_two_databases_never_share_entries(tmp_path):
    a, a_ids = _db_with_faces(tmp_path / "a.db", [1])
    b, b_ids = _db_with_faces(tmp_path / "b.db", [2])
    assert a_ids == b_ids  # same face id, different face
    ea = a.get_face_encodings_cached(a_ids)[a_ids[0]]
    eb = b.get_face_encodings_cached(b_ids)[b_ids[0]]
    assert not np.array_equal(ea, eb)


def test_missing_ids_are_simply_absent(tmp_path):
    db, ids = _db_with_faces(tmp_path / "a.db", [1])
    assert set(db.get_face_encodings_cached(ids + [999_999])) == set(ids)


def test_cache_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(dbmod._FaceEncodingCache, "MAX_FACES", 2)
    db, ids = _db_with_faces(tmp_path / "a.db", [1, 2, 3])
    db.get_face_encodings_cached(ids)
    assert len(_FACE_ENCODING_CACHE._entries) == 2
