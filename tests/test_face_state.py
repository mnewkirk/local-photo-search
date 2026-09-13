"""Tests for photosearch.face_state — the replica→NAS face-assignment bridge.

Two databases throughout: ``src`` plays the replica that computed assignments,
``dst`` the NAS that receives them. Both are built from the same fixture rows
so face ids line up, exactly as a replica dump of the NAS would.
"""

import shutil

import pytest

from photosearch.face_state import apply_face_state, export_face_state, read_meta


def _faces(db):
    return {r["id"]: dict(r) for r in db.conn.execute(
        "SELECT id, person_id, cluster_id, match_source FROM faces ORDER BY id")}


@pytest.fixture
def pair(db, tmp_path):
    """(src, dst, face_ids, person_id): dst is a byte-copy of src."""
    from photosearch.db import PhotoDB

    db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    # Start from a known state: every face unmatched and unclustered.
    db.conn.execute("UPDATE faces SET person_id = NULL, cluster_id = NULL, match_source = NULL")
    db.conn.execute("DELETE FROM ignored_clusters")
    db.conn.commit()
    db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    dst_path = str(tmp_path / "nas.db")
    shutil.copy(db.db_path, dst_path)
    dst = PhotoDB(dst_path)
    face_ids = sorted(_faces(db))
    assert len(face_ids) >= 3, "fixture needs at least 3 faces"
    person_id = db.conn.execute("SELECT id FROM persons LIMIT 1").fetchone()[0]
    yield db, dst, face_ids, person_id
    dst.close()


def _export(src, tmp_path, meta=None):
    path = str(tmp_path / "face-state.db")
    export_face_state(src, path, meta=meta)
    return path


def test_export_roundtrips_meta(pair, tmp_path):
    src, _dst, face_ids, _pid = pair
    path = _export(src, tmp_path, meta={"stages": {"match_faces": {"last_run_at": "t"}}})
    assert read_meta(path) == {"stages": {"match_faces": {"last_run_at": "t"}}}


def test_export_without_meta_reads_as_empty(pair, tmp_path):
    src, _dst, _ids, _pid = pair
    assert read_meta(_export(src, tmp_path)) == {}


def test_read_meta_rejects_a_non_sqlite_file(tmp_path):
    bogus = tmp_path / "junk.db"
    bogus.write_bytes(b"definitely not sqlite" * 100)
    with pytest.raises(ValueError):
        read_meta(str(bogus))


def test_read_meta_rejects_sqlite_without_assignments(tmp_path):
    import sqlite3
    other = tmp_path / "other.db"
    conn = sqlite3.connect(other)
    conn.execute("CREATE TABLE unrelated (x)")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError):
        read_meta(str(other))


def test_additive_persons_fill_only_unmatched_faces(pair, tmp_path):
    src, dst, (f1, f2, f3, *_), pid = pair
    other_pid = dst.conn.execute(
        "SELECT id FROM persons WHERE id <> ? LIMIT 1", (pid,)).fetchone()
    other_pid = other_pid[0] if other_pid else pid

    # Replica matched f1 and f2.
    src.conn.execute("UPDATE faces SET person_id = ?, match_source = 'temporal' "
                     "WHERE id IN (?, ?)", (pid, f1, f2))
    src.conn.commit()
    # NAS curated f2 differently since the sync — must survive.
    dst.conn.execute("UPDATE faces SET person_id = ?, match_source = 'manual' WHERE id = ?",
                     (other_pid, f2))
    dst.conn.commit()

    summary = apply_face_state(dst, _export(src, tmp_path), apply_clusters=False)
    faces = _faces(dst)
    assert summary["persons"] == 1
    assert faces[f1]["person_id"] == pid and faces[f1]["match_source"] == "temporal"
    assert faces[f2]["person_id"] == other_pid and faces[f2]["match_source"] == "manual"
    assert faces[f3]["person_id"] is None


def test_additive_fill_never_undoes_dedupe(pair, tmp_path):
    src, dst, (f1, *_), pid = pair
    src.conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (pid, f1))
    src.conn.commit()
    dst.conn.execute("UPDATE faces SET match_source = 'dedupe_unmatched' WHERE id = ?", (f1,))
    dst.conn.commit()

    apply_face_state(dst, _export(src, tmp_path))
    assert _faces(dst)[f1]["person_id"] is None


def test_dry_run_writes_nothing(pair, tmp_path):
    src, dst, (f1, *_), pid = pair
    src.conn.execute("UPDATE faces SET person_id = ?, cluster_id = 7 WHERE id = ?", (pid, f1))
    src.conn.commit()
    before = _faces(dst)
    summary = apply_face_state(dst, _export(src, tmp_path), renumbered=True, apply=False)
    assert summary["applied"] is False
    assert summary["persons"] == 1
    assert _faces(dst) == before


def test_faces_absent_from_the_file_get_no_person(pair, tmp_path):
    """A face detected on the NAS after the sync isn't in the file at all."""
    src, dst, (f1, *_), pid = pair
    src.conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (pid, f1))
    src.conn.commit()
    path = _export(src, tmp_path)
    # Simulate: the file predates f1 entirely.
    import sqlite3
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM face_assignments WHERE face_id = ?", (f1,))
    conn.commit()
    conn.close()

    apply_face_state(dst, path)
    assert _faces(dst)[f1]["person_id"] is None


def test_renumbered_apply_clears_stale_ids_and_remaps_ignored(pair, tmp_path):
    src, dst, (f1, f2, f3, *rest), pid = pair

    # NAS old numbering: f1+f2 in cluster 5 (ignored), f3 in cluster 9, and a
    # named face still carrying a stale id from its life as an unknown.
    dst.conn.execute("UPDATE faces SET cluster_id = 5 WHERE id IN (?, ?)", (f1, f2))
    dst.conn.execute("UPDATE faces SET cluster_id = 9 WHERE id = ?", (f3,))
    dst.conn.execute("INSERT INTO ignored_clusters (cluster_id) VALUES (5)")
    dst.conn.commit()

    # Replica recluster: f1+f2 are new cluster 0; f3 is new cluster 5 — the
    # SAME number as the NAS's ignored cluster, which a naive apply would hide.
    src.conn.execute("UPDATE faces SET cluster_id = 0 WHERE id IN (?, ?)", (f1, f2))
    src.conn.execute("UPDATE faces SET cluster_id = 5 WHERE id = ?", (f3,))
    src.conn.commit()
    path = _export(src, tmp_path)

    if rest:
        # A face absent from the file, carrying an old id that collides.
        import sqlite3
        conn = sqlite3.connect(path)
        conn.execute("DELETE FROM face_assignments WHERE face_id = ?", (rest[0],))
        conn.commit()
        conn.close()
        dst.conn.execute("UPDATE faces SET cluster_id = 0 WHERE id = ?", (rest[0],))
        dst.conn.commit()

    summary = apply_face_state(dst, path, apply_persons=False, renumbered=True)
    faces = _faces(dst)
    ignored = {r[0] for r in dst.conn.execute("SELECT cluster_id FROM ignored_clusters")}

    assert faces[f1]["cluster_id"] == 0 and faces[f2]["cluster_id"] == 0
    assert faces[f3]["cluster_id"] == 5
    assert ignored == {0}, "the ignore follows the faces, not the number"
    assert summary["ignored_before"] == 1 and summary["ignored_after"] == 1
    if rest:
        assert faces[rest[0]]["cluster_id"] is None
        assert summary["cleared_absent"] == 1


def test_renumbered_apply_does_not_ignore_a_minority_merge(pair, tmp_path):
    """One ignored face joining a bigger cluster must not hide the cluster."""
    src, dst, (f1, f2, f3, *_), _pid = pair
    dst.conn.execute("UPDATE faces SET cluster_id = 5 WHERE id = ?", (f1,))
    dst.conn.execute("INSERT INTO ignored_clusters (cluster_id) VALUES (5)")
    dst.conn.commit()
    src.conn.execute("UPDATE faces SET cluster_id = 1 WHERE id IN (?, ?, ?)", (f1, f2, f3))
    src.conn.commit()

    apply_face_state(dst, _export(src, tmp_path), apply_persons=False, renumbered=True)
    assert dst.conn.execute("SELECT COUNT(*) FROM ignored_clusters").fetchone()[0] == 0


def test_renumbered_apply_strips_cluster_from_named_faces(pair, tmp_path):
    src, dst, (f1, *_), pid = pair
    src.conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (pid, f1))
    src.conn.commit()
    dst.conn.execute("UPDATE faces SET cluster_id = 3 WHERE id = ?", (f1,))
    dst.conn.commit()

    apply_face_state(dst, _export(src, tmp_path), renumbered=True)
    face = _faces(dst)[f1]
    assert face["person_id"] == pid
    assert face["cluster_id"] is None


def test_before_commit_failure_rolls_everything_back(pair, tmp_path):
    src, dst, (f1, *_), pid = pair
    src.conn.execute("UPDATE faces SET person_id = ?, cluster_id = 2 WHERE id = ?", (pid, f1))
    src.conn.commit()
    before = _faces(dst)

    def boom(_summary):
        raise RuntimeError("watermark write failed")

    with pytest.raises(RuntimeError):
        apply_face_state(dst, _export(src, tmp_path), renumbered=True, before_commit=boom)
    assert _faces(dst) == before
    # And the connection is usable again (DETACH ran, no dangling transaction).
    assert not dst.conn.in_transaction
    dst.conn.execute("SELECT 1 FROM faces").fetchone()
