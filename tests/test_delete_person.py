"""Deleting a stray person name ('Asa', 'cars' — an early Enter). Only a name
no face is labelled with can go; a real label is refused."""

import pytest


def _person(db, name):
    pid = db.add_person(name)
    db.conn.commit()
    return pid


def test_empty_person_is_deleted_with_its_references(db):
    pid = _person(db, "Asa")
    db.conn.execute("INSERT INTO face_references (person_id, source_path) VALUES (?, 'x')", (pid,))
    db.conn.commit()
    out = db.delete_empty_person(pid)
    assert out == {"id": pid, "name": "Asa", "references_removed": 1}
    assert db.conn.execute("SELECT COUNT(*) FROM persons WHERE id = ?", (pid,)).fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM face_references WHERE person_id = ?",
                           (pid,)).fetchone()[0] == 0


def test_a_named_face_blocks_the_delete(db):
    pid = _person(db, "Calvin Test")
    face = db.conn.execute("SELECT id FROM faces LIMIT 1").fetchone()
    if face is None:
        pytest.skip("fixture DB has no faces")
    db.conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (pid, face[0]))
    db.conn.commit()
    with pytest.raises(ValueError, match="still labelled"):
        db.delete_empty_person(pid)
    assert db.conn.execute("SELECT COUNT(*) FROM persons WHERE id = ?", (pid,)).fetchone()[0] == 1


def test_undo_snapshot_rows_go_too(db):
    pid = _person(db, "cars")
    db.conn.execute("CREATE TABLE IF NOT EXISTS face_dedupe_undo (face_id INTEGER PRIMARY KEY, "
                    "person_id INTEGER, match_source TEXT)")
    db.conn.execute("INSERT INTO face_dedupe_undo VALUES (999999, ?, 'temporal')", (pid,))
    db.conn.commit()
    db.delete_empty_person(pid)
    assert db.conn.execute("SELECT COUNT(*) FROM face_dedupe_undo WHERE person_id = ?",
                           (pid,)).fetchone()[0] == 0


def test_api(client, db):
    pid = _person(db, "Asa")
    assert client.delete(f"/api/persons/{pid}").json()["name"] == "Asa"
    assert client.delete(f"/api/persons/{pid}").status_code == 404
    face = db.conn.execute("SELECT id FROM faces LIMIT 1").fetchone()
    if face is not None:
        busy = _person(db, "Real Person")
        db.conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (busy, face[0]))
        db.conn.commit()
        r = client.delete(f"/api/persons/{busy}")
        assert r.status_code == 409 and "still labelled" in r.json()["detail"]
