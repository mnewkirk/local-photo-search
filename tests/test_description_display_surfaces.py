"""The surfaces that show the named text: the photo detail endpoint, Ask/MCP
hits, book captions, the M28 mirror, and the Google caption staying RAW
(docs/plans/description-names.md)."""
import json

import pytest

from photosearch import people_bind as PB
from photosearch.db import PhotoDB
from tests.test_worker_submit_resilience import _open, client  # noqa: F401  (fixture)

DESC = "A young boy with reddish hair walks across a grassy field."
SPAN = "A young boy with reddish hair"
BOX = (100, 400, 400, 100)


def _bind(db, pid, person_name="Zack Azerad", source="manual"):
    person = db.get_person_by_name(person_name) or {"id": db.add_person(person_name)}
    fid = db.add_face(pid, BOX, [], person_id=person["id"], det_score=0.9)
    db.conn.execute("UPDATE faces SET match_source=? WHERE id=?", (source, fid))
    db.conn.execute("UPDATE photos SET description=? WHERE id=?", (DESC, pid))
    db.conn.execute(
        "INSERT OR REPLACE INTO description_subjects(photo_id, description_sha, slots_json, "
        "method, bind_version) VALUES (?,?,?,?,1)",
        (pid, PB.sha_of(DESC), json.dumps([{"slot": "s1", "kind": "person",
         "spans": [{"start": 0, "end": len(SPAN), "text": SPAN}],
         "faces": [{"face_id": fid, "bbox": list(BOX)}]}]), "rule"))
    db.conn.commit()
    return fid


def test_detail_endpoint_flag_off_is_unchanged_and_flag_on_adds_the_named_text(client, monkeypatch):
    with _open() as d:
        fid = _bind(d, 1)
    monkeypatch.delenv("PHOTOSEARCH_DISPLAY_DESCRIPTIONS", raising=False)
    off = client.get("/api/photos/1").json()
    assert off["description"] == DESC
    assert "description_display" not in off and "description_subjects" not in off
    monkeypatch.setenv("PHOTOSEARCH_DISPLAY_DESCRIPTIONS", "1")
    on = client.get("/api/photos/1").json()
    assert on["description"] == DESC                                   # raw is never replaced
    assert on["description_display"].startswith("Zack walks across a grassy field")
    assert on["description_subjects"][0]["state"] == "named"
    assert on["description_subjects"][0]["face_ids"] == [fid]
    # the relabel Koa -> Calvin, through the real assign path: the text follows
    with _open() as d:
        calvin = d.add_person("Calvin Newkirk")
        d.assign_face_to_person(fid, calvin, "manual")
        d.conn.commit()
    assert client.get("/api/photos/1").json()["description_display"].startswith("Calvin walks")


def test_ask_and_mcp_hits_carry_the_named_text_only_when_the_flag_is_on(client, monkeypatch):
    from photosearch import tools
    with _open() as d:
        _bind(d, 1)
    with _open() as d:
        monkeypatch.setitem(tools._REGISTRY, "fake_search", tools.ToolSpec(
            name="fake_search", description="x", parameters={"type": "object"},
            handler=lambda db, args: {"results": [tools._compact_hit(db.get_photo(1))]}))
        monkeypatch.delenv("PHOTOSEARCH_DISPLAY_DESCRIPTIONS", raising=False)
        assert tools.call_tool(d, "fake_search")["results"][0]["description"] == DESC
        monkeypatch.setenv("PHOTOSEARCH_DISPLAY_DESCRIPTIONS", "1")
        hit = tools.call_tool(d, "fake_search")["results"][0]
        assert hit["description"].startswith("Zack walks") and hit["named"] is True


def test_a_temporal_label_never_reaches_ask(client, monkeypatch):
    from photosearch import tools
    with _open() as d:
        _bind(d, 1, source="temporal")
        monkeypatch.setitem(tools._REGISTRY, "fake_search", tools.ToolSpec(
            name="fake_search", description="x", parameters={"type": "object"},
            handler=lambda db, args: {"results": [tools._compact_hit(db.get_photo(1))]}))
        monkeypatch.setenv("PHOTOSEARCH_DISPLAY_DESCRIPTIONS", "1")
        assert tools.call_tool(d, "fake_search")["results"][0]["description"] == DESC


def test_book_caption_context_uses_the_named_text(client, monkeypatch):
    from photosearch import book_ai
    with _open() as d:
        _bind(d, 1)
        monkeypatch.setenv("PHOTOSEARCH_DISPLAY_DESCRIPTIONS", "1")
        lines, _, _ = book_ai._photo_context(d, [1])
        assert lines[0].startswith("- Zack walks")
        monkeypatch.delenv("PHOTOSEARCH_DISPLAY_DESCRIPTIONS")
        assert book_ai._photo_context(d, [1])[0][0].startswith("- A young boy")


def test_google_caption_source_reads_the_raw_description():
    import inspect
    from photosearch import google_photos
    src = inspect.getsource(google_photos)
    assert "import description_render" not in src and "attach_display" not in src


def test_mirror_fields_carry_the_row_and_the_replica_applies_it(client):
    with _open() as d:
        _bind(d, 1)
    fields = client.get("/api/photos/1/mirror-fields").json()
    ds = fields["description_subjects"]
    assert ds["description_sha"] == PB.sha_of(DESC) and ds["method"] == "rule"
    from photosearch import rerun
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "replica.db")
    with PhotoDB(path) as r:
        pid = r.add_photo(filepath="x/a.jpg", filename="a.jpg", description=DESC)
        fields = dict(fields, faces=[])      # no faces: nothing to remap
        rerun._apply_mirror(r, pid, dict(fields, description_subjects=dict(ds)))
        row = r.conn.execute("SELECT description_sha FROM description_subjects "
                             "WHERE photo_id=?", (pid,)).fetchone()
        assert row[0] == PB.sha_of(DESC)
        rerun._apply_mirror(r, pid, {"description_subjects": None})      # NAS cleared it
        assert r.conn.execute("SELECT COUNT(*) FROM description_subjects").fetchone()[0] == 0
        # an older NAS (no key at all) leaves the local row alone
        rerun._apply_mirror(r, pid, dict(fields, description_subjects=dict(ds)))
        rerun._apply_mirror(r, pid, {"description": DESC})
        assert r.conn.execute("SELECT COUNT(*) FROM description_subjects").fetchone()[0] == 1
