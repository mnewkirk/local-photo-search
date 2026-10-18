"""Video main-subject share surfaced in search (docs/plans/video-support.md
step 6), hidden behind PHOTOSEARCH_VIDEO_SUBJECT_SHARE until the holdout
eval passes (D16).

- A person's share is read GROUP BY faces.person_id at query time, so a
  relabel in /faces moves it with no recompute.
- A subject label is normalized like the stored ones ("the dog" -> dog).
- Flag off: the HTTP API rejects the params, the tool schemas omit them and a
  stray tool arg is ignored, the agent's locked filters drop them, and the
  typeahead endpoint 404s. The PhotoModal strip's endpoint is not gated.
"""

import pytest

from photosearch import agent, tools
from photosearch.features import VIDEO_SUBJECT_SHARE_ENV
from photosearch.search import search_combined

FLAG = VIDEO_SUBJECT_SHARE_ENV


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)


def _video(db, name, duration, date="2026-09-26 10:00:00"):
    vid = db.add_photo(filepath=f"2026/v/{name}", filename=name,
                       date_taken=date, media_type="video", camera_model="VCAM")
    db.conn.execute("INSERT INTO video_meta (photo_id, probe_version, kind, duration_s) "
                    "VALUES (?, 1, 'clip', ?)", (vid, duration))
    return vid


def _subject(db, vid, ix, label, category, share, duration, face_id=None):
    db.conn.execute(
        "INSERT INTO video_subjects (photo_id, subject_ix, label, category, face_id, "
        "focus_s, share, n_segments) VALUES (?,?,?,?,?,?,?,1)",
        (vid, ix, label, category, face_id, share * duration, share))


@pytest.fixture
def vids(db):
    kid = db.add_person("Vidkid")
    sis = db.add_person("Vidsis")
    # A still of the kid: never a focus result.
    still = db.add_photo(filepath="2026/v/still.jpg", filename="still.jpg",
                         date_taken="2026-09-26 09:00:00")
    db.add_face(still, (0, 10, 10, 0), [], person_id=kid)
    # V1, 60 s: kid the subject for half; a dog for 30%.
    v1 = _video(db, "V1.MP4", 60)
    f1 = db.add_face(v1, (0, 10, 10, 0), [], person_id=kid)
    _subject(db, v1, 0, "person", "person", 0.5, 60, f1)
    _subject(db, v1, 1, "dog", "animal", 0.3, 60)
    # V2, 600 s: kid only 10% (60 s), the sister 60%.
    v2 = _video(db, "V2.MP4", 600, "2026-09-27 10:00:00")
    f2 = db.add_face(v2, (0, 10, 10, 0), [], person_id=kid)
    f2b = db.add_face(v2, (20, 30, 30, 20), [], person_id=sis)
    _subject(db, v2, 0, "person", "person", 0.1, 600, f2)
    _subject(db, v2, 1, "person", "person", 0.6, 600, f2b)
    # V3: a mountain lake, no people.
    v3 = _video(db, "V3.MP4", 20)
    _subject(db, v3, 0, "mountain lake", "landscape", 0.9, 20)
    db.conn.commit()
    return dict(kid=kid, sis=sis, still=still, v1=v1, v2=v2, v3=v3, f1=f1)


def _ids(res):
    page, total = res
    return [r["id"] for r in page]


def _run(db, **kw):
    return search_combined(db, limit=100, with_total=True, **kw)


# ---------------------------------------------------------------------------
# search_combined
# ---------------------------------------------------------------------------

def test_min_focus_share_needs_the_person_to_be_the_main_subject(db, vids):
    page, total = _run(db, person="Vidkid", min_focus_share=0.4)
    assert [r["id"] for r in page] == [vids["v1"]]
    hit = page[0]
    assert hit["focus_share"] == 0.5 and hit["focus_seconds"] == 30.0
    assert hit["focus_people"][0]["name"] == "Vidkid"
    # Any share: both videos, never the still.
    assert set(_ids(_run(db, person="Vidkid", min_focus_share=0.0))) == {
        vids["v1"], vids["v2"]}


def test_a_relabel_moves_the_share_without_a_recompute(db, vids):
    assert _ids(_run(db, person="Vidkid", min_focus_share=0.4)) == [vids["v1"]]
    db.conn.execute("UPDATE faces SET person_id = ? WHERE id = ?",
                    (vids["sis"], vids["f1"]))
    db.conn.commit()
    assert _ids(_run(db, person="Vidkid", min_focus_share=0.4)) == []
    assert set(_ids(_run(db, person="Vidsis", min_focus_share=0.4))) == {
        vids["v1"], vids["v2"]}


def test_min_focus_seconds_is_the_long_video_question(db, vids):
    assert _ids(_run(db, person="Vidkid", min_focus_seconds=50)) == [vids["v2"]]
    assert _ids(_run(db, person="Vidkid", min_focus_seconds=50,
                     min_focus_share=0.2)) == []


def test_every_named_person_must_pass(db, vids):
    both = _ids(_run(db, person_ids=[vids["kid"], vids["sis"]], min_focus_share=0.05))
    assert both == [vids["v2"]]
    assert _ids(_run(db, person_ids=[vids["kid"], vids["sis"]],
                     min_focus_share=0.2)) == []


def test_focus_sort_ranks_by_share(db, vids):
    page, _ = _run(db, person="Vidkid", sort="focus_desc")
    assert [r["id"] for r in page] == [vids["v1"], vids["v2"]]


@pytest.mark.parametrize("label", ["dog", "the dog", "Dogs", "our dog's"])
def test_subject_label_is_normalized_at_query_time(db, vids, label):
    page, _ = _run(db, subject=label)
    assert [r["id"] for r in page] == [vids["v1"]]
    assert page[0]["subject_label"] == "dog" and page[0]["subject_share"] == 0.3


def test_min_subject_share_and_subject_sort(db, vids):
    assert _ids(_run(db, subject="dog", min_subject_share=0.5)) == []
    assert _ids(_run(db, subject="mountain lake", min_subject_share=0.5)) == [vids["v3"]]
    assert _ids(_run(db, subject="dog", sort="subject_desc",
                     date_from="2026-09-26", date_to="2026-09-26")) == [vids["v1"]]


def test_share_filters_respect_media(db, vids):
    assert _ids(_run(db, person="Vidkid", min_focus_share=0.0, media="photo")) == []


@pytest.mark.parametrize("kw", [
    dict(min_focus_share=0.5),
    dict(sort="focus_desc", camera="VCAM"),
    dict(min_subject_share=0.5, camera="VCAM"),
    dict(subject="the", camera="VCAM"),
    dict(person="Vidkid", min_focus_share=1.5),
    dict(person="Vidkid", min_focus_seconds=-1),
])
def test_unsatisfiable_share_requests_are_errors(db, vids, kw):
    with pytest.raises(ValueError):
        _run(db, **kw)


# ---------------------------------------------------------------------------
# HTTP: flag-gated
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("qs", [
    "person=Vidkid&min_focus_share=0.5", "person=Vidkid&min_focus_seconds=5",
    "subject=dog", "subject=dog&min_subject_share=0.2",
    "person=Vidkid&sort=focus_desc", "subject=dog&sort=subject_desc"])
def test_flag_off_rejects_every_share_param(client, vids, flag_off, qs):
    r = client.get("/api/search?" + qs)
    assert r.status_code == 400
    assert FLAG in r.json()["detail"]


def test_flag_on_search_returns_the_share(client, vids, flag_on):
    body = client.get("/api/search?person=Vidkid&min_focus_share=0.4").json()
    assert [r["id"] for r in body["results"]] == [vids["v1"]]
    assert body["results"][0]["focus_share"] == 0.5
    body = client.get("/api/search?subject=the%20dog").json()
    assert body["results"][0]["subject_label"] == "dog"
    # Unsatisfiable: a clear 400, not a 500.
    assert client.get("/api/search?min_subject_share=0.2&camera=VCAM").status_code == 400


def test_flag_off_leaves_ordinary_search_untouched(client, vids, flag_off):
    body = client.get("/api/search?person=Vidkid").json()
    assert vids["v1"] in [r["id"] for r in body["results"]]
    assert "focus_share" not in body["results"][0]


def test_stats_split_photos_and_videos_and_report_the_flag(client, db, vids, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    body = client.get("/api/stats").json()
    total = db.conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    assert body["videos"] == 3 and body["stills"] == total - 3
    assert body["video_subject_share"] is False
    monkeypatch.setenv(FLAG, "1")
    assert client.get("/api/stats").json()["video_subject_share"] is True


def test_stats_video_count_reads_the_partial_index(db, vids):
    plan = " ".join(r[3] for r in db.conn.execute(
        "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM photos WHERE media_type IS NOT NULL"))
    assert "idx_photos_media_video" in plan, plan


def test_subject_typeahead(client, vids, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert client.get("/api/videos/subject-labels").status_code == 404
    monkeypatch.setenv(FLAG, "1")
    labels = client.get("/api/videos/subject-labels").json()["labels"]
    # Non-person subjects only, with their video counts.
    assert {x["label"] for x in labels} == {"dog", "mountain lake"}
    got = client.get("/api/videos/subject-labels?q=The%20Dogs").json()["labels"]
    assert [x["label"] for x in got] == ["dog"]
    got = client.get("/api/videos/subject-labels?q=lake").json()["labels"]
    assert [x["label"] for x in got] == ["mountain lake"]


def test_subject_strip_endpoint_is_not_gated(client, vids, flag_off):
    body = client.get(f"/api/videos/{vids['v1']}/subjects").json()
    assert body["photo_id"] == vids["v1"]


def test_face_records_from_a_video_are_marked(db, vids):
    from photosearch.web import _mark_video_faces
    rows = [{"face_id": 1, "photo_id": vids["v1"]},
            {"face_id": 2, "photo_id": vids["still"]}]
    _mark_video_faces(db.conn, rows)
    assert rows[0]["media_type"] == "video" and "media_type" not in rows[1]


def test_intent_for_the_typeahead_and_share_search():
    from photosearch.request_intent import infer_intent
    assert "subject labels" in infer_intent("GET", "/api/videos/subject-labels", "q=dog")
    text = infer_intent("GET", "/api/search", "person=Calvin&min_focus_share=0.5")
    assert "main subject share" in text


# ---------------------------------------------------------------------------
# Ask / MCP tools
# ---------------------------------------------------------------------------

_SHARE_PROPS = {"min_focus_share", "min_focus_seconds", "subject", "min_subject_share"}


def _search_schema(specs, key):
    for s in specs:
        if (s.get("name") or s.get("function", {}).get("name")) == "search_photos":
            return s[key] if key in s else s["function"]["parameters"]


def test_tool_schemas_omit_the_share_props_while_the_flag_is_off(flag_off):
    for spec in tools.mcp_tools(include_writes=True):
        props = spec["inputSchema"].get("properties", {})
        assert not (_SHARE_PROPS & set(props)), spec["name"]
        assert not ({"focus_desc", "subject_desc"}
                    & set((props.get("sort") or {}).get("enum") or [])), spec["name"]
    for spec in tools.openai_tools(consolidated=True):
        props = spec["function"]["parameters"].get("properties", {})
        assert not (_SHARE_PROPS & set(props))


def test_tool_schemas_offer_them_when_on(flag_on):
    props = _search_schema(tools.mcp_tools(), "inputSchema")["properties"]
    assert _SHARE_PROPS <= set(props)
    assert {"focus_desc", "subject_desc"} <= set(props["sort"]["enum"])


def test_search_photos_tool_honours_the_flag(db, vids, monkeypatch):
    args = {"people": ["Vidkid"], "min_focus_share": 0.4}
    monkeypatch.setenv(FLAG, "1")
    out = tools.call_tool(db, "search_photos", args)
    assert [r["id"] for r in out["results"]] == [vids["v1"]]
    assert out["results"][0]["focus_share"] == 0.5
    # A model saying 40 means 40%.
    out = tools.call_tool(db, "search_photos", {**args, "min_focus_share": 40})
    assert [r["id"] for r in out["results"]] == [vids["v1"]]
    # No people: the floor is dropped with a note rather than failing.
    out = tools.call_tool(db, "search_photos", {"min_focus_share": 0.4, "media": "video"})
    assert "needs `people`" in out["note"]
    monkeypatch.delenv(FLAG)
    out = tools.call_tool(db, "search_photos", args)
    assert vids["still"] in [r["id"] for r in out["results"]]  # ignored, not applied


def test_filter_sql_share_clauses_and_media_switch(db, vids, monkeypatch):
    monkeypatch.setenv(FLAG, "1")

    def ids(args):
        where, params = tools._build_filter_sql(db, args)
        return {r[0] for r in db.conn.execute(
            f"SELECT id FROM photos WHERE {where or '1'}", params)}

    # The book/curation default is stills; a share filter switches it to videos.
    assert ids({"people": ["Vidkid"], "min_focus_share": 0.4}) == {vids["v1"]}
    assert ids({"subject": "The Dog"}) == {vids["v1"]}
    assert ids({"subject": "dog", "min_subject_share": 0.5}) == set()
    assert vids["v1"] not in ids({"people": ["Vidkid"]})  # stills by default
    monkeypatch.delenv(FLAG)
    assert vids["v1"] not in ids({"people": ["Vidkid"], "min_focus_share": 0.4})


def test_agent_locks_the_share_filters_only_when_on(monkeypatch):
    pinned = {"people": ["Vidkid"], "min_focus_share": 0.5, "subject": "dog",
              "sort": "focus_desc"}
    monkeypatch.delenv(FLAG, raising=False)
    out = agent._normalize_locked(pinned)
    assert "min_focus_share" not in out and "subject" not in out and "sort" not in out
    monkeypatch.setenv(FLAG, "1")
    out = agent._normalize_locked(pinned)
    assert out["min_focus_share"] == 0.5 and out["subject"] == "dog"
    assert out["sort"] == "focus_desc"
