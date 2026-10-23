"""The video faces-row gate (video_analysis.face_row_eligible) and the
targeted re-scan (`video-rescan-fragmented`).

An identity below the gate keeps its `video_tracks` row (face_id NULL) but
gets no `faces` row; every consumer of `video_tracks.face_id` must take the
NULL; human labels still carry across a re-scan (D13), promoting a gated
identity; the re-scan CLI is read-only by default and re-queues exactly the
affected videos, never their description."""

import sqlite3

from click.testing import CliRunner

from photosearch import video_analysis as VA
from photosearch.video_scan_store import (SCAN_VERSION, fragmented_videos,
                                          requeue_scan)
from test_video_scan_submit import (_faces, _ident, _meta, _result, _submit,  # noqa: F401
                                    _unit, env)

FRAG = dict(n_det=1, presence_s=0.25)


def _tracks(db, pid):
    return [dict(r) for r in db.conn.execute(
        "SELECT * FROM video_tracks WHERE photo_id = ? ORDER BY ident_ix", (pid,))]


# --- the gate ----------------------------------------------------------------

def test_gate_rule():
    assert VA.face_row_eligible(3, 0.1)
    assert VA.face_row_eligible(2, 0.5)
    assert not VA.face_row_eligible(2, 0.4)
    assert not VA.face_row_eligible(1, 0.25)
    # At 2 fps one sample cell IS 0.5 s: presence alone never admits a
    # single detection.
    assert not VA.face_row_eligible(1, 0.501)
    assert VA.face_row_eligible(None, None)          # unknown never drops a face
    loose = VA.AnalysisParams(face_min_detections=1)
    assert VA.face_row_eligible(1, 0.1, loose)
    assert VA.face_row_eligible(1, 0.1, min_detections=1)


def test_one_detection_identity_keeps_its_track_but_gets_no_faces_row(env):
    db = env["db"]
    pid = env["video"]("A.MOV")
    resp = _submit(env["client"], [_result(pid, [_ident(0, _unit(0)),
                                                 _ident(1, _unit(1), **FRAG)])])
    info = resp["video_scan"][0]
    assert info["faces"] == 1 and info["gated"] == 1
    faces = _faces(db, pid)
    assert len(faces) == 1
    tracks = _tracks(db, pid)
    assert [t["ident_ix"] for t in tracks] == [0, 1]
    assert tracks[0]["face_id"] == faces[0]["id"]
    assert tracks[1]["face_id"] is None and tracks[1]["n_det"] == 1
    assert db.conn.execute("SELECT COUNT(*) FROM face_encodings").fetchone()[0] == 1


def test_the_workers_verdict_wins_over_the_default_gate(env):
    db = env["db"]
    pid = env["video"]("A.MOV")
    _submit(env["client"], [_result(pid, [_ident(0, _unit(0), face_row=False),
                                          _ident(1, _unit(1), face_row=True, **FRAG)])])
    assert [t["face_id"] is None for t in _tracks(db, pid)] == [True, False]


def test_build_scan_result_flags_each_identity():
    from types import SimpleNamespace
    from photosearch import video_scan as VS
    frames = []
    for i in range(8):
        faces = [{"bbox": [100, 100, 220, 240], "det_score": 0.9,
                  "embedding": _unit(0), "sharpness": 1.0}]
        if i == 4:      # a stranger seen once
            faces.append({"bbox": [600, 100, 700, 220], "det_score": 0.9,
                          "embedding": _unit(1), "sharpness": 1.0})
        frames.append({"t": 0.125 + i * 0.25, "faces": faces})
    m = {"shift_x": 0.0, "shift_y": 0.0, "sigma": 1.0, "clip_hi": 0.0, "clip_lo": 0.0}
    scan = SimpleNamespace(frames=frames, width=1280, height=720, duration=2.0,
                           sample_fps=4.0, frame_measures=[m] * 8, _stored={},
                           keyframe_ix=[], segment_ix=[], segments=[],
                           fps=30.0, n_decoded=8, store_scale=1.0, timings={})
    out = VS.build_scan_result(scan, with_clip=False)
    flags = sorted((i["n_det"], i["face_row"]) for i in out["result"]["identities"])
    assert flags == [(1, False), (8, True)]


# --- consumers of video_tracks.face_id ----------------------------------------

def test_consumers_take_a_null_face_id(env):
    from photosearch import rerun
    from photosearch.video_describe_store import subject_summary
    from photosearch.video_subjects_store import apply_subjects_result, read_subjects
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    _submit(client, [_result(pid, [_ident(0, _unit(0)), _ident(1, _unit(1), **FRAG)])])
    fid = _faces(db, pid)[0]["id"]
    cal = db.add_person("Calvin")
    db.assign_face_to_person(fid, cal, match_source="manual")
    gen = _meta(db, pid)["scan_generation"]
    segs = [[0.0, 3.0], [3.0, 6.0]]
    info = apply_subjects_result(db.conn, {
        "photo_id": pid, "scan_version": SCAN_VERSION, "scan_generation": gen,
        "subjects": [{"ix": 0, "label": "person", "category": "person", "face_ix": 0,
                      "focus_s": 3.0, "share": 0.33},
                     {"ix": 1, "label": "person", "category": "person", "face_ix": 1,
                      "focus_s": 3.0, "share": 0.33}],
        "analysis": {"segments": segs, "primaries": [0, 1],
                     "subjects": [{"ix": 0, "face_ix": 0, "category": "person"},
                                  {"ix": 1, "face_ix": 1, "category": "person"}]},
        "model": "m"})
    db.conn.commit()
    assert info["unmapped_face_ix"] == []          # a gated track is not "unmapped"
    rows = db.conn.execute("SELECT subject_ix, face_id FROM video_subjects "
                           "WHERE photo_id=? ORDER BY subject_ix", (pid,)).fetchall()
    assert [tuple(r) for r in rows] == [(0, fid), (1, None)]

    j = read_subjects(db.conn, pid)
    assert [s["display"] for s in j["subjects"]] == ["Calvin", "unidentified person"]
    assert [p["name"] for p in j["people"]] == ["Calvin"]
    assert client.get(f"/api/videos/{pid}/subjects").status_code == 200
    assert "Calvin" in subject_summary(db.conn, pid)

    # Subject share: grouped by faces.person_id — the gated track adds nothing.
    from photosearch.search import _PERSON_FOCUS_SQL
    got = db.conn.execute(_PERSON_FOCUS_SQL + " GROUP BY f.photo_id", (cal,)).fetchall()
    assert [(r[0], r[1]) for r in got] == [(pid, 0.33)]

    # mirror-fields ships the NULL; the replica's apply keeps it NULL.
    mf = client.get(f"/api/photos/{pid}/mirror-fields").json()
    assert [t["face_id"] for t in mf["video_tracks"]] == [fid, None]
    rerun._apply_mirror(db, pid, {"video_tracks": mf["video_tracks"]})
    assert [t["face_id"] for t in _tracks(db, pid)] == [fid, None]


# --- carry-over across a gated re-scan ----------------------------------------

def test_a_human_label_promotes_a_gated_identity_and_exclusions_still_carry(env):
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    _submit(client, [_result(pid, [_ident(0, _unit(0)), _ident(1, _unit(1)),
                                   _ident(2, _unit(2))])])
    f0, f1, f2 = [f["id"] for f in _faces(db, pid)]
    cal = db.add_person("Calvin")
    ellie = db.add_person("Ellie")
    db.assign_face_to_person(f0, cal, match_source="manual")
    db.conn.execute("UPDATE faces SET person_id=?, match_source='strict' WHERE id=?",
                    (ellie, f1))
    db.conn.execute("INSERT INTO face_person_exclusions (face_id, person_id, reason) "
                    "VALUES (?, ?, 'resolve_dups')", (f2, cal))
    db.conn.commit()

    # Re-scan: Calvin's identity is now a 1-detection fragment, Ellie's
    # (strict) too, and the excluded face is a full identity.
    resp = _submit(client, [_result(pid, [_ident(0, _unit(0), **FRAG),
                                          _ident(1, _unit(1), **FRAG),
                                          _ident(2, _unit(2))])])
    info = resp["video_scan"][0]
    assert info["promoted_by_label"] == 1 and info["gated"] == 1
    faces = _faces(db, pid)
    assert len(faces) == 2
    by = {f["person_id"]: f for f in faces}
    assert by[cal]["match_source"] == "manual"          # human label survived the gate
    assert ellie not in by                              # strict: re-matched nightly
    other = [f for f in faces if f["person_id"] is None][0]
    assert db.conn.execute("SELECT COUNT(*) FROM face_person_exclusions WHERE face_id=? "
                           "AND person_id=?", (other["id"], cal)).fetchone()[0] == 1
    assert [t["face_id"] is None for t in _tracks(db, pid)] == [False, True, False]


# --- the targeted re-scan ------------------------------------------------------

def _two_videos(env):
    db, client = env["db"], env["client"]
    frag = env["video"]("F.MOV")
    clean = env["video"]("C.MOV")
    # Both scanned under the OLD rule (every identity a faces row): force it
    # with the worker verdict.
    _submit(client, [_result(frag, [_ident(0, _unit(0)),
                                    _ident(1, _unit(1), face_row=True, **FRAG),
                                    _ident(2, _unit(2), face_row=True, **FRAG)]),
                     _result(clean, [_ident(0, _unit(3))])])
    db.conn.execute("UPDATE photos SET description='a boy kicks a ball' WHERE id IN (?, ?)",
                    (frag, clean))
    db.conn.execute("UPDATE video_meta SET subjects_version=1 WHERE photo_id IN (?, ?)",
                    (frag, clean))
    db.conn.commit()
    return frag, clean


def test_fragmented_videos_counts_only_rows_below_the_gate(env):
    frag, clean = _two_videos(env)
    conn = sqlite3.connect(f"file:{env['db_path']}?mode=ro", uri=True)
    try:
        info = fragmented_videos(conn)
        assert info["photo_ids"] == [frag]
        assert info["rows_below_gate"] == 2 and info["rows_disappear"] == 2
        assert info["subjects_done"] == 1 and info["described"] == 1
        # A looser gate admits the singletons: nothing to re-scan.
        assert fragmented_videos(conn, min_detections=1)["videos"] == 0
    finally:
        conn.close()


def test_rescan_cli_dry_run_writes_nothing(env):
    import cli
    frag, clean = _two_videos(env)
    before = _meta(env["db"], frag)
    r = CliRunner().invoke(cli.cli, ["video-rescan-fragmented", "--db", env["db_path"]])
    assert r.exit_code == 0, r.output
    assert "videos to re-scan:                1" in r.output
    assert "would disappear:                2" in r.output
    assert "Dry run" in r.output
    assert _meta(env["db"], frag) == before


def test_rescan_cli_apply_requeues_exactly_those_videos_and_keeps_descriptions(env):
    import cli
    db = env["db"]
    frag, clean = _two_videos(env)
    r = CliRunner().invoke(cli.cli, ["video-rescan-fragmented", "--db", env["db_path"],
                                     "--save-collection", "frag", "--apply"])
    assert r.exit_code == 0, r.output
    assert _meta(db, frag)["scan_version"] is None
    assert _meta(db, clean)["scan_version"] == SCAN_VERSION
    assert [p["id"] for p in db.get_unprocessed_photos("video-scan", limit=5)] == [frag]
    # faces stay until the re-scan replaces them; the description is untouched
    assert len(_faces(db, frag)) == 3
    assert db.conn.execute("SELECT description FROM photos WHERE id=?",
                           (frag,)).fetchone()[0] == "a boy kicks a ball"
    c = db.get_collection_by_name("frag")
    assert [r[0] for r in db.conn.execute(
        "SELECT photo_id FROM collection_photos WHERE collection_id=?", (c["id"],))] == [frag]

    # The re-scan landing under the gate drops the fragments and re-queues
    # video-subjects; the description still stands.
    _submit(env["client"], [_result(frag, [_ident(0, _unit(0)), _ident(1, _unit(1), **FRAG),
                                           _ident(2, _unit(2), **FRAG)])])
    assert len(_faces(db, frag)) == 1 and len(_tracks(db, frag)) == 3
    assert _meta(db, frag)["subjects_version"] is None
    assert db.conn.execute("SELECT description FROM photos WHERE id=?",
                           (frag,)).fetchone()[0] == "a boy kicks a ball"


def test_requeue_scan_of_no_ids_touches_nothing(env):
    frag, clean = _two_videos(env)
    assert requeue_scan(env["db"].conn, []) == 0
    assert _meta(env["db"], frag)["scan_version"] == SCAN_VERSION
