"""db.PASS_MEDIA — which passes may see which media (schema v35).

docs/plans/video-support.md step 1. A video is a `photos` row with
media_type='video'; every image pass's claim predicate would otherwise pick it
up (faces is `NOT EXISTS faces`, describe is `description IS NULL`, verify is
`description IS NOT NULL AND verified_at IS NULL` — it would claim a video the
moment a video description lands). "Which media a pass applies to" is ONE
shared fact, like faces.MATCHABLE_SQL, and these tests iterate EVERY worker
pass so a new pass cannot forget it.
"""

import pytest

from photosearch import db as dbmod
from photosearch.batch_state import WORKER_PASSES, batch_state
from photosearch.db import (MAX_PROCESS_ATTEMPTS, PASS_MEDIA, STILL_SQL,
                            VIDEO_SQL, media_clause, media_sql)
from photosearch.ingest_batches import register_batch

_DIR = "2091/2091-10-17_phone-matt"


def _make_eligible(db, pass_type, ids):
    """The fixture describes everything (so the gated passes are eligible);
    describe itself then needs the description cleared; video-scan needs a
    current probe (it only claims probed, playable clips); video-subjects a
    current, clean scan; video-describe a clean scan and no description."""
    if pass_type in ("video-subjects", "video-describe"):
        from photosearch.video_probe import PROBE_VERSION
        from photosearch.video_scan_store import SCAN_VERSION
        for pid in ids:
            db.conn.execute("INSERT OR IGNORE INTO video_meta (photo_id) VALUES (?)", (pid,))
            db.conn.execute("UPDATE video_meta SET probe_version = ?, kind = 'clip', "
                            "scan_version = ? WHERE photo_id = ?",
                            (PROBE_VERSION, SCAN_VERSION, pid))
        db.conn.commit()
    if pass_type == "video-scan":
        from photosearch.video_probe import PROBE_VERSION
        for pid in ids:
            db.conn.execute("INSERT OR IGNORE INTO video_meta (photo_id) VALUES (?)", (pid,))
            db.conn.execute("UPDATE video_meta SET probe_version = ?, kind = 'clip' "
                            "WHERE photo_id = ?", (PROBE_VERSION, pid))
        db.conn.commit()
    if pass_type in ("describe", "video-describe"):
        db.conn.execute(
            f"UPDATE photos SET description = NULL WHERE id IN ({','.join('?' * len(ids))})",
            ids)
        db.conn.commit()


def _all_worker_pass_lists():
    from photosearch import rerun, worker_api
    return {"batch_state.WORKER_PASSES": WORKER_PASSES,
            "worker_api._ALL_PASSES": worker_api._ALL_PASSES,
            "rerun.ALL_PASSES": rerun.ALL_PASSES}


@pytest.fixture
def mixed(db):
    """Two stills + two videos in one batch folder. Everything a gated pass
    needs is present (a description), so a video row is ELIGIBLE for every
    pass by column test alone — only the media fragment keeps it out."""
    stills = [db.add_photo(filepath=f"{_DIR}/IMG_{i}.JPG", filename=f"IMG_{i}.JPG",
                           date_taken=f"2091-10-17 10:0{i}:00",
                           description="a still")
              for i in range(2)]
    videos = [db.add_photo(filepath=f"{_DIR}/VID_{i}.MP4", filename=f"VID_{i}.MP4",
                           date_taken=f"2091-10-17 11:0{i}:00",
                           description="a video", media_type="video")
              for i in range(2)]
    batch_id = register_batch(db, _DIR)
    return db, batch_id, stills, videos


def test_every_worker_pass_has_a_media_entry():
    for name, passes in _all_worker_pass_lists().items():
        for p in passes:
            assert p in PASS_MEDIA, f"{p} (from {name}) has no db.PASS_MEDIA entry"
            assert PASS_MEDIA[p] in ("still", "video", "any")


def test_the_map_says_what_the_plan_says():
    # verify stays still-only (D18); category-content / keywords read any
    # description since video-describe (step 5) writes one for videos.
    still = {"clip", "faces", "quality", "aesthetics", "describe", "verify",
             "category-visual"}
    for p in still:
        assert PASS_MEDIA[p] == "still"
        assert media_sql(p) == STILL_SQL
    for p in ("category-content", "keywords"):
        assert PASS_MEDIA[p] == "any"
        assert media_sql(p) is None


def test_an_any_pass_gets_no_media_fragment(monkeypatch):
    monkeypatch.setitem(PASS_MEDIA, "keywords", "any")
    assert media_sql("keywords") is None and media_clause("keywords") == ""
    for p in ("video-probe", "video-scan", "video-subjects", "video-describe"):
        assert PASS_MEDIA[p] == "video"
        assert media_sql(p, "p") == f"p.{VIDEO_SQL}"
    # all four video passes are live (2b, 3, 3b, 5)
    for p in ("video-probe", "video-scan", "video-subjects", "video-describe"):
        assert p in WORKER_PASSES


def test_a_pass_missing_from_the_map_fails_loudly():
    with pytest.raises(KeyError):
        media_sql("no-such-pass")


@pytest.mark.parametrize("pass_type", WORKER_PASSES)
@pytest.mark.parametrize("scoped", [False, True])
def test_video_never_offered_to_a_still_pass(mixed, pass_type, scoped):
    db, _bid, stills, videos = mixed
    _make_eligible(db, pass_type, stills + videos)
    scope = stills + videos if scoped else None
    claimed = {r["id"] for r in db.get_unprocessed_photos(
        pass_type, photo_ids=scope, limit=1000)}
    n = db.count_unprocessed_photos(pass_type, photo_ids=scope)
    vids_claimed = claimed & set(videos)
    if PASS_MEDIA[pass_type] == "still":
        assert not vids_claimed, f"{pass_type} offered a video row"
        # and the stills still are (the fragment narrows, never empties)
        assert set(stills) <= claimed
    else:
        assert vids_claimed == set(videos), f"{pass_type} must see videos"
    # the count agrees with the claim
    if scoped:
        assert n == len(claimed)


@pytest.mark.parametrize("pass_type", WORKER_PASSES)
def test_large_scope_branch_is_media_isolated_too(mixed, monkeypatch, pass_type):
    """count_unprocessed_photos switches to `+id IN (...)` above a threshold."""
    db, _bid, stills, videos = mixed
    _make_eligible(db, pass_type, stills + videos)
    monkeypatch.setattr(dbmod, "_SCOPE_DRIVE_FROM_INDEX_AT", 0)
    n = db.count_unprocessed_photos(pass_type, photo_ids=stills + videos)
    expected = {"still": len(stills), "video": len(videos)}.get(
        PASS_MEDIA[pass_type], len(stills) + len(videos))
    assert n == expected


def test_video_fragment_selects_only_videos(mixed):
    db, _bid, stills, videos = mixed
    got = {r[0] for r in db.conn.execute(
        f"SELECT id FROM photos WHERE {media_sql('video-probe')}")}
    assert got == set(videos)
    got = {r[0] for r in db.conn.execute(
        f"SELECT p.id FROM photos p WHERE p.id IN ({','.join('?' * 4)}) "
        f"AND {media_sql('clip', 'p')}", stills + videos)}
    assert got == set(stills)


@pytest.mark.parametrize("pass_type", WORKER_PASSES)
def test_batch_state_agrees_with_the_claim_predicate(mixed, pass_type):
    db, bid, stills, videos = mixed
    step = next(s for s in batch_state(db, bid)["steps"] if s["step"] == pass_type)
    assert step["remaining"] == db.count_unprocessed_photos(
        pass_type, photo_ids=stills + videos)
    expect_total = {"still": len(stills), "video": len(videos)}.get(
        PASS_MEDIA[pass_type], len(stills) + len(videos))
    assert step["total"] == expect_total


def _finish_still_passes(db, ids):
    for i, pid in enumerate(ids):
        vec = [0.0] * 512
        vec[i] = 1.0
        db.add_clip_embedding(pid, vec)
        db.conn.execute(
            "UPDATE photos SET aesthetic_score = 5, aesthetic_concepts = '[]', "
            "aes_overall = 6, verified_at = datetime('now'), visual_tags = '[]', "
            "categories = '[]', keywords = '[]' WHERE id = ?", (pid,))
    db.mark_processed(ids, "faces", terminal=True)
    db.conn.commit()


def test_still_passes_complete_with_unprocessed_videos_in_the_batch(mixed):
    db, bid, stills, videos = mixed
    _finish_still_passes(db, stills)
    steps = {s["step"]: s for s in batch_state(db, bid)["steps"]}
    for p in WORKER_PASSES:
        if PASS_MEDIA[p] == "still":
            assert steps[p]["state"] == "completed", (p, steps[p])
        elif p == "video-scan":
            # unprobed videos are not claimable yet: it waits on the probe
            assert steps[p]["state"] == "waiting", steps[p]
            assert steps[p]["waiting_on"] == "video-probe"
        elif p in ("video-subjects", "video-describe"):
            # and the subjects / description wait on the scan
            assert steps[p]["state"] == "waiting", steps[p]
            assert steps[p]["waiting_on"] == "video-scan"
        else:
            assert steps[p]["remaining"] == len(videos)


def test_a_video_only_batch_reads_still_passes_complete(db):
    vid = db.add_photo(filepath=f"{_DIR}/ONLY.MOV", filename="ONLY.MOV",
                       media_type="video", date_taken="2091-10-17 12:00:00")
    bid = register_batch(db, _DIR)
    steps = {s["step"]: s for s in batch_state(db, bid)["steps"]}
    for p in WORKER_PASSES:
        if PASS_MEDIA[p] == "still":
            assert steps[p]["total"] == 0
            assert steps[p]["state"] == "completed", (p, steps[p])
    # a video never sits in the stacking step's remaining count
    assert steps["stacking"]["remaining"] == 0
    assert vid


def test_exhausted_video_not_counted_as_a_still_failure(mixed):
    """`failed` is media-scoped too — a video with a stray exhausted ledger
    row must not read a still pass as blocked."""
    db, bid, stills, videos = mixed
    db.conn.execute(
        "INSERT INTO worker_processed (photo_id, pass_type, attempts) VALUES (?, 'describe', ?)",
        (videos[0], MAX_PROCESS_ATTEMPTS))
    db.conn.execute("UPDATE photos SET description = NULL WHERE id = ?", (videos[0],))
    db.conn.commit()
    step = next(s for s in batch_state(db, bid)["steps"] if s["step"] == "describe")
    assert step["failed"] == 0
