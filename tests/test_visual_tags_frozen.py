"""`sharp` / `blurry` are FROZEN: not derived, not asked, not deleted.

They were derived from `aes_sharpness` on the strength of agreement with the
VLM's own `blurry` tag — which is two models agreeing, not ground truth. On the
live DB that score's coverage turned out to be 98.9% (not the July copy's
2.4%), so the thresholds would have stamped `blurry` on 10,103 photos
(+8,819) and `sharp` on 15,076 (+11,666). Four photos scoring <= 2 were looked
at by hand: two genuinely blurry, two not (a tack-sharp phone photo of
construction formwork, a dark noisy GoPro night scene). `aes_sharpness`
conflates sharpness with general technical quality.

So: `derive_tags` never emits them, the prompt never offers them, and the
backfill leaves existing values exactly as they are.
"""

import json
import os

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from photosearch import describe as D
from photosearch import visual_tags_derive as V

DAYLIGHT = dict(exposure_time="1/800", f_number="28/5", iso=160,
                image_width=6000, image_height=4000)
NIGHT = dict(exposure_time="4", f_number="8", iso=100,
             image_width=6000, image_height=4000)


def _insert(db, pid, year, **exif):
    cols = ["id", "filepath", "filename", "date_taken"] + list(exif)
    vals = [pid, f"{year}/p{pid}.jpg", f"p{pid}.jpg",
            f"{year}-05-04T12:00:00"] + [exif[k] for k in exif]
    db.conn.execute(
        f"INSERT INTO photos ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        vals)


def _row(**kw):
    base = {"exposure_time": None, "f_number": None, "iso": None,
            "image_width": None, "image_height": None, "aes_sharpness": None}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# The four-way partition
# ---------------------------------------------------------------------------

def test_the_vocabulary_splits_four_ways_with_no_overlap():
    from photosearch.vocab_visual import VISUAL_VOCABULARY

    groups = {
        "derived": set(V.CAPTURE_FACT_TAGS),
        "retired": set(V.RETIRED_TAGS),
        "frozen": set(V.FROZEN_TAGS),
        "perceived": set(V.PERCEIVED_VOCABULARY),
    }
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (groups[a] & groups[b]), f"{a} overlaps {b}"
    union = set().union(*groups.values())
    assert union == set(VISUAL_VOCABULARY)


def test_the_groups_are_exactly_these_terms():
    assert set(V.CAPTURE_FACT_TAGS) == {"long-exposure", "low-light", "panoramic"}
    assert set(V.RETIRED_TAGS) == {"motion-blur"}
    assert set(V.FROZEN_TAGS) == {"sharp", "blurry"}
    assert len(V.PERCEIVED_VOCABULARY) == 30


# ---------------------------------------------------------------------------
# Never derived
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score", [1, 2, 3, 5, 7, 8, 9, 10])
def test_aes_sharpness_derives_nothing_at_any_score(score):
    assert V.derive_tags(_row(aes_sharpness=score)) == []


def test_a_fully_populated_row_derives_only_the_three_capture_facts():
    row = _row(exposure_time="4", f_number="8", iso=100,
               image_width=9000, image_height=3000, aes_sharpness=1)
    assert V.derive_tags(row) == ["long-exposure", "low-light", "panoramic"]


def test_no_threshold_constants_remain_for_the_frozen_terms():
    assert not hasattr(V, "SHARP_MIN_AES_SHARPNESS")
    assert not hasattr(V, "BLURRY_MAX_AES_SHARPNESS")


# ---------------------------------------------------------------------------
# Never asked of the VLM
# ---------------------------------------------------------------------------

def test_the_prompt_never_offers_a_frozen_term():
    prompt = D._build_visual_prompt(V.PERCEIVED_VOCABULARY)
    for term in V.FROZEN_TAGS:
        assert term not in prompt


def test_the_parser_drops_a_frozen_term_the_model_volunteers():
    vocab = set(V.PERCEIVED_VOCABULARY)
    assert D._parse_visual_response("sharp, sunny, blurry", vocab) == ["sunny"]


# ---------------------------------------------------------------------------
# Never deleted — the vlm vs stored distinction
# ---------------------------------------------------------------------------

def test_merge_vlm_answer_drops_a_frozen_term_from_fresh_output():
    assert V.merge_vlm_answer(["sharp", "peaceful"], _row()) == ["peaceful"]


def test_merge_vlm_answer_carries_over_a_previously_stored_frozen_term():
    """A re-tag must not silently delete data we have not decided about."""
    assert V.merge_vlm_answer(["peaceful"], _row(),
                              existing=["blurry", "joyful"]) == \
        ["blurry", "peaceful"]
    # Only the FROZEN half is carried; a stale perceived tag is replaced.
    assert "joyful" not in V.merge_vlm_answer(["peaceful"], _row(),
                                              existing=["joyful"])


def test_merge_vlm_answer_ignores_a_frozen_term_the_model_adds_on_top():
    assert V.merge_vlm_answer(["sharp", "peaceful"], _row(),
                              existing=["blurry"]) == ["blurry", "peaceful"]


def test_merge_stored_tags_passes_frozen_terms_through():
    assert V.merge_stored_tags(["sharp", "peaceful"], _row()) == \
        ["peaceful", "sharp"]
    assert V.merge_stored_tags(["blurry", "long-exposure"], _row()) == ["blurry"]


def test_merge_stored_tags_still_strips_derived_and_retired():
    row = _row(exposure_time="1/800", f_number="28/5", iso=160)
    assert V.merge_stored_tags(
        ["long-exposure", "low-light", "motion-blur", "blurry", "joyful"],
        row) == ["blurry", "joyful"]


def test_merge_stored_tags_accepts_the_raw_json_string():
    assert V.merge_stored_tags('["sharp", "peaceful"]', _row()) == \
        ["peaceful", "sharp"]


def test_both_entry_points_are_idempotent():
    row = _row(exposure_time="4", f_number="8", iso=100)
    once = V.merge_stored_tags(["sharp", "peaceful"], row)
    assert V.merge_stored_tags(once, row) == once
    v = V.merge_vlm_answer(["peaceful"], row, existing=["sharp"])
    assert V.merge_vlm_answer(["peaceful"], row, existing=v) == v


# ---------------------------------------------------------------------------
# The write path carries a stored frozen tag across a re-tag
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "x.db")
    monkeypatch.setenv("PHOTOSEARCH_DB", db_path)
    from photosearch import web, worker_api
    monkeypatch.setattr(web, "_db_path", db_path, raising=False)
    monkeypatch.setattr(worker_api, "_db_path", db_path, raising=False)
    monkeypatch.setattr(worker_api, "_shutting_down", False, raising=False)

    from photosearch.db import PhotoDB
    with PhotoDB(db_path) as db:
        _insert(db, 1, 2090, **DAYLIGHT)
        _insert(db, 2, 2091, **NIGHT)
        db.conn.commit()
    return TestClient(web.app)


def _submit(client, results):
    claim = client.post("/api/worker/claim-batch", json={
        "worker_id": "w1", "pass_type": "category-visual", "limit": 10,
    }).json()
    return client.post("/api/worker/submit-results", json={
        "batch_id": claim["batch_id"], "worker_id": "w1",
        "pass_type": "category-visual", "category_visual_results": results,
    })


def _stored(pid):
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        raw = db.conn.execute(
            "SELECT visual_tags FROM photos WHERE id=?", (pid,)).fetchone()[0]
    return None if raw is None else json.loads(raw)


def test_a_retag_keeps_a_frozen_tag_already_on_the_photo(client):
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        db.conn.execute("UPDATE photos SET visual_tags=? WHERE id=1",
                        (json.dumps(["blurry", "moody"]),))
        db.conn.commit()
    r = _submit(client, [{"photo_id": 1, "model": "llava",
                          "visual_tags": ["sunny", "joyful"]}])
    assert r.status_code == 200, r.text
    assert _stored(1) == ["blurry", "joyful", "sunny"]


def test_a_first_tag_gains_no_frozen_term(client):
    _submit(client, [{"photo_id": 2, "model": "llava",
                      "visual_tags": ["sharp", "peaceful"]}])
    assert _stored(2) == ["long-exposure", "low-light", "peaceful"]


def test_generations_never_records_a_frozen_term(client):
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        db.conn.execute("UPDATE photos SET visual_tags=? WHERE id=1",
                        (json.dumps(["blurry"]),))
        db.conn.commit()
    _submit(client, [{"photo_id": 1, "model": "llava", "visual_tags": ["sunny"]}])
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        text = db.conn.execute(
            "SELECT generated_text FROM generations "
            "WHERE photo_id=1 AND text_type='category-visual'").fetchone()[0]
    assert json.loads(text) == ["sunny"]


# ---------------------------------------------------------------------------
# The backfill leaves them at delta 0
# ---------------------------------------------------------------------------

@pytest.fixture
def backfill_db(tmp_path):
    from photosearch.db import PhotoDB
    path = str(tmp_path / "b.db")
    with PhotoDB(path) as db:
        _insert(db, 1, 2090, **DAYLIGHT)
        _insert(db, 2, 2091, **NIGHT)
        _insert(db, 3, 2090, aes_sharpness=1)
        db.conn.execute("UPDATE photos SET visual_tags=? WHERE id=1",
                        (json.dumps(["blurry", "long-exposure", "joyful"]),))
        db.conn.execute("UPDATE photos SET visual_tags=? WHERE id=2",
                        (json.dumps(["sharp", "motion-blur", "peaceful"]),))
        db.conn.execute("UPDATE photos SET visual_tags=? WHERE id=3",
                        (json.dumps(["peaceful"]),))
        db.conn.commit()
    return path


def _run(path, *args):
    from cli import cli
    return CliRunner().invoke(cli, ["derive-visual-tags", "--db", path, *args])


def test_the_dry_run_table_shows_no_frozen_delta(backfill_db):
    res = _run(backfill_db)
    assert res.exit_code == 0, res.output
    rows = {ln.split()[0]: ln for ln in res.output.splitlines()
            if ln and ln.split() and ln.split()[0] in
            ("sharp", "blurry", "long-exposure", "low-light", "motion-blur",
             "panoramic")}
    assert rows["sharp"].split()[-1] == "+0", rows["sharp"]
    assert rows["blurry"].split()[-1] == "+0", rows["blurry"]
    assert rows["motion-blur"].split()[-1] == "-1", rows["motion-blur"]


def test_the_backfill_preserves_frozen_tags_on_apply(backfill_db):
    _run(backfill_db, "--apply")
    from photosearch.db import PhotoDB
    with PhotoDB(backfill_db) as db:
        rows = {r[0]: json.loads(r[1]) for r in
                db.conn.execute("SELECT id, visual_tags FROM photos "
                                "WHERE visual_tags IS NOT NULL")}
    assert rows[1] == ["blurry", "joyful"]          # long-exposure stripped
    assert rows[2] == ["long-exposure", "low-light", "peaceful", "sharp"]
    assert rows[3] == ["peaceful"]                  # no frozen tag invented


def test_the_backfill_never_invents_a_frozen_tag_from_aes_sharpness(backfill_db):
    _run(backfill_db, "--apply")
    from photosearch.db import PhotoDB
    with PhotoDB(backfill_db) as db:
        raw = db.conn.execute(
            "SELECT visual_tags FROM photos WHERE id=3").fetchone()[0]
    assert "blurry" not in json.loads(raw)
