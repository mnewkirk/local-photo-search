import json
import os

import pytest
from fastapi.testclient import TestClient

from photosearch import aesthetics as A


def _valid_payload(**overrides):
    """A well-formed model response covering all 11 sub-attributes."""
    payload = {
        "technical": {"critique": "sharp, well exposed",
                      "sharpness": 8, "exposure": 7,
                      "depth_of_field": 6, "white_balance": 7},
        "composition": {"critique": "strong thirds",
                        "framing": 7, "leading_lines": 6,
                        "rule_of_thirds": 8, "balance": 7},
        "impact": {"critique": "striking moment",
                   "emotion": 9, "originality": 7, "wow": 8},
        "style": {"lighting": "golden-hour backlight", "mood": "serene",
                  "tonal_character": "warm", "color": "muted earth tones",
                  "composition_notes": "layered foreground"},
        "style_tags": ["golden-hour", "serene", "warm-tones"],
    }
    payload.update(overrides)
    return payload


# --- prompt ---------------------------------------------------------------

def test_prompt_mentions_all_subattrs_and_rubric():
    prompt = A.build_aesthetics_prompt()
    for sub in A.ALL_SUBATTRS:
        assert sub in prompt
    # rubric anchors present so the model uses the full range
    assert "MOST photos" in prompt
    assert "1 to 10" in prompt


def test_prompt_embeds_style_vocab():
    prompt = A.build_aesthetics_prompt(["golden-hour", "moody"])
    assert "golden-hour" in prompt
    assert "moody" in prompt


# --- JSON extraction ------------------------------------------------------

def test_extract_json_from_fenced_and_prefixed():
    raw = 'Sure! Here you go:\n```json\n{"a": 1, "b": "x"}\n```\nthanks'
    assert A._extract_json(raw) == {"a": 1, "b": "x"}


def test_extract_json_handles_braces_in_strings():
    raw = '{"note": "a } brace { inside", "n": 3}'
    assert A._extract_json(raw) == {"note": "a } brace { inside", "n": 3}


def test_extract_json_returns_none_on_garbage():
    assert A._extract_json("no json here") is None
    assert A._extract_json("") is None


# --- score coercion -------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (7, 7.0),
    (7.5, 7.5),
    ("8", 8.0),
    ("score: 9/10", 9.0),
    (0, 1.0),      # clamp low
    (99, 10.0),    # clamp high
    (True, None),  # bool rejected
    ("N/A", None),
    (None, None),
])
def test_coerce_score(raw, expected):
    assert A._coerce_score(raw) == expected


# --- parse ----------------------------------------------------------------

def test_parse_valid_response():
    raw = "```json\n" + json.dumps(_valid_payload()) + "\n```"
    out = A.parse_aesthetics_response(raw)
    assert out is not None
    assert set(out["sub_scores"]) == set(A.ALL_SUBATTRS)
    assert out["sub_scores"]["emotion"] == 9.0
    assert set(out["dim_scores"]) == {"technical", "composition", "impact"}
    assert out["critiques"]["impact"] == "striking moment"
    assert out["style"]["mood"] == "serene"
    assert out["style_tags"] == ["golden-hour", "serene", "warm-tones"]
    # overall is weighted, dominated by impact (weight 0.40)
    assert 6.5 < out["overall"] < 8.5


def test_parse_missing_subattr_returns_none():
    payload = _valid_payload()
    del payload["technical"]["sharpness"]
    assert A.parse_aesthetics_response(json.dumps(payload)) is None


def test_parse_missing_dimension_returns_none():
    payload = _valid_payload()
    del payload["impact"]
    assert A.parse_aesthetics_response(json.dumps(payload)) is None


def test_parse_out_of_range_scores_are_clamped_not_rejected():
    payload = _valid_payload()
    payload["technical"]["sharpness"] = 15
    payload["technical"]["exposure"] = -3
    out = A.parse_aesthetics_response(json.dumps(payload))
    assert out["sub_scores"]["sharpness"] == 10.0
    assert out["sub_scores"]["exposure"] == 1.0


def test_parse_dedupes_style_tags_case_insensitively():
    payload = _valid_payload(style_tags=["Golden-Hour", "golden-hour", "MOODY"])
    out = A.parse_aesthetics_response(json.dumps(payload))
    assert out["style_tags"] == ["golden-hour", "moody"]


def test_parse_non_json_returns_none():
    assert A.parse_aesthetics_response("the photo is nice") is None
    assert A.parse_aesthetics_response("") is None


# --- compute_overall ------------------------------------------------------

def test_compute_overall_weights_impact_highest():
    subs = {s: 5.0 for s in A.ALL_SUBATTRS}
    subs["emotion"] = subs["originality"] = subs["wow"] = 10.0  # impact dim = 10
    dims, overall = A.compute_overall(subs)
    assert dims["technical"] == 5.0
    assert dims["composition"] == 5.0
    assert dims["impact"] == 10.0
    # 0.3*5 + 0.3*5 + 0.4*10 = 7.0
    assert overall == 7.0


def test_compute_overall_technical_override():
    subs = {s: 5.0 for s in A.ALL_SUBATTRS}
    dims, overall = A.compute_overall(subs, technical_override=9.0)
    assert dims["technical"] == 9.0
    # 0.3*9 + 0.3*5 + 0.4*5 = 6.2
    assert overall == 6.2


def test_compute_overall_custom_weights():
    subs = {s: 5.0 for s in A.ALL_SUBATTRS}
    subs["emotion"] = subs["originality"] = subs["wow"] = 8.0
    _, overall = A.compute_overall(
        subs, weights={"technical": 1.0, "composition": 0.0, "impact": 0.0})
    assert overall == 5.0  # only technical counts


# --- percentile -----------------------------------------------------------

def test_percentile_ranks_spread_and_order():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    pct = A.percentile_ranks(vals)
    assert pct == sorted(pct)          # monotonic with value
    assert pct[0] == 10.0              # 100*(0+0.5)/5
    assert pct[-1] == 90.0             # 100*(4+0.5)/5
    assert pct[2] == 50.0              # median at ~50


def test_percentile_ranks_ties_share_rank():
    pct = A.percentile_ranks([5.0, 5.0, 9.0])
    assert pct[0] == pct[1]            # tied values share percentile
    assert pct[2] > pct[0]


def test_percentile_ranks_edge_cases():
    assert A.percentile_ranks([]) == []
    assert A.percentile_ranks([7.0]) == [50.0]


# --- score_photo_aesthetics (mocked chat) ---------------------------------

def test_score_photo_aesthetics_happy_path(monkeypatch, tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0notreallyajpeg")
    from photosearch import describe as d
    monkeypatch.setattr(d, "_encode_image_for_ollama", lambda p: "b64data")
    monkeypatch.setattr(
        d, "_ollama_chat_with_retry",
        lambda **kw: json.dumps(_valid_payload()))
    out = A.score_photo_aesthetics(str(img))
    assert out is not None
    assert out["overall"] > 0


def test_score_photo_aesthetics_retries_then_gives_up(monkeypatch, tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0x")
    calls = []
    from photosearch import describe as d
    monkeypatch.setattr(d, "_encode_image_for_ollama", lambda p: "b64")

    def _bad(**kw):
        calls.append(1)
        return "not json at all"
    monkeypatch.setattr(d, "_ollama_chat_with_retry", _bad)
    out = A.score_photo_aesthetics(str(img))
    assert out is None
    assert len(calls) == 2  # one retry with a temperature bump


def test_score_photo_aesthetics_missing_file_returns_none():
    assert A.score_photo_aesthetics("/nonexistent/nope.jpg") is None


# --- worker pass integration (claim -> submit -> DB) ----------------------

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
        db.conn.executemany(
            "INSERT INTO photos (id, filepath, filename) VALUES (?, ?, ?)",
            [(1, "a.jpg", "a.jpg"), (2, "b.jpg", "b.jpg")],
        )
        db.conn.commit()
    return TestClient(web.app)


def test_aesthetics_claim_and_submit_writes_columns(client):
    claim = client.post("/api/worker/claim-batch", json={
        "worker_id": "w1", "pass_type": "aesthetics", "limit": 10,
    }).json()
    assert claim["pass_type"] == "aesthetics"
    assert {p["id"] for p in claim["photos"]} == {1, 2}
    batch_id = claim["batch_id"]

    scores = {f"aes_{s}": 7.0 for s in A.ALL_SUBATTRS}
    scores.update({"aes_technical": 7.0, "aes_composition": 7.0,
                   "aes_impact": 8.0, "aes_overall": 7.4})
    r = client.post("/api/worker/submit-results", json={
        "batch_id": batch_id, "worker_id": "w1", "pass_type": "aesthetics",
        "aesthetics_results": [
            {"photo_id": 1, "scores": scores,
             "aes_style": json.dumps({"facets": {"mood": "serene"}}),
             "aes_style_tags": json.dumps(["golden-hour"]),
             "model": "qwen2.5-vl"},
            # photo 2 was deferred (empty scores) — should mark processed but
            # write no aes_overall
            {"photo_id": 2, "scores": {}},
        ],
    })
    assert r.status_code == 200, r.text

    with A_db() as db:
        rows = {row["id"]: row for row in db.conn.execute(
            "SELECT id, aes_overall, aes_impact, aes_style_tags, aes_model, aes_scored_at "
            "FROM photos").fetchall()}
        assert rows[1]["aes_overall"] == 7.4
        assert rows[1]["aes_impact"] == 8.0
        assert json.loads(rows[1]["aes_style_tags"]) == ["golden-hour"]
        assert rows[1]["aes_model"] == "qwen2.5-vl"
        assert rows[1]["aes_scored_at"] is not None
        assert rows[2]["aes_overall"] is None  # deferred, not written
        # generations provenance row for the scored photo
        tt = {r[0] for r in db.conn.execute(
            "SELECT text_type FROM generations").fetchall()}
        assert "aesthetics" in tt


def test_aesthetics_scored_photo_not_reclaimed(client):
    """Once aes_overall is set, the photo drops out of the claim set."""
    claim = client.post("/api/worker/claim-batch", json={
        "worker_id": "w1", "pass_type": "aesthetics", "limit": 10,
    }).json()
    batch_id = claim["batch_id"]
    scores = {f"aes_{s}": 5.0 for s in A.ALL_SUBATTRS}
    scores["aes_overall"] = 5.0
    client.post("/api/worker/submit-results", json={
        "batch_id": batch_id, "worker_id": "w1", "pass_type": "aesthetics",
        "aesthetics_results": [{"photo_id": 1, "scores": scores}],
    })
    # photo 1 is done; only photo 2 remains claimable
    reclaim = client.post("/api/worker/claim-batch", json={
        "worker_id": "w2", "pass_type": "aesthetics", "limit": 10,
    }).json()
    assert {p["id"] for p in reclaim["photos"]} == {2}


def A_db():
    from photosearch.db import PhotoDB
    return PhotoDB(os.environ["PHOTOSEARCH_DB"])


# --- backfill helpers (recompute overall + normalize percentile) ----------

def _seed_scored(db, rows):
    """rows: list of (id, overall_base) — fills all sub-attrs to base."""
    for pid, base in rows:
        cols = {f"aes_{s}": base for s in A.ALL_SUBATTRS}
        cols["aes_overall"] = base
        db.conn.execute("INSERT INTO photos (id, filepath, filename) VALUES (?,?,?)",
                        (pid, f"{pid}.jpg", f"{pid}.jpg"))
        db.update_photo(pid, **cols)
    db.conn.commit()


def test_recompute_overall_applies_new_weights(tmp_path):
    from photosearch.db import PhotoDB
    p = str(tmp_path / "r.db")
    with PhotoDB(p) as db:
        # photo with impact higher than technical/composition
        cols = {f"aes_{s}": 5.0 for s in A.ALL_SUBATTRS}
        cols["aes_emotion"] = cols["aes_originality"] = cols["aes_wow"] = 10.0
        cols["aes_overall"] = 0.0  # stale
        db.conn.execute("INSERT INTO photos (id, filepath, filename) VALUES (1,'a','a')")
        db.update_photo(1, **cols)
        db.conn.commit()
        n = A.recompute_overall_scores(db, apply=True)
        assert n == 1
        row = db.conn.execute(
            "SELECT aes_overall, aes_impact FROM photos WHERE id=1").fetchone()
        assert row["aes_impact"] == 10.0
        # default weights: 0.3*5 + 0.3*5 + 0.4*10 = 7.0
        assert row["aes_overall"] == 7.0


def test_normalize_overall_writes_percentiles(tmp_path):
    from photosearch.db import PhotoDB
    p = str(tmp_path / "n.db")
    with PhotoDB(p) as db:
        _seed_scored(db, [(1, 3.0), (2, 5.0), (3, 9.0)])
        # dry-run writes nothing
        assert A.normalize_overall(db, apply=False) == 3
        assert db.conn.execute(
            "SELECT COUNT(*) FROM photos WHERE aes_overall_pct IS NOT NULL"
        ).fetchone()[0] == 0
        A.normalize_overall(db, apply=True)
        pcts = {r["id"]: r["aes_overall_pct"] for r in db.conn.execute(
            "SELECT id, aes_overall_pct FROM photos").fetchall()}
        assert pcts[1] < pcts[2] < pcts[3]
        assert pcts[3] > 80  # top photo reads top-tier


def test_normalize_overall_by_day_ranks_within_each_day(tmp_path):
    from photosearch.db import PhotoDB
    p = str(tmp_path / "day.db")
    with PhotoDB(p) as db:
        # Day A has a low(3) + high(9); day B a low(4) + high(5). Globally 9>5>4>3,
        # but per-day each day's best should read high and worst low.
        seed = [(1, 3.0, "2026-06-28"), (2, 9.0, "2026-06-28"),
                (3, 4.0, "2026-06-29"), (4, 5.0, "2026-06-29")]
        for pid, base, day in seed:
            cols = {f"aes_{s}": base for s in A.ALL_SUBATTRS}
            cols["aes_overall"] = base
            db.conn.execute(
                "INSERT INTO photos (id, filepath, filename, date_taken) VALUES (?,?,?,?)",
                (pid, f"{pid}.jpg", f"{pid}.jpg", day + " 10:00:00"))
            db.update_photo(pid, **cols)
        db.conn.commit()

        assert A.normalize_overall_by_day(db, apply=False) == 4  # dry-run no write
        assert db.conn.execute(
            "SELECT COUNT(*) FROM photos WHERE aes_overall_day_pct IS NOT NULL"
        ).fetchone()[0] == 0

        A.normalize_overall_by_day(db, apply=True)
        d = {r["id"]: r["aes_overall_day_pct"] for r in db.conn.execute(
            "SELECT id, aes_overall_day_pct FROM photos").fetchall()}
        # Each day's worst == 25, best == 75 (n=2 per day) — so photo 4 (global
        # score 5, day B's best) outranks photo 1 (score 3) AND photo 3 (score 4),
        # which a library-wide percentile would not do.
        assert d[1] < d[2] and d[3] < d[4]
        assert d[2] == d[4]        # both days' best rank equally per-day
        assert d[1] == d[3]        # both days' worst rank equally per-day


def test_normalize_by_day_skips_undated(tmp_path):
    from photosearch.db import PhotoDB
    with PhotoDB(str(tmp_path / "u.db")) as db:
        cols = {f"aes_{s}": 5.0 for s in A.ALL_SUBATTRS}
        cols["aes_overall"] = 5.0
        db.conn.execute("INSERT INTO photos (id, filepath, filename) VALUES (1,'a','a')")
        db.update_photo(1, **cols)  # no date_taken/date_created
        db.conn.commit()
        A.normalize_overall_by_day(db, apply=True)
        assert db.conn.execute(
            "SELECT aes_overall_day_pct FROM photos WHERE id=1").fetchone()[0] is None


def test_search_aesthetic_sort_and_filters(client):
    # score both photos, normalize percentiles
    from photosearch.db import PhotoDB
    with A_db() as db:
        for pid, base, tags in [(1, 3.0, ["moody"]), (2, 9.0, ["golden-hour"])]:
            cols = {f"aes_{s}": base for s in A.ALL_SUBATTRS}
            cols.update({"aes_overall": base, "aes_technical": base,
                         "aes_composition": base, "aes_impact": base,
                         "aes_style_tags": json.dumps(tags)})
            db.update_photo(pid, **cols)
        db.conn.commit()
        A.normalize_overall(db, apply=True)

    # sort=aesthetic_desc → highest percentile (photo 2) first
    r = client.get("/api/search", params={"sort": "aesthetic_desc"}).json()
    ids = [it["id"] for it in r["results"]]
    assert ids[0] == 2

    # min_aesthetic percentile filter keeps only the top photo
    r = client.get("/api/search", params={"min_aesthetic": 60}).json()
    assert {it["id"] for it in r["results"]} == {2}

    # style_tag filter
    r = client.get("/api/search", params={"style_tag": "golden-hour"}).json()
    assert {it["id"] for it in r["results"]} == {2}

    # per-dimension filter
    r = client.get("/api/search", params={"min_impact": 8}).json()
    assert {it["id"] for it in r["results"]} == {2}


def test_photo_detail_returns_aesthetics_breakdown(client):
    with A_db() as db:
        cols = {f"aes_{s}": 7.0 for s in A.ALL_SUBATTRS}
        cols.update({"aes_overall": 7.4, "aes_overall_pct": 88.0,
                     "aes_technical": 7.0, "aes_composition": 7.0, "aes_impact": 8.0,
                     "aes_style": json.dumps({"facets": {"mood": "serene"},
                                              "critiques": {"impact": "striking"}}),
                     "aes_style_tags": json.dumps(["golden-hour"]),
                     "aes_model": "qwen2.5-vl"})
        db.update_photo(1, **cols)
        db.conn.commit()
    detail = client.get("/api/photos/1").json()
    aes = detail["aesthetics"]
    assert aes["overall"] == 7.4
    assert aes["overall_pct"] == 88.0
    assert aes["dimensions"]["impact"]["score"] == 8.0
    assert aes["dimensions"]["impact"]["critique"] == "striking"
    assert aes["style"]["mood"] == "serene"
    assert aes["style_tags"] == ["golden-hour"]


def test_normalize_aesthetics_maintenance_stage(tmp_path):
    from photosearch.db import PhotoDB
    from photosearch.maintenance import run_maintenance_sweep
    p = str(tmp_path / "m.db")
    with PhotoDB(p) as db:
        _seed_scored(db, [(1, 4.0), (2, 8.0)])
        res = run_maintenance_sweep(db, apply=True)
    stages = {s["stage"]: s for s in res["stages"]}
    assert "normalize_aesthetics" in stages
    assert stages["normalize_aesthetics"]["applied"] == 2
    with PhotoDB(p) as db:
        assert db.conn.execute(
            "SELECT COUNT(*) FROM photos WHERE aes_overall_pct IS NULL"
        ).fetchone()[0] == 0
