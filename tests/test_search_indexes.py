"""Schema v34 search indexes and the composed search scope.

docs/plans/search-indexes.md. On the NAS a search's cost is bytes read from a
spinning disk with a cold cache, and every un-indexed filter was a ~545 MB scan
of `photos`; date x person x camera x location read ~1.9 GB because each filter
ran over the whole library before the sets were intersected in Python.

Two kinds of test:

- **Plans.** The SQL production actually ran is captured with
  `set_trace_callback` and EXPLAINed, so a reformat that silently stops an index
  being used (expression and partial indexes match textually) fails here.
- **Answers.** The composed scope is an optimisation only: every combination
  must return exactly what the unscoped path returns.
"""

import sqlite3

import pytest

from photosearch import search as search_mod
from photosearch.db import (
    PhotoDB, SCHEMA_VERSION, _SEARCH_INDEXES, _SUPERSEDED_INDEXES,
)
from photosearch.search import search_combined


@pytest.fixture(autouse=True)
def _no_nominatim(monkeypatch):
    # Location search falls back to a Nominatim bbox; never hit the network.
    monkeypatch.setattr(search_mod, "_resolve_location_bbox", lambda db, name: None)


@pytest.fixture
def lib(tmp_path):
    db = PhotoDB(str(tmp_path / "lib.db"))
    calvin = db.add_person("Calvin")
    ellie = db.add_person("Ellie")
    rows = [
        # (camera, date_taken, place, country, admin2, persons, visual, sources)
        ("ILCE-7RM6", "2026-09-26 09:00:00", "San Rafael, California, US", "US",
         "Marin County", [calvin], '["sunny"]', "strict"),
        ("ILCE-7RM6", "2026-09-26 23:59:59", "San Rafael, California, US", "US",
         "Marin County", [calvin, ellie], '["sunny", "joyful"]', "manual"),
        ("ILCE-7RM6", "2026-09-27 12:00:00", "Varenna, Lombardy, IT", "IT",
         "Lecco", [ellie], '["moody"]', "strict"),
        ("ILCE-7RM6", "2026-09-28 00:00:00", "San Rafael, California, US", "US",
         "Marin County", [calvin], '["sunny"]', "temporal"),
        ("ILCE-7M4", "2026-09-26 10:00:00", "San Rafael, California, US", "US",
         "marin county", [calvin], '["Sunny"]', "strict"),
        ("ILCE-7M4", "2025-06-01 10:00:00", "Varenna, Lombardy, IT", "IT",
         "Lecco", [calvin, ellie], "[]", "strict"),
        ("ILCE-7RM6", None, "San Rafael, California, US", "US",
         "Marin County", [calvin], '["sunny"]', "strict"),
        ("Pixel 8", "2026-09-27 08:00:00", None, None, None, [], None, None),
    ]
    ids = []
    for i, (cam, dt, place, country, admin2, persons, vis, source) in enumerate(rows):
        pid = db.add_photo(
            filepath=f"2026/f/IMG_{i}.JPG", filename=f"IMG_{i}.JPG",
            camera_model=cam, date_taken=dt, place_name=place,
            country=country, admin2=admin2, visual_tags=vis,
            categories='["sports"]' if i % 2 else '["travel"]',
            keywords='["soccer game"]' if i < 3 else '["lake"]',
            aesthetic_score=4.0 + i * 0.5, file_hash=f"h{i}")
        ids.append(pid)
        for person in persons:
            fid = db.add_face(pid, (0, 10, 10, 0), [], person_id=person)
            db.conn.execute("UPDATE faces SET match_source = ? WHERE id = ?",
                            (source, fid))
    # A stranger's face beside Calvin, so person EXISTS has to discriminate.
    db.add_face(ids[0], (20, 30, 30, 20), [])
    # Videos (schema v35): a playable clip, a not-yet-probed one, and the two
    # kinds default search hides — a paired Live clip and an unreadable file.
    # Kept OUT of `ids` so the per-photo fixtures below are unchanged.
    for name, cam, dt, meta in (
            ("CLIP.MP4", "ILCE-7RM6", "2026-09-26 11:00:00",
             {"probe_version": 1, "kind": "clip", "duration_s": 42.0}),
            ("NEW.MOV", "Pixel 8", "2026-09-27 09:00:00", {}),
            ("IMG_0.MOV", "ILCE-7RM6", "2026-09-26 09:00:01",
             {"probe_version": 1, "kind": "live", "live_of_photo_id": ids[0],
              "duration_s": 2.5}),
            ("BAD.MP4", "ILCE-7RM6", "2026-09-27 10:00:00",
             {"probe_version": 1, "probe_error": "moov atom not found"})):
        vid = db.add_photo(filepath=f"2026/f/{name}", filename=name,
                           camera_model=cam, date_taken=dt, media_type="video",
                           keywords='["soccer game"]', categories='["sports"]',
                           visual_tags='["sunny"]', country="US",
                           admin2="Marin County",
                           place_name="San Rafael, California, US")
        cols = ["photo_id"] + list(meta)
        db.conn.execute(
            f"INSERT INTO video_meta ({', '.join(cols)}) VALUES "
            f"({', '.join('?' * len(cols))})", [vid, *meta.values()])
    db.conn.commit()
    yield db, ids
    db.conn.close()


def _traced(db, fn):
    """Run fn(), returning every SQL statement it executed (bound values
    expanded, so each one can be EXPLAINed as-is)."""
    seen: list[str] = []
    db.conn.set_trace_callback(seen.append)
    try:
        fn()
    finally:
        db.conn.set_trace_callback(None)
    return [s for s in seen if s.lstrip().upper().startswith("SELECT")]


def _plan(db, sql):
    return " | ".join(r[3] for r in db.conn.execute("EXPLAIN QUERY PLAN " + sql))


def _assert_one_face_seek(plan):
    """The person EXISTS must be a point lookup on BOTH columns. With only
    idx_faces_person it walked every face of the person for every candidate
    photo (16 GB read for Calvin + two days, measured 2026-10-04)."""
    assert ("(person_id=? AND photo_id=?)" in plan
            or "(photo_id=? AND person_id=?)" in plan), plan


def _plans_touching(db, statements, needle):
    return [_plan(db, s) for s in statements if needle in s]


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def test_v33_db_migrates_to_v34_indexes(tmp_path):
    path = str(tmp_path / "v33.db")
    with PhotoDB(path) as db:
        pid = db.add_photo(filepath="a.jpg", filename="a.jpg", camera_model="X")
    # Rebuild the v33 index set.
    conn = sqlite3.connect(path)
    for name, _ in _SEARCH_INDEXES:
        conn.execute(f"DROP INDEX IF EXISTS {name}")
    conn.executescript("""
        CREATE INDEX idx_faces_photo ON faces(photo_id);
        CREATE INDEX idx_faces_person ON faces(person_id);
        CREATE INDEX idx_photos_country ON photos(country);
        CREATE INDEX idx_photos_admin1 ON photos(admin1);
        CREATE INDEX idx_photos_admin2 ON photos(admin2);
        CREATE INDEX idx_photos_locality ON photos(locality);
        CREATE INDEX idx_stack_members_photo ON stack_members(photo_id);
        CREATE INDEX idx_stack_members_stack ON stack_members(stack_id);
        CREATE INDEX idx_collection_photos_coll ON collection_photos(collection_id);
        UPDATE schema_info SET value = '33' WHERE key = 'version';
    """)
    conn.commit()
    conn.close()

    with PhotoDB(path) as db:
        names = {r[0] for r in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'")}
        assert {n for n, _ in _SEARCH_INDEXES} <= names
        assert not names & set(_SUPERSEDED_INDEXES)
        # Not a duplicate of anything: must survive.
        assert "idx_collection_photos_photo" in names
        assert db.conn.execute("SELECT camera_model FROM photos WHERE id = ?",
                               (pid,)).fetchone()[0] == "X"
        version = db.conn.execute(
            "SELECT value FROM schema_info WHERE key = 'version'").fetchone()[0]
        assert int(version) == SCHEMA_VERSION == 38


def test_fresh_db_has_v34_indexes_and_no_superseded_ones(tmp_path):
    with PhotoDB(str(tmp_path / "fresh.db")) as db:
        names = {r[0] for r in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert {n for n, _ in _SEARCH_INDEXES} <= names
    assert not names & set(_SUPERSEDED_INDEXES)


# ---------------------------------------------------------------------------
# Plans: the SQL production runs uses the indexes
# ---------------------------------------------------------------------------

def test_camera_and_date_use_the_composite_index(lib):
    db, _ = lib
    stmts = _traced(db, lambda: search_combined(
        db, camera="ILCE-7RM6", date_from="2026-09-26", date_to="2026-09-27"))
    plans = _plans_touching(db, stmts, "camera_model")
    assert plans
    for plan in plans:
        assert "idx_photos_camera_date (camera_model=? AND date_taken>" in plan
        assert "SCAN photos" not in plan


def test_composed_scope_checks_people_with_one_seek_per_photo(lib):
    db, _ = lib
    stmts = _traced(db, lambda: search_combined(
        db, person="Calvin", camera="ILCE-7RM6",
        date_from="2026-09-26", date_to="2026-09-27"))
    scope = [s for s in stmts if s.startswith("SELECT p.id FROM photos p")]
    assert len(scope) == 1
    plan = _plan(db, scope[0])
    assert "idx_photos_camera_date" in plan
    _assert_one_face_seek(plan)


def test_people_only_scope_is_driven_from_faces(lib):
    db, _ = lib
    stmts = _traced(db, lambda: search_combined(db, query="Calvin and Ellie",
                                                location="San Rafael"))
    scope = [s for s in stmts if "FROM faces f0" in s]
    assert len(scope) == 1
    plan = _plan(db, scope[0])
    assert "idx_faces_person_photo (person_id=?)" in plan
    _assert_one_face_seek(plan.split("CORRELATED", 1)[1])


def test_structured_location_uses_nocase_indexes(lib):
    db, _ = lib
    stmts = _traced(db, lambda: search_combined(db, location="Marin County"))
    plans = _plans_touching(db, stmts, "COLLATE NOCASE")
    assert plans
    for name in ("country", "admin1", "admin2", "locality"):
        assert f"idx_photos_{name}_nc" in plans[0]
    assert "MULTI-INDEX OR" in plans[0]


def test_tag_filter_scans_a_covering_index_even_with_a_wide_date_range(lib):
    db, _ = lib
    stmts = _traced(db, lambda: search_combined(
        db, visual_tag="sunny", date_from="2000-01-01", date_to="2030-12-31"))
    plans = _plans_touching(db, stmts, "visual_tags IS NOT NULL")
    assert plans
    assert "COVERING INDEX idx_photos_visual_tags" in plans[0]
    assert "idx_photos_date" not in plans[0]


def test_min_quality_uses_the_expression_index(lib):
    db, _ = lib
    stmts = _traced(db, lambda: search_combined(db, min_quality=5.0))
    plans = _plans_touching(db, stmts, "COALESCE(aes_overall, aesthetic_score)")
    assert plans and "idx_photos_raw_quality" in plans[0]


def test_subject_aesthetic_sort_uses_the_expression_index(lib):
    db, _ = lib
    stmts = _traced(db, lambda: search_combined(db, sort="subject_aesthetic_desc"))
    plans = _plans_touching(db, stmts, "aes_subject_overall_pct")
    assert plans and "idx_photos_subject_aes" in plans[0]
    assert "TEMP B-TREE" not in plans[0]


@pytest.mark.parametrize("pass_type,index", [
    ("describe", "idx_photos_need_describe"),
    ("verify", "idx_photos_need_verify"),
    ("quality", "idx_photos_need_quality"),
])
def test_worker_counts_use_the_partial_indexes(lib, pass_type, index):
    db, _ = lib
    stmts = _traced(db, lambda: db.count_unprocessed_photos(pass_type))
    stmts += _traced(db, lambda: db.get_unprocessed_photos(pass_type))
    plans = [_plan(db, s) for s in stmts if "FROM photos" in s]
    assert plans
    for plan in plans:
        assert index in plan, plan


def test_ingest_hash_lookup_and_map_query_are_indexed(lib):
    db, _ = lib
    import inspect
    from photosearch import ingest, web
    lookup = "SELECT id, filepath FROM photos WHERE file_hash = ? LIMIT 1"
    assert lookup in inspect.getsource(ingest)
    assert "idx_photos_file_hash" in _plan(db, lookup.replace("?", "'h1'"))

    geojson = ("SELECT id, gps_lat, gps_lon, location_source, date_taken, place_name "
               "FROM photos WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL")
    web_src = inspect.getsource(web.api_photos_geojson)
    assert all(part.strip() in web_src for part in geojson.split("FROM"))
    assert "COVERING INDEX idx_photos_gps_cover" in _plan(db, geojson)


def test_photo_stack_lookup_still_indexed_after_dropping_duplicates(lib):
    db, ids = lib
    db.create_stack(ids[:2])
    stmts = _traced(db, lambda: db.get_photo_stack(ids[0]))
    plans = [_plan(db, s) for s in stmts if "stack_members" in s]
    assert len(plans) == 2
    for plan in plans:
        assert "SCAN" not in plan, plan


# ---------------------------------------------------------------------------
# Answers: the composed scope never changes what a search returns
# ---------------------------------------------------------------------------

COMBOS = [
    dict(camera="ILCE-7RM6", date_from="2026-09-26", date_to="2026-09-27"),
    dict(camera="ILCE-7RM6", date_from="2026-09-26"),
    dict(person="Calvin", date_from="2026-09-26", date_to="2026-09-26"),
    dict(person="Calvin", camera="ILCE-7RM6", date_from="2026-09-26",
         date_to="2026-09-28"),
    dict(person="Calvin", camera="ILCE-7RM6", location="San Rafael",
         date_from="2026-09-26", date_to="2026-09-27"),
    dict(person="Calvin", location="marin county"),
    dict(person="Calvin", match_source="strict", date_from="2025-01-01"),
    dict(query="Calvin and Ellie", camera="ILCE-7M4"),
    dict(query="Calvin and Ellie", date_from="2025-01-01", date_to="2026-12-31"),
    dict(person_ids=[1, 2], location="Varenna"),
    dict(location="US", date_from="2026-09-26", date_to="2026-09-27"),
    dict(visual_tag="sunny", person="Calvin"),
    dict(visual_tag="SUNNY", date_from="2026-09-26", date_to="2026-09-28"),
    dict(keyword="soccer", camera="ILCE-7RM6"),
    dict(category="sports", person="Ellie", date_from="2025-01-01"),
    dict(person="Nobody", camera="ILCE-7RM6"),
    dict(camera="Pixel 8", person="Calvin"),
    # Media (video-support step 2): Videos composes into the scope.
    dict(media="video", date_from="2026-09-26", date_to="2026-09-27"),
    dict(media="video", camera="ILCE-7RM6"),
    dict(media="video", location="Marin County"),
    dict(media="video", keyword="soccer"),
    dict(media="video", person="Calvin"),
    dict(media="photo", camera="ILCE-7RM6", date_from="2026-09-26"),
    dict(media="photo", visual_tag="sunny", date_from="2026-09-26"),
]


@pytest.mark.parametrize("kw", COMBOS, ids=lambda kw: ",".join(sorted(kw)))
def test_scoped_results_match_unscoped(lib, monkeypatch, kw):
    db, _ = lib

    def run():
        page, total = search_combined(db, limit=100, with_total=True,
                                      sort="date_desc", **kw)
        return total, sorted(p["id"] for p in page)

    scoped = run()
    monkeypatch.setattr(search_mod, "_SCOPE_MAX_IDS", -1)  # never scope
    unscoped = run()
    assert scoped == unscoped


def test_too_broad_a_scope_is_dropped(lib, monkeypatch):
    db, _ = lib
    monkeypatch.setattr(search_mod, "_SCOPE_MAX_IDS", 1)
    assert search_mod._compose_scope(
        db, date_from="2026-09-26", date_to="2026-09-28", camera=None,
        person_ids=[], match_source=None) is None
    assert search_mod._compose_scope(
        db, date_from="2026-09-28", date_to="2026-09-28", camera=None,
        person_ids=[], match_source=None) == [4]


def test_date_bounds_match_the_python_date_filter(lib):
    db, _ = lib
    rows = [dict(r) for r in db.conn.execute("SELECT * FROM photos")]
    for lo, hi in [("2026-09-26", "2026-09-26"), ("2026-09-26", None),
                   ("2026-09-28", "2026-09-28"), ("2025-06-01", "2026-09-27")]:
        want = {r["id"] for r in search_mod._filter_by_date(
            rows, lo, hi or search_mod._OPEN_DATE_HI)}
        b = search_mod._date_bounds(lo, hi)
        got = {r[0] for r in db.conn.execute(
            "SELECT id FROM photos WHERE date_taken >= ? AND date_taken <= ?", b)}
        assert got == want, (lo, hi)


def test_nocase_location_matches_lower(lib):
    db, _ = lib
    for name in ("marin county", "MARIN COUNTY", "Lecco", "us"):
        lower = {r[0] for r in db.conn.execute(
            "SELECT id FROM photos WHERE LOWER(admin2) = LOWER(?) "
            "OR LOWER(country) = LOWER(?)", (name, name))}
        nocase = {r[0] for r in db.conn.execute(
            "SELECT id FROM photos WHERE admin2 = ? COLLATE NOCASE "
            "OR country = ? COLLATE NOCASE", (name, name))}
        assert lower == nocase and lower, name


# ---------------------------------------------------------------------------
# Phase 2: id-first rewrites and SQL pagination
# ---------------------------------------------------------------------------

def test_people_queries_read_rows_by_id_from_the_faces_index(lib):
    db, _ = lib
    stmts = _traced(db, lambda: search_combined(db, query="Calvin and Ellie"))
    stmts += _traced(db, lambda: search_combined(db, person="Calvin"))
    plans = _plans_touching(db, stmts, "f.person_id")
    assert len(plans) >= 2
    for plan in plans:
        assert "idx_faces_person_photo" in plan, plan
        assert "SCAN p " not in plan and "SCAN photos" not in plan, plan


@pytest.mark.parametrize("call,index", [
    (lambda db: search_combined(db, query="IMG_3"), "sqlite_autoindex_photos_1"),
    (lambda db: search_combined(db, location="Varenna"), "idx_photos_place"),
])
def test_substring_likes_scan_a_narrow_index(lib, call, index):
    db, _ = lib
    stmts = _traced(db, lambda: call(db))
    plans = _plans_touching(db, stmts, " LIKE ")
    assert plans
    assert f"COVERING INDEX {index}" in plans[0], plans[0]


@pytest.fixture
def scored(lib):
    """lib plus aesthetic scores with ties, zeros, NULLs and undated rows —
    the cases where SQL and _apply_sort/_filter_aesthetic could disagree."""
    db, ids = lib
    vals = [  # pct, subj, day, subj_day, tech, impact, aes_overall
        (90.0, None, 50.0, None, 8.0, 7.0, 7.0),
        (90.0, 95.0, None, 99.0, 0.0, 8.0, 7.0),
        (0.0, None, 10.0, None, None, 3.0, 5.5),
        (40.0, 40.0, 40.0, 40.0, 5.0, 5.0, None),
        (90.0, None, 70.0, 30.0, 8.0, 7.0, 7.0),  # subject day pct wins
        (75.0, 10.0, 80.0, None, 9.0, 9.0, 6.2),
        (75.0, None, None, None, 9.0, 9.0, 6.2),  # undated row (ids[6])
        (None, None, None, None, None, None, None),
    ]
    for pid, v in zip(ids, vals):
        db.conn.execute(
            "UPDATE photos SET aes_overall_pct=?, aes_subject_overall_pct=?, "
            "aes_overall_day_pct=?, aes_subject_overall_day_pct=?, "
            "aes_technical=?, aes_impact=?, aes_overall=? WHERE id=?", (*v, pid))
    db.conn.commit()
    return db, ids


PAGED = [
    dict(min_aesthetic=50),
    dict(min_aesthetic=0),
    dict(min_aesthetic=-5),
    dict(min_technical=0.0),
    dict(min_technical=6, min_impact=6),
    dict(min_subject_aesthetic=20),
    dict(min_day_aesthetic=45),
    dict(min_day_aesthetic=0),
    dict(min_aesthetic=10, date_from="2026-09-26", date_to="2026-09-27"),
    dict(min_quality=5.0),
    dict(min_quality=5.0, min_aesthetic=80),
    dict(min_quality=5.0, date_from="2026-01-01"),
    dict(),  # browse: only with an aesthetic sort
]
SORTS = ["date_desc", "date_asc", "aesthetic_desc", "subject_aesthetic_desc",
         "quality_desc", "day_quality_desc", "relevance"]


@pytest.mark.parametrize("sort", SORTS)
@pytest.mark.parametrize("kw", PAGED, ids=lambda kw: ",".join(
    f"{k}={v}" for k, v in sorted(kw.items())) or "browse")
@pytest.mark.parametrize("offset,limit", [(0, 100), (1, 2), (3, 0)])
def test_sql_pagination_matches_the_python_path(scored, monkeypatch, kw, sort,
                                                 offset, limit):
    db, _ = scored

    def run():
        return search_combined(db, sort=sort, offset=offset, limit=limit,
                               with_total=True, **kw)

    page, total = run()
    monkeypatch.setattr(search_mod, "_sort_sql", lambda *a, **k: None)
    old_page, old_total = run()
    assert total == old_total
    assert [p["id"] for p in page] == [p["id"] for p in old_page]


def test_aesthetic_sort_reads_only_the_page(scored):
    db, _ = scored
    stmts = _traced(db, lambda: search_combined(db, sort="aesthetic_desc", limit=2))
    page = [s for s in stmts if s.startswith("SELECT * FROM photos WHERE")]
    assert len(page) == 1 and "LIMIT 2" in page[0]
    plan = _plan(db, page[0])
    assert "idx_photos_aes_overall_pct" in plan and "TEMP B-TREE" not in plan


# ---------------------------------------------------------------------------
# People-only searches: narrow rows, full rows for the page only
# ---------------------------------------------------------------------------

NARROW = [
    dict(person="Calvin"),
    dict(person="Calvin", match_source="strict"),
    dict(query="Calvin and Ellie"),
    dict(person_ids=[1, 2]),
    dict(person="Calvin", person_ids=[2]),
    dict(person="Calvin", date_from="2026-09-26", date_to="2026-09-27"),
    dict(person="Calvin", min_quality=5.0),
    dict(person="Calvin", min_aesthetic=50, min_day_aesthetic=10),
    dict(person="Calvin", style_tag="golden-hour"),
    dict(person="Nobody"),
]


@pytest.mark.parametrize("sort", SORTS)
@pytest.mark.parametrize("kw", NARROW, ids=lambda kw: ",".join(
    f"{k}={v}" for k, v in sorted(kw.items())))
@pytest.mark.parametrize("offset,limit", [(0, 100), (1, 2)])
def test_people_only_page_matches_full_rows(scored, monkeypatch, kw, sort,
                                            offset, limit):
    db, ids = scored
    db.conn.execute("UPDATE photos SET aes_style_tags = '[\"golden-hour\"]' "
                    "WHERE id IN (?, ?)", (ids[0], ids[4]))
    # A duplicate copy of a Calvin photo: hash dedupe must still apply.
    dup = db.add_photo(filepath="2026/copy/IMG_0.JPG", filename="IMG_0.JPG",
                       date_taken="2026-09-26 09:00:00", file_hash="h0",
                       camera_model="ILCE-7RM6")
    db.add_face(dup, (0, 10, 10, 0), [], person_id=1)
    db.conn.commit()

    # Recency decay (relevance sort) reads the clock; pin it so the two runs
    # compare rrf_score exactly.
    import datetime as real_dt

    class _FrozenDatetime(real_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 4, 12, 0, 0)
    monkeypatch.setattr(real_dt, "datetime", _FrozenDatetime)

    def run():
        return search_combined(db, sort=sort, offset=offset, limit=limit,
                               with_total=True, **kw)

    page, total = run()
    monkeypatch.setattr(search_mod, "_NARROW_COLUMNS", "p.*")
    full_page, full_total = run()
    assert total == full_total
    assert page == full_page  # same rows, same order, same keys and values


def test_people_only_search_reads_full_rows_for_the_page_only(scored):
    db, _ = scored
    stmts = _traced(db, lambda: search_combined(db, person="Calvin", limit=2))
    wide = [s for s in stmts if s.startswith("SELECT * FROM photos")
            or "SELECT p.*" in s]
    assert len(wide) == 1 and wide[0].count(",") == 1  # IN (id, id)


# ---------------------------------------------------------------------------
# Media filter (video-support step 2): rowid-only predicates keep stills cheap
# ---------------------------------------------------------------------------

def _names(page):
    return sorted(p["filename"] for p in page)


def test_default_search_hides_live_and_unreadable_videos(lib):
    db, _ = lib
    page = search_combined(db, date_from="2026-09-26", date_to="2026-09-27",
                           limit=100)
    names = _names(page)
    assert "CLIP.MP4" in names and "NEW.MOV" in names
    assert "IMG_0.MOV" not in names and "BAD.MP4" not in names
    assert "IMG_0.JPG" in names  # the Live clip's still is untouched


@pytest.mark.parametrize("kw", [
    dict(camera="ILCE-7RM6"), dict(keyword="soccer"), dict(location="Marin County"),
    dict(visual_tag="sunny"), dict(min_quality=0.0),
    dict(date_from="2026-09-26"),
], ids=lambda kw: ",".join(sorted(kw)))
def test_media_modes_partition_the_results(lib, kw):
    db, _ = lib
    def run(media):
        return {p["filename"] for p in search_combined(db, limit=1000,
                                                       media=media, **kw)}
    every, photos, videos = run(None), run("photo"), run("video")
    assert photos | videos == every and not photos & videos
    assert all(n.endswith(".JPG") for n in photos)
    assert videos <= {"CLIP.MP4", "NEW.MOV"}


def test_videos_only_browse_and_scope(lib):
    db, _ = lib
    page, total = search_combined(db, media="video", limit=100, with_total=True)
    assert _names(page) == ["CLIP.MP4", "NEW.MOV"] and total == 2
    assert search_mod._compose_scope(
        db, date_from=None, date_to=None, camera=None, person_ids=[],
        match_source=None, videos_only=True) is not None


def test_unknown_media_value_is_an_error(lib):
    db, _ = lib
    with pytest.raises(ValueError):
        search_combined(db, camera="ILCE-7RM6", media="movies")


def test_media_predicate_keeps_the_sql_page_on_its_index(lib):
    """The hidden-video predicate is `id NOT IN (<tiny video_meta set>)`:
    rowid-only, so the aesthetic browse stays a COVERING scan of its index and
    no still pays a join."""
    db, _ = lib
    db.conn.execute("UPDATE photos SET aes_overall_pct = 50")
    stmts = _traced(db, lambda: search_combined(db, sort="aesthetic_desc",
                                                limit=2, with_total=True))
    page = [s for s in stmts if s.startswith("SELECT * FROM photos WHERE")]
    count = [s for s in stmts if s.startswith("SELECT COUNT(*) FROM photos WHERE")]
    assert len(page) == 1 and len(count) == 1
    assert "video_meta" in page[0]
    plan = _plan(db, page[0])
    assert "idx_photos_aes_overall_pct" in plan and "TEMP B-TREE" not in plan
    assert "LIST SUBQUERY" in plan and "SCAN video_meta" in plan
    assert "COVERING INDEX idx_photos_aes_overall_pct" in _plan(db, count[0])
    photo_only = _traced(db, lambda: search_combined(
        db, sort="aesthetic_desc", limit=2, with_total=True, media="photo"))
    cplan = _plan(db, [s for s in photo_only
                       if s.startswith("SELECT COUNT(*) FROM photos WHERE")][0])
    assert "COVERING INDEX idx_photos_aes_overall_pct" in cplan
    assert "idx_photos_media_video" in cplan


def test_videos_only_scope_is_driven_from_the_partial_index(lib):
    db, _ = lib
    stmts = _traced(db, lambda: search_combined(db, media="video",
                                                location="Marin County"))
    scope = [s for s in stmts
             if s.startswith("SELECT id FROM photos WHERE media_type") and "LIMIT" in s]
    assert len(scope) == 1
    assert "idx_photos_media_video" in _plan(db, scope[0])
    stmts = _traced(db, lambda: search_combined(db, media="video",
                                                camera="ILCE-7RM6"))
    scope = [s for s in stmts if s.startswith("SELECT p.id FROM photos p")]
    assert len(scope) == 1
    plan = _plan(db, scope[0])
    assert "idx_photos_camera_date" in plan and "idx_photos_media_video" in plan


# ---------------------------------------------------------------------------
# Video quality sorts (video-support step 7): the shake/noise index + scope
# ---------------------------------------------------------------------------

@pytest.fixture
def quality_lib(lib):
    """`lib` plus shake/noise on CLIP.MP4 (measured) — NEW.MOV stays
    unmeasured (NULL), which is what proves "NULLs last"."""
    db, _ = lib
    clip = db.conn.execute(
        "SELECT id FROM photos WHERE filename = 'CLIP.MP4'").fetchone()[0]
    db.conn.execute("UPDATE video_meta SET shake = 1.2, noise = 4.0 "
                    "WHERE photo_id = ?", (clip,))
    db.conn.commit()
    return db, clip


@pytest.mark.parametrize("sort,col,index", [
    ("steadiest", "shake", "idx_video_meta_shake"),
    ("least_noisy", "noise", "idx_video_meta_noise"),
])
def test_video_quality_sort_uses_the_partial_index(quality_lib, sort, col, index):
    db, _ = quality_lib
    stmts = _traced(db, lambda: search_combined(db, sort=sort))
    plans = _plans_touching(db, stmts, f"FROM video_meta WHERE {col}")
    assert plans
    assert index in plans[0]
    assert "SCAN video_meta" not in plans[0]


def test_steadiest_sort_implies_media_video(quality_lib):
    db, clip = quality_lib
    stmts = _traced(db, lambda: search_combined(db, sort="steadiest"))
    scope = [s for s in stmts
             if s.startswith("SELECT * FROM photos WHERE media_type IS NOT NULL")]
    assert len(scope) == 1
    assert "idx_photos_media_video" in _plan(db, scope[0])


VIDEO_QUALITY_COMBOS = [
    dict(sort="steadiest"),
    dict(sort="least_noisy"),
    dict(sort="steadiest", camera="ILCE-7RM6"),
    dict(sort="least_noisy", date_from="2026-09-26", date_to="2026-09-27"),
    dict(sort="steadiest", media="photo"),  # overridden to video regardless
    dict(sort="steadiest", person="Calvin"),
]


@pytest.mark.parametrize("kw", VIDEO_QUALITY_COMBOS, ids=lambda kw: ",".join(sorted(kw)))
def test_video_quality_scoped_results_match_unscoped(quality_lib, monkeypatch, kw):
    db, _ = quality_lib

    def run():
        page, total = search_combined(db, limit=100, with_total=True,
                                      **{"sort": "date_desc", **kw})
        return total, [p["id"] for p in page]  # order matters here

    scoped = run()
    monkeypatch.setattr(search_mod, "_SCOPE_MAX_IDS", -1)  # never scope
    assert scoped == run()


# ---------------------------------------------------------------------------
# Video main-subject share (video-support step 6): composed scope + plans
# ---------------------------------------------------------------------------

@pytest.fixture
def subj_lib(lib):
    db, ids = lib
    clip = db.conn.execute(
        "SELECT id FROM photos WHERE filename = 'CLIP.MP4'").fetchone()[0]
    calvin = db.get_person_by_name("Calvin")["id"]
    fid = db.add_face(clip, (0, 10, 10, 0), [], person_id=calvin)
    db.conn.execute("UPDATE faces SET match_source = 'strict' WHERE id = ?", (fid,))
    for ix, label, cat, face, share in ((0, "person", "person", fid, 0.6),
                                        (1, "dog", "animal", None, 0.3)):
        db.conn.execute(
            "INSERT INTO video_subjects (photo_id, subject_ix, label, category, "
            "face_id, focus_s, share) VALUES (?,?,?,?,?,?,?)",
            (clip, ix, label, cat, face, share * 42.0, share))
    db.conn.commit()
    return db, clip


SUBJECT_COMBOS = [
    dict(person="Calvin", min_focus_share=0.5),
    dict(person="Calvin", min_focus_share=0.5, date_from="2026-09-26",
         date_to="2026-09-26"),
    dict(person="Calvin", min_focus_seconds=10, camera="ILCE-7RM6"),
    dict(person="Calvin", match_source="strict", sort="focus_desc",
         location="Marin County"),
    dict(subject="the dog"),
    dict(subject="dog", date_from="2026-09-26", date_to="2026-09-27"),
    dict(subject="dog", min_subject_share=0.2, camera="ILCE-7RM6"),
    dict(subject="dog", person="Calvin", min_focus_share=0.1),
    dict(subject="dog", keyword="soccer", sort="subject_desc"),
    dict(subject="cat", camera="ILCE-7RM6"),
]


@pytest.mark.parametrize("kw", SUBJECT_COMBOS, ids=lambda kw: ",".join(sorted(kw)))
def test_subject_share_scoped_results_match_unscoped(subj_lib, monkeypatch, kw):
    db, clip = subj_lib

    def run():
        page, total = search_combined(db, limit=100, with_total=True,
                                      **{"sort": "date_desc", **kw})
        return total, sorted(p["id"] for p in page)

    scoped = run()
    monkeypatch.setattr(search_mod, "_SCOPE_MAX_IDS", -1)  # never scope
    assert scoped == run()
    if kw.get("subject") != "cat":
        assert scoped[1] == [clip]


def test_person_share_is_driven_from_the_faces_index(subj_lib):
    db, _ = subj_lib
    stmts = _traced(db, lambda: search_combined(
        db, person="Calvin", min_focus_share=0.5, date_from="2026-09-26"))
    plans = _plans_touching(db, stmts, "JOIN video_subjects")
    assert plans
    for plan in plans:
        assert "SCAN f " not in plan + " " and "SCAN faces" not in plan, plan
        assert "idx_video_subjects_face" in plan and "idx_faces_person" in plan, plan


def test_subject_share_and_its_scope_use_the_label_index(subj_lib):
    db, _ = subj_lib
    stmts = _traced(db, lambda: search_combined(
        db, subject="dog", camera="ILCE-7RM6", date_from="2026-09-26"))
    label_stmts = [s for s in stmts if "video_subjects" in s]
    assert len(label_stmts) >= 2  # the composed scope and the share query
    for s in label_stmts:
        assert "idx_video_subjects_label" in _plan(db, s), s
    scope = [s for s in stmts if s.startswith("SELECT p.id FROM photos p")]
    assert len(scope) == 1 and "idx_photos_camera_date" in _plan(db, scope[0])
