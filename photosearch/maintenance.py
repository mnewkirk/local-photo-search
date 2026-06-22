"""Maintenance sweep + data-integrity validation/repair (M25).

A single idempotent, dependency-ordered backfill that runs the lightweight
CPU-side derived-data passes that have no worker pass and otherwise fall
through the cracks when new photos arrive (reverse-geocode place_name,
structured location columns, inferred GPS, dominant colors, library-wide
stacking, face->person matching, one-person-per-photo enforcement).

Heavy GPU passes (CLIP / faces detection / quality / describe / category /
keywords / verify) stay on the worker fleet — the sweep never runs those.

Every stage is gated by a "what's still missing" SQL predicate, so a sweep
over a fully-enriched library is nearly free (counts come back zero, nothing
runs). The whole thing is cancellable + streams progress via the
``on_progress`` / ``should_abort`` callback pair established by
``stacking.py`` (the reference shape for instrumenting long-running jobs).

This module orchestrates the *existing* module functions rather than
reimplementing them, so each pass stays the single source of truth.
"""

import json
import logging
import os
import re
import time
from typing import Callable, Optional

logger = logging.getLogger("photosearch.maintenance")

# Stages that run by default in dependency order. ``stacking`` and ``colors``
# are toggleable; ``recluster`` is opt-in (it renumbers every unknown cluster
# and clears ignored_clusters, so a nightly auto-run would wipe "ignore"
# decisions — gate it to a slower/manual cadence per the M25 plan).
SWEEP_STAGE_ORDER = (
    "geocode",
    "normalize",
    "infer",
    "normalize_inferred",
    "colors",
    "stacking",
    "match_faces",
    "resolve_dups",
    "recluster",
)

# JSON array columns the validator checks for malformed values.
_JSON_ARRAY_COLUMNS = ("tags", "categories", "visual_tags", "keywords", "dominant_colors")

# A date_taken is considered well-formed when its first 10 chars are a
# YYYY-MM-DD date (covers both "YYYY-MM-DD HH:MM:SS" and the ISO "T" form,
# plus fractional-second / timezone tails). The corruption seen in M24a was
# stray control bytes (e.g. '\x18u') which fail this prefix check. Using a
# prefix check (not a full-string match) avoids false-positiving valid rows
# that carry a non-space separator.
_VALID_DATE_PREFIX_GLOB = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
_FOLDER_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_date_taken(s: Optional[str]) -> bool:
    """True if ``s`` looks like a real date_taken (YYYY-MM-DD prefix)."""
    if not s or len(s) < 10:
        return False
    head = s[:10]
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", head))


def _folder_date_from_path(filepath: str) -> Optional[str]:
    """Pull a YYYY-MM-DD date out of a stored filepath (the dated-folder
    convention /photos/YYYY/YYYY-MM-DD_<suffix>/). Returns
    'YYYY-MM-DD 00:00:00' or None."""
    if not filepath:
        return None
    m = _FOLDER_DATE_RE.search(filepath)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)} 00:00:00"


def _make_emit(on_progress: Optional[Callable[[dict], None]]):
    def _emit(event: dict):
        if on_progress:
            try:
                on_progress(event)
            except Exception:  # never let a progress sink kill the job
                logger.debug("on_progress sink raised", exc_info=True)
    return _emit


def _make_abort(should_abort: Optional[Callable[[], bool]]):
    def _check():
        if should_abort and should_abort():
            raise InterruptedError("maintenance cancelled")
    return _check


# ---------------------------------------------------------------------------
# Individual sweep stages — each returns a dict {would, applied, status, ...}
# ---------------------------------------------------------------------------

def _stage_geocode(db, apply, emit, check_abort, batch_size=2000):
    """Reverse-geocode place_name for GPS-bearing rows missing it."""
    rows = db.conn.execute(
        "SELECT id, gps_lat, gps_lon FROM photos "
        "WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL "
        "AND (place_name IS NULL OR place_name = '')"
    ).fetchall()
    would = len(rows)
    if would == 0 or not apply:
        return {"stage": "geocode", "would": would, "applied": 0,
                "status": "skipped" if would == 0 else "preview"}
    from .geocode import reverse_geocode_batch
    applied = 0
    for start in range(0, would, batch_size):
        check_abort()
        batch = rows[start:start + batch_size]
        coords = [(r["gps_lat"], r["gps_lon"]) for r in batch]
        places = reverse_geocode_batch(coords)
        for r, place in zip(batch, places):
            if place:
                db.conn.execute(
                    "UPDATE photos SET place_name = ? WHERE id = ? AND "
                    "(place_name IS NULL OR place_name = '')",
                    (place, r["id"]),
                )
                applied += 1
        db.conn.commit()
        emit({"phase": "sweep", "stage": "geocode", "status": "running",
              "done": min(start + batch_size, would), "total": would})
    return {"stage": "geocode", "would": would, "applied": applied, "status": "done"}


def _stage_normalize(db, apply, emit, check_abort, batch_size=2000, stage="normalize"):
    """Backfill structured country/admin1/admin2/locality columns (v19)."""
    rows = db.conn.execute(
        "SELECT id, gps_lat, gps_lon FROM photos "
        "WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL "
        "AND (country IS NULL OR admin1 IS NULL)"
    ).fetchall()
    would = len(rows)
    if would == 0 or not apply:
        return {"stage": stage, "would": would, "applied": 0,
                "status": "skipped" if would == 0 else "preview"}
    import reverse_geocoder as rg
    applied = 0
    for start in range(0, would, batch_size):
        check_abort()
        batch = rows[start:start + batch_size]
        coords = [(r["gps_lat"], r["gps_lon"]) for r in batch]
        try:
            results = rg.search(coords)
        except Exception as err:
            logger.warning("reverse_geocoder error: %s", err)
            continue
        for photo_row, rg_row in zip(batch, results):
            db.conn.execute(
                "UPDATE photos SET country=?, admin1=?, admin2=?, locality=? WHERE id=?",
                (rg_row.get("cc"), rg_row.get("admin1"),
                 rg_row.get("admin2"), rg_row.get("name"), photo_row["id"]),
            )
            applied += 1
        db.conn.commit()
        emit({"phase": "sweep", "stage": stage, "status": "running",
              "done": min(start + batch_size, would), "total": would})
    return {"stage": stage, "would": would, "applied": applied, "status": "done"}


def _stage_infer(db, apply, emit, check_abort,
                 window_minutes=30, max_drift_km=25.0, min_confidence=0.0):
    """Infer GPS for no-GPS rows from temporal neighbors (M19)."""
    from .infer_location import infer_locations
    result = infer_locations(
        db,
        window_minutes=window_minutes,
        max_drift_km=max_drift_km,
        min_confidence=min_confidence,
        cascade=True,
    )
    candidates = result["candidates"]
    would = len(candidates)
    if would == 0 or not apply:
        return {"stage": "infer", "would": would, "applied": 0,
                "status": "skipped" if would == 0 else "preview"}
    from .geocode import reverse_geocode_batch
    coords = [(c["lat"], c["lon"]) for c in candidates]
    places = reverse_geocode_batch(coords)
    applied = 0
    cur = db.conn.cursor()
    for c, place in zip(candidates, places):
        check_abort()
        cur.execute(
            "UPDATE photos SET gps_lat=?, gps_lon=?, "
            "place_name=COALESCE(place_name, ?), location_source='inferred', "
            "location_confidence=? WHERE id=? AND gps_lat IS NULL",
            (c["lat"], c["lon"], place, c["confidence"], c["photo_id"]),
        )
        applied += cur.rowcount
    db.conn.commit()
    emit({"phase": "sweep", "stage": "infer", "status": "running",
          "done": would, "total": would})
    return {"stage": "infer", "would": would, "applied": applied, "status": "done"}


def _stage_colors(db, apply, emit, check_abort, batch_size=100):
    """Extract dominant colors for rows missing them."""
    rows = db.conn.execute(
        "SELECT id, filepath FROM photos WHERE dominant_colors IS NULL"
    ).fetchall()
    would = len(rows)
    if would == 0 or not apply:
        return {"stage": "colors", "would": would, "applied": 0,
                "status": "skipped" if would == 0 else "preview"}
    from .colors import colors_to_json, extract_dominant_colors
    applied = 0
    for i, r in enumerate(rows, 1):
        if i % 50 == 0:
            check_abort()
        abs_path = db.resolve_filepath(r["filepath"]) or r["filepath"]
        try:
            colors = extract_dominant_colors(abs_path)
            if colors:
                db.conn.execute(
                    "UPDATE photos SET dominant_colors = ? WHERE id = ?",
                    (colors_to_json(colors), r["id"]),
                )
                applied += 1
        except Exception as err:
            logger.debug("color extraction failed for %s: %s", abs_path, err)
        if i % batch_size == 0:
            db.conn.commit()
            emit({"phase": "sweep", "stage": "colors", "status": "running",
                  "done": i, "total": would})
    db.conn.commit()
    return {"stage": "colors", "would": would, "applied": applied, "status": "done"}


def _abort_flag(check_abort) -> bool:
    """Adapt a raising check into a boolean for should_abort consumers."""
    try:
        check_abort()
        return False
    except InterruptedError:
        return True


def _stage_stacking(db, apply, emit, check_abort):
    """Library-wide burst/bracket stack detection (idempotent re-detect)."""
    from .stacking import run_stacking
    # "would" here is the count of timestamped photos eligible for detection;
    # stacking is a full re-detect rather than a missing-only backfill.
    would = db.conn.execute(
        "SELECT COUNT(*) FROM photos WHERE date_taken IS NOT NULL"
    ).fetchone()[0]
    if not apply:
        return {"stage": "stacking", "would": would, "applied": 0, "status": "preview"}

    def _on_prog(ev):
        ev = dict(ev)
        ev["phase"] = "sweep"
        ev["stage"] = "stacking"
        ev["status"] = "running"
        emit(ev)

    stacks = run_stacking(
        db, dry_run=False,
        on_progress=_on_prog, should_abort=lambda: _abort_flag(check_abort),
    )
    return {"stage": "stacking", "would": would,
            "applied": len(stacks), "status": "done",
            "photos_stacked": sum(len(s) for s in stacks)}


def _stage_match_faces(db, apply, emit, check_abort):
    """Match unassigned faces to known persons (strict + temporal)."""
    would = db.conn.execute(
        "SELECT COUNT(*) FROM faces WHERE person_id IS NULL"
    ).fetchone()[0]
    if would == 0:
        return {"stage": "match_faces", "would": 0, "applied": 0, "status": "skipped"}
    # No registered persons -> nothing to match against.
    has_refs = db.conn.execute("SELECT 1 FROM face_references LIMIT 1").fetchone()
    if not has_refs:
        return {"stage": "match_faces", "would": would, "applied": 0,
                "status": "skipped", "message": "no person references"}
    if not apply:
        return {"stage": "match_faces", "would": would, "applied": 0, "status": "preview"}
    try:
        from .faces import (
            match_faces_to_persons, match_faces_temporal, check_available,
        )
        check_available()
    except Exception as err:
        return {"stage": "match_faces", "would": would, "applied": 0,
                "status": "skipped", "message": f"insightface unavailable: {err}"}
    matched = match_faces_to_persons(db)
    emit({"phase": "sweep", "stage": "match_faces", "status": "running",
          "done": matched, "total": would})
    matched += match_faces_temporal(db)
    return {"stage": "match_faces", "would": would, "applied": matched, "status": "done"}


def _stage_resolve_dups(db, apply, emit, check_abort):
    """Enforce one-person-per-photo: keep one face per (photo, person)."""
    c = db.conn
    dups = c.execute(
        "SELECT photo_id, person_id FROM faces WHERE person_id IS NOT NULL "
        "GROUP BY photo_id, person_id HAVING COUNT(*) >= 2"
    ).fetchall()
    PRIORITY = {"manual": 0, "strict": 1}  # temporal/other/NULL -> 2
    to_unmatch = []
    for d in dups:
        faces = c.execute(
            "SELECT id, match_source AS src, det_score, "
            "(bbox_bottom - bbox_top) * (bbox_right - bbox_left) AS area "
            "FROM faces WHERE photo_id = ? AND person_id = ?",
            (d["photo_id"], d["person_id"]),
        ).fetchall()
        ranked = sorted(faces, key=lambda f: (
            PRIORITY.get(f["src"], 2), -(f["det_score"] or 0.0), -(f["area"] or 0)))
        for f in ranked[1:]:
            to_unmatch.append(f["id"])
    would = len(to_unmatch)
    if would == 0 or not apply:
        return {"stage": "resolve_dups", "would": would, "applied": 0,
                "status": "skipped" if would == 0 else "preview",
                "groups": len(dups)}
    # Reversible snapshot (same on-demand table the CLI dedup commands use).
    c.execute("CREATE TABLE IF NOT EXISTS face_dedupe_undo ("
              "face_id INTEGER PRIMARY KEY, person_id INTEGER, match_source TEXT, "
              "unmatched_at TEXT DEFAULT (datetime('now')))")
    for fid in to_unmatch:
        row = c.execute("SELECT person_id, match_source FROM faces WHERE id = ?",
                        (fid,)).fetchone()
        if row and row["person_id"] is not None:
            c.execute("INSERT OR REPLACE INTO face_dedupe_undo"
                      "(face_id, person_id, match_source) VALUES (?, ?, ?)",
                      (fid, row["person_id"], row["match_source"]))
        c.execute("UPDATE faces SET person_id = NULL, "
                  "match_source = 'dedupe_unmatched' WHERE id = ?", (fid,))
    db.conn.commit()
    return {"stage": "resolve_dups", "would": would, "applied": would,
            "status": "done", "groups": len(dups)}


def _stage_recluster(db, apply, emit, check_abort):
    """Global DBSCAN over unknown faces (opt-in: wipes ignored_clusters)."""
    would = db.conn.execute(
        "SELECT COUNT(*) FROM faces WHERE person_id IS NULL"
    ).fetchone()[0]
    if would == 0:
        return {"stage": "recluster", "would": 0, "applied": 0, "status": "skipped"}
    if not apply:
        return {"stage": "recluster", "would": would, "applied": 0, "status": "preview"}
    try:
        from .faces import recluster_unknown_faces
    except Exception as err:
        return {"stage": "recluster", "would": would, "applied": 0,
                "status": "skipped", "message": f"unavailable: {err}"}
    summary = recluster_unknown_faces(db, dry_run=False)
    return {"stage": "recluster", "would": would,
            "applied": summary.get("cluster_count", 0), "status": "done",
            "face_count": summary.get("face_count", 0)}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_maintenance_sweep(
    db,
    *,
    apply: bool = False,
    do_colors: bool = True,
    do_stacking: bool = True,
    do_match: bool = True,
    do_recluster: bool = False,
    window_minutes: int = 30,
    max_drift_km: float = 25.0,
    min_confidence: float = 0.0,
    on_progress: Optional[Callable[[dict], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> dict:
    """Run the dependency-ordered backfill sweep over only-the-missing rows.

    Args:
        db: Open PhotoDB.
        apply: When False (default) every stage reports how many rows it
            *would* touch and writes nothing. When True, stages run.
        do_colors / do_stacking: toggle those (heavier) stages.
        do_recluster: opt-in — clears ignored_clusters, so off by default.
        window_minutes / max_drift_km / min_confidence: infer-locations tuning.
        on_progress: callback(dict) invoked per stage (and inside long stages)
            with a {"phase": "sweep", "stage": ..., "status": ...} event.
        should_abort: callable -> bool; checked between + inside stages.
            When it returns True, the sweep raises InterruptedError.

    Returns a dict {"stages": [per-stage dicts...], "apply": bool}.
    """
    emit = _make_emit(on_progress)
    check_abort = _make_abort(should_abort)

    plan = [
        ("geocode", lambda: _stage_geocode(db, apply, emit, check_abort)),
        ("normalize", lambda: _stage_normalize(db, apply, emit, check_abort)),
        ("infer", lambda: _stage_infer(
            db, apply, emit, check_abort,
            window_minutes=window_minutes, max_drift_km=max_drift_km,
            min_confidence=min_confidence)),
        # Re-normalize structured cols for the rows infer just gave GPS to.
        ("normalize_inferred", lambda: _stage_normalize(
            db, apply, emit, check_abort, stage="normalize_inferred")),
    ]
    if do_colors:
        plan.append(("colors", lambda: _stage_colors(db, apply, emit, check_abort)))
    if do_stacking:
        plan.append(("stacking", lambda: _stage_stacking(db, apply, emit, check_abort)))
    # match_faces is the heaviest CPU stage (ArcFace distance over every
    # unmatched face); gate it so the interactive/live sweep can skip it and
    # leave it to an off-hours cron or the worker fleet.
    if do_match:
        plan.append(("match_faces", lambda: _stage_match_faces(db, apply, emit, check_abort)))
    plan.append(("resolve_dups", lambda: _stage_resolve_dups(db, apply, emit, check_abort)))
    if do_recluster:
        plan.append(("recluster", lambda: _stage_recluster(db, apply, emit, check_abort)))

    stages = []
    for name, fn in plan:
        check_abort()
        emit({"phase": "sweep", "stage": name, "status": "scanning"})
        t0 = time.monotonic()
        result = fn()
        result["seconds"] = round(time.monotonic() - t0, 2)
        stages.append(result)
        emit({"phase": "sweep", "stage": name, "status": result.get("status", "done"),
              "would": result.get("would", 0), "applied": result.get("applied", 0),
              "message": result.get("message")})
        logger.info("sweep stage %s: %s", name, result)

    return {"apply": apply, "stages": stages}


# ---------------------------------------------------------------------------
# Data-integrity validation + repair
# ---------------------------------------------------------------------------

def validate_data(db, sample: int = 5) -> dict:
    """Scan for invalid rows. Read-only. Returns a per-category report.

    Categories that this module can repair (date_taken / GPS / JSON) carry a
    ``count`` + ``sample``; categories that already have a dedicated command
    (orphaned vec rows, garbage tag sets) are reported with a ``hint``
    pointing at it rather than duplicating its logic.
    """
    c = db.conn
    report = {}

    # 1. Corrupt date_taken (bad YYYY-MM-DD prefix — e.g. control bytes).
    bad_dates = c.execute(
        "SELECT id, filepath, date_taken FROM photos "
        "WHERE date_taken IS NOT NULL "
        f"AND substr(date_taken, 1, 10) NOT GLOB '{_VALID_DATE_PREFIX_GLOB}'"
    ).fetchall()
    report["corrupt_date_taken"] = {
        "count": len(bad_dates),
        "sample": [
            {"id": r["id"], "date_taken": repr(r["date_taken"])}
            for r in bad_dates[:sample]
        ],
    }

    # 2. Out-of-range / zero GPS.
    bad_gps = c.execute(
        "SELECT id, gps_lat, gps_lon FROM photos "
        "WHERE (gps_lat IS NOT NULL OR gps_lon IS NOT NULL) AND ("
        "  (gps_lat = 0 AND gps_lon = 0) "
        "  OR gps_lat < -90 OR gps_lat > 90 "
        "  OR gps_lon < -180 OR gps_lon > 180)"
    ).fetchall()
    report["bad_gps"] = {
        "count": len(bad_gps),
        "sample": [
            {"id": r["id"], "gps_lat": r["gps_lat"], "gps_lon": r["gps_lon"]}
            for r in bad_gps[:sample]
        ],
    }

    # 3. Malformed JSON array columns.
    json_report = {}
    for col in _JSON_ARRAY_COLUMNS:
        rows = c.execute(
            f"SELECT id, {col} AS val FROM photos "
            f"WHERE {col} IS NOT NULL AND {col} != ''"
        ).fetchall()
        bad = []
        for r in rows:
            try:
                parsed = json.loads(r["val"])
                if not isinstance(parsed, list):
                    bad.append(r["id"])
            except (ValueError, TypeError):
                bad.append(r["id"])
        json_report[col] = {"count": len(bad), "sample": bad[:sample]}
    report["malformed_json"] = json_report

    # 4. Orphaned vec rows — report only, point at cleanup-orphans.
    orphans = db.cleanup_vec_orphans(dry_run=True)
    report["orphaned_vec_rows"] = {
        "clip_embeddings": orphans["clip_orphans"],
        "face_encodings": orphans["face_orphans"],
        "hint": "run `photosearch cleanup-orphans` to delete these",
    }

    # 5. Orphan faces (photo deleted) — report only.
    orphan_faces = c.execute(
        "SELECT COUNT(*) FROM faces WHERE photo_id NOT IN (SELECT id FROM photos)"
    ).fetchone()[0]
    report["orphan_faces"] = {
        "count": orphan_faces,
        "hint": "run `photosearch cleanup-orphan-faces --apply` to delete these",
    }

    # 6. Garbage tag sets (regurgitated vocabulary) — report only. Heuristic:
    #    >=16 tags in the legacy `tags` column. Defer the real fix.
    garbage = 0
    for r in c.execute("SELECT tags FROM photos WHERE tags IS NOT NULL AND tags != '[]'"):
        try:
            t = json.loads(r["tags"])
            if isinstance(t, list) and len(t) >= 16:
                garbage += 1
        except (ValueError, TypeError):
            pass
    report["garbage_tag_sets"] = {
        "count": garbage,
        "hint": "run `photosearch clean-garbage-tags` to clear these",
    }

    return report


def repair_data(
    db,
    *,
    apply: bool = False,
    on_progress: Optional[Callable[[dict], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> dict:
    """Repair the auto-fixable categories from :func:`validate_data`.

    Repairs (only when ``apply=True``):
      - corrupt date_taken -> cascade EXIF -> folder-name date -> file mtime -> NULL
      - out-of-range/zero GPS -> NULL gps + dependent place/structured columns
      - malformed JSON array columns -> NULL

    Read-only otherwise. Returns a per-category summary of repaired counts.
    Orphaned-vec / garbage-tag categories are left to their dedicated
    commands (reported by ``validate_data``, never auto-fixed here).
    """
    emit = _make_emit(on_progress)
    check_abort = _make_abort(should_abort)
    c = db.conn
    summary = {"apply": apply}

    # --- 1. Corrupt date_taken --------------------------------------------
    bad_dates = c.execute(
        "SELECT id, filepath, date_created, date_taken FROM photos "
        "WHERE date_taken IS NOT NULL "
        f"AND substr(date_taken, 1, 10) NOT GLOB '{_VALID_DATE_PREFIX_GLOB}'"
    ).fetchall()
    date_summary = {"count": len(bad_dates), "from_exif": 0, "from_folder": 0,
                    "from_mtime": 0, "nulled": 0}
    if bad_dates and apply:
        try:
            from .exif import extract_exif
        except Exception:
            extract_exif = None
        for i, r in enumerate(bad_dates, 1):
            if i % 50 == 0:
                check_abort()
            abs_path = db.resolve_filepath(r["filepath"]) or r["filepath"]
            new_date = None
            source = None
            # (a) EXIF
            if extract_exif is not None:
                try:
                    cand = extract_exif(abs_path).get("date_taken")
                    if _valid_date_taken(cand):
                        new_date, source = cand, "from_exif"
                except Exception:
                    pass
            # (b) folder-name date
            if new_date is None:
                cand = _folder_date_from_path(r["filepath"])
                if cand:
                    new_date, source = cand, "from_folder"
            # (c) existing valid date_created, else file mtime
            if new_date is None:
                if _valid_date_taken(r["date_created"]):
                    new_date, source = r["date_created"], "from_mtime"
                else:
                    try:
                        from datetime import datetime
                        ts = os.path.getmtime(abs_path)
                        new_date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                        source = "from_mtime"
                    except OSError:
                        pass
            # (d) give up -> NULL so it sorts to the tail (no control bytes)
            if new_date is None:
                c.execute("UPDATE photos SET date_taken = NULL WHERE id = ?", (r["id"],))
                date_summary["nulled"] += 1
            else:
                c.execute("UPDATE photos SET date_taken = ? WHERE id = ?",
                          (new_date, r["id"]))
                date_summary[source] += 1
        db.conn.commit()
        emit({"phase": "repair", "stage": "date_taken", "done": len(bad_dates)})
    summary["corrupt_date_taken"] = date_summary

    # --- 2. Out-of-range / zero GPS ---------------------------------------
    bad_gps = c.execute(
        "SELECT id FROM photos "
        "WHERE (gps_lat IS NOT NULL OR gps_lon IS NOT NULL) AND ("
        "  (gps_lat = 0 AND gps_lon = 0) "
        "  OR gps_lat < -90 OR gps_lat > 90 "
        "  OR gps_lon < -180 OR gps_lon > 180)"
    ).fetchall()
    gps_summary = {"count": len(bad_gps), "nulled": 0}
    if bad_gps and apply:
        ids = [r["id"] for r in bad_gps]
        for fid in ids:
            c.execute(
                "UPDATE photos SET gps_lat=NULL, gps_lon=NULL, place_name=NULL, "
                "location_source=NULL, location_confidence=NULL, "
                "country=NULL, admin1=NULL, admin2=NULL, locality=NULL "
                "WHERE id=?",
                (fid,),
            )
        gps_summary["nulled"] = len(ids)
        db.conn.commit()
        emit({"phase": "repair", "stage": "gps", "done": len(ids)})
    summary["bad_gps"] = gps_summary

    # --- 3. Malformed JSON array columns ----------------------------------
    json_summary = {}
    for col in _JSON_ARRAY_COLUMNS:
        rows = c.execute(
            f"SELECT id, {col} AS val FROM photos "
            f"WHERE {col} IS NOT NULL AND {col} != ''"
        ).fetchall()
        bad_ids = []
        for r in rows:
            try:
                parsed = json.loads(r["val"])
                if not isinstance(parsed, list):
                    bad_ids.append(r["id"])
            except (ValueError, TypeError):
                bad_ids.append(r["id"])
        if bad_ids and apply:
            for fid in bad_ids:
                c.execute(f"UPDATE photos SET {col} = NULL WHERE id = ?", (fid,))
            db.conn.commit()
        json_summary[col] = {"count": len(bad_ids), "nulled": len(bad_ids) if apply else 0}
    summary["malformed_json"] = json_summary

    return summary
