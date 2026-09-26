"""Photo stacking — detect and group near-identical burst/bracket shots.

A "stack" is a group of photos taken within a short time window that are
also visually near-identical (very high CLIP similarity).  The best photo
(by aesthetic score) is promoted to "top" and shown in the UI; the rest
are collapsed behind a count badge.

Detection requires BOTH conditions to be met:
  1. Temporal proximity: date_taken within ``time_window_sec`` seconds.
  2. Visual similarity: CLIP cosine distance < ``clip_threshold``.

This is deliberately much tighter than review clustering (which uses
distance ~0.20 for "visually similar").  Stacking at 0.05 targets
"obviously same burst frame, maybe slightly different exposure or focus."
"""

import logging
import struct
import time
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np

from .db import PhotoDB, CLIP_DIMENSIONS, _deserialize_float_list

# Minimum seconds between throttled progress callbacks. Phase transitions
# (start/end of a phase) bypass the throttle so the UI always sees them.
_PROGRESS_INTERVAL = 0.25

logger = logging.getLogger("photosearch.stacking")


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def _parse_date(date_str: str | None) -> datetime | None:
    """Parse a date_taken string into a datetime, tolerating common formats."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    # Try with fractional seconds
    try:
        return datetime.strptime(date_str[:19], "%Y-%m-%dT%H:%M:%S")
    except (ValueError, IndexError):
        return None


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two unit vectors: 1 - dot(a, b)."""
    return float(1.0 - np.dot(a, b))


def detect_stacks(
    db: PhotoDB,
    time_window_sec: float = 5.0,
    clip_threshold: float = 0.05,
    directory: str | None = None,
    max_stack_span_sec: float = 10.0,
    photo_ids: list[int] | None = None,
    on_progress: Optional[Callable[[dict], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> list[list[int]]:
    """Detect photo stacks based on temporal proximity + CLIP similarity.

    Algorithm:
      1. Fetch all photos with date_taken, sorted chronologically.
      2. Slide a window: for each photo, compare against subsequent photos
         within ±time_window_sec.
      3. For candidate pairs within the time window, check CLIP cosine
         distance < clip_threshold.
      4. Build connected components via union-find — if A~B and B~C then
         {A, B, C} form one stack.
      5. Enforce max_stack_span_sec: every member of a stack must be within
         this many seconds of the earliest member.  If a component spans
         wider, it is trimmed from the tail until it fits.
      6. For each stack, pick the photo with highest aesthetic_score as top.

    Args:
        db: Open PhotoDB instance.
        time_window_sec: Max seconds between *consecutive* shots to consider.
        clip_threshold: Max CLIP cosine distance (1 - similarity).
        directory: If set, only consider photos whose resolved filepath
            starts with this directory.
        max_stack_span_sec: Hard cap on total span of a stack — all members
            must be within this many seconds of the earliest member.
        photo_ids: If set, only consider these specific photo IDs
            (e.g. from a collection). Mutually exclusive with directory.

    Returns:
        List of stacks, where each stack is a list of photo IDs
        ordered by aesthetic score descending (best first).
    """
    def _emit(event: dict):
        if on_progress:
            on_progress(event)

    def _check_abort():
        if should_abort and should_abort():
            raise InterruptedError("stacking cancelled")

    _emit({"phase": "scan", "message": "Loading photos"})
    _check_abort()

    # 1. Fetch photos with timestamps
    rows = db.conn.execute(
        "SELECT id, filepath, date_taken, aesthetic_score FROM photos "
        "WHERE date_taken IS NOT NULL "
        "ORDER BY date_taken"
    ).fetchall()

    photo_id_set = set(photo_ids) if photo_ids else None

    photos = []
    for r in rows:
        dt = _parse_date(r["date_taken"])
        if dt is None:
            continue
        # Optional photo_ids filter (collection mode)
        if photo_id_set is not None:
            if r["id"] not in photo_id_set:
                continue
        # Optional directory filter
        elif directory:
            abs_path = db.resolve_filepath(r["filepath"])
            if not abs_path or not abs_path.startswith(directory.rstrip("/") + "/"):
                continue
        photos.append({
            "id": r["id"],
            "date_taken": dt,
            "aesthetic_score": r["aesthetic_score"] or 0.0,
        })

    _emit({"phase": "scan", "scanned": len(photos)})

    if len(photos) < 2:
        return []

    logger.info("Stacking: %d photos with timestamps, window=%gs, clip_threshold=%.3f",
                len(photos), time_window_sec, clip_threshold)

    # 2. Load CLIP embeddings in bulk
    photo_ids = [p["id"] for p in photos]
    embeddings = _load_embeddings_bulk(
        db, photo_ids, on_progress=on_progress, should_abort=should_abort,
    )
    logger.info("Loaded %d CLIP embeddings", len(embeddings))

    # 3. Sliding window: find candidate pairs
    window = timedelta(seconds=time_window_sec)

    # Union-find for building connected components
    parent = {p["id"]: p["id"] for p in photos}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    _emit({"phase": "pairs", "processed": 0, "total": len(photos), "pairs_found": 0})
    last_emit = time.monotonic()

    pair_count = 0
    for i, p in enumerate(photos):
        # Throttled progress + abort check (every ~1000 iterations keeps the
        # per-photo overhead negligible even for 500k-photo libraries).
        if i % 1000 == 0:
            _check_abort()
            now = time.monotonic()
            if now - last_emit >= _PROGRESS_INTERVAL:
                _emit({
                    "phase": "pairs",
                    "processed": i,
                    "total": len(photos),
                    "pairs_found": pair_count,
                })
                last_emit = now
        if p["id"] not in embeddings:
            continue
        emb_i = embeddings[p["id"]]
        # Look ahead at subsequent photos within the time window
        for j in range(i + 1, len(photos)):
            q = photos[j]
            time_diff = q["date_taken"] - p["date_taken"]
            if time_diff > window:
                break  # sorted by date, so all further are outside window
            if q["id"] not in embeddings:
                continue
            dist = _cosine_distance(emb_i, embeddings[q["id"]])
            if dist < clip_threshold:
                union(p["id"], q["id"])
                pair_count += 1

    _emit({
        "phase": "pairs",
        "processed": len(photos),
        "total": len(photos),
        "pairs_found": pair_count,
    })
    logger.info("Found %d near-identical pairs", pair_count)

    _check_abort()
    _emit({"phase": "group", "message": "Grouping into stacks"})

    # 4. Group into connected components
    from collections import defaultdict
    groups = defaultdict(list)
    for p in photos:
        root = find(p["id"])
        groups[root].append(p)

    # 5. Enforce max stack span: all members must be within max_stack_span_sec
    #    of the earliest member.  Sort chronologically, keep only photos
    #    that fall within the span from the first.
    span = timedelta(seconds=max_stack_span_sec)
    trimmed_groups = []
    for members in groups.values():
        if len(members) < 2:
            continue
        # Sort by date_taken for span check
        members.sort(key=lambda p: p["date_taken"])
        earliest = members[0]["date_taken"]
        kept = [m for m in members if (m["date_taken"] - earliest) <= span]
        if len(kept) >= 2:
            trimmed_groups.append(kept)

    # 6. Sort by aesthetic score and collect IDs
    stacks = []
    for members in trimmed_groups:
        members.sort(key=lambda p: p["aesthetic_score"], reverse=True)
        stacks.append([m["id"] for m in members])

    logger.info("Detected %d stacks (%d total stacked photos)",
                len(stacks), sum(len(s) for s in stacks))
    _emit({
        "phase": "group",
        "stacks_detected": len(stacks),
        "photos_stacked": sum(len(s) for s in stacks),
    })
    return stacks


def _load_embeddings_bulk(
    db: PhotoDB,
    photo_ids: list[int],
    on_progress: Optional[Callable[[dict], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> dict[int, np.ndarray]:
    """Load CLIP embeddings for a list of photo IDs.

    Returns {photo_id: normalized_numpy_array}.
    """
    result = {}
    batch_size = 500
    total = len(photo_ids)
    last_emit = time.monotonic()
    if on_progress:
        on_progress({"phase": "load_embeddings", "loaded": 0, "total": total})
    for i in range(0, total, batch_size):
        if should_abort and should_abort():
            raise InterruptedError("stacking cancelled")
        batch = photo_ids[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        rows = db.conn.execute(
            f"SELECT photo_id, embedding FROM clip_embeddings WHERE photo_id IN ({placeholders})",
            batch,
        ).fetchall()
        for r in rows:
            vec = np.array(
                _deserialize_float_list(r["embedding"], CLIP_DIMENSIONS),
                dtype=np.float32,
            )
            # Ensure unit-normalized for cosine distance
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            result[r["photo_id"]] = vec
        now = time.monotonic()
        if on_progress and now - last_emit >= _PROGRESS_INTERVAL:
            on_progress({
                "phase": "load_embeddings",
                "loaded": min(i + batch_size, total),
                "total": total,
            })
            last_emit = now
    if on_progress:
        on_progress({"phase": "load_embeddings", "loaded": total, "total": total})
    return result


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_stacks(
    db: PhotoDB,
    stacks: list[list[int]],
    on_progress: Optional[Callable[[dict], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
    scope_photo_ids: list[int] | None = None,
):
    """Clear existing stacks and save new ones.

    Each stack list is ordered by aesthetic score descending —
    the first element becomes the top photo.

    scope_photo_ids: when set (a scoped detection, e.g. an ingest indexing one
    folder), clear ONLY stacks overlapping that scope instead of wiping the whole
    library. This is the fix for the ingest cron silently deleting every stack in
    the library each night — a scoped run must replace only its own stacks.
    """
    if scope_photo_ids is not None:
        # Scoped run — clear only stacks overlapping this scope. An empty scope
        # clears NOTHING (an empty list is still a scope, not a full-library
        # wipe); only scope_photo_ids=None means "full re-detect, clear all".
        db.clear_stacks_for_photos(scope_photo_ids)
    else:
        db.clear_stacks()
    total = len(stacks)
    last_emit = time.monotonic()
    if on_progress:
        on_progress({"phase": "save", "saved": 0, "total": total})
    for idx, stack_ids in enumerate(stacks, 1):
        if should_abort and should_abort():
            raise InterruptedError("stacking cancelled")
        db.create_stack(stack_ids, top_photo_id=stack_ids[0])
        now = time.monotonic()
        if on_progress and now - last_emit >= _PROGRESS_INTERVAL:
            on_progress({"phase": "save", "saved": idx, "total": total})
            last_emit = now
    if on_progress:
        on_progress({"phase": "save", "saved": total, "total": total})
    logger.info("Saved %d stacks to database", total)


def _resolve_directory_photo_ids(db: PhotoDB, directory: str) -> list[int]:
    """Resolve the photo IDs whose files live under `directory`, matching the
    same prefix logic detect_stacks uses to scope a directory run. Used to build
    the clear-scope for a directory-scoped stacking run so it replaces only that
    folder's stacks instead of wiping the whole library."""
    prefix = directory.rstrip("/") + "/"
    rows = db.conn.execute("SELECT id, filepath FROM photos").fetchall()
    ids = []
    for r in rows:
        abs_path = db.resolve_filepath(r["filepath"])
        if abs_path and abs_path.startswith(prefix):
            ids.append(r["id"])
    return ids


# ---------------------------------------------------------------------------
# Incremental (missing-only) stacking — the maintenance-sweep path
# ---------------------------------------------------------------------------

# Chunk size for IN (...) lists — well under SQLITE_MAX_VARIABLE_NUMBER.
_SQL_CHUNK = 900


def _chunks(seq, n=_SQL_CHUNK):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def run_incremental_stacking(
    db: PhotoDB,
    time_window_sec: float = 5.0,
    clip_threshold: float = 0.05,
    max_stack_span_sec: float = 10.0,
    dry_run: bool = False,
    on_progress: Optional[Callable[[dict], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> dict:
    """Re-detect stacks only where something changed, and write only the diff.

    The nightly maintenance sweep used to call ``run_stacking`` unscoped: a
    full-library re-detect that cleared and re-created every stack (27,165 on
    the NAS) every night on an unchanged library. This is the missing-only
    replacement.

    **What is "dirty".** ``stacking_seen`` records every photo detection has
    already considered, with the ``date_taken`` it had then. A photo is dirty
    when it is eligible (``date_taken`` set + a CLIP embedding) and either has
    no ledger row or its ``date_taken`` has changed (a clock retime). Also
    dirty: the members of any stack that lost a member to a deleted/undated
    photo, or that is down to fewer than two members.

    **Why the scope is exact, not a heuristic.** detect_stacks only ever
    unions two photos whose timestamps are within ``time_window_sec``. So the
    library splits into *sessions* — maximal runs of eligible photos whose
    consecutive gaps are all <= ``time_window_sec`` — and no stack can span two
    sessions. Re-detecting just the sessions that contain a dirty photo
    therefore reproduces exactly what a full-library run would produce for
    them. Existing stacks overlapping that scope are pulled in whole (and their
    sessions with them, to a fixpoint), so a stale stack can never be half
    inside the scope.

    **Why it writes a diff.** Detected stacks are compared to the affected
    existing stacks by *membership*: an identical stack is left untouched
    (same id, and a user's hand-picked top survives), a vanished one is
    deleted, a new one is created. ``applied`` is the number of stacks
    actually created + removed, so the log can show "nothing changed".

    **It never clears the library.** There is no ``clear_stacks()`` call and
    no unscoped ``detect_stacks``: deletions are by explicit stack id, drawn
    only from stacks overlapping the scope, and an empty scope skips
    detection entirely (``detect_stacks`` treats ``photo_ids=[]`` as "no
    scope" and would fall through to the whole library — the bug that wiped
    the library's stacks twice).

    The first run after deploy has an empty ledger, so every eligible photo is
    dirty: it costs one full detection (what every night used to cost) but
    still writes only the diff, then fills the ledger.

    ``dry_run`` computes the dirty set only (cheap: no embeddings, no
    detection) and writes nothing.

    Returns a dict: ``dirty``, ``scope_photos``, ``stacks_detected``,
    ``stacks_created``, ``stacks_removed``, ``stacks_unchanged``, ``applied``.
    """
    def _emit(event: dict):
        if on_progress:
            on_progress(event)

    def _check_abort():
        if should_abort and should_abort():
            raise InterruptedError("stacking cancelled")

    conn = db.conn
    eligible_rows = conn.execute(
        "SELECT p.id, p.date_taken, s.photo_id AS seen_id, s.date_taken AS seen_date "
        "FROM photos p LEFT JOIN stacking_seen s ON s.photo_id = p.id "
        "WHERE p.date_taken IS NOT NULL "
        "AND p.id IN (SELECT photo_id FROM clip_embeddings)"
    ).fetchall()
    eligible = {r["id"]: r["date_taken"] for r in eligible_rows}
    dirty = {r["id"] for r in eligible_rows
             if r["seen_id"] is None or r["seen_date"] != r["date_taken"]}

    # Ledger rows for photos that are no longer eligible (deleted rows cascade
    # away on their own; this is the date cleared / embedding removed case).
    gone = {r[0] for r in conn.execute("SELECT photo_id FROM stacking_seen")
            if r[0] not in eligible}

    # Existing stacks, in memory (~76k member rows on the NAS — cheap).
    stack_members: dict[int, set[int]] = {}
    photo_stack: dict[int, int] = {}
    for r in conn.execute("SELECT stack_id, photo_id FROM stack_members"):
        stack_members.setdefault(r[0], set()).add(r[1])
        photo_stack[r[1]] = r[0]

    # A stack that lost members (a photo deleted, undated, or un-embedded) or
    # is down to one member is stale: re-detect its session.
    for sid, members in stack_members.items():
        if len(members) < 2 or any(m not in eligible for m in members):
            dirty.update(members)

    result = {
        "dirty": len(dirty), "scope_photos": 0, "stacks_detected": 0,
        "stacks_created": 0, "stacks_removed": 0, "stacks_unchanged": 0,
        "applied": 0,
    }
    if dry_run or (not dirty and not gone):
        return result

    # Sessions over eligible photos with a parseable date.
    parsed = []
    for pid, ds in eligible.items():
        dt = _parse_date(ds)
        if dt is not None:
            parsed.append((dt, pid))
    parsed.sort()
    session_of: dict[int, int] = {}
    sessions: list[list[int]] = []
    window = timedelta(seconds=time_window_sec)
    prev = None
    for dt, pid in parsed:
        if prev is None or dt - prev > window:
            sessions.append([])
        sessions[-1].append(pid)
        session_of[pid] = len(sessions) - 1
        prev = dt

    # Scope = sessions containing a dirty photo, closed over existing stacks.
    scope: set[int] = set()
    affected: set[int] = set()   # existing stack ids the diff may remove
    frontier = set(dirty)
    while frontier:
        new_sessions = {session_of[p] for p in frontier if p in session_of}
        added = set()
        for si in new_sessions:
            for p in sessions[si]:
                if p not in scope:
                    scope.add(p)
                    added.add(p)
        # Photos that are dirty but have no session (unparseable date, or a
        # member of a stale stack that is no longer eligible) still pull their
        # stack into the affected set.
        touched = added | {p for p in frontier if p not in session_of}
        frontier = set()
        for p in touched:
            sid = photo_stack.get(p)
            if sid is not None and sid not in affected:
                affected.add(sid)
                frontier.update(m for m in stack_members[sid]
                                if m in session_of and m not in scope)
    result["scope_photos"] = len(scope)
    _check_abort()

    detected: list[list[int]] = []
    if scope:
        # Non-empty by construction here. NEVER pass an empty list:
        # detect_stacks reads photo_ids=[] as "no scope" -> whole library.
        detected = detect_stacks(
            db, time_window_sec, clip_threshold, None, max_stack_span_sec,
            photo_ids=sorted(scope), on_progress=on_progress,
            should_abort=should_abort,
        )
    result["stacks_detected"] = len(detected)

    existing_by_key = {frozenset(stack_members[sid]): sid for sid in affected}
    detected_keys = {frozenset(s) for s in detected}
    to_remove = [sid for key, sid in existing_by_key.items() if key not in detected_keys]
    to_create = [s for s in detected if frozenset(s) not in existing_by_key]
    result["stacks_unchanged"] = len(affected) - len(to_remove)

    _check_abort()
    _emit({"phase": "save", "saved": 0, "total": len(to_create) + len(to_remove)})
    # Removals first: stack_members.photo_id is UNIQUE.
    for chunk in _chunks(to_remove):
        ph = ",".join("?" * len(chunk))
        conn.execute(f"DELETE FROM stack_members WHERE stack_id IN ({ph})", chunk)
        conn.execute(f"DELETE FROM photo_stacks WHERE id IN ({ph})", chunk)
        conn.commit()
    last_emit = time.monotonic()
    for idx, stack_ids in enumerate(to_create, 1):
        if idx % 200 == 0:
            _check_abort()
        db.create_stack(stack_ids, top_photo_id=stack_ids[0])
        now = time.monotonic()
        if on_progress and now - last_emit >= _PROGRESS_INTERVAL:
            on_progress({"phase": "save", "saved": len(to_remove) + idx,
                         "total": len(to_create) + len(to_remove)})
            last_emit = now
    result["stacks_removed"] = len(to_remove)
    result["stacks_created"] = len(to_create)
    result["applied"] = len(to_remove) + len(to_create)

    # Ledger last: an abort anywhere above leaves these photos dirty, so the
    # next run redoes their sessions and converges.
    seen_rows = [(pid, eligible[pid]) for pid in (scope | dirty) if pid in eligible]
    for chunk in _chunks(seen_rows):
        conn.executemany(
            "INSERT INTO stacking_seen (photo_id, date_taken) VALUES (?, ?) "
            "ON CONFLICT(photo_id) DO UPDATE SET date_taken = excluded.date_taken, "
            "seen_at = datetime('now')",
            chunk,
        )
    for chunk in _chunks(gone):
        ph = ",".join("?" * len(chunk))
        conn.execute(f"DELETE FROM stacking_seen WHERE photo_id IN ({ph})", chunk)
    conn.commit()
    _emit({"phase": "save", "saved": result["applied"], "total": result["applied"]})
    logger.info(
        "Incremental stacking: %d dirty, %d in scope, %d detected, "
        "%d created, %d removed, %d unchanged",
        result["dirty"], result["scope_photos"], result["stacks_detected"],
        result["stacks_created"], result["stacks_removed"],
        result["stacks_unchanged"])
    return result


def run_stacking(
    db: PhotoDB,
    time_window_sec: float = 5.0,
    clip_threshold: float = 0.05,
    directory: str | None = None,
    dry_run: bool = False,
    max_stack_span_sec: float = 10.0,
    photo_ids: list[int] | None = None,
    on_progress: Optional[Callable[[dict], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> list[list[int]]:
    """Full stacking pipeline: detect + save.

    Args:
        db: Open PhotoDB.
        time_window_sec: Temporal window for grouping.
        clip_threshold: CLIP distance threshold.
        directory: Restrict to photos in this directory.
        dry_run: If True, detect but don't save to DB.
        max_stack_span_sec: Hard cap on total time span of a stack.
        photo_ids: Restrict to these specific photo IDs (e.g. from a collection).
        on_progress: Optional callback invoked with progress events (throttled
            to every ~0.25s inside hot loops; phase transitions always fire).
            Each event is a dict with a "phase" key and phase-specific fields.
        should_abort: Optional callable. If it returns True mid-run, the
            function raises InterruptedError. Checked at phase transitions
            and throttled inside hot loops.

    Returns:
        List of detected stacks (each a list of photo IDs, best first).
    """
    stacks = detect_stacks(
        db, time_window_sec, clip_threshold, directory,
        max_stack_span_sec, photo_ids=photo_ids,
        on_progress=on_progress, should_abort=should_abort,
    )
    if not dry_run and stacks:
        # Scoped runs clear only their own stacks; a full-library run clears
        # everything. Prevents the ingest cron from wiping all stacks. The scope
        # is whatever restricted this run: an explicit photo_ids set (collection
        # mode) OR the photos under `directory` (the ingest path — index_directory
        # calls run_stacking(directory=...), which is what nightly ingest hits).
        # Resolving directory→ids here is the missing half of the original fix,
        # which only threaded photo_ids and left the directory path wiping the
        # whole library.
        scope_photo_ids = photo_ids
        if scope_photo_ids is None and directory is not None:
            scope_photo_ids = _resolve_directory_photo_ids(db, directory)
        save_stacks(db, stacks, on_progress=on_progress, should_abort=should_abort,
                    scope_photo_ids=scope_photo_ids)
    return stacks
