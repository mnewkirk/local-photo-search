"""Per-photo audit log: what changed on a photo, when, and who asked.

One row per change in ``photo_audit`` (schema v40). Every writer reaches it
through a ``PhotoDB`` method (or ``set_photo_filepath`` /
``unmatch_faces_as_duplicates``), which calls :func:`record` on the SAME
connection as the change and does not commit — the audit row lands in the
caller's transaction, so a rolled-back change leaves no row and a committed one
cannot lack it.

Who asked comes from a context variable, not from arguments, so the db layer
needs no new parameters:

* the web middleware sets ``source`` (``ui`` / ``claude`` / ``worker`` / ...,
  from ``request_intent.classify_source``) and the stated or inferred intent;
* MCP / Ask write tools add the tool name (and the MCP caller's ``intent``);
* the CLI group sets ``source='cli'`` with the command name;
* anything else records ``source='internal'`` (a background thread that lost
  the context — start threads through ``request_intent.carry_context``).

Replica: the NAS is the only writer and ``sync-replica.sh`` swaps the whole
database, so a row written on a replica would vanish at the next sync and
would duplicate the NAS's own row until then. ``record`` is a no-op wherever
``PHOTOSEARCH_NAS_URL`` is set or a request ran in replica mode
(``suppressed``); the replica reads the NAS's log through the API instead.

History starts at deploy: nothing here reconstructs earlier changes.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
from typing import Iterable, Optional

MAX_FIELD = 300      # source / intent / detail / collection_name
MAX_VALUE = 4000     # old_value / new_value (tag lists can be long)

# Actions. Kept as constants so readers and tests share one vocabulary.
PHOTO_ADDED = "photo_added"
FILEPATH_CHANGED = "filepath_changed"
PHOTO_DELETED = "photo_deleted"      # written by a DB trigger, see db._init_schema
FIELD_CHANGED = "field_changed"          # date_taken, gps_*, place_name via update_photo
LOCATION_SET = "location_set"
TAGS_SET = "tags_set"
COLLECTION_ADDED = "collection_added"
COLLECTION_REMOVED = "collection_removed"
FACE_ASSIGNED = "face_assigned"
FACE_CLEARED = "face_cleared"            # a human "no": match_source='rejected'
FACE_UNMATCHED = "face_unmatched"        # automatic: duplicate resolver, bulk-unmatch undo

ACTIONS = (PHOTO_ADDED, PHOTO_DELETED, FILEPATH_CHANGED, FIELD_CHANGED, LOCATION_SET, TAGS_SET,
           COLLECTION_ADDED, COLLECTION_REMOVED, FACE_ASSIGNED, FACE_CLEARED,
           FACE_UNMATCHED)

# Columns update_photo() audits. Everything else it writes (descriptions,
# scores, tags from the passes) is machine output with its own provenance
# (`generations`) and would bury the changes someone made on purpose.
AUDITED_PHOTO_COLUMNS = ("filepath", "date_taken", "gps_lat", "gps_lon", "place_name")

_UNSET = object()
_ctx: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "photosearch_audit_ctx", default=None)


# ---------------------------------------------------------------------------
# Context: who is asking, and why
# ---------------------------------------------------------------------------

def current() -> dict:
    """``{"source", "intent", "suppress"}`` for the running call."""
    c = _ctx.get() or {}
    return {"source": c.get("source") or "internal",
            "intent": c.get("intent"),
            "suppress": bool(c.get("suppress"))}


def set_context(source=_UNSET, intent=_UNSET, suppress=_UNSET):
    """Set the audit context (unspecified keys are inherited). Returns the
    token for :func:`reset_context`."""
    new = dict(_ctx.get() or {})
    for k, v in (("source", source), ("intent", intent), ("suppress", suppress)):
        if v is not _UNSET:
            new[k] = v
    return _ctx.set(new)


def reset_context(token) -> None:
    _ctx.reset(token)


@contextlib.contextmanager
def context(source=_UNSET, intent=_UNSET, suppress=_UNSET):
    token = set_context(source, intent, suppress)
    try:
        yield
    finally:
        reset_context(token)


@contextlib.contextmanager
def tool_context(tool: str, intent: Optional[str] = None, source: Optional[str] = None):
    """Wrap a write-tool call: keep the caller's source (MCP, Ask's request),
    and name the tool in the intent so the log says which tool wrote."""
    cur = current()
    why = intent or cur["intent"]
    label = f"tool {tool}" + (f": {why}" if why else "")
    with context(source=source or cur["source"], intent=label):
        yield


def suppressed():
    """No audit rows inside the block — for replica mirror writes."""
    return context(suppress=True)


def enabled() -> bool:
    if current()["suppress"]:
        return False
    return not (os.environ.get("PHOTOSEARCH_NAS_URL") or "").strip()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _clip(v, limit: int) -> Optional[str]:
    if v is None:
        return None
    if not isinstance(v, str):
        v = json.dumps(v, sort_keys=True, default=str)
    return v if len(v) <= limit else v[:limit - 1] + "…"


def record(conn, photo_id: int, action: str, *, field: Optional[str] = None,
           old=None, new=None, collection_id: Optional[int] = None,
           collection_name: Optional[str] = None, face_id: Optional[int] = None,
           detail: Optional[str] = None) -> None:
    """Append one audit row on ``conn``. Does NOT commit. Never raises on a
    missing table (an old database opened by a script that bypassed
    ``PhotoDB``): the change it describes matters more than its log line."""
    record_many(conn, [dict(photo_id=photo_id, action=action, field=field, old=old,
                            new=new, collection_id=collection_id,
                            collection_name=collection_name, face_id=face_id,
                            detail=detail)])


def record_many(conn, events: Iterable[dict]) -> int:
    """Append several rows (same keys as :func:`record`) in one statement."""
    if not enabled():
        return 0
    c = current()
    source = _clip(c["source"], MAX_FIELD)
    intent = _clip(c["intent"], MAX_FIELD)
    rows = [(e["photo_id"], e["action"], e.get("field"),
             _clip(e.get("old"), MAX_VALUE), _clip(e.get("new"), MAX_VALUE),
             e.get("collection_id"), _clip(e.get("collection_name"), MAX_FIELD),
             e.get("face_id"), _clip(e.get("detail"), MAX_FIELD), source, intent)
            for e in events]
    if not rows:
        return 0
    try:
        conn.executemany(
            "INSERT INTO photo_audit (photo_id, action, field, old_value, new_value, "
            "collection_id, collection_name, face_id, detail, source, intent) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    except Exception as err:  # noqa: BLE001
        if "no such table" in str(err):
            return 0
        raise
    return len(rows)


def person_names(conn, person_ids: Iterable[Optional[int]]) -> dict:
    """``{person_id: name}`` for the ids that exist."""
    ids = sorted({int(p) for p in person_ids if p is not None})
    out: dict = {}
    for i in range(0, len(ids), 500):
        batch = ids[i:i + 500]
        for r in conn.execute(
                f"SELECT id, name FROM persons WHERE id IN ({','.join('?' * len(batch))})",
                batch):
            out[r[0]] = r[1]
    return out


def face_label_events(conn, old_rows, new_person_id: Optional[int],
                      new_source: Optional[str]) -> list[dict]:
    """Audit events for faces about to get ``new_person_id`` / ``new_source``.

    ``old_rows``: ``(face_id, photo_id, person_id, match_source)`` as they are
    BEFORE the write. A face whose person is unchanged is skipped unless its
    source changed (re-confirming a label as ``manual`` is a change worth
    keeping: it is what later protects the label from auto-matching)."""
    names = person_names(conn, [r[2] for r in old_rows] + [new_person_id])
    if new_person_id is not None:
        action = FACE_ASSIGNED
    elif new_source == "rejected":
        action = FACE_CLEARED
    else:
        action = FACE_UNMATCHED
    new_name = names.get(new_person_id) if new_person_id is not None else None
    events = []
    for face_id, photo_id, old_pid, old_source in old_rows:
        if old_pid == new_person_id and old_source == new_source:
            continue
        events.append(dict(
            photo_id=photo_id, action=action, field="person", face_id=face_id,
            old=names.get(old_pid) if old_pid is not None else None,
            new=new_name, detail=f"match_source {old_source or '-'} -> {new_source or '-'}"))
    return events


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

_COLS = ("id, ts, photo_id, action, field, old_value, new_value, collection_id, "
         "collection_name, face_id, detail, source, intent")


def _rows(conn, sql: str, params: list) -> list[dict]:
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    except Exception as err:  # noqa: BLE001
        if "no such table" in str(err):
            return []
        raise


def query(conn, *, photo_id: Optional[int] = None, collection_id: Optional[int] = None,
          collection_name: Optional[str] = None, action: Optional[str] = None,
          source: Optional[str] = None, since: Optional[str] = None,
          before_id: Optional[int] = None, limit: int = 200) -> list[dict]:
    """Newest first. ``since`` is a ``YYYY-MM-DD[ HH:MM:SS]`` UTC lower bound."""
    where, params = [], []
    for col, val in (("photo_id", photo_id), ("collection_id", collection_id),
                     ("action", action), ("source", source)):
        if val is not None:
            where.append(f"{col} = ?")
            params.append(val)
    if collection_name:
        where.append("collection_name = ? COLLATE NOCASE")
        params.append(collection_name)
    if since:
        where.append("ts >= ?")
        params.append(since)
    if before_id is not None:
        where.append("id < ?")
        params.append(before_id)
    limit = max(1, min(int(limit), 2000))
    sql = (f"SELECT {_COLS} FROM photo_audit"
           + (" WHERE " + " AND ".join(where) if where else "")
           + " ORDER BY id DESC LIMIT ?")
    return _rows(conn, sql, params + [limit])


def removed_from_collection(conn, collection_id: int) -> list[dict]:
    """Photos whose LAST membership event in this collection is a removal and
    that are not in it now — i.e. "which photo is no longer in the collection".
    A photo removed and then re-added is not listed; one removed from a since
    deleted collection is (the collection's id is all that is left of it)."""
    return _rows(conn, """
        SELECT a.photo_id, a.ts, a.source, a.intent, a.detail, a.collection_name, a.id
        FROM photo_audit a
        WHERE a.collection_id = ? AND a.action = ?
          AND a.id = (SELECT MAX(b.id) FROM photo_audit b
                      WHERE b.collection_id = a.collection_id
                        AND b.photo_id = a.photo_id AND b.action IN (?, ?))
          AND NOT EXISTS (SELECT 1 FROM collection_photos cp
                          WHERE cp.collection_id = a.collection_id
                            AND cp.photo_id = a.photo_id)
        ORDER BY a.id DESC""",
        [collection_id, COLLECTION_REMOVED, COLLECTION_ADDED, COLLECTION_REMOVED])
