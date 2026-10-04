"""Find photos whose description-derived passes predate their description.

category-content and keywords are extracted FROM `photos.description`; verify
checks it. When the description is written over (a verify rewrite, an M28
re-describe) after those passes ran, they describe text that is gone. Until
2026-10-04 the fleet ran verify LAST, so every verify rewrite did exactly this:
13,637 photos on the replica copy, all of them verify rewrites. The server now
re-queues dependents on every description write
(`PhotoDB.invalidate_description_dependents`); this finds the historical ones.

Evidence comes from `generations`, the per-artifact provenance log:

  text changed at   T = max(created_at) over 'describe' + 'verify' rows
                        ('verify' rows ARE rewritten descriptions)
  categories stale  latest 'category-content' row < T
  keywords stale    latest 'keywords' row < T
  verify stale      latest 'describe' row > verified_at, both in UTC

Traps, each measured:
- created_at comes in two spellings ('2026-04-12T20:04:19' and
  '2026-04-12 20:04:19'); compared raw, a 'T' row sorts after every space row
  of the same day. Normalized with replace(…,'T',' ').
- verified_at was the WORKER's clock with no zone; generations are UTC. The
  clock is recoverable exactly: a verify REWRITE logs a 'verify' generation
  (UTC, server) in the same submit as its verified_at, so their difference is
  the worker's offset. Measured over all ~14,000 rewrites: every stamp from
  2026-05-25 on is Pacific (-7 h in PDT; the few -7.25..-8 are submit
  latency, never a third clock). April is mixed: 04-09 is Pacific (60 of 141
  stamps precede their own photo's description if read as UTC, 0 as
  Pacific), the 04-12 rewrites are UTC (offset 0, the in-process NAS run),
  04-10/11 carry no evidence either way.
  So `_verified_utc` reads every zone-less stamp as Pacific, with real DST
  rules. That is EXACT for everything but the April UTC stamps, and for those
  it is the later of the two readings — a photo is flagged only if stale
  under both, never wrongly; it could miss a re-describe within 7 h of an
  April-12 verification, and there are none (measured). NOT a margin: an 8 h
  margin also hid every real re-describe inside it, and verify normally runs
  within an hour of describe. Stamps written since the fix carry an explicit
  Z (worker._utc_stamp). No margin is needed after conversion: a photo cannot
  be claimed for verify until its description exists, so a describe logged
  after the verification is a genuine re-describe.
- A photo with categories but NO category-content row predates logging; there
  is nothing to compare, so it is counted (`unknown_*`) and never flagged.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

TEXT_PASSES = ("category-content", "keywords", "verify")

# Zone-less verified_at stamps are the fleet's local clock (module docstring).
_PACIFIC = ZoneInfo("America/Los_Angeles")


def _verified_utc(stamp: str) -> Optional[str]:
    """`verified_at` as 'YYYY-MM-DD HH:MM:SS' UTC — the generations format."""
    if not stamp:
        return None
    s = stamp.strip()
    try:
        if s.endswith("Z") or s.endswith("+00:00"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s.replace(" ", "T"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_PACIFIC)
    except ValueError:
        return None
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _scope_sql(folder: Optional[str], collection_id: Optional[int]) -> tuple[str, list]:
    if folder:
        f = folder.strip("/")
        return ("AND (p.folder = ? OR (p.folder >= ? AND p.folder < ?))",
                [f, f + "/", f + "0"])
    if collection_id is not None:
        return ("AND p.id IN (SELECT photo_id FROM collection_photos "
                "WHERE collection_id = ?)", [collection_id])
    return "", []


def find_stale(conn: sqlite3.Connection, folder: Optional[str] = None,
               collection_id: Optional[int] = None) -> dict:
    """{pass: [photo ids]} for each text pass, plus `unknown_*` counts."""
    scope, params = _scope_sql(folder, collection_id)
    rows = conn.execute(f"""
        WITH g AS (
            SELECT photo_id, text_type, MAX(replace(created_at, 'T', ' ')) AS t
              FROM generations
             WHERE text_type IN ('describe', 'verify', 'category-content', 'keywords')
             GROUP BY photo_id, text_type),
        d AS (SELECT photo_id, MAX(t) AS t FROM g
               WHERE text_type IN ('describe', 'verify') GROUP BY photo_id)
        SELECT p.id,
               p.categories IS NOT NULL AND cc.t IS NOT NULL AND cc.t < d.t,
               p.keywords   IS NOT NULL AND kw.t IS NOT NULL AND kw.t < d.t,
               p.verified_at, ds.t,
               p.categories IS NOT NULL AND cc.t IS NULL,
               p.keywords   IS NOT NULL AND kw.t IS NULL
          FROM photos p
          JOIN d ON d.photo_id = p.id
          LEFT JOIN g cc ON cc.photo_id = p.id AND cc.text_type = 'category-content'
          LEFT JOIN g kw ON kw.photo_id = p.id AND kw.text_type = 'keywords'
          LEFT JOIN g ds ON ds.photo_id = p.id AND ds.text_type = 'describe'
         WHERE 1 = 1 {scope}
    """, params).fetchall()
    out = {p: [] for p in TEXT_PASSES}
    out["unknown_category-content"] = 0
    out["unknown_keywords"] = 0
    for pid, cc, kw, verified_at, describe_t, ucc, ukw in rows:
        v_utc = _verified_utc(verified_at) if describe_t else None
        vf = v_utc is not None and describe_t > v_utc
        if cc:
            out["category-content"].append(pid)
        if kw:
            out["keywords"].append(pid)
        if vf:
            out["verify"].append(pid)
        out["unknown_category-content"] += bool(ucc)
        out["unknown_keywords"] += bool(ukw)
    return out


def stale_ids(found: dict) -> list[int]:
    return sorted(set().union(*(found[p] for p in TEXT_PASSES)))


# Columns + ledger rows each pass owns — the same set clear-pass resets.
_PASS_COLUMNS = {
    "category-content": ("categories",),
    "keywords": ("keywords",),
    "verify": ("verified_at", "verification_status", "hallucination_flags"),
}


def requeue(conn: sqlite3.Connection, found: dict, chunk: int = 500) -> dict:
    """Clear exactly the stale pass on each photo so the fleet re-claims it.

    Per pass, not per photo: a photo whose categories are stale but whose
    verification is current keeps its verification. Chunked commits so a
    13k-photo requeue never holds the NAS write lock long (see the chunked
    percentile refresh).
    """
    counts = {}
    for pass_type in TEXT_PASSES:
        ids = found[pass_type]
        sets = ", ".join(f"{c} = NULL" for c in _PASS_COLUMNS[pass_type])
        for i in range(0, len(ids), chunk):
            part = ids[i:i + chunk]
            ph = ",".join("?" * len(part))
            conn.execute(f"UPDATE photos SET {sets} WHERE id IN ({ph})", part)
            conn.execute(f"DELETE FROM worker_processed WHERE pass_type = ? "
                         f"AND photo_id IN ({ph})", [pass_type, *part])
            conn.commit()
        counts[pass_type] = len(ids)
    return counts
