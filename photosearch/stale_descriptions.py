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
  verify stale      latest 'describe' row > verified_at + 8 h

Traps, each measured:
- created_at comes in two spellings ('2026-04-12T20:04:19' and
  '2026-04-12 20:04:19'); compared raw, a 'T' row sorts after every space row
  of the same day. Normalized with replace(…,'T',' ').
- verified_at is the WORKER's local clock with no zone, generations are UTC.
  Raw comparison flags 26,154 photos; with an 8 h margin (> the PDT/PST
  offset) it is 1 — the rest are verify running within hours of describe.
- A photo with categories but NO category-content row predates logging; there
  is nothing to compare, so it is counted (`unknown_*`) and never flagged.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

TEXT_PASSES = ("category-content", "keywords", "verify")
_VERIFY_TZ_MARGIN = "+8 hours"


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
               p.verified_at IS NOT NULL AND ds.t IS NOT NULL
                 AND ds.t > datetime(replace(p.verified_at, 'T', ' '), ?),
               p.categories IS NOT NULL AND cc.t IS NULL,
               p.keywords   IS NOT NULL AND kw.t IS NULL
          FROM photos p
          JOIN d ON d.photo_id = p.id
          LEFT JOIN g cc ON cc.photo_id = p.id AND cc.text_type = 'category-content'
          LEFT JOIN g kw ON kw.photo_id = p.id AND kw.text_type = 'keywords'
          LEFT JOIN g ds ON ds.photo_id = p.id AND ds.text_type = 'describe'
         WHERE 1 = 1 {scope}
    """, [_VERIFY_TZ_MARGIN, *params]).fetchall()
    out = {p: [] for p in TEXT_PASSES}
    out["unknown_category-content"] = 0
    out["unknown_keywords"] = 0
    for pid, cc, kw, vf, ucc, ukw in rows:
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
