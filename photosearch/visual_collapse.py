"""Detect category-visual COLLAPSE — one tag set stamped on a whole shoot.

Each answer the visual pass gives can be plausible on its own and still be
wrong in aggregate: on 2026-10-03 minicpm gave 1,211 of 1,260 photos the
identical set `colorful, sunny, vibrant` (7 distinct sets in the folder). No
per-photo guard can see that — it is a property of the batch. So this is a
REPORT, never a gate: it flags a folder for a human to look at (and a targeted
re-run), it changes nothing.

Thresholds, measured on the replica 2026-10-06 over the 540 folders with
>= 50 visually-tagged photos: the top tag set covers a median 24% of a folder,
p75 34%, p90 50%, p95 64%. 59 folders sit at >= 50% — the 2026-10-03 shoot at
96%, older cohorts at 70-97% (`colorful, joyful, vibrant` on kids' events,
`long-exposure, low-light, moody, overcast` from the pre-derive tagger). The
flag is >= 50% at >= 50 photos: the p90, and well clear of the median, so it
names the clear cases without calling every sunny match collapsed. A genuinely
uniform shoot (all `sunny`) can trip it; that is the cost of a flag that a
human reads, and why it is not a gate.

Only non-empty sets count: a stored `'[]'` is the model answering "nothing
notable", which is a legitimate and common answer, not a collapse.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Iterable, Optional

COLLAPSE_MIN_PHOTOS = 50
COLLAPSE_TOP_SHARE = 0.5

_TAGGED = "visual_tags IS NOT NULL AND visual_tags != '[]'"


def _set_label(raw: str) -> str:
    try:
        tags = json.loads(raw)
    except (TypeError, ValueError):
        return str(raw)
    return ", ".join(tags) if isinstance(tags, list) else str(raw)


def summarize_sets(raw_sets: Iterable[str],
                   min_photos: int = COLLAPSE_MIN_PHOTOS,
                   top_share: float = COLLAPSE_TOP_SHARE) -> Optional[dict]:
    """Stats over stored `visual_tags` JSON strings, or None if there are none.

    Stored arrays are sorted and deduped by `visual_tags_derive.merge_tags`,
    so the raw JSON string is a canonical key for the set.
    """
    counts = Counter(raw_sets)
    n = sum(counts.values())
    if not n:
        return None
    top_raw, top_n = counts.most_common(1)[0]
    share = top_n / n
    return {
        "tagged": n,
        "distinct_sets": len(counts),
        "top_set": _set_label(top_raw),
        "top_count": top_n,
        "top_share": round(share, 4),
        "collapsed": n >= min_photos and share >= top_share,
    }


def stats_for_ids(conn, ids: list[int], **kw) -> Optional[dict]:
    """Collapse stats for one photo-id scope (a batch). Chunked under the
    SQLite variable cap."""
    raw: list[str] = []
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        ph = ",".join("?" * len(chunk))
        raw.extend(r[0] for r in conn.execute(
            f"SELECT visual_tags FROM photos WHERE id IN ({ph}) AND {_TAGGED}", chunk))
    return summarize_sets(raw, **kw)


def batch_warning(stats: Optional[dict]) -> tuple[Optional[str], Optional[str]]:
    """(short warning for the flow node, full text for its tooltip)."""
    if not stats or not stats["collapsed"]:
        return None, None
    pct = round(100 * stats["top_share"])
    return (f"⚠ {pct}% share one tag set",
            f"category-visual collapse: {stats['top_count']:,} of "
            f"{stats['tagged']:,} tagged photos are exactly "
            f"[{stats['top_set']}] ({stats['distinct_sets']} distinct sets). "
            f"Consider re-running category-visual on this folder.")


def collapsed_folders(conn, min_photos: int = COLLAPSE_MIN_PHOTOS,
                      top_share: float = COLLAPSE_TOP_SHARE,
                      folder_prefix: Optional[str] = None) -> list[dict]:
    """Every folder at or over the bar, worst first."""
    sql = f"SELECT folder, visual_tags FROM photos WHERE {_TAGGED}"
    args: list = []
    if folder_prefix:
        p = folder_prefix.rstrip("/")
        # Same range trick as get_directory_photo_ids: `p` or any subfolder.
        sql += " AND (folder = ? OR (folder >= ? AND folder < ?))"
        args += [p, p + "/", p + "0"]
    by_folder: dict[str, list[str]] = {}
    for folder, raw in conn.execute(sql, args):
        by_folder.setdefault(folder or "", []).append(raw)
    out = []
    for folder, sets in by_folder.items():
        s = summarize_sets(sets, min_photos=min_photos, top_share=top_share)
        if s and s["collapsed"]:
            out.append({"folder": folder, **s})
    out.sort(key=lambda r: (-r["top_share"], -r["tagged"]))
    return out


def folder_photo_ids(conn, folders: list[str], only_top_set: bool = False) -> list[int]:
    """Ids of the TAGGED photos in `folders` — the cohort a targeted re-run
    would clear. `only_top_set` narrows to photos carrying each folder's
    dominant set (the photos the collapse actually stamped)."""
    ids: list[int] = []
    for folder in folders:
        rows = conn.execute(
            f"SELECT id, visual_tags FROM photos WHERE folder = ? AND {_TAGGED}",
            (folder,)).fetchall()
        if only_top_set and rows:
            top = Counter(r[1] for r in rows).most_common(1)[0][0]
            rows = [r for r in rows if r[1] == top]
        ids.extend(r[0] for r in rows)
    return ids
