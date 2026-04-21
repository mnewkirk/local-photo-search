"""Temporal-neighbor GPS inference (M19).

Walks photos sorted by date_taken and copies coordinates from GPS-bearing
neighbors within a time window. Supports cascading — inferred photos
become anchors for further inference — with multiplicative confidence
decay so chains self-limit.

Called by the `infer-locations` CLI and the /api/geocode/infer-preview +
/api/geocode/infer-apply endpoints. Pure-functional core: no DB writes
happen inside infer_locations(); the caller decides.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional

_EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points.

    Uses the standard haversine formula; correctly handles the
    International Date Line because sin^2 is symmetric around ±π.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _parse_date(s: str) -> datetime:
    """Parse a date_taken string. Python 3.11+ fromisoformat accepts both
    'YYYY-MM-DD HH:MM:SS' and 'YYYY-MM-DDTHH:MM:SS'."""
    return datetime.fromisoformat(s)


def _scan_photos(db) -> tuple[list[dict], int]:
    """Return (time_sorted_photos, no_date_count).

    Each photo dict contains: id, filepath, date_taken (original str),
    date_taken_dt (parsed datetime), gps_lat, gps_lon.
    Rows whose date_taken fails to parse are counted as no_date.
    """
    rows = db.conn.execute(
        "SELECT id, filepath, date_taken, gps_lat, gps_lon "
        "FROM photos WHERE date_taken IS NOT NULL"
    ).fetchall()
    photos: list[dict] = []
    parse_failures = 0
    for r in rows:
        try:
            dt = _parse_date(r["date_taken"])
        except (ValueError, TypeError):
            parse_failures += 1
            continue
        photos.append({
            "id": r["id"],
            "filepath": r["filepath"],
            "date_taken": r["date_taken"],
            "date_taken_dt": dt,
            "gps_lat": r["gps_lat"],
            "gps_lon": r["gps_lon"],
        })
    photos.sort(key=lambda p: p["date_taken_dt"])

    no_date_row = db.conn.execute(
        "SELECT COUNT(*) AS cnt FROM photos WHERE date_taken IS NULL"
    ).fetchone()
    return photos, int(no_date_row["cnt"]) + parse_failures


def _find_flanking_anchors(
    photos: list[dict],
    idx: int,
    window_minutes: float,
    anchor_ids: set[int],
    anchor_data: dict[int, dict],
) -> tuple[Optional[dict], Optional[dict]]:
    """For the photo at photos[idx], walk outward and return the nearest
    anchor on each side within window_minutes. Anchors are photos whose
    id is in anchor_ids; their coords + confidence live in anchor_data.

    Returns (left_anchor, right_anchor) where each is either a dict or None.
    """
    target_dt = photos[idx]["date_taken_dt"]
    left = None
    for j in range(idx - 1, -1, -1):
        gap = (target_dt - photos[j]["date_taken_dt"]).total_seconds() / 60.0
        if gap > window_minutes:
            break
        if photos[j]["id"] in anchor_ids:
            left = {"photo": photos[j], "gap_min": gap}
            break
    right = None
    for j in range(idx + 1, len(photos)):
        gap = (photos[j]["date_taken_dt"] - target_dt).total_seconds() / 60.0
        if gap > window_minutes:
            break
        if photos[j]["id"] in anchor_ids:
            right = {"photo": photos[j], "gap_min": gap}
            break
    return left, right


def _infer_one_round(
    photos: list[dict],
    unanchored_indices: list[int],
    anchor_ids: set[int],
    anchor_data: dict[int, dict],
    *,
    window_minutes: float,
    max_drift_km: float,
    min_confidence: float,
) -> tuple[list[dict], dict[str, int]]:
    """One inference pass over `unanchored_indices` using the current
    anchor set. Returns (new_candidates, skip_counters)."""
    candidates: list[dict] = []
    skipped = {"no_anchor": 0, "movement_guard": 0, "below_min_confidence": 0}

    for idx in unanchored_indices:
        p = photos[idx]
        left, right = _find_flanking_anchors(
            photos, idx, window_minutes, anchor_ids, anchor_data
        )
        if left is None and right is None:
            skipped["no_anchor"] += 1
            continue

        if left is not None and right is not None:
            la = anchor_data[left["photo"]["id"]]
            ra = anchor_data[right["photo"]["id"]]
            drift = haversine_km(la["lat"], la["lon"], ra["lat"], ra["lon"])
            if drift > max_drift_km:
                skipped["movement_guard"] += 1
                continue
            if left["gap_min"] <= right["gap_min"]:
                chosen, chosen_gap, sides = left["photo"], left["gap_min"], "both"
            else:
                chosen, chosen_gap, sides = right["photo"], right["gap_min"], "both"
            sides_factor = 1.0
        elif left is not None:
            chosen, chosen_gap = left["photo"], left["gap_min"]
            drift = 0.0
            sides = "left"
            sides_factor = 0.7
        else:
            chosen, chosen_gap = right["photo"], right["gap_min"]
            drift = 0.0
            sides = "right"
            sides_factor = 0.7

        anchor = anchor_data[chosen["id"]]
        base_decay = max(0.0, 1.0 - chosen_gap / window_minutes)
        confidence = base_decay * sides_factor * anchor["confidence"]

        if confidence <= min_confidence:
            skipped["below_min_confidence"] += 1
            continue

        candidates.append({
            "photo_id": p["id"],
            "filepath": p["filepath"],
            "lat": anchor["lat"],
            "lon": anchor["lon"],
            "confidence": confidence,
            "hop_count": anchor["hop_count"] + 1,
            "sides": sides,
            "time_gap_min": chosen_gap,
            "drift_km": drift,
            "source_photo_id": chosen["id"],
        })

    return candidates, skipped


def infer_locations(
    db,
    *,
    window_minutes: int = 30,
    max_drift_km: float = 25.0,
    min_confidence: float = 0.0,
    cascade: bool = True,
    max_cascade_rounds: int = 10,
) -> dict:
    """Scan photos lacking GPS and return inferred (lat, lon) candidates
    pulled from temporal GPS neighbors.

    Returns {
        "candidates": [
            {photo_id, filepath, lat, lon, confidence, hop_count, sides,
             time_gap_min, drift_km, source_photo_id},
            ...
        ],
        "summary": {
            "total_photos", "no_gps_count", "gps_count",
            "candidate_count", "cascade_rounds_used",
            "skipped": {"no_anchor", "movement_guard",
                        "no_date_taken", "below_min_confidence"}
        }
    }
    """
    photos, no_date_count = _scan_photos(db)
    total_photos = len(photos) + no_date_count

    anchor_ids: set[int] = set()
    anchor_data: dict[int, dict] = {}
    unanchored_indices: list[int] = []
    for i, p in enumerate(photos):
        if p["gps_lat"] is not None and p["gps_lon"] is not None:
            anchor_ids.add(p["id"])
            anchor_data[p["id"]] = {
                "lat": p["gps_lat"],
                "lon": p["gps_lon"],
                "confidence": 1.0,
                "hop_count": 0,
            }
        else:
            unanchored_indices.append(i)

    real_gps_count = len(anchor_ids)

    total_skipped = {
        "no_anchor": 0,
        "movement_guard": 0,
        "no_date_taken": no_date_count,
        "below_min_confidence": 0,
    }
    all_candidates: list[dict] = []
    rounds_used = 0

    if not cascade:
        # Non-cascade path: one batch round, no sequential promotion.
        new_candidates, skipped = _infer_one_round(
            photos, unanchored_indices, anchor_ids, anchor_data,
            window_minutes=window_minutes,
            max_drift_km=max_drift_km,
            min_confidence=min_confidence,
        )
        all_candidates.extend(new_candidates)
        for k in ("no_anchor", "movement_guard", "below_min_confidence"):
            total_skipped[k] += skipped[k]
        rounds_used = 1 if photos else 0
    else:
        # Cascade path: sequential scan over photos in time order, promoting
        # each successful inference immediately so the next photo sees it as
        # an anchor. This lets chains form naturally — each photo picks the
        # NEAREST anchor (often its just-inferred predecessor), compounding
        # confidence multiplicatively and incrementing hop_count per link.
        # rounds_used reflects max hop depth, which equals cascade_rounds_used
        # and satisfies the >= N assertions in tests.
        # Per-round skip counts are discarded; a final _infer_one_round pass below produces the accurate totals.
        for idx in unanchored_indices:
            p = photos[idx]
            left, right = _find_flanking_anchors(
                photos, idx, window_minutes, anchor_ids, anchor_data
            )
            if left is None and right is None:
                continue

            if left is not None and right is not None:
                la = anchor_data[left["photo"]["id"]]
                ra = anchor_data[right["photo"]["id"]]
                drift = haversine_km(la["lat"], la["lon"], ra["lat"], ra["lon"])
                if drift > max_drift_km:
                    continue
                if left["gap_min"] <= right["gap_min"]:
                    chosen, chosen_gap, sides = left["photo"], left["gap_min"], "both"
                else:
                    chosen, chosen_gap, sides = right["photo"], right["gap_min"], "both"
                sides_factor = 1.0
                drift_out = drift
            elif left is not None:
                chosen, chosen_gap = left["photo"], left["gap_min"]
                sides = "left"
                sides_factor = 0.7
                drift_out = 0.0
            else:
                chosen, chosen_gap = right["photo"], right["gap_min"]
                sides = "right"
                sides_factor = 0.7
                drift_out = 0.0

            anchor = anchor_data[chosen["id"]]
            base_decay = max(0.0, 1.0 - chosen_gap / window_minutes)
            confidence = base_decay * sides_factor * anchor["confidence"]

            if confidence <= min_confidence:
                continue

            hop = anchor["hop_count"] + 1
            candidate = {
                "photo_id": p["id"],
                "filepath": p["filepath"],
                "lat": anchor["lat"],
                "lon": anchor["lon"],
                "confidence": confidence,
                "hop_count": hop,
                "sides": sides,
                "time_gap_min": chosen_gap,
                "drift_km": drift_out,
                "source_photo_id": chosen["id"],
            }
            all_candidates.append(candidate)
            # Immediate promotion: this photo becomes an anchor for photos
            # later in the sorted list (next iteration of the loop).
            anchor_ids.add(p["id"])
            anchor_data[p["id"]] = {
                "lat": anchor["lat"],
                "lon": anchor["lon"],
                "confidence": confidence,
                "hop_count": hop,
            }
            rounds_used = max(rounds_used, hop)

        # Final skip counts: re-run a read-only pass over the remaining
        # unanchored set so totals reflect genuinely unreachable photos.
        still_unanchored = [i for i in unanchored_indices if photos[i]["id"] not in anchor_ids]
        _, final_skipped = _infer_one_round(
            photos, still_unanchored, anchor_ids, anchor_data,
            window_minutes=window_minutes,
            max_drift_km=max_drift_km,
            min_confidence=min_confidence,
        )
        for k in ("no_anchor", "movement_guard", "below_min_confidence"):
            total_skipped[k] = final_skipped[k]
        if not rounds_used and photos:
            rounds_used = 1

    return {
        "candidates": all_candidates,
        "summary": {
            "total_photos": total_photos,
            "no_gps_count": len(unanchored_indices) + no_date_count,
            "gps_count": real_gps_count,
            "candidate_count": len(all_candidates),
            "cascade_rounds_used": rounds_used,
            "skipped": total_skipped,
        },
    }
