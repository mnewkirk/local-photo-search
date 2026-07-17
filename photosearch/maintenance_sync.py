"""Reconcile replica-computed maintenance results back to the NAS.

Background: maintenance stages write to whatever PHOTOSEARCH_DB points at. On
the local replica that's photo_index.db.local, which sync-replica.sh replaces
wholesale (NAS -> replica, mv over the file). So a replica-side sweep is lost on
the next sync unless its results are pushed up. This module is that push.

Two modes, decided per stage:

  trigger  — the NAS recomputes the stage itself over its own current data. No
             payload and no fingerprint guard: a stale-input mismatch is
             structurally impossible. Right for anything cheap + deterministic.
             geocode MUST be here rather than transfer — the replica lacks the
             /data/geonames rich dataset, so replica-computed place names would
             silently downgrade the NAS's labels.
  transfer — the replica ships computed rows, because recomputing on the N100 is
             precisely what we're avoiding. Only stacking earns this.
  excluded — cannot run on the replica at all (see EXCLUDED_STAGES).

Kept separate from maintenance.py so that module stays about RUNNING stages and
this one about RECONCILING them.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Cheap + deterministic: the NAS redoes these itself in ~a second.
TRIGGER_STAGES = frozenset({
    "geocode",
    "normalize",
    "infer",
    "normalize_inferred",
    "resolve_dups",
    "normalize_aesthetics",
    "normalize_subject_aesthetics",
})

# Expensive to recompute on the N100 -> ship the rows instead.
TRANSFER_STAGES = frozenset({"stacking"})

# Must not run on the replica at all:
#   colors       — reads pixels; the replica has no originals (PHOTO_ROOT unset,
#                  images proxy from the NAS), so it cannot be correct locally.
#   dedup_photos — DELETEs photos; destructive cross-machine ops are out of scope.
#   match_faces,
#   recluster    — already served by export-face-state / apply-face-state.
#   requeue      — clears worker_processed markers, but the fleet claims from the
#                  NAS, so a local run is a no-op with a misleading success.
EXCLUDED_STAGES = frozenset({
    "colors",
    "dedup_photos",
    "match_faces",
    "recluster",
    "requeue",
})


def push_mode(stage: str) -> str:
    """Return 'trigger' | 'transfer' | 'excluded' for a sweep stage name."""
    if stage in TRANSFER_STAGES:
        return "transfer"
    if stage in TRIGGER_STAGES:
        return "trigger"
    if stage in EXCLUDED_STAGES:
        return "excluded"
    raise ValueError(f"unknown maintenance stage: {stage!r}")


def photo_fingerprint(db) -> dict:
    """Cheap 'is the photo index the same?' fingerprint — two indexed queries.

    Deliberately NOT a per-row update-date comparison: this is ~7ms on a 150k
    library. Pragmatic, not cryptographic — defeating it needs a delete plus an
    insert with a higher id between checks. The only in-tree deleter is dedup,
    which is excluded from push and opt-in. Accepted.
    """
    row = db.conn.execute(
        "SELECT COUNT(*) AS photo_count, MAX(id) AS photo_max_id FROM photos"
    ).fetchone()
    return {"photo_count": row["photo_count"], "photo_max_id": row["photo_max_id"]}


def fingerprints_match(a: dict, b: dict) -> bool:
    """True when two fingerprints describe the same photo index."""
    return (a.get("photo_count") == b.get("photo_count")
            and a.get("photo_max_id") == b.get("photo_max_id"))
