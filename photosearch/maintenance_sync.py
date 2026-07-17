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

import json
import logging

import requests

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


def eligible_stages(stage_results: list) -> list:
    """Stage names from a sweep result that may be pushed to the NAS.

    Only 'done' stages qualify: 'skipped' changed nothing, 'preview' was a
    dry-run, and 'cancelled' left partial state that must never ship. Excluded
    stages are dropped defensively — they should have been rejected before the
    sweep ever ran them.
    """
    out = []
    for result in stage_results:
        name = result.get("stage")
        if result.get("status") != "done":
            continue
        if not name or push_mode(name) == "excluded":
            continue
        out.append(name)
    return out


def collect_stacking_rows(db) -> list:
    """Every stack as {"members": [{"photo_id", "is_top"}, ...]}.

    Stack ids are deliberately NOT carried: they're local autoincrement values
    and the NAS re-mints its own on apply. Photo ids ARE stable (AUTOINCREMENT,
    and the replica is a dump of the NAS), so they're safe join keys.
    """
    grouped: dict = {}
    for row in db.conn.execute(
        "SELECT stack_id, photo_id, is_top FROM stack_members "
        "ORDER BY stack_id, photo_id"
    ):
        grouped.setdefault(row["stack_id"], []).append(
            {"photo_id": row["photo_id"], "is_top": row["is_top"]}
        )
    return [{"members": members} for members in grouped.values()]


def collect_payload(db, stage_results: list) -> dict:
    """Build the maintenance-apply request body from a sweep's stages list."""
    names = eligible_stages(stage_results)
    runs = db.get_maintenance_runs()
    stages = {}
    for name in names:
        run = runs.get(name)
        if not run:
            # Not stamped -> the sweep didn't consider it applied. Skip.
            logger.warning("stage %s eligible but unstamped; skipping push", name)
            continue
        stages[name] = {"mode": push_mode(name), "last_run_at": run["last_run_at"]}

    payload = {
        "fingerprint": photo_fingerprint(db),
        "stages": stages,
        "stacking": None,
    }
    if "stacking" in stages:
        payload["stacking"] = collect_stacking_rows(db)
    return payload


def fetch_nas_fingerprint(nas_url: str, timeout: int = 10) -> dict:
    """GET the NAS's fingerprint. Raises on any failure.

    Lives here (rather than inline in web.py) so both the pre-flight gate and
    the drift panel share one implementation, and so tests can patch a single
    seam.

    Note on restarts: web.py's shutdown middleware only 503s /api/worker/* and
    /api/photos/*/full — NOT /api/admin/*. So a NAS mid-restart surfaces here as
    a connection error, not a 503 with Retry-After. That's why there's no
    retry/backoff in this module: the caller treats it as 'unreachable' and the
    user retries. Do not add a second backoff for a case that can't arise.
    """
    r = requests.get(f"{nas_url.rstrip('/')}/api/admin/maintenance-fingerprint",
                     timeout=timeout)
    r.raise_for_status()
    return r.json()


def _sse_has_fatal(body: str) -> bool:
    """True if an SSE body contains a terminal 'fatal' event.

    Parses the `data:` lines rather than substring-matching the raw text: the
    latter silently depends on json.dumps' default separators, so a compact
    re-serialization server-side would make a REAL failure read as success.
    Unparseable lines are ignored — a malformed frame is not a fatal event.
    """
    for line in (body or "").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[len("data:"):].strip())
        except (ValueError, TypeError):
            continue
        if isinstance(event, dict) and event.get("type") == "fatal":
            return True
    return False


def push_to_nas(db, nas_url: str, stage_results: list, *,
                timeout: int = 900) -> dict:
    """Reconcile a completed local sweep to the NAS. Returns per-stage results.

    Transfer first (fast + bounded), triggers second (geocode can be slow on the
    N100). The two modes are separate calls, so they can partially succeed
    relative to each other — hence per-stage results rather than one boolean.
    """
    nas_url = (nas_url or "").rstrip("/")
    payload = collect_payload(db, stage_results)
    stages = payload["stages"]
    if not stages:
        return {"ok": True, "stages": {}}

    out: dict = {}

    # --- transfer ---------------------------------------------------------
    transfer = {n: i for n, i in stages.items() if i["mode"] == "transfer"}
    if transfer:
        body = {"fingerprint": payload["fingerprint"],
                "stages": transfer,
                "stacking": payload["stacking"]}
        try:
            r = requests.post(f"{nas_url}/api/admin/maintenance-apply",
                              json=body, timeout=timeout)
        except requests.exceptions.RequestException as e:
            logger.warning("maintenance push (transfer) failed: %s", e)
            return {"ok": False, "error": "unreachable", "stages": out}
        if r.status_code == 409:
            return {"ok": False, "error": "fingerprint_mismatch", "stages": out}
        if r.status_code != 200:
            return {"ok": False, "error": f"http_{r.status_code}", "stages": out}
        try:
            applied = r.json().get("applied") or {}
        except (ValueError, TypeError) as e:
            logger.warning("maintenance push (transfer) bad response body: %s", e)
            return {"ok": False, "error": "bad_response", "stages": out}
        out.update(applied)

    # --- trigger ----------------------------------------------------------
    trigger = sorted(n for n, i in stages.items() if i["mode"] == "trigger")
    if trigger:
        try:
            # The NAS sweep endpoint is SSE; requests reads the finite stream to
            # completion and hands back the whole body.
            r = requests.post(f"{nas_url}/api/admin/maintenance-sweep",
                              json={"apply": True, "stages": trigger},
                              timeout=timeout)
        except requests.exceptions.RequestException as e:
            logger.warning("maintenance push (trigger) failed: %s", e)
            for name in trigger:
                out[name] = {"status": "failed", "reason": "unreachable"}
            return {"ok": False, "error": "unreachable", "stages": out}
        if r.status_code != 200 or _sse_has_fatal(r.text):
            for name in trigger:
                out[name] = {"status": "failed", "reason": f"http_{r.status_code}"}
            return {"ok": False, "error": f"http_{r.status_code}", "stages": out}
        for name in trigger:
            out[name] = {"status": "triggered"}

    ok = all(v.get("status") in ("applied", "triggered", "skipped")
             for v in out.values())
    return {"ok": ok, "stages": out}
