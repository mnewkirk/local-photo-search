# Maintenance Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make maintenance-sweep results computed on the local replica land on the NAS too, instead of being silently destroyed by the next `sync-replica.sh`, and schedule the sweep so derived data stops going stale.

**Architecture:** Each sweep stage declares a *push mode*. `trigger` stages (cheap, deterministic) are re-run on the NAS over its own data — no payload, correct by construction. `transfer` stages (`stacking` only) ship rows. `excluded` stages can't run on the replica at all. A `maintenance_runs` watermark table (schema v29) on both machines records per-stage last-run times; a 7ms `(COUNT(*), MAX(id))` photo fingerprint gates transfers.

**Tech Stack:** Python 3.11, FastAPI, SQLite (WAL), pytest, plain React UMD (no build step).

**Spec:** `docs/superpowers/specs/2026-07-17-maintenance-push-up-design.md`

## Global Constraints

- `SCHEMA_VERSION` goes 28 → 29. Migrations are idempotent DDL in `_init_schema()`; there is a fast-path `return` at `db.py:277` when the stored version is already `>= SCHEMA_VERSION`, so a new `CREATE TABLE IF NOT EXISTS` plus the version bump *is* the whole migration.
- Every CLI command's `--db` option must carry `envvar="PHOTOSEARCH_DB"`.
- `PhotoDB.conn` uses `sqlite3.Row`; index columns by name (`row["photo_count"]`), never by position.
- Frontend has **no build step**. Use `React.createElement` (aliased `e`), never JSX. Components attach to the `PS` namespace (`frontend/dist/shared.js:27`).
- Do **not** add HTTP retry/backoff to this work. The NAS's shutdown middleware guards only `/api/worker/*` and `/api/photos/*/full` (`web.py:85`); `/api/admin/*` keeps serving during a drain, so a mid-restart NAS surfaces as a plain connection error. Treat it as `unreachable` and let the user retry. (The spec's first draft claimed otherwise; it has been corrected.)
- Timestamps are UTC ISO8601 strings, produced by `datetime.now(timezone.utc).isoformat()`.
- Do not enable `--recluster` or `--dedup-photos` anywhere in this work. Recluster clears `ignored_clusters`; dedup DELETEs photos.
- **Test command.** This worktree has no venv of its own — the interpreter lives in the main checkout. Export this once per shell, then use `"$VENV"/pytest` everywhere:

  ```bash
  export VENV=/Users/mattnewkirk/Documents/Claude/Projects/photo_organization/local-photo-search/venv/bin
  ```

  Run from the worktree root. Verified: this resolves the **worktree's** `photosearch`, not the main checkout's. Call the binary directly; do **not** `source venv/bin/activate`.
- **Baseline:** `"$VENV"/pytest tests/test_maintenance.py -q` → **23 passed** before any change. A task that leaves this red is not done.

## File Structure

| File | Responsibility |
|---|---|
| `photosearch/db.py` | **Modify.** v29 table + two accessor methods. Follows the `log_generation` precedent — all SQL lives here. |
| `photosearch/maintenance_sync.py` | **Create.** The entire sync concern: fingerprint, push-mode taxonomy, payload collection, push orchestration. Keeps `maintenance.py` about *running* stages and this module about *reconciling* them. |
| `photosearch/maintenance.py` | **Modify.** Stamp `maintenance_runs` per completed stage; accept a `stages` subset. |
| `photosearch/admin_api.py` | **Modify.** `maintenance-fingerprint`, `maintenance-apply`, `maintenance-push-status`. |
| `photosearch/web.py` | **Modify.** Replica-mode gating + pre-flight + background push on the existing sweep endpoint. |
| `frontend/dist/shared.js` | **Modify.** `PS.MaintenanceSyncPanel`. |
| `frontend/dist/status.html`, `frontend/dist/admin_maintenance.html` | **Modify.** Render the panel; disable excluded stages in replica mode. |
| `tests/test_maintenance_sync.py` | **Create.** Unit + endpoint coverage. |
| `tests/test_db.py` | **Modify.** v28 → v29 migration test. |
| `CLAUDE.md`, `.claude/skills/photo-search/SKILL.md` | **Modify.** Cron runbook + the replica-mode rule. |

---

### Task 1: Schema v29 — the `maintenance_runs` watermark

**Files:**
- Modify: `photosearch/db.py:82` (version), `photosearch/db.py:586` (DDL, insert before `photo_stacks`)
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: nothing.
- Produces: table `maintenance_runs(stage TEXT PRIMARY KEY, last_run_at TEXT NOT NULL, photo_count INTEGER, photo_max_id INTEGER, applied INTEGER, source TEXT)`; `PhotoDB.record_maintenance_run(stage, last_run_at, photo_count=None, photo_max_id=None, applied=None, source="nas", commit=True) -> None` — `commit=False` leaves the write in the caller's open transaction (Task 6 needs this); `PhotoDB.get_maintenance_runs() -> dict[str, dict]` keyed by stage, each value `{"last_run_at", "photo_count", "photo_max_id", "applied", "source"}`.

- [ ] **Step 1: Write the failing migration test**

Add to `tests/test_db.py`:

```python
def test_v28_db_migrates_to_v29_maintenance_runs(tmp_path):
    """A v28 DB gains maintenance_runs on open, and the version is stamped 29."""
    import sqlite3
    from photosearch.db import PhotoDB, SCHEMA_VERSION

    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_info (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO schema_info (key, value) VALUES ('version', '28')")
    conn.commit()
    conn.close()

    with PhotoDB(path) as db:
        cols = {r["name"] for r in db.conn.execute("PRAGMA table_info(maintenance_runs)")}
        assert cols == {"stage", "last_run_at", "photo_count",
                        "photo_max_id", "applied", "source"}
        version = db.conn.execute(
            "SELECT value FROM schema_info WHERE key = 'version'"
        ).fetchone()["value"]
        assert int(version) == SCHEMA_VERSION == 29


def test_record_and_get_maintenance_runs(db):
    db.record_maintenance_run(
        stage="stacking", last_run_at="2026-07-17T09:00:00+00:00",
        photo_count=10, photo_max_id=99, applied=3, source="replica",
    )
    runs = db.get_maintenance_runs()
    assert runs["stacking"]["source"] == "replica"
    assert runs["stacking"]["applied"] == 3
    assert runs["stacking"]["photo_max_id"] == 99

    # Same stage again -> upsert, not a second row.
    db.record_maintenance_run(
        stage="stacking", last_run_at="2026-07-17T10:00:00+00:00",
        photo_count=11, photo_max_id=100, applied=4, source="nas",
    )
    runs = db.get_maintenance_runs()
    assert len(runs) == 1
    assert runs["stacking"]["last_run_at"] == "2026-07-17T10:00:00+00:00"
    assert runs["stacking"]["source"] == "nas"


def test_record_maintenance_run_commit_false_is_rollback_able(db):
    """commit=False must leave the write in the caller's transaction.

    /api/admin/maintenance-apply stamps the watermark inside the same
    transaction that replaces the stacks; if a rollback left the stamp behind,
    the NAS would claim work it doesn't have.
    """
    db.conn.execute("BEGIN IMMEDIATE")
    db.record_maintenance_run(
        stage="stacking", last_run_at="2026-07-17T09:00:00+00:00",
        photo_count=1, photo_max_id=1, applied=1, source="replica",
        commit=False,
    )
    db.conn.rollback()
    assert db.get_maintenance_runs() == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
"$VENV"/pytest tests/test_db.py -k "maintenance_runs" -v
```

Expected: FAIL — `sqlite3.OperationalError: no such table: maintenance_runs` on the first, `AttributeError: 'PhotoDB' object has no attribute 'record_maintenance_run'` on the second.

- [ ] **Step 3: Add the table DDL**

In `photosearch/db.py`, immediately **before** the `CREATE TABLE IF NOT EXISTS photo_stacks` block (currently line 586):

```python
        # M-sync: per-stage watermark for the maintenance sweep. Lives in the
        # main DB on BOTH machines. A replica re-sync overwrites the replica's
        # copy with the NAS's, which is correct: after a successful push both
        # sides already agree, and after a failed push the sync wipes the local
        # results AND the record of them together, so they stay consistent.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_runs (
                stage            TEXT PRIMARY KEY,
                last_run_at      TEXT NOT NULL,
                photo_count      INTEGER,
                photo_max_id     INTEGER,
                applied          INTEGER,
                source           TEXT
            )
        """)
```

- [ ] **Step 4: Bump the schema version**

In `photosearch/db.py`, line 82:

```python
SCHEMA_VERSION = 29
```

- [ ] **Step 5: Add the accessor methods**

In `photosearch/db.py`, add near `log_generation`:

```python
    def record_maintenance_run(self, stage: str, last_run_at: str,
                               photo_count: Optional[int] = None,
                               photo_max_id: Optional[int] = None,
                               applied: Optional[int] = None,
                               source: str = "nas",
                               commit: bool = True) -> None:
        """Stamp a stage's watermark. Upserts — one row per stage.

        ``commit=False`` leaves the write in the caller's open transaction —
        needed by /api/admin/maintenance-apply, which stamps inside the same
        BEGIN IMMEDIATE that replaces the stacks, so a failure can't leave the
        watermark claiming work that was rolled back.
        """
        self.conn.execute(
            "INSERT INTO maintenance_runs "
            "  (stage, last_run_at, photo_count, photo_max_id, applied, source) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(stage) DO UPDATE SET "
            "  last_run_at=excluded.last_run_at, "
            "  photo_count=excluded.photo_count, "
            "  photo_max_id=excluded.photo_max_id, "
            "  applied=excluded.applied, "
            "  source=excluded.source",
            (stage, last_run_at, photo_count, photo_max_id, applied, source),
        )
        if commit:
            self.conn.commit()

    def get_maintenance_runs(self) -> dict:
        """Every stage watermark, keyed by stage name."""
        rows = self.conn.execute(
            "SELECT stage, last_run_at, photo_count, photo_max_id, applied, source "
            "FROM maintenance_runs"
        ).fetchall()
        return {
            r["stage"]: {
                "last_run_at": r["last_run_at"],
                "photo_count": r["photo_count"],
                "photo_max_id": r["photo_max_id"],
                "applied": r["applied"],
                "source": r["source"],
            }
            for r in rows
        }
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
"$VENV"/pytest tests/test_db.py -k "maintenance_runs" -v
```

Expected: 2 passed.

- [ ] **Step 7: Run the full db suite for regressions**

```bash
"$VENV"/pytest tests/test_db.py -q
```

Expected: all pass. The version bump touches every DB open, so this must be green before moving on.

- [ ] **Step 8: Commit**

```bash
git add photosearch/db.py tests/test_db.py
git commit -m "feat(db): schema v29 — maintenance_runs watermark table"
```

---

### Task 2: Fingerprint + push-mode taxonomy

**Files:**
- Create: `photosearch/maintenance_sync.py`
- Test: `tests/test_maintenance_sync.py`

**Interfaces:**
- Consumes: `PhotoDB.conn` from Task 1.
- Produces: `photo_fingerprint(db) -> {"photo_count": int, "photo_max_id": int | None}`; `push_mode(stage: str) -> "trigger" | "transfer" | "excluded"`; module constants `TRIGGER_STAGES`, `TRANSFER_STAGES`, `EXCLUDED_STAGES` (all `frozenset[str]`); `fingerprints_match(a: dict, b: dict) -> bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_maintenance_sync.py`:

```python
"""Tests for the maintenance sync layer (replica -> NAS push).

Covers the fingerprint, the stage push-mode taxonomy, payload collection,
the timestamp comparison rules, and the NAS-side apply path.
"""

import pytest

from photosearch.maintenance_sync import (
    EXCLUDED_STAGES,
    TRANSFER_STAGES,
    TRIGGER_STAGES,
    fingerprints_match,
    photo_fingerprint,
    push_mode,
)


def test_photo_fingerprint_counts_and_max_id(db):
    fp = photo_fingerprint(db)
    expected_n = db.conn.execute("SELECT COUNT(*) AS n FROM photos").fetchone()["n"]
    expected_max = db.conn.execute("SELECT MAX(id) AS m FROM photos").fetchone()["m"]
    assert fp == {"photo_count": expected_n, "photo_max_id": expected_max}


def test_photo_fingerprint_on_empty_db(tmp_db_path):
    from photosearch.db import PhotoDB
    with PhotoDB(tmp_db_path) as empty:
        assert photo_fingerprint(empty) == {"photo_count": 0, "photo_max_id": None}


def test_fingerprints_match_is_exact():
    a = {"photo_count": 10, "photo_max_id": 99}
    assert fingerprints_match(a, {"photo_count": 10, "photo_max_id": 99})
    assert not fingerprints_match(a, {"photo_count": 11, "photo_max_id": 99})
    assert not fingerprints_match(a, {"photo_count": 10, "photo_max_id": 100})


@pytest.mark.parametrize("stage", sorted(TRIGGER_STAGES))
def test_trigger_stages(stage):
    assert push_mode(stage) == "trigger"


def test_stacking_is_the_only_transfer_stage():
    assert TRANSFER_STAGES == frozenset({"stacking"})
    assert push_mode("stacking") == "transfer"


@pytest.mark.parametrize("stage", ["colors", "dedup_photos", "match_faces",
                                   "recluster", "requeue"])
def test_excluded_stages(stage):
    assert push_mode(stage) == "excluded"


def test_taxonomy_covers_every_sweep_stage_exactly_once():
    """Every stage the sweep can emit must have exactly one push mode.

    Guards against a new stage being added to maintenance.py without a
    reconciliation decision — which would silently mean 'lost on next sync'.
    """
    from photosearch.maintenance import SWEEP_STAGE_ORDER
    known = TRIGGER_STAGES | TRANSFER_STAGES | EXCLUDED_STAGES
    assert set(SWEEP_STAGE_ORDER) - known == set(), "sweep stage with no push mode"
    assert not (TRIGGER_STAGES & TRANSFER_STAGES)
    assert not (TRIGGER_STAGES & EXCLUDED_STAGES)
    assert not (TRANSFER_STAGES & EXCLUDED_STAGES)


def test_push_mode_rejects_unknown_stage():
    with pytest.raises(ValueError, match="unknown maintenance stage"):
        push_mode("not_a_stage")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
"$VENV"/pytest tests/test_maintenance_sync.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'photosearch.maintenance_sync'`.

- [ ] **Step 3: Write the module**

Create `photosearch/maintenance_sync.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
"$VENV"/pytest tests/test_maintenance_sync.py -v
```

Expected: all pass. If `test_taxonomy_covers_every_sweep_stage_exactly_once` fails, a stage name in `maintenance.py:SWEEP_STAGE_ORDER` is missing from the taxonomy — add it to the correct set rather than loosening the test.

- [ ] **Step 5: Commit**

```bash
git add photosearch/maintenance_sync.py tests/test_maintenance_sync.py
git commit -m "feat(sync): photo fingerprint + per-stage push-mode taxonomy"
```

---

### Task 3: The sweep stamps `maintenance_runs` and accepts a stage subset

**Files:**
- Modify: `photosearch/maintenance.py:688-790` (`run_maintenance_sweep`)
- Test: `tests/test_maintenance_sync.py`

**Interfaces:**
- Consumes: `PhotoDB.record_maintenance_run` (Task 1), `photo_fingerprint` (Task 2).
- Produces: `run_maintenance_sweep(db, *, stages=None, source="nas", ...)`. `stages: Optional[Sequence[str]]` filters the plan to those names; `source: str` is recorded on each watermark. Return shape is unchanged: `{"apply": bool, "stages": [{"stage", "would", "applied", "status", "seconds", ...}]}`. Only stages with `status == "done"` are stamped.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_maintenance_sync.py`:

```python
# ---------------------------------------------------------------------------
# The sweep stamps watermarks
# ---------------------------------------------------------------------------

def test_sweep_stamps_watermark_only_for_done_stages(db):
    from photosearch.maintenance import run_maintenance_sweep

    result = run_maintenance_sweep(db, apply=True, do_colors=False,
                                   do_stacking=False, do_match=False,
                                   source="replica")
    runs = db.get_maintenance_runs()
    done = {s["stage"] for s in result["stages"] if s["status"] == "done"}
    skipped = {s["stage"] for s in result["stages"] if s["status"] != "done"}

    assert done, "expected at least one stage to run against the fixture"
    assert done <= set(runs), "every done stage must be stamped"
    assert not (skipped & set(runs)), "skipped stages must not be stamped"
    for stage in done:
        assert runs[stage]["source"] == "replica"
        assert runs[stage]["last_run_at"]


def test_dry_run_stamps_nothing(db):
    from photosearch.maintenance import run_maintenance_sweep

    run_maintenance_sweep(db, apply=False, do_colors=False,
                          do_stacking=False, do_match=False)
    assert db.get_maintenance_runs() == {}


def test_stages_subset_runs_only_those_stages(db):
    from photosearch.maintenance import run_maintenance_sweep

    result = run_maintenance_sweep(
        db, apply=True, stages=["normalize_aesthetics"],
    )
    assert [s["stage"] for s in result["stages"]] == ["normalize_aesthetics"]


def test_unknown_stage_subset_is_rejected(db):
    from photosearch.maintenance import run_maintenance_sweep

    with pytest.raises(ValueError, match="unknown maintenance stage"):
        run_maintenance_sweep(db, apply=True, stages=["bogus_stage"])
```

- [ ] **Step 2: Run to verify they fail**

```bash
"$VENV"/pytest tests/test_maintenance_sync.py -k "sweep or stages_subset or dry_run or unknown_stage" -v
```

Expected: FAIL — `TypeError: run_maintenance_sweep() got an unexpected keyword argument 'source'`.

- [ ] **Step 3: Add the parameters to the signature**

In `photosearch/maintenance.py`, in `run_maintenance_sweep`'s signature (after `min_confidence: float = 0.0,`):

```python
    stages: Optional[Sequence[str]] = None,
    source: str = "nas",
```

Ensure `Sequence` is imported at the top of the file:

```python
from typing import Callable, Optional, Sequence
```

- [ ] **Step 4: Filter the plan and stamp watermarks**

In `photosearch/maintenance.py`, replace the execution loop (currently lines 777-790) with:

```python
    if stages is not None:
        from .maintenance_sync import push_mode
        for name in stages:
            push_mode(name)  # raises ValueError on an unknown stage name
        wanted = set(stages)
        plan = [(name, fn) for name, fn in plan if name in wanted]

    from datetime import datetime, timezone
    from .maintenance_sync import photo_fingerprint

    stage_results = []
    for name, fn in plan:
        check_abort()
        emit({"phase": "sweep", "stage": name, "status": "scanning"})
        t0 = time.monotonic()
        result = fn()
        result["seconds"] = round(time.monotonic() - t0, 2)
        stage_results.append(result)
        # Stamp the watermark only for stages that actually did work. A skipped
        # or preview stage changed nothing, and a cancelled one left partial
        # state — neither is eligible to push.
        if apply and result.get("status") == "done":
            fp = photo_fingerprint(db)
            db.record_maintenance_run(
                stage=name,
                last_run_at=datetime.now(timezone.utc).isoformat(),
                photo_count=fp["photo_count"],
                photo_max_id=fp["photo_max_id"],
                applied=result.get("applied"),
                source=source,
            )
        emit({"phase": "sweep", "stage": name, "status": result.get("status", "done"),
              "would": result.get("would", 0), "applied": result.get("applied", 0),
              "message": result.get("message")})
        logger.info("sweep stage %s: %s", name, result)

    return {"apply": apply, "stages": stage_results}
```

- [ ] **Step 5: Run to verify they pass**

```bash
"$VENV"/pytest tests/test_maintenance_sync.py -v
```

Expected: all pass.

- [ ] **Step 6: Run the existing maintenance suite for regressions**

```bash
"$VENV"/pytest tests/test_maintenance.py -q
```

Expected: all pass — the return shape is unchanged and `source` defaults to `"nas"`.

- [ ] **Step 7: Commit**

```bash
git add photosearch/maintenance.py tests/test_maintenance_sync.py
git commit -m "feat(maint): stamp maintenance_runs per done stage; add stages subset"
```

---

### Task 4: Payload collection + push eligibility

**Files:**
- Modify: `photosearch/maintenance_sync.py`
- Test: `tests/test_maintenance_sync.py`

**Interfaces:**
- Consumes: `photo_fingerprint`, `push_mode` (Task 2); `PhotoDB.get_maintenance_runs` (Task 1).
- Produces: `eligible_stages(stage_results: list[dict]) -> list[str]` (names with `status == "done"` whose mode is not `excluded`); `collect_stacking_rows(db) -> list[dict]` where each item is `{"members": [{"photo_id": int, "is_top": int}, ...]}`; `collect_payload(db, stage_results) -> dict` shaped `{"fingerprint": {...}, "stages": {name: {"mode": str, "last_run_at": str}}, "stacking": [...] | None}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_maintenance_sync.py`:

```python
# ---------------------------------------------------------------------------
# Payload collection
# ---------------------------------------------------------------------------

from photosearch.maintenance_sync import (  # noqa: E402
    collect_payload,
    collect_stacking_rows,
    eligible_stages,
)


def test_eligible_stages_only_includes_done_and_pushable():
    results = [
        {"stage": "normalize_aesthetics", "status": "done"},
        {"stage": "geocode", "status": "skipped"},
        {"stage": "stacking", "status": "done"},
        {"stage": "colors", "status": "done"},      # excluded mode
        {"stage": "infer", "status": "preview"},
    ]
    assert eligible_stages(results) == ["normalize_aesthetics", "stacking"]


def test_cancelled_stage_is_not_eligible():
    """A half-finished stacking run must never ship."""
    results = [{"stage": "stacking", "status": "cancelled"}]
    assert eligible_stages(results) == []


def _make_stack(db, photo_ids, top_index=0):
    cur = db.conn.execute("INSERT INTO photo_stacks DEFAULT VALUES")
    stack_id = cur.lastrowid
    for i, pid in enumerate(photo_ids):
        db.conn.execute(
            "INSERT INTO stack_members (stack_id, photo_id, is_top) VALUES (?, ?, ?)",
            (stack_id, pid, 1 if i == top_index else 0),
        )
    db.conn.commit()
    return stack_id


def test_collect_stacking_rows_groups_members_by_stack(db):
    pids = [r["id"] for r in db.conn.execute(
        "SELECT id FROM photos ORDER BY id LIMIT 3")]
    _make_stack(db, pids[:2], top_index=1)

    rows = collect_stacking_rows(db)
    assert len(rows) == 1
    members = rows[0]["members"]
    assert {m["photo_id"] for m in members} == set(pids[:2])
    assert sum(m["is_top"] for m in members) == 1


def test_collect_payload_includes_stacking_only_when_stacking_ran(db):
    db.record_maintenance_run(
        stage="stacking", last_run_at="2026-07-17T09:00:00+00:00",
        photo_count=1, photo_max_id=1, applied=1, source="replica")
    db.record_maintenance_run(
        stage="normalize_aesthetics", last_run_at="2026-07-17T09:00:01+00:00",
        photo_count=1, photo_max_id=1, applied=1, source="replica")

    with_stacking = collect_payload(db, [
        {"stage": "stacking", "status": "done"},
        {"stage": "normalize_aesthetics", "status": "done"},
    ])
    assert with_stacking["stacking"] is not None
    assert with_stacking["stages"]["stacking"]["mode"] == "transfer"
    assert with_stacking["stages"]["normalize_aesthetics"]["mode"] == "trigger"
    assert with_stacking["fingerprint"] == photo_fingerprint(db)

    without = collect_payload(db, [{"stage": "normalize_aesthetics", "status": "done"}])
    assert without["stacking"] is None
    assert "stacking" not in without["stages"]
```

- [ ] **Step 2: Run to verify they fail**

```bash
"$VENV"/pytest tests/test_maintenance_sync.py -k "eligible or collect or cancelled" -v
```

Expected: FAIL — `ImportError: cannot import name 'collect_payload'`.

- [ ] **Step 3: Implement**

Append to `photosearch/maintenance_sync.py`:

```python
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
```

- [ ] **Step 4: Run to verify they pass**

```bash
"$VENV"/pytest tests/test_maintenance_sync.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add photosearch/maintenance_sync.py tests/test_maintenance_sync.py
git commit -m "feat(sync): payload collection + push eligibility rules"
```

---

### Task 5: `GET /api/admin/maintenance-fingerprint`

**Files:**
- Modify: `photosearch/admin_api.py` (add after `admin_replica_status`, which ends around line 460)
- Test: `tests/test_maintenance_sync.py`

**Interfaces:**
- Consumes: `photo_fingerprint` (Task 2), `PhotoDB.get_maintenance_runs` (Task 1).
- Produces: `GET /api/admin/maintenance-fingerprint` → `{"photo_count": int, "photo_max_id": int | None, "stages": {name: {"last_run_at", "source", "applied"}}, "replica_mode": bool}`. Runs on both machines.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_maintenance_sync.py`:

```python
# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def test_fingerprint_endpoint_reports_index_and_stages(client, db):
    db.record_maintenance_run(
        stage="stacking", last_run_at="2026-07-17T09:00:00+00:00",
        photo_count=3, photo_max_id=42, applied=2, source="replica")

    r = client.get("/api/admin/maintenance-fingerprint")
    assert r.status_code == 200
    body = r.json()
    expected = photo_fingerprint(db)
    assert body["photo_count"] == expected["photo_count"]
    assert body["photo_max_id"] == expected["photo_max_id"]
    assert body["stages"]["stacking"]["source"] == "replica"
    assert body["stages"]["stacking"]["last_run_at"] == "2026-07-17T09:00:00+00:00"
    assert body["replica_mode"] is False
```

- [ ] **Step 2: Run to verify it fails**

```bash
"$VENV"/pytest tests/test_maintenance_sync.py -k fingerprint_endpoint -v
```

Expected: FAIL — `assert 404 == 200`.

- [ ] **Step 3: Implement the endpoint**

In `photosearch/admin_api.py`, add after `admin_replica_status`:

```python
@router.get("/maintenance-fingerprint")
def admin_maintenance_fingerprint():
    """Photo-index fingerprint + per-stage watermarks. Cheap — two indexed
    queries (~7ms on 150k photos), unlike /api/stats' full COUNT scans.

    Serves two consumers: the replica's pre-flight guard before a push, and the
    NAS-vs-replica drift panel on /status.
    """
    from .db import PhotoDB
    from .maintenance_sync import photo_fingerprint

    db_path = os.environ.get("PHOTOSEARCH_DB", "photo_index.db")
    with PhotoDB(db_path) as db:
        fp = photo_fingerprint(db)
        runs = db.get_maintenance_runs()

    return {
        "photo_count": fp["photo_count"],
        "photo_max_id": fp["photo_max_id"],
        "stages": {
            name: {
                "last_run_at": r["last_run_at"],
                "source": r["source"],
                "applied": r["applied"],
            }
            for name, r in runs.items()
        },
        "replica_mode": bool((os.environ.get("PHOTOSEARCH_NAS_URL") or "").strip()),
    }
```

- [ ] **Step 4: Run to verify it passes**

```bash
"$VENV"/pytest tests/test_maintenance_sync.py -k fingerprint_endpoint -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add photosearch/admin_api.py tests/test_maintenance_sync.py
git commit -m "feat(api): GET /api/admin/maintenance-fingerprint"
```

---

### Task 6: `POST /api/admin/maintenance-apply` — the NAS-side transfer receiver

**Files:**
- Modify: `photosearch/admin_api.py`
- Test: `tests/test_maintenance_sync.py`

**Interfaces:**
- Consumes: `photo_fingerprint`, `fingerprints_match`, `push_mode` (Task 2); `PhotoDB.record_maintenance_run`, `get_maintenance_runs` (Task 1).
- Produces: `POST /api/admin/maintenance-apply`. Body = the `collect_payload` dict (Task 4). Returns `{"applied": {name: {"status": "applied" | "skipped", "reason"?: str, "stacks"?: int}}}`. Returns **409** with `{"detail": {"error": "fingerprint_mismatch", "local": {...}, "remote": {...}}}` when fingerprints differ.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_maintenance_sync.py`:

```python
def _payload_for(db, *, last_run_at, stacking=None):
    fp = photo_fingerprint(db)
    stages = {"stacking": {"mode": "transfer", "last_run_at": last_run_at}}
    return {"fingerprint": fp, "stages": stages, "stacking": stacking or []}


def test_apply_rejects_fingerprint_mismatch(client, db):
    body = _payload_for(db, last_run_at="2026-07-17T09:00:00+00:00")
    body["fingerprint"]["photo_count"] += 1

    r = client.post("/api/admin/maintenance-apply", json=body)
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "fingerprint_mismatch"


def test_apply_replaces_stacks_and_stamps_watermark(client, db):
    pids = [r["id"] for r in db.conn.execute(
        "SELECT id FROM photos ORDER BY id LIMIT 3")]
    # Pre-existing stack that must be REPLACED, not merged.
    _make_stack(db, pids[:2])

    body = _payload_for(
        db, last_run_at="2026-07-17T09:00:00+00:00",
        stacking=[{"members": [{"photo_id": pids[2], "is_top": 1}]}],
    )
    r = client.post("/api/admin/maintenance-apply", json=body)
    assert r.status_code == 200
    assert r.json()["applied"]["stacking"]["status"] == "applied"

    rows = db.conn.execute(
        "SELECT photo_id, is_top FROM stack_members").fetchall()
    assert [r["photo_id"] for r in rows] == [pids[2]]
    assert db.conn.execute(
        "SELECT COUNT(*) AS n FROM photo_stacks").fetchone()["n"] == 1

    runs = db.get_maintenance_runs()
    assert runs["stacking"]["source"] == "replica"
    assert runs["stacking"]["last_run_at"] == "2026-07-17T09:00:00+00:00"


def test_apply_skips_stage_when_nas_is_fresher(client, db):
    db.record_maintenance_run(
        stage="stacking", last_run_at="2026-07-17T12:00:00+00:00",
        photo_count=1, photo_max_id=1, applied=1, source="nas")
    pids = [r["id"] for r in db.conn.execute("SELECT id FROM photos LIMIT 1")]

    body = _payload_for(
        db, last_run_at="2026-07-17T09:00:00+00:00",  # older than the NAS
        stacking=[{"members": [{"photo_id": pids[0], "is_top": 1}]}],
    )
    r = client.post("/api/admin/maintenance-apply", json=body)
    assert r.status_code == 200
    assert r.json()["applied"]["stacking"]["status"] == "skipped"
    assert r.json()["applied"]["stacking"]["reason"] == "nas_fresher"
    # Watermark untouched.
    assert db.get_maintenance_runs()["stacking"]["source"] == "nas"


def test_apply_skips_stage_when_timestamps_are_equal(client, db):
    same = "2026-07-17T09:00:00+00:00"
    db.record_maintenance_run(
        stage="stacking", last_run_at=same,
        photo_count=1, photo_max_id=1, applied=1, source="nas")

    body = _payload_for(db, last_run_at=same, stacking=[])
    r = client.post("/api/admin/maintenance-apply", json=body)
    assert r.json()["applied"]["stacking"]["status"] == "skipped"
```

- [ ] **Step 2: Run to verify they fail**

```bash
"$VENV"/pytest tests/test_maintenance_sync.py -k "apply_" -v
```

Expected: FAIL — `assert 404 == 409`.

- [ ] **Step 3: Implement the endpoint**

In `photosearch/admin_api.py`, add after `admin_maintenance_fingerprint`:

```python
@router.post("/maintenance-apply")
def admin_maintenance_apply(payload: dict):
    """Receive replica-computed maintenance results (the transfer path).

    Only 'transfer' stages arrive here; 'trigger' stages are a separate call to
    this host's own maintenance-sweep. Applies in ONE transaction so a failure
    can't leave half-replaced stacks.
    """
    from .db import PhotoDB
    from .maintenance_sync import fingerprints_match, photo_fingerprint, push_mode

    db_path = os.environ.get("PHOTOSEARCH_DB", "photo_index.db")
    remote_fp = payload.get("fingerprint") or {}
    stages = payload.get("stages") or {}
    applied = {}

    with PhotoDB(db_path) as db:
        local_fp = photo_fingerprint(db)
        if not fingerprints_match(local_fp, remote_fp):
            # The photo index moved under the replica (e.g. the nightly ingest
            # landed mid-run). Its results describe a different library, so
            # reject the WHOLE request rather than applying part of it.
            raise HTTPException(status_code=409, detail={
                "error": "fingerprint_mismatch",
                "local": local_fp,
                "remote": remote_fp,
            })

        runs = db.get_maintenance_runs()
        try:
            db.conn.execute("BEGIN IMMEDIATE")
            for name, info in stages.items():
                if push_mode(name) != "transfer":
                    applied[name] = {"status": "skipped", "reason": "not_transfer_mode"}
                    continue

                incoming = info.get("last_run_at") or ""
                existing = (runs.get(name) or {}).get("last_run_at") or ""
                if existing and incoming <= existing:
                    # ISO8601 UTC strings sort lexicographically == chronologically.
                    applied[name] = {"status": "skipped", "reason": "nas_fresher"}
                    continue

                if name == "stacking":
                    rows = payload.get("stacking") or []
                    # Full replace: stacking is a full re-detect, so replace is
                    # its natural semantics. stack_members cascades on delete.
                    db.conn.execute("DELETE FROM photo_stacks")
                    for stack in rows:
                        cur = db.conn.execute("INSERT INTO photo_stacks DEFAULT VALUES")
                        stack_id = cur.lastrowid
                        for member in stack.get("members") or []:
                            db.conn.execute(
                                "INSERT INTO stack_members (stack_id, photo_id, is_top) "
                                "VALUES (?, ?, ?)",
                                (stack_id, member["photo_id"], int(member.get("is_top", 0))),
                            )
                    applied[name] = {"status": "applied", "stacks": len(rows)}

                # commit=False: stamp inside the SAME transaction that replaced
                # the stacks, so a rollback can't leave a watermark claiming
                # work that didn't land.
                db.record_maintenance_run(
                    stage=name,
                    last_run_at=incoming,
                    photo_count=local_fp["photo_count"],
                    photo_max_id=local_fp["photo_max_id"],
                    applied=applied[name].get("stacks"),
                    source="replica",
                    commit=False,
                )
            db.conn.commit()
        except Exception:
            db.conn.rollback()
            raise

    return {"applied": applied}
```

- [ ] **Step 4: Run to verify they pass**

```bash
"$VENV"/pytest tests/test_maintenance_sync.py -k "apply_" -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add photosearch/admin_api.py tests/test_maintenance_sync.py
git commit -m "feat(api): POST /api/admin/maintenance-apply — transactional stack transfer"
```

---

### Task 7: `push_to_nas` orchestration

**Files:**
- Modify: `photosearch/maintenance_sync.py`
- Test: `tests/test_maintenance_sync.py`

**Interfaces:**
- Consumes: `collect_payload` (Task 4); the endpoints from Tasks 5-6.
- Produces: `push_to_nas(db, nas_url: str, stage_results: list, *, timeout: int = 900) -> dict` shaped `{"ok": bool, "stages": {name: {"status": str, "reason"?: str}}, "error"?: str}`. Transfer runs first (fast, bounded), triggers second (geocode is slow on the N100).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_maintenance_sync.py`:

```python
# ---------------------------------------------------------------------------
# push_to_nas orchestration (NAS calls mocked)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or 'event: message\ndata: {"type": "done"}\n\n'

    def json(self):
        return self._payload


def test_push_sends_transfer_before_trigger(db, monkeypatch):
    from photosearch import maintenance_sync

    calls = []

    def fake_post(url, json=None, timeout=None, **kw):
        calls.append(url)
        if url.endswith("/maintenance-apply"):
            return _FakeResponse(200, {"applied": {"stacking": {"status": "applied"}}})
        return _FakeResponse(200, {})

    monkeypatch.setattr(maintenance_sync.requests, "post", fake_post)

    db.record_maintenance_run(
        stage="stacking", last_run_at="2026-07-17T09:00:00+00:00",
        photo_count=1, photo_max_id=1, applied=1, source="replica")
    db.record_maintenance_run(
        stage="normalize_aesthetics", last_run_at="2026-07-17T09:00:01+00:00",
        photo_count=1, photo_max_id=1, applied=1, source="replica")

    result = maintenance_sync.push_to_nas(db, "http://nas:8000", [
        {"stage": "stacking", "status": "done"},
        {"stage": "normalize_aesthetics", "status": "done"},
    ])

    assert result["ok"] is True
    assert calls[0].endswith("/maintenance-apply"), "transfer must go first"
    assert calls[1].endswith("/maintenance-sweep"), "triggers go second"
    assert result["stages"]["stacking"]["status"] == "applied"
    assert result["stages"]["normalize_aesthetics"]["status"] == "triggered"


def test_push_reports_fingerprint_mismatch(db, monkeypatch):
    from photosearch import maintenance_sync

    def fake_post(url, json=None, timeout=None, **kw):
        return _FakeResponse(409, {"detail": {"error": "fingerprint_mismatch"}})

    monkeypatch.setattr(maintenance_sync.requests, "post", fake_post)
    db.record_maintenance_run(
        stage="stacking", last_run_at="2026-07-17T09:00:00+00:00",
        photo_count=1, photo_max_id=1, applied=1, source="replica")

    result = maintenance_sync.push_to_nas(db, "http://nas:8000",
                                          [{"stage": "stacking", "status": "done"}])
    assert result["ok"] is False
    assert result["error"] == "fingerprint_mismatch"


def test_push_with_nothing_eligible_is_a_noop(db, monkeypatch):
    from photosearch import maintenance_sync

    def boom(*a, **kw):
        raise AssertionError("must not call the NAS with nothing to push")

    monkeypatch.setattr(maintenance_sync.requests, "post", boom)
    result = maintenance_sync.push_to_nas(db, "http://nas:8000",
                                          [{"stage": "geocode", "status": "skipped"}])
    assert result["ok"] is True
    assert result["stages"] == {}
```

- [ ] **Step 2: Run to verify they fail**

```bash
"$VENV"/pytest tests/test_maintenance_sync.py -k push_ -v
```

Expected: FAIL — `AttributeError: module 'photosearch.maintenance_sync' has no attribute 'requests'`.

- [ ] **Step 3: Implement**

At the top of `photosearch/maintenance_sync.py`, add to the imports:

```python
import requests
```

Append to `photosearch/maintenance_sync.py`:

```python
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
        except Exception as e:
            logger.warning("maintenance push (transfer) failed: %s", e)
            return {"ok": False, "error": "unreachable", "stages": out}
        if r.status_code == 409:
            return {"ok": False, "error": "fingerprint_mismatch", "stages": out}
        if r.status_code != 200:
            return {"ok": False, "error": f"http_{r.status_code}", "stages": out}
        out.update(r.json().get("applied") or {})

    # --- trigger ----------------------------------------------------------
    trigger = sorted(n for n, i in stages.items() if i["mode"] == "trigger")
    if trigger:
        try:
            # The NAS sweep endpoint is SSE; requests reads the finite stream to
            # completion and hands back the whole body.
            r = requests.post(f"{nas_url}/api/admin/maintenance-sweep",
                              json={"apply": True, "stages": trigger},
                              timeout=timeout)
        except Exception as e:
            logger.warning("maintenance push (trigger) failed: %s", e)
            for name in trigger:
                out[name] = {"status": "failed", "reason": "unreachable"}
            return {"ok": False, "error": "unreachable", "stages": out}
        if r.status_code != 200 or '"type": "fatal"' in (r.text or ""):
            for name in trigger:
                out[name] = {"status": "failed", "reason": f"http_{r.status_code}"}
            return {"ok": False, "error": f"http_{r.status_code}", "stages": out}
        for name in trigger:
            out[name] = {"status": "triggered"}

    ok = all(v.get("status") in ("applied", "triggered", "skipped")
             for v in out.values())
    return {"ok": ok, "stages": out}
```

- [ ] **Step 4: Run to verify they pass**

```bash
"$VENV"/pytest tests/test_maintenance_sync.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add photosearch/maintenance_sync.py tests/test_maintenance_sync.py
git commit -m "feat(sync): push_to_nas — transfer then trigger, per-stage results"
```

---

### Task 8: Replica-mode gating on the sweep endpoint

**Files:**
- Modify: `photosearch/web.py:4121-4190` (`api_maintenance_sweep`)
- Test: `tests/test_maintenance_sync.py`

**Interfaces:**
- Consumes: `EXCLUDED_STAGES` (Task 2).
- Produces: `api_maintenance_sweep` returns **400** `{"detail": {"error": "excluded_stage_in_replica_mode", "stages": [...]}}` when replica mode requests an excluded stage with `apply=true`, and **503** `{"detail": {"error": "nas_unreachable"}}` when replica mode + `apply=true` + the NAS fingerprint is unreachable. (`_push_status` / `_push_lock` are **Task 9's**, not this task's — an earlier draft of this line wrongly claimed them here.)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_maintenance_sync.py`:

```python
# ---------------------------------------------------------------------------
# Replica-mode gating
# ---------------------------------------------------------------------------

def test_apply_blocked_in_replica_mode_when_nas_unreachable(client, monkeypatch):
    """THE gate: never write results that the next sync is guaranteed to wipe."""
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://unreachable:8000")

    from photosearch import maintenance_sync

    def boom(*a, **kw):
        raise OSError("connection refused")

    monkeypatch.setattr(maintenance_sync.requests, "get", boom)

    r = client.post("/api/admin/maintenance-sweep", json={"apply": True})
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "nas_unreachable"


def test_dry_run_allowed_in_replica_mode_without_nas(client, monkeypatch):
    """Dry-runs write nothing, so there's nothing to lose."""
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://unreachable:8000")
    r = client.post("/api/admin/maintenance-sweep", json={"apply": False})
    assert r.status_code == 200


def test_excluded_stage_rejected_in_replica_mode(client, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://nas:8000")
    r = client.post("/api/admin/maintenance-sweep",
                    json={"apply": True, "do_colors": True})
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "excluded_stage_in_replica_mode"
    assert "colors" in r.json()["detail"]["stages"]


def test_excluded_stage_allowed_on_the_nas(client, monkeypatch):
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    r = client.post("/api/admin/maintenance-sweep",
                    json={"apply": True, "do_colors": True})
    assert r.status_code == 200
```

- [ ] **Step 2: Run to verify they fail**

```bash
"$VENV"/pytest tests/test_maintenance_sync.py -k "replica_mode or dry_run_allowed or excluded_stage" -v
```

Expected: FAIL — `assert 200 == 503`.

- [ ] **Step 3: Add the gate**

In `photosearch/web.py`, inside `api_maintenance_sweep`, immediately **after** the `do_requeue = bool(data.get("do_requeue", False))` line and **before** the sweep thread is constructed:

```python
    # --- replica-mode gating ------------------------------------------------
    # A sweep on the replica writes to photo_index.db.local, which the next
    # sync-replica.sh replaces wholesale. So an apply here is only legitimate
    # if we can hand the results to the NAS afterwards.
    from .maintenance_sync import EXCLUDED_STAGES
    _nas_url = (os.environ.get("PHOTOSEARCH_NAS_URL") or "").rstrip("/")
    if _nas_url and apply:
        requested_excluded = [
            name for name, on in (
                ("colors", do_colors),
                ("match_faces", do_match),
                ("recluster", do_recluster),
                ("dedup_photos", do_dedup),
                ("requeue", do_requeue),
            ) if on and name in EXCLUDED_STAGES
        ]
        if requested_excluded:
            # These can't produce a correct result here (colors needs pixels the
            # replica doesn't have) or must not cross machines (dedup DELETEs).
            raise HTTPException(status_code=400, detail={
                "error": "excluded_stage_in_replica_mode",
                "stages": requested_excluded,
                "message": "These stages must run on the NAS. See the "
                           "maintenance-sync spec for why.",
            })
        from .maintenance_sync import fetch_nas_fingerprint
        try:
            fetch_nas_fingerprint(_nas_url)
        except Exception as e:
            raise HTTPException(status_code=503, detail={
                "error": "nas_unreachable",
                "message": (
                    "Refusing to apply: results computed here would be lost on "
                    f"the next replica sync and the NAS is unreachable ({e})."
                ),
            })
```

Going through `fetch_nas_fingerprint` (Task 7) rather than a local `import requests` is deliberate: it gives the tests a single patchable seam. A local import inside the endpoint would make `monkeypatch.setattr(maintenance_sync.requests, "get", ...)` silently ineffective, and the test would pass or fail for the wrong reason.

Confirm `HTTPException` is imported in `web.py`; if not, add it to the existing `from fastapi import ...` line.

- [ ] **Step 4: Run to verify they pass**

```bash
"$VENV"/pytest tests/test_maintenance_sync.py -k "replica_mode or dry_run_allowed or excluded_stage" -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add photosearch/web.py tests/test_maintenance_sync.py
git commit -m "feat(web): refuse a replica apply that the next sync would wipe"
```

---

### Task 9: Pre-flight auto-sync + background push + push-status endpoint

**Files:**
- Modify: `photosearch/web.py` (`api_maintenance_sweep`), `photosearch/admin_api.py`
- Test: `tests/test_maintenance_sync.py`

**Interfaces:**
- Consumes: `push_to_nas` (Task 7), `photo_fingerprint`/`fingerprints_match` (Task 2).
- Produces: `GET /api/admin/maintenance-push-status` → `{"state": "idle" | "running" | "ok" | "failed", "stages": {...}, "error": str | None, "finished_at": str | None}`. The sweep SSE gains `{"type": "push", "state": ...}` events.

Pre-flight ordering is load-bearing: the fingerprint check happens **before** compute, because a sync replaces the whole local DB and would destroy results computed first.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_maintenance_sync.py`:

```python
def test_push_status_starts_idle(client):
    r = client.get("/api/admin/maintenance-push-status")
    assert r.status_code == 200
    assert r.json()["state"] == "idle"


def test_push_status_reflects_last_push(client):
    from photosearch import web

    with web._push_lock:
        web._push_status.update({
            "state": "failed", "error": "fingerprint_mismatch",
            "stages": {"stacking": {"status": "failed"}},
            "finished_at": "2026-07-17T09:00:00+00:00",
        })
    try:
        body = client.get("/api/admin/maintenance-push-status").json()
        assert body["state"] == "failed"
        assert body["error"] == "fingerprint_mismatch"
    finally:
        with web._push_lock:
            web._push_status.update({"state": "idle", "error": None,
                                     "stages": {}, "finished_at": None})
```

- [ ] **Step 2: Run to verify they fail**

```bash
"$VENV"/pytest tests/test_maintenance_sync.py -k push_status -v
```

Expected: FAIL — `assert 404 == 200`.

- [ ] **Step 3: Add the push-status module state**

In `photosearch/web.py`, near the other module-level state (after `_books_db_path`, around line 126):

```python
# Maintenance push state. The push outlives its request (it runs detached so a
# closed tab can't kill it), so the UI polls this to drive the Retry affordance.
_push_lock = threading.Lock()
_push_status: dict = {"state": "idle", "stages": {}, "error": None,
                      "finished_at": None}
```

Confirm `threading` is imported at the top of `web.py`; add it if not.

- [ ] **Step 4: Add the status endpoint**

In `photosearch/admin_api.py`, add after `admin_maintenance_apply`:

```python
@router.get("/maintenance-push-status")
def admin_maintenance_push_status():
    """Last/current replica->NAS push. Backs the Retry affordance."""
    from . import web
    with web._push_lock:
        return dict(web._push_status)
```

- [ ] **Step 5: Run to verify they pass**

```bash
"$VENV"/pytest tests/test_maintenance_sync.py -k push_status -v
```

Expected: 2 passed.

- [ ] **Step 6: Wire pre-flight + background push into the sweep**

In `photosearch/web.py`, inside `api_maintenance_sweep`'s `run()`, replace the `with _get_db() as db:` block that calls `run_maintenance_sweep` so it (a) pre-flights, (b) passes `source`, (c) pushes at the end:

```python
            # Pre-flight BEFORE compute: a sync replaces the whole local DB, so
            # discovering drift after a local stacking run would destroy the very
            # results we intend to push.
            if nas_url_now and apply:
                from .maintenance_sync import (
                    fetch_nas_fingerprint, fingerprints_match, photo_fingerprint,
                )
                remote = fetch_nas_fingerprint(nas_url_now)
                with _get_db() as db:
                    local = photo_fingerprint(db)
                if not fingerprints_match(local, remote):
                    _emit({"type": "progress", "phase": "preflight",
                           "status": "syncing",
                           "message": "Photo index drifted — syncing from the NAS "
                                      "before computing."})
                    from .admin_api import run_replica_sync_blocking
                    run_replica_sync_blocking()
                    with _get_db() as db:
                        local = photo_fingerprint(db)
                    remote = fetch_nas_fingerprint(nas_url_now)
                    if not fingerprints_match(local, remote):
                        _emit({"type": "fatal",
                               "message": "Sync did not reconcile the photo index; "
                                          "refusing to compute against a stale DB."})
                        return

            with _get_db() as db:
                result = run_maintenance_sweep(
                    db,
                    apply=apply,
                    do_colors=do_colors,
                    do_stacking=do_stacking,
                    do_match=do_match,
                    do_recluster=do_recluster,
                    do_dedup=do_dedup,
                    do_requeue=do_requeue,
                    force_normalize_aesthetics=force_normalize_aesthetics,
                    force_normalize_subject_aesthetics=force_normalize_subject_aesthetics,
                    requeue_passes=tuple(requeue_passes) if requeue_passes else None,
                    window_minutes=window_minutes,
                    max_drift_km=max_drift_km,
                    min_confidence=min_confidence,
                    on_progress=_on_progress,
                    should_abort=_should_abort,
                    source="replica" if nas_url_now else "nas",
                )
```

Then, immediately **after** the existing `_emit({"type": "done", ...})` call, add the push. It is deliberately last — one push after ALL stages, never between them:

```python
            if nas_url_now and apply:
                _start_push(result["stages"])
```

- [ ] **Step 7: Add the detached push helper**

In `photosearch/web.py`, add at module level near `_push_status`:

```python
def _start_push(stage_results: list) -> None:
    """Reconcile a finished local sweep to the NAS, detached.

    Non-blocking with respect to the sweep: its results are already reported.
    Detached from the request so closing the tab can't kill an in-flight push.
    """
    nas_url = (os.environ.get("PHOTOSEARCH_NAS_URL") or "").rstrip("/")
    if not nas_url:
        return

    def _run():
        from datetime import datetime, timezone
        from .maintenance_sync import push_to_nas
        with _push_lock:
            _push_status.update({"state": "running", "stages": {}, "error": None,
                                 "finished_at": None})
        try:
            with _get_db() as db:
                res = push_to_nas(db, nas_url, stage_results)
        except Exception as e:  # never let the thread die silently
            logger.exception("maintenance push crashed")
            res = {"ok": False, "error": str(e), "stages": {}}
        with _push_lock:
            _push_status.update({
                "state": "ok" if res.get("ok") else "failed",
                "stages": res.get("stages") or {},
                "error": res.get("error"),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })

    threading.Thread(target=_run, name="maintenance-push", daemon=True).start()
```

- [ ] **Step 8: Extract the blocking replica sync**

In `photosearch/admin_api.py`, find the existing `replica-sync` endpoint. Extract its subprocess invocation into a reusable blocking helper, and have the endpoint call it:

```python
def run_replica_sync_blocking(timeout: int = 1800) -> None:
    """Run sync-replica.sh to completion. Raises CalledProcessError on failure.

    Shared by the /api/admin/replica-sync SSE endpoint and the maintenance
    pre-flight, which must sync BEFORE computing.
    """
    subprocess.run(
        ["./sync-replica.sh"],
        cwd=os.environ.get("PHOTOSEARCH_REPO_DIR", "."),
        check=True, capture_output=True, timeout=timeout,
    )
```

- [ ] **Step 9: Run the full suite**

```bash
"$VENV"/pytest tests/test_maintenance_sync.py tests/test_maintenance.py -q
```

Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add photosearch/web.py photosearch/admin_api.py tests/test_maintenance_sync.py
git commit -m "feat(web): pre-flight auto-sync + detached post-sweep push to the NAS"
```

---

### Task 10: `PS.MaintenanceSyncPanel` — make drift visible

**Files:**
- Modify: `frontend/dist/shared.js`, `frontend/dist/status.html`, `frontend/dist/admin_maintenance.html`
- Test: manual (the frontend has no build step and no JS test harness)

**Interfaces:**
- Consumes: `GET /api/admin/maintenance-fingerprint` (Task 5), `GET /api/admin/maintenance-push-status` (Task 9).
- Produces: `PS.MaintenanceSyncPanel({compact: bool})`.

- [ ] **Step 1: Add the component**

In `frontend/dist/shared.js`, add alongside the other `PS.*` components (the file's `e` alias for `React.createElement` and the `PS` namespace are established at line 27):

```javascript
  // Per-stage NAS-vs-replica drift. "Replica ahead" is the state that matters:
  // it means locally-computed work will be destroyed by the next sync.
  PS.MaintenanceSyncPanel = function MaintenanceSyncPanel(props) {
    var compact = props && props.compact;
    var st = React.useState(null), data = st[0], setData = st[1];
    var ps = React.useState(null), push = ps[0], setPush = ps[1];

    var load = React.useCallback(function () {
      var local = fetch('/api/admin/maintenance-fingerprint').then(function (r) { return r.json(); });
      local.then(function (l) {
        if (!l.replica_mode) { setData({ replica_mode: false }); return; }
        fetch('/api/admin/maintenance-nas-fingerprint')
          .then(function (r) { return r.json(); })
          .then(function (n) { setData({ replica_mode: true, local: l, nas: n }); })
          .catch(function () { setData({ replica_mode: true, local: l, nas: null }); });
      });
      fetch('/api/admin/maintenance-push-status')
        .then(function (r) { return r.json(); }).then(setPush).catch(function () {});
    }, []);

    React.useEffect(function () { load(); }, [load]);

    if (!data || !data.replica_mode) return null;
    var nas = data.nas, local = data.local;
    if (!nas) return e('div', { className: 'deploy-meta' }, 'NAS unreachable — drift unknown.');

    var drift = local.photo_count !== nas.photo_count || local.photo_max_id !== nas.photo_max_id;
    var names = Object.keys(local.stages).concat(Object.keys(nas.stages))
      .filter(function (v, i, a) { return a.indexOf(v) === i; }).sort();

    var rows = names.map(function (name) {
      var l = local.stages[name], n = nas.stages[name];
      var state, cls;
      if (!l && !n) { state = 'Never run'; cls = 'ms-none'; }
      else if (l && (!n || l.last_run_at > n.last_run_at)) {
        state = 'Replica ahead — unpushed'; cls = 'ms-warn';
      } else if (n && (!l || n.last_run_at > l.last_run_at)) {
        state = 'NAS ahead'; cls = 'ms-info';
      } else { state = 'In sync'; cls = 'ms-ok'; }
      return e('tr', { key: name, className: cls },
        e('td', null, name),
        e('td', null, n ? n.last_run_at : '—'),
        e('td', null, l ? l.last_run_at : '—'),
        e('td', null, state));
    });

    var unpushed = names.filter(function (name) {
      var l = local.stages[name], n = nas.stages[name];
      return l && (!n || l.last_run_at > n.last_run_at);
    });

    if (compact) {
      return e('div', { className: 'deploy-meta' },
        drift ? e('span', { className: 'ms-warn' }, 'Photo index drift — sync needed. ') : null,
        unpushed.length
          ? e('a', { href: '/admin_maintenance' },
              unpushed.length + ' stage(s) unpushed — will be lost on next sync →')
          : e('span', { className: 'ms-ok' }, 'Maintenance in sync'));
    }

    return e('div', { className: 'card' },
      e('h3', null, 'Maintenance: NAS vs replica'),
      drift ? e('div', { className: 'ms-warn' },
        'Photo index drift: NAS ' + nas.photo_count + ' photos / max id '
        + nas.photo_max_id + ', replica ' + local.photo_count + ' / '
        + local.photo_max_id + '. A push will auto-sync first.') : null,
      push && push.state === 'failed'
        ? e('div', { className: 'ms-warn' }, 'Last push failed: ' + (push.error || 'unknown'))
        : null,
      e('table', { className: 'ms-table' },
        e('thead', null, e('tr', null,
          e('th', null, 'Stage'), e('th', null, 'NAS'),
          e('th', null, 'Replica'), e('th', null, 'State'))),
        e('tbody', null, rows)));
  };
```

- [ ] **Step 2: Add the NAS-fingerprint proxy the panel reads**

The browser can't reach the NAS directly (CORS / tailnet). In `photosearch/admin_api.py`, add:

```python
@router.get("/maintenance-nas-fingerprint")
def admin_maintenance_nas_fingerprint():
    """Proxy the NAS's fingerprint for the drift panel. Mirrors the existing
    workers/queue-status proxy: the browser talks only to this host."""
    import requests
    nas = (os.environ.get("PHOTOSEARCH_NAS_URL") or "").rstrip("/")
    if not nas:
        raise HTTPException(status_code=400, detail="not in replica mode")
    try:
        r = requests.get(f"{nas}/api/admin/maintenance-fingerprint", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NAS unreachable: {e}")
```

- [ ] **Step 3: Add the styles**

In `frontend/dist/admin_maintenance.html`, inside the existing `<style>` block:

```css
.ms-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.ms-table th, .ms-table td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #2a2a2a; }
.ms-warn { color: #f5a623; font-weight: 600; }
.ms-ok { color: #4caf50; }
.ms-info { color: #888; }
.ms-none { color: #666; }
```

Copy the same rules into `frontend/dist/status.html`'s `<style>` block — per the project convention, styles live inline per page and are copied rather than shared.

- [ ] **Step 4: Render the panel**

In `frontend/dist/admin_maintenance.html`, inside the maintenance card's render, add:

```javascript
            e(PS.MaintenanceSyncPanel, { compact: false }),
```

In `frontend/dist/status.html`, inside the existing Replica card's render, add:

```javascript
            e(PS.MaintenanceSyncPanel, { compact: true }),
```

- [ ] **Step 5: Disable excluded stages in replica mode**

In `frontend/dist/admin_maintenance.html`, the Colors / Match faces checkboxes must be disabled when `replica_mode` is true, so the UI matches the server-side 400 from Task 8. Fetch `replica_mode` from `/api/admin/maintenance-fingerprint` and extend each checkbox's `disabled` prop:

```javascript
            e('label', { title: replicaMode ? 'Needs photo originals — run on the NAS' : '' },
              e('input', { type: 'checkbox', checked: doColors,
                disabled: busy || replicaMode,
                onChange: function(ev) { setColors(ev.target.checked); } }),
              'Colors (heavy)'),
```

Apply the same `disabled: busy || replicaMode` and `title` to the "Match faces (heavy)" checkbox.

- [ ] **Step 6: Syntax-check the changed JS**

There is no JS test harness and no build step, so a syntax error would only surface as a blank page in the browser. Catch it here:

```bash
node --check frontend/dist/shared.js
```

Expected: no output (exit 0). Browser verification is the operator's — see the runbook at the bottom of this plan. Do NOT start a server in this task.

- [ ] **Step 7: Commit**

```bash
git add frontend/dist/shared.js frontend/dist/status.html frontend/dist/admin_maintenance.html photosearch/admin_api.py
git commit -m "feat(ui): NAS-vs-replica maintenance drift panel; disable excluded stages"
```

---

### Task 11: Document the scheduling + replica-mode model (docs only)

**Files:**
- Modify: `CLAUDE.md`, `.claude/skills/photo-search/SKILL.md`

**Interfaces:**
- Consumes: everything above.
- Produces: documentation only. **No code, and no changes to the NAS.**

> **Scope decision (2026-07-17):** installing the cron entry touches production
> — root's crontab and `/var/log` on the live NAS, over SSH that UGOS
> auto-blocks on retry storms. That is **not** an agent action. This task writes
> the docs only; the operator runs the install from the runbook at the bottom of
> this plan. Do NOT ssh anywhere in this task.

- [ ] **Step 1: Document in CLAUDE.md**

Add to the maintenance section of `CLAUDE.md`:

```markdown
### Scheduling + replica-mode maintenance (2026-07-17)

`maintenance-sweep` runs nightly on the NAS from **root's crontab** at
**01:00 UTC** (`CRON_TZ=UTC`, so it never drifts with DST). That is 18:00
America/Los_Angeles — chosen deliberately over the 03:00 Pacific ingest slot;
accepted tradeoff is that it runs during California evening. Log:
`/var/log/photo-maintenance.log`. `--recluster` / `--dedup-photos` stay OFF
(recluster clears `ignored_clusters`; dedup DELETEs photos).

**Replica mode is now gated.** A sweep writes to whatever `PHOTOSEARCH_DB`
points at, and `sync-replica.sh` replaces the replica's DB wholesale — so a
replica-side apply used to be silently destroyed on the next sync. Now:

- `apply=true` on the replica with the NAS unreachable → **503, refuses to run**.
- `colors` / `match_faces` / `dedup` / `recluster` / `requeue` → **400** in
  replica mode (colors needs pixels the replica doesn't have; the rest must not
  cross machines or already have `export-face-state`).
- Everything else: pre-flight fingerprint → auto-sync if drifted → compute →
  one push at the end. Cheap deterministic stages are **re-triggered** on the
  NAS rather than transferred; only `stacking` ships rows.

`maintenance_runs` (schema v29) holds the per-stage watermark on both machines.
The `/status` Replica card and `/admin_maintenance` show per-stage drift;
**"Replica ahead — unpushed"** means local work will be lost on the next sync.

Module: `photosearch/maintenance_sync.py`. Spec:
`docs/superpowers/specs/2026-07-17-maintenance-push-up-design.md`.
```

- [ ] **Step 2: Mirror into the skill doc**

Add the same section to `.claude/skills/photo-search/SKILL.md` under its maintenance heading, so the skill and CLAUDE.md don't disagree.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md .claude/skills/photo-search/SKILL.md
git commit -m "docs: schedule maintenance-sweep at 01:00 UTC; document replica-mode gating"
```

---

## Operator runbook — steps the agents deliberately do NOT perform

These touch production or need human eyes. Agents finish at Task 11; the
operator runs these. Deploy the new code to the NAS first (`/status` → Build →
Restart, or the manual recovery sequence in CLAUDE.md) — the NAS needs the v29
schema and the new endpoints before any push can land.

### 1. Verify `CRON_TZ` is supported — the UTC pin depends on it

```bash
ssh cantimatt@dxp4800-f976 'man 5 crontab | grep -c CRON_TZ'
```

Non-zero → supported. `0` → use the fallback `0 18 * * *` (host-local; drifts to
02:00 UTC in Pacific winter) and correct the `CLAUDE.md` section Task 11 wrote.

**One attempt.** UGOS auto-blocks the client IP on SSH retry storms; if it
fails, diagnose before retrying.

### 2. Smoke-test the exact cron command as a dry run, before trusting it unattended

```bash
ssh cantimatt@dxp4800-f976 'cd /volume1/docker/photosearch && \
  docker compose -f docker-compose.nas.yml run --rm photosearch maintenance-sweep'
```

Expected: per-stage `would` counts, no writes.

### 3. Create the log file

```bash
ssh cantimatt@dxp4800-f976 \
  'sudo touch /var/log/photo-maintenance.log && sudo chown cantimatt:admin /var/log/photo-maintenance.log'
```

### 4. Install the cron entry

Via a temp file — the `( crontab -l; echo ... ) | crontab -` one-liner is
paste-fragile and yields `"-":1: bad minute`:

```bash
ssh cantimatt@dxp4800-f976 'sudo crontab -l > /tmp/rootcron 2>/dev/null; \
  grep -q CRON_TZ /tmp/rootcron || echo "CRON_TZ=UTC" >> /tmp/rootcron; \
  echo "0 1 * * * cd /volume1/docker/photosearch && docker compose -f docker-compose.nas.yml run --rm photosearch maintenance-sweep --apply >> /var/log/photo-maintenance.log 2>&1" >> /tmp/rootcron; \
  sudo crontab /tmp/rootcron && rm /tmp/rootcron && sudo crontab -l'
```

Expected: the printed crontab shows `CRON_TZ=UTC` and `0 1 * * *` alongside the
existing `0 3 * * *` ingest entry.

### 5. Replica → NAS round-trip (the smoke test no automated test covers)

1. `./sync-replica.sh`
2. `./run-local-replica.sh --model qwen/qwen2.5-7b-instruct`
3. `/admin_maintenance` → the drift table renders one row per stage; **Colors**
   and **Match faces** are greyed out with a tooltip.
4. Check **Apply**, heavy stages off, run. The log shows stage results, then a
   `push` event.
5. Confirm the watermark crossed over:
   ```bash
   curl -s http://100.115.143.4:8000/api/admin/maintenance-fingerprint | python3 -m json.tool
   ```
   Pushed stages should read `"source": "replica"`.
6. Reload `/admin_maintenance` — every pushed stage reads **In sync**.
7. `/status` shows the compact one-liner in the Replica card.

### 6. Verify the gate that motivated all of this

Point `PHOTOSEARCH_NAS_URL` at a dead host and try an apply. It must refuse with
a 503 and write nothing. If it writes, the central bug is not fixed.
