# Maintenance sync: replica-computed results pushed to the NAS

**Status:** Designed, not implemented. 2026-07-17.
**Depends on:** M25 (`maintenance-sweep`), M26a (local read-replica + `sync-replica.sh`).
**Schema:** bumps `SCHEMA_VERSION` 28 → 29.

## Problem

Maintenance-sweep stages run against whatever `PHOTOSEARCH_DB` points at. The
endpoint is a plain `with _get_db() as db: run_maintenance_sweep(db, ...)`
(`web.py:4207`), and `_get_db()` opens `PHOTOSEARCH_DB` (`web.py:116`). On the
local replica that is `photo_index.db.local`, so:

1. A sweep run from the replica's maintenance page writes **only** to the
   replica.
2. `sync-replica.sh` dumps the NAS DB and does `mv "${TARGET}.tmp" "${TARGET}"`
   — a whole-file replace, not a merge.
3. The replica-side results are destroyed on the next sync, silently.

The sync direction is NAS → replica only, and it is a full replace, so it can
only ever carry NAS state down. There is no mechanism by which replica state
travels up.

Unlike the M28 "re-run passes" flow (NAS-authoritative + mirror-local) and the
Workers panel (`admin_api.py:822`, which proxies to the NAS in replica mode
precisely because "the local replica DB is a fully-processed synced snapshot"),
the maintenance sweep was never wired for replica mode.

The repo has already been bitten by this once: photobook curation state was
moved to a sidecar file with the comment "curation state lives in a SEPARATE
sidecar sqlite file so a full replica re-sync (sync-replica.sh atomically swaps
_db_path) can't wipe it" (`web.py:121`).

### Contributing root cause

`maintenance-sweep` is **not scheduled anywhere** — it appears in no cron entry,
no shell script, and no compose file, only in documentation. The M25 plan
proposed a cron stage and left it at "Decide before wiring it to cron"
(`docs/plans/backfill-maintenance-sweep.md:86`); that half never shipped. So
these stages only run when a human clicks them, and clicking them on the replica
loses them. The NAS is therefore likely stale on derived data generally, not
merely drifting from the replica.

## Measurements that shaped this design

Taken 2026-07-17 against the Jul 11 replica (151,432 photos):

| Fact | Value | Consequence |
|---|---|---|
| `normalize_overall` inputs | 3,685 rows with `aes_overall` | Read + sort < 10ms |
| `normalize_subject_overall` inputs | 1,065 rows | Same |
| `photo_stacks` / `stack_members` | 43 / 111 rows | Push payload is KB, not MB |
| Fingerprint `COUNT(*), MAX(id)` | 151,432 / 237,307 | 7ms — free to check |

`normalize_overall` (`aesthetics.py:418`) reads `(id, aes_overall)`, computes
percentile ranks, writes back `aes_overall_pct`. It is a **pure deterministic
derivation**: given identical inputs, the NAS and replica produce bit-identical
output. There is nothing in the result the NAS cannot recreate for free.

`_stage_stacking` (`maintenance.py:323`) is a library-wide re-detect over every
timestamped photo, loading CLIP embeddings and doing windowed pairwise
comparison. The maintenance page's own copy says the heavy stages "peg the N100
and starve the server — leave them off here and run the full sweep off-hours via
cron" (`admin_maintenance.html:1381`). This is the stage with a real reason to
compute on the desktop.

## Goals

- Results of a replica-run maintenance sweep end up on **both** machines.
- The freshness check is cheap — no per-row update-date comparison.
- The `/status` page makes NAS-vs-replica drift obvious.
- Heavy stages (stacking) compute on the desktop, not the N100.
- The sweep runs on a schedule, so derived data stops going stale by default.

## Non-goals

- Reverse-syncing the whole DB. The NAS stays authoritative for photos.
- Pushing destructive operations (`dedup`) across machines.
- A second face-state bridge. `export-face-state` / `apply-face-state` already
  covers recluster/match.

## Design

### Stage taxonomy: two push modes

Each stage declares how it reconciles to the NAS.

**`trigger`** — the NAS recomputes the stage itself. The replica ships no data;
it asks the NAS to run the stage over the NAS's own current data. Correct by
construction: a stale-input mismatch is structurally impossible, so these stages
need no fingerprint guard and no payload.

Stages: `normalize_aesthetics`, `normalize_subject_aesthetics`, `geocode`,
`normalize`, `infer`, `normalize_inferred`, `resolve_dups`.

`geocode` **must** be trigger-mode rather than transfer: the replica almost
certainly lacks the `/data/geonames` rich dataset, so replica-computed place
names would silently downgrade the NAS's richer labels (Point Reyes → Inverness,
per the CLAUDE.md note). Letting the NAS geocode with its own dataset avoids
this entirely.

Implementation note: trigger mode is a POST to the NAS's **existing**
`/api/admin/maintenance-sweep`, which needs only a new `stages` subset parameter
so trigger mode can request exactly `normalize_aesthetics` without dragging the
whole default plan along.

**`transfer`** — the replica ships computed rows because recomputing on the N100
is what we are avoiding.

Stages: `stacking` only.

Payload: `photo_stacks` + `stack_members`, applied as a **full replace**.
Stacking is already a full re-detect, so replace is its natural semantics.
Photo ids are stable (`AUTOINCREMENT`, and the replica is a dump of the NAS), so
ids are safe join keys.

**Excluded — not runnable from the replica at all**

These stages are **disabled in the replica's maintenance UI** (checkbox greyed
with a reason), and rejected server-side if requested in replica mode. They are
not "run locally and skip the push" — they must not run locally at all.

| Stage | Reason |
|---|---|
| `colors` | Reads pixels, and the replica has no originals (`PHOTO_ROOT` is deliberately unset so images proxy from the NAS). It cannot produce a correct result locally. Run it on the NAS (cron, off-hours). |
| `dedup` | DELETEs photos. Destructive cross-machine ops are out of scope; a local run would also be silently reverted by the next sync. |
| `recluster`, `match_faces` | Already served by `export-face-state` / `apply-face-state`. No second path. |

### Flow

1. **Pre-flight.** Replica fetches the NAS fingerprint and compares to its own.
2. **Mismatch → auto-sync.** Run `replica-sync` first, streaming into the same
   log, then re-check. This must precede compute: a sync atomically replaces the
   whole local DB, so discovering the mismatch after a local stacking run would
   destroy the very results we intend to push.
3. **Compute.** Run the sweep locally.
4. **Push.** One push after **all** stages complete — never between stages.
   Non-blocking with respect to the sweep: sweep results are reported first, and
   the push runs in a detached background thread that survives client
   disconnect.
5. **Mid-run drift.** If the fingerprint moved while stacking ran (e.g. the 3am
   ingest landed), the push is rejected. Surface it and offer Retry, which
   redoes the whole chain. No auto-loop.

### Fingerprint

`(COUNT(*), MAX(id))` over `photos`. Two cheap queries — 7ms measured locally.
This is the "up to date on the photo index" check, and it deliberately avoids
per-row comparison.

It is a pragmatic fingerprint, not a cryptographic one: defeating it requires
deleting one photo and adding another with a higher id between checks. The only
in-tree operation that deletes photos is `dedup`, which is excluded from push
and is opt-in. Accepted.

### Schema (v29)

Bump `SCHEMA_VERSION` to 29 in `db.py` and add to `_init_schema()`:

```sql
CREATE TABLE IF NOT EXISTS maintenance_runs (
  stage            TEXT PRIMARY KEY,
  last_run_at      TEXT NOT NULL,   -- UTC ISO8601
  photo_count      INTEGER,         -- fingerprint at run time
  photo_max_id     INTEGER,
  applied          INTEGER,
  source           TEXT             -- 'nas' | 'replica'
);
```

Lives in the main DB on both machines. Sync overwrites the replica's copy with
the NAS's, which is **correct, not a bug**: after a successful push both sides
already agree; after a failed push the sync wipes the local results *and* the
record of them together. The two stay consistent either way.

### Endpoints

| Endpoint | Host | Purpose |
|---|---|---|
| `GET /api/admin/maintenance-fingerprint` | both | `{photo_count, photo_max_id, stages: {stage: {last_run_at, source}}}`. Serves the pre-flight guard **and** the status card. |
| `POST /api/admin/maintenance-apply` | NAS | Transfer path. Body: stacking rows + replica fingerprint + per-stage timestamps. Validates, applies in one transaction, stamps `maintenance_runs`. |
| `POST /api/admin/maintenance-sweep` | NAS | Existing. Gains a `stages` subset param for trigger mode. |
| `GET /api/admin/maintenance-push-status` | replica | Push outlives the request; backs the Retry UX. |

The new fingerprint endpoint also sidesteps a wart in the existing Replica card,
which sources the NAS photo count from `/api/stats` (`admin_api.py:447`) — the
same expensive COUNT-scan endpoint that `WorkerClient(probe=False)` deliberately
avoids.

`maintenance-apply` validation, per stage:

1. Replica fingerprint == NAS fingerprint, else reject the whole request (409).
2. Replica's `last_run_at` > NAS's `last_run_at` for that stage, else skip that
   stage (the NAS is fresher).
3. Apply in one transaction; stamp `maintenance_runs` with the replica's
   timestamp and `source='replica'`.

### New module

`photosearch/maintenance_sync.py`:

- `photo_fingerprint(db) -> dict`
- `push_mode(stage) -> 'trigger' | 'transfer' | 'excluded'`
- `collect_payload(db, stages) -> dict`
- `push_to_nas(db, nas_url, stages) -> dict` (per-stage results)

Keeps `maintenance.py` focused on running stages; the sync concern stays
separately testable.

### UI

A shared `PS.MaintenanceSyncPanel` in `shared.js` — compact on `/status`'s
existing Replica card, full detail on `/admin_maintenance`. One row per stage
with both timestamps and a state:

| State | Meaning |
|---|---|
| **In sync** | timestamps match |
| **Replica ahead — unpushed** | ⚠️ the data-loss condition. Prominent. Offers "Push now". |
| **NAS ahead** | replica catches up on next sync — informational |
| **Never run** | neither side has run it |
| **Index drift** | fingerprints differ; a push would auto-sync first |

"Replica ahead — unpushed" is the state that matters: it is exactly the silent
failure this spec exists to fix, and surfacing it is most of the value.

## Error handling

**The gate that prevents the original bug:** when `apply=true` **and** replica
mode **and** the NAS is unreachable, the sweep refuses to run rather than
writing results guaranteed to be wiped. Dry-runs (`apply=false`) are always
allowed — they write nothing.

| Case | Behavior |
|---|---|
| Auto-sync fails | Abort the chain before computing. Never sweep against a known-stale index. |
| Push fails (network) | Local results stand; `maintenance_runs` keeps replica timestamps; card shows "Replica ahead — unpushed"; Retry available. |
| Push rejected (fingerprint moved mid-run) | Same state, same Retry. One user-initiated retry, no auto-loop. |
| Sweep cancelled mid-stage | Only stages reporting `status='done'` are push-eligible. A half-finished stacking run never ships. |
| NAS mid-restart | Surfaces as a **connection error**, not a 503. The shutdown middleware only guards `/api/worker/*` and `/api/photos/*/full` (`web.py:85`) — `/api/admin/*` is deliberately left serving so the browser UI keeps working during a drain. So there is nothing to reuse from `WorkerClient._request()` here, and no backoff to add: the push reports `unreachable` and the user retries. (An earlier draft of this spec claimed admin paths were 503-gated. They are not.) |

**Transactionality.** The transfer path applies in one NAS-side transaction
(replace both tables, stamp `maintenance_runs`, commit). Trigger stages are a
separate call, so the two modes can partially succeed relative to each other.
The push therefore reports **per-stage** results rather than one boolean, and
the card reflects per-stage reality. Transfer runs first (fast, bounded);
triggers after (geocode can be slow on the N100).

## Testing

`tests/test_maintenance_sync.py`, following `tests/test_maintenance.py`
conventions:

- Fingerprint computation.
- Stage push-mode taxonomy.
- Payload collection.
- Timestamp comparison in all three directions: replica newer → push, NAS newer
  → skip, equal → skip.
- Fingerprint mismatch → rejection.
- Cancelled stage → not push-eligible.
- Apply path against a fixture DB: given a stacking payload, assert both tables
  are replaced and `maintenance_runs` is stamped `source='replica'`.

Plus:

- A v28 → v29 migration test in `tests/test_db.py`, matching the existing
  pattern of building a minimal old-version DB and asserting the migration runs.
- Endpoint tests in the style of `tests/test_web_geocode.py`, with the NAS call
  mocked.

**Not covered by automated tests:** a real replica → NAS round-trip. Manual
smoke test, same caveat M24b carries for its LM Studio path.

## Scheduling: wire `maintenance-sweep` to cron on the NAS

The M25 plan proposed this and never shipped it; it is the reason derived data
goes stale. It is in scope here.

### The entry

Runs at **01:00 UTC**, pinned to UTC so it never drifts with anyone's DST:

```cron
CRON_TZ=UTC
0 1 * * * cd /volume1/docker/photosearch && docker compose -f docker-compose.nas.yml run --rm photosearch maintenance-sweep --apply >> /var/log/photo-maintenance.log 2>&1
```

Goes in **root's crontab** (`sudo crontab -l`) alongside the existing ingest
entry — a normal `crontab -l` as `cantimatt` is empty. Load it via a temp file;
the `( crontab -l; echo '...' ) | crontab -` one-liner is paste-fragile (mangled
continuations → `"-":1: bad minute`). The log needs
`sudo chown cantimatt:admin /var/log/photo-maintenance.log`, mirroring
`/var/log/photo-ingest.log`.

### Timing rationale

The NAS runs `TZ=America/Los_Angeles` (`docker-compose.nas.yml:181`), verified
against its HTTP `Date` header. So:

| | |
|---|---|
| 01:00 UTC | the scheduled slot |
| = 03:00 CEST | the requested wall-clock time |
| = 18:00 America/Los_Angeles | when it actually runs at the NAS |
| existing ingest at `0 3 * * *` | 03:00 Pacific — **nine hours clear**, no collision |

**Accepted tradeoff:** 18:00 Pacific is evening prime time where the NAS lives,
and the heavy stages "peg the N100 and starve the server"
(`admin_maintenance.html:1381`). This was chosen deliberately over chaining onto
the ingest line at 03:00 Pacific. Revisit if the household notices the NAS
dragging in the evening — the fix is to move the slot, not to change the design.

`CRON_TZ` is a Vixie/Debian-cron feature. UGOS ships a Debian-derived cron (it
is missing only the setgid `crontab` setup, per CLAUDE.md), so this should work
— but **verify it on the box before relying on it**. If unsupported, the
fallback is `0 18 * * *` in host-local time, which drifts to 02:00 UTC in
Pacific winter; accept the drift or wrap the command in a UTC-aware guard.

### Stage selection

The bare `maintenance-sweep --apply` runs the default plan: geocode, normalize,
infer, normalize_inferred, colors, stacking, match_faces, resolve_dups, plus
both aesthetic normalizes. `--recluster` and `--dedup-photos` are off by default
and **stay off** — recluster clears `ignored_clusters` (re-running it nightly
would wipe ignore decisions every night, the exact hazard the M25 plan flagged),
and dedup DELETEs photos.

### Interaction with the push path

The two compose safely through the same watermark, with no extra machinery:

- Cron keeps the NAS fresh nightly; the replica picks that up on its next sync,
  where it lands as **NAS ahead** — informational, no action.
- If the user runs stacking on the replica *after* the last cron run, their
  `last_run_at` is newer and the push applies.
- If cron ran stacking *more recently* than the replica's local run, the push
  skips that stage — the existing rule ("replica's `last_run_at` must be newer,
  else skip") already covers it. No new conflict logic.

With cron running nightly, the push path becomes the exception rather than the
rule: it only carries work the user explicitly chose to do *now* rather than
wait for tonight.

## Future work

- Extend transfer mode to `match_faces` if the `export-face-state` bridge proves
  awkward in practice. Not before.
- Reconsider the strict fingerprint guard if the sync → run → push window proves
  annoying in daily use. Superset-tolerance (allow when the NAS has only *added*
  photos) was considered and deferred as a subtler mental model.
