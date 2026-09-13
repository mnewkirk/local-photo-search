# Handoff — 2026-09-13 (b): deploy the replica face stages

Continues `docs/HANDOFF-2026-09-13.md`. Its item 1 is **built and on main but
not deployed** — the session that wrote it ran on the Mac, which could not
reach the NAS (Tailscale reported stopped), so the deploy moves to the desktop.

Placeholders: `<nas>` is the NAS web address (`http://<nas>:8000`), `<replica>`
is the desktop replica checkout. Use the IP, not the hostname, for HTTP.

---

## What shipped — `f209884` on main

`match_faces` and `recluster` now run from `/admin/maintenance` on the replica
instead of 400ing. New push mode `face_state` beside trigger / transfer /
excluded:

- The replica computes the stage, exports a face-state file
  (`photosearch/face_state.py`, ~9 MB, with fingerprint + stage watermarks in a
  `face_state_meta` table) and POSTs it to the NAS at
  **`POST /api/admin/maintenance-apply-face-state`**.
- Push leg order is transfer → face_state → trigger; a landed match adds
  `resolve_dups` to the triggers.
- The NAS applies **only what was recomputed**. Match → additive person fill
  (never `dedupe_unmatched`), no cluster copy. Recluster → new cluster ids,
  stale ids cleared on named faces and on faces absent from the file, and
  `ignored_clusters` **remapped** (a new cluster stays ignored when >50% of its
  faces were ignored before) instead of wiped.
- `export-face-state` / `apply-face-state` CLI now call the same module.
- `colors` / `dedup_photos` / `requeue` stay excluded.

Tests: 23 new (`tests/test_face_state.py`, `tests/test_maintenance_sync.py`).
CI-equivalent run: 1104 passed, 66 skipped, nothing ignored. **Never exercised
against the real NAS** — step 3 below is the first real run.

---

## Step 1 — redeploy the NAS (must come first)

A NAS older than `f209884` 404s the new endpoint: the push reports `http_404`
and the replica's face work is overwritten by the next sync.

Python-only change, no `requirements.txt` edit → ~10 s build. Easiest path is
the `/status` **Deployment** card: Fetch → Pull → Build → Restart. API
equivalent:

```bash
curl -s -X POST http://<nas>:8000/api/admin/git-pull
curl -sN -X POST http://<nas>:8000/api/admin/docker-build   # SSE; wait for "done"
curl -s -X POST http://<nas>:8000/api/admin/restart
```

Verify — deployed SHA must be `f209884` (or later):

```bash
curl -s http://<nas>:8000/api/admin/version
curl -s -o /dev/null -w "%{http_code}\n" -X POST \
  http://<nas>:8000/api/admin/maintenance-apply-face-state --data-binary 'x'
# expect 400 (bad_face_state_file). 404 means the old image is still running.
```

Last known NAS SHA was `9e22b8b` (per the earlier handoff; not re-verified).
Restart the worker fleet only if it misbehaves — the 503 handshake covers it.

## Step 2 — update the replica on the desktop

```bash
cd <replica> && git pull
systemctl --user restart photosearch-replica.service \
  || sudo systemctl restart photosearch-replica.service   # whichever it is
./.venv/bin/python -c "from photosearch.maintenance_sync import push_mode; print(push_mode('match_faces'))"
# expect: face_state
```

On `localhost:8001/admin/maintenance`, the **Match faces** and **Recluster
faces** checkboxes should now be enabled.

## Step 3 — first real run: match the 2026-09-12 shoot

Baseline first (NAS, authoritative) so the result is measurable:

```sql
SELECT COUNT(*) FROM faces f JOIN photos p ON p.id = f.photo_id
 WHERE p.date_taken LIKE '2026-09-12%' AND f.person_id IS NOT NULL;
-- was 1 on 2026-09-13
```

Then on the replica `/admin/maintenance`: check **Apply** + **Match faces**,
leave Recluster off for this first run, Run sweep. **Keep the tab open on the
desktop** — the SSE sweep dies with the browser (a phone screen-lock cancelled
one before). The push runs detached once the sweep's `done` arrives.

What should happen:

1. Pre-flight: syncs from the NAS if the photo index drifted.
2. `match_faces` runs locally; trigger stages show `deferred`.
3. Push: face_state leg, then triggers (including `resolve_dups`).
4. `GET localhost:8001/api/admin/maintenance-push-status` → `state: "ok"`,
   `stages.match_faces.status: "applied"` with a `persons` count.
5. Drift panel: `match_faces` reads **In sync**, not "Replica ahead".

Re-run the baseline query on the NAS; it should be well above 1.

If the push fails:

| `error` | meaning | fix |
|---|---|---|
| `http_404` | NAS not redeployed | step 1, then re-run the sweep |
| `fingerprint_mismatch` | an ingest landed mid-run | re-run; pre-flight re-syncs |
| `unreachable` | NAS restarting / network | re-run |
| `partial` state | some stages landed | read per-stage results; don't blindly re-push |

A re-run after a failure recomputes from scratch — the sync replaces the local
DB, so nothing half-applied lingers on the replica.

## Step 4 — then

- `review-faces` for 2026-09-12 — it learns the jersey colour from named faces
  on the date, which step 3 finally provides.
- Recluster from the replica when wanted. Before the first one, note the NAS's
  `SELECT COUNT(*) FROM ignored_clusters` and compare with the push result's
  `ignored_before` / `ignored_after`.

---

## Decisions to review

- **Ignored-cluster remap rule.** The earlier handoff said to reuse the approach
  in `rollback_calvin.py` / the Calvin memory note. Neither exists on the Mac
  (not in the repo, git history or Mac memory) — they are probably on the
  desktop. If that approach differs from the >50% face-membership rule, compare
  and adjust `face_state.apply_face_state`.
- **Clear-then-push window.** A label cleared on the NAS between sync and a
  match push is re-filled: `/api/faces/{id}/clear` sets `match_source = NULL`,
  indistinguishable from never-matched. Narrow, accepted; a distinct
  `match_source` on clear would close it.
- **Writer-lock time on the N100.** The apply holds `BEGIN IMMEDIATE` across the
  face updates (counts are computed before). Watch worker submits during the
  first recluster push; the fleet's 60 s busy timeout should absorb it.
- **Recluster after a replica match.** Duplicate-person faces the NAS's
  `resolve_dups` later unmatches were excluded from the replica's recluster,
  so they land unclustered — reviewable in the `/faces` "Unclustered" bucket.

## Traps hit this session

- `scripts/test-like-ci.sh` is committed without the exec bit — run it as
  `bash scripts/test-like-ci.sh`. The shared `/tmp/photosearch-ci-venv` on the
  Mac had no pytest; `PHOTOSEARCH_CI_VENV=<fresh dir>` rebuilds cleanly.
- The Mac's Tailscale CLI reported `stopped` even after it was enabled in the
  app, and both NAS tailnet addresses timed out.

## Remaining backlog

Unchanged from `docs/HANDOFF-2026-09-13.md` items 2–6: D4 loud ingest
write-failure, S2 + S4 as schema v30, ARW/video folder merge, intra-cluster
congruence review, open questions.
