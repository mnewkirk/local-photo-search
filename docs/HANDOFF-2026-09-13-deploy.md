# Handoff — 2026-09-13 (b): deploy the replica face stages

Continues `docs/HANDOFF-2026-09-13.md`. Its item 1 is **built and on main but
not deployed** — the session that wrote it ran on the Mac, which could not
reach the NAS (Tailscale reported stopped), so the deploy moves to the desktop.

> **STATUS: deployed and run, 2026-09-13.** Steps 1–3 below are complete and
> verified; the 2026-09-12 shoot went from 1 named face to 197. A separate
> pre-existing defect surfaced — `geocode` OOM-kills the NAS web container, so
> every replica push's trigger leg fails. See **"Deploy result"** at the foot of
> this file for what landed, what to distrust, and what is still open.

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

---

# Deploy result — 2026-09-13, from the desktop

Steps 1–3 are **done**. One new, unrelated defect found: the `geocode` stage
OOM-kills the NAS web container, which breaks the trigger leg of *every*
replica push.

## Step 1 — NAS redeployed ✅

`9e22b8b` → `47dee89` via the admin API (pull → build 21 s → restart helper).
Verified: `/api/admin/version` reports `deployed_sha 47dee89`, and the probe
returns **400 `bad_face_state_file`** (not 404), so the new endpoint is live.

## Step 2 — replica updated ✅

`git pull` + `systemctl --user restart photosearch-replica.service`.
`push_mode('match_faces')` → `face_state`; `/api/admin/version` reports
`native`, `deployed 47dee89`. Local test run: 88 passed.

## Step 3 — first real run ✅ (the feature works)

Driven by `curl -N` on the desktop rather than a browser tab, so no screen-lock
risk; SSE written to a file, waited on a marker file (never `pgrep -f`).

| | before | after |
|---|---|---|
| 2026-09-12 faces named | **1** | **197** (26 strict + 170 temporal, all Calvin; + 1 pre-existing manual) |
| photos with a named face | 1 | 197 |
| library faces named | 59,489 | 59,678 |
| duplicate person-in-photo groups | 28 (post-match) | **0** |

Sweep: trigger stages `deferred` as designed, `match_faces` ran locally in
**55.6 s** and matched 6,556 faces. Push: face_state leg **applied**, and
`match_faces` now reads **in sync** on the drift panel — nothing is "Replica
ahead — unpushed".

**Only 217 of the replica's 6,556 matches landed, and that is correct.** The
other ~6,339 are faces whose NAS `match_source` is `dedupe_unmatched` — the
June 2026 over-matching cleanup. The replica's temporal pass happily re-matches
them; the additive apply refuses them. That guard is the whole reason this is
safe behind a button, and this run is the first evidence it fires at scale. The
replica's local re-matches are discarded by the next sync, as intended.

> **Read the 170 temporal matches with suspicion.** The Calvin-temporal memory
> note measured **4 % accuracy** on the 2026-08-29 soccer shoot (strict 14/14
> correct, temporal 1/26). This is the same body (`ILCE-7RM6`) and the same kind
> of shoot. The 26 **strict** matches are the trustworthy ones. Decide whether
> to keep the temporal ones before leaning on them — and note `review-faces`
> (step 4) learns the jersey colour from named faces on the date, so seeding it
> with 170 probably-wrong matches would poison it. Reverse only via a pinned-id
> script; never `restore-unmatched-faces`.

## NEW DEFECT — `geocode` OOM-kills the NAS container

**This is what made the push report `partial`, and it is not the new feature.**

The trigger leg died with every stage `unreachable`. `docker events` shows the
photosearch container **`die … exitCode=137`** (SIGKILL) twice, at
`execDuration=291` and `264`, both while the stream sat on `geocode scanning`:

```
container die 920916e00344 … exitCode=137 … name=photosearch
```

Cause is memory. The NAS has the **rich GeoNames dataset installed**
(`/data/geonames/rg_rich.csv`, 370 MB; `allCountries.txt`, 1.7 GB), and CLAUDE.md
budgets ~1 GB steady-state for the KDTree plus a parse spike. The host has
**7.7 GB total, ~3.2 GB already in use, 4.5 GB available**, and the container
limit is 7 GiB — i.e. *above* host RAM, so the cgroup limit never engages and
the kernel OOM-killer fires first. (No `dmesg` line to quote: UGOS restricts it.)

Every other trigger stage was then run individually and **all succeeded**:

| stage | result |
|---|---|
| `resolve_dups` | done, 28/28 — cleared every duplicate group |
| `normalize` | done, 1,016 |
| `normalize_aesthetics` | done, 155,966 (175 s) |
| `normalize_subject_aesthetics` | done, 69,553 (32 s) |
| `geocode` | **OOM-kills the container** (reproduced twice) |
| `infer`, `normalize_inferred` | **untested** — both reverse-geocode on apply, so both are suspect. Not tested because a fleet had 3 live claims and another OOM kill is not a graceful shutdown. |

**Why this matters beyond today:** a replica sweep defers *all* trigger stages
to the push, and `geocode` is always among them. So **every** replica push will
report `partial` with the server bouncing underneath it, until this is fixed.
It also means the nightly NAS `maintenance-sweep` cron (when installed) would
kill the web server every night.

Worth trying, in order:
1. Run `geocode` out-of-process — `docker compose run --rm photosearch
   normalize-places` gets its own container and its own memory, instead of
   loading the KDTree inside the long-lived web server.
2. Cap the web container (`mem_limit` *below* host RAM) so it fails as a clean
   Python `MemoryError` the SSE stream can report, instead of a silent SIGKILL
   that looks like "NAS unreachable".
3. Stream/chunk the `rg_rich.csv` load in `geonames_rich.py`, or drop back to
   stock `reverse_geocoder` on the NAS and keep the rich labels for backfills
   run from the desktop.

## Decisions to review — answered

- **Ignored-cluster remap rule.** The earlier handoff pointed at
  `rollback_calvin.py` / the Calvin memory note. Both are on this desktop and
  **neither is about cluster remapping** — `rollback_calvin.py` restores person
  matches for 223 pinned face ids from `face_dedupe_undo`, and the memory note
  only says to reverse that unmatch by pinned id rather than
  `restore-unmatched-faces`. So there is **no prior art to reconcile with**: the
  >50 %-face-membership rule in `face_state.apply_face_state` stands on its own,
  and it is a clear improvement on the wipe (`ignored_clusters_backup.json` at
  the repo root is a hand-taken 2026-09-10 backup — someone had already been
  bitten by the wipe). Nothing to change. **Consider this item closed.**
- **Writer-lock time on the N100.** Not observable on this run: the match apply
  touched only 217 rows. It stays open for the first *recluster* push.

## Step 4 — not done, deliberately

- **Recluster from the replica** — untested end-to-end. Before the first one,
  note `SELECT COUNT(*) FROM ignored_clusters` on the NAS (**31** as of now) and
  compare against the push result's `ignored_before` / `ignored_after`.
- **`review-faces` for 2026-09-12** — blocked on the judgement call above: it
  learns from named faces on the date, and 170 of the 197 are suspect temporal.

## Traps hit this session

- **Fetch before trusting `git log`.** This session re-implemented the whole
  `face_state` feature from `HANDOFF-2026-09-13.md` before fetching, because the
  local checkout was two commits stale and `git log` looked authoritative. The
  work was discarded. `git fetch` first when picking up a handoff — the handoff
  you were pointed at may itself be the newer commit.
- The SSE keepalive comments (`: keepalive`) dominate a saved stream; filter
  with `grep -v keepalive` before reading it.
