# Maintenance — history and traps

Geocode memory, vec0 orphans, the clip re-claim, the maintenance sweep and its replica-mode gating. Condensed rules: `.claude/rules/maintenance.md`. Append new history under a new `###` heading; never rename an existing heading.

<!-- moved verbatim from CLAUDE.md (line 100) on 2026-10-19 -->
### The memory cost is per-ROW, not the KDTree — and it OOM-killed the NAS

This section used to claim "~1 GB (the KDTree)". **That was wrong by 5×**, and
the error was load-bearing. Measured 2026-09-13: `RGeocoder.load()` keeps **one
Python dict per row** at ~745 bytes, so the original unfiltered 6.58M-row build
was **4.9 GB** — plus 1.44 GB of transient text, because
`get_rich_geocoder` did `io.StringIO(f.read())`, holding the 370 MB CSV *twice*
as UCS-2 (the file contains CJK punctuation, so Python can't use 1 byte/char).

Peak ~6.3 GB against 4.5 GB free. The `geocode` maintenance stage therefore
**SIGKILLed the whole photosearch container** (`exitCode=137`), reproducibly,
~4-5 min in. Because a replica sweep defers every trigger stage to the push,
this made **every** replica→NAS push report `partial` / `unreachable`, and it
would have killed the web server nightly once the maintenance cron was
installed. Symptom to recognise: `docker events` shows `container die …
exitCode=137` while the SSE stream sits on `geocode scanning`.

Two fixes, both needed:

- **`_MIN_POPULATION = 1`** drops class-P rows GeoNames records as population
  0 — **4.72M of 5.2M** populated places, overwhelmingly hamlets. Named POIs
  are matched by feature *code* and kept whatever their population (parks and
  peaks are always 0), so the labels this dataset exists for are untouched.
  6.58M rows → 1.86M, 4.9 GB → ~1.4 GB. In `build_rich_dataset` the `is_poi`
  check runs **before** the population gate, so a row that is both class P and
  a POI code survives — reversing that order silently drops park/village labels.
  **The rich labels are unaffected** — verified on the NAS after the rebuild:
  Muir Woods still resolves to "Muir Woods National Monument" (not "Mill
  Valley") and Point Reyes to "Mount Wittenberg" (not "Inverness"), which are
  the two examples this whole section exists to justify.
- **Stream the CSV.** `RGeocoder(stream=<file object>)` works —
  its `load()` is a `csv.DictReader`, which iterates line by line. Don't
  "simplify" it back to `StringIO(f.read())`.

**Rebuilding after a filter change does NOT need a re-download.**
`--force` now re-filters from the source files already on disk; the separate
`--refresh-source` is the one that pulls a fresh 400 MB (it used to be
conflated, so re-filtering cost a download that couldn't change the answer):

```bash
$DC run --rm photosearch download-geonames --force        # re-filter, no download
$DC run --rm photosearch download-geonames --refresh-source   # newer GeoNames data
```

Verified on the NAS 2026-09-13 after the rebuild (`download-geonames --force`,
1,856,756 rows): the `geocode` stage completes in **6.4 s** without restarting
the container, and the whole deferred trigger set runs clean. Note the stage is
gated missing-only, so a fully-enriched library **skips before loading the
KDTree at all** and pays nothing.

The durable fix is a **SQLite R-tree** instead of an in-memory KDTree — ~0
resident memory, no per-process load, and all 6.58M rows retained (so the
population gate could go away). `rtree` is already compiled into the SQLite on
both machines. Plan: `docs/plans/geocode-rtree.md`.

<!-- moved verbatim from CLAUDE.md (line 1758) on 2026-10-19 -->
## Vec0 orphan cleanup

`clip_embeddings` and `face_encodings` are sqlite-vec `vec0` virtual tables.
Virtual tables can't participate in foreign-key constraints, so
`ON DELETE CASCADE` on `photos`/`faces` doesn't reach them — when a photo or
face row is deleted, its vector row stays behind. The status page will show
`>100% embedded` when this happens. AUTOINCREMENT on `photos.id` / `faces.id`
prevents ID reuse, so the dangling rows can never silently re-attach to a
different photo.

Also: `add_clip_embedding` uses explicit DELETE+INSERT (not `INSERT OR
REPLACE`) because vec0 doesn't honor `OR REPLACE` — the PK conflict fires
first and raises UNIQUE, which causes worker submits to fail when a claim
TTL expired and another worker re-processed the same photo.

```bash
$DC photosearch cleanup-orphans [--dry-run]
```

<!-- moved verbatim from CLAUDE.md (line 1777) on 2026-10-19 -->
## Non-image rows & the clip-claim infinite re-claim

**FIXED — clip is on the attempts ledger now.** It used to be the only worker
pass with no `worker_processed.attempts >= MAX_PROCESS_ATTEMPTS` cap: the
claim/count predicates just checked `id NOT IN (SELECT photo_id FROM
clip_embeddings)`, the clip branch of `worker_api.submit_results` never called
`mark_processed`, and the worker silently skipped an image it could not load.
So such a photo left *no trace* and was re-claimed every TTL forever. SQLite
returns the claim set in rowid order, so the same unloadable rows sat at the
front of every claim — workers churned at ~290% CPU and `queue_depth.clip`
never reached 0 ("workers won't drain").

Now: both predicates carry `db._CLIP_NOT_EXHAUSTED`; `worker._process_clip`
reports a photo `embed_images_stream` skipped as a failure row (the shared
`failures` list); the server spends one attempt per failure (and per non-lock
write error — a malformed vector), defers only a transient lock, and after
three tries the photo stops being claimed and its batch reads `blocked`. A
**successful** embedding is still its own marker and writes no ledger row
(clip is the highest-volume pass; a row per photo on the N100 buys nothing).
`clear-pass clip` deletes the clip ledger rows along with the embeddings, so a
re-embed starts clean. Same deploy note as the other passes: an old worker
never sends `failures`, so against it clip stays uncapped until the fleet is
updated. Tests: `tests/test_worker_attempts_cap.py`.

The rows are still worth removing rather than leaving capped:

The usual culprit: files whose extension lies about their content — iOS Live
Photo / motion bundles saved as `IMG_xxxx(1).JPG` that are actually **ZIP
archives** wrapping a `.mov` (`PK` magic; PIL: "cannot identify image file").

`index.py:is_real_image()` (magic-byte allowlist) now gates row creation in
`index_directory` and reclassifies such files as move-only companions in
`ingest.py`, so no new bogus rows are created. Purge existing ones (runs on the
**NAS**, where the files live):

```bash
$DC run --rm photosearch purge-nonimage-photos                 # dry-run, no-CLIP set
$DC run --rm photosearch purge-nonimage-photos --audit /data/nonimage.csv --apply
```

Default scans only photos with no CLIP embedding (the stuck set); `--all` scans
everything. `--apply` requires `--audit` (deletes are reversible only by
re-import). Cascades faces/stacks; runs `cleanup-orphans` internally.

<!-- moved verbatim from CLAUDE.md (line 3598) on 2026-10-19 -->
## Maintenance sweep & duplicate pruning (M25 + M28)

`maintenance-sweep` (CLI) / `POST /api/admin/maintenance-sweep` (SSE) /
the `/status` Maintenance-sweep card all run the **same** orchestrator:
`maintenance.run_maintenance_sweep`. It is **one dependency-ordered backfill
over many stages**, not a menu of independent jobs. Dry-run is the default
(`apply=False`) — every stage reports a `would` count and writes nothing.

Default stage order (each gated by a "what's still missing" SQL predicate, so a
fully-enriched library is nearly free): `geocode` → `normalize` → `infer` →
`normalize_inferred` → `colors` → `stacking` → `match_faces` → `resolve_dups`
→ `normalize_aesthetics` → `normalize_subject_aesthetics`.
Opt-in extras flip a single stage on but **do not narrow the sweep**:

**`stacking` is missing-only (2026-09-26) — it used to rewrite every stack
nightly.** The log read `stacking done: would 157815, applied 27165` every
night: the stage called an **unscoped** `run_stacking`, i.e. `db.clear_stacks()`
+ a full-library re-detect + 27,165 `create_stack`s, on an unchanged library.
`would` was the count of dated photos and `applied` the total stack count, not
a change count. (Side effect: every hand-picked stack top was reset nightly.)

It now calls `stacking.run_incremental_stacking`:

- **Ledger** `stacking_seen(photo_id, date_taken, seen_at)` (schema v32) —
  which photos detection has considered, and at which `date_taken`. Dirty =
  eligible (dated + CLIP embedding) and no ledger row or a changed
  `date_taken` (retime), plus members of any stack that lost a member or is
  down to one. `would` = dirty photos; nothing dirty → `skipped`.
- **Scope is exact, not a margin.** `detect_stacks` only unions photos within
  `time_window_sec` of each other, so the library splits into *sessions*
  (maximal runs with every consecutive gap ≤ the window) and no stack crosses
  one. Re-detecting only sessions containing a dirty photo, closed over the
  existing stacks they touch, reproduces the full-run answer for them —
  `test_incremental_batches_converge_to_the_full_detect_answer` pins that.
  A new 1,373-photo shoot has a scope of ~1,373.
- **Writes a diff.** Detected vs affected existing stacks compared by
  membership: identical stacks are untouched (same id, hand-set top kept),
  vanished ones deleted **by id**, new ones created. `applied` = created +
  removed.
- **It can never wipe the library.** No `clear_stacks()`, no unscoped
  `detect_stacks`; an empty scope skips detection entirely (never
  `photo_ids=[]`, which `detect_stacks` reads as "whole library").
  `test_never_clears_the_library_or_runs_unscoped` spies on both.
- The ledger is written **last**, so an abort leaves those photos dirty and
  the next run converges.
- **First run after deploy** has an empty ledger: every photo is dirty, one
  full detection (~15 s on the desktop, what every night used to cost) — and,
  measured on a replica copy, **0 created / 0 removed / 27,165 unchanged**,
  then the second run is `skipped` in 0.3 s.

Known limits: a stack that lost a member but still has ≥ 2 eligible members is
not re-detected (only matters if the deleted photo bridged a chain), and a
re-embedded photo (CLIP re-run) is not dirty. A full re-detect is still the
`/status` stacking form / `photosearch stack` CLI. Other stack writers
(ingest's scoped run, `batch-advance`, the replica's full-replace transfer)
don't touch the ledger; the worst case is the sweep redoing a session.
Tests: `tests/test_maintenance_incremental.py`.

- `do_dedup` ("Prune duplicate photos") — prepends the destructive
  `dedup_photos` stage (DELETEs redundant copies, runs *first* so later stages
  see the deduped set).
- `do_recluster` — appends global DBSCAN (clears `ignored_clusters`).

**Gotcha (the thing that surprises you on `/status`):** toggling "Prune
duplicate photos" still previews/applies *every other default stage too* —
`infer` showing e.g. "would 1,356" is the infer-GPS stage doing its normal
missing-only backfill, not something dedup triggered. Applying the sweep with
dedup on therefore both deletes duplicates *and* writes inferred GPS
(`location_source='inferred'`), re-runs stacking, etc.

**When to use which:**
- **Prune duplicates only, nothing else** → `photosearch find-duplicate-photos`
  (preview, full same-image detection) + its standalone prune. No GPS/stacking
  side effects.
- **Periodic catch-up backfill after imports** → `maintenance-sweep` (the whole
  point: one idempotent pass that fills geocode/infer/colors/stacking/faces for
  new photos). Add `--apply`; leave dedup/recluster off unless you want them.
- **Sweep + prune in one go** → `maintenance-sweep` with dedup on, accepting the
  full-sweep side effects above.
- **One photo / a few photos, recompute a model pass** → M28 "Re-run passes" in
  the image view, *not* the sweep.

<!-- moved verbatim from CLAUDE.md (line 3680) on 2026-10-19 -->
### Scheduling + replica-mode maintenance (2026-07-17)

`maintenance-sweep` **runs nightly on the NAS from the NAS user's own crontab**
(not root's), at **18:30 host-local** (Pacific) — verified on the box
2026-09-19 with `crontab -l`:

```cron
0 4 * * *   … flock -n /tmp/photo-ingest.lock      … ingest-incoming --no-colors …
30 18 * * * … flock -n /tmp/photo-maintenance.lock … maintenance-sweep --apply --no-colors …
```

Both are `flock`-guarded, so a run that overlaps the previous one exits instead
of stacking. Host-local time means it drifts an hour against UTC with DST
(01:30 UTC in summer, 02:30 in winter); the `CRON_TZ=UTC` variant was never
adopted. Log: `/var/log/photo-maintenance.log`. `--recluster` /
`--dedup-photos` stay OFF (recluster clears `ignored_clusters`; dedup DELETEs
photos). (This section used to say the entry was "not yet installed", in root's
crontab, at 01:00 UTC, with ingest at 03:00 — all four were stale.)

**The sweep's `match_faces` stage is STRICT-ONLY.** It used to run the temporal
matcher too, every night, over shoots nobody had reviewed. On 2026-09-19 it
swept a shoot ingested that afternoon and wrote **466 temporal labels (Calvin
382, Ellie 149)** beside 65 strict ones — on exactly the kind of shoot where
temporal is ~4% accurate and tags one kid across both teams. Temporal is now an
explicit opt-in: `maintenance-sweep --match-temporal`, `match_temporal=True` on
`run_maintenance_sweep`, `"match_temporal": true` on the API. Hand-run
`match-faces --temporal` is unchanged. Tests: `tests/test_maintenance_match_faces.py`.

**FIXED (2026-09-19) — the nightly match/unmatch churn loop.** The sweep's log
showed `match_faces` and `resolve_dups` applying the *same* number night after
night — 6134/6134, 6129/6129, 5262/5262, 5858/5327 — while the unmatched pool
grew **202k → 213k over eight runs**. Unmatched by source at the time: NULL
204,043 · `dedupe_unmatched` 5,531 · `rejected` 3,720.

Mechanism: `resolve_dups` keeps one face per (photo, person) and unmatches the
loser as `dedupe_unmatched`, which is **deliberately still matchable** (the
loser may be a *different* person — filtering it wholesale was considered and
rejected, it would freeze ~9.6k faces out of matching forever). But nothing
remembered *which* person the face lost to, so the matcher re-applied the same
label and the resolver stripped it again. Forever, on the sweep's heaviest CPU
stage.

The fix is **`face_person_exclusions(face_id, person_id, reason, created_at)`**
(schema v31): remember the *pairing*, not the face. Load it once per run into
`{face_id: {person_id}}` (`faces.load_match_exclusions`) — never a query per
face; the matchers walk ~213k rows on a 4-core N100. Both matchers consult it,
the same precedent as `MATCHABLE_SQL`, and a test asserts both do.

Things not to simplify back:

- **The two matchers act on it differently, and unifying them is a bug.**
  **Strict** falls through to the NEXT best person inside the same tolerance
  (it walks `match_face`'s already-sorted in-tolerance list past a barred
  person). It has no ambiguity rule, so the runner-up clears exactly the bar
  the winner did — measured, 0 of 600 sampled excluded faces actually fell
  through. **Temporal** ranks the UNFILTERED field and, if the winner is
  barred, leaves the face unmatched. Its `min_gap` check (`TEMPORAL_MIN_GAP`
  0.08) is a judgement about the *whole field*: filtering the barred person
  out first hands the runner-up an unopposed win the real field never gave
  them. A face 1.20 from Calvin and 1.25 from Ellie is a 0.05 gap this matcher
  **refuses** as ambiguous; bar Calvin and Ellie has no one left to be
  ambiguous against. Measured on the 260k-face replica: of 1,500 sampled
  backfill-eligible faces, **34 gained a new label** under fall-through (30
  where the old code chose a different person, 4 where the gap guard had
  rejected outright) and **17 of the 34 were Calvin↔Ellie** — extrapolating to
  the ~5,664 backfilled rows, ~128 new *persistent* temporal labels and ~64
  sibling swaps. The old churn self-cancelled nightly; that terminates in a
  wrong label, which is strictly worse. This is the ArcFace sibling limit the
  face-integrity section describes, so **don't "simplify" temporal to match
  strict.** Tests:
  `test_temporal_exclusion_cannot_promote_an_ambiguous_sibling`,
  `test_temporal_matcher_does_NOT_fall_through_to_the_next_best_person`.
- **One write primitive**: `db.unmatch_faces_as_duplicates` does snapshot +
  exclusion + null, and `cli.py`'s `resolve-duplicate-persons` /
  `dedupe-person-faces` and `maintenance._stage_resolve_dups` all call it. They
  each had their own copy of snapshot-then-null before; a fix to one would have
  left the other churning. `test_the_duplicate_unmatch_write_lives_in_one_place`
  fails if a copy comes back.
- **A human overrules it.** `db.assign_face_to_person` clears the pairing
  unless the source is `strict`/`temporal` (`AUTO_MATCH_SOURCES`) — so assign,
  bulk-assign, `face_edit`, correct-face and the manual-assignment import all
  get it from the one primitive. `correct-face` had to be *moved onto* that
  primitive: it wrote a bare `UPDATE faces SET person_id`, so it neither
  cleared the exclusion nor stamped a source, leaving a hand-corrected face
  barred from the person the human had just named. Merge-accept and the
  replica's `_mirror_face_labels` write their own UPDATE, so they clear it
  explicitly. Restores
  (`restore-unmatched-faces`, `bulk_unmatch.restore`) clear it too: the
  restored pairing's exclusion is spent, and leaving it would make the face
  permanently unmatchable to the person just restored.
- **`face_state.apply_face_state` refuses excluded pairings on the TARGET** —
  the replica recomputes off a synced copy, so its file can carry exactly the
  pairing the NAS has since stripped, restarting the loop by the long route.
  Guarded on the overwrite path too: one-person-per-photo is not something a
  force flag should violate.
- **Rows cascade** on both `face_id` and `person_id` (`PRAGMA foreign_keys` is
  ON in `PhotoDB.__init__`), so they never outlive a deleted face or person.

The ~5,531 already-stuck faces are backfilled from the `face_dedupe_undo`
snapshots **inside the v31 migration** (idempotent, `INSERT OR IGNORE`, does
nothing when that on-demand table was never created), so the fix takes effect
on deploy with no operator step. Re-check or re-run by hand with
`photosearch backfill-face-exclusions [--apply]`.

**That backfill runs inside the PhotoDB constructor, so it is doubly guarded.**
`INSERT OR IGNORE` does **not** suppress a FOREIGN KEY violation (`PRAGMA
foreign_keys` is ON), and `face_dedupe_undo` rows go stale on both sides — a
live snapshot had 2,940 of 8,919 orphaned `face_id`s. So the query joins
**both** `faces` and `persons` to skip dangling ids, and the call is wrapped in
a logged `try/except`: a failed backfill leaves some faces churning, which is
the bug we already had, whereas an exception escaping `_init_schema` would stop
the web server and every CLI container from starting, leave the schema at 30,
and be retried forever. Never make it silent, and never let it raise.

Similarly `unmatch_faces_as_duplicates` stamps only the faces that actually had
a person (it iterates the collected pairs, not the caller's id list) — passing
it an already-unmatched face must not overwrite a `rejected` marker with the
matchable `dedupe_unmatched`, nor inflate the count the sweep reports.

Tests: `tests/test_face_person_exclusions.py` (24 cases, including the
end-to-end `test_match_then_resolve_twice_is_stable` that reproduces the
6134/6134 loop — verified to go red when the matchers stop consulting the
table, along with the four per-matcher cases), plus
`test_v30_db_migrates_to_v31_face_person_exclusions` in `tests/test_db.py`.

**Replica mode is now gated.** A sweep writes to whatever `PHOTOSEARCH_DB`
points at, and `sync-replica.sh` replaces the replica's DB wholesale — so a
replica-side apply used to be silently destroyed on the next sync. Now:

- `apply=true` on the replica with the NAS unreachable → **503, refuses to run**.
- `colors` / `dedup_photos` / `requeue` → **400** in replica mode (colors needs
  pixels the replica doesn't have; dedup DELETEs; requeue is a no-op because
  the fleet claims from the NAS).
- Everything else: pre-flight fingerprint → auto-sync if drifted → compute →
  one push at the end. Push modes (`maintenance_sync.push_mode`): cheap
  deterministic stages are **re-triggered** on the NAS; `stacking` is
  **transferred** as rows; `match_faces` / `recluster` are **face_state** —
  the replica exports a face-state file (`photosearch/face_state.py`, the same
  code as the `export-face-state` / `apply-face-state` CLI pair) and POSTs it
  to `POST /api/admin/maintenance-apply-face-state`. The leg order is transfer
  → face_state → trigger, and a landed match adds `resolve_dups` to the
  triggers.
- face_state apply on the NAS applies **only what the replica recomputed**:
  match → additive person fill (skips `dedupe_unmatched`), no cluster copy (a
  sync-time cluster id would revert a split made on the NAS since); recluster
  → cluster ids for unmatched faces, stale ids cleared on named faces and on
  faces absent from the file, and `ignored_clusters` **remapped** — a new
  cluster stays ignored when >50% of its faces sat in an ignored cluster
  before. Known window: a label cleared on the NAS between sync and push
  (`/clear` sets `match_source=NULL`, indistinguishable) is re-filled by a
  match push.

`maintenance_runs` (schema v29) holds the per-stage watermark on both machines.
The `/status` Replica card and `/admin_maintenance` show per-stage drift;
**"Replica ahead — unpushed"** means local work will be lost on the next sync.

**Redeploy the NAS before the replica** (same shape as M28's `/mirror-fields`
gotcha above) — a stale NAS 404s `GET /api/admin/maintenance-fingerprint`,
which makes the replica's pre-flight gate 503 with "NAS unreachable" and
blocks **every** replica apply, even though the real problem is just an old
NAS image.

The same applies to the face stages: a NAS image older than the face_state
push mode 404s `POST /api/admin/maintenance-apply-face-state`, so the push
reports `http_404` for `match_faces` / `recluster` and the locally computed
face work is overwritten by the next replica sync.

Module: `photosearch/maintenance_sync.py`. Spec:
`docs/superpowers/specs/2026-07-17-maintenance-push-up-design.md`.
