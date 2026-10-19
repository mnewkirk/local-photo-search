---
paths:
  - "photosearch/maintenance*.py"
  - "photosearch/geocode.py"
  - "photosearch/geonames_rich.py"
  - "photosearch/stacking.py"
  - "photosearch/face_state.py"
  - "sync-replica.sh"
---

# Maintenance — traps

Full history and measurements: docs/notes/maintenance.md (cite by heading). Runbooks: .claude/skills/photo-search/reference/nas-operations.md.

- **The memory cost is per-ROW, not the KDTree — and it OOM-killed the NAS.**
  `RGeocoder.load()` keeps one Python dict per row, so row count is the memory
  bill. Keep `_MIN_POPULATION = 1` (drops population-0 hamlets). In
  `build_rich_dataset` the `is_poi` check must run BEFORE the population gate,
  or park/peak labels vanish silently.
- **Stream the CSV.** `get_rich_geocoder` passes the open file as
  `RGeocoder(stream=f)`. Never go back to `io.StringIO(f.read())` — it holds
  the CSV twice in memory.
- **`--force` vs `--refresh-source`.** `download-geonames --force` re-filters
  the files on disk; only `--refresh-source` downloads again. Keep them
  separate. The durable fix is a SQLite R-tree (`docs/plans/geocode-rtree.md`),
  which would let the population gate go.
- **Vec0 orphan cleanup.** `clip_embeddings` and `face_encodings` are vec0
  virtual tables: `ON DELETE CASCADE` never reaches them. After deleting
  photos/faces, run `cleanup-orphans`. `add_clip_embedding` must stay
  DELETE+INSERT — vec0 ignores `INSERT OR REPLACE` and raises UNIQUE.
- **Non-image rows & the clip-claim infinite re-claim.** clip is on the
  attempts ledger: claim and count predicates carry `db._CLIP_NOT_EXHAUSTED`,
  unloadable images come back as failure rows, and a successful embedding is
  its own marker (no ledger row). Never drop the cap from a pass.
- **ZIP-wrapped `.JPG` rows.** Live Photo bundles saved as `.JPG` are ZIPs.
  `index.is_real_image()` gates row creation — keep it on every row-creating
  path. Remove old rows on the NAS with `purge-nonimage-photos`; `--apply`
  requires `--audit` (deletes are undoable only by re-import).
- **Maintenance sweep & duplicate pruning (M25 + M28).** One orchestrator,
  `maintenance.run_maintenance_sweep`, behind the CLI, the SSE API and the UI.
  It is one dependency-ordered, missing-only backfill; dry-run
  (`apply=False`) is the default and writes nothing. Every new stage needs a
  "still missing" gate so an enriched library is nearly free.
- **Opt-in extras do not narrow the sweep.** `do_dedup` prepends the
  destructive `dedup_photos` stage; `do_recluster` appends DBSCAN and clears
  `ignored_clusters`. Every default stage still runs alongside them.
- **`stacking` is missing-only.** The stage calls
  `stacking.run_incremental_stacking` with the `stacking_seen` ledger and
  writes a diff (unchanged stacks keep id and hand-set top). Never call
  `db.clear_stacks()` or an unscoped `detect_stacks`/`run_stacking` from it,
  and never pass `photo_ids=[]` — `detect_stacks` reads that as the whole
  library. Write the ledger LAST so an abort stays dirty.
- **The sweep's `match_faces` stage is STRICT-ONLY.** Temporal matching is
  opt-in only (`--match-temporal`, `match_temporal=True`). Never turn it on
  by default.
- **FIXED — the nightly match/unmatch churn loop.**
  `face_person_exclusions` remembers the (face, person) pairing that
  `resolve_dups` stripped. Load it once per run with
  `faces.load_match_exclusions`, never one query per face. Both matchers must
  consult it.
- **The two matchers act on it differently, and unifying them is a bug.**
  Strict falls through to the next in-tolerance person. Temporal ranks the
  UNFILTERED field and leaves the face unmatched if the winner is barred —
  filtering first defeats `TEMPORAL_MIN_GAP` and promotes ambiguous siblings.
- **One write primitive.** `db.unmatch_faces_as_duplicates` does snapshot +
  exclusion + null for the sweep's `_stage_resolve_dups` and the CLIs. Never
  add a second copy. It stamps only faces that had a person, so a `rejected`
  marker is never overwritten.
- **A human overrules it.** `db.assign_face_to_person` clears the exclusion
  unless the source is in `AUTO_MATCH_SOURCES`. Any write that bypasses it
  (merge-accept, `web._mirror_face_labels`, restores) must clear the
  exclusion itself.
- **`face_state.apply_face_state` refuses excluded pairings on the TARGET**,
  including on the overwrite path — a replica file can carry a pairing the
  NAS has since stripped.
- **That backfill runs inside the PhotoDB constructor, so it is doubly
  guarded.** `backfill_exclusions_from_dedupe_undo` joins `faces` and
  `persons` (`INSERT OR IGNORE` does not suppress FK errors) and is wrapped in
  a logged try/except. It must never raise out of `_init_schema` and never
  fail silently.
- **Replica mode is now gated.** `sync-replica.sh` swaps the replica DB
  wholesale, so local writes are lost unless pushed. `apply` with the NAS
  unreachable → 503. `colors` / `dedup_photos` / `requeue` → 400 on the
  replica. Push modes (`maintenance_sync.push_mode`): cheap stages are
  re-triggered on the NAS, `stacking` is transferred as rows, `match_faces` /
  `recluster` go as `face_state` to `POST
  /api/admin/maintenance-apply-face-state`. Leg order: transfer → face_state →
  trigger. A new stage needs a push mode.
- **face_state apply applies only what the replica recomputed.** A match push
  is an additive person fill (skips `dedupe_unmatched`) with no cluster copy;
  a recluster push remaps `ignored_clusters`.
- **Redeploy the NAS before the replica.** An old NAS 404s `GET
  /api/admin/maintenance-fingerprint` (pre-flight reads as "NAS unreachable",
  blocking every replica apply) and `maintenance-apply-face-state` (face work
  lost on the next sync).
- **Scheduling + replica-mode maintenance (2026-07-17).** The nightly
  `maintenance-sweep --apply --no-colors` cron is `flock`-guarded. Keep
  `--recluster` and `--dedup-photos` off in it.
