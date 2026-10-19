# API endpoints

Read when you need a route's path, parameters or response shape.

<!-- moved verbatim from SKILL.md (line 320) on 2026-10-19 -->
## API Endpoints (50+)

<!-- moved verbatim from SKILL.md (line 322) on 2026-10-19 -->
### Search & Photos
- `GET /api/search` — Combined search (CLIP semantic, color, face, place, date, filename)
  - Params: q, person, color, place, limit, offset, min_score, min_quality,
    sort_quality, sort, tag_match, date_from, date_to, location, match_source,
    camera, category, visual_tag, keyword, min_aesthetic, min_day_aesthetic,
    style_tag, media (`all|photo|video`; default hides Live clips + unreadable
    videos), sort `steadiest|least_noisy` (videos), and — only while
    `PHOTOSEARCH_VIDEO_SUBJECT_SHARE` is on, else **400** — min_focus_share,
    min_focus_seconds, subject, min_subject_share, sort `focus_desc|subject_desc`
    (see "Search performance" for how combinations execute)
  - **Name extraction from `q`** — `search.py:_extract_persons_from_query`
    pulls registered person names out of the free-text query and turns
    them into AND-intersected person filters via the same `result_sets`
    logic that handles `person=`. `?q=Calvin and Ellie` matches photos
    with both people; connector tokens (`and`, `with`, `&`, `,`) are
    stripped, the residual goes to CLIP. Matching is case-insensitive,
    word-bounded, longest-first ("Matt Newkirk" beats "Matt"). Lookbehind
    `(?<!-)` keeps `-Calvin` as a CLIP exclusion rather than a filter.
- `GET /api/photos/{id}` — Photo detail
- `GET /api/photos/{id}/thumbnail` — Cached thumbnail
- `GET /api/photos/{id}/full` — Full resolution
- `GET /api/photos/{id}/preview` — Preview size

<!-- moved verbatim from SKILL.md (line 345) on 2026-10-19 -->
### Video (built 2026-10-17, deployed 2026-10-18 — backfill not started — CLAUDE.md "## Video")
- `GET /api/photos/{id}/full` — for a video: single-range HTTP Range (206/416);
  `?purpose=worker` is throttled (`PHOTOSEARCH_VIDEO_STREAM_MBPS`, default 30
  MB/s) and capped at `PHOTOSEARCH_VIDEO_WORKER_STREAMS` (4) → 503 Retry-After.
  Thumbnail/preview serve the uploaded poster (Live → the still, unprobed /
  unreadable → SVG placeholder).
- `POST /api/videos/{id}/assets` — worker upload of poster/keyframe/crop JPEGs
  (base64 JSON, staged per scan generation)
- `GET /api/videos/{id}/frames/{name}` — keyframes / rep frames (`?v=`, `?g=`)
- `GET /api/videos/{id}/subjects` — subjects, per-person shares (names via
  `faces.person_id` at read time), timeline, dense presence
- `GET /api/videos/subject-labels` — Subject typeahead
- `GET /api/admin/unreadable-videos` — unreadable / ZIP-wrapped list
  (`/admin/maintenance` card)
- `PUT /api/videos/{id}/proxy?version=N` — worker upload of the H.264
  playback proxy (raw `video/mp4` stream, not base64; 409 on the replica —
  upload to the NAS directly)
- `GET /api/videos/{id}/proxy` — serves the playable copy with HTTP Range
  (`?v=` pins a version); the replica forwards Range to the NAS
- CLI: `photosearch register-videos [--folder 2026] [--limit N] [--apply]`
  (NAS; walk + insert only, dry run read-only)
- CLI: `photosearch unwrap-zip-videos [--year 2016] [--folder F] [--mbps N]
  [--apply --audit PATH] [--undo PATH]` (NAS; fixes ZIP-wrapped video files
  in place — see CLAUDE.md "## Video" ZIP unwrap section). Run **before**
  `register-videos` for ZIP-heavy years (2016-17).
- Worker passes (`--native` only — ffprobe/ffmpeg aren't in the Docker worker
  image): `video-probe`, `video-proxy` (needs ffmpeg + ffprobe; D12 fallback
  for clips the browser can't decode as-is), `video-scan`, `video-subjects`
  (role `subject`, `PHOTOSEARCH_LLM_SUBJECT_MODEL`, `--llm-subject-model`),
  `video-describe` (strict role `video`, `PHOTOSEARCH_LLM_VIDEO_MODEL`,
  `--llm-video-model`, no fallback). `clear-pass` accepts all five. Pass
  order: `video-probe, video-proxy, video-scan, ..., video-subjects, ...,
  video-describe`. Runbook: Step 8 of `docs/plans/video-support.md`.
- Eval: `evals/video_eval.py`; labelling `/eval/models?tab=vsubject|vquality`.

<!-- moved verbatim from SKILL.md (line 380) on 2026-10-19 -->
### Faces
- `GET /api/faces/groups` — All face groupings (paginated, similarity-sorted).
  **Content filters** (any combination): `date_from`, `date_to`, `location`, `q`
  (semantic/CLIP), `person` — restrict to groups with a face in matching photos,
  ranked by matching-photo count (`sort=relevance`), singletons shown. Reuses
  `search_combined` for `q`/`person`/`location`, direct date/location SQL
  otherwise; matching photo-ids go into a temp table joined into the group
  queries (`_face_filter_photo_ids` in `web.py`). Backs the `/faces`
  "Filter clusters:" bar — turns cluster triage into a searchable workflow.
- `GET /api/faces/group-info?cluster_id=N | person_id=N` — Single group metadata
  (used by `/faces?cluster_id=N` auto-open; works for hidden singletons + unloaded pages)
- `GET /api/faces/group/{type}/{id}/photos` — Photos for a person or cluster.
  Takes the SAME content filters as `/api/faces/groups`
  (`date_from`/`date_to`/`location`/`q`/`person`/`camera`) and applies them to
  **every** group type. They used to be accepted and silently ignored for
  person/cluster, so opening a name from a one-day `/faces` view loaded that
  person's whole library (Calvin: 18,417 photos). The response carries
  `filtered` + `total_unfiltered` so the UI can say "29 of 18,417" and offer to
  widen — without the denominator, "narrowed to 29" and "this person only has
  29 photos" look identical on screen. Capped at `limit` (default 10,000); the
  UI labels a truncated widen rather than silently delivering less than the
  button promised.
- `GET /api/faces/label-health` — **who carries a suspect `match_source`,
  library-wide**, ranked by count, with the days it is concentrated on. Pure SQL
  (instant); `?person=` adds the distance calibration for one person. The bar is
  the p90 of that person's own **strict** distances, not their manual ones —
  manual labels cluster in the sessions you hand-labelled, so their spread is
  artificially tight. `calibratable: false` + `blocker` is a finding (a person
  with no manual labels can't be judged automatically). Backs the 📊 Label
  health panel.
- `GET /api/faces/unmatch-preview` — candidate faces for a bulk unmatch
  (`person`, `sources` default `temporal`, `min_dist`, date range), sorted
  farthest-from-the-person first; unmeasured faces sort last. **400 without a
  date scope.** Read-only — applying goes through `bulk-assign`.
- `GET /api/faces/verify-labels` — **labelled faces that look more like someone
  else in the same scope** (M32). Scope required. Backs the 🔍 Verify labels
  panel. Needs no eps and no distance threshold: each face is compared against
  every other person present and calibrated against how close those two people
  genuinely get. Returns `margin`, `separation`, `decisive`, `alibi_excluded`
  and the nearest reference face-ids for both identities so the reviewer sees
  the strongest case for each side by side. `?rivals_only=1` keeps only the
  findings where **another face in the SAME photo claims the label better** —
  a person appears in a photo at most once, so those are near-certain, and
  unlike the rest they name the right face (`rival.rival_face_id`) so the
  reviewer swaps instead of adjudicating. The rival may be **unlabelled**,
  which is what the margin test structurally cannot see. `?sources=` picks
  which labels you are ASKED about and defaults to `manual,strict,merge_review`
  — `temporal` is ~4% accurate on these shoots and buried the real findings
  26:1 on 2026-09-12 (108 findings → 4). Use `sources=all` to include it; the
  hidden counts come back in `stats.hidden_by_source`. See
  `docs/plans/face-label-verification.md`.
- `POST /api/faces/review-team` — **SSE**; the `review-faces` CLI as an endpoint
  (M32). Samples the jersey colour below each face, keeps the ones matching the
  team, DBSCANs the survivors into nameable groups. Previews are fetched via
  `web._preview_bytes` (in-process), NOT over HTTP — the NAS would otherwise
  issue ~800 requests to itself, and going through `_get_or_create_preview`
  lets the replica reuse its on-disk preview cache. Terminal events:
  done/cancelled/fatal.
- `GET /api/faces/crop/{face_id}` — Face crop image (disk-cached)
- `GET /api/faces/face-detail/{face_id}` — Photo id + bbox + dimensions for
  overlays. `image_width`/`image_height` are stored EXIF-oriented (swapped for
  orientation 5-8), matching the bbox + preview space so overlays line up.
- `POST /api/faces/{face_id}/assign` — Assign face to person
- `POST /api/faces/{face_id}/clear` — Clear assignment. Writes
  `match_source='rejected'` (`faces.REJECTED_MATCH_SOURCE`), **not NULL** —
  NULL is indistinguishable from never-matched, so the next `match-faces` run
  re-applies the very person just removed. Both matcher candidate queries
  filter it out via the shared `faces.MATCHABLE_SQL`.
- `POST /api/faces/bulk-assign` — `{face_ids, person_name}`; a null/empty name
  CLEARS (and marks `rejected`, as above). The write primitive behind the
  person inspector, team review and label verification.
- `POST /api/faces/bulk-collect` — Bulk assign unassigned faces
- `POST /api/faces/ignore` / `POST /api/faces/unignore` — Ignore/restore clusters
- `POST /api/faces/clusters/{id}/split` — Re-run DBSCAN with tighter eps on
  one cluster (body `{eps, min_samples, dry_run}`)
- `GET /api/faces/person/{id}/inspect` — sub-structure of ONE person's faces
  for spotting wrong ones. Takes `date_from`/`date_to`: DBSCAN runs on the
  **scoped** faces (so the groups are "the different kids that day") but the
  reference core is built from the person's **full trusted set** — scope the
  core too and you measure the suspect faces against themselves. A scoped call
  with no explicit `eps` defaults to **0.90 / min_samples 2**; the library's
  0.50 groups *nothing* on one day (measured on 212 real faces: 0.50 → 0
  clusters/212 noise, 0.90 → 20 clusters/100 grouped).
- `GET /api/faces/label-conflicts` — clusters holding two or more DIFFERENT
  named people. Scope required. **Sweep the eps**: on 2026-09-12 the default
  0.80 reported zero and the real mislabel only appeared at 0.85/0.95.
- `GET /api/faces/manual-assignments` / `POST /api/faces/import-assignments`
  — Export/import manual face-to-person assignments

<!-- moved verbatim from SKILL.md (line 469) on 2026-10-19 -->
### Merge suggestions (M18)
- `GET /api/faces/suggestions` — Read cached JSON (via `PHOTOSEARCH_SUGGESTIONS_JSON`
  or `/data/suggestions.json`), augments each side with up to 4 sample_face_ids,
  filters rows whose source cluster has been collapsed
- `POST /api/faces/suggestions/regenerate` — Re-run the engine with params
  `{centroid_cutoff, min_pair_cutoff, max_members, min_group_size,
  include_ignored}` and overwrite the cached JSON
- `POST /api/faces/merges` — Apply a merge, body `{source: {type:"cluster",id},
  target: {type:"cluster"|"person",id}}`. Returns `{moved_face_count}`. Stamps
  `faces.match_source='merge_review'`.

<!-- moved verbatim from SKILL.md (line 480) on 2026-10-19 -->
### Persons
- `GET /api/persons` — List persons with photo counts

<!-- moved verbatim from SKILL.md (line 483) on 2026-10-19 -->
### Collections
- `GET /api/collections` — List all
- `POST /api/collections` — Create
- `GET /api/collections/{id}` — Detail (includes photos with sort_order)
- `PUT /api/collections/{id}` — Rename
- `DELETE /api/collections/{id}` — Delete
- `POST /api/collections/{id}/photos` — Add photos
- `POST /api/collections/{id}/photos/remove` — Remove photos

<!-- moved verbatim from SKILL.md (line 492) on 2026-10-19 -->
### Stacks (Burst/Bracket Groups)
- `GET /api/stacks` — List all
- `GET /api/stacks/{id}` — Detail with members
- `PUT /api/stacks/{id}/top` — Set top photo
- `DELETE /api/stacks/{id}` — Delete stack
- `POST /api/photos/{id}/unstack` — Remove from stack
- `POST /api/stacks/{id}/add` — Add to stack
- `GET /api/photos/{id}/nearby-stacks` — Find nearby stacks
- `POST /api/stacks/detect` — Run detection synchronously. Body
  `{time_window_sec?, clip_threshold?, max_stack_span_sec?, clear?, dry_run?}`
  (all optional, defaults match the CLI: 5.0 / 0.05 / 10.0). Returns
  `{stacks_created, photos_stacked, cleared, dry_run, duration_seconds, params}`.
- `POST /api/stacks/detect/stream` — SSE variant used by the status-page
  Stacking form. Same body; streams `start` / `progress` / `done` /
  `cancelled` / `fatal` events. Client disconnect flips a `threading.Event`
  that `stacking.py` checks inside its hot loops (≤1s abort latency on 500k
  libraries). Progress events carry `phase` ∈ `{scan, load_embeddings,
  pairs, group, save, cleared}` plus phase-specific counters.

<!-- moved verbatim from SKILL.md (line 511) on 2026-10-19 -->
### Review (Culling)
- `GET /api/review/folders` — Available folders (returns `{path, max_date}` objects, sorted by most recent photo first)
- `GET /api/review/run` — Run culling algorithm
- `GET /api/review/load` — Load saved selections
- `POST /api/review/toggle/{id}` — Toggle photo selection

<!-- moved verbatim from SKILL.md (line 517) on 2026-10-19 -->
### Ingest batches (`/batches`, M "ingest batch" — see CLAUDE.md for the traps)
- `GET /api/batches` — pure SQL: active sweep (if any) + the batch list
  (`include_dismissed`).
- `GET /api/batches/{id}` — derived readiness state (`batch_state.batch_state`),
  memoized 30s behind a non-blocking lock; carries `computed_at`/`stale`, or
  `computing: true` when nothing is memoized yet and the lock is held elsewhere.
- `POST /api/batches/{id}/dismiss` / `POST /api/batches/{id}/ready` — lifecycle writes.
- `POST /api/batches/{id}/jobs` — record job intent `{steps, job_kind}`
  (used by a remote fleet launch in replica mode); steps validated against
  `batch_state.STEP_ORDER`.
- `POST /api/batches/register` — `{directory, source}`, manual registration.
- `POST /api/admin/batch-advance` — SSE; runs the batch's NAS steps (stacking,
  normalize_aesthetics, strict match_faces, resolve_dups, warm_crops,
  rank_measure) in order, id-scoped. `photosearch batch-advance --batch N
  [--apply]` is the CLI equivalent (dry-run by default). `rank_measure` is
  `batch_state.OPTIONAL_STEPS` — it does NOT gate `ready`, but `next_action`
  still offers `advance_nas` while it is the only step left, which is what
  makes the button able to run it. It writes
  `dirname(PHOTOSEARCH_DB)/rank_shoot_<date>.json`, the cache
  `scripts/rank_shoot.py --date <date>` reads for its selection phase
  (measurement itself is `photosearch/rank_measure.py` — `scripts/` is not in
  the Docker image).
- `POST /api/admin/batch-launch-fleet` — `{batch_id, count}`; launches
  `run-workers.sh --native` scoped to the batch's directory for exactly the
  worker passes it still needs (`batch_state.fleet_launch_passes`). 409 if a
  pass already reads `queued`/`running` for the batch.
- In replica mode (`PHOTOSEARCH_NAS_URL` set), every `/api/batches/*` route
  proxies to the NAS with no local fallback (502 if unreachable).

<!-- moved verbatim from SKILL.md (line 546) on 2026-10-19 -->
### Google Photos
- `GET /api/google/status` — OAuth status (configured + authenticated)
- `GET /api/google/authorize` — Start OAuth flow
- `POST /api/google/exchange-code` — Manual code exchange
- `GET /api/google/callback` — OAuth callback
- `DELETE /api/google/disconnect` — Revoke + clear tokens
- `POST /api/google/albums` — Create album
- `POST /api/google/upload-status` — Check which photos are already uploaded to an album
- `POST /api/google/upload` — Upload with SSE streaming progress

<!-- moved verbatim from SKILL.md (line 556) on 2026-10-19 -->
### Utility
- `GET /api/stats` — Database statistics for status page. Response includes
  `photos`, `clip_embedded`, `faces`, `persons`, `described`, `quality_scored`
  (+ `quality_stats`), `concepts_analyzed`, `tagged`, `stacks`, `stacked_photos`,
  `verify_passed`/`verify_failed`/`verify_regenerated`.
- `GET /api/stats/activity` — Hourly index activity, 3-day window
- `GET /api/stats/errors` — Recent indexing errors
- `GET /api/worker/status` — (see Worker System below) polled by the status
  page Workers panel every 5s for per-pass queue depth + active claims

<!-- moved verbatim from SKILL.md (line 566) on 2026-10-19 -->
### Map view
- `GET /api/photos/geojson` — compact point dump of every GPS-bearing
  photo. Returns `{count, points: [[id, lat, lon, source, year], ...]}`
  — tuple format keeps 50k points under 500KB gzipped. `source` is
  `'exif' | 'inferred' | None`; `year` is parsed from `date_taken[:4]`
  or None. Consumed by `/map` (Leaflet + markercluster UMD, no build
  step). **Must be declared before `/api/photos/{photo_id}` in
  `web.py`** or FastAPI's path matcher takes `geojson` as a photo id
  and returns 422.

<!-- moved verbatim from SKILL.md (line 576) on 2026-10-19 -->
### Manual bulk geotagging
- `GET /api/geotag/folders` — folder summary with no-GPS/inferred/exif
  counts per folder, sorted by no-GPS descending. Hides fully-tagged
  folders unless `?include_fully_tagged=true`. Folder path is derived
  from `os.path.dirname`-style rfind of `/` in `filepath`.
- `GET /api/geotag/folder-photos?folder=...&show_inferred=false` —
  photos in one folder. By default returns `gps_lat IS NULL` rows only;
  with `show_inferred=true` also returns `location_source='inferred'`
  rows so users can correct M19 misfires. Never returns `exif` rows.
- `GET /api/geotag/known-places?q=...` — distinct `place_name` values
  from the library matching q (case-insensitive substring), sorted by
  photo count descending. Feeds the "library" section of the /geotag
  typeahead.
- `GET /api/geocode/search?q=...&limit=5` — forward-geocode via
  Nominatim (public OSM endpoint) with a persistent 30-day cache in
  `geocode_cache`. User-Agent: `"local-photo-search/1.0 (…)"`.
  Response shape `{results: [{display_name, lat, lon, country,
  admin1, admin2, locality, type, importance}], source: 'cache'|
  'nominatim'}`.
- `POST /api/photos/bulk-set-location` — body `{photo_ids, lat, lon,
  place_name, overwrite?}`. Writes `gps_lat/gps_lon/place_name/
  location_source='manual'/location_confidence=NULL` in one transaction.
  `overwrite=false` (default) keeps existing GPS; `overwrite=true`
  replaces any source including exif. Returns
  `{updated_count, skipped_count}`.

<!-- moved verbatim from SKILL.md (line 602) on 2026-10-19 -->
### Inferred geotagging (M19)
- `POST /api/geocode/infer-preview` — read-only. Body takes
  `{window_minutes, max_drift_km, min_confidence, cascade, max_cascade_rounds}`
  (all optional, defaults match the CLI: 30 / 25.0 / 0.0 / true / 10).
  Returns candidate counts, `confidence_buckets` (fixed 5-entry list),
  `hop_distribution` (sorted), `skipped` reasons, and up to 10 sample
  candidates with `thumbnail_url` + pre-reverse-geocoded `place_name`.
- `POST /api/geocode/infer-apply` — write path. Same body plus
  `confirm: true` (400 without it). Reverse-geocodes the full candidate
  set and writes in one transaction with
  `WHERE id=? AND gps_lat IS NULL` so rows a concurrent indexer just
  populated are never overwritten. Returns
  `{updated_count, rounds_used, duration_seconds}`.

---
