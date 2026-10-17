# Video support (mp4/mov): faces + main-focus share, description, quality

Status: **planned** (2026-10-17). Built by plan-debate: two planners drafted
independently, critiqued each other over two rounds, and converged on every
contested point. The owner then decided three things (recorded in "Decisions"):
**skip Live Photo clips**, **the whole clip is the share denominator**, and
**all work other than indexing runs on the desktop, never the NAS** — the NAS
only registers rows, stores results and serves files. That last decision moved
the probe and the poster frame off the NAS into a new `video-probe` worker pass.

## Why, and what the owner asked for

The owner wants videos in the photo server, analysed for three things:

1. **Faces** — who is in the video, and especially **how much of the video each
   person is the main focus**. In a sports clip a kid in the background should
   register as present, but the question is "what share of this clip is about
   Calvin".
2. **What the video is about** — what is happening, what we are seeing.
3. **Quality** — subjects in focus, noise, shake.

Plus a rigorous evaluation of candidate models/approaches before any is chosen,
in the same style as the photo passes (`docs/plans/model-eval-harnesses.md`).

## The corpus (measured on the NAS, 2026-10-17)

Videos already live in the library tree but have **no DB rows**: `ingest.py`
moves `VIDEO_EXTENSIONS` as companions and never indexes them.

| fact | value |
|---|---|
| files | **5,577** (3,796 mp4, 1,739 mov, 35 avi, 4 mts, 3 m4v) |
| size | ~70 MB average, ≈ **400 GB** |
| by year | 2016–18 ≈ 1,650 · 2022–24 ≈ 1,200 · **2025: 1,405 · 2026: 471** |
| 2026 sources | phone-nicole 226, phone-matt 97, ILCE-7RM6 39, KODAK PIXPRO 29 |
| unreadable | **111 of 400 sampled (28%)**: `moov atom not found`, all from **2016–2017** (Takeout-era iPhone MOV / Android mp4) |
| codecs (readable, 60-sample) | hevc 32, h264 13 |
| duration (289 readable of 400) | **p50 7.5 s, p75 25 s, p90 54 s, p95 68 s, p99 161 s, max 1,751 s** (one 29-min 2016 clip) |
| duration buckets | 42% under 4 s · 21% 4–15 s · 30% 15–60 s · 7% 1–5 min · 2 files over 5 min |
| total footage | ≈ 30 h library-wide; **clips over 60 s hold 51% of it** |

42% of files are under 4 s, mostly **iPhone Live Photo clips** (paired to
their still and skipped, D3). Long videos are rare by count but half the
footage by time, so the design has to be cheap on a 3 s clip and correct on a
5-minute match. ~1,400 files are probably unreadable and must be **recorded
once, never re-decoded**.

## Facts found in the code that shape the design

- **cv2 decodes video on both machines with no new dependency.** The desktop
  venv's `cv2` 4.13 reports `FFMPEG: YES`; the NAS image installs the same
  `opencv-python` wheel. There is **no `ffmpeg` binary in the Dockerfile** and
  none on the desktop host; `ffmpeg`/`ffprobe` exist only on the NAS *host*
  (`/usr/bin`). After D19 the NAS never decodes, so only the desktop wheel's
  HEVC support matters (spike).
- **Desktop torch is a CUDA build with no usable device.** CLIP and any
  torch model run on CPU there; the AMD GPU is reachable only through
  `onnxruntime-rocm` (InsightFace) and LM Studio. A native video-input VLM via
  HF transformers is therefore out unless the owner sets up ROCm torch (D7).
- **`purge-nonimage-photos` would delete video rows.** Its default scope is
  "no CLIP embedding" and it then calls `is_real_image`, which old MOVs (no
  leading `ftyp`) fail. Must be guarded in step 1; FK cascade would take faces
  and collection links with the row.
- **`/api/photos/{id}/full` hard-codes `image/jpeg`** and in replica mode
  reads the whole file into RAM. Wrong for a 700 MB clip.
- **Every image pass's claim predicate would pick up video rows** (`faces` is
  `NOT EXISTS faces`, `describe` is `description IS NULL`, `verify` is
  `description IS NOT NULL AND verified_at IS NULL` — it would claim a video
  the moment a video description lands). "Which media a pass applies to" must
  be one shared fact, not nine hand edits.
- `describe._to_openai_message` already turns `images: [...]` into a
  multi-`image_url` content list, so N frames in one request need no transport
  code. LM Studio splits context across parallel slots, so frames × resolution
  must be part of the describe eval and a 400 "context exceeded" is a config
  failure, never an answer.
- Ingest already SHA-256s every companion file before the companion branch, so
  new videos get `file_hash` for free. Video folder routing uses mtime, so a
  Live Photo `.MOV` can land in a different dated folder than its `.HEIC`.
- InsightFace runs at `det_size=(640,640)`: a 1920 px frame is shrunk ~3×, so
  a 20 px background face is not found. Mostly the right bias for "main focus";
  detection resolution is an eval axis, not a default.
- CI mocks `cv2`, `insightface`, `onnxruntime`, `torch`, `scipy`. The tracker
  and the focus metric must be pure numpy (greedy association, no
  `linear_sum_assignment`), in a module that imports cv2 lazily.
- The temporal face matcher's ~4% accuracy is about carrying names across
  photos up to 30 minutes apart. Linking boxes 0.25–0.5 s apart inside one
  clip is a different, far stronger signal; that precedent does not forbid a
  tracker.
- Two photo attempts to turn a pixel scalar into a shipped tag failed their
  evals (`sharp`/`blurry` FROZEN, `motion-blur` RETIRED;
  `docs/plans/sharpness-measurement.md`, 0.86 in-sample → 0.55 holdout). Quality
  ships numbers first; a tag needs the same P ≥ 0.8 @ R ≥ 0.5 holdout gate.

## Architecture in one paragraph

**A video is a `photos` row with `media_type='video'`** (NULL = still, so no
backfill of existing rows). Its probe and analysis live in two side tables,
`video_meta` and `video_tracks`. One map, `db.PASS_MEDIA`, says which passes
see videos; image passes never do, and `category-content`/`keywords` run on
video descriptions unchanged. **The NAS runs no decoder and no prober**: it walks the tree, inserts rows,
stores what the workers upload, and serves files with Range support. Three new
worker passes do everything on the desktop: **`video-probe`** (ffprobe over
HTTP Range against the NAS plus one first-frame decode: duration, codec,
rotation, dates, GPS, the unreadable verdict, and the poster at three sizes,
with Live Photo pairing decided server-side from the submitted duration and
date), **`video-scan`** (non-LLM: one rate-capped download,
dense cv2 decode, InsightFace on sampled frames, a numpy tracker, per-person
main-focus share, CLIP mean embedding, quality measurements, keyframes uploaded
back to the NAS) and **`video-describe`** (new LLM role `video`, K keyframes in
one multi-image request). **Each within-video identity becomes one `faces`
row**, so `/faces`, clustering, the strict matcher, verify-labels, the rival
test and the one-person-per-photo invariant apply to videos unchanged, and the
share is grouped by `faces.person_id` at query time so a relabel moves it with
no recompute.

## The main-subject share metric (concrete)

**The owner's requirement is the SUBJECT of the video, not the faces in it**:
"whether that's a dog or a mountain or a person". Faces alone cannot see a dog,
a mountain, or a kid facing away from the camera. So the metric has two layers
that the eval scores separately and together:

1. **Subject layer (sparse, VLM).** One keyframe per **segment** (3 s segments,
   capped at 40 per video, so a 29-min clip gets 44 s segments). Each keyframe
   goes through the existing **`subjects.ground_photo_subjects`** (qwen2.5-vl
   grounding, already in production for photos: returns `[{label, bbox}]` for
   the main subject(s), `[]` for a pure landscape) with the prompt extended to
   return a **category** (`person | animal | landscape | object | activity`)
   and, for `landscape`, a label anyway ("mountain lake"). The **primary
   subject** of a segment is the returned box with the highest salience
   (area × centre weight, below); `[]` is "no subject". Subjects are linked
   across segments into one identity per video by normalized label plus CLIP
   similarity of the box crop, so "dog" in segment 2 and "the dog" in segment 5
   are one subject.
2. **Face layer (dense, no VLM).** InsightFace at 2–4 fps with the numpy tracker
   (below) gives every **person** an identity (a `faces` row, so `/faces`
   labelling applies) and a dense presence/salience signal.

**Joining them**: a `person` subject box that contains a tracked face box in
that keyframe, or within ±1 s via the track's coasting, takes that face's
identity. A person box with no face in it stays an "unidentified person"
subject (a player facing away). This is what the body-tracker candidate F3 was
for; the VLM box now plays that role for free, so F3 is demoted to a fallback.

**Shares.** Each segment carries its duration as weight. For subject *S*:

```
subject_share(S) = Σ_seg w_seg · [primary(seg) == S] / Σ_seg w_seg   # whole-clip denominator (D10)
focus_seconds(S) = Σ_seg w_seg · [primary(seg) == S]
presence_share(P) = dense face-track presence / duration            # people only
```

"Whole clip" means the video's full running time; the sampling only decides how
finely it is estimated. A 60 s clip where Calvin is the subject for 30 s scores
**0.50**; a 10-min match where he is the subject for 2 min scores 0.20 and
`focus_seconds` 120. Both are stored and both are filterable, because on a long
video "≥ 2 minutes of Calvin" is the natural question and on a burst "mostly
Calvin" is.

**Within-segment refinement for people (variant S2).** When the VLM names a
person as a segment's primary subject and that person's face track is present
for only part of the segment, credit only the tracked portion. The eval decides
whether this beats whole-segment credit (S1).

**Face salience** (used to pick among several person boxes, to break ties, and
as the eval's no-VLM baseline F1): for a detection with apparent size
`A = sqrt(w·h) / min(W, H)` and centre offset `(dx, dy)`:

```
S = A × C,   C = exp(−r² / (2σ²)),  σ = [0.35],  r² = (dx/(W/2))² + 0.5·(dy/(H/2))²
```

Eligible for focus at `A ≥ [0.03]`; presence at `det_score ≥ [0.5]` and short
edge ≥ 16 px. Tracks: greedy association (IoU ≥ 0.3 with the last box, or
ArcFace L2 ≤ 0.9 to the running mean within 1.5 s); gaps ≤ 1 s coasted; tracks
merged into identities by complete linkage at L2 ≤ [0.9], never when they
overlap in time. Bracketed numbers are tunables swept on the dev split.

**Stored.** `video_subjects` (`photo_id`, `subject_ix`, `label`, `category`,
`face_id` nullable → `faces`, `focus_s`, `focus_intervals` JSON, `rep_t`,
`rep_frame`, `rep_bbox`, `n_segments`). `video_tracks` (per person identity:
`face_id`, `presence_s`, `intervals`, `rep_t`, `rep_frame`, `n_det`,
`mean_salience`). Person shares are read `GROUP BY faces.person_id` so a
relabel in `/faces` moves the share with no recompute; non-person subjects are
searchable by label (`subject=dog`, `min_subject_share`).

**Cost.** From the 400-file sample, ≈ 8.6 segments per readable video → ≈ 36k
grounding calls library-wide at ~2–3 s each on qwen2.5-vl ≈ **20–30 GPU
hours**, two or three nights of fleet time. The 40-segment cap is what keeps
the 29-min clip from costing 580 calls.

## Steps

Estimates are for one engineer who knows this codebase. **Ship** points are
where the owner gets value.

### Step 0 — Spike and fixtures (1 d)

- Pull ~12 real files to a desktop fixtures dir: iPhone HEVC portrait MOV with
  rotation, an HDR/Dolby Vision clip, a Live Photo MOV + HEIC, a Takeout MP4, a
  `moov atom not found` file, an ILCE-7RM6 4K file, a Kodak PIXPRO file, a 720p
  2016 file, one `.avi`, one `.mts`.
- Install `ffmpeg`/`ffprobe` on the WSL desktop (`apt`; there is none today).
  The Dockerfile is **not** touched: the NAS never probes or decodes (D19).
- Measure on the desktop: cv2 open/rotation/decode fps (1080p and 4K HEVC);
  **bytes read by `ffprobe <http url>` and by a cv2 first-frame grab over HTTP
  Range** for three file shapes — faststart (moov first), moov-at-end, and
  no-moov (the unreadable 25%; ffprobe may read to EOF looking for `moov`, so
  measure it, and check whether a 1 MB head + 1 MB tail `moov` search can
  short-circuit it); HEVC playback in the owner's browsers (Chrome/Edge on
  Windows, iOS Safari) from a static `<video>` page.
- Measure the per-camera offset between `mvhd` creation time and sibling photo
  `date_taken` (feeds the date policy).
- Files: this doc gains a spike table. Nothing in `photosearch/` yet.
- **Gate:** the probe pass must cost well under one whole-file download per
  video on average, or the probe is folded into `video-scan`'s single
  download and unreadable files are detected there instead (the row still
  stores the error either way).

### Step 1 — Schema v35, media isolation, registration, unreadable handling (4 d)

**Schema v35** (`db.py`, migration in `_init_schema`):

- `photos.media_type TEXT`, **NULL means still** — no `UPDATE photos` backfill
  (a full-table rewrite under the NAS write lock is the percentile-refresh
  incident's shape). Partial index `idx_photos_media_video ON photos(id) WHERE
  media_type IS NOT NULL` in `_SEARCH_INDEXES`.
- `video_meta` (`photo_id` PK → `photos` CASCADE): `duration_s`, `fps`,
  `codec`, `container`, `width`, `height`, `rotation`, `bitrate`, `has_audio`,
  `probe_json`, `probe_version`, `probe_error`; `kind` ('clip' | 'live'),
  `live_of_photo_id`; `poster_t`; `scan_version`, `scan_error`, `scanned_at`;
  `shake`, `noise`, `focus_consistency`, `quality_json`.
- `video_tracks` (`id`, `photo_id` CASCADE, `face_id` → `faces` ON DELETE SET
  NULL, `ident_ix`, `t_start`, `t_end`, `n_det`, `presence_s`, `focus_s`,
  `focus_intervals`, `rep_t`, `rep_frame`, `mean_salience`); indexes on
  `(photo_id)` and `(face_id)`.

**One media map.** `db.PASS_MEDIA = {pass: "still" | "video" | "any"}` with
shared fragments `STILL_SQL = "media_type IS NULL"` / `VIDEO_SQL`. Every branch
of `get_unprocessed_photos` / `count_unprocessed_photos` (scoped and unscoped)
appends its pass's fragment: still = `clip`, `faces`, `quality`, `aesthetics`,
`describe`, **`verify`**, `category-visual`; any = `category-content`,
`keywords`. `batch_state` derives `total`/`eligible` from the same map. A test
iterates **every** `WORKER_PASSES` entry and asserts a video row is never
offered to a still pass, a still never to a video pass, and `batch_state`
agrees with the claim predicate (the `MATCHABLE_SQL` shared-fragment pattern).
A future pass without a `PASS_MEDIA` entry fails that test.

**Pixel readers that must skip videos** (guard + test each): `cli.py
purge-nonimage-photos` (**must land in this step** — the one data-loss risk),
`stacking.detect_stacks`, `sharpness_backfill` candidates, `rank_measure`,
`face_review`, `upscale`, `split_export`, `book_export`, `shutterfly_export`,
`google_photos` upload, `subjects`/`aesthetics` if they read pixels.

**Registration (NAS, "indexing" only):**

- **`photosearch/video_registry.py`**: `register_video(db, path, file_hash)`
  inserts the `photos` row via `add_photo` (`media_type='video'`, filename,
  `filepath`/`folder`, `date_taken` = file mtime **provisionally**, replaced by
  the probe pass) and an empty `video_meta` row (`probe_version IS NULL`).
  No ffprobe, no decode, no hashing on the NAS. `register_videos(...)` is the
  backfill driver: walks the photo root for `VIDEO_EXTENSIONS` with no row;
  dry run by default (`mode=ro`); chunked commits every 25; `os.nice(10)` and
  IDLE ioprio out of habit, though a walk + insert is cheap.
- **Ingest** (`ingest.py`): after a video companion moves, call
  `register_video` with the hash ingest already computed; folders that
  received only videos join `new_dirs` so `register_batch` sees them. Routing
  keeps using mtime (the probe date arrives later and only fixes
  `date_taken`; **no refile of existing files**, D15). DB-hash dedup now
  applies to videos too.
- **CLI**: `register-videos [--folder F] [--limit N] [--apply]`,
  `envvar="PHOTOSEARCH_DB"`.

**Probe (desktop, the `video-probe` worker pass — step 2b):**

- `photosearch/video_probe.py` runs `ffprobe -print_format json -show_format
  -show_streams` against the NAS URL (`/full?purpose=worker`, Range-capable,
  so ffprobe reads only the header and the `moov` box) and parses the JSON.
  CI feeds canned JSON fixtures; ffprobe is never on CI. A head+tail `moov`
  pre-check (two 1 MB Range reads) short-circuits the no-`moov` files if the
  spike shows ffprobe otherwise reads them to EOF. `PROBE_VERSION` constant.
  Why ffprobe and not cv2 or a hand-rolled atom parser: it is the tool that
  produced the 15/60 figure, so the stored `probe_error` matches it by
  construction; cv2 cannot read creation time, Apple
  `com.apple.quicktime.creationdate` with its offset, ISO6709 GPS, or the
  rotation display matrix; and it covers `.avi`/`.mts` natively.
- The same pass grabs the **first decoded frame** with cv2 over HTTP (no
  seek) and uploads the poster as thumb/preview/full-size JPEGs, so the NAS
  never resizes anything either. A paired Live clip gets no poster (it shows
  its still); an unreadable file gets none (static placeholder).
- **A probe error is STORED** on submit (`probe_error` + `probe_version` set):
  the row exists, no later pass ever claims it, nothing re-probes unless
  `PROBE_VERSION` bumps (the CLIP re-claim / sharpness rule).
- **Live Photo pairing is decided server-side in the submit branch** from the
  submitted duration and capture time: same stem as a still in the same folder
  or a ±1-day sibling folder, `duration ≤ 4 s`, capture times within 3 s →
  `kind='live'`, `live_of_photo_id`. Pure SQL, so it counts as indexing.
- **Hash policy.** New ingests store the hash ingest already computed. The
  backfill never hashes on the N100: the worker hashes while it streams the
  download in `video-scan` and the server fills `file_hash` when NULL.
  Unreadable rows are never downloaded whole, so they keep `file_hash` NULL;
  their dedup is by path only, which is acceptable for files nothing can play.
- **Date policy** (D14): Apple `creationdate` (local wall time with offset) →
  else `mvhd.creation_time` treated as UTC, corrected by the per-camera offset
  measured in step 0 → else mtime.

Tests: `test_video_probe.py` (fixture JSON: ok, moov-missing, truncated,
rotation 90/180/270, Apple date/GPS), `test_video_registry.py` (Live pairing
positives/negatives, stored error not retried, dry run writes nothing),
`test_pass_media.py`, the v34→v35 migration test, and
`tests/test_search_indexes.py` must still EXPLAIN-match (adding `AND
media_type IS NULL` keeps the partial `need_*` indexes usable because the query
predicate implies the index predicate — verify, don't assume).

Verify: `./scripts/test-like-ci.sh` green; a dry run on a replica copy reports
≈ 5,577 files and per-year counts matching the table above (probe errors and
Live pairs appear only after step 2b); apply on one NAS folder and
`/api/worker/status?passes=…` shows its video rows in the `video-probe` queue
and in **no** still-pass queue; `purge-nonimage-photos` dry run reports 0
video rows.

### Step 2 — Playback, grid, modal, media filter (2.5 d) — **SHIP A**

- `web.py` thumbnails/previews for a video are **never generated on the
  NAS**: paired Live → the still's thumbnail; probe error → a static
  "unreadable video" placeholder; not yet probed → a static "video" placeholder;
  otherwise the poster JPEGs the probe pass uploaded, cached under the usual
  `{id}_thumb.jpg` / `{id}_preview.jpg` names so every grid and the replica's
  thumbnail mirror work unchanged.
- `/full` for a video: correct mime (`video/mp4`, `video/quicktime`),
  `FileResponse` Range passthrough on the NAS, and **`?purpose=worker` streams
  through a token-bucket generator capped at `PHOTOSEARCH_VIDEO_STREAM_MBPS`
  (default 30 MB/s)**. The 2026-09-19 disk starvation was a ~50 MB/s bulk read;
  the backfill is 400 GB of exactly that shape and must never run at LAN
  speed (400 GB at 30 MB/s ≈ 3.7 h). Range requests from ffprobe/cv2 in the
  probe pass go through the same route. Replica mode: a streaming proxy that
  forwards `Range` and returns 206 with `iter_content`, never `.read()`.
- `POST /api/videos/{id}/assets` (worker-authenticated like `submit-results`):
  atomic write of poster/keyframe/crop JPEGs under
  `/data/video_frames/{pid}/v{version}/`, used by the probe and scan passes.
- Frontend: `PS.PhotoModal` renders `<video controls preload=metadata
  poster=…>` for `media_type==='video'`, hides face overlays, adds a Download
  link for `.avi`/`.mts`; grid card gets a "▶ 0:42" badge once the duration is
  known; `index.html` gets **Media: All / Photos / Videos** (`media=` through
  `search_combined` → `_scope_clause`, plus a `COMBOS` case in
  `test_search_indexes.py`). Default search hides `kind='live'` and
  probe-error rows (D2, D3); an "Unreadable videos" list on
  `/admin/maintenance` links to them. `request_intent._RULES` for new
  endpoints; `check-frontend-refs.js` passes.

Verify: an iPhone portrait MOV plays upright on desktop Chrome and iPhone,
seeking works, the replica plays the same clip through the proxy;
`request_log.jsonl` thumbnail p95 unchanged for photo grids; no
`/full?purpose=worker` read exceeds the cap.

**Value:** every registered video is browsable by date/folder/filename and
playable, with a generic poster until step 2b fills the real one.

### Step 2b — `video-probe` worker pass (2 d)

- `worker._process_video_probe`: for each claimed video, the head+tail `moov`
  pre-check, then `ffprobe` over HTTP Range, then one cv2 first-frame grab over
  HTTP, poster resized to thumb/preview/full on the desktop and uploaded via
  the assets endpoint. Claim size 25; the pass is cheap, so `-n 2` is plenty.
  A transport failure is omitted (`_TRANSIENT`); an ffprobe or cv2 failure on
  a complete read is the **probe verdict**, submitted as `probe_error`, not a
  `failures` row — it is the result the row exists to store.
- `worker_api.VideoProbeResult` + `video_probe_results` on `SubmitRequest`.
  The branch goes through `_SubmitOutcome`: write `video_meta` probe columns,
  rewrite `date_taken` per the date policy (D14), `gps_*` with
  `location_source='exif'`, `camera_make/model`, `image_width/height` as
  **display** dims after rotation, then run the Live-pairing SQL. A transient
  lock defers; a non-lock error spends the attempt.
- Claim predicate: `media_type='video' AND probe_version IS NULL`. Later video
  passes require `probe_error IS NULL AND kind != 'live'`.
- `_ALL_PASSES` / `WORKER_PASSES` / `rerun.ALL_PASSES` / `batch-flow.js` /
  `cli.py valid_passes` / fleet checkboxes gain `video-probe` before
  `video-scan`; `batch_state` counts probe errors as done with detail "N
  unreadable" (as `sharpness`), and `video-scan` depends on `video-probe`.
- Tests: `test_video_probe.py` (fixture JSON: ok, moov-missing, truncated,
  rotation 90/180/270, Apple date/GPS; pre-check on synthetic head/tail
  bytes), `test_video_probe_submit.py` (Live pairing positives/negatives,
  stored error not retried, date policy, deferral on lock).
- Verify: on one NAS folder the pass fills ≈ 25% `probe_error`, the Live
  count, and real posters; NAS bytes/sec during the run stays far below the
  cap; a second fleet run claims nothing.

### Step 3 — `video-scan` worker pass (6 d) — **SHIP B**

- **`photosearch/video_analysis.py`** (pure numpy, CI-tested): sampling and
  keyframe schedules, greedy tracker, identity merge, salience, frame focus,
  shares, run-length intervals, the segment summariser the eval uses, and the
  quality measures over per-frame inputs — **shake** = RMS of the high-pass
  global trajectory (per-frame phase-correlation shifts minus their 1 s moving
  average, as % of frame width; pans drop out), **noise** = median Immerkær σ
  on luma with flat-region masking, **focus_consistency** = share of main-focus
  frames whose face-crop sharpness is ≥ 0.5 × the video's own p90 (relative
  within the video, never a library threshold), **exposure** = clipped
  highlight/shadow fractions.
- **`photosearch/video_scan.py`** (worker-only, lazy cv2): decode once,
  sequentially; faces through a new **`faces.detect_faces_bgr(ndarray)`**
  extracted from `detect_faces` with a parity test (the array variant must
  **raise** on error — a detection exception is a failure row, never "nobody
  here"); **K keyframes** (1 for d ≤ 2 s, 3 for ≤ 10 s, rising to 8 at ≥ 60 s,
  at uniform quantile centres nudged to the sharpest frame within ±0.5 s);
  CLIP mean of the L2-normalised keyframe embeddings, renormalised, into
  `clip_embeddings`; the poster is **replaced** by the highest-salience
  keyframe (the probe pass's first frame is often a black lead-in); one
  native-res representative frame JPEG per identity, **plus the face-crop
  JPEGs at the sizes `face_crop.py` serves**, so the NAS never decodes a frame
  even for `/faces` tiles.
- **`worker.py`** `_process_video_scan`: download through
  `/full?purpose=worker`, hashing as it streams, size-checked against
  Content-Length. A transient truncated download is omitted (`_TRANSIENT`);
  cv2 failing to open a complete file is a **terminal `scan_error`** (stored,
  version set, counted "N unreadable" in the batch step); an exception
  mid-analysis is a `failures` row. Claim limit 1 video per batch; `-n 2`.
- **`worker_api.py`**: `VideoScanResult` + `video_scan_results` on
  `SubmitRequest`; the branch goes through `_SubmitOutcome`: upload assets
  through the step 2 endpoint, atomic under
  `/data/video_frames/{pid}/v{SCAN_VERSION}/` (≈ 2–4 MB per video with crops,
  ≈ 12–18 GB library-wide, D5), then in one transaction delete old
  `video_tracks` and their `faces`/`face_encodings`, insert the new `faces`
  rows (bbox in representative-frame coordinates, `det_score`, encoding = mean
  of the top-5 detections), insert `video_tracks`, write `clip_embeddings`
  (DELETE+INSERT — vec0 ignores OR REPLACE), update `video_meta`, fill
  `file_hash` if NULL. Transient lock defers; non-lock error spends the
  attempt. **A re-scan carries human labels**: old `faces` rows with
  `match_source` in (`manual`, `merge_review`, `rejected`) are matched
  one-to-one to new identities at L2 ≤ 0.6 (D13); unmatched labels are logged
  in the response, never silently dropped. `_ALL_PASSES` gains `video-scan`;
  the claim response adds `media_type` and `duration_s`.
- `face_crop.py` serves video face crops straight from the uploaded crop
  files; `warm_crops` is a no-op for videos (already warm on upload). `batch_state.py` +
  `batch-flow.js`: `video-scan` right after `faces`, `_OUTPUT_MISSING`
  (`scan_version IS NULL OR < N`, readable, not live),
  `_REMAINING_FILTERS_ATTEMPTS=True`; `match_faces`/`warm_crops` also depend on
  it; scan errors count done with detail "N unreadable" (as `sharpness`).
  `rerun.py` (`ALL_PASSES`, `_MIRROR_COLUMNS` + a `video_meta`/`video_tracks`
  mirror), `cli.py valid_passes`, the `admin_maintenance.html` fleet checkboxes,
  PhotoModal re-run boxes. `cleanup-orphans` learns stale
  `video_frames/{pid}/v*` directories.

Tests: `test_video_analysis.py` worked examples (one dominant face → ≈ 1; two
equal → 0.5 each; subject leaves at half time → 0.5; a one-frame miss doesn't
flip focus; overlapping tracks never merge); `test_video_scan_submit.py`
(deferral on lock, failure spends an attempt, terminal decode error stored and
never re-claimed, assets atomic, manual-label carry-over, one `faces` row per
identity); the `detect_faces_bgr` parity test.

Verify: run on one sports folder and one family folder on the NAS; `/batches`
shows `video-scan` completed; `/faces?date_from=…` shows video faces in
clusters with working crops; `person=Calvin` returns his videos; "kids on a
trampoline" returns a known video; measure s/video, NAS bytes/sec and the NAS
web p95 in `request-stats` during the run.

**Value:** videos found by person and by text. Dense presence and face
salience are computed here; the **subject** share needs step 3b and is
**hidden in the UI until step 4's eval picks the config** (D16).

### Step 3b — `video-subjects` LLM pass (3 d)

- `photosearch/video_subjects.py`: segment schedule (3 s, cap 40), one
  keyframe per segment taken from `video-scan`'s uploaded frames (so
  `video-scan` must upload a per-segment keyframe set, not just K ≤ 8;
  ≈ 100–200 KB each at 768 px, still ≈ 2–4 MB per typical video), the
  grounding call via `subjects.ground_photo_subjects` with the category
  extension, primary-subject choice, cross-segment linking, the face-track
  join, S1/S2 credit, shares and intervals. Pure-Python scoring over cached
  JSON so it is CI-testable; the VLM call is the only non-deterministic part.
- `describe.py`: role `subject` → `PHOTOSEARCH_LLM_SUBJECT_MODEL`
  (default qwen2.5-vl-7b-instruct, the grounding model photos already use),
  `PASS_ROLES["video-subjects"]="subject"`, per-role `max_tokens` 256, the
  usual guards: an unparseable answer retries once then is a `failures` row;
  `[]` is a real answer ("no subject").
- `worker._process_video_subjects` + `worker_api.VideoSubjectsResult` through
  `_SubmitOutcome`: replace `video_subjects` rows for the photo in one
  transaction, log one `generations` row per video (`text_type='subjects'`,
  the segment JSON) with `effective_model(role='subject')`.
- Pass order: after `video-scan`, before `video-describe`, and its segment
  summary is handed to the describe prompt ("subjects over time: 0–12 s dog,
  12–40 s Calvin, …"). `batch_state`/`batch-flow.js`/`rerun`/`cli` mirrors.
- Tests: `test_video_subjects.py` — linking "dog"/"the dog" into one subject,
  the face join through coasting, S1 vs S2 credit, `[]` segments in the
  denominator, the 40-segment cap, a person box with no face stays
  unidentified.
- Verify: on the two NAS test folders, a dog video reads "dog 0.9", a soccer
  clip reads a per-kid breakdown that matches the eye, a landscape pan reads
  "mountain lake" with no person rows.

### Step 4 — Video eval harness and blind labelling (6 d engineering, ≈ 2.8 h owner)

General rules, inherited from `model-eval-harnesses.md` and the visual-tags
eval: pin the role (`model_eval.pin_role_model`); wrap every LLM call in
`Recorder` (a `TransportError` is never cached; a truncation is `unusable`; a
context-overflow 400 is cached as `rejected` and screens the config out);
labels are files under `evals/model-evals/video/` (git-ignored), never DB rows;
originals from a local cache (≈ 90 × 70 MB ≈ 6 GB) so neither runs nor
labelling need the NAS; one model loaded at a time, fleet stopped; the page is
**blind** (no variant/model name in any response, model output hidden before
the label is saved, `PS.PhotoModal` not used); three-state labels where a
verdict can honestly be uncertain; a seeded 15-video **recheck** set measures
owner self-agreement (κ < 0.4 → fix the definition in `LABELLER_NOTES`, don't
score it); a seeded, stratified **2/3 dev, 1/3 holdout** split where tunables
are chosen on dev and **only holdout numbers decide winners and gates**; ratios
under 3 observations print `n<3`.

**Sample** (`evals/video_eval.py sample`, DB `mode=ro`, seed 1). Strata come
from pre-scan facts only (folder, sibling photos' categories and named faces,
camera, duration, year, Live pairing), never from a candidate's own output:

| stratum | n | definition |
|---|---|---|
| sports-long | 20 | ≥ 20 s, folder whose photos are ≥ 50% sports categories |
| sports-short | 10 | same folders, < 20 s |
| family-multi | 15 | phone, sibling photos have ≥ 2 named people |
| single-subject | 10 | phone, one named person dominant |
| camera-nonsports | 8 | ILCE / Kodak / other camera, not sports |
| unpaired-short | 5 | ≤ 4 s, no Live pair |
| no-people | 7 | sibling photos have no faces |
| old-era | 5 | 2016–2018, ≤ 720p or Takeout |
| live-paired | 10 | auto-paired; owner confirms pairing only |
| **labelled total** | **90** | 60 dev / 30 holdout |
| probe-check | 40 | 20 probe-error + 20 ok: automatic ffprobe comparison, no owner time |

**Main subject (people and non-people).** Face detections are cached per
video (npz) and grounding answers per keyframe (JSON), so rule variants are
rescored with no re-detection and no relabelling.

| id | approach | new model? |
|---|---|---|
| F0 | InsightFace per frame, per-detection identity, no tracking — people only | no |
| F1 | F0 + tracks + coasting + identity merge; main focus by face salience — people only | no |
| S1 | VLM grounding per segment keyframe (qwen2.5-vl), primary = largest salience box, face join for identity, whole-segment credit (**default**) | no (already the photo subject model) |
| S2 | S1 with within-segment credit refined by the person's face track | no |
| S1-m | S1 with the grounding model swapped: minicpm-v-4_5, gemma-4-12b-qat | no |
| F3 | S1 + ONNX person detector for the no-face person case | **yes**, fallback only (D6) |

Rule grid swept on dev: centre σ ∈ {0.25, 0.35, 0.5, off}; A_min ∈ {0.02,
0.03, 0.05}; link threshold for subjects; face-join window {0, ±1 s}. Identity
for every candidate uses the production strict matcher against
`face_references` + manual labels **as of a frozen replica snapshot**.

Owner's "Video subject" tab: watch the clip; answer **"Who is in this video?"**
(multi-select chips with typeahead over `persons`, plus "someone else",
nothing pre-selected); then label each **segment** (3 s, **at most 12 per
video** — long eval videos are labelled on a 12-way coarsening of the
production segments) with the **main subject**: a person from the first
answer, "someone else", **a non-person subject** picked from a blind, shuffled
chip list of every candidate's labels for that segment plus free text, "no
subject", or "can't tell". `LABELLER_NOTES`: "what the camera operator is
framing or following: a person, an animal, a thing, or the scenery; if two are
framed equally, can't tell is fine". About 75 min for 90 videos.

Non-person labels are scored by normalized label match (lowercase, lemmatized,
articles stripped); when the owner typed free text, the comparer model
(gemma-4-26b-a4b, not a candidate) judges "same subject?" and the owner
spot-checks 20 of those judgements.

Metrics (holdout headline, dev shown): segment main-subject accuracy ("can't
tell" excluded, per stratum, and split **person / non-person**); top-subject
agreement per video; **share MAE** per (video, subject); retrieval
precision/recall of `subject_share ≥ 0.5` against the owner's; named-presence
precision/recall; **error diagnosis** per wrong segment (not detected /
detected but unnamed or misnamed / wrong primary / wrong category); s/video.

Winner thresholds: among S1, S2, S1-m the highest holdout segment accuracy; a
gap **< 5 points is a tie** and the simpler or cheaper wins. **F1 must be
reported beside it**: if S1 does not beat F1 by ≥ 5 points on the person
strata, the VLM has not earned its 20–30 GPU hours for people and ships only
for non-person subjects. **F3** is tried only if the diagnosis shows ≥ 10% of
sports segments lost to "person box, no face, misattributed". **Ship gate for
the subject filter** (step 6): top-subject agreement ≥ **0.80**,
named-presence precision ≥ **0.95**, holdout segment accuracy ≥ **0.70** on
both the person and non-person splits, and share MAE ≤ **0.15**. Fallbacks:
presence passes but subject fails → ship presence % only; non-person passes
but person fails → ship non-person subjects and person presence only.

**Description.** Frames come from the production keyframe sampler on the
cached original.

| id | approach |
|---|---|
| D0 | **poster-only baseline**: production `describe_photo` (qwen3.5-9b) on the poster — a video treated as a still |
| D1-m | multi-image, K = 6 @ 768 px, one request, video prompt, for m ∈ {qwen3.5-9b, minicpm-v-4_5, gemma-4-12b-qat, qwen2.5-vl-7b} |
| D1-K | the D1 winner at K ∈ {3, 10} and @ 512/1024 px (context sensitivity) |
| D2 | storyboard: K = 6 tiled 3×2 into one 1536×1024 image, single-image path |
| D3 | per-keyframe `describe_photo` + gemma text summary into one description |
| D4 | native video input (Qwen2.5-VL / MiniCPM-V video mode): **excluded by default** (D7) |

Stage A screens automatically: answered %, unusable/truncated %, **rejected
(context) %**, format leaks, words p50/p90, s/video (solo). More than 10%
unanswered/unusable/rejected screens out. At most 4 finalists go to the owner,
D0 always among them. Stage B (owner): the existing **Differences** machinery
(`describe_eval.py disputes`-style, comparer gemma-4-26b-a4b which is not a
candidate, ≤ 6 checkable disagreements per video on 50 non-Live videos, the
`<video>` playing beside them) plus one new blind 3-state question per distinct
description, keyed by `text_sha`: **"Does it say what happens over the clip,
not just one moment?"** (yes / partly / no). About 60 min. Metrics:
right-when-answered on disputed points, wrong claims per video, the temporal
yes-rate, a per-video sign test against D0 and the runner-up. Winner: highest
right-rate; **< 0.05 is a tie**, broken by (1) same model as `describe` (no
extra load), (2) lower s/video. **Gate against D0:** the winner must beat
poster-only by ≥ 0.20 on the temporal yes-rate **or** ≥ 0.05 right-rate at
sign p < 0.1; otherwise production uses D0 and multi-frame hasn't earned its
keep.

**Quality.** Candidates: **Q-m** (measured shake, noise σ, focus consistency,
absolute Tenengrad max for comparison only, exposure clipping; logistic
thresholds fitted on dev only); **Q-v** (VLM rubric, K frames → JSON
`{subject_in_focus: yes|partly|no, shaky: yes|no, noisy: yes|no}` for
minicpm-v-4_5, qwen2.5-vl-7b, gemma-4-12b-qat — sparse frames can't show
jitter, so Q-m is expected to win shake and the VLM might win focus; the eval
decides); **Q-mv** (per-flag OR/AND). Owner's "Video quality" tab, all 90
videos, four three-state questions: main subject in focus (judged on a
native-resolution frame grab in `PS.Loupe`), shaky, noisy/grainy, keeper
(keep / meh / delete). About 30 min. Metrics: per-flag holdout P/R with
"debatable" as neither; keeper Spearman ρ with a Fisher-z CI. **Gate: a flag
becomes a tag only at holdout P ≥ 0.8 with R ≥ 0.5**, the gate `blurry`
failed. If nothing passes, step 7 ships numbers and sorts only.

**Registration checks** (automatic): the probe pass over HTTP Range vs
`ffprobe` on the local original for the 40 probe-check files must agree 100%
on readable/unreadable (the Range path must not misread a moov-at-end file); every probe-ok file opens in
cv2; **Live pairing precision must be 10/10** on the owner-confirmed 10 pairs
+ 5 near-misses (same stem, Δt 3–10 s) — wrongly hiding a real video is the
costly error; tighten Δt or require Apple's content identifier if it fails.

**Owner time**: focus ~60 min, Differences ~45 min, temporal ratings ~15 min,
quality ~30 min, recheck ~15 min, Live pairs ~5 min ≈ **2.8 h**. (A per-frame,
per-person label grid was considered and rejected: 6–8 h for a worse signal.)

Files: `evals/video_eval.py` (`sample | fetch-originals | run-faces |
run-describe | run-quality | disputes | report | agreement`); video stores and
a detection cache in `model_eval.py`; video tabs + Range-served originals + a
native frame grab in `model_eval_api.py`; `eval_models.html` tabs **Video
focus | Video quality**, and Differences shows a `<video>`;
`eval_api.LABELLER_NOTES` entries. Tests `tests/test_video_eval.py`: page is
blind, labels keyed by `(pid, seg_version, seg_ix)`, `TransportError` never
cached, context-overflow cached as `rejected`, holdout split deterministic.

### Step 5 — `video-describe` pass (2.5 d) — **SHIP C**

- `describe.py`: role `video` (`PHOTOSEARCH_LLM_VIDEO_MODEL`),
  `PASS_ROLES["video-describe"]="video"`, `_OPENAI_MAX_TOKENS["video"]=768`,
  **no silent fallback to the VISUAL model** (a missing role var is a startup
  error in the fleet env); `describe_video(frame_paths, ts, model, prompt,
  raise_unusable)` builds one message with K images through
  `_to_openai_message` and the same `_description_problem` guards; the prompt
  states the frame count and timestamps and asks what *happens*.
- `rerun.FLEET_ROLE_MODELS["video"]` = the eval winner, plus `_PASS_LLM` and
  `admin_api._fleet_env`. `worker._process_video_describe` fetches keyframes
  from `GET /api/videos/{id}/keyframes`; `UnusableAnswer` → failure row;
  `_provenance_kwargs` logs `generations` with `effective_model(role='video')`.
  `worker_api` writes `description` and calls
  `invalidate_description_dependents` so `category-content`/`keywords` run
  next; `verify` stays still-only (D18). `video-describe` sits **right after
  `describe`** in the pass order so a shared model loads once; depends on
  `video-scan`. `PASS_ROLES` is pinned by the existing test.
- Verify: description written and `generations.model_used` = the effective
  model; categories/keywords appear on the next pass; stale-description
  tooling ignores videos for `verify`; a deliberately tiny `max_tokens`
  produces a failure row, not stored truncated text.

### Step 6 — Surface the subject share (3.5 d) — **SHIP D**

- `search_combined`: `min_focus_share` and **`min_focus_seconds`** (both need
  a person; restrict to videos), `subject=<label>` with `min_subject_share`
  for non-person subjects, sorts `focus_desc` / `subject_desc`, composed into
  `_compose_scope`; `tools._build_filter_sql` + `_CURATION_FILTER_PROPS` so
  Ask and MCP get `media`, `min_focus_share`, `min_focus_seconds`, `subject`;
  book tools default to `media=photo`.
- `index.html`: with a person selected and Media ∈ {All, Videos}, a "Main
  subject ≥ [x]% / ≥ [n] s" control; a Subject typeahead over
  `video_subjects.label`; cards show "Calvin 63%" or "dog 90%".
- PhotoModal **subject strip** for a video: a **timeline** of segments
  coloured by primary subject (person name, "dog", "mountain lake", "no
  subject"), per-person presence and focus bars beneath, click to seek. This
  is the owner's fastest way to audit the metric.
- `/faces`: a video badge on face tiles; verify-labels / rival tools work
  unchanged. `/api/stats` and the status card split photos and videos.
- Verify: "Calvin ≥ 50% main subject" and "dog ≥ 50%" return the
  owner-labelled holdout positives; EXPLAIN on the new filters (`COMBOS`); the
  timeline matches the clip by eye on 5 videos.

### Step 7 — Quality surfacing (1.5 d)

Always store and show the numbers (shake, noise, focus consistency) in the
modal with sorts `steadiest` / `least_noisy`. **No `shaky` / `out-of-focus` /
`noisy` tag unless that flag passed the step 4 holdout gate.** If a VLM rubric
wins a flag, it becomes a `video-quality` LLM pass in step 5's shape.

### Step 8 — Backfill runbook, docs, deploy (1.5 d)

- **Deploy the NAS first** (new endpoints and schema; the M28/batches lesson),
  then the workers, together (the `failures` semantics).
- Backfill: `register-videos --apply` per year starting with 2026 (a walk +
  inserts) → fleet `-p video-probe -n 2` (header reads only; the spike says
  how many bytes) → `-p video-scan -n 2` overnight (≈ 4,200 readable videos,
  ≈ 300 GB at 30 MB/s ≈ 3 h transfer + 1–2 h analysis) → `video-describe` →
  advance batches (match_faces; warm_crops skips videos). Report the cost up
  front before each year, the way the gemma-4-26b one-off did.
- CLAUDE.md gains a "Video" section: `PASS_MEDIA`, the unreadable rule, the
  throttle, Live pairing, the focus metric, the "nothing decodes on the NAS"
  rule, what not to simplify back.

## Estimate and ordering

| step | days | ships |
|---|---|---|
| 0 spike | 1 | — |
| 1 schema / isolation / registration | 3 | — |
| 2 playback / filter | 2.5 | **A**: videos visible + playable |
| 2b video-probe pass | 2 | real posters, dates, unreadable list |
| 3 video-scan | 6 | **B**: person + CLIP search find videos |
| 3b video-subjects | 3 | subject per segment (dog / mountain / person) |
| 4 eval harness + labels | 6 | picks configs |
| 5 video-describe | 2.5 | **C**: text/category/keyword search |
| 6 subject share UI + filters | 3.5 | **D**: "videos where X is the main subject" |
| 7 quality | 1.5 | numbers (tags only if gated) |
| 8 backfill / docs | 1.5 | — |
| **total** | **≈ 32.5** (≈ 35 with review slack) | |

Owner ≈ 2.8 h labelling. GPU/CPU ≈ 4–6 h for the eval, ≈ 5–8 h for the
scan backfill plus **20–30 h** for the subject pass. 0 → 1 → 2 → 2b is serial; step 4's harness can start alongside steps 3/3b
(both call the pure `video_analysis.py` / `video_subjects.py`); step 5 needs step 4's describe
result and step 6 needs its faces result; 5 and 6 are independent — the owner
picks which ships first.

## Risks

1. **NAS disk starvation during the backfill** — a 400 GB read is the
   2026-09-19 incident's shape. The 30 MB/s cap, `-n 2`, 1 video per claim,
   IDLE ioprio with refuse/yield, overnight runs, and `request-stats` p95
   checked on the first night.
2. **Deleting video rows through image-era tooling** (purge-nonimage, a
   stacking clear, dedupe prune). Guarded in step 1 with a test per tool. The
   one risk that loses data.
3. **HEVC in cv2 or the browser.** The spike settles both. Fallbacks: desktop
   ffmpeg via subprocess for decode (it is installed for ffprobe anyway); H.264
   720p proxies made on the desktop for playback (D12).
3b. **Probe bandwidth on no-`moov` files.** ffprobe may read a whole 70 MB
   file hunting for `moov`; 25% of the corpus is that shape, ≈ 100 GB. The
   head+tail pre-check and the stream cap bound it; the spike measures it.
4. **HDR / Dolby Vision iPhone clips** decode to washed-out 8-bit in cv2.
   Faces unaffected; CLIP/VLM slightly worse. Spike checks; a levels stretch
   before keyframe JPEG if needed.
5. **Dates.** `mvhd` is UTC on phones and often local on cameras; a wrong
   offset breaks Live pairing and timelines. Spike measures per-camera
   offsets; D14 applies them.
6. **Split identity vs one-person-per-video.** A failed merge gives one person
   two `faces` rows, `resolve-duplicate-persons` unmatches one, the share is
   under-counted. The merge threshold is in the eval sweep; a test pins that
   two tracks of one embedding cluster merge; the modal timeline makes it
   visible.
7. **Re-scan erasing human labels** — carry-over plus a test; unmatched labels
   reported in the submit response.
8. **Context overflow on multi-image requests** (LM Studio's slot split).
   Measured per K/resolution in the eval; production pins the winner's K/px;
   `video` role models need the per-model context default (16384+).
9. **Pass-predicate drift** — a future pass without a `PASS_MEDIA` entry
   would claim videos; the coverage test fails.
10. **Payload size** — `video-scan` submits are 3–5 MB of base64 JPEGs; keep
    ≤ 8 keyframes at 1024 px q85; mind the 120 s submit timeout.
11. **Small eval sets** — 90 videos, ~540 segments, holdout 30. Report CIs,
    use tie bands, say "tie" rather than overclaim.
12. **The subject pass is the expensive one** (≈ 36k VLM calls). The
    40-segment cap bounds it; the eval's F1-vs-S1 comparison decides whether
    people even need it. Run it after `video-scan` has already made videos
    searchable, so nothing waits on it.
13. **Grounding labels are free text.** "dog" / "puppy" / "our dog" must link
    into one subject; the normalizer + CLIP-crop similarity handle most of it
    and the eval measures the rest. Searching by label needs the same
    normalization at query time.
14. **The quality tag may simply fail its eval**, as it did twice for photos.
    "Ship numbers, no tag" is the expected outcome, not a worst case.

## Decisions

The two planners resolved every point they contested; the merged positions
are recorded first. The owner-facing forks below them were not planner
disagreements; the recommended default is adopted and the owner can override
any of them before the relevant step starts.

### Resolved between the planners

| topic | Planner A | Planner B | outcome | date |
|---|---|---|---|---|
| Probe toolchain | ffmpeg/ffprobe apt package in the image | cv2 + a pure-Python atom parser | **ffprobe for metadata, cv2 for decoding** — both on the **desktop** after the owner's D19 (the planners had put ffprobe in the NAS image). ffprobe is the tool behind the 25% figure, reads creation time/GPS/rotation cv2 can't, covers .avi/.mts; atom parser dropped. | 2026-10-17 |
| Where frames are decoded | NAS seek-extracts 3–12 sparse JPEG frames, ships those | Desktop downloads each file once (30 MB/s cap), decodes densely at 2–4 fps, uploads keyframes | **B.** No `/dev/dri` passthrough, so container decode is software HEVC on an OOM-killed N100; 3–12 frames make the share too coarse; A's three passes re-extracted each video three times. After D19 the NAS does not even probe or grab the poster; a `video-probe` pass does both over HTTP Range. | 2026-10-17 |
| Video face storage | Separate per-frame `video_faces`; only named people can be main focus | One `faces` row per identity + `video_tracks`; share grouped by `faces.person_id` | **B.** A's design kept video faces out of `/faces`, clustering, verify-labels and the rival test, and had a correctness bug (an unknown dominant kid credits a small named one). Focus is computed over **all** faces; identity attached afterward. | 2026-10-17 |
| Tracking | None (citing the ~4% temporal matcher) | Dependency-free numpy tracker | **B as default F1, A as baseline F0 in the eval**; < 5 pts = tie, simpler wins. The temporal matcher links photos 30 min apart; boxes 0.25–0.5 s apart are a different problem. | 2026-10-17 |
| Live Photo clips | No decision | Pair to the still, hide, don't analyse | **B.** Changes how much of the backlog needs analysis; pairing precision gated at 10/10. | 2026-10-17 |
| `media_type` migration | `UPDATE photos SET media_type='photo'` | NULL means still | **B** — no full-table rewrite under the NAS write lock. | 2026-10-17 |
| `verify` on videos | reuse unchanged | still-only | **B** — `_process_verify` reads pixels and would claim a video once it has a description. | 2026-10-17 |
| `purge-nonimage-photos` | not addressed | guard in step 1 | **B** — the one data-loss risk. | 2026-10-17 |
| Owner labelling shape | per-frame, per-person grid (6–8 h) | per-segment main focus, ≤ 12 segments (≈ 2.8 h) | **B** — better signal and a third of the time. | 2026-10-17 |
| Describe metric | add a "names the main action" yes-rate | "says what happens over the clip" 3-state + gate vs poster-only | **Merged**: B's question and D0 gate carry A's intent. | 2026-10-17 |
| Focus ship gate | share MAE ≤ 0.15 | top-person ≥ 0.80, presence P ≥ 0.95, segment acc ≥ 0.70 | **Merged**: all four. | 2026-10-17 |

### Owner-facing forks (default adopted; override before the step starts)

| # | question | adopted default | alternative | choice | date |
|---|---|---|---|---|---|
| D1 | Data model | `photos` row + `media_type`, side tables `video_meta`/`video_tracks` | separate `videos` table (zero blast radius, every UI surface rebuilt) | default | 2026-10-17 |
| D2 | Unreadable 25% | register a row with a stored `probe_error`; hidden from search; listed on `/admin/maintenance` | report file only, no rows | default | 2026-10-17 |
| D3 | Live Photo clips | register, hide behind the still ("▶ Live" in the modal), don't analyse | first-class videos, fully analysed | **owner: skip them** (default confirmed) | 2026-10-17 |
| D4 | Decode location | desktop, after a 30 MB/s-capped download | NAS-side extraction | default; superseded by D19 for the probe/poster too | 2026-10-17 |
| D5 | Keyframes on NAS `/data` | yes, ≈ 12–15 GB | desktop-only cache (breaks re-runs from another machine, face crops) | default | 2026-10-17 |
| D6 | Body tracker (F3) | fallback only, tried if the diagnosis shows ≥ 10% of sports segments lost to "person box, no face" (the VLM subject box now covers that case) | always / never | default, demoted by D21 | 2026-10-17 |
| D7 | Native video-input VLM | excluded (desktop torch has no GPU; LM Studio route is image-only) | owner sets up ROCm torch or vLLM to include it | default | 2026-10-17 |
| D8 | Role for video describe | new `video` role even if the same model | reuse `describe` | default | 2026-10-17 |
| D9 | Videos in default search | yes, with a Media filter | opt-in (Media=Photos default) | default | 2026-10-17 |
| D10 | Share denominator | whole clip duration = the video's full running time (the sample only sets the estimate's resolution); `focus_seconds` stored and filterable beside the share for long videos | people-frames only (show both in the modal) | **owner: whole clip** (confirmed after clarifying it is not the sample, "for now") | 2026-10-17 |
| D11 | Quality tags | numbers only unless P ≥ 0.8 / R ≥ 0.5 on holdout | ship VLM tags un-gated | default | 2026-10-17 |
| D12 | HEVC playback | rely on browser HEVC (spike confirms) | H.264 720p proxies made on the desktop | default | 2026-10-17 |
| D13 | Re-scan and manual labels | carry over by one-to-one encoding match ≤ 0.6 | freeze `faces` rows once any is manual | default | 2026-10-17 |
| D14 | Video date source | Apple creationdate > mvhd + per-camera offset > mtime | mtime (current ingest behaviour) | default | 2026-10-17 |
| D15 | Existing mis-foldered videos | leave files in place; fix only `date_taken` | refile (same care as `refile-unknown-camera`) | default | 2026-10-17 |
| D16 | Before the eval | compute the share, hide the filter | show it as "beta" immediately | default | 2026-10-17 |
| D17 | Hash for backfilled videos | worker hashes during the scan download; NAS hashes only unreadable ones | NAS whole-file hash of 400 GB (throttled ~3–4 h) | default | 2026-10-17 |
| D18 | `verify` on videos | off (still-only) | adapt verify to keyframes later | default | 2026-10-17 |
| D19 | Where the work runs | **everything except registering rows, storing uploads, serving files and the Live-pairing SQL runs on the desktop**: probe, poster, keyframes, face crops, all models. No ffmpeg in the NAS image; no decode on the N100 ever. | planners' version: NAS probes with ffprobe and grabs one poster frame | **owner decision** | 2026-10-17 |
| D21 | What "main focus" means | **the SUBJECT, whatever it is** — person, animal, landscape, object — via VLM grounding per segment keyframe (`subjects.ground_photo_subjects`, already the photo subject model), with faces supplying identity for people | faces only (the planners' version; cannot see a dog, a mountain, or a kid facing away) | **owner decision** | 2026-10-17 |
| D20 | ffprobe on the desktop | `apt install ffmpeg` in WSL (one-time; matches the measurement tool) | the `av` (PyAV) wheel, no sudo, reads the same metadata and decodes; costs a new pinned dependency | default | 2026-10-17 |
