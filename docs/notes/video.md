# Video — history and traps

Video as `photos` rows, PASS_MEDIA, the five video passes, playback proxies, ZIP unwrap. Backfill runbook: `docs/plans/video-support.md` Step 8. Condensed rules: `.claude/rules/video.md`. Append new history under a new `###` heading; never rename an existing heading.

<!-- moved verbatim from CLAUDE.md (line 1529) on 2026-10-19 -->
## Video

`docs/plans/video-support.md` (plan, Step 0 spike results, decisions D1–D21,
as-built table, and the **Step 8 backfill runbook**). Built 2026-10-17,
**deployed 2026-10-18** (NAS + replica on schema v39) — **backfill not
started**: nothing registered, no fleet run, no sampling yet (owner's
instruction). Corpus: 5,577 files, ~400 GB, median 3 s (mostly Live Photo
clips), ~25% "unreadable".

**Data model.** A video is a `photos` row with `media_type='video'`; **NULL
means still**, so no existing row was rewritten (an `UPDATE … ='photo'` would
have held the NAS write lock over the whole table). Side tables: `video_meta`
(probe + scan + subjects state and quality numbers), `video_tracks` (one per
within-video person identity), `video_subjects` (one per main subject). Schema
v35 added `media_type` + the three tables; v36 `video_meta.poster_version` +
`video_tracks.presence_intervals`; v37 `subjects_version/at/json` +
`video_subjects.share/rep_seg`; v38 `video_meta.scan_generation` (staged
uploads, below); v39 `video_meta.proxy_needed/proxy_version/proxy_bytes/
proxy_w/proxy_h/proxy_at` (playback proxies, below), backfilled from
`probe_json` for rows probed before it.

**`db.PASS_MEDIA` says which rows each pass may claim** (`still` / `video` /
`any`), and every claim/count branch appends its fragment. Image passes are
`still`; `category-content`/`keywords` are `any` (they read
`photos.description`, which `video-describe` writes); `verify` is `still` (it
reads pixels and would otherwise claim a video once described, D18). A pass
missing from the map **raises**, and `tests/test_pass_media.py` iterates every
`WORKER_PASSES` entry — so a new pass cannot silently start claiming videos.
`batch_state` derives per-pass totals from the same map, so a batch holding
videos still completes its image passes. Every pixel reader that predates video
skips them (purge-nonimage — the one data-loss risk — stacking, sharpness,
rank_measure, team review, upscale, split, book export, Google upload, colors,
orientation, cull, verify CLI, `similar_by_photo(media=…)`).

**The NAS never decodes or probes (owner decision D19).** It walks the tree
(`register-videos`, and `ingest-incoming` for new files), inserts rows, stores
what workers upload (`POST /api/videos/{id}/assets`, atomic, validated before
any write, under `<DB dir>/video_frames/{id}/v{ver}/g{gen}/`), serves files with
HTTP Range, and does the Live-pairing SQL. No ffmpeg in the image; no decode on
the N100 ever. Don't "simplify" by moving the probe back onto the NAS.

**Five worker passes**, order (all mirrors of the pass list, `batch_state.
WORKER_PASSES`): `clip, faces, quality, video-probe, video-proxy, video-scan,
aesthetics, video-subjects, describe, video-describe, verify,
category-content, keywords, category-visual`. `video-proxy` sits right after
`video-probe` (it reads `proxy_needed`, set at probe submit) and before
`video-scan`; video-describe drains before the text passes that read it;
aesthetics → video-subjects share one qwen2.5-vl load, describe →
video-describe one qwen3.5.

| pass | does | model / env |
|---|---|---|
| `video-probe` (claim 25) | 12-byte magic read, ffprobe over Range, per-camera date, GPS, rotation-corrected dims, one cv2 frame → poster ×3, decides `proxy_needed` | none (ffprobe) |
| `video-proxy` (claim 1) | rate-capped download, ffmpeg transcode to H.264 720p-class, ffprobe-verify, streamed PUT — only clips `needs_proxy` flagged | none (ffmpeg/ffprobe; needs both on the worker host) |
| `video-scan` (claim 1) | one throttled download (hashed), dense decode at 4/3/2 fps by length, InsightFace + numpy tracker, CLIP mean, quality, keyframes, crops | none (CPU InsightFace) |
| `video-subjects` (claim 2) | VLM grounding of one keyframe per 3 s segment (cap 40), joined to face tracks | role `subject`: `PHOTOSEARCH_LLM_SUBJECT_MODEL` (falls back to the aesthetics model) |
| `video-describe` (claim 4) | K keyframes in ONE request + the subjects timeline → `photos.description` | role `video`: `PHOTOSEARCH_LLM_VIDEO_MODEL`, **strict — no fallback** |

<!-- moved verbatim from CLAUDE.md (line 1587) on 2026-10-19 -->
### Playback proxies (`video-proxy`, D12, owner-approved 2026-10-18)

The owner's Windows Chrome/Edge decodes **no HEVC at all**
(`MediaSource.isTypeSupported('video/mp4; codecs="hvc1.1.6.L93.B0"')` is
false), measured 2026-10-18 — played H.264 8-bit mp4/mov; did NOT play HEVC
Main10 HLG HDR iPhone `.mov`, 8K HEVC Sony `.mp4`, MPEG-4 ASP `.avi` (Canon),
or H.264 `.mts`. HEVC is the dominant codec among readable library videos, so
without a proxy most clips were a broken player.

- **`video_proxy.needs_proxy`** (pure) decides at probe submit: proxy iff NOT
  (h264 AND yuv420p/yuvj420p AND container mp4/mov/m4v AND long edge <=
  3840) — read from the already-stored `probe_json`, no new probe field. A
  ZIP that `video-scan` already unwrapped is marked `proxy_needed=1` (the NAS
  file is still the ZIP); a row still `probe_error='zip_wrapped'` is skipped
  until `unwrap-zip-videos` + re-probe.
- **Output**: H.264 High 8-bit `yuv420p`, long edge <= 1280
  (`PHOTOSEARCH_VIDEO_PROXY_LONG_EDGE`), CRF 23 `veryfast`, AAC 128k stereo
  (or no track), `+faststart`, <= 60 fps, rotation baked into the pixels,
  HDR (HLG/PQ) tone-mapped to BT.709 with zscale + mobius — chosen over
  hable, which came out a stop dark; mobius keeps cv2's brightness and
  saturation on the HLG fixtures (mean luma/sat 202/48 vs cv2's 205/32 on a
  short clip, 190/67 vs 194/47 on a 58 s clip). Whites show a warm cast —
  owner to eyeball on real footage.
- **Where it runs (D19 unchanged)**: the desktop. The worker downloads the
  original once (rate-capped), transcodes in a temp dir, verifies with
  ffprobe, and streams a PUT. No ffmpeg on the NAS.
- **Endpoints**: `PUT /api/videos/{id}/proxy?version=N` (raw `video/mp4`
  streamed to a temp file under `<DB dir>/video_proxies/`, `ftyp` magic + 2 GB
  cap, fsync + `os.replace`; 409 on the replica — upload to the NAS
  directly). `GET /api/videos/{id}/proxy` serves it with HTTP Range
  (un-throttled — this is browser playback); the replica forwards Range
  through to the NAS rather than caching it locally.
- **Source choice in the UI**: `PhotoModal`'s `PS.canPlayVideo` (an RFC 6381
  codec string from the probe through `MediaSource.isTypeSupported`) decides,
  and `PS.chooseVideoSource` picks the proxy when the original can't play
  here, the original otherwise with the proxy as the error fallback; while
  neither is ready it shows "Preparing a playable copy — or ↓ download".
- **Storage**: smoke timings show 0.1–0.25× real time to transcode (8K HEVC
  1.4×); proxies were made for exactly the 5 unplayable fixtures of 12.
  Library estimate ≈ 15–40 GB (likely ≈ 25 GB), ≈ 6–10 h with one worker.
  `cleanup-orphans` removes stale proxy versions + crashed temp files.

<!-- moved verbatim from CLAUDE.md (line 1629) on 2026-10-19 -->
### ZIP-wrapped video unwrap (`unwrap-zip-videos`, NAS CLI)

~58% of the 2016-17 "moov atom not found" videos are actually ZIP archives
under a video extension (the iOS Live-bundle bug CLAUDE.md already documents
for `.JPG`). `video-scan` already unwraps these on the fly for analysis;
`unwrap-zip-videos` fixes the **files on disk** so every consumer — browser
playback, ffprobe, the poster — sees a real video, not a ZIP.

- Per file: validate (exactly one safe video member, size/ratio caps, shared
  with the scan via `photosearch/zip_video.py` so the two can't disagree),
  extract to a temp file in the same directory (`O_EXCL`, sha256 while
  writing, fsync file + dir), give the zip a second name `<name>.zip` via
  `os.link` (`EEXIST` refuses, never overwrites), then `os.replace` the
  extracted video under the **original filename** — `filepath`/`folder`
  never change. The zip is **never deleted**.
- DB (`--apply` only): resets the matching `video_meta` row for re-probe and
  sets `photos.file_hash` to the inner file's sha; a prior scan's results are
  **kept** (it already analysed the unwrapped bytes) unless its
  `scan_error` starts `zip:`.
- Safety: runs on the NAS (stdlib `zipfile`, no decode — an owner-approved
  exception to D19), throttled (nice + IDLE ioprio + `--mbps`), refuses to
  start while an ingest sweep holds its lock, and `--apply` **holds** that
  lock itself. `--apply` requires `--audit` (CSV, fsynced per row); `--undo
  <audit.csv> [--apply]` restores the zip under the original name only when
  the current file there matches the recorded size + sha256.

```bash
photosearch unwrap-zip-videos --year 2016                                     # dry run
photosearch unwrap-zip-videos --year 2016 --apply --audit /data/unwrap-2016.csv
photosearch unwrap-zip-videos --undo /data/unwrap-2016.csv [--apply]
```

NAS free space 2026-10-18: 4.7 TB of 11 TB; unwrapping adds roughly 56–100 GB
(the recovered videos, since the ZIPs are kept too).

`rerun.FLEET_ROLE_MODELS`: subject `qwen2.5-vl-7b-instruct`, video
`qwen/qwen3.5-9b` (until the eval picks). An unset strict role raises
`RoleModelNotConfigured`; with `ContextOverflow` it halts the pass
(`PassHalted`: finished results submitted, the rest released with no attempt)
instead of burning attempts. **The video passes need `--native`**: ffprobe is
not in the Docker worker image (`PHOTOSEARCH_FFPROBE`, else PATH, else
`~/.local/bin/ffprobe`); `video-proxy` also needs ffmpeg there
(`PHOTOSEARCH_FFMPEG`, else PATH, else `~/.local/bin`) and drops itself if
either tool is missing. The static ffprobe build **segfaults resolving
`localhost`** — point workers at `127.0.0.1` when the server is local.

**Unreadable is stored once, never re-decoded.** A probe verdict is a stored
`probe_error` (the CLIP re-claim trap otherwise); transport errors are omitted
(deferred), a tool failure is a failure row. Unreadable/Live rows count as
done in batch state and are hidden from default search; they are listed at
`GET /api/admin/unreadable-videos` (card on `/admin/maintenance`).

**Most "moov atom not found" files are ZIPs, not corruption.** 23/40 sampled
2016–17 `.mov`/`.mp4` start `PK\x03\x04` (the iOS Live-bundle bug CLAUDE.md
already knew for `.JPG`), each wrapping one playable video. Probe stores
`probe_error='zip_wrapped'`; **scan unwraps** inside its existing download
(unwrapping costs a full decompress — DEFLATE over H.264 saves 0.03% — so it
does not belong in the cheap probe). Safety: one video member only (ignores
`__MACOSX/` and `._*`, takes the MOV of a JPG+MOV bundle, refuses 0 or 2+),
member ≤ 16 GiB **and** ≤ 2× the ZIP's own size + 16 MB (1.2× refused four 2018 HTC Zoe clips that deflate ~1.3×) (so a `compress_size 0`
header can't dodge the bomb check); dates/codec come from the inner file.

**Dates are a per-camera-family dispatch, not one offset (D14 as built).**
Apple `com.apple.quicktime.creationdate` verbatim; Android/Pixel
`creation_time − duration` (the muxer stamps the END) in `PHOTOSEARCH_TZ`;
Sony XAVC and Kodak PIXPRO write local wall time with a `Z` — treat as local,
**never convert** (converting moves every clip a whole timezone the wrong
way); else keep mtime. Measured on n≈10/family — re-validate on a bigger sample
before bumping `PROBE_VERSION`.

**Live pairing** is server-side at probe submit: same stem, same or ±1-day
sibling folder, ≤ 4 s long, within 3 s of the still. Paired clips are hidden
behind the still and never analysed (owner: skip them, D3).

**Throttle.** `?purpose=worker` reads of `/full` go through a process-wide
token bucket, `PHOTOSEARCH_VIDEO_STREAM_MBPS` (default 30 MB/s, ranged reads
included), and at most `PHOTOSEARCH_VIDEO_WORKER_STREAMS` (default 4)
concurrent worker streams — over the cap is `503 Retry-After: 10`, which
`WorkerClient` already backs off on. This is the 2026-09-19 disk-starvation
incident's shape (400 GB of reads); keep `-n 2` and watch `request-stats`.

**Faces: one `faces` row per within-video identity**, so `/faces`,
clustering, the strict matcher, verify-labels, the rival test and
one-person-per-photo apply unchanged. The bbox is in **native display px**
(the space of `photos.image_width/height`) — rep-frame px made 4K faces look
tiny to `min_bbox_edge`. All video coordinate systems are documented once, in
`video_scan_store`. Crops are **served from the uploaded crop / rep frame,
never the video**; `warm_crops` skips video faces. A re-scan carries
`manual`/`merge_review`/`rejected` labels one-to-one at L2 ≤ 0.6, plus every
matched face's `face_person_exclusions`; unmatched labels are reported.
**Uploads are staged** (v38): scan writes into a new generation dir and
installs nothing; only the committed submit records the generation, bumps
`poster_version` and installs posters + crops — so a deferred or failed
re-scan leaves the live files untouched. video-subjects/-describe carry
`scan_generation` and **defer** a result computed from a replaced one rather
than mapping `face_ix` onto new tracks. The tracker vetoes an IoU match whose
embeddings are > 1.2 apart (`track_iou_veto_l2`).

**Subject share** (D10, D21): per segment, the VLM's most salient box is the
primary subject (person / animal / landscape / object / activity); person
boxes take a face identity. `share = Σ seg seconds where S is primary / whole
clip duration`, `focus_s` beside it. Person shares are read `GROUP BY
faces.person_id` **at query time**, so a relabel moves the share with no
recompute. Search (`min_focus_share`, `min_focus_seconds`, `subject`,
`min_subject_share`, `sort=focus_desc|subject_desc`) is behind
**`PHOTOSEARCH_VIDEO_SUBJECT_SHARE`** (off until the eval passes, D16): off,
`/api/search` **400s** on those params, tool schemas omit them, stray tool args
are ignored. The modal's subject timeline (`GET /api/videos/{id}/subjects`)
always shows — it is the audit tool.

**Quality** (D11): shake / noise / focus consistency / exposure as **numbers
only**, shown in the modal; `sort=steadiest|least_noisy` (ungated, implies
`media=video`). No tag until a holdout eval gates one — the photo quality tag
failed its eval twice.

**Search.** `media=all|photo|video` (`/api/search`, Ask locked filters, tools).
Default hides paired Live clips and unreadable rows via rowid-only predicates
(`db.media_filter_sql`); curation/book tools default to stills.

**Evals:** `evals/video_eval.py` (sample → fetch-originals → run-faces /
run-describe / run-quality → report, holdout headline, CIs, `agreement`);
labelling on `/eval/models?tab=vsubject` and `?tab=vquality`.

**Deploy the NAS first** (schema v35–v39, new endpoints) — done 2026-10-18 on
both the NAS and the replica — **then the workers, together**, and run video
passes `--native` only. The runbook with exact commands and cost estimates
(including `unwrap-zip-videos` before registration) is Step 8 of the plan;
nothing has been registered or backfilled yet.
