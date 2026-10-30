# Indexing passes

Read for what each indexing pass does, its models, parallelism, collection re-indexing and stacking parameters.

<!-- moved verbatim from SKILL.md (line 701) on 2026-10-19 -->
## Stacking System

Burst/bracket detection using union-find over time-sorted photos:
- Two photos linked if taken within `time_window_sec` (default 5.0) AND
  CLIP cosine distance < `clip_threshold` (default 0.05)
- Span enforcement: max `max_stack_span_sec` (default 10.0) from earliest
  to latest member
- Top photo selected by highest aesthetic score
- Full CRUD API for manual stack management

Three invocation paths:
1. **CLI** — `photosearch stack` (see command table below)
2. **Blocking API** — `POST /api/stacks/detect`
3. **SSE API** — `POST /api/stacks/detect/stream` (used by the status-page
   Stacking form). Streams `progress` events with `phase` labels
   (`scan`, `load_embeddings`, `pairs`, `group`, `save`) throttled to
   ~0.25s, terminates with `done` / `cancelled` / `fatal`.

`stacking.py:run_stacking`, `detect_stacks`, `save_stacks`, and
`_load_embeddings_bulk` all accept `on_progress: Callable[[dict], None]`
and `should_abort: Callable[[], bool]`. Abort is checked at phase
transitions and every ~1000 photos inside the pairs loop, so a `cancel_event`
set by `request.is_disconnected()` stops the run within ~1s even on a
500k-photo library. The pair of hooks is the reference shape for adding
SSE progress + cancel to other long-running jobs.

---

<!-- moved verbatim from SKILL.md (line 1338) on 2026-10-19 -->
## Indexing Types — Detailed Reference

The `index` command has two modes: **directory mode** (scan a folder for new photos and
index them) and **collection mode** (re-index existing photos in a collection). Collection
mode skips EXIF extraction/insertion since photos are already in the DB, making it ideal
for re-running specific passes (e.g. testing a different LLM model for descriptions).

```
# Directory mode — scan and index new photos
photosearch index <photo_dir> [OPTIONS]

# Collection mode — re-index existing photos by collection ID
photosearch index --collection <ID> [OPTIONS]

Common options:
  --clip / --force-clip        CLIP semantic embeddings (ViT-B/16, 512-dim)
  --faces / --force-faces      Face detection + ArcFace encoding (InsightFace buffalo_l)
  --quality / --force-quality  Aesthetic scoring (ViT-L/14 + MLP) + concept analysis
  --describe / --force-describe  LLaVA scene descriptions (requires Ollama)
  --describe-model MODEL       Ollama model name (default: llama3.2-vision)
  --category-content / --force-category-content  Content categories from descriptions (text-only LLM)
  --category-visual / --force-category-visual    Visual-quality tags from images (vision LLM)
  --keywords / --force-keywords                  Free-form keywords from descriptions (text-only LLM)
  --no-colors                  Disable dominant color extraction (on by default)
  --full                       Enable all: --faces --describe --quality --category-content --category-visual --keywords
  --verify                     Run hallucination verification after other passes
  --batch-size N               Batch size for CLIP/quality (default: 8)
  --db PATH                    Database path (default: PHOTOSEARCH_DB env var)

Collection-only options:
  --collection ID              Re-index photos in this collection (replaces PHOTO_DIR)
  --expand-stacks              Also include other photos from the same stacks
```

<!-- moved verbatim from SKILL.md (line 1372) on 2026-10-19 -->
### Pass details

| Pass | Flag | Model | Speed on N100 | Dependencies |
|------|------|-------|---------------|-------------|
| EXIF + hash | (always) | — | Fast, ~1000/min | None |
| CLIP embeddings | `--clip` | ViT-B/16 (512-dim, ~330 MB) | ~1000 photos/hr | None |
| Dominant colors | (default on) | ColorThief | Fast, ~1000/min | None |
| Face detection | `--faces` | InsightFace buffalo_l (~300 MB) | ~0.5-2s/photo | None |
| Quality scoring | `--quality` | ViT-L/14 (768-dim) + MLP | ~1000 photos/hr | None |
| Concept analysis | (auto with quality) | Same ViT-L/14 | Runs after scoring | Quality pass |
| Descriptions | `--describe` | `llama3.2-vision` via Ollama (`--describe-model`) | 30-200s/photo CPU, ~4s GPU | Ollama running |
| Category (visual) | `--category-visual` | `llava` via Ollama (`--category-visual-model`) — **perceived** visual qualities only (mood / colour / light / atmosphere / viewpoint / composition); the capture facts are derived from EXIF, not asked of the model | 30-200s/photo CPU, ~1s GPU | Ollama running |
| Category (content) | `--category-content` | `llama3.2:3b` via Ollama (`--category-content-model`) — text-only, from description | ~text-only | Description exists |
| Keywords | `--keywords` | `llama3.2:3b` via Ollama (`--keywords-model`) — text-only, from description | ~text-only | Description exists |
| Critique | (auto) | Same as describe model | 30-200s/photo | Quality + describe |
| Stacking | (auto after CLIP) | — | Fast (DB only) | CLIP embeddings |
| Geocoding | (auto) | GeoNames (offline) | Fast | GPS data in EXIF |
| Verification | `--verify` | `llava` checker + `llama3.2-vision` regen | Slow (2 LLM passes) | Descriptions exist |
| Video probe | worker `video-probe` only | ffprobe over Range + 1 cv2 frame (desktop) | header reads | `register-videos` row |
| Video proxy | worker `video-proxy` only | ffmpeg transcode to H.264 720p-class (desktop); only clips `needs_proxy` | 0.1–0.25x real time (8K HEVC 1.4x) | video-probe |
| Video scan | worker `video-scan` only | InsightFace + tracker + CLIP (desktop CPU) | ~7–43 s/video | video-probe |
| Video subjects | worker `video-subjects` only | role `subject` (qwen2.5-vl) | ~0.7–1.9 s/keyframe | video-scan |
| People bind | worker `people-bind` only (not a batch step; `db.NON_BATCH_PASSES`) | role `bind` (provisional qwen2.5-vl-7b; the eval picks), rule tier needs no model | ~0 s rule, ~2 calls VLM | verify settled; a trusted-label face; <=6 faces |
| Video describe | worker `video-describe` only | role `video` (qwen3.5-9b), strict | ~3–8 s/video | video-scan |

The old single `tags` pass was removed and split into `category-visual`
(vision), `category-content` (text), and `keywords` (text). The per-pass
models are decoupled because no single model wins all — see "Per-pass model
strategy" under Distributed Indexing, and CLAUDE.md. `describe.py` has
model-aware Ollama options + a degeneration detect-retry-fallback for
`llama3.2-vision`, and a regurgitation guard on the visual-tag task.
(`clean-garbage-tags` still exists to clear historical regurgitated tag sets.)

**`category-visual` asks only about PERCEIVED qualities.** Three capture-fact
terms — `long-exposure`, `low-light`, `panoramic` — are computed
deterministically from EXIF (`photosearch/visual_tags_derive.py`) and override
whatever the model said; `motion-blur` was retired from the vocabulary
entirely; `sharp` / `blurry` are **frozen** (never produced, never deleted)
pending a hand-labelled eval. A VLM cannot read a shutter speed off the tile it is
shown, and the measurements said it did not try (`long-exposure` was tagged on
64% of a 1/125–1/800 s daytime soccer folder). The merge happens **server-side**
in `worker_api.submit_results` and both `index.py` writers; storage is still
the single `photos.visual_tags` JSON column, so `visual_tag=`, `list_vocab`
and the Ask tools are unaffected. Full detail + thresholds in CLAUDE.md,
"`category-visual`: derived capture facts vs perceived qualities".

Backfill existing rows without any VLM work (dry run by default; never touches
a NULL `visual_tags`, which is what drives the worker queue):

```bash
photosearch derive-visual-tags            # before/after frequency table
photosearch derive-visual-tags --apply    # chunked, guarded, idempotent
```

<!-- moved verbatim from SKILL.md (line 1425) on 2026-10-19 -->
### Parallelization

These can run as separate `nohup` processes simultaneously (different models, no conflicts):
- `--clip`, `--faces`, `--quality`

These share Ollama and now use **different** models (`llama3.2-vision` for
describe, `llava` for tags). Run them as **separate sequential runs** — bundling
them forces Ollama to thrash-swap models between every batch. Critiques use the
describe model, so they batch fine alongside `--describe`.

<!-- moved verbatim from SKILL.md (line 1435) on 2026-10-19 -->
### Collection-based re-indexing

All indexing commands support `--collection <ID>` to scope work to a specific collection
instead of an entire directory. This is useful for A/B testing models, re-running a pass
on a curated set, or iterating on a small batch without re-processing thousands of photos.

```bash
# Re-describe collection 2 with a different model
photosearch index --collection 2 --describe --force-describe --describe-model moondream

# Re-score quality for collection 2, including burst stack members
photosearch index --collection 2 --quality --force-quality --expand-stacks

# Match faces only within a collection
photosearch match-faces --collection 2 --temporal

# Detect stacks only among a collection's photos
photosearch stack --collection 2
```

Collection mode in `index` skips EXIF extraction and photo insertion (photos are already
in the DB) and runs all selected passes over the collection's photos. The `--expand-stacks`
flag expands the photo set to include other members of the same stacks.

**Implementation:** `index_directory()` dispatches to `_index_collection()` when
`collection_id` is set. The collection path uses `PhotoDB.get_collection_photo_pairs()`
to get `(photo_id, abs_path)` tuples. `match_faces_to_persons()`, `match_faces_temporal()`,
`detect_stacks()`, and `run_stacking()` all accept an optional `photo_ids` parameter for
scoping. `PhotoDB.expand_to_stacks()` expands photo IDs to include stack members.

<!-- moved verbatim from SKILL.md (line 1465) on 2026-10-19 -->
### Key model details

- **Search CLIP (ViT-B/16):** 512-dim embeddings, different from quality CLIP. Force-regen
  with `--force-clip` when switching models to avoid stale embeddings.
- **Quality CLIP (ViT-L/14):** 768-dim embeddings fed to linear MLP scorer. Score range
  in practice: 3.68–5.99 (mean 4.81). Both models unloaded after each pass to free RAM.
- **InsightFace:** Downloads buffalo_l to `INSIGHTFACE_HOME` (~300 MB). Images >3500px
  long-edge are downsampled before detection. 512-dim ArcFace embeddings.
- **LLaVA/moondream:** Images resized to 1024px max before sending to Ollama. moondream is
  5-10x faster but significantly worse quality — hallucinates frequently and ignores
  structured prompts (especially critique prompts). Use llava for anything quality-sensitive.

<!-- moved verbatim from SKILL.md (line 1477) on 2026-10-19 -->
### Stacking parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `time_window_sec` | 5.0 | Max seconds between consecutive burst shots |
| `clip_threshold` | 0.05 | Max CLIP cosine distance (very tight = same scene) |
| `max_stack_span_sec` | 10.0 | Hard cap on total stack duration |

The `stack` CLI command also supports `--collection ID` and `--expand-stacks` to scope
detection to a specific collection's photos.

**Progress + cancel hooks** — `detect_stacks`, `_load_embeddings_bulk`,
`save_stacks`, and `run_stacking` all take
`on_progress: Callable[[dict], None] | None` and
`should_abort: Callable[[], bool] | None`. Progress events are dicts with
a `phase` key (`scan` / `load_embeddings` / `pairs` / `group` / `save`)
plus phase-specific counters (`loaded`/`processed`/`saved` vs `total`,
`pairs_found`, `stacks_detected`). Emission inside hot loops is throttled
to `_PROGRESS_INTERVAL = 0.25s`; phase transitions bypass the throttle.
`should_abort()` is checked every ~1000 iterations and at every phase
transition; returning `True` causes `InterruptedError` to propagate out.
The SSE endpoint `/api/stacks/detect/stream` wires these up to an
`asyncio.Queue` + `threading.Event` — the exact pattern to copy when
adding cancel + progress to any other long-running job (recluster,
describe, quality, verify).
