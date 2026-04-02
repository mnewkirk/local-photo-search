# local-photo-search

A fully local, offline photo search engine. Search your photo library by person, place, description, or color — without sending a single photo to the cloud.

Built collaboratively by Matt Newkirk and Claude (Anthropic) as a proof of concept on macOS, targeting eventual deployment on a UGREEN NAS (Intel N100, 8GB RAM) in Docker.


## Why this exists

Photo libraries grow fast and become impossible to browse. Commercial solutions (Google Photos, Apple Photos, Amazon Photos) solve this with cloud-based AI, but that means uploading every personal photo to someone else's servers. This project proves that the same search capabilities — semantic natural language queries, face recognition, scene understanding — can run entirely on local hardware using open-source models.

The guiding principles:

1. **Never modify or destroy a photo.** All annotations live in a separate SQLite database.
2. **Never send photos to an external API.** CLIP runs locally via PyTorch, LLaVA runs locally via Ollama, face detection runs locally via InsightFace.
3. **Keep the database fast to access.** sqlite-vec provides vector similarity search directly inside SQLite — no separate vector database to manage.
4. **Make results easy to find and visualize.** Start with a CLI, plan for a web UI.


## What it can do today

**Semantic search** — describe what you're looking for in natural language and get ranked results:

```bash
python cli.py search -q "people outdoors" --limit 50
python cli.py search -q "kids playing on the beach"
python cli.py search -q "bird on a branch"
```

**Person search** — find all photos of a specific person:

```bash
python cli.py add-person "Alex" --photo reference/alex.jpg
python cli.py match-faces --temporal
python cli.py search --person "Alex"
```

**Color search** — find photos dominated by a particular color:

```bash
python cli.py search --color blue
python cli.py search --color "#ff8800"
```

**Face reference search** — find photos containing a face similar to one in a reference image:

```bash
python cli.py search --face /path/to/reference.jpg
```

**Quality filtering** — find your best photos using aesthetic scoring:

```bash
# Show only high-quality photos from any search
python cli.py search -q "beach sunset" --min-quality 6.0

# Find the best photos in your entire collection
python cli.py search --min-quality 7.0 --limit 20

# Sort any search by quality instead of relevance
python cli.py search --person "Alex" --sort-quality

# Score range: 1-10 (3-5 average, 5-7 good, 7+ excellent)
```

**Combined search** — intersect multiple criteria:

```bash
python cli.py search -q "beach" --person "Alex" --color blue
python cli.py search -q "kids playing" --min-quality 5.5
```

Search results are written to a timestamped `results/` subfolder as symlinks (preserving the originals) alongside JPEG thumbnails for quick browsing in Finder.


## Architecture

```
local-photo-search/
├── photosearch/           # Core Python package
│   ├── clip_embed.py      # CLIP embedding (ViT-B/16 + OpenAI weights)
│   ├── db.py              # SQLite schema, vector tables, all queries
│   ├── search.py          # Hybrid search: CLIP + face boost + description scoring
│   ├── index.py           # Indexing pipeline orchestrator
│   ├── faces.py           # InsightFace detection, ArcFace encoding, clustering
│   ├── describe.py        # LLaVA scene descriptions + semantic tagging via Ollama
│   ├── colors.py          # Dominant color extraction (colorthief)
│   ├── quality.py         # Aesthetic quality scoring (LAION predictor + ViT-L/14)
│   ├── verify.py          # Hallucination detection (CLIP + cross-model LLM)
│   ├── cull.py            # Shoot review: clustering + selection algorithm
│   ├── stacking.py        # Photo stacking: burst/bracket detection + grouping
│   └── exif.py            # EXIF/GPS metadata extraction
├── frontend/dist/         # Web UI (static HTML + React)
│   ├── index.html         # Main search interface
│   ├── faces.html         # Face management (grouping, naming, merging, ignore)
│   ├── review.html        # Shoot review / culling interface
│   ├── collections.html   # Collections / albums interface
│   └── shared.js          # Shared components (PhotoModal)
├── cli.py                 # Click-based CLI with all commands
├── tests/
│   ├── test_face_matching.py   # Face accuracy + semantic ranking tests
│   ├── test_stacking.py        # Stacking algorithm + API tests
│   ├── test_cull.py             # Shoot review algorithm tests
│   ├── test_db.py               # Database layer tests
│   ├── test_api.py              # Web API endpoint tests
│   └── ...
├── scripts/
│   └── setup_sample.py    # Automated sample dataset setup for testing
└── requirements.txt
```

### Data flow

```
Photo files on disk
        │
        ▼
   ┌─────────────────────────────────────┐
   │          Indexing pipeline           │
   │                                     │
   │  1. EXIF extraction (exifread)      │
   │  2. CLIP embedding (open-clip)      │
   │  3. Dominant colors (colorthief)    │
   │  4. Face detection (InsightFace)    │  ← opt-in: --faces
   │  5. Scene description (LLaVA)       │  ← opt-in: --describe
   │  6. Aesthetic scoring (ViT-L/14)    │  ← opt-in: --quality
   └──────────────┬──────────────────────┘
                  │
                  ▼
           photo_index.db
       ┌──────────────────────┐
       │  photos              │  EXIF metadata, description, tags, colors, aesthetic_score
       │  clip_embeddings     │  512-dim vectors (sqlite-vec)
       │  faces               │  bounding boxes, cluster IDs
       │  face_encodings      │  512-dim ArcFace vectors (sqlite-vec)
       │  persons             │  named people
       │  face_references     │  reference photos for known people
       │  face_ref_encodings  │  reference ArcFace vectors (sqlite-vec)
       │  review_selections   │  shoot review picks + cluster assignments
       │  collections         │  named photo collections
       │  collection_photos   │  photos in each collection
       │  photo_stacks        │  stack groups (burst/bracket detection)
       │  stack_members       │  photos in each stack + top-photo designation
       └──────────────────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │     Search engine     │
       │                       │
       │  CLIP similarity      │  visual match to query
       │  + Face boost         │  per-face bonus for people queries
       │  + Description score  │  text match / negation / absence
       │  = Combined ranking   │
       └──────────────────────┘
```


## How search actually works

Semantic search is the most complex part of the system. A query like "people outdoors" seems simple, but CLIP embeddings alone can't reliably distinguish it — in a beach photo set, all photos are "outdoors" and CLIP scores cluster tightly. We solve this by combining three complementary signals.

### Signal 1: CLIP visual similarity

The query is embedded with the same CLIP model (ViT-B/16, OpenAI pretrained weights) used to embed all indexed photos. sqlite-vec finds the nearest photos by L2 distance. The score is `1 - distance`, so higher is better.

We chose ViT-B/16 over ViT-B/32 after testing showed B/32 couldn't separate "people outdoors" from "outdoors" at all — the 16px patch size gives 4x more image patches and substantially better fine-grained understanding. Both produce 512-dim vectors, so no schema change was needed.

### Signal 2: Face-aware reranking

When the query mentions people-related keywords ("people", "child", "family", etc.), each photo gets a score bonus of `+0.02` per detected face. This is enough to lift a people-containing photo above its neighbors in a tight CLIP score cluster, without being so aggressive that any random face-having photo leapfrogs a genuinely relevant result.

### Signal 3: LLaVA description scoring

Each photo has a natural language description generated by LLaVA (running locally via Ollama). The description is matched against the query using a three-tier scoring system:

- **Positive boost (+0.05):** All query words appear in the description. "People outdoors" matches a description mentioning "person" and "outdoor." This is stronger than the face boost because a text match is direct evidence.

- **Negation penalty (−0.04):** The description explicitly contradicts the query. "No people visible", "nobody", "empty beach", "no visible presence of people" all trigger this. We use regex-based matching to catch flexible phrasings that rigid substring matching would miss, while excluding "no other people" (which implies one person IS present).

- **People-absence penalty (−0.04):** For people queries specifically, if LLaVA described the entire scene without mentioning any people-related word, that's strong evidence nobody is in the photo — even if other query words match (e.g., a description saying "outdoor hillside with a bird" matches "outdoor" but not "people").

- **General absence penalty (−0.02):** The description exists but matches none of the query words. A weaker signal — the description might just use different vocabulary.

- **CLIP gate:** Description boosts are only applied when the CLIP score is above −0.05. This prevents hallucinated descriptions (LLaVA claiming a person exists in a landscape photo) from surfacing visually irrelevant results.

### Why this hybrid approach

Each signal covers the others' blind spots:

- CLIP sees visual similarity but can't distinguish "beach with people" from "beach without people" when the scenes look similar.
- Face detection gives a hard binary signal (faces present or not) but misses people from behind, at a distance, or with obscured faces.
- LLaVA descriptions capture semantic content ("a young girl walking on a trail") but occasionally hallucinate people or objects that aren't there.

The combination — CLIP provides the base ranking, faces boost confirmed-person photos, and descriptions push down explicitly empty scenes — produces search results that are far more accurate than any single signal alone.


## Face recognition pipeline

Face recognition uses InsightFace's `buffalo_l` model pack, which includes RetinaFace for detection and ArcFace for encoding. This replaced the original plan to use `face_recognition` (dlib) because InsightFace produced better embeddings for our photo set.

### Detection

High-resolution photos (Sony A7 IV at 7008px) are downsampled to 3500px on the long edge before detection — this keeps memory manageable and detection fast while preserving enough detail for ArcFace's 512-dim embeddings. Bounding boxes are scaled back to original coordinates for storage.

### Matching

ArcFace produces L2-normalized 512-dimensional vectors. Matching uses L2 distance with a calibrated threshold:

- **Standard matching (tolerance 1.15):** Calibrated on sample photos where same-person distances ranged 0.88–1.11 and different-person distances started at 1.31+. The 1.15 threshold captures all confirmed matches with a 0.16+ gap before the first false positive.

- **Temporal propagation (tolerance 1.45):** A second pass for faces that are too small or angled for confident auto-matching. Uses EXIF timestamps to check whether the best-matching person appears in a photo taken within 30 minutes. Requires both a clear distance gap to the second-best person AND temporal proximity — two constraints that prevent false positives in crowds.

### Workflow

```bash
# 1. Index photos with face detection
python cli.py index /path/to/photos --faces

# 2. Register known people from reference photos
python cli.py add-person "Alex" --photo ref/alex.jpg
python cli.py add-person "Jamie" --photo ref/jamie1.jpg --photo ref/jamie2.jpg

# 3. Match detected faces to known people
python cli.py match-faces --temporal

# 4. Review and correct mistakes
python cli.py diagnose-photo DSC04922.JPG
python cli.py correct-face DSC04907.JPG 2 "Alex"

# 5. Search by person
python cli.py search --person "Alex"
```


## LLaVA scene descriptions

Each photo gets a 2–4 sentence description from LLaVA running locally through Ollama. The description prompt is tuned for search: it asks the model to describe who/what is present, what they're doing, the setting, and whether people are present or absent.

We use `llava:13b` in production for better accuracy. The 7B model had frequent issues: describing children as adults, misidentifying gender, and hallucinating objects like surfboards. The 13B model is meaningfully better, though occasional hallucinations still occur (describing a person on an empty beach). The descriptions are stored in the database and used by the search engine's text-matching logic.

Ollama runs descriptions entirely locally — the `ollama` Python client communicates with the local Ollama server, and we verified through network monitoring that no photos are sent to external services.

```bash
# Generate descriptions for all photos
python cli.py index /path/to/photos --describe --describe-model llava:13b

# View descriptions
python cli.py show-descriptions
python cli.py show-descriptions DSC04907.JPG
```


## Aesthetic quality scoring

Photos can be scored on a 1–10 scale for general aesthetic quality using the LAION improved aesthetic predictor. This model was trained on millions of images with human aesthetic ratings and evaluates composition, lighting, visual appeal, and overall photographic quality.

The scoring uses a separate CLIP model (ViT-L/14) from the one used for semantic search (ViT-B/16). ViT-L/14 produces 768-dimensional embeddings that are passed through a pretrained linear head to produce a single quality score. This is a one-time computation per photo — the score is stored in the database and used instantly at search time.

Score ranges are roughly:

- **1–3:** Poor quality — blurry, bad exposure, random snapshots
- **3–5:** Average — typical phone photos, unremarkable
- **5–7:** Good — well-composed, pleasant lighting
- **7–9:** Excellent — professional quality, striking composition
- **9–10:** Exceptional — gallery-worthy, extraordinary

The ViT-L/14 model is loaded only during scoring, then released to free memory. On a Mac with Apple Silicon, scoring takes roughly 1–2 seconds per photo in batches. On the N100 NAS (CPU-only), expect 3–5 seconds per photo.

```bash
# Score all photos during indexing
python cli.py index /path/to/photos --quality

# Filter searches to only high-quality results
python cli.py search -q "sunset" --min-quality 6.0

# Find the absolute best photos across your entire collection
python cli.py search --min-quality 7.0 --limit 50 --sort-quality
```


## Shoot review (culling)

After a shoot, you often have hundreds of photos with many near-duplicates. The review system automatically selects the best, most representative photos — typically 10–15% of the folder — so you can quickly identify your keepers.

### How it works

The algorithm uses CLIP embeddings to cluster visually similar photos (via agglomerative hierarchical clustering with an adaptive distance threshold), then selects the best photo from each cluster based on aesthetic quality scores. The key design choices:

**Adaptive clustering.** Rather than using a fixed distance threshold (which breaks on same-day shoots where all photos are visually similar), the algorithm binary-searches for the threshold that produces the right number of clusters for the target selection count. This means it automatically adapts to tight-distribution shoots (beach day) and wide-distribution shoots (travel across multiple locations).

**Represent-all-then-trim.** Every cluster gets a representative. If there are more clusters than the selection budget, low-quality singletons are trimmed first, and clusters with 3+ members (real content themes) are protected. This ensures no significant content type is invisible in the selection.

**Tag-based diversity.** Within large clusters, photos with rare tags (from the ~60-tag semantic vocabulary) get selected as diversity picks. If a cluster of 20 beach photos contains one with a "bird" tag, that photo surfaces even though CLIP considers it visually similar to the others.

**ARW pairing.** Raw files (ARW) are tracked as paired files — same name, different extension. The index uses the JPG for analysis (faster, smaller), but export can include the corresponding ARW files.

### Usage

```bash
# Review a shoot folder (CLI)
python cli.py review ../Photos/2026-03-13

# Adjust target percentage
python cli.py review ../Photos/2026-03-13 --target-pct 15

# Export selected JPGs to a new folder
python cli.py review ../Photos/2026-03-13 --export ~/Desktop/selects

# Also export ARW raw files
python cli.py review ../Photos/2026-03-13 \
    --export ~/Desktop/selects \
    --export-raw ~/Desktop/selects/raw
```

### Web UI

The review page (`/review`) provides a visual grid interface with:

- **Folder picker** and adjustable target percentage
- **Four view modes:** All, Obscure Unselected (dims non-selected), Selected Only, and Clusters (groups by cluster with headers)
- **Click-to-toggle** selection on any photo, or use S/X keyboard shortcuts in the photo modal
- **Export** with optional ARW inclusion — copies file paths to clipboard for use with `cp` or `rsync`
- **Cluster badges** (center-top), **stack badges** (top-right), **quality scores**, and **RAW indicators** on every thumbnail
- **Stack expand/collapse** — view burst/bracket members inline in the grid


## Photo stacking

When shooting bursts or brackets, you end up with clusters of near-identical photos taken within seconds of each other. The stacking feature detects these and collapses them — showing only the best photo (the "top") while keeping the others accessible with one click.

### How it works

Stacking uses a union-find algorithm over photos sorted by timestamp. Two adjacent photos are linked if they were taken within a configurable time gap (default 3 seconds) AND have high CLIP similarity (L2 distance < 0.15). After building connected components, a span enforcement step limits each stack to 10 seconds from its earliest member — this prevents transitive chaining where A→B→C→D could span far longer than the intended window.

The top photo in each stack is selected by highest aesthetic quality score. All other members are hidden behind it in the UI, shown as a count badge on the thumbnail.

### Stack management in the UI

Stacking is available on all three main pages (Search, Review, Collections):

- **Stack badge** — clickable count badge (top-right) expands/collapses the stack inline in the grid. Expanded members show a white outline to distinguish them from regular photos.
- **Expand/Close all stacks** — bulk toggle button appears when stacks are detected.
- **Photo modal** — shows all stack members as chronologically-sorted thumbnails. "Make this top" promotes any member to top photo. "Unstack" removes a photo from the stack entirely.
- **Add to Stack** — unstacked photos show an "Add to Stack" button that finds nearby stacks (within 60 seconds) and lets you add the photo to one.
- **Selection persistence** — selected stack members remain visible in the grid even when their stack is collapsed.

### Usage

```bash
# Detect stacks in all indexed photos
python cli.py detect-stacks

# Detect stacks with custom parameters
python cli.py detect-stacks --max-gap 3.0 --max-dist 0.15

# Stacking API endpoints (used by web UI)
# GET  /api/stacks                      — list all stacks
# GET  /api/stacks/{id}                 — stack details with members
# POST /api/stacks/{id}/top             — set top photo
# POST /api/stacks/{id}/remove          — remove photo from stack
# POST /api/stacks/{id}/add             — add photo to stack
# GET  /api/photos/{id}/nearby-stacks   — find stacks within 60s
```


## Semantic tagging

Photos are tagged from a fixed ~60-tag vocabulary by LLaVA during indexing. Tags cover living things (animal, bird, wildlife), people (person, child, group), activities (playing, jumping, running), scenes (beach, mountain, forest), mood (dramatic, peaceful), and photography style (close-up, aerial). Tags are stored in the database and used by the shoot review algorithm for within-cluster diversity detection, and by the search engine for tag-based matching.

```bash
# Generate tags during indexing
python cli.py index /path/to/photos --tags

# Re-tag all photos (e.g., after improving the tag prompt)
python cli.py index /path/to/photos --tags --force-tags
```


## Hallucination detection (verification)

LLaVA occasionally hallucinates — describing a frisbee that isn't there, or claiming a bird is a hummingbird when it's a sparrow. The verification pipeline catches these errors using a three-pass approach with cross-model checking.

### Three-pass pipeline

**Pass 1 — CLIP scoring.** Extract visual nouns from the description and score each against the photo's CLIP embedding. Uses both an absolute threshold and a relative threshold (1.5 std deviations below the median) to flag suspicious items. In batch mode, only photos with flagged items proceed to Pass 2. In `--photo` or `--llm-all` mode, all photos proceed.

**Pass 2 — Cross-model LLM check.** Send the photo (not the description) to a *different* vision model (minicpm-v by default) and ask it to identify errors in the original description. Using a different model is critical: the same model that hallucinated an object will confirm its own hallucination when asked to verify. Different models have different biases, so cross-checking catches errors that self-verification misses.

**Pass 3 — CLIP cross-check.** For each item the LLM flagged as wrong, embed it with CLIP and check its similarity to the photo. If CLIP gives it an above-median score (suggesting the object really is in the image), the LLM's finding is overridden. This prevents the verification model from over-flagging — both CLIP and the LLM must agree before a hallucination is confirmed.

When hallucinations are confirmed, the description and tags are automatically regenerated with a stricter prompt that explicitly excludes the hallucinated objects.

### Why cross-model verification

We discovered through testing that same-model verification doesn't work for hallucination detection. When LLaVA described a beach photo as "playing with a frisbee" (no frisbee present), asking LLaVA to verify always returned "YES, frisbee is visible." Switching verification to minicpm-v (a completely different architecture: SigLip + Qwen2 vs CLIP + Vicuna) immediately caught the error.

```bash
# Verify all unverified photos
python cli.py verify

# Verify a specific photo (verbose output, checks everything)
python cli.py verify --photo DSC04929.JPG

# Re-verify all photos
python cli.py verify --force

# Flag only, don't regenerate
python cli.py verify --no-regenerate --force --limit 50
```

### Required models

The verification pipeline requires two Ollama vision models:

- **Description model** (default: `llava`) — generates and regenerates descriptions/tags
- **Verification model** (default: `minicpm-v`) — cross-checks descriptions for errors

```bash
ollama pull llava:13b    # or llava for the 7B variant
ollama pull minicpm-v    # ~5 GB, SigLip + Qwen2 architecture
```


## Prerequisites

**Python 3.10+** is required. The project is tested on Python 3.11.

**Ollama** is required for scene descriptions and hallucination verification. It runs as a local server that serves vision language models. Install it from [ollama.com](https://ollama.com), then start it with `ollama serve` (it runs on port 11434 by default).

You'll need two models — the initial download takes a while but only happens once:

```bash
ollama pull llava:13b      # ~8 GB — scene descriptions (better quality than 7B)
ollama pull minicpm-v      # ~5 GB — hallucination verification (cross-model check)
```

If you're tight on disk or RAM, you can start with just `ollama pull llava` (the 7B variant, ~4 GB) and skip `minicpm-v` — descriptions will work but verification won't.


## Quick start

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Index a folder of photos (full pipeline)
python cli.py index /path/to/photos --faces --describe --describe-model llava:13b --quality

# Launch the web UI
python cli.py serve              # opens at http://localhost:8000

# Or search from the CLI
python cli.py search -q "people outdoors" --limit 50
python cli.py search --person "Alex"
python cli.py search --color blue
python cli.py search -q "sunset" --min-quality 6.0
python cli.py stats

# See all available commands
python cli.py --help
```

The database (`photo_index.db`) is created automatically on first run in the current directory. All photo annotations are stored there — the original photos are never modified.


## CLI reference

| Command | Purpose |
|---------|---------|
| `index <dir>` | Index photos: EXIF, CLIP, colors, faces (opt-in), descriptions (opt-in) |
| `search` | Search by query, person, place, color, or face reference |
| `add-person <name>` | Register a named person with reference photo(s) |
| `match-faces` | Match detected faces to registered persons |
| `list-persons` | Show registered persons and photo counts |
| `face-clusters` | Show unidentified face clusters |
| `correct-face` | Fix a wrong face match |
| `tag-photo` | Manually tag a person in a photo (bypasses face detection) |
| `diagnose-photo` | Show face detection details and match distances |
| `stats` | Database statistics |
| `show-descriptions` | View LLaVA-generated descriptions |
| `verify` | Verify descriptions/tags for hallucinations and auto-regenerate |
| `review <dir>` | Shoot review — select best representative photos from a folder |
| `detect-stacks` | Detect and group burst/bracket shots into stacks |
| `serve` | Launch the web UI (default port 8000) |

Key flags for `index`:

| Flag | Effect |
|------|--------|
| `--faces` | Enable face detection and encoding |
| `--describe` | Generate LLaVA scene descriptions |
| `--describe-model llava:13b` | Use a specific Ollama model |
| `--quality` | Compute aesthetic quality scores (1–10) |
| `--tags` | Generate semantic category tags |
| `--force-clip` | Regenerate all CLIP embeddings (required after model change) |
| `--force-describe` | Regenerate all descriptions |
| `--force-faces` | Re-run face detection on all photos |
| `--force-quality` | Rescore quality for all photos |
| `--force-tags` | Regenerate tags for all photos |
| `--no-clip` | Skip CLIP embedding |
| `--no-colors` | Skip color extraction |

Key flags for `verify`:

| Flag | Effect |
|------|--------|
| `--photo <path>` | Verify a single photo (auto-enables verbose + llm-all) |
| `--verify-model minicpm-v` | Vision model for verification (default: minicpm-v) |
| `--model llava` | Model for regeneration (default: llava) |
| `--threshold 0.18` | CLIP similarity threshold for flagging |
| `--force` | Re-verify even previously verified photos |
| `--no-regenerate` | Flag hallucinations but don't auto-regenerate |
| `--llm-all` | Send all nouns to LLM, not just CLIP-flagged |
| `--limit N` | Max photos to verify (0 = all) |
| `-v` / `--verbose` | Show detailed CLIP scores for every noun/tag |

Key flags for `review`:

| Flag | Effect |
|------|--------|
| `--target-pct 10` | Target selection percentage (default 10%) |
| `--threshold 0.0` | Clustering distance threshold (0 = adaptive) |
| `--export <dir>` | Copy selected JPGs to a directory |
| `--export-raw <dir>` | Copy selected ARW files to a directory |
| `--list-only` | Print selected file paths without copying |


## Testing

Tests cover face matching accuracy and semantic search ranking against a 7-photo sample set with known ground truth:

```bash
# Full test suite (requires CLIP model and sample photos)
python tests/test_face_matching.py --setup

# Or with pytest
pytest tests/test_face_matching.py -v

# Face tests only (no CLIP needed)
pytest tests/test_face_matching.py -v -m "not semantic"
```

The semantic search tests validate ranking rather than exact thresholds — photos with people must rank above landscape photos for "people outdoors". This makes tests resilient to minor score changes while still catching regressions in the hybrid scoring logic.


## Scale testing

The system has been tested on 196 Sony A7 IV JPEGs (4608×3072, ~2 GB total) from a single beach outing. At this scale:

- CLIP embedding takes ~2–3 minutes (batched, on Apple Silicon MPS)
- LLaVA descriptions take ~30–60 seconds per photo with `llava:13b`
- Face detection and encoding takes ~1–2 seconds per photo
- Search queries return in under a second
- The SQLite database with sqlite-vec handles all vector operations without issue


## Design decisions and their rationale

**SQLite + sqlite-vec over a dedicated vector database.** The entire index is a single file. No Postgres, no Pinecone, no Chroma server to manage. sqlite-vec adds vector similarity search as a virtual table — it lives inside the same database as the metadata. For a personal photo library (even 100k photos), this is more than fast enough, and the deployment story for Docker on a NAS is dramatically simpler.

**CLIP ViT-B/16 (OpenAI) over ViT-B/32 (LAION).** We started with ViT-B/32 + LAION2B because it was the default in many examples. It couldn't distinguish "people outdoors" from "outdoors" at all — all 7 test photos scored within 0.018 points of each other. Switching to ViT-B/16 with OpenAI's pretrained weights gave meaningfully better separation. The 16px patch size provides 4x more image patches for fine-grained understanding. Both produce 512-dim vectors, so the switch required no schema change — just `--force-clip` to regenerate embeddings.

**InsightFace over face_recognition (dlib).** The original requirements.txt specified `face_recognition` and `dlib`, but InsightFace (RetinaFace + ArcFace) produced better embeddings for our photo set. ArcFace's 512-dim L2-normalized vectors give clean distance thresholds (same-person under 1.15, different-person above 1.31).

**LLaVA 13B over 7B.** The 7B model had significant accuracy issues: describing a 10-year-old boy as "an adult woman", hallucinating surfboards, and frequently misidentifying basic scene elements. The 13B model is meaningfully better at age/gender estimation and object identification, though still imperfect. The extra VRAM and inference time is worth it for search accuracy.

**Hybrid search scoring over any single signal.** No single signal is good enough alone. CLIP can't distinguish "beach with people" from "beach without people" when scenes look similar. Face detection misses people from behind. LLaVA hallucinates. The three-signal combination — with calibrated boost/penalty values — solves all of these problems in practice.

**Negation detection as regex over static phrases.** We started with a list of exact phrases ("no people", "no visible people"). LLaVA generates natural language that varies: "no visible presence of people", "no visible human activity", "untouched by people". A regex with bounded gaps (`\bno\b.{0,30}\bpeople\b`) catches these variants without an ever-growing phrase list, while a "no other" exclusion pattern prevents false negatives for descriptions like "no other people visible" (which implies one person IS present).

**CLIP gate on description boosts.** LLaVA occasionally hallucinates — describing a person walking on a beach where nobody exists. Without a gate, the description boost (+0.05) alone is enough to surface a completely irrelevant photo. The CLIP gate (minimum CLIP score of −0.05 for description boost to apply) ensures that only photos CLIP considers at least somewhat relevant get the description boost. Penalties always apply regardless of CLIP score.


## How we built this

This project was built collaboratively through conversation. Matt defined the goals and constraints (local-only, non-destructive, NAS-deployable), made architectural decisions, and tested against real photos. Claude wrote the code, debugged scoring issues, and iterated on the search algorithm.

The development process was iterative in a way that's hard to capture in commits alone. Some examples:

**The CLIP model switch** started because "people outdoors" returned all 7 test photos with scores within 0.018 points. Matt ran `--json-output --min-score -1.0` to dump raw scores, which revealed the clustering problem. Claude analyzed the scores, identified that LAION-pretrained weights were optimized for different tasks than natural language photo search, and suggested switching to OpenAI-pretrained ViT-B/16. This fixed the spread but introduced new issues — too many results now scored similarly — which led to the face boost and description scoring systems.

**The negation detection** evolved through several rounds. First pass: check for "no people" in descriptions. Failed because DSC04880 said "no one visible" instead. Second pass: add "no one", "nobody" to a phrase list. Failed because DSC04903 said "no visible presence of people" — the word "presence" broke the substring match. Third pass: regex with bounded gaps. This caught the flexible phrasing while "no other people" was deliberately excluded after DSC04895's description — "no other people visible" — turned out to mean "one person present, no additional ones."

**The people-absence penalty** was the final refinement. A bird photo (DSC04892) kept appearing in "people outdoors" results because its description mentioned "outdoor" (matching one of two query words), giving it a neutral score of 0.0 instead of a penalty. The fix: for people queries, check separately whether ANY people-related word appears in the description, regardless of other word matches. If LLaVA described the scene and never mentioned a person, that's a strong negative signal.

**The LLaVA prompt** was tightened after DSC04907's description mentioned a surfboard that wasn't in the photo. Adding "Only describe what you can clearly see — do not guess at objects you are unsure about" to the prompt reduced hallucinations. Matt then upgraded from llava:7b to llava:13b which further improved accuracy, though it still occasionally hallucinates (describing a person on DSC05036 when the beach is empty).


## Development milestones

| Milestone | Status | Description |
|-----------|--------|-------------|
| **M1** | Done | EXIF + CLIP indexing + color extraction. Semantic search works. |
| **M2** | Done | Face detection, encoding, clustering, temporal matching. |
| **M3** | Done | LLaVA descriptions via Ollama with hybrid search scoring. |
| **M4** | Done | Full CLI with all search modes (semantic, person, place, color, face). |
| **M5** | Done | Scale test — 196 photos indexed with descriptions and search validated. |
| **M6** | Done | Web UI (FastAPI + simple frontend). |
| **M7** | Done | Docker packaging for UGREEN NAS deployment. |
| **M8** | Done | Aesthetic quality scoring — pretrained model scores photos 1–10 for composition, lighting, and visual appeal. Filter/sort by quality in search. |
| **M9** | Done | Semantic tagging — LLM-generated tags from a fixed ~60-tag vocabulary at index time. Tags stored in the photos table and used for search matching and shoot review diversity detection. |
| **M10** | Done | Shoot review — adaptive CLIP clustering + quality-based selection for post-shoot culling. Web UI with grid view, cluster view, toggle selection, and export. CLI with export to directory. |
| **M11** | Done | Portable photo paths — photo_root system stores relative paths in DB, resolves at runtime. Supports moving the database between machines without re-indexing. |
| **M12** | Done | Hallucination detection — three-pass verification (CLIP scoring → cross-model LLM check → CLIP cross-check) catches and auto-regenerates hallucinated descriptions. Uses a separate vision model (minicpm-v) to avoid same-model confirmation bias. |
| **M13** | Done | Collections / albums — persistent named collections stored in the same SQLite DB. Full CRUD API. Web UI with dedicated collections page, save-from-review, add-from-search (single + multi-select). |
| **M14** | Done | Photo stacking — burst/bracket detection via union-find (time gap + CLIP similarity), 10-second span enforcement. Stack management UI on all pages (expand/collapse, set top, unstack, add to stack). Schema v11. |
| **M15** | Done | Shared header component — extracted `PS.SharedHeader` into `shared.js`. All pages use the same logo, nav links, and layout. Page-specific controls passed as children. |
| **M16** | Done | Shared photo detail modal — extracted `PS.PhotoModal` into `shared.js`. Unified modal with configurable feature flags, face editing, collection management, stacking UI, and keyboard navigation. Removed ~1500 lines of duplicated code. |


## Known limitations

- **Place search has no data.** The Sony camera that took the test photos didn't have GPS enabled. The place search infrastructure (GPS extraction, schema columns, search query) is wired up and ready for photos that include location EXIF data.

- **LLaVA hallucinations.** The 13B model occasionally describes people, objects, or activities that aren't in the photo. The CLIP gate and face confirmation mitigate this for search. The `verify` command (M12) catches and auto-regenerates most hallucinated descriptions using cross-model verification, but some false positives and false negatives remain.

- **No incremental CLIP updates.** Switching CLIP models requires `--force-clip` to regenerate all embeddings. There's no versioning to detect stale embeddings automatically.

- **Single-threaded LLaVA descriptions.** Ollama processes one image at a time. For 196 photos at ~45 seconds each, that's about 2.5 hours. This could be parallelized if Ollama supported batching.


## Target deployment

Proof of concept runs on macOS (Apple Silicon). Production target is a **UGREEN NAS** with:

- Intel N100 (4 cores, no GPU)
- 8 GB RAM
- Docker container

The stack is chosen to work within these constraints: SQLite needs no server, CLIP inference runs on CPU (slower but functional), and Ollama can serve LLaVA in CPU-only mode. Face detection via InsightFace uses ONNX Runtime's CPU provider.


## License

MIT — see [LICENSE](LICENSE).
