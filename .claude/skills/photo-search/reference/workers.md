# Worker fleet

Read before launching or debugging workers: run-workers.sh, LM Studio routing, GPU setup, the worker API.

<!-- moved verbatim from CLAUDE.md (line 340) on 2026-10-19 -->
## Distributed Indexing (Worker)

Offload heavy indexing to a fast laptop while the NAS keeps the DB + photos.
Worker claims batches via HTTP, downloads photos, processes locally, POSTs results back.

**Docker worker fleet (recommended)** — avoids PyTorch MPS memory leak on macOS:
```bash
./run-workers.sh -s http://<NAS-IP>:8000 -p clip -d /photos/2026 -n 4
./run-workers.sh --status    # containers + memory + progress
./run-workers.sh --logs      # tail all workers live
./run-workers.sh --stop      # stop all workers
```
Worker log lines are **timestamped** (`worker.install_log_timestamps`, applied
by `cli.py worker`). The OpenAI-route reachability check
(`describe.check_available`) now retries ~2.5 min: LM Studio stops answering
while it JIT-loads a model, and that single un-retried check used to kill a
worker whose pass started right after its model was ejected.

**Workers drain passes sequentially by default** (since 2026-09-26): each pass
in `-p` order is drained completely before the next starts, so a new shoot is
CLIP-searchable before the slow LLM passes begin and model weights aren't
reloaded every batch. `--round-robin` (cli.py worker, run-workers.sh) restores
the old one-batch-per-pass-per-cycle interleave; the `/admin/maintenance`
Sequential checkbox now defaults on.

**Startup probe is `GET /api/health`** (no DB), never `/api/stats`: every
worker in a fleet probes at once, and `/api/stats` cost 3 × 117 s on a cold NAS
(2026-10-07). A 404 from an older server counts as reachable. `/api/stats` is
itself now memoized (`web._cached_stats`, 60 s, stale-while-revalidate: only
the first request after a start computes inline) and reads `photos` with one
table scan instead of four (`web._compute_stats`; `tests/test_api_stats_cache.py`
pins it). The replica card counts NAS photos via `/api/admin/maintenance-fingerprint`.
Scoped queue counts (`count_unprocessed_photos(photo_ids=...)`, behind
`/api/worker/status?collection_id=…`) write `+id IN (...)` above 2,000 ids
(`db._scope_col`) so they drive from the need-index rather than reading one
`photos` leaf per id (16k-id collection: 19–33 s on the NAS). Checking the
queue by hand? Pass `passes=` — all nine counts is what was slow.

Uses CPU-only PyTorch with 3GB hard memory limit per container. Use NAS IP address
(not hostname) — Docker containers can't resolve local DNS names.

**Bare-metal / native** — required when the worker has a GPU (the Docker fleet
is CPU-only). Two ways to launch:

```bash
# Easier: same command on every machine; auto-picks native on WSL2, Docker on Mac/Linux.
./run-workers.sh -s http://<NAS-IP>:8000 -p clip,faces,quality -d /photos/2026 -n 3
# Override with --native / --docker. --ollama-host URL for non-default Ollama.

# Or run cli.py directly:
python cli.py worker -s http://<NAS-IP>:8000 -p clip,faces,quality,describe,category-content,category-visual,keywords,verify -d /photos/2026
```

**`-d` gotcha:** `-d` must be a real subdirectory (e.g. `/photos/2026` or
`/photos/2026/2026-04-09`). `-d /photos` — the photo *root* — always 404s
("No photos found in directory /photos"): `get_directory_photo_ids` strips
the photo_root prefix, leaving an empty prefix that matches no relative DB
path. To process the whole library, **omit `-d` entirely.**

`get_directory_photo_ids` is on the fleet's **per-claim** path, so it ranges over
the indexed `photos.folder` column (`folder = p OR (folder >= 'p/' AND folder <
'p0')` — `'0'` is the character after `'/'`) rather than `filepath LIKE 'p/%'`,
which SQLite cannot index and was a full scan of `photos` per claimed batch. The
range is exactly "in `p` or any subfolder", so year-level dirs work unchanged;
a miss retries the LIKE only because LIKE is ASCII case-insensitive. Don't
"simplify" it to `folder = ?` — that silently drops subfolders.
`tests/test_directory_scope.py` asserts the query plan uses `idx_photos_folder`.

Native mode launches `cli.py worker` processes from the project venv (GPU
auto-detected via torch+rocm / cuda) with `HSA_ENABLE_DXG_DETECTION=1` and
`OLLAMA_HOST` resolved to the Windows-host gateway on WSL2. Worker logs land
in `/tmp/photosearch-worker-fleet/worker-N.log`; `--status` / `--logs` /
`--stop` work in both modes.

**Filter-scoped runs (no collection needed):** besides `--collection` /
`--directory`, the fleet can be scoped by a **structured filter set** — the
same vocabulary as `search`: `--from` / `--to` (date), `--person` (repeatable,
AND-intersected), `--location`, `--min-quality`, `--min-aesthetic`, `--camera`,
`--category`, `--visual-tag`, `--keyword`, `--style-tag`. The three scope kinds
are mutually exclusive. Available on `cli.py worker`, `run-workers.sh`, and the
`/admin/maintenance` **Worker fleet** panel (a filter row that overrides the
collection dropdown). This replaces the old "make a collection from a date
range, then run a job against it" workflow — e.g. score aesthetics for a trip:
`./run-workers.sh --native -s <NAS> -p aesthetics --from 2026-06-27 --to
2026-08-06 -n 2`. `cli.py worker … --dry-run` prints per-pass queue depth for
the scope without claiming.

The filter set is resolved **server-side** to a photo-id scope via
`worker_api._resolve_scope_ids` (reuses `tools._build_filter_sql`), so the NAS
must run this code — an older NAS image ignores `filters` and would claim the
whole library. A broad scope (a whole camera, a frequent person) can be tens of
thousands of ids; `_unprocessed_scoped` / `_count_scoped` chunk the scope under
`SQLITE_MAX_VARIABLE_NUMBER` so large scopes don't blow the SQL parameter cap.

Key files: `photosearch/worker.py` (client), `photosearch/worker_api.py` (server endpoints),
`run-workers.sh` (Docker fleet launcher). API routes under `/api/worker/*`.
Claims have TTL (default 30min) for crash recovery. `submit_results` for the
LLM passes (describe / category-* / keywords / verify) also logs to the
`generations` table — the worker sends `model` + `model_version` so provenance
is captured per artifact.

<!-- moved verbatim from CLAUDE.md (line 984) on 2026-10-19 -->
### Routing LLM passes to LM Studio (OpenAI-compatible)

`describe.py` can route **all** LLM passes (text *and* vision) to an
OpenAI-compatible `/chat/completions` endpoint instead of Ollama. Set
`PHOTOSEARCH_TEXT_LLM_URL` (base must end in `/v1`) and the per-pass model is
chosen by **role** env var (not by model name):

| Pass | role | Env var |
|---|---|---|
| describe (+ regen) | `describe` | `PHOTOSEARCH_LLM_DESCRIBE_MODEL` |
| verify | `verify` | `PHOTOSEARCH_LLM_VERIFY_MODEL` |
| category-visual | `visual` | `PHOTOSEARCH_LLM_VISUAL_MODEL` |
| category-content / keywords | `text` | `PHOTOSEARCH_LLM_TEXT_MODEL` |

**The fleet's per-pass models live in one place: `rerun.FLEET_ROLE_MODELS`**,
chosen by the model evals (`docs/plans/model-eval-harnesses.md`). As of
2026-10-02: **describe `qwen/qwen3.5-9b`**, **verify `google/gemma-4-12b-qat`**,
**visual `minicpm-v-4_5`**, aesthetics `qwen2.5-vl-7b-instruct`, **text
(category-content + keywords) `google/gemma-4-12b-qat`**. The web-UI fleet (`admin_api._fleet_env`, used by
`/admin/maintenance` and `/batches`) **sets every role explicitly** from it; override
one role with `PHOTOSEARCH_FLEET_<ROLE>_MODEL`. It used to `setdefault` from the
server's `PHOTOSEARCH_LLM_VISUAL_MODEL`, which the replica sets for rerank. So from
2026-09-20 to 09-28, describe, verify, visual and aesthetics **all ran on
qwen2.5-vl-7b**, not the per-role models this section used to list (visible in
`generations.model_used`). `run-local-replica.sh` now exports the same role models
for its in-server re-runs, and rerank/photobook read **`PHOTOSEARCH_LLM_RERANK_MODEL`**
first (still qwen2.5-vl), so changing the visual pass doesn't move them.

**Each pass unloads its model from LM Studio when it drains**
(`worker._release_llm_models` → `describe.unload_openai_models`, LM Studio's
`/api/v1/models/unload`). LM Studio never evicts a JIT-loaded model on its own,
so before this a full fleet run ended with every role model resident at once
(2026-10-04: qwen2.5-vl, qwen3.5-9b, minicpm and gemma, all at 16k context, on
one 24 GB card). Rules: a model the **next** pass shares stays (gemma serves
category-content → keywords → verify); a model **another worker** still holds a
live claim on stays — so the last worker out unloads; a failed status call or
the Ollama route unloads nothing. Needs LM Studio **JIT loading on** — the next
request reloads whatever was unloaded (incl. the replica's rerank / Ask models).

**A model loaded while VRAM was full STAYS spilled.** LM Studio places a model's
layers when it loads and never rebalances, so a model JIT-loaded behind three
others is partly on CPU for life. 2026-10-04: gemma-4-12b ran at **3 t/s**, every
`category-content` call blew the 10 s cap even with reasoning off; unloading it
(`POST :1234/api/v1/models/unload {"instance_id": …}`) and letting JIT reload it
into empty VRAM gave **17 t/s** under 3 concurrent workers, ~3 s/photo. Symptom:
text passes all `deferring (timeout/error)` with `reasoning_tokens` 0.

**A model the fleet JIT-loads needs an LM Studio per-model context default**
(`%USERPROFILE%\.lmstudio\.internal\user-concrete-model-default-config\…json`,
`llm.load.contextLength`). Otherwise it loads at 4096, split across 4 parallel
slots, and vision requests 400. Set to 16384 for gemma-4-12b-qat and minicpm-v-4_5.

`run-workers.sh` has no flag for these — `export` them before a hand-run `--native`
launch (the native launcher inherits exported env):

```bash
export PHOTOSEARCH_TEXT_LLM_URL=http://<host>:1234/v1
export PHOTOSEARCH_LLM_DESCRIBE_MODEL=qwen/qwen3.5-9b
export PHOTOSEARCH_LLM_VERIFY_MODEL=google/gemma-4-12b-qat
export PHOTOSEARCH_LLM_VISUAL_MODEL=minicpm-v-4_5
export PHOTOSEARCH_LLM_AESTHETICS_MODEL=qwen2.5-vl-7b-instruct
export PHOTOSEARCH_LLM_TEXT_MODEL=google/gemma-4-12b-qat
export PHOTOSEARCH_LLM_REASONING_EFFORT=none   # REQUIRED for gemma-4 / minicpm, see below
./run-workers.sh --native -s http://<NAS-IP>:8000 \
    -p clip,faces,quality,describe,category-content,category-visual,keywords,verify -n 2
```

**Reasoning models need `PHOTOSEARCH_LLM_REASONING_EFFORT=none`.** gemma-4 and
qwen3.x think by default, and on a short tag/JSON task they spend the ENTIRE
`max_tokens` budget on hidden reasoning and return `''` — measured on
gemma-4-26b-a4b (2026-09-21): 765 of 768 tokens reasoning, empty content,
~80 s/photo, every photo UNANSWERED. With the env set, `describe.
_openai_chat_with_retry` adds `reasoning_effort` to the request (3.6 s, real
answer). `chat_template_kwargs.enable_thinking=false` did nothing on LM
Studio. Unset by default so backends that reject unknown fields never see it.
**The fleet's role models are thinking models too** — gemma-4-12b-qat (text,
verify), minicpm-v-4_5 (visual), qwen3.5-9b (describe) — and the evals that
chose them all ran with `none`. The UI fleet (`admin_api._fleet_env`) now sets it
from `rerun.FLEET_REASONING_EFFORT`, and `run-local-replica.sh` exports it. On
2026-10-03 it was missing: gemma took 70 s per text call (297/300 tokens
reasoning, `''` back), so every `category-content` call hit the 10 s cap and
deferred forever (0 of 1,259, no attempt spent, no error logged); minicpm ran
out of `max_tokens` mid-thought (`finish_reason=length`) on 307 photos and
retired them `blocked` with **no `index_errors` row** — the "no usable answer"
path. Symptom: `category-*` blocked or stuck "running" with `done` flat.

This was adopted because Ollama proved unstable on a single 24 GB AMD GPU
(`model runner has unexpectedly stopped` under VRAM contention). LM Studio
caveats: enable JIT loading + max-loaded-models ≥3 + TTL off, and **raise each
model's context length above the 4096 JIT default** (LM Studio splits context
across parallel slots, so a vision describe request 400s with `Context size has
been exceeded`; qwen3.5-9b→16384, gemma→8192 worked).

<!-- moved verbatim from CLAUDE.md (line 1434) on 2026-10-19 -->
### GPU acceleration

CLIP / quality / faces run on the worker's GPU automatically — `clip_embed.py`
and `faces.py` auto-detect CUDA/ROCm (faces is env-gated via
`PHOTOSEARCH_DEVICE`, mirroring clip_embed). The LLM passes (describe /
category-* / keywords / verify) run through Ollama by default — point
`OLLAMA_HOST` at whichever host has a GPU-capable Ollama — or through an
OpenAI-compatible backend (LM Studio) via `PHOTOSEARCH_TEXT_LLM_URL` (see
"Routing LLM passes to LM Studio" above). See the photo-search SKILL.md "GPU
acceleration" section for per-machine setup (Mac Metal, WSL2 + AMD via
librocdxg, WSL2 + NVIDIA).

<!-- moved verbatim from SKILL.md (line 1715) on 2026-10-19 -->
## Distributed Indexing (Worker System)

The worker system lets you offload heavy indexing passes (CLIP, faces, quality,
describe, category-content, category-visual, keywords, verify) to a fast
laptop/desktop while the NAS remains the single source of truth (DB + photos).

The valid pass tokens (enforced by `valid_passes` in `cli.py`) are: `clip`,
`faces`, `quality`, `describe`, `category-content`, `category-visual`,
`keywords`, `verify`. **The old `tags` pass was removed — `--passes tags` now
errors.** It was split into `category-visual` (vision tags), `category-content`
(text categories), and `keywords` (text keywords).

<!-- moved verbatim from SKILL.md (line 1727) on 2026-10-19 -->
### Architecture

- **NAS** runs the web server with `/api/worker/*` endpoints (`worker_api.py`)
- **Worker** (laptop) runs `python cli.py worker`, claims batches via HTTP,
  downloads photo bytes, processes locally, and POSTs results back
- Claims have a TTL (default 30 min) — if a worker dies, photos are auto-reclaimed
- Multiple workers can run concurrently on different passes

<!-- moved verbatim from SKILL.md (line 1735) on 2026-10-19 -->
### Worker fleet — `run-workers.sh` (auto-picks mode)

`./run-workers.sh` auto-detects whether to run the **Docker fleet** (default
on Mac/Linux) or **native** mode (default on WSL2 — launches `cli.py worker`
processes directly from the project venv, GPU-capable, with `OLLAMA_HOST`
auto-resolved to the Windows-host gateway). Same command works on every
machine. Override with `--native` / `--docker`; point at a non-default Ollama
with `--ollama-host URL`. `--status` / `--logs` / `--stop` work in both modes.

#### Docker mode (default on Mac/Linux)

Running workers bare-metal on macOS causes PyTorch's MPS (Metal) allocator to leak
memory over time, eventually crashing the machine. The Docker fleet avoids this by
using CPU-only PyTorch with hard memory limits per container.

```bash
# Launch 4 workers for CLIP embeddings:
./run-workers.sh -s http://<NAS-IP>:8000 -p clip -d /photos/2026 -n 4

# Multiple passes:
./run-workers.sh -s http://<NAS-IP>:8000 -p clip,quality,faces -d /photos/2026

# Fewer workers for heavier passes:
./run-workers.sh -s http://<NAS-IP>:8000 -p describe --collection 3 -n 2

# More memory per worker if needed:
./run-workers.sh -s http://<NAS-IP>:8000 -p quality -d /photos/2026 -m 4g -n 3

# Monitor:
./run-workers.sh --status    # containers + memory usage + recent progress
./run-workers.sh --logs      # tail all worker logs live (Ctrl-C to stop)
./run-workers.sh --stop      # stop all workers
```

**Important:** Use the NAS IP address (not hostname) — Docker containers run in an
isolated network and cannot resolve local DNS names like `nas.local` or mDNS hostnames.

Key options: `-n` (number of workers, default 4), `-m` (memory limit, default 3g),
`--batch-size`, `--model-batch-size`, `--ttl`, `--force`, `--describe-model`,
`--verify-model` (the `cli.py worker` command also has `--category-content-model`,
`--category-visual-model`, `--keywords-model`, though `run-workers.sh` does not
forward them yet).

**`-d` gotcha:** `-d` must be a real subdirectory like `/photos/2026`. `-d /photos`
(the photo root) always 404s — `get_directory_photo_ids` strips the photo_root
prefix and the empty remainder matches no relative DB path. Omit `-d` entirely to
process the whole queue.

CPU-only inference is ~2-3x slower per worker than MPS, but 4 containers running
concurrently with stable memory is faster than 1-2 MPS workers that eventually crash.

#### Ollama for the LLM passes (describe / category-* / keywords / verify)

If `--passes` includes any LLM pass (`describe`, `category-content`,
`category-visual`, `keywords`, `verify`), the script checks `localhost:11434`
and either reuses an existing Ollama or starts a managed container. It then
pre-pulls the required models into the Ollama volume:

- `describe` → pulls `${DESCRIBE_MODEL:-llama3.2-vision}`
- `category-visual` → pulls `${CATEGORY_VISUAL_MODEL:-llava}` (the constrained
  visual-tag task — a different model wins it than free-form describe)
- `category-content` / `keywords` → pull `llama3.2:3b` (text-only)
- `verify` → pulls **all of** `${VERIFY_MODEL:-llava}` (verifier),
  `${DESCRIBE_MODEL:-llama3.2-vision}` (regeneration model), and the visual model.
  Ollama does not auto-pull on request, so they must be present before the pass.

Because describe and category-visual use different vision models, **don't bundle
both vision passes in one `--passes` run on a VRAM-constrained host** — Ollama
would swap models between batches. Run heavy vision passes as separate
invocations, or route to LM Studio (below).

#### Routing the LLM passes to LM Studio (OpenAI-compatible)

Instead of Ollama, the LLM passes can route to an OpenAI-compatible
`/chat/completions` server (LM Studio, llama-server) by setting
`PHOTOSEARCH_TEXT_LLM_URL` (base must end in `/v1`). When set, **all** text and
vision calls go there. The per-pass model is chosen by role env var:

| Pass | role | Env var |
|---|---|---|
| describe (+ regen) | `describe` | `PHOTOSEARCH_LLM_DESCRIBE_MODEL` |
| verify | `verify` | `PHOTOSEARCH_LLM_VERIFY_MODEL` |
| category-visual | `visual` | `PHOTOSEARCH_LLM_VISUAL_MODEL` |
| category-content / keywords | `text` | `PHOTOSEARCH_LLM_TEXT_MODEL` |

`run-workers.sh` has no flag for these — `export` them before a `--native` launch
(the native launcher inherits exported env):

```bash
export PHOTOSEARCH_TEXT_LLM_URL=http://<host>:1234/v1
export PHOTOSEARCH_LLM_DESCRIBE_MODEL=qwen/qwen3.5-9b
export PHOTOSEARCH_LLM_VERIFY_MODEL=google/gemma-4-e2b
export PHOTOSEARCH_LLM_VISUAL_MODEL=google/gemma-4-e2b
export PHOTOSEARCH_LLM_TEXT_MODEL=llama-3.2-3b-instruct
./run-workers.sh --native -s http://<NAS-IP>:8000 \
    -p clip,faces,quality,describe,category-content,category-visual,keywords,verify -n 2
```

Adopted because Ollama was unstable on the single 24 GB AMD GPU (`model runner
has unexpectedly stopped` under VRAM contention). **LM Studio config that
matters:** JIT loading on, max-loaded-models ≥3, model TTL off — and **raise
each model's context length above the 4096 JIT default**. LM Studio splits the
loaded context across parallel slots, so a ~1367-token vision describe request
400s with `Context size has been exceeded` at 4096/N-slots; set qwen3.5-9b →
16384, gemma-4-e2b → 8192, llama-3.2-3b → 8192. JIT loads at the 4096 default,
so save per-model load config (or load manually with TTL off) to make it stick.

**Prefer native Ollama, on the GPU.** Ollama performance is dominated by
whether inference runs on a GPU vs CPU. Switching from CPU-only Docker
Ollama to GPU-accelerated native Ollama is a 3-6x throughput jump and
usually the single biggest perf lever for the describe pass. `run-workers.sh`
checks `localhost:11434` first and reuses any reachable Ollama (managed
container or external), so the setup is one-time per machine and the
script needs no change. See **GPU acceleration: per-machine tuning** below
for the concrete recipes.

**If you must use the managed Ollama container:** raise Docker Desktop memory
(Settings → Resources → Memory) to at least ~24 GiB for the default fleet
(4 workers × 3 GiB = 12 GiB + Ollama ~4-5 GiB + daemon overhead). ~16 GiB has
been observed to still OOM-kill the llama runner. Restart Docker Desktop fully
after changing the limit.

**Diagnostic hint:** `photosearch/describe.py` detects the
`"llama runner process has terminated"` error pattern and prints a one-time
hint to stderr explaining the likely cause and fixes (see
`_maybe_print_runner_oom_hint()`).

**Text-pass Ollama stall (`category-content` / `keywords`):** these
`llama3.2:3b` text passes intermittently trigger a multi-minute Ollama
runner freeze under sustained `NUM_PARALLEL=1` load (serves nothing for
minutes, then bulk-releases). Investigated 2026-05-20: **not** GPU/ROCm
(a CPU-only WSL2 Ollama froze identically), **not** poison input (normal
descriptions), **not** runaway generation (stuck photos replay in ~1s,
`eval_count` ~20). Root cause still open — an Ollama-internal stall
common to both 0.23.x-CPU and 0.24-GPU. Mitigation:
`describe.py:_TEXT_OLLAMA_TIMEOUT_S = 10` bounds these calls to 10s (vs
the 120s `_DEFAULT_OLLAMA_TIMEOUT_S` the vision passes keep) so a stall
aborts fast and the photo is retried later instead of blocking the
single slot. The vision passes (`describe`, `category-visual`) have not
shown this. A restart-on-stall watchdog was rejected — it crashes
workers via the un-retry-wrapped `check_available` and only masks the
symptom.

#### GPU acceleration: per-machine tuning

Measured per-photo describe times on this project's fleet (LLaVA 7B Q4_0,
3 workers, `OLLAMA_NUM_PARALLEL=3`):

| Machine | Path | Per-photo describe |
|---|---|---|
| Mac M-series, 14 cores | Docker `ollama/ollama`, CPU-only | 30-60s |
| Mac M-series, 14 cores | Native `ollama serve` via brew (Metal/MPS) | ~10s |
| WSL2 + RX 7900 XTX | Windows-native Ollama (GPU), worker `OLLAMA_HOST` → Windows host | ~0.8s llava / ~4s llama3.2-vision (warm) |
| WSL2, no supported GPU | Docker `ollama/ollama`, CPU-only | 30-90s |

`OLLAMA_NUM_PARALLEL` should match the per-machine worker count — `run-workers.sh`
sizes it automatically when it creates the managed container, and warns if a
pre-existing container's value doesn't match. Env vars on a running container
are immutable; recreation is required to apply changes (`docker rm -f
photosearch-worker-ollama` then re-run the script).

##### Mac: native Ollama via brew services

3-6x faster than Docker on Apple Silicon (Metal/MPS vs Linux-VM CPU). Steps:

```bash
brew install ollama                          # if not already

# Set NUM_PARALLEL in launchd env BEFORE starting — env is captured at spawn,
# not picked up on the fly.
launchctl setenv OLLAMA_NUM_PARALLEL 3
brew services start ollama
ollama pull llava                            # pull into native Ollama
curl -s http://localhost:11434/api/tags >/dev/null && echo "Ollama OK"
```

Logs land at `/opt/homebrew/var/log/ollama.log`. For reboot persistence of
the env var, edit `~/Library/LaunchAgents/homebrew.mxcl.ollama.plist` to
add an `EnvironmentVariables` dict; otherwise `launchctl setenv` resets at
logout and brew services restarts with `NUM_PARALLEL=1`.

Then start workers normally — `run-workers.sh` detects native Ollama and
skips its Docker fallback:

```bash
./run-workers.sh -s http://<NAS-IP>:8000 -p describe -n 3 ...
```

Operating point: 3 workers + NUM_PARALLEL=3 saturates the M-series GPU at
~97% with describe times steady at ~10s. Adding slots beyond 3 splits the
GPU and inflates per-photo time without throughput gain. Verify with
Activity Monitor → Window → GPU History, or:

```bash
sudo powermetrics --samplers gpu_power -n 3 -i 1000 2>&1 | grep -E "GPU.*active"
```

##### WSL2 Ubuntu with AMD Radeon (ROCm)

Officially supported WSL2 ROCm cards (per AMD docs, early 2026): **RX 7900
XTX/XT/GRE**, **W7900/W7800/W6800**, Instinct MI series. Older RDNA1/2 cards
have no official WSL2 ROCm support.

**The split that works (verified on RX 7900 XTX, 2026-05):**

| Pass | How it reaches the GPU |
|---|---|
| clip, quality | PyTorch-ROCm in the worker's venv |
| faces | onnxruntime-rocm in the worker's venv |
| describe, category-*, keywords, verify | **Windows-native Ollama** (or LM Studio via `PHOTOSEARCH_TEXT_LLM_URL`) — on the Windows host |

**Do NOT run Ollama inside WSL2 for AMD GPUs.** The modern ROCm-on-WSL stack
routes HSA through `/dev/dxg` via AMD's `librocdxg` shim (there is no
`/dev/kfd`). Raw ROCm/HIP/PyTorch work fine through it, but Ollama's
`--ollama-engine` GPU-verification subprocess is incompatible with librocdxg —
it times out at a hardcoded 30 s and Ollama silently falls back to CPU. The
Docker `ollama/ollama:rocm` image doesn't help; the `/dev/kfd` it expects
isn't there. Windows-*native* Ollama, by contrast, sees the GPU through the
normal Windows AMD driver and just works.

Setup:

1. **Windows side:** install the standard AMD Adrenalin driver (the consumer
   one — no Pro Edition needed) and Windows-native Ollama. `ollama pull llava`
   and `ollama pull llama3.2-vision`. Confirm it sees the GPU:
   `%LOCALAPPDATA%\Ollama\server.log` should show
   `inference compute ... library=ROCm ... "AMD Radeon RX 7900 XTX"`.
2. **WSL2 ROCm runtime (for PyTorch/onnxruntime):**
   ```bash
   sudo amdgpu-install -y --usecase=rocm --no-dkms   # the `wsl` usecase was removed in ROCm 7.2
   # then install librocdxg — the rocdxg-roct .deb from
   #   github.com/ROCm/librocdxg/releases  (drops librocdxg.so in /opt/rocm/lib)
   rocminfo | grep -E "Name:|Marketing Name:"        # must list gfx1100 as an HSA agent
   ```
   Every ROCm process on this box needs `HSA_ENABLE_DXG_DETECTION=1` in its
   environment, or HSA init fails.
3. **Worker venv:** `pip install torch torchvision --index-url
   https://download.pytorch.org/whl/rocm7.x` (pin the full `+rocm7.x` version
   or pip won't replace a `+cpu` build). For faces, install AMD's
   `onnxruntime_rocm` wheel from `repo.radeon.com/rocm/manylinux/rocm-rel-7.x/`
   (the PyPI build targets ROCm 6.x and won't load on a 7.x box); it also
   wants `librocm_smi64.so.7` — symlink the system `.so.1` into
   `onnxruntime_rocm.libs/` if missing.
4. **Run the worker** bare-metal with both env vars set:
   ```bash
   HSA_ENABLE_DXG_DETECTION=1 \
   OLLAMA_HOST=http://$(ip route show default | awk '{print $3}'):11434 \
     python cli.py worker -s http://<NAS-IP>:8000 \
     -p clip,faces,quality,describe,category-content,category-visual,keywords,verify -d /photos/<year>
   ```
   The WSL2 NAT gateway IP (`ip route show default`) is the Windows host —
   it can change across WSL restarts, so resolve it dynamically.

The 7900 XTX has 24 GB VRAM. Windows Ollama is a single process (not the
`-n` worker fleet), so describe throughput is tuned via `OLLAMA_NUM_PARALLEL`
on the Windows side, not `run-workers.sh`.

##### WSL2 Ubuntu with NVIDIA (CUDA)

Better-supported path than AMD. Install NVIDIA's WSL2 driver on Windows
(the same driver that powers CUDA in WSL2 Ubuntu), then either:

- Use the standard `ollama/ollama:latest` image with `--gpus all`. Docker
  Desktop + WSL2 NVIDIA passthrough is automatic when the Windows driver
  is present.
- Or install native Ollama in WSL2 (`curl -fsSL https://ollama.com/install.sh | sh`).
  Auto-detects CUDA.

##### Fallback: route describe to whichever machine has the GPU

If a machine can't reach GPU acceleration, leave it on CPU-only passes
(`-p clip,faces,quality`) and push the LLM passes to the GPU machine
alone (`-p describe,category-visual`). The `BEGIN IMMEDIATE` claim-batch
serialization (commit `64eb1ac`) keeps this safe under concurrency —
multiple machines claiming describe batches serialize on the NAS writer
lock without duplicate work.

<!-- moved verbatim from SKILL.md (line 2012) on 2026-10-19 -->
### Bare-Metal Worker CLI

Use the bare-metal worker (not the Docker fleet) when the machine has a **GPU** —
the Docker fleet is CPU-only PyTorch, so a GPU box must run the worker directly
to pick up CUDA/ROCm. For the LLM passes (describe / category-* / keywords /
verify), set `OLLAMA_HOST` to whichever host runs a GPU-capable Ollama (or
`PHOTOSEARCH_TEXT_LLM_URL` for an LM Studio backend). (The Docker fleet is still
preferred on Mac for its memory-stability under MPS.)

```bash
# Run CLIP embeddings for a directory:
python cli.py worker -s http://nas.local:8000 -p clip -d /photos/2026/2026-04-09

# Run full pipeline on a directory:
python cli.py worker -s http://nas.local:8000 -p clip,faces,quality,describe,category-content,category-visual,keywords,verify -d /photos/2026

# Run CLIP + quality for a collection:
python cli.py worker -s http://nas.local:8000 -p clip,quality --collection 3

# Run descriptions with a non-default model:
python cli.py worker -s http://nas.local:8000 -p describe --describe-model moondream

# Run visual tags with a specific model (decoupled from describe):
python cli.py worker -s http://nas.local:8000 -p category-visual --category-visual-model llava

# GPU worker on WSL2 — CLIP/quality/faces via ROCm, describe via Windows Ollama:
HSA_ENABLE_DXG_DETECTION=1 OLLAMA_HOST=http://<windows-host>:11434 \
  python cli.py worker -s http://<NAS-IP>:8000 -p clip,faces,quality -d /photos/2026

# Quick test — one batch only:
python cli.py worker -s http://localhost:8000 -p clip -d /photos/2026/2026-04-09 --one-shot

# Force re-process (clears existing data first):
python cli.py worker -s http://nas.local:8000 -p clip --collection 3 --force
```

Key options: `--batch-size` (photos per claim, default 16), `--model-batch-size`
(inference batch, default 8), `--ttl` (claim TTL minutes, default 30), `--one-shot`
(single batch then exit), `--force` (clear + reprocess, requires --collection or
--directory), `--describe-model` / `--verify-model` / `--category-content-model`
/ `--category-visual-model` / `--keywords-model`.

<!-- moved verbatim from SKILL.md (line 2054) on 2026-10-19 -->
### Worker API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /api/worker/claim-batch` | Claim N unprocessed photos for a pass type |
| `POST /api/worker/submit-results` | Submit processing results for a claimed batch |
| `POST /api/worker/clear-pass` | Clear processing state to allow re-processing |

`submit-results` accepts optional `model` + `model_version` fields (the worker
populates them for the LLM passes — describe / category-* / keywords / verify);
those branches log a row to the `generations` provenance table via
`db.log_generation()`.
| `GET /api/worker/photo-detail/{id}` | Get photo metadata + CLIP embedding (for verify) |
| `GET /api/worker/status` | Queue depth and active claims |

<!-- moved verbatim from SKILL.md (line 2069) on 2026-10-19 -->
### Worker Loop

1. Claims a batch of unprocessed photos from the server
2. Downloads photo bytes to a temp directory
3. Runs the specified indexing pass locally (fast GPU/CPU)
4. POSTs results back to the server
5. Cleans up temp files, repeats until queue is empty

Network retries are built in (exponential backoff on transient errors like
sleep/wake disconnections). If submit fails after retries, photos are released
after TTL expires and will be reclaimed by the next worker.

<!-- moved verbatim from SKILL.md (line 2081) on 2026-10-19 -->
### Key Files

- `run-workers.sh` — Docker fleet launcher script (start/status/logs/stop)
- `photosearch/worker.py` — Client-side worker loop + per-pass processing functions
- `photosearch/worker_api.py` — Server-side FastAPI router (`/api/worker/*`)
- `cli.py` — `worker` command with all CLI options

<!-- moved verbatim from SKILL.md (line 2088) on 2026-10-19 -->
### DB Tables (Worker)

| Table | Purpose |
|-------|---------|
| `worker_claims` | Active batch claims (batch_id, worker_id, photo_ids, expires_at) |
| `worker_processed` | Per-photo processing ledger (photo_id, pass_type) for faces/describe/tags |

---
