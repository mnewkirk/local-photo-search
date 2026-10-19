# Troubleshooting

Read when asked why something was slow or what a request was for (the request log).

<!-- moved verbatim from SKILL.md (line 2097) on 2026-10-19 -->
## Troubleshooting

<!-- moved verbatim from SKILL.md (line 2099) on 2026-10-19 -->
### Request log — "why was that slow?" / "what was that for?"

Every `/api/*` request appends one JSON line to `request_log.jsonl` beside the
DB (`/data/request_log.jsonl` on the NAS, `./request_log.jsonl` on the
replica); MCP tool calls go to `request_log.mcp.jsonl`. Rotated at 20 MB × 5;
written by a background thread so a starved disk never stalls a request.
`PHOTOSEARCH_REQUEST_LOG=<path>|0` moves or disables it. Fields: `ts`,
`method`, `path`, `query`, `status`, `ms`, `source`, `intent`,
`intent_inferred`, `page` (the Referer), `client`, `streaming` (SSE: `ms` is
time to headers).

`source`: `ui` (browser, has a Referer) · `claude` (sent
`X-Photosearch-Source`) · `claude-mcp` · `agent` (Ask; each tool call logged
with the question) · `replica` (the replica calling the NAS — every such
call site uses `request_intent.outbound_headers(purpose)`, so its intent reads
"Fetch preview of photo N … - for: <the UI request that needed it>") ·
`script` (any other Python HTTP client) · `worker` · `other`. A new replica→NAS
call must send `outbound_headers`; a new background thread or pool must be
wrapped in `carry_context` (`web._carry_context`) or its NAS calls lose the
"for:" part. `intent`: a stated header / the Ask question /
inferred from page + endpoint + parameters by `request_intent.infer_intent`
(add a rule there for a new endpoint).

```bash
$DC run --rm photosearch request-stats --last 500             # per-endpoint latency
$DC run --rm photosearch request-stats --recent 50            # who and why, newest last
$DC run --rm photosearch request-stats --source ui --slowest 20
./.venv/bin/python cli.py request-stats --db photo_index.db.local   # replica
```

Page polling and worker traffic are hidden unless `--include-polling`. The
container's own stdout log is discarded on every redeploy; this file is not.

**"Error: No such command 'python'" / "'sh' / '-c'" when running ad-hoc commands** —
`docker-entrypoint.sh` routes every non-"serve"/"index" first arg to `python cli.py <arg> …`,
so `docker compose … run --rm photosearch python -c "…"` becomes `python cli.py python -c "…"`
and Click rejects it. Override the entrypoint — but note `--entrypoint` is a `docker compose
run` flag, so it must come *before* the service name (it can't be tacked onto an existing
`$DC="docker compose … run --rm photosearch"` alias). Either re-expand the full command
or use dedicated aliases:
```bash
docker compose -f docker-compose.nas.yml run --rm --entrypoint python photosearch -c "<snippet>"
docker compose -f docker-compose.nas.yml run --rm --entrypoint bash   photosearch -c "<shell cmd>"

# Or as aliases:
DCPY="docker compose -f docker-compose.nas.yml run --rm --entrypoint python"
DCSH="docker compose -f docker-compose.nas.yml run --rm --entrypoint bash"
$DCPY photosearch -c "<snippet>"
```
Same pattern for sqlite3, pip, etc. — anything that isn't a `cli.py` subcommand needs
`--entrypoint`. This trips up diagnostics constantly; reach for it first when an in-container
one-liner fails.

**Empty / all-zero results from `PhotoDB()` ad-hoc snippets** — `PhotoDB()` with no args
defaults to the relative path `"photo_index.db"`, which inside the container resolves to
`/app/photo_index.db` (a fresh empty DB), not the real `/data/photo_index.db`. The
`PHOTOSEARCH_DB` env var is only honored by `cli.py`'s `--db` defaults, not by the `PhotoDB`
constructor. Always pass it explicitly:
```python
import os
from photosearch.db import PhotoDB
with PhotoDB(os.environ['PHOTOSEARCH_DB']) as db:
    ...
```
If you hit this, also `rm -f /app/photo_index.db` afterwards — the empty stub gets
persisted into the container's writable layer on first access.

**"database is locked"** — Another process holds a write lock. Check `docker ps` for
concurrent jobs. WAL mode + `PRAGMA busy_timeout=60000` resolves short locks automatically.

**"Database not found: photo_index.db"** — `PHOTOSEARCH_DB` env var not set. Confirm
Dockerfile has `ENV PHOTOSEARCH_DB=/data/photo_index.db`.

**InsightFace downloading every run** — `/data/.insightface/` not persisted. Confirm Docker
volume mount and `ENV INSIGHTFACE_HOME=/data/.insightface`.

**No SSE progress reaching browser** — Three common causes: (1) using `asyncio.get_event_loop()`
instead of `asyncio.get_running_loop()` in async endpoints, (2) missing terminal event
type in the generate() stream check, (3) `InterruptedError` swallowed by generic
`except Exception` instead of being re-raised.

**Worker containers exit immediately with "Connection refused"** — Docker containers
can't resolve local hostnames (mDNS, `.local`, NAS hostnames). Use the NAS IP address
instead: `./run-workers.sh -s http://192.168.x.x:8000 ...` Find it with `ping nas-hostname`.

**Worker memory leak / Mac crash** — PyTorch MPS allocator leaks memory in long-running
workers. Use the Docker fleet (`./run-workers.sh`) instead of bare `python cli.py worker`
for sustained multi-worker runs. Docker forces CPU-only with hard memory limits.

**"llama runner process has terminated: %!w(<nil>)" (status 500)** — The llama
runner subprocess inside Ollama was OOM-killed, almost always because the managed
Docker Ollama container is sharing Docker Desktop's VM memory with worker containers
and LLaVA's ~4.3 GiB working set doesn't fit. Fixes (easiest first):
1. Use native `ollama serve` on the host. Stop the fleet (`./run-workers.sh --stop`),
   run `ollama serve` (or launch Ollama.app), `ollama pull llama3.2-vision && ollama pull llava`
   (describe + tags/verify models), then relaunch the fleet — it will detect and
   reuse the native Ollama.
2. Raise Docker Desktop memory to ~24 GiB (Settings → Resources → Memory) and
   restart Docker Desktop fully. ~16 GiB is too tight for the default 4×3g fleet
   plus Ollama.
3. Reduce fleet pressure: `./run-workers.sh -n 2 -m 2g ...`.

`photosearch/describe.py` prints a one-time diagnostic hint when this error pattern
is detected, so you'll see this guidance in the worker logs as well.

**Google Photos upload shows 0 uploaded, N re-synced** — `batchAddMediaItems` silently
fails for photos removed via Google Photos UI. Use "Re-upload" with specific photos selected.

**Panel-triggered restart leaves a container stuck in "Created" with a
rename-suffix name** (e.g. `54350640f483_photosearch`) — almost always means
the compose subprocess was killed mid-restart because it was running inside
the container being recreated. With the helper-container architecture this
shouldn't happen anymore; if it does, recovery is:
```bash
docker rm -f <stuck-container-id>
docker rm -f photosearch-restart-helper   # if it didn't self-clean
docker compose -f docker-compose.nas.yml up -d --no-deps photosearch
```
To diagnose what happened during a failed restart, walk the docker event log:
```bash
docker events --since 10m --until 30s 2>&1 | grep photosearch | tail -30
```
You'll see the full lifecycle: `container create` (new), `container kill
signal=15` (SIGTERM old), 60s wait, `container kill signal=9` (SIGKILL old),
`container die exitCode=137`, and then either `container start` (new — happy
path) or nothing (compose died before reaching that step).

**Old container exits 137 on restart but new one IS up** — uvicorn's
"waiting for connections to close" couldn't drain in `stop_grace_period: 60s`
because workers (or other clients) kept opening fresh connections. The 503-
on-shutdown middleware fixes this for worker traffic; for browser polling
(e.g. `/api/worker/status` every 5s) the connections are short-lived enough
not to matter. If you've raised drain time further and still hit 137,
something is hanging an HTTP request open — check `docker logs photosearch
2>&1 | tail -50` for the last accepted requests before shutdown.

**Workers crash on `RuntimeError: claim-batch failed (503)`** — old worker
code (pre-577a3b3) raised on any non-2xx. New `WorkerClient._request()`
transparently sleeps `Retry-After` and retries. Pull + restart your worker
fleet after deploying the server-side shutdown middleware.

**Deploy panel shows "git repo not mounted at /repo"** — `REPO_DIR` was set
to empty string in the shell that ran `docker compose up`. Bash's
`${REPO_DIR:-default}` resolves to the default ONLY when unset, not when
empty, so the volume source became `:/repo` and docker silently dropped
the mount. Fix: `unset REPO_DIR` before the compose call (or set it
explicitly), then recreate the container.

---
