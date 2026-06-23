# Local Read-Replica Deployment + Write Tools (M26)

**Status:** 🟡 **M26a partially shipped** (image-proxy fallback, sync script,
sync/status endpoints + `/status` card). **M26b (writes) shipped 2026-06-23** —
all three agent-facing write tools `set_photo_location` / `set_photo_tags` /
`add_to_collection` now live in the shared tool layer (`photosearch/tools.py`)
with the full read-local / write-NAS-authoritative / mirror-local dual-write
loop and all the §3.2 guardrails (explicit `photo_ids`, dry-run-by-default
`confirm`, affected-count cap via `PHOTOSEARCH_WRITE_MAX_ROWS`,
reversible+audited; collection-create additionally needs `create=true`). Both
the in-app agent (`/api/ask`) and the MCP server advertise the write tools and
the gate is re-checked at each call boundary. **Writes are ON by default** —
the deployment lives behind Tailscale (same trust boundary as the deploy
panel); set `PHOTOSEARCH_ALLOW_WRITES=0` to make a deployment read-only. The
agent's system prompt enforces preview→confirm. NAS endpoints:
`POST /api/photos/bulk-set-location` + `bulk-set-tags` (return canonical
`applied`/`results` for the mirror, landed 2026-06-22) and
`POST /api/collections/add-photos` (resolve-or-create by name, returns the
canonical collection id so the mirror re-creates it under the same id via
`PhotoDB.ensure_collection`). Tests: `tests/test_write_tools.py` (guardrails +
mocked-NAS dual-write+mirror for all three tools), `tests/test_web_writes.py`
(endpoints), `tests/test_tools.py` + `tests/test_agent.py` (write-gate).
**Deferred:** structured confirm-over-SSE button (conversational confirm is
enough for v1). Independent of M25. (Status authority: see
[the roadmap index](README.md).)

**M26a shipped so far:**
- `web.py` image routes fall back to the NAS (`PHOTOSEARCH_NAS_URL`) when the
  local original is absent, caching thumbnails/previews locally (replica mode).
  Tests: `tests/test_web_replica.py`.
- `sync-replica.sh` pulls a consistent DB snapshot (dump-db + cat-stream, atomic
  swap) — **verified live: 1.6 GB in ~128 s, all 163,330 photos + embeddings**.
- `POST /api/admin/replica-sync` (SSE) + `GET /api/admin/replica-status`
  (freshness/drift) + a "Local replica" card on `/status` (hidden on the NAS).

**M26a still TODO:** schedule the nightly pull (cron/Task Scheduler); optional
thumbnail pre-warm (bulk mirror) — lazy proxy covers it for now; a real
end-to-end run of `serve` off the replica with local LM Studio.

**Measured reality (correcting §2 estimates):** the DB is **~1.6 GB** (not
~1 GB). The sync is a **full snapshot each run** (~128 s), not a delta — fine
nightly; if sub-daily freshness is wanted, Litestream/WAL-shipping is the delta
path (rsync-into-`/data` is blocked by UGREEN perms, which is why we stream a
container-made snapshot).

## 1. Motivation

The hardware is asymmetric: the NAS (Intel N100, no GPU, 8 GB) is the source of
truth (photos + DB + the sole writer), while the daily-driver desktop has a GPU
and runs LM Studio + the worker stack already. Running the LLM-driven search
*on the NAS* means CLIP text-embedding and vec KNN execute on the N100 and the
agent depends on a network call to the desktop's LM Studio.

Because **search is read-only**, we can flip this: keep a **local read-replica**
of the SQLite DB (+ a thumbnail mirror) on the strong machine and run the whole
experience there — web UI, `/api/ask` agent, and MCP server — off the replica,
with the local LM Studio. The NAS becomes a pure source-of-truth that syncs
nightly. Search compute (CLIP embed on the GPU, KNN, SQL) all moves to the fast
machine; the desktop is self-contained once synced.

Writes (M26b) are the exception — they must reach the NAS (the disposable
replica can't be the system of record) — but we mirror each confirmed write
into the local DB immediately so search reflects it without waiting for a sync.

```
  NAS (dxp4800) — source of truth, SOLE writer        Desktop (GPU) — daily driver
  ┌───────────────────────────────────┐               ┌─────────────────────────────┐
  │ photo_index.db   (writer)          │  nightly rsync│ photo_index.db  (read replica)│
  │ /data/thumbnails (append-only)     │──────────────▶│ thumbnails mirror (~12 GB)    │
  │ /photos (originals)                │   + "sync now" │ photosearch serve  ◀─ browser │
  │ ingest + worker submits = writes   │               │ /api/ask agent ◀─ local LM    │
  │ HTTP write endpoints  ◀────────────┼── write (auth)─┤ MCP server (local clients)    │
  └───────────────────────────────────┘   mirror-local └─────────────────────────────┘
                                                         phone ──Tailscale──▶ desktop
```

## 2. M26a — Local read-replica deployment

**Sync (NAS → desktop).** Reuse the rsync approach `debug-db.sh` already
proves. For a *consistent* snapshot, checkpoint or copy cleanly before pulling:

- On the NAS: `PRAGMA wal_checkpoint(TRUNCATE)` (or `VACUUM INTO snapshot.db`)
  so the replica isn't a torn mid-write file, then rsync `photo_index.db` and
  the `thumbnails/` dir. rsync block-delta ships only changed blocks.
- **DB ≈ 1 GB** (CLIP + face vectors dominate); **thumbnails ≈ 12 GB**
  (measured: ~74 KB avg × 163k). Thumbnails are **append-only** — once
  generated they never change — so after the first pull the nightly delta is
  just new photos (hundreds of MB). Previews (~78 GB) and full-res are **not**
  mirrored; fetch those rare large views from the NAS on demand.

**Cadence:** nightly scheduled pull (cron on the NAS pushing, or a scheduled
task on the desktop pulling) **plus a "sync now" trigger** — a local endpoint
(`POST /api/replica/sync`, SSE) + button for an on-demand refresh (handy right
after a write, or to pull a fresh ingest).

**Run the app off the replica.** No core code change — just env:
```
PHOTOSEARCH_DB=<local replica path>
PHOTOSEARCH_TEXT_LLM_URL=http://localhost:1234/v1     # local LM Studio
PHOTOSEARCH_LLM_AGENT_MODEL=<tool-capable model loaded in LM Studio>
```
`photosearch serve` gives the web UI + `/api/ask` agent locally; optionally run
`python -m photosearch.mcp_server` off the same replica for external MCP
clients (Claude Desktop, etc.). The agent's CLIP text-embed runs on the GPU.

**Thumbnails — mirror (recommended) or lazy-proxy.** The desktop has **no
original photo files**, so it cannot *generate* thumbnails — it must either
have the mirror or proxy to the NAS. Default: mirror the `thumbnails/` dir
(self-contained; phone-over-Tailscale gets thumbnails from the desktop even if
the NAS is asleep). Fallback for disk-constrained laptops: a **lazy
proxy-and-cache** — the thumbnail route, on a local miss, fetches from the
NAS's `/api/photos/{id}/thumbnail` and caches it (self-warming). Previews/full
always proxy to the NAS.

**Cross-platform + remote.** The stack runs on Windows (Docker Desktop / WSL2)
and Mac (Docker) — the worker fleet already runs on both. The phone reaches
whichever machine is up via Tailscale. Multiple replicas (desktop *and* laptop)
are fine — they're disposable read copies; each reconciles on its own sync.

**Scope:** mostly ops. New code is small: the sync script, the `POST
/api/replica/sync` SSE endpoint + button, and the thumbnail-serve-from-mirror-
or-proxy branch.

## 3. M26b — Write tools (read-local / write-NAS-authoritative / mirror-local)

Let the agent *act* on results, not just find them: "find all photos of Calvin
on the beach and update the tags to suit", "set the geocode for everything in
this folder."

### 3.1 Execution model — dual write, NAS-authoritative

The local replica is disposable (overwritten by sync), so it **cannot** be the
system of record. Each confirmed write runs in this order:

1. **NAS HTTP write (authoritative).** POST the change to the NAS endpoint. The
   endpoint applies it and **returns the canonical applied values** (e.g. the
   resolved `place_name` after reverse-geocoding), not just an ack.
2. **Mirror to the local replica (immediate visibility).** Apply *those exact
   returned values* to the local DB via direct SQL — so local search reflects
   the change instantly, with no full sync, and stays byte-identical to what
   the NAS computed (we write the NAS's values, never re-derive).
3. **Nightly sync reconciles.** Since the NAS already has the change, the next
   rsync is a no-op for those rows — the mirror was just an optimistic
   fast-path.

Failure handling keeps the two from diverging dangerously:
- NAS write fails → **abort, do not touch local.** (Local never holds a change
  the NAS lacks.)
- NAS write succeeds, local mirror fails → log it; the change is durable on the
  NAS and lands on the next sync. Worst case: that one edit is invisible
  locally until the nightly pull.

This is CQRS-flavored: reads from the fast local replica, writes to the
authoritative NAS, with an optimistic local mirror for immediacy and the sync
as the reconciliation backstop.

### 3.2 Guardrails (mandatory — LLM-driven bulk mutation)

Bulk-editing a 163k library from natural language is powerful and easy to get
wrong, so every write tool enforces:

- **Explicit id-set scoping.** The write tool takes a concrete `photo_ids` list
  (from a search the agent already ran and the user saw). It NEVER re-runs a
  search internally — the set being mutated is exactly what was shown.
- **Dry-run / preview by default.** `confirm=false` (default) returns the
  affected count + a before→after sample and writes nothing. `confirm=true`
  applies. The agent is system-prompted to never pass `confirm=true` without
  explicit user approval in the conversation — so confirmation is a normal
  chat turn ("update 412 photos? yes"), no special SSE protocol needed for v1.
- **Affected-count cap.** Above a threshold (e.g. 1000 rows) require a second,
  stronger acknowledgement.
- **Reversible + audited.** Location writes use `location_source='manual'`
  (already nullable → one-query rollback); tag writes log to the `generations`
  provenance table. A bad bulk edit is undoable.

### 3.3 Tools + NAS endpoints

| Write tool | NAS endpoint | Status |
|---|---|---|
| `set_photo_location(ids, lat/lon or place)` | `POST /api/photos/bulk-set-location` | ✅ shipped (geocodes `place`, returns + mirrors canonical `applied`) |
| `set_photo_tags(ids, categories?/visual_tags?/keywords?, mode=add|replace)` | `POST /api/photos/bulk-set-tags` | ✅ shipped (logs to `generations`, mirror via replace w/o double-log) |
| `add_to_collection(ids, collection, create?)` | **new** `POST /api/collections/add-photos` | ✅ shipped (resolve-or-create by name; mirror re-creates under the NAS's canonical id via `ensure_collection`) |

Each tool: dry-run preview → confirmed NAS write returning canonical values →
local mirror SQL. The dry-run/confirm + mirror logic lives in the tool layer so
both the MCP server and the in-app agent get it for free.

### 3.4 Agent UX

Conversational: search (read, local) → agent shows results → "update the tags"
→ agent calls the write tool with `confirm=false` → previews "would change N,
here's the diff" → user says yes → agent re-calls with `confirm=true` → applied
on the NAS + mirrored locally → results refresh. A structured confirm
event/button over the `/api/ask` SSE is a future nicety; the conversational
turn (using existing history) is enough for v1.

## 4. Risks / open questions

- **Divergence window.** Between a NAS write and the local mirror (or if the
  mirror fails), local search can briefly differ from the NAS. Bounded by the
  next sync; acceptable for a personal archive. The "sync now" button is the
  manual escape hatch.
- **Two writers via two replicas.** If a desktop and a laptop both issue writes,
  both go to the authoritative NAS (serialized by its writer lock) — safe. Each
  machine mirrors its own write locally and reconciles others on sync.
- **Confirm-over-SSE** (the structured button) deferred — conversational
  confirm first.
- **Security.** The NAS write endpoints are mutation surface; same LAN/tailnet
  trust boundary as the existing deploy panel. If ever exposed wider, gate with
  a token (ties into the MCP auth item in `llm-driven-search.md`).
- **Snapshot method.** `wal_checkpoint(TRUNCATE)` + rsync vs `VACUUM INTO` vs
  Litestream for near-real-time — measure on the real ~1 GB DB; nightly rsync is
  the default, Litestream only if you want sub-daily freshness.

## 5. Sequencing

1. **M26a** — replica sync + run serve/MCP off it + thumbnail mirror + "sync
   now". Read-only, immediately useful, ~no core code change.
2. **M26b** — write tools with the dual-write model + guardrails (needs the new
   `bulk-set-tags` NAS endpoint).

M25 (backfill/maintenance sweep) is independent; pick the order by which pain is
louder. The write tools (M26b) also make some M25 backfills user-triggerable
("re-geocode this folder") rather than cron-only.

## Related

- `docs/plans/llm-driven-search.md` — M24 (shipped). §8 listed write tools as
  future work; this is that. The tool layer + agent it built are the base here.
- `docs/plans/backfill-maintenance-sweep.md` — M25. Overlaps on geocode/tag
  backfills; M26b makes some of them interactive.
- `debug-db.sh` — the existing rsync-the-prod-DB-locally tool the M26a sync
  productionizes.
