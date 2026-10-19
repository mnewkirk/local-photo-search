# Plan: shrinking the fixed per-session context for /home/mattn/local-photo-search

Date 2026-10-18. Claude Code 2.1.302.

**Status 2026-10-19: steps 0-7 EXECUTED** (tag `pre-context-diet`; commits `8441ace`, `4fd6ee7`, `9bb2210`). CLAUDE.md 258,266 → 7,803 B; SKILL.md 147,932 → 5,905 B; MEMORY.md 18,218 → 6,719 B; nine rules 5.5-7.1 KB. Deviations: 168 headings (3 sections were newer than this plan: category-content grounding, playback proxies, ZIP unwrap); every section was COPIED verbatim to docs/notes/ or reference/ (nothing condensed in place), so reference/ holds runbooks rather than hand-written summaries; the optional PostToolUse hook (5d) was not added. Step 8 (measure) via `evals/context_eval/` before/after runs.

## 0. Mechanics verified (code.claude.com/docs/en/memory and /skills, fetched today)

| Mechanism | Verified behaviour | Design consequence |
|---|---|---|
| `CLAUDE.md` | Loaded in full at launch (up to 4 MiB). Docs: "target under 200 lines"; longer files "reduce adherence". | Must physically shrink; cannot be hidden. |
| `@path` imports | Expanded and loaded at launch, "don't reduce context cost". | Do not use imports to "organize" the big file. |
| `.claude/rules/*.md` | Rules with `paths:` frontmatter load only when Claude Reads/Writes/Edits (or `cat`/`head`s) a matching file; rules WITHOUT `paths:` load at launch like CLAUDE.md. Glob syntax: `paths:\n  - "photosearch/face*.py"`. Discovered recursively. | Subsystem traps (class b) go here. A rule only fires on file access, not on Grep or Bash ops, so CLAUDE.md keeps a one-line index naming each rule file. |
| Skills | Only name+description (combined cap 1,536 chars) are in context at startup. On trigger the whole SKILL.md body enters as one message and persists for the session. Supporting files (`reference/*.md`) are not loaded until Claude reads them. Docs: "Keep SKILL.md under 500 lines". | SKILL.md becomes a short navigator; bulk goes to `reference/`. |
| Auto memory | `MEMORY.md`: first 200 lines or 25 KB loaded every session; topic files loaded on recall. Not loaded into subagents. | Index lines must be one short line each. |
| `/context` | Lists Memory files, rules, skills and their token counts. | Measurement tool. |

Repo facts that matter: no `.claude/rules/` exists yet; `.claude/settings.json` has no hooks; the only MCP server is user-scope `photosearch` (18 tools + a long instructions block); `tests/test_frontend_refs.py` already shows the pattern for a repo-file-checking pytest; `docs/plans/README.md` declares itself "the single source of truth for plan/milestone status" yet CLAUDE.md carries a 9.6 KB "Planned milestones" copy and SKILL.md a 6.3 KB one.

## 1. Target budget

| File | Now | Target (hard cap enforced by test) | Why |
|---|---|---|---|
| `CLAUDE.md` | 246,797 B / 4,173 lines / ~62k tok | ≤ 24,576 B, ≤ 300 lines (~6k tok); aim ~16 KB | Docs recommend <200 lines; this repo has ~15 genuinely universal rules plus a subsystem index, which fits in ~150 lines. 12k tok (the prompt's example) is too generous: it still leaves room for narrative creep. |
| `.claude/skills/photo-search/SKILL.md` | 146,890 B / 2,647 lines / ~37k tok | ≤ 20,480 B, ≤ 300 lines (~5k tok) | Navigator + project layout + where-to-look table. Everything procedural goes to `reference/*.md`, read on demand. |
| `.claude/rules/*.md` (new) | 0 | each ≤ 8,192 B; all carry `paths:`; typical session touches 1-2 → ~2-4k tok | Loaded only when the subsystem's code is opened. |
| `MEMORY.md` | 17,510 B / 60 lines (~4.4k tok) | ≤ 8,192 B, ≤ 60 lines, ≤ 140 chars/line (~2k tok) | Index role only; detail already lives in topic files. |
| Fixed cost at session start | ~66k tok (103k with skill) | ~8k (≈13k with skill, ≈16k with two rules) | ~85% reduction. |

## 2. CLAUDE.md disposition

Destinations:
- **KEEP** — stays in CLAUDE.md, rewritten to 1-6 lines.
- **R:x** — condensed trap list in `.claude/rules/x.md` (path-scoped; one bullet per trap, heading name kept verbatim as the bullet's bold lead).
- **N:x** — verbatim relocation of the full section (heading intact) to `docs/notes/x.md`. This is the archive that makes the preservation check trivial; rules and skill references then point to it by heading.
- **REF:x** — operator runbook to `.claude/skills/photo-search/reference/x.md`.
- **DEL** — delete, content exists elsewhere (noted where).

Pointer lines that remain in CLAUDE.md (one "subsystem index" block, ~25 lines total):

```
## Subsystem traps (path-scoped rules load when you open the code; read them first if you only run ops)
- faces      → .claude/rules/faces.md       (rival test · trusted labels only · unmatch-person · one-person-per-photo invariant · match_source='rejected' · quality pre-filter · Unclustered bucket · encoding cache)
- llm-passes → .claude/rules/llm-passes.md  (Bad LLM output is a failure · provenance: log the model that RAN · Ollama stall timeout · new description re-queues · failed SERVER-side write must not burn a retry · category-visual derived vs perceived · LM Studio routing · percentile refresh is chunked)
- ingest     → .claude/rules/ingest.md      (phone ingest · refile-unknown-camera · six batch states · faces "no output" means done · never /workers/fleet-status for "queued" · rank_measure optional · deploy NAS before desktop)
- maintenance→ .claude/rules/maintenance.md (geocode memory is per-ROW · sweep STRICT-ONLY match_faces · replica mode gated · vec0 orphans · clip-claim infinite re-claim · crontab perms)
- workers    → .claude/rules/workers.md     (worker passes · -d /photos 404 · aesthetics pass · GPU/ROCm · never pgrep -f a bare string)
- video      → .claude/rules/video.md
- print      → .claude/rules/print.md       (Topaz Autopilot artifacts · strength blend · seam mode = mounting · seam misalignment integer-dpi grid)
- frontend   → .claude/rules/frontend.md    (no linter: check-frontend-refs.js · PS.poll never setInterval · status live actions · deploy panel helper container · replica freshness card)
- search     → .claude/rules/search.md      (v34 indexes match query TEXT · _compose_scope · name extraction · request timing log source+intent · inferred geotagging M19)
Full history for every trap: docs/notes/<same name>.md. Runbooks: .claude/skills/photo-search/reference/.
```

Section table (bytes measured today, code-fence `#` lines excluded):

| # | Section (L=line) | B | Class | Dest | Remains in CLAUDE.md |
|---|---|---|---|---|---|
| 1 | `# Photo Search` + `## Tech Stack` (L1) | 446 | a | KEEP | as is |
| 2 | `## Key Patterns — Follow These` (L12) | 666 | a | KEEP | merged with SKILL.md patterns 7-10 → 10 one-liners |
| 3 | `## Database` (L23) | 2909 | a/b | KEEP 6 lines (file name, SCHEMA_VERSION source of truth, bump rule, key tables); rest R:search + N:search | "Schema v38; indexes/attempts/generations: rules/search.md" |
| 4 | `## Richer reverse geocoding (optional)` (L69) | 1192 | c | REF:nas-operations | index line maintenance |
| 5 | `### The memory cost is per-ROW, not the KDTree — and it OOM-killed the NAS` (L100) | 3190 | b+d | R:maintenance (3 lines) + N:maintenance | trap name in index |
| 6 | `## API request timing log` (L155) | 3450 | b | R:search (5 lines) + REF:troubleshooting | "every curl sends X-Photosearch-Source/Intent" stays as Key Pattern |
| 7 | `## Debugging against the prod DB locally` (L217) | 1236 | c | REF:nas-operations | "debug-db.sh / run-local-replica.sh: reference/nas-operations.md" |
| 8 | `## NAS Docker Commands` (L244) | 2568 | a | KEEP ~12 lines: the two aliases, `--entrypoint` before service name, `PhotoDB()` needs explicit path, git `-c safe.directory` one-liner; rest REF:nas-operations | trimmed block |
| 9 | `### Watching a long job: never pgrep -f a bare string` (L298) | 1898 | a | KEEP 2 lines + N:workers | "never `pgrep -f` a bare string; poll a marker/PID" |
| 10 | `## Distributed Indexing (Worker)` (L340) | 5993 | c | REF:workers | index line workers |
| 11 | `### Worker passes` (L441) | 1209 | b | R:workers | — |
| 12 | `### VLM aesthetic scoring (the aesthetics pass)` (L466) | 3341 | b | R:workers (3 lines) + N:llm-passes | — |
| 13 | `### The percentile refresh is chunked — and must stay that way` (L519) | 3350 | b+d | R:llm-passes (2 lines) + N:llm-passes | trap name |
| 14 | `### Per-pass model strategy (Ollama defaults)` (L573) | 1422 | b | R:llm-passes (table) | — |
| 15 | `### category-visual: derived capture facts vs perceived qualities` (L599) | 23784 | d (+b) | R:llm-passes (5 lines: DERIVED/RETIRED/FROZEN/PERCEIVED groups, derived wins, PROMPT_SECTIONS must partition) + N:llm-passes verbatim | trap name |
| 16 | `### Routing LLM passes to LM Studio (OpenAI-compatible)` (L984) | 5856 | c | REF:workers + R:llm-passes (env-var table) | — |
| 17 | `### Model evals for every LLM pass (/eval/models)` (L1077) | 2145 | c | REF:evals (new) | — |
| 18 | `### A new description re-queues what was derived from it` (L1112) | 3644 | b | R:llm-passes (2 lines) + N:llm-passes | trap name |
| 19 | `### Bad LLM output is a failure, not a result` (L1169) | 7832 | b+d | R:llm-passes (6 lines: stop reason, per-role max_tokens, UnusableAnswer → failure row) + N:llm-passes | trap name |
| 20 | `### Provenance: log the model that RAN, not the one configured` (L1301) | 1751 | b | R:llm-passes (1 line) + N | trap name |
| 21 | `### Ollama stall on the text passes — tight per-call timeout` (L1333) | 1188 | b | R:llm-passes (1 line) + N | trap name |
| 22 | `### GPU acceleration` (L1354) | 627 | c | REF:workers | — |
| 23 | `### A failed SERVER-side write must not burn a retry attempt` (L1366) | 5406 | b+d | R:workers (2 lines) + N:workers | trap name |
| 24 | `## Video` (L1449) | 8999 | b/c | R:video (traps, ≤ 2 KB) + REF:video (deploy/backfill runbook, or point at `docs/plans/video-support.md` Step 8) + N:video | index line video |
| 25 | `## Vec0 orphan cleanup` (L1589) | 825 | b | R:maintenance | trap name |
| 26 | `## Non-image rows & the clip-claim infinite re-claim` (L1608) | 2478 | b+d | R:maintenance (2 lines) + N | trap name |
| 27 | `## Phone-photo daily ingest` (L1652) | 4630 | c | REF:nas-operations (dedupe with SKILL.md "Phone-photo ingest", 4580 B near-duplicate) | index line ingest |
| 28 | `### Repairing the folders already misfiled — refile-unknown-camera` (L1728) | 14713 | c+d | REF:ingest-runbooks (the command + collision semantics, ~2 KB) + N:ingest verbatim | trap name |
| 29 | `### Scheduling / crontab perms on the NAS` (L1959) | 1507 | c | REF:nas-operations (also in SKILL.md "Cron / scheduling") | — |
| 30 | `### Syncthing receiver` (L1988) | 798 | c | REF:nas-operations | — |
| 31 | `## Adding Features` (L2002) | 1055 | a | KEEP | as is (the longer SKILL.md "Adding New Features" moves to REF:adding-features) |
| 32 | `## Name extraction in search queries` (L2018) | 615 | b | R:search | — |
| 33 | `## Inferred geotagging (M19)` (L2029) | 4383 | b/c | R:search (3 lines) + N:geotagging (merge with SKILL.md's 2615 B section of the same heading) | trap name |
| 34 | `## Status page live actions` (L2114) | 1452 | b | R:frontend | — |
| 35 | `## Replica freshness card + "Sync from NAS" (M26a UI)` (L2140) | 590 | b | R:frontend | — |
| 36 | `## On-demand pass re-runs + status-page fleet launcher (M28)` (L2150) | 2396 | b | R:frontend (2 lines) + N:frontend | — |
| 37 | `## Running the tests the way CI does` (L2188) | 1215 | a | KEEP 4 lines (`scripts/test-like-ci.sh`; 3 divergences as one line each; worktree gitdir caveat) | trimmed |
| 38 | `## Local Topaz upscaling (M30)` (L2211) | 4466 | b/c | R:print + REF:print | index line print |
| 39 | `### Artifacts: Autopilot merges its own passes in` (L2286) | 2550 | b+d | R:print (1 line) + N:print | trap name |
| 40 | `### Strength: Topaz has none, so we blend` (L2335) | 2715 | b+d | R:print (1 line) + N:print | trap name |
| 41 | `## Split — multi-panel print planner (M31)` (L2383) | 2879 | b/c | R:print + REF:print | — |
| 42 | `### Seam mode decides how the sheets must be MOUNTED` (L2431) | 1172 | b | R:print (1 line) + N | trap name |
| 43 | `### Seam misalignment in print (files were fine)` (L2452) | 7265 | d | R:print (1 line: integer-dpi grid, `test_panels_are_exactly_sheet_sized`) + N:print verbatim | trap name |
| 44 | `## Deploy panel (Version / Build / Restart)` (L2570) | 2954 | b | R:frontend (3 lines incl. REPO_DIR empty-string trap) + N:frontend; dedupe with SKILL.md /status block | trap name |
| 45 | `## Face clustering` (L2634) | 1292 | b | R:faces | — |
| 46 | `### Quality pre-filter (schema v20)` (L2662) | 1638 | b | R:faces | trap name |
| 47 | `## Reviewing a shoot in the UI: person inspector + team review` (L2700) | 2792 | c | REF:faces-review (new) | — |
| 48 | `## Face-label integrity: is this face actually that person?` (L2750) | 3036 | b | R:faces (3 lines) + N:faces | — |
| 49 | `### The rival test — the right face is usually in the same photo` (L2803) | 3465 | b+d | R:faces (2 lines) + N:faces | trap name |
| 50 | `### The panel asks about TRUSTED labels only` (L2863) | 1921 | b | R:faces (1 line) + N | trap name |
| 51 | `### Bulk-undoing one person's bad labels — unmatch-person` (L2897) | 4276 | c | REF:face-management-cli + R:faces (1 line: reversible, marks rejected, RUN ON NAS) | trap name |
| 52 | `### Finding more of one person: ➕ More of this kid on /faces` (L2971) | 1332 | c | REF:faces-review | — |
| 53 | `### The high-level view: 📊 Label health on /faces` (L2992) | 2352 | c | REF:faces-review | — |
| 54 | `### match_source='rejected' — a human "no" that survives auto-matching` (L3034) | 1635 | b | R:faces (1 line) + N | trap name |
| 55 | `## Per-shoot "best of" curation — scripts/rank_shoot.py` (L3062) | 4527 | c | REF:face-management-cli (dedupe with SKILL.md "Per-shoot curation", 1765 B) | — |
| 56 | `## Frontend has no linter — scripts/check-frontend-refs.js` (L3139) | 1151 | a | KEEP 2 lines (run it; shared helpers go on `PS.*`) + N:frontend for the IIFE story | Key Pattern |
| 57 | `### Poll with PS.poll, never setInterval` (L3163) | 1302 | a | KEEP 1 line + N:frontend (the 2026-09-19 incident) | Key Pattern |
| 58 | `## Splitting attractor clusters (M18)` (L3186) | 848 | c | REF:face-management-cli | — |
| 59 | `## Face merge suggestions (M18)` (L3205) | 2616 | b/c | REF:face-management-cli (dedupe with SKILL.md "Unknown-face clustering & merge suggestions") | — |
| 60 | `## Face over-matching cleanup & the one-person-per-photo invariant` (L3260) | 3391 | b | R:faces (3 lines) + N:faces | trap name |
| 61 | `### Face-encoding cache` (L3315) | 783 | b | R:faces (1 line) | trap name |
| 62 | `### Desktop-as-client face recompute (heavy clustering off the NAS)` (L3329) | 1704 | c | REF:face-management-cli | — |
| 63 | `### Face detection / matching gotchas` (L3359) | 583 | b | R:faces | — |
| 64 | `### Reviewing never-clustered faces (the "Unclustered" bucket)` (L3368) | 2235 | b/c | R:faces (1 line) + REF:faces-review | trap name |
| 65 | `### Searchable face clusters` (L3406) | 880 | b | R:faces | — |
| 66 | `### EXIF-oriented image dimensions` (L3420) | 497 | b | R:faces | — |
| 67 | `## Maintenance sweep & duplicate pruning (M25 + M28)` (L3429) | 4875 | b/c | R:maintenance (4 lines) + REF:nas-operations | index line maintenance |
| 68 | `### Scheduling + replica-mode maintenance (2026-07-17)` (L3511) | 10526 | c+d | REF:nas-operations (cron lines, flock, STRICT-ONLY, replica gating ≈ 2 KB) + N:maintenance verbatim; DELETE the SKILL.md copy (4313 B, same heading) | trap names |
| 69 | `## Ingest batches (/batches)` (L3681) | 1678 | b | R:ingest | index line ingest |
| 70 | `### The six states, and the two ways count_unprocessed == 0 lies` (L3708) | 2991 | b | R:ingest (3 lines) + N:ingest | trap name |
| 71 | `### faces is the one pass where "no output" means done` (L3757) | 1483 | b | R:ingest (1 line) + N | trap name |
| 72 | `### Where "queued" comes from — never /workers/fleet-status` (L3780) | 1696 | b | R:ingest (1 line) + N | trap name |
| 73 | `### The non-blocking-lock cache — the incident it answers` (L3807) | 1147 | d | N:ingest | trap name |
| 74 | `### rank_measure is a NAS step, and it is OPTIONAL` (L3825) | 1631 | b | R:ingest (1 line) + N | trap name |
| 75 | `### What's deliberately not in "ready" — and why` (L3852) | 4225 | d | N:ingest | trap name |
| 76 | `### Endpoints and CLI` (L3916) | 1445 | c | REF:api-endpoints (Batches block) | — |
| 77 | `### Deploy the NAS before the desktop` (L3942) | 742 | b | R:ingest (1 line) | trap name |
| 78 | `## Measured sharpness (schema v33) — column + throttled backfill` (L3955) | 3825 | b/c | R:workers (2 lines) + N:sharpness (or append to `docs/plans/sharpness-measurement.md` as "As built") | — |
| 79 | `## Planned milestones (see docs/plans/)` (L4019) | 9641 | e | DEL after folding any status newer than README's "Last reconciled 2026-06-23" (video BUILT 10-17, search-indexes SHIPPED 10-04, ingest-batch SHIPPED 09-19, M25/M26/M27 shipped) into `docs/plans/README.md` | "Plan status: docs/plans/README.md only" |
| 80 | `## Detailed Reference` (L4170) | 214 | a | KEEP, reworded to point at the skill's `reference/` | 2 lines |

Totals by destination: KEEP ≈ 7-9 KB after rewrite; rules ≈ 40 KB across 9 files (each ≤ 8 KB; `llm-passes` and `faces` are the ones to watch); `docs/notes/` ≈ 150 KB verbatim; `reference/` receives ≈ 45 KB from CLAUDE.md plus ≈ 115 KB from SKILL.md.

## 3. SKILL.md disposition

Target shape: frontmatter (unchanged description, it triggers well) + ≤ 300 lines: 1-paragraph purpose, `## Project Layout` tree (trimmed to the modules list), `## Where to look` table (topic → `reference/<file>.md` → one-line summary), `## Before you operate` (5 bullets: read the matching `.claude/rules/*.md`; send X-Photosearch-Source/Intent; writes go NAS-authoritative; `docs/plans/README.md` is status truth; new narratives go to `docs/notes/`).

| SKILL.md section | B | Class | Dest |
|---|---|---|---|
| Tech Stack table | 652 | e (dup of CLAUDE.md) | DEL; CLAUDE.md's 3 lines suffice |
| Project Layout | 3316 | a (for skill) | KEEP, trimmed to ~2 KB |
| Database Schema (v38) + Search performance indexes (v34) | 8653 | c/b | `reference/schema.md`; the 5 index rules also in R:search |
| Key Patterns 1-10 | 3400 | a | DEL from skill; patterns 7 (SSE terminal events), 8, 9 (face IDs stable, cluster IDs not), 10 (state intent) merge into CLAUDE.md Key Patterns |
| API Endpoints (all 14 subsections incl. Video, Faces 6282 B, Batches, geotag) | ~19.5 KB | c | `reference/api-endpoints.md` |
| Inferred geotagging (M19) | 2615 | b/c | merge into N:geotagging with CLAUDE.md's section; 3 lines in R:search |
| Google Photos Integration (3 subsections) | 1129 | c | `reference/google-photos.md` (+ memory `project_gphotos_upload_500storm_recovery.md` pointer) |
| Stacking System; Shoot Review / Culling; Frontend stack filtering | 2351 | c | `reference/indexing-passes.md` (stacking) and `reference/frontend-pages.md` (review) |
| Shared Frontend Components; /status page (7021); /faces review panels (3645); encoding cache; no-linter; /faces URL params; /merges architecture | ~15 KB | c | `reference/frontend-pages.md`; the REPO_DIR empty-string trap duplicated in R:frontend |
| Per-shoot curation and label checking (CLI) | 1765 | c | `reference/face-management-cli.md` (dedupe with CLAUDE.md §55) |
| NAS Operations: Docker commands, Volume layout, Full indexing sequence, Phone-photo ingest, Cron, Scheduling + replica-mode maintenance | ~15 KB | c | `reference/nas-operations.md`; Docker-commands block deduped with CLAUDE.md §8 (verbatim duplicate today) |
| Indexing Types — Detailed Reference: Pass details, Parallelization, Collection re-indexing, Key model details, Stacking parameters | ~9.9 KB | c | `reference/indexing-passes.md` |
| Face management commands; Unknown-face clustering & merge suggestions; Merge review page; Vec0 orphan cleanup; Face over-matching cleanup, dedup & desktop recompute | ~12.4 KB | c | `reference/face-management-cli.md`; vec0 + over-matching invariant lines also in R:faces / R:maintenance |
| Distributed Indexing: Architecture, Worker fleet run-workers.sh (13713), Bare-Metal Worker CLI, Worker API, Worker Loop, Key Files, DB Tables | ~18.6 KB | c | `reference/workers.md` (merge with CLAUDE.md §10, §16, §22) |
| Troubleshooting → Request log | 8423 | c | `reference/troubleshooting.md` |
| MCP server (M24a); In-app Ask agent (M24b); Running off a local read-replica (M26a) | ~13.6 KB | c | `reference/mcp-ask-replica.md` |
| Planned milestones | 6301 | e | DEL (docs/plans/README.md) |
| Adding New Features (6 subsections) | 3770 | a/c | `reference/adding-features.md`; CLAUDE.md keeps its 12-line version |

Resulting `reference/` set: `schema.md`, `api-endpoints.md`, `frontend-pages.md`, `nas-operations.md`, `indexing-passes.md`, `face-management-cli.md`, `faces-review.md`, `workers.md`, `troubleshooting.md`, `mcp-ask-replica.md`, `print.md`, `evals.md`, `adding-features.md`, `google-photos.md`, `video.md`. Each starts with a 3-line "when to read this" so the navigator table in SKILL.md can be generated from the first lines.

## 4. Memory hygiene

Current: 60 index lines, 17,301 B; 23 lines exceed 300 chars (max 647); 23 entries are closed (DONE/SHIPPED/FIXED/COMPLETE).

Rules for the rewrite: one line per entry, ≤ 140 chars: `- [title](file.md) — <one clause: what to do or where it is>`. Detail stays in the topic file. Status words go at the END of the title in brackets, not as a prefix, so grep for a topic still works.

| Action | Entries (files in `~/.claude/projects/-home-mattn-local-photo-search/memory/`) |
|---|---|
| **Pointer-ize** (canonical text now lives in repo docs/notes or rules; line becomes "see docs/notes/<x>.md § <heading>") | `project_geocode_oom_kills_nas`, `project_face_rival_test`, `project_face_overmatch_cleanup`, `project_worker_fleet_lmstudio`, `project_deploy_panel_helper_container`, `project_phone_ingest_pipeline`, `project_describe_model_strategy`, `project_maintenance_sweep_colors_wedge`, `project_stacking_wipe_cron`, `project_rerun_passes_feature`, `project_worker_panel_on_maintenance`, `project_replica_face_stages_blocked`, `project_video_support_plan`, `project_mcp_server_connection`, `project_calvin_temporal_matches_unreliable`, `project_cluster_label_diagnostic_tool`, `project_worker_fleet_drain_and_exit`, `project_text_pass_safeguards` |
| **Archive** (move topic file to `memory/archive/`, drop index line; outcome is in git history/docs) | `project_rank_shoot_rerun_2026_09_12`, `project_visual_retag_collapsed_folders`, `project_model_eval_plan` (keep ONE line "final fleet models: rerun.FLEET_ROLE_MODELS"), `project_visual_tag_eval`, `project_vlm_aesthetics_scoring`, `project_face_crop_warming`, `project_recluster_hnsw_backend`, `project_category_content_drainer` (9 KB; drainer is a one-off), `project_shutterfly_proof_capture` (books done 07-07), older handoff `HANDOFF-2026-09-13` line (keep only the latest handoff line) |
| **Keep as-is, shorten line** (behavioural preferences and open items) | all `feedback_*`, `user_*`, `project_todo_franklin_vs_koa`, `project_intra_cluster_congruence_review`, `project_arw_unknown_camera_folder`, `project_stale_text_passes_collection`, `project_lce_7rm6_truncated_source`, `project_native_ollama_wsl_rocm` (23 KB topic file — split into ≤ 3 KB summary + archive), `project_2026_europe_trip_books`, `project_photobook_*` (merge the two into one line), `project_mcp_rerank_no_vision_model`, `project_manual_face_labels_also_wrong`, `project_chrome_smb_hammer_starves_ingest`, `project_wsl_*`, `project_gphotos_upload_500storm_recovery`, `project_duplicate_photos_bad_merges`, `project_replica_null_photo_root_path_matching`, `project_trip_clock_fix_a7rvi`, `project_subject_aware_quality`, `project_model_eval_one_model_at_a_time`, `project_mllama_means_ollama_fallback` |

Expected: ~48 lines, ~6-7 KB. Add as the FIRST index line: `- [index rules](MEMORY.md) — one line ≤140 chars per entry; detail in topic files; repo facts belong in docs/notes, not here`.

## 5. Guardrails

**5a. pytest** `tests/test_context_budget.py` (mirrors `tests/test_frontend_refs.py` style: `ROOT = Path(__file__).resolve().parent.parent`):

- `CLAUDE.md` ≤ 24,576 bytes and ≤ 300 lines.
- `.claude/skills/photo-search/SKILL.md` ≤ 20,480 bytes and ≤ 300 lines.
- every `.claude/rules/*.md` ≤ 8,192 bytes AND its frontmatter contains `paths:` (a rule without `paths:` loads every session; fail with message "rule loads unconditionally — add paths: or move to CLAUDE.md").
- `CLAUDE.md` contains no line matching `^\s*-?\s*\*\*?\d{4}-\d{2}-\d{2}` and no `###` heading (dates and h3s are the signature of narratives).
- Failure message names this plan: "CLAUDE.md is a rules file. Narratives → docs/notes/, runbooks → skill reference/, subsystem traps → .claude/rules/. See CLAUDE.md § Where knowledge goes."

Runs in the existing GitHub Actions `tests` job automatically (no ML deps needed).

**5b. CLAUDE.md rule** (exact wording, place directly under Key Patterns):

```
## Where knowledge goes (tests/test_context_budget.py enforces the caps)
- CLAUDE.md (≤ 24 KB): rules every task needs. One line each. No dates, no narratives, no `###`.
- A trap specific to one subsystem → `.claude/rules/<subsystem>.md` (path-scoped, ≤ 8 KB). One bullet per trap; bold the trap's name.
- How to operate something → `.claude/skills/photo-search/reference/<topic>.md`.
- What happened, measurements, why a decision was made → `docs/notes/<subsystem>.md` under its own dated `###` heading, then ONE line in the rule or reference file that names it.
- Plan or milestone status → `docs/plans/README.md` only.
Test before appending here: would a session editing an unrelated file (say `cli.py`) need this line? If not, it goes elsewhere.
```

**5c. Convention line for commits** (in `docs/notes/README.md`): "Each file here is append-only history for one subsystem. Headings are stable identifiers: never rename a `###`; rules and memory link to them by name."

**5d. Optional hook** (`.claude/settings.json`, `PostToolUse` on `Edit|Write` matcher, command `python3 -m pytest tests/test_context_budget.py -q`): instant feedback the moment a session appends to CLAUDE.md, instead of at CI. Cost: ~10 lines of JSON, ~1 s per edit.

## 6. Measurement

| Step | Method |
|---|---|
| Before | Fresh `claude` session in the repo, run `/context`; record the Memory files block (CLAUDE.md tokens), auto-memory line, and skills line. Then trigger the skill with a trivial repo question and run `/context` again. Also `wc -c CLAUDE.md .claude/skills/photo-search/SKILL.md ~/.claude/projects/-home-mattn-local-photo-search/memory/MEMORY.md` (today: 246,797 / 146,890 / 17,510). |
| After | Same three `/context` readings, plus one after `Read photosearch/faces.py` to see a rule load. `wc -c` on the same files plus `.claude/rules/*.md` and `reference/*.md`. |
| Expected per session start | ~66k tok → ~8k (CLAUDE.md ~6k, MEMORY.md ~2k). Skill trigger adds ~5k instead of ~37k. Each rule touched adds ~1.5-2k. Typical dev session: ~103k → ~15k fixed. |
| Per turn | System prompt + CLAUDE.md are re-sent (cached) on every API call; ~90k fewer cached-read tokens per turn, and compaction thresholds are reached ~90k tokens later, so long sessions compact less often. |
| Per compaction | The fixed block survives compaction untouched, so after each compaction the working budget regains ~90k tokens that were previously fixed. |

## 7. Execution order, verification, rollback

| # | Step | Verify | Rollback |
|---|---|---|---|
| 0 | `git tag pre-context-diet`; record `/context` and `wc -c` baselines; write `docs/notes/_headings-2026-10-18.txt` = every `##`/`###` heading of today's CLAUDE.md and SKILL.md (code fences excluded; the awk/python one-liner used in this analysis). | file has 80 + 95 headings | — |
| 1 | Create `docs/notes/{faces,llm-passes,ingest,maintenance,workers,video,print,frontend,search,geotagging,sharpness}.md` by MOVING sections verbatim (cut by line ranges from the table above). Commit. | `scripts/check_headings_preserved.py`: every heading in `_headings-2026-10-18.txt` appears verbatim as a heading in some file under `docs/`, `.claude/rules/`, `.claude/skills/`, or CLAUDE.md. Must report 0 missing. | `git checkout pre-context-diet -- CLAUDE.md` |
| 2 | Fold CLAUDE.md §79 and SKILL.md "Planned milestones" status into `docs/plans/README.md`; delete both. | README "Last reconciled" bumped to 2026-10-18 | tag |
| 3 | Write the 9 `.claude/rules/*.md` (condensed; each bullet's bold lead = the `###` heading name). Suggested `paths:` — faces: `photosearch/face*.py, photosearch/faces.py, photosearch/bulk_unmatch.py, frontend/dist/faces.html, frontend/dist/merges.html, scripts/rank_shoot.py`; llm-passes: `photosearch/describe.py, verify.py, rerun.py, visual_tags_derive.py, stale_descriptions.py, visual_collapse.py, model_eval*.py, vocab_*.py, subjects.py`; ingest: `photosearch/ingest*.py, refile.py, batch_*.py, frontend/dist/batches.html, frontend/dist/batch-flow.js`; maintenance: `photosearch/maintenance*.py, geocode.py, geonames_rich.py, sync-replica.sh`; workers: `photosearch/worker*.py, aesthetics.py, sharpness*.py, run-workers.sh, setup-wsl-gpu.sh`; video: `photosearch/video_*.py`; print: `photosearch/upscale.py, split_export.py, book*.py, shutterfly_export.py, frontend/dist/split*.{html,js}, frontend/dist/book.html`; frontend: `frontend/dist/*.{html,js}`; search: `photosearch/search.py, db.py, tools.py, agent.py, request_*.py, infer_location.py, web.py`. | `/context` after `Read photosearch/faces.py` shows `rules/faces.md` loaded; each file < 8 KB | delete `.claude/rules/` |
| 4 | Rewrite CLAUDE.md to the KEEP set + subsystem index + "Where knowledge goes". | `wc -c` ≤ 24,576; `tests/test_context_budget.py` passes | tag |
| 5 | Split SKILL.md into navigator + `reference/*.md`. | skill still triggers on "how do I index 2024?"; `/context` shows ~5k on trigger | tag |
| 6 | Memory hygiene (section 4). | `wc -c MEMORY.md` ≤ 8,192; every link resolves | memory dir is outside git: copy it to `memory/archive/MEMORY.md.2026-10-18` first |
| 7 | Add `tests/test_context_budget.py`, `scripts/check_headings_preserved.py`, optional hook. Run `./scripts/test-like-ci.sh`. | CI green | — |
| 8 | Measure (section 6); write `docs/HANDOFF-2026-10-18-context-diet.md` with before/after numbers. | — | — |

Risks: (1) a path-scoped rule does not fire for Bash-only ops sessions — mitigated by the CLAUDE.md index naming every rule and the skill's "Before you operate" line; (2) condensing loses nuance — mitigated because `docs/notes/` holds the verbatim text and rules cite headings; (3) the plan-debate and other sessions in `.claude/worktrees/` share CLAUDE.md, so do the rewrite on main and rebase worktrees; (4) `paths:` brace groups count against a 1,000-pattern budget — the globs above use at most two braces.

## 8. Harness-level levers beyond file size

| Lever | Saving | Cost to try |
|---|---|---|
| Duplicate skill listing: `photo-search` appears twice in the skills list (local `.claude/skills/photo-search` and `anthropic-skills:photo-search`). Remove the plugin copy or the local one. | ~400 tok/session, and removes a trigger ambiguity | 1 min |
| `photosearch` MCP server is user-scope, so its 18 tool schemas + instructions block (~2-3k tok) load in every session including pure dev work. Move it to project `.mcp.json` and toggle with `/mcp`, or `claude --strict-mcp-config` for dev sessions. | 2-3k tok/session | 5 min |
| Subagents (`Explore`/general-purpose Task) for `docker logs`, `request_log.jsonl` digs, test-suite output, and `cat` of large plan docs (video-support.md is 80 KB, categories-keywords-implementation.md 128 KB). Only the summary returns to the main context. | 10-100k tok per heavy read | 0; behavioural |
| Start a fresh session per task from the latest `docs/HANDOFF-*.md` (already the owner's habit) rather than continuing through compactions. | avoids compaction summaries carrying stale narrative | 0 |
| `settings.local.json` is 42 KB of permission allowlist — not loaded into context, no saving; leave it. | 0 | — |
| rtk (output filtering) — evaluated at ~3%; after this plan the fixed block is ~15k so rtk's share would rise slightly, still minor. | small | already evaluated |
| Hook-based enforcement instead of CLAUDE.md prose for "always send X-Photosearch-Intent" (PreToolUse on `Bash(curl *)`) lets those lines leave CLAUDE.md entirely. | ~200 tok + better compliance | 15 min |

### Critical files for implementation
- /home/mattn/local-photo-search/CLAUDE.md
- /home/mattn/local-photo-search/.claude/skills/photo-search/SKILL.md
- /home/mattn/local-photo-search/docs/plans/README.md
- /home/mattn/.claude/projects/-home-mattn-local-photo-search/memory/MEMORY.md
- /home/mattn/local-photo-search/tests/test_frontend_refs.py (pattern for the new tests/test_context_budget.py)
