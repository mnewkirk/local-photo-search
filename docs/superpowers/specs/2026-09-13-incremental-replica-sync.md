# The photo database, evaluated — and incremental replica sync

**Status:** Part II (incremental sync): designed, not implemented, 2026-09-13.
Part I (holistic schema evaluation + forward recommendations): added in the
2026-09-13 revision, after a read-only review of the v29 schema against
everything the system has become.
**Depends on:** M26a (`sync-replica.sh`, local read-replica), the maintenance-sync
push model (schema v29, `photosearch/maintenance_sync.py`).
**Schema:** one shared v29 → 30 bump carries everything in this document that
needs a migration — Part I's S2/S4/R2 and, if the sync is built, Part II
Phase 2's tracking columns. Sync Phase 1 needs no schema change.

This lives in `docs/superpowers/specs/` (dated design specs — same shape as
`2026-07-17-maintenance-push-up-design.md`), not `docs/plans/` (milestone
plans/backlogs). It is a design for a specific mechanism, so the specs
convention applies. The 2026-09-13 revision widened the scope: the sync
mechanism is now Part II of a document that also carries the forward design
of the database itself, because the review kept finding that the sync
design, the write-lock failures, and the schema's growth pains are one
subject.

# Part I — the v29 database, evaluated

Reviewed 2026-09-13, read-only, against the Sep 13 replica
(`photo_index.db.local`, 2,063,765,504 bytes = 503,849 × 4 KB pages, schema
v29) plus the write paths in `db.py`, `worker_api.py`, `maintenance.py`,
`faces.py`, `aesthetics.py`, `search.py`, `cli.py`. Every number below was
measured in this pass unless it explicitly cites the briefing.

## What this database is now

One 2 GB file doing at least seven jobs that arrived years apart: (1) photo
identity + EXIF (the original "photo search tool"), (2) two vector-search
indexes (sqlite-vec, ~980 MB — half the file), (3) a face-recognition
workbench (`faces`/`persons`/clusters/references/undo), (4) a VLM aesthetics
store (32 columns bolted onto `photos`), (5) an LLM text store plus an
append-only provenance log (`generations`, 847,062 rows), (6)
distributed-fleet bookkeeping (`worker_processed` / `worker_claims`), and
(7) operational logs (`index_activity`, `index_errors`,
`maintenance_runs`). Its consumers now include the web UI, the MCP server,
the Ask agent, the worker fleet, three cron jobs, and a second machine
holding a wholesale-swapped copy of the whole file.

The one-line summary: **the relational design has held up remarkably well;
the physical layout and the write-concurrency behaviour have not.** Most of
what follows is about the second half of that sentence.

## What the design gets right — leave these alone

- **AUTOINCREMENT everywhere; IDs never reused.** Quietly load-bearing for
  at least four features: vec0 orphan safety, append-only delta extraction
  (Part II), the face-crop caches, and tombstones. Never change this.
- **The pragmas are already right.** `db.py:122–127` sets WAL,
  `busy_timeout=60000`, `synchronous=NORMAL`, a 64 MB page cache, and
  autocheckpoint. The briefing's remedy list opens with "WAL tuning /
  `busy_timeout`" — that lever was pulled long ago. The lock failures happen
  *despite* a 60-second retry, which is diagnostic: they are long-hold and
  hot-acquirer problems, not configuration problems.
- **FK cascades work where declared.** `generations` has **0 orphans**
  against 847,062 rows and a library that has deleted tens of thousands of
  photos. (Contrast `worker_processed`, which has no FK — §3.)
- **Relative filepaths + `photo_root` indirection** — the single decision
  that made the replica architecture possible at all.
- **`generations` as append-only provenance** is the right shape. Its
  problems are weight and indexing (§4), not design.
- **The migration ladder** has survived real mess (the stray-v24 state, the
  v23 tags split) and keeps every deployed DB upgradeable in place. Verbose
  at v29, not broken (verdict in R3).
- **Sidecar discipline** (`photobooks.db`, the split-planner settings DB) is
  the correct response to swap-the-file sync. The pattern should be named
  and kept — not unified away (R5).

## What it is fighting — measured

### 1. The write lock: the dominant failure, and it is three specific behaviours, not a platform limit

SQLite allows one writer per database file. That is not going away, and no
engine swap is warranted (Sequencing, do-nothing list). The failures traced
to the one bad day each have a specific owner:

- **Idle workers are writers.** `claim-batch` takes `BEGIN IMMEDIATE`
  *unconditionally* (`worker_api.py:305`) — before knowing whether anything
  is claimable — and the query it then runs while holding the writer lock
  (`_unprocessed_scoped`, a NOT-EXISTS / NOT-IN scan) is the slow kind. An
  idle fleet polling an empty queue therefore serializes every other writer
  on the box through its polling loop. Two idle pollers were the measured
  root cause of the face-assign 500s, the 30-minute collection insert, and
  (by starving `ingest-incoming`) the 2,040-file zero-row ingest. Fix: S1 —
  ~20 lines, no schema change.
- **Long transactions blow past the 60 s retry.** `recluster-faces`
  renumbers every unknown face in ONE transaction (`faces.py:800–813`:
  `executemany` over ~181k UPDATEs plus the `ignored_clusters` wipe) — a
  ~20-minute hold on the N100. No `busy_timeout` value survives a 20-minute
  hold. Fix: S3.
- **Whole-row rewrites amplify small writes ~100×.** `normalize-aesthetics`
  writes ~20 B of floats per photo and dirties ~503 MB, because SQLite
  rewrites the full 2,258 B row. Fix: S2 — the `photo_scores` split already
  designed in Part II, Phase 1.5.

The general levers, in the order they pay here: **hold the lock briefly**
(S3), **acquire it less often** (S1), **write fewer bytes per acquisition**
(S2). A fourth lever — a single-writer queue daemon in front of the file —
is seductive and rejected: SQLite already *is* the write queue; the problem
was never two well-behaved writers colliding, it was individual writers
behaving badly, and a daemon adds an availability dependency to every CLI,
cron, and ad-hoc container path that today needs only the file.

### 2. `photos` is five tables wearing one rowid

69 columns, 503 MB, 2,258 B/row, and at least five distinct mutability
classes riding a single row: (a) immutable identity/EXIF (~24 columns), (b)
write-once bulky LLM text (`description` 364 B/row, `aes_style` 336,
`aes_subject` 308, `aesthetic_concepts` 293, `keywords` 135), (c) volatile
scalars rewritten library-wide (~20 B/row — the percentiles and recomputed
means), (d) occasionally-corrected location fields, and (e) dead columns
(§6). Every UPDATE to any class rewrites all of them. There are **40
`UPDATE photos SET` call sites across 6 modules** (db 9, maintenance 9,
worker_api 8, cli 8, aesthetics 4, verify 2).

The fix is the mutability split already designed in Part II ("Narrowing
`photos`"): extract the ~6 volatile scalars to `photo_scores`. The *fuller*
three-way split (identity / scores / text blobs) was evaluated and rejected
(S6): the text columns are write-once, so they cause zero rewrite
amplification; moving them buys only sequential-scan speed on a search path
that is CLIP-KNN-first anyway, and costs a join in every result-hydration
site. `description`, `keywords`, `categories`, `subject_boxes` stay where
they are.

### 3. `worker_processed`: bookkeeping that outgrew its design

1,232,215 rows; 52.8 MB plus a **29.3 MB duplicate autoindex** (the table is
keyed `(photo_id, pass_type)` but is a rowid table, so the PK is stored
twice). Measured rot:

- **228,194 rows (18.5%) are orphans** of deleted photos — no FK, so every
  purge and dedup leaves rows forever. The briefing's "133% of photos have a
  `faces` row" verified exactly (210,517 distinct vs 158,419 photos).
- **78,812 rows (6.4%) belong to the removed `tags` pass.** Nothing will
  ever read them.

Is the table the right design at all? **In role, yes** — "which (photo,
pass) pairs are done, with attempts" is exactly what a resumable fleet
needs, and `processed_at` is already the delta key Part II relies on. **In
physical form, no.** Rebuild once (S4): `WITHOUT ROWID` (the PK b-tree
becomes the storage; the 29 MB autoindex disappears) + `REFERENCES
photos(id) ON DELETE CASCADE` (orphans become impossible) + purge the
orphans and the dead pass on the way through. ~82 MB → ~40–45 MB, and two
whole rot classes designed out for the price of one migration step.

### 4. `generations`: right log, wrong weight

847,062 rows; 250 MB table + 30.4 MB of indexes = **~14% of the file** for
data read from exactly two places: the per-photo modal
(`web.py:1433,1494` — indexed by `photo_id`, fine) and rare admin backfills
(`cli.py:4798–4931` — the only consumers of `WHERE text_type = …`). It
grows ~5.3 rows/photo and never stops; every re-run appends more. Three
observations with three different verdicts:

- **`idx_generations_type` (19.8 MB) exists for one-off admin CLIs** that
  can afford a 2–10 s scan of 847k rows. Drop it (S5): one statement,
  −20 MB, and one fewer index-maintenance write per LLM artifact forever.
- **Retention / compression: do nothing.** Average artifact is 216 B
  (183 MB of text total). Deleting history surrenders the one thing the
  table exists for — provenance across model re-runs; compressing it makes
  every consumer decode to save ~100 MB that isn't hurting anything.
- **Moving it to an ATTACHed `provenance.db` (S8)** is the honest
  "eventually": −280 MB from the file every full dump copies, **zero effect
  on the write lock** (inserts already flow through one chokepoint,
  `log_generation`), small permanent tax (consumers must ATTACH; the FK
  cascade is lost, so orphan cleanup moves to `cleanup-orphans`). Bundle it
  with other work or skip it indefinitely; if Part II ships and full dumps
  become weekly, the payoff halves.

### 5. Index economics: one scan hiding in a hot path, several indexes of doubtful worth

`photos` carries 18 secondary indexes (~55 MB). Two findings:

- **`min_quality` is an unindexed full-table scan.** `search.py:1844–46`
  filters *and orders* on `COALESCE(aes_overall, aesthetic_score)` — an
  expression that neither `idx_photos_aes_overall` nor
  `idx_photos_aesthetic` can serve. `EXPLAIN QUERY PLAN` confirms
  `SCAN photos`: 503 MB per query using that filter, on the N100, from the
  search page. The fix rides S2 — materialise the floor as a real column in
  `photo_scores` and index it there. (A zero-migration interim exists — an
  indexed generated column — but if `photo_scores` is happening, don't
  build the floor twice.)
- **Several aes indexes may be dead weight.** On the semantic path, ranking
  and the `min_aesthetic` floor are applied *in Python* on the fetched
  candidate set (`search.py:1182, 1347–1374`), so
  `idx_photos_aes_technical` / `_composition` / `_impact` only pay on the
  pure-structured path. `idx_photos_aesthetic` (legacy `aesthetic_score`,
  3.2 MB) is very likely fully dead — its column now appears only inside
  the COALESCE that cannot use it. Verify-then-drop (R4). The win is small
  (~10–15 MB plus write overhead), which is why it batches with v30 rather
  than shipping alone.

### 6. Vestigial schema, measured

| item | what | size | verdict |
|---|---|---|---|
| `aes_technical_iqa`, `aes_overall_iqa` | reserved for the rejected IQA anchor; **0 non-NULL values in 158,419 rows** | ~0 (NULLs) | drop at the v30 rebuild — they cost nothing but make the schema lie about capabilities |
| `tags_v22_backup` | v23 migration snapshot — a **column**, not a table; 41,887 non-null | 2.8 MB | drop at v30; the re-tag it insured has long since run (192k category-pass rows) |
| `tags` | pre-v23 column; 7,572 stragglers | 0.6 MB | drop at v30 after confirming the stragglers were re-tagged |
| `aesthetic_critique` | **0 non-NULL** — yet still referenced by live code (`index.py:751,772`, `worker_api.py:760`, `rerun.py:278`) | 0 | delete the code paths first, *then* the column — a trap for a bare DROP |
| `worker_processed` rows for pass `tags` | removed pass | 78,812 rows | purge (S4, or the interim cleanup-orphans extension) |
| `photo_index.db.before-org-move-20260427` | stray file in NAS `/data` (briefed; not verifiable from the replica) | ~2 GB disk | delete after confirming its age — it predates four schema versions |
| `photobooks.db.local.bak-*` × 7 | hand-named ad-hoc backups in the repo root | ~100 MB | superseded by D1's rotation; gitignore the pattern |

Checked and **not** vestigial: `hallucination_flags` (84,473 rows, 9.0 MB —
verify UI reads it) and `raw_filepath` (64,079 rows — the RAW badge and
download path).

### 7. Durability is currently an accident of the replica

What the file holds that **no re-run can regenerate** — the human labour:
5,212 manual + 1,341 merge-review face assignments, 54 persons, **60,517
manually geotagged photos** (more than double the 28,597 with EXIF GPS),
9,370 review selections, 56,223 collection memberships across 30
collections, the ignored-clusters curation, and the whole `photobooks.db`
sidecar. Second tier, regenerable only at real cost: 847k LLM artifacts,
158k descriptions, 264k face encodings, and both vector indexes — call it
**weeks of GPU time** to rebuild.

The current safety net for all of that: a nightly replica pull whose target
**overwrites itself** (so an upstream corruption or a bad migration
propagates to the only copy within 24 h), one ad-hoc undo table
(`face_dedupe_undo`), audit CSVs that some destructive CLIs write, and
seven hand-named `.bak` files. There is no point-in-time recovery of any
kind, while the destructive-operation inventory keeps growing:
`dedup_photos` DELETEs, `purge-nonimage-photos` DELETEs, `recluster-faces`
wipes `ignored_clusters` unconditionally, `apply-face-state
--overwrite-persons` exists.

And the Sep-12 ingest proved that writes can fail **silently at full
scale**: 2,040 files moved, zero rows written, discovered a day later by
absence. That is not a lock bug; it is a missing invariant — "files moved
must equal rows written, or scream" (D4). A related operational hole:
`index_activity`'s newest row on this replica is **2026-08-31**, a
two-week silence over a period that included ingests — activity logging is
evidently not reaching the log on some paths, which is exactly the
condition D4 exists to make loud.

## Corrections to the briefing

Stated prominently, as asked:

1. **"WAL tuning / busy_timeout / retry policy" is already done** —
   `db.py:122–127` has had WAL + 60 s busy_timeout + NORMAL sync all along.
   The lock problem is behavioural (S1/S2/S3), not configurational;
   re-tuning pragmas would change nothing measurable.
2. **Splitting the vec0 tables into a separate ATTACHed file would not help
   the write lock at all.** Vectors are write-once at index time and
   contend for nothing afterward. A separate file would only shrink the
   full dump — which Part II achieves without file surgery. (A separate
   *generations* file is a real but modest idea: S8.)
3. `tags_v22_backup` is a **column on `photos`**, not a table.
4. The `UPDATE photos SET` count is **40 sites across 6 modules** today —
   the briefed 32/5 omitted `cli.py`'s 8. The argument it supports (that
   convention-based change tracking is dead on arrival) only strengthens.
5. Everything else verified exactly: the 133% `worker_processed` faces
   ratio, the dbstat top sizes (±2% of this pass's numbers), the 2,258 B/row
   `photos` payload, the aes column-group split, the 20 B of volatile
   scalars.

# Part I recommendations

Organised by the axis they serve; each carries a cost/benefit. Conflicts
between axes are called out afterward; build order — including the explicit
do-nothings — is the Sequencing section.

## Speed — meaning "stop losing to the write lock", then reads

**S1. `claim-batch`: stop taking the write lock to learn there is no
work.** Restructure to read-first: run `_unprocessed_scoped` *outside* any
transaction (WAL readers never block or hold anything); if the result is
empty — the overwhelmingly common case for an idle fleet — return without
ever touching the writer lock. Only when candidates exist, `BEGIN
IMMEDIATE` and **re-run the claim query inside the lock** before inserting
claim rows, preserving the serialization the current comment defends. Add
±25% jitter to the worker poll sleep while in the file. Cost: ~20 lines in
`worker_api.py`, no schema change, one race test (two concurrent claims
must not overlap). Benefit: removes the single highest-frequency
`BEGIN IMMEDIATE` acquirer on the NAS — the measured root cause of the bad
day. **Highest value-per-line change in this document.**

**S2. The `photo_scores` split** — fully designed in Part II ("Narrowing
`photos`", Phase 1.5) and unchanged; it is listed here because it is a
Part-I-class fix that happens to have been designed there first: 503 MB →
~7 MB for every normalize night, and the `min_quality` scan (§5) gets its
index for free by materialising `COALESCE(aes_overall, aesthetic_score)` as
a real column. It is the anchor tenant of the v30 migration.

**S3. Cap every write transaction at roughly a second of hold.** The known
offender: `recluster-faces`' single ~181k-row transaction → chunk to ~10k
rows per commit (~18 commits; cluster IDs are ephemeral and the documented
recovery for a failed recluster is "rerun", so whole-run atomicity buys
nothing a rerun doesn't). Audit `maintenance.py` stage batch sizes while
there — it commits per stage today, and a stage over a 40k backlog is the
same shape of hold. Cost: ~30 lines + audit. Benefit: no writer can be
starved past `busy_timeout` again — the 60 s retry becomes sufficient *by
construction*. Trade-off noted under Conflicts.

**S4. Rebuild `worker_processed` at v30**: `WITHOUT ROWID`, FK `ON DELETE
CASCADE`, purge the 228k orphans and 79k dead-pass rows in the same
migration step. Cost: one ~80 MB table rebuild, ~1 min on the N100, inside
the existing ladder. Benefit: ~82 MB → ~40–45 MB, the orphan class extinct
by construction, the duplicate autoindex gone. `processed_at` unchanged —
zero Part II impact. Interim (ships this week, no migration): extend
`cleanup-orphans` with `DELETE FROM worker_processed WHERE photo_id NOT IN
(SELECT id FROM photos)` and `WHERE pass_type='tags'`.

**S5. `DROP INDEX idx_generations_type`.** One statement, −19.8 MB, one
fewer index write per artifact forever; its only consumers are one-off
backfill CLIs that can afford a table scan. Can ship today.

**S6. Do NOT split the bulky write-once text out of `photos`.** Evaluated
(the maximal reading of the row-width question) and rejected — see §2.
Revisit only if a profile shows `photos` scan time dominating a real query
*after* S2 lands.

**S7. Do NOT move vec0 to a separate file.** See Corrections #2. The
1 GB of vectors is inert weight, not contention; Part II already stops
shipping it nightly.

**S8. `generations` → ATTACHed `provenance.db`** — real but optional;
−280 MB of dump weight, zero lock effect, small permanent ATTACH tax.
Bundle with a future migration or skip indefinitely (§4). If built, Part
II's taxonomy is unaffected in kind — noted inline there.

## Durability

**D1 — SHIPPED 2026-09-13.** Implemented in `sync-replica.sh`: snapshots the
OUTGOING replica before the swap, zstd -3, `KEEP_NIGHTLY=7` / `KEEP_WEEKLY=4`,
`SNAPSHOTS=0` to disable. Measured on the real 2.06 GB file: **984 MB per
snapshot (52%)**, matching the predicted ~53%; a decompressed snapshot passes
`PRAGMA integrity_check` and carries all 60,517 manual geotags and 6,696
manual/merge face assignments. Two corrections to the design below: the cull
sorts by the **datestamp in the filename**, not `ls -t` — mtime ordering silently
keeps the OLDEST N once a file is copied or restored (caught in test) — and the
Sunday weekly is a **hardlink**, not a copy, so it costs nothing until the
nightly is pruned out from under it.

**D1. Rotate replica snapshots instead of overwriting the only copy.**
`sync-replica.sh` currently `mv`s over the previous replica. Change the
swap to rotate: keep ~7 nightly + 4 weekly, compressed (`zstd -3` at the
measured ~53% ratio ≈ ~1 GB each, ~11 GB total on the desktop). Cost: ~15
script lines + disk. Benefit: **point-in-time recovery exists for the first
time**, covering both machines — including upstream corruption and
bad-migration scenarios that today propagate into the only copy within
24 h. Cheapest durability win available; needs no NAS changes at all.

**D2. `export-curation` — a daily dump of the rows no re-run can
regenerate.** The §7 human-labour set (persons, manual/merge face
assignments, manual locations, collections, review selections, ignored
clusters, settings) fits a ~10–30 MB sidecar SQLite file.
`export-face-state` already proves the pattern; widen it, cron it on the
NAS, rotate 30 deep. Cost: ~1 day. Benefit: months of eyeball work survives
even a scenario that loses both full DBs; restore is ATTACH + set-based
UPDATE, exactly like `apply-face-state`.

**D3. One undo convention instead of N ad-hoc ones.** Generalise
`face_dedupe_undo` into a single `op_undo` table — `(op_id, created_at,
table_name, row_pk, column, old_value)` — plus `photosearch undo <op_id>`,
adopted by each destructive CLI as it is next touched: dedup snapshots
deleted rows, recluster snapshots `ignored_clusters` before wiping,
`apply-face-state --overwrite-persons` snapshots what it overwrites. Cost:
~150-line helper + incremental adoption. Benefit: "recluster ate my ignored
clusters" becomes recoverable, and destructive ops stop needing bespoke
undo design each time. Explicitly *not* an event-sourcing rewrite — just
the existing in-tree pattern, named and reused.

**D4. Make silent write failure structurally loud.** `ingest-incoming`
must compare files-moved vs rows-written per run, and on divergence exit
non-zero, log at ERROR, and surface on the `/status` ingest card. ~20
lines. Same treatment for the `index_activity` silence (§7): if an ingest
run logs no activity row, that is a failure, not a quiet night.

**D5. Long jobs must not die with the browser.** The sweep killed by a
phone lock-screen is an SSE-lifecycle bug, not a DB bug, but it destroys
durability of *operations*: run sweeps detached with a reconnectable
progress channel — the `on_progress` / `should_abort` machinery already
exists; the abort must come from an explicit cancel, never from transport
disconnect. Moderate cost; schedule with normal feature work.

## Readability

**R1. Generate a data dictionary; stop making the schema archaeology.**
One page (`docs/db-schema.md`) generated from `PRAGMA table_info` plus a
maintained one-liner per column (which pass writes it; when it went dead),
with a CI test asserting the doc and a fresh `_init_schema()` agree on
columns. Cost: half a day. Benefit: the next "is `hallucination_flags`
dead?" takes ten seconds instead of a grep session — this review needed
exactly that session.

**R2. Retire the dead schema at v30** (the §6 table): drop `aes_*_iqa`,
`tags_v22_backup`, `tags`; delete the `aesthetic_critique` *code paths*
first, then the column. `photos` goes 69 → ~59 columns (→ ~53 after S2
moves the volatile scalars out). Rides the v30 rebuild — do not run a
standalone 503 MB table rewrite to reclaim 3 MB.

**R3. Keep the migration ladder; add a canonical snapshot.** Verdict on
"is `_init_schema()` still maintainable at v29": **yes** — linear,
battle-tested against real deployed states, and every framework alternative
adds a dependency for zero user-visible gain. The real gap is that nothing
states the *current* schema in one place; that is R1, not Alembic.

**R4. Verify-then-drop the doubtful indexes** (`idx_photos_aesthetic`; the
per-dimension aes indexes if EXPLAIN shows the structured path never
reaches them). Small (~10–15 MB); batch with v30.

**R5. Do-nothing verdicts, for the record:** no column renames
(`date_taken` / `date_created` / `indexed_at` confuse, but they are touched
from 6 modules plus the frontend plus the MCP tool schemas — a rename buys
aesthetics at real breakage risk); no table renames; no ORM; keep
`photobooks.db` separate (its isolation from the sync swap is a feature,
not a wart); keep the two-file NAS/replica model.

## Where the axes conflict

- **S2 vs readability:** every hot filter path gains a JOIN, and the shared
  filter vocabulary (`search.py` + `tools._build_filter_sql`) must move in
  lockstep — the exact risk Part II documents. Paid once, at migration
  time, with tests; the alternative is paying 503 MB per normalize forever.
- **S3 vs durability:** a crashed recluster now leaves a half-renumbered
  state instead of rolling back. Acceptable *because* cluster IDs are
  ephemeral and the recovery is "rerun"; D3's `ignored_clusters` snapshot
  covers the one non-ephemeral casualty. This trade would be wrong for
  `photos` content writes — do not generalise it blindly.
- **S8 vs readability:** every `generations` consumer must ATTACH, and the
  FK cascade is lost. That cost is why S8 is optional-bundle, not
  recommended-now.
- **S4 (`WITHOUT ROWID`) vs familiarity:** subtly different storage rules;
  one comment in the DDL suffices.
- **D1 vs disk:** ~11 GB on the desktop. Accepted without ceremony.

## Sequencing

| when | items | cost | benefit |
|---|---|---|---|
| **Now — no schema change, ship this week** | ✅ S1 claim-batch read-first (a935e5b, deployed) · ✅ S5 drop `idx_generations_type` · ✅ D1 snapshot rotation · D4 loud ingest · interim `cleanup-orphans` extension for `worker_processed` | ~2 days total | kills the measured #1 lock source; first-ever point-in-time recovery; the silent-failure class closed; −20 MB |
| **v30 — ONE migration, one deploy** | S2 `photo_scores` (anchor) · S4 `worker_processed` rebuild · R2 column retirement · R4 index drops · **plus Part II Phase 2's tracking columns if the sync is being built** | ~1–2 weeks incl. search-path tests | normalize writes 503 MB → ~7 MB; `min_quality` indexed; file −~120 MB; two orphan classes extinct; schema stops lying |
| **With normal feature work** | S3 transaction caps · D3 undo convention · D5 detached jobs · R1 data dictionary · D2 export-curation | ~1 week, spread out | lock holds bounded by construction; destructive ops reversible; curation backed up independently of both DBs |
| **Only if bundled — maybe never** | S8 `provenance.db` split | ~2–3 days | −280 MB of dump weight; halves in value if Part II ships |
| **Do nothing — deliberate verdicts** | vec0 file split (S7) · single-writer daemon · engine swap (nothing here exceeds SQLite used well) · three-way `photos` split (S6) · `generations` retention/compression · column renames · migration framework | 0 | churn avoided on a system in daily, successful use |

The governing rule: **there should be exactly one more disruptive
migration**, carrying everything above that needs one. On a two-machine,
sync-coupled deployment, each schema bump costs a NAS redeploy + a replica
redeploy + a full sync, in the right order, with the documented
stale-image failure modes in between. Batching v30 is not tidiness; it is
the deployment model.

---

# Part II — Incremental replica sync (the original design)

*Kept intact from the original spec. Part I revisions are marked
**[rev 2026-09-13]** where they change something; nothing else was reworded.*

## Problem

A full replica sync moves 2,063,765,504 bytes and takes ~295 s — but the
network is not the problem. Measured raw ssh throughput NAS→desktop is
~95 MB/s (210 MB in 2.2 s), so the 2 GB transfer accounts for only ~22 s. The
other ~250 s is `dump-db` itself: the N100 reading the whole 2 GB DB through
the sqlite backup API and writing all 2 GB back to the same slow disk before a
single byte leaves the box. Compression attacks the wrong leg — gzip -1 gets
~53% on this data, saving ~10 s of a 22 s transfer while the 250 s dump stands.
The only fix that matters is to stop dumping bytes that didn't change — and
~65% of the file (vector chunks + `generations`) is immutable or append-only,
while the week's real change rate is under 1.5% of photos.

## Measurements (re-verified 2026-09-13 against the Sep 13 replica)

All of the briefing's numbers checked out; verification was done read-only
against `photo_index.db.local` (2,063,765,504 bytes, schema v29, exactly the
file the brief measured). Corrections and *useful additions* the brief missed:

- **`worker_processed` already has a delta key.** `processed_at` has
  `DEFAULT (datetime('now'))` *and* is explicitly re-stamped by the
  `mark_processed` UPSERT (`db.py:1913`). No schema change needed to delta
  this table — the brief listed it as an untracked mutable.
- **`faces` is not a delta-killer.** The table is only ~9.1 MB (264,874 rows ≈
  36 B/row). Even the recluster that rewrote `cluster_id` on 181,485 faces is
  a ~7 MB delta. The *only* library-wide rewrites that threaten the delta are
  the ones touching `photos` (~3.3 KB/row × 158,419 = ~503 MB):
  `normalize-aesthetics`, `recompute-aesthetic-overall`, `backfill-folders`,
  `normalize-places --force`.
- **vec0 delta extraction is empirically viable** (the brief's biggest open
  question — details in the vec0 section): a rowid-range query on
  `clip_embeddings` returned 8,264 embeddings in 0.64 s; `IN (1000 ids)`
  point-lookups took 0.04 s.
- dbstat sizes match the brief (face_encodings chunks 583 MiB, photos
  503 MiB, clip chunks 397 MiB, generations 250 MiB, worker_processed
  53 MiB + a 29 MiB autoindex, faces 9.1 MiB). Change rate matches: 1,020
  photos indexed in 7 days, 2,314 in 30, of 158,419.
- `photos` has `indexed_at` and **no** `updated_at` — confirmed
  (`db.py:299-324`; the only `updated_at` in the schema is on `collections`).
- `grep -c 'UPDATE photos SET'` finds **40 call sites across 6 modules**
  (db.py 9, maintenance.py 9, worker_api.py 8, cli.py 8, aesthetics.py 4,
  verify.py 2; **[rev 2026-09-13]** the original count here — 32 across 5 —
  missed cli.py and went stale within the week it was written, which is
  itself the argument) — plus the
  documented ad-hoc `$DCPY` snippet culture. Any design that asks writers to
  remember anything is dead on arrival; this number is why.

## Expected saving — the arithmetic

Today's 295 s ≈ **250 s dump** + **22 s stream** + **~20 s fixed overhead**
(4–5 `docker compose run` container spins + ssh round trips).

A typical night's delta: ~146 new photos (1,020/wk ÷ 7) → 146 × (3.3 KB photos
row + 2 KB CLIP vector) + ~300 new faces × (36 B + 2 KB encoding) + a few
hundred `generations` rows + `worker_processed` upserts ≈ **2–5 MB**; a
fleet-drain night with tens of thousands of worker rows is still tens of MB.
Extraction is indexed lookups + vec0 point reads — seconds on the N100.
Streaming is sub-second. Apply is seconds.

What's left is the floor: fixed per-container/ssh overhead (~15–20 s across
3 remote invocations) plus a local 2 GB copy-before-apply (~3–8 s on the
desktop). So the honest Phase 2 number is **295 s → ~20–35 s (≈10×)**, not
10–20 s. The brief's 10–20 s becomes true only after Phase 3 collapses the
remote legs into one container invocation and drops the local copy — worth
doing only if the interactive button still feels slow at 30 s. Phase 1 alone
(no schema change) gets **295 s → ~120–140 s (≈2.3×)**: the dump shrinks to
the ~774 MB of genuinely mutable bytes.

## Goals

- Typical sync in tens of seconds, not minutes — the interactive paths
  (`/status` **Sync from NAS**, the maintenance pre-flight auto-sync in
  replica mode) block a human for the full 295 s today.
- The full dump remains the fallback for *every* failure, and keeps getting
  exercised on a schedule so it can't rot.
- A partially-applied delta can never be visible to the running replica —
  same safety property as today's `.tmp` + size check + atomic `mv`.
- Bounded divergence: any silent drift is corrected within N days by
  construction, not by luck.

## Non-goals

- Replacing the full sync. It stays in the script, unchanged in behaviour.
- Touching the face-crop mirror leg (already incremental via its own
  mtime watermark; see "Interactions").
- Reverse sync — `maintenance_sync.py` owns replica→NAS reconciliation.
- Real replication (litestream/LiteFS-class). One writer (NAS), one
  pull-consumer (desktop), nightly-to-hourly cadence: a bespoke delta file
  over the mandatory cat-stream transport is less machinery than adapting a
  streaming replicator to UGREEN's blocked inbound rsync/scp/SFTP and the
  swap-the-file replica model.

## Design

### Shape: a delta *file* over the same cat-stream transport

`photosearch dump-delta` (new CLI, runs on the NAS) writes a small standalone
SQLite file containing changed rows + tombstones + a manifest, to
`/data/replica-delta.db`. The script streams it out exactly like today
(`docker compose run --entrypoint cat` — UGREEN blocks scp/SFTP/rsync inbound,
so cat-streaming is mandatory, not stylistic). `photosearch apply-delta` (runs
on the desktop) ATTACHes it and applies set-based — deliberately the same
shape as `export-face-state` / `apply-face-state`, the in-tree precedent for
"ship a small SQLite file of rows, apply with ATTACH".

### Table taxonomy — the core decision

Every table falls into one of four classes. Nothing is left implicit; an
unknown table in a future schema version fails the manifest check (below)
rather than silently not syncing.

| class | tables | delta key | share of DB |
|---|---|---|---|
| **delta-by-timestamp** | `photos`, `faces` (Phase 2, new `updated_at`); `worker_processed` (existing `processed_at`) | `updated_at > W` / `processed_at > W` | ~29% |
| **delta-by-id** (append-only) | `generations`, `index_activity`, `index_errors` | `id > last max id` | ~14% |
| **keyed-off-parent** (vec0) | `clip_embeddings`, `face_encodings` | parent-row delta + rowid range | ~50% |
| **full-copy every sync** | everything else: `persons`, `collections`, `collection_photos`, `photo_stacks`, `stack_members`, `review_selections`, `google_photos_uploads`, `ignored_clusters`, `face_references`, `face_ref_encodings`, `worker_claims`, `maintenance_runs`, `geocode_cache`, `face_dedupe_undo`, `schema_info` | none — ship whole | ~1–2% (a few MB total) |

Full-copying the long tail is a feature, not laziness: it costs less than
tracking those tables (largest is `stack_members` at 48k tiny rows) and it
reconciles their deletes and rewrites (stacking `clear`, collection edits,
recluster's `ignored_clusters` wipe) for free, with zero new invariants.

**[rev 2026-09-13]** Part I items adjust this table if built first, all
benignly: `photo_scores` (S2 / Phase 1.5) joins the **full-copy** class — at
~3–10 MB it is cheaper to ship whole than to track, so the split needs **no
`updated_at` column and no trigger of its own**. A `provenance.db` split (S8)
leaves `generations` in the delta-by-id class unchanged; `dump-delta` just
ATTACHes one more file. The `worker_processed` v30 rebuild (S4) keeps
`processed_at`, so its delta key is untouched. Nothing in Part I moves a
table between classes.

### Narrowing `photos`: split the volatile scores into their own table

**Proposed by Matt 2026-09-13, after the taxonomy above was drafted. It removes
the single worst case this design has to absorb, and it is worth doing whether
or not incremental sync ever ships.**

SQLite is row-oriented: changing one column rewrites the entire row. `photos`
averages **2,258 B of payload per row**, so `normalize-aesthetics` updating a
single 8-byte float per photo dirties ~503 MB. That one fact is why escape
hatch #1 exists.

Measured column payloads (158,419 rows):

| group | payload | B/row | mutability |
|---|---|---|---|
| `aes_style` | 53.3 MB | 336 | written once by the VLM pass |
| `aes_subject` | 48.8 MB | 308 | written once |
| `aesthetic_concepts` | 46.4 MB | 293 | written once |
| `aes_style_tags` | 8.2 MB | 52 | written once |
| **volatile scalars** | **3.2 MB** | **20** | **rewritten by every normalize / recompute** |
| all 32 `aes_*` together | 174.3 MB | 1,100 | (49% of the table) |
| everything else (37 cols) | 183.5 MB | 1,158 | |

The columns actually rewritten library-wide are all floats:
`aes_overall_pct` and `aes_subject_overall_pct` (`normalize-aesthetics`,
aesthetics.py:432/453), plus `aes_technical`/`aes_composition`/`aes_impact`/
`aes_overall` (`recompute_overall_scores`, aesthetics.py:388).

**Split on mutability, not on topic.** Moving all 32 `aes_*` columns into one
table is the obvious reading and the wrong one — a normalize night would then
rewrite 174 MB, because the VLM's text critiques ride along for no reason. A
narrow `photo_scores` table holding *only* the recomputed scalars:

| approach | normalize-night delta |
|---|---|
| today | ~503 MB |
| all aesthetics in one table | ~174 MB |
| **volatile scalars only** | **~7–10 MB** |

**Consequences for this design.** Escape hatch #1 stops being load-bearing:
with the volatile scalars extracted, the library-wide rewrites that could push
`photos` over the 10% threshold are only `backfill-folders` and
`normalize-places --force`, both one-off backfills rather than routine
maintenance. The hatch stays as a safety net, but the nightly stops needing it,
and the "~½ a full sync" worst case largely disappears.

**The bigger win is NAS-side and independent of sync.** `normalize-aesthetics`
today rewrites 503 MB on the N100 while holding SQLite's single write lock.
That is the same contention class that produced the `500 database is locked`
face assignments, the 30-minute collection writes and the silently-failed
2026-09-12 ingest. At ~7 MB it becomes sub-second. Moving the ~157 MB of aes
text out also shrinks `photos` to ~330 MB, making every full scan of it faster
— and search does many.

**Costs.** A join on the hot filter paths: `sort=aesthetic_desc`,
`min_aesthetic`, `min_day_aesthetic`, `min_quality`, `min_technical` /
`min_composition` / `min_impact`, `style_tag` currently read straight from
`photos` and would need `photo_scores` joined in `search.py` and
`tools._build_filter_sql`. On an INTEGER PRIMARY KEY that is a rowid lookup and
may be net-faster given narrower `photos` rows — but it touches the shared
filter vocabulary that structured Search, the MCP tools and the Ask agent all
rely on staying in lockstep, so it wants its own tests. `rerun.mirror_photos`
(M28) must also learn the new table, or a replica mirror silently stops
carrying scores.

**Recommendation.** Do the narrow version (volatile scalars only, ~6 columns),
not the wholesale aesthetics extraction: ~98% of the benefit for a fraction of
the migration and search-path risk. Sequence it **before** Phase 2 if both are
being built — it shrinks what Phase 2's triggers have to cover — but it is not
a prerequisite, and it earns its keep on lock-contention grounds alone.
**[rev 2026-09-13]** Sync bonus found in the Part I review: at ~3–10 MB the
new table simply joins the full-copy class, so the split adds **zero**
change-tracking machinery to this design — no `updated_at`, no trigger.

### Change tracking for `photos`/`faces`: trigger-maintained `updated_at`

**Recommendation: SQL triggers.** The alternatives, and why they lose:

- **"Every writer touches `updated_at`" by convention** — rejected outright.
  32 `UPDATE photos SET` call sites across 5 modules, plus the documented
  ad-hoc `$DCPY` snippet workflow on the NAS. One forgotten writer = silent,
  unbounded replica staleness with no error anywhere. This is the fragile
  option the brief warned about, and it is exactly as fragile as feared.
- **SQLite session extension / changesets** — rejected. Verified: CPython's
  `sqlite3` exposes no session API (`dir(sqlite3)` has nothing), so this
  means apsw in the NAS image *and* a session object attached to **every
  writing connection** — web workers, every one-shot CLI container, worker
  submits. A writer that opens a plain connection bypasses it silently. Same
  failure mode as the convention approach, with a new dependency on both
  machines. (It also produces changesets per-connection that something must
  collect and order — more machinery than the whole rest of this design.)
- **Per-row hashing** — rejected as the *tracking* mechanism: computing row
  hashes means reading the full ~503 MB photos table on the N100 every sync,
  which re-creates a third of the problem being solved. It survives as the
  *verification* tool (see Testing) where it runs on the desktop, rarely.

Triggers fire at the SQL layer, so they cover every code path — including
ad-hoc container snippets and one-off surgery like the a7R VI clock fix. Cost
on the N100 is one extra in-page write + one index update per updated row;
even the 158k-row normalize adds only seconds (and that night falls back to
table-full-copy anyway, below).

Schema v30, on the NAS (and inherited by every future full dump):

```sql
ALTER TABLE photos ADD COLUMN updated_at TEXT;   -- backfill: COALESCE(indexed_at, datetime('now'))
ALTER TABLE faces  ADD COLUMN updated_at TEXT;   -- backfill: datetime('now')
CREATE INDEX idx_photos_updated_at ON photos(updated_at);
CREATE INDEX idx_faces_updated_at  ON faces(updated_at);

CREATE TRIGGER photos_touch_updated AFTER UPDATE ON photos
FOR EACH ROW WHEN NEW.updated_at IS OLD.updated_at
BEGIN UPDATE photos SET updated_at = datetime('now') WHERE id = NEW.id; END;
-- same pair (AFTER UPDATE + AFTER INSERT default) on faces
```

Two deliberate details:

- `recursive_triggers` is OFF by default in SQLite, so the trigger's own
  UPDATE cannot re-fire it. Do not turn that pragma on globally.
- The `WHEN NEW.updated_at IS OLD.updated_at` guard means a writer that
  *explicitly sets* `updated_at` wins. This is load-bearing: `apply-delta`
  and `rerun._apply_mirror` must write the **NAS's** `updated_at` verbatim on
  the replica (see Interactions) — without the guard, the replica-side
  trigger would re-stamp local time and break the freshness comparison.

**The one blind spot triggers cannot cover: vec0.** SQLite refuses triggers
on virtual tables, so a re-embed of an *existing* photo (M28 re-run clip:
DELETE+INSERT into `clip_embeddings`, no `photos` row touched) is invisible.
Mitigation: `add_clip_embedding` and `add_face_encoding` are the **single
chokepoints** in `db.py` (verified — nothing else INSERTs those tables), so
each gains one line touching the parent row's `updated_at`. This is the only
"remember to touch" surface in the design: two functions, both in `db.py`,
both unit-tested. Accepted.

### Deletions: tombstones

```sql
CREATE TABLE sync_tombstones (
    tbl        TEXT NOT NULL,       -- 'photos' | 'faces'
    row_id     INTEGER NOT NULL,
    deleted_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (tbl, row_id)
);
-- AFTER DELETE triggers on photos and faces INSERT OR REPLACE a tombstone.
```

The delta ships tombstones with `deleted_at > W`; apply deletes those ids
plus their vec0 rows explicitly (FKs don't reach vec0 — same reason
`cleanup_orphans` exists) and lets the replica's real FK cascades handle the
rest. This covers `purge-nonimage-photos`, duplicate pruning, prune-missing,
`delete_face`, and face re-detection (which deletes + re-inserts face rows
under new AUTOINCREMENT ids). Full-copy tables need no tombstones. The
table is pruned of rows older than ~90 days during `dump-delta` — it stays
tiny because deletes are rare.

### vec0 delta: keyed off the parent, applied DELETE+INSERT

The brief asked whether vec0 can be delta'd safely at all. **Yes — measured.**
You never diff the `*_vector_chunks00` shadow tables; you query the virtual
table by its INTEGER PRIMARY KEY (= rowid), which sqlite-vec supports:

- `SELECT photo_id, embedding FROM clip_embeddings WHERE photo_id > ?` —
  query plan is a rowid scan (`SCAN ... VIRTUAL TABLE INDEX 0:1`) but only
  matching rows materialize their vector: 8,264 rows came back in 0.64 s
  against the full 397 MiB table. The rowid walk itself touches only the
  small `_rowids` shadow table.
- `WHERE photo_id IN (…)` decomposes to point lookups
  (`INDEX 3:2!___`): 1,000 vectors in 0.04 s.

Extraction rule: new vectors via rowid range (`> replica's max id`, ids are
AUTOINCREMENT and never reused — CLAUDE.md's orphan-cleanup section already
relies on this), *re-embedded* vectors via `IN (changed-photo ids)` from the
`photos` delta (whose `updated_at` was touched by the chokepoint, above).
The delta file stores them as ordinary rows `(photo_id, embedding BLOB)`.

Apply rule: explicit `DELETE` then `INSERT` per row — never `INSERT OR
REPLACE`, which vec0 does not honor (the PK conflict fires first; this is
the documented `add_clip_embedding` pattern and the same bug class that once
broke worker submits). After tombstone deletes, run the `cleanup_orphans`
DELETE against the applied copy so the replica can't accumulate the >100%
embedded drift the NAS is prone to.

`face_ref_encodings` (12 rows) is full-copied via the same DELETE+INSERT.

### The delta file format

```
_sync_manifest   (single row): db_uuid, schema_version, mode ('delta'|'full-table-mix'),
                 since, until, created_at
_sync_tables     (per table): name, mode ('delta'|'full'), rows_shipped,
                 post_count, post_max_pk     -- fingerprints at snapshot time
photos, faces, clip_embeddings, ...          -- plain tables of shipped rows
_sync_tombstones -- (tbl, row_id) to delete
```

- **`db_uuid`** is a new `schema_info` key minted once at the v30 migration
  (random hex). Full dumps carry it automatically (they're byte copies).
  `apply-delta` refuses a delta whose `db_uuid` differs from the replica's —
  this is what makes "NAS restored from backup" or "pointed at the wrong
  host" a loud full-dump fallback instead of silent corruption.
- **`since`/`until`** are NAS-clock (`datetime('now')`, UTC). Extraction runs
  inside one read transaction (WAL — doesn't block writers) so the delta is
  a consistent snapshot, mirroring what the backup API gives the full dump.
  The next sync requests `since = until − 5 minutes`: the overlap costs a
  few re-shipped rows and buys immunity to writes that committed during
  extraction and to modest clock steps, because **apply is idempotent** (all
  upserts / DELETE+INSERT / re-deletes).
- The per-table `post_count` / `post_max_pk` fingerprints are the same shape
  as `maintenance_sync.photo_fingerprint` — the pattern is proven; this
  extends it per-table.

### Apply: copy → apply → verify → swap

```
cp photo_index.db.local photo_index.db.local.tmp        # ~3–8 s local NVMe
apply-delta --db .tmp --from replica-delta.db           # one transaction
verify: every _sync_tables (post_count, post_max_pk) matches the applied copy
mv .tmp → photo_index.db.local                          # atomic, as today
```

The live replica file is **never** written. A crash, a bad delta, or a failed
verification leaves the current replica byte-for-byte intact and deletes the
watermark, so the next sync is a full dump. This is strictly the same safety
property as today's 4 KB-check + `mv` (which both remain, unchanged, on the
full path). The verification is meaningful because extraction snapshotted the
fingerprints *after* the delta's `until` inside the same read transaction: a
correct apply must land exactly on them; any mismatch means a bug or
divergence, and the answer to both is the full dump.

Rejected: applying in a transaction directly against the live file. It's
atomic in SQLite's sense and skips the 2 GB copy, but it abandons the
"live file is never a write target" invariant that has made the current
script incorruptible, and it takes a multi-second write lock against the
replica web server's per-request connections. Revisit in Phase 3 only if the
copy shows up in profiles.

Rows land via `INSERT OR REPLACE` — except `photos`, which uses an UPSERT
guarded by freshness:

```sql
INSERT INTO photos (...) VALUES (...)
ON CONFLICT(id) DO UPDATE SET ...
WHERE excluded.updated_at >= photos.updated_at OR photos.updated_at IS NULL;
```

That guard is the M26b/M28 protection (see Interactions).

### Escape hatches: when the delta stops being worth it

Decision inputs are three indexed COUNTs and two MAX() lookups (<100 ms on
the N100) — never a table scan. Byte estimates use **constants baked from
this spec's dbstat measurements** (photos ≈ 3.3 KB/row, faces ≈ 36 B,
clip ≈ 2.6 KB, face-enc ≈ 2.3 KB, generations ≈ 310 B), refreshed whenever
someone re-measures; precision is unnecessary because the thresholds are
coarse.

1. **Per-table full-copy:** if a table's changed-row count exceeds **10%** of
   its rows, ship that table complete (sequential scan) instead of row-picking
   (random IO through the `updated_at` index). Crossover for random-vs-
   sequential on this disk is somewhere in the 10–20% band; 10% is the
   conservative edge. This is what absorbs `normalize-aesthetics` /
   `backfill-folders` nights: photos ships whole (~503 MB, ~70–90 s extract)
   while the 1 GB of vectors still stays home — that night costs ~½ a full
   sync, not a full one. It also absorbs recluster trivially (faces whole =
   9 MB). **If the `photo_scores` split above is built, this hatch stops being
   load-bearing for the nightly** — `normalize-aesthetics` drops from ~503 MB
   to ~7 MB and no longer trips the 10% threshold at all; only the one-off
   backfills (`backfill-folders`, `normalize-places --force`) still would.
2. **Global full-dump fallback:** if estimated total delta bytes exceed
   **40%** of the DB, do a full dump. Rationale for 40: with per-table
   full-copy already keeping extraction sequential, the incremental path's
   only remaining advantage is skipped bytes; below ~2× projected saving the
   full path's simplicity and decade of battle-testing win. (At 40% of 2 GB
   the delta ships ~800 MB ≈ 110 s vs 295 s — right at that 2× line.)
3. **Scheduled full:** every Nth sync (`SYNC_FULL_EVERY`, default 7) is a
   forced full dump. This bounds any undetected divergence at a week *and*
   keeps the fallback path continuously exercised — a fallback that only
   runs after a failure is a fallback that has rotted.
4. **Anything abnormal → full:** missing/corrupt watermark file, `db_uuid`
   mismatch, `schema_version` mismatch, manifest verification failure,
   truncated stream, *or an old NAS image that doesn't know `dump-delta`*
   (the script treats "no such command" as "fall back", so deploy order
   can't break the nightly — same failure family as the M28
   `/mirror-fields` 404 gotcha, handled rather than documented this time).
5. **Manual:** `SYNC_FULL=1 ./sync-replica.sh`.

### `sync-replica.sh` changes

State lives in `${TARGET}.sync-state` (JSON sidecar next to the replica,
gitignored like the DB): `{db_uuid, watermark, syncs_since_full}`. A sidecar,
not a table in the replica, because the replica file is wholesale-replaced by
full syncs and the state must describe the file from outside it (same
reasoning as the face-crop `.last_sync` marker and the photobooks sidecar).

Flow: read sync-state → decide full-vs-incremental (ladder above) →
incremental: `dump-delta --since W --to /data/replica-delta.db` remotely,
cat-stream, local copy+apply+verify+swap, advance watermark to the manifest's
`until` **only on full success** → full: exactly today's four steps, then
initialize sync-state from the manifest `dump-db` now also prints (`db_uuid`,
snapshot time). The face-crop leg (steps [5/5]) is untouched.

## Interactions with existing machinery

- **M26b/M28 mirror writes** (`rerun.mirror_photos`): the NAS's
  `/api/photos/{id}/mirror-fields` payload and `_MIRROR_COLUMNS` gain
  `updated_at`, and `_apply_mirror` writes it verbatim (the trigger's WHEN
  guard makes that stick). Then the delta's guarded UPSERT cannot regress a
  freshly mirrored row: a delta extracted at T₁ carries `updated_at ≤ T₁`,
  a mirror that ran at T₂ > T₁ carries the NAS's later stamp, and the guard
  keeps the newer one. Against a pre-v30 NAS the mirror payload has no
  `updated_at` and behaves as today (unguarded overwrite) — acceptable,
  since that NAS also can't produce deltas.
- **maintenance_sync:** `maintenance_runs` is in the full-copy class, so the
  drift panel stays truthful after an incremental sync. "Replica ahead —
  unpushed" rows are overwritten by a delta exactly as a full sync
  overwrites them today — the db.py comment above `maintenance_runs` already
  declares that correct; semantics unchanged. The **pre-flight auto-sync**
  in replica-mode maintenance applies (`run_replica_sync_blocking`) is the
  single biggest beneficiary of this whole design — it currently inserts
  295 s into a human's click.
- **Face crops:** unchanged; independently incremental already. The two
  watermarks stay separate files on purpose — crop mtimes are desktop-clock
  relative to a tar, DB watermarks are NAS-clock relative to a snapshot, and
  fusing them once would create a shared failure mode for no saving.
- **`debug-db.sh`** keeps using plain `dump-db`; nothing there changes.

## Concrete interface changes

| where | change |
|---|---|
| `photosearch/db.py` | `SCHEMA_VERSION` 29→30; migration adds `photos.updated_at`, `faces.updated_at` (+ backfill), 2 indexes, 4 triggers, `sync_tombstones`, `db_uuid` in `schema_info`; `add_clip_embedding`/`add_face_encoding` touch parent `updated_at` |
| new `photosearch/replica_delta.py` | `extract_delta(db, since, out_path)` (taxonomy, thresholds, manifest), `apply_delta(db_path, delta_path)` (copy/apply/verify), shared table-class registry |
| `cli.py` | `dump-delta --since TS --to PATH [--full]`, `apply-delta --from PATH [--db …]` — both with `envvar="PHOTOSEARCH_DB"`; `dump-db` additionally prints `db_uuid` + snapshot time as JSON |
| `sync-replica.sh` | incremental default, fallback ladder, `${TARGET}.sync-state`, `SYNC_FULL` / `SYNC_FULL_EVERY` env knobs |
| `photosearch/web.py` | `mirror-fields` payload += `updated_at` |
| `photosearch/rerun.py` | `_MIRROR_COLUMNS` += `updated_at` |
| deploy order | **NAS first**, as with M28/maintenance-sync — but an old NAS degrades to full dumps rather than erroring |

## Testing

`./scripts/test-like-ci.sh` is the pre-push gate; CI installs `sqlite-vec`
(verified in `tests.yml`), so every test below runs against real vec0 tables,
no mocks.

1. **Trigger coverage, parametrized over real writers.** For each in-tree
   writer path (`update_photo`, `aesthetics` normalize, `verify`, worker
   `submit_results`, `infer-locations` apply, `bulk-set-location`, raw
   `conn.execute` as the ad-hoc stand-in): assert `updated_at` advances.
   Plus: WHEN-guard lets explicit stamps through; recursion doesn't fire;
   chokepoint touch on re-embed; tombstone rows on delete.
2. **Round-trip equivalence — the core test.** Build a seeded DB; take full
   dump A (today's path). Mutate the source with a representative script:
   inserts, scattered updates, a bulk aes rewrite, deletes, a re-embed, a
   recluster-style faces rewrite. Then (a) delta-extract + apply onto A, and
   (b) take fresh full dump B. Assert **logical equivalence** of (a) and (b):
   per-table ordered row comparison over every real table, with vec0 compared
   via `SELECT pk, vector ORDER BY pk`. *Byte-identical files are explicitly
   not the bar* — page layout, freelists, and vec0 chunk packing legitimately
   differ between a backup and an applied delta; equivalence of all queryable
   content is the property that matters and the property search runs on.
3. **Idempotency & overlap:** applying the same delta twice is a no-op;
   a row updated inside the 5-minute overlap window appears in two
   consecutive deltas and converges.
4. **Fallback ladder:** each rung (thresholds, uuid mismatch, schema
   mismatch, unknown-command, truncated stream, corrupted manifest) ends in
   a full dump and an intact replica.
5. **Crash safety:** kill apply mid-transaction → live replica untouched,
   watermark cleared.
6. **Mirror interaction:** mirror a row with a newer `updated_at`, apply an
   older delta → mirrored values survive.
7. **Manual acceptance before flipping the cron:** shadow-run on the real
   pair — incremental into a scratch target, full dump into another, compare
   with `sqldiff` / the test-2 comparator; run for a week alongside the
   existing nightly.

## Risks and failure modes

- **Silent divergence is the risk class the full dump structurally cannot
  have.** Everything here — snapshot fingerprints verified at apply,
  `db_uuid`, scheduled fulls — *bounds* that risk (at ≤ `SYNC_FULL_EVERY`
  days) rather than eliminating it. If that bound is unacceptable, don't
  build Phase 2.
- **Trigger blind spots.** Triggers cover all SQL, including ad-hoc
  container snippets. What they don't cover: direct vec0 writes outside the
  two chokepoints (none exist today; a new one is a code-review catch), and
  `writable_schema`-class surgery (if you're doing that, you know to force a
  full sync).
- **Clock steps on the NAS.** A backwards NTP jump larger than the 5-minute
  overlap can hide writes until the next scheduled full. Accepted; widening
  the overlap is a one-constant fix if it ever bites.
- **Schema drift mid-flight.** A v31 NAS producing deltas for a v30 replica
  is refused by the manifest check → full dump → the full dump *carries* the
  new schema, and the replica is upgraded wholesale. Self-healing.
- **The 40%/10% constants are folklore, not physics.** They're derived from
  one disk's rough random-vs-sequential crossover and one 2× "is it worth
  it" judgment. They're env-overridable and the wrong value costs seconds,
  not correctness.
- **Is this worth building at all?** For the nightly cron alone — **no**.
  295 s unattended at 3 AM is fine forever, and that alone would not justify
  ~600 lines plus a schema bump plus a new invariant class. What justifies
  it: (1) the two *interactive* paths — the Sync-from-NAS button and the
  replica-maintenance pre-flight auto-sync — put the full 295 s in front of
  a waiting human today; (2) the dump hammers the N100's disk for 4+ minutes
  against the same spindle serving the library, nightly, adjacent to the
  ingest cron; (3) a ~25 s sync makes *hourly* replica freshness practical,
  which shrinks the window where M26b mirrors and nightly reconciliation
  have to paper over staleness. If none of those three matter, stop after
  Phase 1 — or build nothing and close this spec. **One caveat to that
  verdict:** the `photo_scores` split (Phase 1.5) is justified independently of
  everything else here. Even with this spec closed unbuilt, cutting
  `normalize-aesthetics` from a 503 MB write to ~7 MB is worth doing on its own,
  because that write holds the NAS's single SQLite write lock and lock
  contention has already cost a face-assignment session, a collection build and
  an entire overnight ingest. **[rev 2026-09-13]** And S1 (claim-batch
  read-first, Part I) attacks the same lock with ~20 lines and no schema
  change; it should ship before anything in this Part, whatever happens to
  the sync.

## Phasing

**Phase 1 — no schema change, no triggers (~2.3×: 295 s → ~120–140 s).**
Build the whole skeleton: delta file + manifest, transport, copy/apply/
verify/swap, sync-state, fallback ladder. Delta only the classes that need
no tracking: vec0 by rowid range off the parent tables' max ids,
`generations`/`index_activity`/`index_errors` by max id, `worker_processed`
by its existing `processed_at`; **full-copy `photos` and `faces`** (and the
long tail). Requires a NAS redeploy for the new CLI, but no migration. Known,
accepted staleness: a re-embedded *old* photo's vector lags until the next
scheduled full (M28 re-runs are rare and manual). Worth shipping alone: it
banks half the win, and it battle-tests every risky moving part (transport,
apply, verify, fallback) while the change-tracking risk is still zero.

**Phase 2 — schema v30 (~10×: → ~20–35 s).** Triggers + `updated_at` +
tombstones + `db_uuid`; `photos`/`faces` move to delta-by-timestamp; vec0
gains the re-embed path via the chokepoint touch; mirror guard lands. This is
the phase that carries all the correctness risk, which is why it rides on a
Phase 1 skeleton that's already been syncing for a while.
**[rev 2026-09-13]** If Phase 2 is built, its schema bump must be the *same*
v30 migration that carries Part I's S2/S4/R2 (see Sequencing, Part I) — on a
two-machine sync-coupled deployment every extra disruptive migration costs a
NAS redeploy + a replica redeploy + a full sync; batching is the deployment
model, not tidiness.

**Phase 1.5 — `photo_scores` split (schema bump, independent).** See
"Narrowing `photos`" above. Not a prerequisite for either phase, and it earns
its keep on NAS lock-contention grounds with no sync at all. But if Phase 2 is
going to be built, do this first: it shrinks the table the triggers have to
cover and retires escape hatch #1 as a routine path. If only ONE thing from
this spec ever gets built, this is the strongest candidate — it is the smallest
change here and the only one whose main benefit (a 503 MB → 7 MB write while
holding the global write lock) lands on the problem that has actually bitten
this system repeatedly.

**Phase 3 — optional, chases the ~15 s floor.** Collapse the three remote
invocations into one (`--entrypoint sh -c 'python cli.py dump-delta … >&2 &&
cat …'` — one container spin, delta on stdout), and/or replace the local 2 GB
copy with direct transactional apply. Only if the button still feels slow;
the copy is also the safety story, so this trades an invariant for seconds.
