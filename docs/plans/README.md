# Roadmap & Plan Index

**This file is the single source of truth for plan/milestone status.** Each
detail doc in this folder is a deep-dive; the *status* you should trust lives
here. When a plan's state changes, update the row here (and the detail doc's own
status header if you like, but this index is authoritative). The README only
links here — it no longer carries a milestone table.

**Status legend:** ✅ Shipped · 🟡 Partial / in progress · ⬜ Not started ·
📐 Design only

_Last reconciled: 2026-10-19_ (folded in the "Planned milestones" lists CLAUDE.md and the
photo-search SKILL.md used to carry; their frozen copies are in
`docs/notes/_archive-planned-milestones.md`).

---

## Open / active

These are the things actually left to do, roughly by impact-per-effort.

| Plan | Status | What's left | Detail |
|---|---|---|---|
| **Search accuracy & ranking** | ⬜ | RRF fusion across filters + recency decay + *ranking* use of the structured `country`/`admin1`/`admin2`/`locality` columns (the columns shipped in v19; the ranking didn't). Fixes "Calvin in France ranks ancient photos first" and "Marin County returns nothing." ~200-line bundle. | [search-accuracy-improvements.md](search-accuracy-improvements.md) |
| **Search / LLM-search backlog** | 🟡 | Running, prioritized list. Top items already shipped (summarize, VLM rerank, subject-prominence, prompt tuning, face-framing metadata filters `only_these_people`/`faces_in_frame`, and `representatives` `max_buckets` for "top-N, one per location" — last two from the 2026-06-23 Ask-log review). Remainder: fuzzy name/place matching, LLM query rewriter, self-hosted Nominatim. | [search-improvement-backlog.md](search-improvement-backlog.md) |
| **Categories + keywords** (v22/v23) | 🟡 | Three-field tag model (`categories`/`visual_tags`/`keywords`) and worker passes shipped (schema v22/v23, replacing M9's single `tags`). **Remaining:** vocab curation + backfill via the `mine-vocab` → `group-vocab` → `/admin/vocab` pipeline. | [redesign](categories-keywords-redesign.md) · [implementation](categories-keywords-implementation.md) |
| **Inferred-geotag refinements** | ⬜ | Post-M19 cascade tuning: cap hop depth (776-deep chains observed), downweight inferred anchors on re-scan so decay protects against cross-run compounding, drop the dead `--max-cascade-rounds` flag. Localized to `infer_location.py`. | [infer-location-refinements.md](infer-location-refinements.md) |
| **Timeline view + LLM summaries** (M21) | ⬜ | Chronological UI grouping photos by period, each segment annotated with an LLM summary (where, from GPS/place; what, from descriptions/tags). No detail doc yet. | _(none)_ |
| **Video backfill** | 🟡 | Built 2026-10-17, deployed 2026-10-18 (schema v39, NAS + replica). **Backfill not started** — nothing registered, no fleet run, no sampling until the owner asks. Step 8 of the plan is the runbook (incl. `unwrap-zip-videos` before registration). Main-subject share **parked 2026-10-28** (both eval rounds failed every gate); video-face margin rule + 2 s presence shipped. | [video-support.md](video-support.md) |
| **Batch blocked re-run** | ⬜ | Planned 2026-10-18; nothing built. | [batch-blocked-rerun.md](batch-blocked-rerun.md) |
| **Names in descriptions (face-label reconciliation)** | 📐 | Design approved 2026-10-30; nothing built. Raw `photos.description` untouched; new label-free `people-bind` pass stores spans keyed on face id; names rendered at read time from current trusted labels (schema v40). Gated on a hand-labelled eval (0 wrong names) before any backfill. | [description-names.md](description-names.md) |
| **Geocode R-tree** | ⬜ | Queued. Durable replacement for the in-memory KDTree (the OOM is already mitigated by the population gate + streaming). | [geocode-rtree.md](geocode-rtree.md) |
| **Subject-aware quality** | 📐 | Prototype validated 2026-07-09 (subject-crop aesthetic re-score); design proposed, not built. | [subject-aware-quality.md](subject-aware-quality.md) |
| **Search indexes — remainder** | 🟡 | v34 shipped 2026-10-04 (below). Open: date pushdown into the remaining standalone filters. | [search-indexes.md](search-indexes.md) |
| **Test-isolation fixes** | ⬜ | 6 pre-existing test failures `@pytest.mark.skip`-ped 2026-06-20 so real regressions stay visible. Test-harness debt, not product bugs. | [test-isolation-fixes.md](test-isolation-fixes.md) |

---

## Shipped

Compact history — one line each. Detail docs linked where they exist.

**Foundations (M1–M17)**

- **M1–M7** ✅ — EXIF + CLIP indexing + color extraction; face detect/encode/cluster/temporal-match; LLaVA descriptions + hybrid search scoring; full CLI (semantic/person/place/color/face); first scale test; FastAPI web UI; Docker packaging for the UGREEN NAS.
- **M8** ✅ — Aesthetic quality scoring (1–10), filter/sort by quality.
- **M9** ✅ — Semantic tagging from a fixed ~60-tag vocab (later replaced by the categories/keywords redesign, above).
- **M10** ✅ — Shoot review / culling: adaptive CLIP clustering + quality selection, grid/cluster UI, export.
- **M11** ✅ — Portable photo paths (`photo_root`, relative paths resolved at runtime).
- **M12** ✅ — Hallucination detection: three-pass cross-model verification + auto-regenerate.
- **M13** ✅ — Collections / albums (CRUD API + dedicated page).
- **M14** ✅ — Photo stacking: burst/bracket union-find detection + stack-management UI (schema v11).
- **M15** ✅ — Shared header component (`PS.SharedHeader` in `shared.js`).
- **M16** ✅ — Shared photo detail modal (`PS.PhotoModal`), ~1500 dup lines removed.
- **M17** ✅ — Distributed indexing: worker claim/submit API + HTTP worker loop with TTL crash recovery.

**Recent (M18–M27)**

- **M18** ✅ — Face clustering overhaul: session-stacking recluster pass, `suggest-face-merges` + `/merges` review page, `split-cluster` for attractor clusters. [faces-clustering-and-perf.md](faces-clustering-and-perf.md)
- **M19** ✅ — Inferred geotagging (`infer-locations`, cascade + movement guard, schema v17), the `/geotag` manual UI (M19.1), and the `/map` view (M19.2). [bulk-set-location.md](bulk-set-location.md)
- **M20** ✅ — Google Photos / Takeout import into the dated-folder layout; feeds the phone-ingest pipeline. [google-photos-import.md](google-photos-import.md)
- **M24** ✅ — LLM-driven search: shared tool layer + streamable-HTTP MCP server (M24a) and in-app `/api/ask` agent + "✨ Ask" mode (M24b). [llm-driven-search.md](llm-driven-search.md)
- **VLM re-ranking** ✅ — `rerank_photos` per-image vision scoring tool for "find THE photo" (the doc self-labels M27; it's really part of the M24 search line). [vlm-reranking.md](vlm-reranking.md)
- **M25** ✅ — `maintenance-sweep` / `validate-data` / `repair-data` (CPU backfills, dependency-ordered, dry-run default) + SSE endpoint + `/status` card (shipped 2026-06-21; the detail doc's "DO NOT START" header is stale). [backfill-maintenance-sweep.md](backfill-maintenance-sweep.md)
- **M27** ✅ — `/review` + `/geotag` folder-picker performance: indexed `photos.folder` column (schema v25) + `GROUP BY folder` rework (shipped 2026-06-23). [review-folders-perf.md](review-folders-perf.md)
- **M26** ✅ — Local read-replica + write tools. M26a: replica sync (`sync-replica.sh`, consistent snapshot + atomic swap), lazy thumbnail proxy, sync/status endpoints + `/status` card, nightly Windows Task Scheduler pull, end-to-end serve-off-replica with local LM Studio (all verified 2026-06-23; optional thumbnail bulk pre-warm deferred). M26b: agent-facing `set_photo_location`/`set_photo_tags`/`add_to_collection` write tools with read-local / write-NAS-authoritative / mirror-local dual-write + guardrails (id-set scoping, dry-run→confirm, affected-count cap, reversible+audited). [local-replica-and-writes.md](local-replica-and-writes.md)
**Since 2026-07 (by name; numbers M28–M32 were reused loosely)**

- **Ingest-batch readiness** ✅ — `/batches` per-batch status flow, six step states, one-click `batch-advance` / fleet launch (shipped 2026-09-19). [ingest-batch-readiness.md](ingest-batch-readiness.md)
- **Search indexes (v34)** ✅ — composed id scope in `search_combined`, request log with source + intent, face-encoding cache (shipped 2026-10-04). [search-indexes.md](search-indexes.md)
- **Face-label verification** ✅ — `verify-face-labels`, per-pair calibration, the in-photo rival test + Swap. [face-label-verification.md](face-label-verification.md)
- **Model evals for every LLM pass** ✅ — harnesses + `/eval/models`; final fleet models in `rerun.FLEET_ROLE_MODELS` (complete 2026-10-03). [model-eval-harnesses.md](model-eval-harnesses.md)
- **Category-content grounding** ✅ — description-decided drops + recitation verdict; deployed on the NAS at `79a14f7`, gemma backfill applied, holdout 0.958 (2026-10-18). [category-grounding.md](category-grounding.md)
- **Measured sharpness** ✅/closed — schema v33 column + throttled backfill built; the `blurry` gate failed its eval three times, so `sharp`/`blurry` stay FROZEN (closed 2026-09-26). [sharpness-measurement.md](sharpness-measurement.md)
- **Video support** ✅ built + deployed (backfill open, above). [video-support.md](video-support.md)
- **Context diet** ✅ — CLAUDE.md 258 KB → 8 KB, SKILL.md 148 KB → 6 KB, path-scoped `.claude/rules/`; Sonnet eval 207 → 210/231 at −70% cost (2026-10-19). [context-diet.md](context-diet.md) · [handoff](../HANDOFF-2026-10-19-context-diet.md)
- **M28** ✅ on-demand pass re-runs + fleet launcher · **M29** ✅ `/book` photobook builder · **M30** ✅ local Topaz upscaling (also the photobook authoring wizard) · **M31** ✅ `/split` print planner · **M32** ✅ `/faces` review panels. No detail docs; history is in `docs/notes/`.

---

## Notes on numbering

Milestone numbers have gaps: **M22/M23 were never used** (numbering jumped M21 →
M24), and the categories/keywords redesign that produced schema v22/v23 was never
given a milestone number — it's tracked by name above. "M27" is claimed by both
the review-folders work and the VLM-rerank doc; treat the names, not the numbers,
as canonical.
