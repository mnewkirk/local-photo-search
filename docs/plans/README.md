# Roadmap & Plan Index

**This file is the single source of truth for plan/milestone status.** Each
detail doc in this folder is a deep-dive; the *status* you should trust lives
here. When a plan's state changes, update the row here (and the detail doc's own
status header if you like, but this index is authoritative). The README only
links here — it no longer carries a milestone table.

**Status legend:** ✅ Shipped · 🟡 Partial / in progress · ⬜ Not started ·
📐 Design only

_Last reconciled: 2026-06-23._

---

## Open / active

These are the things actually left to do, roughly by impact-per-effort.

| Plan | Status | What's left | Detail |
|---|---|---|---|
| **Local read-replica + write tools** (M26) | 🟡 | M26a (replica sync, image proxy, `/status` card) partly shipped. M26b foundation landed 2026-06-22 (NAS `bulk-set-tags` + `bulk-set-location` dual-write endpoints, `set_photo_location`/`set_photo_tags` db helpers, tests). **Remaining:** agent-facing write tools, the read-local/write-NAS/mirror-local dual-write loop, and the guardrails (id-set scoping, dry-run→confirm, affected-count cap, audit). | [local-replica-and-writes.md](local-replica-and-writes.md) |
| **Search accuracy & ranking** | ⬜ | RRF fusion across filters + recency decay + *ranking* use of the structured `country`/`admin1`/`admin2`/`locality` columns (the columns shipped in v19; the ranking didn't). Fixes "Calvin in France ranks ancient photos first" and "Marin County returns nothing." ~200-line bundle. | [search-accuracy-improvements.md](search-accuracy-improvements.md) |
| **Search / LLM-search backlog** | 🟡 | Running, prioritized list. Top items already shipped (summarize, VLM rerank, subject-prominence, prompt tuning, face-framing metadata filters `only_these_people`/`faces_in_frame`, and `representatives` `max_buckets` for "top-N, one per location" — last two from the 2026-06-23 Ask-log review). Remainder: fuzzy name/place matching, LLM query rewriter, self-hosted Nominatim. | [search-improvement-backlog.md](search-improvement-backlog.md) |
| **Categories + keywords** (v22/v23) | 🟡 | Three-field tag model (`categories`/`visual_tags`/`keywords`) and worker passes shipped (schema v22/v23, replacing M9's single `tags`). **Remaining:** vocab curation + backfill via the `mine-vocab` → `group-vocab` → `/admin/vocab` pipeline. | [redesign](categories-keywords-redesign.md) · [implementation](categories-keywords-implementation.md) |
| **Inferred-geotag refinements** | ⬜ | Post-M19 cascade tuning: cap hop depth (776-deep chains observed), downweight inferred anchors on re-scan so decay protects against cross-run compounding, drop the dead `--max-cascade-rounds` flag. Localized to `infer_location.py`. | [infer-location-refinements.md](infer-location-refinements.md) |
| **Timeline view + LLM summaries** (M21) | ⬜ | Chronological UI grouping photos by period, each segment annotated with an LLM summary (where, from GPS/place; what, from descriptions/tags). No detail doc yet. | _(none)_ |
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

---

## Notes on numbering

Milestone numbers have gaps: **M22/M23 were never used** (numbering jumped M21 →
M24), and the categories/keywords redesign that produced schema v22/v23 was never
given a milestone number — it's tracked by name above. "M27" is claimed by both
the review-folders work and the VLM-rerank doc; treat the names, not the numbers,
as canonical.
