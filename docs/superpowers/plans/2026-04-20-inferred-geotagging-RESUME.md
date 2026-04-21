# M19 Inferred Geotagging — Resume Notes

**Paused:** 2026-04-20 after Task 6.
**Branch:** `m19-inferred-geotagging` (7 commits ahead of `main`, not pushed).
**Plan:** `docs/superpowers/plans/2026-04-20-inferred-geotagging.md`
**Spec:** `docs/superpowers/specs/2026-04-20-inferred-geotagging-design.md`

---

## What's done (Tasks 1–6)

| # | Task | Commit | Status |
|---|---|---|---|
| 1 | Schema v17 migration (`location_source` + `location_confidence` columns, exif backfill) | `a04c2c7` | reviewed ✅ |
| 2 | Haversine helper + 6 multi-region tests | `933d47f` | reviewed ✅ |
| 3 | `_scan_photos` + `_parse_date` helpers | `12de94c` | reviewed ✅ |
| 4 | Direct inference (round 1) — `_find_flanking_anchors`, `_infer_one_round`, `infer_locations` skeleton | `ded86e0` | reviewed ✅ |
| 5 | Cascade loop (fixpoint-ish sequential promote-as-you-go) + bug fix | `2735d0a` + `1edd9a7` | reviewed ✅ |
| 6 | UTF-8 `place_name` roundtrip regression guard | `edccc3f` | skipped review (trivial) |

**Test state:** 19 tests in `tests/test_infer_location.py` pass. Full suite (excluding `test_face_matching.py` and `test_integration.py`) passes: 346/346.

**Design deviations from the plan that got accepted:**

- **Task 5 — algorithm restructure.** The plan described fixpoint rounds; the implementation is sequential time-ordered promote-as-you-go. Equivalent correctness, strictly higher confidence for chains because each photo anchors to its *nearest* just-inferred predecessor instead of the more-distant real GPS. `_find_flanking_anchors` walks both directions so two-sided flanking works regardless of scan order.
- **Task 5 — test timestamp fixes.** Plan's `test_cascade_three_hop_chain` used 20-min intervals but asserted `2/3 * 0.7` confidence, which requires 10-min intervals. Corrected to T10:10/T10:20/T10:30. Plan's `test_cascade_movement_guard_transitive` put Portland at T12:00 (outside t90's window); moved to T11:15 so the movement guard can actually fire.
- **Task 5 — cleanup.** Dead `skipped_counts` accumulator inside the cascade loop was removed; the post-loop `_infer_one_round` recount produces the authoritative skip totals.

**Open minor concerns carried forward:**

- **`window_minutes` type annotation inconsistency.** Public signature in `infer_locations` says `int`, helpers (`_find_flanking_anchors`, `_infer_one_round`) type-hint `float`. Works at runtime; may want to unify to `float` when Task 7 adds the CLI flag (Click's `type=int` vs `type=float` matters here).
- **`cascade_rounds_used` semantics.** Computed as `max(hop_count)` across candidates. With `cascade=False` it returns 1 when photos exist and 0 when not — arguably confusing, but the field was for UI display and is consistent.

---

## What's left (Tasks 7–10)

All task text + full code blocks live in the plan at `docs/superpowers/plans/2026-04-20-inferred-geotagging.md`. Summaries:

### Task 7 — CLI command `photosearch infer-locations`

- **Files:** modify `cli.py`, create `tests/test_cli_infer.py`.
- **Adds:** a `@cli.command("infer-locations")` with `--window-minutes`, `--max-drift-km`, `--min-confidence`, `--cascade/--no-cascade`, `--max-cascade-rounds`, `--apply` flags.
- **Behavior:** dry-run prints candidate summary + confidence/hop histograms + 10 sample inferences. `--apply` calls `reverse_geocode_batch` on the coords and writes `gps_lat`/`gps_lon`/`place_name`/`location_source='inferred'`/`location_confidence` in one transaction, with `WHERE gps_lat IS NULL` to guard against overwrites.
- **Tests:** 3 CLI tests using `click.testing.CliRunner` — dry-run reports candidates, apply writes stamped inferred rows, apply does not overwrite pre-existing GPS.
- **Note:** plan text has `cli.py` insertion pointer near line 962 — verify before pasting; earlier tasks haven't touched it.

### Task 8 — API endpoints

- **Files:** modify `photosearch/web.py`, create `tests/test_web_geocode.py`.
- **Adds:** `POST /api/geocode/infer-preview` (read-only summary + 10 samples with thumbnails + pre-reverse-geocoded `place_name`) and `POST /api/geocode/infer-apply` (requires `confirm: true`, writes in one transaction).
- **Uses:** project's existing `data: dict` POST body convention (not Pydantic BaseModel — the plan was revised during self-review). A `_parse_infer_params` helper coerces int/float types from JSON.
- **Tests:** 4 integration tests using the `client` + `db` fixtures from `tests/conftest.py`. Note the fixture has pre-seeded no-GPS photos, so one test uses `min_confidence=0.99` to get a "no candidates" state instead of assuming an empty fixture.

### Task 9 — Frontend `InferLocationForm` panel

- **File:** modify `frontend/dist/status.html`.
- **Adds:** a React component (vanilla UMD, `React.createElement` no JSX) shaped like the existing `StackingForm`. Params inputs → **Preview** button → summary + histograms + 10 sample thumbnails → **Apply** button (disabled until current params have been previewed; re-enables on param change → re-preview).
- **No automated tests** — project has no frontend test toolchain. Verify manually in a browser after rebuilding the Docker image.

### Task 10 — Documentation pass

- **Files:** modify `CLAUDE.md`, `.claude/skills/photo-search/SKILL.md`.
- **Bumps:** schema-version callouts from 16 → 17.
- **Adds:** an "Inferred geotagging (M19)" section describing the CLI command, API endpoints, `/status` panel, schema columns, and the rollback SQL snippet.

---

## Resumption checklist for tomorrow

1. `git -C /Users/mattnewkirk/Documents/Claude/Projects/photo_organization/local-photo-search status` — confirm branch is `m19-inferred-geotagging`, clean.
2. `pytest tests/test_infer_location.py -v` — confirm 19 green.
3. Re-enter subagent-driven-development flow. Next up: **Task 7 (CLI)**.
4. Keep using the combined spec + code-quality review subagent pattern — it's faster than two separate calls.
5. Task 7 subagent should notice the `window_minutes: int` annotation (flagged in Task 4 review) — if adding `--window-minutes` to Click, use `type=int`. No float-width needed for the CLI surface. The internal float-ish usage is fine because Python's division is float anyway.
6. After Task 10, dispatch the final whole-branch code reviewer per the subagent-driven-development skill, then use `superpowers:finishing-a-development-branch`.

## Branch merge plan (for when Task 10 lands)

- Option A: `git checkout main && git merge --ff-only m19-inferred-geotagging` — fast-forward, preserves per-task commits.
- Option B: squash-merge if you want a single history entry on `main`.
- The feature branch has no remote push yet — whether to push depends on your workflow.
