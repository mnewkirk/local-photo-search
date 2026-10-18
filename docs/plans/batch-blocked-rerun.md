# `/batches`: show blocked photos and re-run them from the step box

**Status: PLANNED 2026-10-18.** Nothing built yet. Companion to
`docs/plans/category-grounding.md` (which fixes WHY the motivating photo
fails); this plan fixes HOW an owner acts on a blocked step at all.

## Problem

A worker-pass step on `/batches` reads `blocked` when every remaining photo
has `worker_processed.attempts >= MAX_PROCESS_ATTEMPTS`. Today the page shows
the count and nothing else. Acting on it means: query `worker_processed`
joined to the batch folder by hand to find the ids, read `index_errors` for
the reason, then either open each photo's modal and use "Re-run passes" or
call `POST /api/admin/rerun-passes {photo_ids, passes, mode}` yourself. The
alternative is Mark ready, which accepts the loss without naming it.

Real case, 2026-10-18: batch 6 (`2026/2026-10-17_ILCE-7RM6`, 1,074 photos),
`category-content` blocked on one photo (251357) after three "vocabulary
recitation" failures. Nothing on the page named the photo or the reason. A
hand-made re-run then failed a fourth time, and the sync result came back as
`written: 0, processed: 1`, which the existing callers (`PS.PhotoModal`'s
`submitRerun`) report as success.

## Root cause recap

- `batch_state._worker_step` already computes `failed` with the exact
  photo-level predicate; it keeps only the count and throws away the ids.
- `faces` can never read `blocked`: `_EMPTY_OUTPUT_IS_DONE` zeros its
  `failed` (an exhausted photo with no face rows is DONE, not failed). Any
  ids query must apply the same guard so those photos are never listed.
- `rerun.run_pass_sync` has the failure message in hand (`worker._failure_row`
  → `worker._is_failure_row`) before it submits, and drops it when building
  its return dict. That is why `written: 0` prints as success.

## Routing

- **Read (blocked list):** `batch_state` + a new `GET /api/batches/{id}/blocked`
  on the existing batches router, so it inherits the "every `/api/batches`
  route proxies to the NAS in replica mode" rule. The NAS always computes it
  (directly, or because the replica proxied), the same way the rest of the
  page avoids a stale local copy.
- **Write (re-run):** reuse `POST /api/admin/rerun-passes` unchanged. It is
  deliberately NOT on the batches router and NOT proxied: `mode=sync` must
  run where LM Studio is (the desktop replica), and `mode=queue` already
  branches on `rerun.nas_base()` to call the NAS's `clear-pass` from either
  side. No new re-run endpoint.
- **Cache refresh:** a new `POST /api/batches/{id}/invalidate` on the proxied
  router, called by the UI after a re-run or queue completes, then the page
  re-fetches detail. The actual writes land through `worker_api`, which does
  not know `batch_api`'s memo exists; invalidating from the client through the
  proxied route reaches the right process's memo in both deployment shapes.

## Steps

1. **`photosearch/batch_state.py`: ids and detail.**
   `blocked_photo_ids(db, batch_id, step)`: the `_OUTPUT_MISSING` + media
   scope + attempts-cap SQL of `_worker_step`'s `failed`, selecting `p.id`,
   chunked like `_count_where`. Returns `[]` at once when `step not in
   WORKER_PASSES or step in _EMPTY_OUTPUT_IS_DONE` (the explicit faces guard,
   against a stale or hand-typed step).
   `blocked_photo_detail(db, batch_id, step, limit=200)` →
   `{"total": N, "truncated": bool, "photos": [{"photo_id", "filename",
   "attempts", "last_error"}]}`. `attempts` from `worker_processed`;
   `last_error` is the NEWEST `index_errors` row with `pass_type = step AND
   filepath = str(photo_id)`. That column holds the stringified photo id only
   for the two worker-failure writers (`worker_api._record_worker_failures`,
   `_record_write_failure`); other `log_error` callers store a basename, so
   the match is specific to exactly these failures.
   Verify: unit tests against both functions with a batch holding exhausted
   `category-content` rows and a planted `index_errors` row; `faces` returns
   `[]` even with exhausted no-face rows present.
2. **`photosearch/batch_api.py`: two routes.**
   `GET /{batch_id}/blocked?step=`: proxy when `_nas_url()`; else 400 if the
   step is not a worker pass, 404 if no such batch, else `blocked_photo_detail`.
   `POST /{batch_id}/invalidate`: proxy when `_nas_url()`; else 404 if no such
   batch, else `_invalidate(_key(batch_id))`, returns
   `{"batch_id": …, "invalidated": true}`.
   Verify: `tests/test_batch_api.py` cases mirroring the proxy and cache
   tests; a GET after `/invalidate` recomputes (extend the "two GETs call
   batch_state once" harness to assert a second call after invalidate).
3. **`photosearch/rerun.py`: stop reporting a failure as success.**
   In `run_pass_sync`, `failure_row = next((r for r in results if
   W._is_failure_row(r)), None)` and add `"error": failure_row["error"] if
   failure_row else None` to the returned dict beside `written` / `processed`
   / `deferred`. No NAS round trip: the message was computed locally before
   submission. This also fixes `PS.PhotoModal`'s `submitRerun` "Re-ran N
   pass(es)" message in `shared.js`, which has the same bug (only a raised
   exception counts as failure today). Fix it once here, not in two JS files.
   Verify: a worker-raising stub asserts `error` is the message and
   `written == 0`; a success asserts `error is None`.
4. **`photosearch/request_intent.py`: two rules.**
   `GET /api/batches/{id}/blocked` → "List blocked photos for batch {id}";
   `POST /api/batches/{id}/invalidate` → "Refresh cached state for batch
   {id}". Existing rules are `$`-anchored, so `…/6/blocked` never matches the
   bare `…/6` rule.
5. **`frontend/dist/batch-flow.js`: pure helpers.**
   `canInspectBlocked(step)`: `state === 'blocked'` AND the step is in
   `WORKER_PASSES` (excludes `ingest`'s sweep-failed blocked and every NAS
   step). `rerunResultLine(result)`: one sync result (now carrying `error`) →
   `{text, cls}`: `"still failing: <error>"` / `l-err` when `error` is set or
   `written === 0 && !deferred`, else `"done"` / `l-ok`. Tested in
   `frontend/__tests__/batch-flow.test.js` like `advanceLogLine`.
6. **`frontend/dist/batches.html`: the button and panel.**
   In `Node`, when `BF.canInspectBlocked(s)`, a small button under `node-cap`:
   "Show blocked (N)" (N = `s.failed`) → `onInspect(s.step)` (new page state
   `inspectStep`). Below `Flow` / `Legend`, when set, a `BlockedPanel` card:
   fetches `GET /api/batches/{id}/blocked?step=` on mount and step change
   (plain `fetch`, not `PS.poll`: operator-opened, not an ambient poll); a
   thumbnail grid (`/api/photos/{id}/thumbnail`, click opens `PS.PhotoModal`
   as `review.html` / `faces.html` do), each row's `attempts` and
   `last_error`, and "N of TOTAL shown" when truncated. Three actions, each
   disabled while busy:
   - **Run now**: per-photo loop, `mode: "sync"`, one call per visible photo
     (so one poison photo cannot hide the others' results), each line via
     `BF.rerunResultLine`.
   - **Queue for fleet**: `mode: "queue"`, one call with every blocked id.
     Panel copy says "resets attempts; a worker fleet run picks it up", so
     nobody expects instant results.
   - **Accept**: closes the panel. Mark ready is untouched and remains the
     whole-batch version of the same choice.
   After either action settles: `POST /{id}/invalidate`, then `loadDetail()`
   (the existing `act()` pattern for dismiss/ready) and re-fetch the blocked
   list so the panel reflects fresh state too.
   Verify: manually against a seeded blocked batch on the dev server;
   `node scripts/check-frontend-refs.js` passes (new calls resolve through
   `PS.BatchFlow`, never as bare names).

## Risks

- **Double launch:** Run now runs in-process and never touches the fleet
  launcher's 409 guard. Queue only resets ledger rows via `clear-pass` and
  launches nothing; the step flips to `needs_queue` and the existing Advance
  / launch button, already governed by the 409 check, claims it later.
- **Burning attempts on a photo that will fail again:** each failed Run now
  bumps `attempts` past the cap (harmless to the `>=` predicate, but a wasted
  LM Studio call). No auto-retry anywhere: one photo, one click, one result
  line, and `last_error` is shown up front so the owner can tell "vocabulary
  recitation" (worth a try once the grounding guard ships) from file
  corruption (not worth one).
- **A huge blocked list on a bad batch:** `limit=200` + `total` / `truncated`
  bound the endpoint and the grid. Queue still clears all blocked ids
  server-side (cheap); only Run now and the display are capped.
- **`faces`' exhausted no-face photos:** excluded twice, by `_worker_step`
  never emitting `blocked` for faces and by `blocked_photo_ids` refusing any
  step in `_EMPTY_OUTPUT_IS_DONE`.

## Tests

- `tests/test_batch_state.py`: `blocked_photo_ids` / `blocked_photo_detail`:
  correct ids for an exhausted pass; `[]` for `faces`; `last_error` and
  `attempts` shape; truncation at `limit`.
- `tests/test_batch_api.py`: the GET proxies in replica mode; 400 on a
  non-worker-pass step; `/invalidate` clears the memo (next GET recomputes);
  `/invalidate` proxies in replica mode.
- `tests/test_rerun.py`: `run_pass_sync` returns `error` + `written: 0` on a
  worker-reported failure, `error: None` on success.
- `frontend/__tests__/batch-flow.test.js`: `canInspectBlocked` true only for
  a worker-pass blocked row; `rerunResultLine` distinguishes done from
  still failing.
- `tests/test_request_intent.py`: both new routes classify.

## Estimate

Backend (steps 1 to 4) 0.5 to 1 day; frontend (steps 5 and 6) with manual
verification against a seeded blocked batch about 1 day. Total 1.5 to 2
days, plus `./scripts/test-like-ci.sh` and `scripts/check-frontend-refs.js`
before pushing. Deploy: NAS first (the proxied routes), then the replica.
