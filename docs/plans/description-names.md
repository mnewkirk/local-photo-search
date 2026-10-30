# Names in photo descriptions that follow the face labels

**Status: 🟡 built 2026-10-30 up to the eval gate; no model run, nothing deployed, no backfill.** (Status lives in `README.md`; this header is a convenience.)

Produced by a plan-debate (two independent planners, two critique rounds). Both agreed on almost everything; the
three points that stayed open are decided in the table below. No code, schema or live data was touched.

## Problem

`photos.description` comes from a vision LLM and describes people generically ("A young boy with reddish hair walks
across a grassy field…", photo 248080) even when the face in the photo is hand-labelled (Zack Azerad, `manual`).
Matt also corrects labels by hand (2026-10-30: a face labelled Koa was Calvin), so any named text must **follow label
changes**, not be baked in. Goal: render "Zack walks across a grassy field…" with no LLM re-run when a label or name
changes, and never name the wrong child.

## Design in one paragraph

Raw `photos.description` is never modified. A new label-free pass, **`people-bind`**, shows a vision model the photo
with every sizable face drawn as an anonymous numbered box plus the raw description, and records *which phrase of the
raw text refers to which face* as character spans keyed on **face id + bbox fingerprint**. Names are filled in **at
read time** by one pure render function that joins the face's *current* `person_id` / `match_source` to `persons.name`.
So a relabel, rename or person merge changes the displayed text immediately; nothing stored depends on a label. With
every span generic the render is byte-identical to the raw text, and any doubt (untrusted source, stale text, moved
face, ambiguity) falls back to the original words. Search, verify, categories, keywords and `generations` keep reading
the raw text.

Matt's proposal (1) (separate display text, raw preserved) is accepted. Proposal (2) (subject ids, rename without LLM)
is accepted in intent; the stored artefact is **span annotations**, not an LLM-written template with placeholders
(Decision 3). The `{s1}` template form is a derived debug view of raw + spans.

## Data model (schema v40)

```sql
CREATE TABLE IF NOT EXISTS description_subjects (
    photo_id        INTEGER PRIMARY KEY REFERENCES photos(id) ON DELETE CASCADE,
    description_sha TEXT    NOT NULL,   -- sha1 of the raw description the spans index into
    slots_json      TEXT    NOT NULL,   -- '[]' = "looked, nothing bindable" (a real result)
    method          TEXT    NOT NULL,   -- 'rule' | 'vlm'
    bind_version    INTEGER NOT NULL,   -- bump to re-bind everything (cf. SHARPNESS_VERSION)
    model TEXT, model_version TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);
```

`slots_json` element: `{"slot":"s1","kind":"person|group","spans":[{"start","end","text"}],"faces":[{"face_id","bbox":[top,right,bottom,left]}]}`.
`group` = 2-3 faces; more is never bound. Pronouns ("he", "they") are never bound or substituted.

New partial index in `db._SEARCH_INDEXES` (own commit; predicate text verbatim in the claim SQL):
`idx_faces_trusted_photo ON faces(photo_id) WHERE person_id IS NOT NULL AND match_source IN ('manual','strict','merge_review')`.

## Render rules (`photosearch/description_render.py`, pure, never raises → raw)

A slot becomes a name only if ALL hold; otherwise its spans keep the generic words:

1. `sha1(raw description) == description_sha`.
2. Every bound face still exists with the stored bbox (±2 px); a missing face is re-paired one-to-one by IoU ≥ 0.7 among the photo's current faces, else generic.
3. Every bound face has `person_id` and `match_source` in the **name sources** (default `face_verify.TRUSTED_LABEL_SOURCES` = manual, strict, merge_review; env `PHOTOSEARCH_DISPLAY_NAME_SOURCES` narrows). `temporal`, `rejected`, `dedupe_unmatched`, NULL never name anyone.
4. **A person who owns two or more live faces in the photo (any source except rejected/dedupe_unmatched) is never named in that photo**, whichever slot the faces are in.
5. A group is all-or-nothing ("Zack and Calvin"; never "Zack and another player").
6. Name = first name; full `persons.name` if another person in the photo (or an ambiguous registered first name) shares it. Possessives keep their `'s`.

Properties (tested): all-generic render == raw; text differs from raw only inside spans; deterministic.

## Mapping (the part that must not guess)

Two tiers, reported and gated **separately**; a tier is dropped if its wrong-bind count on the holdout is above 0.

- **Rule tier (no model).** Binds only when: exactly one face passes the size/quality floor; exactly one singular
  person noun phrase from a closed head-noun list and no plural/collective/other-person word; **`subject_boxes` is
  present**, the face centre lies inside the largest subject box and that box's label is a person word. NULL
  `subject_boxes` → VLM tier (the lone detected face may be a sideline parent while the described child's face
  was undetected). Eval stratum includes "parent face detected, boy's face not".
- **VLM tier.** Numbered anonymous boxes (labelled *and* unlabelled faces, number order shuffled, no names anywhere in
  the prompt). The model lists for every person phrase **every box it could equally describe** (`could_be`), plus
  `kind` and `all_members_boxed`. **Code decides**: bind a singular phrase only if `|could_be| == 1`, an N-person
  phrase only if `|could_be| == N`; larger sets are ambiguous → no slot. On photos with ≥2 offered faces a **second
  call with re-shuffled numbering** must give identical phrase→face sets or the slot is dropped. Phrases must be exact
  substrings of the description; face ids must belong to the photo; violations raise `UnusableAnswer`.
- Not changing the describe prompt (placeholders in raw text would force a 250k re-describe, pollute search and
  verify input, and describe runs before labels exist).
- Scope limits: stills only; photos with more than 6 offered faces are **excluded in the claim/count predicate** (one
  shared SQL fragment, same quality floor as the boxes), never written as `'[]'` from the claim path.

## Consumers

| Reader | Text |
|---|---|
| Search (`_description_relevance`, `search_descriptions`, `search_text_fields`, exclusion/negation, people-word logic), verify, category-content, keywords, `generations`, eval/vocab/admin pages | **raw**. "Zack" queries already route through trusted face labels (`_extract_persons_from_query`); indexing names would add an untrusted second path. |
| `PS.PhotoModal`, collection cards | **display**, "show original" toggle, small "names from face labels" chip |
| Ask / MCP hits (`tools._compact_hit`) | **display**. ~9 call sites (`tools.py` and `web.py`): route all through one batched `_compact_hits(db, rows)`; a grep test requires bare `_compact_hit(` calls to go through it |
| Book captions (`book_ai.py:43`, `book_authoring.py:75`) | **display** (Decision 2) |
| Google Photos captions (`google_photos.py`), Shutterfly export (`shutterfly_export.py`) | **raw** unless the caller opts in per upload |

API: an additive `description_display` key; `description` is unchanged key-for-key. The detail endpoint also returns
`description_subjects: [{slot, generic, name, state, face_ids}]` with `state` ∈ named / unlabelled / untrusted_source /
rejected / stale / ambiguous_name, so the UI can say why a kid is not named. Frontend trap: the modal's `syncFaces`
(`shared.js`) re-fetches `/api/photos/{id}` after a face edit but keeps only `d.faces`; it must also keep
`description_display` / `description_subjects` or the text will not follow a relabel until reopened.

## Worker pass `people-bind`

- Role `bind` (`describe.PASS_ROLES`, `rerun._PASS_LLM`, `rerun.FLEET_ROLE_MODELS["bind"]`, `_OPENAI_MAX_TOKENS["bind"]`). **Add `"bind"` to the vision-role fallback tuple in `describe._resolve_openai_model`** (`("describe","verify","visual","aesthetics","subject")`) or an unset env var 400s. The eval picks the model; no rule that it differ from describe.
- `PASS_MEDIA["people-bind"] = "still"`. Position: after `verify` in all four lists kept equal by tests (`batch_state.WORKER_PASSES` = `rerun.ALL_PASSES` = `worker_api._ALL_PASSES` = `batch-flow.js`) plus `cli.py valid_passes`.
- **Not a gating batch step**: labels keep arriving after a batch, so a gating step would flip between waiting and blocked, and `fleet_launch_passes` assumes `WORKER_PASSES` is topologically ordered with NAS steps after it. No `DEPENDS_ON` on NAS steps. Keep it out of `ready` via `OPTIONAL_STEPS` or a `NON_BATCH_PASSES` tuple (Step 0 spike decides).
- **Claim/count predicate**: description present, still media, ≥1 trusted-source face (`idx_faces_trusted_photo`), ≤6 offered faces, `(verified_at IS NOT NULL OR verify attempts at the cap)` (pass order does not protect scoped or partial runs), no `description_subjects` row, `attempts < MAX_PROCESS_ATTEMPTS`; `ORDER BY +p.id`. A new trusted label makes a photo claimable with no hook.
- Claim payload (`people_bind_input`): face ids, bboxes, det_score, image size, `subject_boxes`, `description_sha`. **No names or person ids.** Rule tier needs no image fetch; the image is fetched only for the VLM tier (`SELF_READ_PASSES`).
- Submit branch (through `_SubmitOutcome`): recompute the sha of the *current* description and re-validate against the *current* faces; sha mismatch → `outcome.defer` with no attempt spent (precedent: video-describe `is_stale`, `worker_api.py` ~L1495; terminates because description changes are each capped by their own attempts and each re-queues via `invalidate_description_dependents`). Invalid → per-photo failure (one attempt + `index_errors`). Valid → upsert + `db.log_generation(pid, "people-bind", slots_json, …)` with provenance via `describe.effective_model` / `_provenance_kwargs`. `'[]'` is a result. Transient lock only defers (`_is_transient_db_error`, never a bare except). Truncated/unparseable/all-invalid output raises `TruncatedOutput` / `UnusableAnswer`. Fleet runs with `PHOTOSEARCH_LLM_REASONING_EFFORT=none`.
- `invalidate_description_dependents` also deletes the `description_subjects` row and the `people-bind` ledger row, **including when `include_verify=False`** (verify's own rewrite changes the text the spans index into). `clear-pass people-bind` deletes rows + ledger.
- Face ids die only on re-detection (`clear-pass faces`, `index.py` force, `face_edit.delete_face`, `rerun._apply_mirror`). No write hooks (rejected: fragile, miss future paths). Render-time validation (rule 2) is the safety net; a **maintenance sweep** calls the same re-pair function, rebinds by IoU ≥ 0.7, else deletes the row and ledger entry to re-queue; `validate-data` counts dangling slots.
- Replica: the NAS is the only writer. The replica renders from its own copied tables; `_mirror_face_labels` already keeps labels current, and `rerun._apply_mirror` keeps NAS face ids when free locally, so no new mirror code for relabels. `/api/photos/{id}/mirror-fields` + `_apply_mirror` carry the `description_subjects` row for M28 re-runs. The replica must tolerate a NAS DB older than v40 (`no such table` → raw).

## Steps

| # | Step | Est. |
|---|---|---|
| 0 | Spikes: fresh replica counts (scope, strict vs manual share, faces-per-photo); read `batch_state.py` L260-460 + `fleet_launch_passes` to confirm a worker pass can sit outside `ready`; confirm LM Studio serves the candidate vision models at reasoning effort none | 0.5 d |
| 1 | Schema v40 + index + `PASS_MEDIA` + claim/count predicates + `invalidate_description_dependents` (tests: migration from v39, invalidation both modes, EXPLAIN uses the index, pass-list equality) | 1 d |
| 2 | Pure modules `people_bind.py` (prompt, parser, validator, rule tier) and `description_render.py` + tests (accuracy cases from 248080; two-person refuse; possessive; plural groups; collisions; invariants) | 2.5 d |
| 3 | Eval harness + hand-labelled sets (Step "Eval"); Matt labels ~2 h | 3 d |
| 4 | Worker pass, submit branch, pass lists, role fallback, `clear-pass`, sweep | 2 d |
| 5 | Render seam, `description_display`, `_compact_hits`, book captions, raw-by-default Google/Shutterfly, flag, mirror-fields | 1.5 d |
| 6 | Frontend (modal, `syncFaces`, chip, per-viewer toggle in try/catch localStorage, dragged re-run set) + `node scripts/check-frontend-refs.js` | 0.5 d |
| 7 | Audit seam, docs, deploy + pilot (see below) | 1 d + GPU time |

≈ 11 dev days + GPU hours. Every step lands behind `PHOTOSEARCH_DISPLAY_DESCRIPTIONS` (default off); the pass only runs when someone launches it.

## Eval (gate before any backfill)

Follows `model-eval-harnesses.md`: labels are **files** (`evals/people-bind/labels.json`, ids only, no paths/hosts), pinned role model (`model_eval.pin_role_model`), transport errors raise and never cache, **dev/holdout split with no tuning on holdout** (the sharpness 0.86→0.55 lesson). The harness calls the production `people_bind` and `description_render` functions, not copies.

Sample: soccer 2026, stratified: 1-face (30+), 2-3 faces (50), 4-6 faces (40), adversarial (30: back-turned subject with a spectator face, "parent face, boy undetected", identical jerseys, goalkeeper vs field players, face cut by frame edge, "two players in blue" with one visible). Gold per phrase: the set of face ids, NONE, or AMBIGUOUS, marked from a generated contact sheet that separates *mapping errors* from *wrong face labels*.

Metrics: correct-bind, **wrong-bind** (bound to anything but gold, or bound when gold is NONE/AMBIGUOUS), right-abstain, coverage, render-level **user-visible wrong-name rate split by `strict` vs `manual` source**, invariant failures over 2k random photos, seconds and calls per photo. Ablations: rule vs VLM, one vs two calls, 1024 vs 1536 px, candidate models (qwen3.5-9b, gemma-4-12b, qwen2.5-vl-7b).

Gates (all on holdout unless noted): wrong-bind **0** across ≥150 bound mentions (rule of three ≈ 2% bound) including 0 on the adversarial set; rule tier separately 0; coverage ≥ 50% on multi-face and ≥ 80% on single-face; null/abstain correctness ≥ 95%; all-generic == raw at 100%; render p95 < 5 ms per 100-photo page. `strict` labels stay in the name sources only if the eval shows **0 strict-caused wrong names** (Decision 1). The VLM tier may fail while the rule tier ships.

## Rollout

Deploy NAS (additive v40 migration, flag off) → replica → workers together (`workers.md`). Pilot: Windows GPU only (`--native`, LM Studio), scoped via the existing collection/date scope to the 2026 soccer collection (~1-2k photos), never `-d /photos`; Matt reviews 100 random renders with the flag on in his own view; **zero wrong names** required. Then widen: all collection photos → 2025-26 trusted-label photos → rest (optional). Not on Mac Ollama (wedges), not on the N100. Rollback: flag off / `clear-pass people-bind` / `DROP TABLE`; raw text was never changed.

Backfill cost (re-measure in Step 0): the snapshot (stale, schema v28, 2026-07-10, 151k photos) had 29.5k photos with a trusted-source face (~1/3 single-face → rule tier, no GPU); today probably 35-55k. VLM tier ≈ 40-140 GPU-hours depending on seconds per call (unmeasured; an eval output).

## Audit-log seam

One module, `photosearch/description_audit.py: record(db, photo_id, kind, details)`, kinds `bound` / `unbound` / `rebound`; a no-op log line today, only its body changes when the per-photo audit branch merges. Relabels are already that log's own event; **rendered-text changes are not stored as events** (the audit view can compute "text now" by calling the render).

## Docs on landing (per "Where knowledge goes")

`.claude/rules/llm-passes.md` bullet (**people-bind: spans not text, label-free binding**); `.claude/rules/faces.md` bullet (**names render only for trusted sources; bbox fingerprint**); `.claude/rules/search.md` bullet (**search reads raw on purpose**); schema reference row; eval numbers under a dated `###` in `docs/notes/llm-passes.md`. CLAUDE.md: nothing.

## Risks and unknowns

- Upstream label error: a `strict` mislabel in a lookalike-kid set shows as a wrong name (one click from a fix). Mitigated by the eval's per-source split, the env narrowing knob, and the pilot review.
- VLM accuracy on tiny soccer faces (~37 px at 1024 px) and seconds per call: unmeasured; ablate resolution.
- Closed-list noun phrases miss "the midfielder", "the keeper in yellow" → lost coverage, not accuracy.
- Whether a worker pass can sit outside `ready` (Step 0); fallback `NON_BATCH_PASSES`.
- Today's real scope and source mix: stale snapshot only.
- Frontend edits need an image rebuild.

**Out of scope, flagged:** `video_describe_store.describe_input` bakes `persons.name` into *video* descriptions at claim time with **no `match_source` filter** (temporal and dedupe_unmatched included). Those names are frozen in `photos.description`, flow into keywords/categories, and never follow relabels. Fix later with the same source filter or by moving videos to this scheme.

## Decisions

Points the debate resolved by agreement are in the "Resolved in debate" list; the rest were put to Matt.

| Topic | Option A | Option B | Choice | Date |
|---|---|---|---|---|
| Which labels may name a child | Trusted incl. `strict` (manual+strict+merge_review), narrowable by env, strict kept only if the eval shows 0 strict-caused wrong names | `manual` + `merge_review` only (near-zero auto-match risk, far lower coverage) | **A** | 2026-10-30 |
| Names in outbound/printed text | Books use named text; Google Photos and Shutterfly-export captions raw unless opted in per upload | Everything raw until opted in, books included | **A** | 2026-10-30 |
| Stored template shape | Spans over the raw text, rendered at read time (no stored display column) | LLM-written `{face:id}` template in a stored display column | **A** | 2026-10-30 |

Resolved in debate (no decision needed): rule tier kept but requires `subject_boxes` and is gated separately; VLM returns `could_be` sets and code decides abstention, with a shuffled second call; render-time face validation + sweep instead of `faces_changed` write hooks; non-gating pass whose claim waits for verify; crowd cap in the claim predicate; a person on two faces is never named; `render_changed` audit events dropped; bind model chosen by the eval; table `description_subjects`, flag `PHOTOSEARCH_DISPLAY_DESCRIPTIONS`. Left to the eval: group size (2-3 faces) and whether groups ship at all.

## Build status (2026-10-30)

Built on branch `claude/silly-shaw-64bd05`, all tests green, **nothing deployed and no model run** (LM Studio was busy).
Differences from the plan above, all deliberate:

- `people-bind` is in `rerun.ALL_PASSES` / `worker_api._ALL_PASSES` but NOT in `batch_state.WORKER_PASSES`; the spike resolved to `db.NON_BATCH_PASSES` (no `OPTIONAL_STEPS` use). It sits last in the fleet order, after verify.
- The claim predicate uses `p.id IN (SELECT photo_id FROM faces WHERE <trusted>)`, not a correlated EXISTS (the EXISTS form scanned all of `photos`); `test_scoped_queue_counts` exempts this pass for that reason.
- Ask/MCP names are applied once in `tools.call_tool` (`_with_display_names`), not by a `_compact_hits` wrapper at the ~9 call sites. The Google caption is raw by simply not reading the display text; the per-upload opt-in is not built.
- The VLM-slot subject-box guard is not implemented (it can only lower coverage); revisit if the eval shows a wrong bind it would catch.
- Fleet role model for `bind` is provisional (`qwen2.5-vl-7b-instruct`); the eval picks it.
- **Schema v40 is claimed twice**: the per-photo audit branch (`claude/stoic-kapitsa-99628c`) also adds v40 (`photo_audit`). Whichever merges second renumbers to v41 and bumps the `== 40` test assertions. `description_audit.record` is the only seam to change when the audit module lands.

Still to do before any backfill (needs the GPU box and ~2 h of the owner's labelling): `evals/people_bind_eval.py sample` / `sheet` / label / `run` / `report --split holdout`; fresh live scope counts; deploy NAS first, then replica, then workers (flag off).
