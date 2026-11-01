---
paths:
  - "photosearch/people_bind*.py"
  - "photosearch/description_render.py"
  - "photosearch/description_audit.py"
  - "evals/people_bind_eval.py"
  - "evals/people-bind/**"
---
# people-bind: names in descriptions — traps

Plan and decisions: docs/plans/description-names.md. History: docs/notes/llm-passes.md ("people-bind").

- **The raw `photos.description` is never edited, and search reads it on purpose.** Names appear only in `description_display` / `description_subjects`, added at read time by `description_render`. Search, verify, category-content, keywords and `generations` stay on raw text (`_PEOPLE_KEYWORDS` needs "boy/girl/people"; "Zack" queries already route through trusted face labels). Never write a name into `photos.description`.
- **Spans, not a rewritten template.** `description_subjects.slots_json` holds character spans of the raw text plus face id + bbox. With every slot generic the render is byte-identical to raw (a property test). An LLM must never write display text.
- **The pass is label-free.** `people_bind_input` carries face ids, boxes and scores, never a name or person id. Giving the model names would make the binding depend on labels and force a re-run on every relabel.
- **The code decides, not the model.** The VLM lists every box a phrase *could* describe; bind only when the count equals the people the phrase names (`decide`). A phrase occurring twice, a pronoun, a crowd, a group of more than 3 never binds. Multi-face photos need a second call with re-shuffled numbering to agree (`merge_calls`).
- **Rule tier needs `subject_boxes`.** No boxes means the VLM tier, never a guess: the lone detected face may be a sideline parent. Both tiers are gated separately by the eval; drop a tier whose wrong-bind count is above 0.
- **Only `db.DISPLAY_NAME_SOURCES` name a child** (manual, strict, merge_review; env `PHOTOSEARCH_DISPLAY_NAME_SOURCES` may only narrow). `temporal`, `rejected`, `dedupe_unmatched`, NULL never do. A person on two live faces in the photo is never named there. Groups are all-or-nothing.
- **A missing face is re-paired by IoU >= 0.7, one to one, only when unique** (`repair_faces`), at render time AND in `sweep_subjects` (one function). No write hooks on face deletes: a missed sweep shows generic words, never a wrong name.
- **`description_sha` guards every use.** A stale row renders raw; the submit defers (no attempt) when the sha no longer matches. `invalidate_description_dependents` drops the row even for verify's own rewrite (`include_verify=False`).
- **Not a batch step.** `db.NON_BATCH_PASSES`: in `rerun.ALL_PASSES` / `worker_api._ALL_PASSES`, never in `batch_state.WORKER_PASSES` (labels arrive after a batch; `fleet_launch_passes` needs a topological order). The claim waits for verify (`verified_at` or verify at the attempts cap) because pass order does not protect scoped runs.
- **Claim SQL drives from the faces list** (`p.id IN (SELECT … trusted faces)`); the correlated EXISTS form scanned all of `photos`. `tests/test_scoped_queue_counts.py` exempts this pass for that reason. >6 offered faces are excluded in the predicate (`PEOPLE_BIND_SCOPE_SQL`), never written as `'[]'` from a claim.
- **Outbound text stays raw.** Google Photos / Shutterfly captions never read the display text (children's names would leave the machine); book captions do.
- **Schema: v41 `description_subjects`, v42 person attributes, v43 `person_boxes`.** The tests pin `SCHEMA_VERSION == 43`. The people-context tables (`body_attrs`, `body_identity`, `body_labels`, `hints_json`) are v44.
- **The claim waits for `bodies`** (rows, or the bodies pass at its cap), as it waits for verify. Bodies in the claim = body mode + overlap check; none stored = face mode. `bodies` is a non-batch GPU pass (`--native`, `HSA_ENABLE_DXG_DETECTION=1` on the desktop), never on the NAS.
- **Two checks guard a body-mode bind** (docs/notes/llm-passes.md, "people context: Step 0"): the keypoint-verified link (`keypoint_contested`: the linked body, and no other, has a head keypoint in the UNPADDED face box) and the overlap second call (`overlap_checked`: a chosen box >= 30% inside a non-member's box needs both swapped-letter A/B answers to pick it). Eval body files without a `keypoints` key skip the first check.
- **Rule 7 (`attribute_conflict`) reads `persons.gender` / `birth_year` in the ONE persons query** (`load_persons_with_attributes`) and `date_taken` from the joined `description_subjects` query; never a per-slot query. No attributes set must render byte-identically to before (property test). The attributes never leave the machine (not in MCP/Ask output or Google/Shutterfly code; grep test). Notes: docs/notes/llm-passes.md, "people-bind R4".
- **Replica attribute writes go NAS-first** (`web._nas_person_attributes`, then `_mirror_person_attributes` copies the NAS's values). A pre-v42 NAS answers 405/404 → 409 "deploy the NAS first"; never fall back to a local-only write.
