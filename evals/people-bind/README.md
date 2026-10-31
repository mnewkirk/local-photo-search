# people-bind eval

Hand-labelled ground truth and cached model runs for the people-bind pass
(`docs/plans/description-names.md`). Labels are FILES, never DB rows.

- `sample.json`  - the photo ids, strata and dev/holdout split (committed; ids only)
- `labels.json`  - gold per person phrase (committed once labelled; ids and phrases, no paths or hosts)
- `sheet/`, `runs/` - generated, git-ignored. Originals are cached in `evals/model-evals/originals/`.

The harness opens the DB `mode=ro` only; never point it at the live NAS DB.

## Workflow

    DB=photo_index.db.local          # read-only is enough; any copy of the replica works
    python evals/people_bind_eval.py sample --db $DB --date-from 2026-01-01 --collection '%soccer%' --adversarial 30
    python evals/people_bind_eval.py sheet  --db $DB     # labelling page + labels.json skeleton
    # label in the browser (below), download labels.json, then:
    python evals/people_bind_eval.py check  --db $DB --import ~/Downloads/labels.json
    export PHOTOSEARCH_TEXT_LLM_URL=http://localhost:1234/v1
    python evals/people_bind_eval.py run --db $DB --variant rule --tier rule
    python evals/people_bind_eval.py run --db $DB --variant <name> --tier both --model <LM Studio id>
    python evals/people_bind_eval.py report --db $DB --split dev

`run` sets `PHOTOSEARCH_LLM_REASONING_EFFORT=none` unless set, never caches a
transport failure (it stops after 5 photos in a row with no answer), records an
unusable answer as an error that abstains (what production would show), and
refuses to resume a variant under a different spec: ablations get their own
variant name (`--max-px 1536`, `--one-call`). Pick on `--split dev`; run
`--split holdout` ONCE per decision.

Phrases match by character OVERLAP, not exact text: the model picks its own
boundaries ("One child" labelled, "One child in blue" bound is the same
mention). `check` rejects overlapping phrases in one photo.

The 150-bound-mentions gate is on the holdout only. A ~200-photo sample has ~75
holdout photos and will read INCOMPLETE on that gate; top it up with holdout-only
photos (`sample ... --only-split holdout --single N --multi-2-3 N --multi-4-6 N`),
after importing your labels (the page keeps browser edits per sample, so a new
sample starts from labels.json).

## Labelling guide

Open `sheet/index.html` in a browser (from Windows:
`\\wsl.localhost\Ubuntu\home\mattn\local-photo-search\evals\people-bind\sheet\index.html`).
Each photo shows the image with numbered red boxes, the raw description, the
name and label source behind each box, and the person phrases found
automatically. Edits are saved in the browser as you go; "Download labels.json"
any time, then `check --import` it.

For every phrase, mark what it refers to **in the pixels**, ignoring the names:

- **box number(s)**: the person the phrase describes is that box. A phrase
  naming 2-3 people ("two boys") gets all their boxes.
- **NONE**: the person has no box (back turned, face hidden or undetected,
  out of frame), or the phrase is a crowd / more than 3 people.
- **AMBIGUOUS**: the description does not say which of several boxes it means
  ("a player in green" with two players in green), or you cannot tell.

Then:

- **Add every phrase that refers to a specific person** that is not listed
  ("the midfielder", "a man in a cap", "another child in dark clothing"):
  select the words in the description and click "+ add selected text". A bound
  phrase nobody listed fails the gate. Skip pronouns and crowds ("others",
  "spectators", "the team"). "remove" is only for a listed phrase that is not
  a person.
- **One entry per reference.** "A young boy ... The boy ..." are two entries
  with the same box. Do not list overlapping phrases.
- **Tick "name wrong"** on a box whose shown NAME is not that child. This is
  separate from the phrase gold: a phrase can point at the right box (gold =
  that box) while the box carries the wrong name. The first is a mapping
  question, the second a face-label error; the report counts the second as a
  user-visible wrong name, split by `strict` vs `manual`.
- Tick "every person phrase is listed and labelled" when the photo is done.
  "only unfinished" hides finished photos.

About 200 photos and ~325 phrases: budget 2 hours (half a minute a photo; the
4-6-face photos take longer). Adversarial photos carry the heuristic reason
they were picked (`lone_face_no_subject_boxes`, `counted_group_fewer_boxes`,
...); label them like the rest.
