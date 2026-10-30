# people-bind eval

Hand-labelled ground truth and cached model runs for the people-bind pass
(`docs/plans/description-names.md`). Labels are FILES, never DB rows.

- `sample.json`  - the photo ids, strata and dev/holdout split (committed; ids only)
- `labels.json`  - gold per person phrase (committed; ids and phrases, no paths or hosts)
- `sheet/`, `runs/`, `originals/` - generated, git-ignored

Workflow (never run against the live NAS DB; use a replica copy opened read-only):

    python evals/people_bind_eval.py sample --db photo_index.db.local --date-from 2026-01-01
    python evals/people_bind_eval.py sheet  --db photo_index.db.local     # contact sheet + labels skeleton
    # fill `gold` in labels.json from the sheet (about 2 hours), then:
    export PHOTOSEARCH_TEXT_LLM_URL=http://localhost:1234/v1
    python evals/people_bind_eval.py run --variant rule --tier rule --db photo_index.db.local
    python evals/people_bind_eval.py run --variant qwen --tier vlm --model qwen2.5-vl-7b-instruct --db photo_index.db.local
    python evals/people_bind_eval.py report --db photo_index.db.local --split holdout

Run `--split dev` while tuning; run `--split holdout` ONCE per decision.
