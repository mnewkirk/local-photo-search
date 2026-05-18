# Categories + Keywords Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single 78-word `photos.tags` column with three richer fields — `categories` (mined controlled vocabulary), `visual_tags` (mood/light/composition vocab), and `keywords` (free-form noun phrases extracted from descriptions) — so 98% of described photos get useful, searchable text metadata.

**Architecture:** Three new worker passes claim against `worker_processed`. Two are text-only (read `photos.description`, ~32 photos/batch); one is vision (`llava`, mirrors the existing tags pass). A new `mine-vocab` → `group-vocab` → `/admin/vocab` curator pipeline produces the content + visual vocabularies and a query-expansion map as committed Python modules. Search adds three match functions and three new filters. Old tags pass + frontend bits are deleted cleanly; one schema migration adds the four new columns plus a backup snapshot for one-way rollback.

**Tech Stack:** Python 3.11, FastAPI, SQLite (schema v22 → v23), Ollama (llama3.2 text-only for content/keywords; llava for visual), spaCy `en_core_web_sm` (vocab mining), plain React (UMD) for curator UI.

**Related docs:**
- Design spec: `docs/plans/categories-keywords-redesign.md` — read first for "why".
- Worker pattern reference: `photosearch/worker_api.py:244` (`submit_results`), `photosearch/db.py:1501` (`mark_processed`), `photosearch/db.py:1466` (`log_generation`).

---

## File Map

**New files:**
- `photosearch/vocab_mining.py` — spaCy noun-chunk extractor + frequency sort.
- `photosearch/vocab_grouping.py` — LLM-based clustering of mined candidates into draft buckets.
- `photosearch/vocab_compile.py` — turns a curated draft JSON into three Python modules.
- `photosearch/vocab_admin.py` — admin API router (5 endpoints) + helpers.
- `photosearch/vocab_content.py` — **generated**, list of content terms.
- `photosearch/vocab_visual.py` — **generated**, list of visual-quality terms.
- `photosearch/vocab_query_expansion.py` — **generated**, `_QUERY_TO_CATEGORIES` dict.
- `photosearch/bakeoff.py` — keyword-extraction model bakeoff harness.
- `frontend/dist/admin_vocab.html` — curator UI.
- `tests/test_vocab_mining.py`
- `tests/test_vocab_compile.py`
- `tests/test_web_vocab.py`
- `tests/test_describe_extraction.py` — covers `extract_categories_from_description`, `extract_keywords_from_description`, refactored visual tag function.
- `tests/test_worker_categories.py` — end-to-end claim/submit for the three new passes.

**Modified files:**
- `photosearch/db.py` — schema v23 migration; pass-type values added to `get_unprocessed_photos` / `count_unprocessed_photos` / `mark_processed`.
- `photosearch/describe.py` — add `extract_categories_from_description`, `extract_keywords_from_description`, rename `tag_photo` → `tag_visual_photo`, drop `TAG_VOCABULARY`/`TAG_PROMPT`/`_parse_tag_response` (replaced by vocab-driven equivalents).
- `photosearch/worker.py` — new branch arms for the three passes (~lines 461–800).
- `photosearch/worker_api.py` — `SubmitRequest` fields, three new submit handlers, `clear-pass` arms.
- `photosearch/index.py` — `index_directory()` and `_index_collection()` enable flags.
- `photosearch/search.py` — three new match functions; rename `tag_match` → `text_match`; new filter params.
- `photosearch/web.py` — `/api/search` param renames + new filters; `tag=X → category=X` redirect; `/api/stats` field changes; mount admin router; serve `/admin/vocab`.
- `cli.py` — `--category-content` / `--category-visual` / `--keywords` flags on `index`; `mine-vocab`, `group-vocab`, `compile-vocab`, `bakeoff-keywords` subcommands; `--tags` flag removed; `valid_passes` set updated.
- `frontend/dist/shared.js` — `PS.PhotoModal` three new rows; `PS.SharedHeader` "Admin" entry; new `activePage: 'admin'`.
- `frontend/dist/index.html` — three new filter controls.
- `frontend/dist/status.html` — split Tags stat card into three; replace run-command rows.
- `requirements.txt` — add `spacy>=3.7,<4` and `en_core_web_sm` install instruction (model is downloaded separately, not pip-installable).
- `Dockerfile` — add `python -m spacy download en_core_web_sm` step in the runtime stage.

**Generated artifacts (not committed to repo by this plan; produced by tasks):**
- `/data/vocab_candidates.json` (Phase 2, Task 2.6 output)
- `/data/vocab_proposal.json` (Phase 2, Task 2.10 output)
- `/data/vocab_draft.json` (Phase 3, Task 3.10 output during curation)

---

## Phase 0 — Keyword extraction bakeoff

**Goal:** Pick the model used by the `keywords` pass (text-only) by running two candidates on a 30-description sample and inspecting outputs.

### Task 0.1: Bakeoff harness module

**Files:**
- Create: `photosearch/bakeoff.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_describe_extraction.py`:

```python
import pytest
from photosearch.bakeoff import build_keyword_prompt, parse_keywords_response


def test_build_keyword_prompt_embeds_description():
    desc = "A golden retriever runs along Stinson Beach at sunset."
    prompt = build_keyword_prompt(desc)
    assert desc in prompt
    assert "5-15" in prompt  # range from design


def test_parse_keywords_lowercases_and_trims():
    raw = "Golden Retriever, Stinson Beach, Sunset, dog, beach, Pacific Ocean"
    out = parse_keywords_response(raw)
    assert out == [
        "golden retriever", "stinson beach", "sunset",
        "dog", "beach", "pacific ocean",
    ]


def test_parse_keywords_dedupes_and_drops_empties():
    raw = "dog, , Dog,  beach , beach"
    out = parse_keywords_response(raw)
    assert out == ["dog", "beach"]


def test_parse_keywords_handles_newline_and_bullet_responses():
    raw = "- Golden Retriever\n- beach\n* sunset"
    out = parse_keywords_response(raw)
    assert out == ["golden retriever", "beach", "sunset"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_describe_extraction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'photosearch.bakeoff'`

- [ ] **Step 3: Implement `photosearch/bakeoff.py`**

```python
"""Phase 0 keyword-extraction bakeoff helpers.

Run `photosearch bakeoff-keywords` to compare two candidate models against
30 sample descriptions. Outputs side-by-side JSON for review before
hardcoding the chosen model in worker.py.
"""
from __future__ import annotations

import re
from typing import Optional

_KEYWORD_PROMPT_TEMPLATE = """\
Extract 5-15 keywords or short phrases from this photo description.

Rules:
- Include proper nouns (people, places), breeds, multi-word phrases (e.g. "golden retriever", "pacific ocean").
- Lowercase everything.
- Return ONLY a comma-separated list. No bullets, no explanation, no numbering.
- Skip vague words like "thing", "scene", "image", "photo".

Description:
{description}
"""


def build_keyword_prompt(description: str) -> str:
    return _KEYWORD_PROMPT_TEMPLATE.format(description=description.strip())


_BULLET_RE = re.compile(r"^[\-\*•]\s*")


def parse_keywords_response(raw: Optional[str]) -> list[str]:
    """Lowercased, deduped, ordered list of keywords from the LLM's raw text."""
    if not raw:
        return []
    # Normalise bullets / newlines to commas so the same splitter handles both.
    cleaned_lines = []
    for line in raw.splitlines():
        stripped = _BULLET_RE.sub("", line.strip())
        if stripped:
            cleaned_lines.append(stripped)
    text = ", ".join(cleaned_lines) if cleaned_lines else raw
    seen: set[str] = set()
    out: list[str] = []
    for token in text.split(","):
        kw = token.strip().lower()
        if not kw or kw in seen:
            continue
        seen.add(kw)
        out.append(kw)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_describe_extraction.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add photosearch/bakeoff.py tests/test_describe_extraction.py
git commit -m "feat(bakeoff): keyword prompt + response parser"
```

### Task 0.2: Bakeoff CLI

**Files:**
- Modify: `cli.py` — add `bakeoff-keywords` command near other admin commands (search for `clean-garbage-tags` to find the area).

- [ ] **Step 1: Implement the CLI command**

Append to `cli.py` (place near `clean-garbage-tags`):

```python
@cli.command("bakeoff-keywords")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB")
@click.option("--sample", default=30, show_default=True,
              help="Number of descriptions to sample.")
@click.option("--models", default="llama3.2:3b,llama3.2-vision",
              show_default=True,
              help="Comma-separated model list to evaluate.")
@click.option("--out", default="/data/bakeoff_keywords.json",
              show_default=True,
              help="Output JSON path with side-by-side results.")
@click.option("--seed", default=17, show_default=True,
              help="Random seed for reproducible sampling.")
def bakeoff_keywords(db, sample, models, out, seed):
    """Run two candidate models on N descriptions; emit JSON for review."""
    import json
    import random
    import time
    from photosearch.db import PhotoDB
    from photosearch.bakeoff import build_keyword_prompt, parse_keywords_response
    from photosearch.describe import _ollama_chat_with_retry, HAS_OLLAMA

    if not HAS_OLLAMA:
        click.echo("Ollama client not available; install `ollama` Python package.", err=True)
        raise SystemExit(1)

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    rng = random.Random(seed)

    with PhotoDB(db) as photo_db:
        rows = photo_db.conn.execute(
            "SELECT id, description FROM photos "
            "WHERE description IS NOT NULL AND length(description) > 40 "
            "ORDER BY id"
        ).fetchall()
    pool = [(r["id"], r["description"]) for r in rows]
    if len(pool) < sample:
        click.echo(f"Only {len(pool)} described photos found; using all.")
        chosen = pool
    else:
        chosen = rng.sample(pool, sample)

    results = []
    for photo_id, description in chosen:
        prompt = build_keyword_prompt(description)
        per_model = {}
        for model in model_list:
            t0 = time.time()
            try:
                raw = _ollama_chat_with_retry(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0},
                )
            except Exception as exc:
                per_model[model] = {"error": str(exc), "elapsed_s": time.time() - t0}
                continue
            per_model[model] = {
                "raw": raw,
                "parsed": parse_keywords_response(raw),
                "elapsed_s": round(time.time() - t0, 3),
            }
        results.append({
            "photo_id": photo_id,
            "description": description,
            "models": per_model,
        })
        click.echo(f"photo {photo_id}: " + " | ".join(
            f"{m}={len(per_model[m].get('parsed', []))} kw, "
            f"{per_model[m].get('elapsed_s', '-')}s"
            for m in model_list
        ))

    with open(out, "w") as f:
        json.dump({"sample": sample, "seed": seed, "models": model_list, "rows": results},
                  f, indent=2)
    click.echo(f"\nWrote {out} — review side-by-side and pick a model.")
```

- [ ] **Step 2: Smoke-check that the command parses**

Run: `venv/bin/python cli.py bakeoff-keywords --help`
Expected: help text listing all options.

- [ ] **Step 3: Commit**

```bash
git add cli.py
git commit -m "feat(cli): bakeoff-keywords command for Phase 0 model comparison"
```

### Task 0.3: Run the bakeoff and record the decision

- [ ] **Step 1: Run on the NAS against 30 photos**

```bash
DC="docker compose -f docker-compose.nas.yml run --rm"
$DC photosearch bakeoff-keywords --sample 30 --out /data/bakeoff_keywords.json
```

Expected: per-photo lines showing each model's keyword count + elapsed seconds; final "Wrote /data/bakeoff_keywords.json" line.

- [ ] **Step 2: Pull the JSON locally and inspect**

```bash
rsync nas.local:/volume1/docker/photosearch/data/bakeoff_keywords.json .
venv/bin/python -m json.tool bakeoff_keywords.json | less
```

Look for: per-model parse rate (non-empty results), avg keyword count, presence of proper nouns / multi-word phrases, response shape (clean comma list vs prose).

- [ ] **Step 3: Record the decision in this plan**

Edit this file: replace the placeholder in Task 4.2, Step 3 (`KEYWORDS_MODEL = "<TBD>"`) with the chosen model string. Add a one-line note under "Status" at the top of `docs/plans/categories-keywords-redesign.md` recording the choice and date.

- [ ] **Step 4: Commit the decision**

```bash
git add docs/plans/categories-keywords-redesign.md docs/plans/categories-keywords-implementation.md
git commit -m "docs(plans): record Phase 0 keyword-model decision"
```

---

## Phase 1 — Schema v22 → v23 migration

**Goal:** Add the four new columns, snapshot existing `tags`, mark legacy `generations` rows, and bump `SCHEMA_VERSION` — all idempotent and inside one transaction so a v22 database upgrades cleanly without losing data.

### Task 1.1: Migration test (TDD)

**Files:**
- Modify: `tests/test_db.py`

- [ ] **Step 1: Add the failing migration test**

Append to `tests/test_db.py`:

```python
def test_schema_v23_migration_from_v22(tmp_path):
    """Build a minimal v22 DB by hand, open it as v23, verify migration."""
    import sqlite3, json
    db_path = tmp_path / "v22.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT,
            description TEXT,
            tags TEXT
        );
        CREATE TABLE schema_info (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id INTEGER, text_type TEXT, generated_text TEXT,
            model_used TEXT, model_version TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        INSERT INTO schema_info (key, value) VALUES ('version', '22');
        INSERT INTO photos (id, filepath, description, tags)
            VALUES (1, 'a.jpg', 'a dog', '["pet","animal"]'),
                   (2, 'b.jpg', NULL,    NULL);
        INSERT INTO generations (photo_id, text_type, generated_text, model_used)
            VALUES (1, 'tags', '["pet","animal"]', 'llava');
    """)
    conn.commit()
    conn.close()

    from photosearch.db import PhotoDB, SCHEMA_VERSION
    assert SCHEMA_VERSION == 23

    with PhotoDB(str(db_path)) as db:
        # New columns exist and are nullable.
        cols = {row[1] for row in db.conn.execute("PRAGMA table_info(photos)").fetchall()}
        assert {"categories", "visual_tags", "keywords", "tags_v22_backup"} <= cols

        # Pre-migration tag values are snapshotted then nulled.
        row = db.conn.execute(
            "SELECT tags, tags_v22_backup, categories FROM photos WHERE id=1"
        ).fetchone()
        assert row["tags"] is None
        assert json.loads(row["tags_v22_backup"]) == ["pet", "animal"]
        assert row["categories"] is None  # awaits the new pass

        # Legacy generations row got re-tagged.
        text_types = {r[0] for r in db.conn.execute(
            "SELECT DISTINCT text_type FROM generations"
        ).fetchall()}
        assert "category-content-legacy" in text_types
        assert "tags" not in text_types

        # Schema version stamped.
        v = db.conn.execute(
            "SELECT value FROM schema_info WHERE key='version'"
        ).fetchone()[0]
        assert int(v) == 23
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_db.py::test_schema_v23_migration_from_v22 -v`
Expected: FAIL with `assert SCHEMA_VERSION == 23` (currently 22).

### Task 1.2: Implement the migration

**Files:**
- Modify: `photosearch/db.py` — `SCHEMA_VERSION` constant (line 63), `_init_schema` method (line 236).

- [ ] **Step 1: Bump `SCHEMA_VERSION`**

Change `photosearch/db.py:63`:

```python
SCHEMA_VERSION = 23
```

- [ ] **Step 2: Append the v23 migration block**

In `photosearch/db.py`, inside `_init_schema()`, find the last existing migration (the `worker_processed.attempts` block at lines 546–549) and append immediately after it (before any `_ensure_schema_info_table` / version stamping):

```python
        # Schema v23: categories / visual_tags / keywords / tags_v22_backup.
        # The old 78-word `tags` content is incompatible with the new vocab,
        # so we snapshot it into tags_v22_backup, null `tags`, and re-tag
        # historical `generations` rows as 'category-content-legacy' so the
        # provenance log keeps a clean lineage. See
        # docs/plans/categories-keywords-redesign.md for full rationale.
        try:
            cur.execute("SELECT categories FROM photos LIMIT 1")
        except sqlite3.OperationalError:
            # All four ADDs + the data moves run inside the connection's
            # implicit transaction that wraps _init_schema; partial failure
            # rolls back cleanly.
            cur.execute("ALTER TABLE photos ADD COLUMN categories TEXT")
            cur.execute("ALTER TABLE photos ADD COLUMN visual_tags TEXT")
            cur.execute("ALTER TABLE photos ADD COLUMN keywords TEXT")
            cur.execute("ALTER TABLE photos ADD COLUMN tags_v22_backup TEXT")
            cur.execute(
                "UPDATE photos SET tags_v22_backup = tags "
                "WHERE tags IS NOT NULL AND tags != ''"
            )
            cur.execute("UPDATE photos SET tags = NULL")
            cur.execute(
                "UPDATE generations SET text_type = 'category-content-legacy' "
                "WHERE text_type = 'tags'"
            )
```

- [ ] **Step 3: Run the migration test**

Run: `venv/bin/pytest tests/test_db.py::test_schema_v23_migration_from_v22 -v`
Expected: PASS.

- [ ] **Step 4: Run the full db test module to catch regressions**

Run: `venv/bin/pytest tests/test_db.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add photosearch/db.py tests/test_db.py
git commit -m "feat(db): schema v23 — categories/visual_tags/keywords + tags backup"
```

### Task 1.3: Expand `count_unprocessed_photos` and `get_unprocessed_photos` for the new passes

**Files:**
- Modify: `photosearch/db.py` — extend the `describe`/`tags` branches (lines ~1537 / ~1716) to also handle `category-content`, `category-visual`, `keywords`.

- [ ] **Step 1: Add tests**

Append to `tests/test_db.py`:

```python
def _seed_three_described(db):
    db.conn.executemany(
        "INSERT INTO photos (filepath, description) VALUES (?, ?)",
        [("a.jpg", "a dog"), ("b.jpg", "a beach"), ("c.jpg", None)],
    )
    db.conn.commit()


def test_count_unprocessed_category_content_gates_on_description(tmp_path):
    from photosearch.db import PhotoDB
    with PhotoDB(str(tmp_path / "x.db")) as db:
        _seed_three_described(db)
        # Two described, one not — only the described ones are eligible.
        assert db.count_unprocessed_photos("category-content") == 2


def test_count_unprocessed_keywords_gates_on_description(tmp_path):
    from photosearch.db import PhotoDB
    with PhotoDB(str(tmp_path / "x.db")) as db:
        _seed_three_described(db)
        assert db.count_unprocessed_photos("keywords") == 2


def test_count_unprocessed_category_visual_includes_all(tmp_path):
    from photosearch.db import PhotoDB
    with PhotoDB(str(tmp_path / "x.db")) as db:
        _seed_three_described(db)
        # visual pass doesn't require a description.
        assert db.count_unprocessed_photos("category-visual") == 3


def test_mark_processed_blocks_after_max_attempts_category_content(tmp_path):
    from photosearch.db import PhotoDB, MAX_PROCESS_ATTEMPTS
    with PhotoDB(str(tmp_path / "x.db")) as db:
        _seed_three_described(db)
        photo_ids = [r[0] for r in db.conn.execute("SELECT id FROM photos WHERE description IS NOT NULL")]
        for _ in range(MAX_PROCESS_ATTEMPTS):
            db.mark_processed(photo_ids, "category-content")
        assert db.count_unprocessed_photos("category-content") == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_db.py -k category -v`
Expected: 3 FAIL (the count tests) — `count_unprocessed_photos` raises or returns 0 wrong for unknown pass types.

- [ ] **Step 3: Extend `count_unprocessed_photos`**

In `photosearch/db.py`, find the `elif pass_type in ("describe", "tags"):` branch (around line 1716) and change it. Replace that entire branch with:

```python
        elif pass_type in ("describe", "tags", "category-content", "keywords"):
            # All four passes gate on the same condition: photos.<col> IS NULL
            # AND no exhausted worker_processed row exists. The legacy "tags"
            # case stays for historical claim cleanup; new code targets the
            # three v23 columns.
            col = {
                "describe": "description",
                "tags": "tags",
                "category-content": "categories",
                "keywords": "keywords",
            }[pass_type]
            # category-content and keywords require a non-null description.
            extra = ""
            if pass_type in ("category-content", "keywords"):
                extra = " AND description IS NOT NULL"
            if photo_ids:
                placeholders = ",".join("?" * len(photo_ids))
                row = self.conn.execute(
                    f"""SELECT COUNT(*) FROM photos
                        WHERE id IN ({placeholders})
                        AND {col} IS NULL{extra}
                        AND NOT EXISTS (SELECT 1 FROM worker_processed wp
                                        WHERE wp.photo_id = photos.id AND wp.pass_type = ?
                                          AND wp.attempts >= {MAX_PROCESS_ATTEMPTS})""",
                    list(photo_ids) + [pass_type],
                ).fetchone()
            else:
                row = self.conn.execute(
                    f"""SELECT COUNT(*) FROM photos
                        WHERE {col} IS NULL{extra}
                        AND NOT EXISTS (SELECT 1 FROM worker_processed wp
                                        WHERE wp.photo_id = photos.id AND wp.pass_type = ?
                                          AND wp.attempts >= {MAX_PROCESS_ATTEMPTS})""",
                    (pass_type,),
                ).fetchone()
```

Then add a separate branch for `category-visual` (place it next to the others, e.g. directly under the block above):

```python
        elif pass_type == "category-visual":
            # Visual pass has no description dependency.
            if photo_ids:
                placeholders = ",".join("?" * len(photo_ids))
                row = self.conn.execute(
                    f"""SELECT COUNT(*) FROM photos
                        WHERE id IN ({placeholders})
                        AND visual_tags IS NULL
                        AND NOT EXISTS (SELECT 1 FROM worker_processed wp
                                        WHERE wp.photo_id = photos.id AND wp.pass_type = 'category-visual'
                                          AND wp.attempts >= {MAX_PROCESS_ATTEMPTS})""",
                    list(photo_ids),
                ).fetchone()
            else:
                row = self.conn.execute(
                    f"""SELECT COUNT(*) FROM photos
                        WHERE visual_tags IS NULL
                        AND NOT EXISTS (SELECT 1 FROM worker_processed wp
                                        WHERE wp.photo_id = photos.id AND wp.pass_type = 'category-visual'
                                          AND wp.attempts >= {MAX_PROCESS_ATTEMPTS})"""
                ).fetchone()
```

- [ ] **Step 4: Mirror the same expansion in `get_unprocessed_photos`**

Find the same branch shape in `get_unprocessed_photos` (around line 1537+; same `elif pass_type in ("describe", "tags"):` structure) and apply the equivalent changes — the SELECT shape is the same except returning `id, filepath` and respecting `claimed` / `limit + len(claimed)` as the existing code does.

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_db.py -k category -v`
Expected: all 3 category tests + the mark_processed test pass.

- [ ] **Step 6: Run the full db test module**

Run: `venv/bin/pytest tests/test_db.py -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add photosearch/db.py tests/test_db.py
git commit -m "feat(db): category-content/visual/keywords pass types in claim queries"
```

### Task 1.4: Add new pass types to validation

**Files:**
- Modify: `cli.py:3221` — `valid_passes` set in the `worker` command.

- [ ] **Step 1: Update the set**

Find `valid_passes = {"clip", "faces", "quality", "describe", "tags", "verify"}` at `cli.py:3221` and change to:

```python
    valid_passes = {
        "clip", "faces", "quality", "describe", "verify",
        "category-content", "category-visual", "keywords",
    }
    # Note: "tags" is removed. Old clients that still pass --passes tags
    # will be rejected here with a clear error. They should be upgraded.
```

- [ ] **Step 2: Smoke check**

Run: `venv/bin/python cli.py worker --help`
Expected: help text loads without import error.

Run: `venv/bin/python cli.py worker --passes tags --server http://example/ --directory /tmp 2>&1 | head -3`
Expected: `Error: unknown pass type 'tags'. Valid: category-content, category-visual, clip, describe, faces, keywords, quality, verify`

- [ ] **Step 3: Commit**

```bash
git add cli.py
git commit -m "feat(cli): worker valid_passes — add three new pass types, drop tags"
```

---

## Phase 2 — Vocab mining and proposal

**Goal:** Produce `/data/vocab_candidates.json` (mined noun phrases with frequency ≥ 50) and `/data/vocab_proposal.json` (LLM-grouped draft buckets) — the raw inputs for human curation.

### Task 2.1: Add spaCy dependency

**Files:**
- Modify: `requirements.txt`
- Modify: `Dockerfile`

- [ ] **Step 1: Add spaCy to requirements**

In `requirements.txt`, after the existing `numpy` / `pillow` lines, add:

```
spacy>=3.7,<4
```

- [ ] **Step 2: Add the model download to the Dockerfile**

In `Dockerfile`, in the runtime stage (search for `python cli.py` or the final `CMD`), add before any model-cache `RUN` step:

```dockerfile
RUN python -m spacy download en_core_web_sm
```

- [ ] **Step 3: Install locally**

```bash
venv/bin/pip install 'spacy>=3.7,<4'
venv/bin/python -m spacy download en_core_web_sm
```

Expected: no error; `venv/bin/python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('ok')"` prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt Dockerfile
git commit -m "deps: spaCy + en_core_web_sm for vocab mining"
```

### Task 2.2: Mining module (TDD)

**Files:**
- Create: `photosearch/vocab_mining.py`
- Create: `tests/test_vocab_mining.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_vocab_mining.py`:

```python
import pytest


def test_extract_noun_phrases_lemmatizes_and_lowercases():
    from photosearch.vocab_mining import extract_noun_phrases
    out = extract_noun_phrases(
        "Two golden retrievers running on Stinson Beach during sunset."
    )
    # Multi-word phrases preserved; nouns lemmatized; lowercase.
    assert "golden retriever" in out
    assert "stinson beach" in out
    assert "sunset" in out


def test_extract_noun_phrases_skips_stopword_only_chunks():
    from photosearch.vocab_mining import extract_noun_phrases
    out = extract_noun_phrases("The thing in the place looks nice.")
    # 'thing' / 'place' filtered as vague.
    assert "thing" not in out
    assert "place" not in out


def test_mine_corpus_returns_frequency_sorted_filtered_list():
    from photosearch.vocab_mining import mine_corpus
    descriptions = [
        "A dog on the beach.",
        "Two dogs at the beach.",
        "A child on the beach.",
        "A bird in the sky.",
    ]
    # min_count=2 → 'beach' (3) and 'dog' (2) qualify; 'bird' (1), 'sky' (1), 'child' (1) drop.
    out = mine_corpus(descriptions, min_count=2)
    terms = [row["term"] for row in out]
    assert terms[0] == "beach"
    assert "dog" in terms
    assert "bird" not in terms
    # Frequency monotone non-increasing.
    counts = [row["count"] for row in out]
    assert counts == sorted(counts, reverse=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_vocab_mining.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `photosearch/vocab_mining.py`**

```python
"""Mine candidate vocabulary terms from photo descriptions.

Uses spaCy noun-chunk extraction + lemmatization. Designed to be run once
per vocab refresh; output is reviewed in /admin/vocab (Phase 3).
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

_VAGUE_WORDS = {
    "thing", "things", "place", "stuff", "scene", "image", "photo",
    "picture", "view", "side", "part", "area", "background",
}


def _get_nlp():
    import spacy
    # Reuse a single instance per process; spaCy load is expensive (~2s).
    global _NLP
    try:
        return _NLP  # type: ignore[name-defined]
    except NameError:
        pass
    _NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    # Noun-chunk extraction requires the parser; re-enable lightly.
    _NLP.enable_pipe("parser") if "parser" in _NLP.disabled else None
    return _NLP


def extract_noun_phrases(text: str) -> list[str]:
    """Return lemmatized, lowercased noun chunks from one description."""
    if not text:
        return []
    nlp = _get_nlp()
    doc = nlp(text)
    seen: set[str] = set()
    out: list[str] = []
    for chunk in doc.noun_chunks:
        toks = [t for t in chunk if not t.is_stop and not t.is_punct]
        if not toks:
            continue
        if all(t.lemma_.lower() in _VAGUE_WORDS for t in toks):
            continue
        phrase = " ".join(t.lemma_.lower() for t in toks)
        if not phrase or phrase in _VAGUE_WORDS:
            continue
        if phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
    return out


def mine_corpus(descriptions: Iterable[str], min_count: int = 50) -> list[dict]:
    """Return [{term, count}, ...] sorted by count desc, count >= min_count."""
    counter: Counter[str] = Counter()
    for desc in descriptions:
        for phrase in extract_noun_phrases(desc):
            counter[phrase] += 1
    return [
        {"term": term, "count": count}
        for term, count in counter.most_common()
        if count >= min_count
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_vocab_mining.py -v`
Expected: 3 passed. (May be slow first time as spaCy loads the model — ~3-5s.)

- [ ] **Step 5: Commit**

```bash
git add photosearch/vocab_mining.py tests/test_vocab_mining.py
git commit -m "feat(vocab): noun-phrase mining via spaCy"
```

### Task 2.3: `mine-vocab` CLI

**Files:**
- Modify: `cli.py`

- [ ] **Step 1: Implement the command**

Append to `cli.py` (place near `bakeoff-keywords`):

```python
@cli.command("mine-vocab")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB")
@click.option("--out", default="/data/vocab_candidates.json", show_default=True)
@click.option("--min-count", default=50, show_default=True,
              help="Drop phrases that occur fewer than N times.")
@click.option("--limit", default=0,
              help="Optional cap on descriptions processed (0 = all).")
def mine_vocab(db, out, min_count, limit):
    """Mine noun-phrase candidates from photo descriptions."""
    import json
    import time
    from photosearch.db import PhotoDB
    from photosearch.vocab_mining import mine_corpus

    with PhotoDB(db) as photo_db:
        rows = photo_db.conn.execute(
            "SELECT description FROM photos "
            "WHERE description IS NOT NULL AND length(description) > 20"
        ).fetchall()
    descriptions = [r["description"] for r in rows]
    if limit:
        descriptions = descriptions[:limit]

    click.echo(f"Mining noun phrases from {len(descriptions)} descriptions...")
    t0 = time.time()
    candidates = mine_corpus(descriptions, min_count=min_count)
    elapsed = time.time() - t0

    payload = {
        "source_count": len(descriptions),
        "min_count": min_count,
        "elapsed_s": round(elapsed, 1),
        "candidates": candidates,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    click.echo(f"Wrote {len(candidates)} terms (count >= {min_count}) to {out}")
    click.echo(f"Top 20: {', '.join(c['term'] for c in candidates[:20])}")
```

- [ ] **Step 2: Smoke-check**

Run: `venv/bin/python cli.py mine-vocab --help`
Expected: help text.

- [ ] **Step 3: Commit**

```bash
git add cli.py
git commit -m "feat(cli): mine-vocab — extract noun-phrase candidates"
```

### Task 2.4: Vocab grouping module + CLI

**Files:**
- Create: `photosearch/vocab_grouping.py`
- Modify: `cli.py`

- [ ] **Step 1: Implement the grouping module**

Create `photosearch/vocab_grouping.py`:

```python
"""LLM-based clustering of candidate vocab into semantic buckets.

Reads a flat candidate list (output of mine-vocab) and asks Llama to
sort terms into draft buckets. The candidate list won't fit in one
context window for 4k+ terms, so we chunk by frequency band.
"""
from __future__ import annotations

import json
from typing import Optional

_GROUP_PROMPT = """\
You are organising photo-description vocabulary. Sort these terms into 6-12 semantic buckets like:
animals, people, activities, landscapes, weather, food, vehicles, architecture, mood, photography, miscellaneous.

Rules:
- Return ONLY valid JSON: {{"bucket_name": ["term1", "term2"], ...}}
- Every input term must appear in exactly one bucket.
- Use existing bucket names when terms fit; only invent new ones if necessary.

Terms (one per line):
{terms}

Existing buckets so far (merge into these where possible):
{existing}
"""


def group_terms(
    terms: list[str],
    chunk_size: int = 200,
    model: str = "llama3.2:3b",
    chat_fn=None,
) -> dict[str, list[str]]:
    """Group `terms` into semantic buckets via repeated LLM calls.

    chat_fn is the Ollama chat callable; defaults to describe._ollama_chat_with_retry.
    Injectable so tests can pass a deterministic stub.
    """
    if chat_fn is None:
        from photosearch.describe import _ollama_chat_with_retry as chat_fn  # type: ignore

    grouped: dict[str, list[str]] = {}
    for start in range(0, len(terms), chunk_size):
        chunk = terms[start : start + chunk_size]
        existing_summary = ", ".join(
            f"{k} ({len(v)})" for k, v in grouped.items()
        ) or "(none yet)"
        prompt = _GROUP_PROMPT.format(
            terms="\n".join(chunk),
            existing=existing_summary,
        )
        raw = chat_fn(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        parsed = _parse_grouping_json(raw, fallback_terms=chunk)
        for bucket, items in parsed.items():
            grouped.setdefault(bucket, []).extend(items)
    return grouped


def _parse_grouping_json(raw: Optional[str], fallback_terms: list[str]) -> dict[str, list[str]]:
    """Tolerantly parse the LLM's JSON; on failure, dump terms into a fallback bucket."""
    if not raw:
        return {"unsorted": list(fallback_terms)}
    # Strip code fences if present.
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {"unsorted": list(fallback_terms)}
    if not isinstance(obj, dict):
        return {"unsorted": list(fallback_terms)}
    out: dict[str, list[str]] = {}
    for k, v in obj.items():
        if isinstance(v, list):
            out[str(k)] = [str(t).strip().lower() for t in v if str(t).strip()]
    return out or {"unsorted": list(fallback_terms)}
```

- [ ] **Step 2: Add tests**

Append to `tests/test_vocab_mining.py`:

```python
def test_group_terms_uses_stub_chat_and_aggregates():
    from photosearch.vocab_grouping import group_terms

    def fake_chat(model, messages, options):
        # Pretend the LLM split the chunk evenly.
        prompt = messages[0]["content"]
        terms_block = prompt.split("Terms (one per line):\n", 1)[1].split("\n\n", 1)[0]
        terms = [t for t in terms_block.splitlines() if t]
        half = len(terms) // 2 or 1
        return '{"animals": ' + str(terms[:half]).replace("'", '"') + \
               ', "landscapes": ' + str(terms[half:]).replace("'", '"') + "}"

    grouped = group_terms(["dog", "cat", "beach", "mountain"], chunk_size=4, chat_fn=fake_chat)
    assert sorted(grouped["animals"]) == ["cat", "dog"]
    assert sorted(grouped["landscapes"]) == ["beach", "mountain"]


def test_group_terms_chunks_and_passes_existing_summary():
    from photosearch.vocab_grouping import group_terms
    calls = []

    def fake_chat(model, messages, options):
        calls.append(messages[0]["content"])
        return '{"misc": ["x"]}'

    group_terms(["a", "b", "c", "d"], chunk_size=2, chat_fn=fake_chat)
    assert len(calls) == 2
    assert "(none yet)" in calls[0]
    assert "misc" in calls[1]


def test_group_terms_falls_back_to_unsorted_on_bad_json():
    from photosearch.vocab_grouping import group_terms

    def bad_chat(model, messages, options):
        return "not json at all"

    grouped = group_terms(["dog", "cat"], chunk_size=10, chat_fn=bad_chat)
    assert grouped == {"unsorted": ["dog", "cat"]}
```

- [ ] **Step 3: Run tests**

Run: `venv/bin/pytest tests/test_vocab_mining.py -v`
Expected: 6 passed total.

- [ ] **Step 4: Add `group-vocab` CLI**

Append to `cli.py`:

```python
@cli.command("group-vocab")
@click.option("--in", "in_path", default="/data/vocab_candidates.json",
              show_default=True, help="Output of mine-vocab.")
@click.option("--out", default="/data/vocab_proposal.json", show_default=True)
@click.option("--model", default="llama3.2:3b", show_default=True)
@click.option("--chunk-size", default=200, show_default=True)
def group_vocab(in_path, out, model, chunk_size):
    """Group mined candidates into semantic buckets via LLM."""
    import json
    from photosearch.vocab_grouping import group_terms

    with open(in_path) as f:
        payload = json.load(f)
    terms = [c["term"] for c in payload["candidates"]]
    click.echo(f"Grouping {len(terms)} terms via {model} (chunk size {chunk_size})...")
    grouped = group_terms(terms, chunk_size=chunk_size, model=model)
    with open(out, "w") as f:
        json.dump({
            "source_path": in_path,
            "model": model,
            "buckets": grouped,
        }, f, indent=2)
    click.echo(f"Wrote {len(grouped)} buckets to {out}")
    for bucket, items in grouped.items():
        click.echo(f"  {bucket}: {len(items)} terms")
```

- [ ] **Step 5: Smoke-check**

Run: `venv/bin/python cli.py group-vocab --help`
Expected: help text.

- [ ] **Step 6: Commit**

```bash
git add photosearch/vocab_grouping.py cli.py tests/test_vocab_mining.py
git commit -m "feat(vocab): LLM-based candidate grouping + group-vocab CLI"
```

### Task 2.5: Run mining + grouping on the live library (operational)

- [ ] **Step 1: Run mining on the NAS**

```bash
DC="docker compose -f docker-compose.nas.yml run --rm"
$DC photosearch mine-vocab --min-count 50 --out /data/vocab_candidates.json
```

Expected: "Wrote N terms (count >= 50) to /data/vocab_candidates.json" with 2,000-4,000 terms per design.

- [ ] **Step 2: Run grouping**

```bash
$DC photosearch group-vocab --in /data/vocab_candidates.json --out /data/vocab_proposal.json
```

Expected: 6-12 buckets, sum of bucket sizes ≈ candidate count.

- [ ] **Step 3: Sanity-check by eye**

```bash
ssh nas.local 'sudo cat /volume1/docker/photosearch/data/vocab_proposal.json' | venv/bin/python -m json.tool | head -60
```

Look for: reasonable bucket names; terms placed sensibly; "unsorted" bucket size small (indicates LLM JSON parsing worked).

No commit — these are data files in `/data/`, intentionally not committed to the repo.

---

## Phase 3 — Vocab curator (API + UI)

**Goal:** Build the `/admin/vocab` page so a human can edit the proposal interactively, preview coverage, and compile the three Python vocab modules. Curated `vocab_content.py` / `vocab_visual.py` / `vocab_query_expansion.py` land in the repo at the end.

### Task 3.1: Compile module (TDD)

**Files:**
- Create: `photosearch/vocab_compile.py`
- Create: `tests/test_vocab_compile.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_vocab_compile.py`:

```python
import pytest


def test_compile_renders_content_module(tmp_path):
    from photosearch.vocab_compile import render_content_module
    src = render_content_module(
        terms=["beach", "mountain", "golden retriever"],
        draft_hash="abc123",
        timestamp="2026-05-17T18:00:00",
    )
    # Module is importable Python.
    namespace = {}
    exec(src, namespace)
    assert namespace["CONTENT_VOCABULARY"] == ["beach", "golden retriever", "mountain"]
    # Header comment present with metadata.
    assert "abc123" in src
    assert "2026-05-17" in src


def test_compile_renders_query_expansion_module(tmp_path):
    from photosearch.vocab_compile import render_query_expansion_module
    src = render_query_expansion_module(
        expansions={"dog": ["puppy", "pup"], "ocean": ["sea", "pacific ocean"]},
        draft_hash="def456",
        timestamp="2026-05-17T18:00:00",
    )
    namespace = {}
    exec(src, namespace)
    assert namespace["_QUERY_TO_CATEGORIES"] == {
        "dog": {"puppy", "pup"},
        "ocean": {"sea", "pacific ocean"},
    }


def test_compile_validates_disjoint_vocabs():
    from photosearch.vocab_compile import validate_draft, VocabError
    draft = {
        "content": ["beach", "mountain"],
        "visual": ["beach", "dramatic"],  # 'beach' in both → error
        "expansions": {},
    }
    with pytest.raises(VocabError, match="appear in both"):
        validate_draft(draft)


def test_compile_validates_content_floor():
    from photosearch.vocab_compile import validate_draft, VocabError
    draft = {
        "content": ["beach", "mountain"],  # < 50 terms
        "visual": ["dramatic"],
        "expansions": {},
    }
    with pytest.raises(VocabError, match="at least 50"):
        validate_draft(draft)


def test_compile_validates_nonempty_visual():
    from photosearch.vocab_compile import validate_draft, VocabError
    draft = {
        "content": ["x"] * 60,
        "visual": [],
        "expansions": {},
    }
    with pytest.raises(VocabError, match="visual vocab is empty"):
        validate_draft(draft)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_vocab_compile.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `photosearch/vocab_compile.py`**

```python
"""Compile a curated vocab draft into three importable Python modules.

The draft shape:
    {
        "content": [str, ...],        # ≥ 50 entries
        "visual":  [str, ...],        # ≥ 1 entry
        "expansions": {str: [str, ...], ...},   # query → category aliases
    }

Validation runs before any file write. The generated modules carry a
header comment with the draft hash + timestamp so the deployed vocab
is traceable back to the curation session that produced it.
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone


class VocabError(ValueError):
    """Raised when a draft fails validation."""


def validate_draft(draft: dict) -> None:
    content = [t for t in draft.get("content", []) if t]
    visual = [t for t in draft.get("visual", []) if t]
    if len(content) < 50:
        raise VocabError(
            f"Content vocab must have at least 50 terms; got {len(content)}."
        )
    if not visual:
        raise VocabError("visual vocab is empty; add mood/light/composition terms.")
    overlap = set(content) & set(visual)
    if overlap:
        raise VocabError(
            f"Terms appear in both content and visual vocab: {sorted(overlap)[:5]}"
        )


def draft_hash(draft: dict) -> str:
    serialised = json.dumps(draft, sort_keys=True).encode()
    return hashlib.sha256(serialised).hexdigest()[:12]


def _header(kind: str, draft_hash_: str, timestamp: str) -> str:
    return (
        f'"""GENERATED FILE — do not edit by hand.\n\n'
        f"Kind: {kind}\n"
        f"Draft hash: {draft_hash_}\n"
        f"Generated at: {timestamp}\n\n"
        f"Source: /admin/vocab curator (see docs/plans/categories-keywords-redesign.md).\n"
        f"Regenerate via POST /api/admin/vocab/compile or `photosearch compile-vocab`.\n"
        f'"""\n\n'
    )


def render_content_module(terms: list[str], draft_hash: str, timestamp: str) -> str:
    body = ",\n    ".join(repr(t) for t in sorted(set(terms)))
    return (
        _header("content vocabulary", draft_hash, timestamp)
        + f"CONTENT_VOCABULARY: list[str] = [\n    {body},\n]\n"
    )


def render_visual_module(terms: list[str], draft_hash: str, timestamp: str) -> str:
    body = ",\n    ".join(repr(t) for t in sorted(set(terms)))
    return (
        _header("visual vocabulary", draft_hash, timestamp)
        + f"VISUAL_VOCABULARY: list[str] = [\n    {body},\n]\n"
    )


def render_query_expansion_module(
    expansions: dict[str, list[str]],
    draft_hash: str,
    timestamp: str,
) -> str:
    items = []
    for query, cats in sorted(expansions.items()):
        cat_set = "{" + ", ".join(repr(c) for c in sorted(set(cats))) + "}"
        items.append(f"    {query!r}: {cat_set},")
    body = "\n".join(items)
    return (
        _header("query → categories expansion", draft_hash, timestamp)
        + f"_QUERY_TO_CATEGORIES: dict[str, set[str]] = {{\n{body}\n}}\n"
    )


def compile_draft(
    draft: dict,
    repo_dir: str,
) -> dict:
    """Validate, render, and write the three modules. Returns metadata."""
    validate_draft(draft)
    h = draft_hash(draft)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    files = {
        f"{repo_dir}/photosearch/vocab_content.py":
            render_content_module(draft["content"], h, ts),
        f"{repo_dir}/photosearch/vocab_visual.py":
            render_visual_module(draft["visual"], h, ts),
        f"{repo_dir}/photosearch/vocab_query_expansion.py":
            render_query_expansion_module(draft.get("expansions", {}), h, ts),
    }
    for path, src in files.items():
        with open(path, "w") as f:
            f.write(src)
    return {"draft_hash": h, "timestamp": ts, "files": list(files.keys())}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_vocab_compile.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add photosearch/vocab_compile.py tests/test_vocab_compile.py
git commit -m "feat(vocab): compile/validate draft into three Python modules"
```

### Task 3.2: Admin vocab API router

**Files:**
- Create: `photosearch/vocab_admin.py`
- Create: `tests/test_web_vocab.py`
- Modify: `photosearch/web.py` — mount the new router.

- [ ] **Step 1: Write failing API tests**

Create `tests/test_web_vocab.py`:

```python
import json
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_data(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_DB", str(tmp_path / "x.db"))
    # /data dir for json files.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("PHOTOSEARCH_DATA_DIR", str(data_dir))

    # Seed candidates file.
    (data_dir / "vocab_candidates.json").write_text(json.dumps({
        "source_count": 3,
        "min_count": 1,
        "candidates": [{"term": "beach", "count": 3}, {"term": "dog", "count": 2}],
    }))

    # Seed a couple of described photos for coverage preview.
    from photosearch.db import PhotoDB
    with PhotoDB(str(tmp_path / "x.db")) as db:
        db.conn.executemany(
            "INSERT INTO photos (filepath, description) VALUES (?, ?)",
            [("a.jpg", "a dog at the beach"), ("b.jpg", "an empty mountain road")],
        )
        db.conn.commit()

    from photosearch.web import app
    return TestClient(app)


def test_get_candidates_returns_seeded_file(client_with_data):
    r = client_with_data.get("/api/admin/vocab/candidates")
    assert r.status_code == 200
    body = r.json()
    assert body["candidates"][0]["term"] == "beach"


def test_get_draft_returns_empty_when_missing(client_with_data):
    r = client_with_data.get("/api/admin/vocab/draft")
    assert r.status_code == 200
    assert r.json() == {"content": [], "visual": [], "expansions": {}}


def test_put_draft_persists_then_reads_back(client_with_data):
    draft = {
        "content": ["beach", "dog"],
        "visual": ["dramatic"],
        "expansions": {"sea": ["beach"]},
    }
    r = client_with_data.put("/api/admin/vocab/draft", json=draft)
    assert r.status_code == 200
    r2 = client_with_data.get("/api/admin/vocab/draft")
    assert r2.json() == draft


def test_coverage_preview_uses_draft_against_descriptions(client_with_data):
    # Both photos contain at least one content term.
    draft = {"content": ["beach", "mountain"], "visual": [], "expansions": {}}
    r = client_with_data.post("/api/admin/vocab/coverage-preview",
                              json={"draft": draft, "sample_size": 10})
    body = r.json()
    assert body["sample_size"] == 2
    assert body["covered_count"] == 2
    assert body["coverage_pct"] == 100.0


def test_test_photo_returns_matched_terms(client_with_data):
    draft = {"content": ["beach", "dog"], "visual": [], "expansions": {}}
    r = client_with_data.post("/api/admin/vocab/test-photo/1", json={"draft": draft})
    body = r.json()
    assert set(body["matched_categories"]) == {"beach", "dog"}


def test_compile_rejects_empty_visual(client_with_data, tmp_path):
    draft = {"content": ["x"] * 60, "visual": [], "expansions": {}}
    r = client_with_data.post("/api/admin/vocab/compile",
                              json={"draft": draft, "repo_dir": str(tmp_path)})
    assert r.status_code == 400
    assert "visual" in r.json()["detail"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_web_vocab.py -v`
Expected: FAIL — endpoints don't exist.

- [ ] **Step 3: Implement `photosearch/vocab_admin.py`**

```python
"""Admin curator API for the categories/visual/keywords vocab.

Endpoints under /api/admin/vocab/* — used by frontend/dist/admin_vocab.html.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Body

from .vocab_compile import compile_draft, VocabError


router = APIRouter(prefix="/api/admin/vocab", tags=["admin-vocab"])


def _data_dir() -> Path:
    return Path(os.environ.get("PHOTOSEARCH_DATA_DIR", "/data"))


def _candidates_path() -> Path:
    return _data_dir() / "vocab_candidates.json"


def _draft_path() -> Path:
    return _data_dir() / "vocab_draft.json"


def _get_db():
    # Local import avoids circulars at module import time.
    from .web import _get_db as web_get_db
    return web_get_db()


@router.get("/candidates")
def get_candidates():
    path = _candidates_path()
    if not path.exists():
        raise HTTPException(404, "No vocab_candidates.json. Run `photosearch mine-vocab` first.")
    with path.open() as f:
        return json.load(f)


@router.get("/draft")
def get_draft():
    path = _draft_path()
    if not path.exists():
        return {"content": [], "visual": [], "expansions": {}}
    with path.open() as f:
        return json.load(f)


@router.put("/draft")
def put_draft(draft: dict = Body(...)):
    # Coerce / validate shape; don't enforce vocab-size rules here, that's compile's job.
    content = list(draft.get("content", []))
    visual = list(draft.get("visual", []))
    expansions = dict(draft.get("expansions", {}))
    payload = {"content": content, "visual": visual, "expansions": expansions}
    with _draft_path().open("w") as f:
        json.dump(payload, f, indent=2)
    return {"ok": True, "content_count": len(content), "visual_count": len(visual)}


def _photo_terms(description: Optional[str], terms: list[str]) -> list[str]:
    """Match terms against a description (case-insensitive whole-word)."""
    if not description:
        return []
    text = description.lower()
    matched = []
    for term in terms:
        pat = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pat, text):
            matched.append(term)
    return matched


@router.post("/coverage-preview")
def coverage_preview(payload: dict = Body(...)):
    """Sample N described photos; return % that get ≥1 category from the draft."""
    draft = payload.get("draft") or {}
    sample_size = int(payload.get("sample_size") or 1000)
    content_terms = list(draft.get("content", []))

    with _get_db() as db:
        rows = db.conn.execute(
            "SELECT id, description FROM photos "
            "WHERE description IS NOT NULL "
            "ORDER BY RANDOM() LIMIT ?",
            (sample_size,),
        ).fetchall()

    covered = 0
    samples_uncovered = []
    for row in rows:
        if _photo_terms(row["description"], content_terms):
            covered += 1
        elif len(samples_uncovered) < 10:
            samples_uncovered.append({"id": row["id"], "description": row["description"][:140]})

    actual = len(rows)
    pct = round(100.0 * covered / actual, 1) if actual else 0.0
    return {
        "sample_size": actual,
        "covered_count": covered,
        "coverage_pct": pct,
        "samples_uncovered": samples_uncovered,
    }


@router.post("/test-photo/{photo_id}")
def test_photo(photo_id: int, payload: dict = Body(...)):
    draft = payload.get("draft") or {}
    with _get_db() as db:
        row = db.conn.execute(
            "SELECT description FROM photos WHERE id = ?", (photo_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, f"Photo {photo_id} not found.")
    desc = row["description"]
    return {
        "photo_id": photo_id,
        "description": desc,
        "matched_categories": _photo_terms(desc, draft.get("content", [])),
        "matched_visual": _photo_terms(desc, draft.get("visual", [])),
    }


@router.post("/compile")
def compile_(payload: dict = Body(...)):
    draft = payload.get("draft") or {}
    repo_dir = payload.get("repo_dir") or os.environ.get("PHOTOSEARCH_REPO_DIR", "/repo")
    try:
        result = compile_draft(draft, repo_dir=repo_dir)
    except VocabError as exc:
        raise HTTPException(400, str(exc))
    return result
```

- [ ] **Step 4: Mount the router in `web.py`**

In `photosearch/web.py`, find the line where the FastAPI app is created (search for `app = FastAPI(`). After all existing `app.include_router(...)` lines (or after the existing router imports), add:

```python
from .vocab_admin import router as vocab_admin_router  # noqa: E402
app.include_router(vocab_admin_router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_web_vocab.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add photosearch/vocab_admin.py photosearch/web.py tests/test_web_vocab.py
git commit -m "feat(api): admin/vocab curator endpoints"
```

### Task 3.3: Curator frontend page

**Files:**
- Create: `frontend/dist/admin_vocab.html`
- Modify: `frontend/dist/shared.js` — add `'admin'` to `PS.SharedHeader` navLinks.
- Modify: `photosearch/web.py` — serve `/admin/vocab`.

- [ ] **Step 1: Add the route in `web.py`**

Find the block of `@app.get("/<page>")` routes (e.g. `/faces`, `/map`, `/geotag`). After the last one, append:

```python
@app.get("/admin/vocab", response_class=HTMLResponse)
async def admin_vocab_page():
    return FileResponse("frontend/dist/admin_vocab.html",
                        headers={"Cache-Control": "no-cache"})
```

- [ ] **Step 2: Add the Admin nav entry to `shared.js`**

In `frontend/dist/shared.js`, find `PS.SharedHeader`'s `navLinks` array. Append:

```javascript
    { id: "admin", href: "/admin/vocab", label: "Admin" },
```

(`activePage: 'admin'` is the marker pages will pass.)

- [ ] **Step 3: Create the curator page**

Create `frontend/dist/admin_vocab.html` with this content:

```html
<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Vocab Curator</title>
<script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<script src="/shared.js"></script>
<style>
  body { font: 14px/1.4 system-ui, sans-serif; margin: 0; background: #f5f5f7; }
  .layout { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; min-height: calc(100vh - 60px); }
  .pane { background: white; border-radius: 8px; padding: 12px; overflow-y: auto; max-height: calc(100vh - 100px); }
  .coverage-bar { background: #eee; height: 18px; border-radius: 9px; overflow: hidden; }
  .coverage-fill { background: #36a36a; height: 100%; transition: width 0.3s; }
  .bucket { margin-bottom: 12px; }
  .bucket h4 { margin: 6px 0; font-size: 13px; color: #555; text-transform: uppercase; }
  .term { display: inline-block; padding: 3px 8px; margin: 2px; border: 1px solid #ccc;
          border-radius: 12px; cursor: pointer; user-select: none; background: #fff; }
  .term.included-content { background: #d8e9ff; border-color: #5994f0; }
  .term.included-visual  { background: #ffe0d8; border-color: #f08259; }
  .term .count { color: #888; font-size: 11px; margin-left: 4px; }
  .tabs { display: flex; gap: 8px; margin-bottom: 8px; }
  .tab { padding: 4px 10px; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; }
  .tab.active { background: #333; color: white; }
  .toolbar { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; }
  button { padding: 6px 12px; cursor: pointer; }
  .filter-input { flex: 1; padding: 4px 8px; }
  .status-line { padding: 6px 0; color: #666; }
  .error { color: #c00; font-weight: 600; }
  .test-photo-input { padding: 4px 8px; }
</style>
</head><body>
<div id="header-root"></div>
<div class="layout">
  <div class="pane" id="left-pane"></div>
  <div class="pane" id="right-pane"></div>
</div>

<script>
const { useState, useEffect, useMemo, createElement: h } = React;

// -- API helpers ----------------------------------------------------
const api = {
  candidates: () => fetch('/api/admin/vocab/candidates').then(r => r.ok ? r.json() : null),
  getDraft:   () => fetch('/api/admin/vocab/draft').then(r => r.json()),
  putDraft:   d  => fetch('/api/admin/vocab/draft', {method: 'PUT', headers: {'Content-Type':'application/json'}, body: JSON.stringify(d)}).then(r => r.json()),
  coverage:   (draft, n) => fetch('/api/admin/vocab/coverage-preview', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({draft, sample_size: n})}).then(r => r.json()),
  compile:    draft => fetch('/api/admin/vocab/compile', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({draft})}).then(async r => { const b = await r.json(); if (!r.ok) throw new Error(b.detail || 'compile failed'); return b; }),
  testPhoto:  (id, draft) => fetch('/api/admin/vocab/test-photo/' + id, {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({draft})}).then(r => r.json()),
};

// -- Coverage preview -----------------------------------------------
function CoveragePreview({ draft }) {
  const [pct, setPct] = useState(null);
  const [n, setN] = useState(0);
  const [loading, setLoading] = useState(false);
  const run = async () => {
    setLoading(true);
    const r = await api.coverage(draft, 1000);
    setPct(r.coverage_pct);
    setN(r.sample_size);
    setLoading(false);
  };
  return h('div', { style: { padding: '6px 0' } },
    h('div', { className: 'toolbar' },
      h('button', { onClick: run, disabled: loading }, loading ? 'Sampling...' : 'Preview coverage (1000 sample)'),
      pct !== null && h('span', { style: { fontWeight: 600 } }, pct + '% of ' + n + ' covered'),
    ),
    pct !== null && h('div', { className: 'coverage-bar' },
      h('div', { className: 'coverage-fill', style: { width: pct + '%' } })),
  );
}

// -- Left pane: candidate list grouped by bucket --------------------
function Candidates({ candidates, draft, onToggle, filter }) {
  // Candidate file is flat (term + count); buckets come from the LLM-grouped
  // proposal (vocab_proposal.json). For the curator we treat each candidate
  // as its own row and let the user filter/search.
  const visible = candidates.filter(c =>
    !filter || c.term.toLowerCase().includes(filter.toLowerCase())
  );
  const contentSet = useMemo(() => new Set(draft.content), [draft.content]);
  const visualSet  = useMemo(() => new Set(draft.visual),  [draft.visual]);
  return h('div', null,
    h('div', { className: 'status-line' }, visible.length + ' / ' + candidates.length + ' candidates'),
    visible.map(c => {
      const cls = contentSet.has(c.term) ? 'term included-content'
                : visualSet.has(c.term)  ? 'term included-visual'
                : 'term';
      return h('span', {
        key: c.term, className: cls,
        title: 'click: toggle content | shift-click: toggle visual',
        onClick: e => onToggle(c.term, e.shiftKey ? 'visual' : 'content'),
      }, c.term, h('span', { className: 'count' }, c.count));
    })
  );
}

// -- Right pane: draft state ----------------------------------------
function DraftEditor({ draft, setDraft, saveDraft, lastSaved, compileStatus, onCompile }) {
  const [tab, setTab] = useState('content');
  const [testId, setTestId] = useState('');
  const [testResult, setTestResult] = useState(null);
  const list = draft[tab] || [];
  const removeTerm = term => {
    setDraft({ ...draft, [tab]: list.filter(t => t !== term) });
  };
  const runTest = async () => {
    if (!testId) return;
    const r = await api.testPhoto(testId, draft);
    setTestResult(r);
  };
  return h('div', null,
    h(CoveragePreview, { draft }),
    h('div', { className: 'tabs' },
      ['content', 'visual'].map(t =>
        h('div', { key: t, className: 'tab' + (tab === t ? ' active' : ''),
                   onClick: () => setTab(t) }, t + ' (' + (draft[t] || []).length + ')')),
    ),
    h('div', null,
      list.map(term => h('span', {
        key: term, className: 'term included-' + tab,
        onClick: () => removeTerm(term),
        title: 'click to remove'
      }, term))
    ),
    h('div', { className: 'toolbar', style: { marginTop: 16 } },
      h('button', { onClick: saveDraft }, 'Save draft'),
      lastSaved && h('span', { className: 'status-line' }, 'saved ' + lastSaved),
    ),
    h('div', { className: 'toolbar' },
      h('button', { onClick: onCompile }, 'Compile vocab modules'),
      compileStatus && h('span', { className: compileStatus.error ? 'error' : 'status-line' },
        compileStatus.error || ('compiled ' + (compileStatus.draft_hash || ''))),
    ),
    h('div', { className: 'toolbar', style: { marginTop: 16 } },
      h('input', {
        className: 'test-photo-input', placeholder: 'Photo ID to test',
        value: testId, onChange: e => setTestId(e.target.value),
      }),
      h('button', { onClick: runTest }, 'Test on photo'),
    ),
    testResult && h('div', { className: 'status-line' },
      h('div', null, h('em', null, (testResult.description || '').slice(0, 240))),
      h('div', null, 'categories: ', (testResult.matched_categories || []).join(', ') || '(none)'),
      h('div', null, 'visual: ',     (testResult.matched_visual || []).join(', ') || '(none)'),
    ),
  );
}

// -- Root app -------------------------------------------------------
function App() {
  const [candidates, setCandidates] = useState([]);
  const [draft, setDraft] = useState({ content: [], visual: [], expansions: {} });
  const [filter, setFilter] = useState('');
  const [lastSaved, setLastSaved] = useState(null);
  const [compileStatus, setCompileStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.candidates().then(r => {
      if (!r) { setError('No vocab_candidates.json. Run `photosearch mine-vocab` on the NAS first.'); return; }
      setCandidates(r.candidates || []);
    });
    api.getDraft().then(setDraft);
  }, []);

  const onToggle = (term, kind) => {
    setDraft(d => {
      const list = new Set(d[kind] || []);
      if (list.has(term)) list.delete(term); else list.add(term);
      // Toggle removes from the other list to enforce disjointness.
      const otherKey = kind === 'content' ? 'visual' : 'content';
      const other = (d[otherKey] || []).filter(t => t !== term);
      return { ...d, [kind]: Array.from(list), [otherKey]: other };
    });
  };

  const saveDraft = async () => {
    await api.putDraft(draft);
    setLastSaved(new Date().toLocaleTimeString());
  };

  const onCompile = async () => {
    setCompileStatus(null);
    try {
      const r = await api.compile(draft);
      setCompileStatus(r);
    } catch (e) {
      setCompileStatus({ error: e.message });
    }
  };

  const header = h(PS.SharedHeader, { activePage: 'admin' });

  return [
    h('div', { className: 'toolbar', style: { padding: '8px 16px', background: 'white' } },
      h('input', { className: 'filter-input', placeholder: 'Filter candidates...',
                   value: filter, onChange: e => setFilter(e.target.value) }),
      error && h('span', { className: 'error' }, error),
    ),
    h('div', { className: 'layout' },
      h('div', { className: 'pane' }, h(Candidates, { candidates, draft, onToggle, filter })),
      h('div', { className: 'pane' }, h(DraftEditor, {
        draft, setDraft, saveDraft, lastSaved, compileStatus, onCompile,
      })),
    ),
  ];
}

ReactDOM.createRoot(document.getElementById('header-root')).render(
  h(PS.SharedHeader, { activePage: 'admin' })
);
ReactDOM.createRoot(document.querySelector('.layout').parentNode).render(h(App));
</script>
</body></html>
```

(Note: the page is intentionally lean — there is no LLM-bucket grouping pane in v1; users filter and toggle one term at a time. The proposal JSON is informational background; if Step 3.6 shows it's slow to curate without bucket headers, add them as a follow-up.)

- [ ] **Step 4: Rebuild + restart the container, then load `/admin/vocab` in a browser**

Run:
```bash
docker compose -f docker-compose.nas.yml build photosearch
docker compose -f docker-compose.nas.yml up -d photosearch
```

Open `http://<NAS-IP>:8000/admin/vocab`. Expected:
- Header has an "Admin" link highlighted.
- Left pane shows candidate terms with frequencies (assuming Phase 2 was run).
- Filter input narrows the list.
- Click adds to content (blue); shift-click moves to visual (orange).
- "Preview coverage" shows a green bar with %.
- "Save draft" persists; refresh → draft survives.
- "Compile" with a tiny draft errors with "at least 50 content terms".

- [ ] **Step 5: Commit**

```bash
git add frontend/dist/admin_vocab.html frontend/dist/shared.js photosearch/web.py
git commit -m "feat(ui): /admin/vocab curator page"
```

### Task 3.4: Curate the vocab (operational)

- [ ] **Step 1: Use `/admin/vocab` to build the curated vocab**

Time budget per spec: 1–2 hours.

Working in the UI:
- Toggle ~150–250 content terms (animals, landscapes, activities, objects, etc.).
- Toggle ~20–30 visual terms (dramatic, peaceful, foggy, silhouette, close-up, aerial, etc.).
- Click "Preview coverage" — iterate until ≥95% of the 1000-sample is covered.
- Click "Save draft" frequently.
- Click "Test on photo" with 3-4 sample IDs to spot-check.

- [ ] **Step 2: Edit expansions by hand (no UI yet)**

The curator UI doesn't expose query expansions in v1. Edit `/data/vocab_draft.json` on the NAS directly:

```bash
ssh nas.local 'sudo nano /volume1/docker/photosearch/data/vocab_draft.json'
```

Set the `"expansions"` key to a dict like:
```json
"expansions": {
  "ocean": ["beach", "sea", "coast"],
  "dog":   ["pet", "puppy"],
  "kid":   ["child"]
}
```

- [ ] **Step 3: Click "Compile vocab modules" in the UI**

Expected: success message with a draft hash; the three vocab Python files appear in the repo.

- [ ] **Step 4: Inspect the generated modules**

Run:
```bash
head -20 photosearch/vocab_content.py photosearch/vocab_visual.py photosearch/vocab_query_expansion.py
```

Expected: header comments with hash + timestamp; sorted term lists; importable.

Run:
```bash
venv/bin/python -c "
from photosearch.vocab_content import CONTENT_VOCABULARY
from photosearch.vocab_visual import VISUAL_VOCABULARY
from photosearch.vocab_query_expansion import _QUERY_TO_CATEGORIES
print(len(CONTENT_VOCABULARY), 'content;', len(VISUAL_VOCABULARY), 'visual;', len(_QUERY_TO_CATEGORIES), 'expansions')
"
```

- [ ] **Step 5: Commit the generated vocab**

```bash
git add photosearch/vocab_content.py photosearch/vocab_visual.py photosearch/vocab_query_expansion.py
git commit -m "feat(vocab): initial curated content/visual/expansion modules"
```

---

## Phase 4 — Backend passes and search integration

**Goal:** Wire up `category-content`, `category-visual`, and `keywords` end-to-end (worker → API → DB → search). Delete the old `--tags` pass and its frontend hooks cleanly.

### Task 4.1: Content extraction function (TDD)

**Files:**
- Modify: `photosearch/describe.py` — add `extract_categories_from_description`.

- [ ] **Step 1: Add tests**

Append to `tests/test_describe_extraction.py`:

```python
def test_extract_categories_returns_only_vocab_terms(monkeypatch):
    from photosearch import describe as d
    monkeypatch.setattr(d, "_ollama_chat_with_retry",
                        lambda **kw: "beach, dog, glorbnox, mountain")
    monkeypatch.setattr("photosearch.vocab_content.CONTENT_VOCABULARY",
                        ["beach", "dog", "mountain", "sky"])
    out = d.extract_categories_from_description("a dog at the beach")
    assert set(out) == {"beach", "dog", "mountain"}
    # 'glorbnox' silently dropped.


def test_extract_categories_returns_empty_on_empty_response(monkeypatch):
    from photosearch import describe as d
    monkeypatch.setattr(d, "_ollama_chat_with_retry", lambda **kw: "")
    monkeypatch.setattr("photosearch.vocab_content.CONTENT_VOCABULARY",
                        ["beach", "dog"])
    assert d.extract_categories_from_description("a dog") == []


def test_extract_categories_returns_empty_on_none_description(monkeypatch):
    from photosearch import describe as d
    assert d.extract_categories_from_description(None) == []
    assert d.extract_categories_from_description("") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_describe_extraction.py -k extract_categories -v`
Expected: FAIL — function doesn't exist; vocab module may not exist either.

- [ ] **Step 3: Implement `extract_categories_from_description` in `describe.py`**

Add near the existing `tag_photo` (line ~477):

```python
CATEGORY_CONTENT_MODEL = "llama3.2:3b"  # text-only, default per Phase 0


def _build_category_prompt(description: str, vocab: list[str]) -> str:
    vocab_str = ", ".join(vocab)
    return (
        "From the vocabulary below, return ONLY tags that apply to this photo description.\n"
        "Rules:\n"
        "- Return a comma-separated list with no commentary.\n"
        "- Only use tags from the vocabulary; ignore anything else.\n\n"
        f"Vocabulary: {vocab_str}\n\n"
        f"Description: {description.strip()}\n"
    )


def extract_categories_from_description(
    description: Optional[str],
    model: str = CATEGORY_CONTENT_MODEL,
) -> list[str]:
    """Map a description → list of in-vocab categories via a text-only LLM."""
    if not description or not description.strip():
        return []
    from .vocab_content import CONTENT_VOCABULARY
    if not HAS_OLLAMA:
        return []
    vocab_set = set(CONTENT_VOCABULARY)
    prompt = _build_category_prompt(description, CONTENT_VOCABULARY)
    try:
        raw = _ollama_chat_with_retry(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
    except Exception:
        return []
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for token in raw.split(","):
        t = token.strip().lower().rstrip(".")
        if t in vocab_set and t not in seen:
            seen.add(t)
            out.append(t)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_describe_extraction.py -k extract_categories -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add photosearch/describe.py tests/test_describe_extraction.py
git commit -m "feat(describe): extract_categories_from_description (text-only LLM)"
```

### Task 4.2: Keywords extraction function (TDD)

**Files:**
- Modify: `photosearch/describe.py`

- [ ] **Step 1: Add tests**

Append to `tests/test_describe_extraction.py`:

```python
def test_extract_keywords_uses_bakeoff_parser(monkeypatch):
    from photosearch import describe as d
    monkeypatch.setattr(d, "_ollama_chat_with_retry",
                        lambda **kw: "Golden Retriever, Stinson Beach, sunset")
    out = d.extract_keywords_from_description("a dog at the beach")
    assert out == ["golden retriever", "stinson beach", "sunset"]


def test_extract_keywords_returns_empty_on_none(monkeypatch):
    from photosearch import describe as d
    assert d.extract_keywords_from_description(None) == []
    assert d.extract_keywords_from_description("") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_describe_extraction.py -k keywords -v`
Expected: FAIL — function missing.

- [ ] **Step 3: Implement `extract_keywords_from_description`**

In `photosearch/describe.py`, after the `extract_categories_from_description` you just added:

```python
KEYWORDS_MODEL = "<TBD — set after Phase 0 bakeoff, e.g. 'llama3.2:3b'>"


def extract_keywords_from_description(
    description: Optional[str],
    model: str = KEYWORDS_MODEL,
) -> list[str]:
    """Extract 5-15 free-form lowercased keywords from a description."""
    from .bakeoff import build_keyword_prompt, parse_keywords_response
    if not description or not description.strip():
        return []
    if not HAS_OLLAMA:
        return []
    prompt = build_keyword_prompt(description)
    try:
        raw = _ollama_chat_with_retry(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
    except Exception:
        return []
    return parse_keywords_response(raw)
```

**Replace `<TBD ...>` with the Phase 0 winner before the test pass step.**

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_describe_extraction.py -k keywords -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add photosearch/describe.py tests/test_describe_extraction.py
git commit -m "feat(describe): extract_keywords_from_description (text-only LLM)"
```

### Task 4.3: Visual-tag function (rename + vocab swap)

**Files:**
- Modify: `photosearch/describe.py` — rename `tag_photo` → `tag_visual_photo`, drop `TAG_VOCABULARY`/`TAG_PROMPT`/`_parse_tag_response`, point at `VISUAL_VOCABULARY`, drop `_MAX_PLAUSIBLE_TAGS` to 12.

- [ ] **Step 1: Add tests**

Append to `tests/test_describe_extraction.py`:

```python
def test_tag_visual_photo_uses_visual_vocab(monkeypatch, tmp_path):
    from photosearch import describe as d
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")  # minimal jpeg stub
    monkeypatch.setattr("photosearch.vocab_visual.VISUAL_VOCABULARY",
                        ["dramatic", "peaceful", "foggy"])
    monkeypatch.setattr(d, "_ollama_chat_with_retry",
                        lambda **kw: "dramatic, peaceful, nope")
    monkeypatch.setattr(d, "_encode_image_for_ollama", lambda p: "encoded")
    out = d.tag_visual_photo(str(img))
    assert out == ["dramatic", "peaceful"]


def test_tag_visual_photo_regurgitation_guard_at_12(monkeypatch, tmp_path):
    from photosearch import describe as d
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    big_vocab = [f"v{i}" for i in range(25)]
    monkeypatch.setattr("photosearch.vocab_visual.VISUAL_VOCABULARY", big_vocab)
    # First call returns 12 (regurgitation); retry returns the same 12 → guard returns None.
    monkeypatch.setattr(d, "_ollama_chat_with_retry",
                        lambda **kw: ", ".join(big_vocab[:12]))
    monkeypatch.setattr(d, "_encode_image_for_ollama", lambda p: "encoded")
    out = d.tag_visual_photo(str(img))
    assert out is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_describe_extraction.py -k tag_visual -v`
Expected: FAIL — function missing.

- [ ] **Step 3: Refactor `describe.py`**

In `photosearch/describe.py`, delete `TAG_VOCABULARY` (lines 415–441), `TAG_PROMPT` (lines 443–453), `_MAX_PLAUSIBLE_TAGS = 16` (line 463), `_parse_tag_response` (lines 466–474), and `tag_photo` (lines 477–524). Replace with:

```python
_VISUAL_MAX_PLAUSIBLE_TAGS = 12  # tighter than the old 16; smaller vocab.


def _build_visual_prompt(vocab: list[str]) -> str:
    return (
        "Pick visual-quality tags for this photo from this list: "
        + ", ".join(vocab)
        + "\n\nRules:\n"
        "- Return ONLY a comma-separated list of tags from the list above.\n"
        "- Mood / light / composition only. Don't describe content.\n"
        "- Include every tag that clearly applies.\n"
    )


def _parse_visual_response(raw: str, vocab_set: set[str]) -> list[str]:
    out = []
    seen = set()
    for token in (raw or "").split(","):
        t = token.strip().lower().rstrip(".")
        if t in vocab_set and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def tag_visual_photo(
    image_path: str,
    model: str = TAGS_MODEL,
) -> Optional[list[str]]:
    """Generate visual-quality tags for a single photo via Ollama (vision).

    Mirrors the old tag_photo regurgitation guard at threshold 12 (smaller
    vocab). Returns None on guard trip / generation failure / empty result.
    """
    from .vocab_visual import VISUAL_VOCABULARY
    vocab_set = set(VISUAL_VOCABULARY)
    prompt = _build_visual_prompt(VISUAL_VOCABULARY)
    raw = describe_photo(image_path, model=model, prompt=prompt)
    if not raw:
        return None
    tags = _parse_visual_response(raw, vocab_set)
    if len(tags) >= _VISUAL_MAX_PLAUSIBLE_TAGS:
        # Retry with temp bump (same shape as old tag_photo).
        if not HAS_OLLAMA:
            return None
        from pathlib import Path
        path = Path(image_path)
        if not path.exists():
            return None
        encoded = _encode_image_for_ollama(str(path))
        image_ref = encoded if encoded is not None else str(path)
        retry_opts = dict(_options_for_model(model))
        retry_opts["temperature"] = 0.4
        retry_opts.setdefault("repeat_penalty", 1.3)
        try:
            raw2 = _ollama_chat_with_retry(
                model=model,
                messages=[{"role": "user", "content": prompt, "images": [image_ref]}],
                options=retry_opts,
            )
        except Exception:
            raw2 = None
        if not raw2:
            return None
        tags2 = _parse_visual_response(raw2, vocab_set)
        if len(tags2) >= _VISUAL_MAX_PLAUSIBLE_TAGS or not tags2:
            return None
        tags = tags2
    return tags if tags else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_describe_extraction.py -k tag_visual -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add photosearch/describe.py tests/test_describe_extraction.py
git commit -m "refactor(describe): tag_photo → tag_visual_photo on visual vocab"
```

### Task 4.4: Worker handlers for the three new passes

**Files:**
- Modify: `photosearch/worker.py` — branches near lines 461 and 780.

- [ ] **Step 1: Find the existing tags branch and the imports at the top**

Read `photosearch/worker.py:461-525` for the in-process tags handler shape and lines 461–800 for the dispatch table.

- [ ] **Step 2: Add new branch arms in the worker dispatcher**

In `photosearch/worker.py`, find the `if pass_type == "clip":` chain around line 76 (the small one that lists supported passes) and extend it:

```python
    if pass_type == "clip":
        ...  # existing
    elif pass_type == "quality":
        ...  # existing
    elif pass_type == "faces":
        ...  # existing
    elif pass_type == "verify":
        ...  # existing
    elif pass_type in ("describe", "tags", "category-visual"):
        ...  # passes that need image bytes
    elif pass_type in ("category-content", "keywords"):
        ...  # text-only passes — no image download
```

Adjust the existing branch boundaries: the text-only passes do NOT require image bytes. Mirror the existing pattern (the existing function probably skips download for some passes already; look for `if pass_type in (...)` before the `download_photo` call).

- [ ] **Step 3: Add the per-pass result wiring around line 780**

In the long dispatch (search for `if pass_type == "describe":` in the result-builder section, ~line 790), add three new branches that mirror the shape:

```python
                elif pass_type == "category-content":
                    from .describe import extract_categories_from_description, CATEGORY_CONTENT_MODEL
                    # Fetch the description for each claimed photo from the server.
                    result_payload = []
                    for photo_meta in batch:
                        desc = photo_meta.get("description")
                        cats = extract_categories_from_description(desc, model=category_content_model)
                        result_payload.append({
                            "photo_id": photo_meta["id"],
                            "categories": cats,
                            "model": category_content_model,
                        })
                    submit_kwargs["category_content_results"] = result_payload
                elif pass_type == "category-visual":
                    from .describe import tag_visual_photo, TAGS_MODEL
                    result_payload = []
                    for photo_meta, path in zip(batch, downloaded_paths):
                        tags = tag_visual_photo(path, model=tags_model)
                        result_payload.append({
                            "photo_id": photo_meta["id"],
                            "visual_tags": tags or [],
                            "model": tags_model,
                        })
                    submit_kwargs["category_visual_results"] = result_payload
                elif pass_type == "keywords":
                    from .describe import extract_keywords_from_description, KEYWORDS_MODEL
                    result_payload = []
                    for photo_meta in batch:
                        kws = extract_keywords_from_description(photo_meta.get("description"),
                                                                model=keywords_model)
                        result_payload.append({
                            "photo_id": photo_meta["id"],
                            "keywords": kws,
                            "model": keywords_model,
                        })
                    submit_kwargs["keywords_results"] = result_payload
```

For `category-content` and `keywords`, the claim-batch response needs to include `description`. Update `photosearch/worker_api.py:claim_batch` (line 165) to also return `description` for these pass types — extend the SELECT to include `p.description`.

- [ ] **Step 4: Add CLI options for the three model knobs in worker invocation**

Find the `worker` Click command (search for `@cli.command("worker")` in `cli.py`). Next to `--describe-model`, `--tags-model`, `--verify-model`, add:

```python
@click.option("--category-content-model", default=None,
              help="Model for category-content pass (default: llama3.2:3b).")
@click.option("--category-visual-model", default=None,
              help="Model for category-visual pass (default: llava).")
@click.option("--keywords-model", default=None,
              help="Model for keywords pass (default: Phase 0 bakeoff winner).")
```

Pass them through to `worker.run_worker(...)` and into the branches above.

- [ ] **Step 5: Local smoke test (single batch)**

Start a local server in another shell:
```bash
venv/bin/uvicorn photosearch.web:app --port 8090 &
```

Then run one batch:
```bash
venv/bin/python cli.py worker -s http://localhost:8090 -p category-content --one-shot --directory /tmp/empty
```

Expected: "no work claimable" message (empty DB) — no crash.

- [ ] **Step 6: Commit**

```bash
git add photosearch/worker.py photosearch/worker_api.py cli.py
git commit -m "feat(worker): dispatch the three new pass types"
```

### Task 4.5: API submit handlers (TDD)

**Files:**
- Modify: `photosearch/worker_api.py` — `SubmitRequest` (~line 244), submit branches.
- Create: `tests/test_worker_categories.py`

- [ ] **Step 1: Write failing end-to-end test**

Create `tests/test_worker_categories.py`:

```python
import json
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_DB", str(tmp_path / "x.db"))
    from photosearch.db import PhotoDB
    with PhotoDB(str(tmp_path / "x.db")) as db:
        db.conn.executemany(
            "INSERT INTO photos (id, filepath, description) VALUES (?, ?, ?)",
            [(1, "a.jpg", "a dog at the beach"),
             (2, "b.jpg", "a misty mountain morning")],
        )
        db.conn.commit()
    from photosearch.web import app
    return TestClient(app)


def test_claim_batch_returns_description_for_text_passes(client):
    r = client.post("/api/worker/claim-batch", json={
        "worker_id": "w1", "pass_type": "category-content", "limit": 10,
    })
    body = r.json()
    assert body["pass_type"] == "category-content"
    descs = {p["id"]: p["description"] for p in body["photos"]}
    assert descs == {1: "a dog at the beach", 2: "a misty mountain morning"}


def test_submit_category_content_writes_to_photos_and_generations(client):
    claim = client.post("/api/worker/claim-batch", json={
        "worker_id": "w1", "pass_type": "category-content", "limit": 10,
    }).json()
    batch_id = claim["batch_id"]
    r = client.post("/api/worker/submit-results", json={
        "batch_id": batch_id,
        "worker_id": "w1",
        "pass_type": "category-content",
        "category_content_results": [
            {"photo_id": 1, "categories": ["beach", "dog"], "model": "llama3.2:3b"},
            {"photo_id": 2, "categories": ["mountain"], "model": "llama3.2:3b"},
        ],
    })
    assert r.status_code == 200

    from photosearch.db import PhotoDB
    import os
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        rows = {
            r["id"]: json.loads(r["categories"])
            for r in db.conn.execute("SELECT id, categories FROM photos").fetchall()
        }
        assert rows == {1: ["beach", "dog"], 2: ["mountain"]}
        # Generations log entries.
        text_types = {r[0] for r in db.conn.execute(
            "SELECT DISTINCT text_type FROM generations"
        ).fetchall()}
        assert "category-content" in text_types


def test_submit_keywords_writes_lowercased(client):
    claim = client.post("/api/worker/claim-batch", json={
        "worker_id": "w1", "pass_type": "keywords", "limit": 10,
    }).json()
    r = client.post("/api/worker/submit-results", json={
        "batch_id": claim["batch_id"],
        "worker_id": "w1",
        "pass_type": "keywords",
        "keywords_results": [
            {"photo_id": 1, "keywords": ["dog", "pacific ocean", "stinson beach"],
             "model": "llama3.2:3b"},
        ],
    })
    assert r.status_code == 200
    from photosearch.db import PhotoDB
    import os
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        row = db.conn.execute("SELECT keywords FROM photos WHERE id=1").fetchone()
        assert json.loads(row[0]) == ["dog", "pacific ocean", "stinson beach"]


def test_empty_results_still_mark_processed(client):
    """Parse-empty must mark processed so the photo isn't re-claimed forever."""
    claim = client.post("/api/worker/claim-batch", json={
        "worker_id": "w1", "pass_type": "category-content", "limit": 10,
    }).json()
    client.post("/api/worker/submit-results", json={
        "batch_id": claim["batch_id"],
        "worker_id": "w1",
        "pass_type": "category-content",
        "category_content_results": [
            {"photo_id": 1, "categories": [], "model": "llama3.2:3b"},
            {"photo_id": 2, "categories": [], "model": "llama3.2:3b"},
        ],
    })
    from photosearch.db import PhotoDB
    import os
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        # categories column stays NULL but worker_processed has attempts=1.
        rows = db.conn.execute(
            "SELECT photo_id, attempts FROM worker_processed WHERE pass_type='category-content'"
        ).fetchall()
        assert {r[0]: r[1] for r in rows} == {1: 1, 2: 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_worker_categories.py -v`
Expected: FAIL — claim doesn't recognise the new pass type / submit fields don't exist.

- [ ] **Step 3: Extend `claim_batch` to include `description`**

In `photosearch/worker_api.py:claim_batch` (line 165), find the section that builds the `photos` response. Modify it so when `pass_type in ("category-content", "keywords", "describe", "verify")` the SELECT also returns `description`. The simplest path: always include `description` in the response (3-4 KB per photo is negligible).

```python
        # Build response with description for text passes.
        photo_meta = []
        for photo in claimed_photos:
            row = db.conn.execute(
                "SELECT id, filepath, description FROM photos WHERE id = ?",
                (photo["id"],),
            ).fetchone()
            photo_meta.append({
                "id": row["id"],
                "filepath": row["filepath"],
                "description": row["description"],
            })
```

(Or, if claim_batch already returns dicts from a wider SELECT, just confirm `description` is present.)

- [ ] **Step 4: Add SubmitRequest fields**

In `photosearch/worker_api.py`, find the `class SubmitRequest(BaseModel):` (near line 244). Add:

```python
class CategoryContentResult(BaseModel):
    photo_id: int
    categories: list[str]
    model: Optional[str] = None
    model_version: Optional[str] = None


class CategoryVisualResult(BaseModel):
    photo_id: int
    visual_tags: list[str]
    model: Optional[str] = None
    model_version: Optional[str] = None


class KeywordsResult(BaseModel):
    photo_id: int
    keywords: list[str]
    model: Optional[str] = None
    model_version: Optional[str] = None


class SubmitRequest(BaseModel):
    # ... existing fields ...
    category_content_results: Optional[list[CategoryContentResult]] = None
    category_visual_results: Optional[list[CategoryVisualResult]] = None
    keywords_results: Optional[list[KeywordsResult]] = None
```

- [ ] **Step 5: Add submit handler branches**

In `photosearch/worker_api.py:submit_results` (line 244), find the chain of `elif req.pass_type == "tags":` (line 357) and add three parallel branches after the existing `verify` block:

```python
            elif req.pass_type == "category-content":
                processed_photo_ids = []
                for r in (req.category_content_results or []):
                    cats_json = json.dumps(r.categories)
                    if r.categories:  # only write non-empty results to photos.
                        db.conn.execute(
                            "UPDATE photos SET categories = ? WHERE id = ?",
                            (cats_json, r.photo_id),
                        )
                        db.log_generation(
                            r.photo_id, "category-content", cats_json,
                            model_used=r.model, model_version=r.model_version,
                        )
                    processed_photo_ids.append(r.photo_id)
                db.mark_processed(processed_photo_ids, "category-content")

            elif req.pass_type == "category-visual":
                processed_photo_ids = []
                for r in (req.category_visual_results or []):
                    tags_json = json.dumps(r.visual_tags)
                    if r.visual_tags:
                        db.conn.execute(
                            "UPDATE photos SET visual_tags = ? WHERE id = ?",
                            (tags_json, r.photo_id),
                        )
                        db.log_generation(
                            r.photo_id, "category-visual", tags_json,
                            model_used=r.model, model_version=r.model_version,
                        )
                    processed_photo_ids.append(r.photo_id)
                db.mark_processed(processed_photo_ids, "category-visual")

            elif req.pass_type == "keywords":
                processed_photo_ids = []
                for r in (req.keywords_results or []):
                    kws_json = json.dumps(r.keywords)
                    if r.keywords:
                        db.conn.execute(
                            "UPDATE photos SET keywords = ? WHERE id = ?",
                            (kws_json, r.photo_id),
                        )
                        db.log_generation(
                            r.photo_id, "keywords", kws_json,
                            model_used=r.model, model_version=r.model_version,
                        )
                    processed_photo_ids.append(r.photo_id)
                db.mark_processed(processed_photo_ids, "keywords")
```

- [ ] **Step 6: Add clear-pass branches**

In `photosearch/worker_api.py:clear_pass` (line ~466), add parallel branches mirroring the existing `tags` shape:

```python
        elif req.pass_type == "category-content":
            db.conn.execute(
                f"UPDATE photos SET categories = NULL WHERE id IN ({placeholders})",
                list(req.photo_ids),
            )
            db.conn.execute(
                f"DELETE FROM worker_processed WHERE pass_type = 'category-content' AND photo_id IN ({placeholders})",
                list(req.photo_ids),
            )
        elif req.pass_type == "category-visual":
            db.conn.execute(
                f"UPDATE photos SET visual_tags = NULL WHERE id IN ({placeholders})",
                list(req.photo_ids),
            )
            db.conn.execute(
                f"DELETE FROM worker_processed WHERE pass_type = 'category-visual' AND photo_id IN ({placeholders})",
                list(req.photo_ids),
            )
        elif req.pass_type == "keywords":
            db.conn.execute(
                f"UPDATE photos SET keywords = NULL WHERE id IN ({placeholders})",
                list(req.photo_ids),
            )
            db.conn.execute(
                f"DELETE FROM worker_processed WHERE pass_type = 'keywords' AND photo_id IN ({placeholders})",
                list(req.photo_ids),
            )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_worker_categories.py -v`
Expected: 4 passed.

- [ ] **Step 8: Commit**

```bash
git add photosearch/worker_api.py tests/test_worker_categories.py
git commit -m "feat(api): claim/submit/clear-pass for three new pass types"
```

### Task 4.6: Indexer integration

**Files:**
- Modify: `photosearch/index.py` — `index_directory()` and `_index_collection()`.
- Modify: `cli.py` — `index` command flags.

- [ ] **Step 1: Add three pairs of flags to `index_directory`**

In `photosearch/index.py`, find the `index_directory` signature (search for `def index_directory(`). Add parameters after the existing `enable_tags` / `force_tags`:

```python
    enable_category_content: bool = False,
    force_category_content: bool = False,
    enable_category_visual: bool = False,
    force_category_visual: bool = False,
    enable_keywords: bool = False,
    force_keywords: bool = False,
    category_content_model: str = "llama3.2:3b",
    keywords_model: str = "llama3.2:3b",  # update after Phase 0
```

Find the existing tags-pass block (search for `if enable_tags:` around line 416) and add three parallel blocks. Each follows the streaming pattern: fetch candidates with `db.get_unprocessed_photos(pass_type)`, call the extraction function, write per-batch (`UPDATE photos SET <col> = ?` + `db.log_generation` + `db.mark_processed`).

Example shape for `category-content` (write the same for `category-visual` and `keywords`):

```python
        if enable_category_content:
            from .describe import extract_categories_from_description, check_available as ollama_check
            if not ollama_check():
                print("Ollama not available; skipping category-content pass.")
            else:
                model = category_content_model
                candidates = db.conn.execute(
                    """SELECT id, description FROM photos
                       WHERE description IS NOT NULL
                         AND categories IS NULL
                         AND NOT EXISTS (SELECT 1 FROM worker_processed wp
                                         WHERE wp.photo_id = photos.id
                                           AND wp.pass_type = 'category-content'
                                           AND wp.attempts >= 3)""",
                ).fetchall()
                if force_category_content:
                    candidates = db.conn.execute(
                        "SELECT id, description FROM photos WHERE description IS NOT NULL"
                    ).fetchall()
                print(f"\nExtracting categories for {len(candidates)} photo(s)...")
                for row in candidates:
                    cats = extract_categories_from_description(row["description"], model=model)
                    if cats:
                        cats_json = json.dumps(cats)
                        db.conn.execute("UPDATE photos SET categories=? WHERE id=?",
                                        (cats_json, row["id"]))
                        db.log_generation(row["id"], "category-content", cats_json,
                                          model_used=model)
                    db.mark_processed([row["id"]], "category-content")
```

Apply the same pattern to `_index_collection` (around line 1062).

- [ ] **Step 2: Add CLI flags on `index`**

In `cli.py`, find the `@cli.command("index")` definition. Next to the existing `--tags` / `--force-tags`, add:

```python
@click.option("--category-content", "enable_category_content", is_flag=True, default=False)
@click.option("--force-category-content", is_flag=True, default=False)
@click.option("--category-visual", "enable_category_visual", is_flag=True, default=False)
@click.option("--force-category-visual", is_flag=True, default=False)
@click.option("--keywords", "enable_keywords", is_flag=True, default=False)
@click.option("--force-keywords", is_flag=True, default=False)
@click.option("--category-content-model", default="llama3.2:3b", show_default=True)
@click.option("--keywords-model", default="llama3.2:3b", show_default=True)
```

Pass them through to `index_directory`.

Update the `--full` aggregate flag (search for the `--full` handler) to also enable the three new passes:

```python
    if full:
        enable_clip = True
        enable_faces = True
        enable_describe = True
        enable_quality = True
        enable_category_content = True
        enable_category_visual = True
        enable_keywords = True
        # Note: --tags is removed; --full no longer enables it.
```

- [ ] **Step 3: Smoke check**

Run: `venv/bin/python cli.py index --help`
Expected: help text includes the new flags.

- [ ] **Step 4: Commit**

```bash
git add photosearch/index.py cli.py
git commit -m "feat(index): three new pass flags wired into index_directory/_index_collection"
```

### Task 4.7: Search match functions (TDD)

**Files:**
- Modify: `photosearch/search.py`

- [ ] **Step 1: Add tests**

Append to `tests/test_search_logic.py` (or create the file if absent — it exists per `ls tests/`):

```python
import json


def test_categories_match_uses_exact_then_expansion(monkeypatch):
    from photosearch import search as s
    monkeypatch.setattr("photosearch.search._QUERY_TO_CATEGORIES",
                        {"sea": {"beach", "ocean"}})
    cats_json = json.dumps(["beach", "dog"])
    # Exact match.
    assert s._categories_match_query(cats_json, "beach") > 0
    # Expansion: "sea" → {beach, ocean}; beach is present → match.
    assert s._categories_match_query(cats_json, "sea") > 0
    # No expansion + not present.
    assert s._categories_match_query(cats_json, "moon") == 0


def test_keywords_match_handles_multiword_phrases():
    from photosearch import search as s
    kws_json = json.dumps(["golden retriever", "stinson beach"])
    assert s._keywords_match_query(kws_json, "golden retriever") > 0
    # Token in query, full phrase present → match.
    assert s._keywords_match_query(kws_json, "stinson") > 0
    # Wrong phrase order — still matches because phrase token is checked as substring.
    assert s._keywords_match_query(kws_json, "retriever") > 0
    # Unrelated.
    assert s._keywords_match_query(kws_json, "moon") == 0


def test_visual_match_returns_score_on_exact():
    from photosearch import search as s
    vis_json = json.dumps(["dramatic", "foggy"])
    assert s._visual_match_query(vis_json, "dramatic") > 0
    assert s._visual_match_query(vis_json, "cheerful") == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_search_logic.py -k "categories_match or keywords_match or visual_match" -v`
Expected: FAIL — functions don't exist.

- [ ] **Step 3: Implement the three functions in `search.py`**

In `photosearch/search.py`, near the existing `_tags_match_query` (around line 432), add:

```python
from typing import Optional
import json
import re


try:
    from .vocab_query_expansion import _QUERY_TO_CATEGORIES
except ImportError:
    _QUERY_TO_CATEGORIES: dict[str, set[str]] = {}


def _categories_match_query(categories_json: Optional[str], query: str) -> float:
    """Score a categories array against a free-text query."""
    if not categories_json or not query:
        return 0.0
    try:
        cats = set(json.loads(categories_json))
    except (ValueError, TypeError):
        return 0.0
    if not cats:
        return 0.0
    q_lower = query.lower().strip()
    score = 0.0
    if q_lower in cats:
        score += 1.0
    for word in q_lower.split():
        if word in cats:
            score += 0.5
        expansion = _QUERY_TO_CATEGORIES.get(word, set())
        if expansion & cats:
            score += 0.4
    return score


def _visual_match_query(visual_json: Optional[str], query: str) -> float:
    if not visual_json or not query:
        return 0.0
    try:
        tags = set(json.loads(visual_json))
    except (ValueError, TypeError):
        return 0.0
    q_lower = query.lower().strip()
    score = 0.0
    if q_lower in tags:
        score += 0.8
    for word in q_lower.split():
        if word in tags:
            score += 0.4
    return score


def _keywords_match_query(keywords_json: Optional[str], query: str) -> float:
    """Score: full-phrase contains > word-in-phrase substring."""
    if not keywords_json or not query:
        return 0.0
    try:
        keywords = json.loads(keywords_json)
    except (ValueError, TypeError):
        return 0.0
    if not keywords:
        return 0.0
    q_lower = query.lower().strip()
    score = 0.0
    for kw in keywords:
        kw_l = kw.lower()
        if kw_l == q_lower:
            score += 1.2
        elif q_lower in kw_l or kw_l in q_lower:
            score += 0.6
        else:
            # word-level overlap.
            kw_words = set(kw_l.split())
            q_words = set(q_lower.split())
            if kw_words & q_words:
                score += 0.3
    return score
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_search_logic.py -k "categories_match or keywords_match or visual_match" -v`
Expected: 3 passed (or 6 passed if the test file already exists with other tests).

- [ ] **Step 5: Commit**

```bash
git add photosearch/search.py tests/test_search_logic.py
git commit -m "feat(search): _categories/_visual/_keywords_match_query"
```

### Task 4.8: Wire match functions into `search_combined`; rename `tag_match → text_match`

**Files:**
- Modify: `photosearch/search.py` — `search_combined` (and the helper at line 662).
- Modify: `photosearch/web.py` — `api_search` param.

- [ ] **Step 1: Add test for the parameter rename**

Append to `tests/test_search_logic.py`:

```python
def test_search_combined_accepts_text_match_param_categories_only(monkeypatch):
    """text_match='categories' should use only the categories signal."""
    # End-to-end is heavy; just verify the function signature accepts the kw.
    from photosearch.search import search_combined
    import inspect
    sig = inspect.signature(search_combined)
    assert "text_match" in sig.parameters
    # Backwards-compat: tag_match should NOT still be there (clean rename).
    assert "tag_match" not in sig.parameters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_search_logic.py::test_search_combined_accepts_text_match_param_categories_only -v`
Expected: FAIL — old param still named `tag_match`.

- [ ] **Step 3: Rename + extend in `search.py`**

In `photosearch/search.py`, find `tag_match: str = "both"` (lines 662 and 1339) and rename to `text_match: str = "all"` in both. Where the body currently dispatches:

```python
    if tag_match in ("tags", "both"):
```

replace the whole block (lines ~729–810) with a richer dispatcher:

```python
    # Build the per-photo text relevance using all four signals.
    # text_match controls which subset contributes:
    #   "all"        (default) — union of categories + visual + keywords + dict.
    #   "categories" — only categories signal.
    #   "visual"     — only visual signal.
    #   "keywords"   — only keywords signal.
    #   "dict"       — only description-literal _tags_match_query equivalent.
    #   "off"        — text relevance disabled.
    use_categories = text_match in ("all", "categories")
    use_visual     = text_match in ("all", "visual")
    use_keywords   = text_match in ("all", "keywords")
    use_dict       = text_match in ("all", "dict")
    if text_match not in ("all", "categories", "visual", "keywords", "dict", "off"):
        text_match = "all"
        use_categories = use_visual = use_keywords = use_dict = True

    # Replace the old tag_cache with three caches.
    cat_cache, visual_cache, kw_cache = {}, {}, {}
    if use_categories:
        for row in db.conn.execute("SELECT id, categories FROM photos WHERE categories IS NOT NULL"):
            cat_cache[row["id"]] = row["categories"]
    if use_visual:
        for row in db.conn.execute("SELECT id, visual_tags FROM photos WHERE visual_tags IS NOT NULL"):
            visual_cache[row["id"]] = row["visual_tags"]
    if use_keywords:
        for row in db.conn.execute("SELECT id, keywords FROM photos WHERE keywords IS NOT NULL"):
            kw_cache[row["id"]] = row["keywords"]

    # Per-photo combine (in the existing per-result loop).
    for photo_id in photo_ids:
        rel = 0.0
        if use_dict and photo_id in desc_cache:
            rel += _description_match_query(desc_cache.get(photo_id), positive_query) * 0.6
        if use_categories and photo_id in cat_cache:
            rel += _categories_match_query(cat_cache.get(photo_id), positive_query) * 1.0
        if use_visual and photo_id in visual_cache:
            rel += _visual_match_query(visual_cache.get(photo_id), positive_query) * 0.7
        if use_keywords and photo_id in kw_cache:
            rel += _keywords_match_query(kw_cache.get(photo_id), positive_query) * 1.2
        text_relevance[photo_id] = rel
```

(Adapt to the existing loop variable names; the gist is: replace the single `tag_cache` lookup with three caches and a weighted sum.)

Also propagate the rename through any docstrings + the `search_combined` call at line 1467.

Delete `_tags_match_query` (line 432) and `_QUERY_TO_TAGS` (line 380) — they are now dead.

- [ ] **Step 4: Run search tests + signature test**

Run: `venv/bin/pytest tests/test_search_logic.py -v`
Expected: all pass.

- [ ] **Step 5: Update `web.py` `/api/search`**

In `photosearch/web.py:216`, rename:

```python
    tag_match: str = Query("both", description="..."),
```

to:

```python
    text_match: str = Query(
        "all",
        description="Text matching mode: all, categories, visual, keywords, dict, off.",
    ),
    # Legacy alias kept for one release so bookmarked URLs don't 422.
    tag_match: Optional[str] = Query(None, deprecated=True),
```

In the body, add a one-liner translation:

```python
    if tag_match and not request.query_params.get("text_match"):
        # Old "both" → new "all"; "tags" → "categories"; pass through "dict"/"off".
        text_match = {"both": "all", "tags": "categories"}.get(tag_match, tag_match)
```

Then pass `text_match=text_match` into `search_combined` (line 257).

- [ ] **Step 6: Smoke test the API**

Start a local server, then:
```bash
curl -s 'http://localhost:8000/api/search?q=beach&text_match=categories&limit=2' | head -20
```

Expected: JSON response with `results` array (may be empty if backfill hasn't run yet).

- [ ] **Step 7: Commit**

```bash
git add photosearch/search.py photosearch/web.py tests/test_search_logic.py
git commit -m "feat(search): text_match modes; new categories/visual/keywords scoring"
```

### Task 4.9: New filter params + `tag=X → category=X` redirect

**Files:**
- Modify: `photosearch/web.py` — `api_search` and `api_photo_detail`.
- Modify: `photosearch/search.py` — `search_combined` filter args.

- [ ] **Step 1: Add the three new filter params**

In `photosearch/web.py:api_search`, add to the function signature:

```python
    category: Optional[str] = Query(None, description="Exact-match category filter."),
    visual_tag: Optional[str] = Query(None, description="Exact-match visual tag filter."),
    keyword: Optional[str] = Query(None, description="Substring keyword filter."),
    tag: Optional[str] = Query(None, deprecated=True),  # back-compat alias
```

At the top of the body:

```python
    if tag and not category:
        category = tag  # legacy /?tag=beach → /?category=beach
```

Pass `category=category, visual_tag=visual_tag, keyword=keyword` into `search_combined`.

- [ ] **Step 2: Add the filters to `search_combined`**

In `photosearch/search.py:search_combined`, add params + intersect them into `result_sets`. Pattern (copy from existing `person=` handling):

```python
    if category:
        cat_ids = set()
        for row in db.conn.execute(
            "SELECT id, categories FROM photos WHERE categories IS NOT NULL"
        ):
            try:
                if category.lower() in {c.lower() for c in json.loads(row["categories"])}:
                    cat_ids.add(row["id"])
            except (ValueError, TypeError):
                pass
        result_sets.append(cat_ids)
    if visual_tag:
        vis_ids = set()
        for row in db.conn.execute(
            "SELECT id, visual_tags FROM photos WHERE visual_tags IS NOT NULL"
        ):
            try:
                if visual_tag.lower() in {c.lower() for c in json.loads(row["visual_tags"])}:
                    vis_ids.add(row["id"])
            except (ValueError, TypeError):
                pass
        result_sets.append(vis_ids)
    if keyword:
        kw_ids = set()
        needle = keyword.lower()
        for row in db.conn.execute(
            "SELECT id, keywords FROM photos WHERE keywords IS NOT NULL"
        ):
            try:
                if any(needle in k.lower() for k in json.loads(row["keywords"])):
                    kw_ids.add(row["id"])
            except (ValueError, TypeError):
                pass
        result_sets.append(kw_ids)
```

- [ ] **Step 3: Update `api_photo_detail` to include the three new fields**

Find `api_photo_detail` in `photosearch/web.py`. In the response dict (search for `tags` being read into the response), add `categories`, `visual_tags`, `keywords` parallel reads (each is a JSON string column, parse with `json.loads(...) if value else []`). Remove the `tags` field from the response — the column is now always NULL.

- [ ] **Step 4: Smoke test**

```bash
curl -s 'http://localhost:8000/api/search?category=beach&limit=2'
curl -s 'http://localhost:8000/api/search?keyword=tahoe&limit=2'
curl -s 'http://localhost:8000/api/search?tag=beach&limit=2'  # legacy alias
```

Expected: all return valid JSON (empty until backfill).

- [ ] **Step 5: Commit**

```bash
git add photosearch/web.py photosearch/search.py
git commit -m "feat(api): category/visual_tag/keyword filters; tag→category redirect"
```

### Task 4.10: Stats endpoint update

**Files:**
- Modify: `photosearch/web.py:1878` — stats endpoint.

- [ ] **Step 1: Replace the `tagged` field with three new ones**

Find `tagged = db.conn.execute(...)` near line 1878. Replace with:

```python
        tagged_categories = db.conn.execute(
            "SELECT COUNT(*) FROM photos WHERE categories IS NOT NULL AND categories != '[]'"
        ).fetchone()[0]
        tagged_visual = db.conn.execute(
            "SELECT COUNT(*) FROM photos WHERE visual_tags IS NOT NULL AND visual_tags != '[]'"
        ).fetchone()[0]
        tagged_keywords = db.conn.execute(
            "SELECT COUNT(*) FROM photos WHERE keywords IS NOT NULL AND keywords != '[]'"
        ).fetchone()[0]
```

In the response dict (line ~1903), replace `"tagged": tagged,` with:

```python
        "category_tagged": tagged_categories,
        "visual_tagged": tagged_visual,
        "keyword_tagged": tagged_keywords,
```

- [ ] **Step 2: Smoke check**

```bash
curl -s http://localhost:8000/api/stats | venv/bin/python -m json.tool | grep -E "tagged|categor|visual|keyword"
```

Expected: three new keys; no `"tagged": ...` line.

- [ ] **Step 3: Commit**

```bash
git add photosearch/web.py
git commit -m "feat(api): stats endpoint — split tagged into three new counters"
```

---

## Phase 5 — Backfill (operational, no code)

**Goal:** Run the three new passes against the full library so the backend has data to test against in Phase 6.

### Task 5.1: Backfill on the NAS

- [ ] **Step 1: Pre-flight — full DB snapshot**

```bash
ssh nas.local
cd /volume1/docker/photosearch
DC="docker compose -f docker-compose.nas.yml run --rm"
$DC photosearch dump-db --to /data/pre-v23.db
ls -lh /volume1/docker/photosearch/data/pre-v23.db
```

Expected: snapshot file with size matching live DB.

- [ ] **Step 2: Launch backfill via worker fleet (Mac side)**

```bash
# Locally on Mac, with run-workers.sh:
./run-workers.sh -s http://<NAS-IP>:8000 -p category-content -n 4
./run-workers.sh -s http://<NAS-IP>:8000 -p keywords -n 4         # run sequentially after category-content
./run-workers.sh -s http://<NAS-IP>:8000 -p category-visual -n 3  # GPU-bound; smaller fleet
```

Monitor: `/status` page; workers panel shows progress.

Expected duration on 135k photos:
- `category-content`: 2–4h text passes.
- `keywords`: 2–4h.
- `category-visual`: ~3h with 3 Mac-native LLaVA workers.

- [ ] **Step 3: Verify coverage**

```bash
ssh nas.local 'curl -s http://localhost:8000/api/stats | python3 -m json.tool' | grep -E "category|visual|keyword"
```

Expected: `category_tagged` ≥ 95% of described photos; `keyword_tagged` similar; `visual_tagged` lower (vision-bound).

No commit — operational only.

---

## Phase 6 — Frontend

**Goal:** Photo modal shows three rows; search page has three filter controls; status page has three new stat cards.

### Task 6.1: PhotoModal three rows

**Files:**
- Modify: `frontend/dist/shared.js` — `PS.PhotoModal`.

- [ ] **Step 1: Find the existing tags row**

In `frontend/dist/shared.js`, search for `"Tags"` or `tags && tags.length`. The pattern is a section that renders a single row of clickable chips.

- [ ] **Step 2: Replace with three rows**

Replace the single row with:

```javascript
        photo.categories && photo.categories.length ? h('div', { className: 'modal-row' },
          h('span', { className: 'modal-label' }, 'Categories'),
          photo.categories.map(c => h('a', {
            key: c, className: 'modal-chip',
            href: '/?category=' + encodeURIComponent(c),
          }, c)),
        ) : null,
        photo.visual_tags && photo.visual_tags.length ? h('div', { className: 'modal-row' },
          h('span', { className: 'modal-label' }, 'Visual'),
          photo.visual_tags.map(c => h('a', {
            key: c, className: 'modal-chip',
            href: '/?visual_tag=' + encodeURIComponent(c),
          }, c)),
        ) : null,
        photo.keywords && photo.keywords.length ? h('div', { className: 'modal-row' },
          h('span', { className: 'modal-label' }, 'Keywords'),
          photo.keywords.map(c => h('a', {
            key: c, className: 'modal-chip modal-chip-kw',
            href: '/?keyword=' + encodeURIComponent(c),
          }, c)),
        ) : null,
```

Add a small CSS rule (in whichever `<style>` block `shared.js` is paired with or in each consuming page's style block):

```css
.modal-chip-kw { font-size: 11px; opacity: 0.85; border-style: dashed; }
```

- [ ] **Step 3: Manual verification**

Rebuild + restart NAS (or run uvicorn locally). Open any photo in the modal. Expected: three rows where the data exists; each row hides itself when empty. Clicking a category chip navigates to `/?category=...` and the search reloads with results.

- [ ] **Step 4: Commit**

```bash
git add frontend/dist/shared.js
git commit -m "feat(ui): PhotoModal — three rows for categories/visual/keywords"
```

### Task 6.2: Search page filter controls

**Files:**
- Modify: `frontend/dist/index.html` — toolbar area where filters live.

- [ ] **Step 1: Find the existing filter row**

In `frontend/dist/index.html`, search for `person` or `place` filter selects — there's a row of `<select>` / `<input>` controls above the result grid.

- [ ] **Step 2: Add three new controls**

Insert near the existing controls:

```html
<select id="category-filter" class="filter" onChange="applyFilters()">
  <option value="">All categories</option>
  <!-- populated by JS from /api/admin/vocab/draft or /vocab_content.py exposure -->
</select>
<select id="visual-filter" class="filter" onChange="applyFilters()">
  <option value="">All visual tags</option>
</select>
<input id="keyword-filter" class="filter" type="text" placeholder="keyword..." onInput="debouncedApply()">
```

Populate the two selects on page load:

```javascript
fetch('/api/admin/vocab/draft').then(r => r.json()).then(d => {
  const catSel = document.getElementById('category-filter');
  (d.content || []).sort().forEach(t => {
    const o = document.createElement('option');
    o.value = t; o.textContent = t;
    catSel.appendChild(o);
  });
  const visSel = document.getElementById('visual-filter');
  (d.visual || []).sort().forEach(t => {
    const o = document.createElement('option');
    o.value = t; o.textContent = t;
    visSel.appendChild(o);
  });
});
```

In `applyFilters()` / the search URL builder, include `category`, `visual_tag`, `keyword` parameters when non-empty.

- [ ] **Step 3: Manual verification**

Reload `/`. Expected:
- Three new controls visible.
- Selecting a category narrows results to that category.
- Typing in the keyword box (with debounce) narrows further.
- Combination is AND.

- [ ] **Step 4: Commit**

```bash
git add frontend/dist/index.html
git commit -m "feat(ui): search filter controls for category/visual/keyword"
```

### Task 6.3: Status page split

**Files:**
- Modify: `frontend/dist/status.html`

- [ ] **Step 1: Find the Tags stat card**

Search for `tagged` or `"Tags"` in the stat-grid React component.

- [ ] **Step 2: Replace with three cards**

Replace the single card definition with three parallel ones:

```javascript
        h(StatCard, { title: 'Categories', count: stats.category_tagged, total: stats.described,
                      hint: 'Photos with at least one content category.' }),
        h(StatCard, { title: 'Visual tags', count: stats.visual_tagged, total: stats.photos,
                      hint: 'Photos with at least one visual-quality tag.' }),
        h(StatCard, { title: 'Keywords',   count: stats.keyword_tagged, total: stats.described,
                      hint: 'Photos with at least one free-form keyword.' }),
```

- [ ] **Step 3: Update the run-command snippets**

Search for the row that lists `--tags`. Replace with three rows:

```
photosearch index /photos/YEAR --category-content
photosearch index /photos/YEAR --keywords
photosearch index /photos/YEAR --category-visual

# Or run all three:
photosearch index /photos/YEAR --category-content --keywords --category-visual
```

- [ ] **Step 4: Manual verification**

Reload `/status`. Expected: three cards in the stat grid; run-command section lists the three new commands.

- [ ] **Step 5: Commit**

```bash
git add frontend/dist/status.html
git commit -m "feat(ui): status page — split Tags card; new run snippets"
```

### Task 6.4: Delete dead code from describe.py and search.py exports

- [ ] **Step 1: Grep for remaining references**

```bash
grep -rn "tag_photo\|TAG_VOCABULARY\|TAG_PROMPT\|_QUERY_TO_TAGS\|_tags_match_query\|_parse_tag_response" \
  photosearch/ cli.py frontend/dist/ tests/ 2>/dev/null
```

Expected: zero matches in `photosearch/` and `cli.py` (only stale tests remain, if any).

- [ ] **Step 2: Address any remaining references**

Delete or rename each remaining hit. Common holdouts: `cli.py:clean-garbage-tags` (delete the whole subcommand — no longer relevant), any import of `tag_photo` in `worker.py` (replace with `tag_visual_photo`), the `--tags-model` arg on `worker` (deprecate in favour of `--category-visual-model`).

- [ ] **Step 3: Re-run the full test suite**

Run: `venv/bin/pytest`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove dead tag_photo / TAG_VOCABULARY / _QUERY_TO_TAGS"
```

---

## Phase 7 — Drop the backup column

**Goal:** After a week of confidence with the new fields, drop `tags_v22_backup` to reclaim space.

Schedule: at minimum 7 days after Phase 5 backfill completes and no regressions surface.

### Task 7.1: Schema v24 migration

**Files:**
- Modify: `photosearch/db.py`

- [ ] **Step 1: Bump SCHEMA_VERSION to 24**

```python
SCHEMA_VERSION = 24
```

- [ ] **Step 2: Append migration block**

After the v23 block in `_init_schema()`:

```python
        # Schema v24: drop tags_v22_backup + the now-unused `tags` column.
        # Reaches here only after >1 week of confidence (manual decision).
        if int(self._get_schema_version() or 0) < 24:
            try:
                cur.execute("SELECT tags_v22_backup FROM photos LIMIT 1")
            except sqlite3.OperationalError:
                pass  # already dropped
            else:
                # SQLite DROP COLUMN supported since 3.35 (Mar 2021). The
                # Docker python image's sqlite is well past that.
                cur.execute("ALTER TABLE photos DROP COLUMN tags_v22_backup")
                cur.execute("ALTER TABLE photos DROP COLUMN tags")
```

- [ ] **Step 3: Add test**

Append to `tests/test_db.py`:

```python
def test_schema_v24_drops_backup_and_tags(tmp_path):
    # Build a v23 DB by hand, open as v24, verify columns gone.
    import sqlite3
    db_path = tmp_path / "v23.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT, tags TEXT, tags_v22_backup TEXT,
            categories TEXT, visual_tags TEXT, keywords TEXT
        );
        CREATE TABLE schema_info (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO schema_info VALUES ('version', '23');
    """)
    conn.commit()
    conn.close()

    from photosearch.db import PhotoDB
    with PhotoDB(str(db_path)) as db:
        cols = {row[1] for row in db.conn.execute("PRAGMA table_info(photos)").fetchall()}
        assert "tags" not in cols
        assert "tags_v22_backup" not in cols
        assert "categories" in cols
```

- [ ] **Step 4: Run test + full suite**

Run: `venv/bin/pytest tests/test_db.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add photosearch/db.py tests/test_db.py
git commit -m "feat(db): schema v24 — drop tags + tags_v22_backup"
```

### Task 7.2: Deploy

- [ ] **Step 1: Use the deploy panel**

Open `/status`, scroll to the Deploy panel, click Build → wait for done → click Restart.

Verify the migration ran on container startup:

```bash
ssh nas.local 'docker logs photosearch 2>&1 | tail -50' | grep -i 'schema\|migrat'
```

Expected: a line confirming schema version 24.

- [ ] **Step 2: Spot-check the columns**

```bash
./debug-db.sh pull
./debug-db.sh query "PRAGMA table_info(photos);" | grep -E "tags|categories|visual|keywords"
```

Expected: `categories`, `visual_tags`, `keywords` present; `tags`, `tags_v22_backup` absent.

No commit needed (operational).

---

## Self-Review Notes

This plan was sanity-checked against the design spec on 2026-05-17. Open items left to executor judgement:

- **Bakeoff outcome wires into Task 4.2 Step 3.** If `llama3.2-vision` wins, also adjust the model arg on `extract_categories_from_description` (Task 4.1, Step 3) to match — Phase 4 lands as a single branch.
- **Curator UI is intentionally minimal.** If toggling 200+ terms one-at-a-time gets tedious during Task 3.4, add bucket headers from `vocab_proposal.json` in a follow-up tweak — not in this plan's scope.
- **Search ranking weights are starting guesses** (`keywords > categories > visual > description literal`). Tune empirically after Phase 5 backfill produces real data; record any change as a one-commit follow-up rather than re-entering this plan.
- **The `tag=X → category=X` and `tag_match → text_match` aliases** stay one release, then can be removed. Schedule a tiny cleanup task for the release after Phase 7.

---

## Execution Handoff

**Plan complete and saved to `docs/plans/categories-keywords-implementation.md`.**

Two execution options:

1. **Subagent-Driven (recommended for a plan this size)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.
