# What each context ablation removes

This is the exact change behind each row of the ablation results in
[FINDINGS.md](FINDINGS.md). It comes from `harness.ABLATIONS` and
`harness.apply_ablation`. File sizes are from the tree the runs used: `1429bb9`
for Sonnet and Opus, `35dcc9e` for Haiku. Between those two commits, 4 of these
files changed (24 lines added, 3 removed).

## How an ablation runs

`harness.py run --ablate <name>` checks out a detached worktree of the ref under
`/tmp`. It deletes the listed paths there and commits the deletion inside the
worktree. Then it runs all four suites in that worktree. The repo checkout is
never touched. Task grading diffs against the ablated commit, so the deletion
does not count as part of the model's change.

| ablation (config suffix) | short name in the tables | what is deleted | bytes removed |
|---|---|---|--:|
| `head` | none (baseline) | nothing | 0 |
| `no-rules` | rules | `.claude/rules/` (9 files) **and** the rule-injection hook in `.claude/settings.json` | 57,317 + hook |
| `no-notes` | notes | `docs/notes/` (14 files) | 244,165 |
| `no-rules-notes` | rules + notes | both of the above, plus the hook | 301,482 + hook |
| `no-reference` | runbooks | `.claude/skills/photo-search/reference/` (13 files). `SKILL.md` stays, so the skill can still be invoked, but the files it points to are gone. | 188,773 |
| `no-claude-md` | CLAUDE.md | `CLAUDE.md` | 10,484 |
| `bare` | everything | all of the above, plus the hook | 500,739 + hook |

Terms used in the tables:

- **rules**: path-scoped files in `.claude/rules/`. Claude Code loads one
  when the session opens a file that matches its `paths:`. A `PreToolUse` hook
  (`scripts/hooks/inject_rules.py`) also injects one when a Bash/Grep/Glob call
  names its subsystem.
- **notes**: `docs/notes/<subsystem>.md`. These hold history and
  measurements. Nothing loads them automatically. A session finds them only
  by search or by following a pointer.
- **runbooks**: the photo-search skill's `reference/` files, the "how to
  operate X" pages. `SKILL.md` is a short navigator that points at them.

## The hook change (every ablation that removes the rules)

`no-rules`, `no-rules-notes` and `bare` also drop the hook. Without that, the
hook would still try to inject rules that no longer exist.

```diff
--- a/.claude/settings.json
+++ b/.claude/settings.json
@@
-{
-  "hooks": {
-    "PreToolUse": [
-      {
-        "matcher": "Bash|Grep|Glob",
-        "hooks": [
-          {
-            "type": "command",
-            "command": "python3 \"$CLAUDE_PROJECT_DIR\"/scripts/hooks/inject_rules.py"
-          }
-        ]
-      }
-    ]
-  }
-}
+{}
```

## Deleted files, by layer

Each deletion removes the whole file. The diff for each file is its entire
content, so it is not repeated here. Read the files at `1429bb9` to see what a
run lost, e.g. `git show 1429bb9:.claude/rules/workers.md`.

**`.claude/rules/`** (rules): `faces.md` 5,944 · `frontend.md` 5,514 ·
`ingest.md` 6,075 · `llm-passes.md` 7,651 · `maintenance.md` 5,916 ·
`print.md` 5,611 · `search.md` 5,853 · `video.md` 7,797 · `workers.md` 6,956.

**`docs/notes/`** (notes): `README.md` 596 · `_archive-planned-milestones.md`
16,408 · `_headings-2026-10-19.txt` 6,833 · `faces.md` 22,540 · `frontend.md`
11,213 · `geotagging.md` 7,343 · `ingest.md` 38,114 · `llm-passes.md` 53,426 ·
`maintenance.md` 22,643 · `print.md` 21,932 · `search.md` 10,554 ·
`sharpness.md` 4,494 · `video.md` 18,751 · `workers.md` 9,318.

**`.claude/skills/photo-search/reference/`** (runbooks): `adding-features.md`
9,796 · `api-endpoints.md` 19,731 · `evals.md` 5,425 · `face-management-cli.md`
29,175 · `faces-review.md` 6,871 · `frontend-pages.md` 16,747 ·
`google-photos.md` 1,452 · `indexing-passes.md` 11,951 · `mcp-ask-replica.md`
14,401 · `nas-operations.md` 23,504 · `schema.md` 8,958 · `troubleshooting.md`
8,745 · `workers.md` 32,015.

**`CLAUDE.md`**: 10,484 bytes. With it gone, nothing in the session's starting
context points to the rules, notes or skill. The skill is still listed by
Claude Code, because the listing comes from `SKILL.md`'s description.

## What every ablation leaves in place, `bare` included

`bare` is **not** "code only". It removes the curated context layers. About
1.4 MB of other prose stays in the tree (Markdown and JSONL at `1429bb9`):

| still present | bytes | files |
|---|--:|--:|
| `docs/plans/` | 559,808 | 28 |
| `docs/superpowers/` (older plans) | 333,934 | 11 |
| `evals/context_eval/` (this eval: `recall.jsonl` with its expected patterns and `pass_example` answers, `REPORT.md`, `SUITES.md`) | 197,729 | 7 |
| `docs/HANDOFF-*.md` | 93,164 | 8 |
| `README.md`, `NAS_SETUP.md`, `GOOGLE_PHOTOS_SETUP.md` | 66,638 | 3 |
| `evals/ask_eval/` and `evals/category-grounding/` write-ups, two `.claude/skills/` files (incl. `SKILL.md`), one script README | 146,703 | 6 |

The eval's own answer key is in that list. The models do read it: see
"The `bare` floor is not code-only" in [FINDINGS.md](FINDINGS.md).

## Reproduce this page (offline)

```bash
python3 -c "import sys; sys.path.insert(0,'evals/context_eval'); import harness; print(harness.ABLATIONS)"
git ls-tree -r -l 1429bb9 -- CLAUDE.md .claude/rules docs/notes .claude/skills/photo-search/reference
git show 1429bb9:.claude/settings.json
```
