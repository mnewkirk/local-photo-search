# Context diet: the test suites, item by item

This report lists every test item in the context-diet evaluation and shows how each one did
before and after the change, for each model. "Before" is the repo at git tag `pre-context-diet`,
where Claude Code preloads the 258 KB `CLAUDE.md` and the 148 KB photo-search `SKILL.md`; "after"
is HEAD, where those are ~8 KB and ~6 KB and the detail lives in path-scoped
`.claude/rules/*.md`, `docs/notes/*.md` and `.claude/skills/photo-search/reference/*.md`.
Every number comes from the stored run rows in `results/<config>/<suite>.jsonl`, written by
`harness.py` and re-scored with the final graders. Costs are what `claude -p` reports at API
prices; the runs were billed to a Max plan, so the dollar figures are notional. For the
method, the per-model summaries and the conclusions, read [REPORT.md](REPORT.md).

## What each suite tests

All four suites drive Claude Code headless (`claude -p`), one fresh session per run, with MCP
servers disabled (`--strict-mcp-config`), so the photosearch MCP tools cannot answer for the
model. A session starts with whatever context the checked-out ref gives it: before the diet,
the full `CLAUDE.md`; after it, the slim `CLAUDE.md`, whose pointers the model must follow to
find the detail. Git writes and `gh` are blocked in every run since the Haiku task runs pushed commits on
2026-10-19 (the Haiku task suites were re-run with the block; the earlier runs had it unset). Each item ran 3 times per
config (231 runs per config: 77 items × 3). The local gemma model ran only the first 10 recall
items, once each.

### recall (49 items)

- **What the model is asked:** one direct question about a repo fact or trap, such as "What
  happens when the worker fleet is started with `-d /photos`?" Each item covers one section
  that the diet moved out of `CLAUDE.md`.
- **Context and tools:** an appended system prompt tells it that it is answering a maintainer's
  question about this repository, in at most 8 lines, naming exact identifiers, files, numbers
  or flags, without changing any file. It may use Read, Grep, Glob, Skill, Task and read-only
  Bash (`grep`, `cat`, `sed`, `git log`, ...).
- **How a pass is decided:** case-insensitive regexes run over the final answer. Every `expect`
  pattern must match, and each `any` group needs at least one match. A run that ends in an
  error (for example, hitting the turn cap) fails.
- **Why it matters for the diet:** before, the answer was already in the prompt; after, the
  model must notice it does not know and look the fact up. This suite measures whether the
  pointers in the slim `CLAUDE.md` still lead to the moved text.

### rules (9 items)

- **What the model is asked:** the harness prefixes every question with "First read the whole
  file `<path>` with the Read tool. Then answer: ...". Each item names one subsystem file
  (`photosearch/face_verify.py`, `photosearch/describe.py`, `frontend/dist/shared.js`, ...) and
  asks about a trap in that subsystem.
- **Context and tools:** the same system prompt, tools and grader as recall. After the diet,
  reading the file makes Claude Code load the matching path-scoped `.claude/rules/<name>.md`;
  before it, the answer comes from the preloaded `CLAUDE.md`.
- **How a pass is decided:** the same regex grading as recall.
- **Why it matters for the diet:** path-scoped rules only load when a matching file is opened.
  This suite checks that opening a subsystem file still brings that subsystem's trap knowledge
  into the session.

### skills (8 items)

- **What the model is asked:** one of the vague prompts that the photo-search skill's
  description promises to catch, such as "how do I index 2024?" or "recluster faces".
- **Context and tools:** the same system prompt and read-only tools as recall; the Skill tool
  and the photo-search skill are offered.
- **How a pass is decided:** the transcript must contain a Skill tool call whose skill name
  contains `photo-search`. The answer itself is not graded.
- **Why it matters for the diet:** after the diet the skill body is a short navigator, so the
  skill was meant to be a main route to the runbooks. In practice models almost never call it,
  before or after: they answer from `CLAUDE.md` and the reference files instead. REPORT.md notes
  that after the diet 7 of 8 Sonnet and 19 of 24 Opus skill answers cited the right
  `reference/` file without calling the skill. A low pass rate here therefore says the Skill
  tool went unused, not that the answer was wrong.

### tasks (11 items)

- **What the model is asked:** a small coding change in which a documented trap bites, such as
  "add a CLI command", "add a column to the photos table" or "write a script that waits for a
  background job".
- **Context and tools:** each run gets a fresh throwaway git worktree of the ref, with no
  project memory. A system prompt tells the model to make the change directly, not run tests,
  not commit and not ask questions, and to follow the repository's conventions. It may also use
  Edit and Write (`--permission-mode acceptEdits`), with a cap of 25 turns, $1.50 and 900 s.
- **How a pass is decided:** deterministic checks on the lines the run added (the `+` lines of
  `git diff HEAD` plus any new untracked file), optionally limited to named files. An
  `added_grep` pattern must appear, an `added_not_grep` pattern must not appear outside comments
  and docstrings, and an `any_added_grep` needs one of its patterns. A run fails if it moved
  HEAD (committed) or ended in an error such as hitting the turn cap. The checks test the
  convention, not whether the code works.
- **Why it matters for the diet:** this is the real adherence test. Conventions that used to be
  preloaded may now live only in a rule, which loads when a matching file is opened, or in a
  note the model has to find. The suite checks that the model still follows them when it
  changes code.

## Results per item

How to read the tables:

- **pass**: passing runs out of scored runs, before → after. ▲ marks an improvement and ▼ a
  regression.
- **cost/run**: mean notional `cost_usd` per run. For gemma it is "—": the model ran locally,
  and the cost Claude Code reports for it is meaningless.
- **wall/run**: mean elapsed seconds of one `claude -p` run.
- **$/s**: total cost ÷ total wall seconds for that item and config.
- The subtotal and total rows use the same formulas over all runs in the suite or config.
  There were no infrastructure failures left in the stored rows; every item has 3 runs per
  config (1 for gemma).

The four totals match REPORT.md: Sonnet 207 → 210 of 231 and $64.43 → $19.52; Opus 5.5
207 → 206 and $125.90 → $36.27; Haiku 4.5 197 → 197 and $28.05 → $11.28; gemma 10/10 → 9/10.

### Sonnet

| suite | test | what it checks | pass before → after | cost/run before → after | wall/run before → after | $/s before → after |
|---|---|---|---|---|---|---|
| recall | `cli-db-envvar` | Every CLI `--db` carries `envvar="PHOTOSEARCH_DB"` | 3/3 → 3/3 | $0.037 → $0.059 | 7.7 s → 5.3 s | $0.0048 → $0.0111 |
| recall | `db-filename-schema` | DB file name; `SCHEMA_VERSION` in `db.py` | 3/3 → 3/3 | $0.172 → $0.063 | 5.2 s → 4.3 s | $0.0333 → $0.0148 |
| recall | `frontend-no-build` | No JSX (`React.createElement`); frontend changes need an image rebuild | 3/3 → 3/3 | $0.174 → $0.026 | 6.3 s → 4.7 s | $0.0276 → $0.0056 |
| recall | `sse-loop-and-events` | SSE uses `get_running_loop`; emits done/fatal/cancelled | 3/3 → 3/3 | $0.173 → $0.026 | 6.0 s → 4.5 s | $0.0290 → $0.0056 |
| recall | `photodb-no-args` | `PhotoDB()` with no args opens an empty stub DB | 3/3 → 3/3 | $0.329 → $0.045 | 6.6 s → 4.1 s | $0.0501 → $0.0108 |
| recall | `compose-entrypoint` | `--entrypoint` goes before the service name | 3/3 → 3/3 | $0.331 → $0.026 | 7.9 s → 5.6 s | $0.0419 → $0.0047 |
| recall | `nas-git-safe-directory` | NAS git "dubious ownership" needs `safe.directory` | 3/3 → 3/3 | $0.172 → $0.025 | 4.5 s → 5.7 s | $0.0386 → $0.0045 |
| recall | `pgrep-self-match` | `pgrep -f` matches the watcher itself; watch a PID or marker | 3/3 → 3/3 | $0.331 → $0.025 | 5.5 s → 4.7 s | $0.0605 → $0.0054 |
| recall | `worker-d-photo-root` | `-d /photos` 404s; omit `-d` for the whole library | 3/3 → 3/3 | $0.174 → $0.060 | 5.2 s → 12.9 s | $0.0336 → $0.0047 |
| recall | `worker-health-probe` | Workers probe `/api/health`, not the slow `/api/stats` | 3/3 → 3/3 | $0.172 → $0.037 | 4.8 s → 7.1 s | $0.0362 → $0.0052 |
| recall | `tags-pass-removed` | `tags` pass is gone; names the three passes replacing it | 3/3 → 3/3 | $0.173 → $0.091 | 5.9 s → 12.8 s | $0.0291 → $0.0071 |
| recall | `aesthetics-primary-signal` | qwen2.5-vl scores aesthetics; LAION score tops out near 6.2 | 3/3 → 3/3 | $0.177 → $0.065 | 8.5 s → 11.9 s | $0.0207 → $0.0054 |
| recall | `percentile-chunking` | Percentile writes chunked for lock time, guarded on source score | 3/3 → 3/3 | $0.331 → $0.095 | 6.2 s → 13.9 s | $0.0536 → $0.0068 |
| recall | `visual-derived-tags` | Three EXIF-derived visual tags and their thresholds | 3/3 → 3/3 | $0.172 → $0.093 | 4.5 s → 12.4 s | $0.0385 → $0.0075 |
| recall | `visual-frozen-sharp-blurry` | `sharp`/`blurry` are frozen until a labelled eval decides | 3/3 → 3/3 | $0.333 → $0.069 | 8.5 s → 10.4 s | $0.0393 → $0.0066 |
| recall | `visual-retired-tag` | `motion-blur` was retired and stripped | 3/3 → 3/3 | $0.172 → $0.042 | 6.5 s → 7.2 s | $0.0265 → $0.0058 |
| recall | `visual-derived-wins` | EXIF strips a model's wrong `long-exposure` tag | 3/3 → 3/3 | $0.176 → $0.112 | 9.0 s → 14.5 s | $0.0196 → $0.0077 |
| recall | `lmstudio-visual-role-env` | LM Studio role env vars for visual and text passes | 3/3 → 3/3 | $0.329 → $0.040 | 5.5 s → 8.3 s | $0.0601 → $0.0049 |
| recall | `reasoning-effort-none` | Thinking models need reasoning effort `none`, else empty answers | 3/3 → 3/3 | $0.176 → $0.080 | 8.4 s → 13.0 s | $0.0209 → $0.0062 |
| recall | `fleet-role-models-location` | Role models live in `rerun.FLEET_ROLE_MODELS`; per-role override | 3/3 → 3/3 | $0.184 → $0.054 | 7.8 s → 9.4 s | $0.0236 → $0.0057 |
| recall | `provenance-effective-model` | `effective_model` logs the model that actually ran | 3/3 → 3/3 | $0.174 → $0.072 | 7.3 s → 11.3 s | $0.0237 → $0.0064 |
| recall | `text-pass-timeout` | 10 s timeout on text passes bounds Ollama stalls | 3/3 → 3/3 | $0.329 → $0.045 | 6.4 s → 9.0 s | $0.0512 → $0.0050 |
| recall | `attempts-cap` | 3-attempt cap in `worker_processed.attempts` | 3/3 → 3/3 | $0.173 → $0.064 | 4.4 s → 8.1 s | $0.0394 → $0.0078 |
| recall | `server-write-failure-attempt` | A transient DB lock on submit defers; no attempt spent | 3/3 → 3/3 | $0.333 → $0.083 | 9.6 s → 13.4 s | $0.0347 → $0.0062 |
| recall | `faces-empty-terminal` | Empty faces result is terminal; detection errors are failure rows | 3/3 → 3/3 | $0.175 → $0.058 | 6.2 s → 11.7 s | $0.0282 → $0.0049 |
| recall | `video-nas-never-decodes` | NAS never probes video; native workers run ffprobe | 3/3 → 3/3 | $0.332 → $0.044 | 8.4 s → 8.2 s | $0.0397 → $0.0053 |
| recall | `pass-media-map` | `PASS_MEDIA` limits passes by media; a missing pass raises | 3/3 → 3/3 | $0.182 → $0.060 | 9.9 s → 15.7 s | $0.0183 → $0.0038 |
| recall | `video-passes-native` | Video passes need `--native`: no ffprobe in the Docker image | 3/3 → 3/3 | $0.175 → $0.049 | 6.0 s → 7.4 s | $0.0292 → $0.0066 |
| recall | `clip-reclaim-cap` | Unloadable photos were re-claimed forever; clip now attempt-capped | 3/3 → 3/3 | $0.175 → $0.085 | 5.8 s → 15.5 s | $0.0300 → $0.0055 |
| recall | `ingest-no-index-trap` | `--no-index` leaves moved photos with no row; use `--no-clip` | 3/3 → 3/3 | $0.173 → $0.034 | 4.3 s → 6.6 s | $0.0399 → $0.0051 |
| recall | `refile-link-not-rename` | `os.link` fails with EEXIST instead of overwriting | 3/3 → 3/3 | $0.175 → $0.060 | 5.0 s → 10.9 s | $0.0352 → $0.0055 |
| recall | `maintenance-cron-time` | Nightly sweep runs at 18:30 from the NAS user's crontab | 3/3 → 3/3 | $0.173 → $0.070 | 4.5 s → 11.3 s | $0.0384 → $0.0062 |
| recall | `sweep-match-strict-only` | Sweep matches strict only; `--match-temporal` opts in | 3/3 → 3/3 | $0.173 → $0.070 | 5.8 s → 20.4 s | $0.0299 → $0.0034 |
| recall | `exclusions-churn-fix` | `face_person_exclusions` remembers face-person pairings, ending the churn | 3/3 → 3/3 | $0.175 → $0.056 | 5.4 s → 12.9 s | $0.0322 → $0.0043 |
| recall | `exclusion-strict-vs-temporal` | Strict falls through to next best; temporal leaves face unmatched | 3/3 → 3/3 | $0.175 → $0.057 | 6.3 s → 9.2 s | $0.0277 → $0.0062 |
| recall | `rejected-match-source` | `rejected` is a human "no" the matchers skip | 3/3 → 3/3 | $0.176 → $0.069 | 6.3 s → 12.0 s | $0.0279 → $0.0057 |
| recall | `rival-test` | Rival test: a better claimant in the same photo, even unlabelled | 3/3 → 3/3 | $0.175 → $0.056 | 7.0 s → 9.8 s | $0.0250 → $0.0057 |
| recall | `leave-one-burst-out` | Exclude the whole burst so a same-burst mislabel can't vouch | 3/3 → 3/3 | $0.174 → $0.038 | 5.9 s → 7.7 s | $0.0295 → $0.0049 |
| recall | `ps-poll-not-setinterval` | Poll with `PS.poll`; `setInterval` stacks requests on a slow server | 3/3 → 3/3 | $0.175 → $0.042 | 5.4 s → 9.5 s | $0.0321 → $0.0045 |
| recall | `frontend-ref-check` | `check-frontend-refs.js` catches calls to undefined helpers | 3/3 → 3/3 | $0.330 → $0.045 | 5.5 s → 8.6 s | $0.0596 → $0.0052 |
| recall | `restart-helper-container` | Restart runs compose in a detached helper container | 3/3 → 3/3 | $0.175 → $0.051 | 6.7 s → 11.3 s | $0.0261 → $0.0045 |
| recall | `batch-zero-unprocessed-lies` | Zero unprocessed can mean exhausted attempts or waiting on describe | 3/3 → 3/3 | $0.173 → $0.053 | 4.2 s → 10.0 s | $0.0408 → $0.0053 |
| recall | `batch-queued-source` | `queued` comes from `ingest_batch_jobs`, never fleet-status | 3/3 → 3/3 | $0.330 → $0.036 | 6.7 s → 7.1 s | $0.0489 → $0.0051 |
| recall | `detect-stacks-empty-list` | `photo_ids=[]` means the whole library and can wipe stacks | 3/3 → 3/3 | $0.345 → $0.063 | 9.5 s → 14.8 s | $0.0362 → $0.0042 |
| recall | `geocode-oom-per-row` | Geocode OOM was per-row dicts; population gate plus streaming fixed it | 3/3 → 3/3 | $0.175 → $0.048 | 6.4 s → 9.6 s | $0.0275 → $0.0049 |
| recall | `vec0-orphans` | vec0 tables don't cascade; `cleanup-orphans` removes leftovers | 3/3 → 3/3 | $0.174 → $0.063 | 5.4 s → 12.3 s | $0.0325 → $0.0051 |
| recall | `api-intent-headers` | Claude's curl calls send source and intent headers | 3/3 → 3/3 | $0.173 → $0.034 | 4.9 s → 7.5 s | $0.0354 → $0.0045 |
| recall | `sharpness-error-vs-absent` | Decode error is stored; an absent file records nothing | 3/3 → 3/3 | $0.174 → $0.071 | 5.5 s → 12.3 s | $0.0319 → $0.0058 |
| recall | `mirror-fields-needs-nas` | `mirrored:0` means the NAS lacks `/mirror-fields`; redeploy it | 3/3 → 3/3 | $0.174 → $0.053 | 5.1 s → 10.7 s | $0.0340 → $0.0050 |
| rules | `rule-faces-separation` | `separation` is a 5th percentile, robust to one bad label | 3/3 → 3/3 | $0.367 → $0.073 | 20.4 s → 10.0 s | $0.0180 → $0.0074 |
| rules | `rule-llm-unusable-answer` | `UnusableAnswer` spends an attempt; a transport timeout defers | 3/3 → 3/3 | $0.394 → $0.262 | 15.5 s → 25.6 s | $0.0254 → $0.0102 |
| rules | `rule-ingest-faces-done` | Only `faces` counts exhausted no-output photos as done | 3/3 → 3/3 | $0.263 → $0.089 | 10.8 s → 11.3 s | $0.0243 → $0.0079 |
| rules | `rule-maintenance-stacking-incremental` | Stacking is missing-only via `stacking_seen`; never clears the library | 3/3 → 3/3 | $0.285 → $0.111 | 11.5 s → 12.9 s | $0.0248 → $0.0086 |
| rules | `rule-workers-failures-list` | Worker exceptions go in `failures`; the server spends one attempt | 3/3 → 3/3 | $0.349 → $0.218 | 15.0 s → 18.7 s | $0.0233 → $0.0117 |
| rules | `rule-video-probe-owner` | Workers probe and scan video; the NAS stores and serves | 3/3 → 3/3 | $0.219 → $0.068 | 9.0 s → 11.2 s | $0.0243 → $0.0061 |
| rules | `rule-print-topaz-exit-code` | Topaz success means an output file appeared, not the exit code | 3/3 → 3/3 | $0.221 → $0.087 | 13.4 s → 10.6 s | $0.0165 → $0.0082 |
| rules | `rule-frontend-ps-poll` | `PS.poll` re-arms after settling; names the disk-starvation incident | 3/3 → 3/3 | $0.434 → $0.247 | 11.9 s → 17.1 s | $0.0364 → $0.0144 |
| rules | `rule-search-compose-scope` | `_compose_scope` folds date, camera and people into one id scope | 3/3 → 3/3 | $0.415 → $0.256 | 13.5 s → 10.7 s | $0.0308 → $0.0240 |
| skills | `skill-index-2024` | "how do I index 2024?" calls the photo-search skill | 0/3 → 0/3 | $0.188 → $0.077 | 11.0 s → 15.9 s | $0.0172 → $0.0048 |
| skills | `skill-search-slow` | "why is search slow?" calls the photo-search skill | 0/3 → 0/3 | $0.177 → $0.064 | 8.8 s → 10.2 s | $0.0202 → $0.0063 |
| skills | `skill-upload-google` | Google Photos upload question calls the photo-search skill | 0/3 → 0/3 | $0.240 → $0.072 | 17.5 s → 15.7 s | $0.0137 → $0.0046 |
| skills | `skill-add-filter` | New search filter request calls the photo-search skill | 0/3 → 0/3 | $0.204 → $0.056 | 12.7 s → 13.5 s | $0.0161 → $0.0041 |
| skills | `skill-last-queries` | "what were my last queries?" calls the photo-search skill | 0/3 → 3/3 ▲ | $0.294 → $0.065 | 29.8 s → 12.4 s | $0.0099 → $0.0053 |
| skills | `skill-fleet-laptop` | Laptop worker fleet question calls the photo-search skill | 0/3 → 0/3 | $0.331 → $0.064 | 6.2 s → 13.5 s | $0.0537 → $0.0047 |
| skills | `skill-recluster` | Bare "recluster faces" calls the photo-search skill | 0/3 → 0/3 | $0.332 → $0.046 | 9.0 s → 10.1 s | $0.0367 → $0.0045 |
| skills | `skill-schema-question` | "what tables are in photo_index.db?" calls the photo-search skill | 0/3 → 0/3 | $0.198 → $0.039 | 12.7 s → 8.6 s | $0.0156 → $0.0046 |
| tasks | `task-cli-envvar` | New CLI command's `--db` uses `envvar="PHOTOSEARCH_DB"` | 3/3 → 3/3 | $0.566 → $0.124 | 21.0 s → 26.7 s | $0.0270 → $0.0046 |
| tasks | `task-watcher-no-pgrep` | Wait script uses a PID or `wait`, never `pgrep -f` | 3/3 → 3/3 | $0.503 → $0.074 | 15.8 s → 11.6 s | $0.0319 → $0.0063 |
| tasks | `task-frontend-poll` | Health poll uses `PS.poll`, not `setInterval` | 3/3 → 3/3 | $0.585 → $0.139 | 27.9 s → 36.6 s | $0.0209 → $0.0038 |
| tasks | `task-nas-snippet-db-path` | NAS script passes `PHOTOSEARCH_DB` to `PhotoDB`, never `PhotoDB()` | 3/3 → 3/3 | $0.524 → $0.084 | 16.9 s → 14.5 s | $0.0310 → $0.0058 |
| tasks | `task-register-pass` | New pass lands in db.py, batch_state/rerun, batch-flow.js and cli.py | 3/3 → 3/3 | $0.733 → $0.663 | 48.9 s → 147.6 s | $0.0150 → $0.0045 |
| tasks | `task-sse-endpoint` | SSE endpoint avoids `get_event_loop()` and emits `done` | 3/3 → 3/3 | $0.576 → $0.125 | 24.4 s → 25.5 s | $0.0236 → $0.0049 |
| tasks | `task-curl-intent-headers` | Smoke-test curls send source and intent headers | 3/3 → 3/3 | $0.513 → $0.098 | 27.7 s → 21.1 s | $0.0185 → $0.0047 |
| tasks | `task-schema-bump` | New column bumps `SCHEMA_VERSION` and adds an `ADD COLUMN` migration | 3/3 → 3/3 | $0.610 → $0.173 | 33.7 s → 35.3 s | $0.0181 → $0.0049 |
| tasks | `task-filepath-writer` | Path change goes through `set_photo_filepath`, never a bare UPDATE | 3/3 → 3/3 | $0.563 → $0.128 | 26.3 s → 21.9 s | $0.0214 → $0.0058 |
| tasks | `task-restack-folder` | Restack passes `photo_ids` and guards the empty list | 3/3 → 3/3 | $0.556 → $0.149 | 22.6 s → 29.8 s | $0.0246 → $0.0050 |
| tasks | `task-nas-git-pull` | NAS git-pull script uses `safe.directory` | 3/3 → 3/3 | $0.533 → $0.098 | 22.0 s → 24.1 s | $0.0243 → $0.0041 |
| **recall** | **subtotal** |  | 147/147 → 147/147 | $0.210 → $0.056 | 6.3 s → 10.0 s | $0.0332 → $0.0057 |
| **rules** | **subtotal** |  | 27/27 → 27/27 | $0.327 → $0.157 | 13.5 s → 14.2 s | $0.0243 → $0.0110 |
| **skills** | **subtotal** |  | 0/24 → 3/24 ▲ | $0.246 → $0.060 | 13.5 s → 12.5 s | $0.0182 → $0.0048 |
| **tasks** | **subtotal** |  | 33/33 → 33/33 | $0.569 → $0.169 | 26.1 s → 35.9 s | $0.0218 → $0.0047 |
| **all** | **total** |  | 207/231 → 210/231 ▲ | $0.279 → $0.085 | 10.7 s → 14.4 s | $0.0260 → $0.0059 |

### Opus 5.5

| suite | test | what it checks | pass before → after | cost/run before → after | wall/run before → after | $/s before → after |
|---|---|---|---|---|---|---|
| recall | `cli-db-envvar` | Every CLI `--db` carries `envvar="PHOTOSEARCH_DB"` | 3/3 → 3/3 | $0.634 → $0.097 | 10.1 s → 7.3 s | $0.0628 → $0.0134 |
| recall | `db-filename-schema` | DB file name; `SCHEMA_VERSION` in `db.py` | 3/3 → 3/3 | $0.362 → $0.060 | 7.8 s → 8.6 s | $0.0464 → $0.0070 |
| recall | `frontend-no-build` | No JSX (`React.createElement`); frontend changes need an image rebuild | 3/3 → 3/3 | $0.644 → $0.083 | 8.7 s → 10.4 s | $0.0738 → $0.0080 |
| recall | `sse-loop-and-events` | SSE uses `get_running_loop`; emits done/fatal/cancelled | 3/3 → 3/3 | $0.332 → $0.056 | 6.3 s → 6.9 s | $0.0530 → $0.0081 |
| recall | `photodb-no-args` | `PhotoDB()` with no args opens an empty stub DB | 3/3 → 3/3 | $0.334 → $0.064 | 5.8 s → 9.6 s | $0.0580 → $0.0066 |
| recall | `compose-entrypoint` | `--entrypoint` goes before the service name | 3/3 → 3/3 | $0.633 → $0.052 | 7.6 s → 5.2 s | $0.0833 → $0.0101 |
| recall | `nas-git-safe-directory` | NAS git "dubious ownership" needs `safe.directory` | 3/3 → 3/3 | $0.331 → $0.069 | 4.9 s → 9.0 s | $0.0675 → $0.0077 |
| recall | `pgrep-self-match` | `pgrep -f` matches the watcher itself; watch a PID or marker | 3/3 → 3/3 | $0.635 → $0.130 | 9.5 s → 12.9 s | $0.0669 → $0.0101 |
| recall | `worker-d-photo-root` | `-d /photos` 404s; omit `-d` for the whole library | 3/3 → 3/3 | $0.346 → $0.077 | 8.2 s → 8.5 s | $0.0422 → $0.0091 |
| recall | `worker-health-probe` | Workers probe `/api/health`, not the slow `/api/stats` | 3/3 → 3/3 | $0.357 → $0.105 | 10.5 s → 11.4 s | $0.0341 → $0.0092 |
| recall | `tags-pass-removed` | `tags` pass is gone; names the three passes replacing it | 3/3 → 3/3 | $0.365 → $0.160 | 6.1 s → 16.2 s | $0.0598 → $0.0099 |
| recall | `aesthetics-primary-signal` | qwen2.5-vl scores aesthetics; LAION score tops out near 6.2 | 3/3 → 3/3 | $0.342 → $0.163 | 11.3 s → 15.8 s | $0.0302 → $0.0103 |
| recall | `percentile-chunking` | Percentile writes chunked for lock time, guarded on source score | 3/3 → 3/3 | $0.667 → $0.134 | 8.5 s → 13.7 s | $0.0787 → $0.0098 |
| recall | `visual-derived-tags` | Three EXIF-derived visual tags and their thresholds | 3/3 → 3/3 | $0.678 → $0.137 | 10.0 s → 13.4 s | $0.0676 → $0.0102 |
| recall | `visual-frozen-sharp-blurry` | `sharp`/`blurry` are frozen until a labelled eval decides | 3/3 → 3/3 | $0.672 → $0.171 | 12.0 s → 11.3 s | $0.0562 → $0.0152 |
| recall | `visual-retired-tag` | `motion-blur` was retired and stripped | 3/3 → 3/3 | $0.344 → $0.085 | 6.4 s → 12.4 s | $0.0535 → $0.0069 |
| recall | `visual-derived-wins` | EXIF strips a model's wrong `long-exposure` tag | 3/3 → 3/3 | $0.669 → $0.210 | 11.2 s → 18.6 s | $0.0598 → $0.0113 |
| recall | `lmstudio-visual-role-env` | LM Studio role env vars for visual and text passes | 3/3 → 3/3 | $0.335 → $0.123 | 6.3 s → 10.0 s | $0.0529 → $0.0123 |
| recall | `reasoning-effort-none` | Thinking models need reasoning effort `none`, else empty answers | 3/3 → 3/3 | $0.651 → $0.184 | 10.5 s → 15.5 s | $0.0618 → $0.0118 |
| recall | `fleet-role-models-location` | Role models live in `rerun.FLEET_ROLE_MODELS`; per-role override | 3/3 → 3/3 | $0.416 → $0.128 | 11.0 s → 13.2 s | $0.0378 → $0.0097 |
| recall | `provenance-effective-model` | `effective_model` logs the model that actually ran | 3/3 → 3/3 | $0.368 → $0.087 | 9.9 s → 10.2 s | $0.0373 → $0.0085 |
| recall | `text-pass-timeout` | 10 s timeout on text passes bounds Ollama stalls | 3/3 → 3/3 | $0.724 → $0.103 | 14.3 s → 11.4 s | $0.0507 → $0.0090 |
| recall | `attempts-cap` | 3-attempt cap in `worker_processed.attempts` | 3/3 → 3/3 | $0.367 → $0.109 | 11.9 s → 11.3 s | $0.0309 → $0.0096 |
| recall | `server-write-failure-attempt` | A transient DB lock on submit defers; no attempt spent | 3/3 → 3/3 | $0.369 → $0.150 | 9.5 s → 16.2 s | $0.0388 → $0.0093 |
| recall | `faces-empty-terminal` | Empty faces result is terminal; detection errors are failure rows | 3/3 → 3/3 | $0.367 → $0.122 | 8.5 s → 14.7 s | $0.0434 → $0.0083 |
| recall | `video-nas-never-decodes` | NAS never probes video; native workers run ffprobe | 3/3 → 3/3 | $0.339 → $0.073 | 8.4 s → 8.4 s | $0.0405 → $0.0088 |
| recall | `pass-media-map` | `PASS_MEDIA` limits passes by media; a missing pass raises | 3/3 → 3/3 | $0.383 → $0.099 | 13.0 s → 11.3 s | $0.0294 → $0.0088 |
| recall | `video-passes-native` | Video passes need `--native`: no ffprobe in the Docker image | 3/3 → 3/3 | $0.360 → $0.125 | 9.7 s → 11.8 s | $0.0373 → $0.0106 |
| recall | `clip-reclaim-cap` | Unloadable photos were re-claimed forever; clip now attempt-capped | 3/3 → 3/3 | $0.670 → $0.107 | 11.8 s → 18.3 s | $0.0569 → $0.0059 |
| recall | `ingest-no-index-trap` | `--no-index` leaves moved photos with no row; use `--no-clip` | 3/3 → 3/3 | $0.333 → $0.070 | 7.1 s → 8.6 s | $0.0467 → $0.0081 |
| recall | `refile-link-not-rename` | `os.link` fails with EEXIST instead of overwriting | 3/3 → 3/3 | $0.338 → $0.113 | 7.4 s → 12.5 s | $0.0454 → $0.0090 |
| recall | `maintenance-cron-time` | Nightly sweep runs at 18:30 from the NAS user's crontab | 3/3 → 3/3 | $0.335 → $0.087 | 7.0 s → 10.7 s | $0.0479 → $0.0082 |
| recall | `sweep-match-strict-only` | Sweep matches strict only; `--match-temporal` opts in | 3/3 → 3/3 | $0.366 → $0.143 | 9.3 s → 14.1 s | $0.0395 → $0.0101 |
| recall | `exclusions-churn-fix` | `face_person_exclusions` remembers face-person pairings, ending the churn | 3/3 → 3/3 | $0.337 → $0.083 | 13.0 s → 17.7 s | $0.0258 → $0.0047 |
| recall | `exclusion-strict-vs-temporal` | Strict falls through to next best; temporal leaves face unmatched | 3/3 → 3/3 | $0.338 → $0.116 | 9.6 s → 17.0 s | $0.0354 → $0.0068 |
| recall | `rejected-match-source` | `rejected` is a human "no" the matchers skip | 3/3 → 3/3 | $0.371 → $0.121 | 10.2 s → 16.4 s | $0.0364 → $0.0073 |
| recall | `rival-test` | Rival test: a better claimant in the same photo, even unlabelled | 3/3 → 3/3 | $0.359 → $0.077 | 11.1 s → 12.1 s | $0.0323 → $0.0064 |
| recall | `leave-one-burst-out` | Exclude the whole burst so a same-burst mislabel can't vouch | 3/3 → 3/3 | $0.367 → $0.088 | 9.5 s → 10.9 s | $0.0386 → $0.0081 |
| recall | `ps-poll-not-setinterval` | Poll with `PS.poll`; `setInterval` stacks requests on a slow server | 3/3 → 3/3 | $0.346 → $0.077 | 6.9 s → 11.4 s | $0.0499 → $0.0068 |
| recall | `frontend-ref-check` | `check-frontend-refs.js` catches calls to undefined helpers | 3/3 → 3/3 | $0.353 → $0.099 | 7.2 s → 9.6 s | $0.0488 → $0.0104 |
| recall | `restart-helper-container` | Restart runs compose in a detached helper container | 3/3 → 3/3 | $0.338 → $0.102 | 7.2 s → 13.6 s | $0.0471 → $0.0075 |
| recall | `batch-zero-unprocessed-lies` | Zero unprocessed can mean exhausted attempts or waiting on describe | 3/3 → 3/3 | $0.334 → $0.098 | 5.8 s → 8.6 s | $0.0580 → $0.0114 |
| recall | `batch-queued-source` | `queued` comes from `ingest_batch_jobs`, never fleet-status | 3/3 → 3/3 | $0.336 → $0.062 | 7.4 s → 8.0 s | $0.0452 → $0.0078 |
| recall | `detect-stacks-empty-list` | `photo_ids=[]` means the whole library and can wipe stacks | 3/3 → 3/3 | $0.388 → $0.165 | 9.8 s → 17.9 s | $0.0397 → $0.0092 |
| recall | `geocode-oom-per-row` | Geocode OOM was per-row dicts; population gate plus streaming fixed it | 3/3 → 3/3 | $0.338 → $0.085 | 7.1 s → 10.6 s | $0.0476 → $0.0080 |
| recall | `vec0-orphans` | vec0 tables don't cascade; `cleanup-orphans` removes leftovers | 3/3 → 3/3 | $0.646 → $0.121 | 10.9 s → 19.2 s | $0.0594 → $0.0063 |
| recall | `api-intent-headers` | Claude's curl calls send source and intent headers | 3/3 → 3/3 | $0.334 → $0.082 | 7.0 s → 9.7 s | $0.0475 → $0.0084 |
| recall | `sharpness-error-vs-absent` | Decode error is stored; an absent file records nothing | 3/3 → 3/3 | $0.664 → $0.097 | 11.7 s → 12.0 s | $0.0566 → $0.0081 |
| recall | `mirror-fields-needs-nas` | `mirrored:0` means the NAS lacks `/mirror-fields`; redeploy it | 3/3 → 3/3 | $0.358 → $0.080 | 8.5 s → 9.8 s | $0.0420 → $0.0082 |
| rules | `rule-faces-separation` | `separation` is a 5th percentile, robust to one bad label | 3/3 → 3/3 | $0.405 → $0.099 | 12.4 s → 10.7 s | $0.0326 → $0.0093 |
| rules | `rule-llm-unusable-answer` | `UnusableAnswer` spends an attempt; a transport timeout defers | 3/3 → 3/3 | $0.631 → $0.343 | 13.7 s → 17.2 s | $0.0461 → $0.0199 |
| rules | `rule-ingest-faces-done` | Only `faces` counts exhausted no-output photos as done | 3/3 → 3/3 | $0.428 → $0.136 | 13.4 s → 12.8 s | $0.0320 → $0.0106 |
| rules | `rule-maintenance-stacking-incremental` | Stacking is missing-only via `stacking_seen`; never clears the library | 3/3 → 3/3 | $0.484 → $0.210 | 11.1 s → 18.0 s | $0.0435 → $0.0117 |
| rules | `rule-workers-failures-list` | Worker exceptions go in `failures`; the server spends one attempt | 3/3 → 3/3 | $0.621 → $0.298 | 15.1 s → 17.1 s | $0.0412 → $0.0174 |
| rules | `rule-video-probe-owner` | Workers probe and scan video; the NAS stores and serves | 3/3 → 3/3 | $0.405 → $0.113 | 10.9 s → 9.6 s | $0.0373 → $0.0117 |
| rules | `rule-print-topaz-exit-code` | Topaz success means an output file appeared, not the exit code | 3/3 → 3/3 | $0.404 → $0.135 | 11.6 s → 11.5 s | $0.0347 → $0.0117 |
| rules | `rule-frontend-ps-poll` | `PS.poll` re-arms after settling; names the disk-starvation incident | 3/3 → 3/3 | $0.683 → $0.447 | 16.5 s → 16.7 s | $0.0415 → $0.0267 |
| rules | `rule-search-compose-scope` | `_compose_scope` folds date, camera and people into one id scope | 3/3 → 3/3 | $0.611 → $0.369 | 18.4 s → 11.9 s | $0.0333 → $0.0311 |
| skills | `skill-index-2024` | "how do I index 2024?" calls the photo-search skill | 0/3 → 0/3 | $0.361 → $0.087 | 9.7 s → 12.7 s | $0.0372 → $0.0068 |
| skills | `skill-search-slow` | "why is search slow?" calls the photo-search skill | 0/3 → 0/3 | $0.371 → $0.122 | 17.4 s → 14.9 s | $0.0213 → $0.0082 |
| skills | `skill-upload-google` | Google Photos upload question calls the photo-search skill | 0/3 → 0/3 | $0.543 → $0.148 | 22.6 s → 19.4 s | $0.0240 → $0.0076 |
| skills | `skill-add-filter` | New search filter request calls the photo-search skill | 0/3 → 0/3 | $0.393 → $0.083 | 11.7 s → 13.4 s | $0.0337 → $0.0062 |
| skills | `skill-last-queries` | "what were my last queries?" calls the photo-search skill | 0/3 → 0/3 | $0.492 → $0.157 | 22.7 s → 64.8 s | $0.0217 → $0.0024 |
| skills | `skill-fleet-laptop` | Laptop worker fleet question calls the photo-search skill | 0/3 → 0/3 | $0.339 → $0.173 | 8.6 s → 19.5 s | $0.0392 → $0.0088 |
| skills | `skill-recluster` | Bare "recluster faces" calls the photo-search skill | 0/3 → 0/3 | $0.672 → $0.110 | 11.9 s → 14.8 s | $0.0566 → $0.0075 |
| skills | `skill-schema-question` | "what tables are in photo_index.db?" calls the photo-search skill | 0/3 → 0/3 | $0.461 → $0.098 | 15.2 s → 12.3 s | $0.0304 → $0.0080 |
| tasks | `task-cli-envvar` | New CLI command's `--db` uses `envvar="PHOTOSEARCH_DB"` | 3/3 → 3/3 | $1.058 → $0.224 | 18.3 s → 15.9 s | $0.0578 → $0.0141 |
| tasks | `task-watcher-no-pgrep` | Wait script uses a PID or `wait`, never `pgrep -f` | 3/3 → 3/3 | $1.014 → $0.170 | 12.5 s → 13.9 s | $0.0814 → $0.0122 |
| tasks | `task-frontend-poll` | Health poll uses `PS.poll`, not `setInterval` | 3/3 → 3/3 | $1.117 → $0.291 | 31.7 s → 24.1 s | $0.0352 → $0.0120 |
| tasks | `task-nas-snippet-db-path` | NAS script passes `PHOTOSEARCH_DB` to `PhotoDB`, never `PhotoDB()` | 3/3 → 3/3 | $1.037 → $0.188 | 18.8 s → 32.7 s | $0.0550 → $0.0057 |
| tasks | `task-register-pass` | New pass lands in db.py, batch_state/rerun, batch-flow.js and cli.py | 3/3 → 2/3 ▼ | $1.390 → $1.196 | 62.6 s → 126.4 s | $0.0222 → $0.0095 |
| tasks | `task-sse-endpoint` | SSE endpoint avoids `get_event_loop()` and emits `done` | 3/3 → 3/3 | $1.166 → $0.287 | 27.4 s → 37.6 s | $0.0426 → $0.0076 |
| tasks | `task-curl-intent-headers` | Smoke-test curls send source and intent headers | 3/3 → 3/3 | $1.095 → $0.215 | 30.0 s → 25.2 s | $0.0365 → $0.0085 |
| tasks | `task-schema-bump` | New column bumps `SCHEMA_VERSION` and adds an `ADD COLUMN` migration | 3/3 → 3/3 | $1.210 → $0.355 | 30.8 s → 36.2 s | $0.0393 → $0.0098 |
| tasks | `task-filepath-writer` | Path change goes through `set_photo_filepath`, never a bare UPDATE | 3/3 → 3/3 | $1.150 → $0.315 | 37.0 s → 45.6 s | $0.0311 → $0.0069 |
| tasks | `task-restack-folder` | Restack passes `photo_ids` and guards the empty list | 3/3 → 3/3 | $1.088 → $0.284 | 21.8 s → 28.6 s | $0.0498 → $0.0099 |
| tasks | `task-nas-git-pull` | NAS git-pull script uses `safe.directory` | 3/3 → 3/3 | $1.059 → $0.206 | 22.8 s → 22.5 s | $0.0465 → $0.0092 |
| **recall** | **subtotal** |  | 147/147 → 147/147 | $0.434 → $0.107 | 9.0 s → 12.1 s | $0.0480 → $0.0088 |
| **rules** | **subtotal** |  | 27/27 → 27/27 | $0.519 → $0.239 | 13.7 s → 14.0 s | $0.0380 → $0.0171 |
| **skills** | **subtotal** |  | 0/24 → 0/24 | $0.454 → $0.122 | 15.0 s → 21.5 s | $0.0303 → $0.0057 |
| **tasks** | **subtotal** |  | 33/33 → 32/33 ▼ | $1.126 → $0.339 | 28.5 s → 37.2 s | $0.0395 → $0.0091 |
| **all** | **total** |  | 207/231 → 206/231 ▼ | $0.545 → $0.157 | 13.0 s → 16.9 s | $0.0420 → $0.0093 |

### Haiku 4.5

| suite | test | what it checks | pass before → after | cost/run before → after | wall/run before → after | $/s before → after |
|---|---|---|---|---|---|---|
| recall | `cli-db-envvar` | Every CLI `--db` carries `envvar="PHOTOSEARCH_DB"` | 3/3 → 3/3 | $0.119 → $0.016 | 8.6 s → 6.1 s | $0.0139 → $0.0026 |
| recall | `db-filename-schema` | DB file name; `SCHEMA_VERSION` in `db.py` | 3/3 → 3/3 | $0.065 → $0.010 | 5.3 s → 4.5 s | $0.0122 → $0.0021 |
| recall | `frontend-no-build` | No JSX (`React.createElement`); frontend changes need an image rebuild | 3/3 → 3/3 | $0.065 → $0.010 | 7.3 s → 7.0 s | $0.0090 → $0.0015 |
| recall | `sse-loop-and-events` | SSE uses `get_running_loop`; emits done/fatal/cancelled | 3/3 → 3/3 | $0.066 → $0.016 | 6.3 s → 8.3 s | $0.0104 → $0.0019 |
| recall | `photodb-no-args` | `PhotoDB()` with no args opens an empty stub DB | 3/3 → 3/3 | $0.120 → $0.011 | 8.2 s → 5.9 s | $0.0146 → $0.0018 |
| recall | `compose-entrypoint` | `--entrypoint` goes before the service name | 3/3 → 3/3 | $0.068 → $0.011 | 12.0 s → 6.4 s | $0.0057 → $0.0017 |
| recall | `nas-git-safe-directory` | NAS git "dubious ownership" needs `safe.directory` | 3/3 → 3/3 | $0.066 → $0.011 | 7.1 s → 7.4 s | $0.0093 → $0.0015 |
| recall | `pgrep-self-match` | `pgrep -f` matches the watcher itself; watch a PID or marker | 3/3 → 3/3 | $0.068 → $0.010 | 13.6 s → 5.8 s | $0.0050 → $0.0018 |
| recall | `worker-d-photo-root` | `-d /photos` 404s; omit `-d` for the whole library | 3/3 → 3/3 | $0.067 → $0.024 | 8.4 s → 13.7 s | $0.0080 → $0.0017 |
| recall | `worker-health-probe` | Workers probe `/api/health`, not the slow `/api/stats` | 3/3 → 3/3 | $0.066 → $0.037 | 8.8 s → 16.0 s | $0.0076 → $0.0023 |
| recall | `tags-pass-removed` | `tags` pass is gone; names the three passes replacing it | 3/3 → 2/3 ▼ | $0.066 → $0.050 | 7.2 s → 22.2 s | $0.0092 → $0.0022 |
| recall | `aesthetics-primary-signal` | qwen2.5-vl scores aesthetics; LAION score tops out near 6.2 | 3/3 → 3/3 | $0.074 → $0.041 | 26.8 s → 21.5 s | $0.0028 → $0.0019 |
| recall | `percentile-chunking` | Percentile writes chunked for lock time, guarded on source score | 3/3 → 3/3 | $0.070 → $0.035 | 16.4 s → 17.2 s | $0.0043 → $0.0020 |
| recall | `visual-derived-tags` | Three EXIF-derived visual tags and their thresholds | 3/3 → 3/3 | $0.068 → $0.034 | 12.4 s → 12.4 s | $0.0055 → $0.0027 |
| recall | `visual-frozen-sharp-blurry` | `sharp`/`blurry` are frozen until a labelled eval decides | 3/3 → 3/3 | $0.069 → $0.046 | 14.6 s → 21.3 s | $0.0047 → $0.0022 |
| recall | `visual-retired-tag` | `motion-blur` was retired and stripped | 3/3 → 2/3 ▼ | $0.065 → $0.034 | 7.2 s → 14.5 s | $0.0091 → $0.0023 |
| recall | `visual-derived-wins` | EXIF strips a model's wrong `long-exposure` tag | 3/3 → 3/3 | $0.069 → $0.045 | 15.0 s → 24.3 s | $0.0046 → $0.0019 |
| recall | `lmstudio-visual-role-env` | LM Studio role env vars for visual and text passes | 3/3 → 3/3 | $0.066 → $0.020 | 6.3 s → 12.6 s | $0.0104 → $0.0016 |
| recall | `reasoning-effort-none` | Thinking models need reasoning effort `none`, else empty answers | 3/3 → 3/3 | $0.069 → $0.023 | 16.6 s → 12.5 s | $0.0042 → $0.0018 |
| recall | `fleet-role-models-location` | Role models live in `rerun.FLEET_ROLE_MODELS`; per-role override | 3/3 → 3/3 | $0.093 → $0.043 | 16.8 s → 17.9 s | $0.0056 → $0.0024 |
| recall | `provenance-effective-model` | `effective_model` logs the model that actually ran | 3/3 → 3/3 | $0.068 → $0.041 | 11.4 s → 21.9 s | $0.0059 → $0.0018 |
| recall | `text-pass-timeout` | 10 s timeout on text passes bounds Ollama stalls | 3/3 → 3/3 | $0.067 → $0.037 | 11.7 s → 16.9 s | $0.0058 → $0.0022 |
| recall | `attempts-cap` | 3-attempt cap in `worker_processed.attempts` | 3/3 → 3/3 | $0.067 → $0.035 | 10.9 s → 14.0 s | $0.0061 → $0.0025 |
| recall | `server-write-failure-attempt` | A transient DB lock on submit defers; no attempt spent | 3/3 → 2/3 ▼ | $0.073 → $0.068 | 30.3 s → 26.3 s | $0.0024 → $0.0026 |
| recall | `faces-empty-terminal` | Empty faces result is terminal; detection errors are failure rows | 3/3 → 2/3 ▼ | $0.067 → $0.055 | 11.2 s → 16.6 s | $0.0060 → $0.0033 |
| recall | `video-nas-never-decodes` | NAS never probes video; native workers run ffprobe | 3/3 → 3/3 | $0.069 → $0.043 | 14.6 s → 14.2 s | $0.0047 → $0.0030 |
| recall | `pass-media-map` | `PASS_MEDIA` limits passes by media; a missing pass raises | 3/3 → 3/3 | $0.066 → $0.037 | 8.5 s → 14.7 s | $0.0078 → $0.0025 |
| recall | `video-passes-native` | Video passes need `--native`: no ffprobe in the Docker image | 3/3 → 3/3 | $0.121 → $0.021 | 11.3 s → 12.1 s | $0.0106 → $0.0017 |
| recall | `clip-reclaim-cap` | Unloadable photos were re-claimed forever; clip now attempt-capped | 2/3 → 3/3 ▲ | $0.064 → $0.063 | 14.5 s → 23.9 s | $0.0044 → $0.0027 |
| recall | `ingest-no-index-trap` | `--no-index` leaves moved photos with no row; use `--no-clip` | 3/3 → 3/3 | $0.178 → $0.032 | 13.0 s → 11.2 s | $0.0138 → $0.0028 |
| recall | `refile-link-not-rename` | `os.link` fails with EEXIST instead of overwriting | 3/3 → 3/3 | $0.069 → $0.066 | 16.0 s → 16.0 s | $0.0043 → $0.0041 |
| recall | `maintenance-cron-time` | Nightly sweep runs at 18:30 from the NAS user's crontab | 3/3 → 3/3 | $0.069 → $0.031 | 13.6 s → 11.1 s | $0.0050 → $0.0028 |
| recall | `sweep-match-strict-only` | Sweep matches strict only; `--match-temporal` opts in | 3/3 → 3/3 | $0.067 → $0.029 | 9.8 s → 12.9 s | $0.0068 → $0.0023 |
| recall | `exclusions-churn-fix` | `face_person_exclusions` remembers face-person pairings, ending the churn | 3/3 → 3/3 | $0.068 → $0.026 | 13.4 s → 12.3 s | $0.0051 → $0.0021 |
| recall | `exclusion-strict-vs-temporal` | Strict falls through to next best; temporal leaves face unmatched | 3/3 → 3/3 | $0.071 → $0.058 | 20.1 s → 25.2 s | $0.0035 → $0.0023 |
| recall | `rejected-match-source` | `rejected` is a human "no" the matchers skip | 3/3 → 3/3 | $0.070 → $0.047 | 16.8 s → 21.3 s | $0.0042 → $0.0022 |
| recall | `rival-test` | Rival test: a better claimant in the same photo, even unlabelled | 3/3 → 3/3 | $0.124 → $0.031 | 20.7 s → 15.0 s | $0.0060 → $0.0021 |
| recall | `leave-one-burst-out` | Exclude the whole burst so a same-burst mislabel can't vouch | 3/3 → 3/3 | $0.068 → $0.062 | 13.0 s → 20.2 s | $0.0052 → $0.0031 |
| recall | `ps-poll-not-setinterval` | Poll with `PS.poll`; `setInterval` stacks requests on a slow server | 3/3 → 2/3 ▼ | $0.069 → $0.020 | 14.5 s → 13.2 s | $0.0048 → $0.0015 |
| recall | `frontend-ref-check` | `check-frontend-refs.js` catches calls to undefined helpers | 3/3 → 3/3 | $0.067 → $0.025 | 10.4 s → 12.0 s | $0.0064 → $0.0021 |
| recall | `restart-helper-container` | Restart runs compose in a detached helper container | 3/3 → 3/3 | $0.067 → $0.061 | 11.4 s → 12.2 s | $0.0059 → $0.0050 |
| recall | `batch-zero-unprocessed-lies` | Zero unprocessed can mean exhausted attempts or waiting on describe | 3/3 → 3/3 | $0.120 → $0.067 | 8.3 s → 13.7 s | $0.0144 → $0.0049 |
| recall | `batch-queued-source` | `queued` comes from `ingest_batch_jobs`, never fleet-status | 3/3 → 3/3 | $0.067 → $0.018 | 9.8 s → 7.2 s | $0.0069 → $0.0026 |
| recall | `detect-stacks-empty-list` | `photo_ids=[]` means the whole library and can wipe stacks | 3/3 → 3/3 | $0.070 → $0.052 | 16.6 s → 32.8 s | $0.0042 → $0.0016 |
| recall | `geocode-oom-per-row` | Geocode OOM was per-row dicts; population gate plus streaming fixed it | 3/3 → 3/3 | $0.070 → $0.034 | 18.0 s → 13.7 s | $0.0039 → $0.0025 |
| recall | `vec0-orphans` | vec0 tables don't cascade; `cleanup-orphans` removes leftovers | 3/3 → 3/3 | $0.067 → $0.065 | 9.9 s → 24.3 s | $0.0068 → $0.0027 |
| recall | `api-intent-headers` | Claude's curl calls send source and intent headers | 3/3 → 3/3 | $0.068 → $0.011 | 12.0 s → 7.5 s | $0.0057 → $0.0015 |
| recall | `sharpness-error-vs-absent` | Decode error is stored; an absent file records nothing | 3/3 → 3/3 | $0.068 → $0.032 | 15.2 s → 12.9 s | $0.0045 → $0.0025 |
| recall | `mirror-fields-needs-nas` | `mirrored:0` means the NAS lacks `/mirror-fields`; redeploy it | 3/3 → 3/3 | $0.067 → $0.036 | 8.3 s → 19.7 s | $0.0080 → $0.0018 |
| rules | `rule-faces-separation` | `separation` is a 5th percentile, robust to one bad label | 3/3 → 3/3 | $0.094 → $0.033 | 15.5 s → 12.8 s | $0.0061 → $0.0026 |
| rules | `rule-llm-unusable-answer` | `UnusableAnswer` spends an attempt; a transport timeout defers | 3/3 → 3/3 | $0.195 → $0.081 | 20.1 s → 14.9 s | $0.0097 → $0.0054 |
| rules | `rule-ingest-faces-done` | Only `faces` counts exhausted no-output photos as done | 3/3 → 3/3 | $0.109 → $0.049 | 18.9 s → 16.1 s | $0.0058 → $0.0031 |
| rules | `rule-maintenance-stacking-incremental` | Stacking is missing-only via `stacking_seen`; never clears the library | 3/3 → 2/3 ▼ | $0.118 → $0.060 | 20.6 s → 17.6 s | $0.0057 → $0.0034 |
| rules | `rule-workers-failures-list` | Worker exceptions go in `failures`; the server spends one attempt | 3/3 → 3/3 | $0.213 → $0.074 | 20.6 s → 19.3 s | $0.0103 → $0.0039 |
| rules | `rule-video-probe-owner` | Workers probe and scan video; the NAS stores and serves | 3/3 → 3/3 | $0.096 → $0.036 | 13.4 s → 11.7 s | $0.0071 → $0.0031 |
| rules | `rule-print-topaz-exit-code` | Topaz success means an output file appeared, not the exit code | 3/3 → 3/3 | $0.153 → $0.041 | 16.7 s → 16.7 s | $0.0092 → $0.0024 |
| rules | `rule-frontend-ps-poll` | `PS.poll` re-arms after settling; names the disk-starvation incident | 3/3 → 3/3 | $0.153 → $0.103 | 16.0 s → 14.2 s | $0.0096 → $0.0072 |
| rules | `rule-search-compose-scope` | `_compose_scope` folds date, camera and people into one id scope | 3/3 → 3/3 | $0.224 → $0.097 | 19.3 s → 22.8 s | $0.0116 → $0.0043 |
| skills | `skill-index-2024` | "how do I index 2024?" calls the photo-search skill | 0/3 → 0/3 | $0.067 → $0.030 | 8.9 s → 16.0 s | $0.0075 → $0.0019 |
| skills | `skill-search-slow` | "why is search slow?" calls the photo-search skill | 0/3 → 0/3 | $0.069 → $0.065 | 15.1 s → 25.8 s | $0.0045 → $0.0025 |
| skills | `skill-upload-google` | Google Photos upload question calls the photo-search skill | 0/3 → 0/3 | $0.069 → $0.088 | 14.5 s → 32.3 s | $0.0048 → $0.0027 |
| skills | `skill-add-filter` | New search filter request calls the photo-search skill | 0/3 → 0/3 | $0.068 → $0.037 | 12.1 s → 18.8 s | $0.0056 → $0.0020 |
| skills | `skill-last-queries` | "what were my last queries?" calls the photo-search skill | 0/3 → 0/3 | $0.075 → $0.035 | 11.6 s → 15.5 s | $0.0065 → $0.0023 |
| skills | `skill-fleet-laptop` | Laptop worker fleet question calls the photo-search skill | 0/3 → 0/3 | $0.069 → $0.065 | 12.4 s → 20.7 s | $0.0055 → $0.0032 |
| skills | `skill-recluster` | Bare "recluster faces" calls the photo-search skill | 0/3 → 0/3 | $0.067 → $0.040 | 10.6 s → 16.3 s | $0.0063 → $0.0024 |
| skills | `skill-schema-question` | "what tables are in photo_index.db?" calls the photo-search skill | 0/3 → 0/3 | $0.099 → $0.028 | 15.6 s → 12.6 s | $0.0063 → $0.0022 |
| tasks | `task-cli-envvar` | New CLI command's `--db` uses `envvar="PHOTOSEARCH_DB"` | 3/3 → 3/3 | $0.362 → $0.097 | 43.2 s → 39.5 s | $0.0084 → $0.0025 |
| tasks | `task-watcher-no-pgrep` | Wait script uses a PID or `wait`, never `pgrep -f` | 3/3 → 3/3 | $0.184 → $0.027 | 10.2 s → 9.1 s | $0.0180 → $0.0030 |
| tasks | `task-frontend-poll` | Health poll uses `PS.poll`, not `setInterval` | 2/3 → 3/3 ▲ | $0.416 → $0.142 | 53.4 s → 41.2 s | $0.0078 → $0.0035 |
| tasks | `task-nas-snippet-db-path` | NAS script passes `PHOTOSEARCH_DB` to `PhotoDB`, never `PhotoDB()` | 3/3 → 3/3 | $0.186 → $0.027 | 12.4 s → 8.3 s | $0.0150 → $0.0033 |
| tasks | `task-register-pass` | New pass lands in db.py, batch_state/rerun, batch-flow.js and cli.py | 0/3 → 0/3 | $0.560 → $0.247 | 78.0 s → 72.1 s | $0.0072 → $0.0034 |
| tasks | `task-sse-endpoint` | SSE endpoint avoids `get_event_loop()` and emits `done` | 1/3 → 3/3 ▲ | $0.456 → $0.103 | 67.2 s → 38.2 s | $0.0068 → $0.0027 |
| tasks | `task-curl-intent-headers` | Smoke-test curls send source and intent headers | 1/3 → 3/3 ▲ | $0.201 → $0.030 | 17.9 s → 11.8 s | $0.0112 → $0.0026 |
| tasks | `task-schema-bump` | New column bumps `SCHEMA_VERSION` and adds an `ADD COLUMN` migration | 3/3 → 3/3 | $0.426 → $0.125 | 53.2 s → 39.7 s | $0.0080 → $0.0031 |
| tasks | `task-filepath-writer` | Path change goes through `set_photo_filepath`, never a bare UPDATE | 3/3 → 2/3 ▼ | $0.436 → $0.159 | 64.4 s → 65.2 s | $0.0068 → $0.0024 |
| tasks | `task-restack-folder` | Restack passes `photo_ids` and guards the empty list | 2/3 → 3/3 ▲ | $0.249 → $0.079 | 32.2 s → 32.9 s | $0.0077 → $0.0024 |
| tasks | `task-nas-git-pull` | NAS git-pull script uses `safe.directory` | 3/3 → 3/3 | $0.210 → $0.032 | 20.2 s → 13.4 s | $0.0104 → $0.0024 |
| **recall** | **subtotal** |  | 146/147 → 142/147 ▼ | $0.076 → $0.035 | 12.7 s → 15.0 s | $0.0060 → $0.0024 |
| **rules** | **subtotal** |  | 27/27 → 26/27 ▼ | $0.151 → $0.064 | 17.9 s → 16.2 s | $0.0084 → $0.0039 |
| **skills** | **subtotal** |  | 0/24 → 0/24 | $0.073 → $0.048 | 12.6 s → 19.8 s | $0.0058 → $0.0024 |
| **tasks** | **subtotal** |  | 24/33 → 29/33 ▲ | $0.335 → $0.097 | 41.1 s → 33.8 s | $0.0081 → $0.0029 |
| **all** | **total** |  | 197/231 → 197/231 | $0.121 → $0.049 | 17.3 s → 18.3 s | $0.0070 → $0.0027 |

### gemma-4-12b-qat (local, LM Studio)

| suite | test | what it checks | pass before → after | cost/run before → after | wall/run before → after | $/s before → after |
|---|---|---|---|---|---|---|
| recall | `cli-db-envvar` | Every CLI `--db` carries `envvar="PHOTOSEARCH_DB"` | 1/1 → 1/1 | — | 624.0 s → 33.0 s | — |
| recall | `db-filename-schema` | DB file name; `SCHEMA_VERSION` in `db.py` | 1/1 → 1/1 | — | 883.9 s → 33.9 s | — |
| recall | `frontend-no-build` | No JSX (`React.createElement`); frontend changes need an image rebuild | 1/1 → 1/1 | — | 955.9 s → 31.2 s | — |
| recall | `sse-loop-and-events` | SSE uses `get_running_loop`; emits done/fatal/cancelled | 1/1 → 1/1 | — | 957.6 s → 44.4 s | — |
| recall | `photodb-no-args` | `PhotoDB()` with no args opens an empty stub DB | 1/1 → 1/1 | — | 887.1 s → 33.3 s | — |
| recall | `compose-entrypoint` | `--entrypoint` goes before the service name | 1/1 → 1/1 | — | 895.3 s → 33.1 s | — |
| recall | `nas-git-safe-directory` | NAS git "dubious ownership" needs `safe.directory` | 1/1 → 1/1 | — | 879.9 s → 32.5 s | — |
| recall | `pgrep-self-match` | `pgrep -f` matches the watcher itself; watch a PID or marker | 1/1 → 1/1 | — | 894.9 s → 33.0 s | — |
| recall | `worker-d-photo-root` | `-d /photos` 404s; omit `-d` for the whole library | 1/1 → 0/1 ▼ | — | 878.6 s → 51.0 s | — |
| recall | `worker-health-probe` | Workers probe `/api/health`, not the slow `/api/stats` | 1/1 → 1/1 | — | 886.3 s → 323.8 s | — |
| **recall** | **subtotal** |  | 10/10 → 9/10 ▼ | — | 874.4 s → 64.9 s | — |
| **all** | **total** |  | 10/10 → 9/10 ▼ | — | 874.4 s → 64.9 s | — |

## Reading the tables

- **Sonnet and Opus pass every recall and rules item in all 3 runs, before and after.** Their
  only per-item changes are Sonnet's `skill-last-queries` (0/3 → 3/3, the only skills item any
  model ever passed) and Opus's `task-register-pass` (3/3 → 2/3; the failing run hit the
  25-turn cap).
- **Recall and rules regressions appear only for the weaker models.** Haiku lost one of three
  runs on six items (`tags-pass-removed`, `visual-retired-tag`, `server-write-failure-attempt`,
  `faces-empty-terminal`, `ps-poll-not-setinterval`, `rule-maintenance-stacking-incremental`)
  and gained one on `clip-reclaim-cap`. gemma failed `worker-d-photo-root`, the fact REPORT.md
  traces to a path-scoped rule that a Q&A session never opens.
- **Haiku's tasks improved on four items** (`task-frontend-poll`, `task-sse-endpoint`,
  `task-curl-intent-headers`, `task-restack-folder`) and regressed on one
  (`task-filepath-writer`). Six of its nine failed task runs before the diet ended at the turn
  cap. `task-register-pass` failed all three runs on Haiku both before and after, every time at
  the turn cap.
- **`task-register-pass` is the most expensive item in every cloud config, and the diet barely
  cut it.** Per run it went $0.733 → $0.663 on Sonnet and $1.390 → $1.196 on Opus, while its
  wall time tripled on Sonnet (48.9 s → 147.6 s) and doubled on Opus (62.6 s → 126.4 s). Every
  other Sonnet and Opus task fell to 15-29% of its before cost.
- **Rules items got the smallest cost cut.** Per run, rules fell 52% on Sonnet ($0.327 →
  $0.157) and 54% on Opus ($0.519 → $0.239), against 70-76% for recall, skills and tasks. As a
  result, rules grew from 14% to 22% of Sonnet's total spend and from 11% to 18% of Opus's.
- **On gemma every item got faster.** Runs took 624-958 s before and 31-51 s after, except
  `worker-health-probe` at 324 s. The cloud models went the other way: recall runs got slower
  on 41 of 49 items for Sonnet, 42 of 49 for Opus and 28 of 49 for Haiku.
