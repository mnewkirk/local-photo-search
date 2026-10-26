"""Grading and transcript parsing of evals/context_eval/harness.py, no Claude calls."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "evals" / "context_eval" / "harness.py"

spec = importlib.util.spec_from_file_location("ctx_harness", HARNESS)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

_real_run = subprocess.run


@pytest.fixture(autouse=True)
def _never_call_claude(monkeypatch):
    """Guard: no test in this file may start a real `claude` process."""
    def guarded(cmd, *a, **kw):
        if cmd and cmd[0] == "claude":
            raise AssertionError("test tried to run the real claude CLI")
        return _real_run(cmd, *a, **kw)
    monkeypatch.setattr(subprocess, "run", guarded)


def test_item_files_parse_and_have_unique_ids():
    for suite in h.SUITES:
        items = h.load_items(suite)
        assert items, suite
        for it in items:
            assert it["id"] and it["prompt"], (suite, it)
    for it in h.load_items("tasks"):
        assert it["checks"], it["id"]
        for chk in it["checks"]:
            assert chk["type"] in ("added_grep", "added_not_grep", "any_added_grep",
                                   "added_gt_baseline"), (it["id"], chk)
    for it in h.load_items("rules"):
        assert (ROOT / it["read"]).exists(), f"{it['id']}: {it['read']} missing"


def test_recall_items_have_at_least_one_grader():
    for it in h.load_items("recall"):
        assert it.get("expect") or it.get("any"), it["id"]


def test_grade_text_expect_any_reject():
    item = {"expect": ["PHOTOSEARCH_DB"], "any": [["envvar", "environment"]], "reject": ["photos\\.db"]}
    ok, d = h.grade_text(item, "Use envvar=\"PHOTOSEARCH_DB\" on --db.")
    assert ok and not d["expect_missing"]
    ok, d = h.grade_text(item, "Use PHOTOSEARCH_DB.")
    assert not ok and d["any_missing"]
    ok, d = h.grade_text(item, "envvar PHOTOSEARCH_DB; the file is photos.db")
    assert not ok and d["reject_hit"]


def test_grade_task_paths_and_types():
    added = {"cli.py": ['@click.option("--db", envvar="PHOTOSEARCH_DB")', "def count_videos():"],
             "scripts/x.sh": ["pgrep -f recluster"]}
    item = {"id": "t", "checks": [
        {"type": "added_grep", "pattern": "envvar=\"PHOTOSEARCH_DB\"", "paths": ["cli.py"]},
        {"type": "added_not_grep", "pattern": "pgrep\\s+-f", "paths": ["cli.py"]},
        {"type": "any_added_grep", "patterns": ["count_videos", "count-videos"]},
    ]}
    ok, d = h.grade_task(item, added)
    assert ok and d["files_touched"] == ["cli.py", "scripts/x.sh"]
    item["checks"][1]["paths"] = ["scripts/*.sh"]
    ok, _ = h.grade_task(item, added)
    assert not ok


def test_grade_task_unknown_type_raises():
    with pytest.raises(ValueError):
        h.grade_task({"id": "t", "checks": [{"type": "nope", "pattern": "x"}]}, {})


def test_added_lines_tracked_and_untracked(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "a.py").write_text("one\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)
    (tmp_path / "a.py").write_text("one\ntwo\n")
    (tmp_path / "new.sh").write_text("echo hi\n")
    added = h.added_lines(tmp_path)
    assert added["a.py"] == ["two"]
    assert added["new.sh"] == ["echo hi"]


def test_parse_stream_collects_result_tools_skills():
    lines = [
        {"type": "system", "subtype": "init", "model": "claude-sonnet-5"},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Skill", "input": {"skill": "photo-search"}},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "x"}}]}},
        {"type": "result", "subtype": "success", "result": "envvar PHOTOSEARCH_DB",
         "total_cost_usd": 0.0123, "num_turns": 3, "duration_ms": 4200,
         "usage": {"input_tokens": 10, "cache_read_input_tokens": 60000, "output_tokens": 50}},
    ]
    raw = "noise\n" + "\n".join(json.dumps(l) for l in lines)
    p = h.parse_stream(raw)
    assert p["text"] == "envvar PHOTOSEARCH_DB"
    assert p["tools"] == ["Skill", "Read"] and p["skills"] == ["photo-search"]
    assert p["cost_usd"] == 0.0123 and p["num_turns"] == 3 and p["model"] == "claude-sonnet-5"
    assert h._input_tokens(p["usage"]) == 60010


def test_parse_stream_error_subtype_is_marked():
    raw = json.dumps({"type": "result", "subtype": "error_max_turns", "is_error": True})
    p = h.parse_stream(raw)
    assert p["is_error"] and p["text"] == "<error_max_turns>"


def test_claude_cmd_shape():
    cmd = h.claude_cmd("q", model="sonnet", max_turns=5, budget=1.0, edit=False, system="s")
    assert cmd[:2] == ["claude", "-p"] and "--strict-mcp-config" in cmd
    assert "Edit" not in cmd and "--permission-mode" not in cmd
    cmd = h.claude_cmd("q", model="", max_turns=5, budget=1.0, edit=True, system="s")
    assert "--model" not in cmd and "acceptEdits" in cmd and "Edit" in cmd


# ------------------------------------------------ turn cap (suggestion 7) --

def test_max_turns_default_is_per_suite_and_explicit_overrides_all():
    assert h.max_turns_for("tasks", None) == 40
    for suite in ("recall", "rules", "skills"):
        assert h.max_turns_for(suite, None) == 25
    assert h.max_turns_for("tasks", 10) == 10 and h.max_turns_for("recall", 60) == 60


def test_max_turns_flag_defaults_to_none_so_suites_can_differ(capsys):
    assert h.main(["run", "--config", "x", "--dry-run", "--repeat", "1", "--limit", "1"]) == 0
    out = capsys.readouterr().out
    assert "tasks" in out and "max_turns=40" in out and "max_turns=25" in out


def _row(suite="tasks", rid="t1", ok=False, subtype="success", **kw):
    return {"suite": suite, "id": rid, "rep": 1, "pass": ok, "subtype": subtype, **kw}


def test_agg_counts_turn_cap_hits_separately():
    rows = [_row(subtype="error_max_turns"), _row(subtype="error_max_turns"),
            _row(ok=True), _row(subtype="success")]
    a = h._agg(rows)
    assert a["n"] == 4 and a["turn_cap"] == 2 and a["pass_rate"] == 0.25
    assert h.is_turn_cap(rows[0]) and not h.is_turn_cap(rows[2])


def test_report_has_turn_cap_column_and_lists_hits(tmp_path, monkeypatch, capsys):
    d = tmp_path / "cfg"
    d.mkdir()
    rows = [_row(subtype="error_max_turns", num_turns=40, max_turns=40, answer="<error_max_turns>"),
            _row(rid="t2", ok=True)]
    (d / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(h, "RESULTS", tmp_path)
    assert h.main(["report", "--configs", "cfg", "--failures"]) == 0
    out = capsys.readouterr().out
    assert "turn-cap hits" in out
    assert "## Turn-cap hits" in out and "cfg / tasks / t1 rep1: 40 turns (cap 40)" in out
    assert "[TURN CAP]" in out
    assert "regex-only" not in out  # no judge verdicts -> no judge columns


# ----------------------------------------------- judge (suggestion 8) ------

ITEM = {"id": "q", "prompt": "Why never pgrep -f?",
        "expect": ["PHOTOSEARCH_DB", "\\bkill -0\\b"],
        "any": [["itself", "own process"], ["marker", "pid"]],
        "reject": ["photos\\.db"]}


def test_rubric_renders_regexes_as_readable_text():
    txt = h.rubric_text(ITEM)
    assert "must mention: PHOTOSEARCH_DB" in txt
    assert "must mention: kill -0" in txt          # \b stripped
    assert "at least one of: itself; own process" in txt
    assert "must NOT claim: photos.db" in txt      # \. unescaped
    assert "\\" not in txt
    assert h._readable("get_running_loop|a.*b") == "get_running_loop or a ... b"


def test_rubric_prefers_facts_and_prompt_contains_all_parts():
    item = {**ITEM, "facts": "  The watcher matches itself.  "}
    assert h.rubric_text(item) == "The watcher matches itself."
    p = h.judge_prompt(item, "my answer text")
    assert "Why never pgrep -f?" in p and "The watcher matches itself." in p
    assert "my answer text" in p and '"pass"' in p


def _stdout(result, **extra):
    return json.dumps({"type": "result", "is_error": False, "result": result,
                       "total_cost_usd": 0.002, **extra})


def test_parse_judge_output_good_fenced_and_with_prose():
    assert h.parse_judge_output(_stdout('{"pass": true, "reason": "ok"}')) == (True, "ok")
    assert h.parse_judge_output(_stdout('```json\n{"pass": false, "reason": "missing"}\n```')) == (False, "missing")
    assert h.parse_judge_output(_stdout('Verdict: {"pass": true, "reason": "x"}')) == (True, "x")
    events = json.dumps([{"type": "system"}, {"type": "result", "result": '{"pass": true, "reason": "r"}'}])
    assert h.parse_judge_output(events) == (True, "r")


@pytest.mark.parametrize("raw", [
    "not json at all",
    _stdout("I think it passes."),
    _stdout('{"pass": "yes", "reason": "x"}'),      # pass must be a real bool
    _stdout('{"pass": true}'),                      # reason required
    _stdout('{"pass": true, "reason": 3}'),
    _stdout("{broken json"),
    json.dumps({"type": "result", "is_error": True, "result": "Not logged in"}),
    "",
])
def test_parse_judge_output_malformed_raises(raw):
    with pytest.raises(ValueError):
        h.parse_judge_output(raw)


class _Proc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def _runner(proc, calls=None):
    def run(cmd, **kw):
        if calls is not None:
            calls.append((cmd, kw))
        return proc
    return run


def test_judge_cmd_has_no_tools_one_turn_and_git_write_block():
    cmd = h.judge_cmd("PROMPT", "haiku")
    assert cmd[:3] == ["claude", "-p", "PROMPT"]
    assert cmd[cmd.index("--model") + 1] == "haiku"
    assert cmd[cmd.index("--max-turns") + 1] == "1"
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--allowedTools" not in cmd
    assert cmd[cmd.index("--disallowedTools") + 1:] == h.NO_GIT_WRITES


def test_judge_answer_success_uses_clean_env_and_scratch_cwd(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    calls = []
    out = h.judge_answer(ITEM, "a", runner=_runner(
        _Proc(_stdout('{"pass": true, "reason": "paraphrase"}')), calls))
    assert out["pass"] is True and out["reason"] == "paraphrase" and out["cost_usd"] == 0.002
    cmd, kw = calls[0]
    assert "CLAUDECODE" not in kw["env"] and kw["stdin"] == subprocess.DEVNULL
    assert str(kw["cwd"]) != str(ROOT)


@pytest.mark.parametrize("proc", [
    _Proc(_stdout("sure thing")),                                        # malformed reply
    _Proc("", "Not logged in - please run /login", 1),                   # auth failure
    _Proc(json.dumps({"is_error": True, "result": "OAuth session expired and could not be refreshed"}), "", 1),
    _Proc(_stdout('{"pass": true, "reason": "x"}'), "boom", 2),          # non-zero exit
])
def test_judge_answer_failures_become_error_not_verdict(proc):
    calls = []
    out = h.judge_answer(ITEM, "a", runner=_runner(proc, calls))
    assert "error" in out and "pass" not in out
    assert len(calls) == 1  # never retried


def test_judge_answer_timeout_is_error():
    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)
    assert h.judge_answer(ITEM, "a", runner=boom)["error"] == "judge timeout"


# --- regrade --judge: regex pre-filter, overturn, keep-on-error, report ---

def _write_rows(tmp_path, monkeypatch, rows, suite="recall"):
    d = tmp_path / "cfg"
    d.mkdir()
    (d / f"{suite}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(h, "RESULTS", tmp_path)
    return d / f"{suite}.jsonl"


def _items(monkeypatch, **extra):
    base = {"id": "q1", "prompt": "Q?", "expect": ["PHOTOSEARCH_DB"]}
    real = h.load_items
    monkeypatch.setattr(h, "load_items",
                        lambda s: [{**base, **extra}] if s == "recall" else real(s))


def _regrade(*extra):
    return h.main(["regrade", "--configs", "cfg", *extra])


def test_regrade_judge_overturns_regex_failure_and_keeps_both_verdicts(tmp_path, monkeypatch):
    _items(monkeypatch)
    path = _write_rows(tmp_path, monkeypatch, [
        {"suite": "recall", "id": "q1", "rep": 1, "pass": False, "answer": "the env var DB path"},
        {"suite": "recall", "id": "q1", "rep": 2, "pass": True, "answer": "PHOTOSEARCH_DB"},
    ])
    seen = []
    monkeypatch.setattr(h, "judge_answer", lambda item, ans, **kw: (
        seen.append(ans), {"pass": True, "reason": "same fact", "model": "haiku", "cost_usd": 0.001})[1])
    assert _regrade("--judge") == 0
    assert seen == ["the env var DB path"]          # the regex PASS was not sent to the judge
    r1, r2 = [json.loads(l) for l in path.read_text().splitlines()]
    assert r1["pass"] is True and r1["detail"]["regex_pass"] is False
    assert r1["detail"]["judge"]["pass"] is True and r1["detail"]["judge"]["reason"] == "same fact"
    assert r2["pass"] is True and r2["detail"]["regex_pass"] is True and "judge" not in r2["detail"]


def test_regrade_judge_can_confirm_the_failure(tmp_path, monkeypatch):
    _items(monkeypatch)
    path = _write_rows(tmp_path, monkeypatch, [
        {"suite": "recall", "id": "q1", "rep": 1, "pass": False, "answer": "wrong"}])
    monkeypatch.setattr(h, "judge_answer", lambda *a, **k: {"pass": False, "reason": "no", "model": "haiku"})
    _regrade("--judge")
    r = json.loads(path.read_text())
    assert r["pass"] is False and r["detail"]["judge"]["pass"] is False


def test_regrade_malformed_judge_keeps_regex_verdict_and_records_error(tmp_path, monkeypatch):
    _items(monkeypatch)
    path = _write_rows(tmp_path, monkeypatch, [
        {"suite": "recall", "id": "q1", "rep": 1, "pass": False, "answer": "wrong"}])
    monkeypatch.setattr(h.subprocess, "run", _runner(_Proc(_stdout("probably fine"))))
    _regrade("--judge")
    r = json.loads(path.read_text())
    assert r["pass"] is False
    assert "judge_error" in r["detail"] and "judge" not in r["detail"]
    assert r["detail"]["regex_pass"] is False


def test_regrade_stops_calling_judge_after_repeated_failures(tmp_path, monkeypatch):
    _items(monkeypatch)
    rows = [{"suite": "recall", "id": "q1", "rep": n, "pass": False, "answer": f"w{n}"}
            for n in range(1, 7)]
    path = _write_rows(tmp_path, monkeypatch, rows)
    calls = []
    monkeypatch.setattr(h, "judge_answer", lambda *a, **k: (calls.append(1), {"error": "judge auth failure"})[1])
    _regrade("--judge")
    assert len(calls) == h.JUDGE_ABORT_STREAK
    assert all(not json.loads(l)["pass"] for l in path.read_text().splitlines())


def test_regrade_without_judge_flag_calls_nothing_but_reuses_stored_verdicts(tmp_path, monkeypatch):
    _items(monkeypatch)
    stored = {"pass": True, "reason": "earlier", "model": "haiku"}
    path = _write_rows(tmp_path, monkeypatch, [
        {"suite": "recall", "id": "q1", "rep": 1, "pass": True, "answer": "paraphrase",
         "detail": {"regex_pass": False, "judge": stored}},
        {"suite": "recall", "id": "q1", "rep": 2, "pass": False, "answer": "other"}])
    monkeypatch.setattr(h, "judge_answer", lambda *a, **k: pytest.fail("judge must not be called"))
    _regrade()
    r1, r2 = [json.loads(l) for l in path.read_text().splitlines()]
    assert r1["pass"] is True and r1["detail"]["judge"] == stored
    assert r2["pass"] is False


def test_report_shows_regex_only_and_judged_rates(tmp_path, monkeypatch, capsys):
    _write_rows(tmp_path, monkeypatch, [
        {"suite": "recall", "id": "q1", "rep": 1, "pass": True,
         "detail": {"regex_pass": False, "judge": {"pass": True, "reason": "r", "model": "haiku"}}},
        {"suite": "recall", "id": "q1", "rep": 2, "pass": False,
         "detail": {"regex_pass": False, "judge": {"pass": False, "reason": "missing fact", "model": "haiku"}}},
    ])
    h.main(["report", "--configs", "cfg", "--failures"])
    out = capsys.readouterr().out
    assert "regex-only pass rate" in out and "| 50% | 0% | 2 |" in out
    assert "(regex 0/2)" in out
    assert "judge (haiku): fail - missing fact" in out


def test_judged_items_have_facts_or_regexes():
    for suite in ("recall", "rules"):
        for it in h.load_items(suite):
            assert h.rubric_text(it).strip(), it["id"]
            if "facts" in it:
                assert isinstance(it["facts"], str) and len(it["facts"]) > 30, it["id"]


# ------------------------------------- grader audit (2026-10-25) ----------
# Every recall/rules item carries `counter` (plausible WRONG answers its grader
# must fail) and `pass_example` (a real stored answer that passed, so CI can
# check a tightened grader still passes right answers without results/).

def _text_items():
    return [(s, it) for s in ("recall", "rules") for it in h.load_items(s)]


@pytest.mark.parametrize("suite,item", _text_items(), ids=lambda v: v["id"] if isinstance(v, dict) else v)
def test_counter_answers_fail(suite, item):
    assert item.get("counter"), f"{item['id']}: no counter answers"
    for wrong in item["counter"]:
        ok, d = h.grade_text(item, wrong)
        assert not ok, f"{item['id']}: grader passes a wrong answer: {wrong!r}"


@pytest.mark.parametrize("suite,item", _text_items(), ids=lambda v: v["id"] if isinstance(v, dict) else v)
def test_pass_example_still_passes(suite, item):
    assert item.get("pass_example"), f"{item['id']}: no pass_example"
    ok, d = h.grade_text(item, item["pass_example"])
    assert ok, (item["id"], d)


def test_contradicting_answers_from_the_review_fail():
    items = {it["id"]: it for _, it in _text_items()}
    for iid, wrong in [
        ("server-write-failure-attempt", "Yes, it burns one attempt; the batch is deferred."),
        ("video-nas-never-decodes", "The NAS runs ffprobe on new videos; the workers don't, it's not native."),
        ("detect-stacks-empty-list", "An empty list is a no-op, nothing wiped; the library is safe."),
        ("sse-loop-and-events", "Use get_event_loop(), not get_running_loop; emit done, fatal, cancelled."),
    ]:
        assert not h.grade_text(items[iid], wrong)[0], iid


# --- task mutants: a correct change passes; the convention-breaking one fails ---

_BASE = {"photosearch/db.py": "SCHEMA_VERSION = 39\n"}
_DB = ["PASS_MEDIA_EXTRA = 1"]
TASK_CASES = {
    "task-cli-envvar": (
        {"cli.py": ['@cli.command("count-videos")',
                    '@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB")',
                    "def count_videos(db):"]},
        {"cli.py": ['@cli.command("count-videos")',
                    '@click.option("--db", default="photo_index.db")  # envvar="PHOTOSEARCH_DB"',
                    "def count_videos(db):"]}),
    "task-frontend-poll": (
        {"frontend/dist/status.html": ["const h = PS.poll(() => fetch('/api/health').then(show), 5000);"]},
        {"frontend/dist/status.html": ["// PS.poll would be nicer here",
                                       "fetchHealth(); setTimeout(loop, 5000);"]}),
    "task-sse-endpoint": (
        {"photosearch/web.py": ['@app.post("/api/admin/noop-stream")',
                                "    loop = asyncio.get_running_loop()",
                                '        yield _sse({"type": "done"})']},
        {"photosearch/web.py": ['@app.post("/api/admin/noop-stream")',
                                "    loop = asyncio.get_running_loop()",
                                '        yield _sse({"type": "progress"})  # done']}),
    "task-restack-folder": (
        {"photosearch/stacking.py": [
            "def restack_folder(db, folder):",
            '    photo_ids = [r[0] for r in db.conn.execute("SELECT id FROM photos WHERE folder = ?", (folder,))]',
            "    if not photo_ids:",
            '        logger.info("no photos in %s", folder)',
            "        return []",
            "    return run_stacking(db, photo_ids=photo_ids)"]},
        {"photosearch/stacking.py": [
            "def restack_folder(db, folder):",
            '    photo_ids = [r[0] for r in db.conn.execute("SELECT id FROM photos WHERE folder = ?", (folder,))]',
            "    if not photo_ids:",
            "        photo_ids = None",
            "    return run_stacking(db, photo_ids=photo_ids)"]}),
    "task-schema-bump": (
        {"photosearch/db.py": ["SCHEMA_VERSION = 40",
                               '            self.conn.execute("ALTER TABLE photos ADD COLUMN rating INTEGER")']},
        {"photosearch/db.py": ["SCHEMA_VERSION = 12",
                               "            # ALTER TABLE photos ADD COLUMN rating INTEGER"]}),
    "task-register-pass": (
        {"photosearch/db.py": ['    "exif-rescan",', '    "exif-rescan": "still",'],
         "photosearch/batch_state.py": ['    "exif-rescan",'],
         "frontend/dist/batch-flow.js": ["  'exif-rescan',"],
         "cli.py": ['    "exif-rescan",']},
        {"photosearch/db.py": ['    "exif-rescan",'],
         "photosearch/batch_state.py": ['    "exif-rescan",'],
         "frontend/dist/batch-flow.js": ["  'exif-rescan',"],
         "cli.py": ['    "exif-rescan",']}),
    "task-filepath-writer": (
        {"cli.py": ["def set_filepath(db, photo_id, path):", "    db.set_photo_filepath(photo_id, path)"]},
        {"cli.py": ["def set_filepath(db, photo_id, path):",
                    '    db.conn.execute("UPDATE photos SET filepath = ? WHERE id = ?", (path, photo_id))',
                    "    # not db.set_photo_filepath"]}),
    "task-watcher-no-pgrep": (
        {"scripts/wait_recluster.sh": ["python cli.py recluster-faces --dry-run > /tmp/recluster.log 2>&1 &",
                                       "pid=$!", 'while kill -0 "$pid" 2>/dev/null; do sleep 2; done', "echo DONE"]},
        {"scripts/wait_recluster.sh": ["python cli.py recluster-faces --dry-run > /tmp/recluster.log 2>&1 &",
                                       "while pgrep -f recluster-faces >/dev/null; do sleep 2; done",
                                       "echo DONE  # wait $!"]}),
}


@pytest.mark.parametrize("tid", sorted(TASK_CASES))
def test_task_mutant_fails_and_correct_change_passes(tid):
    item = {it["id"]: it for it in h.load_items("tasks")}[tid]
    good, mutant = TASK_CASES[tid]
    ok, d = h.grade_task(item, good, baseline=_BASE.get)
    assert ok, (tid, [c for c in d["checks"] if not c["ok"]])
    ok, _ = h.grade_task(item, mutant, baseline=_BASE.get)
    assert not ok, f"{tid}: mutant passes"


def test_restack_rejects_inline_none_fallback_and_unscoped_call():
    item = {it["id"]: it for it in h.load_items("tasks")}["task-restack-folder"]
    good = TASK_CASES["task-restack-folder"][0]["photosearch/stacking.py"]
    inline = [l.replace("photo_ids=photo_ids)", "photo_ids=photo_ids or None)") for l in good]
    unscoped = good + ["    run_stacking(db)"]
    for lines in (inline, unscoped):
        assert not h.grade_task(item, {"photosearch/stacking.py": lines})[0]


def test_schema_bump_needs_a_known_baseline_and_a_higher_version():
    item = {it["id"]: it for it in h.load_items("tasks")}["task-schema-bump"]
    good = TASK_CASES["task-schema-bump"][0]
    assert not h.grade_task(item, good)[0]                          # no baseline -> fail
    assert not h.grade_task(item, good, baseline=lambda rel: "SCHEMA_VERSION = 40\n")[0]
    ok, d = h.grade_task(item, good, baseline=lambda rel: "SCHEMA_VERSION = 39\n")
    assert ok and d["checks"][0]["baseline"] == 39 and d["checks"][0]["found"] == [40]


def test_git_baseline_reads_the_committed_file(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "db.py").write_text("SCHEMA_VERSION = 7\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)
    (tmp_path / "db.py").write_text("SCHEMA_VERSION = 8\n")         # the run's edit
    read = h.git_baseline(tmp_path)
    assert read("db.py") == "SCHEMA_VERSION = 7\n" and read("missing.py") is None


def test_strip_comments_by_file_type():
    py = '#!/usr/bin/env python3\n"""Uses envvar="PHOTOSEARCH_DB"."""\nx = a // b  # PS.poll\nconn.execute(\n    """ALTER TABLE t ADD COLUMN r"""\n)'
    out = h._strip_comments(py, "x.py")
    assert "PHOTOSEARCH_DB" not in out and "PS.poll" not in out
    assert "a // b" in out and "ADD COLUMN r" in out               # floor division and SQL stay
    js = "/* PS.poll(\n f) */ fetch(u); // PS.poll\nlet c = '#fff'; <!-- PS.poll -->"
    out = h._strip_comments(js, "frontend/dist/status.html")
    assert "PS.poll" not in out and "#fff" in out


# --- regrade: infra reclassification and turn caps ---

def test_regrade_reclassifies_stored_session_limit_as_infra(tmp_path, monkeypatch, capsys):
    _items(monkeypatch)
    usage = {"input_tokens": 10, "cache_read_input_tokens": 100}
    path = _write_rows(tmp_path, monkeypatch, [
        {"suite": "recall", "id": "q1", "rep": 1, "pass": False, "subtype": "success", "exit_code": 1,
         "usage": usage, "answer": "You've hit your session limit · resets 4:50pm"},
        {"suite": "recall", "id": "q1", "rep": 2, "pass": True, "subtype": "success", "exit_code": 0,
         "usage": usage, "answer": "PHOTOSEARCH_DB"},
    ])
    _regrade()
    r1, r2 = [json.loads(l) for l in path.read_text().splitlines()]
    assert r1["infra_error"].startswith("gateway:") and r1["pass"] is False
    assert not r2.get("infra_error") and r2["pass"] is True
    assert "1 reclassified as infra" in capsys.readouterr().err
    a = h._agg([r1, r2])
    assert a["n"] == 1 and a["infra"] == 1 and a["pass_rate"] == 1.0


def test_infra_needs_a_failed_run_even_for_stored_rows():
    ok_row = {"answer": "the server returns rate_limit_error to callers", "exit_code": 0,
              "subtype": "success", "usage": {"input_tokens": 5}}
    assert h.infra_error(h.stored_result(ok_row)) is None


def test_regrade_keeps_turn_cap_fail_but_records_checks_passed(tmp_path, monkeypatch, capsys):
    item = {"id": "t1", "prompt": "p", "checks": [{"type": "added_grep", "pattern": "PS\\.poll"}]}
    real = h.load_items
    monkeypatch.setattr(h, "load_items", lambda s: [item] if s == "tasks" else real(s))
    _write_rows(tmp_path, monkeypatch, [
        _row(subtype="error_max_turns", exit_code=1, usage={"input_tokens": 9}, added={"a.js": ["PS.poll(f, 5)"]}),
        _row(rid="t1", ok=True, exit_code=0, usage={"input_tokens": 9}, added={"a.js": ["PS.poll(f, 5)"]}),
    ], suite="tasks")
    _regrade()
    capsys.readouterr()
    h.main(["report", "--configs", "cfg"])
    out = capsys.readouterr().out
    assert "checks passed but hit turn cap" in out
    assert "| 50% |" in out and "| 1 | 1 | 0 |" in out              # 1 turn-cap hit, 1 with checks ok
    assert "[1 turn-cap, 1 checks ok]" in out


# --- report: power ---

def test_regressed_needs_a_drop_of_half_the_runs(tmp_path, monkeypatch, capsys):
    rows = {"a": [], "b": []}
    for rep in (1, 2, 3):
        rows["a"] += [{"suite": "recall", "id": "one", "rep": rep, "pass": True},
                      {"suite": "recall", "id": "two", "rep": rep, "pass": True}]
        rows["b"] += [{"suite": "recall", "id": "one", "rep": rep, "pass": rep != 1},
                      {"suite": "recall", "id": "two", "rep": rep, "pass": rep == 3}]
    for c, rs in rows.items():
        (tmp_path / c).mkdir()
        (tmp_path / c / "recall.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    monkeypatch.setattr(h, "RESULTS", tmp_path)
    h.main(["report", "--configs", "a", "b"])
    out = capsys.readouterr().out
    assert "| recall | one | 3/3 | 2/3 | lower (noise) |" in out
    assert "| recall | two | 3/3 | 1/3 | REGRESSED |" in out
    assert "## Pass rates with 95% intervals (a vs b)" in out
    assert "6/6 100% [61%–100%]" in out and "3/6 50%" in out
    assert "| 0.18 | 2 / 0 | 0.50 |" in out


def test_stats_helpers():
    lo, hi = h.wilson(146, 147)
    assert 0.96 < lo < 0.97 and hi > 0.998
    assert h.wilson(0, 0) == (0.0, 1.0)
    assert abs(h.fisher_exact(3, 0, 0, 3) - 0.1) < 1e-9          # textbook 2x2
    assert h.fisher_exact(5, 5, 5, 5) == 1.0
    assert abs(h.sign_test(0, 5) - 0.0625) < 1e-9 and h.sign_test(0, 0) == 1.0
