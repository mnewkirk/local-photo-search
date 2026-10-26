"""evals/ask_eval: grading, the fleet guard and the runner's write block. No
model, LM Studio or NAS call — an autouse fixture fails any network attempt."""
import datetime as dt
import importlib.util
import json
import os
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "evals" / "ask_eval"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


checks = _load("ask_checks", EVAL / "checks.py")
h = _load("ask_harness", EVAL / "harness.py")
runner = _load("ask_runner", EVAL / "runner.py")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*a, **kw):
        raise AssertionError("test tried to open a network connection")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def _row(calls, answer="", **kw):
    trace = [{"k": "call", "tool": t, "args": a} for t, a in calls]
    return {"trace": trace, "answer": answer, "ran_at": "2026-10-25T12:00:00", **kw}


# ---------------------------------------------------------------- items --

def test_items_validate_and_ids_are_unique():
    items = h.load_items()
    assert len(items) >= 30
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))
    for i in items:
        assert checks.validate_item(i) == [], i["id"]


def test_validate_flags_unknown_op_and_empty_item():
    assert checks.validate_item({"id": "x", "suite": "s", "prompt": "p",
                                 "checks": [{"op": "nope", "tool": "t"}]})
    assert checks.validate_item({"id": "x", "suite": "s", "prompt": "p"})


# --------------------------------------------------------------- checks --

def test_consolidated_search_grades_as_the_family_tool():
    calls = checks.canonical_calls(_row([("search", {"mode": "chapters", "date_from": "2025-06-16"}),
                                         ("_fallback", {})])["trace"])
    assert calls == [{"tool": "group_into_chapters", "args": {"date_from": "2025-06-16"}}]


def test_set_eq_tolerates_a_stringified_list():
    spec = {"op": "set_eq", "tool": "search_photos", "arg": "people", "value": ["Calvin"]}
    for v in (["calvin"], '["Calvin"]', "Calvin"):
        assert checks.check_trace(spec, [{"tool": "search_photos", "args": {"people": v}}], {})


def test_negative_checks_need_a_call():
    spec = {"op": "absent", "tool": "search_photos", "arg": "min_quality"}
    assert not checks.check_trace(spec, [], {})
    assert checks.check_trace(spec, [{"tool": "search_photos", "args": {}}], {})
    assert not checks.check_trace(spec, [{"tool": "search_photos", "args": {"min_quality": 5}}], {})


def test_multi_arg_contains():
    spec = {"op": "contains", "tool": "search_photos", "arg": ["query", "category"], "value": "beach"}
    assert checks.check_trace(spec, [{"tool": "search_photos", "args": {"category": ["beach"]}}], {})
    assert not checks.check_trace(spec, [{"tool": "search_photos", "args": {"query": "park"}}], {})


def test_date_tokens_follow_the_run_day():
    t = checks.date_tokens(dt.date(2026, 10, 25))
    assert t["last_summer"] == "2026" and t["last_month_start"] == "2026-09-01"
    assert t["last_month_end"] == "2026-09-30"
    assert checks.date_tokens(dt.date(2026, 3, 1))["last_summer"] == "2025"


def test_grade_uses_the_day_the_run_happened():
    item = {"checks": [{"op": "date_between", "tool": "search_photos", "arg": "date_from",
                        "value": ["{today}", "{today}"]}]}
    ok, _ = checks.grade(item, _row([("search_photos", {"date_from": "2026-10-25"})]))
    assert ok


def test_fallback_and_errors_fail_the_run():
    item = {"checks": [{"op": "called", "tool": "search_photos"}]}
    row = _row([("search_photos", {})])
    assert checks.grade(item, row)[0]
    assert not checks.grade(item, {**row, "single_shot": True})[0]
    assert not checks.grade(item, {**row, "error": "LLM error"})[0]


def test_answer_sql_accepts_thousands_separators():
    item = {"answer_sql": [{"name": "n", "sql": "SELECT 1"}]}
    assert checks.grade(item, _row([], answer="About 16,500 photos."), lambda q: 16500)[0]
    assert checks.grade(item, _row([], answer="16500 photos"), lambda q: 16500)[0]
    assert not checks.grade(item, _row([], answer="165000 photos"), lambda q: 16500)[0]


def test_results_sql_substitutes_ids_and_kinds_are_tagged():
    seen = []
    item = {"results_sql": [{"name": "bad", "sql": "SELECT COUNT(*) WHERE id IN ({ids})", "expect": 0}],
            "answer_expect": ["Calvin"]}
    ok, d = checks.grade(item, _row([], answer="Calvin", result_ids=[3, 4]),
                         lambda q: seen.append(q) or 0)
    assert ok and "IN (3,4)" in seen[0]
    assert {c["kind"] for c in d["checks"]} == {"results", "answer"}


def test_db_checks_are_skipped_without_a_db():
    item = {"answer_sql": [{"name": "n", "sql": "SELECT 1"}], "answer_expect": ["x"]}
    ok, d = checks.grade(item, _row([], answer="x"))
    assert ok and d["skipped"] == ["n"]


# ---------------------------------------------------------------- guard --

def _fake_proc(tmp_path, pid, argv):
    d = tmp_path / str(pid)
    d.mkdir()
    (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")


def test_local_worker_pids_finds_fleet_processes_not_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(h.glob, "glob", lambda pat, _g=h.glob.glob: (
        _g(pat) if pat.startswith(str(tmp_path)) else []))
    _fake_proc(tmp_path, 101, ["python", "cli.py", "worker", "--server", "x"])
    _fake_proc(tmp_path, 102, ["bash", "./run-workers.sh", "--native"])
    _fake_proc(tmp_path, 103, ["python", "cli.py", "serve"])
    _fake_proc(tmp_path, os.getpid(), ["python", "cli.py", "worker"])
    found = h.local_worker_pids(str(tmp_path))
    assert [f.split(":")[0] for f in sorted(found)] == ["pid 101", "pid 102"]


def test_nas_guard_fails_closed():
    reasons = h.nas_claims("http://nas.invalid:8000")
    assert reasons and "could not verify" in reasons[0]


def test_run_refuses_and_starts_nothing_while_busy(monkeypatch, tmp_path):
    monkeypatch.setattr(h, "fleet_busy", lambda nas: ["pid 1: cli.py worker"])
    monkeypatch.setattr(h, "RESULTS", tmp_path)

    def boom(*a, **kw):
        raise AssertionError("runner started while the fleet was busy")
    monkeypatch.setattr(h, "run_runner", boom)
    monkeypatch.setattr(h, "Tree", boom)
    for extra in ([], ["--dry-run"]):
        assert h.main(["run", "--config", "t", *extra]) == 2
    assert not any(tmp_path.iterdir())


def test_run_rechecks_the_fleet_before_every_chunk(monkeypatch, tmp_path):
    calls = iter([[], ["NAS claim: describe x16"]])
    monkeypatch.setattr(h, "fleet_busy", lambda nas: next(calls))
    monkeypatch.setattr(h, "RESULTS", tmp_path)

    class FakeTree:
        def __init__(self, ref):
            self.path, self.sha, self.dirty = ROOT, "abc", False

        def __enter__(self):
            return self

        def __exit__(self, *e):
            pass
    monkeypatch.setattr(h, "Tree", FakeTree)
    monkeypatch.setattr(h, "measure_tree", lambda tree, args: {"surfaces": {}})
    monkeypatch.setattr(h, "lmstudio_models", lambda lm: {})
    monkeypatch.setattr(h, "_run_chunk", lambda *a: (_ for _ in ()).throw(
        AssertionError("chunk ran after the fleet became busy")))
    db = tmp_path / "db.sqlite"
    db.write_bytes(b"")
    assert h.main(["run", "--config", "t", "--db", str(db), "--only", "r01-person"]) == 2


# --------------------------------------------------------------- runner --

def test_runner_blocks_confirmed_writes_and_stubs_rerank(monkeypatch):
    from photosearch import agent
    from photosearch import tools as toolmod

    monkeypatch.setattr(agent, "_chat", agent._chat)  # restored after the test
    real = []
    monkeypatch.setattr(toolmod, "call_tool", lambda db, name, args=None: real.append(name) or {})
    rec = runner._Recorder()
    runner._install_wrappers(rec, allow_rerank=False)

    out = toolmod.call_tool(None, "set_photo_tags", {"photo_ids": [1], "confirm": True})
    assert "disabled" in out["error"] and rec.blocked_writes[0]["tool"] == "set_photo_tags"
    toolmod.call_tool(None, "set_photo_tags", {"photo_ids": [1]})  # dry run executes
    assert "error" in toolmod.call_tool(None, "rerank_photos", {"photo_ids": [1], "criteria": "x"})
    assert real == ["set_photo_tags"] and rec.stubbed == ["rerank_photos"]


def test_runner_records_lm_studio_error_body(monkeypatch):
    import io
    import urllib.error
    from photosearch import agent
    from photosearch import tools as toolmod

    def overflow(*a, **kw):
        raise urllib.error.HTTPError("u", 400, "Bad Request", {}, io.BytesIO(
            b'{"error":"Context size has been exceeded."}'))
    monkeypatch.setattr(agent, "_chat", overflow)
    monkeypatch.setattr(toolmod, "call_tool", toolmod.call_tool)
    rec = runner._Recorder()
    runner._install_wrappers(rec, allow_rerank=False)
    with pytest.raises(urllib.error.HTTPError):
        agent._chat([{"role": "system", "content": "s"}], [])
    assert "Context size" in rec.rounds[0]["error_body"]


def test_runner_row_feeds_the_grader(db, monkeypatch):
    """A scripted agent run end to end: runner.run_job's row grades the way an
    item expects (conftest `db`: Alex has 3 photos)."""
    from photosearch import agent
    from photosearch import tools as toolmod

    replies = iter([
        {"content": None, "tool_calls": [{"id": "c1", "name": "search_photos",
                                          "arguments": {"people": ["Alex"]}}],
         "usage": {"prompt_tokens": 11700, "completion_tokens": 20}},
        {"content": "Found 3 photos of Alex.", "tool_calls": [],
         "usage": {"prompt_tokens": 11900, "completion_tokens": 9}},
    ])
    monkeypatch.setenv("PHOTOSEARCH_LLM_AGENT_MODEL", "unset")  # run_job sets it
    monkeypatch.setattr(agent, "_chat", lambda *a, **kw: next(replies))
    monkeypatch.setattr(toolmod, "call_tool", toolmod.call_tool)
    rec = runner._Recorder()
    runner._install_wrappers(rec, allow_rerank=False)
    item = {"id": "t", "suite": "routing", "prompt": "photos of Alex",
            "checks": [{"op": "set_eq", "tool": "search_photos", "arg": "people", "value": ["Alex"]}],
            "results_min": 3, "answer_expect": [r"\b3\b"]}
    row = runner.run_job(db, {"item": item, "rep": 0, "model": "m"}, rec)
    assert row["first_prompt_tokens"] == 11700 and row["max_prompt_tokens"] == 11900
    assert row["prompt"]["n_tools"] > 0 and row["prompt"]["system_chars"] > 0
    ok, detail = checks.grade(item, row)
    assert ok, detail
