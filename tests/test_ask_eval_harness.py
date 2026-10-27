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
mutants = _load("ask_mutants", EVAL / "mutants.py")
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

def test_runner_blocks_confirmed_writes_and_fakes_rerank(monkeypatch):
    from photosearch import agent
    from photosearch import tools as toolmod

    monkeypatch.setattr(agent, "_chat", agent._chat)  # restored after the test
    monkeypatch.setenv("PHOTOSEARCH_LLM_RERANK_MODEL", "vision-model")
    real, seen_env = [], []

    def fake_call(db, name, args=None):
        real.append(name)
        if name == "rerank_photos":
            seen_env.append(os.environ.get("PHOTOSEARCH_LLM_RERANK_MODEL"))
            return {"reranked": False, "criteria": "x", "results": [{"id": 1}, {"id": 2}]}
        return {}
    monkeypatch.setattr(toolmod, "call_tool", fake_call)
    rec = runner._Recorder()
    runner._install_wrappers(rec, allow_rerank=False)

    out = toolmod.call_tool(None, "set_photo_tags", {"photo_ids": [1], "confirm": True})
    assert "disabled" in out["error"] and rec.blocked_writes[0]["tool"] == "set_photo_tags"
    toolmod.call_tool(None, "set_photo_tags", {"photo_ids": [1]})  # dry run executes
    out = toolmod.call_tool(None, "rerank_photos", {"photo_ids": [1, 2], "criteria": "x"})
    # The real handler ran with no vision model, and its result is dressed as scored.
    assert seen_env == [None] and os.environ["PHOTOSEARCH_LLM_RERANK_MODEL"] == "vision-model"
    assert out["reranked"] is True and [h["id"] for h in out["results"]] == [1, 2]
    assert all(isinstance(h["rerank_score"], float) for h in out["results"])
    assert real == ["set_photo_tags", "rerank_photos"] and rec.stubbed == ["rerank_photos"]


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



# ---------------------------------------------------------- new checks --

def test_allowed_args_catches_the_june_false_passes():
    """Two June bakeoff traces the first graders passed: qwen set a quality floor
    for "good photos", gemma added the family to "our best photos"."""
    items = {i["id"]: i for i in h.load_items()}
    qwen = _row([("search_photos", {"people": ["Calvin", "Ellie"], "category": "soccer",
                                    "sort": "quality_desc", "min_quality": 6, "limit": 30})])
    ok, d = checks.grade(items["r05-kids-soccer"], qwen)
    assert not ok and any("min_quality" in c["name"] for c in d["checks"] if not c["ok"])
    gemma = _row([("search_photos", {"date_from": "2025-06-01", "date_to": "2025-08-31",
                                     "people": ["Calvin", "Ellie", "Nicole", "Matt"],
                                     "sort": "quality_desc"})], ran_at="2026-06-20")
    assert not checks.grade(items["r03-best-last-summer"], gemma)[0]


def test_allowed_args_ignores_pinned_filters():
    item = {"allowed_args": ["sort"], "locked": {"camera": "X"}}
    assert checks.grade(item, _row([("search_photos", {"sort": "quality_desc", "camera": "X"})]))[0]


def test_default_counts_an_omitted_arg_as_the_tool_default():
    spec = {"op": "eq", "tool": "representatives", "arg": "bucket", "value": "year", "default": "year"}
    assert checks.check_trace(spec, [{"tool": "representatives", "args": {}}], {})
    assert not checks.check_trace(spec, [{"tool": "representatives", "args": {"bucket": "month"}}], {})


def test_every_has_needs_every_call():
    spec = {"op": "every_has", "tool": "search_photos", "arg": "people", "value": ["Calvin"]}
    one = {"tool": "search_photos", "args": {"people": ["Calvin"]}}
    none = {"tool": "search_photos", "args": {}}
    assert checks.check_trace(spec, [one, one], {})
    assert not checks.check_trace(spec, [one, none], {})
    assert not checks.check_trace(spec, [], {})


def test_called_after_and_num_between():
    calls = [{"tool": "search_photos", "args": {}}, {"tool": "rerank_photos", "args": {"top_n": "3"}}]
    after = {"op": "called_after", "tool": "rerank_photos", "value": ["search_photos"]}
    assert checks.check_trace(after, calls, {})
    assert not checks.check_trace(after, calls[::-1], {})
    num = {"op": "num_between", "tool": "rerank_photos", "arg": "top_n", "value": [3, 3]}
    assert checks.check_trace(num, calls, {})


def test_count_with_match():
    spec = {"op": "count_ge", "tool": "search_photos", "arg": "date_from", "match": "-04-27$", "n": 2}
    calls = [{"tool": "search_photos", "args": {"date_from": f"{y}-04-27"}} for y in (2019, 2020)]
    assert checks.check_trace(spec, calls, {})
    assert not checks.check_trace(spec, calls[:1], {})


# -------------------------------------------------------------- mutants --

@pytest.fixture
def prompt(db):
    from photosearch import agent
    from photosearch import tools as toolmod
    tools = toolmod.openai_tools(include_images=False, include_writes=True, consolidated=False)
    return agent._system_prompt(db, allow_writes=True, consolidated=False), tools


def test_every_registered_mutant_finds_its_text(prompt):
    """mutants.RULES/BLOCKS track the prompt's wording. After a trim rewrites a
    rule, update its regex here so the baseline's mutants stay addressable."""
    system, tools = prompt
    for mid in list(mutants.RULES) + list(mutants.BLOCKS):
        if mid in ("block.facts", "block.hints"):
            continue  # depend on the DB and the env, checked below
        new, _, removed = mutants.apply(mid, system, tools)
        assert removed > 0 and len(new) < len(system), mid
    assert len(mutants.example_ids(system)) >= 9


def test_rule_mutants_remove_exactly_one_line(prompt):
    system, tools = prompt
    for mid in mutants.RULES:
        new, _, _ = mutants.apply(mid, system, tools)
        assert len(system.split("\n")) - len(new.split("\n")) == 1, mid


def test_examples_block_removes_every_example(prompt):
    system, tools = prompt
    new, _, _ = mutants.apply("block.examples", system, tools)
    assert "Examples:" not in new and not mutants.example_ids(new)
    assert "Rules:" in new and "WRITES" in new


def test_param_mutant_does_not_leak_through_shared_schemas(prompt):
    """openai_tools() splices the same filter-prop dicts into several tools; a
    mutant on one tool must not strip the others (or the caller's copy)."""
    system, tools = prompt
    before = json.dumps(tools)
    _, out, _ = mutants.apply("paramdesc.representatives", system, tools)
    assert json.dumps(tools) == before
    by = {t["function"]["name"]: t for t in out}
    assert "description" in by["search_photos"]["function"]["parameters"]["properties"]["people"]
    assert "description" not in by["representatives"]["function"]["parameters"]["properties"]["people"]


def test_filter_descriptions_keep_one_copy(prompt):
    system, tools = prompt
    _, out, removed = mutants.apply("tools.filter-descriptions", system, tools)
    by = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in out}
    assert removed > 0
    assert "description" in by["search_photos"]["people"]
    assert "description" not in by["daily_highlights"]["people"]


def test_absent_mutant_removes_nothing():
    _, _, removed = mutants.apply("rule.foreground", "no rules here", [])
    assert removed == 0
    with pytest.raises(KeyError):
        mutants.apply("nope", "", [])


def test_runner_applies_the_mutant_only_to_tool_calling_requests(monkeypatch):
    from photosearch import agent
    from photosearch import tools as toolmod

    sent = []
    monkeypatch.setattr(agent, "_chat", lambda messages, tools, *a, **kw: sent.append(
        (messages[0]["content"], tools)) or {"content": "x", "tool_calls": []})
    monkeypatch.setattr(toolmod, "call_tool", toolmod.call_tool)
    rec = runner._Recorder()
    runner._install_wrappers(rec, allow_rerank=False, mutant="drop.get_photo")
    tools = [{"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
             for n in ("search_photos", "get_photo")]
    agent._chat([{"role": "system", "content": "s"}], tools)
    agent._chat([{"role": "system", "content": "s"}], None)  # single-shot: untouched
    assert [t["function"]["name"] for t in sent[0][1]] == ["search_photos"]
    assert sent[1][1] is None and rec.mutant_removed > 0


# ------------------------------------------------------ mutation report --

def _write_rows(root, config, rows):
    d = root / config
    d.mkdir(parents=True)
    (d / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _r(item, passed, **kw):
    return {"model": "m", "item_id": item, "passed": passed, "suite": "s", "rep": 0,
            "detail": {"checks_passed": int(passed), "checks_total": 1, "checks": []}, **kw}


def test_mutation_kills_on_a_half_drop(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(h, "RESULTS", tmp_path)
    _write_rows(tmp_path, "before", [_r("a", True)] * 3 + [_r("b", True), _r("b", True), _r("b", False)])
    _write_rows(tmp_path, "abl-rule.x", [_r("a", False, mutant_removed=120), _r("b", True)])
    _write_rows(tmp_path, "abl-rule.y", [_r("a", True, mutant_removed=900), _r("b", True)])
    assert h.main(["mutation", "--baseline", "before"]) == 0
    out = capsys.readouterr().out
    assert "| rule.x | 120 | 2 | yes | a |" in out
    assert "| rule.y | 900 | 2 | no |" in out
    assert "Mutation score: 1/2" in out and "rule.y: 900 chars" in out


def test_sign_test():
    assert h.sign_test(0, 0) == 1.0
    assert h.sign_test(6, 0) == pytest.approx(0.03125)
    assert h.sign_test(3, 3) == 1.0


# ------------------------------------------------------------- coverage --

def test_every_cover_id_is_known():
    for i in h.load_items():
        assert i.get("covers"), i["id"]
        for c in i["covers"]:
            assert (c in mutants.RULES or c in mutants.BLOCKS or c in mutants.USAGE
                    or c.startswith("example.")), (i["id"], c)


def test_real_asks_and_behaviours(tmp_path):
    (tmp_path / "ask-2026-01-01.md").write_text(
        "**Ask:** best photo of Matt, one for each year, in the foreground\n"
        "**Ask:** test ask from today\nnoise\n")
    asks = h.real_asks(tmp_path)
    assert asks == ["best photo of Matt, one for each year, in the foreground"]
    assert {"rule.representatives", "rule.foreground", "rule.best-quality"} <= h.behaviours(asks[0])


def test_covering_rolls_rules_up_into_their_block():
    cov = h._covering([{"id": "a", "covers": ["rule.dates"]},
                       {"id": "b", "covers": ["example.photos-of-calvin"]}])
    assert cov["block.rules"] == ["a"] and cov["block.examples"] == ["b"]


def test_mutation_discounts_items_that_flip_under_a_control(tmp_path, monkeypatch, capsys):
    """An item that flips under an edit that removes nothing is noise: its flip
    under a real mutant is not a kill."""
    monkeypatch.setattr(h, "RESULTS", tmp_path)
    _write_rows(tmp_path, "before", [_r("a", True)] * 3 + [_r("b", True)] * 3)
    _write_rows(tmp_path, "abl-control.newline", [_r("a", False), _r("b", True)])
    _write_rows(tmp_path, "abl-rule.x", [_r("a", False, mutant_removed=50), _r("b", True)])
    _write_rows(tmp_path, "abl-rule.y", [_r("a", True, mutant_removed=70), _r("b", False)])
    assert h.main(["mutation", "--baseline", "before"]) == 0
    out = capsys.readouterr().out
    assert "Fragile items (flip under a control, so excluded as killers): a." in out
    assert "| rule.x | 50 | 2 | no |" in out          # only the fragile item flipped
    assert "| rule.y | 70 | 2 | yes | b |" in out
    assert "Mutation score: 1/2" in out


def test_controls_remove_nothing():
    system = "You are a photo-library search assistant.\nExamples:\n  'a' -> x\n  'b' -> y"
    for mid in mutants.CONTROLS:
        new, tools, removed = mutants.apply(mid, system, [])
        assert removed == 0 and new != system
        assert sorted(new.split()) == sorted(system.split()) or mid == "control.reword"
