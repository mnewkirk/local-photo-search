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


def test_item_files_parse_and_have_unique_ids():
    for suite in h.SUITES:
        items = h.load_items(suite)
        assert items, suite
        for it in items:
            assert it["id"] and it["prompt"], (suite, it)
    for it in h.load_items("tasks"):
        assert it["checks"], it["id"]
        for chk in it["checks"]:
            assert chk["type"] in ("added_grep", "added_not_grep", "any_added_grep"), (it["id"], chk)
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
