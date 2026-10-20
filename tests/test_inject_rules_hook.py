"""The PreToolUse hook that injects .claude/rules/*.md on Bash/Grep/Glob.

Stdlib-only: nothing here imports photosearch, so CI's mocked torch/cv2 are
irrelevant.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HOOK = ROOT / "scripts" / "hooks" / "inject_rules.py"

_spec = importlib.util.spec_from_file_location("inject_rules", HOOK)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def _run(payload, state_dir):
    out = hook.build_output(payload, ROOT, state_dir)
    return json.loads(out) if out else None


def _bash(cmd, sid="s1"):
    return {"session_id": sid, "tool_name": "Bash", "tool_input": {"command": cmd}}


def _context(result):
    assert result["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert "permissionDecision" not in result["hookSpecificOutput"]
    return result["hookSpecificOutput"]["additionalContext"]


def test_workers_command_injects_workers_rule(tmp_path):
    ctx = _context(_run(_bash("bash scripts/run-workers.sh -p clip"), tmp_path))
    assert "Workers" in ctx and "MAX_PROCESS_ATTEMPTS" in ctx
    assert "---\npaths:" not in ctx  # frontmatter stripped


def test_unrelated_command_injects_nothing(tmp_path):
    assert _run(_bash("ls -la"), tmp_path) is None
    assert _run(_bash("git status && python -m pytest -q tests/test_x.py"), tmp_path) is None


def test_second_call_in_same_session_injects_nothing(tmp_path):
    assert _run(_bash("./run-workers.sh"), tmp_path) is not None
    assert _run(_bash("./run-workers.sh --help"), tmp_path) is None
    # a different session still gets it
    assert _run(_bash("./run-workers.sh", sid="s2"), tmp_path) is not None


def test_other_rule_still_injected_after_first(tmp_path):
    assert _run(_bash("./run-workers.sh"), tmp_path) is not None
    ctx = _context(_run(_bash("python cli.py maintenance-sweep --apply"), tmp_path))
    assert "Maintenance" in ctx and "Workers —" not in ctx


@pytest.mark.parametrize("cmd,heading", [
    ("python cli.py recluster-faces --db x", "Faces"),
    ("python cli.py ingest-incoming", "Ingest"),
    ("python cli.py derive-visual-tags", "LLM passes"),
    ("python cli.py register-videos", "Video"),
    ("python cli.py upscale a.jpg", "Print"),
    ("node scripts/check-frontend-refs.js", "Frontend"),
    ("curl -H 'X-Photosearch-Intent: x' localhost:8000/api/search", "Search"),
])
def test_keywords_route_to_rule(tmp_path, cmd, heading):
    assert heading in _context(_run(_bash(cmd), tmp_path))


def test_path_globs_come_from_frontmatter(tmp_path):
    # photosearch/worker*.py is a workers glob; absolute and ./ prefixes match.
    for cmd in ("cat photosearch/worker_api.py",
                f"sed -n 1,5p {ROOT}/photosearch/worker.py",
                "head ./photosearch/sharpness_backfill.py"):
        ctx = _context(_run(_bash(cmd, sid=cmd), tmp_path))
        assert "Workers" in ctx, cmd
    # quality.py must not match a .pyc
    assert _run(_bash("ls photosearch/quality.pyc"), tmp_path) is None


def test_grep_and_glob_inputs(tmp_path):
    grep = {"session_id": "g", "tool_name": "Grep",
            "tool_input": {"pattern": "def run_incremental_stacking", "path": "photosearch/stacking.py"}}
    assert "Maintenance" in _context(_run(grep, tmp_path))
    glob = {"session_id": "g2", "tool_name": "Glob",
            "tool_input": {"pattern": "photosearch/video_*.py"}}
    assert "Video" in _context(_run(glob, tmp_path))


def test_other_tools_ignored(tmp_path):
    p = {"session_id": "r", "tool_name": "Read",
         "tool_input": {"file_path": "photosearch/worker.py", "command": "run-workers"}}
    assert _run(p, tmp_path) is None


def test_output_stays_under_documented_cap(tmp_path):
    cmd = ("run-workers recluster-faces maintenance-sweep ingest-incoming "
           "derive-visual-tags register-videos upscale check-frontend-refs /api/search")
    seen = []
    for _ in range(10):  # leftovers arrive on later calls
        r = _run(_bash(cmd), tmp_path)
        if not r:
            break
        ctx = _context(r)
        assert len(ctx) < 10_000
        seen.append(ctx)
    assert len(seen) >= 2
    assert sum(ctx.count("Project rule .claude/rules/") for ctx in seen) == 9


def test_every_keyword_key_is_a_rule():
    stems = {p.stem for p in (ROOT / ".claude" / "rules").glob("*.md")}
    assert set(hook.KEYWORDS) == stems


def test_every_rule_glob_matches_its_own_files():
    for f in (ROOT / ".claude" / "rules").glob("*.md"):
        globs, _ = hook.parse_rule(f)
        assert globs, f
        for g in globs:
            files = [p for p in ROOT.glob(g) if p.is_file()]
            if not files:
                continue
            rel = files[0].relative_to(ROOT).as_posix()
            assert hook.glob_to_regex(g).search(rel), (g, rel)


def _spawn(stdin, tmp_path):
    env = {"PATH": "/usr/bin:/bin", "TMPDIR": str(tmp_path),
           "CLAUDE_PROJECT_DIR": str(ROOT)}
    return subprocess.run([sys.executable, str(HOOK)], input=stdin, text=True,
                          capture_output=True, env=env, timeout=30)


def test_process_end_to_end_and_never_fails(tmp_path):
    r = _spawn(json.dumps(_bash("./run-workers.sh", sid="e2e")), tmp_path)
    assert r.returncode == 0
    assert "Workers" in json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"]
    r = _spawn(json.dumps(_bash("./run-workers.sh", sid="e2e")), tmp_path)
    assert r.returncode == 0 and r.stdout == ""
    for junk in ("", "not json", "[]", '{"tool_name": "Bash", "tool_input": 5}'):
        r = _spawn(junk, tmp_path)
        assert r.returncode == 0 and r.stdout == ""


def test_settings_registers_hook():
    cfg = json.loads((ROOT / ".claude" / "settings.json").read_text())
    entries = cfg["hooks"]["PreToolUse"]
    assert any(e["matcher"] == "Bash|Grep|Glob"
               and "scripts/hooks/inject_rules.py" in e["hooks"][0]["command"]
               for e in entries)
