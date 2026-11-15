"""healthwatch's incident state machine, driven by fake checks.

healthwatch/healthwatch.py is a stdlib-only script (it runs on the NAS host,
outside Docker), so it is loaded by path rather than imported as a package.
"""
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "healthwatch" / "healthwatch.py"
_spec = importlib.util.spec_from_file_location("healthwatch", _PATH)
hw = importlib.util.module_from_spec(_spec)
sys.modules["healthwatch"] = hw  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(hw)


class Script:
    """A check whose results (and fix outcome) are scripted per call."""

    def __init__(self, results, fix_heals=False):
        self.results = list(results)
        self.calls = 0
        self.fixes = 0
        self.fix_heals = fix_heals
        self.healed = False

    def run(self, ctx):
        self.calls += 1
        if self.healed:
            return hw.Result(True, "healthy again")
        r = self.results.pop(0) if self.results else self.last
        self.last = r
        return r

    def fix(self, ctx):
        self.fixes += 1
        self.healed = self.fix_heals
        return "did the thing"


@pytest.fixture
def ctx(tmp_path):
    sent = []
    c = hw.Ctx(role="nas", state_dir=tmp_path, sleep=lambda s: None,
               notifier=lambda title, body, prio, tags: sent.append((title, prio)) or True)
    c.sent = sent
    return c


def _check(script, **kw):
    kw.setdefault("hint", "Known cause: gremlins.")
    return hw.Check("widget", script.run, fix=kw.pop("fix", None), **kw)


def _incident(ctx):
    path = ctx.state["checks"]["widget"]["incident"]
    return hw.Incident.load(Path(path)) if path else None


FAIL = hw.Result(False, "widget is broken", {"widget-log": "line 1\nline 2"})
OK = hw.Result(True, "widget fine")


def test_debounce_then_incident_with_report(ctx):
    s = Script([FAIL, FAIL])
    checks = [_check(s, debounce=2)]
    hw.run_cycle(ctx, checks)
    assert ctx.state["checks"]["widget"]["incident"] is None
    assert ctx.sent == []
    hw.run_cycle(ctx, checks)
    inc = _incident(ctx)
    assert inc.data["status"] == "OPEN"
    report = (inc.path / "report.md").read_text()
    for section in ("## Known causes and runbook", "gremlins", "## Timeline",
                    "## Evidence (captured when it happened)", "line 2",
                    "## Every check at the time it opened", "## For a Claude session",
                    "check --role nas widget"):
        assert section in report, section
    assert (inc.path / "evidence" / "widget-log.txt").read_text() == "line 1\nline 2\n"
    assert ctx.sent == [("Problem: widget", "high")]
    assert (ctx.state_dir / "latest-incident").resolve() == inc.path.resolve()
    assert "FAILING" in (ctx.state_dir / "status.md").read_text()


def test_recovery_closes_incident(ctx):
    s = Script([FAIL, OK])
    checks = [_check(s, debounce=1)]
    hw.run_cycle(ctx, checks)
    path = Path(ctx.state["checks"]["widget"]["incident"])
    hw.run_cycle(ctx, checks)
    inc = hw.Incident.load(path)
    assert inc.data["status"] == "RESOLVED"
    assert "recovered" in inc.data["resolution"]
    assert ctx.state["checks"]["widget"] == {**ctx.state["checks"]["widget"],
                                             "fails": 0, "incident": None}
    assert ctx.sent[-1] == ("Recovered: widget", "default")


def test_successful_fix_closes_as_auto_fixed(ctx):
    s = Script([FAIL], fix_heals=True)
    checks = [_check(s, debounce=1, fix=s.fix, fix_label="poke it")]
    hw.run_cycle(ctx, checks)
    assert s.fixes == 1
    assert ctx.state["checks"]["widget"]["incident"] is None
    inc_dir = next((ctx.state_dir / "incidents").iterdir())
    inc = hw.Incident.load(inc_dir)
    assert inc.data["resolution"] == "auto-fixed by: poke it (attempt 1)"
    assert "fix-1-output.txt" in inc.data["evidence"]
    assert [t for t, _ in ctx.sent] == ["Problem: widget", "Fixed: widget"]


def test_fix_attempts_are_capped_and_escalate_once(ctx):
    s = Script([FAIL])
    checks = [_check(s, debounce=1, fix=s.fix, fix_label="poke it", max_fixes=2)]
    for _ in range(5):
        hw.run_cycle(ctx, checks)
    assert s.fixes == 2
    inc = _incident(ctx)
    assert inc.data["fix_attempts"] == 2
    titles = [t for t, _ in ctx.sent]
    assert titles.count("Needs attention: widget") == 1
    assert ("Needs attention: widget", "urgent") in ctx.sent


def test_no_fix_when_fixes_disabled(ctx):
    ctx.fixes = False
    s = Script([FAIL])
    hw.run_cycle(ctx, [_check(s, debounce=1, fix=s.fix, fix_label="poke it")])
    assert s.fixes == 0
    assert _incident(ctx).data["status"] == "OPEN"


def test_mitigation_does_not_claim_failure(ctx):
    s = Script([FAIL])
    checks = [_check(s, debounce=1, fix=s.fix, fix_label="detour", max_fixes=1,
                     mitigation=True)]
    hw.run_cycle(ctx, checks)
    hw.run_cycle(ctx, checks)
    titles = [t for t, _ in ctx.sent]
    assert "Worked around: widget" in titles
    assert "Needs attention: widget" not in titles
    assert s.fixes == 1
    assert any("Mitigation 1 applied" in text for _, text in _incident(ctx).data["timeline"])


def test_skipped_neither_opens_nor_closes(ctx):
    s = Script([FAIL, hw.Result(None, "skipped: dep down"), hw.Result(None, "skipped")])
    checks = [_check(s, debounce=1)]
    hw.run_cycle(ctx, checks)
    path = ctx.state["checks"]["widget"]["incident"]
    hw.run_cycle(ctx, checks)
    hw.run_cycle(ctx, checks)
    assert ctx.state["checks"]["widget"]["incident"] == path
    assert hw.Incident.load(Path(path)).data["status"] == "OPEN"


def test_crashing_check_is_a_failure(ctx):
    def boom(ctx):
        raise RuntimeError("kaboom")
    hw.run_cycle(ctx, [hw.Check("widget", boom, hint="h", debounce=1)])
    assert "check crashed: RuntimeError: kaboom" in _incident(ctx).data["summary"]


def test_daily_reminder(ctx):
    s = Script([FAIL])
    checks = [_check(s, debounce=1)]
    hw.run_cycle(ctx, checks)
    hw.run_cycle(ctx, checks)
    assert [t for t, _ in ctx.sent] == ["Problem: widget"]
    old = (dt.datetime.now().astimezone() - dt.timedelta(hours=25)).isoformat()
    ctx.state["checks"]["widget"]["reminded"] = old
    hw.run_cycle(ctx, checks)
    assert ctx.sent[-1][0] == "Still open: widget"


def test_replica_lan_stopgap_lifecycle(ctx, tmp_path, monkeypatch):
    dropin = tmp_path / "stopgap.conf"
    monkeypatch.setattr(hw, "REPLICA_DROPIN", dropin)
    ctx.state["nas_paths"] = {"tailscale": False, "lan": True}
    assert hw.check_replica_nas_url(ctx).ok is True          # no stopgap
    dropin.write_text("x")
    assert hw.check_replica_nas_url(ctx).ok is True          # stopgap while TS down
    ctx.state["nas_paths"] = {"tailscale": True, "lan": True}
    assert hw.check_replica_nas_url(ctx).ok is False         # TS back: remove it


def test_stopgap_fix_refuses_while_workers_run(ctx, tmp_path, monkeypatch):
    dropin = tmp_path / "stopgap.conf"
    monkeypatch.setattr(hw, "REPLICA_DROPIN", dropin)
    monkeypatch.setattr(hw, "live_worker_pids", lambda: [4242])
    ctx.state["nas_paths"] = {"tailscale": False, "lan": True}
    assert "skipped: worker fleet running" in hw.fix_nas_reachable(ctx)
    assert not dropin.exists()


def test_check_command_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HEALTHWATCH_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(hw, "REPLICA_DROPIN", tmp_path / "absent.conf")
    (tmp_path / "state.json").write_text(json.dumps({"checks": {}}))
    assert hw.main(["check", "--role", "desktop", "replica_nas_url"]) == 0
    assert "replica_nas_url" in capsys.readouterr().out
    assert json.loads((tmp_path / "state.json").read_text()) == {"checks": {}}
    assert not (tmp_path / "incidents").exists()


def test_go_version_reader(tmp_path):
    f = tmp_path / "bin"
    f.write_bytes(b"\x7fELF" + b"\0" * 64 + b"\xff Go buildinf:" + b"\0" * 8 + b"go1.27.3\0")
    assert hw._go_version(str(f)) == (1, 27, 3)
    # Seen on the NAS: snapshot_tool (go1.9) also contains "go1.190119941996".
    h = tmp_path / "snapshot_tool"
    h.write_bytes(b"\x7fELF" + b"\0" * 8 + b"\xff Go buildinf:" + b"go1.190119941996\0" + b"go1.9\0")
    assert hw._go_version(str(h)) == (1, 9, 0)
    g = tmp_path / "notgo"
    g.write_bytes(b"\x7fELF" + b"\0" * 64)
    assert hw._go_version(str(g)) is None
