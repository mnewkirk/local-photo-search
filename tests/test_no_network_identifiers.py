"""Guard: no device hostnames / private IPs / MACs in tracked files.

The repo is public. The NAS's hostname, IPs and login belong in the git-ignored
``nas.env`` (see ``nas.env.example``), never in a tracked file; docs use
``<nas-host>`` / ``<nas-user>`` placeholders.

The patterns match *shapes* — a vendor device name like ``abc1234-0f9e``, any
IPv4 literal in a private or CGNAT range, a MAC address. They deliberately do
NOT spell out the real values: hard-coding those here would itself be the leak.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_OCTET = r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
PATTERNS = {
    # NAS/router factory hostnames: model number + last MAC bytes.
    "device hostname": re.compile(r"(?<![0-9a-z])[a-z]{2,5}\d{4}-[0-9a-f]{4}(?![0-9a-z])", re.I),
    "private/CGNAT IPv4": re.compile(
        r"(?<![\d.])(?:"
        rf"10\.{_OCTET}\.{_OCTET}\.{_OCTET}"
        rf"|192\.168\.{_OCTET}\.{_OCTET}"
        rf"|172\.(?:1[6-9]|2\d|3[01])\.{_OCTET}\.{_OCTET}"
        rf"|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.{_OCTET}\.{_OCTET}"
        r")(?![\d.])"
    ),
    "MAC address": re.compile(r"(?<![0-9A-Fa-f:-])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f:-])"),
}


def _tracked_files():
    if not shutil.which("git"):
        pytest.skip("git not available")
    res = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True)
    if res.returncode != 0:
        pytest.skip("not a git checkout (e.g. inside the Docker image)")
    return [ROOT / p for p in res.stdout.decode().split("\0") if p]


def _scan(text):
    return [(kind, m.group(0)) for kind, rx in PATTERNS.items() for m in rx.finditer(text)]


def test_patterns_catch_the_shapes_and_spare_the_placeholders():
    # Built from parts so this file never contains a matching literal itself.
    bad = [
        "ssh someone@" + "192.168" + ".1.50",
        "http://" + "10.0" + ".0.7:8000",
        "gateway " + "172.20" + ".176.1",
        "tailnet " + "100.101" + ".102.103",
        "host " + "abc4800" + "-f0e1" + ":8000",
        "mac " + ":".join(["aa", "bb", "cc", "dd", "ee", "0f"]),
    ]
    for line in bad:
        assert _scan(line), f"pattern missed: {line!r}"
    ok = [
        "ssh <nas-user>@<nas-host>", "http://<nas-host>:8000", "http://nas.local:8000",
        "127.0.0.1", "0.0.0.0", "8.8.8.8", "172.15.0.1", "100.63.0.1", "100.128.0.1",
        "version 3.10.12.1", "2026-09-19", "UGREEN DXP4800", "12:34:56",
    ]
    for line in ok:
        assert not _scan(line), f"false positive: {line!r} -> {_scan(line)}"


def test_no_network_identifiers_in_tracked_files():
    hits = []
    for path in _tracked_files():
        try:
            data = path.read_bytes()
        except OSError:          # deleted-but-tracked, broken symlink
            continue
        if b"\0" in data[:8192]:  # binary
            continue
        for n, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            for kind, val in _scan(line):
                hits.append(f"{path.relative_to(ROOT)}:{n}: {kind}: {val}")
    assert not hits, (
        "Network/device identifiers in tracked files — use <nas-host>/<nas-user> "
        "placeholders or nas.env (see nas.env.example):\n  " + "\n  ".join(hits[:40])
    )


def test_nas_env_is_gitignored():
    if not shutil.which("git"):
        pytest.skip("git not available")
    res = subprocess.run(["git", "check-ignore", "-q", "nas.env"], cwd=ROOT)
    if res.returncode == 128:
        pytest.skip("not a git checkout")
    assert res.returncode == 0, "nas.env must be git-ignored"


# --- the loaders -----------------------------------------------------------

def test_python_loader_env_wins_then_file_then_clear_error(tmp_path, monkeypatch):
    from photosearch import nas_config

    f = tmp_path / "nas.env"
    f.write_text('# comment\r\nexport NAS_HOST="u@h.example"\r\nPHOTOSEARCH_NAS_URL=http://h.example:8000/\n')
    monkeypatch.setenv("NAS_ENV_FILE", str(f))
    monkeypatch.delenv("NAS_HOST", raising=False)
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    assert nas_config.nas_setting("NAS_HOST") == "u@h.example"
    assert nas_config.require_nas_url() == "http://h.example:8000"
    monkeypatch.setenv("NAS_HOST", "env@wins")
    assert nas_config.nas_setting("NAS_HOST") == "env@wins"

    monkeypatch.setenv("NAS_ENV_FILE", str(tmp_path / "absent.env"))
    with pytest.raises(SystemExit) as e:
        nas_config.require_nas_url()
    assert "PHOTOSEARCH_NAS_URL" in str(e.value) and "nas.env.example" in str(e.value)


def _bash(script, env):
    base = {k: v for k, v in os.environ.items() if k not in ("NAS_HOST", "PHOTOSEARCH_NAS_URL")}
    return subprocess.run(["bash", "-c", script], cwd=ROOT, env={**base, **env},
                          capture_output=True, text=True)


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
def test_shell_loader_matches_the_python_one(tmp_path):
    f = tmp_path / "nas.env"
    f.write_text('# comment\r\nexport NAS_HOST="u@h.example"\r\nEVIL=$(touch pwned)\n')
    src = "set -eu; . scripts/nas-env.sh; "
    r = _bash(src + 'echo "$NAS_HOST"', {"NAS_ENV_FILE": str(f)})
    assert r.stdout.strip() == "u@h.example", r.stderr
    assert not (ROOT / "pwned").exists()          # parsed, never executed
    r = _bash(src + 'echo "$NAS_HOST"', {"NAS_ENV_FILE": str(f), "NAS_HOST": "env@wins"})
    assert r.stdout.strip() == "env@wins"
    r = _bash(src + 'nas_env_require NAS_HOST "ssh target"; echo reached',
              {"NAS_ENV_FILE": str(tmp_path / "absent.env")})
    assert r.returncode == 2 and "NAS_HOST is not set" in r.stderr and "reached" not in r.stdout


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
@pytest.mark.parametrize("cmd,var", [
    ("./sync-replica.sh", "NAS_HOST"),
    ("./debug-db.sh pull", "NAS_HOST"),
    ("./run-local-replica.sh", "PHOTOSEARCH_NAS_URL"),
])
def test_scripts_fail_fast_naming_the_variable(tmp_path, cmd, var):
    r = _bash(cmd, {"NAS_ENV_FILE": str(tmp_path / "absent.env")})
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert f"{var} is not set" in r.stderr
