"""This machine's NAS connection settings — never hard-coded.

The repo is public, so the NAS's hostname / IP / login stay out of tracked
files. They come from the environment, or from a git-ignored ``nas.env`` at the
repo root (see ``nas.env.example``). ``scripts/nas-env.sh`` is the shell twin of
this module; keep the two in step (same file, same precedence: env wins).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def nas_env_file() -> Path:
    override = os.environ.get("NAS_ENV_FILE")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "nas.env"


def _read_file() -> dict[str, str]:
    path = nas_env_file()
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, val = line.partition("=")
        key = key.strip()
        if not sep or not _KEY.match(key):
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        out[key] = val
    return out


def nas_setting(name: str) -> str | None:
    """Environment first, then ``nas.env``. Empty counts as unset."""
    return os.environ.get(name) or _read_file().get(name) or None


def require_nas_setting(name: str, what: str = "") -> str:
    """The value, or exit naming the variable.

    There is deliberately no built-in default: a wrong host fails slowly and
    confusingly, an unset one should fail at once and say what to set.
    """
    val = nas_setting(name)
    if not val:
        raise SystemExit(
            f"ERROR: {name} is not set{' — ' + what if what else ''}.\n"
            f"  Set it in {nas_env_file()} (copy nas.env.example), or export {name}."
        )
    return val


def require_nas_url() -> str:
    return require_nas_setting(
        "PHOTOSEARCH_NAS_URL", "the NAS web URL, e.g. http://<nas-host>:8000"
    ).rstrip("/")
