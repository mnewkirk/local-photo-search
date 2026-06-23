#!/usr/bin/env python3
"""Syntax-check the inline <script> blocks of the no-build-step frontend HTML.

The frontend uses React.createElement (no JSX, no bundler), so each page's logic
lives in an inline <script> tag. `node --check` validates JS *syntax* (undefined
browser globals like React/document are fine — it doesn't resolve references),
which is exactly what catches the breakage from moving components between files.

Usage: scripts/check_inline_js.py frontend/dist/status.html [more.html ...]
Exit non-zero if any inline script fails to parse.
"""
import re
import subprocess
import sys
import tempfile

SCRIPT_RE = re.compile(r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>", re.DOTALL | re.IGNORECASE)

# UPPER_SNAKE identifiers (must contain an underscore) — the module-level
# const style in this codebase (STACK_DEFAULTS, PASS_COLORS, …). `node --check`
# only validates syntax, so a component referencing one that wasn't moved with
# it passes the parse but throws ReferenceError at runtime. This catches that.
CONST_RE = re.compile(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]*[A-Z0-9]\b")
DECL_RE = re.compile(r"\b(?:var|let|const|function)\s+([A-Z][A-Z0-9]*_[A-Z0-9_]*[A-Z0-9])\b")
# strip line/block comments so commented-out names don't count as "declared"
COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def undeclared_consts(body: str) -> set:
    code = COMMENT_RE.sub("", body)
    declared = set(DECL_RE.findall(code))
    used = set(CONST_RE.findall(code))
    return used - declared


def check(path: str) -> bool:
    html = open(path, encoding="utf-8").read()
    ok = True
    for i, m in enumerate(SCRIPT_RE.finditer(html)):
        if "src=" in m.group("attrs"):
            continue  # external script (React CDN, shared.js)
        body = m.group("body").strip()
        if not body:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(body)
            tmp = fh.name
        r = subprocess.run(["node", "--check", tmp], capture_output=True, text=True)
        if r.returncode != 0:
            ok = False
            print(f"✗ {path} (inline script #{i}):\n{r.stderr.strip()}\n")
        missing = undeclared_consts(body)
        if missing:
            ok = False
            print(f"✗ {path} (inline script #{i}): referenced but undeclared "
                  f"UPPER_SNAKE const(s): {', '.join(sorted(missing))}\n")
    if ok:
        print(f"✓ {path}: inline JS valid")
    return ok

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(0 if all(check(p) for p in sys.argv[1:]) else 1)
