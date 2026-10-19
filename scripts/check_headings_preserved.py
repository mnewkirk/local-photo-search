#!/usr/bin/env python3
"""Check that the context diet lost no section.

Every `##`/`###` heading CLAUDE.md and the photo-search SKILL.md had before the
diet (docs/notes/_headings-2026-10-19.txt) must still appear verbatim as a
heading in CLAUDE.md, docs/, .claude/rules/ or .claude/skills/. Exits 1 and
lists the missing ones otherwise. See docs/plans/context-diet.md step 1.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "docs/notes/_headings-2026-10-19.txt"
SEARCH = [ROOT / "CLAUDE.md", ROOT / "docs", ROOT / ".claude/rules", ROOT / ".claude/skills"]


def headings(path: Path) -> set[str]:
    found, fence = set(), False
    for line in path.read_text(errors="replace").splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
        elif not fence:
            m = re.match(r"^#{2,3} (.+)$", line)
            if m:
                found.add(m.group(1).strip())
    return found


def main() -> int:
    wanted = [l.lstrip("#").strip() for l in SNAPSHOT.read_text().splitlines()
              if re.match(r"^#{2,3} ", l)]
    have: set[str] = set()
    for base in SEARCH:
        for p in [base] if base.is_file() else sorted(base.rglob("*.md")):
            have |= headings(p)
    missing = [h for h in wanted if h not in have]
    print(f"{len(wanted) - len(missing)}/{len(wanted)} headings preserved")
    for h in missing:
        print(f"  missing: {h}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
