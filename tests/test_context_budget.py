"""Guard: the per-session context stays small.

CLAUDE.md, the photo-search SKILL.md body and every `.claude/rules/*.md` are
loaded into Claude Code sessions. Before 2026-10-19 CLAUDE.md had grown to
258 KB (~62k tokens in every session) because every incident write-up was
appended to it. docs/plans/context-diet.md moved that history to docs/notes/,
runbooks to the skill's reference/, and subsystem traps to path-scoped rules;
these caps keep it from growing back.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = ROOT / "CLAUDE.md"
SKILL_MD = ROOT / ".claude" / "skills" / "photo-search" / "SKILL.md"
RULES = sorted((ROOT / ".claude" / "rules").glob("*.md"))

HOW_TO_FIX = (
    "CLAUDE.md is a rules file. Narratives → docs/notes/, runbooks → skill "
    "reference/, subsystem traps → .claude/rules/. See CLAUDE.md § Where "
    "knowledge goes.")


def _size(path):
    text = path.read_text()
    return len(text.encode()), text.count("\n") + (not text.endswith("\n"))


def test_claude_md_within_budget():
    nbytes, nlines = _size(CLAUDE_MD)
    assert nbytes <= 24_576, f"CLAUDE.md is {nbytes} B (cap 24,576). {HOW_TO_FIX}"
    assert nlines <= 300, f"CLAUDE.md is {nlines} lines (cap 300). {HOW_TO_FIX}"


def test_claude_md_has_no_narratives():
    lines, fence = [], False
    for line in CLAUDE_MD.read_text().splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
        elif not fence:
            lines.append(line)
    h3 = [l for l in lines if l.startswith("### ")]
    dated = [l for l in lines if re.match(r"^\s*-?\s*\*\*?\d{4}-\d{2}-\d{2}", l)]
    assert not h3, f"CLAUDE.md has ### headings {h3[:3]}. {HOW_TO_FIX}"
    assert not dated, f"CLAUDE.md has dated entries {dated[:3]}. {HOW_TO_FIX}"


def test_skill_md_within_budget():
    nbytes, nlines = _size(SKILL_MD)
    assert nbytes <= 20_480, (
        f"SKILL.md is {nbytes} B (cap 20,480). Move detail to "
        f".claude/skills/photo-search/reference/.")
    assert nlines <= 300, f"SKILL.md is {nlines} lines (cap 300)."


def test_rules_exist():
    assert RULES, "no .claude/rules/*.md found"


@pytest.mark.parametrize("rule", RULES, ids=lambda p: p.name)
def test_rule_is_path_scoped_and_small(rule):
    text = rule.read_text()
    nbytes = len(text.encode())
    assert nbytes <= 8_192, (
        f"{rule.name} is {nbytes} B (cap 8,192). Keep one bullet per trap; "
        f"history goes to docs/notes/.")
    fm = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert fm and re.search(r"^paths:", fm.group(1), re.M), (
        f"{rule.name}: rule loads unconditionally — add paths: or move to CLAUDE.md")


@pytest.mark.parametrize("rule", RULES, ids=lambda p: p.name)
def test_rule_paths_match_real_files(rule):
    """A glob that matches nothing means the rule never loads."""
    fm = re.match(r"^---\n(.*?)\n---\n", rule.read_text(), re.S)
    globs = re.findall(r'^\s*-\s*"([^"]+)"', fm.group(1), re.M) if fm else []
    dead = [g for g in globs if not any(ROOT.glob(g))]
    assert not dead, f"{rule.name}: paths match no file: {dead}"
