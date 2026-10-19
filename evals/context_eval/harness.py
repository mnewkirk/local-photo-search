#!/usr/bin/env python3
"""Context eval harness: does a trimmed CLAUDE.md still make Claude Code behave?

Runs Claude Code headless (`claude -p --output-format stream-json`) against this
repo and scores four suites, recording cost and token usage per run so the
before/after of a CLAUDE.md restructure can be compared on behaviour AND cost,
not on file sizes.

Suites (one JSONL file each, beside this script):

  recall          trap-recall questions; graded by regex on the final answer
  rules           "read <file> then answer"; graded by regex (exercises
                  path-scoped .claude/rules/*.md once they exist)
  skills          vague prompts; pass = the photo-search skill was invoked
  tasks           small coding tasks in a throwaway git worktree; graded by
                  deterministic checks on the diff

Usage:
  python evals/context_eval/harness.py run --config before [--suite recall]
        [--only ID ...] [--repeat 3] [--model sonnet] [--parallel 2]
        [--ref HEAD] [--limit N] [--dry-run]
  python evals/context_eval/harness.py report --configs before after

`--config` is a free label; results land in results/<config>/<suite>.jsonl
(appended), each row self-describing (git sha, CLAUDE.md bytes, model).
Read-only suites run in the repo root (so the real CLAUDE.md, skill and
memory load). `tasks` run in a fresh `git worktree` of `--ref`, so the
CLAUDE.md under test must be COMMITTED on that ref.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS = HERE / "results"
SUITES = ("recall", "rules", "skills", "tasks")

READ_ONLY_TOOLS = [
    "Read", "Grep", "Glob", "Skill", "Task",
    "Bash(grep *)", "Bash(rg *)", "Bash(cat *)", "Bash(sed *)", "Bash(head *)",
    "Bash(tail *)", "Bash(ls *)", "Bash(find *)", "Bash(wc *)", "Bash(git log *)",
    "Bash(git show *)", "Bash(git diff *)", "Bash(git status *)",
]
EDIT_TOOLS = READ_ONLY_TOOLS + ["Edit", "Write", "MultiEdit", "NotebookEdit"]

RECALL_SYSTEM = (
    "You are answering a question about THIS repository for a maintainer. "
    "Answer in at most 8 lines, naming the exact identifiers, files, numbers "
    "or flags involved. Do not modify any file."
)
TASK_SYSTEM = (
    "Make the requested change directly in the working tree. Do not run the "
    "test suite, do not commit, do not ask questions; make reasonable choices "
    "and finish. Follow this repository's conventions."
)


# ---------------------------------------------------------------- grading --

def _any(patterns, text):
    return any(re.search(p, text, re.I | re.S) for p in patterns)


def grade_text(item: dict, text: str) -> tuple[bool, dict]:
    """`expect`: every regex must match. `any`: list of groups, each group
    needs >= 1 match. `reject`: no regex may match."""
    detail = {"expect_missing": [], "any_missing": [], "reject_hit": []}
    for p in item.get("expect", []):
        if not re.search(p, text, re.I | re.S):
            detail["expect_missing"].append(p)
    for group in item.get("any", []):
        if not _any(group, text):
            detail["any_missing"].append(group)
    for p in item.get("reject", []):
        if re.search(p, text, re.I | re.S):
            detail["reject_hit"].append(p)
    ok = not (detail["expect_missing"] or detail["any_missing"] or detail["reject_hit"])
    return ok, detail


def added_lines(worktree: Path) -> dict[str, list[str]]:
    """Added lines per path: `+` lines of `git diff HEAD` for tracked files,
    whole content for untracked (non-ignored) files."""
    out: dict[str, list[str]] = {}
    diff = subprocess.run(["git", "diff", "HEAD", "--unified=0", "--no-color"],
                          cwd=worktree, capture_output=True, text=True).stdout
    cur = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            p = line[4:]
            cur = p[2:] if p.startswith("b/") else None
        elif line.startswith("+") and not line.startswith("+++") and cur:
            out.setdefault(cur, []).append(line[1:])
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                               cwd=worktree, capture_output=True, text=True).stdout.split()
    for rel in untracked:
        p = worktree / rel
        try:
            out.setdefault(rel, []).extend(p.read_text(errors="replace").splitlines())
        except (OSError, IsADirectoryError):
            pass
    return out


def grade_task(item: dict, added: dict[str, list[str]]) -> tuple[bool, dict]:
    """Checks: {type: added_grep|added_not_grep|any_added_grep, pattern|patterns,
    paths?: [globs]}. `paths` restricts which files' added lines are searched."""
    results = []
    for chk in item["checks"]:
        globs = chk.get("paths")
        text = "\n".join(
            l for path, lines in added.items()
            if not globs or any(fnmatch.fnmatch(path, g) for g in globs)
            for l in lines
        )
        t = chk["type"]
        if t == "added_grep":
            ok = re.search(chk["pattern"], text, re.I | re.M) is not None
        elif t == "added_not_grep":
            ok = re.search(chk["pattern"], text, re.I | re.M) is None
        elif t == "any_added_grep":
            ok = _any(chk["patterns"], text)
        else:
            raise ValueError(f"unknown check type {t!r} in task {item['id']}")
        results.append({"check": chk, "ok": ok})
    return all(r["ok"] for r in results), {"checks": results, "files_touched": sorted(added)}


# ---------------------------------------------------------- claude runner --

def claude_cmd(prompt: str, *, model: str, max_turns: int, budget: float,
               edit: bool, system: str) -> list[str]:
    cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose",
           "--strict-mcp-config", "--max-turns", str(max_turns),
           "--max-budget-usd", f"{budget:.2f}", "--append-system-prompt", system]
    if model:
        cmd += ["--model", model]
    if edit:
        cmd += ["--permission-mode", "acceptEdits", "--allowedTools", *EDIT_TOOLS]
    else:
        cmd += ["--allowedTools", *READ_ONLY_TOOLS]
    return cmd


def parse_stream(raw: str) -> dict:
    """Pull the pieces we score from a stream-json transcript."""
    tools, skills, result, init = [], [], None, {}
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = ev.get("type")
        if t == "system" and ev.get("subtype") == "init":
            init = ev
        elif t == "assistant":
            for block in (ev.get("message") or {}).get("content") or []:
                if block.get("type") == "tool_use":
                    tools.append(block.get("name"))
                    if block.get("name") == "Skill":
                        skills.append(str((block.get("input") or {}).get("skill", "")))
        elif t == "result":
            result = ev
    text = (result or {}).get("result") or ""
    if not text and result and result.get("subtype") not in (None, "success"):
        text = f"<{result.get('subtype')}>"
    offered = init.get("tools") or []
    return {
        "text": text,
        "tools": tools,
        "skills": skills,
        "skill_tool_offered": ("Skill" in offered) if offered else None,
        "offered_skills": [s for s in (init.get("skills") or init.get("slash_commands") or [])
                           if "photo" in str(s).lower()],
        "usage": (result or {}).get("usage") or {},
        "model_usage": (result or {}).get("modelUsage") or {},
        "cost_usd": (result or {}).get("total_cost_usd"),
        "num_turns": (result or {}).get("num_turns"),
        "duration_ms": (result or {}).get("duration_ms"),
        "subtype": (result or {}).get("subtype"),
        "session_id": (result or {}).get("session_id"),
        "model": init.get("model"),
        "is_error": bool((result or {}).get("is_error")),
    }


def run_claude(prompt: str, cwd: Path, *, model: str, max_turns: int, budget: float,
               edit: bool, system: str, timeout: int) -> dict:
    cmd = claude_cmd(prompt, model=model, max_turns=max_turns, budget=budget,
                     edit=edit, system=system)
    env = dict(os.environ)
    env.pop("CLAUDECODE", None)  # allow nesting when launched from inside a session
    # A local gateway streams nothing while it prefills a ~100k-token prompt
    # (~9 min on LM Studio), so Claude Code's 5-min first-byte wait kills and
    # retries it from zero forever. The run's own --timeout is the real cap.
    for var in ("CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS", "API_TIMEOUT_MS"):
        env.setdefault(var, str(timeout * 1000))
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout, env=env, stdin=subprocess.DEVNULL)
        parsed = parse_stream(proc.stdout)
        parsed["exit_code"] = proc.returncode
        parsed["stderr_tail"] = proc.stderr[-2000:]
    except subprocess.TimeoutExpired as e:
        parsed = parse_stream(e.stdout or "")
        parsed.update(exit_code=None, stderr_tail="TIMEOUT", subtype="timeout", is_error=True)
    parsed["wall_s"] = round(time.time() - t0, 1)
    return parsed


# -------------------------------------------------------------- suites -----

def load_items(suite: str) -> list[dict]:
    path = HERE / {"recall": "recall.jsonl", "rules": "rule_triggers.jsonl",
                   "skills": "skill_triggers.jsonl", "tasks": "tasks.jsonl"}[suite]
    items = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if line and not line.startswith("#"):
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{n}: bad JSON: {e}")
    ids = [i["id"] for i in items]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise SystemExit(f"{path}: duplicate ids {sorted(dup)}")
    return items


def context_fingerprint(root: Path) -> dict:
    def size(rel):
        p = root / rel
        return p.stat().st_size if p.exists() else None
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                         capture_output=True, text=True).stdout.strip()
    rules = sorted((root / ".claude" / "rules").glob("*.md")) if (root / ".claude" / "rules").exists() else []
    return {
        "git_sha": sha,
        "claude_md_bytes": size("CLAUDE.md"),
        "skill_md_bytes": size(".claude/skills/photo-search/SKILL.md"),
        "rules_files": len(rules),
        "rules_bytes": sum(p.stat().st_size for p in rules),
    }


class Worktree:
    def __init__(self, ref: str):
        self.ref = ref
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="ctxeval-"))
        subprocess.run(["git", "worktree", "add", "--detach", "--quiet", str(self.path), self.ref],
                       cwd=REPO, check=True, capture_output=True, text=True)
        return self.path

    def __exit__(self, *exc):
        subprocess.run(["git", "worktree", "remove", "--force", str(self.path)],
                       cwd=REPO, capture_output=True, text=True)
        shutil.rmtree(self.path, ignore_errors=True)


INFRA_ABORT_STREAK = 3
_INFRA_MARKERS = ("exceed_context_size", "exceeds the available context",
                  "check your inference gateway", "connection refused",
                  "overloaded_error", "rate_limit_error")


def infra_error(res: dict) -> str | None:
    """Why a run says nothing about the context under test (gateway down,
    model context overflow, timeout), else None. These must not count as
    task FAILs: they would read as the context regressing."""
    text = res.get("text") or ""
    if res.get("subtype") == "timeout":
        return "timeout"
    if res.get("is_error") and text.startswith("API Error"):
        m = re.search(r'"message":"([^"]{0,160})', text)
        return "api_error: " + (m.group(1) if m else text[:160])
    if any(k in text.lower() for k in _INFRA_MARKERS):
        return "gateway: " + text[:160]
    if res.get("is_error") and not sum(int(v or 0) for v in (res.get("usage") or {}).values()
                                       if isinstance(v, (int, float))):
        return "no_usage: " + (res.get("stderr_tail") or "")[-160:].strip()
    return None


def run_one(suite: str, item: dict, rep: int, args) -> dict:
    row = {"suite": suite, "id": item["id"], "config": args.config, "rep": rep,
           "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "requested_model": args.model}
    if suite == "tasks":
        with Worktree(args.ref) as wt:
            row["context"] = context_fingerprint(wt)
            res = run_claude(item["prompt"], wt, model=args.model, max_turns=args.max_turns,
                             budget=args.budget, edit=True, system=TASK_SYSTEM,
                             timeout=args.timeout)
            ok, detail = grade_task(item, added_lines(wt))
    else:
        cwd = REPO
        row["context"] = context_fingerprint(cwd)
        prompt = item["prompt"]
        if suite == "rules":
            prompt = (f"First read the whole file {item['read']} with the Read tool. "
                      f"Then answer: {item['prompt']}")
        res = run_claude(prompt, cwd, model=args.model, max_turns=args.max_turns,
                         budget=args.budget, edit=False, system=RECALL_SYSTEM,
                         timeout=args.timeout)
        if suite == "skills":
            ok = any("photo-search" in s for s in res["skills"])
            detail = {"skills_invoked": res["skills"],
                      "skill_tool_offered": res["skill_tool_offered"],
                      "photo_skills_offered": res["offered_skills"]}
        else:
            ok, detail = grade_text(item, res["text"])
    infra = infra_error(res)
    row.update({
        "infra_error": infra,
        "pass": ok and not res["is_error"] and not infra,
        "detail": detail,
        "answer": res["text"][:4000],
        "tools": res["tools"],
        "skills": res["skills"],
        "usage": res["usage"],
        "model_usage": res["model_usage"],
        "cost_usd": res["cost_usd"],
        "num_turns": res["num_turns"],
        "duration_ms": res["duration_ms"],
        "wall_s": res["wall_s"],
        "model": res["model"],
        "subtype": res["subtype"],
        "exit_code": res["exit_code"],
        "error": infra or (res["stderr_tail"] if res["is_error"] else None),
    })
    return row


def cmd_run(args) -> int:
    suites = [args.suite] if args.suite else list(SUITES)
    jobs = []
    for suite in suites:
        items = load_items(suite)
        if args.only:
            items = [i for i in items if i["id"] in set(args.only)]
        if args.limit:
            items = items[: args.limit]
        for item in items:
            for rep in range(1, args.repeat + 1):
                jobs.append((suite, item, rep))
    if not jobs:
        print("nothing to run", file=sys.stderr)
        return 2
    print(f"{len(jobs)} runs: config={args.config} model={args.model or 'default'} "
          f"ref={args.ref} parallel={args.parallel}", file=sys.stderr)
    if args.dry_run:
        for suite, item, rep in jobs:
            print(f"  {suite:7s} {item['id']:34s} rep{rep}")
        return 0
    outdir = RESULTS / args.config
    outdir.mkdir(parents=True, exist_ok=True)
    passed = total = infra_n = infra_streak = 0
    cost = 0.0

    def work(job):
        suite, item, rep = job
        try:
            return run_one(suite, item, rep, args)
        except Exception as e:  # keep the batch going; the row records the failure
            return {"suite": suite, "id": item["id"], "config": args.config, "rep": rep,
                    "pass": False, "error": f"harness: {type(e).__name__}: {e}",
                    "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}

    with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for row in pool.map(work, jobs):
            with open(outdir / f"{row['suite']}.jsonl", "a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
            passed += bool(row.get("pass"))
            infra_n += bool(row.get("infra_error"))
            cost += row.get("cost_usd") or 0.0
            mark = ("PASS" if row.get("pass")
                    else "INFRA" if row.get("infra_error") else "FAIL")
            extra = f" ({row['error'][:80]})" if row.get("error") else ""
            print(f"[{total}/{len(jobs)}] {mark} {row['suite']}/{row['id']} rep{row['rep']} "
                  f"${(row.get('cost_usd') or 0):.3f} {row.get('num_turns') or '-'}t"
                  f" {row.get('wall_s') or '-'}s{extra}", file=sys.stderr)
            # A broken gateway fails every job the same way; don't spend the
            # whole batch proving it.
            infra_streak = infra_streak + 1 if row.get("infra_error") else 0
            if infra_streak >= INFRA_ABORT_STREAK:
                print(f"\naborting: {infra_streak} infra errors in a row — fix the "
                      f"gateway/model first ({row['infra_error'][:120]})", file=sys.stderr)
                pool.shutdown(wait=False, cancel_futures=True)
                break
    scored = total - infra_n
    print(f"\n{passed}/{scored} passed ({infra_n} infra errors not scored), "
          f"${cost:.2f} total → {outdir}", file=sys.stderr)
    return 0


# -------------------------------------------------------------- report -----

def _load_results(config: str) -> list[dict]:
    rows = []
    d = RESULTS / config
    for f in sorted(d.glob("*.jsonl")) if d.exists() else []:
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _input_tokens(u: dict) -> int:
    return sum(int(u.get(k) or 0) for k in
               ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))


def _agg(rows):
    infra = sum(bool(r.get("infra_error")) for r in rows)
    rows = [r for r in rows if not r.get("infra_error")]
    n = len(rows)
    if not n:
        return {"n": 0, "infra": infra}
    return {
        "n": n,
        "infra": infra,
        "pass_rate": sum(bool(r.get("pass")) for r in rows) / n,
        "cost": sum(r.get("cost_usd") or 0 for r in rows) / n,
        "input_tok": sum(_input_tokens(r.get("usage") or {}) for r in rows) / n,
        "cache_read": sum(int((r.get("usage") or {}).get("cache_read_input_tokens") or 0) for r in rows) / n,
        "turns": sum(r.get("num_turns") or 0 for r in rows) / n,
        "errors": sum(bool(r.get("error")) for r in rows),
    }


def cmd_report(args) -> int:
    data = {c: _load_results(c) for c in args.configs}
    for c, rows in data.items():
        if not rows:
            print(f"warning: no results for config {c!r}", file=sys.stderr)
    print("# Context eval report\n")
    for c, rows in data.items():
        if rows:
            fp = next((r.get("context") for r in rows if r.get("context")), {})
            print(f"- **{c}**: {len(rows)} runs; git {fp.get('git_sha')}, CLAUDE.md "
                  f"{fp.get('claude_md_bytes')} B, SKILL.md {fp.get('skill_md_bytes')} B, "
                  f"rules {fp.get('rules_files')} files / {fp.get('rules_bytes')} B; "
                  f"models {sorted({r.get('model') for r in rows if r.get('model')})}")
    print("\n## Per suite\n")
    print("| suite | config | runs | pass rate | mean cost $ | mean input tok | mean cache-read tok | mean turns | errors |")
    print("|---|---|---|---|---|---|---|---|---|")
    for suite in SUITES:
        for c, rows in data.items():
            a = _agg([r for r in rows if r.get("suite") == suite])
            if a["n"] or a["infra"]:
                if not a["n"]:
                    print(f"| {suite} | {c} | 0 | — | — | — | — | — | {a['infra']} infra |")
                    continue
                infra_note = f" ({a['infra']} infra, unscored)" if a["infra"] else ""
                print(f"| {suite} | {c} | {a['n']} | {a['pass_rate']:.0%} | {a['cost']:.3f} | "
                      f"{a['input_tok']:,.0f} | {a['cache_read']:,.0f} | {a['turns']:.1f} | "
                      f"{a['errors']}{infra_note} |")
    print("\n## Per item (pass count / runs per config)\n")
    ids = sorted({(r["suite"], r["id"]) for rows in data.values() for r in rows})
    print("| suite | id | " + " | ".join(args.configs) + " | note |")
    print("|---|---|" + "---|" * len(args.configs) + "---|")
    for suite, iid in ids:
        cells, rates = [], []
        for c in args.configs:
            rs = [r for r in data[c] if r["suite"] == suite and r["id"] == iid
                  and not r.get("infra_error")]
            if rs:
                p = sum(bool(r.get("pass")) for r in rs)
                cells.append(f"{p}/{len(rs)}")
                rates.append(p / len(rs))
            else:
                cells.append("—")
        note = ""
        if len(rates) >= 2 and rates[-1] < rates[0]:
            note = "REGRESSED"
        elif len(rates) >= 2 and rates[-1] > rates[0]:
            note = "improved"
        print(f"| {suite} | {iid} | " + " | ".join(cells) + f" | {note} |")
    if args.failures:
        print("\n## Failure details\n")
        for c, rows in data.items():
            for r in rows:
                if not r.get("pass") and not r.get("infra_error"):
                    print(f"### {c} / {r['suite']} / {r['id']} rep{r.get('rep')}\n")
                    if r.get("error"):
                        print(f"error: `{r['error'][:300]}`\n")
                    print("```json\n" + json.dumps(r.get("detail"), indent=1)[:1500] + "\n```\n")
                    if r.get("answer"):
                        print("answer:\n\n> " + r["answer"][:800].replace("\n", "\n> ") + "\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one or all suites")
    r.add_argument("--config", required=True, help="label for this configuration, e.g. before / after")
    r.add_argument("--suite", choices=SUITES)
    r.add_argument("--only", nargs="*", help="item ids")
    r.add_argument("--limit", type=int, default=0, help="first N items of each suite")
    r.add_argument("--repeat", type=int, default=3)
    r.add_argument("--model", default="sonnet", help="claude --model value; '' for the session default")
    r.add_argument("--parallel", type=int, default=2)
    r.add_argument("--ref", default="HEAD", help="git ref the tasks worktree is created from")
    r.add_argument("--max-turns", type=int, default=25)
    r.add_argument("--budget", type=float, default=1.50, help="per-run --max-budget-usd")
    r.add_argument("--timeout", type=int, default=900, help="seconds per run")
    r.add_argument("--dry-run", action="store_true", help="list the runs, call nothing")
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("report", help="compare configs")
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--failures", action="store_true", help="append failure details")
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
