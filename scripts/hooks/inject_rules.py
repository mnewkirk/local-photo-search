#!/usr/bin/env python3
"""PreToolUse hook: inject the matching `.claude/rules/*.md` on Bash/Grep/Glob.

Claude Code loads a path-scoped rule only when the session *opens* a file that
matches the rule's `paths:` globs. A session that only runs commands (Bash) or
searches (Grep/Glob) never opens code, so it never gets the rule. This hook
closes that gap: when a tool call names a subsystem (a file path matching a
rule's own `paths:` frontmatter, or one of the hand-written keywords below),
the rule body is returned as `hookSpecificOutput.additionalContext`.

Contract (https://code.claude.com/docs/en/hooks):
  stdin  : {"session_id", "tool_name", "tool_input", ...}
  stdout : {"hookSpecificOutput": {"hookEventName": "PreToolUse",
            "additionalContext": "..."}}   (must stay under 10,000 chars)
  exit 0 always. The hook never blocks, never sets a permission decision and
  never changes the tool input. Any error -> no output, exit 0.

Each rule is injected at most once per session; the injected names are kept in
a file under the temp dir, keyed by session_id.

Stdlib only; Python 3.11+.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

MAX_CONTEXT_CHARS = 9_500  # the documented cap is 10,000

# Hand-written keywords per rule (the file stem under .claude/rules/). Regexes,
# matched case-insensitively against the Bash command / Grep pattern+path+glob /
# Glob pattern+path. Keep them specific: a rule is injected only once per
# session, but a vague keyword would inject it into unrelated sessions.
KEYWORDS: dict[str, list[str]] = {
    "workers": [
        r"run-workers", r"setup-wsl-gpu", r"/api/worker", r"photosearch\.worker",
        r"worker_api", r"worker_processed", r"worker_claims", r"worker_client",
        r"MAX_PROCESS_ATTEMPTS", r"(?:cli\.py|photosearch)\s+worker\b",
        r"--round-robin", r"--native\b", r"clear-pass", r"\bsharpness\b",
        r"clip_embed", r"fleet-status", r"/api/health",
    ],
    "maintenance": [
        r"maintenance-sweep", r"maintenance_sweep", r"maintenance_sync",
        r"/api/admin/maintenance", r"\bgeocod", r"geonames", r"normalize-places",
        r"\bstacking\b", r"\bstacks?\b", r"\bdetect_stacks\b", r"sync-replica",
        r"run-local-replica", r"cleanup-orphans?\b", r"cleanup-orphan-faces",
        r"purge-nonimage", r"dedup[-_]photos", r"find-duplicate-photos",
        r"face_state", r"export-face-state", r"apply-face-state",
        r"backfill-dates", r"prune-missing", r"validate-data", r"repair-data",
    ],
    "faces": [
        r"recluster-faces", r"verify-face-labels", r"verify-person-matches",
        r"match-faces", r"add-persons?\b", r"list-persons", r"unmatch-person",
        r"restore-unmatch", r"dedupe-person-faces", r"resolve-duplicate-persons",
        r"review-faces", r"split-cluster", r"suggest-face-merges", r"face-clusters",
        r"correct-face", r"clear-matches", r"(?:export|import)-face-assignments",
        r"warm-face-crops", r"export-face-crops", r"person-coverage",
        r"backfill-face-exclusions", r"face_person_exclusions", r"face_verify",
        r"face_references", r"/api/faces", r"/api/persons", r"match_source",
        r"cluster_id", r"\bfrom\s+faces\b", r"\bupdate\s+faces\b", r"face_encodings",
    ],
    "ingest": [
        r"ingest-incoming", r"ingest_batches", r"\bingest\b", r"_incoming",
        r"refile-unknown-camera", r"unknown-camera", r"batch-advance",
        r"/api/batches", r"register_batch", r"batch_state", r"batch_api",
    ],
    "llm-passes": [
        r"derive-visual-tags", r"ground-categories", r"retry-failed-describe",
        r"stale-description-passes", r"visual-tag-collapse", r"bakeoff-keywords",
        r"mine-vocab", r"group-vocab", r"generation-history", r"backfill-generations",
        r"clean-garbage-tags", r"show-descriptions", r"recompute-aesthetic-overall",
        r"normalize-(?:subject-)?aesthetics", r"category-(?:visual|content)",
        r"lm[ -]?studio", r"lmstudio", r"\bollama\b", r"PHOTOSEARCH_LLM_",
        r"PHOTOSEARCH_TEXT_LLM", r"REASONING_EFFORT", r"model_eval",
        r"visual_tags", r"\bgenerations\b", r"rerun-passes",
    ],
    "video": [
        r"register-videos", r"unwrap-zip-videos", r"video-(?:probe|proxy|scan|subjects|describe)",
        r"media_type", r"\bffprobe\b", r"\bffmpeg\b", r"PASS_MEDIA",
        r"/api/videos", r"video_probe", r"video_proxy",
    ],
    "print": [
        r"\bupscale\b", r"split-export", r"shutterfly", r"\btopaz\b", r"\btpai",
        r"gigapixel", r"PHOTOSEARCH_TOPAZ", r"export_root", r"split_export",
        r"split-geometry", r"book_export", r"PHOTOSEARCH_BOOKS_DB",
    ],
    "frontend": [
        r"check-frontend-refs", r"check:refs", r"frontend/(?:dist|__tests__)",
        r"shared\.js", r"\bPS\.poll\b", r"\bPS\.PhotoModal\b", r"docker-build",
        r"/api/admin/(?:version|deploy|restart|rerun-passes|mirror-photos|replica-status|workers)",
        r"\bREPO_DIR\b",
    ],
    "search": [
        r"search_combined", r"request_log", r"request_intent", r"X-Photosearch-",
        r"/api/search", r"/api/ask", r"photo_index\.db", r"debug-db", r"\bsqlite3?\b",
        r"SCHEMA_VERSION", r"_init_schema", r"_SEARCH_INDEXES", r"mcp_server",
        r"set_photo_filepath", r"infer-locations", r"infer_location",
        r"idx_photos", r"\bPhotoDB\(", r"PHOTOSEARCH_DB\b",
    ],
}


def _project_dir() -> Path:
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


def parse_rule(path: Path) -> tuple[list[str], str]:
    """Return (paths globs, body) for a rule file. Parses the frontmatter itself
    so the glob table can never drift from what Claude Code loads."""
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n?", text, re.S)
    if not m:
        return [], text
    globs = re.findall(r'^\s*-\s*["\']?([^"\'\n]+?)["\']?\s*$', m.group(1), re.M)
    return globs, text[m.end():].lstrip("\n")


def glob_to_regex(glob: str) -> re.Pattern:
    out, i = [], 0
    while i < len(glob):
        c = glob[i]
        if c == "*":
            if glob[i:i + 2] == "**":
                out.append(".*")
                i += 2
                if glob[i:i + 1] == "/":
                    i += 1
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(c))
        i += 1
    # Anchored at a path-component boundary at the front, so absolute paths and
    # `./` prefixes match; anchored at the end so `worker*.py` != `worker.pyc`.
    return re.compile(r"(?:^|/)" + "".join(out) + r"$")


def tool_text(tool_name: str, tool_input: dict) -> str:
    if tool_name == "Bash":
        keys = ("command",)
    elif tool_name in ("Grep", "Glob"):
        keys = ("pattern", "path", "glob")
    else:
        return ""
    return "\n".join(str(tool_input[k]) for k in keys if tool_input.get(k))


_TOKEN_SPLIT = re.compile(r"[\s\"'`=:,;|&<>()\[\]{}]+")


def path_tokens(text: str) -> list[str]:
    toks = []
    for t in _TOKEN_SPLIT.split(text):
        t = t.removeprefix("./").rstrip("/")
        if t:
            toks.append(t)
    return toks


def load_rules(project_dir: Path) -> dict[str, tuple[list[re.Pattern], str]]:
    rules = {}
    for f in sorted((project_dir / ".claude" / "rules").glob("*.md")):
        globs, body = parse_rule(f)
        rules[f.stem] = ([glob_to_regex(g) for g in globs], body)
    return rules


def match_rules(text: str, rules: dict) -> list[str]:
    """Rule names that match `text`, best match (most hits) first."""
    toks = path_tokens(text)
    scores: dict[str, int] = {}
    for name, (globs, _body) in rules.items():
        hits = sum(1 for t in toks for g in globs if g.search(t))
        for kw in KEYWORDS.get(name, ()):
            if re.search(kw, text, re.I):
                hits += 1
        if hits:
            scores[name] = hits
    return sorted(scores, key=lambda n: (-scores[n], n))


def _state_file(session_id: str, state_dir: Path) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)[:100]
    return state_dir / f"photosearch-rules-{safe}.json"


def _load_state(f: Path) -> list[str]:
    try:
        data = json.loads(f.read_text())
        return [s for s in data if isinstance(s, str)]
    except (OSError, ValueError, TypeError):
        return []


def build_output(payload: dict, project_dir: Path, state_dir: Path) -> str | None:
    tool_name = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    text = tool_text(tool_name, tool_input)
    if not text:
        return None
    rules = load_rules(project_dir)
    names = match_rules(text, rules)
    if not names:
        return None

    session_id = str(payload.get("session_id") or "")
    state = _state_file(session_id, state_dir) if session_id else None
    seen = _load_state(state) if state else []
    todo = [n for n in names if n not in seen]
    if not todo:
        return None

    chunks, injected, size = [], [], 0
    for n in todo:
        chunk = (
            f"Project rule .claude/rules/{n}.md (Claude Code loads it when a "
            f"matching file is opened; this tool call names the {n} subsystem):\n\n"
            f"{rules[n][1].strip()}"
        )
        if chunks and size + len(chunk) > MAX_CONTEXT_CHARS:
            continue  # stays un-injected; arrives on the next matching call
        chunks.append(chunk)
        injected.append(n)
        size += len(chunk) + 2
    if not chunks:
        return None

    if state:
        try:
            state.write_text(json.dumps(seen + injected))
        except OSError:
            pass  # at worst a rule is injected twice
    context = "\n\n".join(chunks)
    return json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "additionalContext": context[:MAX_CONTEXT_CHARS + 600],
    }})


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        out = build_output(payload, _project_dir(), Path(tempfile.gettempdir()))
        if out:
            sys.stdout.write(out)
    except Exception:  # a broken hook must never break a session
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
