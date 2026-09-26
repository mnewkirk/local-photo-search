"""Shared storage + instrumentation for the model-eval harnesses.

One contract for every pass the new LM Studio models are measured on
(aesthetics, describe, category-content, keywords, verify) — the harness CLIs
in `evals/`, the labelling API (`photosearch/model_eval_api.py`) and the
`/eval/models` page all read and write through here, so they cannot disagree
about a file format. Plan: `docs/plans/model-eval-harnesses.md`.

Layout under `eval_dir()` (`PHOTOSEARCH_MODEL_EVAL_DIR`, default
./evals/model-evals, git-ignored):

    originals/<photo_id>          full-size bytes, fetched ONCE from the server
    <pass>/runs/<variant>.json    one cached run per (pass, variant)
    <pass>/*.json                 per-pass samples / labels / inputs

Same rules as the visual-tag eval (`photosearch/visual_tag_eval.py`), which
this generalises rather than copies:

- Files, not DB rows: `sync-replica.sh` swaps the replica DB wholesale, and
  hand labels are the one thing here that cannot be regenerated.
- Every write is atomic: a kill mid-dump must not cost labels or GPU time.
- A run records the model that ACTUALLY ran (`describe.effective_model`), and
  refuses to resume under a different model / prompt / input.
- A photo that failed for transport reasons is never cached, so re-running
  without --force fills exactly the gaps.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

DEFAULT_SERVER = "http://localhost:8001"
STORED = "stored"            # pseudo-variant: what the DB holds today
# Below this many observations a ratio is noise (same bar as the visual eval).
MIN_N = 3
# This many consecutive failures looks like LM Studio dropping, not bad photos
# (2026-09-26: ~80 photos in a row got 400s after a dropped connection).
CONSECUTIVE_FAIL_WARN = 5

VARIANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def eval_dir() -> Path:
    return Path(os.environ.get("PHOTOSEARCH_MODEL_EVAL_DIR", "./evals/model-evals"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Atomic JSON
# --------------------------------------------------------------------------

def read_json(path: Path, default):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def pass_dir(pass_: str) -> Path:
    if not VARIANT_RE.match(pass_):
        raise ValueError(f"bad pass name {pass_!r}")
    return eval_dir() / pass_


def read_pass_file(pass_: str, name: str, default):
    return read_json(pass_dir(pass_) / name, default)


def write_pass_file(pass_: str, name: str, data) -> None:
    write_json_atomic(pass_dir(pass_) / name, data)


# --------------------------------------------------------------------------
# Run cache
# --------------------------------------------------------------------------

def check_variant(variant: str) -> None:
    if not VARIANT_RE.match(variant or "") or variant == STORED:
        raise SystemExit(f"Bad variant name {variant!r} "
                         f"(letters, digits, . _ - ; '{STORED}' is reserved).")


def run_path(pass_: str, variant: str) -> Path:
    return pass_dir(pass_) / "runs" / f"{variant}.json"


def load_run(pass_: str, variant: str) -> Optional[dict]:
    return read_json(run_path(pass_, variant), None)


def save_run(pass_: str, variant: str, run: dict) -> None:
    write_json_atomic(run_path(pass_, variant), run)


def list_variants(pass_: str) -> list[str]:
    d = pass_dir(pass_) / "runs"
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json") if not p.name.startswith("."))


def open_run(pass_: str, variant: str, *, force: bool = False, **identity) -> dict:
    """Load `variant`'s cached run, or start a new one.

    `identity` is what makes a run ONE experiment (effective_model,
    prompt_sha, input_source, ...). Resuming under different values would mix
    two experiments in a file that reports as one, so it refuses."""
    check_variant(variant)
    run = None if force else load_run(pass_, variant)
    if run is not None:
        for key, now in identity.items():
            if run.get(key) != now:
                raise SystemExit(
                    f"{pass_} variant {variant!r} was run with {key}={run.get(key)!r}, "
                    f"now {now!r}. Use a new --variant name, or --force to "
                    "discard the cached run.")
        return run
    return {"pass": pass_, "variant": variant, "created": now_iso(),
            "items": {}, **identity}


class FailureCounter:
    """The loud end-of-run line the handoff asks for, plus a warning when a
    streak of failures looks like the backend dying rather than bad photos."""

    def __init__(self, log=print):
        self.log = log
        self.failed = 0
        self.streak = 0
        self.warned = False

    def ok(self):
        self.streak = 0

    def fail(self, pid, err):
        self.failed += 1
        self.streak += 1
        self.log(f"  ! {pid}: {err}")
        if self.streak >= CONSECUTIVE_FAIL_WARN and not self.warned:
            self.warned = True
            self.log(f"  !!! {self.streak} failures in a row — is LM Studio still "
                     "up? The run keeps going; re-run afterwards to fill the gaps.")

    def summary(self):
        if self.failed:
            self.log(f"[run] {self.failed} photo(s) failed and were NOT cached — "
                     "re-run (without --force) to fill the gaps")


# --------------------------------------------------------------------------
# Model pinning
# --------------------------------------------------------------------------

def pin_role_model(role: str, model: Optional[str]) -> None:
    """Make an explicit --model the one that actually runs for `role`.

    On the LM Studio route the name passed to the call is IGNORED — the role
    env var picks the model, and the vision roles fall back to
    PHOTOSEARCH_LLM_VISUAL_MODEL when their own var is unset
    (`describe._resolve_openai_model`). So a bake-off passing `--model X` while
    the shell exports VISUAL=qwen would score qwen under X's name — which is
    exactly what may have happened to the 2026-07-09 aesthetics ρ 0.70.

    --model values are LM Studio ids, so pinning without the LM Studio route
    is refused rather than quietly handed to Ollama as a model name."""
    if not model:
        return
    if not os.environ.get("PHOTOSEARCH_TEXT_LLM_URL"):
        raise SystemExit("--model needs PHOTOSEARCH_TEXT_LLM_URL (the LM Studio "
                         "route); without it the id would go to Ollama.")
    os.environ[f"PHOTOSEARCH_LLM_{role.upper()}_MODEL"] = model


# --------------------------------------------------------------------------
# Originals cache
# --------------------------------------------------------------------------

def fetch_from_server(server: str, photo_id: int, kind: str = "full",
                      timeout: float = 120) -> bytes:
    """Bytes of one photo from a running photosearch server. `full` is what
    the worker downloads, so production code then does the same re-encode it
    does in the fleet."""
    import urllib.request
    url = f"{server.rstrip('/')}/api/photos/{int(photo_id)}/{kind}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def originals_dir() -> Path:
    return eval_dir() / "originals"


def cached_original(photo_id: int) -> Optional[Path]:
    p = originals_dir() / str(int(photo_id))
    return p if p.exists() else None


def original_path(photo_id: int, server: str = DEFAULT_SERVER,
                  fetch: Optional[Callable] = None) -> Path:
    """Local path of `photo_id`'s full-size bytes, fetched on first use.

    The replica pulls originals from the NAS; three models x several passes
    would otherwise pull the same ~70 photos from the N100 over and over (and
    502 whenever it restarts). A failed fetch raises and writes nothing."""
    hit = cached_original(photo_id)
    if hit is not None:
        return hit
    data = (fetch or fetch_from_server)(server, int(photo_id), "full")
    if not data:
        raise RuntimeError(f"empty response for photo {photo_id}")
    d = originals_dir()
    d.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=f".{int(photo_id)}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, d / str(int(photo_id)))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return d / str(int(photo_id))


def prefetch(photo_ids: Iterable[int], server: str = DEFAULT_SERVER,
             fetch: Optional[Callable] = None, log=print) -> tuple[int, int, int]:
    """Pull every id into the cache. Returns (already_cached, fetched, failed)."""
    have = got = bad = 0
    for pid in photo_ids:
        if cached_original(pid) is not None:
            have += 1
            continue
        try:
            original_path(pid, server, fetch)
            got += 1
        except Exception as e:
            bad += 1
            log(f"  ! {pid}: {e}")
    log(f"[originals] {have} cached, {got} fetched, {bad} failed")
    return have, got, bad


# --------------------------------------------------------------------------
# What LM Studio has loaded (for the solo/shared latency label)
# --------------------------------------------------------------------------

def lmstudio_loaded(base_url: Optional[str] = None, timeout: float = 5) -> Optional[list[str]]:
    """Ids LM Studio reports as loaded, or None if it can't say.

    Best effort: LM Studio's native REST API (`/api/v0/models`) lists every
    downloaded model with a `state`. The OpenAI-compatible `/v1/models` does
    not distinguish loaded from JIT-loadable, so it is not a fallback."""
    import urllib.request
    base = base_url or os.environ.get("PHOTOSEARCH_TEXT_LLM_URL")
    if not base:
        return None
    root = re.sub(r"/v1/?$", "", base.rstrip("/"))
    try:
        with urllib.request.urlopen(root + "/api/v0/models", timeout=timeout) as r:
            data = json.loads(r.read())
    except Exception:
        return None
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return None
    return sorted(str(m.get("id")) for m in rows
                  if isinstance(m, dict) and m.get("state") == "loaded")


def latency_label(effective: str, loaded_start, loaded_end, solo_flag: bool = False) -> str:
    """`solo` only when LM Studio said so at both ends of the run (or the
    operator asserted it with --solo). gemma-4-26b ran 7x slower with qwen also
    loaded — a latency number without this label compares memory pressure."""
    if loaded_start is None or loaded_end is None:
        return "solo (asserted)" if solo_flag else "unknown"
    others = sorted((set(loaded_start) | set(loaded_end)) - {effective})
    if not others:
        return "solo"
    return "shared(" + ",".join(others) + ")"


# --------------------------------------------------------------------------
# Recorder — tell a dead backend from a useless answer
# --------------------------------------------------------------------------

class TransportError(RuntimeError):
    """The backend could not be reached / refused the request. Says nothing
    about the photo, so it must not be cached as a result."""


class Recorder:
    """Wrap `describe._ollama_chat_with_retry` (the single chat entry point
    every pass funnels through, including verify/aesthetics which import it at
    call time) and the per-attempt hook, for one production call.

    Production code turns transport failures into answers: `describe_photo`
    returns None, `llm_verify_description` returns [] (== "ALL CORRECT"), the
    text extractors return None. `check(result)` raises TransportError when no
    call was answered, so none of those get cached.

    `calls` = [{"model", "role", "attempts": [...], "raw" | "error"}]."""

    def __init__(self):
        self.calls: list[dict] = []

    @contextmanager
    def active(self):
        from photosearch import describe
        real_chat = describe._ollama_chat_with_retry
        real_hook = describe._ATTEMPT_HOOK

        def recording(*a, **kw):
            model = kw.get("model", a[0] if a else None)
            entry = {"model": model, "role": kw.get("role"), "attempts": []}
            self.calls.append(entry)
            try:
                entry["raw"] = real_chat(*a, **kw)
            except Exception as e:
                entry["error"] = f"{e.__class__.__name__}: {e}"
                raise
            return entry["raw"]

        def hook(**kw):
            if self.calls:
                att = {k: kw.get(k) for k in ("attempt", "outcome", "completion_tokens",
                                              "finish_reason")}
                att["elapsed_s"] = round(float(kw.get("elapsed") or 0), 3)
                self.calls[-1]["attempts"].append(att)

        describe._ollama_chat_with_retry = recording
        describe._ATTEMPT_HOOK = hook
        try:
            yield self
        finally:
            describe._ollama_chat_with_retry = real_chat
            describe._ATTEMPT_HOOK = real_hook

    def answered(self) -> bool:
        return any("error" not in c for c in self.calls)

    def check(self, result_is_empty: bool, what: str = "the production call") -> None:
        """Raise TransportError if the result is empty-ish AND no call answered."""
        if not result_is_empty:
            return
        if not self.calls:
            raise TransportError(f"{what} made no model call "
                                 "(is the `ollama` package installed? unreadable file?)")
        if not self.answered():
            raise TransportError(self.calls[-1]["error"])

    # ---- per-attempt summaries -------------------------------------------

    def attempts(self) -> list[dict]:
        return [a for c in self.calls for a in c["attempts"]]

    def count(self, outcome: str) -> int:
        return sum(1 for a in self.attempts() if a.get("outcome") == outcome)

    def truncated(self, max_tokens: int = 768) -> bool:
        """The last answered attempt ran out of tokens (thinking models burn
        the whole budget; a long describe gets cut mid-sentence)."""
        ok = [a for a in self.attempts() if a.get("outcome") == "ok"]
        if not ok:
            return False
        last = ok[-1]
        return last.get("finish_reason") == "length" or \
            (last.get("completion_tokens") or 0) >= max_tokens


# --------------------------------------------------------------------------
# Report helpers
# --------------------------------------------------------------------------

def fmt_ratio(num: float, den: float, digits: int = 2) -> str:
    if den < MIN_N:
        return "n<3"
    return f"{num / den:.{digits}f}"


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def percentile(xs, p: float):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def latencies(items: dict, skip_first: bool = True) -> list[float]:
    """Per-item latency in insertion order; the first call pays the JIT load,
    so it is dropped by default."""
    xs = [it.get("latency_s") for it in items.values() if it.get("latency_s") is not None]
    return xs[1:] if skip_first and len(xs) > 1 else xs
