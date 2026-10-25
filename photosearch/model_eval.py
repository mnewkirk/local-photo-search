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

import io
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


_THREAD_LOCKS: dict = {}
_THREAD_LOCKS_GUARD = __import__("threading").Lock()


@contextmanager
def _file_lock(path: Path):
    """Exclusive lock for one JSON store: a thread lock (the web server runs
    sync handlers in a pool) plus flock on a sidecar file (a CLI run at the same
    time). Held around read-modify-write so parallel saves take turns."""
    import threading
    key = str(Path(path).resolve())
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.setdefault(key, threading.Lock())
    with lock:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            import fcntl
        except ImportError:          # not on the platforms this runs on
            yield
            return
        with open(f"{path}.lock", "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


def update_pass_file(pass_: str, name: str, default, fn):
    """Locked read-modify-write of one store: `fn(data)` mutates `data` and
    returns the caller's result. Every label save goes through here — the
    labelling page saves a photo's labels as PARALLEL requests, and an unlocked
    read-modify-write let them overwrite each other (2026-09-28: 126 answers
    submitted, 38 survived)."""
    path = pass_dir(pass_) / name
    with _file_lock(path):
        data = read_json(path, default)
        result = fn(data)
        write_json_atomic(path, data)
        return result


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
# Samples + text identity
# --------------------------------------------------------------------------

def load_sample(pass_: str) -> dict:
    return read_pass_file(pass_, "sample.json", {"created": None, "photos": []})


def save_sample(pass_: str, photos: list[dict], **meta) -> dict:
    """`photos` is [{"photo_id": int, "stratum": str}, ...]."""
    data = {"created": now_iso(), **meta,
            "photos": [{"photo_id": int(p["photo_id"]), "stratum": str(p["stratum"])}
                       for p in photos]}
    write_pass_file(pass_, "sample.json", data)
    return data


def sample_ids(pass_: str) -> list[int]:
    return [p["photo_id"] for p in load_sample(pass_)["photos"]]


def text_sha(text: Optional[str]) -> Optional[str]:
    """Identity of a generated text, whitespace-normalised. Labels on
    generated text are keyed by this, not by variant: the labelling page never
    needs a model name (blind by construction), two models that wrote the same
    words share one label, and a re-run that changes the words orphans the old
    label instead of being scored with it."""
    if text is None:
        return None
    import hashlib
    norm = " ".join(text.split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def open_db_readonly(path: Optional[str]):
    """sqlite3 connection with mode=ro. A missing file is an error, not a new
    stub DB — and never PhotoDB, which migrates on open. sqlite-vec is loaded
    when available so `clip_embeddings` can be read."""
    import sqlite3
    from urllib.parse import quote
    if not path:
        raise SystemExit("No DB given — pass --db (or set PHOTOSEARCH_DB).")
    uri = "file:" + quote(os.path.abspath(path)) + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.execute("SELECT 1 FROM photos LIMIT 1")
    except sqlite3.OperationalError as e:
        raise SystemExit(f"Cannot open {path} read-only: {e}")
    conn.row_factory = sqlite3.Row
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception:
        pass
    return conn


def clip_embedding(conn, photo_id: int) -> Optional[list[float]]:
    import struct
    try:
        row = conn.execute("SELECT embedding FROM clip_embeddings WHERE photo_id = ?",
                           (int(photo_id),)).fetchone()
    except Exception:
        return None
    if not row or row[0] is None:
        return None
    blob = row[0]
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


# --------------------------------------------------------------------------
# Model pinning
# --------------------------------------------------------------------------

def pin_role_model(role: str, model: Optional[str]) -> None:
    """Make an explicit --model the one that actually runs for `role`.

    On the LM Studio route the name passed to the call is IGNORED — the role
    env var picks the model, and the vision roles fall back to
    PHOTOSEARCH_LLM_VISUAL_MODEL when their own var is unset
    (`describe._resolve_openai_model`). So a bake-off passing `--model X` while
    the shell exports VISUAL=qwen would score qwen under X's name. (The
    aesthetics fallback arrived in 4cfc202, 2026-07-10 — after the 07-09
    bake-off, whose qwen ρ 0.70 is therefore genuinely qwen's.)

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
            except describe.UnusableAnswer as e:
                # The model ANSWERED (cut off at its token limit) — a real
                # outcome to score and cache, not a dead backend.
                entry["raw"] = getattr(e, "text", None)
                entry["unusable"] = str(e)
                raise
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
        """Raise TransportError if the result is empty-ish AND the LAST call
        errored (or none was made). An earlier answered-but-unparseable call
        followed by a dead retry is still unknown, not a parse failure."""
        if not result_is_empty:
            return
        if not self.calls:
            raise TransportError(f"{what} made no model call "
                                 "(is the `ollama` package installed? unreadable file?)")
        if "error" in self.calls[-1]:
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


# --------------------------------------------------------------------------
# Describe labels: wrong claims, pairwise preference, visible text
# --------------------------------------------------------------------------
#
#   describe/claims.json  {"labels": {<text_sha>: {"photo_id", "n_segments",
#                          "wrong": [segment idx], "other_wrong": bool,
#                          "done": bool, "updated_at"}}}
#   describe/pairs.json   {"created", "seed", "baseline",
#                          "pairs": [{"key", "photo_id", "a_sha", "b_sha",
#                                     "a_variant", "b_variant"}]}
#   describe/prefs.json   {"prefs": {<key>: {"winner_sha": sha | None (tie),
#                          "updated_at"}}}
#   describe/text_truth.json {<photo_id>: {"text", "done", "updated_at"}}
#
# Claims and prefs are keyed by TEXT, never by variant — see text_sha().

_SENTENCE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”)]))\s+(?=[A-Z\"“(])")
# Clause joins that almost always start a new factual claim. Deliberately
# narrow: splitting on a bare "and" would cut "salt and pepper" in half.
_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St|Mt|Jr|Sr|vs|etc|e\.g|i\.e)\.\s*$")
_CLAUSE = re.compile(r"(?:;\s+|,\s+(?=(?:and|while|with|where|which|but)\s))")


def segment_claims(text: Optional[str]) -> list[str]:
    """Split a description into clickable claim chunks. Deterministic, and
    lossless: "".join(segment_claims(t)) == t, so the page can render the
    description exactly as written with each chunk clickable."""
    if not text:
        return []
    sents: list[str] = []
    for sent in _split_keep(text, _SENTENCE):
        if sents and _ABBREV.search(sents[-1]):
            sents[-1] += sent          # "Mr. " is not the end of a sentence
        else:
            sents.append(sent)
    out: list[str] = []
    for sent in sents:
        out.extend(_split_keep(sent, _CLAUSE))
    return [s for s in out if s]


def _split_keep(text: str, pattern) -> list[str]:
    """Split at `pattern`, keeping each separator on the chunk before it."""
    parts, last = [], 0
    for m in pattern.finditer(text):
        parts.append(text[last:m.end()])
        last = m.end()
    parts.append(text[last:])
    return parts


def load_claims() -> dict:
    return read_pass_file("describe", "claims.json", {"labels": {}})["labels"]


def save_claim(sha: str, photo_id: int, n_segments: int, wrong: Iterable[int],
               other_wrong: bool = False, done: bool = True) -> dict:
    wrong = sorted(set(int(i) for i in wrong))
    if any(i < 0 or i >= n_segments for i in wrong):
        raise ValueError(f"segment index out of range 0..{n_segments - 1}")
    label = {"photo_id": int(photo_id), "n_segments": int(n_segments), "wrong": wrong,
             "other_wrong": bool(other_wrong), "done": bool(done), "updated_at": now_iso()}

    def put(data):
        data["labels"][sha] = label
        return label
    return update_pass_file("describe", "claims.json", {"labels": {}}, put)


def claim_errors(label: Optional[dict]) -> Optional[int]:
    """Wrong claims in one labelled description, or None if not done."""
    if not label or not label.get("done"):
        return None
    return len(label.get("wrong") or []) + (1 if label.get("other_wrong") else 0)


def pair_key(a_sha: str, b_sha: str) -> str:
    return "~".join(sorted((a_sha, b_sha)))


def load_pairs() -> dict:
    return read_pass_file("describe", "pairs.json", {"pairs": []})


def load_prefs() -> dict:
    return read_pass_file("describe", "prefs.json", {"prefs": {}})["prefs"]


def save_pref(key: str, winner_sha: Optional[str]) -> dict:
    def put(data):
        data["prefs"][key] = {"winner_sha": winner_sha, "updated_at": now_iso()}
        return data["prefs"][key]
    return update_pass_file("describe", "prefs.json", {"prefs": {}}, put)


def load_text_truth() -> dict:
    return read_pass_file("describe", "text_truth.json", {})


def save_text_truth(photo_id: int, text: str, done: bool = True) -> dict:
    def put(data):
        data[str(int(photo_id))] = {"text": text, "done": bool(done), "updated_at": now_iso()}
        return data[str(int(photo_id))]
    return update_pass_file("describe", "text_truth.json", {}, put)


# --------------------------------------------------------------------------
# Text-pass inputs + labels (category-content, keywords)
# --------------------------------------------------------------------------
#
#   text/inputs-<name>.json   {"name", "source", "source_effective_model",
#                              "created", "items": {<pid>: {"text", "text_sha"}}}
#   text/category_labels.json {<pid>:<text_sha>: {"yes": [term], "done", "updated_at"}}
#   text/keyword_labels.json  {<pid>:<text_sha>: {"wrong": [kw], "judged": [kw],
#                              "done", "updated_at"}}
#
# Labels are keyed to the frozen TEXT, so refreezing orphans them rather than
# scoring a new description against a judgement of the old one.

TEXT_PASSES = ("category-content", "keywords")


def load_inputs(name: str = "main") -> Optional[dict]:
    return read_pass_file("text", f"inputs-{name}.json", None)


def save_inputs(name: str, source: str, source_effective_model: Optional[str],
                items: dict, force: bool = False) -> dict:
    if not VARIANT_RE.match(name):
        raise ValueError(f"bad inputs name {name!r}")
    if load_inputs(name) is not None and not force:
        raise SystemExit(f"inputs {name!r} already frozen; labels are keyed to it. "
                         "--force to refreeze (orphans those labels).")
    data = {"name": name, "source": source, "source_effective_model": source_effective_model,
            "created": now_iso(),
            "items": {str(int(k)): {"text": v, "text_sha": text_sha(v)}
                      for k, v in items.items() if v}}
    write_pass_file("text", f"inputs-{name}.json", data)
    return data


def inputs_identity(inputs: dict) -> str:
    """What a text run is keyed to: the input set's name + a hash of its texts."""
    import hashlib
    h = hashlib.sha256()
    for pid in sorted(inputs["items"], key=int):
        h.update(f"{pid}:{inputs['items'][pid]['text_sha']};".encode())
    return f"{inputs['name']}@{h.hexdigest()[:12]}"


def label_key(photo_id, sha: str) -> str:
    return f"{int(photo_id)}:{sha}"


def load_category_labels() -> dict:
    return read_pass_file("text", "category_labels.json", {})


def save_category_label(key: str, yes: Iterable[str], done: bool = True,
                        missing: Iterable[str] = ()) -> dict:
    """`yes` = right categories from the vocabulary (scored). `missing` = right
    categories the owner added that the vocabulary LACKS: no model may emit
    them, so scoring them would charge every model the same miss. They are
    kept as vocabulary-gap evidence instead (text_passes_eval report)."""
    from .vocab_content import CONTENT_VOCABULARY
    vocab = set(CONTENT_VOCABULARY)
    yes = sorted(set(yes))
    bad = [t for t in yes if t not in vocab]
    if bad:
        raise ValueError(f"not in the content vocabulary: {bad}")
    missing = sorted({" ".join(str(t).lower().split()) for t in missing} - vocab - {""})
    def put(data):
        data[key] = {"yes": yes, "missing": missing, "done": bool(done),
                     "updated_at": now_iso()}
        return data[key]
    return update_pass_file("text", "category_labels.json", {}, put)


def load_keyword_labels() -> dict:
    return read_pass_file("text", "keyword_labels.json", {})


def save_keyword_label(key: str, judged: Iterable[str], wrong: Iterable[str],
                       done: bool = True) -> dict:
    """`judged` is the pool the owner saw; a keyword in it and not in `wrong`
    is judged right. A keyword a later run adds is simply unjudged."""
    judged = sorted(set(judged))
    wrong = sorted(set(wrong))
    if not set(wrong) <= set(judged):
        raise ValueError("wrong keywords must come from the judged pool")
    def put(data):
        data[key] = {"judged": judged, "wrong": wrong, "done": bool(done),
                     "updated_at": now_iso()}
        return data[key]
    return update_pass_file("text", "keyword_labels.json", {}, put)


# --------------------------------------------------------------------------
# Verify sets
# --------------------------------------------------------------------------
#
#   verify/sets.json  {"source_variant", "source_effective_model", "created",
#                      "items": [{"id", "photo_id", "kind": clean|planted|real,
#                                 "type": object|colour|count|None, "text",
#                                 "spans": [str], "confirmed": bool|None}]}
#
# A planted error counts only once the owner has confirmed it really is false
# for the photo (`confirmed` true) — a templated plant can be accidentally true.

def load_verify_sets() -> Optional[dict]:
    return read_pass_file("verify", "sets.json", None)


def save_verify_sets(data: dict) -> None:
    write_pass_file("verify", "sets.json", data)


def confirm_planted(item_id: str, confirmed: bool) -> dict:
    def put(data):
        if data is None:
            raise KeyError("no verify sets")
        for it in data["items"]:
            if it["id"] == item_id and it["kind"] == "planted":
                it["confirmed"] = bool(confirmed)
                it["confirmed_at"] = now_iso()
                return it
        raise KeyError(item_id)
    return update_pass_file("verify", "sets.json", None, put)


# --------------------------------------------------------------------------
# Describe disputes: only the facts the descriptions DISAGREE on
# --------------------------------------------------------------------------
#
# Labelling every description is wasted effort when most of them agree. A text
# model reads one photo's distinct descriptions (blind, as D1..Dn) and lists
# the checkable facts they disagree on, with each description's answer. The
# owner picks the right answer(s) looking at the photo; each description is
# then right, wrong or silent on each point.
#
#   describe/disputes.json        {<pid>: {"shas": [sha...], "model", "created",
#                                  "questions": [{"key", "question",
#                                                 "answers": {<sha>: str|None}}]}}
#   describe/dispute_labels.json  {<key>: {"correct": [answer...], "cant_tell",
#                                  "updated_at"}}
#
# A question key hashes the photo, the question text and the texts it was asked
# of, so regenerating disputes for changed descriptions orphans old labels.

NOT_MENTIONED = None


def norm_answer(a) -> Optional[str]:
    if a is None:
        return None
    s = " ".join(str(a).split()).strip().rstrip(".").lower()
    if not s or s in ("not mentioned", "n/a", "none mentioned", "unspecified",
                      "not stated", "does not say", "not specified"):
        return None
    return s


def dispute_key(photo_id, question: str, shas) -> str:
    import hashlib
    h = hashlib.sha256(f"{int(photo_id)}|{question}|{','.join(sorted(shas))}".encode())
    return f"{int(photo_id)}-{h.hexdigest()[:12]}"


def load_disputes(pass_: str = "describe") -> dict:
    """`pass_` is `describe` (photos) or `video-describe` (step 4 of
    docs/plans/video-support.md): same format, separate files."""
    return read_pass_file(pass_, "disputes.json", {})


def save_disputes(data: dict, pass_: str = "describe") -> None:
    write_pass_file(pass_, "disputes.json", data)


def load_dispute_labels(pass_: str = "describe") -> dict:
    return read_pass_file(pass_, "dispute_labels.json", {})


def dispute_options(q: dict) -> list[str]:
    """Distinct answers to one disputed point; silence is not an option."""
    return sorted({a for a in q["answers"].values() if a is not None})


def save_dispute_label(key: str, correct: Iterable[str], cant_tell: bool = False,
                       options_seen: Optional[Iterable[str]] = None,
                       pass_: str = "describe") -> dict:
    """`options_seen` = the answers the owner was shown. A description added
    later whose answer is not among them is UNJUDGED, never wrong by default —
    "None right" said nothing about an answer that was not on screen."""
    def put(data):
        data[key] = {"correct": sorted(set(correct)), "cant_tell": bool(cant_tell),
                     "updated_at": now_iso()}
        if options_seen is not None:
            data[key]["options_seen"] = sorted(set(options_seen))
        return data[key]
    return update_pass_file(pass_, "dispute_labels.json", {}, put)


def dispute_verdict(answer: Optional[str], label: Optional[dict]) -> Optional[str]:
    """'right' | 'wrong' | 'silent' for one description's answer, or None when
    the question is unlabelled or the owner could not tell."""
    if not label or label.get("cant_tell"):
        return None
    a = norm_answer(answer)
    if a is None:
        return "silent"
    if a in set(label.get("correct") or []):
        return "right"
    seen = label.get("options_seen")
    if seen is not None and a not in set(seen):
        return None                     # an answer the owner never saw
    return "wrong"


def dispute_label_current(q: dict, label: Optional[dict]) -> bool:
    """Is this point fully judged — labelled, and no answer added since?"""
    if not label:
        return False
    if label.get("cant_tell"):
        return True
    seen = label.get("options_seen")
    return seen is None or set(dispute_options(q)) <= set(seen)


# --------------------------------------------------------------------------
# Video eval (docs/plans/video-support.md, step 4)
# --------------------------------------------------------------------------
#
# Everything lives under eval_dir()/video (git-ignored), never in the DB:
#
#   video/sample.json             {"created", "seed", "photos": [{"photo_id",
#                                  "stratum", "split": dev|holdout|check,
#                                  "filename", "duration_s", "width", "height",
#                                  "live_still_id"}], "probe_check": [{...}],
#                                  "recheck": [pid], "shortfall": {stratum: n}}
#   video/subject_labels[-recheck].json
#                                 {"who": {<pid>: {"people": [person_id],
#                                          "someone_else", "live_pair",
#                                          "done", "updated_at"}},
#                                  "segments": {"<pid>:<seg_version>:<ix>":
#                                          {"kind", "person_id", "label",
#                                           "free_text"}}}
#   video/quality_labels[-recheck].json  {<pid>: {"in_focus", "shaky", "noisy",
#                                          "keeper", "done", "updated_at"}}
#   video/temporal_labels.json    {<text_sha>: {"photo_id", "answer", "updated_at"}}
#   video/detections/<tag>/<pid>.npz + .json   the face-detection cache
#   video/frames/<pid>/<ms>-<px>.jpg           grabbed keyframes
#   video-grounding/runs/<v>.json, video-subject/runs/<v>.json,
#   video-describe/runs/<v>.json, video-quality/runs/<v>.json  (run caches)
#
# Segment labels are keyed by (pid, seg_version, seg_ix): a change to the
# segment schedule or its 12-way coarsening changes seg_version and orphans
# the old labels instead of scoring them against different time ranges.

VIDEO = "video"
VIDEO_LABEL_MAX_SEGMENTS = 12
VIDEO_HOLDOUT_FRAC = 1 / 3
VIDEO_LABEL_SETS = ("main", "recheck")
SUBJECT_KINDS = ("person", "someone_else", "subject", "none", "cant_tell")
THREE_STATE = ("yes", "debatable", "no")
QUALITY_FLAGS = ("in_focus", "shaky", "noisy")
KEEPER = ("keep", "meh", "delete")
TEMPORAL = ("yes", "partly", "no")


def video_seg_version() -> str:
    """Identity of the labelling segmentation. Derived from the production
    schedule's constants, so changing them orphans labels automatically."""
    from .video_analysis import SEGMENT_CAP, SEGMENT_S
    return f"s{SEGMENT_S:g}c{SEGMENT_CAP}m{VIDEO_LABEL_MAX_SEGMENTS}"


def video_production_segments(duration: float) -> list:
    from .video_analysis import segment_schedule
    return [tuple(s) for s in segment_schedule(float(duration or 0.0))]


def video_label_segments(duration: float) -> list:
    """The <= 12 segments the owner labels: the production 3 s / cap-40
    schedule, coarsened on production boundaries."""
    from .video_analysis import coarsen_segments
    return [tuple(s) for s in coarsen_segments(video_production_segments(duration),
                                               VIDEO_LABEL_MAX_SEGMENTS)]


def coarsen_keys(prod_segments, keys, label_segments) -> list:
    """Map one key per production segment onto the label segments: each label
    segment takes the key holding the most time inside it (None = no subject
    counts as a key; ties go to the earliest)."""
    out = []
    for a, b in label_segments:
        acc: dict = {}
        order: list = []
        for (pa, pb), k in zip(prod_segments, keys):
            ov = min(b, pb) - max(a, pa)
            if ov <= 1e-9:
                continue
            kk = json.dumps(k, sort_keys=True)
            if kk not in acc:
                order.append(kk)
            acc[kk] = acc.get(kk, 0.0) + ov
        if not acc:
            out.append(None)
            continue
        best = max(order, key=lambda kk: (acc[kk], -order.index(kk)))
        out.append(json.loads(best))
    return out


def holdout_split(entries: Iterable[dict], seed: int = 1,
                  frac: float = VIDEO_HOLDOUT_FRAC) -> dict:
    """{photo_id: "dev"|"holdout"} — stratified and deterministic: within each
    stratum the ids are sorted, shuffled with a seed derived from (seed,
    stratum), and the first round(n * frac) are holdout. Input order and the
    other strata never change a video's split."""
    import random
    by: dict = {}
    for e in entries:
        by.setdefault(str(e["stratum"]), []).append(int(e["photo_id"]))
    out = {}
    for stratum, ids in by.items():
        ids = sorted(set(ids))
        random.Random(f"video-split-{seed}-{stratum}").shuffle(ids)
        k = int(round(len(ids) * frac))
        for i, pid in enumerate(ids):
            out[pid] = "holdout" if i < k else "dev"
    return out


def load_video_sample() -> dict:
    return read_pass_file(VIDEO, "sample.json", {"created": None, "photos": [],
                                                  "probe_check": [], "recheck": []})


def save_video_sample(data: dict) -> dict:
    data = dict(data)
    data.setdefault("created", now_iso())
    write_pass_file(VIDEO, "sample.json", data)
    return data


def video_sample_entry(photo_id: int) -> Optional[dict]:
    s = load_video_sample()
    for p in list(s["photos"]) + list(s.get("probe_check") or []):
        if int(p["photo_id"]) == int(photo_id):
            return p
    return None


def is_labelled_clip(entry: dict) -> bool:
    """Live clips are paired to their still and only get a pairing question."""
    return not str(entry.get("stratum", "")).startswith("live")


def _label_file(base: str, label_set: str) -> str:
    if label_set not in VIDEO_LABEL_SETS:
        raise ValueError(f"unknown label set {label_set!r}")
    return f"{base}.json" if label_set == "main" else f"{base}-{label_set}.json"


def seg_label_key(photo_id, seg_version: str, seg_ix: int) -> str:
    return f"{int(photo_id)}:{seg_version}:{int(seg_ix)}"


def load_video_subject_labels(label_set: str = "main") -> dict:
    return read_pass_file(VIDEO, _label_file("subject_labels", label_set),
                          {"who": {}, "segments": {}})


def _clean_seg_label(s: dict, people: set) -> dict:
    kind = s.get("kind")
    if kind not in SUBJECT_KINDS:
        raise ValueError(f"segment kind must be one of {SUBJECT_KINDS}")
    out = {"kind": kind}
    if kind == "person":
        pid = s.get("person_id")
        if pid is None or int(pid) not in people:
            raise ValueError("a person segment must name someone from 'who is in this video'")
        out["person_id"] = int(pid)
    elif kind == "subject":
        lab = " ".join(str(s.get("label") or "").split()).strip().lower()
        if not lab:
            raise ValueError("a subject segment needs a label")
        out["label"] = lab
        out["free_text"] = bool(s.get("free_text"))
    return out


def save_video_subject_label(photo_id: int, *, seg_version: str, n_segments: int,
                             people: Iterable[int] = (), someone_else: bool = False,
                             segments: Optional[dict] = None, live_pair: Optional[str] = None,
                             done: bool = True, label_set: str = "main") -> dict:
    """Save one video's answers. `segments` = {seg_ix: {"kind", ...}}; every
    key is written as (pid, seg_version, seg_ix). Refuses a seg_version that
    is not the current one — a page built on an old schedule must not label
    the new one's time ranges."""
    if seg_version != video_seg_version():
        raise ValueError(f"stale segmentation {seg_version!r}; reload the page")
    if live_pair is not None and live_pair not in ("yes", "no", "cant_tell"):
        raise ValueError("live_pair must be yes, no or cant_tell")
    people = sorted({int(p) for p in people})
    cleaned = {}
    for ix, s in (segments or {}).items():
        ix = int(ix)
        if ix < 0 or ix >= n_segments:
            raise ValueError(f"segment index out of range 0..{n_segments - 1}")
        cleaned[ix] = _clean_seg_label(s, set(people))
    who = {"people": people, "someone_else": bool(someone_else),
           "seg_version": seg_version, "n_segments": int(n_segments),
           "done": bool(done), "updated_at": now_iso()}
    if live_pair is not None:
        who["live_pair"] = live_pair

    def put(data):
        data.setdefault("who", {})[str(int(photo_id))] = who
        segs = data.setdefault("segments", {})
        prefix = f"{int(photo_id)}:{seg_version}:"
        for k in [k for k in segs if k.startswith(prefix)]:
            del segs[k]
        for ix, lab in cleaned.items():
            segs[seg_label_key(photo_id, seg_version, ix)] = lab
        return {"who": who, "segments": {str(i): v for i, v in cleaned.items()}}
    return update_pass_file(VIDEO, _label_file("subject_labels", label_set),
                            {"who": {}, "segments": {}}, put)


def video_segment_labels(labels: dict, photo_id: int, seg_version: str,
                         n_segments: int) -> list:
    """One label (or None) per segment, for the CURRENT seg_version only."""
    segs = labels.get("segments") or {}
    return [segs.get(seg_label_key(photo_id, seg_version, i)) for i in range(n_segments)]


def load_video_quality_labels(label_set: str = "main") -> dict:
    return read_pass_file(VIDEO, _label_file("quality_labels", label_set), {})


def save_video_quality_label(photo_id: int, answers: dict, done: bool = True,
                             label_set: str = "main") -> dict:
    lab = {}
    for f in QUALITY_FLAGS:
        v = answers.get(f)
        if v is not None and v not in THREE_STATE:
            raise ValueError(f"{f} must be one of {THREE_STATE}")
        lab[f] = v
    k = answers.get("keeper")
    if k is not None and k not in KEEPER:
        raise ValueError(f"keeper must be one of {KEEPER}")
    lab["keeper"] = k
    if done and any(lab[f] is None for f in QUALITY_FLAGS + ("keeper",)):
        raise ValueError("answer all four questions before marking done")
    lab.update(done=bool(done), updated_at=now_iso())

    def put(data):
        data[str(int(photo_id))] = lab
        return lab
    return update_pass_file(VIDEO, _label_file("quality_labels", label_set), {}, put)


def load_temporal_labels() -> dict:
    return read_pass_file(VIDEO, "temporal_labels.json", {})


def save_temporal_label(sha: str, photo_id: int, answer: str) -> dict:
    """"Does it say what happens over the clip, not just one moment?" — keyed
    by the description's text_sha, like every label on generated text."""
    if answer not in TEMPORAL:
        raise ValueError(f"answer must be one of {TEMPORAL}")
    lab = {"photo_id": int(photo_id), "answer": answer, "updated_at": now_iso()}

    def put(data):
        data[sha] = lab
        return lab
    return update_pass_file(VIDEO, "temporal_labels.json", {}, put)


def video_recheck_ids(n: int = 15, seed: int = 7) -> list[int]:
    """The seeded recheck subset: clips only, stable for a given sample."""
    import random
    ids = sorted(int(p["photo_id"]) for p in load_video_sample()["photos"]
                 if is_labelled_clip(p))
    random.Random(f"video-recheck-{seed}").shuffle(ids)
    return sorted(ids[:n])


# ---- failure classification ---------------------------------------------

_CONTEXT_RE = re.compile(r"context (size|length|window)|exceed(s|ed)? (the )?context|"
                         r"too many tokens|maximum context|n_ctx", re.I)


def is_context_overflow(err) -> bool:
    """A 400 "context exceeded" from LM Studio (its slot split): a CONFIG
    failure that screens the variant out, cached as `rejected`, never retried
    as if the backend were down."""
    texts = [str(err)]
    body = getattr(err, "body_text", None)
    if body:
        texts.append(str(body))
    return any(_CONTEXT_RE.search(t) for t in texts)


def http_error_body(exc) -> Optional[str]:
    """Best-effort body of a urllib HTTPError (the message alone is
    'HTTP Error 400: Bad Request', which does not say why)."""
    try:
        read = getattr(exc, "read", None)
        if read is None:
            return None
        data = read()
        return data.decode("utf-8", "replace") if isinstance(data, bytes) else str(data)
    except Exception:
        return None


def classify_failure(exc) -> str:
    """'rejected' (context overflow — cache it) or 'transport' (never cache)."""
    if getattr(exc, "body_text", None) is None:
        body = http_error_body(exc)
        if body:
            try:
                exc.body_text = body
            except Exception:
                pass
    return "rejected" if is_context_overflow(exc) else "transport"


# ---- detection cache ------------------------------------------------------

def _det_dir(tag: str) -> Path:
    if not VARIANT_RE.match(tag or ""):
        raise ValueError(f"bad detection tag {tag!r}")
    return pass_dir(VIDEO) / "detections" / tag


def save_detections(photo_id: int, tag: str, frames: list, meta: dict) -> Path:
    """Cache one video's per-frame face detections (video_analysis input
    format) as npz + a JSON sidecar, so rule variants are rescored without
    re-decoding or re-detecting. Atomic: npz first, then the sidecar that
    marks the entry complete."""
    import numpy as np
    d = _det_dir(tag)
    d.mkdir(parents=True, exist_ok=True)
    t, fi, bbox, score, emb, has_emb, sharp = [], [], [], [], [], [], []
    for i, fr in enumerate(frames):
        t.append(float(fr["t"]))
        for det in fr.get("faces") or []:
            fi.append(i)
            bbox.append([float(x) for x in det["bbox"]])
            score.append(float(det.get("det_score") or 0.0))
            e = det.get("embedding")
            has_emb.append(e is not None)
            emb.append(np.zeros(512, np.float32) if e is None else np.asarray(e, np.float32))
            s = det.get("sharpness")
            sharp.append(np.nan if s is None else float(s))
    arrays = dict(t=np.asarray(t, np.float64), frame_ix=np.asarray(fi, np.int32),
                  bbox=np.asarray(bbox, np.float32).reshape(-1, 4),
                  det_score=np.asarray(score, np.float32),
                  embedding=np.asarray(emb, np.float32).reshape(-1, 512),
                  has_embedding=np.asarray(has_emb, bool),
                  sharpness=np.asarray(sharp, np.float64))
    npz = d / f"{int(photo_id)}.npz"
    fd, tmp = tempfile.mkstemp(dir=d, prefix=f".{int(photo_id)}.", suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, npz)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    write_json_atomic(d / f"{int(photo_id)}.json", {**meta, "n_frames": len(t),
                                                    "saved": now_iso()})
    return npz


def load_detections(photo_id: int, tag: str) -> Optional[tuple]:
    """(frames, meta) or None when not cached (or only half-written)."""
    import numpy as np
    d = _det_dir(tag)
    meta = read_json(d / f"{int(photo_id)}.json", None)
    npz = d / f"{int(photo_id)}.npz"
    if meta is None or not npz.exists():
        return None
    z = np.load(npz)
    frames = [{"t": float(x), "faces": []} for x in z["t"]]
    for k in range(len(z["frame_ix"])):
        s = float(z["sharpness"][k])
        frames[int(z["frame_ix"][k])]["faces"].append({
            "bbox": [float(x) for x in z["bbox"][k]],
            "det_score": float(z["det_score"][k]),
            "embedding": z["embedding"][k].astype(float).tolist() if z["has_embedding"][k] else None,
            "sharpness": None if s != s else s})
    return frames, meta


# ---- originals + frames ----------------------------------------------------

def fetch_video_original(photo_id: int, server: str = DEFAULT_SERVER, *,
                         intent: str = "Video eval: cache an original for labelling",
                         timeout: float = 600, chunk: int = 1 << 20,
                         retries: int = 5) -> Path:
    """Stream one video into the originals cache via
    /api/photos/{id}/full?purpose=worker (the fleet's rate-capped read), with
    the request-log source/intent headers. Size-checked against
    Content-Length; a short or failed read writes nothing. A 503 (worker
    stream cap) is waited out per Retry-After."""
    import time as _time
    import urllib.error
    import urllib.request
    hit = cached_original(photo_id)
    if hit is not None:
        return hit
    url = f"{server.rstrip('/')}/api/photos/{int(photo_id)}/full?purpose=worker"
    req = urllib.request.Request(url, headers={
        "X-Photosearch-Source": "claude",
        "X-Photosearch-Intent": f"{intent} (video {int(photo_id)})"})
    d = originals_dir()
    d.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code == 503 and attempt < retries - 1:
                _time.sleep(float(e.headers.get("Retry-After") or 10))
                continue
            raise
        break
    want = resp.headers.get("Content-Length")
    fd, tmp = tempfile.mkstemp(dir=d, prefix=f".{int(photo_id)}.", suffix=".tmp")
    try:
        got = 0
        with os.fdopen(fd, "wb") as f, resp:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                got += len(buf)
            f.flush()
            os.fsync(f.fileno())
        if want is not None and got != int(want):
            raise RuntimeError(f"short read for video {photo_id}: {got} of {want} bytes")
        if got == 0:
            raise RuntimeError(f"empty response for video {photo_id}")
        os.replace(tmp, d / str(int(photo_id)))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return d / str(int(photo_id))


def copy_video_original(photo_id: int, src: str) -> Path:
    """Cache an original from a local path (the NAS share mounted, or the
    photo root on the machine that holds the files). Hardlink when possible."""
    import shutil
    hit = cached_original(photo_id)
    if hit is not None:
        return hit
    d = originals_dir()
    d.mkdir(parents=True, exist_ok=True)
    dest = d / str(int(photo_id))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=f".{int(photo_id)}.", suffix=".tmp")
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        if os.path.getsize(tmp) != os.path.getsize(src):
            raise RuntimeError(f"short copy of {src}")
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return dest


BLACK_FRAME_MEAN = 3.0


def ffmpeg_frame(path, t: float):
    """One frame via the ffmpeg CLI (PIL RGB, auto-rotated), or None. The
    fallback for what cv2 seeks badly: on an AVCHD .m2ts it returned BLACK
    frames at every timestamp (2026-10-25, video 254431)."""
    import subprocess
    from PIL import Image
    from .video_proxy import find_ffmpeg
    exe = find_ffmpeg()
    if not exe:
        return None
    try:
        proc = subprocess.run([exe, "-v", "error", "-ss", f"{max(0.0, float(t)):.3f}",
                               "-i", str(path), "-frames:v", "1", "-f", "image2pipe",
                               "-vcodec", "png", "-"], capture_output=True, timeout=180)
    except Exception:
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        return Image.open(io.BytesIO(proc.stdout)).convert("RGB")
    except Exception:
        return None


def grab_frame(path, t: float, max_px: int = 0):
    """One frame at `t` seconds as a PIL RGB image (cv2, imported lazily —
    never on the NAS). max_px 0 = native resolution (for the loupe). A
    missing or all-black cv2 frame is re-read with ffmpeg (`ffmpeg_frame`)."""
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(path))
    im = None
    try:
        if not cap.isOpened():
            raise RuntimeError("cv2 cannot open this video")
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t)) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None or float(frame.mean()) < BLACK_FRAME_MEAN:
            im = ffmpeg_frame(path, t)
        if im is None and (not ok or frame is None):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if im is None and (not ok or frame is None):
            raise RuntimeError(f"no frame at {t:.2f}s")
    finally:
        cap.release()
    if im is None:
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if max_px and max(im.size) > max_px:
        im.thumbnail((max_px, max_px), Image.LANCZOS)
    return im


def cached_frame_jpeg(photo_id: int, t: float, max_px: int = 768,
                      grab: Optional[Callable] = None) -> Path:
    """Path of a keyframe JPEG under video/frames, grabbed on first use."""
    d = pass_dir(VIDEO) / "frames" / str(int(photo_id))
    p = d / f"{int(round(float(t) * 1000))}-{int(max_px)}.jpg"
    if p.exists():
        return p
    src = cached_original(photo_id)
    if src is None:
        raise FileNotFoundError(f"video {photo_id} is not in the originals cache")
    im = (grab or grab_frame)(src, t, max_px)
    d.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".f", suffix=".jpg")
    os.close(fd)
    try:
        im.save(tmp, "JPEG", quality=85)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return p
