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
    data = read_pass_file("describe", "claims.json", {"labels": {}})
    label = {"photo_id": int(photo_id), "n_segments": int(n_segments), "wrong": wrong,
             "other_wrong": bool(other_wrong), "done": bool(done), "updated_at": now_iso()}
    data["labels"][sha] = label
    write_pass_file("describe", "claims.json", data)
    return label


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
    data = read_pass_file("describe", "prefs.json", {"prefs": {}})
    data["prefs"][key] = {"winner_sha": winner_sha, "updated_at": now_iso()}
    write_pass_file("describe", "prefs.json", data)
    return data["prefs"][key]


def load_text_truth() -> dict:
    return read_pass_file("describe", "text_truth.json", {})


def save_text_truth(photo_id: int, text: str, done: bool = True) -> dict:
    data = load_text_truth()
    data[str(int(photo_id))] = {"text": text, "done": bool(done), "updated_at": now_iso()}
    write_pass_file("describe", "text_truth.json", data)
    return data[str(int(photo_id))]


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


def save_category_label(key: str, yes: Iterable[str], done: bool = True) -> dict:
    from .vocab_content import CONTENT_VOCABULARY
    vocab = set(CONTENT_VOCABULARY)
    yes = sorted(set(yes))
    bad = [t for t in yes if t not in vocab]
    if bad:
        raise ValueError(f"not in the content vocabulary: {bad}")
    data = load_category_labels()
    data[key] = {"yes": yes, "done": bool(done), "updated_at": now_iso()}
    write_pass_file("text", "category_labels.json", data)
    return data[key]


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
    data = load_keyword_labels()
    data[key] = {"judged": judged, "wrong": wrong, "done": bool(done), "updated_at": now_iso()}
    write_pass_file("text", "keyword_labels.json", data)
    return data[key]


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
    data = load_verify_sets()
    if data is None:
        raise KeyError("no verify sets")
    for it in data["items"]:
        if it["id"] == item_id and it["kind"] == "planted":
            it["confirmed"] = bool(confirmed)
            it["confirmed_at"] = now_iso()
            save_verify_sets(data)
            return it
    raise KeyError(item_id)
