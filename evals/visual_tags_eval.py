#!/usr/bin/env python
"""Visual-tag eval — score the `category-visual` pass against HUMAN labels.

The prompt that ships today was chosen on 11 photos labelled by the person who
wrote it, after seeing the failures it was meant to fix. This harness is the
honest version: a stratified sample, labelled by the owner on
`/eval/visual-tags`, scored as per-tag precision / recall, and re-runnable for
any named VARIANT — so the same labels settle the production prompt, an
alternate prompt, and a model bake-off across whatever is loaded in LM Studio.

Three steps, one subcommand each:

  sample   pick the photos (stratified over the known failure areas) and write
           `sample.json`. Done once: labels are keyed to it.
               python evals/visual_tags_eval.py sample --db photo_index.db.local

  run      get one variant's predictions for every labelled photo and cache
           them under `runs/<variant>.json` (resumable). The default variant is
           PRODUCTION — the real `describe.tag_visual_photo`, real parser, real
           guard — so the number describes what ships, not a reimplementation:
               export PHOTOSEARCH_TEXT_LLM_URL=http://localhost:1234/v1
               python evals/visual_tags_eval.py run --variant production
               python evals/visual_tags_eval.py run --variant gemma --model google/gemma-4-e2b
               python evals/visual_tags_eval.py run --variant promptC --prompt-file c.txt
           `run --probe --model M` sends one synthetic image and says whether M
           can see at all — check that before spending a bake-off on it.

  report   score the cached runs side by side, plus a `stored` pseudo-variant
           (what `photos.visual_tags` holds right now — no model run needed):
               python evals/visual_tags_eval.py report --db photo_index.db.local

Storage lives in photosearch/visual_tag_eval.py (shared with the labelling
page); the directory is `PHOTOSEARCH_VISUAL_EVAL_DIR`, default
./evals/visual-tags, git-ignored.

The DB is only ever opened READ-ONLY (`mode=ro`). Never through `PhotoDB`,
which migrates on open and would write to the copy being measured.
"""
import argparse
import hashlib
import html
import json
import os
import random
import re
import sqlite3
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from urllib.parse import quote

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

from photosearch import visual_tag_eval as store  # noqa: E402
from photosearch.visual_tags_derive import (  # noqa: E402
    DERIVE_COLUMNS, PERCEIVED_VOCABULARY, PROMPT_SECTIONS, derive_tags)

_PERCEIVED = frozenset(PERCEIVED_VOCABULARY)

STORED = "stored"            # pseudo-variant: the column as it is today
DEFAULT_VARIANT = "production"
DEFAULT_SERVER = "http://localhost:8001"
# Below this many observations a ratio is noise, and printing "1.00" for 1/1
# invites exactly the over-reading the 11-photo A/B suffered from.
MIN_N = 3

_VARIANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# --------------------------------------------------------------------------
# DB access — read-only, always
# --------------------------------------------------------------------------

def open_db_readonly(path):
    """Open `path` with mode=ro. A missing file is an error, not a new stub DB."""
    if not path:
        raise SystemExit("No DB given — pass --db (or set PHOTOSEARCH_DB).")
    uri = "file:" + quote(os.path.abspath(path)) + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.execute("SELECT 1 FROM photos LIMIT 1")
    except sqlite3.OperationalError as e:
        raise SystemExit(f"Cannot open {path} read-only: {e}")
    conn.row_factory = sqlite3.Row
    return conn


def _json_list(raw):
    try:
        val = json.loads(raw) if raw else []
    except ValueError:
        return []
    return [str(t) for t in val] if isinstance(val, list) else []


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

# The shoots where the collapse was measured (`centered` 88-94%, one verbatim
# tag set on 101 photos). Prefix match, so `2026/2026-09-12_ILCE-7RM6` counts.
SPORTS_FOLDERS = ("2026/2026-08-29", "2026/2026-09-12", "2026/2026-09-19")

_LANDSCAPE_CATEGORIES = frozenset({
    "landscape", "natural landscape", "scenic view", "natural beauty",
    "mountain", "ocean", "beach", "forest"})


def rare_terms():
    """The over-applied terms, read from the prompt's own RARE section so a
    vocabulary edit moves the strata with it."""
    for header, groups in PROMPT_SECTIONS:
        if header.startswith("RARE"):
            return [g for g in groups]
    return []


def build_strata(sports_folders=SPORTS_FOLDERS, home_country="US"):
    """[(name, quota-per-60, predicate(row))], in assignment order.

    Order matters: a photo goes to the FIRST stratum that draws it, so the
    narrow strata come first and the uniform remainder last. Quotas are per 60
    photos and scale with --n.
    """
    strata = []
    for prefix in sports_folders:
        strata.append((f"sports:{prefix.rsplit('/', 1)[-1]}", 3,
                       lambda r, p=prefix: (r["folder"] or "").startswith(p)))
    # One stratum per RARE term the model over-applies; the trailing group
    # (reflection / composite / split-screen) is rare in the library too, so
    # it shares one.
    for group in rare_terms():
        name = f"rare:{group[0]}" if len(group) == 1 else "rare:other"
        strata.append((name, 2, lambda r, g=frozenset(group): bool(r["tags"] & g)))
    # 45% / 37% of the library — suspected over-selected, so they need enough
    # predicted positives for a precision figure to mean anything.
    strata.append(("mood:peaceful", 4, lambda r: "peaceful" in r["tags"]))
    strata.append(("mood:moody", 4, lambda r: "moody" in r["tags"]))
    # From EXIF, not from the stored tag: a copy that predates the
    # derive-visual-tags backfill still carries the VLM's guess in the column.
    strata.append(("low-light", 4, lambda r: "low-light" in r["derived"]))
    strata.append(("indoor", 5, lambda r: "indoor" in r["cats"]))
    strata.append(("landscape", 4, lambda r: bool(r["cats"] & _LANDSCAPE_CATEGORIES)))
    strata.append(("travel", 3,
                   lambda r: bool(r["country"]) and r["country"] != home_country))
    strata.append(("empty", 4, lambda r: not r["tags"]))
    return strata


def _load_candidates(conn):
    cols = ", ".join(DERIVE_COLUMNS)
    rows = []
    for r in conn.execute(
            f"SELECT id, folder, visual_tags, categories, country, {cols} "
            "FROM photos WHERE visual_tags IS NOT NULL ORDER BY id"):
        rows.append({"id": r["id"], "folder": r["folder"], "country": r["country"],
                     "tags": frozenset(_json_list(r["visual_tags"])),
                     "cats": frozenset(_json_list(r["categories"])),
                     "derived": frozenset(derive_tags(r))})
    return rows


def choose_sample(conn, n=60, seed=1, sports_folders=SPORTS_FOLDERS,
                  home_country="US"):
    """Return [{"photo_id", "stratum"}] — deterministic for (DB, n, seed)."""
    rows = _load_candidates(conn)
    chosen, taken = [], set()

    def draw(name, pool, k):
        pool = [r for r in pool if r["id"] not in taken]
        # A per-stratum RNG: adding or resizing one stratum must not reshuffle
        # the others, or a tweak to the strata invalidates every label.
        rng = random.Random(f"{seed}:{name}")
        for r in rng.sample(pool, min(k, len(pool))):
            taken.add(r["id"])
            chosen.append({"photo_id": r["id"], "stratum": name})

    for name, per60, pred in build_strata(sports_folders, home_country):
        k = min(max(1, round(per60 * n / 60)), n - len(chosen))
        if k <= 0:
            break
        draw(name, [r for r in rows if pred(r)], k)
    # Whatever a short stratum could not supply lands here too.
    draw("random", rows, n - len(chosen))
    return chosen


def cmd_sample(args):
    existing = store.load_sample()
    if existing["photos"] and not args.force:
        raise SystemExit(
            f"{store.eval_dir() / 'sample.json'} already holds "
            f"{len(existing['photos'])} photos. Labels are keyed to it — "
            "re-sampling orphans them. Pass --force if that is what you want.")
    conn = open_db_readonly(args.db)
    try:
        photos = choose_sample(conn, n=args.n, seed=args.seed,
                               sports_folders=tuple(args.sports_folder or SPORTS_FOLDERS),
                               home_country=args.home_country)
    finally:
        conn.close()
    store.save_sample(photos, args.seed)
    counts = {}
    for p in photos:
        counts[p["stratum"]] = counts.get(p["stratum"], 0) + 1
    print(f"[sample] {len(photos)} photos (seed {args.seed}) -> "
          f"{store.eval_dir() / 'sample.json'}")
    for name, c in counts.items():
        print(f"  {name:<28} {c:>3}")
    if len(photos) < args.n:
        print(f"  ! only {len(photos)} of {args.n}: the DB has too few tagged photos")


# --------------------------------------------------------------------------
# Run cache
# --------------------------------------------------------------------------

def run_path(variant):
    return store.eval_dir() / "runs" / f"{variant}.json"


def load_run(variant):
    path = run_path(variant)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_run(variant, data):
    # Same atomic shape as the label store: a kill mid-dump must not cost an
    # hour of model calls.
    path = run_path(variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{variant}.", suffix=".tmp")
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


def list_variants():
    d = store.eval_dir() / "runs"
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json") if not p.name.startswith("."))


def predictions_of(run):
    """{photo_id: tags-or-None} from a cached run. `None` (no usable answer)
    and `[]` (the model answered "none") are different results and both
    survive the JSON round trip — null vs []."""
    return {int(pid): (None if p["tags"] is None else list(p["tags"]))
            for pid, p in run["predictions"].items()}


# --------------------------------------------------------------------------
# Pixels + the production tagger
# --------------------------------------------------------------------------

def fetch_image(server, photo_id, kind="full", timeout=120):
    """Bytes of one photo from a running photosearch server.

    `full` is what the production worker downloads (`WorkerClient.
    download_photo`), so `tag_visual_photo` then does the very same 1024-px
    LANCZOS / JPEG q85 re-encode it does in the fleet. `preview` (1920 px,
    q82) is ~10x lighter but adds a JPEG generation the fleet never sees.
    """
    import urllib.request
    url = f"{server.rstrip('/')}/api/photos/{int(photo_id)}/{kind}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


@contextmanager
def prompt_override(text):
    """Swap ONLY the prompt text, inside this process.

    `tag_visual_photo` looks `_build_visual_prompt` up as a module global on
    every call, so rebinding it reaches the real code path — same image
    encode, parser, guard and retry — without an optional parameter in
    describe.py that production would then have to keep honest.
    """
    from photosearch import describe
    if text is None:
        yield
        return
    original = describe._build_visual_prompt
    describe._build_visual_prompt = lambda vocab: text
    try:
        yield
    finally:
        describe._build_visual_prompt = original


@contextmanager
def vocab_override(extra):
    """Let the parser ACCEPT candidate tags, inside this process.

    `tag_visual_photo` imports PERCEIVED_VOCABULARY from visual_tags_derive on
    every call and builds its parser's allow-set from it, so a candidate the
    prompt offers would otherwise be silently dropped as off-vocabulary. The
    prompt text is a separate matter — pair this with --prompt-file.
    """
    from photosearch import visual_tags_derive as vtd
    if not extra:
        yield
        return
    original = vtd.PERCEIVED_VOCABULARY
    vtd.PERCEIVED_VOCABULARY = list(original) + [t for t in extra if t not in original]
    try:
        yield
    finally:
        vtd.PERCEIVED_VOCABULARY = original


class TransportError(RuntimeError):
    """The backend could not be reached / refused the request — says nothing
    about the photo, so it must not be cached as an "unanswered" result."""


def production_tagger(path, model):
    """(tags-or-None, calls) from the real `describe.tag_visual_photo`.

    The chat function is wrapped for the duration so the raw answers are kept
    (for reading failures afterwards) and so a transport error is told apart
    from a model that answered uselessly: `tag_visual_photo` returns None for
    both, and caching a dead LM Studio as 60 "unanswered" photos would quietly
    score as zero recall.
    """
    from photosearch import describe
    calls = []
    real = describe._ollama_chat_with_retry

    def recording(*a, **kw):
        entry = {"temperature": (kw.get("options") or {}).get("temperature")}
        calls.append(entry)
        try:
            entry["raw"] = real(*a, **kw)
        except Exception as e:
            entry["error"] = f"{e.__class__.__name__}: {e}"
            raise
        return entry["raw"]

    describe._ollama_chat_with_retry = recording
    try:
        tags = describe.tag_visual_photo(path, model=model)
    finally:
        describe._ollama_chat_with_retry = real
    if tags is None and calls and all("error" in c for c in calls):
        raise TransportError(calls[-1]["error"])
    if tags is None and not calls:
        # tag_visual_photo bailed before asking: no `ollama` package (it gates
        # on HAS_OLLAMA even on the LM Studio route) or an unreadable file.
        raise TransportError("tag_visual_photo made no model call "
                             "(is the `ollama` package installed?)")
    return tags, calls


def _pin_model(model):
    """Make an explicit --model the one that actually runs.

    On the LM Studio route the name passed to the call is IGNORED — the role
    env var picks the model (see `describe._resolve_openai_model`). A bake-off
    that passes `--model gemma` while PHOTOSEARCH_LLM_VISUAL_MODEL says qwen
    would score qwen under gemma's name. Setting the role var in THIS process
    keeps the production resolution path and makes it resolve to the request.
    """
    if model and os.environ.get("PHOTOSEARCH_TEXT_LLM_URL"):
        os.environ["PHOTOSEARCH_LLM_VISUAL_MODEL"] = model


def run_variant(variant, *, model=None, prompt_text=None, prompt_file=None,
                extra_vocab=(), server=DEFAULT_SERVER, pixels="full", limit=None, force=False,
                tagger=None, fetch=None, log=print):
    """Predict every `done`-labelled sample photo; cache as we go. Returns the
    run dict. `tagger` / `fetch` are injectable so tests need no model or
    network."""
    from photosearch import describe
    if not _VARIANT_RE.match(variant) or variant == STORED:
        raise SystemExit(f"Bad variant name {variant!r} "
                         f"(letters, digits, . _ - ; '{STORED}' is reserved).")
    tagger = tagger or production_tagger
    fetch = fetch or fetch_image

    _pin_model(model)
    nominal = model or describe.TAGS_MODEL
    effective = describe.effective_model(nominal, "visual")
    prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12] \
        if prompt_text is not None else None

    extra_vocab = sorted(set(extra_vocab))
    unknown = [t for t in extra_vocab if t not in store.CANDIDATE_TAGS]
    if unknown:
        raise SystemExit(f"--extra-vocab: not candidate tags: {unknown} "
                         f"(have: {sorted(store.CANDIDATE_TAGS)})")
    if extra_vocab and prompt_text is None:
        # The production prompt never mentions a candidate, so the model
        # would never emit it and the run would score it at recall 0.
        raise SystemExit("--extra-vocab needs --prompt-file: the shipped "
                         "prompt does not offer candidate tags.")

    run = None if force else load_run(variant)
    if run is not None:
        # One variant = one (model, prompt). Resuming under a different one
        # would mix two experiments in a file that reports as one.
        for key, now in (("effective_model", effective), ("prompt_sha", prompt_sha),
                         ("extra_vocab", extra_vocab)):
            if (run.get(key) or ([] if key == "extra_vocab" else None)) != now:
                raise SystemExit(
                    f"variant {variant!r} was run with {key}={run.get(key)!r}, "
                    f"now {now!r}. Use a new --variant name, or --force to "
                    "discard the cached run.")
    else:
        run = {"variant": variant, "model": nominal, "effective_model": effective,
               "prompt_file": prompt_file, "prompt_sha": prompt_sha,
               "extra_vocab": extra_vocab, "pixels": pixels, "server": server,
               "created": datetime.now(timezone.utc).isoformat(),
               "predictions": {}}

    labels = store.scoreable_labels()
    todo = [p["photo_id"] for p in store.load_sample()["photos"]
            if p["photo_id"] in labels and str(p["photo_id"]) not in run["predictions"]]
    if limit is not None:
        todo = todo[:limit]
    log(f"[run] {variant}: model={effective} prompt="
        f"{prompt_file or 'production'} — {len(todo)} to do, "
        f"{len(run['predictions'])} cached, {len(labels)} labelled")

    errors = 0
    with prompt_override(prompt_text), vocab_override(extra_vocab), \
            tempfile.TemporaryDirectory() as tmp:
        for i, pid in enumerate(todo, 1):
            path = os.path.join(tmp, f"{pid}.jpg")
            try:
                with open(path, "wb") as f:
                    f.write(fetch(server, pid, pixels))
                t0 = time.time()
                tags, calls = tagger(path, nominal)
                latency = time.time() - t0
            except Exception as e:
                # Not cached, so the next run retries it.
                errors += 1
                log(f"  ! {pid}: {e}")
                continue
            finally:
                if os.path.exists(path):
                    os.unlink(path)
            run["predictions"][str(pid)] = {
                "tags": None if tags is None else list(tags),
                "latency_s": round(latency, 3),
                "effective_model": effective,
                "calls": calls,
            }
            save_run(variant, run)
            shown = "UNANSWERED" if tags is None else (", ".join(tags) or "none")
            log(f"  [{i}/{len(todo)}] {pid}: {shown}  ({latency:.1f}s)")
    if errors:
        log(f"[run] {errors} photo(s) failed and were NOT cached — re-run to retry")
    return run


# --------------------------------------------------------------------------
# Probe — can this model see at all?
# --------------------------------------------------------------------------

def _probe_image_b64():
    import base64
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (512, 512), (220, 20, 20)).save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def probe(model=None, chat=None):
    """Send ONE solid-red image and ask its colour.

    "Accepted the request" is not the test: a text-only model behind an
    OpenAI-compatible server may take the payload and silently ignore the
    image. Only an answer that names the colour proves pixels reached it.
    Returns {"model", "verdict": sees|blind|rejected, "answer"|"error"}.
    """
    from photosearch import describe
    _pin_model(model)
    nominal = model or describe.TAGS_MODEL
    out = {"model": describe.effective_model(nominal, "visual")}
    chat = chat or describe._ollama_chat_with_retry
    try:
        answer = chat(
            model=nominal,
            messages=[{"role": "user",
                       "content": "What single colour fills this image? "
                                  "Answer with one word.",
                       "images": [_probe_image_b64()]}],
            options={"temperature": 0, "num_predict": 20},
            retries=1, role="visual")
    except Exception as e:
        return {**out, "verdict": "rejected", "error": f"{e.__class__.__name__}: {e}"}
    out["answer"] = (answer or "").strip()
    out["verdict"] = "sees" if "red" in out["answer"].lower() else "blind"
    return out


def cmd_run(args):
    if args.probe:
        res = probe(args.model)
        print(f"[probe] model={res['model']} -> {res['verdict'].upper()}")
        print(f"        {res.get('answer') or res.get('error')!r}")
        if res["verdict"] == "blind":
            print("        accepted the image but did not name its colour — "
                  "treat as NOT vision-capable")
        raise SystemExit(0 if res["verdict"] == "sees" else 1)
    prompt_text = None
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt_text = f.read()
    run_variant(args.variant, model=args.model, prompt_text=prompt_text,
                prompt_file=args.prompt_file,
                extra_vocab=[t.strip() for t in (args.extra_vocab or "").split(",") if t.strip()],
                server=args.server, pixels=args.pixels, limit=args.limit, force=args.force)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def stored_predictions(conn, photo_ids):
    """What `photos.visual_tags` holds for `photo_ids`, PERCEIVED terms only
    (derived / frozen terms are not the model's to be scored on). A NULL
    column is None — unanswered — because that is what the library holds."""
    out = {}
    ids = list(photo_ids)
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        marks = ",".join("?" * len(chunk))
        for r in conn.execute(
                f"SELECT id, visual_tags FROM photos WHERE id IN ({marks})", chunk):
            raw = r["visual_tags"]
            out[r["id"]] = None if raw is None else \
                sorted(set(_json_list(raw)) & _PERCEIVED)
    return out


def _summarise(predicted, labels, sample, extra=()):
    """score() plus the things it does not carry: tags/photo and strata.
    `extra` = the candidate tags this variant offered (scored for it alone)."""
    res = store.score(predicted, labels, extra_tags=extra)
    allowed = _PERCEIVED | set(extra)
    answered = [set(t) & allowed for pid, t in predicted.items()
                if pid in labels and t is not None]
    res["avg_tags"] = (sum(len(t) for t in answered) / len(answered)) if answered else None
    res["strata"] = {}
    by_stratum = {}
    for p in sample["photos"]:
        by_stratum.setdefault(p["stratum"], []).append(p["photo_id"])
    for name, pids in by_stratum.items():
        sub_labels = {pid: labels[pid] for pid in pids if pid in labels}
        sub = store.score({pid: predicted[pid] for pid in sub_labels if pid in predicted},
                          sub_labels, extra_tags=extra)
        res["strata"][name] = {"photos_scored": sub["photos_scored"],
                               "unanswered": sub["unanswered"], **sub["overall"]}
    return res


def build_report(variant_predictions, labels=None, sample=None, extras=None):
    """{"variants": [names], "results": {name: summary}, "labelled": n}."""
    labels = store.scoreable_labels() if labels is None else labels
    sample = store.load_sample() if sample is None else sample
    return {"variants": list(variant_predictions), "labelled": len(labels),
            "sampled": len(sample["photos"]),
            "results": {name: _summarise(pred, labels, sample,
                                         (extras or {}).get(name, ()))
                        for name, pred in variant_predictions.items()}}


def fmt_ratio(num, den):
    """`0.67 (4/6)`, or `n<3` — the denominator is always shown, and below
    MIN_N the ratio is withheld rather than printed as a confident 1.00."""
    if den < MIN_N:
        return f"n<{MIN_N} ({num}/{den})" if den else "—"
    return f"{num / den:.2f} ({num}/{den})"


def _precision(c):
    return fmt_ratio(c["tp"], c["tp"] + c["fp"])


def _recall(c):
    return fmt_ratio(c["tp"], c["tp"] + c["fn"])


def _tag_order(report):
    """Tags by total false positives across variants (the failure this eval
    exists to find), dropping tags nobody predicted or labelled."""
    res = list(report["results"].values())
    # Union, not PERCEIVED_VOCABULARY: a candidate tag appears only in the
    # variants that offered it.
    tags = list(dict.fromkeys(t for r in res for t in r["per_tag"]))
    totals = {t: [sum(r["per_tag"].get(t, {}).get(k, 0) for r in res)
                  for k in ("fp", "fn", "tp", "debatable")]
              for t in tags}
    live = [t for t, v in totals.items() if any(v)]
    return sorted(live, key=lambda t: (-totals[t][0], -totals[t][1], t))


def report_tables(report):
    """[(title, header-row, [rows])] — one source for the text and HTML views."""
    names = report["variants"]
    res = report["results"]
    tables = []

    rows = []
    for n in names:
        r = res[n]
        avg = "—" if r["avg_tags"] is None else f"{r['avg_tags']:.2f}"
        rows.append([n, r["photos_scored"], r["unanswered"], avg,
                     _precision(r["overall"]), _recall(r["overall"]),
                     r["overall"]["fp"], r["overall"]["fn"], r["overall"]["debatable"]])
    tables.append(("Overall", ["variant", "photos", "unanswered", "tags/photo",
                               "precision", "recall", "fp", "fn", "debatable"], rows))

    head = ["tag"]
    for n in names:
        head += [f"{n}: tp/fp/fn/deb", "precision", "recall"]
    rows = []
    for t in _tag_order(report):
        row = [t]
        for n in names:
            c = res[n]["per_tag"].get(t)
            if c is None:          # a candidate this variant never offered
                row += ["not offered", "—", "—"]
                continue
            row += [f"{c['tp']}/{c['fp']}/{c['fn']}/{c['debatable']}",
                    _precision(c), _recall(c)]
        rows.append(row)
    tables.append(("Per tag (sorted by false positives)", head, rows))

    head = ["stratum"]
    for n in names:
        head += [f"{n}: photos", "precision", "recall"]
    strata = []
    for n in names:
        strata += [s for s in res[n]["strata"] if s not in strata]
    rows = []
    for s in strata:
        row = [s]
        for n in names:
            c = res[n]["strata"].get(s)
            row += [c["photos_scored"], _precision(c), _recall(c)] if c else ["—"] * 3
        rows.append(row)
    tables.append(("Per stratum", head, rows))
    return tables


def render_text(report):
    out = [f"{report['labelled']} labelled (done) of {report['sampled']} sampled. "
           f"Ratios are shown with their counts; below {MIN_N} they are withheld. "
           "Debatable tags count as neither hit nor miss."]
    for title, head, rows in report_tables(report):
        table = [[str(c) for c in r] for r in [head] + rows]
        widths = [max(len(r[i]) for r in table) for i in range(len(head))]
        out.append(f"\n=== {title} ===")
        for r in table:
            out.append("  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                                 for i, (c, w) in enumerate(zip(r, widths))).rstrip())
    return "\n".join(out)


def render_html(report):
    parts = []
    for title, head, rows in report_tables(report):
        parts.append(f"<h2>{html.escape(title)}</h2><table><thead><tr>"
                     + "".join(f"<th>{html.escape(str(h))}</th>" for h in head)
                     + "</tr></thead><tbody>")
        for r in rows:
            parts.append("<tr>" + "".join(
                f"<td{' class=low' if str(c).startswith('n<') else ''}>"
                f"{html.escape(str(c))}</td>" for c in r) + "</tr>")
        parts.append("</tbody></table>")
    return f"""<!DOCTYPE html><html><head><meta charset=utf-8>
<title>Visual-tag eval</title><style>
 body{{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
 h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}}
 table{{border-collapse:collapse;margin:12px 0}}
 th,td{{border:1px solid #333;padding:5px 10px;text-align:right}}
 td:first-child,th:first-child{{text-align:left}}
 td.low{{color:#777}}
</style></head><body>
<h1>Visual-tag eval — {report['labelled']} labelled of {report['sampled']} sampled</h1>
<p style="color:#999">Ratios carry their counts; below {MIN_N} observations they
are withheld. Debatable tags count as neither hit nor miss.</p>
{''.join(parts)}
</body></html>"""


def cmd_report(args):
    labels = store.scoreable_labels()
    if not labels:
        raise SystemExit("No `done` labels yet — label some photos on "
                         "/eval/visual-tags first.")
    wanted = [v.strip() for v in args.variants.split(",") if v.strip()] \
        if args.variants else [STORED] + list_variants()
    preds = {}
    extras = {}
    for name in wanted:
        if name == STORED:
            if not args.db:
                # Only an explicit request is an error; by default the model
                # runs are still worth reporting without a DB to hand.
                if args.variants:
                    raise SystemExit("`stored` needs --db (or PHOTOSEARCH_DB).")
                print("[report] no --db: skipping the `stored` pseudo-variant")
                continue
            conn = open_db_readonly(args.db)
            try:
                preds[STORED] = stored_predictions(conn, labels)
            finally:
                conn.close()
            continue
        run = load_run(name)
        if run is None:
            raise SystemExit(f"No cached run for variant {name!r} "
                             f"(have: {', '.join(list_variants()) or 'none'}).")
        preds[name] = predictions_of(run)
        extras[name] = run.get("extra_vocab") or []
        print(f"[report] {name}: model={run.get('effective_model')} "
              f"prompt={run.get('prompt_file') or 'production'} "
              f"({len(run['predictions'])} cached)")
    if not preds:
        raise SystemExit("Nothing to report — no runs cached and no --db.")
    report = build_report(preds, labels, extras=extras)
    print(render_text(report))
    if args.html:
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(render_html(report))
        print(f"\n[report] {args.html}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("sample", help="Choose the stratified sample (once).")
    sp.add_argument("--db", default=os.environ.get("PHOTOSEARCH_DB"),
                    help="photo_index.db — opened READ-ONLY.")
    sp.add_argument("--n", type=int, default=60)
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--force", action="store_true",
                    help="Overwrite an existing sample (orphans its labels).")
    sp.add_argument("--sports-folder", action="append",
                    help=f"Folder prefix of a sports shoot (repeatable). "
                         f"Default: {', '.join(SPORTS_FOLDERS)}")
    sp.add_argument("--home-country", default="US",
                    help="Photos outside it form the `travel` stratum.")
    sp.set_defaults(func=cmd_sample)

    rp = sub.add_parser("run", help="Predict the labelled photos for one variant.")
    rp.add_argument("--variant", default=DEFAULT_VARIANT)
    rp.add_argument("--model", help="Model id. On the LM Studio route this is "
                                    "pinned as the visual role model for this process.")
    rp.add_argument("--extra-vocab", help="Comma-separated CANDIDATE tags (see "
                    "photosearch/visual_tag_eval.py) the parser should accept and "
                    "the report should score for this variant. Needs --prompt-file.")
    rp.add_argument("--prompt-file", help="Alternate prompt text (replaces only "
                                          "the prompt; parser and guard stay production).")
    rp.add_argument("--server", default=os.environ.get("PHOTOSEARCH_EVAL_SERVER",
                                                       DEFAULT_SERVER),
                    help="photosearch server to fetch pixels from.")
    rp.add_argument("--pixels", choices=("full", "preview"), default="full",
                    help="full = the original, as the worker fleet downloads "
                         "(default); preview = 1920px, lighter, one extra JPEG pass.")
    rp.add_argument("--limit", type=int, help="Predict at most N new photos.")
    rp.add_argument("--force", action="store_true",
                    help="Discard this variant's cached run and start over.")
    rp.add_argument("--probe", action="store_true",
                    help="Send ONE synthetic image and report whether the model "
                         "can see it. Exit 0 only if it can.")
    rp.set_defaults(func=cmd_run)

    tp = sub.add_parser("report", help="Score cached runs against the labels.")
    tp.add_argument("--variants", help=f"Comma-separated; default = `{STORED}` "
                                       "+ every cached run.")
    tp.add_argument("--db", default=os.environ.get("PHOTOSEARCH_DB"),
                    help=f"For the `{STORED}` pseudo-variant — opened READ-ONLY.")
    tp.add_argument("--html", help="Also write an HTML report here.")
    tp.set_defaults(func=cmd_report)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
