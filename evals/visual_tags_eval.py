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
           Every row is scored on the photos ALL variants cover, next to
           constant baselines (top-1..3 most-labelled tags, recite-all, empty)
           with micro/macro-F1 and paired-bootstrap ΔF1 CIs vs the reference
           and the best baseline; runs whose prompt / guard+parser hash is
           missing or differs from today's code are marked STALE.

  spec     replay the cached answers, reformatted / contradicted / padded,
           through the PRODUCTION parser, guard and `tag_visual_photo` (chat
           scripted in-process — no model). Also a section of `report`.
               python evals/visual_tags_eval.py spec

Plus the sharpness ground truth (docs/plans/sharpness-measurement.md, step 1):

  sample-sharpness   a SEPARATE, blurry-weighted 60 under sharpness/, labelled
                     for `sharp` / `blurry` at /eval/visual-tags?set=sharpness
                     in a 100% loupe on the original. Never touches sample.json.
               python evals/visual_tags_eval.py sample-sharpness --db photo_index.db.local
  agreement --set all   labeller self-consistency on `blurry` over both samples.

`sharp` / `blurry` are MEASURED tags: no model variant is ever scored on them.

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
    CONTRADICTORY_PAIRS, DERIVE_COLUMNS, PERCEIVED_VOCABULARY, PROMPT_SECTIONS,
    derive_tags, parse_exif_number)

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
# Sharpness sample — ground truth for the measured `sharp` / `blurry` tags
# --------------------------------------------------------------------------
#
# docs/plans/sharpness-measurement.md, step 1. A SEPARATE sample
# (store.load_sample("sharpness"), files under sharpness/) so drawing it never
# orphans the visual sample, its labels or its cached runs. Stratified to
# OVERSAMPLE likely-blurry photos: a uniform draw would hold a handful of
# genuinely blurry frames and the eval gate needs at least 25 positives.

SHARPNESS_CAMERAS = ("ILCE-7RM6", "ILCE-7M4")        # a7R VI, a7 IV
_PHONE_MAKES = ("apple", "google", "samsung", "oneplus", "motorola", "xiaomi",
                "huawei", "lg", "sony mobile", "nothing")
_PHONE_MODEL_PREFIXES = ("iphone", "pixel", "sm-", "galaxy")
# Chrome cannot draw these, so the 100% loupe cannot show them — a photo the
# labeller cannot judge at native resolution must not be in this sample.
_UNVIEWABLE_EXT = (".heic", ".heif", ".arw", ".dng", ".cr2", ".cr3", ".nef",
                   ".raf", ".orf", ".rw2")
_INDOOR_SPORT_TERMS = ("indoor soccer", "soccer indoor", "futsal", "gymnasium",
                       "gymnastics", "gym", "basketball", "volleyball",
                       "badminton", "hockey rink", "ice rink", "swimming pool")
NIGHT_HOURS = frozenset(list(range(21, 24)) + list(range(0, 6)))   # 21:00-06:00
HIGH_ISO = 3200
BOKEH_MAX_F = 2.8
AES_SHARPNESS_MAX = 2
_SHARPNESS_COLUMNS = ("folder", "filepath", "visual_tags", "categories", "keywords",
                      "aes_sharpness", "subject_boxes", "date_taken",
                      "camera_make", "camera_model") + tuple(DERIVE_COLUMNS)
_HOUR_RE = re.compile(r"[T ](\d{2}):\d{2}")


def camera_class(make, model):
    """'ILCE-7RM6' | 'ILCE-7M4' | 'phone' | 'gopro' | None."""
    make_l, model_s = (make or "").strip().lower(), (model or "").strip()
    model_l = model_s.lower()
    if model_s in SHARPNESS_CAMERAS:
        return model_s
    if "gopro" in make_l or model_l.startswith("hero"):
        return "gopro"
    if make_l.startswith(_PHONE_MAKES) or model_l.startswith(_PHONE_MODEL_PREFIXES):
        return "phone"
    return None


def _hour(date_taken):
    m = _HOUR_RE.search(date_taken or "")
    return int(m.group(1)) if m else None


def _load_sharpness_candidates(conn):
    have = {r[1] for r in conn.execute("PRAGMA table_info(photos)")}
    # A column an older DB lacks reads as NULL, so its stratum is simply empty.
    cols = ", ".join(c if c in have else f"NULL AS {c}" for c in _SHARPNESS_COLUMNS)
    faces = {}
    try:
        for pid, n in conn.execute("SELECT photo_id, COUNT(*) FROM faces GROUP BY photo_id"):
            faces[pid] = n
    except sqlite3.OperationalError:
        pass
    rows = []
    for r in conn.execute(f"SELECT id, {cols} FROM photos ORDER BY id"):
        if (r["filepath"] or "").lower().endswith(_UNVIEWABLE_EXT):
            continue
        text = " ".join(_json_list(r["categories"]) + _json_list(r["keywords"])).lower()
        derived = derive_tags(r)
        rows.append({
            "id": r["id"],
            "tags": frozenset(_json_list(r["visual_tags"])),
            "aes_sharpness": r["aes_sharpness"],
            "long_exposure": "long-exposure" in derived,
            "indoor_sport": any(t in text for t in _INDOOR_SPORT_TERMS)
                            or ("indoor" in text and "sport" in text),
            "iso": _parse_float(r["iso"]),
            "hour": _hour(r["date_taken"]),
            "f": _parse_float(r["f_number"]),
            "faces": faces.get(r["id"], 0),
            # '[]' = grounded, no clear subject; NULL = never grounded.
            "no_subject": _is_empty_json_list(r["subject_boxes"]),
            "camera": camera_class(r["camera_make"], r["camera_model"]),
        })
    return rows


def _parse_float(raw):
    return parse_exif_number(raw)


def _is_empty_json_list(raw):
    if raw is None:
        return False
    try:
        return json.loads(raw) == []
    except (TypeError, ValueError):
        return False


def build_sharpness_strata():
    """[(name, quota-per-60, predicate(row))] in assignment order — the
    likely-blurry strata first, so they are filled before the camera and
    random strata take anything."""
    return [
        # What the VLM already called blurry (1,284 photos live): a biased
        # pool, but it is where positives are densest.
        ("stored-blurry", 9, lambda r: "blurry" in r["tags"]),
        # Half of a 4-photo hand check was a false positive; label it anyway —
        # it is the baseline step 3 must beat.
        ("aes-sharpness<=2", 8, lambda r: r["aes_sharpness"] is not None
                                          and r["aes_sharpness"] <= AES_SHARPNESS_MAX),
        # Intentional motion counts as blurry (eval_api.LABELLER_NOTES) but is
        # reported separately: a measurement SHOULD fire on it, and a reader
        # must be able to see how much of the recall it accounts for.
        ("long-exposure", 4, lambda r: r["long_exposure"]),
        ("indoor-sports", 7, lambda r: r["indoor_sport"]),
        # Noise is NOT blur: the GoPro night street false positive lives here.
        ("iso>=3200-or-night", 7, lambda r: (r["iso"] is not None and r["iso"] >= HIGH_ISO)
                                            or r["hour"] in NIGHT_HOURS),
        # Bokeh is NOT blur: a sharp face over a melted background.
        ("bokeh-portrait", 6, lambda r: r["faces"] > 0 and r["f"] is not None
                                        and r["f"] <= BOKEH_MAX_F),
        # No subject: the whole frame decides, so a whole-frame metric is
        # tested where it is supposed to work.
        ("no-subject-landscape", 4, lambda r: r["no_subject"] and r["faces"] == 0),
        ("camera:a7RVI", 3, lambda r: r["camera"] == "ILCE-7RM6"),
        ("camera:a7IV", 3, lambda r: r["camera"] == "ILCE-7M4"),
        ("camera:phone", 3, lambda r: r["camera"] == "phone"),
        ("camera:gopro", 3, lambda r: r["camera"] == "gopro"),
    ]


def choose_sharpness_sample(conn, n=60, seed=1, exclude=()):
    """[{"photo_id", "stratum"}] — deterministic for (DB, n, seed, exclude).
    `exclude` = ids already in the visual sample, so the owner's combined
    labelling set is n + 60 DISTINCT photos."""
    excluded = set(int(i) for i in exclude)
    rows = [r for r in _load_sharpness_candidates(conn) if r["id"] not in excluded]
    chosen, taken = [], set()

    def draw(name, pool, k):
        pool = [r for r in pool if r["id"] not in taken]
        rng = random.Random(f"{seed}:sharpness:{name}")    # per-stratum, as above
        for r in rng.sample(pool, min(k, len(pool))):
            taken.add(r["id"])
            chosen.append({"photo_id": r["id"], "stratum": name})

    for name, per60, pred in build_sharpness_strata():
        k = min(max(1, round(per60 * n / 60)), n - len(chosen))
        if k <= 0:
            break
        draw(name, [r for r in rows if pred(r)], k)
    draw("random", rows, n - len(chosen))
    return chosen


#: `sample-sharpness --extend N` draws only from these: on the first 120
#: labels they held 9 of 21 blurry photos at a ~50% hit rate, against ~5%
#: everywhere else. Suffixed so the report can tell the draws apart.
EXTEND_STRATA = ("stored-blurry", "aes-sharpness<=2")


def extend_sharpness_sample(conn, existing, n, seed=1, exclude=(),
                            strata=EXTEND_STRATA):
    """`existing` + n NEW photos split evenly over EXTEND_STRATA, never
    repeating an existing or excluded id, so labels already keyed to the
    sample stay valid (appending never orphans them)."""
    taken = {int(p["photo_id"]) for p in existing} | {int(i) for i in exclude}
    rows = [r for r in _load_sharpness_candidates(conn) if r["id"] not in taken]
    preds = {name: pred for name, _q, pred in build_sharpness_strata()}
    out = list(existing)
    for i, name in enumerate(strata):
        k = n // len(strata) + (1 if i < n % len(strata) else 0)
        pool = [r for r in rows if preds[name](r) and r["id"] not in taken]
        rng = random.Random(f"{seed}:sharpness-extend:{name}:{len(existing)}")
        for r in rng.sample(pool, min(k, len(pool))):
            taken.add(r["id"])
            out.append({"photo_id": r["id"], "stratum": f"{name}+ext"})
    return out


def cmd_sample_sharpness(args):
    existing = store.load_sample("sharpness")
    path = store.eval_dir() / store.SAMPLE_FILES["sharpness"]
    if args.extend:
        if not existing["photos"]:
            raise SystemExit("nothing to extend: draw the sample first")
        exclude = [p["photo_id"] for p in store.load_sample("visual")["photos"]]
        conn = open_db_readonly(args.db)
        try:
            photos = extend_sharpness_sample(conn, existing["photos"], args.extend,
                                             seed=existing.get("seed") or args.seed,
                                             exclude=exclude,
                                             strata=tuple(args.extend_from.split(",")))
        finally:
            conn.close()
        store.save_sample(photos, existing.get("seed") or args.seed, sample="sharpness")
        print(f"[sample-sharpness] extended {len(existing['photos'])} -> {len(photos)} "
              f"photos -> {path}")
        return
    if existing["photos"] and not args.force:
        raise SystemExit(
            f"{path} already holds {len(existing['photos'])} photos. Labels are "
            "keyed to it — re-sampling orphans them. Pass --force if that is "
            "what you want.")
    exclude = [p["photo_id"] for p in store.load_sample("visual")["photos"]]
    conn = open_db_readonly(args.db)
    try:
        photos = choose_sharpness_sample(conn, n=args.n, seed=args.seed, exclude=exclude)
    finally:
        conn.close()
    store.save_sample(photos, args.seed, sample="sharpness")
    counts = {}
    for p in photos:
        counts[p["stratum"]] = counts.get(p["stratum"], 0) + 1
    print(f"[sample-sharpness] {len(photos)} photos (seed {args.seed}, "
          f"{len(exclude)} visual-sample ids excluded) -> {path}")
    for name, _q, _p in build_sharpness_strata() + [("random", 0, None)]:
        print(f"  {name:<28} {counts.get(name, 0):>3}")
    print("Label at /eval/visual-tags?set=sharpness "
          "(blind recheck: ?set=sharpness-recheck).")


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
    from photosearch import model_eval
    if kind == "full":
        # Read once through the shared originals cache: every variant x model
        # otherwise re-pulls the same photos from the NAS via the replica, and
        # a NAS restart 502'd 28 of them mid-run on 2026-09-26.
        return model_eval.original_path(photo_id, server).read_bytes()
    return model_eval.fetch_from_server(server, photo_id, kind, timeout)


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


# --------------------------------------------------------------------------
# Code fingerprint — what produced a cached run
# --------------------------------------------------------------------------
#
# A run used to record `prompt_sha: None` whenever there was no --prompt-file,
# so a `production` run resumed after a describe.py prompt edit passed the
# "mixing two experiments" check and half its photos answered a different
# prompt. Both hashes come from the LIVE production code — the real prompt
# builder and the real guard/parser source — never from a copy.

#: What a raw answer passes through after the model. A change to any of them
#: changes what the same raw answer becomes, so it changes `code_sha`.
GUARD_PARSER_FUNCS = ("_parse_visual_response", "_is_explicit_empty_answer",
                      "_visual_contradictions", "_visual_answer_problem",
                      "_drop_visual_contradictions", "tag_visual_photo")
_GUARD_PARSER_CONSTANTS = ("_VISUAL_MAX_TAGS", "_VISUAL_MAX_PLAUSIBLE_TAGS",
                           "_VISUAL_SYNONYMS", "_VISUAL_EMPTY_ANSWERS",
                           "_VISUAL_WRAPPERS", "_VISUAL_BULLET_RE", "_VISUAL_LABEL_RE")


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def production_prompt_sha():
    """sha of the prompt `tag_visual_photo` would send right now — the real
    `_build_visual_prompt` over the live vocabulary, no model involved. Call
    it OUTSIDE `prompt_override`, or it hashes the override."""
    from photosearch import describe, visual_tags_derive as vtd
    return _sha(describe._build_visual_prompt(list(vtd.PERCEIVED_VOCABULARY)))


def _canon(value):
    if isinstance(value, re.Pattern):
        return value.pattern
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, dict):
        return sorted(value.items())
    return value


def guard_parser_sha():
    """sha of the guard + parser SOURCE and the constants they read, plus the
    perceived vocabulary (the parser's allow-set) and CONTRADICTORY_PAIRS."""
    import inspect
    from photosearch import describe, visual_tags_derive as vtd
    parts = []
    for name in GUARD_PARSER_FUNCS:
        fn = getattr(describe, name, None)
        try:
            parts.append(inspect.getsource(fn))
        except (OSError, TypeError):
            parts.append(repr(fn))
    parts.append(repr([(n, _canon(getattr(describe, n, None)))
                       for n in _GUARD_PARSER_CONSTANTS]))
    parts.append(repr(sorted(vtd.PERCEIVED_VOCABULARY)))
    parts.append(repr(vtd.CONTRADICTORY_PAIRS))
    return _sha("\n".join(parts))


def run_settings(nominal, effective):
    """Generation settings a run is made under: the first-call temperature
    (the rule is 0) and the reasoning effort — and whether it is actually SENT
    for this model (`describe._model_has_no_reasoning_mode`)."""
    from photosearch import describe
    effort = os.environ.get("PHOTOSEARCH_LLM_REASONING_EFFORT", "").strip() or None
    return {"temperature": describe._options_for_model(nominal).get("temperature"),
            "reasoning_effort": effort,
            "reasoning_effort_sent": bool(effort) and not
            describe._model_has_no_reasoning_mode(effective)}


def _resolve_prompt_file(path):
    for p in (path, os.path.join(PROJECT, path)):
        if p and os.path.isfile(p):
            return p
    return None


def run_staleness(run, live_prompt=None, live_code=None):
    """{"prompt", "code", "stale"} — does this cached run still describe what
    the code would do now? A hash that was never recorded is STALE too: the
    run cannot show it is current."""
    live_prompt = production_prompt_sha() if live_prompt is None else live_prompt
    live_code = guard_parser_sha() if live_code is None else live_code
    sha, pfile = run.get("prompt_sha"), run.get("prompt_file")
    if sha is None:
        prompt = "STALE: prompt not recorded"
    elif sha == live_prompt:
        prompt = "production (current)"
    elif pfile is None:
        prompt = "STALE: production prompt changed"
    else:
        found = _resolve_prompt_file(pfile)
        if found is None:
            prompt = f"alt {os.path.basename(pfile)} (file gone)"
        else:
            with open(found, encoding="utf-8") as f:
                now = _sha(f.read())
            prompt = (f"alt {os.path.basename(pfile)}" if now == sha
                      else f"STALE: {os.path.basename(pfile)} edited since")
    code = run.get("code_sha")
    code = ("STALE: not recorded" if code is None
            else "current" if code == live_code else "STALE: guard/parser changed")
    return {"prompt": prompt, "code": code,
            "stale": prompt.startswith("STALE") or code.startswith("STALE")}


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
    # No --prompt-file = the LIVE production prompt, hashed from the real
    # builder, so a describe.py prompt edit between two resumes is caught.
    prompt_sha = _sha(prompt_text) if prompt_text is not None else production_prompt_sha()
    code_sha = guard_parser_sha()
    settings = run_settings(nominal, effective)

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
        # A run made before these were recorded (None) cannot show it matches,
        # so it is refused too: its first half may answer different code.
        for key, now in (("effective_model", effective), ("prompt_sha", prompt_sha),
                         ("extra_vocab", extra_vocab), ("code_sha", code_sha),
                         ("temperature", settings["temperature"]),
                         ("reasoning_effort", settings["reasoning_effort"])):
            then = (run.get(key) or []) if key == "extra_vocab" else run.get(key)
            if then != now:
                raise SystemExit(
                    f"variant {variant!r} was run with {key}={run.get(key)!r}, "
                    f"now {now!r}. Use a new --variant name, or --force to "
                    "discard the cached run.")
    else:
        run = {"variant": variant, "model": nominal, "effective_model": effective,
               "prompt_file": prompt_file, "prompt_sha": prompt_sha,
               "prompt_source": "file" if prompt_text is not None else "production",
               "code_sha": code_sha, **settings,
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

    from photosearch import model_eval
    loaded_start = model_eval.lmstudio_loaded()
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
    # Speed depends on what else is loaded (gemma-4-26b: 0.45 s alone, 3.1 s
    # beside qwen); record it so the model-eval summary can say which.
    loaded_end = model_eval.lmstudio_loaded()
    if loaded_start is not None or loaded_end is not None or "latency_label" not in run:
        run["latency_label"] = model_eval.latency_label(effective, loaded_start, loaded_end)
        save_run(variant, run)
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


#: Macro-F1 averages only the tags with at least this many labelled positives
#: in the scored set; below it one tag's F1 is one or two photos' worth of
#: noise. Tags under it are listed as "unscored (n<5)", never silently dropped.
MACRO_MIN_POS = 5
#: Labeller self-agreement (blind recheck) below this kappa = the tag is not
#: defined tightly enough to score a model on — same threshold `agreement` uses.
KAPPA_MIN = 0.4
#: Paired bootstrap: resample PHOTOS (a photo's tags are not independent
#: draws), fixed seed so a re-run of the report prints the same interval.
BOOT_N = 2000
BOOT_SEED = 0
BASELINE_PREFIX = "baseline:"
BASELINE_TOP_K = (1, 2, 3)


def micro_f1(c):
    d = 2 * c["tp"] + c["fp"] + c["fn"]
    return 2 * c["tp"] / d if d else None


def label_counts(labels, vocab=PERCEIVED_VOCABULARY):
    """{tag: {"pos", "neg", "debatable"}} over `labels` (done photos only)."""
    out = {}
    for t in vocab:
        pos = sum(1 for lab in labels.values() if t in lab["yes"])
        deb = sum(1 for lab in labels.values()
                  if t in lab["debatable"] and t not in lab["yes"])
        out[t] = {"pos": pos, "debatable": deb, "neg": len(labels) - pos - deb}
    return out


def macro_f1(per_tag, counts, min_pos=MACRO_MIN_POS):
    """Mean per-tag F1 over the perceived tags with >= `min_pos` positives."""
    f = [micro_f1(per_tag[t]) or 0.0 for t, c in counts.items()
         if c["pos"] >= min_pos and t in per_tag]
    return sum(f) / len(f) if f else None


def baseline_predictions(labels):
    """{name: {photo_id: tags}} — answers that look at no pixels at all.

    `topK` is the K most frequent labelled tags on every photo. The frequency
    is read off THESE labels, so it is the best a constant answer of that size
    could do here — an upper bound for "the model learnt the label prior".
    `recite-all` is the whole perceived vocabulary (production's guard would
    reject it; as a score it is the recall ceiling), `empty` answers "none"."""
    counts = label_counts(labels)
    by_freq = sorted((t for t in PERCEIVED_VOCABULARY if counts[t]["pos"]),
                     key=lambda t: (-counts[t]["pos"], t))
    out = {}
    for k in BASELINE_TOP_K:
        if len(by_freq) >= k:
            out[f"{BASELINE_PREFIX}top{k}"] = {pid: list(by_freq[:k]) for pid in labels}
    out[f"{BASELINE_PREFIX}recite-all"] = {pid: list(PERCEIVED_VOCABULARY) for pid in labels}
    out[f"{BASELINE_PREFIX}empty"] = {pid: [] for pid in labels}
    return out


def photo_counts(predicted, labels):
    """{photo_id: (tp, fp, fn)} over the PERCEIVED vocabulary — the unit the
    bootstrap resamples. Candidate tags are left out so a variant that offered
    extra tags is compared with production on the same tags."""
    out = {}
    for pid, lab in labels.items():
        if pid in predicted:
            o = store.score({pid: predicted[pid]}, {pid: lab})["overall"]
            out[pid] = (o["tp"], o["fp"], o["fn"])
    return out


def _f1_rows(rows):
    tp = sum(r[0] for r in rows)
    d = 2 * tp + sum(r[1] for r in rows) + sum(r[2] for r in rows)
    return 2 * tp / d if d else 0.0


def paired_bootstrap(counts_a, counts_b, n_boot=BOOT_N, seed=BOOT_SEED):
    """micro-F1(a) − micro-F1(b) on the photos both scored, with a percentile
    95% CI from resampling photos (both variants get the SAME resample)."""
    ids = sorted(set(counts_a) & set(counts_b))
    if not ids:
        return None
    a = [counts_a[i] for i in ids]
    b = [counts_b[i] for i in ids]
    rng = random.Random(seed)
    n = len(ids)
    ds = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ds.append(_f1_rows([a[i] for i in idx]) - _f1_rows([b[i] for i in idx]))
    ds.sort()
    return {"delta": _f1_rows(a) - _f1_rows(b), "lo": ds[int(0.025 * n_boot)],
            "hi": ds[min(n_boot - 1, int(0.975 * n_boot))], "n": n}


def verdict(ci):
    """beats / worse / tie, from where the CI sits relative to 0."""
    if ci is None:
        return "—"
    if ci["lo"] > 0:
        return "beats"
    if ci["hi"] < 0:
        return "worse"
    return "tie (CI spans 0)"


def label_reliability():
    """({tag: kappa-or-None}, photos rechecked) from the blind recheck."""
    try:
        res = store.self_agreement(("recheck",))
    except (OSError, ValueError, KeyError):
        return {}, 0
    return {t: c["kappa"] for t, c in res["per_tag"].items()}, res["photos"]


def _ts(raw):
    try:
        return datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None


def recheck_contamination(main=None, recheck=None):
    """Was a MAIN label re-saved after its blind recheck? Then main-vs-recheck
    agreement no longer measures a first answer against a blind second one.

    {"rechecked", "resaved": [photo ids], "main_span_min": minutes between the
    first and last main save (a few minutes for all of them = a bulk re-save,
    whose content may or may not have changed)}."""
    main = store.scoreable_labels("main") if main is None else main
    recheck = store.scoreable_labels("recheck") if recheck is None else recheck
    both = sorted(set(main) & set(recheck))
    resaved = []
    for pid in both:
        a, b = _ts(main[pid].get("updated_at")), _ts(recheck[pid].get("updated_at"))
        if a and b and a > b:
            resaved.append(pid)
    stamps = [t for t in (_ts(v.get("updated_at")) for v in main.values()) if t]
    span = (max(stamps) - min(stamps)).total_seconds() / 60 if stamps else None
    return {"rechecked": len(both), "resaved": resaved, "main_span_min": span}


def contamination_note(c):
    if not c["resaved"]:
        return None
    span = "" if c["main_span_min"] is None else (
        f" All main labels were saved within {c['main_span_min']:.0f} min of each "
        "other — a bulk re-save; compare against a backup of labels.json to see "
        "whether their content changed.")
    return (f"WARNING: {len(c['resaved'])} of {c['rechecked']} recheck photos had their "
            "MAIN label re-saved AFTER the blind recheck, so the self-agreement "
            "(kappa) below may be contaminated: the first answer may have been "
            f"edited with the recheck in view.{span}")


def _summarise(predicted, labels, sample, extra=()):
    """score() plus the things it does not carry: tags/photo, strata, and the
    perceived-only F1s. `extra` = the candidate tags this variant offered
    (scored for it alone in P/R; F1 is perceived-only so rows compare)."""
    res = store.score(predicted, labels, extra_tags=extra)
    allowed = _PERCEIVED | set(extra)
    answered = [set(t) & allowed for pid, t in predicted.items()
                if pid in labels and t is not None]
    res["avg_tags"] = (sum(len(t) for t in answered) / len(answered)) if answered else None
    perceived = res if not extra else store.score(predicted, labels)
    res["perceived"] = perceived["overall"]
    res["f1"] = micro_f1(perceived["overall"])
    res["macro_f1"] = macro_f1(perceived["per_tag"], label_counts(labels))
    res["perceived_per_tag"] = perceived["per_tag"]
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


def build_report(variant_predictions, labels=None, sample=None, extras=None, *,
                 reference=None, baselines=True, kappa=None, meta=None,
                 coverage=None, n_boot=BOOT_N, seed=BOOT_SEED):
    """{"variants": [names], "results": {name: summary}, "labelled": n, ...}.

    Baseline rows (`baseline:*`) are scored on the same labels and kept out
    of `variants`. Every variant is compared, by paired bootstrap, with
    `reference` (when given and present) and with the best baseline."""
    labels = store.scoreable_labels() if labels is None else labels
    sample = store.load_sample() if sample is None else sample
    extras = extras or {}
    results = {name: _summarise(pred, labels, sample, extras.get(name, ()))
               for name, pred in variant_predictions.items()}
    base = baseline_predictions(labels) if baselines and labels else {}
    for name, pred in base.items():
        results[name] = _summarise(pred, labels, sample)
    best = max(base, key=lambda n: (results[n]["f1"] or 0.0, n)) if base else None
    counts = {name: photo_counts(pred, labels)
              for name, pred in list(variant_predictions.items()) + list(base.items())}
    if reference not in variant_predictions:
        reference = None
    comparisons = {}
    for name in variant_predictions:
        comparisons[name] = {
            "vs_reference": (paired_bootstrap(counts[name], counts[reference], n_boot, seed)
                             if reference and name != reference else None),
            "vs_best_baseline": (paired_bootstrap(counts[name], counts[best], n_boot, seed)
                                 if best else None)}
    lc = label_counts(labels)
    model_runs = [n for n in variant_predictions if n != STORED]
    fp_runs = {t: sum(results[n]["perceived_per_tag"][t]["fp"] for n in model_runs)
               for t in PERCEIVED_VOCABULARY}
    return {"variants": list(variant_predictions), "labelled": len(labels),
            "sampled": len(sample["photos"]), "results": results,
            "baselines": list(base), "baseline_tags": {n: next(iter(p.values()), [])
                                                      for n, p in base.items()},
            "best_baseline": best, "reference": reference,
            "comparisons": comparisons, "label_counts": lc,
            "kappa": kappa or {}, "fp_across_runs": fp_runs, "model_runs": model_runs,
            "pair_both": {f"{a}/{b}": sum(1 for lab in labels.values()
                                          if a in lab["yes"] and b in lab["yes"])
                          for a, b in CONTRADICTORY_PAIRS},
            "meta": meta or {}, "coverage": coverage or {}, "notes": [],
            "n_boot": n_boot, "seed": seed}


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


def _f(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


def _fmt_ci(ci):
    if ci is None:
        return "—"
    return f"{ci['delta']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]"


def _kappa_cell(report, tag):
    if not report["kappa"]:
        return "—"
    k = report["kappa"].get(tag)
    if k is None:
        return "— (no recheck call)"
    return f"{k:.2f}" + (" unreliable label" if k < KAPPA_MIN else "")


def _tag_order(report):
    """Tags by total false positives across variants (the failure this eval
    exists to find), dropping tags nobody predicted or labelled. Baselines
    are left out: recite-all would put every tag on the list."""
    res = [report["results"][n] for n in report["variants"]]
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

    if report["meta"]:
        rows = []
        for n in names:
            m = report["meta"].get(n)
            if not m:
                continue
            rows.append([n, m.get("model") or "—", m["prompt"], m["code"],
                         m.get("temperature", "—"), m.get("reasoning_effort") or "—",
                         report["coverage"].get(n, "—")])
        tables.append(("Runs (STALE = the run cannot show it matches today's "
                       "prompt / guard+parser)",
                       ["variant", "model", "prompt", "guard/parser", "temperature",
                        "reasoning effort", "cached of labelled"], rows))

    rows = []
    for n in names + report["baselines"]:
        r = res[n]
        avg = "—" if r["avg_tags"] is None else f"{r['avg_tags']:.2f}"
        rows.append([n, r["photos_scored"], r["unanswered"], avg,
                     _precision(r["overall"]), _recall(r["overall"]),
                     r["overall"]["fp"], r["overall"]["fn"], r["overall"]["debatable"],
                     _f(r["f1"]), _f(r["macro_f1"])])
    tables.append(("Overall", ["variant", "photos", "unanswered", "tags/photo",
                               "precision", "recall", "fp", "fn", "debatable",
                               "micro-F1", f"macro-F1 (pos>={MACRO_MIN_POS})"], rows))

    if report["baselines"]:
        ref, best = report["reference"], report["best_baseline"]
        rows = []
        for n in names:
            c = report["comparisons"][n]
            v = verdict(c["vs_best_baseline"])
            rows.append([n, res[n]["photos_scored"], _f(res[n]["f1"]),
                         _fmt_ci(c["vs_reference"]) if ref else "—",
                         _fmt_ci(c["vs_best_baseline"]),
                         v if v == "beats" else f"DOES NOT BEAT BASELINE: {v}"])
        tables.append((f"micro-F1 vs reference `{ref or 'none'}` and vs best baseline "
                       f"`{best}` ({', '.join(report['baseline_tags'][best]) or 'none'}) — "
                       f"paired bootstrap over photos, {report['n_boot']} resamples, "
                       f"seed {report['seed']}, 95% CI",
                       ["variant", "photos", "micro-F1", "ΔF1 vs reference [95% CI]",
                        "ΔF1 vs best baseline [95% CI]", "verdict"], rows))

    lc = report["label_counts"]
    head = ["tag", "pos/neg", "kappa"]
    for n in names:
        head += [f"{n}: tp/fp/fn/deb", "precision", "recall"]
    rows = []
    for t in _tag_order(report):
        c0 = lc.get(t)
        row = [t, f"{c0['pos']}/{c0['neg']}" if c0 else "—", _kappa_cell(report, t)]
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

    rows = []
    for t in sorted(PERCEIVED_VOCABULARY, key=lambda t: (-lc[t]["pos"], t)):
        c = lc[t]
        macro = "scored" if c["pos"] >= MACRO_MIN_POS else f"unscored (n<{MACRO_MIN_POS})"
        rows.append([t, c["pos"], c["neg"], c["debatable"], macro,
                     _kappa_cell(report, t), report["fp_across_runs"][t]])
    tables.append((f"Spec coverage — labelled positives per perceived tag; FP summed "
                   f"over {len(report['model_runs'])} model run(s)",
                   ["tag", "positives", "negatives", "debatable", "macro-F1",
                    "kappa", "FP across runs"], rows))

    rows = []
    for a, b in CONTRADICTORY_PAIRS:
        rows.append([f"{a} / {b}", lc[a]["pos"], lc[b]["pos"],
                     "yes" if lc[a]["pos"] or lc[b]["pos"] else "NO — untested by labels",
                     report["pair_both"].get(f"{a}/{b}", 0),
                     report["fp_across_runs"][a] + report["fp_across_runs"][b]])
    tables.append(("CONTRADICTORY_PAIRS coverage (a photo labelled yes on BOTH "
                   "members means the pair is not exclusive)",
                   ["pair", "pos a", "pos b", "labelled member?", "labelled both",
                    "FP across runs"], rows))

    spec = report.get("spec")
    if spec:
        tables.append(spec_table(spec))
    return tables


def report_notes(report):
    """Lines that must be read before any number: label coverage, the macro
    rule, zero-positive tags, and every warning collected on the way."""
    lc = report["label_counts"]
    zero = [t for t in PERCEIVED_VOCABULARY if lc[t]["pos"] == 0]
    out = list(report.get("notes") or [])
    out.append(f"micro-F1 / macro-F1 / CIs are over the PERCEIVED vocabulary only "
               f"(candidate tags excluded) so every row is on the same tags. "
               f"macro-F1 averages the {sum(1 for t in lc if lc[t]['pos'] >= MACRO_MIN_POS)} "
               f"tags with >= {MACRO_MIN_POS} labelled positives.")
    if zero:
        out.append("Zero-positive tags (recall undefined; every prediction is a "
                   "measurable false positive): " + ", ".join(
                       f"{t} (FP {report['fp_across_runs'][t]})" for t in zero))
    unreliable = sorted(t for t, k in report["kappa"].items()
                        if k is not None and k < KAPPA_MIN and t in _PERCEIVED)
    if unreliable:
        out.append(f"Unreliable labels (recheck kappa < {KAPPA_MIN}): "
                   + ", ".join(unreliable) + " — their per-tag scores measure the "
                   "labeller as much as the model.")
    stale = [n for n, m in report["meta"].items() if m.get("stale")]
    if stale:
        out.append(f"STALE runs ({len(stale)}): {', '.join(stale)} — re-run them "
                   "before quoting their numbers as today's code.")
    losers = [n for n in report["variants"]
              if verdict(report["comparisons"][n]["vs_best_baseline"]) != "beats"]
    if report["baselines"] and losers:
        out.append(f"Do NOT beat the best constant baseline `{report['best_baseline']}` "
                   f"(95% CI includes 0 or below): {', '.join(losers)}")
    return out


def _text_table(title, head, rows):
    table = [[str(c) for c in r] for r in [head] + rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(head))]
    out = [f"\n=== {title} ==="]
    for r in table:
        out.append("  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                             for i, (c, w) in enumerate(zip(r, widths))).rstrip())
    return out


def render_text(report):
    out = [f"{report['labelled']} labelled (done) of {report['sampled']} sampled. "
           f"Ratios are shown with their counts; below {MIN_N} they are withheld. "
           "Debatable tags count as neither hit nor miss."]
    out += [f"* {line}" for line in report_notes(report)]
    for title, head, rows in report_tables(report):
        out += _text_table(title, head, rows)
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
    notes = "".join(f"<li>{html.escape(line)}</li>" for line in report_notes(report))
    return f"""<!DOCTYPE html><html><head><meta charset=utf-8>
<title>Visual-tag eval</title><style>
 body{{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
 h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}}
 table{{border-collapse:collapse;margin:12px 0}}
 th,td{{border:1px solid #333;padding:5px 10px;text-align:right}}
 td:first-child,th:first-child{{text-align:left}}
 td.low{{color:#777}} ul.notes{{color:#fc9;max-width:80em}}
</style></head><body>
<h1>Visual-tag eval — {report['labelled']} labelled of {report['sampled']} sampled</h1>
<p style="color:#999">Ratios carry their counts; below {MIN_N} observations they
are withheld. Debatable tags count as neither hit nor miss.</p>
<ul class=notes>{notes}</ul>
{''.join(parts)}
</body></html>"""


# --------------------------------------------------------------------------
# Guard / parser spec — the production code, exercised without a model
# --------------------------------------------------------------------------
#
# Re-parsing the cached first answers with no guard, or with a naive comma
# split, moved micro-F1 by <= 0.006: the model answers in clean comma lists and
# rarely trips the guard, so the scored runs cannot see the guard or the
# tolerant parser at all. This section can. Seeds are the clean answers the
# cached runs hold (their stored `tags`, NOT a re-parse — a broken parser must
# not shrink its own test set); each is reformatted, contradicted and padded,
# then fed to the REAL `_parse_visual_response`, `_visual_answer_problem` and
# `tag_visual_photo` (chat scripted in-process — no model, no network). The
# expected outcomes are computed here from the rules (CONTRADICTORY_PAIRS,
# `_VISUAL_MAX_PLAUSIBLE_TAGS`), never from the functions under test.

SPEC_FORMATS = {
    "comma": lambda tags: ", ".join(tags),
    "newlines": lambda tags: "\n".join(tags),
    "semicolons": lambda tags: "; ".join(tags),
    "bullets": lambda tags: "\n".join(f"- {t}" for t in tags),
    "numbered": lambda tags: "\n".join(f"{i}. {t}" for i, t in enumerate(tags, 1)),
    "label-prefix": lambda tags: "Tags: " + ", ".join(tags),
    "label-lines": lambda tags: "\n".join(f"Tag {i}: {t}" for i, t in enumerate(tags, 1)),
    "json": lambda tags: json.dumps(tags),
    "fenced": lambda tags: "```\n" + ", ".join(tags) + "\n```",
}


def _pairs_in(tags):
    have = set(tags)
    return [(a, b) for a, b in CONTRADICTORY_PAIRS if a in have and b in have]


def spec_seeds(runs, limit=None):
    """Distinct clean cached answers (perceived tags only): fewer than
    `limit` tags and no contradictory pair."""
    from photosearch import describe
    limit = describe._VISUAL_MAX_PLAUSIBLE_TAGS if limit is None else limit
    seen = {}
    for name, run in runs.items():
        for p in run.get("predictions", {}).values():
            tags = [t for t in (p.get("tags") or []) if t in _PERCEIVED]
            if tags and len(tags) < limit and not _pairs_in(tags):
                seen.setdefault(tuple(tags), name)
    return [list(k) for k in seen]


def _contradict(tags):
    """`tags` + the partner of one member (or a whole pair), and the tags the
    guard must then drop: both members of EVERY pair now present."""
    for a, b in CONTRADICTORY_PAIRS:
        if (a in tags) != (b in tags):
            mut = tags + [b if a in tags else a]
            break
    else:
        a, b = CONTRADICTORY_PAIRS[0]
        mut = tags + [a, b]
    doomed = {t for pair in _pairs_in(mut) for t in pair}
    return mut, [t for t in mut if t not in doomed]


def _pad(tags, size):
    """`tags` padded with vocabulary terms (none contradicting) to `size`."""
    out = list(tags)
    for t in PERCEIVED_VOCABULARY:
        if len(out) >= size:
            break
        if t not in out and not _pairs_in(out + [t]):
            out.append(t)
    return out if len(out) == size else None


@contextmanager
def scripted_chat(answers):
    """Run the REAL `tag_visual_photo` against a scripted chat: each call
    pops the next answer. Yields the list of calls made. In-process only."""
    from photosearch import describe
    queue, calls = list(answers), []

    def chat(*a, **kw):
        calls.append((kw.get("options") or {}).get("temperature"))
        if not queue:
            raise RuntimeError("tag_visual_photo asked more often than scripted")
        return queue.pop(0)

    saved = describe._ollama_chat_with_retry, describe.HAS_OLLAMA
    describe._ollama_chat_with_retry, describe.HAS_OLLAMA = chat, True
    try:
        yield calls
    finally:
        describe._ollama_chat_with_retry, describe.HAS_OLLAMA = saved


def _flow(image, answers):
    """(result, calls) of the real `tag_visual_photo` on scripted answers."""
    from photosearch import describe
    with scripted_chat(answers) as calls:
        got = describe.tag_visual_photo(image, model="spec")
    return got, list(calls)


def _same(got, want):
    if got is None or want is None:
        return got is want
    return sorted(got) == sorted(want)


def guard_parser_spec(runs, image_path=None):
    """{"seeds", "checks": {name: {"cases", "pass", "fail"}}, "replay", "ok"}."""
    from photosearch import describe
    vocab = set(PERCEIVED_VOCABULARY)
    limit = describe._VISUAL_MAX_PLAUSIBLE_TAGS
    seeds = spec_seeds(runs, limit)
    checks = {}

    def record(name, ok, example):
        c = checks.setdefault(name, {"cases": 0, "pass": 0, "fail": None})
        c["cases"] += 1
        if ok:
            c["pass"] += 1
        elif c["fail"] is None:
            c["fail"] = example

    comma = SPEC_FORMATS["comma"]
    with tempfile.TemporaryDirectory() as tmp:
        image = image_path
        if image is None:
            from PIL import Image
            image = os.path.join(tmp, "spec.jpg")
            Image.new("RGB", (32, 24), (90, 140, 200)).save(image, "JPEG")
        for s in seeds:
            for fmt, render in SPEC_FORMATS.items():
                raw = render(s)
                got = describe._parse_visual_response(raw, vocab)
                record(f"parse: {fmt}", got == s, f"{raw!r} -> {got}")
            record("guard: accepts a clean answer", describe._visual_answer_problem(s) is None,
                   f"{s} -> {describe._visual_answer_problem(s)}")
            got, calls = _flow(image, [comma(s)])
            record("flow: clean answer, no retry", _same(got, s) and len(calls) == 1,
                   f"{s} -> {got} in {len(calls)} call(s)")

            mut, kept = _contradict(s)
            if len(mut) < limit:
                prob = describe._visual_answer_problem(mut)
                record("guard: flags a contradictory pair",
                       prob == describe._CONTRADICTION, f"{mut} -> {prob}")
                want = kept or None
                got, calls = _flow(image, [comma(mut), comma(mut)])
                record("flow: contradiction twice -> drop both members",
                       _same(got, want) and len(calls) == 2,
                       f"{mut} -> {got} (want {want}) in {len(calls)} call(s)")
                got, calls = _flow(image, [comma(mut), comma(s)])
                record("flow: contradiction then clean retry -> retry",
                       _same(got, s) and len(calls) == 2, f"{mut} then {s} -> {got}")
                if calls and len(calls) == 2:
                    record("flow: first call at temperature 0", calls[0] == 0,
                           f"temperature {calls[0]}")

            padded = _pad(s, limit)
            if padded:
                prob = describe._visual_answer_problem(padded)
                record(f"guard: rejects {limit} tags (over-selection)",
                       prob == describe._OVER_SELECTION, f"{padded} -> {prob}")
                got, calls = _flow(image, [comma(padded), comma(padded)])
                record("flow: over-selection twice -> no answer (None)",
                       got is None and len(calls) == 2, f"{padded} -> {got}")
                got, calls = _flow(image, [comma(padded), comma(s)])
                record("flow: over-selection then clean retry -> retry",
                       _same(got, s), f"{padded} then {s} -> {got}")
                prob = describe._visual_answer_problem(padded[:-1])
                record(f"guard: accepts {limit - 1} tags", prob is None,
                       f"{padded[:-1]} -> {prob}")

        everything = comma(list(PERCEIVED_VOCABULARY))
        got, _ = _flow(image, [everything, everything])
        record("flow: recite-all -> no answer (None)", got is None, f"-> {got}")
        got, _ = _flow(image, ["none"])
        record("flow: explicit 'none' -> [] (answered, no tags)", got == [], f"-> {got}")
        got, _ = _flow(image, ["I can't tell from this image.", "Sorry, unclear."])
        record("flow: prose twice -> no answer (None)", got is None, f"-> {got}")

    # Informational: would today's parser read the cached first answers the
    # way they were stored? (A STALE run's numbers drift by this much.)
    replay = {"answers": 0, "differ": 0, "example": None}
    for name, run in runs.items():
        allow = vocab | set(run.get("extra_vocab") or [])
        for p in run.get("predictions", {}).values():
            calls = p.get("calls") or []
            raw = calls[0].get("raw") if len(calls) == 1 else None
            if not raw or p.get("tags") is None or describe._is_explicit_empty_answer(raw):
                continue
            replay["answers"] += 1
            now = describe._parse_visual_response(raw, allow)
            if sorted(now) != sorted(p["tags"]):
                replay["differ"] += 1
                replay["example"] = replay["example"] or f"{name}: {raw[:60]!r} -> {now} (cached {p['tags']})"
    ok = bool(checks) and bool(seeds) and all(c["pass"] == c["cases"] for c in checks.values())
    return {"seeds": len(seeds), "checks": checks, "replay": replay, "ok": ok}


def spec_table(spec):
    rows = [[name, c["cases"], c["pass"], "PASS" if c["pass"] == c["cases"] else "FAIL",
             c["fail"] or ""] for name, c in spec["checks"].items()]
    r = spec["replay"]
    rows.append(["replay: cached single-call answers re-parse as stored (info)",
                 r["answers"], r["answers"] - r["differ"], "info", r["example"] or ""])
    return (f"Guard / parser spec — {spec['seeds']} distinct cached answers mutated and "
            f"fed to the production code, no model "
            f"({'ALL PASS' if spec['ok'] else 'FAILURES'})",
            ["check", "cases", "pass", "result", "first failure"], rows)


def cmd_spec(args):
    names = [v.strip() for v in args.variants.split(",") if v.strip()] \
        if args.variants else list_variants()
    runs = {n: load_run(n) for n in names}
    runs = {n: r for n, r in runs.items() if r is not None}
    spec = guard_parser_spec(runs)
    print("\n".join(_text_table(*spec_table(spec))))
    if not spec["ok"]:
        raise SystemExit(1)


def pick_reference(runs, live_prompt):
    """(name, why) of the run to compare every variant with: `production` if
    cached, else the newest run of the shipped prompt on the fleet's visual
    model (rerun.FLEET_ROLE_MODELS), else None."""
    if DEFAULT_VARIANT in runs:
        return DEFAULT_VARIANT, "named `production`"
    try:
        from photosearch.rerun import FLEET_ROLE_MODELS
        fleet = FLEET_ROLE_MODELS.get("visual")
    except Exception:
        fleet = None
    cands = [n for n, r in runs.items()
             if r.get("prompt_sha") == live_prompt and not r.get("extra_vocab")
             and (fleet is None or r.get("effective_model") == fleet)]
    if not cands:
        return None, "no cached run of the shipped prompt on the fleet model"
    name = max(cands, key=lambda n: (runs[n].get("created") or "", n))
    return name, f"shipped prompt on the fleet model {fleet}"


def _run_meta(run, live_prompt, live_code):
    m = run_staleness(run, live_prompt, live_code)
    temp = run.get("temperature")
    if temp is None:
        seen = {c.get("temperature") for p in run.get("predictions", {}).values()
                for c in (p.get("calls") or [])[:1]}
        temp = (f"{seen.pop()} (first calls)" if len(seen) == 1 else
                "not recorded")
    effort = run.get("reasoning_effort")
    if "reasoning_effort" not in run:
        effort = "not recorded"
    elif effort and run.get("reasoning_effort_sent") is False:
        effort = f"{effort} (not sent: no reasoning mode)"
    return {**m, "model": run.get("effective_model"), "temperature": temp,
            "reasoning_effort": effort}


def cmd_report(args):
    labels = store.scoreable_labels()
    if not labels:
        raise SystemExit("No `done` labels yet — label some photos on "
                         "/eval/visual-tags first.")
    wanted = [v.strip() for v in args.variants.split(",") if v.strip()] \
        if args.variants else [STORED] + list_variants()
    preds, extras, runs = {}, {}, {}
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
        runs[name] = run
        preds[name] = predictions_of(run)
        extras[name] = run.get("extra_vocab") or []
        print(f"[report] {name}: model={run.get('effective_model')} "
              f"prompt={run.get('prompt_file') or 'production'} "
              f"({len(run['predictions'])} cached)")
    if not preds:
        raise SystemExit("Nothing to report — no runs cached and no --db.")

    notes = []
    coverage = {n: sum(1 for pid in labels if pid in p) for n, p in preds.items()}
    short = {n: c for n, c in coverage.items() if c < len(labels)}
    scored = labels
    if short and not args.all_photos:
        common = set(labels)
        for p in preds.values():
            common &= set(p)
        scored = {pid: labels[pid] for pid in sorted(common)}
        notes.append(f"WARNING: incomplete run(s) {short} of {len(labels)} labelled — "
                     f"EVERY row is scored on the {len(scored)} photos all variants "
                     "cover. --all-photos scores each on its own set (not comparable).")
    elif short:
        notes.append(f"WARNING: --all-photos with incomplete run(s) {short}: rows are "
                     "scored on DIFFERENT photo sets and do not compare.")

    live_prompt, live_code = production_prompt_sha(), guard_parser_sha()
    meta = {n: _run_meta(r, live_prompt, live_code) for n, r in runs.items()}
    reference, why = (args.reference, "--reference") if args.reference \
        else pick_reference(runs, live_prompt)
    if reference is not None and reference not in preds:
        raise SystemExit(f"--reference {reference!r} is not among the reported variants.")
    notes.append(f"Reference for ΔF1: {reference or 'none'} ({why}).")

    kappa, rechecked = label_reliability()
    if rechecked:
        cont = contamination_note(recheck_contamination())
        if cont:
            notes.append(cont)
    else:
        notes.append("No blind recheck labelled: per-tag label reliability (kappa) unknown.")

    report = build_report(preds, scored, extras=extras, reference=reference,
                          kappa=kappa, meta=meta, coverage=coverage)
    report["notes"] = notes
    if runs and not args.no_spec:
        report["spec"] = guard_parser_spec(runs)
    print(render_text(report))
    if args.html:
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(render_html(report))
        print(f"\n[report] {args.html}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_agreement(args):
    """Labeller self-consistency: first answer vs the blind recheck, per tag."""
    which = getattr(args, "agreement_set", "visual")
    res = store.self_agreement(AGREEMENT_SETS[which])
    if not res["photos"]:
        page = {"visual": "?set=recheck", "sharpness": "?set=sharpness-recheck",
                "all": "?set=recheck and ?set=sharpness-recheck"}[which]
        raise SystemExit("No photos labelled done in BOTH sets — label the "
                         f"recheck subset at /eval/visual-tags{page} first.")
    print(f"{res['photos']} photos labelled twice. Agreement is on yes/no calls; "
          f"a debatable on either side is set aside. kappa < 0.4 = the tag is "
          f"not tightly defined enough to score a model on.")
    rows = []
    for t, c in sorted(res["per_tag"].items(),
                       key=lambda kv: (kv[1]["kappa"] is None, kv[1]["kappa"] or 0)):
        k = "—" if c["kappa"] is None else f"{c['kappa']:.2f}"
        rows.append([t, str(c["n"]), fmt_ratio(c["agree_yes"] + c["agree_no"], c["n"]),
                     str(c["agree_yes"]), str(c["yes_then_no"]), str(c["no_then_yes"]),
                     str(c["debatable"]), k])
    head = ["tag", "n", "agreement", "yes both", "yes→no", "no→yes", "debatable", "kappa"]
    widths = [max(len(r[i]) for r in [head] + rows) for i in range(len(head))]
    for r in [head] + rows:
        print("  ".join(x.ljust(w) for x, w in zip(r, widths)))


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

    ss = sub.add_parser("sample-sharpness",
                        help="Choose the blurry-weighted sharpness sample (once), "
                             "beside the visual one.")
    ss.add_argument("--db", default=os.environ.get("PHOTOSEARCH_DB"),
                    help="photo_index.db — opened READ-ONLY.")
    ss.add_argument("--n", type=int, default=60)
    ss.add_argument("--seed", type=int, default=1)
    ss.add_argument("--force", action="store_true",
                    help="Overwrite an existing sharpness sample (orphans its labels).")
    ss.add_argument("--extend", type=int, default=0, metavar="N",
                    help="APPEND N photos from the blurry-rich strata "
                         "(EXTEND_STRATA); existing labels stay valid.")
    ss.add_argument("--extend-from", default=",".join(EXTEND_STRATA),
                    help="comma-separated strata --extend draws from "
                         "(default: %(default)s)")
    ss.set_defaults(func=cmd_sample_sharpness)

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
    tp.add_argument("--reference", help="Variant every other is compared with "
                    "(ΔF1 CI). Default: `production` if cached, else the newest "
                    "run of the shipped prompt on the fleet's visual model.")
    tp.add_argument("--all-photos", action="store_true",
                    help="Score each variant on every photo it covers instead of "
                         "the set ALL variants cover (rows then do not compare).")
    tp.add_argument("--no-spec", action="store_true",
                    help="Skip the guard/parser spec section.")
    tp.set_defaults(func=cmd_report)

    sp = sub.add_parser("spec", help="Exercise the production guard + parser on "
                        "mutated cached answers (no model). Exit 1 on a failure.")
    sp.add_argument("--variants", help="Comma-separated runs to seed from "
                                       "(default: every cached run).")
    sp.set_defaults(func=cmd_spec)
    _add_agreement(sub)
    return ap


AGREEMENT_SETS = {"visual": ("recheck",), "sharpness": ("sharpness-recheck",),
                  "all": ("recheck", "sharpness-recheck")}


def _add_agreement(sub):
    ap = sub.add_parser("agreement", help="Labeller self-consistency (first "
                        "answer vs the blind recheck set).")
    ap.add_argument("--set", dest="agreement_set", choices=sorted(AGREEMENT_SETS),
                    default="visual",
                    help="visual = main vs recheck (default); sharpness = the "
                         "sharpness sample vs its recheck; all = both pooled "
                         "(the one to read for `blurry`).")
    ap.set_defaults(func=cmd_agreement)


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
