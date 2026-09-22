#!/usr/bin/env python
"""Visual-tag eval, second set — RECALL against Unsplash PHOTOGRAPHER keywords.

The owner-labelled eval (`evals/visual_tags_eval.py`) is the one that settles
precision, and it has 60 photos. This set is the other axis: the Unsplash
Research Dataset Lite carries keywords the photographers themselves attached
to 25k photos, and a photographer who wrote "golden hour" or "silhouette" on
their own upload is a positive label at a scale 60 hand labels cannot reach.

What it can and cannot say — read this before quoting a number:

  * Keywords are POSITIVE-ONLY. A photo without "overcast" is not a "no"; the
    photographer just did not write it. So this set reports per-tag RECALL and
    a CONTRADICTION check (the model said `sunny` on a photographer-keyworded
    `overcast`), never precision.
  * Pixels are the 400-px thumbs already on disk (`THUMBS`), not the fleet's
    1024-px re-encode of an original. That keeps the run server-free and
    download-free, and it means absolute numbers are NOT comparable to the
    owner-set numbers — compare variants against each other, on this set.
  * The keyword -> tag MAPPING below is a judgement call (edit it).

Three subcommands mirror the owner harness, plus a contact sheet:

  sample   draw up to --per-tag photos per tag (seeded, per-tag RNG) whose
           human keywords map to it; only photos with a thumb.
               python evals/visual_tags_unsplash.py sample --per-tag 40
  run      tag every sampled photo with the PRODUCTION tagger (same
           `describe.tag_visual_photo`, parser, guard, retry as the fleet),
           cached and resumable under `unsplash/runs/<variant>.json`.
               export PHOTOSEARCH_TEXT_LLM_URL=http://localhost:1234/v1
               python evals/visual_tags_unsplash.py run --variant production
  report   per-tag recall + contradictions, variants side by side.
  sheet    an HTML contact sheet for one tag — what photographers mean by it.
               python evals/visual_tags_unsplash.py sheet --tag moody

Storage is `visual_tag_eval.eval_dir()/unsplash/` (git-ignored with the rest
of `evals/visual-tags/`). The dataset and the thumbs are READ-ONLY inputs:
nothing here writes, copies or downloads under either.
"""
import argparse
import csv
import hashlib
import html
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
# evals/ is a directory of scripts, not a package; the owner harness is
# imported as a module so its tagger plumbing is reused, not re-implemented.
sys.path.insert(0, HERE)

import visual_tags_eval as owner  # noqa: E402
from photosearch import visual_tag_eval as store  # noqa: E402
from photosearch.visual_tags_derive import (  # noqa: E402
    CONTRADICTORY_PAIRS, PERCEIVED_VOCABULARY)

DATASET = os.environ.get(
    "PHOTOSEARCH_UNSPLASH_DATASET",
    "/mnt/c/Users/mattn/Downloads/unsplash-research-dataset-lite-latest")
THUMBS = os.environ.get(
    "PHOTOSEARCH_UNSPLASH_THUMBS",
    os.path.expanduser("~/unsplash-quality-eval/thumbs"))

# Photographer keyword -> our tag. A JUDGEMENT CALL, and the owner may edit
# it: each entry says "a photographer who wrote this keyword would accept our
# tag on the photo". Matching is case-insensitive on the whole keyword.
# Note `monochrome` maps to BOTH `monochromatic` and `black-and-white`: on
# Unsplash the word is used for either a B&W conversion or a single-hue
# colour photo, and nothing in the keyword row says which, so a photo
# keyworded `monochrome` is a positive for both and the model gets recall
# credit for either.
MAPPING = {
    "sunny": ("sunny",),
    "overcast": ("overcast",),
    "golden-hour": ("golden hour", "golden-hour"),
    "backlit": ("backlit", "backlight"),
    "silhouette": ("silhouette",),
    "harsh-light": ("harsh light", "hard light"),
    "soft-light": ("soft light",),
    "overexposed": ("overexposed",),
    "foggy": ("foggy", "fog"),
    "hazy": ("hazy", "haze"),
    "snowy": ("snowy", "snow"),
    "colorful": ("colorful", "colourful"),
    "vibrant": ("vibrant",),
    "muted": ("muted",),
    "monochromatic": ("monochromatic", "monochrome"),
    "black-and-white": ("black and white", "black & white", "monochrome"),
    "dramatic": ("dramatic",),
    "joyful": ("joyful", "joy"),
    "melancholy": ("melancholy", "melancholic"),
    "moody": ("moody",),
    "peaceful": ("peaceful", "calm", "serene"),
    "close-up": ("close up", "close-up", "closeup"),
    "macro": ("macro",),
    "wide-angle": ("wide angle", "wide-angle", "fisheye"),
    "aerial": ("aerial", "aerial view", "drone"),
    "centered": ("centered", "centred"),
    "symmetrical": ("symmetrical", "symmetry"),
    "reflection": ("reflection",),
    # Candidate tag (visual_tag_eval.CANDIDATE_TAGS): sampled and reported,
    # but scored only for a variant that offered it via --extra-vocab.
    "action": ("action",),
}

_PERCEIVED = frozenset(PERCEIVED_VOCABULARY)
_SCORABLE = _PERCEIVED | frozenset(store.CANDIDATE_TAGS)
_unknown = sorted(set(MAPPING) - _SCORABLE)
assert not _unknown, f"MAPPING names tags that are neither perceived nor candidate: {_unknown}"

# {lower-cased keyword: [our tags]} — the reverse of MAPPING, built once.
KEYWORD_TO_TAGS = {}
for _tag, _kws in MAPPING.items():
    for _kw in _kws:
        KEYWORD_TO_TAGS.setdefault(_kw.lower(), []).append(_tag)

# {tag: partners} from the production guard's own list, so a contradiction
# here is exactly what the guard would call one.
CONTRADICTS = {}
for _a, _b in CONTRADICTORY_PAIRS:
    CONTRADICTS.setdefault(_a, set()).add(_b)
    CONTRADICTS.setdefault(_b, set()).add(_a)

DEFAULT_VARIANT = "production"
MIN_N = owner.MIN_N
fmt_ratio = owner.fmt_ratio

# Unsplash descriptions run long; the default 128 KiB limit trips on them.
csv.field_size_limit(10 * 1024 * 1024)


# --------------------------------------------------------------------------
# Dataset (read-only)
# --------------------------------------------------------------------------

def _tsv(name):
    return os.path.join(DATASET, name)


def load_human_keywords():
    """{photo_id: [keyword, ...]} over rows a HUMAN added (`suggested_by_user
    == 't'`). AI-suggested keywords are the very thing under test's cousins
    and never count as a label here."""
    path = _tsv("keywords.tsv000")
    if not os.path.exists(path):
        raise SystemExit(f"Unsplash keywords file not found: {path} "
                         "(set PHOTOSEARCH_UNSPLASH_DATASET)")
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("suggested_by_user") != "t":
                continue
            kw = (row.get("keyword") or "").strip()
            if kw:
                out.setdefault(row["photo_id"], []).append(kw)
    return out


def load_photo_urls():
    """{photo_id: photo_url} for the contact sheet's links; {} if the photos
    file is absent (the sheet still renders from the thumbs)."""
    path = _tsv("photos.tsv000")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("photo_url"):
                out[row["photo_id"]] = row["photo_url"]
    return out


def thumb_path(photo_id):
    return os.path.join(THUMBS, f"{photo_id}.jpg")


def match_keywords(keywords):
    """{our tag: [matching raw keywords]} for one photo's human keywords."""
    hits = {}
    for kw in keywords:
        for tag in KEYWORD_TO_TAGS.get(kw.lower(), ()):
            hits.setdefault(tag, [])
            if kw not in hits[tag]:
                hits[tag].append(kw)
    return hits


def photos_by_tag(human_keywords):
    """{tag: {photo_id: [matching keywords]}} — every photo whose human
    keywords map to the tag, thumb or not (sampling filters on the thumb)."""
    by_tag = {tag: {} for tag in MAPPING}
    for pid, kws in human_keywords.items():
        for tag, matched in match_keywords(kws).items():
            by_tag[tag][pid] = matched
    return by_tag


# --------------------------------------------------------------------------
# Storage — eval_dir()/unsplash/
# --------------------------------------------------------------------------

def unsplash_dir():
    return store.eval_dir() / "unsplash"


def sample_path():
    return unsplash_dir() / "sample.json"


def _write_atomic(path, data):
    # Same shape as the label store's `_write_atomic` (a private, so copied
    # rather than imported): a kill mid-dump must not leave truncated JSON
    # where a sample or an hour of model calls used to be.
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


def load_sample():
    path = sample_path()
    if not path.exists():
        return {"created": None, "seed": None, "per_tag": None, "photos": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_sample(photos, seed, per_tag):
    data = {"created": datetime.now(timezone.utc).isoformat(), "seed": seed,
            "per_tag": per_tag, "photos": photos}
    _write_atomic(sample_path(), data)
    return data


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def choose_sample(by_tag, per_tag=40, seed=1, has_thumb=None):
    """({photo_id: {"tags", "keywords", "human_keywords"?}}, {tag: (available, drawn)}).

    Deterministic for (dataset, per_tag, seed). One RNG per tag, seeded
    `f"{seed}:{tag}"`, so editing one MAPPING entry re-draws that tag alone
    and every other tag's photos — and any run cached for them — stay put.
    A photo can be positive for several tags; it is drawn independently under
    each and its entry carries the union.
    """
    has_thumb = has_thumb or (lambda pid: os.path.exists(thumb_path(pid)))
    photos, counts = {}, {}
    for tag in MAPPING:
        pool = sorted(pid for pid in by_tag.get(tag, {}) if has_thumb(pid))
        rng = random.Random(f"{seed}:{tag}")
        drawn = rng.sample(pool, min(per_tag, len(pool)))
        counts[tag] = (len(pool), len(drawn))
        for pid in drawn:
            entry = photos.setdefault(pid, {"tags": [], "keywords": []})
            entry["tags"].append(tag)
            for kw in by_tag[tag][pid]:
                if kw not in entry["keywords"]:
                    entry["keywords"].append(kw)
    for entry in photos.values():
        entry["tags"].sort()
        entry["keywords"].sort()
    return photos, counts


def cmd_sample(args):
    existing = load_sample()
    if existing["photos"] and not args.force:
        raise SystemExit(
            f"{sample_path()} already holds {len(existing['photos'])} photos. "
            "Cached runs are keyed to it — re-sampling orphans them. Pass "
            "--force if that is what you want.")
    if not os.path.isdir(THUMBS):
        raise SystemExit(f"Thumbs directory not found: {THUMBS} "
                         "(set PHOTOSEARCH_UNSPLASH_THUMBS)")
    human = load_human_keywords()
    photos, counts = choose_sample(photos_by_tag(human), per_tag=args.per_tag,
                                   seed=args.seed)
    # The photographer's FULL keyword list rides along for the contact sheet:
    # "moody" next to "portrait, studio" means something different from
    # "moody" next to "storm, coast", and the sheet exists to show that.
    for pid, entry in photos.items():
        entry["human_keywords"] = sorted(set(human.get(pid, ())))
    save_sample(photos, args.seed, args.per_tag)
    print(f"[sample] {len(photos)} photos (seed {args.seed}, per-tag {args.per_tag}) "
          f"-> {sample_path()}")
    print(f"  {'tag':<18} {'available':>9} {'drawn':>5}")
    for tag, (avail, drawn) in counts.items():
        print(f"  {tag:<18} {avail:>9} {drawn:>5}")
    short = [t for t, (a, d) in counts.items() if d < MIN_N]
    if short:
        print(f"  ! under {MIN_N} photos (recall withheld): {', '.join(short)}")


# --------------------------------------------------------------------------
# Run cache
# --------------------------------------------------------------------------

def run_path(variant):
    return unsplash_dir() / "runs" / f"{variant}.json"


def load_run(variant):
    path = run_path(variant)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_run(variant, data):
    _write_atomic(run_path(variant), data)


def list_variants():
    d = unsplash_dir() / "runs"
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json") if not p.name.startswith("."))


def predictions_of(run):
    """{photo_id: tags-or-None}. `None` (no usable answer) and `[]` (the
    model answered "none") are different results; null vs [] in the JSON."""
    return {pid: (None if p["tags"] is None else list(p["tags"]))
            for pid, p in run["predictions"].items()}


def run_variant(variant, *, model=None, prompt_text=None, prompt_file=None,
                extra_vocab=(), limit=None, force=False, tagger=None, log=print):
    """Predict every sampled photo from its local thumb; cache as we go.
    Returns the run dict. `tagger` is injectable so tests need no model."""
    from photosearch import describe
    if not owner._VARIANT_RE.match(variant) or variant == owner.STORED:
        raise SystemExit(f"Bad variant name {variant!r} "
                         f"(letters, digits, . _ - ; '{owner.STORED}' is reserved).")
    tagger = tagger or owner.production_tagger

    owner._pin_model(model)
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
               "extra_vocab": extra_vocab, "pixels": "unsplash-thumb-400px",
               "created": datetime.now(timezone.utc).isoformat(),
               "predictions": {}}

    sample = load_sample()
    if not sample["photos"]:
        raise SystemExit(f"No sample at {sample_path()} — run `sample` first.")
    todo = [pid for pid in sorted(sample["photos"]) if pid not in run["predictions"]]
    if limit is not None:
        todo = todo[:limit]
    log(f"[run] {variant}: model={effective} prompt={prompt_file or 'production'} "
        f"— {len(todo)} to do, {len(run['predictions'])} cached, "
        f"{len(sample['photos'])} sampled")

    errors = 0
    with owner.prompt_override(prompt_text), owner.vocab_override(extra_vocab):
        for i, pid in enumerate(todo, 1):
            # The thumb IS the input — read in place, never copied or touched.
            path = thumb_path(pid)
            try:
                if not os.path.exists(path):
                    raise FileNotFoundError(path)
                t0 = time.time()
                tags, calls = tagger(path, nominal)
                latency = time.time() - t0
            except Exception as e:
                # Not cached, so the next run retries it.
                errors += 1
                log(f"  ! {pid}: {e}")
                continue
            run["predictions"][pid] = {
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


def cmd_run(args):
    prompt_text = None
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt_text = f.read()
    run_variant(args.variant, model=args.model, prompt_text=prompt_text,
                prompt_file=args.prompt_file,
                extra_vocab=[t.strip() for t in (args.extra_vocab or "").split(",") if t.strip()],
                limit=args.limit, force=args.force)


# --------------------------------------------------------------------------
# Report — recall and contradictions only
# --------------------------------------------------------------------------

def score(predicted, sample, extra_tags=()):
    """Per-tag recall + contradiction counts of `predicted` {photo_id: tags}
    against the sample's positive-only labels.

    For each tag T, over the sampled photos positive for T that have a
    prediction: `found` = the model emitted T; `contradiction` = it emitted a
    CONTRADICTORY_PAIRS partner of T instead/as well. A None prediction (no
    usable answer) counts as neither — it is a miss, since that is what the
    library would hold. `extra_tags` are the candidate tags this variant
    offered; any other candidate is left out of `per_tag` entirely, so
    production is never charged a miss for a tag it was never asked about.
    """
    extra = [t for t in extra_tags if t in store.CANDIDATE_TAGS]
    allowed = _PERCEIVED | frozenset(extra)
    tags = [t for t in MAPPING if t in allowed]
    per = {t: {"positives": 0, "scored": 0, "unanswered": 0, "found": 0,
               "contradiction": 0} for t in tags}
    answered_sizes = []
    scored = unanswered = 0
    for pid, entry in sample["photos"].items():
        for t in entry["tags"]:
            if t in per:
                per[t]["positives"] += 1
        if pid not in predicted:
            continue
        scored += 1
        pred = predicted[pid]
        if pred is None:
            unanswered += 1
            pred_set = None
        else:
            pred_set = set(pred) & allowed
            answered_sizes.append(len(pred_set))
        for t in entry["tags"]:
            if t not in per:
                continue
            c = per[t]
            c["scored"] += 1
            if pred_set is None:
                c["unanswered"] += 1
                continue
            if t in pred_set:
                c["found"] += 1
            if pred_set & CONTRADICTS.get(t, set()):
                c["contradiction"] += 1

    def _rates(c):
        return {**c,
                "recall": (c["found"] / c["scored"]) if c["scored"] else None,
                "contradiction_rate": (c["contradiction"] / c["scored"]) if c["scored"] else None}

    total = {k: sum(c[k] for c in per.values())
             for k in ("positives", "scored", "unanswered", "found", "contradiction")}
    return {"photos_scored": scored, "unanswered": unanswered,
            "avg_tags": (sum(answered_sizes) / len(answered_sizes)) if answered_sizes else None,
            "overall": _rates(total), "per_tag": {t: _rates(c) for t, c in per.items()}}


def sample_positives(sample):
    """{tag: photos positive for it} over the whole sample — from the sample,
    not from any run, so a candidate tag no variant offered still shows its
    row (as "not offered") instead of vanishing."""
    counts = {t: 0 for t in MAPPING}
    for entry in sample["photos"].values():
        for t in entry["tags"]:
            if t in counts:
                counts[t] += 1
    return counts


def build_report(variant_predictions, sample=None, extras=None):
    sample = load_sample() if sample is None else sample
    return {"variants": list(variant_predictions), "sampled": len(sample["photos"]),
            "positives": sample_positives(sample),
            "results": {name: score(pred, sample, (extras or {}).get(name, ()))
                        for name, pred in variant_predictions.items()}}


def _tag_order(report):
    """Tags by positives in the sample, descending — this set is about
    coverage, so the best-supported numbers come first. A tag with no
    positives has nothing to say and is dropped."""
    positives = report["positives"]
    return sorted((t for t, n in positives.items() if n), key=lambda t: (-positives[t], t))


def report_tables(report):
    """[(title, header-row, [rows])] — one source for the text and HTML views."""
    names = report["variants"]
    res = report["results"]
    tables = []

    rows = []
    for n in names:
        r = res[n]
        avg = "—" if r["avg_tags"] is None else f"{r['avg_tags']:.2f}"
        o = r["overall"]
        rows.append([n, r["photos_scored"], r["unanswered"], avg,
                     fmt_ratio(o["found"], o["scored"]),
                     fmt_ratio(o["contradiction"], o["scored"])])
    tables.append(("Overall", ["variant", "photos", "unanswered", "tags/photo",
                               "recall (all tags)", "contradiction"], rows))

    head = ["tag", "positives"]
    for n in names:
        head += [f"{n}: found/scored", "recall", "contradiction"]
    rows = []
    for t in _tag_order(report):
        row = [t, report["positives"][t]]
        for n in names:
            c = res[n]["per_tag"].get(t)
            if c is None:          # a candidate this variant never offered
                row += ["not offered", "—", "—"]
                continue
            contra = fmt_ratio(c["contradiction"], c["scored"]) if CONTRADICTS.get(t) else "—"
            row += [f"{c['found']}/{c['scored']}", fmt_ratio(c["found"], c["scored"]), contra]
        rows.append(row)
    tables.append(("Per tag (sorted by positives)", head, rows))
    return tables


_HEADER = ("RECALL ONLY. Photographer keywords are positive-only labels: a photo "
           "without a keyword is not a \"no\", so NO PRECISION is reported here — "
           "the owner-labelled set is the one that measures it. Pixels are 400-px "
           "Unsplash thumbs, not the fleet's 1024-px encode, so these numbers are "
           "not comparable to the owner-set numbers, only to each other.")


def render_text(report):
    out = [f"{report['sampled']} sampled Unsplash photos.", _HEADER,
           f"Ratios carry their counts; below {MIN_N} they are withheld. "
           "`contradiction` = the model emitted a CONTRADICTORY_PAIRS partner of "
           "the keyworded tag (e.g. `sunny` on a photographer's `overcast`)."]
    for title, head, rows in report_tables(report):
        table = [[str(c) for c in r] for r in [head] + rows]
        widths = [max(len(r[i]) for r in table) for i in range(len(head))]
        out.append(f"\n=== {title} ===")
        for r in table:
            out.append("  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                                 for i, (c, w) in enumerate(zip(r, widths))).rstrip())
    return "\n".join(out)


_CSS = """
 body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}
 h1{font-size:20px} h2{font-size:16px;margin-top:28px}
 p.note{color:#999;max-width:70em}
 table{border-collapse:collapse;margin:12px 0}
 th,td{border:1px solid #333;padding:5px 10px;text-align:right}
 td:first-child,th:first-child{text-align:left}
 td.low{color:#777}
"""


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
<title>Visual-tag eval — Unsplash recall</title><style>{_CSS}</style></head><body>
<h1>Visual-tag eval — Unsplash photographer keywords, {report['sampled']} sampled</h1>
<p class=note><b>{html.escape(_HEADER)}</b></p>
<p class=note>Ratios carry their counts; below {MIN_N} observations they are
withheld. <i>contradiction</i> = the model emitted a CONTRADICTORY_PAIRS partner
of the keyworded tag.</p>
{''.join(parts)}
</body></html>"""


def cmd_report(args):
    sample = load_sample()
    if not sample["photos"]:
        raise SystemExit(f"No sample at {sample_path()} — run `sample` first.")
    wanted = [v.strip() for v in args.variants.split(",") if v.strip()] \
        if args.variants else list_variants()
    preds, extras = {}, {}
    for name in wanted:
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
        raise SystemExit("Nothing to report — no runs cached.")
    report = build_report(preds, sample, extras=extras)
    print(render_text(report))
    if args.html:
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(render_html(report))
        print(f"\n[report] {args.html}")


# --------------------------------------------------------------------------
# Contact sheet — what do photographers mean by this term?
# --------------------------------------------------------------------------

def sheet_photos(sample, tag, n=24):
    """The first `n` sampled photos positive for `tag`, in id order (ids are
    opaque strings, so this is an arbitrary but stable pick)."""
    return [pid for pid in sorted(sample["photos"])
            if tag in sample["photos"][pid]["tags"]][:n]


def render_sheet(tag, pids, sample, runs, urls=None):
    """HTML: thumb (linked in place from THUMBS), the keywords that put the
    photo here in bold, the photographer's other keywords, and one line per
    cached variant with the tag highlighted and any contradiction in red."""
    urls = urls or {}
    partners = CONTRADICTS.get(tag, set())
    cards = []
    for pid in pids:
        entry = sample["photos"][pid]
        thumb = "file://" + thumb_path(pid)
        img = f'<img src="{html.escape(thumb)}" alt="{html.escape(pid)}" loading=lazy>'
        if pid in urls:
            img = f'<a href="{html.escape(urls[pid])}" target=_blank>{img}</a>'
        matched = set(entry.get("keywords", ()))
        others = [k for k in entry.get("human_keywords", ()) if k not in matched]
        kw = ", ".join(f"<b>{html.escape(k)}</b>" for k in sorted(matched))
        if others:
            kw += " · " + html.escape(", ".join(others))
        lines = []
        for name, run in runs.items():
            p = run["predictions"].get(pid)
            if p is None:
                shown = "<i>not run</i>"
            elif p["tags"] is None:
                shown = "<i>UNANSWERED</i>"
            elif not p["tags"]:
                shown = "none"
            else:
                shown = ", ".join(
                    f"<span class=hit>{html.escape(t)}</span>" if t == tag else
                    f"<span class=bad>{html.escape(t)}</span>" if t in partners else
                    html.escape(t) for t in p["tags"])
            lines.append(f"<div class=pred><span class=v>{html.escape(name)}</span> {shown}</div>")
        cards.append(f"<div class=card>{img}<div class=kw>{kw}</div>{''.join(lines)}"
                     f"<div class=id>{html.escape(pid)}</div></div>")
    return f"""<!DOCTYPE html><html><head><meta charset=utf-8>
<title>Unsplash contact sheet — {html.escape(tag)}</title><style>{_CSS}
 .grid{{display:flex;flex-wrap:wrap;gap:14px}}
 .card{{width:260px;background:#1a1a1a;border:1px solid #333;padding:8px;font-size:12px}}
 .card img{{width:244px;display:block;background:#000}}
 .kw{{margin:6px 0;color:#ccc}} .pred{{margin:2px 0}} .v{{color:#8ab;margin-right:4px}}
 .hit{{color:#6d6;font-weight:bold}} .bad{{color:#e66;font-weight:bold}}
 .id{{color:#666;margin-top:4px}}
</style></head><body>
<h1>What photographers mean by <code>{html.escape(tag)}</code> — {len(pids)} photos</h1>
<p class=note>Positive-only: every photo here carries a photographer keyword that
maps to <code>{html.escape(tag)}</code> (bold). Other keywords are the photographer's
own, for context. Per variant: <span class=hit>green</span> = the model emitted the
tag, <span class=bad>red</span> = it emitted a contradicting partner
({html.escape(', '.join(sorted(partners)) or 'none defined')}).</p>
<div class=grid>{''.join(cards)}</div>
</body></html>"""


def cmd_sheet(args):
    sample = load_sample()
    if not sample["photos"]:
        raise SystemExit(f"No sample at {sample_path()} — run `sample` first.")
    if args.tag not in MAPPING:
        raise SystemExit(f"Unknown tag {args.tag!r} (have: {', '.join(MAPPING)})")
    pids = sheet_photos(sample, args.tag, args.n)
    if not pids:
        raise SystemExit(f"No sampled photos are positive for {args.tag!r}.")
    runs = {v: load_run(v) for v in list_variants()}
    out = args.out or str(unsplash_dir() / f"sheet-{args.tag}.html")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_sheet(args.tag, pids, sample, runs, load_photo_urls()))
    print(f"[sheet] {args.tag}: {len(pids)} photos, {len(runs)} variant(s) -> {out}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("sample", help="Draw per-tag photos from the photographer keywords.")
    sp.add_argument("--per-tag", type=int, default=40)
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--force", action="store_true",
                    help="Overwrite an existing sample (orphans its cached runs).")
    sp.set_defaults(func=cmd_sample)

    rp = sub.add_parser("run", help="Tag every sampled photo for one variant.")
    rp.add_argument("--variant", default=DEFAULT_VARIANT)
    rp.add_argument("--model", help="Model id. On the LM Studio route this is "
                                    "pinned as the visual role model for this process.")
    rp.add_argument("--extra-vocab", help="Comma-separated CANDIDATE tags the parser "
                    "should accept and the report should score. Needs --prompt-file.")
    rp.add_argument("--prompt-file", help="Alternate prompt text (replaces only "
                                          "the prompt; parser and guard stay production).")
    rp.add_argument("--limit", type=int, help="Predict at most N new photos.")
    rp.add_argument("--force", action="store_true",
                    help="Discard this variant's cached run and start over.")
    rp.set_defaults(func=cmd_run)

    tp = sub.add_parser("report", help="Per-tag recall + contradictions across cached runs.")
    tp.add_argument("--variants", help="Comma-separated; default = every cached run.")
    tp.add_argument("--html", help="Also write an HTML report here.")
    tp.set_defaults(func=cmd_report)

    hp = sub.add_parser("sheet", help="HTML contact sheet of sampled photos for one tag.")
    hp.add_argument("--tag", required=True)
    hp.add_argument("--n", type=int, default=24)
    hp.add_argument("--out", help="Default: <eval dir>/unsplash/sheet-<tag>.html")
    hp.set_defaults(func=cmd_sheet)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
