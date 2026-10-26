#!/usr/bin/env python
"""Text-pass eval — category-content and keywords, from a FROZEN description.

Both passes only ever see the description, so a model comparison is fair only
if every model reads the same text. `freeze` snapshots one set of descriptions
(by default what production holds today) into a file that never changes; every
text model is run on exactly that, and the owner's labels are keyed to it.

  freeze   snapshot the descriptions of the describe (or visual-tags) sample:
               python evals/text_passes_eval.py freeze --db photo_index.db.local
           or, once a describe winner exists, a second input set for the
           describe→text chain check:
               python evals/text_passes_eval.py freeze --name chain --from qwen9b
  run      the PRODUCTION extractor, one model per run:
               export PHOTOSEARCH_TEXT_LLM_URL=http://localhost:1234/v1
               python evals/text_passes_eval.py run --pass category-content \\
                   --variant e4b --model google/gemma-4-e4b
  report   automatic metrics for every run, plus owner-label precision /
           pooled recall once /eval/models?tab=categories|keywords is labelled:
               python evals/text_passes_eval.py report --db photo_index.db.local

Automatic metrics (no owner time):
  - timeouts per 100 ATTEMPTS against the production 10 s per-call limit
    (_TEXT_OLLAMA_TIMEOUT_S) — the retry loop hides recovered ones otherwise —
    the deferred rate (extractor returned None: production retries later),
    and p50 / p95 latency of answered attempts.
  - unsupported-tag rate: tags with no word-level support in the
    description. This is the "soccer on a sailboat photo" failure, caught
    without a label. Strict = every content word appears; lenient also
    accepts a small hand-written synonym map.
  - tags per description; off-vocabulary answers the parser dropped.

Owner labels are judged against the PHOTO (the description is shown). A wrong
category the description DOES support is the describe model's fault; one it
does not support is the text model's. The report splits them.

Storage: photosearch/model_eval.py (passes `text`, `category-content`,
`keywords`). The DB is only opened read-only.
"""
import argparse
import json
import os
import re
import sys
import time
from contextlib import contextmanager

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

from photosearch import describe  # noqa: E402
from photosearch import model_eval as me  # noqa: E402

PASSES = me.TEXT_PASSES
TEXT_TIMEOUT_S = 10          # describe._TEXT_OLLAMA_TIMEOUT_S

_STOP = frozenset("""a an the of and or in on at to for with by from as is are be
    over under near into onto up down out off this that these those its it their
    his her some any other""".split())

# Lenient support: a tag word counts as supported when the description has any
# of these words. Hand-written and deliberately small; add entries only for
# words a description plainly means (not "related" concepts).
SUPPORT_SYNONYMS = {
    "male": {"man", "men", "boy", "boys", "he", "his", "gentleman"},
    "female": {"woman", "women", "girl", "girls", "she", "her", "lady"},
    "adult": {"man", "men", "woman", "women", "adults", "parent", "father", "mother"},
    "child": {"kid", "kids", "boy", "girl", "children", "toddler", "baby"},
    "children": {"kid", "kids", "boys", "girls", "child"},
    "people": {"person", "persons", "man", "woman", "men", "women", "crowd",
               "group", "family", "individuals", "children", "kids"},
    "person": {"man", "woman", "boy", "girl", "child", "individual", "people"},
    "dog": {"puppy", "canine"},
    "cat": {"kitten", "feline"},
    "water": {"lake", "river", "ocean", "sea", "pond", "bay", "stream", "waves"},
    "beach": {"shore", "sand", "seashore", "coast"},
    "car": {"vehicle", "sedan", "suv", "truck"},
    "food": {"meal", "dish", "plate", "snack", "dinner", "lunch", "breakfast"},
    "outdoor": {"outside", "outdoors", "park", "field", "garden", "street", "sky"},
    "indoor": {"inside", "indoors", "room", "kitchen", "living", "hall"},
    "sports": {"sport", "soccer", "baseball", "basketball", "football", "tennis",
               "game", "match"},
    "night": {"dark", "evening", "nighttime"},
}


# --------------------------------------------------------------------------
# Word support
# --------------------------------------------------------------------------

def _stem(w):
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _words(text):
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _STOP]


def supported(tag, description, lenient=False):
    """Does every content word of `tag` appear (stemmed) in `description`?"""
    desc_words = set(_words(description))
    desc = {_stem(w) for w in desc_words} | desc_words
    words = _words(tag)
    if not words:
        return True
    for w in words:
        if w in desc or _stem(w) in desc:
            continue
        if lenient and (SUPPORT_SYNONYMS.get(w, set()) | SUPPORT_SYNONYMS.get(_stem(w), set())) & desc:
            continue
        # Compounds: "sailboat" supports "boat", "footballer" does not support "ball".
        if lenient and len(w) >= 4 and any(d.endswith(w) or d.endswith(_stem(w))
                                           for d in desc_words):
            continue
        return False
    return True


# --------------------------------------------------------------------------
# Freeze
# --------------------------------------------------------------------------

def sample_photo_ids():
    ids = me.sample_ids("describe")
    if ids:
        return ids
    from photosearch import visual_tag_eval
    return [p["photo_id"] for p in visual_tag_eval.load_sample()["photos"]]


def freeze(name="main", source=me.STORED, *, db=None, force=False):
    ids = sample_photo_ids()
    if not ids:
        raise SystemExit("No sample: draw the describe sample first "
                         "(python evals/describe_eval.py sample).")
    if source == me.STORED:
        conn = me.open_db_readonly(db)
        items = {}
        for pid in ids:
            row = conn.execute("SELECT description FROM photos WHERE id = ?", (pid,)).fetchone()
            if row and row["description"]:
                items[pid] = row["description"]
        eff = "(production, mixed)"
    else:
        run = me.load_run("describe", source)
        if run is None:
            raise SystemExit(f"no describe run {source!r}")
        items = {int(k): v.get("text") for k, v in run["items"].items() if int(k) in ids}
        eff = run.get("effective_model")
    return me.save_inputs(name, source, eff, items, force=force)


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def production_extractor(pass_, text, nominal):
    from photosearch import describe
    fn = (describe.extract_categories_from_description if pass_ == "category-content"
          else describe.extract_keywords_from_description)
    return fn(text, model=nominal)


def nominal_model(pass_):
    from photosearch import describe
    return describe.CATEGORY_CONTENT_MODEL if pass_ == "category-content" \
        else describe.KEYWORDS_MODEL


def extract_one(pass_, text, nominal, extractor=None):
    """(item) for one description. Raises TransportError when the backend was
    unreachable; a call that TIMED OUT on every attempt is a real outcome here
    (production defers the photo) and is returned as deferred."""
    extractor = extractor or production_extractor
    rec = me.Recorder()
    t0 = time.time()
    unusable = None
    with rec.active():
        try:
            tags = extractor(pass_, text, nominal)
        except describe.UnusableAnswer as e:
            # Production spends an attempt and stores nothing. Scored like a
            # deferral (no tags to judge), but recorded with its reason.
            tags, unusable = None, str(e)
    latency = time.time() - t0
    last = rec.calls[-1] if rec.calls else None
    timed_out = (tags is None and last is not None and "error" in last
                 and last["attempts"]
                 and all(a.get("outcome") == "timeout" for a in last["attempts"]))
    if not timed_out and unusable is None:
        rec.check(tags is None, pass_)
    raw = (last or {}).get("raw")
    off_vocab = 0
    if pass_ == "category-content" and raw and tags is not None:
        tokens = [t.strip().lower().rstrip(".") for t in raw.split(",") if t.strip()]
        off_vocab = sum(1 for t in tokens if t not in set(tags))
    return {"tags": None if tags is None else list(tags), "deferred": tags is None,
            "unusable": unusable, "raw": raw, "off_vocab": off_vocab, "attempts": rec.attempts(),
            "timeouts": rec.count("timeout"), "latency_s": round(latency, 3)}


def run_variant(pass_, variant, *, model=None, inputs_name="main", limit=None,
                force=False, force_stale=False, solo=False, extractor=None, log=print):
    from photosearch import describe
    if pass_ not in PASSES:
        raise SystemExit(f"--pass must be one of {PASSES}")
    inputs = me.load_inputs(inputs_name)
    if inputs is None:
        raise SystemExit(f"no frozen inputs {inputs_name!r} — run `freeze` first")
    me.pin_role_model("text", model)
    nominal = nominal_model(pass_)
    effective = describe.effective_model(model or nominal, "text")
    effort = me.sent_reasoning_effort(effective)
    spec = me.make_spec(me.spec_parts(pass_, reasoning_effort=effort))
    run = me.open_run(pass_, variant, force=force, spec=spec, force_stale=force_stale,
                      effective_model=effective, input_source=me.inputs_identity(inputs))
    run.setdefault("model", model or nominal)
    run.setdefault("reasoning_effort", effort)
    todo = [pid for pid in inputs["items"] if pid not in run["items"]]
    if limit is not None:
        todo = todo[:limit]
    loaded_start = me.lmstudio_loaded()
    log(f"[run] {pass_} {variant}: model={effective} inputs={inputs_name} — "
        f"{len(todo)} to do, {len(run['items'])} cached")
    fc = me.FailureCounter(log)
    for i, pid in enumerate(todo, 1):
        text = inputs["items"][pid]["text"]
        try:
            item = extract_one(pass_, text, nominal, extractor)
        except Exception as e:
            fc.fail(pid, e)
            continue
        fc.ok()
        item["text_sha"] = inputs["items"][pid]["text_sha"]
        run["items"][pid] = item
        me.save_run(pass_, variant, run)
        shown = (f"UNUSABLE ({item['unusable']})" if item.get("unusable")
                 else "DEFERRED (timeout)" if item["deferred"]
                 else ", ".join(item["tags"]) or "none")
        log(f"  [{i}/{len(todo)}] {pid}: {shown}  ({item['latency_s']:.1f}s)")
    fc.summary()
    loaded_end = me.lmstudio_loaded()
    run["loaded_models_start"], run["loaded_models_end"] = loaded_start, loaded_end
    run["latency_label"] = me.latency_label(effective, loaded_start, loaded_end, solo)
    me.save_run(pass_, variant, run)
    return run


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def stored_run(conn, pass_, inputs):
    """What the DB holds today — meaningful only for inputs frozen FROM stored
    (otherwise the stored tags came from a different description)."""
    col = "categories" if pass_ == "category-content" else "keywords"
    items = {}
    for pid, inp in inputs["items"].items():
        row = conn.execute(f"SELECT {col} FROM photos WHERE id = ?", (int(pid),)).fetchone()
        try:
            tags = json.loads(row[col]) if row and row[col] else None
        except (TypeError, ValueError):
            tags = None
        if isinstance(tags, list):
            items[pid] = {"tags": [str(t) for t in tags], "deferred": False,
                          "text_sha": inp["text_sha"]}
    return {"variant": me.STORED, "effective_model": "(production, mixed)", "items": items,
            "latency_label": "—"}


def vocabulary_gaps(inputs_name="main"):
    """[(term, photos)] the owner marked right that the content vocabulary
    lacks — most frequent first. Evidence for growing the vocabulary; never
    scored, since no model is allowed to emit them."""
    inputs = me.load_inputs(inputs_name)
    if inputs is None:
        return []
    keys = {me.label_key(pid, it["text_sha"]) for pid, it in inputs["items"].items()}
    counts = {}
    for key, lab in me.load_category_labels().items():
        if key in keys and lab.get("done"):
            for t in lab.get("missing") or []:
                counts[t] = counts.get(t, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def summarize(pass_, run, inputs, pool_runs=()):
    items = run["items"]
    n = len(items)
    answered = {pid: it for pid, it in items.items()
                if not it.get("deferred") and it.get("tags") is not None}
    tags_n = unsup = unsup_len = 0
    for pid, it in answered.items():
        desc = inputs["items"].get(pid, {}).get("text", "")
        for t in it["tags"]:
            tags_n += 1
            unsup += not supported(t, desc)
            unsup_len += not supported(t, desc, lenient=True)
    attempts = [a for it in items.values() for a in it.get("attempts") or []]
    ok_lat = [a["elapsed_s"] for a in attempts if a.get("outcome") == "ok"]
    row = {"pass": pass_, "variant": run.get("variant"),
           "effective_model": run.get("effective_model"), "n": n,
           "n_full": len(inputs["items"]),
           "deferred": n - len(answered),
           "unusable": sum(1 for it in items.values() if it.get("unusable")),
           "attempts": len(attempts),
           "timeouts": sum(1 for a in attempts if a.get("outcome") == "timeout"),
           "lat_p50": me.median(ok_lat), "lat_p95": me.percentile(ok_lat, 0.95),
           "tags": tags_n, "tags_per": tags_n / len(answered) if answered else None,
           "unsupported": unsup, "unsupported_lenient": unsup_len,
           "off_vocab": sum(it.get("off_vocab") or 0 for it in items.values()),
           "latency_label": run.get("latency_label", "unknown"),
           "spec": me.spec_status(pass_, run)}
    row.update(score_labels(pass_, run, inputs, pool_runs))
    return row


def _shown_pool(lab, pid, pool_runs):
    """(pool, how) — the categories the owner was shown for this photo.

    `recorded`: the label stores its pool (saved since 2026-10-25).
    `reconstructed`: the label's yes-set plus the tags of every run created
    before the label was saved. That covers every tag those runs emitted, but
    the stored column on screen at the time is unknown, so a tag outside it
    is unjudged AND makes the precision approximate."""
    if lab.get("pool") is not None:
        return set(lab["pool"]) | set(lab.get("yes") or []), "recorded"
    at = lab.get("updated_at") or ""
    pool = set(lab.get("yes") or [])
    for r in pool_runs:
        if r.get("created") and at and r["created"] <= at:
            pool |= set(((r.get("items") or {}).get(pid) or {}).get("tags") or [])
    return pool, "reconstructed"


def score_labels(pass_, run, inputs, pool_runs=()):
    """Owner-label scores over the FULL labelled sample: a labelled photo the
    run deferred / found unusable / never cached scores as no tags (every
    right category a miss), so a failure never raises a score. Returns the
    pooled counts plus one record per labelled photo (for the bootstrap).

    Categories: precision + pooled recall, the describe-vs-text split of
    wrong ones, and `unjudged` — tags outside the checklist the owner saw,
    excluded from precision (a model added after labelling is not charged
    for terms nobody judged). Keywords: precision; `judged` was always
    recorded, so unjudged is exact."""
    items = run["items"]
    pool_runs = list(pool_runs) or [run]      # a run's own tags were on screen

    def got_of(pid):
        it = items.get(pid) or {}
        if it.get("deferred") or it.get("tags") is None:
            return None
        return set(it["tags"])

    records = []
    if pass_ == "category-content":
        labels = me.load_category_labels()
        fp_desc = 0
        approx = False
        for pid, inp in inputs["items"].items():
            lab = labels.get(me.label_key(pid, inp["text_sha"]))
            if not lab or not lab.get("done"):
                continue
            got = got_of(pid)
            yes = set(lab["yes"])
            pool, how = _shown_pool(lab, pid, pool_runs)
            r = {"pid": pid, "answered": got is not None, "tp": 0, "fp": 0, "unjudged": 0,
                 "fn": len(yes)}
            if got is not None:
                r["tp"], r["fn"] = len(got & yes), len(yes - got)
                for t in got - yes:
                    if t not in pool:
                        r["unjudged"] += 1
                        approx |= how == "reconstructed"
                        continue
                    r["fp"] += 1
                    fp_desc += supported(t, inp.get("text", ""), lenient=True)
            records.append(r)
        out = {k: sum(r[k] for r in records) for k in ("tp", "fp", "fn", "unjudged")}
        return {"labelled": len(records),
                "answered_labelled": sum(r["answered"] for r in records),
                **out, "fp_from_description": fp_desc, "precision_approx": approx,
                "records": records}
    labels = me.load_keyword_labels()
    for pid, inp in inputs["items"].items():
        lab = labels.get(me.label_key(pid, inp["text_sha"]))
        if not lab or not lab.get("done"):
            continue
        got = got_of(pid)
        judged, bad = set(lab["judged"]), set(lab["wrong"])
        r = {"pid": pid, "answered": got is not None, "right": 0, "wrong": 0, "unjudged": 0}
        for k in got or ():
            if k in bad:
                r["wrong"] += 1
            elif k in judged:
                r["right"] += 1
            else:
                r["unjudged"] += 1
        records.append(r)
    return {"labelled": len(records), "answered_labelled": sum(r["answered"] for r in records),
            "kw_right": sum(r["right"] for r in records),
            "kw_wrong": sum(r["wrong"] for r in records),
            "kw_unjudged": sum(r["unjudged"] for r in records), "records": records}


def category_scores(records):
    """(precision, pooled recall, F1) from per-photo category records."""
    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    p = tp / (tp + fp) if tp + fp else None
    rc = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * rc / (p + rc) if p and rc else (0.0 if p is not None and rc is not None else None)
    return p, rc, f1


def category_f1(records):
    return category_scores(records)[2]


def keyword_photo_precision(records):
    """Mean per-photo keyword precision over the full labelled sample; a
    photo with no usable answer scores 0 (one with only unjudged keywords is
    skipped — there is nothing to score)."""
    vals = []
    for r in records:
        if not r["answered"]:
            vals.append(0.0)
        elif r["right"] + r["wrong"]:
            vals.append(r["right"] / (r["right"] + r["wrong"]))
    return sum(vals) / len(vals) if vals else None


# --------------------------------------------------------------------------
# Replay: today's guards on the cached answers
# --------------------------------------------------------------------------

@contextmanager
def _canned_answer(raw):
    """Every production text call returns `raw` — the retry included, so a
    replay assumes the model would answer the same way twice. No model, no
    network."""
    real_chat, real_has = describe._text_chat_with_retry, describe.HAS_OLLAMA
    describe._text_chat_with_retry = lambda *a, **kw: raw
    describe.HAS_OLLAMA = True
    try:
        yield
    finally:
        describe._text_chat_with_retry, describe.HAS_OLLAMA = real_chat, real_has


def replay_item(pass_, description, item):
    """The cached RAW answer run through the CURRENT production extractor
    (category grounding + recitation / runaway guards, or the keyword
    guards): {"tags", "unusable"}, or None when there is nothing to replay (a
    deferral, the stored pseudo-variant)."""
    raw = item.get("raw")
    if raw is None:
        return None
    fn = (describe.extract_categories_from_description if pass_ == "category-content"
          else describe.extract_keywords_from_description)
    with _canned_answer(raw):
        try:
            tags = fn(description, model="replay")
        except describe.UnusableAnswer as e:
            return {"tags": None, "deferred": True, "unusable": str(e)}
    if tags is None:
        return None
    return {"tags": list(tags), "deferred": False, "unusable": None}


def replay_run(pass_, run, inputs):
    """A copy of `run` with every replayable item re-judged by today's code,
    or None when nothing could be replayed."""
    items, replayed = {}, 0
    for pid, it in run["items"].items():
        rep = replay_item(pass_, inputs["items"].get(pid, {}).get("text", ""), it)
        if rep is None:
            items[pid] = it
            continue
        replayed += 1
        items[pid] = {**it, **rep, "off_vocab": it.get("off_vocab")}
    if not replayed:
        return None
    return {**run, "items": items, "replayed": replayed}


def build_report(*, db=None, inputs_name="main", replay=True):
    inputs = me.load_inputs(inputs_name)
    if inputs is None:
        return []
    ident = me.inputs_identity(inputs)
    conn = me.open_db_readonly(db) if db else None
    rows = []
    for pass_ in PASSES:
        runs = [me.load_run(pass_, v) for v in me.list_variants(pass_)]
        runs = [r for r in runs if r.get("input_source") == ident]
        pool_runs = list(runs)
        if conn is not None and inputs.get("source") == me.STORED:
            runs.append(stored_run(conn, pass_, inputs))
        for r in runs:
            row = summarize(pass_, r, inputs, pool_runs)
            rep = replay_run(pass_, r, inputs) if replay else None
            if rep is not None:
                g = summarize(pass_, rep, inputs, pool_runs)
                row["replay"] = {k: g[k] for k in (
                    "deferred", "unusable", "tags", "tags_per", "labelled", "answered_labelled",
                    "tp", "fp", "fn", "unjudged", "kw_right", "kw_wrong", "kw_unjudged",
                    "records", "precision_approx") if k in g}
                row["replay"]["replayed"] = rep["replayed"]
            rows.append(row)
    return rows


def _f(v, d=2):
    return "—" if v is None else f"{v:.{d}f}"


def _label_cols(pass_, r):
    """The owner-label columns for one row (raw or replayed)."""
    rec = r.get("records") or []
    if pass_ == "category-content":
        p, rc, f1 = category_scores(rec)
        ci = me.bootstrap_ci(rec, category_f1)
        return (f"{r['answered_labelled']:>3}/{r['labelled']:<3} "
                f"{_f(p):>5}{'~' if r.get('precision_approx') else ' '} {_f(rc):>6} "
                f"{_f(f1):>5} {('[' + _f(ci[0]) + ',' + _f(ci[1]) + ']') if ci else '—':>11} "
                f"{r['unjudged']:>8}")
    prec = r["kw_right"] / (r["kw_right"] + r["kw_wrong"]) if r["kw_right"] + r["kw_wrong"] else None
    pp = keyword_photo_precision(rec)
    ci = me.bootstrap_ci(rec, keyword_photo_precision)
    return (f"{r['answered_labelled']:>3}/{r['labelled']:<3} {_f(prec):>9} {_f(pp):>9} "
            f"{('[' + _f(ci[0]) + ',' + _f(ci[1]) + ']') if ci else '—':>11} {r['kw_unjudged']:>8}")


def render(rows):
    out = []
    for pass_ in PASSES:
        rs = [r for r in rows if r["pass"] == pass_]
        if not rs:
            continue
        out.append(f"=== {pass_} ===")
        out.append(f"{'variant':<16} {'effective model':<26} {'n':>3} {'defer':>5} "
                   f"{'t/o per 100 att':>15} {'p50/p95 s':>11} {'tags/desc':>9} "
                   f"{'unsupp':>6} {'lenient':>7} {'offvoc':>6}  latency")
        for r in rs:
            to = (f"{100 * r['timeouts'] / r['attempts']:.1f}" if r["attempts"] else "—")
            tag = me.stale_tag(r.get("spec"))
            out.append(
                f"{str(r['variant']):<16} {str(r['effective_model'])[:26]:<26} {r['n']:>3} "
                f"{me.fmt_ratio(r['deferred'], r['n']):>5} {to:>15} "
                f"{_f(r['lat_p50'], 1) + '/' + _f(r['lat_p95'], 1):>11} {_f(r['tags_per'], 1):>9} "
                f"{me.fmt_ratio(r['unsupported'], r['tags']):>6} "
                f"{me.fmt_ratio(r['unsupported_lenient'], r['tags']):>7} {r['off_vocab']:>6}  "
                f"{r['latency_label']}{'   ' + tag if tag else ''}")
        lab = [r for r in rs if r.get("labelled")]
        if lab and pass_ == "category-content":
            out.append(f"  {'variant':<16} {'':<9} {'ans/lab':>7} {'prec':>6} {'recall':>6} "
                       f"{'F1':>5} {'F1 95% CI':>11} {'unjudged':>8} "
                       f"{'wrong: desc said so / text model':>34}")
            for r in lab:
                out.append(f"  {str(r['variant']):<16} {'as run':<9} {_label_cols(pass_, r)} "
                           f"{r['fp_from_description']:>17} / {r['fp'] - r['fp_from_description']:<16}")
                if r.get("replay"):
                    g = r["replay"]
                    out.append(f"  {'':<16} {'replayed':<9} {_label_cols(pass_, g)}   "
                               f"unusable now {g['unusable']}, tags/desc {_f(g['tags_per'], 1)}")
            out.append("  pooled recall = recall against every category ANY model proposed "
                       "plus those the owner added — not true recall. Unanswered labelled photos "
                       "count as all misses. unjudged = tags outside the checklist the owner saw "
                       "(excluded from precision); prec~ = that checklist was reconstructed "
                       "(label predates recorded pools), so precision is approximate. replayed = "
                       "the cached raw answer re-judged with CURRENT guards (category_grounding."
                       "ground, recitation/runaway checks; a retry is assumed to repeat the "
                       "answer).")
        elif lab:
            out.append(f"  {'variant':<16} {'':<9} {'ans/lab':>7} {'precision':>9} "
                       f"{'per-photo':>9} {'95% CI':>11} {'unjudged':>8}")
            for r in lab:
                out.append(f"  {str(r['variant']):<16} {'as run':<9} {_label_cols(pass_, r)}")
                if r.get("replay"):
                    g = r["replay"]
                    out.append(f"  {'':<16} {'replayed':<9} {_label_cols(pass_, g)}   "
                               f"unusable now {g['unusable']}")
            out.append("  per-photo = mean keyword precision per labelled photo, an unanswered "
                       "photo scoring 0. replayed = re-judged with CURRENT keyword guards.")
        out.append("")
    gaps = vocabulary_gaps()
    if gaps:
        out.append("Vocabulary gaps (right categories the 360-term vocabulary lacks, "
                   "by photos): " + ", ".join(f"{t} ({n})" for t, n in gaps[:30]))
        out.append("")
    out.append(f"defer = extractor returned None (timed out every attempt at the {TEXT_TIMEOUT_S}s "
               "limit) or the answer was unusable: production retries the photo later. unsupp = "
               "share of tags with no word in the description — a SCREEN, not a verdict: "
               "abstract categories (landscape, natural beauty) are often fair inferences. The "
               f"owner labels decide. {me.STALE} = the run's spec hash (production prompt + "
               "guards + options) is missing or differs from the current code.")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    db_default = os.environ.get("PHOTOSEARCH_DB")
    f = sub.add_parser("freeze")
    f.add_argument("--name", default="main")
    f.add_argument("--from", dest="source", default=me.STORED,
                   help="'stored' (default) or a describe variant")
    f.add_argument("--db", default=db_default)
    f.add_argument("--force", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--pass", dest="pass_", required=True, choices=PASSES)
    r.add_argument("--variant", required=True)
    r.add_argument("--model", help="LM Studio id; pins PHOTOSEARCH_LLM_TEXT_MODEL.")
    r.add_argument("--inputs", default="main")
    r.add_argument("--limit", type=int)
    r.add_argument("--force", action="store_true")
    r.add_argument("--force-stale", action="store_true",
                   help="append to a run cached under a different production spec "
                        "(it stays marked STALE)")
    r.add_argument("--solo", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--db", default=db_default)
    p.add_argument("--inputs", default="main")
    args = ap.parse_args(argv)
    if args.cmd == "freeze":
        data = freeze(args.name, args.source, db=args.db, force=args.force)
        print(f"[freeze] {len(data['items'])} descriptions from {data['source']} "
              f"→ inputs {data['name']!r}")
    elif args.cmd == "run":
        run_variant(args.pass_, args.variant, model=args.model, inputs_name=args.inputs,
                    limit=args.limit, force=args.force, force_stale=args.force_stale,
                    solo=args.solo)
    else:
        print(render(build_report(db=args.db, inputs_name=args.inputs)))


if __name__ == "__main__":
    main()
