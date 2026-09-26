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

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

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
    with rec.active():
        tags = extractor(pass_, text, nominal)
    latency = time.time() - t0
    last = rec.calls[-1] if rec.calls else None
    timed_out = (tags is None and last is not None and "error" in last
                 and last["attempts"]
                 and all(a.get("outcome") == "timeout" for a in last["attempts"]))
    if not timed_out:
        rec.check(tags is None, pass_)
    raw = (last or {}).get("raw")
    off_vocab = 0
    if pass_ == "category-content" and raw and tags is not None:
        tokens = [t.strip().lower().rstrip(".") for t in raw.split(",") if t.strip()]
        off_vocab = sum(1 for t in tokens if t not in set(tags))
    return {"tags": None if tags is None else list(tags), "deferred": tags is None,
            "raw": raw, "off_vocab": off_vocab, "attempts": rec.attempts(),
            "timeouts": rec.count("timeout"), "latency_s": round(latency, 3)}


def run_variant(pass_, variant, *, model=None, inputs_name="main", limit=None,
                force=False, solo=False, extractor=None, log=print):
    from photosearch import describe
    if pass_ not in PASSES:
        raise SystemExit(f"--pass must be one of {PASSES}")
    inputs = me.load_inputs(inputs_name)
    if inputs is None:
        raise SystemExit(f"no frozen inputs {inputs_name!r} — run `freeze` first")
    me.pin_role_model("text", model)
    nominal = nominal_model(pass_)
    effective = describe.effective_model(model or nominal, "text")
    run = me.open_run(pass_, variant, force=force, effective_model=effective,
                      input_source=me.inputs_identity(inputs))
    run.setdefault("model", model or nominal)
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
        shown = "DEFERRED (timeout)" if item["deferred"] else ", ".join(item["tags"]) or "none"
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


def summarize(pass_, run, inputs):
    items = run["items"]
    n = len(items)
    answered = {pid: it for pid, it in items.items() if not it.get("deferred")}
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
           "deferred": n - len(answered),
           "attempts": len(attempts),
           "timeouts": sum(1 for a in attempts if a.get("outcome") == "timeout"),
           "lat_p50": me.median(ok_lat), "lat_p95": me.percentile(ok_lat, 0.95),
           "tags": tags_n, "tags_per": tags_n / len(answered) if answered else None,
           "unsupported": unsup, "unsupported_lenient": unsup_len,
           "off_vocab": sum(it.get("off_vocab") or 0 for it in items.values()),
           "latency_label": run.get("latency_label", "unknown")}
    row.update(score_labels(pass_, answered, inputs))
    return row


def score_labels(pass_, answered, inputs):
    """Precision + pooled recall (categories) or precision (keywords) over the
    owner's done labels, and the describe-vs-text split of wrong categories."""
    if pass_ == "category-content":
        labels = me.load_category_labels()
        tp = fp = fn = fp_desc = 0
        labelled = 0
        for pid, it in answered.items():
            lab = labels.get(me.label_key(pid, it["text_sha"]))
            if not lab or not lab.get("done"):
                continue
            labelled += 1
            yes = set(lab["yes"])
            got = set(it["tags"])
            tp += len(got & yes)
            fn += len(yes - got)
            desc = inputs["items"].get(pid, {}).get("text", "")
            for t in got - yes:
                fp += 1
                fp_desc += supported(t, desc, lenient=True)
        return {"labelled": labelled, "tp": tp, "fp": fp, "fn": fn,
                "fp_from_description": fp_desc}
    labels = me.load_keyword_labels()
    right = wrong = unjudged = labelled = 0
    for pid, it in answered.items():
        lab = labels.get(me.label_key(pid, it["text_sha"]))
        if not lab or not lab.get("done"):
            continue
        labelled += 1
        judged, bad = set(lab["judged"]), set(lab["wrong"])
        for k in it["tags"]:
            if k in bad:
                wrong += 1
            elif k in judged:
                right += 1
            else:
                unjudged += 1
    return {"labelled": labelled, "kw_right": right, "kw_wrong": wrong, "kw_unjudged": unjudged}


def build_report(*, db=None, inputs_name="main"):
    inputs = me.load_inputs(inputs_name)
    if inputs is None:
        return []
    ident = me.inputs_identity(inputs)
    conn = me.open_db_readonly(db) if db else None
    rows = []
    for pass_ in PASSES:
        runs = [me.load_run(pass_, v) for v in me.list_variants(pass_)]
        runs = [r for r in runs if r.get("input_source") == ident]
        if conn is not None and inputs.get("source") == me.STORED:
            runs.append(stored_run(conn, pass_, inputs))
        rows += [summarize(pass_, r, inputs) for r in runs]
    return rows


def _f(v, d=2):
    return "—" if v is None else f"{v:.{d}f}"


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
            out.append(
                f"{str(r['variant']):<16} {str(r['effective_model'])[:26]:<26} {r['n']:>3} "
                f"{me.fmt_ratio(r['deferred'], r['n']):>5} {to:>15} "
                f"{_f(r['lat_p50'], 1) + '/' + _f(r['lat_p95'], 1):>11} {_f(r['tags_per'], 1):>9} "
                f"{me.fmt_ratio(r['unsupported'], r['tags']):>6} "
                f"{me.fmt_ratio(r['unsupported_lenient'], r['tags']):>7} {r['off_vocab']:>6}  "
                f"{r['latency_label']}")
        lab = [r for r in rs if r.get("labelled")]
        if lab and pass_ == "category-content":
            out.append(f"  {'variant':<16} {'labelled':>8} {'precision':>9} "
                       f"{'pooled recall':>13} {'wrong: desc said so / text model':>34}")
            for r in lab:
                out.append(f"  {str(r['variant']):<16} {r['labelled']:>8} "
                           f"{me.fmt_ratio(r['tp'], r['tp'] + r['fp']):>9} "
                           f"{me.fmt_ratio(r['tp'], r['tp'] + r['fn']):>13} "
                           f"{r['fp_from_description']:>17} / {r['fp'] - r['fp_from_description']:<16}")
            out.append("  pooled recall = recall against every category ANY model proposed "
                       "plus those the owner added — not true recall.")
        elif lab:
            out.append(f"  {'variant':<16} {'labelled':>8} {'precision':>9} {'unjudged':>8}")
            for r in lab:
                out.append(f"  {str(r['variant']):<16} {r['labelled']:>8} "
                           f"{me.fmt_ratio(r['kw_right'], r['kw_right'] + r['kw_wrong']):>9} "
                           f"{r['kw_unjudged']:>8}")
        out.append("")
    out.append(f"defer = extractor returned None (timed out every attempt at the {TEXT_TIMEOUT_S}s "
               "limit): production retries the photo later. unsupp = share of tags with no "
               "word in the description — a SCREEN, not a verdict: abstract categories "
               "(landscape, natural beauty) are often fair inferences. The owner labels "
               "decide.")
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
                    limit=args.limit, force=args.force, solo=args.solo)
    else:
        print(render(build_report(db=args.db, inputs_name=args.inputs)))


if __name__ == "__main__":
    main()
