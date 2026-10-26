#!/usr/bin/env python
"""Verify eval — does the verify model catch false claims without crying wolf?

Verify asks a vision model to name the WRONG claims in a description; the
worker then regenerates. To score it you need descriptions whose false claims
are already known, which the describe labels provide:

  clean    descriptions the owner marked all-correct  -> false-rejection rate
  planted  a clean description with ONE templated false claim added (an
           absent object, a swapped colour, a changed count), each confirmed
           false by the owner on /eval/models?tab=planted  -> catch rate
  real     descriptions the owner found wrong claims in  -> catch rate on the
           errors models actually make

  plant    build the sets from a describe variant's labelled descriptions:
               python evals/verify_eval.py plant --source qwen9b
  run      the production check, one verify model per run:
               python evals/verify_eval.py run --variant e2b --model google/gemma-4-e2b
           --mode llm       the verify model's own answer (the comparison)
           --mode pipeline  CLIP gate + LLM + CLIP override, what ships
                            (needs --db for the photos' CLIP embeddings)
  report   catch rate (any flag, and MATCHED: a flag names the planted or
           labelled error), per error type, and false-rejection rate.

The verify model must differ from the model that wrote the descriptions, so
the check is independent; `run` refuses a same-model pairing.

Absent-object nouns are drawn from a fixed list WITHOUT consulting CLIP —
choosing nouns CLIP already rejects would flatter the pipeline mode.
"""
import argparse
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

from photosearch import model_eval as me  # noqa: E402

PASS = "verify"
NOMINAL_VERIFY_MODEL = "llava"        # the worker's --verify-model default
MODES = ("llm", "pipeline")

ABSENT_NOUNS = ["bicycle", "umbrella", "guitar", "traffic cone", "red balloon",
                "horse", "fire hydrant", "surfboard", "laptop", "kite", "parrot",
                "wheelbarrow", "snowman", "telescope", "vending machine"]
COLOURS = ["red", "blue", "green", "yellow", "white", "black", "orange", "purple",
           "pink", "brown", "gray", "grey"]
NUMBERS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10}
_NUM_WORD = {v: k for k, v in NUMBERS.items()}
_STOP = frozenset("a an the of and or in on at to with there is are some".split())


def _content(text):
    return {w for w in re.findall(r"[a-z]+", (text or "").lower()) if w not in _STOP}


# --------------------------------------------------------------------------
# Plant
# --------------------------------------------------------------------------

# One fixed sentence would let a verifier learn the SENTENCE instead of
# checking the photo, so the planted claim's wording varies too.
OBJECT_TEMPLATES = [
    "There is {a} {noun} in the background.",
    "{A} {noun} is visible near the edge of the frame.",
    "Off to one side sits {a} {noun}.",
    "{A} {noun} can be seen behind them.",
    "Nearby, {a} {noun} rests on the ground.",
]


def plant_object(text, rnd, used=None, tpl_rnd=None):
    """Append one absent object. `used` (a Counter shared across one build)
    makes the pick the LEAST-used eligible noun, so the same object is not
    planted four times (it was: 4 of 11 plants were 'parrot'). Note a change
    here DOES shift the colour/count plants drawn later from the same rnd —
    random.choice consumes a variable amount of randomness — so a rebuild keeps
    only confirmations whose planted text came out identical."""
    have = _content(text)
    choices = [n for n in ABSENT_NOUNS if not (_content(n) & have)]
    if used is not None:
        least = min(used.get(n, 0) for n in choices)
        choices = [n for n in choices if used.get(n, 0) == least]
    noun = rnd.choice(choices)
    if used is not None:
        used[noun] = used.get(noun, 0) + 1
    a = "an" if noun[0] in "aeiou" else "a"
    tpl = (tpl_rnd or rnd).choice(OBJECT_TEMPLATES) if tpl_rnd is not None else OBJECT_TEMPLATES[0]
    return text.rstrip() + " " + tpl.format(a=a, A=a.capitalize(), noun=noun), [noun]


def plant_colour(text, rnd):
    m = re.search(r"\b(" + "|".join(COLOURS) + r")\b(\s+[a-z]+)?", text, re.I)
    if not m:
        return None
    old = m.group(1).lower()
    new = rnd.choice([c for c in COLOURS if c not in (old, "gray" if old == "grey" else "grey")])
    new = new.capitalize() if m.group(1)[0].isupper() else new
    span = new.lower() + (m.group(2) or "")
    return text[:m.start(1)] + new + text[m.end(1):], [span.strip()]


def plant_count(text, rnd):
    m = re.search(r"\b(" + "|".join(NUMBERS) + r"|[2-9])\b(\s+[a-z]+)?", text, re.I)
    if not m:
        return None
    tok = m.group(1).lower()
    n = NUMBERS.get(tok) or int(tok)
    new_n = n + 2 if n + 2 <= 10 else n - 2
    new = _NUM_WORD[new_n] if tok in NUMBERS else str(new_n)
    new = new.capitalize() if m.group(1)[0].isupper() else new
    span = new.lower() + (m.group(2) or "")
    return text[:m.start(1)] + new + text[m.end(1):], [span.strip()]


PLANTERS = {"object": plant_object, "colour": plant_colour, "count": plant_count}


def description_errors(pid, sha, text, claims, disputes, dlabels):
    """(errors, spans) for one description from the owner's labels, or None
    when it has not been judged.

    Two sources, combined: 'All claims' labels (clicked segments) and the
    Differences labels (disputed facts — its WRONG answers are the spans). A
    description whose every disputed point is labelled and answered right or
    silent counts as clean: facts all the descriptions agreed on were never
    disputed, and are taken as right."""
    judged, errs, spans = False, 0, []
    lab = claims.get(sha)
    if me.claim_errors(lab) is not None:
        judged = True
        errs += me.claim_errors(lab)
        segs = me.segment_claims(text)
        spans += [segs[i].strip() for i in lab.get("wrong") or [] if i < len(segs)]
    d = disputes.get(str(pid))
    if d and sha in d.get("shas", []):
        qs = d["questions"]
        if qs and all(q["key"] in dlabels for q in qs):
            judged = True
            for q in qs:
                if me.dispute_verdict(q["answers"].get(sha), dlabels.get(q["key"])) == "wrong":
                    errs += 1
                    spans.append(q["answers"][sha])
    return (errs, spans) if judged else None


def build_sets(source, *, seed=13):
    """Sets from `source`'s descriptions and the owner's describe labels."""
    run = me.load_run("describe", source)
    if run is None:
        raise SystemExit(f"no describe run {source!r}")
    claims, disputes, dlabels = me.load_claims(), me.load_disputes(), me.load_dispute_labels()
    rnd = random.Random(seed)
    used = {}
    previous = {it["id"]: it for it in (me.load_verify_sets() or {}).get("items", [])}
    items = []
    order = ["object", "colour", "count"]
    k = 0
    for pid, it in sorted(run["items"].items(), key=lambda kv: int(kv[0])):
        text = it.get("text")
        if not text:
            continue
        sha = it.get("text_sha") or me.text_sha(text)
        judged = description_errors(pid, sha, text, claims, disputes, dlabels)
        if judged is None:
            continue
        errs, spans = judged
        if errs == 0:
            items.append({"id": f"clean-{pid}", "photo_id": int(pid), "kind": "clean",
                          "type": None, "text": text, "spans": [], "confirmed": None})
            want = order[k % 3]
            k += 1
            tpl_rnd = random.Random(f"{seed}-tpl-{pid}")
            planted = (plant_object(text, rnd, used, tpl_rnd) if want == "object"
                       else PLANTERS[want](text, rnd))
            if planted is None:
                want, planted = "object", plant_object(text, rnd, used, tpl_rnd)
            new_text, pspans = planted
            old = previous.get(f"planted-{pid}")
            # A confirmation is about one exact sentence: carry it over only
            # when the planted text is unchanged.
            kept = old.get("confirmed") if old and old.get("text") == new_text else None
            items.append({"id": f"planted-{pid}", "photo_id": int(pid), "kind": "planted",
                          "type": want, "text": new_text, "spans": pspans, "confirmed": kept})
        else:
            items.append({"id": f"real-{pid}", "photo_id": int(pid), "kind": "real",
                          "type": None, "text": text, "spans": spans, "confirmed": None})
    data = {"source_variant": source, "source_effective_model": run.get("effective_model"),
            "created": me.now_iso(), "seed": seed, "items": items}
    me.save_verify_sets(data)
    return data


def scored_items(sets):
    """Clean + real always; planted only once the owner confirmed it false."""
    return [it for it in sets["items"]
            if it["kind"] != "planted" or it.get("confirmed") is True]


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def production_checker(path, text, emb, nominal, llm_all):
    from photosearch import verify
    return verify.check_description(path, text, [], emb, verify_model=nominal,
                                    llm_all=llm_all)


def check_one(path, text, emb, nominal, mode, checker=None):
    from photosearch import describe
    checker = checker or production_checker
    rec = me.Recorder()
    t0 = time.time()
    with rec.active():
        try:
            chk = checker(path, text, emb, nominal, mode == "llm")
        except describe.UnusableAnswer as e:
            # The model answered, but no verdict can be stored (cut off at
            # the token limit): production records a failure row. Cached, so
            # it COUNTS in the denominators instead of silently shrinking them.
            return {"stage": "unusable", "unusable": str(e), "flagged": [], "raw_wrong": [],
                    "latency_s": round(time.time() - t0, 3), "attempts": rec.attempts()}
    latency = time.time() - t0
    # llm_verify_description returns [] on ANY error — the same as "ALL
    # CORRECT". A cleared result whose call errored is a dead backend, not a
    # verdict, and must not be cached as a perfect false-rejection score.
    rec.check(chk["stage"] == "llm_cleared", "llm_verify_description")
    raw = (rec.calls[-1].get("raw") if rec.calls else None) or ""
    raw_wrong = [m.group(1).strip() for m in re.finditer(r"(?im)^\s*WRONG:\s*(.+)$", raw)]
    return {"stage": chk["stage"], "flagged": [c["noun"] for c in chk["confirmed"]],
            "raw_wrong": raw_wrong, "latency_s": round(latency, 3),
            "attempts": rec.attempts()}


def run_variant(variant, *, model=None, mode="llm", db=None, server=me.DEFAULT_SERVER,
                limit=None, force=False, force_stale=False, solo=False, checker=None,
                fetch=None, log=print):
    from photosearch import describe
    if mode not in MODES:
        raise SystemExit(f"--mode must be one of {MODES}")
    sets = me.load_verify_sets()
    if sets is None:
        raise SystemExit("no verify sets — run `plant` first")
    me.pin_role_model("verify", model)
    effective = describe.effective_model(model or NOMINAL_VERIFY_MODEL, "verify")
    if effective == sets.get("source_effective_model"):
        raise SystemExit(f"verify model {effective!r} wrote these descriptions; the check "
                         "must be independent. Pick a different verify model.")
    conn = me.open_db_readonly(db) if mode == "pipeline" else None
    effort = me.sent_reasoning_effort(effective)
    spec = me.make_spec(me.spec_parts(PASS, mode=mode, reasoning_effort=effort))
    run = me.open_run(PASS, variant, force=force, spec=spec, force_stale=force_stale,
                      effective_model=effective, mode=mode,
                      source_variant=sets["source_variant"])
    run.setdefault("reasoning_effort", effort)
    todo = [it for it in scored_items(sets) if it["id"] not in run["items"]]
    if limit is not None:
        todo = todo[:limit]
    loaded_start = me.lmstudio_loaded()
    log(f"[run] verify {variant}: model={effective} mode={mode} — {len(todo)} to do")
    fc = me.FailureCounter(log)
    for i, it in enumerate(todo, 1):
        try:
            path = me.original_path(it["photo_id"], server, fetch)
            emb = me.clip_embedding(conn, it["photo_id"]) if conn is not None else None
            if mode == "pipeline" and emb is None:
                raise RuntimeError("no CLIP embedding (pipeline mode needs --db)")
            res = check_one(str(path), it["text"], emb, model or NOMINAL_VERIFY_MODEL,
                            mode, checker)
        except Exception as e:
            fc.fail(it["id"], e)
            continue
        fc.ok()
        run["items"][it["id"]] = res
        me.save_run(PASS, variant, run)
        log(f"  [{i}/{len(todo)}] {it['id']}: {res['stage']} {res['flagged'] or ''}")
    fc.summary()
    loaded_end = me.lmstudio_loaded()
    run["latency_label"] = me.latency_label(effective, loaded_start, loaded_end, solo)
    me.save_run(PASS, variant, run)
    return run


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

# Words that never say WHAT is wrong: dispute answers like "yes" / "no",
# counts, pronouns, prepositions, and nouns so generic that any two claims
# share them. Matching on these is how "post" once matched a thermometer flag.
_MATCH_IGNORE = frozenset("""
    yes no not none both either neither all any each
    one two three four five six seven eight nine ten several many few
    he him his she her hers they them their it its we us you i me my our
    behind beside near next above below under over front back left right top bottom
    into onto from off out up down through across along around between against
    as if while but also just only very appears appear seems seem visible present shown
    seen there here this that these those what which who something someone
    image photo picture scene background foreground frame side edge area part
    thing things object objects
    """.split()) | _STOP
_SUBPHRASE = re.compile(r"\s*(?:[,;/]|\band\b|\bor\b|\bwith\b|\bwhile\b|\bbut\b)\s*", re.I)


def _sing(w):
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if w == "children":
        return "child"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _match_words(phrase):
    """Content words of a phrase, singular, minus `_MATCH_IGNORE`."""
    return [_sing(w) for w in re.findall(r"[a-z]+", (phrase or "").lower())
            if len(w) > 2 and w not in _MATCH_IGNORE]


def head_nouns(phrase):
    """The last content word of each sub-phrase ("red collar and striped" ->
    {collar, striped}) — a crude head noun, no tagger needed."""
    heads = set()
    for sub in _SUBPHRASE.split(phrase or ""):
        ws = _match_words(sub)
        if ws:
            heads.add(ws[-1])
    return heads


def matchable(spans):
    """Can a flag name this error at all? "yes", "five" or "no" alone say
    nothing a verifier could repeat."""
    return any(head_nouns(s) for s in spans or ())


def matched(result, spans):
    """Does a flag NAME the error? Only the shipped `flagged` nouns count (the
    raw WRONG: sentences share words with anything), and a flag matches a span
    when the head noun of one is a content word of the other: "lighthouse"
    names "with a lighthouse behind him"; "two children" does not name "red
    collar and striped"."""
    flags = list(result.get("flagged") or [])
    if not flags or not spans:
        return False
    for f in flags:
        f_heads, f_words = head_nouns(f), set(_match_words(f))
        for s in spans:
            if (f_heads & set(_match_words(s))) or (head_nouns(s) & f_words):
                return True
    return False


def summarize(run, sets):
    """Every scored item counts, whatever happened to it: an item the run
    never cached (a transport failure, `missing`) or could not judge (an
    unusable verdict, `failed`) stays in the denominator as not caught — and,
    on the clean set, as not passed."""
    records = []
    lat = []
    for it in scored_items(sets):
        res = run["items"].get(it["id"])
        failed = res is not None and res.get("stage") == "unusable"
        answered = res is not None and not failed
        flags = list(res.get("flagged") or []) if answered else []
        if answered:
            lat.append(res.get("latency_s"))
        records.append({"kind": it["kind"], "type": it.get("type"),
                        "flagged": bool(flags), "n_flags": len(flags),
                        "matched": bool(flags) and matched(res, it["spans"]),
                        "matchable": it["kind"] == "clean" or matchable(it["spans"]),
                        "failed": failed, "missing": res is None})

    def tally(rs):
        return [len(rs), sum(r["flagged"] for r in rs), sum(r["matched"] for r in rs)]
    by = {k: [r for r in records if r["kind"] == k] for k in ("clean", "planted", "real")}
    planted_by_type = {}
    for r in by["planted"]:
        planted_by_type.setdefault(r["type"], []).append(r)
    return {"variant": run.get("variant"), "effective_model": run.get("effective_model"),
            "mode": run.get("mode"), "planted": tally(by["planted"]),
            "planted_by_type": {t: tally(rs) for t, rs in planted_by_type.items()},
            "real": tally(by["real"]), "clean": tally(by["clean"])[:2],
            "real_matchable": sum(r["matchable"] for r in by["real"]),
            "clean_flags": sum(r["n_flags"] for r in by["clean"]),
            "failed": {k: sum(r["failed"] for r in rs) for k, rs in by.items()},
            "missing": {k: sum(r["missing"] for r in rs) for k, rs in by.items()},
            "records": records,
            "lat_p50": me.median(lat[1:] if len(lat) > 1 else lat),
            "latency_label": run.get("latency_label", "unknown"),
            "spec": me.spec_status(PASS, run)}


def headline(records):
    """Matched planted catch rate minus the share of clean descriptions that
    did not pass (flagged, failed or never run), both over the full set."""
    planted = [r for r in records if r["kind"] == "planted"]
    clean = [r for r in records if r["kind"] == "clean"]
    if not planted or not clean:
        return None
    catch = sum(r["matched"] for r in planted) / len(planted)
    reject = sum(r["flagged"] or r["failed"] or r["missing"] for r in clean) / len(clean)
    return catch - reject


def build_report():
    sets = me.load_verify_sets()
    if sets is None:
        return None, []
    rows = []
    for v in me.list_variants(PASS):
        run = me.load_run(PASS, v)
        if run.get("source_variant") == sets["source_variant"]:
            rows.append(summarize(run, sets))
    return sets, rows


def render(sets, rows):
    if sets is None:
        return "no verify sets — run `plant` first"
    kinds = {}
    for it in sets["items"]:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    unconf = sum(1 for it in sets["items"] if it["kind"] == "planted" and it.get("confirmed") is None)
    out = [f"descriptions by {sets['source_effective_model']} ({sets['source_variant']}): "
           f"{kinds.get('clean', 0)} clean, {kinds.get('planted', 0)} planted "
           f"({unconf} awaiting confirmation), {kinds.get('real', 0)} with real errors", ""]
    out.append(f"{'variant':<16} {'verify model':<24} {'mode':<8} {'planted caught':>14} "
               f"{'matched':>7} {'real caught':>11} {'matched':>12} {'false reject':>12} "
               f"{'flags/clean':>11} {'failed':>6} {'s/ph':>5}  latency")
    for r in rows:
        p, rl, c = r["planted"], r["real"], r["clean"]
        failed = sum(r["failed"].values()) + sum(r["missing"].values())
        tag = me.stale_tag(r.get("spec"))
        out.append(f"{str(r['variant']):<16} {str(r['effective_model'])[:24]:<24} {r['mode']:<8} "
                   f"{me.fmt_ratio(p[1], p[0]):>14} {me.fmt_ratio(p[2], p[0]):>7} "
                   f"{me.fmt_ratio(rl[1], rl[0]):>11} "
                   f"{me.fmt_ratio(rl[2], r['real_matchable']) + ' of ' + str(r['real_matchable']):>12} "
                   f"{me.fmt_ratio(c[1], c[0]):>12} {me.fmt_ratio(r['clean_flags'], c[0]):>11} "
                   f"{failed:>6} "
                   f"{('%.1f' % r['lat_p50']) if r['lat_p50'] is not None else '—':>5}  "
                   f"{r['latency_label']}{'   ' + tag if tag else ''}")
        for t, (n, f, m) in sorted(r["planted_by_type"].items()):
            out.append(f"{'':<16} {'  planted ' + t + ' (n=' + str(n) + ')':<24} {'':<8} "
                       f"{me.fmt_ratio(f, n):>14} {me.fmt_ratio(m, n):>7}")
    out.append("")
    out.append("caught = flagged anything; matched = a SHIPPED flag (the cleaned `flagged` noun, "
               "never the raw WRONG: sentence) names the error by head noun, ignoring yes/no, "
               "counts and generic words. Real matched is over the errors a flag could name at "
               "all (\"of N\"; a bare \"yes\" or \"five\" cannot be). false reject = a clean "
               "description flagged; flags/clean = flags per clean description (a verifier that "
               "flags everything also 'catches' everything). failed = unusable verdicts + items "
               "never cached; both stay in every denominator. Only same-source pairings are "
               f"shown, and run refuses a verify model that wrote the descriptions. {me.STALE} = "
               "the run's spec hash is missing or differs from the current code.")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plant")
    pl.add_argument("--source", required=True, help="describe variant whose labelled "
                                                     "descriptions to use")
    pl.add_argument("--seed", type=int, default=13)
    pl.add_argument("--force", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--variant", required=True)
    r.add_argument("--model", help="LM Studio id; pins PHOTOSEARCH_LLM_VERIFY_MODEL.")
    r.add_argument("--mode", default="llm", choices=MODES)
    r.add_argument("--db", default=os.environ.get("PHOTOSEARCH_DB"))
    r.add_argument("--server", default=me.DEFAULT_SERVER)
    r.add_argument("--limit", type=int)
    r.add_argument("--force", action="store_true")
    r.add_argument("--force-stale", action="store_true",
                   help="append to a run cached under a different production spec "
                        "(it stays marked STALE)")
    r.add_argument("--solo", action="store_true")
    sub.add_parser("report")
    args = ap.parse_args(argv)
    if args.cmd == "plant":
        if me.load_verify_sets() is not None and not args.force:
            raise SystemExit("verify sets exist (confirmations are stored in them). "
                             "--force to rebuild.")
        data = build_sets(args.source, seed=args.seed)
        n = sum(1 for it in data["items"] if it["kind"] == "planted")
        print(f"[plant] {len(data['items'])} items, {n} planted — confirm them on "
              "/eval/models?tab=planted")
    elif args.cmd == "run":
        run_variant(args.variant, model=args.model, mode=args.mode, db=args.db,
                    server=args.server, limit=args.limit, force=args.force,
                    force_stale=args.force_stale, solo=args.solo)
    else:
        print(render(*build_report()))


if __name__ == "__main__":
    main()
