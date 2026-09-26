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

def plant_object(text, rnd):
    have = _content(text)
    choices = [n for n in ABSENT_NOUNS if not (_content(n) & have)]
    noun = rnd.choice(choices)
    article = "an" if noun[0] in "aeiou" else "a"
    return text.rstrip() + f" There is {article} {noun} in the background.", [noun]


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


def build_sets(source, *, seed=13):
    """Sets from `source`'s descriptions and the owner's claim labels."""
    run = me.load_run("describe", source)
    if run is None:
        raise SystemExit(f"no describe run {source!r}")
    labels = me.load_claims()
    rnd = random.Random(seed)
    items = []
    order = ["object", "colour", "count"]
    k = 0
    for pid, it in sorted(run["items"].items(), key=lambda kv: int(kv[0])):
        text = it.get("text")
        if not text:
            continue
        sha = it.get("text_sha") or me.text_sha(text)
        lab = labels.get(sha)
        errs = me.claim_errors(lab)
        if errs is None:
            continue
        if errs == 0:
            items.append({"id": f"clean-{pid}", "photo_id": int(pid), "kind": "clean",
                          "type": None, "text": text, "spans": [], "confirmed": None})
            want = order[k % 3]
            k += 1
            planted = PLANTERS[want](text, rnd)
            if planted is None:
                want, planted = "object", plant_object(text, rnd)
            new_text, spans = planted
            items.append({"id": f"planted-{pid}", "photo_id": int(pid), "kind": "planted",
                          "type": want, "text": new_text, "spans": spans, "confirmed": None})
        else:
            segs = me.segment_claims(text)
            spans = [segs[i].strip() for i in lab.get("wrong") or [] if i < len(segs)]
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
    checker = checker or production_checker
    rec = me.Recorder()
    t0 = time.time()
    with rec.active():
        chk = checker(path, text, emb, nominal, mode == "llm")
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
                limit=None, force=False, solo=False, checker=None, fetch=None, log=print):
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
    run = me.open_run(PASS, variant, force=force, effective_model=effective, mode=mode,
                      source_variant=sets["source_variant"])
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

def matched(result, spans):
    """A flag names the error: any content word shared between a flagged noun
    (or the raw WRONG: line) and a labelled/planted span."""
    want = set().union(*(_content(s) for s in spans)) if spans else set()
    if not want:
        return False
    said = set().union(*(_content(x) for x in result["flagged"] + result["raw_wrong"])) \
        if (result["flagged"] or result["raw_wrong"]) else set()
    return bool(want & said)


def summarize(run, sets):
    by_id = {it["id"]: it for it in scored_items(sets)}
    rows = {"clean": [0, 0], "real": [0, 0, 0]}
    planted = {}
    lat = []
    for iid, res in run["items"].items():
        it = by_id.get(iid)
        if it is None:
            continue                 # a plant the owner has since rejected
        flagged = bool(res["flagged"])
        lat.append(res.get("latency_s"))
        if it["kind"] == "clean":
            rows["clean"][0] += 1
            rows["clean"][1] += flagged
        elif it["kind"] == "real":
            rows["real"][0] += 1
            rows["real"][1] += flagged
            rows["real"][2] += flagged and matched(res, it["spans"])
        else:
            p = planted.setdefault(it["type"], [0, 0, 0])
            p[0] += 1
            p[1] += flagged
            p[2] += flagged and matched(res, it["spans"])
    tot = [sum(v[i] for v in planted.values()) for i in range(3)]
    return {"variant": run.get("variant"), "effective_model": run.get("effective_model"),
            "mode": run.get("mode"), "planted": tot, "planted_by_type": planted,
            "real": rows["real"], "clean": rows["clean"],
            "lat_p50": me.median(lat[1:] if len(lat) > 1 else lat),
            "latency_label": run.get("latency_label", "unknown")}


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
    out.append(f"{'variant':<14} {'verify model':<26} {'mode':<8} {'planted caught':>14} "
               f"{'matched':>7} {'real caught':>11} {'matched':>7} {'false reject':>12} "
               f"{'s/ph':>5}  latency")
    for r in rows:
        p, rl, c = r["planted"], r["real"], r["clean"]
        out.append(f"{str(r['variant']):<14} {str(r['effective_model'])[:26]:<26} {r['mode']:<8} "
                   f"{me.fmt_ratio(p[1], p[0]):>14} {me.fmt_ratio(p[2], p[0]):>7} "
                   f"{me.fmt_ratio(rl[1], rl[0]):>11} {me.fmt_ratio(rl[2], rl[0]):>7} "
                   f"{me.fmt_ratio(c[1], c[0]):>12} "
                   f"{('%.1f' % r['lat_p50']) if r['lat_p50'] is not None else '—':>5}  "
                   f"{r['latency_label']}")
        for t, (n, f, m) in sorted(r["planted_by_type"].items()):
            out.append(f"{'':<14} {'  planted ' + t:<26} {'':<8} {me.fmt_ratio(f, n):>14} "
                       f"{me.fmt_ratio(m, n):>7}")
    out.append("")
    out.append("caught = flagged anything; matched = a flag names the actual error. "
               "false reject = a clean description flagged. Only same-source pairings are "
               "shown, and run refuses a verify model that wrote the descriptions.")
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
                    server=args.server, limit=args.limit, force=args.force, solo=args.solo)
    else:
        print(render(*build_report()))


if __name__ == "__main__":
    main()
