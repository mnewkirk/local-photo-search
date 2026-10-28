#!/usr/bin/env python
"""Cross-pass summary — one table per role, from every harness's cached runs.

Production runs ONE model loaded at a time with the passes in sequence
(owner decision 2026-09-26, docs/plans/model-eval-harnesses.md), so each
role is chosen independently: a candidate only has to fit on the GPU alone,
and verify must not be the model that wrote the descriptions. Nothing here
picks a winner — it lays out the evidence and marks the Pareto set (no other
model is both better on the headline metric and faster).

  init-models   write <eval dir>/models.json listing every model seen in a
                run, for the owner to fill in during Phase 0:
                    {"<id>": {"vram_gb": 9.8, "swap_s": 12, "ctx": 8192,
                              "reasoning_effort": "none", "sees_images": true}}
  report        the tables, plus a sequential fleet-time estimate:
                    python evals/model_eval_summary.py report --db photo_index.db.local
                    python evals/model_eval_summary.py report \\
                        --assign describe=qwen/qwen3.5-9b,verify=google/gemma-4-e2b,...

Headline metric per role (higher is better; over the FULL sample, so a
deferred / unusable / failed item counts against the model; 95% CI from a
seeded per-photo bootstrap; s/photo is always solo when measured that way —
check the latency column):
  describe          right on the ORIGINAL disputed questions (owner; an
                    unanswered photo's questions count wrong) — else claims
                    clean rate — else 1 - bad-output share (the report's screen)
  category-content  F1 of precision / pooled recall (owner), replayed with the
                    current guards — else 1 - unsupported share (lenient)
  keywords          per-photo precision (owner), replayed — else 1 - unsupported
  verify            matched planted catch rate - clean not-passed rate
  visual            F1 of precision / recall on the owner's 60 labels
  aesthetics        Spearman ρ vs the hand ranking (a gap < 0.1 is a tie)
"""
import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

from photosearch import model_eval as me  # noqa: E402

GPU_GB = 24.0
DEFAULT_BATCH = 1373          # the 2026-09-19 shoot
# Production order when passes run one model at a time: text passes need the
# description; verify reads it too.
PASS_ORDER = ["describe", "category-content", "keywords", "verify", "category-visual",
              "aesthetics"]
ROLE_OF_PASS = {"describe": "describe", "category-content": "category-content",
                "keywords": "keywords", "verify": "verify",
                "category-visual": "visual", "aesthetics": "aesthetics"}


def _script(name):
    spec = importlib.util.spec_from_file_location(f"summary_{name}",
                                                  os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ratio(num, den):
    return num / den if den else None


# --------------------------------------------------------------------------
# Collect: {role: [{"model", "variant", "headline", "ci", "n_answered",
#                   "n_full", "s_per_photo", "latency", "detail", "flags",
#                   "stale"}]}
#
# Every headline is over the FULL sample (a deferred / unusable / failed item
# counts against the model), with a seeded per-photo bootstrap 95% CI.
# --------------------------------------------------------------------------

def _stale_flags(r):
    tag = me.stale_tag(r.get("spec"))
    return [tag] if tag else []


def collect_describe(db):
    D = _script("describe_eval")
    rows = []
    for r in D.build_report(db=db, use_clip=False):
        if r["variant"] == me.STORED:
            continue
        n = r["n"] or 0
        recs = r.get("disp_records") or []
        orig = r.get("disp_orig_right", 0) + r.get("disp_orig_wrong", 0)
        ci = None
        if orig:
            # The ORIGINAL questions are the ones every model faced; points
            # added for one model scrutinise only its extra details
            # (qwenvl7b: 0.60 on all points, 0.50 on the original ones).
            head = D.fair_score(recs, original=True)
            ci = me.bootstrap_ci(recs, lambda s: D.fair_score(s, original=True))
            allp = D.fair_score(recs, original=False)
            basis = (f"orig qs right {r['disp_orig_right']}/{orig}"
                     f" (all pts {allp:.2f})" if allp is not None else "")
        elif r.get("claims_n"):
            head, basis = _ratio(r["clean"], r["claims_n"]), f"clean {r['clean']}/{r['claims_n']}"
        else:
            head, basis = (1 - r["bad"] / n) if n else None, "no labels: 1 - bad output"
        flags = ["SCREEN OUT"] if r["screen_out"] else []
        if r.get("replay_screen_out"):
            flags.append(f"SCREEN OUT replayed ({r['replay_bad']}/{n} bad now)")
        rows.append({"model": r["effective_model"], "variant": r["variant"], "headline": head,
                     "proxy": not orig and not r.get("claims_n"),
                     "ci": ci, "n_answered": r["answered"], "n_full": len(recs) or n,
                     "s_per_photo": r["lat_p50"], "latency": r["latency_label"],
                     "detail": basis, "flags": flags + _stale_flags(r),
                     "stale": bool((r.get("spec") or {}).get("stale"))})
    return rows


def collect_text(db):
    X = _script("text_passes_eval")
    out = {"category-content": [], "keywords": []}
    for r in X.build_report(db=db):
        if r["variant"] == me.STORED:
            continue
        flags = []
        ci = None
        n_ans, n_full = r["n"] - r["deferred"], r["n_full"]
        # Score what TODAY's code would store from the cached answers when it
        # can be replayed; the as-run number goes in the basis.
        g = r.get("replay") or r
        how = "replayed" if r.get("replay") else "as run"
        if r["pass"] == "category-content" and r.get("labelled"):
            recs = g["records"]
            p, rc, head = X.category_scores(recs)
            ci = me.bootstrap_ci(recs, X.category_f1)
            raw_f1 = X.category_scores(r["records"])[2]
            basis = (f"F1 {how}: P {p and round(p, 2)} R {rc and round(rc, 2)}"
                     f" (as run F1 {raw_f1 and round(raw_f1, 2)})")
            n_ans, n_full = g["answered_labelled"], g["labelled"]
            if g.get("precision_approx"):
                flags.append(f"precision approx ({g['unjudged']} unjudged)")
        elif r["pass"] == "keywords" and r.get("labelled"):
            recs = g["records"]
            head = X.keyword_photo_precision(recs)
            ci = me.bootstrap_ci(recs, X.keyword_photo_precision)
            pooled = _ratio(g["kw_right"], g["kw_right"] + g["kw_wrong"])
            basis = f"per-photo precision {how} (pooled {pooled and round(pooled, 2)})"
            n_ans, n_full = g["answered_labelled"], g["labelled"]
        else:
            key = "unsupported_lenient" if r["pass"] == "category-content" else "unsupported"
            head = 1 - r[key] / r["tags"] if r["tags"] else None
            basis = "no labels: 1 - unsupported"
        if r["attempts"] and r["timeouts"] / r["attempts"] > 0.05:
            flags.append(f"{100 * r['timeouts'] / r['attempts']:.0f}% attempts time out")
        out[r["pass"]].append({"model": r["effective_model"], "variant": r["variant"],
                               "headline": head, "proxy": basis.startswith("no labels"),
                               "ci": ci, "n_answered": n_ans,
                               "n_full": n_full, "s_per_photo": r["lat_p50"],
                               "latency": r["latency_label"], "detail": basis,
                               "flags": flags + _stale_flags(r),
                               "stale": bool((r.get("spec") or {}).get("stale"))})
    return out


def collect_verify():
    V = _script("verify_eval")
    sets, rows = V.build_report()
    out = []
    for r in rows:
        p, c = r["planted"], r["clean"]
        recs = r["records"]
        head = V.headline(recs)
        ci = me.bootstrap_ci(recs, V.headline, strata=lambda x: x["kind"])
        failed = sum(r["failed"].values()) + sum(r["missing"].values())
        flags = [f"{failed} failed/not run"] if failed else []
        out.append({"model": r["effective_model"], "variant": r["variant"], "headline": head,
                    "ci": ci, "n_answered": len(recs) - failed, "n_full": len(recs),
                    "s_per_photo": r["lat_p50"], "latency": r["latency_label"],
                    "detail": f"{r['mode']}: catch {p[2]}/{p[0]} − reject {c[1]}/{c[0]}"
                              f", {_ratio(r['clean_flags'], c[0]) or 0:.2f} flags/clean",
                    "flags": flags + _stale_flags(r),
                    "stale": bool((r.get("spec") or {}).get("stale")),
                    "describe_source": sets["source_effective_model"]})
    return out


def collect_visual():
    T = _script("visual_tags_eval")
    from photosearch import visual_tag_eval as store
    labels = store.scoreable_labels()
    out = []
    if not labels:
        return out
    for v in T.list_variants():
        run = T.load_run(v)
        preds = T.predictions_of(run)
        rep = T.build_report({v: preds}, labels, extras={v: run.get("extra_vocab") or []})
        o = rep["results"][v]["overall"]
        prec = _ratio(o["tp"], o["tp"] + o["fp"])
        rec = _ratio(o["tp"], o["tp"] + o["fn"])
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
        lat = me.latencies(run.get("predictions", {}))
        out.append({"model": run.get("effective_model"), "variant": v, "headline": f1,
                    "ci": None,          # visual_tags_eval reports no per-photo CI
                    "s_per_photo": me.median(lat),
                    "latency": run.get("latency_label", "not recorded"),
                    "detail": f"P {prec and round(prec, 2)} / R {rec and round(rec, 2)}"
                              f"{' (' + run['prompt_file'] + ')' if run.get('prompt_file') else ''}",
                    "flags": []})
    return out


def collect_aesthetics(out_dir):
    A = _script("aesthetics_bakeoff")
    v2 = me.read_json(os.path.join(out_dir, "scores-v2.json"), {})
    gt_path = os.path.join(out_dir, "ranked.csv")
    gt = A.load_ground_truth(gt_path) if os.path.exists(gt_path) else {}
    out = []
    for variant, entry in v2.items():
        r = A.vlm_summary(entry, gt)
        flags = []
        if r["n"] and r["final_fail"] / r["n"] > 0.05:
            flags.append(f"{r['final_fail']}/{r['n']} unparsed")
        out.append({"model": entry.get("effective_model"), "variant": variant,
                    "headline": r["rho"], "ci": r.get("rho_ci"),
                    "s_per_photo": r["lat_median"],
                    "latency": r["latency_label"],
                    "detail": f"spread std {r['spread']['std'] and round(r['spread']['std'], 2)}",
                    "flags": flags})
    return out


def collect(db=None, aesthetics_dir=None):
    roles = {"describe": collect_describe(db)}
    roles.update(collect_text(db))
    roles["verify"] = collect_verify()
    roles["visual"] = collect_visual()
    roles["aesthetics"] = collect_aesthetics(
        aesthetics_dir or os.path.join(HERE, "aesthetics-bakeoff"))
    return roles


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def _sig_better(o, r):
    """o's headline beats r's by more than the uncertainty: the CIs do not
    overlap. With a CI missing on either side, a plain > (the old rule)."""
    if o.get("ci") and r.get("ci"):
        return o["ci"][0] > r["ci"][1]
    return o["headline"] > r["headline"]


def rankable(rows):
    """The rows a role's ranking may use. A proxy headline (no owner labels:
    1 - bad output, 1 - unsupported) is not on the same scale as a labelled
    one: an unlabelled re-run scored 1.000 for describe against 0.75 for the
    labelled best and became the example pick. When any row is labelled, the
    proxies are shown but not ranked; with none labelled, proxies rank among
    themselves."""
    real = [r for r in rows if r["headline"] is not None and not r.get("proxy")]
    return real or [r for r in rows if r["headline"] is not None]


def pareto(rows):
    """Rows no other row beats on headline AND s/photo. A headline lead
    inside the 95% CI is not a lead: a row within the CI of a faster one is
    dominated, so no star is awarded on noise."""
    ok = [r for r in rankable(rows) if r["s_per_photo"] is not None]
    keep = []
    for r in ok:
        dominated = any(o is not r and not _sig_better(r, o)
                        and o["s_per_photo"] <= r["s_per_photo"]
                        and (_sig_better(o, r) or o["s_per_photo"] < r["s_per_photo"])
                        for o in ok)
        if not dominated:
            keep.append(r)
    return keep


def ties_with_best(rows):
    """{id(row)} of rows whose CI overlaps the best headline's (not the best
    itself): a statistical tie, not a loss."""
    ok = rankable(rows)
    if not ok:
        return set()
    best = max(ok, key=lambda r: r["headline"])
    return {id(r) for r in ok if r is not best and r.get("ci") and best.get("ci")
            and me.ci_overlap(r["ci"], best["ci"])}


def fits_alone(model, models, headroom):
    info = models.get(model) or {}
    vram = info.get("vram_gb")
    if vram is None:
        return None
    return vram <= GPU_GB - headroom


def schedule(assign, roles, models, photos=DEFAULT_BATCH):
    """Sequential fleet time: Σ photos × s/photo per pass, plus one model load
    each time the model changes between consecutive passes. Returns
    (total_s, lines, problems)."""
    total, lines, problems, prev = 0.0, [], [], None
    for pass_ in PASS_ORDER:
        role = ROLE_OF_PASS[pass_]
        model = assign.get(role)
        if not model:
            problems.append(f"no model assigned for {role}")
            continue
        cands = [r for r in roles.get(role, []) if r["model"] == model and r["s_per_photo"]]
        if not cands:
            problems.append(f"{role}: no measured run for {model}")
            continue
        sp = min(c["s_per_photo"] for c in cands)
        swap = 0.0
        if model != prev:
            swap = (models.get(model) or {}).get("swap_s")
            if swap is None:
                problems.append(f"{model}: swap_s unknown in models.json (counted as 0)")
                swap = 0.0
        total += photos * sp + swap
        lines.append(f"  {pass_:<17} {model:<30} {photos} × {sp:.2f}s + load {swap:.0f}s "
                     f"= {(photos * sp + swap) / 60:.1f} min")
        prev = model
    if assign.get("verify") and assign.get("verify") == assign.get("describe"):
        problems.append("verify and describe are the same model — the check is not independent")
    return total, lines, problems


def _ci(ci):
    return f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci else "—"


def render(roles, models, *, headroom=1.5, assign=None, photos=DEFAULT_BATCH):
    out = []
    for role in ["describe", "category-content", "keywords", "verify", "visual", "aesthetics"]:
        rows = roles.get(role) or []
        out.append(f"=== {role} ===")
        if not rows:
            out.append("  no runs yet")
            out.append("")
            continue
        front = {id(r) for r in pareto(rows)}
        tied = ties_with_best(rows)
        out.append(f"  {'model':<30} {'variant':<16} {'headline':>9} {'95% CI':>13} {'ans/n':>7} "
                   f"{'s/photo':>8} {'VRAM':>6} {'fits':>5}  {'basis':<44} flags")
        for r in sorted(rows, key=lambda r: -(r["headline"] if r["headline"] is not None else -9)):
            vram = (models.get(r["model"]) or {}).get("vram_gb")
            fit = fits_alone(r["model"], models, headroom)
            flags = list(r["flags"])
            if role == "verify" and r.get("describe_source") == r["model"]:
                flags.append("SAME AS DESCRIBE")
            head = ("%.3f" % r["headline"]) if r["headline"] is not None else "—"
            if id(r) in tied:
                head = "≈" + head
            if r.get("proxy") and r not in rankable(rows):
                flags.append("UNLABELLED: proxy score, not ranked")
            ansn = (f"{r['n_answered']}/{r['n_full']}" if r.get("n_full") is not None else "—")
            out.append(
                f"{'*' if id(r) in front else ' '} {str(r['model'])[:30]:<30} {str(r['variant'])[:16]:<16} "
                f"{head:>9} {_ci(r.get('ci')):>13} {ansn:>7} "
                f"{('%.2f' % r['s_per_photo']) if r['s_per_photo'] is not None else '—':>8} "
                f"{('%.1f' % vram) if vram is not None else '?':>6} "
                f"{'?' if fit is None else 'yes' if fit else 'NO':>5}  {r['detail'][:44]:<44} "
                f"{'; '.join(flags)}  [{r['latency']}]")
        out.append("")
    out.append("* = Pareto (nothing is both better and faster; a lead inside the 95% CI is not "
               "better). ≈ = CI overlaps the best headline's: a tie, not a loss. 95% CI = per-photo "
               "bootstrap (seeded). ans/n = answered / full sample — failures count against the "
               "model. fits = VRAM alone ≤ "
               f"{GPU_GB:.0f} GB - {headroom} GB headroom; '?' = fill in models.json. "
               f"{me.STALE} = measured against older production prompts/guards — re-run before "
               "trusting it.")
    if assign is None:
        assign = {}
        for role, rows in roles.items():
            ok = [r for r in rankable(rows)
                  if fits_alone(r["model"], models, headroom) is not False
                  and not (role == "verify" and r.get("describe_source") == r["model"])]
            if ok:
                assign[role] = max(ok, key=lambda r: r["headline"])["model"]
        label = ("EXAMPLE — best headline per role, not a recommendation; check the ≈ ties "
                 "and STALE flags above")
    else:
        label = "as assigned"
    total, lines, problems = schedule(assign, roles, models, photos)
    out.append("")
    out.append(f"Sequential fleet time for a {photos}-photo batch ({label}):")
    out += lines
    out.append(f"  total ≈ {total / 3600:.1f} h (verify's regeneration of rejected "
               "descriptions is not included)")
    out += [f"  ! {p}" for p in problems]
    return "\n".join(out)


def all_models(roles):
    return sorted({r["model"] for rows in roles.values() for r in rows if r.get("model")})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    db_default = os.environ.get("PHOTOSEARCH_DB")
    i = sub.add_parser("init-models")
    i.add_argument("--db", default=db_default)
    r = sub.add_parser("report")
    r.add_argument("--db", default=db_default)
    r.add_argument("--aesthetics-dir")
    r.add_argument("--headroom", type=float, default=1.5)
    r.add_argument("--photos", type=int, default=DEFAULT_BATCH)
    r.add_argument("--assign", help="role=model,... (roles: describe, category-content, "
                                    "keywords, verify, visual, aesthetics)")
    args = ap.parse_args(argv)
    path = me.eval_dir() / "models.json"
    if args.cmd == "init-models":
        models = me.read_json(path, {})
        for m in all_models(collect(args.db)):
            models.setdefault(m, {"vram_gb": None, "swap_s": None, "ctx": None,
                                  "reasoning_effort": None, "sees_images": None})
        me.write_json_atomic(path, models)
        print(f"[models] {len(models)} models in {path} — fill in vram_gb (loaded alone) "
              "and swap_s (cold load time)")
        return
    assign = None
    if args.assign:
        assign = dict(kv.split("=", 1) for kv in args.assign.split(",") if "=" in kv)
    roles = collect(args.db, args.aesthetics_dir)
    print(render(roles, me.read_json(path, {}), headroom=args.headroom, assign=assign,
                 photos=args.photos))


if __name__ == "__main__":
    main()
