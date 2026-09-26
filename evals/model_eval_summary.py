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

Headline metric per role (higher is better; s/photo is always solo when
measured that way — check the latency column):
  describe          clean-description rate (owner) — else 1 - bad-output share
  category-content  precision (owner) — else 1 - unsupported share (lenient)
  keywords          precision (owner) — else 1 - unsupported share
  verify            matched planted catch rate - false-rejection rate
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
# Collect: {role: [{"model", "variant", "headline", "s_per_photo", "latency",
#                   "detail", "flags"}]}
# --------------------------------------------------------------------------

def collect_describe(db):
    rows = []
    for r in _script("describe_eval").build_report(db=db, use_clip=False):
        if r["variant"] == me.STORED:
            continue
        n = r["n"] or 0
        bad = (n - r["answered"]) + r["degenerate_final"] + r["truncated"]
        if r.get("claims_n"):
            head, basis = _ratio(r["clean"], r["claims_n"]), f"clean {r['clean']}/{r['claims_n']}"
        else:
            head, basis = (1 - bad / n) if n else None, "no labels: 1 - bad output"
        rows.append({"model": r["effective_model"], "variant": r["variant"], "headline": head,
                     "s_per_photo": r["lat_p50"], "latency": r["latency_label"],
                     "detail": basis, "flags": ["SCREEN OUT"] if r["screen_out"] else []})
    return rows


def collect_text(db):
    out = {"category-content": [], "keywords": []}
    for r in _script("text_passes_eval").build_report(db=db):
        if r["variant"] == me.STORED:
            continue
        if r["pass"] == "category-content" and r.get("labelled"):
            head = _ratio(r["tp"], r["tp"] + r["fp"])
            basis = f"precision, {r['labelled']} labelled"
        elif r["pass"] == "keywords" and r.get("labelled"):
            head = _ratio(r["kw_right"], r["kw_right"] + r["kw_wrong"])
            basis = f"precision, {r['labelled']} labelled"
        else:
            key = "unsupported_lenient" if r["pass"] == "category-content" else "unsupported"
            head = 1 - r[key] / r["tags"] if r["tags"] else None
            basis = "no labels: 1 - unsupported"
        flags = []
        if r["attempts"] and r["timeouts"] / r["attempts"] > 0.05:
            flags.append(f"{100 * r['timeouts'] / r['attempts']:.0f}% attempts time out")
        out[r["pass"]].append({"model": r["effective_model"], "variant": r["variant"],
                               "headline": head, "s_per_photo": r["lat_p50"],
                               "latency": r["latency_label"], "detail": basis,
                               "flags": flags})
    return out


def collect_verify():
    V = _script("verify_eval")
    sets, rows = V.build_report()
    out = []
    for r in rows:
        p, c = r["planted"], r["clean"]
        catch, reject = _ratio(p[2], p[0]), _ratio(c[1], c[0])
        head = catch - reject if catch is not None and reject is not None else None
        out.append({"model": r["effective_model"], "variant": r["variant"], "headline": head,
                    "s_per_photo": r["lat_p50"], "latency": r["latency_label"],
                    "detail": f"{r['mode']}: catch {catch if catch is None else round(catch, 2)}"
                              f" − reject {reject if reject is None else round(reject, 2)}",
                    "flags": [], "describe_source": sets["source_effective_model"]})
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
                    "headline": r["rho"], "s_per_photo": r["lat_median"],
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

def pareto(rows):
    """Rows no other row beats on headline AND s/photo (ties don't dominate)."""
    ok = [r for r in rows if r["headline"] is not None and r["s_per_photo"] is not None]
    keep = []
    for r in ok:
        dominated = any(o is not r and o["headline"] >= r["headline"]
                        and o["s_per_photo"] <= r["s_per_photo"]
                        and (o["headline"] > r["headline"] or o["s_per_photo"] < r["s_per_photo"])
                        for o in ok)
        if not dominated:
            keep.append(r)
    return keep


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
        out.append(f"  {'model':<30} {'variant':<16} {'headline':>8} {'s/photo':>8} "
                   f"{'VRAM':>6} {'fits':>5}  {'basis':<34} flags")
        for r in sorted(rows, key=lambda r: -(r["headline"] if r["headline"] is not None else -9)):
            vram = (models.get(r["model"]) or {}).get("vram_gb")
            fit = fits_alone(r["model"], models, headroom)
            flags = list(r["flags"])
            if role == "verify" and r.get("describe_source") == r["model"]:
                flags.append("SAME AS DESCRIBE")
            out.append(
                f"{'*' if id(r) in front else ' '} {str(r['model'])[:30]:<30} {str(r['variant'])[:16]:<16} "
                f"{('%.3f' % r['headline']) if r['headline'] is not None else '—':>8} "
                f"{('%.2f' % r['s_per_photo']) if r['s_per_photo'] is not None else '—':>8} "
                f"{('%.1f' % vram) if vram is not None else '?':>6} "
                f"{'?' if fit is None else 'yes' if fit else 'NO':>5}  {r['detail'][:34]:<34} "
                f"{'; '.join(flags)}  [{r['latency']}]")
        out.append("")
    out.append("* = Pareto (nothing is both better and faster). fits = VRAM alone ≤ "
               f"{GPU_GB:.0f} GB - {headroom} GB headroom; '?' = fill in models.json.")
    if assign is None:
        assign = {}
        for role, rows in roles.items():
            ok = [r for r in rows if r["headline"] is not None
                  and fits_alone(r["model"], models, headroom) is not False
                  and not (role == "verify" and r.get("describe_source") == r["model"])]
            if ok:
                assign[role] = max(ok, key=lambda r: r["headline"])["model"]
        label = "EXAMPLE — best headline per role, not a recommendation"
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
