#!/usr/bin/env python3
"""Read-only eval of `photosearch.category_grounding` over the replica DB.

Step 2 of docs/plans/category-grounding.md (the ship gate). No model calls,
no writes: opens `file:photo_index.db.local?mode=ro` and reports what the
rules WOULD do.

    ./.venv/bin/python evals/category_grounding_eval.py [--db photo_index.db.local] [--seed 7]

`score` is a separate entry point (not folded into the flags above): prints
the owner-labelling score of the holdout sample
(`evals/category-grounding/samples-holdout-seed11.json`, labelled at
`/eval/category-grounding`) — the same table `GET
/api/eval/category-grounding/score` serves to the labelling page, since both
call `photosearch.category_grounding_eval_api.compute_score`:

    ./.venv/bin/python evals/category_grounding_eval.py score

Queries:
  Q1  current gemma rows: stored categories == a gemma generation (and,
      separately, every gemma generation joined to the CURRENT description)
  Q2  llama generations, split at >= 53 / 20-52 / < 20 tags
  Q3  backfill preview: every stored row attributed to its latest
      category-content generation, eligible when stored set <= generation set
  Q4  the committed fixtures (251357 at temp 0 / 0.4, 251356, 251358)

Answers logged after grounding shipped are already grounded, so re-grounding
them shows nothing. Each query therefore reads the `category-content-raw`
generation (the answer before grounding, logged by worker_api.submit_results
in the same batch, immediately before the grounded row: id = g.id - 1) in
place of the logged list wherever one exists (`_RAW_JOIN`, `_ANSWER`).

Writes evals/category-grounding/summary.md (committed) and samples.json
(git-ignored). Never SELECTs from the vec0 tables.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

# The model's answer before grounding, where the server logged one (see the
# module docstring); else the logged list. Grounded rows from after the ship
# then still show what the rules removed.
_RAW_JOIN = ("LEFT JOIN generations r ON r.photo_id = g.photo_id "
             "AND r.text_type = 'category-content-raw' AND r.id = g.id - 1 ")
_ANSWER = "COALESCE(r.generated_text, g.generated_text)"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from photosearch import category_grounding as cg  # noqa: E402

OUT = ROOT / "evals" / "category-grounding"
LLAMA = ("llama3.2:3b", "llama-3.2-3b-instruct")
_SENT = re.compile(r"(?<=[.!?])\s+")


def _load(text):
    try:
        v = json.loads(text) if text else None
    except (ValueError, TypeError):
        return None
    return [t for t in v if isinstance(t, str)] if isinstance(v, list) else None


def _sentence(desc: str, m) -> str:
    """The sentence of `desc` containing match `m`, with the cue bracketed."""
    if m is None:
        return "no cue"
    bounds = [0] + [x.end() for x in _SENT.finditer(desc)] + [len(desc)]
    for a, b in zip(bounds, bounds[1:]):
        if a <= m.start() < b:
            return f"[{m.group(0)}] {desc[a:b].strip()}"
    return f"[{m.group(0)}] {desc[:200]}"


def _group_of(name):
    return next(g for g in cg.GROUPS if g.name == name)


def _evidence(desc: str, term: str, reason: str):
    """(matched cue sentence or 'no cue', matched cue text or None)."""
    gname, _, rest = reason.partition(": ")
    g = _group_of(gname)
    if "none cued" in rest:
        return "no cue", None
    if " weakly cued" in rest:
        cued_names = [n.strip() for n in rest.split(" weakly cued")[0].split(",")]
        cues = [c for s in g.sides if s.name in cued_names for c in s.partial_cues]
    else:
        cued_names = [n.strip() for n in rest.split(" cued")[0].split(",")]
        cues = [c for s in g.sides if s.name in cued_names for c in s.cues + s.partial_cues]
    m = cg.first_cue_match(cg.scrub(desc), cues)   # same offsets as desc
    return _sentence(desc, m), (m.group(0).lower() if m else None)


class Report:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, s: str = ""):
        print(s)
        self.lines.append(s)

    def table(self, header, rows):
        self("| " + " | ".join(header) + " |")
        self("|" + "|".join("---" for _ in header) + "|")
        for r in rows:
            self("| " + " | ".join(str(x) for x in r) + " |")
        self()


def _trunc(s: str, n: int = 170) -> str:
    s = s.replace("|", "/").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def analyse_groups(rows, rep: Report, rng: random.Random, label: str, samples: dict):
    """rows: [(photo_id, description, categories)]"""
    per_group_rows = Counter()
    per_group_tags = Counter()
    per_group_terms = defaultdict(Counter)
    per_group_cue = defaultdict(Counter)
    per_group_drops = defaultdict(list)
    drop_hist = Counter()
    before = Counter()
    after = Counter()
    total_drop = 0
    rows_hit = 0
    for pid, desc, cats in rows:
        before.update(cats)
        g = cg.ground(desc, cats)
        after.update(g.kept)
        n = len(g.dropped)
        drop_hist[n] += 1
        total_drop += n
        rows_hit += 1 if n else 0
        groups_here = set()
        for term, reason in g.dropped.items():
            gname = reason.split(":")[0]
            groups_here.add(gname)
            per_group_tags[gname] += 1
            per_group_terms[gname][term] += 1
            sent, cue = _evidence(desc, term, reason)
            per_group_cue[gname][cue or "(none cued)"] += 1
            per_group_drops[gname].append((pid, term, reason, sent))
        for gname in groups_here:
            per_group_rows[gname] += 1

    n_tags = sum(before.values())
    rep(f"### {label}: {len(rows):,} rows, {n_tags:,} tags")
    rep()
    rep(f"Rows with at least one drop: **{rows_hit:,}** ({rows_hit / max(1, len(rows)):.1%}); "
        f"tags dropped: **{total_drop:,}** ({total_drop / max(1, n_tags):.2%}).")
    rep()
    rep.table(["group", "rows hit", "tags dropped", "top terms", "rival cue that fired (top)"],
              [(gn.name, per_group_rows[gn.name], per_group_tags[gn.name],
                ", ".join(f"{t} {c}" for t, c in per_group_terms[gn.name].most_common(6)),
                ", ".join(f"{t} {c}" for t, c in per_group_cue[gn.name].most_common(8)))
               for gn in cg.GROUPS])
    rep("Drop-count distribution (tags dropped per row):")
    rep()
    rep.table(["dropped", "rows"], sorted(drop_hist.items()))
    return {"per_group_drops": per_group_drops, "before": before, "after": after,
            "rows_hit": rows_hit, "total_drop": total_drop,
            "per_group_rows": per_group_rows, "per_group_tags": per_group_tags}


def print_group_samples(res, rep: Report, rng: random.Random, samples: dict, key: str, k=25):
    for g in cg.GROUPS:
        drops = res["per_group_drops"][g.name]
        pick = rng.sample(drops, min(k, len(drops)))
        samples.setdefault(key, {})[g.name] = [
            {"photo_id": p, "term": t, "reason": r, "evidence": s} for p, t, r, s in pick]
        rep(f"#### {g.name}: {len(pick)} of {len(drops)} drops (seeded random)")
        rep()
        rep.table(["photo", "term", "reason", "cue sentence"],
                  [(p, t, r, _trunc(s)) for p, t, r, s in pick])


def write_holdout(res, seed: int, sample_a: dict, rep: Report, k=25):
    """Sample B: 25 drops per group, disjoint from sample A, for owner labels."""
    rng = random.Random(seed)
    out = {"_note": (f"HOLDOUT candidate (seed {seed}) for owner labelling: 25 random drops "
                     "per group from current gemma rows, disjoint from the seed-7 tuning "
                     "sample. Label each drop correct / wrong / debatable; a group ships "
                     "only at >= 90% correct (docs/plans/category-grounding.md)."),
           "seed": seed, "groups": {}}
    for g in cg.GROUPS:
        taken = {(x["photo_id"], x["term"]) for x in sample_a.get(g.name, [])}
        pool = [d for d in res["per_group_drops"][g.name] if (d[0], d[1]) not in taken]
        pick = rng.sample(pool, min(k, len(pool)))
        out["groups"][g.name] = [{"photo_id": p, "term": t, "reason": r, "evidence": s,
                                  "label": None} for p, t, r, s in pick]
    path = OUT / f"samples-holdout-seed{seed}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    rep(f"HOLDOUT candidate for owner labelling: `{path.name}` (seed {seed}, "
        + ", ".join(f"{g.name} {len(out['groups'][g.name])}" for g in cg.GROUPS)
        + "; disjoint from the samples above).")
    rep()


def before_after(res, rep: Report, top=40):
    deltas = [(t, res["before"][t], res["after"][t], res["after"][t] - res["before"][t])
              for t in res["before"] if res["after"][t] != res["before"][t]]
    deltas.sort(key=lambda r: r[3])
    rep(f"Before / after per term ({len(deltas)} terms change; all listed):")
    rep()
    rep.table(["term", "before", "after", "delta"], deltas[:top] if top else deltas)


def recitation_section(rep: Report, gemma_cur, gemma_gen, llama_rows, samples):
    rep("## Recitation verdict")
    rep()
    rep(f"`order >= {cg.RECITATION_MIN_ORDER} AND unsupported_obj >= "
        f"{cg.RECITATION_MIN_UNSUPPORTED} AND unsupported_obj/obj >= "
        f"{cg.RECITATION_MIN_UNSUPPORTED_SHARE}`, on the RAW answer.")
    rep()
    buckets = {">=53": [], "20-52": [], "<20": []}
    for pid, desc, cats, model in llama_rows:
        n = len(cats)
        b = ">=53" if n >= 53 else ("20-52" if n >= 20 else "<20")
        buckets[b].append((pid, desc, cats, model))
    rows = []
    over_missed = []
    for b, items in buckets.items():
        flagged = [it for it in items if cg.is_recitation(it[1], it[2])]
        rows.append((f"llama {b}", len(items), len(flagged),
                     f"{len(flagged) / max(1, len(items)):.1%}"))
        if b == ">=53":
            over_missed = [it for it in items if not cg.is_recitation(it[1], it[2])]
    g_flag_cur = [r for r in gemma_cur if cg.is_recitation(r[1], r[2])]
    g_flag_gen = [r for r in gemma_gen if cg.is_recitation(r[1], r[2])]
    rows.append(("gemma current rows (Q1)", len(gemma_cur), len(g_flag_cur),
                 f"{len(g_flag_cur) / max(1, len(gemma_cur)):.2%}"))
    rows.append(("gemma all generations vs current description", len(gemma_gen),
                 len(g_flag_gen), f"{len(g_flag_gen) / max(1, len(gemma_gen)):.2%}"))
    rep.table(["cohort", "rows", "flagged recitation", "share"], rows)

    # Which conjunct stops the gemma rows: how close do they come?
    near = Counter()
    for pid, desc, cats in gemma_cur:
        s = cg.recitation_stats(desc, cats)
        o = s["order"] >= cg.RECITATION_MIN_ORDER
        u = s["unsupported_obj"] >= cg.RECITATION_MIN_UNSUPPORTED
        sh = s["obj"] and s["unsupported_obj"] / s["obj"] >= cg.RECITATION_MIN_UNSUPPORTED_SHARE
        near[(bool(o), bool(u), bool(sh))] += 1
    rep("gemma current rows by which conjuncts hold (order, unsupported>=10, share>=0.5):")
    rep()
    rep.table(["order", "unsup>=10", "share>=0.5", "rows"],
              [(*k, v) for k, v in sorted(near.items(), key=lambda kv: -kv[1])])

    if over_missed:
        rep(f"llama >= 53 recitations NOT caught ({len(over_missed)}):")
        rep()
        for pid, desc, cats, model in over_missed[:15]:
            s = cg.recitation_stats(desc, cats)
            rep(f"- {pid} ({model}, {len(cats)} tags, order {s['order']:.2f}, "
                f"obj {s['obj']}, unsupported {s['unsupported_obj']}): {_trunc(desc or '(no description)', 200)}")
        rep()

    rep(f"### gemma rows flagged ({len(g_flag_gen)} generations; owner: judge each)")
    rep()
    seen = set()
    flagged_dump = []
    for pid, desc, cats in g_flag_gen:
        if pid in seen:
            continue
        seen.add(pid)
        s = cg.recitation_stats(desc, cats)
        flagged_dump.append({"photo_id": pid, "description": desc, "categories": cats,
                             "stats": s})
        rep(f"**{pid}** — {len(cats)} tags, order {s['order']:.2f}, obj {s['obj']}, "
            f"unsupported {s['unsupported_obj']} ({', '.join(s['unsupported'])})")
        rep()
        rep(f"> {desc}")
        rep()
        rep(f"`{', '.join(cats)}`")
        rep()
    samples["gemma_flagged_recitation"] = flagged_dump
    llama_mid = [it for it in buckets["20-52"] if cg.is_recitation(it[1], it[2])]
    samples["llama_20_52_flagged"] = [
        {"photo_id": p, "model": m, "description": d, "categories": c}
        for p, d, c, m in random.Random(11).sample(llama_mid, min(25, len(llama_mid)))]
    rep(f"llama 20-52 flagged: {len(llama_mid)}; 8 seeded samples (all 25 in samples.json):")
    rep()
    for p, d, c, m in random.Random(11).sample(llama_mid, min(8, len(llama_mid))):
        rep(f"- {p} ({len(c)} tags): {_trunc(d or '', 150)} -> `{_trunc(', '.join(c), 200)}`")
    rep()


def _verdict(desc, cats, obj_terms, min_unsup, min_share):
    if not desc or not desc.strip():
        return False
    d = cg.scrub(desc)
    dt = cg.tokens(d)
    obj = [t for t in cats if t in obj_terms]
    uns = [t for t in obj if not cg.is_supported(d, t, dt)]
    return (cg.order_share(cats) >= cg.RECITATION_MIN_ORDER and len(uns) >= min_unsup
            and bool(obj) and len(uns) / len(obj) >= min_share)


def verdict_sensitivity(rep: Report, gemma_cur, llama_rows, fixture):
    """Report-only: how the verdict's counts move under nearby variants."""
    family = frozenset(cg._CONCRETE).union(
        *(terms for terms, _ in cg.REQUIRED_CUE_FAMILIES.values()))
    concrete = cg.OBJECT_TERMS
    O, U, S = cg.OBJECT_TERMS, cg.RECITATION_MIN_UNSUPPORTED, cg.RECITATION_MIN_UNSUPPORTED_SHARE
    variants = [
        ("SHIPPED: objects = concrete nouns only", lambda d, c: _verdict(d, c, O, U, S)),
        ("old plan spec: + family terms as objects", lambda d, c: _verdict(d, c, family, U, S)),
        ("unsupported >= 13", lambda d, c: _verdict(d, c, O, 13, S)),
        ("share >= 0.6", lambda d, c: _verdict(d, c, O, U, 0.6)),
        ("judged on ground().kept", lambda d, c: _verdict(d, cg.ground(d, c).kept, O, U, S)),
        ("concrete only AND unsupported >= 8", lambda d, c: _verdict(d, c, concrete, 8, S)),
        ("+ family terms AND unsupported >= 13", lambda d, c: _verdict(d, c, family, 13, S)),
    ]
    over = [(d, c) for _, d, c, _ in llama_rows if len(c) >= 53]
    mid = [(d, c) for _, d, c, _ in llama_rows if 20 <= len(c) < 53]
    rows = []
    for name, f in variants:
        rows.append((name,
                     f"{sum(f(d, c) for d, c in over)}/{len(over)}",
                     f"{sum(f(d, c) for d, c in mid)}/{len(mid)}",
                     sum(f(d, c) for _, d, c in gemma_cur),
                     f(fixture[0], fixture[1])))
    rep("### Verdict sensitivity (report only; first row is what the module ships)")
    rep()
    rep.table(["variant", "llama >=53 caught", "llama 20-52 flagged", "gemma current flagged",
               "251357 temp-0 (relayed order) flagged"], rows)


def family_section(rep: Report, rows, rng: random.Random, samples: dict):
    rep("## Required-cue families (DIAGNOSTIC — v1 drops none of these)")
    rep()
    hits = defaultdict(list)
    for pid, desc, cats in rows:
        for fam, terms in cg.required_cue_families_diagnostic(desc, cats).items():
            hits[fam].append((pid, terms, desc))
    tab = []
    for fam in cg.REQUIRED_CUE_FAMILIES:
        h = hits[fam]
        tab.append((fam, len(h), sum(len(t) for _, t, _ in h),
                    ", ".join(f"{t} {c}" for t, c in
                              Counter(x for _, ts, _ in h for x in ts).most_common(4))))
    rep.table(["family", "rows hit", "tags that would drop", "top terms"], tab)
    samples["families"] = {}
    for fam in cg.REQUIRED_CUE_FAMILIES:
        h = hits[fam]
        pick = rng.sample(h, min(10, len(h)))
        samples["families"][fam] = [{"photo_id": p, "would_drop": t, "description": d}
                                    for p, t, d in pick]
        if not pick:
            continue
        rep(f"#### {fam}: 10 of {len(h)}")
        rep()
        rep.table(["photo", "would drop", "description"],
                  [(p, ", ".join(t), _trunc(d, 220)) for p, t, d in pick])


def cmd_score(argv=None):
    """`score` subcommand: print the holdout labelling score.

    Prints exactly the table `GET /api/eval/category-grounding/score` serves
    to the labelling page — both call
    `photosearch.category_grounding_eval_api.compute_score`, so the CLI table
    and the live page can never drift apart.
    """
    ap = argparse.ArgumentParser(
        prog="category_grounding_eval.py score",
        description="Print the holdout labelling score (same numbers as "
                    "GET /api/eval/category-grounding/score).")
    ap.add_argument("--eval-dir", default=None,
                    help="Override PHOTOSEARCH_CATEGORY_EVAL_DIR for this run "
                         "(default: ./evals/category-grounding).")
    args = ap.parse_args(argv)
    if args.eval_dir:
        os.environ["PHOTOSEARCH_CATEGORY_EVAL_DIR"] = args.eval_dir

    from photosearch.category_grounding_eval_api import (
        GROUPS as EVAL_GROUPS, MIN_JUDGED, PASS_PRECISION, compute_score,
    )
    result = compute_score()
    rep = Report()
    rep(f"# Category-grounding holdout score (bar: {PASS_PRECISION:.0%} precision, "
        f">= {MIN_JUDGED} judged correct+wrong; debatable excluded from the bar)")
    rep()

    def _row(name, c):
        prec = "—" if c["precision"] is None else f"{c['precision']:.1%}"
        low = c.get("precision_lower_95")
        low = "—" if low is None else f"{low:.1%}"
        return [name, c["n"], c["labelled"], c["correct"], c["wrong"], c["debatable"],
                c["unlabelled"], prec, low, "PASS" if c["passes"] else "no",
                "yes" if c.get("lower_clears_bar") else "no"]

    rows = [_row(g, result["per_group"][g]) for g in EVAL_GROUPS]
    rows.append(_row("OVERALL", result["overall"]))
    rep.table(["group", "n", "labelled", "correct", "wrong", "debatable", "unlabelled",
               "precision", "95% lower", "ships?", "lower >= bar?"], rows)
    rep()
    rep("`ships?` is the bar as decided (point estimate). `95% lower` is the Wilson "
        "lower bound: where it is below the bar, the labels cannot rule out a true "
        "precision under it.")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(ROOT / "photo_index.db.local"))
    ap.add_argument("--seed", type=int, default=7,
                    help="sample A (the tuning sample the cue lists were adjusted on)")
    ap.add_argument("--holdout-seed", type=int, default=11,
                    help="sample B: the HOLDOUT the owner labels (disjoint from A)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rep = Report()
    samples: dict = {}

    rep("# Category grounding eval")
    rep()
    rep(f"DB `{Path(args.db).name}` (read-only), seed {args.seed}. Generated by "
        "`evals/category_grounding_eval.py`; rules in `photosearch/category_grounding.py`. "
        "No model calls.")
    rep()
    rep("**Tuning caveat.** The cue lists were adjusted once after the first run on "
        "these same rows (seed 7 samples = sample A): negation and through-the-window "
        "spans are blanked (`SCRUB_PATTERNS`), `tree-shaped` is not a tree, `floodlit`, "
        "`ambient/pendant/string ... lighting` are artificial cues, and `the sky is clear "
        "blue` / `bright daylight` are sunny cues. The recitation verdict's objects were "
        "then narrowed to concrete nouns (family terms excluded). Owner labelling must "
        "judge the FRESH holdout sample (`samples-holdout-seed11.json`), not these. "
        "Owner decisions 2026-10-18 applied since: basketball court/hoop exempt from the "
        "sport group on indoor-cued text; partly cloudy weakly cues the cloudy side.")
    rep()

    # Q1 ------------------------------------------------------------------
    q1 = conn.execute(
        "SELECT DISTINCT p.id, p.description, " + _ANSWER + " FROM photos p "
        "JOIN generations g ON g.photo_id = p.id AND g.text_type = 'category-content' "
        "AND g.generated_text = p.categories " + _RAW_JOIN
        + "WHERE g.model_used LIKE 'google/gemma%'"
    ).fetchall()
    gemma_cur = [(pid, d, c) for pid, d, raw in q1 if (c := _load(raw)) is not None]
    q1b = conn.execute(
        "SELECT g.photo_id, p.description, " + _ANSWER + " FROM generations g "
        "JOIN photos p ON p.id = g.photo_id " + _RAW_JOIN
        + "WHERE g.text_type = 'category-content' "
        "AND g.model_used LIKE 'google/gemma%'").fetchall()
    gemma_gen = [(pid, d, c) for pid, d, raw in q1b if (c := _load(raw)) is not None]

    rep("## Q1 — gemma (contradiction groups)")
    rep()
    res = analyse_groups(gemma_cur, rep, rng, "gemma current rows (stored == gemma generation)", samples)
    print_group_samples(res, rep, rng, samples, "gemma_group_samples")
    write_holdout(res, args.holdout_seed, samples["gemma_group_samples"], rep)
    before_after(res, rep, top=0)
    res_gen = analyse_groups(gemma_gen, rep, rng,
                             "all gemma generations vs CURRENT description", samples)

    # Q2 ------------------------------------------------------------------
    q2 = conn.execute(
        "SELECT g.photo_id, p.description, " + _ANSWER + ", g.model_used "
        "FROM generations g LEFT JOIN photos p ON p.id = g.photo_id " + _RAW_JOIN
        + "WHERE g.text_type = 'category-content' AND g.model_used IN (?, ?)", LLAMA).fetchall()
    llama_rows = [(pid, d, c, m) for pid, d, raw, m in q2 if (c := _load(raw)) is not None]

    recitation_section(rep, gemma_cur, gemma_gen, llama_rows, samples)

    rep("## Q2 — llama generations (contradiction groups, all sizes)")
    rep()
    analyse_groups([(p, d, c) for p, d, c, _ in llama_rows], rep, rng,
                   "llama generations vs current description", samples)

    # Q3 ------------------------------------------------------------------
    rep("## Q3 — backfill preview (stored rows, latest generation)")
    rep()
    q3 = conn.execute(
        "SELECT p.id, p.description, p.categories, g.generated_text, g.model_used "
        "FROM photos p LEFT JOIN (SELECT photo_id, MAX(id) AS gid FROM generations "
        "WHERE text_type = 'category-content' GROUP BY photo_id) lg ON lg.photo_id = p.id "
        "LEFT JOIN generations g ON g.id = lg.gid WHERE p.categories IS NOT NULL").fetchall()
    by_model = defaultdict(lambda: Counter())
    for pid, desc, raw, gen_raw, model in q3:
        stored = _load(raw)
        gen = _load(gen_raw)
        m = model or "(no generation)"
        b = by_model[m]
        b["rows"] += 1
        if stored is None:
            b["unparseable"] += 1
            continue
        if gen is None or not set(stored) <= set(gen):
            b["ineligible"] += 1
            continue
        b["eligible"] += 1
        if not desc:
            b["no_description"] += 1
        g = cg.ground(desc, stored)
        if g.dropped:
            b["would_change"] += 1
            b["tags_dropped"] += len(g.dropped)
            if not g.kept:
                b["would_empty"] += 1
        if g.recitation:
            b["recitation"] += 1
    rep.table(["model", "rows", "eligible", "ineligible (hand-set / pre-logging)",
               "would change", "tags dropped", "would empty", "recitation"],
              [(m, b["rows"], b["eligible"], b["ineligible"] + b["unparseable"],
                b["would_change"], b["tags_dropped"], b["would_empty"], b["recitation"])
               for m, b in sorted(by_model.items(), key=lambda kv: -kv[1]["rows"])])

    # Q4 ------------------------------------------------------------------
    rep("## Q4 — fixtures")
    rep()
    fx = json.loads((OUT / "fixtures.json").read_text())
    p57 = fx["251357"]
    cases = [("251357 temp 0", p57["description"], p57["temp0"]["categories"]),
             ("251357 temp 0.4", p57["description"], p57["temp04"]["categories"]),
             ("251356", fx["251356"]["description"], fx["251356"]["categories"]),
             ("251358", fx["251358"]["description"], fx["251358"]["categories"])]
    for name, d, c in cases:
        g = cg.ground(d, c)
        s = cg.recitation_stats(d, c)
        rep(f"- **{name}**: {len(c)} -> {len(g.kept)}; recitation={g.recitation} "
            f"(order {s['order']:.2f}, obj {s['obj']}, unsupported {s['unsupported_obj']}); "
            f"dropped: {', '.join(f'{t} ({r})' for t, r in g.dropped.items()) or 'none'}")
    rep()
    rep("(Fixture order caveat: see `_note` in fixtures.json; the temp-0 list is in "
        "relayed near-alphabetical order, not the model's own.)")
    rep()

    verdict_sensitivity(rep, gemma_cur, llama_rows,
                        (p57["description"], p57["temp0"]["categories"]))

    family_section(rep, gemma_cur, rng, samples)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(rep.lines) + "\n")
    (OUT / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT / 'summary.md'} and {OUT / 'samples.json'}", file=sys.stderr)


if __name__ == "__main__":
    # No subparsers on the main eval (it has its own positional-free --db/--seed
    # flags and runs the full Q1-Q4 report); `score` is dispatched by argv[1]
    # instead of folding it into that parser, so `python
    # evals/category_grounding_eval.py [--db ...] [--seed ...]` keeps working
    # unchanged and `score` is the one new entry point.
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        cmd_score(sys.argv[2:])
    else:
        main()
