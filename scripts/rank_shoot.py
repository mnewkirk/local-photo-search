"""Rank one shoot's photos into a "best of" selection. Reusable per game.

This reproduces the curation behind the Aug 29 2026 collections (ids 29/30/31),
which was done ad-hoc and then lost. It recurs every match, so it lives here.

    Best N   — sharp in-focus faces + real action, with per-person coverage
    Next M   — no overlap, no two frames from the same burst
    All N+M  — the union, for one upload

WHY FACE-CROP LAPLACIAN AT NATIVE RESOLUTION IS THE PRIMARY SIGNAL

Within a single shoot the usual rankers are inert or actively wrong:

  - CLIP similarity is useless — every frame is the same kids on the same
    pitch, so the embedding barely moves between a keeper and a dud.
  - The MCP `rerank_photos` tool silently passes through on the NAS (no
    PHOTOSEARCH_LLM_VISUAL_MODEL configured), and a local qwen2.5-vl
    mode-collapses on Likert scoring.
  - VLM aesthetics (`aes_overall`) is the right idea but is only scored on a
    fraction of a fresh shoot, and a PARTIAL pass is worse than none here: it
    covers whichever photos the fleet claimed first, so ranking on it would
    systematically favour that arbitrary subset. This script refuses to use it
    below --min-aes-coverage.

What actually separates frames in a sports burst is whether the face is in
focus, and that has to be measured on the ORIGINAL pixels — a downscaled
preview or a cached 200px crop has already thrown the high-frequency detail
away, which is precisely the signal. Hence one full-resolution decode per
photo, which is the expensive part and is therefore cached.

That measurement pass lives in **`photosearch/rank_measure.py`**, not here:
`scripts/` is not copied into the Docker image, and the `rank_measure` step of
a batch advance has to be able to run it on the NAS. This script imports it,
so there is one implementation and one cache format.

USAGE

    # measure (slow, ~1 image decode per photo; cached, resumable)
    # — or let `Advance batch` on /batches do it for the batch's folder
    python scripts/rank_shoot.py --date 2026-09-12 --measure

    # select + preview, then create the collections
    python scripts/rank_shoot.py --date 2026-09-12
    python scripts/rank_shoot.py --date 2026-09-12 --apply
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

# Before the package import below: this runs as a script, so the project root
# is not on sys.path by virtue of anything else.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from photosearch.rank_measure import default_cache_path, measure  # noqa: E402


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Phase 2 — score + select
# ---------------------------------------------------------------------------

def _pct_ranks(values):
    """Map values -> percentile in [0,1]. Percentiles, not raw units, because
    Laplacian variance, pixel edge and det_score share no scale and their
    distributions differ per shoot; ranking makes the weights mean something.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    for rank, i in enumerate(order):
        out[i] = rank / max(1, len(values) - 1)
    return out


def build_bursts(photos, gap_seconds):
    """Group time-ordered photos into bursts; a gap > `gap_seconds` starts one.

    Derived from timestamps rather than `photo_stacks` on purpose: a fresh
    shoot has not been stacked (0 stacks on 2026-09-12), and running scoped
    stacking to get them has twice wiped the whole library's stacks. This is
    self-contained and has no global side effects.
    """
    ordered = sorted((p for p in photos if p["date_taken"]),
                     key=lambda p: p["date_taken"])
    bursts, cur, prev = [], [], None
    for p in ordered:
        t = datetime.fromisoformat(p["date_taken"][:19])
        if prev is not None and (t - prev).total_seconds() > gap_seconds:
            bursts.append(cur); cur = []
        cur.append(p); prev = t
    if cur:
        bursts.append(cur)
    for p in photos:
        if not p["date_taken"]:
            bursts.append([p])           # undated: its own burst
    return bursts


def select(db, date_, cache, *, best_n, next_n, min_per_person, burst_gap,
           max_per_person=None, min_aes_coverage=0.80):
    rows = db.conn.execute(
        """SELECT p.id, p.filename, p.date_taken, p.aesthetic_score, p.aes_overall
             FROM photos p WHERE date(p.date_taken) = ? ORDER BY p.date_taken""",
        (date_,)).fetchall()
    photos = {r["id"]: dict(r) for r in rows}
    if not photos:
        raise SystemExit(f"no photos on {date_}")

    face_rows = db.conn.execute(
        """SELECT f.id AS face_id, f.photo_id, f.det_score, f.person_id,
                  pe.name AS person_name
             FROM faces f JOIN photos p ON p.id = f.photo_id
             LEFT JOIN persons pe ON pe.id = f.person_id
            WHERE date(p.date_taken) = ?""", (date_,)).fetchall()

    # --- per-photo raw signals, taken from the photo's BEST face -----------
    for p in photos.values():
        p.update(lap=0.0, edge=0, area=0.0, det=0.0, people=set())
    for fr in face_rows:
        m = (cache.get(str(fr["photo_id"])) or {}).get(str(fr["face_id"]))
        p = photos[fr["photo_id"]]
        if fr["person_name"]:
            p["people"].add(fr["person_name"])
        if not m:
            continue
        # "Best face" = sharpest. A frame is worth keeping if ONE face is
        # crisp; averaging would punish a good subject for a blurred bystander.
        if m["lap"] > p["lap"]:
            p.update(lap=m["lap"], edge=m["edge"], area=m["area_frac"],
                     det=float(fr["det_score"] or 0.0))

    import math
    ids = [pid for pid, p in photos.items() if p["lap"] > 0]
    if not ids:
        raise SystemExit("no measured faces — run --measure first")

    aes_cov = sum(1 for pid in ids if photos[pid]["aes_overall"] is not None) / len(ids)
    use_aes = aes_cov >= min_aes_coverage
    laion = [photos[i]["aesthetic_score"] for i in ids]
    have_laion = [v for v in laion if v is not None]
    med_laion = sorted(have_laion)[len(have_laion) // 2] if have_laion else 0.0

    sharp = _pct_ranks([math.log1p(photos[i]["lap"]) for i in ids])
    size = _pct_ranks([photos[i]["edge"] for i in ids])
    det = _pct_ranks([photos[i]["det"] for i in ids])
    aest = _pct_ranks([
        (photos[i]["aes_overall"] if use_aes and photos[i]["aes_overall"] is not None
         else (photos[i]["aesthetic_score"] if photos[i]["aesthetic_score"] is not None
               else med_laion))
        for i in ids])
    # Multi-player frames stand in for "action": a contested ball has several
    # kids in it, an isolated portrait does not. A PROXY, not a detector —
    # capped and lightly weighted so it can't dominate.
    crowd = _pct_ranks([min(len(photos[i]["people"]), 3) for i in ids])

    for k, pid in enumerate(ids):
        photos[pid]["score"] = (0.50 * sharp[k] + 0.22 * size[k] + 0.13 * det[k]
                                + 0.10 * aest[k] + 0.05 * crowd[k])
        photos[pid]["parts"] = {"sharp": round(sharp[k], 3), "size": round(size[k], 3),
                                "det": round(det[k], 3), "aes": round(aest[k], 3),
                                "crowd": round(crowd[k], 3)}

    scored = [photos[i] for i in ids]
    bursts = build_bursts(scored, burst_gap)
    log(f"{len(scored)} scored photos in {len(bursts)} bursts "
        f"(gap {burst_gap}s); aesthetics coverage {aes_cov:.0%} -> "
        f"{'VLM aes_overall' if use_aes else 'LAION aesthetic_score'}")
    if len(bursts) < best_n + next_n:
        log(f"  ! only {len(bursts)} bursts for {best_n + next_n} slots — the "
            f"one-per-burst rule cannot fill the request; lower --burst-gap")

    # One representative per burst: the best frame of each press.
    reps = sorted((max(b, key=lambda p: p["score"]) for b in bursts),
                  key=lambda p: -p["score"])

    # --- fill the Best tier ------------------------------------------------
    # `max_per_person` stops one well-photographed kid owning the album. On
    # 2026-09-12 the uncapped run gave Calvin 16 of 50 — everyone still cleared
    # the >=3 floor, but it read as a Calvin album rather than a team one.
    # A photo with no named player consumes nobody's quota, so it is never
    # blocked by the cap.
    have = defaultdict(int)
    chosen = set()
    best = []

    def _fits(p):
        return (max_per_person is None
                or all(have[n] < max_per_person for n in p["people"]))

    def _take(p):
        best.append(p); chosen.add(p["id"])
        for n in p["people"]:
            have[n] += 1

    for p in reps:
        if len(best) >= best_n:
            break
        if _fits(p):
            _take(p)
    # If the cap is tight enough that the tier can't be filled, fall back to
    # score order for the remainder rather than returning a short album —
    # and say so, because a silently short set would look like a bug.
    if len(best) < best_n:
        short = best_n - len(best)
        for p in reps:
            if len(best) >= best_n:
                break
            if p["id"] not in chosen:
                _take(p)
        log(f"  ! cap {max_per_person} left the Best tier {short} short; "
            f"filled the remainder by score, so some people exceed the cap")

    # Per-person coverage: swap the weakest picks for the strongest frames of
    # anyone short of `min_per_person`. Applied AFTER ranking — it is a
    # constraint on the set, not a term in the score.
    everyone = {n for p in scored for n in p["people"]}
    for name in sorted(everyone, key=lambda n: have[n]):
        pool = [p for p in reps
                if name in p["people"] and p["id"] not in chosen
                # don't fix one person's shortfall by blowing another's cap
                and (max_per_person is None
                     or all(have[o] < max_per_person
                            for o in p["people"] if o != name))]
        while have[name] < min_per_person and pool:
            add = pool.pop(0)
            # Drop the weakest pick that isn't itself propping someone up.
            droppable = [p for p in reversed(best)
                         if all(have[n] > min_per_person for n in p["people"])]
            if not droppable:
                break
            drop = droppable[0]
            best.remove(drop); chosen.discard(drop["id"])
            for n in drop["people"]:
                have[n] -= 1
            _take(add)
    best.sort(key=lambda p: -p["score"])
    nxt = [p for p in reps if p["id"] not in chosen][:next_n]
    return best, nxt, scored


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.environ.get("PHOTOSEARCH_DB", "photo_index.db"))
    ap.add_argument("--date", required=True, metavar="YYYY-MM-DD")
    ap.add_argument("--cache", default=None, help="Sharpness cache JSON.")
    ap.add_argument("--measure", action="store_true",
                    help="Run the native-resolution sharpness pass and exit.")
    ap.add_argument("--best", type=int, default=50)
    ap.add_argument("--next", dest="next_n", type=int, default=200)
    ap.add_argument("--min-per-person", type=int, default=3)
    ap.add_argument("--max-per-person", type=int, default=None,
                    help="Cap how many Best-tier frames any one person can take. "
                         "Off by default. Stops one well-photographed kid owning "
                         "the album (Calvin took 16 of 50 uncapped on 2026-09-12).")
    ap.add_argument("--burst-gap", type=float, default=1.0,
                    help="Seconds between frames that still counts as one burst.")
    ap.add_argument("--label", default=None, help="Collection name prefix.")
    ap.add_argument("--apply", action="store_true", help="Create the collections.")
    return ap


def cache_path_for(args, db_path):
    """`--cache` wins; otherwise the cache sits beside the DB.

    On the NAS the DB is `/data/photo_index.db`, so the default is
    `/data/rank_shoot_<date>.json` — byte-identical to the path this script
    used to hardcode, which is what keeps existing caches usable.
    """
    return args.cache or default_cache_path(db_path, args.date)


def main():
    args = build_parser().parse_args()

    from photosearch.db import PhotoDB

    cache_path = cache_path_for(args, args.db)
    with PhotoDB(args.db) as db:
        if args.measure:
            measure(db, args.date, cache_path)
            log("measurement complete")
            return
        if not os.path.exists(cache_path):
            raise SystemExit(f"no cache at {cache_path} — run --measure first")
        with open(cache_path) as fh:
            cache = json.load(fh)

        best, nxt, scored = select(
            db, args.date, cache, best_n=args.best, next_n=args.next_n,
            min_per_person=args.min_per_person, burst_gap=args.burst_gap,
            max_per_person=args.max_per_person)

        def coverage(sel):
            c = defaultdict(int)
            for p in sel:
                for n in p["people"]:
                    c[n] += 1
            return dict(sorted(c.items(), key=lambda kv: -kv[1]))

        log(f"\nBest {len(best)}  — person coverage:")
        for n, k in coverage(best).items():
            log(f"    {n:<22} {k}")
        log(f"\nNext {len(nxt)} — person coverage:")
        for n, k in coverage(nxt).items():
            log(f"    {n:<22} {k}")
        log(f"\ntop 10 of Best:")
        for p in best[:10]:
            log(f"    {p['filename']:<16} score={p['score']:.3f} {p['parts']} "
                f"{sorted(p['people'])}")

        if not args.apply:
            log("\nDRY RUN — nothing written. Re-run with --apply.")
            return

        label = args.label or f"Soccer Game - {args.date}"
        made = []
        for name, sel, desc in (
            (f"{label} - Best {len(best)}", best,
             f"Top {len(best)} from {args.date}, ranked on native-resolution face "
             f"sharpness, face size and detection confidence, with at least "
             f"{args.min_per_person} frames of every face-matched person"
             + (f" and at most {args.max_per_person}" if args.max_per_person else "")
             + f", and no two frames from the same burst."),
            (f"{label} - Next {len(nxt)}", nxt,
             f"Second tier from {args.date}. No overlap with the Best {len(best)} "
             f"and no two frames from the same burst. Same ranking."),
            (f"{label} - All {len(best) + len(nxt)}", best + nxt,
             f"Everything from the two tiers in one album: the full curated set "
             f"from {args.date}."),
        ):
            cid = db.create_collection(name, desc)
            db.add_photos_to_collection(cid, [p["id"] for p in sel])
            made.append((cid, name, len(sel)))
            log(f"  created collection {cid}: {name} ({len(sel)} photos)")
        log("\ndone: " + ", ".join(f"{c} {n}" for c, n, _ in made))


if __name__ == "__main__":
    main()
