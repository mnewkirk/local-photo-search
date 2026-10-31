"""Scoring for the people-bind eval (docs/plans/description-names.md).

Pure: no DB, no model. ``evals/people_bind_eval.py`` builds the sample, runs the
PRODUCTION binder and calls :func:`score` / :func:`gates` here.

Labels are FILES, never DB rows (llm-passes rules). One entry per photo::

    {"<photo_id>": {"split": "dev"|"holdout",
                    "stratum": "single"|"multi_2_3"|"multi_4_6"|"adversarial",
                    "mentions": [{"phrase": "A young boy ...",
                                  "gold": [<face id>, ...] | "NONE" | "AMBIGUOUS"}],
                    "bad_labels": [<face id whose person label is WRONG>, ...]}}

``gold`` names the face(s) the phrase really refers to, ``"NONE"`` when no
offered box is that person (back turned, out of frame, undetected), and
``"AMBIGUOUS"`` when the description does not say which of several. Predictions
are the binder's slots: ``{"<photo_id>": {"method": "rule"|"vlm"|None,
"slots": [...] | None}}`` (``None`` = deferred, not scored).

A bound phrase the labeller did not list is an ``unscored_bind`` and fails the
gate: the labeller must list every person phrase of a sampled description.
"""

from __future__ import annotations

from typing import Optional

STRATA = ("single", "multi_2_3", "multi_4_6", "adversarial")
MIN_BOUND_FOR_GATE = 150           # rule of three: 0/150 bounds the true rate under 2%
COVERAGE_MULTI = 0.50
COVERAGE_SINGLE = 0.80
ABSTAIN_CORRECT = 0.95


def _overlaps(sp: dict, rng: tuple) -> bool:
    return sp["start"] < rng[1] and rng[0] < sp["end"]


def _pred_for(slots: list, phrase: str, rng: Optional[tuple] = None) -> Optional[list]:
    """Face ids bound to ``phrase`` (None = not bound).

    With ``rng`` (the phrase's character range in the description) a slot
    counts when any of its spans OVERLAPS the phrase: the labeller's phrase
    and the model's need not share boundaries ("One child" vs "One child in
    blue"). Without it, exact span text. Several overlapping slots -> the
    union of their faces (which then cannot equal a 1-face gold)."""
    hit = [s for s in slots or []
           if any((_overlaps(sp, rng) if rng else sp["text"] == phrase) for sp in s["spans"])]
    if not hit:
        return None
    return sorted({f["face_id"] for s in hit for f in s["faces"]})


def mention_range(description: Optional[str], phrase: str) -> Optional[tuple]:
    """Character range of the phrase's first occurrence, or None."""
    if not description:
        return None
    i = description.find(phrase)
    return None if i < 0 else (i, i + len(phrase))


def _score_mention(gold, pred) -> str:
    if pred is None:
        return "missed_abstain" if isinstance(gold, list) else "right_abstain"
    if isinstance(gold, list) and sorted(gold) == sorted(pred):
        return "correct_bind"
    return "wrong_bind"


def score(labels: dict, preds: dict, face_sources: Optional[dict] = None,
          split: Optional[str] = None, descriptions: Optional[dict] = None) -> dict:
    """Metrics over the labelled photos that have a (non-deferred) prediction.

    ``face_sources`` maps face id -> its label's ``match_source`` so the
    user-visible wrong-name count (a correct bind on a WRONGLY labelled face)
    can be split by source. ``descriptions`` (photo id str -> text) switches
    phrase matching from exact span text to character overlap, which is what
    the harness uses: the model chooses its own phrase boundaries."""
    descriptions = descriptions or {}
    face_sources = face_sources or {}
    zero = lambda: {"correct_bind": 0, "wrong_bind": 0, "missed_abstain": 0,
                    "right_abstain": 0}
    total, by_stratum, by_method = zero(), {}, {}
    unscored_bind = 0
    wrong_cases: list = []
    visible_wrong = {"by_source": {}, "total": 0}
    photos = 0
    for pid, lab in labels.items():
        if split and lab.get("split") != split:
            continue
        pr = preds.get(str(pid))
        if pr is None or pr.get("slots") is None:
            continue
        photos += 1
        stratum, method = lab.get("stratum", "?"), pr.get("method") or "?"
        desc = descriptions.get(str(pid))
        rngs = {m["phrase"]: mention_range(desc, m["phrase"]) for m in lab.get("mentions", [])}
        for s in pr["slots"]:
            if not any((_overlaps(sp, r) if r else sp["text"] == ph)
                       for sp in s["spans"] for ph, r in rngs.items()):
                unscored_bind += 1
        bad = set(lab.get("bad_labels", []))
        for m in lab.get("mentions", []):
            if m.get("gold") is None:
                continue                              # not labelled yet
            pred = _pred_for(pr["slots"], m["phrase"], rngs.get(m["phrase"]))
            o = _score_mention(m["gold"], pred)
            for bucket in (total, by_stratum.setdefault(stratum, zero()),
                           by_method.setdefault(method, zero())):
                bucket[o] += 1
            if o == "wrong_bind":
                wrong_cases.append({"photo_id": pid, "phrase": m["phrase"],
                                    "gold": m["gold"], "pred": pred, "method": method})
            if o == "correct_bind" and bad & set(pred):
                # right box, wrong person label: a user-visible wrong name
                for f in sorted(bad & set(pred)):
                    src = face_sources.get(f, "?")
                    visible_wrong["by_source"][src] = visible_wrong["by_source"].get(src, 0) + 1
                    visible_wrong["total"] += 1
    bound = total["correct_bind"] + total["wrong_bind"]
    has_gold = total["correct_bind"] + total["missed_abstain"] + total["wrong_bind"]
    nogold = total["right_abstain"] + total["wrong_bind"]
    return {
        "photos": photos, "mentions": sum(total.values()), "bound": bound,
        "total": total, "by_stratum": by_stratum, "by_method": by_method,
        "unscored_bind": unscored_bind,
        "coverage": _ratio(total["correct_bind"], has_gold),
        "abstain_correct": _ratio(total["right_abstain"], nogold),
        "wrong_bind_rate": _ratio(total["wrong_bind"], bound),
        "wrong_cases": wrong_cases, "visible_wrong": visible_wrong,
    }


def _ratio(a: int, b: int) -> Optional[float]:
    return round(a / b, 4) if b else None


def coverage_for(report: dict, strata: tuple) -> Optional[float]:
    c = g = 0
    for st in strata:
        b = report["by_stratum"].get(st)
        if b:
            c += b["correct_bind"]
            g += b["correct_bind"] + b["missed_abstain"] + b["wrong_bind"]
    return _ratio(c, g)


def gates(report: dict) -> list:
    """The pre-registered gates, on a HOLDOUT report. ``[(name, ok, detail)]``.
    ``ok`` is True/False, or None when there is not enough data to decide.
    Any False fails; any None blocks the backfill just the same."""
    out = []
    rt = report["total"]
    # not enough data is "cannot decide", never a pass: zero wrong out of 20 proves nothing
    out.append(("bound mentions >= %d" % MIN_BOUND_FOR_GATE,
                True if report["bound"] >= MIN_BOUND_FOR_GATE else None, report["bound"]))
    out.append(("wrong binds == 0", rt["wrong_bind"] == 0, rt["wrong_bind"]))
    adv = report["by_stratum"].get("adversarial", {})
    out.append(("adversarial wrong binds == 0", adv.get("wrong_bind", 0) == 0,
                adv.get("wrong_bind", 0)))
    rule = report["by_method"].get("rule")
    out.append(("rule tier wrong binds == 0 (separately gated)",
                True if rule is None else rule["wrong_bind"] == 0,
                None if rule is None else rule["wrong_bind"]))
    out.append(("unscored binds == 0", report["unscored_bind"] == 0, report["unscored_bind"]))
    multi = coverage_for(report, ("multi_2_3", "multi_4_6"))
    single = coverage_for(report, ("single",))
    out.append(("coverage multi-face >= %.0f%%" % (COVERAGE_MULTI * 100),
                None if multi is None else multi >= COVERAGE_MULTI, multi))
    out.append(("coverage single-face >= %.0f%%" % (COVERAGE_SINGLE * 100),
                None if single is None else single >= COVERAGE_SINGLE, single))
    ab = report["abstain_correct"]
    out.append(("abstain-correct >= %.0f%%" % (ABSTAIN_CORRECT * 100),
                None if ab is None else ab >= ABSTAIN_CORRECT, ab))
    strict_wrong = report["visible_wrong"]["by_source"].get("strict", 0)
    out.append(("strict-caused user-visible wrong names == 0 (keeps `strict` in the "
                "name sources)", strict_wrong == 0, strict_wrong))
    return out


def verdict(gate_rows: list) -> str:
    if any(ok is False for _, ok, _ in gate_rows):
        return "FAIL"
    if any(ok is None for _, ok, _ in gate_rows):
        return "INCOMPLETE"
    return "PASS"


GOLD_WORDS = ("NONE", "AMBIGUOUS")


def check_labels(labels: dict, sample: dict, photos: dict) -> dict:
    """Validate a labels file before it is scored.

    ``photos`` maps photo id (str) -> ``{"description", "offered": {face ids
    drawn as boxes}, "faces": {all face ids}}`` as of NOW, or ``None`` when
    the photo is gone / has no description. ``gold`` may still be ``None``
    (not labelled yet): that is progress, not an error.

    Returns ``{"errors": [...], "warnings": [...], "photos", "mentions",
    "labelled", "unlabelled", "missing", "unreviewed"}``; ``missing`` = sampled
    photos with no labels entry; ``unreviewed`` = photos the labeller has not
    ticked "every person phrase is listed" on (the sheet's ``reviewed``)."""
    errors, warnings = [], []
    n_mentions = n_labelled = 0
    for pid, lab in sorted(labels.items(), key=lambda kv: int(kv[0])):
        where = f"photo {pid}"
        meta = sample.get(pid)
        if meta is None:
            errors.append(f"{where}: not in sample.json")
            continue
        for k in ("split", "stratum"):
            if lab.get(k) != meta.get(k):
                errors.append(f"{where}: {k} {lab.get(k)!r} != sample's {meta.get(k)!r}")
        ph = photos.get(pid)
        if ph is None:
            errors.append(f"{where}: gone from the DB or has no description")
            continue
        desc, offered, all_faces = ph["description"], set(ph["offered"]), set(ph["faces"])
        seen = set()
        for m in lab.get("mentions", []):
            n_mentions += 1
            phrase, gold = m.get("phrase"), m.get("gold")
            if not isinstance(phrase, str) or not phrase.strip():
                errors.append(f"{where}: empty phrase")
                continue
            if phrase in seen:
                errors.append(f"{where}: phrase listed twice: {phrase!r}")
            seen.add(phrase)
            if phrase not in desc:
                errors.append(f"{where}: phrase is not an exact substring of the "
                              f"description: {phrase!r}")
            elif desc.count(phrase) > 1:
                warnings.append(f"{where}: {phrase!r} occurs more than once "
                                f"(the binder never binds it)")
            if gold is None:
                continue
            n_labelled += 1
            if isinstance(gold, str):
                if gold not in GOLD_WORDS:
                    errors.append(f"{where}: gold {gold!r} must be a list of face "
                                  f"ids, \"NONE\" or \"AMBIGUOUS\"")
                continue
            if (not isinstance(gold, list) or not gold
                    or not all(isinstance(f, int) and not isinstance(f, bool) for f in gold)):
                errors.append(f"{where}: gold for {phrase!r} must be a non-empty list "
                              f"of face ids")
                continue
            if len(set(gold)) != len(gold) or len(gold) > 3:
                errors.append(f"{where}: gold for {phrase!r} must be 1-3 distinct faces "
                              f"(a bigger group is NONE)")
            not_drawn = sorted(set(gold) - offered)
            if not_drawn:
                errors.append(f"{where}: gold for {phrase!r} names face(s) {not_drawn} "
                              f"that are not drawn boxes of this photo")
        rngs = sorted(r + (m.get("phrase"),) for m in lab.get("mentions", [])
                      if isinstance(m.get("phrase"), str)
                      for r in [mention_range(desc, m["phrase"])] if r)
        for (a0, a1, pa), (b0, b1, pb) in zip(rngs, rngs[1:]):
            if b0 < a1:
                errors.append(f"{where}: phrases overlap ({pa!r} / {pb!r}): list one "
                              f"person reference once")
        bad = lab.get("bad_labels", [])
        if not isinstance(bad, list) or not all(isinstance(f, int) for f in bad):
            errors.append(f"{where}: bad_labels must be a list of face ids")
        elif set(bad) - all_faces:
            errors.append(f"{where}: bad_labels {sorted(set(bad) - all_faces)} are not "
                          f"faces of this photo")
    return {"errors": errors, "warnings": warnings, "photos": len(labels),
            "mentions": n_mentions, "labelled": n_labelled,
            "unlabelled": n_mentions - n_labelled,
            "missing": sorted((p for p in sample if p not in labels), key=int),
            "unreviewed": sorted((p for p, lab in labels.items() if lab.get("reviewed") is not True),
                                 key=int)}


def split_of(photo_id: int) -> str:
    """Stable dev/holdout assignment: ~1/3 holdout, never tuned on."""
    return "holdout" if (int(photo_id) * 2654435761) % 3 == 0 else "dev"
