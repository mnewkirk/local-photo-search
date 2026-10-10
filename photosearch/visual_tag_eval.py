"""Storage for the visual-tag labelled eval — the contract shared by the
labelling page (`/eval/visual-tags`) and the harness (`evals/visual_tags_eval.py`).

Two JSON files in `eval_dir()`:

    sample.json   {"created": iso, "seed": int,
                   "photos": [{"photo_id": int, "stratum": str}, ...]}
    labels.json   {"labels": {"<photo_id>": {"yes": [tag, ...],
                                             "debatable": [tag, ...],
                                             "done": bool,
                                             "measured": bool,  # sharp/blurry asked
                                             "updated_at": iso}}}

A label is THREE-state per tag, and the third state is the point: the A/B that
chose prompt B scored debatable tags as neither right nor wrong, because
forcing a yes/no on `moody` manufactures disagreement that is not the model's
fault. A tag in neither list is a "no" — but only once the photo is `done`;
an un-`done` photo has no opinion at all and must not be scored.

Files, not DB rows: `sync-replica.sh` swaps the replica DB wholesale, and
hand labels are the one thing here that cannot be regenerated. The directory
is git-ignored — labels stay on the owner's machine.

Only PERCEIVED terms are scored against a model. Derived capture facts come
from EXIF and frozen/retired terms are never asked of the model. The one
exception to "never labelled" is MEASURED_TAGS (`sharp` / `blurry`): the owner
labels them as ground truth for the native-resolution sharpness measurement
(docs/plans/sharpness-measurement.md), and no VLM variant is ever scored on
them.

The sharpness eval has its OWN sample, beside the visual one rather than in
place of it, so drawing it never orphans the visual labels or cached runs:

    sharpness/sample.json           same shape as sample.json
    sharpness/labels.json           same shape as labels.json
    sharpness/recheck.json          blind-relabel subset of the sharpness sample
    sharpness/labels-recheck.json

Label SETS name a (sample, labels file) pair — see LABEL_SETS.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .visual_tags_derive import PERCEIVED_VOCABULARY

#: Tags being TRIALLED: the owner labels them, but they are not in the shipped
#: vocabulary or prompt. Collecting the label now is nearly free; collecting it
#: later means re-opening every photo. A candidate is scored only for a variant
#: that actually OFFERED it (`score(..., extra_tags=...)`) — otherwise every
#: labelled `action` would count as a miss against a model never asked.
#:
#: `action` exists because the owner read a dribble and a keeper's save as
#: `dramatic`. That is a fair reading of the word and the wrong tag: `dramatic`
#: is about the LOOK (contrast, visual tension; it contradicts `peaceful`),
#: and on a sports shoot it would go the way of `sunny` (98.7%) and stop
#: discriminating. Action is about the MOMENT. Nothing in categories/keywords
#: carries it either (1,251 of 1,373 frames say "soccer", none say what is
#: happening), and it is the signal `rank_shoot` says it lacks.
CANDIDATE_TAGS: dict[str, str] = {
    "action": ("a moment of real motion caught mid-act: a kick, tackle, save, "
               "leap, or a player driving the ball. Not players standing, "
               "walking, watching or posed - a sports photo is not "
               "automatically an action photo"),
    # 2026-09-26, from the Unsplash `peaceful` sheet: the owner expected "blue
    # sky" and "clouds" on two sky photos, and qwen volunteered `blue-sky,
    # fluffy-clouds` unprompted. Deliberately NOT mutually exclusive with each
    # other, nor with sunny / overcast — no CONTRADICTORY_PAIRS entry.
    "blue-sky": ("clear blue sky is a visible part of the frame, with or "
                 "without clouds in it"),
    "cloudy": ("clouds with visible shape or texture are part of the sky; can "
               "go with blue-sky, sunny or overcast"),
}

#: Tags the owner labels as ground truth for a MEASUREMENT, not for a model.
#: `sharp` / `blurry` are FROZEN in visual_tags_derive: never asked of the
#: VLM, never derived. The labels exist to validate the native-resolution
#: sharpness measurement (docs/plans/sharpness-measurement.md). They sit
#: outside CANDIDATE_TAGS on purpose, so `score()` can never be told to score a
#: VLM variant on them — a VLM reading a 1024 px tile is exactly the judge that
#: already failed. Definitions for the labeller: eval_api.LABELLER_NOTES.
MEASURED_TAGS: tuple[str, ...] = ("sharp", "blurry")

_PERCEIVED = frozenset(PERCEIVED_VOCABULARY)
_MEASURED = frozenset(MEASURED_TAGS)
_LABELLABLE = _PERCEIVED | frozenset(CANDIDATE_TAGS) | _MEASURED
# A measured tag that drifted into either scored group would start charging
# VLM variants for it. Fail at import rather than in a report.
assert not _MEASURED & (_PERCEIVED | frozenset(CANDIDATE_TAGS)), \
    "MEASURED_TAGS must stay outside the perceived and candidate vocabularies"


def eval_dir() -> Path:
    return Path(os.environ.get("PHOTOSEARCH_VISUAL_EVAL_DIR", "./evals/visual-tags"))


def _read(name: str, default: dict) -> dict:
    path = eval_dir() / name
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_atomic(name: str, data: dict) -> None:
    # Same rule as rank_measure's cache: a kill mid-dump must not leave
    # truncated JSON where hand labels used to be. `name` may carry a
    # subdirectory (`sharpness/labels.json`).
    target = eval_dir() / name
    d = target.parent
    d.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


#: SAMPLES. "visual" is the original 60-photo visual-tag sample; "sharpness"
#: is the blurry-weighted sample for the sharpness eval, in its own
#: subdirectory so drawing it cannot touch the visual one.
SAMPLE_FILES = {"visual": "sample.json", "sharpness": "sharpness/sample.json"}
RECHECK_FILES = {"visual": "recheck.json", "sharpness": "sharpness/recheck.json"}


def _sample_file(sample: str) -> str:
    try:
        return SAMPLE_FILES[sample]
    except KeyError:
        raise ValueError(f"unknown sample {sample!r}; have {sorted(SAMPLE_FILES)}")


def load_sample(sample: str = "visual") -> dict:
    return _read(_sample_file(sample), {"created": None, "seed": None, "photos": []})


def save_sample(photos: list[dict], seed: int, sample: str = "visual") -> dict:
    """`photos` is [{"photo_id": int, "stratum": str}, ...]."""
    data = {"created": datetime.now(timezone.utc).isoformat(), "seed": seed,
            "photos": [{"photo_id": int(p["photo_id"]), "stratum": str(p["stratum"])}
                       for p in photos]}
    _write_atomic(_sample_file(sample), data)
    return data


#: Label SETS. "main" is the eval's ground truth. "recheck" is the same
#: labeller re-labelling a blind subset of the same photos, later, without
#: seeing the first answer — the only way to measure how tightly the labeller
#: holds the definitions. A tag the owner disagrees with themself on cannot
#: be one the model is fairly scored against. "sharpness" /
#: "sharpness-recheck" are the same pair over the sharpness sample.
LABEL_SETS = {"main": "labels.json", "recheck": "labels-recheck.json",
              "sharpness": "sharpness/labels.json",
              "sharpness-recheck": "sharpness/labels-recheck.json"}
#: Which sample each label set labels.
SET_SAMPLE = {"main": "visual", "recheck": "visual",
              "sharpness": "sharpness", "sharpness-recheck": "sharpness"}
#: recheck set -> the first-answer set it re-labels blind.
RECHECK_OF = {"recheck": "main", "sharpness-recheck": "sharpness"}
RECHECK_N = 15
#: Larger than the visual recheck: kappa on ONE tag (`blurry`) needs enough
#: positives in the subset to mean anything, and the sample is blurry-weighted.
SHARPNESS_RECHECK_N = 20


def _labels_file(label_set: str) -> str:
    try:
        return LABEL_SETS[label_set]
    except KeyError:
        raise ValueError(f"unknown label set {label_set!r}; have {sorted(LABEL_SETS)}")


def load_labels(label_set: str = "main") -> dict[int, dict]:
    raw = _read(_labels_file(label_set), {"labels": {}})["labels"]
    return {int(k): v for k, v in raw.items()}


def recheck_ids(n: Optional[int] = None, seed: int = 1,
                sample: str = "visual") -> list[int]:
    """The blind-relabel subset: a seeded draw from the sample, persisted in
    recheck.json so it cannot drift under the owner between sessions."""
    _sample_file(sample)                          # validates the name
    existing = _read(RECHECK_FILES[sample], None)
    if existing is not None:
        return [int(i) for i in existing["photo_ids"]]
    if n is None:
        n = SHARPNESS_RECHECK_N if sample == "sharpness" else RECHECK_N
    import random
    ids = [int(p["photo_id"]) for p in load_sample(sample)["photos"]]
    if not ids:
        return []
    # The visual draw keeps its original RNG key so its subset is unchanged;
    # the sharpness one gets its own.
    key = f"{seed}:recheck" if sample == "visual" else f"{seed}:{sample}-recheck"
    chosen = sorted(random.Random(key).sample(ids, min(n, len(ids))))
    _write_atomic(RECHECK_FILES[sample], {"seed": seed, "photo_ids": chosen})
    return chosen


def set_entries(label_set: str = "main") -> list[dict]:
    """The sample entries a label set labels: the whole sample, or for a
    recheck set only its blind subset."""
    _labels_file(label_set)                       # validates the name
    sample = SET_SAMPLE[label_set]
    entries = load_sample(sample).get("photos") or []
    if label_set in RECHECK_OF:
        keep = set(recheck_ids(sample=sample))
        entries = [p for p in entries if int(p["photo_id"]) in keep]
    return entries


def _check(tags: Iterable[str], field: str) -> list[str]:
    tags = sorted(set(tags))
    bad = [t for t in tags if t not in _LABELLABLE]
    if bad:
        raise ValueError(f"{field}: not perceived, candidate or measured "
                         f"visual tags: {bad}")
    return tags


def save_label(photo_id: int, yes: Iterable[str], debatable: Iterable[str],
               done: bool = True, label_set: str = "main",
               measured: bool = False) -> dict:
    """Replace one photo's label. Raises ValueError on a non-labellable tag or
    a tag listed as both yes and debatable.

    `measured` records that the labeller was ASKED about sharp/blurry. Without
    it an absent `blurry` means "never asked" (the first visual labels predate
    the chips), so measured_labels() skips the photo. A sharp/blurry tag in
    the label implies it."""
    yes, debatable = _check(yes, "yes"), _check(debatable, "debatable")
    both = sorted(set(yes) & set(debatable))
    if both:
        raise ValueError(f"tags cannot be both yes and debatable: {both}")
    measured = bool(measured) or bool(_MEASURED & (set(yes) | set(debatable)))
    entry = {"yes": yes, "debatable": debatable, "done": bool(done),
             "measured": measured,
             "updated_at": datetime.now(timezone.utc).isoformat()}
    labels = {str(k): v for k, v in load_labels(label_set).items()}
    labels[str(int(photo_id))] = entry
    _write_atomic(_labels_file(label_set), {"labels": labels})
    return entry


def scoreable_labels(label_set: str = "main") -> dict[int, dict]:
    """Only `done` photos — the only ones whose absent tags mean "no"."""
    return {pid: lab for pid, lab in load_labels(label_set).items() if lab.get("done")}


def self_agreement(recheck_sets: Iterable[str] = ("recheck",)) -> dict:
    """How consistently the labeller applies each tag: first answer vs blind
    recheck on the photos labelled `done` in both. Per tag: `n` photos,
    `agree`, and the disagreements split by direction; a debatable on either
    side is neither. `kappa` is Cohen's kappa on the yes/no calls (None when
    undefined).

    `recheck_sets` pools several pairs — `("recheck", "sharpness-recheck")`
    measures `blurry` over both samples at once."""
    pairs = []
    for rs in recheck_sets:
        if rs not in RECHECK_OF:
            raise ValueError(f"{rs!r} is not a recheck set; have {sorted(RECHECK_OF)}")
        a, b = scoreable_labels(RECHECK_OF[rs]), scoreable_labels(rs)
        pairs += [(a[pid], b[pid]) for pid in sorted(set(a) & set(b))]
    tags = list(PERCEIVED_VOCABULARY) + list(CANDIDATE_TAGS) + list(MEASURED_TAGS)
    per = {}
    for t in tags:
        c = {"n": 0, "agree_yes": 0, "agree_no": 0, "yes_then_no": 0,
             "no_then_yes": 0, "debatable": 0}
        for la, lb in pairs:
            # A label saved before the sharp/blurry chips existed never
            # answered them; its missing `blurry` is not a "no".
            if t in _MEASURED and not (la.get("measured") and lb.get("measured")):
                continue
            ya, da = t in la["yes"], t in la["debatable"]
            yb, db = t in lb["yes"], t in lb["debatable"]
            if da or db:
                c["debatable"] += 1
                continue
            c["n"] += 1
            if ya and yb: c["agree_yes"] += 1
            elif not ya and not yb: c["agree_no"] += 1
            elif ya: c["yes_then_no"] += 1
            else: c["no_then_yes"] += 1
        n = c["n"]
        po = (c["agree_yes"] + c["agree_no"]) / n if n else None
        pa = (c["agree_yes"] + c["yes_then_no"]) / n if n else 0
        pb = (c["agree_yes"] + c["no_then_yes"]) / n if n else 0
        pe = pa * pb + (1 - pa) * (1 - pb)
        c["agreement"] = po
        c["kappa"] = None if n == 0 or pe == 1 else (po - pe) / (1 - pe)
        per[t] = c
    used = {t: c for t, c in per.items()
            if c["agree_yes"] + c["yes_then_no"] + c["no_then_yes"] + c["debatable"]}
    return {"photos": len(pairs), "per_tag": used}


def measured_labels(label_sets: Iterable[str] = ("main", "sharpness")) -> dict[int, dict]:
    """Ground truth for the sharpness measurement: every `done` photo in
    `label_sets`, reduced to MEASURED_TAGS, with its stratum and set.

    {photo_id: {"yes": [...], "debatable": [...], "stratum": str, "set": str}}.
    A done photo with neither tag is a real answer ("neither sharp nor
    blurry"). A photo in both sets keeps the first set's label."""
    out: dict[int, dict] = {}
    for ls in label_sets:
        if ls in RECHECK_OF or ls not in LABEL_SETS:
            raise ValueError(f"{ls!r} is not a first-answer label set")
        strata = {int(p["photo_id"]): p.get("stratum")
                  for p in load_sample(SET_SAMPLE[ls]).get("photos") or []}
        for pid, lab in scoreable_labels(ls).items():
            if pid in out or not lab.get("measured"):
                continue
            out[pid] = {"yes": sorted(set(lab["yes"]) & _MEASURED),
                        "debatable": sorted(set(lab["debatable"]) & _MEASURED),
                        "stratum": strata.get(pid), "set": ls}
    return out


def score(predicted: dict[int, Optional[Iterable[str]]],
          labels: Optional[dict[int, dict]] = None,
          extra_tags: Iterable[str] = ()) -> dict:
    """Per-tag and overall precision/recall of `predicted` {photo_id: tags}.

    Debatable tags are neither a hit nor a miss, in either direction. A photo
    whose prediction is None (no usable answer — see `tag_visual_photo`) is
    counted in `unanswered` and contributes misses for its `yes` tags, since
    that is what the library would actually hold. Non-perceived predicted tags
    are ignored: they never reach the column from the model.

    `extra_tags` are the CANDIDATE_TAGS this variant offered the model; only
    those are scored beyond the perceived vocabulary. A candidate the variant
    never offered is invisible here in both directions. MEASURED_TAGS are
    never scored, whatever is passed: `extra_tags` accepts candidates only,
    and a stored or predicted `sharp`/`blurry` is not a model's answer.
    """
    extra = [t for t in extra_tags if t in CANDIDATE_TAGS]
    vocab = list(PERCEIVED_VOCABULARY) + extra
    allowed = _PERCEIVED | frozenset(extra)
    labels = scoreable_labels() if labels is None else labels
    per = {t: {"tp": 0, "fp": 0, "fn": 0, "debatable": 0} for t in vocab}
    unanswered = scored = 0
    for pid, lab in labels.items():
        if pid not in predicted:
            continue
        scored += 1
        pred = predicted[pid]
        if pred is None:
            unanswered += 1
            pred = ()
        pred = set(pred) & allowed
        yes, deb = set(lab["yes"]) & allowed, set(lab["debatable"]) & allowed
        for t in pred:
            if t in deb:
                per[t]["debatable"] += 1
            elif t in yes:
                per[t]["tp"] += 1
            else:
                per[t]["fp"] += 1
        for t in yes - pred:
            per[t]["fn"] += 1

    def _pr(c: dict) -> dict:
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
        return {**c, "precision": p, "recall": r}

    total = {k: sum(c[k] for c in per.values()) for k in ("tp", "fp", "fn", "debatable")}
    return {"photos_scored": scored, "unanswered": unanswered,
            "overall": _pr(total), "per_tag": {t: _pr(c) for t, c in per.items()}}
