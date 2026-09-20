"""Storage for the visual-tag labelled eval — the contract shared by the
labelling page (`/eval/visual-tags`) and the harness (`evals/visual_tags_eval.py`).

Two JSON files in `eval_dir()`:

    sample.json   {"created": iso, "seed": int,
                   "photos": [{"photo_id": int, "stratum": str}, ...]}
    labels.json   {"labels": {"<photo_id>": {"yes": [tag, ...],
                                             "debatable": [tag, ...],
                                             "done": bool,
                                             "updated_at": iso}}}

A label is THREE-state per tag, and the third state is the point: the A/B that
chose prompt B scored debatable tags as neither right nor wrong, because
forcing a yes/no on `moody` manufactures disagreement that is not the model's
fault. A tag in neither list is a "no" — but only once the photo is `done`;
an un-`done` photo has no opinion at all and must not be scored.

Files, not DB rows: `sync-replica.sh` swaps the replica DB wholesale, and
hand labels are the one thing here that cannot be regenerated. The directory
is git-ignored — labels stay on the owner's machine.

Only PERCEIVED terms are labelled. Derived capture facts come from EXIF and
frozen/retired terms are never asked of the model, so none can be scored.
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
}

_PERCEIVED = frozenset(PERCEIVED_VOCABULARY)
_LABELLABLE = _PERCEIVED | frozenset(CANDIDATE_TAGS)


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
    # truncated JSON where hand labels used to be.
    d = eval_dir()
    d.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=f".{name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, d / name)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_sample() -> dict:
    return _read("sample.json", {"created": None, "seed": None, "photos": []})


def save_sample(photos: list[dict], seed: int) -> dict:
    """`photos` is [{"photo_id": int, "stratum": str}, ...]."""
    data = {"created": datetime.now(timezone.utc).isoformat(), "seed": seed,
            "photos": [{"photo_id": int(p["photo_id"]), "stratum": str(p["stratum"])}
                       for p in photos]}
    _write_atomic("sample.json", data)
    return data


def load_labels() -> dict[int, dict]:
    raw = _read("labels.json", {"labels": {}})["labels"]
    return {int(k): v for k, v in raw.items()}


def _check(tags: Iterable[str], field: str) -> list[str]:
    tags = sorted(set(tags))
    bad = [t for t in tags if t not in _LABELLABLE]
    if bad:
        raise ValueError(f"{field}: not perceived visual tags: {bad}")
    return tags


def save_label(photo_id: int, yes: Iterable[str], debatable: Iterable[str],
               done: bool = True) -> dict:
    """Replace one photo's label. Raises ValueError on a non-perceived tag or
    a tag listed as both yes and debatable."""
    yes, debatable = _check(yes, "yes"), _check(debatable, "debatable")
    both = sorted(set(yes) & set(debatable))
    if both:
        raise ValueError(f"tags cannot be both yes and debatable: {both}")
    entry = {"yes": yes, "debatable": debatable, "done": bool(done),
             "updated_at": datetime.now(timezone.utc).isoformat()}
    labels = {str(k): v for k, v in load_labels().items()}
    labels[str(int(photo_id))] = entry
    _write_atomic("labels.json", {"labels": labels})
    return entry


def scoreable_labels() -> dict[int, dict]:
    """Only `done` photos — the only ones whose absent tags mean "no"."""
    return {pid: lab for pid, lab in load_labels().items() if lab.get("done")}


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
    never offered is invisible here in both directions.
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
