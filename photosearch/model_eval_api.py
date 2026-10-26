"""API behind the model-eval labelling page (`/eval/models`).

The owner labels generated text here (wrong claims in a description, which of
two descriptions is better, the text visible in a photo) and the harnesses in
`evals/` score models against those labels. All storage goes through
`model_eval` — this module adds no format of its own.

Two rules this module exists to keep:

- **Blind.** No response ever carries a variant or model name. Descriptions
  are identified by `text_sha`; the texts of one photo are de-duplicated
  across variants and shuffled with a seed fixed per photo, so a stable order
  gives nothing away either. A test asserts neither string appears.
- **Local-only.** Nothing here proxies to the NAS in replica mode: labels are
  files on whichever machine serves the page (`sync-replica.sh` swaps the DB
  wholesale). The DB is only READ, for stored descriptions and filenames.
  Images come from the local originals cache, so labelling still works while
  the NAS is down.
"""

from __future__ import annotations

import io
import random
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from . import model_eval as me
from . import video_eval_people as vp

router = APIRouter(prefix="/api/eval/models", tags=["eval"])

_SAMPLE_HINT = ("No describe sample yet. Draw one with: "
                "python evals/describe_eval.py sample --db <db>")


class ClaimBody(BaseModel):
    photo_id: int
    wrong: List[int] = []
    other_wrong: bool = False
    done: bool = True


class PrefBody(BaseModel):
    # "a"/"b" are the two sides AS SERVED (left/right), never variants.
    choice: str


class TruthBody(BaseModel):
    text: str = ""
    done: bool = True


def _get_db():
    from . import web
    return web._get_db()


def _stored_descriptions(ids: list[int]) -> dict[int, Optional[str]]:
    out: dict[int, Optional[str]] = {}
    if not ids:
        return out
    try:
        with _get_db() as db:
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                marks = ",".join("?" * len(chunk))
                for pid, desc in db.conn.execute(
                        f"SELECT id, description FROM photos WHERE id IN ({marks})", chunk):
                    out[int(pid)] = desc
    except Exception:
        pass
    return out


def _variants(param: Optional[str]) -> list[str]:
    have = me.list_variants("describe")
    if not param:
        return have
    asked = [v.strip() for v in param.split(",") if v.strip()]
    bad = [v for v in asked if v != me.STORED and v not in have]
    if bad:
        raise HTTPException(400, f"unknown variant(s): {bad}")
    return asked


def texts_by_photo(variants: list[str]) -> dict[int, list[dict]]:
    """{photo_id: [{"sha", "text"}]} — de-duplicated across `variants`,
    shuffled with a per-photo seed. Unanswered (null) texts are skipped:
    there is nothing to label."""
    ids = me.sample_ids("describe")
    found: dict[int, dict[str, str]] = {pid: {} for pid in ids}
    for v in variants:
        if v == me.STORED:
            for pid, text in _stored_descriptions(ids).items():
                if text:
                    found[pid].setdefault(me.text_sha(text), text)
            continue
        run = me.load_run("describe", v) or {"items": {}}
        for pid_s, item in run["items"].items():
            pid = int(pid_s)
            if pid in found and item.get("text"):
                found[pid].setdefault(item.get("text_sha") or me.text_sha(item["text"]),
                                      item["text"])
    out = {}
    for pid, by_sha in found.items():
        texts = [{"sha": s, "text": t} for s, t in sorted(by_sha.items())]
        random.Random(f"describe-claims-{pid}").shuffle(texts)
        out[pid] = texts
    return out


# --------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------

@router.get("/describe/claims")
def get_claims(variants: Optional[str] = Query(None)):
    sample = me.load_sample("describe")
    strata = {p["photo_id"]: p["stratum"] for p in sample["photos"]}
    labels = me.load_claims()
    photos, done_n = [], 0
    for pid, texts in texts_by_photo(_variants(variants)).items():
        entries = [{"sha": t["sha"], "segments": me.segment_claims(t["text"]),
                    "label": labels.get(t["sha"])} for t in texts]
        done = bool(entries) and all((x["label"] or {}).get("done") for x in entries)
        done_n += done
        photos.append({"photo_id": pid, "stratum": strata.get(pid),
                       "texts": entries, "done": done})
    return {"photos": photos, "progress": {"done": done_n, "total": len(photos)},
            "hint": None if photos else _SAMPLE_HINT,
            "eval_dir": str(me.eval_dir())}


@router.put("/describe/claims/{sha}")
def put_claim(sha: str, body: ClaimBody, variants: Optional[str] = Query(None)):
    texts = texts_by_photo(_variants(variants)).get(body.photo_id)
    if texts is None:
        raise HTTPException(404, f"photo {body.photo_id} is not in the describe sample")
    match = [t for t in texts if t["sha"] == sha]
    if not match:
        raise HTTPException(404, "no such description for this photo")
    try:
        label = me.save_claim(sha, body.photo_id, len(me.segment_claims(match[0]["text"])),
                              body.wrong, body.other_wrong, body.done)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"sha": sha, "label": label}


# --------------------------------------------------------------------------
# Pairwise preference
# --------------------------------------------------------------------------

def _served_pairs() -> list[dict]:
    """The pairs as the page sees them: sides a/b in a seeded order, text
    only. Built from pairs.json, whose variant fields never leave the server."""
    texts: dict[str, str] = {}
    for v in me.list_variants("describe"):
        for item in (me.load_run("describe", v) or {"items": {}})["items"].values():
            if item.get("text"):
                texts[item.get("text_sha") or me.text_sha(item["text"])] = item["text"]
    prefs = me.load_prefs()
    out = []
    for p in me.load_pairs()["pairs"]:
        a, b = p["a_sha"], p["b_sha"]
        if a not in texts or b not in texts:
            continue
        if random.Random(p["key"]).random() < 0.5:
            a, b = b, a
        pref = prefs.get(p["key"])
        choice = None
        if pref is not None:
            w = pref.get("winner_sha")
            choice = "tie" if w is None else "a" if w == a else "b"
        out.append({"key": p["key"], "photo_id": p["photo_id"],
                    "a": {"sha": a, "text": texts[a]}, "b": {"sha": b, "text": texts[b]},
                    "choice": choice})
    return out


@router.get("/describe/pairs")
def get_pairs():
    pairs = _served_pairs()
    done = sum(1 for p in pairs if p["choice"])
    return {"pairs": pairs, "progress": {"done": done, "total": len(pairs)},
            "hint": None if pairs else ("No pairs yet. Build them with: python "
                                        "evals/describe_eval.py pairs --baseline <v> "
                                        "--variants <v1>,<v2>")}


@router.put("/describe/pairs/{key}")
def put_pair(key: str, body: PrefBody):
    served = {p["key"]: p for p in _served_pairs()}
    if key not in served:
        raise HTTPException(404, "no such pair")
    if body.choice not in ("a", "b", "tie"):
        raise HTTPException(400, "choice must be a, b or tie")
    p = served[key]
    winner = None if body.choice == "tie" else p[body.choice]["sha"]
    me.save_pref(key, winner)
    return {"key": key, "choice": body.choice}


# --------------------------------------------------------------------------
# Visible text (ground truth for the `text` stratum)
# --------------------------------------------------------------------------

@router.get("/describe/text-truth")
def get_text_truth():
    truth = me.load_text_truth()
    photos = [{"photo_id": p["photo_id"], "truth": truth.get(str(p["photo_id"]))}
              for p in me.load_sample("describe")["photos"] if p["stratum"] == "text"]
    done = sum(1 for p in photos if (p["truth"] or {}).get("done"))
    return {"photos": photos, "progress": {"done": done, "total": len(photos)}}


@router.put("/describe/text-truth/{photo_id}")
def put_text_truth(photo_id: int, body: TruthBody):
    text_ids = {p["photo_id"] for p in me.load_sample("describe")["photos"]
                if p["stratum"] == "text"}
    if photo_id not in text_ids:
        raise HTTPException(404, f"photo {photo_id} is not in the text stratum")
    return {"photo_id": photo_id, "truth": me.save_text_truth(photo_id, body.text, body.done)}


# --------------------------------------------------------------------------
# Pixels from the local originals cache
# --------------------------------------------------------------------------

@router.get("/original/{photo_id}")
def get_original(photo_id: int, max_px: int = Query(1920, ge=200, le=8000)):
    path = me.cached_original(photo_id)
    if path is None:
        raise HTTPException(404, "not in the local originals cache — run "
                                 "`python evals/describe_eval.py fetch-originals`")
    from PIL import Image, ImageOps
    try:
        import photosearch  # noqa: F401 — registers the HEIF opener
        im = Image.open(path)
        try:
            im.draft("RGB", (max_px, max_px))
        except Exception:
            pass
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_px, max_px), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
    except Exception as exc:
        raise HTTPException(500, f"could not decode cached original: {exc}")
    return Response(buf.getvalue(), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=3600"})


# --------------------------------------------------------------------------
# Text passes: pooled category checklist + keyword wrong-marking
# --------------------------------------------------------------------------

class CategoryBody(BaseModel):
    yes: List[str] = []
    done: bool = True


class KeywordBody(BaseModel):
    wrong: List[str] = []
    done: bool = True


def _inputs(name: str) -> dict:
    inputs = me.load_inputs(name)
    if inputs is None:
        raise HTTPException(404, f"no frozen inputs {name!r} — run "
                                 "`python evals/text_passes_eval.py freeze`")
    return inputs


def _stored_column(col: str, ids: list[int]) -> dict[int, list[str]]:
    import json as _json
    out: dict[int, list[str]] = {}
    if not ids:
        return out
    try:
        with _get_db() as db:
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                marks = ",".join("?" * len(chunk))
                for pid, raw in db.conn.execute(
                        f"SELECT id, {col} FROM photos WHERE id IN ({marks})", chunk):
                    try:
                        val = _json.loads(raw) if raw else None
                    except (TypeError, ValueError):
                        val = None
                    if isinstance(val, list):
                        out[int(pid)] = [str(t) for t in val]
    except Exception:
        pass
    return out


def text_pool(pass_: str, inputs: dict) -> dict[str, list[str]]:
    """{pid: sorted union of every run's tags (for these inputs) plus the
    stored column} — unattributed, so the owner judges terms, not models."""
    ident = me.inputs_identity(inputs)
    pool: dict[str, set] = {pid: set() for pid in inputs["items"]}
    for v in me.list_variants(pass_):
        run = me.load_run(pass_, v) or {}
        if run.get("input_source") != ident:
            continue
        for pid, item in run.get("items", {}).items():
            if pid in pool and item.get("tags"):
                pool[pid].update(item["tags"])
    if inputs.get("source") == me.STORED:
        col = "categories" if pass_ == "category-content" else "keywords"
        for pid, tags in _stored_column(col, [int(p) for p in pool]).items():
            pool[str(pid)].update(tags)
    return {pid: sorted(tags) for pid, tags in pool.items()}


@router.get("/text/categories")
def get_categories(inputs: str = Query("main")):
    from .vocab_content import CONTENT_VOCABULARY
    inp = _inputs(inputs)
    labels = me.load_category_labels()
    pool = text_pool("category-content", inp)
    photos = []
    for pid, item in inp["items"].items():
        key = me.label_key(pid, item["text_sha"])
        lab = labels.get(key)
        terms = sorted(set(pool.get(pid, [])) | set((lab or {}).get("yes", []))
                       | set((lab or {}).get("missing", [])))
        photos.append({"photo_id": int(pid), "description": item["text"],
                       "pool": terms, "label": lab})
    done = sum(1 for p in photos if (p["label"] or {}).get("done"))
    return {"photos": photos, "vocabulary": sorted(CONTENT_VOCABULARY),
            "progress": {"done": done, "total": len(photos)}}


@router.put("/text/categories/{photo_id}")
def put_category(photo_id: int, body: CategoryBody, inputs: str = Query("main")):
    inp = _inputs(inputs)
    item = inp["items"].get(str(photo_id))
    if item is None:
        raise HTTPException(404, f"photo {photo_id} is not in inputs {inputs!r}")
    try:
        from .vocab_content import CONTENT_VOCABULARY
        vocab = set(CONTENT_VOCABULARY)
        terms = {" ".join(t.lower().split()) for t in body.yes} - {""}
        # Split, don't reject: a right category the vocabulary lacks is a
        # vocabulary gap, recorded but never scored.
        label = me.save_category_label(me.label_key(photo_id, item["text_sha"]),
                                       sorted(terms & vocab), body.done,
                                       missing=sorted(terms - vocab))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"photo_id": photo_id, "label": label}


@router.get("/text/keywords")
def get_keywords(inputs: str = Query("main")):
    inp = _inputs(inputs)
    labels = me.load_keyword_labels()
    pool = text_pool("keywords", inp)
    photos = []
    for pid, item in inp["items"].items():
        lab = labels.get(me.label_key(pid, item["text_sha"]))
        photos.append({"photo_id": int(pid), "description": item["text"],
                       "pool": pool.get(pid, []), "label": lab})
    done = sum(1 for p in photos if (p["label"] or {}).get("done"))
    return {"photos": photos, "progress": {"done": done, "total": len(photos)}}


@router.put("/text/keywords/{photo_id}")
def put_keyword(photo_id: int, body: KeywordBody, inputs: str = Query("main")):
    inp = _inputs(inputs)
    item = inp["items"].get(str(photo_id))
    if item is None:
        raise HTTPException(404, f"photo {photo_id} is not in inputs {inputs!r}")
    judged = text_pool("keywords", inp).get(str(photo_id), [])
    try:
        label = me.save_keyword_label(me.label_key(photo_id, item["text_sha"]),
                                      judged, body.wrong, body.done)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"photo_id": photo_id, "label": label}


# --------------------------------------------------------------------------
# Verify: confirm each planted claim really is false for the photo
# --------------------------------------------------------------------------

class PlantedBody(BaseModel):
    confirmed: bool


@router.get("/verify/planted")
def get_planted():
    sets = me.load_verify_sets()
    if sets is None:
        return {"items": [], "progress": {"done": 0, "total": 0},
                "hint": "No verify sets yet — run: python evals/verify_eval.py plant --source <variant>"}
    items = [{"id": it["id"], "photo_id": it["photo_id"], "type": it["type"],
              "text": it["text"], "spans": it["spans"], "confirmed": it.get("confirmed")}
             for it in sets["items"] if it["kind"] == "planted"]
    done = sum(1 for it in items if it["confirmed"] is not None)
    return {"items": items, "progress": {"done": done, "total": len(items)}, "hint": None}


@router.put("/verify/planted/{item_id}")
def put_planted(item_id: str, body: PlantedBody):
    try:
        it = me.confirm_planted(item_id, body.confirmed)
    except KeyError:
        raise HTTPException(404, "no such planted item")
    return {"id": item_id, "confirmed": it["confirmed"]}


# --------------------------------------------------------------------------
# Differences: only the facts a photo's descriptions disagree on
# --------------------------------------------------------------------------

class DisputeBody(BaseModel):
    correct: List[str] = []
    cant_tell: bool = False


def _dispute_options(q: dict) -> list[str]:
    """Distinct answers, never attributed. 'Not mentioned' is not an option:
    silence is neither right nor wrong."""
    return me.dispute_options(q)


_DISPUTE_PASSES = ("describe", "video-describe")


def _dispute_pass(name: str) -> str:
    if name not in _DISPUTE_PASSES:
        raise HTTPException(400, f"pass must be one of {_DISPUTE_PASSES}")
    return name


@router.get("/describe/disputes")
def get_disputes(pass_: str = Query("describe", alias="pass")):
    pass_ = _dispute_pass(pass_)
    is_video = pass_ == "video-describe"
    disputes = me.load_disputes(pass_)
    labels = me.load_dispute_labels(pass_)
    strata = {p["photo_id"]: p["stratum"] for p in
              (me.load_video_sample() if is_video else me.load_sample("describe"))["photos"]}
    photos = []
    for pid, d in disputes.items():
        qs = []
        for q in d["questions"]:
            lab = labels.get(q["key"])
            opts = _dispute_options(q)
            seen = (lab or {}).get("options_seen")
            qs.append({"key": q["key"], "question": q["question"], "options": opts,
                       "label": lab,
                       # answers added (by another model) since this was labelled
                       "new_options": [o for o in opts if lab and seen is not None
                                       and o not in seen],
                       "current": me.dispute_label_current(q, lab)})
        if not qs:
            continue
        photos.append({"photo_id": int(pid), "stratum": strata.get(int(pid)), "questions": qs,
                       "media": "video" if is_video else "photo",
                       "done": all(q["current"] for q in qs)})
    # Most disagreement first: that is where a label separates models most.
    photos.sort(key=lambda p: (-len(p["questions"]), p["photo_id"]))
    total_q = sum(len(p["questions"]) for p in photos)
    done_q = sum(1 for p in photos for q in p["questions"] if q["current"])
    return {"photos": photos,
            "progress": {"done": sum(p["done"] for p in photos), "total": len(photos),
                         "points_done": done_q, "points_total": total_q},
            "hint": None if photos else (
                "No disputes yet — run: python evals/"
                + ("video_eval.py" if is_video else "describe_eval.py")
                + " disputes --model <text model>")}


@router.put("/describe/disputes/{key}")
def put_dispute(key: str, body: DisputeBody, pass_: str = Query("describe", alias="pass")):
    pass_ = _dispute_pass(pass_)
    for d in me.load_disputes(pass_).values():
        for q in d["questions"]:
            if q["key"] == key:
                opts = set(_dispute_options(q))
                bad = [c for c in body.correct if c not in opts]
                if bad:
                    raise HTTPException(400, f"not an answer to this question: {bad}")
                return {"key": key, "label": me.save_dispute_label(
                    key, body.correct, body.cant_tell, options_seen=opts, pass_=pass_)}
    raise HTTPException(404, "no such disputed point")


# --------------------------------------------------------------------------
# Video eval (docs/plans/video-support.md step 4): Video subject + Video
# quality tabs, the temporal question, Range-served originals, frame grabs.
#
# Same two rules as above. BLIND: no response carries a variant, model or
# approach name; candidate subject labels are pooled across every run,
# normalized, de-duplicated and shuffled with a per-segment seed, and the
# dev/holdout split is never sent. LOCAL-ONLY: the video bytes come from the
# local originals cache (never the NAS), served with HTTP Range so the
# browser can seek.
# --------------------------------------------------------------------------

_VIDEO_HINT = ("No video sample yet. Draw one with: "
               "python evals/video_eval.py sample --db <db>")
_VIDEO_SUBJECT_PASS = "video-subject"
_VIDEO_DESCRIBE_PASS = "video-describe"


def _video_notes() -> dict:
    from .eval_api import LABELLER_NOTES
    return {k[len("video:"):]: v for k, v in LABELLER_NOTES.items() if k.startswith("video:")}


def _label_set(name: str) -> str:
    if name not in me.VIDEO_LABEL_SETS:
        raise HTTPException(400, f"set must be one of {me.VIDEO_LABEL_SETS}")
    return name


def _video_entries(label_set: str, clips_only: bool = False) -> list[dict]:
    sample = me.load_video_sample()
    entries = list(sample["photos"])
    if label_set == "recheck":
        ids = set(int(i) for i in sample.get("recheck") or [])
        entries = [p for p in entries if int(p["photo_id"]) in ids]
    if clips_only:
        entries = [p for p in entries if me.is_labelled_clip(p)]
    return entries


def _persons() -> list[dict]:
    try:
        with _get_db() as db:
            return [{"id": int(r[0]), "name": r[1]} for r in db.conn.execute(
                "SELECT id, name FROM persons WHERE name IS NOT NULL ORDER BY name COLLATE NOCASE")]
    except Exception:
        return []


def _video_dates(ids) -> dict:
    """{photo_id: 'YYYY-MM-DD HH:MM'} from the DB (capture date, else file
    date). Missing DB / rows -> {} (the page just shows no date)."""
    ids = sorted({int(i) for i in ids})
    if not ids:
        return {}
    try:
        with _get_db() as db:
            out = {}
            for k in range(0, len(ids), 500):
                chunk = ids[k:k + 500]
                for r in db.conn.execute(
                        "SELECT id, COALESCE(date_taken, date_created) FROM photos WHERE id IN (%s)"
                        % ",".join("?" * len(chunk)), chunk):
                    if r[1]:
                        out[int(r[0])] = str(r[1]).replace("T", " ")[:16]
            return out
    except Exception:
        return {}


def candidate_labels(photo_id: int, n_segments: int) -> list[list[str]]:
    """Per label segment, every NON-PERSON subject label any run predicted,
    normalized and de-duplicated, in a seeded shuffled order. Unattributed:
    the labeller judges subjects, never models."""
    from .video_subjects import normalize_label
    seg_v = me.video_seg_version()
    pools: list[set] = [set() for _ in range(n_segments)]
    for v in me.list_variants(_VIDEO_SUBJECT_PASS):
        it = ((me.load_run(_VIDEO_SUBJECT_PASS, v) or {}).get("items") or {}).get(str(photo_id))
        if not it or it.get("seg_version") != seg_v:
            continue
        for i, key in enumerate((it.get("pred") or [])[:n_segments]):
            if key and key[0] == "s" and key[1]:
                pools[i].add(normalize_label(key[1]) or key[1])
    out = []
    for i, pool in enumerate(pools):
        labs = sorted(pool)
        random.Random(f"video-chips-{photo_id}-{seg_v}-{i}").shuffle(labs)
        out.append(labs)
    return out


class VideoSubjectBody(BaseModel):
    seg_version: str
    people: List[int] = []
    someone_else: bool = False
    segments: dict = {}
    live_pair: Optional[str] = None
    done: bool = True


@router.get("/video/subject")
def get_video_subject(set: str = Query("main")):  # noqa: A002 - query name
    label_set = _label_set(set)
    seg_v = me.video_seg_version()
    labels = me.load_video_subject_labels(label_set)
    clusters, names = vp.load_clusters(), vp.load_names()
    videos, done_n = [], 0
    entries = _video_entries(label_set)
    positions = vp.sample_positions()
    dates = _video_dates(p["photo_id"] for p in entries)
    for p in entries:
        pid = int(p["photo_id"])
        live = not me.is_labelled_clip(p)
        segs = [] if live else me.video_label_segments(p.get("duration_s") or 0)
        chips = [] if live else candidate_labels(pid, len(segs))
        who = (labels.get("who") or {}).get(str(pid))
        seg_labels = me.video_segment_labels(labels, pid, seg_v, len(segs))
        done = bool(who and who.get("done") and (live or who.get("seg_version") == seg_v))
        done_n += done
        sigs = vp.segment_signatures(clusters, names, pid, seg_v, len(segs))
        mids = [(a + b) / 2.0 for a, b in segs]
        videos.append({
            "photo_id": pid, "stratum": p["stratum"], "kind": "live" if live else "clip",
            "duration_s": p.get("duration_s"), "live_still_id": p.get("live_still_id"),
            "seg_version": seg_v, "video": vp.page_video_info(pid, router.prefix),
            "segments": [{"ix": i, "start": a, "end": b, "choices": chips[i],
                          "label": seg_labels[i], "people": _sig_people(sigs[i]),
                          "faces": sigs[i][1],
                          "frame_url": f"{router.prefix}/video/frame/{pid}?t={mids[i]:.3f}"
                                       f"&max_px={vp_strip_px}"}
                         for i, (a, b) in enumerate(segs)],
            "runs": [{"segs": r, "people": _sig_people(sigs[r[0]]), "faces": sigs[r[0]][1]}
                     for r in vp.group_runs(sigs)],
            "who": who, "done": done,
            "ident": vp.video_ident(pid, positions, dates)})
    return {"videos": videos, "persons": _persons(), "notes": _video_notes(),
            "set": label_set, "progress": {"done": done_n, "total": len(videos)},
            "clusters_ready": clusters is not None,
            "hint": None if videos else _VIDEO_HINT}


vp_strip_px = 320


def _sig_people(sig) -> list[int]:
    """Named people (persons.id) in one segment signature."""
    return [int(t[1]) for t in sig[0] if t[0] == "p"]


@router.put("/video/subject/{photo_id}")
def put_video_subject(photo_id: int, body: VideoSubjectBody, set: str = Query("main")):  # noqa: A002
    label_set = _label_set(set)
    entry = next((p for p in _video_entries(label_set) if int(p["photo_id"]) == photo_id), None)
    if entry is None:
        raise HTTPException(404, f"video {photo_id} is not in the {label_set} set")
    live = not me.is_labelled_clip(entry)
    if live and body.live_pair is None and body.done:
        raise HTTPException(400, "answer the pairing question")
    n = 0 if live else len(me.video_label_segments(entry.get("duration_s") or 0))
    if not live and body.done:
        missing = [i for i in range(n) if str(i) not in body.segments and i not in body.segments]
        if missing:
            raise HTTPException(400, f"label every segment first (missing {missing})")
    try:
        saved = me.save_video_subject_label(
            photo_id, seg_version=body.seg_version, n_segments=n, people=body.people,
            someone_else=body.someone_else, segments=body.segments,
            live_pair=body.live_pair, done=body.done, label_set=label_set)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"photo_id": photo_id, "label": saved}


class VideoQualityBody(BaseModel):
    in_focus: Optional[str] = None
    shaky: Optional[str] = None
    noisy: Optional[str] = None
    keeper: Optional[str] = None
    done: bool = True


@router.get("/video/quality")
def get_video_quality(set: str = Query("main")):  # noqa: A002
    label_set = _label_set(set)
    labels = me.load_video_quality_labels(label_set)
    videos = []
    for p in _video_entries(label_set, clips_only=True):
        pid = int(p["photo_id"])
        videos.append({"photo_id": pid, "stratum": p["stratum"],
                       "duration_s": p.get("duration_s"), "label": labels.get(str(pid)),
                       "video": vp.page_video_info(pid, router.prefix)})
    done = sum(1 for v in videos if (v["label"] or {}).get("done"))
    return {"videos": videos, "notes": _video_notes(), "set": label_set,
            "progress": {"done": done, "total": len(videos)},
            "hint": None if videos else _VIDEO_HINT}


@router.put("/video/quality/{photo_id}")
def put_video_quality(photo_id: int, body: VideoQualityBody, set: str = Query("main")):  # noqa: A002
    label_set = _label_set(set)
    if photo_id not in {int(p["photo_id"]) for p in _video_entries(label_set, clips_only=True)}:
        raise HTTPException(404, f"video {photo_id} is not in the {label_set} set")
    try:
        lab = me.save_video_quality_label(photo_id, body.dict(exclude={"done"}), body.done,
                                          label_set=label_set)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"photo_id": photo_id, "label": lab}


class TemporalBody(BaseModel):
    photo_id: int
    answer: str


def video_texts() -> dict[int, list[dict]]:
    """{photo_id: [{"sha", "text"}]} over every video-describe run,
    de-duplicated and shuffled per video (blind)."""
    found: dict[int, dict] = {}
    for v in me.list_variants(_VIDEO_DESCRIBE_PASS):
        for pid_s, it in ((me.load_run(_VIDEO_DESCRIBE_PASS, v) or {}).get("items") or {}).items():
            if it.get("text"):
                found.setdefault(int(pid_s), {}).setdefault(
                    it.get("text_sha") or me.text_sha(it["text"]), it["text"])
    out = {}
    for pid, by in found.items():
        texts = [{"sha": s, "text": t} for s, t in sorted(by.items())]
        random.Random(f"video-texts-{pid}").shuffle(texts)
        out[pid] = texts
    return out


@router.get("/video/temporal")
def get_video_temporal():
    labels = me.load_temporal_labels()
    by = video_texts()
    videos = [{"photo_id": pid, "texts": [{"sha": t["sha"], "text": t["text"],
                                           "answer": (labels.get(t["sha"]) or {}).get("answer")}
                                          for t in texts]}
              for pid, texts in sorted(by.items())]
    total = sum(len(v["texts"]) for v in videos)
    done = sum(1 for v in videos for t in v["texts"] if t["answer"])
    return {"videos": videos, "notes": _video_notes(), "progress": {"done": done, "total": total}}


@router.put("/video/temporal/{sha}")
def put_video_temporal(sha: str, body: TemporalBody):
    if not any(t["sha"] == sha for t in video_texts().get(body.photo_id, [])):
        raise HTTPException(404, "no such description for this video")
    try:
        return {"sha": sha, "label": me.save_temporal_label(sha, body.photo_id, body.answer)}
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get("/video/original/{photo_id}")
def get_video_original(photo_id: int, request: Request):
    """The cached original, Range-served (seekable <video>). Never proxied."""
    from . import video_api
    entry = me.video_sample_entry(photo_id)
    path = me.cached_original(photo_id)
    if entry is None or path is None:
        raise HTTPException(404, "not in the local video originals cache — run "
                                 "`python evals/video_eval.py fetch-originals`")
    return video_api.serve_video_file(request.headers.get("range"), str(path),
                                      entry.get("filename") or f"{photo_id}.mp4", None,
                                      {"Cache-Control": "max-age=3600"})


@router.get("/video/frame/{photo_id}")
def get_video_frame(photo_id: int, t: float = Query(0.0, ge=0.0),
                    max_px: int = Query(0, ge=0, le=8000)):
    """One frame of the cached original, at NATIVE resolution by default
    (max_px=0) — focus is judged in PS.Loupe at 1 image px per device px."""
    path = me.cached_original(photo_id)
    if me.video_sample_entry(photo_id) is None or path is None:
        raise HTTPException(404, "not in the local video originals cache")
    if max_px:
        # Downscaled frames (the run strips) are cached on disk —
        # `cluster-faces` pre-writes the keyframes at the strip size.
        try:
            return FileResponse(str(me.cached_frame_jpeg(photo_id, t, max_px)),
                                media_type="image/jpeg",
                                headers={"Cache-Control": "max-age=3600"})
        except Exception as exc:
            raise HTTPException(500, f"could not grab a frame: {exc}")
    try:
        im = me.grab_frame(path, t, max_px)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
    except Exception as exc:
        raise HTTPException(500, f"could not grab a frame: {exc}")
    return Response(buf.getvalue(), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=3600"})


@router.get("/video/info/{photo_id}")
def get_video_info(photo_id: int):
    """What the page needs to pick proxy vs original (PS.chooseVideoSource)."""
    if me.video_sample_entry(photo_id) is None:
        raise HTTPException(404, "not in the video sample")
    return vp.page_video_info(photo_id, router.prefix)


@router.get("/video/proxy/{photo_id}")
def get_video_proxy(photo_id: int, request: Request):
    """The cached H.264 proxy (fetched from the NAS by `video_eval.py
    fetch-proxies`), Range-served. The page plays it when the browser cannot
    play the original (PS.chooseVideoSource); analysis never uses it."""
    from . import video_api
    path = vp.cached_proxy(photo_id)
    if me.video_sample_entry(photo_id) is None or path is None:
        raise HTTPException(404, "no proxy cached — run `python evals/video_eval.py fetch-proxies`")
    return video_api.serve_video_file(request.headers.get("range"), str(path),
                                      f"{photo_id}.mp4", None,
                                      {"Cache-Control": "max-age=3600"})


# ---- "Video people": name each face cluster once --------------------------

_CLUSTER_HINT = ("No face clusters yet. Build them with: "
                 "HSA_ENABLE_DXG_DETECTION=1 python evals/video_eval.py cluster-faces --db <db>")


@router.get("/video/people")
def get_video_people():
    """One card per cross-sample face cluster (largest first), plus the pool
    of crops the owner ejected from a card. Names are keyed by face id."""
    clusters, names = vp.load_clusters(), vp.load_names()
    persons = _persons()
    if not clusters:
        return {"clusters": [], "persons": persons, "hint": _CLUSTER_HINT,
                "progress": {"done": 0, "total": 0}}
    faces = clusters.get("faces") or {}
    by_id = {p["id"]: p["name"] for p in persons}
    positions = vp.sample_positions()
    dates = _video_dates({m["photo_id"] for m in faces.values()})
    cards, done = [], 0
    for c in clusters.get("clusters") or []:
        st = vp.cluster_status(c, names)
        done += st["done"]
        sug = c.get("suggestion")
        per: dict = {}
        for f in c["faces"]:
            if f in faces:
                per[int(faces[f]["photo_id"])] = per.get(int(faces[f]["photo_id"]), 0) + 1
        video_list = [dict(vp.video_ident(pid, positions, dates), faces=n)
                      for pid, n in sorted(per.items(), key=lambda kv: (-kv[1], kv[0]))]
        cards.append({
            "key": c["key"], "size": len(c["faces"]),
            "videos": len(per), "video_list": video_list,
            "faces": [{"id": f, "name": names.get(f)} for f in c["faces"]],
            "suggestion": ({"person_id": sug["person_id"],
                            "name": by_id.get(sug["person_id"]) or sug.get("name")}
                           if sug else None),
            "status": st})
    ejected = sorted(f for f, n in names.items()
                     if (n.get("kind") == "ejected" or n.get("ejected")) and f in faces)
    return {"clusters": cards, "persons": persons,
            "ejected": [{"id": f, "name": names.get(f)} for f in ejected],
            "stats": clusters.get("stats"), "hint": None,
            "progress": {"done": done, "total": len(cards)}}


class FaceNamesBody(BaseModel):
    faces: List[str]
    kind: Optional[str] = None          # person | someone_else | junk | ejected | null = clear
    person_id: Optional[int] = None
    pool: bool = False                  # naming crops from the ejected pool


@router.put("/video/people/names")
def put_video_people_names(body: FaceNamesBody):
    clusters = vp.load_clusters() or {}
    known = clusters.get("faces") or {}
    unknown = [f for f in body.faces if f not in known]
    if unknown:
        raise HTTPException(404, f"unknown face ids {unknown[:3]}")
    ids = {p["id"] for p in _persons()}
    try:
        return vp.save_names(body.faces, body.kind, body.person_id,
                             known_person=(lambda i: i in ids) if ids else None,
                             keep_ejected=body.pool)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get("/video/face-crop/{face_id}")
def get_video_face_crop(face_id: str):
    try:
        path = vp.crop_path(face_id)
    except ValueError:
        raise HTTPException(400, "bad face id")
    if not path.exists():
        raise HTTPException(404, "no such crop")
    return FileResponse(str(path), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})
