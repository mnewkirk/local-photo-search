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

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from . import model_eval as me

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
