"""On-demand re-run of index passes (M28).

Two compute paths, both authoritative-on-NAS / mirror-to-local — the same
read-local / write-NAS / mirror-local model M26b uses for tag/location writes
(see ``photosearch/tools.py`` ``_dual_write_*``). The NAS is a weak N100 with no
GPU, so the *compute* always happens on the desktop (LM Studio / local GPU); the
NAS only stores the result authoritatively.

  ``run_pass_sync``    — compute ONE pass for ONE photo in-process and submit it
                         to the NAS immediately. Instant single-photo feedback.
                         Reuses ``worker._process_*`` + the worker submit
                         endpoint, which applies results by ``photo_id`` even
                         without a live claim, so no claim/heartbeat dance.
  ``requeue_passes``   — clear the (photo, pass) state on the NAS via the
                         existing ``clear-pass`` endpoint so the desktop worker
                         fleet re-processes it. Mirror happens later (poll
                         ``mirror_photos`` once the fleet finishes).
  ``mirror_photos``    — pull authoritative per-photo fields from the NAS and
                         apply them to the local replica DB, so the local UI
                         reflects a re-run without waiting for the nightly
                         ``sync-replica.sh`` full pull.

``nas_base()`` mirrors ``tools._nas_base`` — when ``PHOTOSEARCH_NAS_URL`` is
unset this process IS the authoritative writer (running on the NAS), so there is
nothing to mirror and the sync path has no remote to submit to (callers should
gate sync mode on replica mode).
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from typing import Optional

# Passes this module can re-run, in DEPENDENCY order — the sync re-run and the
# fleet launcher both sort by it (see batch_state.WORKER_PASSES for why verify
# precedes the text passes). Same order as batch_state.WORKER_PASSES.
ALL_PASSES = ("clip", "faces", "quality", "aesthetics", "describe",
              "verify", "category-content", "keywords", "category-visual")

# Passes that read the description from the photo row and need no image download.
TEXT_ONLY_PASSES = {"category-content", "keywords"}

# pass -> (LLM role, Ollama-style default model). The role drives LM Studio model
# selection (PHOTOSEARCH_LLM_<ROLE>_MODEL via describe._resolve_openai_model);
# the default is the Ollama name + what gets logged when no role env is set.
_PASS_LLM = {
    "describe":         ("describe", "llama3.2-vision"),
    "verify":           ("verify",   "llava"),
    "category-content": ("text",     "llama3.2:3b"),
    "keywords":         ("text",     "llama3.2:3b"),
    "category-visual":  ("visual",   "llava"),
    "aesthetics":       ("aesthetics", "qwen2.5-vl"),
}


def nas_base() -> Optional[str]:
    """Authoritative-writer base URL, or None when this process IS the writer
    (running on the NAS). Same env var the thumbnail proxy / write tools use."""
    return (os.environ.get("PHOTOSEARCH_NAS_URL") or "").rstrip("/") or None


# The model each pass runs on LM Studio when the worker fleet is launched from
# the web UI (admin_api._fleet_env: /admin/maintenance and /batches). Chosen by
# the model evals — docs/plans/model-eval-harnesses.md, 2026-09-28:
#   verify -> gemma-4-12b-qat: named 16/20 planted errors and wrongly rejected
#             1/20 clean descriptions; the old verifier (qwen2.5-vl via the VISUAL
#             fallback) had never been measured, gemma-4-e2b named 5/20, rejected 8/20.
#   visual -> minicpm-v-4_5: precision 0.54 / recall 0.56 on the owner's 60 labels
#             vs qwen2.5-vl-7b 0.51 / 0.40.
#   describe -> qwen3.5-9b: right on 0.75 of disputed facts vs qwen2.5-vl-7b 0.50
#             (the model that had actually been describing), 1.04 vs 1.46 wrong
#             claims per photo, fewer wrong on 33 photos to 16 (p 0.02); every
#             other candidate scored lower too.
#   text     -> gemma-4-12b-qat (category-content + keywords share the role): on
#             the owner's 70 labelled descriptions, category precision 0.83 —
#             the same as llama-3.2-3b — with 11.8 right categories per photo
#             vs 4.5 (more right on 64 photos, fewer on 1). Keyword precision a
#             tie (0.93 vs 0.94). Also the verify model, so no swap between them.
# aesthetics stays on what production ran: all four candidates tied. Explicit per role, deliberately: the server's own
# PHOTOSEARCH_LLM_VISUAL_MODEL also drives rerank_photos and the photobook hero
# picks, and letting the fleet's vision roles fall back to it is how describe,
# verify and aesthetics all silently ran on qwen2.5-vl. Override one role for the
# UI fleet with PHOTOSEARCH_FLEET_<ROLE>_MODEL.
FLEET_ROLE_MODELS = {"describe":   "qwen/qwen3.5-9b",
                     "verify":     "google/gemma-4-12b-qat",
                     "visual":     "minicpm-v-4_5",
                     "aesthetics": "qwen2.5-vl-7b-instruct",
                     "text":       "google/gemma-4-12b-qat"}


# Every model eval that picked FLEET_ROLE_MODELS ran with reasoning OFF
# (docs/plans/model-eval-harnesses.md). gemma-4 and minicpm-v-4_5 think by
# default, and with thinking on the fleet is a different, broken system: on the
# 2026-10-03 batch gemma-4-12b spent 297 of 300 tokens reasoning and returned ''
# in 70 s, so every category-content call hit the 10 s text cap and deferred
# forever (0 of 1,259 done); minicpm ran out its 768-token budget mid-thought on
# 307 photos, which burned all three attempts. With "none": ~1-7 s, real answers.
# Override for the UI fleet with PHOTOSEARCH_FLEET_REASONING_EFFORT.
FLEET_REASONING_EFFORT = "none"


def fleet_role_model(role: str, env=None) -> str:
    env = os.environ if env is None else env
    return env.get(f"PHOTOSEARCH_FLEET_{role.upper()}_MODEL") or FLEET_ROLE_MODELS[role]


# LM Studio fallback models when no role env var is configured. These are this
# deployment's loaded models; override with PHOTOSEARCH_LLM_<ROLE>_MODEL. The
# raw Ollama defaults in _PASS_LLM are NOT valid LM Studio ids, so without these
# the OpenAI-compatible call 404s and the pass silently defers.
_LMSTUDIO_DEFAULTS = {"describe": "qwen2.5-vl-7b-instruct",
                      "verify":   "qwen2.5-vl-7b-instruct",
                      "visual":   "qwen2.5-vl-7b-instruct",
                      "aesthetics": "qwen2.5-vl-7b-instruct",
                      "text":     "llama-3.2-3b-instruct"}


def _resolve_model(pass_type: str) -> str:
    """Resolve the model name to pass to the worker processor for ``pass_type``.

    On an OpenAI-compatible backend (LM Studio) the per-role env var wins; absent
    that, vision roles reuse PHOTOSEARCH_LLM_VISUAL_MODEL if set, then fall back
    to this box's loaded models (_LMSTUDIO_DEFAULTS). Otherwise the Ollama
    default (or an explicit PHOTOSEARCH_LLM_<ROLE>_MODEL override) is used."""
    role, default = _PASS_LLM[pass_type]
    if os.environ.get("PHOTOSEARCH_TEXT_LLM_URL"):
        env = (os.environ.get(f"PHOTOSEARCH_LLM_{role.upper()}_MODEL")
               or os.environ.get("PHOTOSEARCH_TEXT_LLM_MODEL"))
        if env:
            return env
        if role in ("describe", "verify", "aesthetics"):
            return os.environ.get("PHOTOSEARCH_LLM_VISUAL_MODEL") or _LMSTUDIO_DEFAULTS[role]
        return _LMSTUDIO_DEFAULTS[role]
    return os.environ.get(f"PHOTOSEARCH_LLM_{role.upper()}_MODEL") or default


def _model_version(model: str) -> Optional[str]:
    """Provenance digest for the generations log.

    Thin alias for the shared ``describe.effective_model_version`` — the worker
    fleet logs through the same helper, so a re-run and a fleet pass can never
    disagree about how a model is identified.
    """
    from .describe import effective_model_version

    return effective_model_version(model)


# ---------------------------------------------------------------------------
# Queue path — re-queue (photo, pass) on the authoritative server.
# ---------------------------------------------------------------------------

def requeue_passes(photo_ids: list[int], passes: list[str],
                   server: Optional[str] = None) -> dict:
    """Clear processing state for each pass on the given photos so a worker
    re-processes them. Targets the NAS in replica mode, else the local DB.

    Returns ``{pass: {cleared, photo_count}}``. Raises on an unknown pass."""
    bad = [p for p in passes if p not in ALL_PASSES]
    if bad:
        raise ValueError(f"unknown pass type(s): {', '.join(bad)}")
    if not photo_ids:
        raise ValueError("photo_ids is empty")

    base = server or nas_base()
    out: dict = {}
    if base:
        from .worker import WorkerClient
        client = WorkerClient(base, probe=False)
        for pass_type in passes:
            # clear_pass takes collection/directory; add photo_ids via raw POST.
            resp = client.session.post(
                f"{client.server_url}/api/worker/clear-pass",
                json={"pass_type": pass_type, "photo_ids": photo_ids},
                timeout=60,
            )
            resp.raise_for_status()
            out[pass_type] = resp.json()
    else:
        # Running on the NAS — clear directly via the in-process endpoint logic.
        from .worker_api import clear_pass, ClearPassRequest
        for pass_type in passes:
            out[pass_type] = clear_pass(
                ClearPassRequest(pass_type=pass_type, photo_ids=photo_ids))
    return out


# ---------------------------------------------------------------------------
# Sync path — compute one pass for one photo in-process, submit to the NAS.
# ---------------------------------------------------------------------------

def run_pass_sync(db, photo_id: int, pass_type: str,
                  server: Optional[str] = None,
                  model_batch_size: int = 8) -> dict:
    """Compute ``pass_type`` for one photo here, submit to the authoritative
    server, and mirror the result into the local DB.

    ``db`` is the LOCAL replica PhotoDB (used to read the photo row + mirror the
    result). ``server`` defaults to ``nas_base()``. Returns
    ``{pass, photo_id, written, mirrored, ...}``.
    """
    if pass_type not in ALL_PASSES:
        raise ValueError(f"unknown pass type: {pass_type}")
    base = server or nas_base()
    if not base:
        raise RuntimeError(
            "synchronous re-run needs an authoritative server "
            "(PHOTOSEARCH_NAS_URL); on the NAS itself, use the worker fleet")

    from . import worker as W

    row = db.get_photo(photo_id)
    if not row:
        raise ValueError(f"photo {photo_id} not found in local DB")
    info = {
        "id": row["id"],
        "filepath": row["filepath"],
        "filename": os.path.basename(row["filepath"]),
        "description": row.get("description"),
    }

    client = W.WorkerClient(base, probe=False)
    needs_image = pass_type not in TEXT_ONLY_PASSES
    tmpdir = tempfile.mkdtemp(prefix="photosearch-rerun-")
    try:
        downloaded = None
        if needs_image:
            downloaded = W._download_batch(client, [info], tmpdir)
            if not downloaded:
                raise RuntimeError(
                    f"could not download photo {photo_id} from {base}")

        if pass_type == "clip":
            results = W._process_clip(downloaded, batch_size=model_batch_size)
            kwargs = W._submit_kwargs("clip_results", results)
        elif pass_type == "quality":
            results = W._process_quality(downloaded, batch_size=model_batch_size)
            kwargs = W._submit_kwargs("quality_results", results)
        elif pass_type == "faces":
            results = W._process_faces(downloaded)
            kwargs = W._submit_kwargs("face_results", results)
        elif pass_type == "describe":
            model = _resolve_model("describe")
            results = W._process_describe(downloaded, model=model)
            kwargs = {"describe_results": results, "model": model,
                      "model_version": _model_version(model)}
        elif pass_type == "verify":
            regen = _resolve_model("describe")
            results = W._process_verify(downloaded, client=client,
                                        verify_model=_resolve_model("verify"),
                                        regen_model=regen)
            kwargs = {**W._submit_kwargs("verify_results", results),
                      "model": regen, "model_version": _model_version(regen)}
        elif pass_type == "category-content":
            model = _resolve_model("category-content")
            results = W._process_category_content([info], model=model)
            mv = _model_version(model)
            for r in results:
                r["model"], r["model_version"] = model, mv
            kwargs = {"category_content_results": results}
        elif pass_type == "category-visual":
            model = _resolve_model("category-visual")
            results = W._process_category_visual(downloaded, model=model)
            mv = _model_version(model)
            for r in results:
                r["model"], r["model_version"] = model, mv
            kwargs = {"category_visual_results": results}
        elif pass_type == "keywords":
            model = _resolve_model("keywords")
            results = W._process_keywords([info], model=model)
            mv = _model_version(model)
            for r in results:
                r["model"], r["model_version"] = model, mv
            kwargs = {"keywords_results": results}
        elif pass_type == "aesthetics":
            model = _resolve_model("aesthetics")
            results = W._process_aesthetics(downloaded, model=model)
            mv = _model_version(model)
            for r in results:
                r["model"], r["model_version"] = model, mv
            kwargs = {"aesthetics_results": results}
        else:  # pragma: no cover - guarded above
            raise ValueError(f"unknown pass type: {pass_type}")

        # Submit to the NAS. submit-results applies by photo_id even without a
        # live claim (it logs a warning), so a synthetic batch_id is fine.
        resp = client.submit_results(uuid.uuid4().hex, pass_type, **kwargs)
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    mirror = mirror_photos(db, [photo_id], server=base)
    return {
        "pass": pass_type,
        "photo_id": photo_id,
        "written": resp.get("written", 0),
        "processed": resp.get("processed", 0),
        # text-only passes omit a result on timeout/error (deferred for retry)
        "deferred": not results,
        "mirrored": mirror.get("mirrored", 0),
        "authority": "nas",
    }


# ---------------------------------------------------------------------------
# Mirror path — pull authoritative fields from the NAS into the local DB.
# ---------------------------------------------------------------------------

# Scalar / text columns mirrored verbatim from /mirror-fields.
def _aes_mirror_columns() -> tuple[str, ...]:
    from .aesthetics import ALL_SUBATTRS, DIMENSIONS
    return (
        "aes_overall", "aes_overall_pct", "aes_technical_iqa", "aes_overall_iqa",
        "aes_style", "aes_style_tags", "aes_model", "aes_scored_at",
        *(f"aes_{s}" for s in ALL_SUBATTRS),
        *(f"aes_{d}" for d in DIMENSIONS),
    )


_MIRROR_COLUMNS = (
    "description", "categories", "visual_tags", "keywords", "tags",
    "verified_at", "verification_status", "hallucination_flags",
    "aesthetic_score", "aesthetic_concepts", "aesthetic_critique",
    *_aes_mirror_columns(),
    # schema v33 — measured on the NAS (sharpness_backfill), carried so a
    # targeted mirror doesn't leave the replica's copy staler than the rest.
    # An older NAS omits them, and `if c in fields` below skips the absent.
    "sharpness", "sharpness_json", "sharpness_version", "sharpness_scored_at",
)


def mirror_photos(db, photo_ids: list[int], server: Optional[str] = None) -> dict:
    """Fetch authoritative per-photo fields from the NAS and apply them to the
    local replica DB. No-op (mirrored=0) when not in replica mode.

    Mirrors text/scalar columns, the CLIP embedding, and face rows so the local
    search index reflects a re-run immediately. Returns
    ``{mirrored, errors, missing}``."""
    base = server or nas_base()
    if not base:
        return {"mirrored": 0, "errors": 0, "missing": 0, "skipped": "not replica"}

    import urllib.request
    import urllib.error
    mirrored = errors = missing = 0
    for pid in photo_ids:
        try:
            with urllib.request.urlopen(
                f"{base}/api/photos/{pid}/mirror-fields", timeout=30) as r:
                fields = json.loads(r.read())
        except urllib.error.HTTPError as e:
            # 404 → no such photo on the NAS (or NAS predates /mirror-fields);
            # count as missing, not a transport error.
            if e.code == 404:
                missing += 1
            else:
                errors += 1
            continue
        except Exception:
            errors += 1
            continue
        try:
            _apply_mirror(db, pid, fields)
            mirrored += 1
        except Exception:
            errors += 1
    return {"mirrored": mirrored, "errors": errors, "missing": missing}


def _apply_mirror(db, photo_id: int, fields: dict) -> None:
    """Apply one photo's authoritative fields to the local DB in a transaction."""
    updates = {c: fields[c] for c in _MIRROR_COLUMNS if c in fields}
    if updates:
        db.update_photo(photo_id, **updates)

    if "clip_embedding" in fields:
        emb = fields["clip_embedding"]
        # DELETE+INSERT — vec0 doesn't honor OR REPLACE (see CLAUDE.md).
        db.conn.execute("DELETE FROM clip_embeddings WHERE photo_id=?", (photo_id,))
        if emb:
            db.add_clip_embedding(photo_id, emb)

    if "faces" in fields:
        faces = fields["faces"] or []
        old = [r[0] for r in db.conn.execute(
            "SELECT id FROM faces WHERE photo_id=?", (photo_id,)).fetchall()]
        if old:
            ph = ",".join("?" * len(old))
            db.conn.execute(f"DELETE FROM face_encodings WHERE face_id IN ({ph})", old)
            db.conn.execute("DELETE FROM faces WHERE photo_id=?", (photo_id,))
        for f in faces:
            fid = db.add_face(photo_id=photo_id, bbox=tuple(f["bbox"]),
                              encoding=f["encoding"], det_score=f.get("det_score"),
                              person_id=f.get("person_id"),
                              cluster_id=f.get("cluster_id"))
            # match_source isn't an add_face param — set it so mirrored manual /
            # temporal assignments keep their provenance locally.
            if f.get("match_source") is not None:
                db.conn.execute("UPDATE faces SET match_source=? WHERE id=?",
                                (f["match_source"], fid))
    db.conn.commit()
