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

# Passes this module can re-run. Mirrors worker_api._ALL_PASSES order.
ALL_PASSES = ("clip", "faces", "quality", "describe",
              "category-content", "category-visual", "keywords", "verify")

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
}


def nas_base() -> Optional[str]:
    """Authoritative-writer base URL, or None when this process IS the writer
    (running on the NAS). Same env var the thumbnail proxy / write tools use."""
    return (os.environ.get("PHOTOSEARCH_NAS_URL") or "").rstrip("/") or None


def _resolve_model(pass_type: str) -> str:
    """Resolve the model name to pass to the worker processor for ``pass_type``.

    On an OpenAI-compatible backend (LM Studio) the per-role env var wins so the
    correct loaded model is used *and* logged; otherwise the Ollama default (or
    an explicit PHOTOSEARCH_LLM_<ROLE>_MODEL override) is used."""
    role, default = _PASS_LLM[pass_type]
    if os.environ.get("PHOTOSEARCH_TEXT_LLM_URL"):
        from .describe import _resolve_openai_model
        return _resolve_openai_model(default, role)
    return os.environ.get(f"PHOTOSEARCH_LLM_{role.upper()}_MODEL") or default


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
            kwargs = {"clip_results": results}
        elif pass_type == "quality":
            results = W._process_quality(downloaded, batch_size=model_batch_size)
            kwargs = {"quality_results": results}
        elif pass_type == "faces":
            results = W._process_faces(downloaded)
            kwargs = {"face_results": results}
        elif pass_type == "describe":
            model = _resolve_model("describe")
            results = W._process_describe(downloaded, model=model)
            kwargs = {"describe_results": results, "model": model,
                      "model_version": W._model_version(model)}
        elif pass_type == "verify":
            regen = _resolve_model("describe")
            results = W._process_verify(downloaded, client=client,
                                        verify_model=_resolve_model("verify"),
                                        regen_model=regen)
            kwargs = {"verify_results": results, "model": regen,
                      "model_version": W._model_version(regen)}
        elif pass_type == "category-content":
            model = _resolve_model("category-content")
            results = W._process_category_content([info], model=model)
            mv = W._model_version(model)
            for r in results:
                r["model"], r["model_version"] = model, mv
            kwargs = {"category_content_results": results}
        elif pass_type == "category-visual":
            model = _resolve_model("category-visual")
            results = W._process_category_visual(downloaded, model=model)
            mv = W._model_version(model)
            for r in results:
                r["model"], r["model_version"] = model, mv
            kwargs = {"category_visual_results": results}
        elif pass_type == "keywords":
            model = _resolve_model("keywords")
            results = W._process_keywords([info], model=model)
            mv = W._model_version(model)
            for r in results:
                r["model"], r["model_version"] = model, mv
            kwargs = {"keywords_results": results}
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
_MIRROR_COLUMNS = (
    "description", "categories", "visual_tags", "keywords", "tags",
    "verified_at", "verification_status", "hallucination_flags",
    "aesthetic_score", "aesthetic_concepts", "aesthetic_critique",
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
    mirrored = errors = missing = 0
    for pid in photo_ids:
        try:
            with urllib.request.urlopen(
                f"{base}/api/photos/{pid}/mirror-fields", timeout=30) as r:
                if r.status == 404:
                    missing += 1
                    continue
                fields = json.loads(r.read())
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
            db.add_face(photo_id=photo_id, bbox=tuple(f["bbox"]),
                        encoding=f["encoding"], det_score=f.get("det_score"))
    db.conn.commit()
