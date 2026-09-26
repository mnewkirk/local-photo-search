"""Remote worker — runs on a fast machine, processes photos from a NAS server.

Usage:
    python cli.py worker --server http://nas.local:8000 --passes clip --collection 3

The worker loop:
  1. Claims a batch of unprocessed photos from the server
  2. Downloads photo bytes to a temp directory
  3. Runs the specified indexing pass locally (fast GPU/CPU)
  4. POSTs results back to the server
  5. Cleans up temp files, repeats
"""

import gc
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import requests
from requests.exceptions import (
    ConnectionError as ReqConnectionError,
    Timeout as ReqTimeout,
    ReadTimeout as ReqReadTimeout,
    ChunkedEncodingError,
    ContentDecodingError,
)

# Transient errors worth retrying (network drop, sleep/wake, mid-stream truncation)
_TRANSIENT = (
    ReqConnectionError,
    ReqTimeout,
    ChunkedEncodingError,
    ContentDecodingError,
    ConnectionError,
    TimeoutError,
)

# Cache of Ollama model name -> short digest, for generation provenance.
_MODEL_VERSION_CACHE: dict = {}


def _model_version(model: str) -> Optional[str]:
    """Best-effort short digest for an Ollama model, for the generations log.

    Returns None on any failure — provenance is nice-to-have, never fatal.
    """
    if model in _MODEL_VERSION_CACHE:
        return _MODEL_VERSION_CACHE[model]
    version = None
    try:
        import ollama
        for m in ollama.list().models:
            name = m.model or ""
            if name == model or name.split(":")[0] == model or name.startswith(model):
                digest = getattr(m, "digest", None)
                version = digest[:12] if digest else None
                break
    except Exception:
        version = None
    _MODEL_VERSION_CACHE[model] = version
    return version


# Passes whose provenance rides on the REQUEST rather than on each result row.
# (`submit_results` reads `req.model` for these and `r.model` for the rest.)
_BATCH_LEVEL_PROVENANCE = {"describe", "verify"}


def _provenance_kwargs(pass_type: str, model: str, results: list,
                       role: Optional[str] = None) -> dict:
    """Stamp `model` / `model_version` for one submitted batch.

    Reports the model that ACTUALLY ran, not the name this worker was
    configured with: on the OpenAI-compatible (LM Studio) route the configured
    name is ignored and the model is chosen by ROLE, so logging the CLI default
    recorded a model that never executed — which is why 159,647 of 159,650
    `category-visual` generations on the live library claim `llava`.

    Shares `describe.effective_model` / `effective_model_version` with
    `rerun.py`, which already resolved this correctly. Don't reintroduce a
    second copy — a wrong provenance string still looks like a string, so
    drift here is invisible until someone asks which model tagged a photo.

    `role` overrides the pass's own role for the case where the logged artifact
    was produced by a DIFFERENT call than the pass is named for: `verify` logs
    the regenerated description, which the regen (describe-role) model wrote,
    not the verifier.

    Returns the kwargs to merge into the submit call; per-result passes are
    stamped in place on `results`.
    """
    role = role or describe_module_roles().get(pass_type)
    if role is None:
        return {}
    from .describe import effective_model, effective_model_version

    resolved = effective_model(model, role)
    version = effective_model_version(resolved)
    if pass_type in _BATCH_LEVEL_PROVENANCE:
        return {"model": resolved, "model_version": version}
    for r in results:
        r["model"] = resolved
        r["model_version"] = version
    return {}


def describe_module_roles() -> dict:
    """`describe.PASS_ROLES`, imported lazily (describe pulls in ollama)."""
    from .describe import PASS_ROLES

    return PASS_ROLES


def _unload_pass_models(pass_type: str) -> None:
    """Release torch models owned by a pass so MPS/CUDA memory is reclaimed.

    Ollama-backed passes (describe/tags/verify) keep their models in the
    sidecar, so there's nothing to unload here for those.
    """
    if pass_type == "clip":
        from .clip_embed import unload_model as _unload
        _unload()
    elif pass_type == "quality":
        from .quality import unload_models as _unload
        _unload()
    elif pass_type == "faces":
        from .faces import unload_model as _unload
        _unload()
    elif pass_type == "verify":
        # Verify borrows clip_embed for its cross-check embeddings.
        from .clip_embed import unload_model as _unload
        _unload()


def _flush_caches() -> None:
    """Drop tensor allocator caches between batches to prevent drift."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def _retry(fn, max_retries=5, base_delay=5, label="request"):
    """Retry a callable on transient network errors (sleep/wake recovery).

    Does NOT retry HTTP 4xx/5xx — only connection-level failures.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except _TRANSIENT as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)  # 5, 10, 20, 40, 80
            print(f"  ⚠ {label} failed ({e.__class__.__name__}), retrying in {delay}s "
                  f"(attempt {attempt + 1}/{max_retries})...")
            time.sleep(delay)


class WorkerClient:
    """HTTP client for the worker API on the NAS."""

    def __init__(self, server_url: str, worker_id: str = None, probe: bool = True):
        self.server_url = server_url.rstrip("/")
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.session = requests.Session()
        # Quick connectivity test. `/api/stats` runs heavy count scans and can
        # take >10s on a cold N100 NAS (full-table COUNT/MIN/MAX over photos,
        # faces, clip_embeddings). A *read* timeout means the TCP connection
        # succeeded — the server is reachable, just slow to compute stats — so
        # it's a false negative to treat it as unreachable (this is why the
        # first fleet launch failed and the second, cache-warm, succeeded).
        # Only genuine connection failures are fatal; a read timeout warns and
        # proceeds (the real claim/download/submit calls have their own timeouts
        # + retries). Callers that just submit a single result (the M28 sync
        # re-run path) pass probe=False to skip it entirely.
        if probe:
            try:
                r = self.session.get(f"{self.server_url}/api/stats", timeout=30)
                r.raise_for_status()
            except ReqReadTimeout:
                print(f"  ⚠ {self.server_url}/api/stats slow to respond (cold cache?) — "
                      f"server is reachable, continuing")
            except Exception as e:
                raise ConnectionError(f"Cannot reach server at {self.server_url}: {e}")

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Send a request, transparently backing off on HTTP 503 (server in
        graceful-shutdown mode signaling 'retry shortly') by sleeping for
        Retry-After seconds and looping. Connection-level errors are NOT
        handled here — caller's _retry() covers those.

        The server returns 503 to /api/worker/* and /api/photos/*/full while
        uvicorn is draining for a restart. Workers see this, back off (so
        uvicorn's drain actually completes), then come back when the new
        container is up and answering 200.
        """
        while True:
            r = self.session.request(method, url, **kwargs)
            if r.status_code != 503:
                return r
            retry_after = 30
            try:
                retry_after = max(1, int(r.headers.get("Retry-After", "30")))
            except (TypeError, ValueError):
                pass
            try:
                r.close()
            except Exception:
                pass
            print(f"  ⏸ server is restarting (503) — backing off {retry_after}s before retry")
            time.sleep(retry_after)

    def claim_batch(self, pass_type: str, limit: int = 16,
                    collection_id: Optional[int] = None,
                    directory: Optional[str] = None,
                    filters: Optional[dict] = None,
                    ttl_minutes: int = 30) -> dict:
        """Claim a batch of photos. Returns {batch_id, pass_type, photos: [...]}."""
        payload = {
            "worker_id": self.worker_id,
            "pass_type": pass_type,
            "limit": limit,
            "ttl_minutes": ttl_minutes,
        }
        if collection_id is not None:
            payload["collection_id"] = collection_id
        if directory is not None:
            payload["directory"] = directory
        if filters:
            payload["filters"] = filters
        r = self._request("POST", f"{self.server_url}/api/worker/claim-batch", json=payload, timeout=30)
        if not r.ok:
            detail = r.text[:200] if r.text else r.reason
            raise RuntimeError(f"claim-batch failed ({r.status_code}): {detail}")
        return r.json()

    def download_photo(self, photo_id: int, dest_path: str) -> int:
        """Download a photo's full-resolution bytes. Returns HTTP status code
        (200 on success, 404 if the NAS can't find the file, etc.)."""
        r = self._request(
            "GET",
            f"{self.server_url}/api/photos/{photo_id}/full",
            timeout=120,
            stream=True,
        )
        if r.status_code != 200:
            return r.status_code
        try:
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
        except _TRANSIENT:
            # Let the caller's retry loop handle it; clean up partial file
            try:
                os.remove(dest_path)
            except OSError:
                pass
            raise
        return 200

    def submit_results(self, batch_id: str, pass_type: str, **kwargs) -> dict:
        """Submit results for a batch. Returns {status, written, batch_id}."""
        payload = {
            "batch_id": batch_id,
            "pass_type": pass_type,
            **kwargs,
        }
        r = self._request(
            "POST",
            f"{self.server_url}/api/worker/submit-results",
            json=payload,
            timeout=120,
        )
        if not r.ok:
            detail = r.text[:200] if r.text else r.reason
            raise RuntimeError(f"submit-results failed ({r.status_code}): {detail}")
        return r.json()

    def get_status(self, collection_id: Optional[int] = None,
                   directory: Optional[str] = None,
                   passes: Optional[list[str]] = None,
                   filters: Optional[dict] = None) -> dict:
        """Get worker queue status.

        If passes is provided, the server only computes queue depth for those
        pass types — much cheaper than counting all six on a large library.
        """
        params: dict = {}
        if collection_id is not None:
            params["collection_id"] = collection_id
        if directory is not None:
            params["directory"] = directory
        if passes:
            params["passes"] = ",".join(passes)
        if filters:
            params["filters"] = json.dumps(filters)
        r = self.session.get(f"{self.server_url}/api/worker/status", params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    def clear_pass(self, pass_type: str, collection_id: Optional[int] = None,
                   directory: Optional[str] = None,
                   filters: Optional[dict] = None) -> dict:
        """Clear processing state for a pass type on a collection, directory, or filter set."""
        payload = {"pass_type": pass_type}
        if collection_id is not None:
            payload["collection_id"] = collection_id
        if directory is not None:
            payload["directory"] = directory
        if filters:
            payload["filters"] = filters
        r = self.session.post(
            f"{self.server_url}/api/worker/clear-pass",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def renew_claim(self, batch_id: str, ttl_minutes: int = 30) -> bool:
        """Extend a claim's TTL (heartbeat). Returns True on success."""
        try:
            r = self._request(
                "POST",
                f"{self.server_url}/api/worker/renew-claim",
                json={"batch_id": batch_id, "ttl_minutes": ttl_minutes},
                timeout=30,
            )
            return r.ok
        except Exception:
            return False

    def get_photo_detail(self, photo_id: int) -> Optional[dict]:
        """Get photo metadata + CLIP embedding for verify pass."""
        r = self.session.get(
            f"{self.server_url}/api/worker/photo-detail/{photo_id}",
            timeout=30,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()


class _ClaimHeartbeat:
    """Background thread that periodically renews a worker claim to prevent expiry."""

    def __init__(self, client: WorkerClient, batch_id: str, ttl_minutes: int):
        self._client = client
        self._batch_id = batch_id
        self._ttl_minutes = ttl_minutes
        # Renew at 40% of TTL, capped at 120s so even short jobs get heartbeats
        self._interval = max(30, min(ttl_minutes * 60 * 0.4, 120))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.healthy = True  # False if renewal has failed repeatedly

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self):
        consecutive_failures = 0
        while not self._stop.wait(self._interval):
            ok = self._client.renew_claim(self._batch_id, self._ttl_minutes)
            if ok:
                consecutive_failures = 0
                print(f"  ♻ Renewed claim {self._batch_id[:8]}... (next in {self._interval:.0f}s)")
            else:
                # Retry once after a short delay before giving up
                if not self._stop.wait(5):
                    ok = self._client.renew_claim(self._batch_id, self._ttl_minutes)
                if ok:
                    consecutive_failures = 0
                    print(f"  ♻ Renewed claim {self._batch_id[:8]}... (retry succeeded)")
                else:
                    consecutive_failures += 1
                    self.healthy = consecutive_failures < 3
                    print(f"  ⚠ Failed to renew claim {self._batch_id[:8]}... ({consecutive_failures} consecutive failures)")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def _download_batch(client: WorkerClient, photos: list[dict], temp_dir: str) -> list[tuple[dict, str]]:
    """Download photos to temp_dir. Returns [(photo_info, local_path)] for successful downloads."""
    downloaded = []
    for photo in photos:
        local_path = os.path.join(temp_dir, f"{photo['id']}_{photo['filename']}")
        print(f"    Downloading {photo['filename']}...", end="", flush=True)
        t0 = time.time()
        try:
            status = _retry(
                lambda pid=photo["id"], lp=local_path: client.download_photo(pid, lp),
                label=f"download {photo['filename']}",
            )
        except _TRANSIENT as e:
            print(f" FAILED after retries ({e.__class__.__name__})")
            continue
        if status == 200:
            elapsed = time.time() - t0
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f" {size_mb:.1f}MB ({elapsed:.1f}s)")
            downloaded.append((photo, local_path))
        else:
            print(f" FAILED (HTTP {status})")
    return downloaded


def _failure_row(photo_id: int, error) -> dict:
    """A photo this worker tried and could NOT process (unloadable image,
    detection/verification raised). Split out of the result list by
    `_submit_kwargs` and sent as `failures`, which spends one of the photo's
    MAX_PROCESS_ATTEMPTS on the server without writing anything — so a poison
    photo is retired by the cap instead of re-claimed forever.

    NOT for timeouts or network errors: those are the worker's problem, not
    the photo's, and are omitted from the payload so they cost nothing.
    """
    return {"photo_id": photo_id, "error": (str(error) or type(error).__name__)[:500]}


def _is_failure_row(row: dict) -> bool:
    return "error" in row


def _submit_kwargs(results_key: str, results: list[dict]) -> dict:
    """Build submit kwargs, moving failure rows into the separate `failures`
    list. Kept out of the per-pass result rows on purpose: an older server
    ignores the unknown `failures` field, whereas a failure folded into a
    result row would 422 (QualityResult.aesthetic_score is required there) or
    be written as a real verification."""
    ok = [r for r in results if not _is_failure_row(r)]
    failed = [r for r in results if _is_failure_row(r)]
    kwargs = {results_key: ok}
    if failed:
        kwargs["failures"] = failed
    return kwargs


def _process_clip(downloaded: list[tuple[dict, str]], batch_size: int = 8) -> list[dict]:
    """Run CLIP embedding on downloaded photos. Returns list of {photo_id, embedding}."""
    from .clip_embed import embed_images_stream

    paths = [path for _, path in downloaded]
    results = []
    embedded = set()
    for idx, emb in embed_images_stream(paths, batch_size=batch_size):
        photo_info = downloaded[idx][0]
        results.append({"photo_id": photo_info["id"], "embedding": emb})
        embedded.add(idx)
    # embed_images_stream skips an image it cannot open. Report it, so the
    # attempts cap retires it — before this it left no trace and headed every
    # clip claim forever (a ZIP-wrapped Live Photo saved as .JPG).
    for idx, (photo_info, _) in enumerate(downloaded):
        if idx not in embedded:
            results.append(_failure_row(
                photo_info["id"], "CLIP embedding failed (image could not be loaded)"))
    return results


def _process_quality(downloaded: list[tuple[dict, str]], batch_size: int = 8) -> list[dict]:
    """Run aesthetic scoring on downloaded photos."""
    from .quality import score_photos_stream, analyze_photos_stream

    paths = [path for _, path in downloaded]
    results = []

    # Score
    scores = {}
    for idx, score in score_photos_stream(paths, batch_size=batch_size):
        photo_info = downloaded[idx][0]
        scores[photo_info["id"]] = score

    # Concept analysis
    concepts = {}
    for idx, concept_data in analyze_photos_stream(paths, batch_size=batch_size):
        photo_info = downloaded[idx][0]
        concepts[photo_info["id"]] = json.dumps(concept_data)

    for photo_info, _ in downloaded:
        pid = photo_info["id"]
        if pid in scores:
            results.append({
                "photo_id": pid,
                "aesthetic_score": scores[pid],
                "aesthetic_concepts": concepts.get(pid),
            })
        else:
            # score_photos_stream skips an image it cannot open. That used to
            # drop the photo silently, so it was re-claimed every TTL forever;
            # report it so the attempts cap can retire it.
            results.append(_failure_row(pid, "quality scoring failed (image could not be loaded)"))
    return results


def _process_faces(downloaded: list[tuple[dict, str]]) -> list[dict]:
    """Run face detection on downloaded photos."""
    from .faces import detect_faces, check_available
    check_available()

    results = []
    for photo_info, path in downloaded:
        try:
            faces = detect_faces(path, use_cnn=False)
            face_data = []
            for face in faces:
                face_data.append({
                    "bbox": list(face["bbox"]),
                    "encoding": face["encoding"],
                    "det_score": face.get("det_score"),
                })
            results.append({
                "photo_id": photo_info["id"],
                "faces": face_data,
            })
        except Exception as e:
            # NEVER `faces: []` here: the server now treats an empty list as a
            # clean "nobody in frame" and retires the photo in one submit. An
            # error is a failure row — it spends ONE attempt, so a transient
            # detection failure still gets retried.
            print(f"    Face detection failed for {photo_info['filename']}: {e}")
            results.append(_failure_row(photo_info["id"], e))
    return results


def _process_describe(downloaded: list[tuple[dict, str]], model: str = "llama3.2-vision") -> list[dict]:
    """Generate scene descriptions via Ollama. Returns list of {photo_id, description}.

    Always includes every photo in results (description may be None) so the
    server can mark them as processed and avoid infinite reclaim loops.
    """
    from .describe import describe_photo, check_available
    check_available(model)

    results = []
    total = len(downloaded)
    for idx, (photo_info, path) in enumerate(downloaded, 1):
        fname = photo_info["filename"]
        print(f"    [{idx}/{total}] {fname} ...", end="", flush=True)
        t0 = time.time()
        try:
            desc = describe_photo(path, model=model)
            elapsed = time.time() - t0
            if desc:
                preview = desc[:80].replace("\n", " ")
                print(f" ({elapsed:.1f}s) {preview}...")
            else:
                print(f" ({elapsed:.1f}s) no description")
            results.append({"photo_id": photo_info["id"], "description": desc})
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({"photo_id": photo_info["id"], "description": None})
    return results


def _process_category_content(
    photos: list[dict],
    model: str = "llama3.2:3b",
) -> list[dict]:
    """Text-only pass: read description from photo dicts, extract in-vocab categories.

    photos is the raw list from claim_batch's response — each dict has 'id',
    'filename', 'filepath', and 'description'. No image download required.
    Returns one entry per SUCCESSFUL photo (categories may be [] = genuinely
    none). Photos whose extraction timed out / errored are OMITTED so the
    server does NOT mark them processed — they get re-claimed and retried later
    instead of being permanently recorded with empty categories on a stall.
    """
    from .describe import extract_categories_from_description, check_available
    check_available(model)
    results = []
    total = len(photos)
    for idx, photo in enumerate(photos, 1):
        fname = photo["filename"]
        print(f"    [{idx}/{total}] {fname} ...", end="", flush=True)
        t0 = time.time()
        try:
            cats = extract_categories_from_description(photo.get("description"), model=model)
        except Exception as e:
            cats, err = None, str(e)
        else:
            err = None
        elapsed = time.time() - t0
        if cats is None:
            # timeout/error → defer (omit from results so it isn't marked done)
            print(f" ({elapsed:.1f}s) deferring (timeout/error{': ' + err if err else ''})")
            continue
        print(f" ({elapsed:.1f}s) {', '.join(cats) if cats else 'no categories'}")
        results.append({"photo_id": photo["id"], "categories": cats})
    return results


def _process_keywords(
    photos: list[dict],
    model: str = "llama3.2:3b",
) -> list[dict]:
    """Text-only pass: read description from photo dicts, extract free-form keywords."""
    from .describe import extract_keywords_from_description, check_available
    check_available(model)
    results = []
    total = len(photos)
    for idx, photo in enumerate(photos, 1):
        fname = photo["filename"]
        print(f"    [{idx}/{total}] {fname} ...", end="", flush=True)
        t0 = time.time()
        try:
            kws = extract_keywords_from_description(photo.get("description"), model=model)
        except Exception as e:
            kws, err = None, str(e)
        else:
            err = None
        elapsed = time.time() - t0
        if kws is None:
            # timeout/error → defer (omit so the server doesn't mark it done)
            print(f" ({elapsed:.1f}s) deferring (timeout/error{': ' + err if err else ''})")
            continue
        print(f" ({elapsed:.1f}s) {', '.join(kws) if kws else 'no keywords'}")
        results.append({"photo_id": photo["id"], "keywords": kws})
    return results


def _process_category_visual(
    downloaded: list[tuple[dict, str]],
    model: str = "llava",
) -> list[dict]:
    """Vision pass: PERCEIVED visual tags (the capture facts are derived from
    EXIF server-side — see photosearch/visual_tags_derive.py).

    `visual_tags: []` and `visual_tags: None` mean different things and must
    not be conflated: `[]` is a real result ("the model looked and found
    nothing"), which the server persists as '[]' so the photo is done in one
    pass. `None` is "no usable answer" — the server leaves the column NULL so
    the photo stays claimable, but still marks it processed so a repeatable
    failure is bounded by MAX_PROCESS_ATTEMPTS. Same shape as the aesthetics
    pass's empty-scores row, and the reason an unparseable response no longer
    retires a photo from the queue forever.
    """
    from . import describe as _describe
    _describe.check_available(model)
    results = []
    total = len(downloaded)
    for idx, (photo, path) in enumerate(downloaded, 1):
        fname = photo["filename"]
        print(f"    [{idx}/{total}] {fname} ...", end="", flush=True)
        t0 = time.time()
        try:
            tags = _describe.tag_visual_photo(path, model=model)
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({"photo_id": photo["id"], "visual_tags": None})
            continue
        elapsed = time.time() - t0
        if tags:
            print(f" ({elapsed:.1f}s) {', '.join(tags)}")
        elif tags is None:
            print(f" ({elapsed:.1f}s) no usable answer (will retry)")
        else:
            print(f" ({elapsed:.1f}s) no visual tags")
        results.append({"photo_id": photo["id"], "visual_tags": tags})
    return results


def _process_aesthetics(
    downloaded: list[tuple[dict, str]],
    model: str = "qwen2.5-vl",
) -> list[dict]:
    """Vision pass: score photos across aesthetic dimensions + style critique.

    Returns one row per photo that scored successfully, each carrying the flat
    scalar columns for the DB plus the JSON style/critique fields. Photos whose
    VLM response couldn't be parsed are OMITTED (not returned with nulls) so the
    server doesn't mark them permanently done — the attempts cap in the claim
    predicate retires genuinely-unparseable photos after N tries.
    """
    from .aesthetics import score_photo_aesthetics, compute_overall
    from .describe import check_available
    check_available(model)
    results = []
    total = len(downloaded)
    for idx, (photo, path) in enumerate(downloaded, 1):
        fname = photo["filename"]
        print(f"    [{idx}/{total}] {fname} ...", end="", flush=True)
        t0 = time.time()
        try:
            aes = score_photo_aesthetics(path, model=model)
            elapsed = time.time() - t0
            if not aes:
                # Send an empty-scores row so the server marks the photo
                # processed (attempts++). The attempts cap then retires
                # genuinely-unparseable photos after N tries, while a still-NULL
                # aes_overall lets transient failures retry until then — avoids
                # the CLIP-style infinite re-claim.
                print(f" ({elapsed:.1f}s) no aesthetics (deferred)")
                results.append({"photo_id": photo["id"], "scores": {}})
                continue
            # Flatten sub-attribute + dimension + overall scores into aes_* cols
            # in a `scores` dict the server passes straight to update_photo.
            scores = {f"aes_{sub}": val for sub, val in aes["sub_scores"].items()}
            scores.update({f"aes_{dim}": val for dim, val in aes["dim_scores"].items()})
            scores["aes_overall"] = aes["overall"]
            row = {
                "photo_id": photo["id"],
                "scores": scores,
                "aes_style": json.dumps(
                    {"facets": aes["style"], "critiques": aes["critiques"]}),
                "aes_style_tags": json.dumps(aes["style_tags"]),
            }
            # Subject-aware quality: ground the primary subject and re-score its
            # crop so wow/impact judge the subject, not the background. Additive —
            # a grounding/crop failure just omits the subject fields (the
            # full-frame score above still lands). See photosearch/subjects.py.
            subj_note = ""
            try:
                from .subjects import score_photo_subject
                subj = score_photo_subject(path, model=model)
                if subj["subject_boxes"] is not None:
                    row["subject_boxes"] = json.dumps(subj["subject_boxes"])
                    sa = subj["subject_aes"]
                    if sa:
                        row["aes_subject_overall"] = sa["overall"]
                        row["aes_subject"] = json.dumps({
                            "dim_scores": sa["dim_scores"],
                            "sub_scores": sa["sub_scores"],
                            "style": sa["style"], "style_tags": sa["style_tags"],
                            "critiques": sa["critiques"]})
                        subj_note = f" subj={sa['overall']}"
                    else:
                        subj_note = f" subj=none({len(subj['subject_boxes'])} box)"
            except Exception as e:
                subj_note = f" subj_err={e}"
            print(f" ({elapsed:.1f}s) overall={aes['overall']}{subj_note}")
            results.append(row)
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({"photo_id": photo["id"], "scores": {}})
    return results


def _process_verify(
    downloaded: list[tuple[dict, str]],
    client: "WorkerClient",
    verify_model: str = "llava",
    regen_model: str = "llama3.2-vision",
) -> list[dict]:
    """Run hallucination verification on downloaded photos.

    This is more complex than other passes because it needs:
    1. The photo's existing CLIP embedding (from NAS DB, via API)
    2. The photo's existing description and tags (from NAS DB, via API)
    3. Local Ollama for LLM verification
    4. Local CLIP for cross-checking

    Returns list of {photo_id, status, description, tags, verified_at, ...}
    """
    from .describe import check_available as desc_check
    desc_check(verify_model)

    results = []
    total = len(downloaded)
    for idx, (photo_info, path) in enumerate(downloaded, 1):
        fname = photo_info["filename"]
        pid = photo_info["id"]
        print(f"    [{idx}/{total}] {fname} ...", end="", flush=True)
        t0 = time.time()

        try:
            # Fetch photo detail from server (includes description, tags, CLIP embedding)
            detail = client.get_photo_detail(pid)
            if not detail:
                print(f" could not fetch detail")
                continue

            description = detail.get("description") or ""
            tags_raw = detail.get("tags")
            tags = json.loads(tags_raw) if tags_raw and isinstance(tags_raw, str) else (tags_raw or [])
            clip_embedding = detail.get("clip_embedding")

            if not description and not tags:
                print(f" no description/tags to verify")
                results.append({
                    "photo_id": pid,
                    "status": "pass",
                    "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "hallucination_flags": None,
                })
                continue

            # Passes 1-3 (CLIP gate, LLM, CLIP override) live in
            # verify.check_description so the model eval measures this exact
            # pipeline (evals/verify_eval.py --mode pipeline).
            from .verify import check_description
            chk = check_description(path, description, tags, clip_embedding,
                                    verify_model=verify_model)
            clip_flags = chk["clip_flags"]
            if chk["stage"] != "confirmed":
                elapsed = time.time() - t0
                label = {"clip_clean": "CLIP clean", "llm_cleared": "LLM cleared",
                         "clip_override": "CLIP override"}[chk["stage"]]
                print(f" ({elapsed:.1f}s) pass ({label})")
                results.append({
                    "photo_id": pid,
                    "status": "pass",
                    "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "hallucination_flags": json.dumps(clip_flags) if clip_flags else None,
                })
                continue
            verified_confirmed = chk["confirmed"]

            # Hallucinations confirmed — regenerate
            confirmed_nouns = {c["noun"] for c in verified_confirmed}
            elapsed = time.time() - t0

            from .describe import describe_photo as _describe, tag_visual_photo as _tag, DESCRIBE_PROMPT
            strict_prompt = DESCRIBE_PROMPT + (
                "\n\nIMPORTANT: A previous description was found to contain "
                "hallucinated objects. Be extra careful to ONLY describe what you "
                "can clearly see. Do NOT mention: "
                + ", ".join(sorted(confirmed_nouns)) + "."
            )
            new_desc = _describe(path, model=regen_model, prompt=strict_prompt)
            new_tags = _tag(path, model=regen_model) if new_desc else None

            # Mirror verify.py: 'regenerated' only if we actually produced a new
            # description. Otherwise the photo's description in the DB is still
            # the one containing hallucinations — mark it 'fail' so it surfaces.
            if new_desc:
                status = "regenerated"
                print(f" ({elapsed:.1f}s) REGENERATED — {', '.join(confirmed_nouns)}")
            else:
                status = "fail"
                print(f" ({elapsed:.1f}s) FAIL — {', '.join(confirmed_nouns)}")

            result = {
                "photo_id": pid,
                "status": status,
                "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "hallucination_flags": json.dumps(
                    [{"noun": n, "llm_says": "NO"} for n in confirmed_nouns]
                ),
            }
            if new_desc:
                result["description"] = new_desc
            if new_tags:
                result["tags"] = new_tags
            results.append(result)

        except _TRANSIENT as e:
            # Network / LLM-backend timeout — the worker's problem, not the
            # photo's. Omit it so the photo comes back without spending an
            # attempt.
            print(f" deferring ({e.__class__.__name__}: {e})")
        except Exception as e:
            print(f" ERROR: {e}")
            results.append(_failure_row(pid, e))
    return results


def run_worker(
    server: str,
    passes: list[str],
    collection_id: Optional[int] = None,
    directory: Optional[str] = None,
    filters: Optional[dict] = None,
    batch_size: int = 16,
    model_batch_size: int = 8,
    ttl_minutes: int = 30,
    one_shot: bool = False,
    stay_alive: bool = False,
    sequential: bool = True,
    force: bool = False,
    describe_model: str = "llama3.2-vision",
    tags_model: str = "llava",
    verify_model: str = "llava",
    category_content_model: str = "llama3.2:3b",
    category_visual_model: str = "llava",
    keywords_model: str = "llama3.2:3b",
    aesthetics_model: str = "qwen2.5-vl",
):
    """Main worker loop. Connects to server and processes photos until queue is empty.

    Args:
        server: NAS server URL (e.g. http://nas.local:8000)
        passes: List of pass types to process (clip, faces, quality, describe, verify, ...)
        collection_id: Optional collection to scope work to
        directory: Optional directory path on the NAS to scope work to
        filters: Optional structured filter set (date_from/date_to, people,
            location, min_quality, min_aesthetic, camera, category, visual_tag,
            keyword, style_tag) resolved server-side to a photo-id scope
        batch_size: Number of photos to claim per batch
        model_batch_size: Batch size for model inference
        ttl_minutes: Claim TTL in minutes
        one_shot: If True, process one batch per pass and exit
        stay_alive: If True, keep polling forever even when every queue is
            empty (the old behaviour). By default a pass is RETIRED the first
            time its queue comes back empty, and the worker exits once every
            pass has been retired — so a fleet started for a specific backlog
            shuts itself down instead of idling. Idle polling is not free: each
            claim opens BEGIN IMMEDIATE on the NAS's SQLite file and takes the
            single write lock, which has previously starved face assignments,
            collection writes and an entire overnight ingest.
        sequential: If True (the default), drain one pass completely before
            starting the next, in the order given. False is round-robin, one
            batch per pass per cycle. Sequential avoids thrashing model weights in and
            out of memory between passes (-p clip,quality otherwise alternates
            ViT-B/16 and ViT-L/14 every batch).
        force: If True, clear existing data and re-process from scratch
        describe_model: Ollama model for descriptions (default: llama3.2-vision)
        tags_model: Ollama model for the (removed) tags pass
        verify_model: Ollama model for verification (default: llava)
        category_content_model: Ollama model for category-content text pass (default: llama3.2:3b)
        category_visual_model: Ollama model for category-visual vision pass (default: llava)
        keywords_model: Ollama model for keywords text pass (default: llama3.2:3b)
        aesthetics_model: Vision model for the aesthetics scoring pass (default: qwen2.5-vl)
    """
    print(f"Connecting to {server}...")
    client = WorkerClient(server)
    print(f"Connected as {client.worker_id}")

    scope_label = (f"collection {collection_id}" if collection_id
                   else f"directory {directory}" if directory
                   else f"filters {filters}" if filters
                   else "all photos")

    # Force mode: clear existing data for the requested passes
    if force:
        if collection_id is None and directory is None and not filters:
            print("Error: --force requires --collection, --directory, or a filter (safety measure).",
                  file=sys.stderr)
            return
        for pass_type in passes:
            print(f"Clearing {pass_type} data for {scope_label}...")
            resp = client.clear_pass(pass_type, collection_id=collection_id,
                                     directory=directory, filters=filters)
            print(f"  Cleared {resp['cleared']} entries across {resp['photo_count']} photos.")

    # Show initial status — informational only, non-fatal. Under concurrent
    # worker startup the status endpoint can be slow (counts scan the library),
    # so we don't want a timeout here to prevent the worker from proceeding.
    try:
        status = client.get_status(
            collection_id=collection_id, directory=directory, passes=passes,
            filters=filters,
        )
        print(f"\nQueue depth:")
        for pass_type, count in status["queue_depth"].items():
            if pass_type in passes:
                marker = " <--" if count > 0 else ""
                print(f"  {pass_type}: {count} photos{marker}")

        if status["active_claims"]:
            print(f"\nActive claims: {len(status['active_claims'])}")
            for claim in status["active_claims"]:
                print(f"  {claim['worker_id']}: {claim['pass_type']} ({claim['photo_count']} photos)")
    except Exception as e:
        print(f"\n(Could not fetch initial queue status: {e} — proceeding to claim loop)")

    temp_base = tempfile.mkdtemp(prefix="photosearch-worker-")
    print(f"\nTemp directory: {temp_base}")

    total_processed = 0
    # Track which pass's torch model is currently resident so we can drop it
    # before loading a different pass's model (prevents both ViT-B/16 and
    # ViT-L/14 being held simultaneously when running -p clip,quality).
    loaded_pass: Optional[str] = None
    # Passes still worth claiming. A pass is removed from here once its queue
    # reports empty (unless --stay-alive); when the list empties, we're done.
    active: list[str] = list(passes)
    seq_idx = 0
    mode = ("sequential" if sequential else "round-robin") + \
           (", stay-alive" if stay_alive else ", exit when drained")
    print(f"\nPass order: {' -> '.join(active)}  ({mode})")

    try:
        while True:
            if not active:
                print(f"\nEvery pass drained. Processed {total_processed} photos total.")
                break

            any_work = False
            drained: list[str] = []

            if sequential:
                seq_idx %= len(active)
                cycle = [active[seq_idx]]
            else:
                cycle = list(active)

            for pass_type in cycle:
                if loaded_pass is not None and loaded_pass != pass_type:
                    _unload_pass_models(loaded_pass)
                    loaded_pass = None
                # Claim a batch (with retry for sleep/wake)
                print(f"\n{'='*60}")
                print(f"Claiming {pass_type} batch (limit={batch_size})...")
                try:
                    batch = _retry(
                        lambda pt=pass_type: client.claim_batch(
                            pass_type=pt, limit=batch_size,
                            collection_id=collection_id, directory=directory,
                            filters=filters, ttl_minutes=ttl_minutes,
                        ),
                        label=f"claim {pass_type}",
                    )
                except _TRANSIENT as e:
                    print(f"  ✗ Cannot reach server after retries: {e}")
                    print(f"  Skipping {pass_type}, will retry next loop.")
                    continue

                if not batch.get("batch_id") or not batch.get("photos"):
                    if batch.get("contended"):
                        # The server found work but lost the claim race to
                        # another worker every retry. Emphatically NOT an empty
                        # queue — retiring here would shrink a busy fleet
                        # precisely when it is busiest.
                        print(f"  {pass_type}: lost the claim race, retrying.")
                        continue
                    # A genuinely EMPTY QUEUE — and only this — retires a pass.
                    # Transport failures take the _TRANSIENT path above and
                    # `continue` without touching `drained`, so a lock, a
                    # timeout or a 503 mid-deploy can never silently kill the
                    # fleet.
                    print(f"  No unprocessed {pass_type} photos in queue.")
                    drained.append(pass_type)
                    continue

                any_work = True
                photos = batch["photos"]
                batch_id = batch["batch_id"]
                remaining = batch.get("remaining", "?")
                print(f"  Claimed {len(photos)} photos (batch {batch_id[:8]}...), {remaining} remaining")

                # Start heartbeat to keep claim alive during long processing
                heartbeat = _ClaimHeartbeat(client, batch_id, ttl_minutes)
                heartbeat.start()

                # Download (with retry per photo — handled inside _download_batch)
                # Text-only passes work off `description` from the claim response
                # and skip the download entirely.
                batch_temp = os.path.join(temp_base, batch_id[:8])
                os.makedirs(batch_temp, exist_ok=True)

                TEXT_ONLY_PASSES = {"category-content", "keywords"}
                needs_images = pass_type not in TEXT_ONLY_PASSES

                if needs_images:
                    print(f"\n  Downloading {len(photos)} photos...")
                    t0 = time.time()
                    downloaded = _download_batch(client, photos, batch_temp)
                    dl_elapsed = time.time() - t0
                    print(f"  Downloaded {len(downloaded)}/{len(photos)} in {dl_elapsed:.1f}s")
                    if not downloaded:
                        print(f"  No photos downloaded, skipping batch.")
                        heartbeat.stop()
                        continue
                else:
                    downloaded = None  # not used by text-only passes

                # Process (local — no network needed except verify)
                print(f"\n  Processing {pass_type}...")
                t0 = time.time()

                if pass_type == "clip":
                    results = _process_clip(downloaded, batch_size=model_batch_size)
                    kwargs = _submit_kwargs("clip_results", results)
                elif pass_type == "quality":
                    results = _process_quality(downloaded, batch_size=model_batch_size)
                    kwargs = _submit_kwargs("quality_results", results)
                elif pass_type == "faces":
                    results = _process_faces(downloaded)
                    kwargs = _submit_kwargs("face_results", results)
                elif pass_type == "describe":
                    results = _process_describe(downloaded, model=describe_model)
                    kwargs = {"describe_results": results,
                              **_provenance_kwargs(pass_type, describe_model, results)}
                elif pass_type == "verify":
                    results = _process_verify(
                        downloaded, client=client,
                        verify_model=verify_model, regen_model=describe_model,
                    )
                    # regen_model == describe_model produces any regenerated
                    # text, so the artifact's provenance is the DESCRIBE role.
                    kwargs = {**_submit_kwargs("verify_results", results),
                              **_provenance_kwargs(pass_type, describe_model,
                                                   results, role="describe")}
                elif pass_type == "category-content":
                    results = _process_category_content(photos, model=category_content_model)
                    _provenance_kwargs(pass_type, category_content_model, results)
                    kwargs = {"category_content_results": results}
                elif pass_type == "category-visual":
                    results = _process_category_visual(downloaded, model=category_visual_model)
                    _provenance_kwargs(pass_type, category_visual_model, results)
                    kwargs = {"category_visual_results": results}
                elif pass_type == "keywords":
                    results = _process_keywords(photos, model=keywords_model)
                    _provenance_kwargs(pass_type, keywords_model, results)
                    kwargs = {"keywords_results": results}
                elif pass_type == "aesthetics":
                    results = _process_aesthetics(downloaded, model=aesthetics_model)
                    _provenance_kwargs(pass_type, aesthetics_model, results)
                    kwargs = {"aesthetics_results": results}
                else:
                    print(f"  Pass type '{pass_type}' not yet implemented in worker.")
                    heartbeat.stop()
                    continue

                loaded_pass = pass_type

                proc_elapsed = time.time() - t0
                print(f"  Processed {len(results)} results in {proc_elapsed:.1f}s")

                # Stop heartbeat before submit (no longer needed)
                heartbeat.stop()

                # Submit (with retry — this is where sleep/wake crashes hit)
                print(f"  Submitting results...")
                try:
                    resp = _retry(
                        lambda: client.submit_results(batch_id, pass_type, **kwargs),
                        label="submit results",
                    )
                except _TRANSIENT as e:
                    print(f"  ✗ Failed to submit after retries: {e}")
                    print(f"  Results lost for this batch — photos will be reclaimed after TTL.")
                    shutil.rmtree(batch_temp, ignore_errors=True)
                    continue

                n_written = resp.get("written", 0)
                n_processed = resp.get("processed", n_written)
                if pass_type == "faces":
                    print(f"  Server processed {n_processed} photos ({n_written} faces found).")
                elif pass_type in ("describe", "category-content", "category-visual", "keywords", "aesthetics"):
                    print(f"  Server processed {n_processed} photos ({n_written} with {pass_type}).")
                else:
                    print(f"  Server wrote {n_written} results.")
                # The server defers a result blocked by a transient DB LOCK (a
                # sweep holding the write lock) instead of spending one of the
                # photo's MAX_PROCESS_ATTEMPTS on its own contention. Any other
                # failure still counts the attempt, so the cap bounds it.
                n_deferred = resp.get("deferred", 0)
                if n_deferred:
                    ids = resp.get("deferred_photo_ids") or []
                    shown = ", ".join(str(i) for i in ids[:8])
                    more = f", +{len(ids) - 8} more" if len(ids) > 8 else ""
                    # Careful with the wording: on a failed batch commit only
                    # the FINAL flush is known not to have landed — earlier
                    # rows may be on disk, in which case the claim predicate
                    # simply won't offer them again.
                    print(f"  ⚠ Server deferred {n_deferred} result(s) it could not "
                          f"commit ({shown}{more}) — no attempt was spent; any "
                          f"that did not land will be reclaimed.")
                total_processed += n_processed

                # Cleanup temp files
                shutil.rmtree(batch_temp, ignore_errors=True)

                # Drop allocator caches (MPS especially) and force a GC so
                # intermediate tensors/PIL buffers don't drift upward.
                del downloaded, results
                _flush_caches()

            # Retire drained passes, or (stay-alive + sequential) move along so
            # the head pass isn't polled forever while later ones have work.
            for pt in drained:
                if not stay_alive:
                    if pt in active:
                        active.remove(pt)
                        left = ", ".join(active) if active else "none"
                        print(f"  → retiring '{pt}' (queue empty). Remaining: {left}")
                elif sequential:
                    seq_idx += 1
            if not active:
                continue    # top of the loop reports completion and breaks

            if not any_work:
                if one_shot:
                    print(f"\nAll queues empty. Processed {total_processed} photos total.")
                    break
                # Queues are empty — drop any resident model so we're not
                # sitting on ~1 GB of weights while idle. Reload cost
                # (~2–5s) is trivial compared to the idle duration.
                if loaded_pass is not None:
                    _unload_pass_models(loaded_pass)
                    loaded_pass = None
                    _flush_caches()
                print(f"\nAll queues empty. Waiting 10s before retrying...")
                time.sleep(10)
            elif one_shot:
                print(f"\nOne-shot mode: processed {total_processed} photos total.")
                break

    except KeyboardInterrupt:
        print(f"\n\nInterrupted. Processed {total_processed} photos total.")
    finally:
        if loaded_pass is not None:
            _unload_pass_models(loaded_pass)
        _flush_caches()
        shutil.rmtree(temp_base, ignore_errors=True)
        print(f"Cleaned up temp directory.")
