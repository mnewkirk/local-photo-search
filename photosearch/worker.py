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

    def __init__(self, server_url: str, worker_id: str = None):
        self.server_url = server_url.rstrip("/")
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.session = requests.Session()
        # Quick connectivity test
        try:
            r = self.session.get(f"{self.server_url}/api/stats", timeout=10)
            r.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Cannot reach server at {self.server_url}: {e}")

    def claim_batch(self, pass_type: str, limit: int = 16,
                    collection_id: Optional[int] = None,
                    directory: Optional[str] = None,
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
        r = self.session.post(f"{self.server_url}/api/worker/claim-batch", json=payload, timeout=30)
        if not r.ok:
            detail = r.text[:200] if r.text else r.reason
            raise RuntimeError(f"claim-batch failed ({r.status_code}): {detail}")
        return r.json()

    def download_photo(self, photo_id: int, dest_path: str) -> int:
        """Download a photo's full-resolution bytes. Returns HTTP status code
        (200 on success, 404 if the NAS can't find the file, etc.)."""
        r = self.session.get(
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
        r = self.session.post(
            f"{self.server_url}/api/worker/submit-results",
            json=payload,
            timeout=120,
        )
        if not r.ok:
            detail = r.text[:200] if r.text else r.reason
            raise RuntimeError(f"submit-results failed ({r.status_code}): {detail}")
        return r.json()

    def get_status(self, collection_id: Optional[int] = None,
                   directory: Optional[str] = None) -> dict:
        """Get worker queue status."""
        params = {}
        if collection_id is not None:
            params["collection_id"] = collection_id
        if directory is not None:
            params["directory"] = directory
        r = self.session.get(f"{self.server_url}/api/worker/status", params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    def clear_pass(self, pass_type: str, collection_id: Optional[int] = None,
                   directory: Optional[str] = None) -> dict:
        """Clear processing state for a pass type on a collection or directory."""
        payload = {"pass_type": pass_type}
        if collection_id is not None:
            payload["collection_id"] = collection_id
        if directory is not None:
            payload["directory"] = directory
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
            r = self.session.post(
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


def _process_clip(downloaded: list[tuple[dict, str]], batch_size: int = 8) -> list[dict]:
    """Run CLIP embedding on downloaded photos. Returns list of {photo_id, embedding}."""
    from .clip_embed import embed_images_stream

    paths = [path for _, path in downloaded]
    results = []
    for idx, emb in embed_images_stream(paths, batch_size=batch_size):
        photo_info = downloaded[idx][0]
        results.append({"photo_id": photo_info["id"], "embedding": emb})
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
                })
            results.append({
                "photo_id": photo_info["id"],
                "faces": face_data,
            })
        except Exception as e:
            print(f"    Face detection failed for {photo_info['filename']}: {e}")
            results.append({"photo_id": photo_info["id"], "faces": []})
    return results


def _process_describe(downloaded: list[tuple[dict, str]], model: str = "llava") -> list[dict]:
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


def _process_tags(downloaded: list[tuple[dict, str]], model: str = "llava") -> list[dict]:
    """Generate semantic tags via Ollama. Returns list of {photo_id, tags: [...]}.

    Always includes every photo in results (tags may be empty list) so the
    server can mark them as processed and avoid infinite reclaim loops.
    """
    from .describe import tag_photo, check_available
    check_available(model)

    results = []
    total = len(downloaded)
    for idx, (photo_info, path) in enumerate(downloaded, 1):
        fname = photo_info["filename"]
        print(f"    [{idx}/{total}] {fname} ...", end="", flush=True)
        t0 = time.time()
        try:
            tags = tag_photo(path, model=model)
            elapsed = time.time() - t0
            if tags:
                print(f" ({elapsed:.1f}s) {', '.join(tags)}")
            else:
                print(f" ({elapsed:.1f}s) no tags")
            results.append({"photo_id": photo_info["id"], "tags": tags or []})
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({"photo_id": photo_info["id"], "tags": []})
    return results


def _process_verify(
    downloaded: list[tuple[dict, str]],
    client: "WorkerClient",
    verify_model: str = "minicpm-v",
    regen_model: str = "llava",
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

            # Pass 1: CLIP scoring
            clip_flags = []
            if clip_embedding:
                from .verify import clip_score_description, clip_score_tags, _flag_by_clip
                desc_scores = clip_score_description(clip_embedding, description) if description else []
                tag_scores = clip_score_tags(clip_embedding, tags) if tags else []
                desc_flagged, tag_flagged, all_clip_items = _flag_by_clip(
                    desc_scores, tag_scores, clip_threshold=0.18
                )
                clip_flags = [item for item in all_clip_items
                              if any(f.get("noun") == item.get("noun") for f in desc_flagged)
                              or any(f.get("tag") == item.get("tag") for f in tag_flagged)]

                if not desc_flagged and not tag_flagged:
                    elapsed = time.time() - t0
                    print(f" ({elapsed:.1f}s) pass (CLIP clean)")
                    results.append({
                        "photo_id": pid,
                        "status": "pass",
                        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "hallucination_flags": json.dumps(clip_flags) if clip_flags else None,
                    })
                    continue

            # Pass 2: LLM verification
            from .verify import llm_verify_description
            confirmed = llm_verify_description(path, description, tags, model=verify_model)

            if not confirmed:
                elapsed = time.time() - t0
                print(f" ({elapsed:.1f}s) pass (LLM cleared)")
                results.append({
                    "photo_id": pid,
                    "status": "pass",
                    "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "hallucination_flags": json.dumps(clip_flags) if clip_flags else None,
                })
                continue

            # Pass 3: CLIP cross-check on LLM findings
            import numpy as np
            verified_confirmed = confirmed
            if clip_embedding:
                from .clip_embed import embed_text
                photo_vec = np.array(clip_embedding, dtype=np.float32)
                desc_scores_sims = [s["similarity"] for s in desc_scores] + [s["similarity"] for s in tag_scores]
                median_sim = float(np.median(desc_scores_sims)) if desc_scores_sims else 0.0

                verified_confirmed = []
                for item in confirmed:
                    text_emb = embed_text(f"a photo of {item['noun']}")
                    if text_emb is not None:
                        text_vec = np.array(text_emb, dtype=np.float32)
                        sim = float(np.dot(photo_vec, text_vec))
                        if sim >= median_sim:
                            continue  # CLIP overrides LLM
                    verified_confirmed.append(item)

            if not verified_confirmed:
                elapsed = time.time() - t0
                print(f" ({elapsed:.1f}s) pass (CLIP override)")
                results.append({
                    "photo_id": pid,
                    "status": "pass",
                    "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "hallucination_flags": json.dumps(clip_flags) if clip_flags else None,
                })
                continue

            # Hallucinations confirmed — regenerate
            confirmed_nouns = {c["noun"] for c in verified_confirmed}
            elapsed = time.time() - t0

            from .describe import describe_photo as _describe, tag_photo as _tag, DESCRIBE_PROMPT
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

        except Exception as e:
            print(f" ERROR: {e}")
    return results


def run_worker(
    server: str,
    passes: list[str],
    collection_id: Optional[int] = None,
    directory: Optional[str] = None,
    batch_size: int = 16,
    model_batch_size: int = 8,
    ttl_minutes: int = 30,
    one_shot: bool = False,
    force: bool = False,
    describe_model: str = "llava",
    verify_model: str = "minicpm-v",
):
    """Main worker loop. Connects to server and processes photos until queue is empty.

    Args:
        server: NAS server URL (e.g. http://nas.local:8000)
        passes: List of pass types to process (clip, faces, quality, describe, tags, verify)
        collection_id: Optional collection to scope work to
        directory: Optional directory path on the NAS to scope work to
        batch_size: Number of photos to claim per batch
        model_batch_size: Batch size for model inference
        ttl_minutes: Claim TTL in minutes
        one_shot: If True, process one batch per pass and exit
        force: If True, clear existing data and re-process from scratch
        describe_model: Ollama model for descriptions/tags (default: llava)
        verify_model: Ollama model for verification (default: minicpm-v)
    """
    print(f"Connecting to {server}...")
    client = WorkerClient(server)
    print(f"Connected as {client.worker_id}")

    scope_label = (f"collection {collection_id}" if collection_id
                   else f"directory {directory}" if directory
                   else "all photos")

    # Force mode: clear existing data for the requested passes
    if force:
        if collection_id is None and directory is None:
            print("Error: --force requires --collection or --directory (safety measure).",
                  file=sys.stderr)
            return
        for pass_type in passes:
            print(f"Clearing {pass_type} data for {scope_label}...")
            resp = client.clear_pass(pass_type, collection_id=collection_id,
                                     directory=directory)
            print(f"  Cleared {resp['cleared']} entries across {resp['photo_count']} photos.")

    # Show initial status
    status = client.get_status(collection_id=collection_id, directory=directory)
    print(f"\nQueue depth:")
    for pass_type, count in status["queue_depth"].items():
        if pass_type in passes:
            marker = " <--" if count > 0 else ""
            print(f"  {pass_type}: {count} photos{marker}")

    if status["active_claims"]:
        print(f"\nActive claims: {len(status['active_claims'])}")
        for claim in status["active_claims"]:
            print(f"  {claim['worker_id']}: {claim['pass_type']} ({claim['photo_count']} photos)")

    temp_base = tempfile.mkdtemp(prefix="photosearch-worker-")
    print(f"\nTemp directory: {temp_base}")

    total_processed = 0
    # Track which pass's torch model is currently resident so we can drop it
    # before loading a different pass's model (prevents both ViT-B/16 and
    # ViT-L/14 being held simultaneously when running -p clip,quality).
    loaded_pass: Optional[str] = None
    try:
        while True:
            any_work = False

            for pass_type in passes:
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
                            ttl_minutes=ttl_minutes,
                        ),
                        label=f"claim {pass_type}",
                    )
                except _TRANSIENT as e:
                    print(f"  ✗ Cannot reach server after retries: {e}")
                    print(f"  Skipping {pass_type}, will retry next loop.")
                    continue

                if not batch.get("batch_id") or not batch.get("photos"):
                    print(f"  No unprocessed {pass_type} photos in queue.")
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
                batch_temp = os.path.join(temp_base, batch_id[:8])
                os.makedirs(batch_temp, exist_ok=True)

                print(f"\n  Downloading {len(photos)} photos...")
                t0 = time.time()
                downloaded = _download_batch(client, photos, batch_temp)
                dl_elapsed = time.time() - t0
                print(f"  Downloaded {len(downloaded)}/{len(photos)} in {dl_elapsed:.1f}s")

                if not downloaded:
                    print(f"  No photos downloaded, skipping batch.")
                    heartbeat.stop()
                    continue

                # Process (local — no network needed except verify)
                print(f"\n  Processing {pass_type}...")
                t0 = time.time()

                if pass_type == "clip":
                    results = _process_clip(downloaded, batch_size=model_batch_size)
                    kwargs = {"clip_results": results}
                elif pass_type == "quality":
                    results = _process_quality(downloaded, batch_size=model_batch_size)
                    kwargs = {"quality_results": results}
                elif pass_type == "faces":
                    results = _process_faces(downloaded)
                    kwargs = {"face_results": results}
                elif pass_type == "describe":
                    results = _process_describe(downloaded, model=describe_model)
                    kwargs = {"describe_results": results}
                elif pass_type == "tags":
                    results = _process_tags(downloaded, model=describe_model)
                    kwargs = {"tags_results": results}
                elif pass_type == "verify":
                    results = _process_verify(
                        downloaded, client=client,
                        verify_model=verify_model, regen_model=describe_model,
                    )
                    kwargs = {"verify_results": results}
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
                elif pass_type in ("describe", "tags"):
                    print(f"  Server processed {n_processed} photos ({n_written} with {pass_type}).")
                else:
                    print(f"  Server wrote {n_written} results.")
                total_processed += n_processed

                # Cleanup temp files
                shutil.rmtree(batch_temp, ignore_errors=True)

                # Drop allocator caches (MPS especially) and force a GC so
                # intermediate tensors/PIL buffers don't drift upward.
                del downloaded, results
                _flush_caches()

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
