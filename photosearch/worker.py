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

import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import requests


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
        r = self.session.post(f"{self.server_url}/api/worker/claim-batch", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def download_photo(self, photo_id: int, dest_path: str) -> bool:
        """Download a photo's full-resolution bytes. Returns True on success."""
        r = self.session.get(
            f"{self.server_url}/api/photos/{photo_id}/full",
            timeout=120,
            stream=True,
        )
        if r.status_code != 200:
            return False
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        return True

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
        r.raise_for_status()
        return r.json()

    def get_status(self, collection_id: Optional[int] = None) -> dict:
        """Get worker queue status."""
        params = {}
        if collection_id is not None:
            params["collection_id"] = collection_id
        r = self.session.get(f"{self.server_url}/api/worker/status", params=params, timeout=10)
        r.raise_for_status()
        return r.json()


def _download_batch(client: WorkerClient, photos: list[dict], temp_dir: str) -> list[tuple[dict, str]]:
    """Download photos to temp_dir. Returns [(photo_info, local_path)] for successful downloads."""
    downloaded = []
    for photo in photos:
        local_path = os.path.join(temp_dir, f"{photo['id']}_{photo['filename']}")
        print(f"    Downloading {photo['filename']}...", end="", flush=True)
        t0 = time.time()
        if client.download_photo(photo["id"], local_path):
            elapsed = time.time() - t0
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f" {size_mb:.1f}MB ({elapsed:.1f}s)")
            downloaded.append((photo, local_path))
        else:
            print(" FAILED")
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


def run_worker(
    server: str,
    passes: list[str],
    collection_id: Optional[int] = None,
    batch_size: int = 16,
    model_batch_size: int = 8,
    ttl_minutes: int = 30,
    one_shot: bool = False,
):
    """Main worker loop. Connects to server and processes photos until queue is empty.

    Args:
        server: NAS server URL (e.g. http://nas.local:8000)
        passes: List of pass types to process (clip, faces, quality)
        collection_id: Optional collection to scope work to
        batch_size: Number of photos to claim per batch
        model_batch_size: Batch size for model inference
        ttl_minutes: Claim TTL in minutes
        one_shot: If True, process one batch per pass and exit
    """
    print(f"Connecting to {server}...")
    client = WorkerClient(server)
    print(f"Connected as {client.worker_id}")

    # Show initial status
    status = client.get_status(collection_id)
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
    try:
        while True:
            any_work = False

            for pass_type in passes:
                # Claim a batch
                print(f"\n{'='*60}")
                print(f"Claiming {pass_type} batch (limit={batch_size})...")
                batch = client.claim_batch(
                    pass_type=pass_type,
                    limit=batch_size,
                    collection_id=collection_id,
                    ttl_minutes=ttl_minutes,
                )

                if not batch.get("batch_id") or not batch.get("photos"):
                    print(f"  No unprocessed {pass_type} photos in queue.")
                    continue

                any_work = True
                photos = batch["photos"]
                batch_id = batch["batch_id"]
                print(f"  Claimed {len(photos)} photos (batch {batch_id[:8]}...)")

                # Download
                batch_temp = os.path.join(temp_base, batch_id[:8])
                os.makedirs(batch_temp, exist_ok=True)

                print(f"\n  Downloading {len(photos)} photos...")
                t0 = time.time()
                downloaded = _download_batch(client, photos, batch_temp)
                dl_elapsed = time.time() - t0
                print(f"  Downloaded {len(downloaded)}/{len(photos)} in {dl_elapsed:.1f}s")

                if not downloaded:
                    print(f"  No photos downloaded, skipping batch.")
                    continue

                # Process
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
                else:
                    print(f"  Pass type '{pass_type}' not yet implemented in worker.")
                    continue

                proc_elapsed = time.time() - t0
                print(f"  Processed {len(results)} results in {proc_elapsed:.1f}s")

                # Submit
                print(f"  Submitting results...")
                resp = client.submit_results(batch_id, pass_type, **kwargs)
                print(f"  Server wrote {resp['written']} results.")
                total_processed += resp["written"]

                # Cleanup temp files
                shutil.rmtree(batch_temp, ignore_errors=True)

            if not any_work:
                if one_shot:
                    print(f"\nAll queues empty. Processed {total_processed} photos total.")
                    break
                print(f"\nAll queues empty. Waiting 10s before retrying...")
                time.sleep(10)
            elif one_shot:
                print(f"\nOne-shot mode: processed {total_processed} photos total.")
                break

    except KeyboardInterrupt:
        print(f"\n\nInterrupted. Processed {total_processed} photos total.")
    finally:
        # Cleanup
        shutil.rmtree(temp_base, ignore_errors=True)
        print(f"Cleaned up temp directory.")
