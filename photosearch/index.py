"""Indexing pipeline — processes a folder of photos and populates the database.

M1: EXIF extraction, CLIP embeddings, dominant colors.
M2: Face detection and encoding.
M3: LLaVA scene descriptions via Ollama.
M8: Aesthetic quality scoring.
"""

import hashlib
import os
import time
from pathlib import Path

from .colors import colors_to_json, extract_dominant_colors
from .clip_embed import embed_image, embed_images_batch, embed_images_stream, unload_model as unload_clip
from .db import PhotoDB
from .exif import extract_exif, find_raw_pair

# File extensions we index
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
RAW_EXTENSIONS = {".arw"}
SUPPORTED_EXTENSIONS = JPEG_EXTENSIONS | RAW_EXTENSIONS


def file_hash(filepath: str, chunk_size: int = 8192) -> str:
    """Compute a fast SHA-256 hash of a file for deduplication."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# Subfolder names that are excluded from indexing
EXCLUDED_DIRS = {"results", "references", ".references", "thumbnails"}


def find_photos(directory: str) -> list[str]:
    """Recursively find all supported image files in a directory.

    Returns JPEG files. ARW files are associated as raw pairs, not indexed separately.
    Skips folders named 'results', 'references', or 'thumbnails'.
    """
    photos = []
    for root, dirs, files in os.walk(directory):
        # Skip excluded subdirectories in-place so os.walk doesn't descend into them
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for fname in sorted(files):
            ext = Path(fname).suffix.lower()
            if ext in JPEG_EXTENSIONS:
                photos.append(os.path.join(root, fname))
    return photos


def index_directory(
    photo_dir: str,
    db_path: str = "photo_index.db",
    batch_size: int = 8,
    skip_existing: bool = True,
    enable_clip: bool = True,
    enable_colors: bool = True,
    enable_faces: bool = False,
    force_faces: bool = False,
    force_clip: bool = False,
    enable_describe: bool = False,
    force_describe: bool = False,
    describe_model: str = "llava",
    enable_quality: bool = False,
    force_quality: bool = False,
    enable_tags: bool = False,
    force_tags: bool = False,
    enable_stacking: bool = True,
):
    """Index all photos in a directory.

    Steps per photo:
      1. Extract EXIF metadata
      2. Check for ARW raw pair
      3. Compute file hash
      4. Insert into database
      5. Generate CLIP embedding (if enable_clip)
      6. Extract dominant colors (if enable_colors)
      7. Detect and encode faces (if enable_faces)
      8. Generate LLaVA description (if enable_describe)
      9. Aesthetic quality scoring (if enable_quality)

    Args:
        photo_dir: Directory to scan for photos.
        db_path: Path to SQLite database file.
        batch_size: Batch size for CLIP embedding.
        skip_existing: If True, skip photos already in the database.
        enable_clip: If True, generate CLIP embeddings.
        enable_colors: If True, extract dominant colors.
        enable_faces: If True, detect and encode faces.
        force_faces: If True, clear existing face data and re-run detection on all photos.
        force_clip: If True, clear existing CLIP embeddings and regenerate for all photos.
                    Use this when switching CLIP models to avoid stale embeddings.
        enable_describe: If True, generate scene descriptions via LLaVA/Ollama.
        force_describe: If True, regenerate descriptions for all photos (even those
                        that already have one).
        describe_model: Ollama model name for descriptions (default: "llava").
        enable_quality: If True, compute aesthetic quality scores.
        force_quality: If True, rescore all photos (even those already scored).
        enable_tags: If True, generate semantic category tags via Ollama.
        force_tags: If True, regenerate tags for all photos (even those already tagged).
    """
    photo_dir = str(Path(photo_dir).resolve())
    photos = find_photos(photo_dir)

    if not photos:
        print(f"No photos found in {photo_dir}")
        return

    print(f"Found {len(photos)} photos in {photo_dir}")

    # Error tracking — continue on individual photo failures
    errors: list[str] = []

    def _report_error(step: str, path: str, exc: Exception):
        msg = f"[{step}] {os.path.basename(path)}: {exc}"
        errors.append(msg)
        print(f" ERROR: {exc}")

    def _eta(t0: float, done: int, total: int) -> str:
        """Estimate time remaining."""
        if done == 0:
            return ""
        elapsed = time.time() - t0
        rate = done / elapsed
        remaining = (total - done) / rate if rate > 0 else 0
        if remaining < 60:
            return f" ~{remaining:.0f}s left"
        elif remaining < 3600:
            return f" ~{remaining / 60:.0f}m left"
        else:
            return f" ~{remaining / 3600:.1f}h left"

    with PhotoDB(db_path) as db:
        # If photo_root is not set yet, store it from the common ancestor of photo_dir.
        # This enables relative paths in the DB for portability across machines.
        if not db.photo_root:
            db.set_photo_root(photo_dir)
            print(f"  Set photo root: {db.photo_root}")

        # Step 1: EXIF + metadata for all photos
        new_photos = []  # list of (photo_id, absolute_path)
        db.begin_batch(batch_size=100)
        skipped = 0
        for i, photo_path in enumerate(photos, 1):
            abs_path = str(Path(photo_path).resolve())
            stored_path = db.relative_filepath(abs_path)
            if skip_existing and db.get_photo_by_path(stored_path):
                skipped += 1
                if skipped % 500 == 0 or i == len(photos):
                    print(f"  [{i}/{len(photos)}] Skipped {skipped} already-indexed photos so far...")
                continue

            try:
                meta = extract_exif(photo_path)
                # Convert to relative path for storage
                meta["filepath"] = stored_path

                # Check for ARW raw pair
                raw_path = find_raw_pair(photo_path)
                if raw_path:
                    meta["raw_filepath"] = db.relative_filepath(str(Path(raw_path).resolve()))

                # File hash for deduplication
                meta["file_hash"] = file_hash(photo_path)

                # Insert into DB
                photo_id = db.add_photo(**meta)
                new_photos.append((photo_id, abs_path))

                if len(new_photos) % 100 == 0:
                    print(f"  [{i}/{len(photos)}] Indexed {len(new_photos)} new photos so far...")

            except Exception as e:
                _report_error("metadata", photo_path, e)

        db.end_batch()

        if skipped:
            print(f"  Skipped {skipped} already-indexed photos.")

        # Build a set of stored paths for O(1) lookups at 200K scale.
        # Paths in the DB may be relative (to photo_root) or absolute.
        dir_photos = None  # lazy-loaded
        _stored_set = None

        def _get_dir_photos():
            """Return [(id, absolute_path)] for photos in the target directory."""
            nonlocal dir_photos, _stored_set
            if dir_photos is None:
                if _stored_set is None:
                    _stored_set = set(
                        db.relative_filepath(str(Path(p).resolve()))
                        for p in photos
                    )
                dir_photos = [
                    (row["id"], db.resolve_filepath(row["filepath"]))
                    for row in db.conn.execute("SELECT id, filepath FROM photos").fetchall()
                    if row["filepath"] in _stored_set
                ]
            return dir_photos

        # force_clip: wipe CLIP embeddings for photos in this directory and re-embed them.
        if force_clip and enable_clip:
            dir_photos_list = _get_dir_photos()
            dir_photo_ids_for_clip = [pid for pid, _ in dir_photos_list]
            print(f"\nClearing CLIP embeddings for {len(dir_photo_ids_for_clip)} photos in directory (--force-clip)...")
            if dir_photo_ids_for_clip:
                placeholders = ",".join("?" * len(dir_photo_ids_for_clip))
                db.conn.execute(
                    f"DELETE FROM clip_embeddings WHERE photo_id IN ({placeholders})",
                    dir_photo_ids_for_clip
                )
                db.conn.commit()
            new_photos = list(dir_photos_list)
            print(f"  Will re-embed {len(new_photos)} photo(s).")
        elif not new_photos and not enable_describe and not enable_faces and not enable_quality and not enable_tags:
            pass

        # When --clip is requested, also queue dir photos already in the DB but missing embeddings.
        if enable_clip and not force_clip:
            dir_photos_list = _get_dir_photos()
            if dir_photos_list:
                already_queued = {pid for pid, _ in new_photos}
                dir_photo_lookup = {pid: path for pid, path in dir_photos_list}
                # Chunk queries to stay within SQLite binding limits
                photos_with_clip = set()
                ids = list(dir_photo_lookup.keys())
                for i in range(0, len(ids), 500):
                    chunk = ids[i:i + 500]
                    placeholders = ",".join("?" * len(chunk))
                    rows = db.conn.execute(
                        f"SELECT photo_id FROM clip_embeddings WHERE photo_id IN ({placeholders})",
                        chunk
                    ).fetchall()
                    photos_with_clip.update(row[0] for row in rows)
                missing = [
                    (pid, path) for pid, path in dir_photos_list
                    if pid not in photos_with_clip and pid not in already_queued
                ]
                if missing:
                    print(f"  Found {len(missing)} photos in directory without CLIP embeddings.")
                    new_photos.extend(missing)

        if new_photos:
            print(f"\nIndexed {len(new_photos)} new photos into database.")

        # Step 1b: Reverse geocode GPS → place names (offline, fast)
        ungeo = db.conn.execute(
            "SELECT id, gps_lat, gps_lon FROM photos "
            "WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL AND place_name IS NULL"
        ).fetchall()
        dir_photo_ids = set(pid for pid, _ in _get_dir_photos())
        ungeo = [r for r in ungeo if r["id"] in dir_photo_ids]

        if ungeo:
            from .geocode import reverse_geocode_batch
            print(f"\nReverse geocoding {len(ungeo)} photo(s) with GPS data...")
            db.begin_batch(batch_size=200)
            coords = [(row["gps_lat"], row["gps_lon"]) for row in ungeo]
            places = reverse_geocode_batch(coords)
            geo_count = 0
            for row, place in zip(ungeo, places):
                if place:
                    db.update_photo(row["id"], place_name=place)
                    geo_count += 1
            db.end_batch()
            print(f"  Geocoded {geo_count}/{len(ungeo)} photos.")

        if not new_photos and not enable_describe and not enable_faces and not enable_quality and not enable_tags:
            if not ungeo:
                print("All photos already indexed. Nothing to do.")
            return

        # Step 2: CLIP embeddings (streamed — store each batch as it completes)
        if enable_clip:
            print(f"\nGenerating CLIP embeddings (batch_size={batch_size})...")
            t0 = time.time()
            paths = [p for _, p in new_photos]
            embedded_count = 0
            db.begin_batch(batch_size=100)
            for idx, emb in embed_images_stream(paths, batch_size=batch_size):
                photo_id, path = new_photos[idx]
                try:
                    db.add_clip_embedding(photo_id, emb)
                    embedded_count += 1
                except Exception as e:
                    _report_error("clip_store", path, e)
            db.end_batch()
            elapsed = time.time() - t0
            print(f"  Embedded {embedded_count}/{len(new_photos)} photos in {elapsed:.1f}s")

            # Free CLIP model memory before loading other models
            unload_clip()
            print("  CLIP model unloaded to free memory.")

        # Step 3: Dominant colors
        if enable_colors:
            print("\nExtracting dominant colors...")
            t0 = time.time()
            db.begin_batch(batch_size=100)
            color_count = 0
            for i, (photo_id, path) in enumerate(new_photos, 1):
                try:
                    colors = extract_dominant_colors(path)
                    if colors:
                        db.update_photo(photo_id, dominant_colors=colors_to_json(colors))
                        color_count += 1
                except Exception as e:
                    _report_error("colors", path, e)
                if i % 500 == 0:
                    print(f"  [{i}/{len(new_photos)}] colors extracted...{_eta(t0, i, len(new_photos))}")

            db.end_batch()
            elapsed = time.time() - t0
            print(f"  Extracted colors for {color_count}/{len(new_photos)} photos in {elapsed:.1f}s")

        # Step 4: Face detection and encoding
        if enable_faces:
            from .faces import detect_faces, cluster_encodings, check_available
            try:
                check_available()
            except RuntimeError as e:
                print(f"\nSkipping face detection: {e}")
            else:
                # Scope to target directory only
                dir_ids = set(pid for pid, _ in _get_dir_photos())
                if force_faces:
                    print("\nClearing existing face data for target directory...")
                    for pid in dir_ids:
                        db.conn.execute("DELETE FROM faces WHERE photo_id = ?", (pid,))
                    db.conn.commit()
                    all_face_candidates = list(_get_dir_photos())
                else:
                    unprocessed_rows = db.conn.execute(
                        """SELECT p.id, p.filepath FROM photos p
                           WHERE NOT EXISTS (
                               SELECT 1 FROM faces f WHERE f.photo_id = p.id
                           )"""
                    ).fetchall()
                    new_ids = {pid for pid, _ in new_photos}
                    all_face_candidates = list(new_photos) + [
                        (row["id"], db.resolve_filepath(row["filepath"]))
                        for row in unprocessed_rows
                        if row["id"] not in new_ids and row["id"] in dir_ids
                    ]

                total = len(all_face_candidates)
                print(f"\nDetecting faces in {total} photo(s)...")
                t0 = time.time()
                all_encodings = []
                all_face_ids = []

                db.begin_batch(batch_size=50)
                face_count = 0
                for idx, (photo_id, path) in enumerate(all_face_candidates, 1):
                    fname = os.path.basename(path)
                    print(f"  [{idx}/{total}] {fname} ...", end="", flush=True)
                    try:
                        t_photo = time.time()
                        faces = detect_faces(path, use_cnn=False)
                        elapsed_photo = time.time() - t_photo
                        if faces:
                            print(f" {len(faces)} face(s) ({elapsed_photo:.1f}s){_eta(t0, idx, total)}")
                        else:
                            print(f" no faces ({elapsed_photo:.1f}s)")
                        for face in faces:
                            face_id = db.add_face(
                                photo_id=photo_id,
                                bbox=face["bbox"],
                                encoding=face["encoding"],
                            )
                            all_encodings.append(face["encoding"])
                            all_face_ids.append(face_id)
                            face_count += 1
                    except Exception as e:
                        _report_error("faces", path, e)

                db.end_batch()
                elapsed = time.time() - t0
                print(f"  Found {face_count} face(s) across {total} photos in {elapsed:.1f}s")

                if all_encodings:
                    print("  Clustering faces...")
                    cluster_ids = cluster_encodings(all_encodings)
                    db.begin_batch(batch_size=200)
                    for face_id, cluster_id in zip(all_face_ids, cluster_ids):
                        db.conn.execute(
                            "UPDATE faces SET cluster_id = ? WHERE id = ?",
                            (cluster_id, face_id),
                        )
                    db.end_batch()
                    n_clusters = len(set(cluster_ids))
                    print(f"  Grouped into {n_clusters} cluster(s)")

        # Step 5: LLaVA scene descriptions via Ollama
        if enable_describe:
            from .describe import describe_photo, check_available as desc_check
            try:
                desc_check(model=describe_model)
            except RuntimeError as e:
                print(f"\nSkipping descriptions: {e}")
            else:
                if force_describe:
                    desc_candidates = list(_get_dir_photos())
                    print(f"\nGenerating descriptions for {len(desc_candidates)} photo(s) (--force-describe)...")
                else:
                    dir_ids = set(pid for pid, _ in _get_dir_photos())
                    undescribed = db.conn.execute(
                        "SELECT id, filepath FROM photos WHERE description IS NULL"
                    ).fetchall()
                    desc_candidates = [(row["id"], db.resolve_filepath(row["filepath"])) for row in undescribed
                                       if row["id"] in dir_ids]
                    if desc_candidates:
                        print(f"\nGenerating descriptions for {len(desc_candidates)} undescribed photo(s)...")
                    else:
                        print("\nAll photos in target directory already have descriptions.")

                if desc_candidates:
                    t0 = time.time()
                    db.begin_batch(batch_size=20)
                    desc_count = 0
                    total = len(desc_candidates)
                    for idx, (photo_id, path) in enumerate(desc_candidates, 1):
                        fname = os.path.basename(path)
                        print(f"  [{idx}/{total}] {fname} ...", end="", flush=True)
                        try:
                            t_photo = time.time()
                            desc = describe_photo(path, model=describe_model)
                            elapsed_photo = time.time() - t_photo
                            if desc:
                                db.update_photo(photo_id, description=desc)
                                preview = desc[:80].replace("\n", " ")
                                print(f" ({elapsed_photo:.1f}s) {preview}...{_eta(t0, idx, total)}")
                                desc_count += 1
                            else:
                                print(f" ({elapsed_photo:.1f}s) no description")
                        except Exception as e:
                            _report_error("describe", path, e)

                    db.end_batch()
                    elapsed = time.time() - t0
                    print(f"  Described {desc_count}/{total} photos in {elapsed:.1f}s")

        # Step 5b: Semantic tags via Ollama (M9)
        if enable_tags:
            import json as _json
            from .describe import tag_photo, check_available as tag_check
            try:
                tag_check(model=describe_model)
            except RuntimeError as e:
                print(f"\nSkipping tags: {e}")
            else:
                if force_tags:
                    tag_candidates = list(_get_dir_photos())
                    print(f"\nGenerating tags for {len(tag_candidates)} photo(s) (--force-tags)...")
                else:
                    dir_ids = set(pid for pid, _ in _get_dir_photos())
                    untagged = db.conn.execute(
                        "SELECT id, filepath FROM photos WHERE tags IS NULL"
                    ).fetchall()
                    tag_candidates = [(row["id"], db.resolve_filepath(row["filepath"])) for row in untagged
                                      if row["id"] in dir_ids]
                    if tag_candidates:
                        print(f"\nGenerating tags for {len(tag_candidates)} untagged photo(s)...")
                    else:
                        print("\nAll photos in target directory already have tags.")

                if tag_candidates:
                    t0 = time.time()
                    db.begin_batch(batch_size=20)
                    tag_count = 0
                    total = len(tag_candidates)
                    for idx, (photo_id, path) in enumerate(tag_candidates, 1):
                        fname = os.path.basename(path)
                        print(f"  [{idx}/{total}] {fname} ...", end="", flush=True)
                        try:
                            t_photo = time.time()
                            tags = tag_photo(path, model=describe_model)
                            elapsed_photo = time.time() - t_photo
                            if tags:
                                db.update_photo(photo_id, tags=_json.dumps(tags))
                                print(f" ({elapsed_photo:.1f}s) {', '.join(tags)}{_eta(t0, idx, total)}")
                                tag_count += 1
                            else:
                                print(f" ({elapsed_photo:.1f}s) no tags")
                        except Exception as e:
                            _report_error("tags", path, e)

                    db.end_batch()
                    elapsed = time.time() - t0
                    print(f"  Tagged {tag_count}/{total} photos in {elapsed:.1f}s")

        # Step 6: Aesthetic quality scoring + concept analysis
        if enable_quality:
            import json as _json
            from .quality import (
                score_photos_batch, analyze_photos_batch,
                unload_models as unload_quality,
            )

            if force_quality:
                quality_candidates = list(_get_dir_photos())
                print(f"\nScoring aesthetic quality for {len(quality_candidates)} photo(s) (--force-quality)...")
            else:
                dir_ids = set(pid for pid, _ in _get_dir_photos())
                unscored = db.conn.execute(
                    "SELECT id, filepath FROM photos WHERE aesthetic_score IS NULL"
                ).fetchall()
                quality_candidates = [(row["id"], db.resolve_filepath(row["filepath"]))
                                      for row in unscored if row["id"] in dir_ids]
                if quality_candidates:
                    print(f"\nScoring aesthetic quality for {len(quality_candidates)} unscored photo(s)...")
                else:
                    print("\nAll photos in target directory already have aesthetic scores.")

            if quality_candidates:
                t0 = time.time()
                paths = [p for _, p in quality_candidates]
                scored_count = 0

                # Stream scores and write to DB per-batch so progress survives a crash.
                from .quality import score_photos_stream as _score_stream
                db.begin_batch(batch_size=100)
                for idx, score in _score_stream(paths, batch_size=batch_size):
                    photo_id = quality_candidates[idx][0]
                    db.update_photo(photo_id, aesthetic_score=score)
                    scored_count += 1
                db.end_batch()

                elapsed = time.time() - t0
                print(f"  Scored {scored_count}/{len(quality_candidates)} photos in {elapsed:.1f}s")

                if scored_count > 0:
                    valid_scores = [s for s in scores if s is not None]
                    print(f"  Score range: {min(valid_scores):.2f} – {max(valid_scores):.2f} "
                          f"(mean: {sum(valid_scores)/len(valid_scores):.2f})")

            # Concept analysis (scoped to target directory)
            if force_quality:
                concept_candidates = quality_candidates
            else:
                dir_ids_c = set(pid for pid, _ in _get_dir_photos())
                unconcept = db.conn.execute(
                    "SELECT id, filepath FROM photos WHERE aesthetic_concepts IS NULL AND aesthetic_score IS NOT NULL"
                ).fetchall()
                concept_candidates = [(row["id"], db.resolve_filepath(row["filepath"]))
                                      for row in unconcept if row["id"] in dir_ids_c]

            if concept_candidates:
                print(f"\nAnalyzing aesthetic concepts for {len(concept_candidates)} photo(s)...")
                t0 = time.time()
                paths = [p for _, p in concept_candidates]
                concepts = analyze_photos_batch(paths, batch_size=batch_size)

                db.begin_batch(batch_size=100)
                concept_count = 0
                for (photo_id, path), concept_data in zip(concept_candidates, concepts):
                    if concept_data is not None:
                        db.update_photo(photo_id, aesthetic_concepts=_json.dumps(concept_data))
                        concept_count += 1

                db.end_batch()
                elapsed = time.time() - t0
                print(f"  Analyzed concepts for {concept_count}/{len(concept_candidates)} photos in {elapsed:.1f}s")

            unload_quality()

        # Step 7: Aesthetic critique via Ollama
        if enable_quality and enable_describe:
            from .describe import critique_photo, check_available as desc_check
            try:
                desc_check(model=describe_model)
            except RuntimeError as e:
                print(f"\nSkipping aesthetic critiques: {e}")
            else:
                if force_quality:
                    critique_candidates = quality_candidates if quality_candidates else []
                else:
                    dir_ids_cr = set(pid for pid, _ in _get_dir_photos())
                    uncritiqued = db.conn.execute(
                        "SELECT id, filepath FROM photos WHERE aesthetic_critique IS NULL AND aesthetic_score IS NOT NULL"
                    ).fetchall()
                    critique_candidates = [(row["id"], db.resolve_filepath(row["filepath"]))
                                           for row in uncritiqued if row["id"] in dir_ids_cr]

                if critique_candidates:
                    print(f"\nGenerating aesthetic critiques for {len(critique_candidates)} photo(s)...")
                    t0 = time.time()
                    db.begin_batch(batch_size=20)
                    crit_count = 0
                    total = len(critique_candidates)
                    for idx, (photo_id, path) in enumerate(critique_candidates, 1):
                        fname = os.path.basename(path)
                        print(f"  [{idx}/{total}] {fname} ...", end="", flush=True)
                        try:
                            t_photo = time.time()
                            critique = critique_photo(path, model=describe_model)
                            elapsed_photo = time.time() - t_photo
                            if critique:
                                db.update_photo(photo_id, aesthetic_critique=critique)
                                preview = critique[:80].replace("\n", " ")
                                print(f" ({elapsed_photo:.1f}s) {preview}...{_eta(t0, idx, total)}")
                                crit_count += 1
                            else:
                                print(f" ({elapsed_photo:.1f}s) no critique")
                        except Exception as e:
                            _report_error("critique", path, e)

                    db.end_batch()
                    elapsed = time.time() - t0
                    print(f"  Generated {crit_count}/{total} critiques in {elapsed:.1f}s")
                else:
                    print("\nAll scored photos already have aesthetic critiques.")

        # ── Step: Photo stacking (burst/bracket detection) ──────────
        # Runs whenever CLIP embeddings are available.
        if enable_stacking and enable_clip:
            print("\n── Photo stacking ──")
            try:
                from .stacking import run_stacking
                stacks = run_stacking(db, directory=photo_dir)
                if stacks:
                    total_stacked = sum(len(s) for s in stacks)
                    print(f"  Detected {len(stacks)} stacks ({total_stacked} photos)")
                else:
                    print("  No stacks detected.")
            except Exception as e:
                print(f"  Stacking failed: {e}")
                errors.append(f"stacking: {e}")

        # Summary
        print(f"\nDone! Database: {db_path} ({db.photo_count()} total photos)")
        if errors:
            print(f"\n{len(errors)} error(s) during indexing:")
            for err in errors[:20]:
                print(f"  - {err}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more.")
