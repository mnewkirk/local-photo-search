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
from .clip_embed import embed_image, embed_images_batch
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
    """
    photo_dir = str(Path(photo_dir).resolve())
    photos = find_photos(photo_dir)

    if not photos:
        print(f"No photos found in {photo_dir}")
        return

    print(f"Found {len(photos)} photos in {photo_dir}")

    with PhotoDB(db_path) as db:
        # Step 1: EXIF + metadata for all photos
        new_photos = []
        for i, photo_path in enumerate(photos, 1):
            if skip_existing and db.get_photo_by_path(str(Path(photo_path).resolve())):
                print(f"  [{i}/{len(photos)}] Skipping (already indexed): {os.path.basename(photo_path)}")
                continue

            print(f"  [{i}/{len(photos)}] Extracting metadata: {os.path.basename(photo_path)}")
            meta = extract_exif(photo_path)

            # Check for ARW raw pair
            raw_path = find_raw_pair(photo_path)
            if raw_path:
                meta["raw_filepath"] = str(Path(raw_path).resolve())
                print(f"    Found RAW pair: {os.path.basename(raw_path)}")

            # File hash for deduplication
            meta["file_hash"] = file_hash(photo_path)

            # Insert into DB
            photo_id = db.add_photo(**meta)
            new_photos.append((photo_id, photo_path))

        # Build a list of all photos in the TARGET DIRECTORY that are in the DB.
        # Used by --force-* flags to scope reprocessing to this directory, not the
        # entire database (which may contain photos from other directories).
        dir_photos = None  # lazy-loaded

        def _get_dir_photos():
            nonlocal dir_photos
            if dir_photos is None:
                resolved = [str(Path(p).resolve()) for p in photos]
                dir_photos = [
                    (row["id"], row["filepath"])
                    for row in db.conn.execute("SELECT id, filepath FROM photos").fetchall()
                    if row["filepath"] in resolved
                ]
            return dir_photos

        # force_clip: wipe all existing CLIP embeddings and re-embed every photo.
        # Required when switching CLIP models — old embeddings live in a different
        # vector space and will give nonsensical search results if mixed with new ones.
        # NOTE: force_clip is global (deletes ALL embeddings) because mixed vector
        # spaces break search. Other force flags are scoped to the target directory.
        all_photos = None  # lazy-loaded list of (id, filepath) for all indexed photos
        if force_clip and enable_clip:
            print("\nClearing existing CLIP embeddings for full regeneration (--force-clip)...")
            db.conn.execute("DELETE FROM clip_embeddings")
            db.conn.commit()
            all_photos = [
                (row["id"], row["filepath"])
                for row in db.conn.execute("SELECT id, filepath FROM photos").fetchall()
            ]
            new_photos = list(all_photos)
            print(f"  Will re-embed {len(new_photos)} photo(s).")
        elif not new_photos and not enable_describe and not enable_faces and not enable_quality:
            print("All photos already indexed. Nothing to do.")
            return

        if new_photos:
            print(f"\nIndexed {len(new_photos)} new photos into database.")

        # Step 2: CLIP embeddings (batched)
        if enable_clip:
            print(f"\nGenerating CLIP embeddings (batch_size={batch_size})...")
            t0 = time.time()
            paths = [p for _, p in new_photos]
            embeddings = embed_images_batch(paths, batch_size=batch_size)

            embedded_count = 0
            for (photo_id, path), emb in zip(new_photos, embeddings):
                if emb is not None:
                    db.add_clip_embedding(photo_id, emb)
                    embedded_count += 1

            elapsed = time.time() - t0
            print(f"  Embedded {embedded_count}/{len(new_photos)} photos in {elapsed:.1f}s")

        # Step 3: Dominant colors
        if enable_colors:
            print("\nExtracting dominant colors...")
            t0 = time.time()
            color_count = 0
            for photo_id, path in new_photos:
                colors = extract_dominant_colors(path)
                if colors:
                    db.update_photo(photo_id, dominant_colors=colors_to_json(colors))
                    color_count += 1

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
                if force_faces:
                    # Wipe all existing face data and re-detect everything
                    print("\nClearing existing face data for re-detection...")
                    db.conn.execute("DELETE FROM faces")
                    db.conn.commit()
                    all_rows = db.conn.execute(
                        "SELECT id, filepath FROM photos"
                    ).fetchall()
                    all_face_candidates = [(row["id"], row["filepath"]) for row in all_rows]
                else:
                    # Only process photos that have never had face detection run
                    unprocessed_rows = db.conn.execute(
                        """SELECT p.id, p.filepath FROM photos p
                           WHERE NOT EXISTS (
                               SELECT 1 FROM faces f WHERE f.photo_id = p.id
                           )"""
                    ).fetchall()
                    new_ids = {pid for pid, _ in new_photos}
                    all_face_candidates = list(new_photos) + [
                        (row["id"], row["filepath"])
                        for row in unprocessed_rows
                        if row["id"] not in new_ids
                    ]

                total = len(all_face_candidates)
                print(f"\nDetecting faces in {total} photo(s)...")
                t0 = time.time()
                all_encodings = []
                all_face_ids = []

                face_count = 0
                for idx, (photo_id, path) in enumerate(all_face_candidates, 1):
                    fname = os.path.basename(path)
                    print(f"  [{idx}/{total}] {fname} ...", end="", flush=True)
                    t_photo = time.time()
                    faces = detect_faces(path, use_cnn=False)
                    elapsed_photo = time.time() - t_photo
                    if faces:
                        print(f" {len(faces)} face(s) found ({elapsed_photo:.1f}s)")
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

                elapsed = time.time() - t0
                print(f"  Found {face_count} face(s) across {len(all_face_candidates)} photos in {elapsed:.1f}s")

                # Cluster all detected faces
                if all_encodings:
                    print("  Clustering faces...")
                    cluster_ids = cluster_encodings(all_encodings)
                    for face_id, cluster_id in zip(all_face_ids, cluster_ids):
                        db.conn.execute(
                            "UPDATE faces SET cluster_id = ? WHERE id = ?",
                            (cluster_id, face_id),
                        )
                    db.conn.commit()
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
                    # Regenerate descriptions for photos in the target directory
                    desc_candidates = list(_get_dir_photos())
                    print(f"\nGenerating descriptions for {len(desc_candidates)} photo(s) in target directory (--force-describe)...")
                else:
                    # Only describe photos that don't have a description yet
                    undescribed = db.conn.execute(
                        "SELECT id, filepath FROM photos WHERE description IS NULL"
                    ).fetchall()
                    desc_candidates = [(row["id"], row["filepath"]) for row in undescribed]
                    if desc_candidates:
                        print(f"\nGenerating descriptions for {len(desc_candidates)} undescribed photo(s)...")
                    else:
                        print("\nAll photos already have descriptions.")

                if desc_candidates:
                    t0 = time.time()
                    desc_count = 0
                    total = len(desc_candidates)
                    for idx, (photo_id, path) in enumerate(desc_candidates, 1):
                        fname = os.path.basename(path)
                        print(f"  [{idx}/{total}] {fname} ...", end="", flush=True)
                        t_photo = time.time()
                        desc = describe_photo(path, model=describe_model)
                        elapsed_photo = time.time() - t_photo
                        if desc:
                            db.update_photo(photo_id, description=desc)
                            preview = desc[:80].replace("\n", " ")
                            print(f" ({elapsed_photo:.1f}s) {preview}...")
                            desc_count += 1
                        else:
                            print(f" ({elapsed_photo:.1f}s) no description")

                    elapsed = time.time() - t0
                    print(f"  Described {desc_count}/{total} photos in {elapsed:.1f}s")

        # Step 6: Aesthetic quality scoring + concept analysis
        if enable_quality:
            import json as _json
            from .quality import (
                score_photos_batch, analyze_photos_batch,
                unload_models as unload_quality,
            )

            if force_quality:
                # Rescore photos in the target directory
                quality_candidates = list(_get_dir_photos())
                print(f"\nScoring aesthetic quality for {len(quality_candidates)} photo(s) in target directory (--force-quality)...")
            else:
                # Only score photos that don't have a score yet
                unscored = db.conn.execute(
                    "SELECT id, filepath FROM photos WHERE aesthetic_score IS NULL"
                ).fetchall()
                quality_candidates = [(row["id"], row["filepath"]) for row in unscored]
                if quality_candidates:
                    print(f"\nScoring aesthetic quality for {len(quality_candidates)} unscored photo(s)...")
                else:
                    print("\nAll photos already have aesthetic scores.")

            if quality_candidates:
                t0 = time.time()
                paths = [p for _, p in quality_candidates]
                scores = score_photos_batch(paths, batch_size=batch_size)

                scored_count = 0
                for (photo_id, path), score in zip(quality_candidates, scores):
                    if score is not None:
                        db.update_photo(photo_id, aesthetic_score=score)
                        scored_count += 1

                elapsed = time.time() - t0
                print(f"  Scored {scored_count}/{len(quality_candidates)} photos in {elapsed:.1f}s")

                # Show score distribution
                if scored_count > 0:
                    valid_scores = [s for s in scores if s is not None]
                    print(f"  Score range: {min(valid_scores):.2f} – {max(valid_scores):.2f} "
                          f"(mean: {sum(valid_scores)/len(valid_scores):.2f})")

            # Concept analysis — runs on the same ViT-L/14 model while it's loaded
            if force_quality:
                concept_candidates = quality_candidates
            else:
                unconcept = db.conn.execute(
                    "SELECT id, filepath FROM photos WHERE aesthetic_concepts IS NULL AND aesthetic_score IS NOT NULL"
                ).fetchall()
                concept_candidates = [(row["id"], row["filepath"]) for row in unconcept]

            if concept_candidates:
                print(f"\nAnalyzing aesthetic concepts for {len(concept_candidates)} photo(s)...")
                t0 = time.time()
                paths = [p for _, p in concept_candidates]
                concepts = analyze_photos_batch(paths, batch_size=batch_size)

                concept_count = 0
                for (photo_id, path), concept_data in zip(concept_candidates, concepts):
                    if concept_data is not None:
                        db.update_photo(photo_id, aesthetic_concepts=_json.dumps(concept_data))
                        concept_count += 1

                elapsed = time.time() - t0
                print(f"  Analyzed concepts for {concept_count}/{len(concept_candidates)} photos in {elapsed:.1f}s")

            # Free the aesthetic model memory before other steps
            unload_quality()

        # Step 7: Aesthetic critique via Ollama (optional, runs after quality scoring)
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
                    uncritiqued = db.conn.execute(
                        "SELECT id, filepath FROM photos WHERE aesthetic_critique IS NULL AND aesthetic_score IS NOT NULL"
                    ).fetchall()
                    critique_candidates = [(row["id"], row["filepath"]) for row in uncritiqued]

                if critique_candidates:
                    print(f"\nGenerating aesthetic critiques for {len(critique_candidates)} photo(s)...")
                    t0 = time.time()
                    crit_count = 0
                    total = len(critique_candidates)
                    for idx, (photo_id, path) in enumerate(critique_candidates, 1):
                        fname = os.path.basename(path)
                        print(f"  [{idx}/{total}] {fname} ...", end="", flush=True)
                        t_photo = time.time()
                        critique = critique_photo(path, model=describe_model)
                        elapsed_photo = time.time() - t_photo
                        if critique:
                            db.update_photo(photo_id, aesthetic_critique=critique)
                            preview = critique[:80].replace("\n", " ")
                            print(f" ({elapsed_photo:.1f}s) {preview}...")
                            crit_count += 1
                        else:
                            print(f" ({elapsed_photo:.1f}s) no critique")
                    elapsed = time.time() - t0
                    print(f"  Generated {crit_count}/{total} critiques in {elapsed:.1f}s")
                else:
                    print("\nAll scored photos already have aesthetic critiques.")

        print(f"\nDone! Database: {db_path} ({db.photo_count()} total photos)")
