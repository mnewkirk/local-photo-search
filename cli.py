#!/usr/bin/env python3
"""CLI entry point for local-photo-search."""

import warnings
# Suppress FutureWarning from insightface/utils/face_align.py (scikit-image API change).
# Must be set before any imports that might trigger it.
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

import json
import logging
import logging.handlers
import os
from pathlib import Path

import click

from photosearch.db import PhotoDB
from photosearch.index import index_directory
from photosearch.search import (
    search_combined,
    symlink_results,
    make_results_subdir,
)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Local photo search — find photos by person, place, description, or color."""
    pass


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("photo_dir", type=click.Path(exists=True))
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
@click.option("--batch-size", default=8, help="Batch size for CLIP embedding.")
@click.option("--clip", is_flag=True, help="Generate CLIP embeddings (required for semantic search).")
@click.option("--no-colors", is_flag=True, help="Skip dominant color extraction.")
@click.option("--faces", is_flag=True, help="Detect and encode faces.")
@click.option("--force-faces", is_flag=True, help="Clear all face data and re-run detection on every photo.")
@click.option("--force-clip", is_flag=True, help="Clear and regenerate CLIP embeddings for photos in this directory. Use to fix stale embeddings.")
@click.option("--describe", is_flag=True, help="Generate scene descriptions via LLaVA (requires Ollama).")
@click.option("--force-describe", is_flag=True, help="Regenerate descriptions for all photos, even those that already have one.")
@click.option("--describe-model", default="llava", show_default=True, help="Ollama model for descriptions.")
@click.option("--quality", is_flag=True, help="Compute aesthetic quality scores (1–10 scale).")
@click.option("--force-quality", is_flag=True, help="Rescore quality for all photos, even those already scored.")
@click.option("--tags", is_flag=True, help="Generate semantic tags via LLaVA (requires --describe or Ollama).")
@click.option("--force-tags", is_flag=True, help="Regenerate tags for all photos, even those that already have them.")
@click.option("--full", is_flag=True, help="Enable all optional pipelines: --faces --describe --quality --tags. Equivalent to passing each flag individually.")
@click.option("--verify", is_flag=True, help="Run hallucination verification after indexing (requires descriptions).")
def index(photo_dir, db, batch_size, clip, no_colors, faces, force_faces, force_clip,
          describe, force_describe, describe_model, quality, force_quality, tags, force_tags, full,
          verify):
    """Index a directory of photos."""
    if full:
        clip = True
        faces = True
        describe = True
        quality = True
        tags = True
    index_directory(
        photo_dir=photo_dir,
        db_path=db,
        batch_size=batch_size,
        enable_clip=clip or force_clip,
        enable_colors=not no_colors,
        enable_faces=faces or force_faces,
        force_faces=force_faces,
        force_clip=force_clip,
        enable_describe=describe or force_describe,
        force_describe=force_describe,
        describe_model=describe_model,
        enable_quality=quality or force_quality,
        force_quality=force_quality,
        enable_tags=tags or force_tags,
        force_tags=force_tags,
    )

    if verify and (describe or force_describe or tags or force_tags or full):
        click.echo("\n--- Verification pass ---")
        from photosearch.db import PhotoDB
        from photosearch.verify import verify_photos
        photo_db = PhotoDB(db, photo_root=photo_dir)
        stats = verify_photos(photo_db, verify_model="minicpm-v", regen_model=describe_model)
        click.echo(f"Verification: {stats['passed']} passed, "
                    f"{stats['regenerated']} regenerated, {stats['failed']} failed")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--query", "-q", help="Semantic search query (natural language).")
@click.option("--person", "-p", help="Search by person name.")
@click.option("--place", help="Search by place name.")
@click.option("--color", "-c", help="Search by dominant color (name or #hex).")
@click.option("--face", type=click.Path(exists=True), help="Search by reference face image.")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
@click.option("--limit", "-n", default=10, help="Max number of results.")
@click.option("--results-dir", default="results", help="Base directory for result subfolders.")
@click.option("--no-results", is_flag=True, help="Don't write a results folder.")
@click.option("--json-output", is_flag=True, help="Output results as JSON.")
@click.option("--min-score", default=-0.25, show_default=True,
              help="Minimum CLIP similarity score to return a result (-1 to 1). "
                   "Raise toward 0 for stricter matching; lower to be more permissive.")
@click.option("--min-quality", type=float, default=None,
              help="Minimum aesthetic quality score (1–10). "
                   "Try 5.0 for good photos, 6.0 for excellent, 7.0 for outstanding.")
@click.option("--sort-quality", is_flag=True, help="Sort results by aesthetic quality (highest first) instead of relevance.")
@click.option("--debug", is_flag=True, help="Write step-by-step search log to debug.log in the results folder.")
@click.option("--tag-match", type=click.Choice(["dict", "tags", "both"]), default="both",
              show_default=True, help="Text matching mode: dict (term expansion), tags (LLM tags), both (take best).")
@click.option("--date", "date_exact", default=None,
              help="Filter by date: YYYY, YYYY-MM, or YYYY-MM-DD. Also supports natural language in --query.")
@click.option("--from", "date_from", default=None,
              help="Filter photos taken on or after this date (YYYY-MM-DD or YYYY-MM).")
@click.option("--to", "date_to", default=None,
              help="Filter photos taken on or before this date (YYYY-MM-DD or YYYY-MM).")
@click.option("--location", "-l", default=None,
              help="Filter by location/place name (matched against geocoded place names).")
def search(query, person, place, color, face, db, limit, results_dir, no_results, json_output, min_score,
           min_quality, sort_quality, debug, tag_match, date_exact, date_from, date_to, location):
    """Search indexed photos."""
    # When --debug is used, capture log lines in a buffer (written to file later).
    debug_handler = None
    if debug:
        debug_handler = logging.handlers.MemoryHandler(capacity=100000, flushLevel=logging.CRITICAL)
        debug_handler.setFormatter(logging.Formatter("%(message)s"))
        search_logger = logging.getLogger("photosearch.search")
        search_logger.setLevel(logging.INFO)
        search_logger.addHandler(debug_handler)

    # Resolve --date into --from / --to range
    if date_exact:
        from photosearch.date_parse import _parse_single_date
        parsed = _parse_single_date(date_exact)
        if parsed:
            date_from = date_from or parsed[0]
            date_to = date_to or parsed[1]
        else:
            click.echo(f"Could not parse date: {date_exact}")
            return

    if not any([query, person, place, color, face, min_quality is not None,
                date_from, date_to, location]):
        click.echo("Please provide at least one search criterion. See --help for options.")
        return

    with PhotoDB(db) as photo_db:
        results = search_combined(
            db=photo_db,
            query=query,
            color=color,
            place=place,
            person=person,
            face_image=face,
            limit=limit,
            min_score=min_score,
            min_quality=min_quality,
            sort_quality=sort_quality,
            debug=debug,
            tag_match=tag_match,
            date_from=date_from,
            date_to=date_to,
            location=location,
        )

        if not results:
            click.echo("No matching photos found.")
            return

        # Always show the summary table in the CLI
        click.echo(f"\nFound {len(results)} matching photos:\n")
        has_quality = any(r.get("aesthetic_score") is not None for r in results)
        if has_quality:
            click.echo(f"{'#':<4} {'Filename':<20} {'Date':<22} {'Score':<8} {'Quality':<8} {'Colors'}")
            click.echo("-" * 90)
        else:
            click.echo(f"{'#':<4} {'Filename':<20} {'Date':<22} {'Score':<8} {'Colors'}")
            click.echo("-" * 80)
        for i, r in enumerate(results, 1):
            filename = r.get("filename", "?")
            date = r.get("date_taken", "")[:19] if r.get("date_taken") else ""
            score = f"{r.get('score', 0):.3f}" if "score" in r else ""
            quality = f"{r['aesthetic_score']:.1f}" if r.get("aesthetic_score") is not None else ""
            colors = ""
            if r.get("dominant_colors"):
                try:
                    color_list = json.loads(r["dominant_colors"])
                    colors = " ".join(color_list[:3])
                except (json.JSONDecodeError, TypeError):
                    pass
            if has_quality:
                click.echo(f"{i:<4} {filename:<20} {date:<22} {score:<8} {quality:<8} {colors}")
            else:
                click.echo(f"{i:<4} {filename:<20} {date:<22} {score:<8} {colors}")

        # If --json-output, also print JSON to stdout (for piping)
        if json_output:
            clean = [{k: v for k, v in r.items() if v is not None} for r in results]
            click.echo(json.dumps(clean, indent=2))

        if not no_results and results:
            subfolder = make_results_subdir(
                results_dir,
                {"query": query, "color": color, "place": place, "person": person},
            )
            result_path = symlink_results(results, output_dir=subfolder)
            click.echo(f"\nResults folder: {result_path}")

            # Always write results.json to the results folder
            json_path = os.path.join(result_path, "results.json")
            clean_results = [{k: v for k, v in r.items() if v is not None} for r in results]
            with open(json_path, "w") as f:
                json.dump(clean_results, f, indent=2)
            click.echo(f"JSON results:   {json_path}")

            # Write debug log to file if --debug was used
            if debug and debug_handler:
                debug_log_path = os.path.join(result_path, "debug.log")
                with open(debug_log_path, "w") as f:
                    for record in debug_handler.buffer:
                        f.write(debug_handler.formatter.format(record) + "\n")
                click.echo(f"Debug log:      {debug_log_path}")

        elif debug and debug_handler:
            # --no-results but --debug: dump debug log to a temp location
            debug_log_path = os.path.join(results_dir, "debug.log")
            os.makedirs(results_dir, exist_ok=True)
            with open(debug_log_path, "w") as f:
                for record in debug_handler.buffer:
                    f.write(debug_handler.formatter.format(record) + "\n")
            click.echo(f"Debug log:      {debug_log_path}")


# ---------------------------------------------------------------------------
# add-person
# ---------------------------------------------------------------------------

@cli.command("add-person")
@click.argument("name")
@click.option("--photo", "photos", multiple=True, type=click.Path(exists=True),
              help="Reference photo(s) containing this person's face. Can be repeated.")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
def add_person(name, photos, db):
    """Register a named person using one or more reference photos.

    \b
    Example:
      python cli.py add-person "Matt" --photo ~/Desktop/matt.jpg
      python cli.py add-person "Matt" --photo photo1.jpg --photo photo2.jpg
    """
    from photosearch.faces import encode_reference_photo, check_available
    from photosearch.db import FACE_DIMENSIONS
    import struct

    try:
        check_available()
    except RuntimeError as e:
        click.echo(str(e))
        return

    with PhotoDB(db) as photo_db:
        # Create or retrieve the person
        existing = photo_db.get_person_by_name(name)
        if existing:
            person_id = existing["id"]
            click.echo(f"Person '{name}' already exists (id={person_id}). Adding new reference(s).")
        else:
            person_id = photo_db.add_person(name)
            click.echo(f"Created person '{name}' (id={person_id}).")

        if not photos:
            click.echo("No reference photos provided. Use --photo to add reference images.")
            click.echo("Run 'match-faces' after adding references to apply them to indexed photos.")
            return

        added = 0
        for photo_path in photos:
            click.echo(f"  Encoding face from: {photo_path}")
            encoding = encode_reference_photo(photo_path)
            if encoding is None:
                click.echo(f"  Skipping — no face found in {photo_path}")
                continue

            # Insert into face_references table
            cur = photo_db.conn.execute(
                "INSERT INTO face_references (person_id, source_path) VALUES (?, ?)",
                (person_id, str(photo_path)),
            )
            ref_id = cur.lastrowid

            # Store encoding in sqlite-vec table
            photo_db.conn.execute(
                "INSERT INTO face_ref_encodings (ref_id, encoding) VALUES (?, ?)",
                (ref_id, struct.pack(f"{FACE_DIMENSIONS}f", *encoding)),
            )
            photo_db.conn.commit()
            added += 1
            click.echo(f"  ✓ Reference added.")

        if added:
            click.echo(f"\nAdded {added} reference(s) for '{name}'.")
            click.echo("Now run: python cli.py match-faces")


# ---------------------------------------------------------------------------
# add-persons  (batch version — reads a YAML config, one model load)
# ---------------------------------------------------------------------------

@cli.command("add-persons")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True),
              help="Path to a YAML file mapping person names to reference photo paths.")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
def add_persons(config_path, db):
    """Register multiple persons from a YAML config — one model load, all at once.

    \b
    Config format (references.yml):
      Matt:
        - /references/matt_reference.jpg
      Calvin:
        - /references/calvin_reference.jpg
        - /references/calvin_reference2.JPG

    \b
    Example (Docker on NAS):
      docker compose -f docker-compose.nas.yml run --rm \\
        -v /home/user/references:/references:ro \\
        photosearch add-persons --config /references/references.yml
    """
    import struct
    try:
        import yaml
    except ImportError:
        click.echo("PyYAML is required: pip install pyyaml")
        return

    from photosearch.faces import encode_reference_photo, check_available, _get_face_app
    from photosearch.db import FACE_DIMENSIONS

    try:
        check_available()
    except RuntimeError as e:
        click.echo(str(e))
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not config or not isinstance(config, dict):
        click.echo("Config file is empty or not a valid YAML mapping.")
        return

    # Load the model once before processing anyone
    click.echo("Loading InsightFace model (buffalo_l)...")
    _get_face_app()
    click.echo("")

    with PhotoDB(db) as photo_db:
        total_added = 0
        for name, photos in config.items():
            if not photos:
                click.echo(f"[{name}] No photos listed — skipping.")
                continue

            click.echo(f"[{name}]")

            existing = photo_db.get_person_by_name(name)
            if existing:
                person_id = existing["id"]
                click.echo(f"  Person already exists (id={person_id}). Adding new reference(s).")
            else:
                person_id = photo_db.add_person(name)
                click.echo(f"  Created person (id={person_id}).")

            added = 0
            for photo_path in photos:
                click.echo(f"  Encoding: {photo_path}")
                encoding = encode_reference_photo(str(photo_path))
                if encoding is None:
                    click.echo(f"  ✗ No face found — skipping.")
                    continue

                cur = photo_db.conn.execute(
                    "INSERT INTO face_references (person_id, source_path) VALUES (?, ?)",
                    (person_id, str(photo_path)),
                )
                ref_id = cur.lastrowid
                photo_db.conn.execute(
                    "INSERT INTO face_ref_encodings (ref_id, encoding) VALUES (?, ?)",
                    (ref_id, struct.pack(f"{FACE_DIMENSIONS}f", *encoding)),
                )
                photo_db.conn.commit()
                added += 1
                click.echo(f"  ✓ Reference added.")

            click.echo(f"  → {added} reference(s) added for '{name}'.\n")
            total_added += added

    click.echo(f"Done. {total_added} total reference(s) added.")
    click.echo("Now run: python cli.py match-faces")


# ---------------------------------------------------------------------------
# match-faces
# ---------------------------------------------------------------------------

@cli.command("match-faces")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
@click.option("--tolerance", default=1.15, help="Match tolerance — L2 distance on 512-dim ArcFace vectors (lower = stricter). Default: 1.15")
@click.option("--temporal", is_flag=True, default=False,
              help="After standard matching, propagate identities to nearby unmatched faces "
                   "using EXIF timestamps (same-session context). Safe for family photos; "
                   "use with caution if photos include team sports or uniformed crowds.")
@click.option("--temporal-tolerance", default=1.45,
              help="Looser ArcFace L2 tolerance used for temporal propagation. Default: 1.45")
@click.option("--temporal-window", default=30,
              help="Time window in minutes defining a 'session' for temporal matching. Default: 30")
def match_faces(db, tolerance, temporal, temporal_tolerance, temporal_window):
    """Match all unidentified faces against registered persons.

    Run this after 'add-person' to apply name labels to indexed faces.

    \b
    Two-pass matching:
      Pass 1 (always): strict ArcFace distance matching against reference photos.
      Pass 2 (--temporal): looser matching for faces where the best-match person
        appears in a photo taken within --temporal-window minutes, AND their
        ArcFace distance is clearly better than the runner-up. This handles small
        or angled faces that fall just outside the strict threshold when photos
        were taken in the same session (same clothes, same lighting).
    """
    from photosearch.faces import (
        match_faces_to_persons, match_faces_temporal, check_available,
    )

    try:
        check_available()
    except RuntimeError as e:
        click.echo(str(e))
        return

    with PhotoDB(db) as photo_db:
        click.echo(f"Matching faces (tolerance={tolerance})...")
        matched = match_faces_to_persons(photo_db, tolerance=tolerance)
        click.echo(f"  Pass 1: matched {matched} face(s) to known persons.")

        if temporal:
            click.echo(
                f"Running temporal propagation "
                f"(tolerance={temporal_tolerance}, window={temporal_window}min)..."
            )
            t_matched = match_faces_temporal(
                photo_db,
                temporal_tolerance=temporal_tolerance,
                window_minutes=temporal_window,
            )
            click.echo(f"  Pass 2: matched {t_matched} additional face(s) via session context.")
            matched += t_matched

        click.echo(f"Total: {matched} face(s) matched.")


# ---------------------------------------------------------------------------
# list-persons
# ---------------------------------------------------------------------------

@cli.command("list-persons")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
def list_persons(db):
    """List all registered persons and their photo counts."""
    with PhotoDB(db) as photo_db:
        rows = photo_db.conn.execute(
            """SELECT p.id, p.name,
                      COUNT(DISTINCT f.photo_id) as photo_count,
                      COUNT(DISTINCT f.id) as face_count
               FROM persons p
               LEFT JOIN faces f ON f.person_id = p.id
               GROUP BY p.id
               ORDER BY p.name"""
        ).fetchall()

        if not rows:
            click.echo("No persons registered. Use 'add-person' to get started.")
            return

        click.echo(f"\n{'ID':<6} {'Name':<24} {'Photos':<10} {'Faces'}")
        click.echo("-" * 50)
        for row in rows:
            click.echo(f"{row['id']:<6} {row['name']:<24} {row['photo_count']:<10} {row['face_count']}")


# ---------------------------------------------------------------------------
# face-clusters
# ---------------------------------------------------------------------------

@cli.command("face-clusters")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
def face_clusters(db):
    """Show a summary of unidentified face clusters.

    Clusters are groups of similar faces not yet matched to a named person.
    Use this to identify who to label with 'add-person'.
    """
    with PhotoDB(db) as photo_db:
        rows = photo_db.conn.execute(
            """SELECT f.cluster_id,
                      COUNT(DISTINCT f.id) as face_count,
                      COUNT(DISTINCT f.photo_id) as photo_count
               FROM faces f
               WHERE f.person_id IS NULL AND f.cluster_id IS NOT NULL
               GROUP BY f.cluster_id
               ORDER BY face_count DESC"""
        ).fetchall()

        if not rows:
            click.echo("No unidentified face clusters found.")
            return

        click.echo(f"\nUnidentified face clusters:\n")
        click.echo(f"{'Cluster':<10} {'Faces':<10} {'Photos'}")
        click.echo("-" * 35)
        for row in rows:
            click.echo(f"{row['cluster_id']:<10} {row['face_count']:<10} {row['photo_count']}")

        click.echo(f"\nTotal: {len(rows)} cluster(s)")
        click.echo("Use 'add-person <name> --photo <ref.jpg>' to identify people.")


# ---------------------------------------------------------------------------
# correct-face
# ---------------------------------------------------------------------------

@cli.command("correct-face")
@click.argument("filename")
@click.argument("face_number", type=int)
@click.argument("correct_person")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
def correct_face(filename, face_number, correct_person, db):
    """Correct a face match in a photo.

    FACE_NUMBER is the face index shown by 'diagnose-photo' (1-based).
    Use CORRECT_PERSON as 'unknown' to clear a wrong match without reassigning.

    \b
    Examples:
      # Reassign face 1 in DSC04922 from Jamie to Alex
      python cli.py correct-face DSC04922.JPG 1 "Alex"

      # Clear a wrong match (mark face as unidentified)
      python cli.py correct-face DSC04922.JPG 2 unknown
    """
    with PhotoDB(db) as photo_db:
        photo_row = photo_db.conn.execute(
            "SELECT id FROM photos WHERE filename = ?", (filename,)
        ).fetchone()
        if not photo_row:
            click.echo(f"Photo '{filename}' not found in database.")
            return

        face_rows = photo_db.conn.execute(
            """SELECT f.id, f.person_id, p.name as person_name
               FROM faces f
               LEFT JOIN persons p ON p.id = f.person_id
               WHERE f.photo_id = ?
               ORDER BY f.id""",
            (photo_row["id"],),
        ).fetchall()

        if not face_rows:
            click.echo(f"No faces found in {filename}.")
            return

        if face_number < 1 or face_number > len(face_rows):
            click.echo(f"Face number must be between 1 and {len(face_rows)}.")
            return

        target_face = face_rows[face_number - 1]
        old_name = target_face["person_name"] or "Unidentified"

        if correct_person.lower() == "unknown":
            # Clear the match
            photo_db.conn.execute(
                "UPDATE faces SET person_id = NULL WHERE id = ?", (target_face["id"],)
            )
            photo_db.conn.commit()
            click.echo(f"✓ Face {face_number} in {filename} cleared (was: {old_name}).")
        else:
            # Reassign to the named person, creating them if needed
            person = photo_db.get_person_by_name(correct_person)
            if not person:
                person_id = photo_db.add_person(correct_person)
                click.echo(f"Created new person '{correct_person}'.")
            else:
                person_id = person["id"]

            photo_db.conn.execute(
                "UPDATE faces SET person_id = ? WHERE id = ?", (person_id, target_face["id"])
            )
            photo_db.conn.commit()
            click.echo(f"✓ Face {face_number} in {filename} reassigned: {old_name} → {correct_person}.")


# ---------------------------------------------------------------------------
# clear-matches
# ---------------------------------------------------------------------------

@cli.command("clear-matches")
@click.argument("photo_dir", type=click.Path(exists=True))
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
@click.option("--person", default=None, help="Only clear matches for this person (by name).")
@click.option("--all-faces", is_flag=True, help="Also delete the detected face data itself, not just the person assignments.")
@click.option("--include-manual", is_flag=True, help="Also clear manual (UI) assignments. By default these are preserved.")
def clear_matches(photo_dir, db, person, all_faces, include_manual):
    """Clear face matches for all photos in a directory.

    By default this resets person_id to NULL on every auto-matched face
    in the directory, preserving manual assignments made through the UI.
    Use --include-manual to clear those too.

    \b
    Examples:
      # Clear auto matches in a folder (keeps manual + face detections)
      python cli.py clear-matches ../Photos/2026-02-08

      # Clear all matches including manual UI assignments
      python cli.py clear-matches ../Photos/2026-02-08 --include-manual

      # Clear only Jamie matches (leave others intact)
      python cli.py clear-matches ../Photos/2026-02-08 --person "Jamie"

      # Nuke everything — clear detections too (will need --faces to re-detect)
      python cli.py clear-matches ../Photos/2026-02-08 --all-faces
    """
    photo_dir = str(Path(photo_dir).resolve())

    with PhotoDB(db) as photo_db:
        # Find photos in the target directory
        all_rows = photo_db.conn.execute(
            "SELECT id, filepath FROM photos"
        ).fetchall()
        dir_photo_ids = []
        for row in all_rows:
            abs_path = photo_db.resolve_filepath(row["filepath"])
            if abs_path and abs_path.startswith(photo_dir + "/"):
                dir_photo_ids.append(row["id"])

        if not dir_photo_ids:
            click.echo(f"No indexed photos found in {photo_dir}")
            return

        # Build the face query
        placeholders = ",".join("?" * len(dir_photo_ids))

        # Manual-assignment filter: by default preserve manual (UI) assignments
        manual_guard = "" if include_manual else " AND (match_source IS NULL OR match_source != 'manual')"

        if all_faces:
            # Delete face records entirely (respects --include-manual)
            if include_manual:
                count = photo_db.conn.execute(
                    f"SELECT COUNT(*) as c FROM faces WHERE photo_id IN ({placeholders})",
                    dir_photo_ids,
                ).fetchone()["c"]
                photo_db.conn.execute(
                    f"DELETE FROM faces WHERE photo_id IN ({placeholders})",
                    dir_photo_ids,
                )
            else:
                count = photo_db.conn.execute(
                    f"SELECT COUNT(*) as c FROM faces WHERE photo_id IN ({placeholders}){manual_guard}",
                    dir_photo_ids,
                ).fetchone()["c"]
                photo_db.conn.execute(
                    f"DELETE FROM faces WHERE photo_id IN ({placeholders}){manual_guard}",
                    dir_photo_ids,
                )
                manual_kept = photo_db.conn.execute(
                    f"SELECT COUNT(*) as c FROM faces WHERE photo_id IN ({placeholders}) AND match_source = 'manual'",
                    dir_photo_ids,
                ).fetchone()["c"]
                if manual_kept:
                    click.echo(f"  Preserved {manual_kept} manual assignment(s). Use --include-manual to clear those too.")
            photo_db.conn.commit()
            click.echo(f"Deleted {count} face detection(s) across {len(dir_photo_ids)} photos in {photo_dir}")
            click.echo("Re-run with --faces to re-detect.")
        elif person:
            # Clear matches for a specific person
            person_row = photo_db.conn.execute(
                "SELECT id FROM persons WHERE name = ?", (person,)
            ).fetchone()
            if not person_row:
                click.echo(f"Person '{person}' not found. Use list-persons to see registered people.")
                return
            person_id = person_row["id"]
            count = photo_db.conn.execute(
                f"SELECT COUNT(*) as c FROM faces WHERE photo_id IN ({placeholders}) AND person_id = ?{manual_guard}",
                dir_photo_ids + [person_id],
            ).fetchone()["c"]
            photo_db.conn.execute(
                f"UPDATE faces SET person_id = NULL, match_source = NULL WHERE photo_id IN ({placeholders}) AND person_id = ?{manual_guard}",
                dir_photo_ids + [person_id],
            )
            photo_db.conn.commit()
            click.echo(f"Cleared {count} '{person}' match(es) across {len(dir_photo_ids)} photos.")
            if not include_manual:
                manual_kept = photo_db.conn.execute(
                    f"SELECT COUNT(*) as c FROM faces WHERE photo_id IN ({placeholders}) AND person_id = ? AND match_source = 'manual'",
                    dir_photo_ids + [person_id],
                ).fetchone()["c"]
                if manual_kept:
                    click.echo(f"  Preserved {manual_kept} manual assignment(s). Use --include-manual to clear those too.")
        else:
            # Clear all person assignments (preserve manual by default)
            count = photo_db.conn.execute(
                f"SELECT COUNT(*) as c FROM faces WHERE photo_id IN ({placeholders}) AND person_id IS NOT NULL{manual_guard}",
                dir_photo_ids,
            ).fetchone()["c"]
            photo_db.conn.execute(
                f"UPDATE faces SET person_id = NULL, match_source = NULL WHERE photo_id IN ({placeholders}) AND person_id IS NOT NULL{manual_guard}",
                dir_photo_ids,
            )
            photo_db.conn.commit()
            click.echo(f"Cleared {count} face match(es) across {len(dir_photo_ids)} photos in {photo_dir}")
            if not include_manual:
                manual_kept = photo_db.conn.execute(
                    f"SELECT COUNT(*) as c FROM faces WHERE photo_id IN ({placeholders}) AND match_source = 'manual'",
                    dir_photo_ids,
                ).fetchone()["c"]
                if manual_kept:
                    click.echo(f"  Preserved {manual_kept} manual assignment(s). Use --include-manual to clear those too.")

        click.echo("Re-run match-faces to re-match.")


# ---------------------------------------------------------------------------
# export / import manual face assignments
# ---------------------------------------------------------------------------

@cli.command("export-face-assignments")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
@click.option("--output", "-o", default="face_assignments.json", help="Output JSON file path.")
def export_face_assignments(db, output):
    """Export manual face-to-person assignments to a JSON file.

    This lets you preserve UI-made assignments across database rebuilds.

    \b
    Example:
      python cli.py export-face-assignments -o my_assignments.json
    """
    import json as _json

    with PhotoDB(db) as photo_db:
        rows = photo_db.conn.execute(
            """SELECT f.id, f.bbox_top, f.bbox_right,
                      f.bbox_bottom, f.bbox_left,
                      p.name as person_name,
                      ph.filepath
               FROM faces f
               JOIN persons p ON p.id = f.person_id
               JOIN photos ph ON ph.id = f.photo_id
               WHERE f.match_source = 'manual'
               ORDER BY ph.filepath, f.id"""
        ).fetchall()

        assignments = []
        for r in rows:
            assignments.append({
                "filepath": r["filepath"],
                "person_name": r["person_name"],
                "bbox": {
                    "top": r["bbox_top"],
                    "right": r["bbox_right"],
                    "bottom": r["bbox_bottom"],
                    "left": r["bbox_left"],
                },
            })

    with open(output, "w") as f:
        _json.dump({"assignments": assignments, "count": len(assignments)}, f, indent=2)

    click.echo(f"Exported {len(assignments)} manual assignment(s) to {output}")


@cli.command("import-face-assignments")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
def import_face_assignments(input_file, db):
    """Re-apply manual face assignments from a previously exported JSON file.

    Matches by filepath + bounding box overlap (>50% IoU) so assignments
    survive face re-detection even if face IDs change.

    \b
    Example:
      python cli.py import-face-assignments face_assignments.json
    """
    import json as _json

    with open(input_file) as f:
        data = _json.load(f)

    assignments = data.get("assignments", [])
    if not assignments:
        click.echo("No assignments found in file.")
        return

    matched = 0
    skipped = 0

    with PhotoDB(db) as photo_db:
        for a in assignments:
            filepath = a.get("filepath")
            person_name = a.get("person_name")
            bbox = a.get("bbox")
            if not filepath or not person_name or not bbox:
                skipped += 1
                continue

            photo = photo_db.conn.execute(
                "SELECT id FROM photos WHERE filepath = ?", (filepath,)
            ).fetchone()
            if not photo:
                skipped += 1
                continue

            person = photo_db.get_person_by_name(person_name)
            if not person:
                pid = photo_db.add_person(person_name)
            else:
                pid = person["id"]

            faces = photo_db.conn.execute(
                """SELECT id, bbox_top, bbox_right, bbox_bottom, bbox_left
                   FROM faces WHERE photo_id = ?""",
                (photo["id"],),
            ).fetchall()

            best_face_id = None
            best_iou = 0.0
            for face in faces:
                if face["bbox_top"] is None:
                    continue
                t1, r1, b1, l1 = bbox["top"], bbox["right"], bbox["bottom"], bbox["left"]
                t2, r2, b2, l2 = face["bbox_top"], face["bbox_right"], face["bbox_bottom"], face["bbox_left"]
                inter_t, inter_l = max(t1, t2), max(l1, l2)
                inter_b, inter_r = min(b1, b2), min(r1, r2)
                inter_area = max(0, inter_b - inter_t) * max(0, inter_r - inter_l)
                union_area = (b1 - t1) * (r1 - l1) + (b2 - t2) * (r2 - l2) - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_face_id = face["id"]

            if best_face_id and best_iou > 0.5:
                photo_db.assign_face_to_person(best_face_id, pid, match_source="manual")
                matched += 1
            else:
                skipped += 1

        photo_db.conn.commit()

    click.echo(f"Imported {matched} assignment(s), skipped {skipped}.")


# ---------------------------------------------------------------------------
# tag-photo
# ---------------------------------------------------------------------------

@cli.command("tag-photo")
@click.argument("filename")
@click.argument("person_name")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
def tag_photo(filename, person_name, db):
    """Manually tag a person as appearing in a photo, bypassing face detection.

    Useful when a face is obscured, wearing a hat, or too small to auto-detect.

    \b
    Example:
      python cli.py tag-photo DSC04894.JPG "Sam"
    """
    with PhotoDB(db) as photo_db:
        # Find the photo
        photo_row = photo_db.conn.execute(
            "SELECT * FROM photos WHERE filename = ?", (filename,)
        ).fetchone()
        if not photo_row:
            click.echo(f"Photo '{filename}' not found in database.")
            return

        # Find or create the person
        person = photo_db.get_person_by_name(person_name)
        if not person:
            person_id = photo_db.add_person(person_name)
            click.echo(f"Created new person '{person_name}'.")
        else:
            person_id = person["id"]

        # Check if already tagged
        existing = photo_db.conn.execute(
            """SELECT id FROM faces
               WHERE photo_id = ? AND person_id = ? AND bbox_top IS NULL""",
            (photo_row["id"], person_id),
        ).fetchone()
        if existing:
            click.echo(f"'{person_name}' is already manually tagged in {filename}.")
            return

        # Insert a face row with no bounding box (manual tag marker)
        photo_db.conn.execute(
            """INSERT INTO faces (photo_id, person_id, bbox_top, bbox_right, bbox_bottom, bbox_left)
               VALUES (?, ?, NULL, NULL, NULL, NULL)""",
            (photo_row["id"], person_id),
        )
        photo_db.conn.commit()
        click.echo(f"✓ Tagged '{person_name}' in {filename}.")


# ---------------------------------------------------------------------------
# diagnose-photo
# ---------------------------------------------------------------------------

@cli.command("diagnose-photo")
@click.argument("filename")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
def diagnose_photo(filename, db):
    """Show face detection details and match distances for a specific photo.

    \b
    Example:
      python cli.py diagnose-photo DSC04922.JPG
    """
    import struct
    import numpy as np
    from photosearch.db import FACE_DIMENSIONS
    from photosearch.faces import MATCH_TOLERANCE

    with PhotoDB(db) as photo_db:
        # Find the photo
        row = photo_db.conn.execute(
            "SELECT * FROM photos WHERE filename = ?", (filename,)
        ).fetchone()
        if not row:
            click.echo(f"Photo '{filename}' not found in database.")
            return

        photo_id = row["id"]
        click.echo(f"\nPhoto: {filename} (id={photo_id})")

        # Faces detected in this photo
        face_rows = photo_db.conn.execute(
            """SELECT f.id, f.person_id, f.cluster_id,
                      f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left,
                      p.name as person_name
               FROM faces f
               LEFT JOIN persons p ON p.id = f.person_id
               WHERE f.photo_id = ?""",
            (photo_id,),
        ).fetchall()

        if not face_rows:
            click.echo("  No faces detected in this photo.")
            click.echo("  Try re-indexing with --faces to detect faces.")
            return

        click.echo(f"  {len(face_rows)} face(s) detected:\n")

        # Load all known person reference encodings
        ref_rows = photo_db.conn.execute(
            """SELECT fr.id, fr.person_id, p.name
               FROM face_references fr
               JOIN persons p ON p.id = fr.person_id"""
        ).fetchall()
        ref_data = []
        for ref_row in ref_rows:
            enc_row = photo_db.conn.execute(
                "SELECT encoding FROM face_ref_encodings WHERE ref_id = ?", (ref_row["id"],)
            ).fetchone()
            if enc_row:
                enc = np.array(struct.unpack(f"{FACE_DIMENSIONS}f", enc_row["encoding"]))
                ref_data.append((ref_row["person_id"], ref_row["name"], enc))

        # Thresholds for the ✓/~/✗ symbols (L2 distance on 512-dim ArcFace)
        match_thresh = MATCH_TOLERANCE          # 0.9 — definite match
        close_thresh = MATCH_TOLERANCE + 0.15   # 1.05 — borderline

        for i, face in enumerate(face_rows, 1):
            label = face["person_name"] or f"Unidentified (cluster {face['cluster_id']})"
            bbox = (face["bbox_top"], face["bbox_right"], face["bbox_bottom"], face["bbox_left"])
            size = (face["bbox_bottom"] - face["bbox_top"], face["bbox_right"] - face["bbox_left"])
            click.echo(f"  Face {i}: {label}")
            click.echo(f"    Bounding box: top={bbox[0]} right={bbox[1]} bottom={bbox[2]} left={bbox[3]}")
            click.echo(f"    Face size:    {size[0]}px × {size[1]}px")

            # Get this face's encoding
            enc_row = photo_db.conn.execute(
                "SELECT encoding FROM face_encodings WHERE face_id = ?", (face["id"],)
            ).fetchone()
            if not enc_row or not ref_data:
                click.echo(f"    (No encoding stored or no persons registered)")
                continue

            face_enc = np.array(struct.unpack(f"{FACE_DIMENSIONS}f", enc_row["encoding"]))
            click.echo(f"    Distances to known persons (ArcFace L2):")

            # One distance per person — keep the best (lowest) across all their references
            seen_persons: dict[str, float] = {}
            for _, person_name, ref_enc in ref_data:
                dist = float(np.linalg.norm(face_enc - ref_enc))
                if person_name not in seen_persons or dist < seen_persons[person_name]:
                    seen_persons[person_name] = dist

            for person_name, dist in sorted(seen_persons.items(), key=lambda x: x[1]):
                symbol = "✓" if dist <= match_thresh else ("~" if dist <= close_thresh else "✗")
                click.echo(f"      {symbol} {person_name}: {dist:.4f}")

            click.echo(
                f"    (✓ = match ≤{match_thresh} | "
                f"~ = borderline ≤{close_thresh:.2f}, try --tolerance {close_thresh:.2f} | "
                f"✗ = no match)"
            )
        click.echo("")


# ---------------------------------------------------------------------------
# find-dupes
# ---------------------------------------------------------------------------

@cli.command("find-dupes")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
@click.option("--output", "-o", default=None, help="Write a list of suggested deletions to this file (one path per line).")
@click.option("--named", is_flag=True, default=False, help="Also report _1/_2 naming-pattern duplicates even if hashes differ.")
def find_dupes(db, output, named):
    """Find duplicate photos in the database.

    \b
    Two detection methods:
      Content duplicates  — same file_hash (byte-for-byte identical, any filename).
      Named duplicates    — files like DSC06241_1.JPG alongside DSC06241.JPG
                           (--named flag; useful for camera copy artifacts).

    \b
    The suggested deletion list (--output) always keeps the file with the
    shorter/simpler name (no _N suffix) and marks the rest for deletion.
    Photos are never deleted automatically — the list is yours to review first.
    """
    import re as _re

    with PhotoDB(db) as photo_db:

        # ── 1. Content duplicates (same file_hash) ──────────────────────────
        rows = photo_db.conn.execute(
            """SELECT file_hash, COUNT(*) as cnt,
                      GROUP_CONCAT(filepath, '|') as paths,
                      GROUP_CONCAT(filename, '|') as names
               FROM photos
               WHERE file_hash IS NOT NULL
               GROUP BY file_hash
               HAVING cnt > 1
               ORDER BY cnt DESC"""
        ).fetchall()

        content_groups = []
        for row in rows:
            paths = row["paths"].split("|")
            names = row["names"].split("|")
            content_groups.append(list(zip(names, paths)))

        total_content_dupes = sum(len(g) - 1 for g in content_groups)
        click.echo(f"\nContent duplicates (same file hash): {len(content_groups)} group(s), "
                   f"{total_content_dupes} extra file(s)\n")

        to_delete = []

        for i, group in enumerate(content_groups[:50], 1):  # cap display at 50 groups
            # Keep shortest filename (usually the original), delete the rest
            group_sorted = sorted(group, key=lambda x: (len(x[0]), x[0]))
            keep_name, keep_path = group_sorted[0]
            click.echo(f"  Group {i}: keep {keep_name}")
            for name, path in group_sorted[1:]:
                click.echo(f"           del  {path}")
                to_delete.append(path)

        if len(content_groups) > 50:
            click.echo(f"  ... and {len(content_groups) - 50} more groups (use --output to see all)")

        # ── 2. Named duplicates (_1, _2 suffix pattern) ─────────────────────
        named_extras = []
        if named:
            _suffix_re = _re.compile(r'^(.+?)_(\d+)(\.[^.]+)$', _re.IGNORECASE)
            all_photos = photo_db.conn.execute(
                "SELECT filename, filepath FROM photos ORDER BY filename"
            ).fetchall()

            # Build a set of all filenames for quick lookup
            all_names = {row["filename"].lower() for row in all_photos}

            for row in all_photos:
                m = _suffix_re.match(row["filename"])
                if not m:
                    continue
                stem, _num, ext = m.groups()
                original_name = f"{stem}{ext}"
                if original_name.lower() in all_names:
                    named_extras.append((original_name, row["filename"], row["filepath"]))

            click.echo(f"\nNamed duplicates (_N suffix with matching original): {len(named_extras)} file(s)\n")
            for orig, dupe_name, dupe_path in named_extras[:100]:
                click.echo(f"  original: {orig}  →  dupe: {dupe_path}")
                if dupe_path not in to_delete:
                    to_delete.append(dupe_path)

            if len(named_extras) > 100:
                click.echo(f"  ... and {len(named_extras) - 100} more (use --output to see all)")

        # ── Summary and optional output file ────────────────────────────────
        click.echo(f"\nTotal suggested deletions: {len(to_delete)} file(s)")
        click.echo("Nothing has been deleted — review the list before acting.")

        if output:
            with open(output, "w") as f:
                for path in to_delete:
                    f.write(path + "\n")
            click.echo(f"Deletion list written to: {output}")
            click.echo("To remove after review:  xargs rm < " + output)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
def stats(db):
    """Show database statistics."""
    if not os.path.exists(db):
        click.echo(f"Database not found: {db}")
        return

    with PhotoDB(db) as photo_db:
        photo_count = photo_db.photo_count()
        clip_count = photo_db.conn.execute("SELECT COUNT(*) as c FROM clip_embeddings").fetchone()["c"]
        face_count = photo_db.conn.execute("SELECT COUNT(*) as c FROM faces").fetchone()["c"]
        person_count = photo_db.conn.execute("SELECT COUNT(*) as c FROM persons").fetchone()["c"]
        unmatched = photo_db.conn.execute(
            "SELECT COUNT(*) as c FROM faces WHERE person_id IS NULL"
        ).fetchone()["c"]
        described = photo_db.conn.execute(
            "SELECT COUNT(*) as c FROM photos WHERE description IS NOT NULL"
        ).fetchone()["c"]
        scored = photo_db.conn.execute(
            "SELECT COUNT(*) as c FROM photos WHERE aesthetic_score IS NOT NULL"
        ).fetchone()["c"]

        click.echo(f"Database:        {db}")
        click.echo(f"Photos indexed:  {photo_count}")
        click.echo(f"CLIP embedded:   {clip_count}/{photo_count}")
        click.echo(f"Faces detected:  {face_count} ({unmatched} unmatched)")
        click.echo(f"Persons named:   {person_count}")
        click.echo(f"Descriptions:    {described}/{photo_count}")
        click.echo(f"Quality scored:  {scored}/{photo_count}")

        if scored > 0:
            score_stats = photo_db.conn.execute(
                """SELECT MIN(aesthetic_score) as min_s, MAX(aesthetic_score) as max_s,
                          AVG(aesthetic_score) as avg_s
                   FROM photos WHERE aesthetic_score IS NOT NULL"""
            ).fetchone()
            click.echo(f"  Score range:   {score_stats['min_s']:.2f} – {score_stats['max_s']:.2f} "
                       f"(mean: {score_stats['avg_s']:.2f})")

        if photo_count > 0:
            rows = photo_db.conn.execute(
                "SELECT filename, date_taken, camera_model FROM photos LIMIT 5"
            ).fetchall()
            click.echo(f"\nSample photos:")
            for row in rows:
                click.echo(f"  {row['filename']}  {row['date_taken'] or ''}  {row['camera_model'] or ''}")


# ---------------------------------------------------------------------------
# show-descriptions
# ---------------------------------------------------------------------------

@cli.command("show-descriptions")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
@click.argument("filename", required=False)
def show_descriptions(db, filename):
    """Show LLaVA-generated descriptions for photos.

    If FILENAME is given, show that photo's description. Otherwise show all.

    \b
    Examples:
      python cli.py show-descriptions
      python cli.py show-descriptions DSC04894.JPG
    """
    with PhotoDB(db) as photo_db:
        if filename:
            row = photo_db.conn.execute(
                "SELECT filename, description FROM photos WHERE filename = ?", (filename,)
            ).fetchone()
            if not row:
                click.echo(f"Photo '{filename}' not found.")
                return
            desc = row["description"] or "(no description)"
            click.echo(f"\n{row['filename']}:\n  {desc}\n")
        else:
            rows = photo_db.conn.execute(
                "SELECT filename, description FROM photos ORDER BY filename"
            ).fetchall()
            for row in rows:
                desc = row["description"] or "(no description)"
                click.echo(f"\n{row['filename']}:")
                click.echo(f"  {desc}")
            click.echo()


# ---------------------------------------------------------------------------
# show-quality
# ---------------------------------------------------------------------------

@cli.command("show-quality")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
@click.option("--sort", "sort_order", type=click.Choice(["score", "name"]), default="score",
              help="Sort by aesthetic score (descending) or filename.")
@click.option("--min", "min_score", type=float, default=None, help="Only show photos above this score.")
@click.option("--detail", is_flag=True, help="Show concept breakdown and critique for each photo.")
@click.argument("directory", required=False, default=None)
def show_quality(db, sort_order, min_score, detail, directory):
    """Show aesthetic quality scores for photos.

    Optionally pass a DIRECTORY to show only photos indexed from that path.

    \b
    Examples:
      python cli.py show-quality
      python cli.py show-quality ../sample
      python cli.py show-quality --sort name
      python cli.py show-quality --min 5.0
      python cli.py show-quality --detail ../sample
    """
    with PhotoDB(db) as photo_db:
        if sort_order == "score":
            order = "aesthetic_score DESC"
        else:
            order = "filename"

        cols = "filename, aesthetic_score"
        if detail:
            cols = "filename, aesthetic_score, aesthetic_concepts, aesthetic_critique, tags"

        # Build WHERE clause
        conditions = []
        params = []

        if directory:
            resolved_dir = str(Path(directory).resolve())
            # Match both absolute and relative paths
            rel_dir = photo_db.relative_filepath(resolved_dir)
            conditions.append("(filepath LIKE ? OR filepath LIKE ?)")
            params.extend([resolved_dir + "/%", rel_dir + "/%"])

        if min_score is not None:
            conditions.append("aesthetic_score IS NOT NULL AND aesthetic_score >= ?")
            params.append(min_score)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        rows = photo_db.conn.execute(
            f"SELECT {cols} FROM photos {where} ORDER BY {order}",
            params,
        ).fetchall()

        if not rows:
            click.echo("No photos with quality scores found. Run 'index --quality' first.")
            return

        scored = [r for r in rows if r["aesthetic_score"] is not None]
        unscored = [r for r in rows if r["aesthetic_score"] is None]

        if detail:
            for row in scored:
                score = row["aesthetic_score"]
                bar = "█" * int(score) + "░" * (10 - int(score))
                click.echo(f"\n{'─' * 70}")
                click.echo(f"  {row['filename']}  —  {score:.2f}  {bar}")

                # Concept breakdown
                if row["aesthetic_concepts"]:
                    try:
                        concepts = json.loads(row["aesthetic_concepts"])
                        strengths = concepts.get("strengths", [])
                        weaknesses = concepts.get("weaknesses", [])
                        if strengths:
                            click.echo(f"  Strengths:  {', '.join(strengths)}")
                        if weaknesses:
                            click.echo(f"  Weaknesses: {', '.join(weaknesses)}")
                    except (json.JSONDecodeError, TypeError):
                        pass

                # LLM critique
                if row["aesthetic_critique"]:
                    click.echo(f"  Critique:   {row['aesthetic_critique']}")

                # Semantic tags
                if row["tags"]:
                    try:
                        tags = json.loads(row["tags"])
                        click.echo(f"  Tags:       {', '.join(tags)}")
                    except (json.JSONDecodeError, TypeError):
                        pass

            click.echo(f"{'─' * 70}")
        else:
            click.echo(f"\n{'Filename':<55} {'Score'}")
            click.echo("-" * 65)
            for row in scored:
                score = row["aesthetic_score"]
                bar = "█" * int(score) + "░" * (10 - int(score))
                click.echo(f"{row['filename']:<55} {score:>5.2f}  {bar}")

        if unscored:
            click.echo(f"\n({len(unscored)} photo(s) not yet scored)")

        # Summary stats
        if scored:
            scores = [r["aesthetic_score"] for r in scored]
            click.echo(f"\n{len(scored)} scored — range: {min(scores):.2f}–{max(scores):.2f}, "
                        f"mean: {sum(scores)/len(scores):.2f}")

        if scored:
            scores = [r["aesthetic_score"] for r in scored]
            click.echo(f"\n{len(scored)} photos scored: "
                       f"{min(scores):.2f} – {max(scores):.2f} "
                       f"(mean: {sum(scores)/len(scores):.2f})")


# ---------------------------------------------------------------------------
# serve (web UI)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
@click.option("--host", default="0.0.0.0", help="Host to bind to.")
@click.option("--port", default=8000, help="Port to listen on.")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development.")
def serve(db, host, port, reload):
    """Launch the web UI for browsing and searching photos.

    \b
    Example:
      python cli.py serve
      python cli.py serve --port 3000
      python cli.py serve --db /path/to/photo_index.db
    """
    import uvicorn
    os.environ["PHOTOSEARCH_DB"] = db
    click.echo(f"Starting photo search UI at http://{host}:{port}")
    click.echo(f"Database: {db}")
    click.echo("Press Ctrl+C to stop.\n")
    uvicorn.run(
        "photosearch.web:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# ---------------------------------------------------------------------------
# remap-paths
# ---------------------------------------------------------------------------

@cli.command("remap-paths")
@click.argument("old_prefix")
@click.argument("new_prefix")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
@click.option("--dry-run", is_flag=True, help="Show what would change without modifying the database.")
def remap_paths(old_prefix, new_prefix, db, dry_run):
    """Rewrite stored filepaths when photos move to a new location.

    Replaces OLD_PREFIX with NEW_PREFIX in all photo filepaths.
    Essential when moving a Mac-built database to a NAS where photos
    are mounted at a different path.

    \b
    Examples:
      # Moving from Mac to Docker container:
      python cli.py remap-paths /Users/matt/Pictures /photos

      # Preview changes without modifying:
      python cli.py remap-paths /Users/matt/Pictures /photos --dry-run

      # Moving to a UGREEN NAS volume:
      python cli.py remap-paths /Users/matt/Pictures /volume1/photos
    """
    with PhotoDB(db) as photo_db:
        rows = photo_db.conn.execute(
            "SELECT id, filepath, raw_filepath FROM photos WHERE filepath LIKE ?",
            (f"{old_prefix}%",),
        ).fetchall()

        if not rows:
            click.echo(f"No photos found with prefix '{old_prefix}'.")
            return

        click.echo(f"Found {len(rows)} photo(s) matching '{old_prefix}'.")

        if dry_run:
            for row in rows[:5]:
                old = row["filepath"]
                new = new_prefix + old[len(old_prefix):]
                click.echo(f"  {old}")
                click.echo(f"  → {new}")
            if len(rows) > 5:
                click.echo(f"  ... and {len(rows) - 5} more")
            click.echo("\nNo changes made (dry run).")
            return

        updated = photo_db.remap_paths(old_prefix, new_prefix)
        click.echo(f"Updated {updated} filepath(s): '{old_prefix}' → '{new_prefix}'.")


@cli.command("set-photo-root")
@click.argument("root_path")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
def set_photo_root(root_path, db):
    """Set or change the photo root directory.

    The photo root is the base directory that stored file paths are relative to.
    When you move the database to a different machine, just update the photo root
    to point to where photos live on that machine.

    \b
    Examples:
      # On your Mac:
      python cli.py set-photo-root /Volumes/personal_folder/Photos

      # On the NAS:
      python cli.py set-photo-root /Photos
    """
    with PhotoDB(db) as photo_db:
        old_root = photo_db.photo_root
        photo_db.set_photo_root(root_path)
        if old_root:
            click.echo(f"Updated photo root: {old_root} → {photo_db.photo_root}")
        else:
            click.echo(f"Set photo root: {photo_db.photo_root}")


# ---------------------------------------------------------------------------
# import-db
# ---------------------------------------------------------------------------

@cli.command("import-db")
@click.argument("source_db", type=click.Path(exists=True))
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Destination database path.")
@click.option("--remap", nargs=2, help="Remap paths: OLD_PREFIX NEW_PREFIX")
def import_db(source_db, db, remap):
    """Import a database built on another machine.

    Copies the source database to the destination path. If --remap is given,
    also rewrites filepaths. This is the recommended way to deploy a
    Mac-built database to a NAS.

    \b
    Examples:
      # Copy and remap in one step:
      python cli.py import-db /mnt/usb/photo_index.db \\
          --db /data/photo_index.db \\
          --remap /Users/matt/Pictures /photos
    """
    import shutil

    click.echo(f"Copying {source_db} → {db}")
    shutil.copy2(source_db, db)

    if remap:
        old_prefix, new_prefix = remap
        # Invoke the remap logic directly
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(remap_paths, [old_prefix, new_prefix, "--db", db])
        click.echo(result.output)
    else:
        click.echo("Done. Run 'remap-paths' if photo paths differ on this machine.")


# ---------------------------------------------------------------------------
# review — shoot culling
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("photo_dir", type=click.Path(exists=True))
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
@click.option("--target-pct", default=10.0, help="Target percentage of photos to select (default 10).")
@click.option("--threshold", default=0.0, help="Clustering distance threshold (0 = adaptive, lower = tighter clusters).")
@click.option("--export", type=click.Path(), default=None,
              help="If set, copy selected JPGs to this directory.")
@click.option("--export-raw", type=click.Path(), default=None,
              help="If set, copy selected ARW files to this directory.")
@click.option("--list-only", is_flag=True, help="Just print selected file paths, don't copy.")
def review(photo_dir, db, target_pct, threshold, export, export_raw, list_only):
    """Review a shoot folder — select the best, most representative photos.

    Clusters visually similar photos using CLIP embeddings and picks the
    highest-quality photo from each cluster. Aims for ~10% of the folder.

    \b
    Examples:
      # Review and see the picks:
      python cli.py review ../Photos/2026-03-13

      # Review and copy selected JPGs to a new folder:
      python cli.py review ../Photos/2026-03-13 --export ~/Desktop/selects

      # Also grab the ARW raw files:
      python cli.py review ../Photos/2026-03-13 \\
          --export ~/Desktop/selects \\
          --export-raw ~/Desktop/selects/raw
    """
    from photosearch.cull import select_best_photos, save_selections

    with PhotoDB(db) as photo_db:
        click.echo(f"Analyzing {photo_dir}...")
        selections = select_best_photos(
            photo_db, photo_dir,
            target_pct=target_pct / 100.0,
            distance_threshold=threshold,
        )

        if not selections:
            click.echo("No indexed photos found in this directory.")
            click.echo("Run 'index' first: python cli.py index " + photo_dir)
            return

        # Save selections
        save_selections(photo_db, str(Path(photo_dir).resolve()), selections)

        total = len(selections)
        selected = [p for p in selections if p["selected"]]
        n_clusters = len(set(p.get("cluster_id") for p in selections if p.get("cluster_id") is not None))

        click.echo(f"\n{total} photos, {n_clusters} clusters, {len(selected)} selected ({len(selected)*100//total}%)")
        click.echo()

        # Print selected photos
        for p in selected:
            score = p.get("aesthetic_score")
            score_str = f" [{score:.1f}]" if score else ""
            raw_str = " +RAW" if p.get("raw_filepath") else ""
            click.echo(f"  \u2713 {p['filename']}{score_str} (cluster {p.get('cluster_id', '?')}){raw_str}")

        if list_only:
            click.echo("\n--- Selected file paths ---")
            for p in selected:
                abs_path = photo_db.resolve_filepath(p["filepath"])
                click.echo(abs_path)
            return

        # Copy files if requested
        if export:
            import shutil
            export_dir = Path(export)
            export_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            for p in selected:
                src = photo_db.resolve_filepath(p["filepath"])
                if src and os.path.exists(src):
                    shutil.copy2(src, export_dir / Path(src).name)
                    copied += 1
            click.echo(f"\nCopied {copied} JPG(s) to {export_dir}")

        if export_raw:
            import shutil
            raw_dir = Path(export_raw)
            raw_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            for p in selected:
                raw_path = p.get("raw_filepath")
                if raw_path:
                    abs_raw = photo_db.resolve_filepath(raw_path)
                    if abs_raw and os.path.exists(abs_raw):
                        shutil.copy2(abs_raw, raw_dir / Path(abs_raw).name)
                        copied += 1
            click.echo(f"Copied {copied} ARW file(s) to {raw_dir}")

        if not export and not export_raw and not list_only:
            click.echo(f"\nSelections saved. View in the web UI at /review")
            click.echo(f"Or use --export <dir> to copy selected JPGs.")


# ---------------------------------------------------------------------------
# verify — hallucination detection
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Path to the SQLite database file.")
@click.option("--threshold", default=0.18, show_default=True,
              help="CLIP similarity threshold. Nouns below this are flagged.")
@click.option("--model", default="llava", show_default=True, help="Ollama model used to regenerate descriptions.")
@click.option("--verify-model", default="minicpm-v", show_default=True,
              help="Ollama vision model for verification (should differ from --model to avoid confirmation bias).")
@click.option("--force", is_flag=True, help="Re-verify even previously verified photos.")
@click.option("--no-regenerate", is_flag=True, help="Flag hallucinations but don't auto-regenerate.")
@click.option("--limit", default=0, help="Max photos to verify (0 = all).")
@click.option("--photo", default=None, type=click.Path(), help="Verify a specific photo by file path.")
@click.option("--llm-all", is_flag=True, help="Send ALL nouns to LLM, not just CLIP-flagged ones (slower but thorough).")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed CLIP scores for every noun/tag.")
def verify(db, threshold, model, verify_model, force, no_regenerate, limit, photo, llm_all, verbose):
    """Verify photo descriptions and tags for hallucinations.

    Two-pass approach:
      1. CLIP check — flags nouns in descriptions that don't match the photo
      2. LLM verify — sends the photo to a DIFFERENT vision model to cross-check

    Uses a separate model for verification (default: minicpm-v) to avoid the
    problem where the same model confirms its own hallucinations.

    Confirmed hallucinations are automatically regenerated unless --no-regenerate.

    \b
    Examples:
      # Verify all unverified photos:
      python cli.py verify

      # Re-verify everything:
      python cli.py verify --force

      # Just flag, don't regenerate:
      python cli.py verify --no-regenerate

      # Verify with stricter threshold:
      python cli.py verify --threshold 0.20

      # Verify a single photo:
      python cli.py verify --photo /path/to/DSC00123.JPG
    """
    from photosearch.db import PhotoDB
    from photosearch.verify import verify_photos
    from photosearch.describe import check_available

    check_available(verify_model)
    if not no_regenerate:
        check_available(model)

    # Enable verbose logging for verify module (auto-enabled for --photo)
    if verbose or photo:
        verify_logger = logging.getLogger("photosearch.verify")
        verify_logger.setLevel(logging.INFO)
        verify_logger.propagate = False  # prevent duplicate output from root logger
        if not verify_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            verify_logger.addHandler(handler)

    photo_db = PhotoDB(db)

    # Target a specific photo by filepath
    photos = None
    if photo:
        photo_path = os.path.abspath(photo)
        # Try absolute path first, then the path as given (may be relative in DB)
        row = photo_db.conn.execute(
            "SELECT * FROM photos WHERE filepath = ?", (photo_path,)
        ).fetchone()
        if row is None:
            row = photo_db.conn.execute(
                "SELECT * FROM photos WHERE filepath = ?", (photo,)
            ).fetchone()
        if row is None:
            # Also try matching just the filename or tail of the path
            row = photo_db.conn.execute(
                "SELECT * FROM photos WHERE filepath LIKE ?", (f"%{os.path.basename(photo)}",)
            ).fetchone()
        if row is None:
            click.echo(f"Error: photo not found in database: {photo}", err=True)
            raise SystemExit(1)
        row = dict(row)
        if row.get("description") is None and row.get("tags") is None:
            click.echo(f"Error: photo has no description or tags to verify: {photo_path}", err=True)
            raise SystemExit(1)
        photos = [row]
        force = True  # always verify when targeting a specific photo
        click.echo(f"Photo: {row['filepath']}")
        click.echo(f"Description: {row.get('description', '(none)')}")
        click.echo(f"Tags: {row.get('tags', '(none)')}")
        click.echo()
    elif limit > 0:
        if force:
            rows = photo_db.conn.execute(
                "SELECT * FROM photos WHERE description IS NOT NULL OR tags IS NOT NULL LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = photo_db.conn.execute(
                """SELECT * FROM photos
                   WHERE (description IS NOT NULL OR tags IS NOT NULL)
                   AND verified_at IS NULL LIMIT ?""",
                (limit,),
            ).fetchall()
        photos = [dict(r) for r in rows]

    # When targeting a single photo, use LLM-all mode (verify every noun, not just CLIP-flagged)
    use_llm_all = llm_all or (photo is not None)
    click.echo(f"Verifying (verify-model={verify_model}, regen-model={model}, threshold={threshold}{', llm-all=true' if use_llm_all else ''})...")
    stats = verify_photos(
        photo_db,
        photos=photos,
        clip_threshold=threshold,
        verify_model=verify_model,
        regen_model=model,
        auto_regenerate=not no_regenerate,
        force=force,
        llm_all=use_llm_all,
    )

    click.echo(f"\nDone. {stats['checked']}/{stats['total']} checked:")
    click.echo(f"  Passed:      {stats['passed']}")
    click.echo(f"  Failed:      {stats['failed']}")
    click.echo(f"  Regenerated: {stats['regenerated']}")


@cli.command()
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB", help="Database path.")
@click.option("--time-window", default=5.0, type=float,
              help="Max seconds between shots to consider stacking (default: 5).")
@click.option("--clip-threshold", default=0.05, type=float,
              help="Max CLIP cosine distance for visual similarity (default: 0.05).")
@click.option("--directory", default=None, type=click.Path(),
              help="Restrict to photos in this directory.")
@click.option("--dry-run", is_flag=True,
              help="Show what would be stacked without saving to DB.")
@click.option("--clear", is_flag=True,
              help="Clear all existing stacks before detecting.")
def stack(db, time_window, clip_threshold, directory, dry_run, clear):
    """Detect and create photo stacks (burst/bracket groups).

    Finds photos taken within --time-window seconds AND with CLIP cosine
    distance < --clip-threshold, groups them into stacks, and picks the
    best photo (by aesthetic score) as the "top" of each stack.
    """
    from photosearch.db import PhotoDB
    from photosearch.stacking import run_stacking

    photo_db = PhotoDB(db)

    if clear and not dry_run:
        photo_db.clear_stacks()
        click.echo("Cleared all existing stacks.")

    if directory:
        directory = os.path.abspath(directory)

    click.echo(f"Detecting stacks (window={time_window}s, clip_threshold={clip_threshold}"
               f"{', directory=' + directory if directory else ''}"
               f"{', dry-run' if dry_run else ''})...")

    stacks = run_stacking(
        photo_db,
        time_window_sec=time_window,
        clip_threshold=clip_threshold,
        directory=directory,
        dry_run=dry_run,
    )

    if not stacks:
        click.echo("No stacks detected.")
        return

    # Print summary
    total_stacked = sum(len(s) for s in stacks)
    click.echo(f"\n{'Would create' if dry_run else 'Created'} {len(stacks)} stacks "
               f"({total_stacked} photos stacked):")
    for i, s in enumerate(stacks[:20]):
        # Fetch filenames for display
        placeholders = ",".join("?" * len(s))
        rows = photo_db.conn.execute(
            f"SELECT id, filename, aesthetic_score FROM photos WHERE id IN ({placeholders})",
            s,
        ).fetchall()
        by_id = {r["id"]: dict(r) for r in rows}
        top = by_id.get(s[0])
        top_name = top["filename"] if top else f"id={s[0]}"
        top_score = f" (score: {top['aesthetic_score']:.1f})" if top and top["aesthetic_score"] else ""
        rest = [by_id.get(pid, {}).get("filename", f"id={pid}") for pid in s[1:]]
        rest_names = ", ".join(rest[:5])
        ellipsis = "..." if len(rest) > 5 else ""
        click.echo(f"  Stack {i+1}: {top_name}{top_score} + {len(rest)} more [{rest_names}{ellipsis}]")
    if len(stacks) > 20:
        click.echo(f"  ... and {len(stacks) - 20} more stacks")


if __name__ == "__main__":
    cli()
