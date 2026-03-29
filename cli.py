#!/usr/bin/env python3
"""CLI entry point for local-photo-search."""

import warnings
# Suppress FutureWarning from insightface/utils/face_align.py (scikit-image API change).
# Must be set before any imports that might trigger it.
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

import json
import os

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
@click.option("--db", default="photo_index.db", help="Path to the SQLite database file.")
@click.option("--batch-size", default=8, help="Batch size for CLIP embedding.")
@click.option("--no-clip", is_flag=True, help="Skip CLIP embedding generation.")
@click.option("--no-colors", is_flag=True, help="Skip dominant color extraction.")
@click.option("--faces", is_flag=True, help="Detect and encode faces.")
@click.option("--force-faces", is_flag=True, help="Clear all face data and re-run detection on every photo.")
def index(photo_dir, db, batch_size, no_clip, no_colors, faces, force_faces):
    """Index a directory of photos."""
    index_directory(
        photo_dir=photo_dir,
        db_path=db,
        batch_size=batch_size,
        enable_clip=not no_clip,
        enable_colors=not no_colors,
        enable_faces=faces or force_faces,
        force_faces=force_faces,
    )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--query", "-q", help="Semantic search query (natural language).")
@click.option("--person", "-p", help="Search by person name.")
@click.option("--place", help="Search by place name.")
@click.option("--color", "-c", help="Search by dominant color (name or #hex).")
@click.option("--face", type=click.Path(exists=True), help="Search by reference face image.")
@click.option("--db", default="photo_index.db", help="Path to the SQLite database file.")
@click.option("--limit", "-n", default=10, help="Max number of results.")
@click.option("--results-dir", default="results", help="Base directory for result subfolders.")
@click.option("--no-results", is_flag=True, help="Don't write a results folder.")
@click.option("--json-output", is_flag=True, help="Output results as JSON.")
def search(query, person, place, color, face, db, limit, results_dir, no_results, json_output):
    """Search indexed photos."""
    if not any([query, person, place, color, face]):
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
        )

        if not results:
            click.echo("No matching photos found.")
            return

        if json_output:
            for r in results:
                for k in list(r.keys()):
                    if r[k] is None:
                        del r[k]
            click.echo(json.dumps(results, indent=2))
        else:
            click.echo(f"\nFound {len(results)} matching photos:\n")
            click.echo(f"{'#':<4} {'Filename':<20} {'Date':<22} {'Score':<8} {'Colors'}")
            click.echo("-" * 80)
            for i, r in enumerate(results, 1):
                filename = r.get("filename", "?")
                date = r.get("date_taken", "")[:19] if r.get("date_taken") else ""
                score = f"{r.get('score', 0):.3f}" if "score" in r else ""
                colors = ""
                if r.get("dominant_colors"):
                    try:
                        color_list = json.loads(r["dominant_colors"])
                        colors = " ".join(color_list[:3])
                    except (json.JSONDecodeError, TypeError):
                        pass
                click.echo(f"{i:<4} {filename:<20} {date:<22} {score:<8} {colors}")

        if not no_results and results:
            subfolder = make_results_subdir(
                results_dir,
                {"query": query, "color": color, "place": place, "person": person},
            )
            result_path = symlink_results(results, output_dir=subfolder)
            click.echo(f"\nResults folder: {result_path}")


# ---------------------------------------------------------------------------
# add-person
# ---------------------------------------------------------------------------

@cli.command("add-person")
@click.argument("name")
@click.option("--photo", "photos", multiple=True, type=click.Path(exists=True),
              help="Reference photo(s) containing this person's face. Can be repeated.")
@click.option("--db", default="photo_index.db", help="Path to the SQLite database file.")
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
# match-faces
# ---------------------------------------------------------------------------

@cli.command("match-faces")
@click.option("--db", default="photo_index.db", help="Path to the SQLite database file.")
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
@click.option("--db", default="photo_index.db", help="Path to the SQLite database file.")
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
@click.option("--db", default="photo_index.db", help="Path to the SQLite database file.")
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
@click.option("--db", default="photo_index.db", help="Path to the SQLite database file.")
def correct_face(filename, face_number, correct_person, db):
    """Correct a face match in a photo.

    FACE_NUMBER is the face index shown by 'diagnose-photo' (1-based).
    Use CORRECT_PERSON as 'unknown' to clear a wrong match without reassigning.

    \b
    Examples:
      # Reassign face 1 in DSC04922 from Ellie to Calvin
      python cli.py correct-face DSC04922.JPG 1 "Calvin"

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
# tag-photo
# ---------------------------------------------------------------------------

@cli.command("tag-photo")
@click.argument("filename")
@click.argument("person_name")
@click.option("--db", default="photo_index.db", help="Path to the SQLite database file.")
def tag_photo(filename, person_name, db):
    """Manually tag a person as appearing in a photo, bypassing face detection.

    Useful when a face is obscured, wearing a hat, or too small to auto-detect.

    \b
    Example:
      python cli.py tag-photo DSC04894.JPG "Nicole"
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
@click.option("--db", default="photo_index.db", help="Path to the SQLite database file.")
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
# stats
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--db", default="photo_index.db", help="Path to the SQLite database file.")
def stats(db):
    """Show database statistics."""
    if not os.path.exists(db):
        click.echo(f"Database not found: {db}")
        return

    with PhotoDB(db) as photo_db:
        photo_count = photo_db.photo_count()
        face_count = photo_db.conn.execute("SELECT COUNT(*) as c FROM faces").fetchone()["c"]
        person_count = photo_db.conn.execute("SELECT COUNT(*) as c FROM persons").fetchone()["c"]
        unmatched = photo_db.conn.execute(
            "SELECT COUNT(*) as c FROM faces WHERE person_id IS NULL"
        ).fetchone()["c"]

        click.echo(f"Database:        {db}")
        click.echo(f"Photos indexed:  {photo_count}")
        click.echo(f"Faces detected:  {face_count} ({unmatched} unmatched)")
        click.echo(f"Persons named:   {person_count}")

        if photo_count > 0:
            rows = photo_db.conn.execute(
                "SELECT filename, date_taken, camera_model FROM photos LIMIT 5"
            ).fetchall()
            click.echo(f"\nSample photos:")
            for row in rows:
                click.echo(f"  {row['filename']}  {row['date_taken'] or ''}  {row['camera_model'] or ''}")


if __name__ == "__main__":
    cli()
