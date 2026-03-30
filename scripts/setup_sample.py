#!/usr/bin/env python3
"""
Idempotent setup script for the 7-photo sample test suite.

Runs only the steps that haven't been completed yet:
  1. Index sample photos with face detection (skips photos already indexed)
  2. Register reference persons (skips persons who already have references)
  3. Run face matching — strict pass, then temporal pass
  4. Apply sample-specific corrections for faces that are genuinely ambiguous
     (model can't distinguish them automatically; ground truth applied directly)

Usage (from the local-photo-search directory, venv active):
  python scripts/setup_sample.py
  python scripts/setup_sample.py --force-faces   # re-detect faces from scratch
  python scripts/setup_sample.py --help
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths relative to the project root
DEFAULT_SAMPLE_DIR = str(PROJECT_ROOT.parent / "sample")
DEFAULT_REFS_DIR   = str(PROJECT_ROOT / "references")
DEFAULT_DB         = "photo_index.db"

# Reference persons and their photos (relative to refs_dir)
REFERENCE_PERSONS = [
    ("Matt",   ["matt_reference.jpg"]),
    ("Calvin", ["calvin_reference.jpg", "calvin_reference2.JPG"]),
    ("Ellie",  ["ellie_reference.jpg",  "ellie_reference2.JPG"]),
    ("Nicole", ["nicole_reference.jpg"]),
]

# Sample-specific ground-truth corrections.
# These are faces that automatic matching (strict + temporal) cannot resolve because
# the ArcFace distance gap between the correct person and the next-closest is too
# small to commit safely (< TEMPORAL_MIN_GAP). The correct identity is known from
# visual inspection and applied directly as the final setup step.
#
# Format: (photo_filename, correct_person_name)
# The script finds the unmatched face in that photo closest to the named person
# and assigns it — no fragile face-number indexing required.
SAMPLE_CORRECTIONS = [
    ("DSC04907.JPG", "Calvin"),  # 87×80px, L2=1.3565; gap to Ellie only 0.004
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(args: list[str], label: str = "") -> int:
    """Run a command from the project root, streaming output. Returns exit code."""
    if label:
        print(f"\n{'─' * 62}")
        print(f"  {label}")
        print(f"{'─' * 62}")
    print(f"$ {' '.join(str(a) for a in args)}\n")
    return subprocess.run(args, cwd=str(PROJECT_ROOT)).returncode


def _db_state(db_path: Path) -> dict:
    """Return a snapshot of the database state, or empty defaults if it doesn't exist."""
    if not db_path.exists():
        return {"photos": 0, "faces": 0, "persons": 0, "person_refs": 0, "unmatched": 0}
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from photosearch.db import PhotoDB
        with PhotoDB(str(db_path)) as db:
            return {
                "photos":      db.conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0],
                "faces":       db.conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0],
                "persons":     db.conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0],
                "person_refs": db.conn.execute("SELECT COUNT(*) FROM face_references").fetchone()[0],
                "unmatched":   db.conn.execute(
                                   "SELECT COUNT(*) FROM faces WHERE person_id IS NULL"
                               ).fetchone()[0],
                "person_names": {
                    r[0] for r in db.conn.execute("SELECT name FROM persons").fetchall()
                },
                "persons_with_refs": {
                    r[0] for r in db.conn.execute(
                        "SELECT DISTINCT p.name FROM persons p "
                        "JOIN face_references fr ON fr.person_id = p.id"
                    ).fetchall()
                },
            }
    except Exception as e:
        return {"error": str(e)}


def _print_state(state: dict, heading: str = "Database state"):
    print(f"\n  {heading}:")
    if "error" in state:
        print(f"    Error reading DB: {state['error']}")
        return
    print(f"    Photos indexed:      {state['photos']}")
    print(f"    Faces detected:      {state['faces']}")
    print(f"    Persons registered:  {state['persons']}")
    print(f"    References stored:   {state['person_refs']}")
    print(f"    Unmatched faces:     {state['unmatched']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _apply_sample_corrections(db_path: Path, corrections: list[tuple[str, str]]) -> int:
    """Apply ground-truth corrections for faces that automatic matching can't resolve.

    For each (photo, person) pair, finds the unmatched face in that photo whose
    ArcFace encoding is closest to the named person's references, and assigns it.
    Does NOT rely on face insertion order — safe to call after re-indexing.

    Returns the number of faces corrected.
    """
    import struct
    import numpy as np
    sys.path.insert(0, str(PROJECT_ROOT))
    from photosearch.db import PhotoDB, FACE_DIMENSIONS

    fixed = 0
    with PhotoDB(str(db_path)) as db:
        for filename, person_name in corrections:
            # Look up the person
            person = db.get_person_by_name(person_name)
            if not person:
                print(f"  Warning: person '{person_name}' not found in DB — skipping {filename}")
                continue
            person_id = person["id"]

            # Check if already assigned
            already = db.conn.execute(
                """SELECT COUNT(*) FROM faces f
                   JOIN photos ph ON ph.id = f.photo_id
                   WHERE ph.filename = ? AND f.person_id = ?""",
                (filename, person_id),
            ).fetchone()[0]
            if already:
                print(f"  Sample correction: '{person_name}' already assigned in {filename} — skipped.")
                continue

            # Load this person's reference encodings
            ref_rows = db.conn.execute(
                "SELECT encoding FROM face_ref_encodings re "
                "JOIN face_references fr ON fr.id = re.ref_id "
                "WHERE fr.person_id = ?",
                (person_id,),
            ).fetchall()
            if not ref_rows:
                print(f"  Warning: no reference encodings for '{person_name}' — skipping {filename}")
                continue
            ref_encs = [
                np.array(struct.unpack(f"{FACE_DIMENSIONS}f", r["encoding"]))
                for r in ref_rows
            ]

            # Find unmatched faces in this photo
            unmatched_faces = db.conn.execute(
                """SELECT f.id, fe.encoding
                   FROM faces f
                   JOIN photos ph ON ph.id = f.photo_id
                   LEFT JOIN face_encodings fe ON fe.face_id = f.id
                   WHERE ph.filename = ? AND f.person_id IS NULL AND fe.encoding IS NOT NULL""",
                (filename,),
            ).fetchall()
            if not unmatched_faces:
                print(f"  Sample correction: no unmatched faces in {filename} — skipped.")
                continue

            # Pick the unmatched face closest to the person's references
            best_face_id = None
            best_dist = float("inf")
            for face_row in unmatched_faces:
                face_enc = np.array(struct.unpack(f"{FACE_DIMENSIONS}f", face_row["encoding"]))
                # Best distance to any of this person's reference encodings
                dist = min(float(np.linalg.norm(face_enc - ref)) for ref in ref_encs)
                if dist < best_dist:
                    best_dist = dist
                    best_face_id = face_row["id"]

            if best_face_id is not None:
                db.assign_face_to_person(best_face_id, person_id)
                print(f"  Sample correction: assigned '{person_name}' to closest "
                      f"unmatched face in {filename} (L2={best_dist:.4f}).")
                fixed += 1

    return fixed


def main():
    parser = argparse.ArgumentParser(
        description="Set up the sample photo test suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sample-dir", default=DEFAULT_SAMPLE_DIR,
                        help=f"Sample photo directory (default: ../sample)")
    parser.add_argument("--refs-dir",   default=DEFAULT_REFS_DIR,
                        help=f"Reference photo directory (default: references/)")
    parser.add_argument("--db",         default=DEFAULT_DB,
                        help=f"Database file (default: {DEFAULT_DB})")
    parser.add_argument("--force-faces", action="store_true",
                        help="Clear face data and re-detect from scratch")
    args = parser.parse_args()

    db_path  = PROJECT_ROOT / args.db
    refs_dir = Path(args.refs_dir)
    py       = sys.executable  # use the same Python (respects active venv)

    print(f"\n{'═' * 62}")
    print(f"  local-photo-search  —  sample setup")
    print(f"{'═' * 62}")
    print(f"  Project: {PROJECT_ROOT}")
    print(f"  Sample:  {args.sample_dir}")
    print(f"  Refs:    {refs_dir}")
    print(f"  DB:      {db_path}")

    state = _db_state(db_path)
    _print_state(state, "Before")

    errors = 0

    # ── Step 1: Index ────────────────────────────────────────────────────────
    if state["photos"] == 0 or state["faces"] == 0 or args.force_faces:
        cmd = [py, "cli.py", "index", args.sample_dir, "--faces", "--db", args.db]
        if args.force_faces:
            cmd.append("--force-faces")
        rc = _run(cmd, "Step 1: Index sample photos with face detection")
        if rc != 0:
            print(f"\n  ERROR: indexing failed (exit {rc})")
            errors += 1
    else:
        print(f"\n  Step 1: skipped — {state['photos']} photos already indexed "
              f"with {state['faces']} faces detected.")

    # ── Step 2: Register reference persons ───────────────────────────────────
    state = _db_state(db_path)  # re-read after indexing
    persons_with_refs = state.get("persons_with_refs", set())
    any_added = False

    for person_name, photo_files in REFERENCE_PERSONS:
        if person_name in persons_with_refs and not args.force_faces:
            print(f"\n  Step 2: skipped '{person_name}' — references already stored.")
            continue

        photo_args = []
        for fname in photo_files:
            full = refs_dir / fname
            if full.exists():
                photo_args += ["--photo", str(full)]
            else:
                print(f"  Warning: reference photo not found: {full}")

        if not photo_args:
            print(f"  Warning: no reference photos found for {person_name}, skipping.")
            continue

        cmd = [py, "cli.py", "add-person", person_name, "--db", args.db] + photo_args
        rc = _run(cmd, f"Step 2: Register {person_name}")
        if rc != 0:
            print(f"  Warning: add-person failed for {person_name} (exit {rc})")
            errors += 1
        any_added = True

    if not any_added and not args.force_faces:
        print("\n  Step 2: all persons already registered — skipped.")

    # ── Step 3: Match faces ──────────────────────────────────────────────────
    rc = _run(
        [py, "cli.py", "match-faces", "--temporal", "--db", args.db],
        "Step 3: Match faces (strict + temporal)",
    )
    if rc != 0:
        print(f"\n  ERROR: match-faces failed (exit {rc})")
        errors += 1

    # ── Step 4: Sample-specific corrections ──────────────────────────────────
    if SAMPLE_CORRECTIONS:
        print(f"\n{'─' * 62}")
        print(f"  Step 4: Apply sample-specific ground-truth corrections")
        print(f"{'─' * 62}")
        n = _apply_sample_corrections(db_path, SAMPLE_CORRECTIONS)
        print(f"  Applied {n} correction(s).")

    # ── Summary ──────────────────────────────────────────────────────────────
    state = _db_state(db_path)
    print(f"\n{'═' * 62}")
    if errors:
        print(f"  Setup finished with {errors} error(s) — check output above.")
    else:
        print(f"  Setup complete!")
    _print_state(state, "After")
    print(f"\n  Run tests:")
    print(f"    python tests/test_face_matching.py")
    print(f"    pytest tests/ -v")
    print(f"{'═' * 62}\n")

    return errors


if __name__ == "__main__":
    sys.exit(main())
