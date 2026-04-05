"""Face detection, encoding, clustering, and matching.

Uses InsightFace (RetinaFace detector + ArcFace recognizer) for all face
operations. All face data is stored in the database — photos are never modified.

On first use, InsightFace will download the 'buffalo_l' model pack (~300 MB)
to ~/.insightface/models/buffalo_l/.

Workflow:
  1. index --faces          → detect faces in photos, store encodings
  2. add-person "Name"      → register a named person from a reference photo
  3. match-faces            → match all unnamed faces against known persons
  4. search --person "Name" → find photos containing that person
"""

import warnings
# Suppress FutureWarning from insightface/utils/face_align.py about scikit-image
# SimilarityTransform.estimate() deprecation — upstream issue, not our code.
warnings.filterwarnings(
    "ignore",
    message="`estimate` is deprecated",
    category=FutureWarning,
)

from typing import Optional
import numpy as np

try:
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False

# Face embedding dimension (ArcFace produces 512-dim L2-normalized vectors).
FACE_ENCODING_DIM = 512

# Match tolerance: L2 distance between two 512-dim normalized ArcFace vectors.
# For normalized vectors, L2 distance = sqrt(2 * (1 - cosine_similarity)).
# Calibrated on sample photos: same-person L2 ranges 0.88–1.11 (smaller faces push
# distances higher due to reduced crop quality); different people consistently > 1.31.
# Using 1.15 captures all confirmed matches with a 0.16+ gap before the false-positive floor.
MATCH_TOLERANCE = 1.15

# Slightly stricter tolerance used when grouping unknown faces into clusters.
CLUSTER_TOLERANCE = 0.75

# Maximum long-edge dimension before downsampling for detection.
# High-res cameras (e.g. Sony A7 IV at 7008px) are downsampled to this
# before detection. Bounding boxes are scaled back to original coordinates
# after detection so they remain accurate in the database.
DETECTION_MAX_DIM = 3500

# Global InsightFace app instance (lazy-loaded on first use).
_face_app: Optional["FaceAnalysis"] = None


def check_available():
    """Raise a clear error if insightface is not installed."""
    if not HAS_INSIGHTFACE:
        raise RuntimeError(
            "insightface is not installed.\n"
            "Run: pip install insightface onnxruntime"
        )


def _get_face_app() -> "FaceAnalysis":
    """Lazy-load and return the InsightFace app (CPU execution)."""
    global _face_app
    if _face_app is None:
        check_available()
        import time as _time
        print("  Loading InsightFace model (buffalo_l)...", end="", flush=True)
        t0 = _time.time()
        app = FaceAnalysis(
            name="buffalo_l",
            # Only load detection + recognition. Skip landmark (1k3d68, 2d106det)
            # and genderage models — we only use bbox + ArcFace embedding.
            # This roughly halves per-face inference time on CPU.
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        # det_size controls the internal resolution of the RetinaFace detector.
        # 640×640 is the standard setting and handles most face sizes well.
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _face_app = app
        print(f" ready ({_time.time() - t0:.1f}s)")
    return _face_app


def _downsample(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Downsample a large image for face detection.

    Returns (resized_image, scale) where scale is the multiplier to recover
    original-image coordinates from detection coordinates.
    If the image is already within DETECTION_MAX_DIM, returns it unchanged
    with scale=1.0.
    """
    import cv2
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= DETECTION_MAX_DIM:
        return image, 1.0
    scale = long_edge / DETECTION_MAX_DIM
    new_w = int(w / scale)
    new_h = int(h / scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _insightface_bbox_to_trbl(bbox) -> tuple:
    """Convert InsightFace [x1, y1, x2, y2] bbox to (top, right, bottom, left)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return (y1, x2, y2, x1)


def _normed_embedding(face) -> Optional[np.ndarray]:
    """Return the L2-normalized 512-dim ArcFace embedding for a detected face.

    InsightFace stores the un-normalized output in face.embedding and the
    normalized version in face.normed_embedding (added in later versions).
    We normalize explicitly so the result is always a unit vector regardless
    of the InsightFace version, which keeps L2 distances in the [0, 2] range
    and makes our MATCH_TOLERANCE thresholds meaningful.
    """
    raw = getattr(face, "normed_embedding", None)
    if raw is None:
        raw = face.embedding
    if raw is None:
        return None
    arr = np.array(raw, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


def _scale_trbl(bbox: tuple, scale: float) -> tuple:
    """Scale a (top, right, bottom, left) bbox back to original image coordinates."""
    if scale == 1.0:
        return bbox
    top, right, bottom, left = bbox
    return (int(top * scale), int(right * scale), int(bottom * scale), int(left * scale))


# ---------------------------------------------------------------------------
# Core detection / encoding
# ---------------------------------------------------------------------------

def detect_faces(image_path: str, use_cnn: bool = False) -> list[dict]:
    """Detect all faces in an image and return their encodings and bounding boxes.

    Args:
        image_path: Path to a JPEG or other supported image file.
        use_cnn: Ignored — InsightFace always uses its CNN-based detector.
                 Kept for API compatibility with the old face_recognition backend.

    High-res images are downsampled to DETECTION_MAX_DIM for detection speed
    and memory efficiency. Bounding boxes are then scaled back to original
    coordinates. InsightFace crops and encodes each face from the downsampled
    image (high enough quality for 128→512-dim ArcFace embeddings).

    Returns a list of dicts:
      {
        "encoding": list[float],   # 512-dim ArcFace encoding (L2-normalized)
        "bbox":     (top, right, bottom, left),  # in original image coords
        "det_score": float,        # detection confidence [0, 1]
      }
    """
    check_available()
    import cv2
    import time as _time

    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"  Warning: could not read image: {image_path}")
            return []

        h, w = img_bgr.shape[:2]
        small_img, scale = _downsample(img_bgr)

        if scale > 1.0:
            print(
                f"    Downsampled {w}×{h} → {small_img.shape[1]}×{small_img.shape[0]} "
                f"(scale 1/{scale:.1f})",
                end="",
                flush=True,
            )

        app = _get_face_app()

        t0 = _time.time()
        faces = app.get(small_img)
        elapsed = _time.time() - t0

        if not faces:
            return []

        results = []
        for face in faces:
            emb = _normed_embedding(face)
            if emb is None:
                continue
            bbox = _insightface_bbox_to_trbl(face.bbox)
            if scale > 1.0:
                bbox = _scale_trbl(bbox, scale)
            results.append({
                "encoding": emb.tolist(),
                "bbox": bbox,
                "det_score": float(face.det_score) if hasattr(face, "det_score") else 1.0,
            })

        return results

    except Exception as e:
        print(f"  Warning: face detection failed for {image_path}: {e}")
        return []


def encode_reference_photo(image_path: str) -> Optional[list[float]]:
    """Extract a single face encoding from a reference photo.

    Tries detection at the standard downsampled resolution. For high-res
    images where no face is found at the downsampled size, retries at full
    resolution (in case the face is very small).

    If multiple faces are found, uses the largest (most prominent) one.
    Returns None if no face is found after all attempts.
    """
    check_available()
    import cv2
    import time as _time

    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"  Warning: could not read image: {image_path}")
            return None

        app = _get_face_app()
        h, w = img_bgr.shape[:2]
        small_img, scale = _downsample(img_bgr)

        strategies: list[tuple[str, np.ndarray]] = [("downsampled" if scale > 1.0 else "standard", small_img)]
        if scale > 1.0:
            # If downsampled detection fails, try full-res as fallback
            strategies.append(("full-res", img_bgr))

        found_faces = None
        for name, img in strategies:
            print(f"    Trying {name}...", end="", flush=True)
            t0 = _time.time()
            faces = app.get(img)
            elapsed = _time.time() - t0
            if faces:
                print(f" found {len(faces)} face(s) ({elapsed:.1f}s)")
                found_faces = faces
                break
            else:
                print(f" nothing ({elapsed:.1f}s)")

        if not found_faces:
            print(f"  No face found in reference photo: {image_path}")
            print(f"  Tip: use a clear, front-facing photo with the face filling most of the frame.")
            return None

        # If multiple faces, use the largest by bounding box area
        if len(found_faces) > 1:
            def _face_area(f):
                x1, y1, x2, y2 = f.bbox
                return (x2 - x1) * (y2 - y1)
            found_faces = [max(found_faces, key=_face_area)]
            print(f"  Multiple faces found — using the largest one.")

        face = found_faces[0]
        emb = _normed_embedding(face)
        if emb is None:
            print(f"  Warning: face detected but no embedding generated for {image_path}")
            return None

        return emb.tolist()

    except Exception as e:
        print(f"  Warning: could not encode reference photo {image_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Matching and clustering
# ---------------------------------------------------------------------------

def match_face(
    query_encoding: list[float],
    known_encodings: list[list[float]],
    tolerance: float = MATCH_TOLERANCE,
) -> list[tuple[int, float]]:
    """Compare a query face encoding against a list of known encodings.

    Computes L2 distance on L2-normalized 512-dim ArcFace vectors.
    Returns a list of (index, distance) tuples for all matches within tolerance,
    sorted by distance (closest first).
    """
    if not known_encodings:
        return []

    query_np = np.array(query_encoding)
    known_np = np.array(known_encodings)

    diffs = known_np - query_np
    distances = np.sqrt((diffs ** 2).sum(axis=1))

    matches = [
        (i, float(dist))
        for i, dist in enumerate(distances)
        if dist <= tolerance
    ]
    matches.sort(key=lambda x: x[1])
    return matches


def cluster_encodings(
    encodings: list[list[float]],
    tolerance: float = CLUSTER_TOLERANCE,
) -> list[int]:
    """Assign cluster IDs to a list of face encodings using greedy nearest-centroid clustering.

    Two faces land in the same cluster when their L2 distance is ≤ tolerance.
    Returns a list of integer cluster IDs (starting at 0) parallel to the input.
    """
    if not encodings:
        return []

    cluster_ids = [-1] * len(encodings)
    cluster_centers: list[np.ndarray] = []

    for i, enc in enumerate(encodings):
        enc_np = np.array(enc)

        if not cluster_centers:
            cluster_ids[i] = 0
            cluster_centers.append(enc_np.copy())
            continue

        centers_np = np.array(cluster_centers)
        diffs = centers_np - enc_np
        distances = np.sqrt((diffs ** 2).sum(axis=1))
        best_idx = int(np.argmin(distances))

        if distances[best_idx] <= tolerance:
            cluster_ids[i] = best_idx
            # Update cluster center to running mean for stability
            n = cluster_ids[:i].count(best_idx) + 1
            cluster_centers[best_idx] = (
                cluster_centers[best_idx] * (n - 1) + enc_np
            ) / n
        else:
            new_id = len(cluster_centers)
            cluster_ids[i] = new_id
            cluster_centers.append(enc_np.copy())

    return cluster_ids


def match_faces_to_persons(
    db,
    tolerance: float = MATCH_TOLERANCE,
) -> int:
    """Match all unassigned faces in the database against all known person references.

    Loads reference encodings from the vector table, then for each unassigned
    face finds the closest reference within tolerance and assigns the person_id.
    Returns the number of faces matched.
    """
    check_available()
    import struct
    from .db import FACE_DIMENSIONS

    # Load all person reference encodings
    ref_rows = db.conn.execute(
        """SELECT fr.id, fr.person_id, p.name
           FROM face_references fr
           JOIN persons p ON p.id = fr.person_id"""
    ).fetchall()

    if not ref_rows:
        print("  No reference encodings found. Add people with: python cli.py add-person")
        return 0

    ref_data = []  # list of (person_id, name, encoding)
    for row in ref_rows:
        enc_row = db.conn.execute(
            "SELECT encoding FROM face_ref_encodings WHERE ref_id = ?", (row["id"],)
        ).fetchone()
        if enc_row:
            enc = list(struct.unpack(f"{FACE_DIMENSIONS}f", enc_row["encoding"]))
            ref_data.append((row["person_id"], row["name"], enc))

    if not ref_data:
        print("  Reference encodings not found in vector table.")
        return 0

    person_ids = [r[0] for r in ref_data]
    ref_encodings = [r[2] for r in ref_data]

    # All faces not yet assigned to a person
    face_rows = db.conn.execute(
        "SELECT id FROM faces WHERE person_id IS NULL"
    ).fetchall()

    if not face_rows:
        print("  All faces already matched.")
        return 0

    matched = 0
    for face_row in face_rows:
        face_id = face_row["id"]
        enc_row = db.conn.execute(
            "SELECT encoding FROM face_encodings WHERE face_id = ?", (face_id,)
        ).fetchone()
        if not enc_row:
            continue

        face_enc = list(struct.unpack(f"{FACE_DIMENSIONS}f", enc_row["encoding"]))
        best_matches = match_face(face_enc, ref_encodings, tolerance=tolerance)

        if best_matches:
            best_idx, _ = best_matches[0]
            db.assign_face_to_person(face_id, person_ids[best_idx], match_source="strict")
            matched += 1

    return matched


# ---------------------------------------------------------------------------
# Temporal proximity matching
# ---------------------------------------------------------------------------

# Looser ArcFace L2 tolerance used for temporal propagation.
# Faces too small or angled for clean auto-matching (e.g. 1.3–1.45 range)
# can still be identified when the same person is confirmed in a nearby photo.
TEMPORAL_TOLERANCE = 1.45

# Minimum L2 gap between the best and second-best person.
# Prevents matches in crowded or uniform-heavy situations where multiple
# people have similar distances (e.g. a team sport where several faces all
# score ~1.35 to a known person).
TEMPORAL_MIN_GAP = 0.08

# Maximum time between two photos to be considered the "same session".
TEMPORAL_WINDOW_MINUTES = 30


def match_faces_temporal(
    db,
    temporal_tolerance: float = TEMPORAL_TOLERANCE,
    min_gap: float = TEMPORAL_MIN_GAP,
    window_minutes: int = TEMPORAL_WINDOW_MINUTES,
) -> int:
    """Propagate person identities to nearby unmatched faces using EXIF timestamps.

    For each still-unmatched face, checks whether the best-matching known person
    (within temporal_tolerance) also appears in a photo taken within window_minutes
    of the current photo. If so, and if the ArcFace distance to that person is
    meaningfully better than the runner-up (by at least min_gap), assigns the
    identity.

    Two-constraint safety:
      1. Distance gap  — the best person must be clearly ahead of the next person.
         This prevents false positives in crowded or uniform scenarios where many
         faces score similarly close to a known person.
      2. Temporal presence — the candidate person must actually appear (confirmed,
         automatically-matched) in another photo from the same session. A face
         that scores 1.40 for "Alex" but Alex isn't in any nearby photo is
         left unassigned.

    Returns the number of faces newly matched.
    """
    from datetime import datetime, timedelta
    import struct
    from .db import FACE_DIMENSIONS

    # ---- Load reference encodings ----
    ref_rows = db.conn.execute(
        """SELECT fr.id, fr.person_id, p.name
           FROM face_references fr
           JOIN persons p ON p.id = fr.person_id"""
    ).fetchall()
    if not ref_rows:
        return 0

    ref_data = []  # list of (person_id, name, encoding_list)
    for row in ref_rows:
        enc_row = db.conn.execute(
            "SELECT encoding FROM face_ref_encodings WHERE ref_id = ?", (row["id"],)
        ).fetchone()
        if enc_row:
            enc = list(struct.unpack(f"{FACE_DIMENSIONS}f", enc_row["encoding"]))
            ref_data.append((row["person_id"], row["name"], enc))
    if not ref_data:
        return 0

    # De-duplicate: keep best encoding per person (for the gap check we need
    # per-person best distances, not per-reference distances).
    unique_persons: dict[int, tuple[str, list[float]]] = {}
    for pid, name, enc in ref_data:
        unique_persons[pid] = (name, enc)  # last reference wins (all are good)

    person_ids_u  = list(unique_persons.keys())
    person_encs_u = [unique_persons[pid][1] for pid in person_ids_u]
    person_names_u = {pid: unique_persons[pid][0] for pid in person_ids_u}

    # ---- Build person → confirmed session timestamps ----
    # Only timestamps where a face was auto-matched (not manually tagged, i.e.
    # bbox is not null) so we don't propagate from a previous manual correction.
    person_sessions: dict[int, list[datetime]] = {pid: [] for pid in person_ids_u}
    for pid in person_ids_u:
        rows = db.conn.execute(
            """SELECT DISTINCT ph.date_taken
               FROM faces f
               JOIN photos ph ON ph.id = f.photo_id
               WHERE f.person_id = ?
                 AND f.bbox_top IS NOT NULL""",
            (pid,),
        ).fetchall()
        for r in rows:
            if r["date_taken"]:
                try:
                    person_sessions[pid].append(
                        datetime.fromisoformat(r["date_taken"][:19])
                    )
                except ValueError:
                    pass

    # ---- Process unmatched faces ----
    face_rows = db.conn.execute(
        """SELECT f.id, ph.date_taken
           FROM faces f
           JOIN photos ph ON ph.id = f.photo_id
           WHERE f.person_id IS NULL"""
    ).fetchall()
    if not face_rows:
        return 0

    window = timedelta(minutes=window_minutes)
    matched = 0

    for face_row in face_rows:
        face_id = face_row["id"]
        enc_row = db.conn.execute(
            "SELECT encoding FROM face_encodings WHERE face_id = ?", (face_id,)
        ).fetchone()
        if not enc_row:
            continue

        face_enc = list(struct.unpack(f"{FACE_DIMENSIONS}f", enc_row["encoding"]))

        # Compute distances to all unique persons (no tolerance filter yet)
        query_np = np.array(face_enc)
        encs_np  = np.array(person_encs_u)
        dists = np.sqrt(((encs_np - query_np) ** 2).sum(axis=1)).tolist()

        # Sort by distance
        ranked = sorted(zip(dists, person_ids_u), key=lambda x: x[0])
        best_dist, best_pid = ranked[0]

        # Check 1: best distance within temporal tolerance
        if best_dist > temporal_tolerance:
            continue

        # Check 2: clear gap to second-best person
        if len(ranked) > 1:
            second_dist, second_pid = ranked[1]
            if second_pid == best_pid:
                # Skip if top-2 are same person (shouldn't happen with unique_persons)
                second_dist = ranked[2][0] if len(ranked) > 2 else float("inf")
            if (second_dist - best_dist) < min_gap:
                continue  # Too ambiguous — could be a uniform/crowd situation

        # Check 3: best-matching person appears in a nearby photo
        face_ts = face_row["date_taken"]
        if face_ts and person_sessions[best_pid]:
            try:
                face_dt = datetime.fromisoformat(face_ts[:19])
                in_session = any(
                    abs((face_dt - sess_dt).total_seconds()) <= window.total_seconds()
                    for sess_dt in person_sessions[best_pid]
                )
                if not in_session:
                    continue  # Person not confirmed in this session
            except ValueError:
                pass  # No parseable timestamp — skip temporal check

        db.assign_face_to_person(face_id, best_pid, match_source="temporal")
        matched += 1

    return matched
