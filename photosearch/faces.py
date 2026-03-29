"""Face detection, encoding, clustering, and matching.

Uses the face_recognition library (built on dlib) for all face operations.
All face data is stored in the database — photos are never modified.

Workflow:
  1. index --faces          → detect faces in photos, store encodings
  2. add-person "Name"      → register a named person from a reference photo
  3. match-faces            → match all unnamed faces against known persons
  4. search --person "Name" → find photos containing that person
"""

import warnings
from typing import Optional
import numpy as np

try:
    # Suppress pkg_resources deprecation warning from face_recognition_models,
    # which is a third-party issue we can't fix directly.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
        import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False

# How close two face encodings must be to be considered the same person.
# Lower = stricter. 0.6 is face_recognition's recommended default; 0.65 is
# a good balance for real-world family photos with varied lighting/angles.
MATCH_TOLERANCE = 0.65

# Tolerance for clustering unknown faces into groups.
CLUSTER_TOLERANCE = 0.55


def check_available():
    """Raise a clear error if face_recognition is not installed."""
    if not HAS_FACE_RECOGNITION:
        raise RuntimeError(
            "face_recognition is not installed.\n"
            "Run: brew install cmake && pip install dlib face_recognition"
        )


def _bbox_overlap(a: tuple, b: tuple, threshold: float = 0.5) -> bool:
    """Return True if two bounding boxes overlap by more than threshold (IoU)."""
    top = max(a[0], b[0])
    right = min(a[1], b[1])
    bottom = min(a[2], b[2])
    left = max(a[3], b[3])
    if bottom <= top or right <= left:
        return False
    inter = (bottom - top) * (right - left)
    area_a = (a[2] - a[0]) * (a[1] - a[3])
    area_b = (b[2] - b[0]) * (b[1] - b[3])
    union = area_a + area_b - inter
    return (inter / union) > threshold if union > 0 else False


# Maximum long-edge dimension used for face detection.
# High-res cameras (e.g. Sony A7 IV at 7008px) are downsampled to this
# before detection to avoid OOM and dramatically speed up CNN.
# A 75px face at 7008px → ~37px at 3500px, comfortably within CNN range.
# Bounding boxes are scaled back to original coordinates after detection.
DETECTION_MAX_DIM = 3500


def _downsample_for_detection(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Downsample a large image for face detection, returning (resized, scale).

    scale is the factor to multiply detection bbox coordinates by to get
    original-image coordinates. If the image is already small enough, returns
    it unchanged with scale=1.0.
    """
    from PIL import Image as PilImage

    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= DETECTION_MAX_DIM:
        return image, 1.0

    scale = long_edge / DETECTION_MAX_DIM
    new_w = int(w / scale)
    new_h = int(h / scale)
    pil_img = PilImage.fromarray(image)
    pil_small = pil_img.resize((new_w, new_h), PilImage.LANCZOS)
    return np.array(pil_small), scale


def _scale_bbox(bbox: tuple, scale: float) -> tuple:
    """Scale a (top, right, bottom, left) bbox back to original image coordinates."""
    if scale == 1.0:
        return bbox
    top, right, bottom, left = bbox
    return (int(top * scale), int(right * scale), int(bottom * scale), int(left * scale))


def detect_faces(image_path: str, use_cnn: bool = False) -> list[dict]:
    """Detect all faces in an image and return their encodings and bounding boxes.

    Strategy:
      - High-res images are downsampled to DETECTION_MAX_DIM for detection
        to avoid OOM and keep processing times reasonable.
      - HOG runs first (fast). CNN supplements to catch small/angled faces.
      - Bounding boxes are scaled back to original coordinates.
      - Encodings are computed on the original full-res image for accuracy.

    Returns a list of dicts:
      {
        "encoding": list[float],  # 128-dim face encoding
        "bbox": (top, right, bottom, left),  # coordinates in original image
      }
    """
    check_available()
    try:
        import time as _time
        original_image = face_recognition.load_image_file(image_path)
        small_image, scale = _downsample_for_detection(original_image)

        h, w = original_image.shape[:2]
        if scale > 1.0:
            print(f"    Downsampled {w}×{h} → {small_image.shape[1]}×{small_image.shape[0]} "
                  f"(scale 1/{scale:.1f}) for detection", end="", flush=True)

        if use_cnn:
            small_locations = face_recognition.face_locations(small_image, model="cnn")
        else:
            hog_locations = face_recognition.face_locations(small_image, model="hog")

            print(f"    CNN pass...", end="", flush=True)
            t_cnn = _time.time()
            cnn_locations = face_recognition.face_locations(small_image, model="cnn")
            print(f" done ({_time.time() - t_cnn:.1f}s)", end="", flush=True)

            small_locations = list(cnn_locations)
            for hog_loc in hog_locations:
                if not any(_bbox_overlap(hog_loc, cnn_loc) for cnn_loc in cnn_locations):
                    small_locations.append(hog_loc)

        if not small_locations:
            return []

        # Scale bounding boxes back to original image coordinates
        original_locations = [_scale_bbox(loc, scale) for loc in small_locations]

        # Encode on the original full-res image for best accuracy
        encodings = face_recognition.face_encodings(
            original_image, original_locations, num_jitters=3
        )
        return [
            {"encoding": enc.tolist(), "bbox": loc}
            for enc, loc in zip(encodings, original_locations)
        ]
    except Exception as e:
        print(f"  Warning: face detection failed for {image_path}: {e}")
        return []


def encode_reference_photo(image_path: str) -> Optional[list[float]]:
    """Extract a single face encoding from a reference photo.

    Tries progressively more thorough detection methods before giving up:
      1. HOG model (fast)
      2. HOG with 2x upsampling (catches smaller/distant faces)
      3. CNN model (most accurate, handles angles and tricky lighting)

    If multiple faces are found, uses the largest (most prominent) one.
    Returns None if no face is found after all attempts.
    """
    check_available()
    try:
        image = face_recognition.load_image_file(image_path)

        import time as _time
        locations = None
        attempts = [
            ("HOG",             dict(model="hog", number_of_times_to_upsample=1)),
            ("HOG 2x upsample", dict(model="hog", number_of_times_to_upsample=2)),
            ("CNN",             dict(model="cnn")),
        ]
        for attempt_name, kwargs in attempts:
            print(f"    Trying {attempt_name}...", end="", flush=True)
            t_attempt = _time.time()
            locs = face_recognition.face_locations(image, **kwargs)
            elapsed = _time.time() - t_attempt
            if locs:
                print(f" found {len(locs)} face(s) ({elapsed:.1f}s)")
                locations = locs
                break
            else:
                print(f" nothing ({elapsed:.1f}s)")

        if not locations:
            print(f"  No face found in reference photo: {image_path}")
            print(f"  Tip: try a clearer, front-facing photo with the face filling more of the frame.")
            return None

        if len(locations) > 1:
            # Pick the largest face by bounding box area
            locations = [max(locations, key=lambda loc: (loc[2] - loc[0]) * abs(loc[1] - loc[3]))]
            print(f"  Multiple faces found — using the largest one.")

        encodings = face_recognition.face_encodings(image, locations, num_jitters=5)
        return encodings[0].tolist() if encodings else None
    except Exception as e:
        print(f"  Warning: could not encode reference photo {image_path}: {e}")
        return None


def match_face(
    query_encoding: list[float],
    known_encodings: list[list[float]],
    tolerance: float = MATCH_TOLERANCE,
) -> list[tuple[int, float]]:
    """Compare a query face encoding against a list of known encodings.

    Returns a list of (index, distance) tuples for all matches within tolerance,
    sorted by distance (closest first).
    """
    if not known_encodings:
        return []

    query_np = np.array(query_encoding)
    known_np = np.array(known_encodings)

    distances = face_recognition.face_distance(known_np, query_np)
    matches = [
        (i, float(dist))
        for i, dist in enumerate(distances)
        if dist <= tolerance
    ]
    matches.sort(key=lambda x: x[1])
    return matches


def cluster_encodings(encodings: list[list[float]], tolerance: float = CLUSTER_TOLERANCE) -> list[int]:
    """Assign cluster IDs to a list of face encodings using greedy clustering.

    Two faces are in the same cluster if their distance is within tolerance.
    Returns a list of integer cluster IDs parallel to the input encodings.
    Cluster IDs start at 0. Returns -1 for any encoding that failed to cluster.
    """
    if not encodings:
        return []

    cluster_ids = [-1] * len(encodings)
    cluster_centers: list[np.ndarray] = []

    for i, enc in enumerate(encodings):
        enc_np = np.array(enc)

        if not cluster_centers:
            cluster_ids[i] = 0
            cluster_centers.append(enc_np)
            continue

        distances = face_recognition.face_distance(np.array(cluster_centers), enc_np)
        best_idx = int(np.argmin(distances))

        if distances[best_idx] <= tolerance:
            cluster_ids[i] = best_idx
            # Update cluster center to running mean
            n = cluster_ids[:i].count(best_idx) + 1
            cluster_centers[best_idx] = (cluster_centers[best_idx] * (n - 1) + enc_np) / n
        else:
            # Start a new cluster
            new_id = len(cluster_centers)
            cluster_ids[i] = new_id
            cluster_centers.append(enc_np)

    return cluster_ids


def match_faces_to_persons(
    db,
    tolerance: float = MATCH_TOLERANCE,
) -> int:
    """Match all unassigned faces in the database against all known person references.

    Updates face rows with person_id where a match is found.
    Returns the number of faces matched.
    """
    check_available()

    # Load all person reference encodings from DB
    ref_rows = db.conn.execute(
        """SELECT fr.id, fr.person_id, p.name
           FROM face_references fr
           JOIN persons p ON p.id = fr.person_id"""
    ).fetchall()

    if not ref_rows:
        print("  No reference encodings found. Add people with: python cli.py add-person")
        return 0

    # Load reference encodings from sqlite-vec
    ref_data = []  # list of (person_id, name, encoding)
    for row in ref_rows:
        enc_row = db.conn.execute(
            "SELECT encoding FROM face_ref_encodings WHERE ref_id = ?", (row["id"],)
        ).fetchone()
        if enc_row:
            import struct
            enc = list(struct.unpack("128f", enc_row["encoding"]))
            ref_data.append((row["person_id"], row["name"], enc))

    if not ref_data:
        print("  Reference encodings not found in vector table.")
        return 0

    person_ids = [r[0] for r in ref_data]
    person_names = [r[1] for r in ref_data]
    ref_encodings = [r[2] for r in ref_data]

    # Load all faces without a person_id
    face_rows = db.conn.execute(
        "SELECT f.id FROM faces f WHERE f.person_id IS NULL"
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

        import struct
        face_enc = list(struct.unpack("128f", enc_row["encoding"]))
        matches = match_face(face_enc, ref_encodings, tolerance=tolerance)

        if matches:
            best_idx, best_dist = matches[0]
            person_id = person_ids[best_idx]
            person_name = person_names[best_idx]
            db.assign_face_to_person(face_id, person_id)
            matched += 1

    return matched
