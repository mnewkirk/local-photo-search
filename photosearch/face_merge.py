"""Suggest merges between face groups.

Identifies pairs of face groups (named persons + unknown clusters) that are
likely the same person, so the user can accept/reject each suggestion instead
of hunting for matches manually.

Two metrics drive the suggestion:

  centroid_dist — L2 between the two groups' normalized mean ArcFace vectors
  min_pair_dist — minimum L2 across all member-to-member pairs

A pair is suggested when ``centroid_dist <= centroid_cutoff`` AND
``min_pair_dist <= min_pair_cutoff``. Centroid gates the shortlist; min-pair
is the primary signal ("at least one crop of each cluster is very similar").

Per-group encodings are capped at ``max_members`` — faces with the largest
bbox (highest-quality crops) are sampled first. This bounds per-pair compute
to O(K²) where K = max_members.

named↔named pairs are never suggested (the user already curated them).
Ignored clusters are skipped. cluster↔person and cluster↔cluster pairs are
canonicalized so the cluster is always on the "left" (the thing to merge)
and the person / larger cluster is on the "right" (the thing to merge into).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from .db import FACE_DIMENSIONS, PhotoDB


# Upper bound on faces per group used in min-pair calculation. Named persons
# can have thousands; clipping keeps O(K²) per-pair bounded. Biggest-bbox
# faces are preferred (cleaner crops → more reliable encodings).
MAX_MEMBERS_PER_GROUP = 60

# Centroid distance ceiling — shortlists pairs worth a min-pair calc.
# Generous; the real acceptance gate is min_pair_cutoff below.
CENTROID_CUTOFF = 0.95

# Min-pair distance ceiling — primary merge signal. Conservative starting
# value; will be tuned against the user's TP/FP examples on the real DB.
MIN_PAIR_CUTOFF = 0.60


@dataclass
class GroupInfo:
    """A named person or an unknown cluster, with its sampled member encodings."""

    key: str                      # "person:<id>" or "cluster:<id>"
    type: str                     # "person" | "cluster"
    id: int
    label: str
    face_ids: list[int] = field(default_factory=list)
    encodings: np.ndarray = field(default_factory=lambda: np.zeros((0, FACE_DIMENSIONS), dtype=np.float32))
    centroid: np.ndarray = field(default_factory=lambda: np.zeros((FACE_DIMENSIONS,), dtype=np.float32))
    face_count: int = 0           # true total member count (not clipped by sampling)
    rep_face_id: Optional[int] = None
    date_min: Optional[str] = None
    date_max: Optional[str] = None


@dataclass
class Suggestion:
    """A proposed merge of ``left`` into ``right``."""

    left: GroupInfo
    right: GroupInfo
    centroid_dist: float
    min_pair_dist: float
    shared_days: Optional[int]    # None if either group lacks dates

    def as_dict(self) -> dict:
        return {
            "left": {
                "key": self.left.key, "type": self.left.type, "id": self.left.id,
                "label": self.left.label, "face_count": self.left.face_count,
                "rep_face_id": self.left.rep_face_id,
                "date_min": self.left.date_min, "date_max": self.left.date_max,
            },
            "right": {
                "key": self.right.key, "type": self.right.type, "id": self.right.id,
                "label": self.right.label, "face_count": self.right.face_count,
                "rep_face_id": self.right.rep_face_id,
                "date_min": self.right.date_min, "date_max": self.right.date_max,
            },
            "centroid_dist": float(self.centroid_dist),
            "min_pair_dist": float(self.min_pair_dist),
            "shared_days": self.shared_days,
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_groups(
    db: PhotoDB,
    include_ignored_clusters: bool = False,
    max_members: int = MAX_MEMBERS_PER_GROUP,
    min_group_size: int = 1,
) -> list[GroupInfo]:
    """Load every face group (named persons + unknown clusters) with member encodings.

    Uses a window function to pull the top ``max_members`` faces per group
    (ranked by bbox area desc). Encodings are fetched in one bulk call and
    reshaped via ``np.frombuffer`` for speed (same pattern as
    ``recluster_unknown_faces``).
    """
    # ----- Aggregate counts + date range per group -----
    rows = db.conn.execute(
        """SELECT
               CASE WHEN f.person_id IS NOT NULL
                    THEN 'p:' || f.person_id
                    ELSE 'c:' || f.cluster_id END   AS key,
               f.person_id, f.cluster_id,
               COUNT(*)                             AS face_count,
               MIN(ph.date_taken)                   AS date_min,
               MAX(ph.date_taken)                   AS date_max
           FROM faces f
           LEFT JOIN photos ph ON ph.id = f.photo_id
           WHERE (f.person_id IS NOT NULL) OR (f.cluster_id IS NOT NULL)
           GROUP BY key"""
    ).fetchall()

    ignored_set = set(
        r["cluster_id"]
        for r in db.conn.execute("SELECT cluster_id FROM ignored_clusters").fetchall()
    )
    persons = {
        r["id"]: r["name"]
        for r in db.conn.execute("SELECT id, name FROM persons").fetchall()
    }

    group_meta: dict[str, GroupInfo] = {}
    for r in rows:
        face_count = int(r["face_count"])
        if face_count < min_group_size:
            continue
        if r["person_id"] is not None:
            pid = int(r["person_id"])
            g = GroupInfo(
                key=r["key"], type="person", id=pid,
                label=persons.get(pid, f"Person #{pid}"),
                face_count=face_count,
                date_min=r["date_min"], date_max=r["date_max"],
            )
        else:
            cid = int(r["cluster_id"])
            if (not include_ignored_clusters) and cid in ignored_set:
                continue
            g = GroupInfo(
                key=r["key"], type="cluster", id=cid,
                label=f"Unknown #{cid}",
                face_count=face_count,
                date_min=r["date_min"], date_max=r["date_max"],
            )
        group_meta[g.key] = g

    if not group_meta:
        return []

    # ----- Pull top-K face ids per group via window function -----
    face_rows = db.conn.execute(
        """WITH ranked AS (
               SELECT f.id AS face_id,
                      f.person_id, f.cluster_id,
                      CASE WHEN f.bbox_top IS NOT NULL
                           THEN (f.bbox_bottom - f.bbox_top)
                                * (f.bbox_right - f.bbox_left)
                           ELSE 0 END AS area,
                      ROW_NUMBER() OVER (
                          PARTITION BY CASE WHEN f.person_id IS NOT NULL
                                            THEN 'p:' || f.person_id
                                            ELSE 'c:' || f.cluster_id END
                          ORDER BY
                              CASE WHEN f.bbox_top IS NOT NULL
                                   THEN (f.bbox_bottom - f.bbox_top)
                                        * (f.bbox_right - f.bbox_left)
                                   ELSE 0 END DESC,
                              f.id
                      ) AS rn
               FROM faces f
               WHERE (f.person_id IS NOT NULL) OR (f.cluster_id IS NOT NULL)
           )
           SELECT face_id, person_id, cluster_id, area
           FROM ranked
           WHERE rn <= ?""",
        (max_members,),
    ).fetchall()

    # Group face ids by their group key
    ids_by_group: dict[str, list[int]] = {k: [] for k in group_meta}
    rep_by_group: dict[str, tuple[int, int]] = {}  # key -> (best_face_id, area)
    for r in face_rows:
        key = ("p:" + str(r["person_id"])) if r["person_id"] is not None \
              else ("c:" + str(r["cluster_id"]))
        if key not in ids_by_group:
            continue  # ignored cluster or sub-min group — skip
        ids_by_group[key].append(int(r["face_id"]))
        area = int(r["area"] or 0)
        cur = rep_by_group.get(key)
        if cur is None or area > cur[1]:
            rep_by_group[key] = (int(r["face_id"]), area)

    # ----- Bulk-load encodings -----
    all_ids: list[int] = []
    for fids in ids_by_group.values():
        all_ids.extend(fids)
    if not all_ids:
        return []

    enc_map = _load_encodings_bulk(db, all_ids)

    # ----- Attach encodings + centroids to GroupInfo objects -----
    groups: list[GroupInfo] = []
    for key, meta in group_meta.items():
        face_ids = [fid for fid in ids_by_group.get(key, []) if fid in enc_map]
        if not face_ids:
            continue
        X = np.stack([enc_map[fid] for fid in face_ids]).astype(np.float32, copy=False)
        centroid = X.mean(axis=0)
        n = float(np.linalg.norm(centroid))
        if n > 0:
            centroid = centroid / n
        meta.face_ids = face_ids
        meta.encodings = X
        meta.centroid = centroid.astype(np.float32, copy=False)
        rep = rep_by_group.get(key)
        meta.rep_face_id = rep[0] if rep else (face_ids[0] if face_ids else None)
        groups.append(meta)

    return groups


def _load_encodings_bulk(db: PhotoDB, face_ids: list[int]) -> dict[int, np.ndarray]:
    """Fetch encodings in chunks and decode via np.frombuffer for speed."""
    result: dict[int, np.ndarray] = {}
    batch = 1000
    for i in range(0, len(face_ids), batch):
        chunk = face_ids[i:i + batch]
        placeholders = ",".join("?" * len(chunk))
        rows = db.conn.execute(
            f"SELECT face_id, encoding FROM face_encodings WHERE face_id IN ({placeholders})",
            chunk,
        ).fetchall()
        for r in rows:
            arr = np.frombuffer(r["encoding"], dtype=np.float32).copy()
            if arr.size == FACE_DIMENSIONS:
                result[int(r["face_id"])] = arr
    return result


# ---------------------------------------------------------------------------
# Suggesting merges
# ---------------------------------------------------------------------------

def compute_suggestions(
    groups: list[GroupInfo],
    centroid_cutoff: float = CENTROID_CUTOFF,
    min_pair_cutoff: float = MIN_PAIR_CUTOFF,
    include_person_pairs: bool = False,
) -> list[Suggestion]:
    """Return suggestions sorted by ascending min-pair distance.

    Pairs are canonicalized so the cluster-to-merge appears on ``left``
    and the person / larger cluster on ``right``.
    """
    if len(groups) < 2:
        return []

    # Pairwise centroid distance matrix (all groups at once — cheap even at G=3000).
    # Encodings are unit-norm, so ||a-b||² = 2 - 2·a·b.
    C = np.stack([g.centroid for g in groups]).astype(np.float32, copy=False)
    sim = C @ C.T
    np.clip(2.0 - 2.0 * sim, 0.0, None, out=sim)
    cent_dists = np.sqrt(sim)  # sim now holds squared L2

    suggestions: list[Suggestion] = []
    n = len(groups)
    for i in range(n):
        gi = groups[i]
        for j in range(i + 1, n):
            gj = groups[j]
            if (not include_person_pairs) and gi.type == "person" and gj.type == "person":
                continue
            cd = float(cent_dists[i, j])
            if cd > centroid_cutoff:
                continue
            md = _min_pair_dist(gi.encodings, gj.encodings)
            if md > min_pair_cutoff:
                continue
            left, right = _canonical_order(gi, gj)
            suggestions.append(Suggestion(
                left=left, right=right,
                centroid_dist=cd, min_pair_dist=md,
                shared_days=_date_overlap_days(left, right),
            ))

    suggestions.sort(key=lambda s: (s.min_pair_dist, s.centroid_dist))
    return suggestions


def score_pair(a: GroupInfo, b: GroupInfo) -> tuple[float, float]:
    """Compute (centroid_dist, min_pair_dist) for an arbitrary group pair.

    Useful for --verify output, which needs scores even for pairs that don't
    reach the suggestion threshold.
    """
    diff = a.centroid - b.centroid
    cd = float(np.sqrt(max(float(diff @ diff), 0.0)))
    md = _min_pair_dist(a.encodings, b.encodings)
    return cd, md


def _min_pair_dist(A: np.ndarray, B: np.ndarray) -> float:
    """Minimum L2 distance between any row of A and any row of B (unit-norm assumed)."""
    if A.size == 0 or B.size == 0:
        return float("inf")
    sims = A @ B.T
    sq = 2.0 - 2.0 * sims
    np.maximum(sq, 0.0, out=sq)
    return float(np.sqrt(sq.min()))


def _canonical_order(a: GroupInfo, b: GroupInfo) -> tuple[GroupInfo, GroupInfo]:
    """Put the 'merge-source' group on the left (cluster, or smaller of two clusters)."""
    if a.type == "person" and b.type == "cluster":
        return b, a
    if a.type == "cluster" and b.type == "person":
        return a, b
    if a.type == "cluster" and b.type == "cluster":
        return (a, b) if a.face_count <= b.face_count else (b, a)
    return a, b


def _date_overlap_days(a: GroupInfo, b: GroupInfo) -> Optional[int]:
    """Days of overlap between the two groups' date ranges (None if any side lacks dates)."""
    if not (a.date_min and a.date_max and b.date_min and b.date_max):
        return None
    try:
        a_min = datetime.fromisoformat(a.date_min[:19])
        a_max = datetime.fromisoformat(a.date_max[:19])
        b_min = datetime.fromisoformat(b.date_min[:19])
        b_max = datetime.fromisoformat(b.date_max[:19])
    except ValueError:
        return None
    lo = max(a_min, b_min)
    hi = min(a_max, b_max)
    if hi <= lo:
        return 0
    return max(1, (hi - lo).days)


# ---------------------------------------------------------------------------
# Verification helpers (for CLI --verify-pair)
# ---------------------------------------------------------------------------

def resolve_group_spec(
    db: PhotoDB,
    groups: list[GroupInfo],
    spec: str,
) -> Optional[GroupInfo]:
    """Resolve a spec like ``cluster:2035`` or ``person:Matt Newkirk`` to a GroupInfo.

    Person names match case-insensitively. Returns None if the group doesn't
    exist or isn't in the loaded set (e.g. filtered out by min_group_size).
    """
    if ":" not in spec:
        return None
    t, v = spec.split(":", 1)
    t = t.strip().lower()
    v = v.strip()
    if t == "cluster":
        try:
            cid = int(v)
        except ValueError:
            return None
        for g in groups:
            if g.type == "cluster" and g.id == cid:
                return g
        return None
    if t == "person":
        row = db.conn.execute(
            "SELECT id FROM persons WHERE LOWER(name) = LOWER(?)", (v,)
        ).fetchone()
        if not row:
            return None
        pid = int(row["id"])
        for g in groups:
            if g.type == "person" and g.id == pid:
                return g
        return None
    return None


def parse_verify_pair(s: str) -> tuple[str, str, bool]:
    """Parse ``cluster:2035=person:Matt Newkirk`` or ``cluster:798!=cluster:745``.

    Returns (left_spec, right_spec, should_match).
    """
    if "!=" in s:
        left, right = s.split("!=", 1)
        return left.strip(), right.strip(), False
    if "=" in s:
        left, right = s.split("=", 1)
        return left.strip(), right.strip(), True
    raise ValueError(f"Pair must contain '=' or '!=': {s!r}")
