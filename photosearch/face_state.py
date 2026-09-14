"""Face-state files: ship computed face assignments from one machine to another.

The desktop replica can afford the global recluster and temporal matching that
the N100 cannot. A face-state file is the bridge — a small SQLite file holding
``(face_id, cluster_id, person_id, match_source)`` for every face (~9 MB at
260k faces), plus an optional ``face_state_meta`` table saying what produced it.

Face ids are stable (AUTOINCREMENT, and the replica is a dump of the NAS), so
they are safe join keys across machines. Faces absent from the file — e.g.
detected on the NAS after the replica synced — are never given a person.

Two consumers share this module so they cannot drift apart: the
``export-face-state`` / ``apply-face-state`` CLI pair, and the maintenance
push's ``face_state`` mode (``maintenance_sync.push_to_nas`` →
``POST /api/admin/maintenance-apply-face-state``).
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Callable, Optional


def export_face_state(db, out_path: str, meta: Optional[dict] = None) -> int:
    """Write every face's assignment columns to a fresh SQLite file.

    ``meta`` (JSON-serializable values) lands in ``face_state_meta`` so the
    file describes itself — the maintenance push uses it to carry the photo
    fingerprint and per-stage watermarks. Returns the number of faces written.
    """
    if os.path.exists(out_path):
        os.remove(out_path)
    c = db.conn
    if c.in_transaction:
        c.commit()  # ATTACH is refused inside an open transaction
    c.execute("ATTACH DATABASE ? AS exp", (out_path,))
    try:
        c.execute("CREATE TABLE exp.face_assignments AS "
                  "SELECT id AS face_id, cluster_id, person_id, match_source FROM faces")
        c.execute("CREATE UNIQUE INDEX exp.ix_fa ON face_assignments(face_id)")
        if meta is not None:
            c.execute("CREATE TABLE exp.face_state_meta (key TEXT PRIMARY KEY, value TEXT)")
            c.executemany("INSERT INTO exp.face_state_meta (key, value) VALUES (?, ?)",
                          [(k, json.dumps(v)) for k, v in meta.items()])
        n = c.execute("SELECT COUNT(*) FROM exp.face_assignments").fetchone()[0]
        c.commit()
    finally:
        c.execute("DETACH DATABASE exp")
    return n


def read_meta(path: str) -> dict:
    """Return the file's meta dict ({} when it has none).

    Raises ValueError when the file is not a face-state file at all — this is
    the validation gate for bytes that arrived over HTTP.
    """
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as e:
        raise ValueError(f"not a face-state file: {e}")
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "face_assignments" not in tables:
            raise ValueError("not a face-state file: no face_assignments table")
        if "face_state_meta" not in tables:
            return {}
        return {k: json.loads(v) for k, v in
                conn.execute("SELECT key, value FROM face_state_meta")}
    except sqlite3.DatabaseError as e:
        raise ValueError(f"not a face-state file: {e}")
    finally:
        conn.close()


# Faces the user deliberately unmatched stay unmatched: an additive fill that
# re-applied a person to them would silently undo dedupe-person-faces /
# resolve-duplicate-persons ('dedupe_unmatched') or an explicit human rejection
# ('rejected' — faces.REJECTED_MATCH_SOURCE).
_ADDITIVE_PERSON_WHERE = (
    "person_id IS NULL "
    "AND IFNULL(match_source, '') NOT IN ('dedupe_unmatched', 'rejected') "
    "AND (SELECT person_id FROM a.face_assignments WHERE face_id = faces.id) IS NOT NULL"
)


def apply_face_state(
    db,
    path: str,
    *,
    apply_persons: bool = True,
    overwrite_persons: bool = False,
    apply_clusters: bool = True,
    renumbered: bool = False,
    apply: bool = True,
    before_commit: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Apply a face-state file to ``db``. Returns counts; writes only if ``apply``.

    apply_persons: fill person assignments. ADDITIVE by default — only faces
        unmatched here get a person, so curation on the target is preserved.
        ``overwrite_persons`` forces every face in the file to match it.
    apply_clusters: copy cluster_id onto faces that remain unmatched here.
    renumbered: the file comes from a global recluster, so every unknown
        cluster id in it is freshly minted and means nothing in the target's
        old numbering. Three consequences, mirroring recluster_unknown_faces:
          - unmatched faces ABSENT from the file lose their cluster_id, or an
            old id would collide with a new cluster of the same number;
          - person-assigned faces lose their cluster_id (the invariant f19d063
            fixed — a stale id on a named face collides the same way);
          - ignored_clusters is REMAPPED rather than wiped: a new cluster stays
            ignored when more than half of its faces were in an ignored cluster
            before. Face ids are the stable anchor that makes this possible.
    before_commit: called with the counts inside the write transaction, so a
        caller can stamp a watermark that rolls back with the data.
    """
    c = db.conn
    if c.in_transaction:
        c.commit()  # ATTACH is refused inside an open transaction
    c.execute("ATTACH DATABASE ? AS a", (path,))
    try:
        summary = {"persons": 0, "clusters": 0, "cleared_absent": 0,
                   "ignored_before": 0, "ignored_after": 0, "applied": False}
        if apply_persons:
            if overwrite_persons:
                summary["persons"] = c.execute(
                    "SELECT COUNT(*) FROM faces f JOIN a.face_assignments x ON x.face_id = f.id "
                    "WHERE IFNULL(f.person_id, -1) <> IFNULL(x.person_id, -1)").fetchone()[0]
            else:
                summary["persons"] = c.execute(
                    f"SELECT COUNT(*) FROM faces WHERE {_ADDITIVE_PERSON_WHERE}").fetchone()[0]
        if apply_clusters:
            summary["clusters"] = c.execute(
                "SELECT COUNT(*) FROM faces f JOIN a.face_assignments x ON x.face_id = f.id "
                "WHERE f.person_id IS NULL "
                "AND IFNULL(f.cluster_id, -1) <> IFNULL(x.cluster_id, -1)").fetchone()[0]
        if renumbered:
            summary["cleared_absent"] = c.execute(
                "SELECT COUNT(*) FROM faces WHERE person_id IS NULL AND cluster_id IS NOT NULL "
                "AND id NOT IN (SELECT face_id FROM a.face_assignments)").fetchone()[0]
            summary["ignored_before"] = c.execute(
                "SELECT COUNT(*) FROM ignored_clusters").fetchone()[0]

        if not apply:
            return summary

        c.execute("BEGIN IMMEDIATE")
        try:
            if renumbered:
                # Snapshot BEFORE any write: which faces sit in an ignored cluster.
                c.execute("DROP TABLE IF EXISTS temp.face_state_ignored")
                c.execute("CREATE TEMP TABLE face_state_ignored AS "
                          "SELECT id AS face_id FROM faces WHERE person_id IS NULL "
                          "AND cluster_id IN (SELECT cluster_id FROM ignored_clusters)")

            # 1) persons first, so newly-matched faces drop out of the cluster step
            if apply_persons:
                src = "(SELECT {col} FROM a.face_assignments WHERE face_id = faces.id)"
                if overwrite_persons:
                    where = "id IN (SELECT face_id FROM a.face_assignments)"
                else:
                    where = _ADDITIVE_PERSON_WHERE
                c.execute(f"UPDATE faces SET person_id = {src.format(col='person_id')}, "
                          f"match_source = {src.format(col='match_source')} WHERE {where}")

            # 2) clusters for faces still unmatched here
            if apply_clusters:
                c.execute("UPDATE faces SET cluster_id = "
                          "(SELECT cluster_id FROM a.face_assignments WHERE face_id = faces.id) "
                          "WHERE person_id IS NULL "
                          "AND id IN (SELECT face_id FROM a.face_assignments)")

            if renumbered:
                c.execute("UPDATE faces SET cluster_id = NULL WHERE person_id IS NULL "
                          "AND cluster_id IS NOT NULL "
                          "AND id NOT IN (SELECT face_id FROM a.face_assignments)")
                c.execute("UPDATE faces SET cluster_id = NULL "
                          "WHERE person_id IS NOT NULL AND cluster_id IS NOT NULL")
                c.execute("DELETE FROM ignored_clusters")
                c.execute(
                    "INSERT INTO ignored_clusters (cluster_id) "
                    "SELECT f.cluster_id FROM faces f "
                    "WHERE f.person_id IS NULL AND f.cluster_id IS NOT NULL "
                    "GROUP BY f.cluster_id "
                    "HAVING 2 * SUM(CASE WHEN f.id IN "
                    "  (SELECT face_id FROM temp.face_state_ignored) THEN 1 ELSE 0 END) > COUNT(*)")
                summary["ignored_after"] = c.execute(
                    "SELECT COUNT(*) FROM ignored_clusters").fetchone()[0]
                c.execute("DROP TABLE temp.face_state_ignored")

            summary["applied"] = True
            if before_commit is not None:
                before_commit(summary)
            c.commit()
        except BaseException:
            c.rollback()
            raise
        return summary
    finally:
        c.execute("DETACH DATABASE a")
