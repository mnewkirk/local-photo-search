"""Interactive photobook builder — persistent, sidecar-backed curation store.

Book/curation state lives in a **separate** SQLite file (default
``photobooks.db.local``) because ``sync-replica.sh`` atomically swaps the whole
replica DB (``mv photo_index.db.local.tmp photo_index.db.local``) and would wipe
any table added there. This store is read/write **local only** — a book is a
desktop curation workspace, not NAS-authoritative data, so there is no dual-write.

Geometry: a book page is ``trim_w`` × ``trim_h`` inches (default 14×11); a
two-page lay-flat spread is the "stage" — ``2*trim_w`` × ``trim_h`` (28×11).
Cell rects (``x, y, w, h``) are inches on that stage.

Crop model (per cell): ``fit`` is ``cover`` | ``contain``; the visible window of
a *cover* photo is centered at ``(crop_cx, crop_cy)`` in source-normalized coords
and tightened by ``crop_zoom`` (≥1). ``crop_min_w`` / ``crop_min_h`` are the
minimum fraction of the source width/height that must stay visible — the 5 UI
presets map to (0,0) no restriction, (0.8,0), (0,0.8), (0.8,0.8), and (1,1) full
view. (1,1) is equivalent to ``fit='contain'``. The browser render/drag layer
enforces these; here we only seed a subject-aware ``(crop_cx, crop_cy)``.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from typing import Any, Optional


# Archetype names mirror photosearch.tools._h_suggest_layout so auto_arrange can
# consume its output directly.
_PANORAMA = {"full-bleed single", "full-spread panorama"}


class BookStore:
    """CRUD + curation over the sidecar photobook DB.

    Methods that need to read photo metadata (dimensions, faces, subject boxes)
    take a live replica ``PhotoDB`` as ``pdb`` — this store never opens or writes
    the replica DB itself.
    """

    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    # -- lifecycle ---------------------------------------------------------
    def _init_schema(self) -> None:
        c = self.conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                subtitle TEXT,
                cover_photo_id INTEGER,
                trim_w REAL NOT NULL DEFAULT 14,
                trim_h REAL NOT NULL DEFAULT 11,
                style_json TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS book_spreads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
                position INTEGER NOT NULL DEFAULT 0,
                label TEXT,
                archetype TEXT,
                bg TEXT NOT NULL DEFAULT '#ffffff',
                caption_json TEXT,
                notes TEXT,
                locked INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS book_cells (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                spread_id INTEGER NOT NULL REFERENCES book_spreads(id) ON DELETE CASCADE,
                position INTEGER NOT NULL DEFAULT 0,
                photo_id INTEGER,
                x REAL NOT NULL DEFAULT 0,
                y REAL NOT NULL DEFAULT 0,
                w REAL NOT NULL DEFAULT 0,
                h REAL NOT NULL DEFAULT 0,
                fit TEXT NOT NULL DEFAULT 'cover',
                crop_cx REAL NOT NULL DEFAULT 0.5,
                crop_cy REAL NOT NULL DEFAULT 0.5,
                crop_zoom REAL NOT NULL DEFAULT 1,
                crop_min_w REAL NOT NULL DEFAULT 0,
                crop_min_h REAL NOT NULL DEFAULT 0,
                align TEXT
            );
            CREATE TABLE IF NOT EXISTS book_decisions (
                book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
                photo_id INTEGER NOT NULL,
                decision TEXT NOT NULL DEFAULT 'include'
                    CHECK (decision IN ('include','exclude')),
                note TEXT,
                updated_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (book_id, photo_id)
            );
            CREATE INDEX IF NOT EXISTS idx_spreads_book ON book_spreads(book_id, position);
            CREATE INDEX IF NOT EXISTS idx_cells_spread ON book_cells(spread_id, position);

            -- M30 authoring pipeline: editable beat outline + per-beat candidates.
            CREATE TABLE IF NOT EXISTS book_beats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
                position INTEGER NOT NULL DEFAULT 0,
                day TEXT,
                title TEXT,
                status TEXT NOT NULL DEFAULT 'in' CHECK (status IN ('in','out')),
                spread_budget INTEGER NOT NULL DEFAULT 1,
                scene_meta TEXT,
                notes TEXT
            );
            CREATE TABLE IF NOT EXISTS book_beat_candidates (
                beat_id INTEGER NOT NULL REFERENCES book_beats(id) ON DELETE CASCADE,
                photo_id INTEGER NOT NULL,
                position INTEGER NOT NULL DEFAULT 0,
                role TEXT NOT NULL DEFAULT 'candidate'
                    CHECK (role IN ('hero','candidate','rejected')),
                vlm_score REAL,
                vlm_reason TEXT,
                crop_mode TEXT NOT NULL DEFAULT 'crop' CHECK (crop_mode IN ('crop','full')),
                PRIMARY KEY (beat_id, photo_id)
            );
            CREATE INDEX IF NOT EXISTS idx_beats_book ON book_beats(book_id, position);
            CREATE INDEX IF NOT EXISTS idx_beatcands_beat ON book_beat_candidates(beat_id, position);
            """
        )
        # Additive column migrations for an existing sidecar (books predates M30).
        for col, ddl in (
            ("back_cover_photo_id", "INTEGER"),
            ("title_page_photo_id", "INTEGER"),
            ("notes", "TEXT"),
            ("target_spreads", "INTEGER"),
        ):
            try:
                c.execute(f"SELECT {col} FROM books LIMIT 1")
            except sqlite3.OperationalError:
                c.execute(f"ALTER TABLE books ADD COLUMN {col} {ddl}")
        c.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _touch(self, book_id: int) -> None:
        self.conn.execute(
            "UPDATE books SET updated_at = datetime('now') WHERE id = ?", (book_id,)
        )

    # -- books -------------------------------------------------------------
    def list_books(self) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT b.*,
                   (SELECT COUNT(*) FROM book_spreads s WHERE s.book_id = b.id) AS spread_count,
                   (SELECT COUNT(*) FROM book_decisions d
                      WHERE d.book_id = b.id AND d.decision = 'include') AS candidate_count
              FROM books b ORDER BY b.updated_at DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def create_book(self, name: str, subtitle: Optional[str] = None,
                    trim_w: float = 14, trim_h: float = 11) -> int:
        cur = self.conn.execute(
            "INSERT INTO books (name, subtitle, trim_w, trim_h) VALUES (?,?,?,?)",
            (name, subtitle, trim_w, trim_h),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_book_row(self, book_id: int) -> Optional[dict]:
        r = self.conn.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()
        return dict(r) if r else None

    def update_book(self, book_id: int, fields: dict) -> None:
        allowed = {"name", "subtitle", "cover_photo_id", "trim_w", "trim_h", "style_json",
                   "back_cover_photo_id", "title_page_photo_id", "notes", "target_spreads"}
        sets, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k} = ?")
                vals.append(v)
        if not sets:
            return
        vals.append(book_id)
        self.conn.execute(f"UPDATE books SET {', '.join(sets)} WHERE id = ?", vals)
        self._touch(book_id)
        self.conn.commit()

    def delete_book(self, book_id: int) -> None:
        self.conn.execute("DELETE FROM books WHERE id = ?", (book_id,))
        self.conn.commit()

    def stage_dims(self, book_id: int) -> tuple[float, float]:
        b = self.get_book_row(book_id)
        tw = (b or {}).get("trim_w") or 14
        th = (b or {}).get("trim_h") or 11
        return tw * 2.0, th

    # -- full document -----------------------------------------------------
    def get_book(self, book_id: int) -> Optional[dict]:
        book = self.get_book_row(book_id)
        if not book:
            return None
        spreads = [dict(r) for r in self.conn.execute(
            "SELECT * FROM book_spreads WHERE book_id = ? ORDER BY position, id",
            (book_id,)).fetchall()]
        by_spread: dict[int, list] = {}
        for r in self.conn.execute(
            """SELECT c.* FROM book_cells c JOIN book_spreads s ON c.spread_id = s.id
               WHERE s.book_id = ? ORDER BY c.spread_id, c.position, c.id""",
            (book_id,)).fetchall():
            by_spread.setdefault(r["spread_id"], []).append(dict(r))
        for sp in spreads:
            sp["caption"] = json.loads(sp["caption_json"]) if sp.get("caption_json") else None
            sp["cells"] = by_spread.get(sp["id"], [])
        decisions = {r["photo_id"]: r["decision"] for r in self.conn.execute(
            "SELECT photo_id, decision FROM book_decisions WHERE book_id = ?",
            (book_id,)).fetchall()}
        sw, sh = self.stage_dims(book_id)
        return {"book": book, "spreads": spreads, "decisions": decisions,
                "stage_w": sw, "stage_h": sh}

    # -- decisions / candidate pool ---------------------------------------
    def add_candidates(self, book_id: int, photo_ids: list[int]) -> int:
        """Add photos to the candidate pool as 'include' — never downgrade an
        existing 'exclude' verdict (INSERT OR IGNORE preserves prior decisions)."""
        added = 0
        for pid in photo_ids:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO book_decisions (book_id, photo_id, decision) "
                "VALUES (?, ?, 'include')", (book_id, int(pid)))
            added += cur.rowcount
        self._touch(book_id)
        self.conn.commit()
        return added

    def set_decision(self, book_id: int, photo_id: int, decision: str,
                     note: Optional[str] = None) -> None:
        if decision not in ("include", "exclude"):
            raise ValueError("decision must be 'include' or 'exclude'")
        self.conn.execute(
            """INSERT INTO book_decisions (book_id, photo_id, decision, note, updated_at)
               VALUES (?,?,?,?, datetime('now'))
               ON CONFLICT(book_id, photo_id)
               DO UPDATE SET decision = excluded.decision, note = excluded.note,
                             updated_at = datetime('now')""",
            (book_id, int(photo_id), decision, note))
        self._touch(book_id)
        self.conn.commit()

    def decision_map(self, book_id: int) -> dict[int, str]:
        return {r["photo_id"]: r["decision"] for r in self.conn.execute(
            "SELECT photo_id, decision FROM book_decisions WHERE book_id = ?",
            (book_id,)).fetchall()}

    def used_photo_ids(self, book_id: int) -> set[int]:
        return {r["photo_id"] for r in self.conn.execute(
            """SELECT DISTINCT c.photo_id FROM book_cells c
               JOIN book_spreads s ON c.spread_id = s.id
               WHERE s.book_id = ? AND c.photo_id IS NOT NULL""",
            (book_id,)).fetchall()}

    def included_ids(self, book_id: int) -> list[int]:
        return [r["photo_id"] for r in self.conn.execute(
            "SELECT photo_id FROM book_decisions WHERE book_id = ? AND decision = 'include' "
            "ORDER BY photo_id", (book_id,)).fetchall()]

    # -- spreads -----------------------------------------------------------
    def _next_spread_pos(self, book_id: int) -> int:
        r = self.conn.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 AS p FROM book_spreads WHERE book_id = ?",
            (book_id,)).fetchone()
        return r["p"]

    def add_spread(self, pdb, book_id: int, archetype: str = "matched 2-up",
                   photo_ids: Optional[list[int]] = None, label: Optional[str] = None,
                   bg: str = "#ffffff", position: Optional[int] = None) -> int:
        if position is None:
            position = self._next_spread_pos(book_id)
        cur = self.conn.execute(
            "INSERT INTO book_spreads (book_id, position, label, archetype, bg) "
            "VALUES (?,?,?,?,?)", (book_id, position, label, archetype, bg))
        spread_id = cur.lastrowid
        self._layout_spread(pdb, book_id, spread_id, archetype, photo_ids or [])
        self._touch(book_id)
        self.conn.commit()
        return spread_id

    def update_spread(self, pdb, spread_id: int, fields: dict) -> None:
        sp = self.conn.execute("SELECT * FROM book_spreads WHERE id = ?",
                               (spread_id,)).fetchone()
        if not sp:
            raise KeyError("spread not found")
        book_id = sp["book_id"]
        sets, vals = [], []
        if "label" in fields:
            sets.append("label = ?"); vals.append(fields["label"])
        if "bg" in fields:
            sets.append("bg = ?"); vals.append(fields["bg"])
        if "notes" in fields:
            sets.append("notes = ?"); vals.append(fields["notes"])
        if "locked" in fields:
            sets.append("locked = ?"); vals.append(1 if fields["locked"] else 0)
        if "position" in fields:
            sets.append("position = ?"); vals.append(int(fields["position"]))
        if "caption" in fields:
            cap = fields["caption"]
            sets.append("caption_json = ?")
            vals.append(json.dumps(cap) if cap else None)
        if sets:
            vals.append(spread_id)
            self.conn.execute(f"UPDATE book_spreads SET {', '.join(sets)} WHERE id = ?", vals)
        # Changing the archetype re-lays out the cells, preserving photo order.
        if "archetype" in fields and fields["archetype"] != sp["archetype"]:
            keep = [r["photo_id"] for r in self.conn.execute(
                "SELECT photo_id FROM book_cells WHERE spread_id = ? ORDER BY position, id",
                (spread_id,)).fetchall()]
            self.conn.execute("UPDATE book_spreads SET archetype = ? WHERE id = ?",
                              (fields["archetype"], spread_id))
            self.conn.execute("DELETE FROM book_cells WHERE spread_id = ?", (spread_id,))
            self._layout_spread(pdb, book_id, spread_id, fields["archetype"], keep)
        self._touch(book_id)
        self.conn.commit()

    def delete_spread(self, spread_id: int) -> None:
        sp = self.conn.execute("SELECT book_id FROM book_spreads WHERE id = ?",
                               (spread_id,)).fetchone()
        self.conn.execute("DELETE FROM book_spreads WHERE id = ?", (spread_id,))
        if sp:
            self._touch(sp["book_id"])
        self.conn.commit()

    def reorder_spreads(self, book_id: int, order: list[int]) -> None:
        for pos, sid in enumerate(order):
            self.conn.execute(
                "UPDATE book_spreads SET position = ? WHERE id = ? AND book_id = ?",
                (pos, int(sid), book_id))
        self._touch(book_id)
        self.conn.commit()

    # -- authoring outline (M30) ------------------------------------------
    def replace_outline(self, book_id: int, beats: list[dict]) -> None:
        """Wipe and rewrite the beat outline. Each beat: {title, day, status,
        spread_budget, scene_meta, candidates:[{photo_id, role, vlm_score,
        vlm_reason, crop_mode}]}."""
        self.conn.execute("DELETE FROM book_beats WHERE book_id = ?", (book_id,))
        for pos, b in enumerate(beats):
            cur = self.conn.execute(
                "INSERT INTO book_beats (book_id, position, day, title, status, "
                "spread_budget, scene_meta) VALUES (?,?,?,?,?,?,?)",
                (book_id, pos, b.get("day"), b.get("title"),
                 b.get("status", "in"), int(b.get("spread_budget") or 1),
                 json.dumps(b.get("scene_meta")) if b.get("scene_meta") is not None else None))
            bid = cur.lastrowid
            for cpos, cand in enumerate(b.get("candidates") or []):
                self.conn.execute(
                    "INSERT OR IGNORE INTO book_beat_candidates (beat_id, photo_id, "
                    "position, role, vlm_score, vlm_reason, crop_mode) VALUES (?,?,?,?,?,?,?)",
                    (bid, int(cand["photo_id"]), cpos, cand.get("role", "candidate"),
                     cand.get("vlm_score"), cand.get("vlm_reason"),
                     cand.get("crop_mode", "crop")))
        self._touch(book_id)
        self.conn.commit()

    def get_outline(self, book_id: int) -> list[dict]:
        beats = [dict(r) for r in self.conn.execute(
            "SELECT * FROM book_beats WHERE book_id = ? ORDER BY position, id",
            (book_id,)).fetchall()]
        cand_by_beat: dict[int, list] = {}
        for r in self.conn.execute(
            """SELECT c.* FROM book_beat_candidates c JOIN book_beats b ON c.beat_id=b.id
               WHERE b.book_id = ? ORDER BY c.beat_id, c.position, c.vlm_score DESC""",
            (book_id,)).fetchall():
            cand_by_beat.setdefault(r["beat_id"], []).append(dict(r))
        for b in beats:
            b["scene_meta"] = json.loads(b["scene_meta"]) if b.get("scene_meta") else None
            b["candidates"] = cand_by_beat.get(b["id"], [])
        return beats

    def update_beat(self, beat_id: int, fields: dict) -> None:
        row = self.conn.execute("SELECT book_id FROM book_beats WHERE id = ?",
                                (beat_id,)).fetchone()
        if not row:
            raise KeyError("beat not found")
        sets, vals = [], []
        for k in ("title", "status", "spread_budget", "notes", "position", "day"):
            if k in fields:
                sets.append(f"{k} = ?"); vals.append(fields[k])
        if sets:
            vals.append(beat_id)
            self.conn.execute(f"UPDATE book_beats SET {', '.join(sets)} WHERE id = ?", vals)
        self._touch(row["book_id"])
        self.conn.commit()

    def set_beat_candidate(self, beat_id: int, photo_id: int, fields: dict) -> None:
        sets, vals = [], []
        for k in ("role", "crop_mode", "position", "vlm_score", "vlm_reason"):
            if k in fields:
                sets.append(f"{k} = ?"); vals.append(fields[k])
        if not sets:
            return
        vals += [beat_id, photo_id]
        self.conn.execute(
            f"UPDATE book_beat_candidates SET {', '.join(sets)} "
            f"WHERE beat_id = ? AND photo_id = ?", vals)
        self.conn.commit()

    def assemble_from_outline(self, pdb, book_id: int, per_spread: int = 3) -> int:
        """Turn the included beats into spreads: a title-page spread (blank left /
        hero right) if set, then each in-beat's non-rejected candidates (hero first)
        chunked into its spread_budget spreads. The hero cell honors crop_mode
        ('full' -> the 100/100 no-crop constraint). Returns spreads created."""
        self.conn.execute("DELETE FROM book_spreads WHERE book_id = ?", (book_id,))
        book = self.get_book_row(book_id) or {}
        sw, sh = self.stage_dims(book_id)
        pos = 0
        # Title page: blank verso, single hero on the recto (right half).
        tp = book.get("title_page_photo_id")
        if tp:
            cur = self.conn.execute(
                "INSERT INTO book_spreads (book_id, position, label, archetype) "
                "VALUES (?,?,?,?)", (book_id, pos, "Title page", "title"))
            cx, cy = self._seed_center(pdb, int(tp))
            m = 0.5
            self.conn.execute(
                "INSERT INTO book_cells (spread_id, position, photo_id, x, y, w, h, "
                "crop_cx, crop_cy) VALUES (?,?,?,?,?,?,?,?,?)",
                (cur.lastrowid, 0, int(tp), sw / 2 + m, m, sw / 2 - 2 * m, sh - 2 * m, cx, cy))
            pos += 1
        for beat in self.get_outline(book_id):
            if beat["status"] != "in":
                continue
            cands = [c for c in beat["candidates"] if c["role"] != "rejected"]
            cands.sort(key=lambda c: (c["role"] != "hero", c["position"]))
            if not cands:
                continue
            budget = max(1, beat.get("spread_budget") or 1)
            chunks = _chunk(cands, budget, per_spread)
            for chunk in chunks:
                if not chunk:
                    continue
                arch = None  # let archetype_layout pick by count
                cur = self.conn.execute(
                    "INSERT INTO book_spreads (book_id, position, label, archetype) "
                    "VALUES (?,?,?,?)",
                    (book_id, pos, beat.get("title"),
                     archetype_for(len(chunk))))
                self._layout_spread(pdb, book_id, cur.lastrowid,
                                    archetype_for(len(chunk)), [c["photo_id"] for c in chunk])
                # Honor crop_mode on the hero (first) cell.
                hero = chunk[0]
                if hero.get("crop_mode") == "full":
                    self.conn.execute(
                        "UPDATE book_cells SET fit='contain', crop_min_w=1, crop_min_h=1 "
                        "WHERE spread_id=? AND position=0", (cur.lastrowid,))
                pos += 1
        self._touch(book_id)
        self.conn.commit()
        return pos

    # -- cells -------------------------------------------------------------
    def set_cell(self, pdb, cell_id: int, fields: dict) -> None:
        cell = self.conn.execute("SELECT * FROM book_cells WHERE id = ?",
                                 (cell_id,)).fetchone()
        if not cell:
            raise KeyError("cell not found")
        book_id = self.conn.execute(
            "SELECT book_id FROM book_spreads WHERE id = ?", (cell["spread_id"],)
        ).fetchone()["book_id"]
        # Assigning a new photo re-seeds the subject-aware crop for this cell.
        if "photo_id" in fields and fields["photo_id"] != cell["photo_id"]:
            pid = fields["photo_id"]
            cx, cy = (0.5, 0.5)
            if pid:
                cx, cy = self._seed_center(pdb, int(pid))
            self.conn.execute(
                "UPDATE book_cells SET photo_id = ?, crop_cx = ?, crop_cy = ?, "
                "crop_zoom = 1 WHERE id = ?", (pid, cx, cy, cell_id))
        allowed = {"x", "y", "w", "h", "fit", "crop_cx", "crop_cy", "crop_zoom",
                   "crop_min_w", "crop_min_h", "align", "position"}
        sets, vals = [], []
        for k in allowed:
            if k in fields:
                sets.append(f"{k} = ?")
                vals.append(fields[k])
        if sets:
            vals.append(cell_id)
            self.conn.execute(f"UPDATE book_cells SET {', '.join(sets)} WHERE id = ?", vals)
        self._touch(book_id)
        self.conn.commit()

    # -- auto-arrange ------------------------------------------------------
    def auto_arrange(self, pdb, book_id: int, photo_ids: Optional[list[int]] = None,
                     spread_count: Optional[int] = None, replace: bool = True) -> int:
        """Materialize spreads/cells from the included candidate pool using the
        deterministic ``tools._h_suggest_layout`` partition. Excluded photos are
        skipped. Returns the number of spreads created."""
        from . import tools  # local import avoids a heavy import at module load

        excluded = {pid for pid, d in self.decision_map(book_id).items() if d == "exclude"}
        ids = photo_ids if photo_ids is not None else self.included_ids(book_id)
        ids = [int(i) for i in ids if int(i) not in excluded]
        if not ids:
            return 0
        plan = tools._h_suggest_layout(pdb, {"photo_ids": ids, "spread_count": spread_count})
        spreads = plan.get("spreads") or []
        if replace:
            self.conn.execute("DELETE FROM book_spreads WHERE book_id = ?", (book_id,))
        base = self._next_spread_pos(book_id)
        for i, sp in enumerate(spreads):
            cur = self.conn.execute(
                "INSERT INTO book_spreads (book_id, position, label, archetype, bg) "
                "VALUES (?,?,?,?,?)",
                (book_id, base + i,
                 f"{sp.get('place') or ''} {sp.get('date') or ''}".strip() or None,
                 sp.get("archetype"), "#ffffff"))
            self._layout_spread(pdb, book_id, cur.lastrowid, sp.get("archetype"),
                                sp.get("photo_ids") or [])
        self._touch(book_id)
        self.conn.commit()
        return len(spreads)

    # -- layout + crop-seed helpers ---------------------------------------
    def _layout_spread(self, pdb, book_id: int, spread_id: int, archetype: str,
                       photo_ids: list[int]) -> None:
        sw, sh = self.stage_dims(book_id)
        n = max(len(photo_ids), 1)
        rects = archetype_layout(archetype, len(photo_ids) or 1, sw, sh)
        for i, rect in enumerate(rects):
            pid = photo_ids[i] if i < len(photo_ids) else None
            cx, cy = (0.5, 0.5)
            if pid:
                cx, cy = self._seed_center(pdb, int(pid))
            self.conn.execute(
                "INSERT INTO book_cells (spread_id, position, photo_id, x, y, w, h, "
                "crop_cx, crop_cy) VALUES (?,?,?,?,?,?,?,?,?)",
                (spread_id, i, pid, rect[0], rect[1], rect[2], rect[3], cx, cy))

    def _seed_center(self, pdb, photo_id: int) -> tuple[float, float]:
        """Subject-aware crop center (0–1) from face + subject boxes (ported from
        mocklib.js coverSA centroid). Falls back to frame center when a photo has
        no detected subject."""
        try:
            row = pdb.conn.execute(
                "SELECT image_width, image_height, subject_boxes FROM photos WHERE id = ?",
                (photo_id,)).fetchone()
        except Exception:
            return (0.5, 0.5)
        if not row:
            return (0.5, 0.5)
        W = row["image_width"] or 0
        H = row["image_height"] or 0
        boxes = []  # normalized (l, t, r, b)
        if W and H:
            try:
                for f in pdb.conn.execute(
                    "SELECT bbox_left, bbox_top, bbox_right, bbox_bottom FROM faces "
                    "WHERE photo_id = ?", (photo_id,)).fetchall():
                    if None in (f["bbox_left"], f["bbox_top"], f["bbox_right"], f["bbox_bottom"]):
                        continue
                    boxes.append((f["bbox_left"] / W, f["bbox_top"] / H,
                                  f["bbox_right"] / W, f["bbox_bottom"] / H))
            except Exception:
                pass
        if row["subject_boxes"]:
            try:
                # subject_boxes items are {"label", "bbox":[x1,y1,x2,y2] normalized,
                # "area_frac"} — corners already in 0–1 EXIF-oriented space.
                for s in json.loads(row["subject_boxes"]) or []:
                    bb = s.get("bbox")
                    if not bb or len(bb) != 4:
                        continue
                    boxes.append((bb[0], bb[1], bb[2], bb[3]))
            except Exception:
                pass
        if not boxes:
            return (0.5, 0.5)
        l = min(b[0] for b in boxes); t = min(b[1] for b in boxes)
        r = max(b[2] for b in boxes); btm = max(b[3] for b in boxes)
        cx = min(1.0, max(0.0, (l + r) / 2.0))
        cy = min(1.0, max(0.0, (t + btm) / 2.0))
        return (cx, cy)


# ---------------------------------------------------------------------------
# Archetype → cell rects (inches on the stage). One source of truth for layout.
# ---------------------------------------------------------------------------

def archetype_for(k: int) -> str:
    """House-style archetype name for a photo count (matches archetype_layout)."""
    if k <= 1:
        return "full-bleed single"
    if k == 2:
        return "matched 2-up"
    if k <= 7:
        return "asymmetric collage"
    return "dense grid"


def _chunk(items: list, budget: int, per_spread: int) -> list[list]:
    """Split up to ``budget*per_spread`` items into ``budget`` roughly-even groups
    (spreads), hero-first order preserved."""
    take = items[:max(1, budget) * max(1, per_spread)]
    n, b = len(take), max(1, budget)
    base, extra = divmod(n, b)
    out, i = [], 0
    for s in range(b):
        k = base + (1 if s < extra else 0)
        if k == 0:
            continue
        out.append(take[i:i + k])
        i += k
    return out


def _grid_rects(n: int, x0: float, y0: float, w: float, h: float,
                cols: int, g: float) -> list[list[float]]:
    cols = max(1, cols)
    rows = max(1, math.ceil(n / cols))
    cw = (w - (cols - 1) * g) / cols
    ch = (h - (rows - 1) * g) / rows
    out = []
    for i in range(n):
        r, c = divmod(i, cols)
        out.append([x0 + c * (cw + g), y0 + r * (ch + g), cw, ch])
    return out


def archetype_layout(archetype: Optional[str], n: int, sw: float, sh: float,
                     m: float = 0.4, g: float = 0.5) -> list[list[float]]:
    """Return ``n`` non-overlapping [x, y, w, h] rects (inches) on the ``sw``×``sh``
    stage, matching the house-style archetypes. ``m`` = outer margin, ``g`` =
    gutter. A single frame runs full-bleed (no margin)."""
    n = max(1, n)
    arch = archetype or ("matched 2-up" if n == 2 else
                         "asymmetric collage" if n <= 7 else "dense grid")
    if arch in _PANORAMA or n == 1:
        return [[0, 0, sw, sh]]
    inner_w, inner_h = sw - 2 * m, sh - 2 * m
    if arch == "matched 2-up" or n == 2:
        cw = (inner_w - g) / 2
        return [[m, m, cw, inner_h], [m + cw + g, m, cw, inner_h]]
    if arch == "dense grid" or n >= 8:
        cols = math.ceil(math.sqrt(n))
        return _grid_rects(n, m, m, inner_w, inner_h, cols, g)
    # asymmetric collage (3–7): one anchor on the left + a right-column grid.
    anchor_w = sw * 0.5
    left = [[m, m, anchor_w - g / 2 - m, inner_h]]
    rest = n - 1
    rx = anchor_w + g / 2
    rw = sw - rx - m
    cols = 1 if rest <= 2 else 2
    right = _grid_rects(rest, rx, m, rw, inner_h, cols, g)
    return left + right
