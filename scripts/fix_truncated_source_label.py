#!/usr/bin/env python3
"""Repair library folders created under a mangled `_incoming/<source>/` label.

Context: the Windows SD-card importer sanitized the EXIF camera model with
`-replace '\\P{IsBasicLatin}'`. PowerShell's `-replace` is case-INSENSITIVE, and
under IgnoreCase .NET treats `I` as case-equivalent to the Turkish dotted/dotless
I (U+0130/U+0131) — which are outside the Basic Latin block — so the pattern ate
every `I`/`i`. `ILCE-7RM6` became `LCE-7RM6`, and ingest, seeing a new source
label, built a parallel set of `YYYY-MM-DD_LCE-7RM6/` folders next to the real
`YYYY-MM-DD_ILCE-7RM6/` ones.

This script folds the bad-label folders back into the good-label ones:

  * file has a byte-identical twin in the good folder  -> delete the bad copy
  * otherwise                                         -> move it into the good
                                                         folder (unique name)
                                                         and repoint its DB row

Dry-run by default. `--apply` needs `--audit PATH`: deletes are undoable only by
re-import, so every action is written to a CSV first.

Usage (on the NAS, where the files live):

    DC="docker compose -f docker-compose.nas.yml"
    $DC run --rm --entrypoint python photosearch scripts/fix_truncated_source_label.py \\
        --bad LCE-7RM6 --good ILCE-7RM6
    $DC run --rm --entrypoint python photosearch scripts/fix_truncated_source_label.py \\
        --bad LCE-7RM6 --good ILCE-7RM6 --audit /data/lce-fix.csv --apply
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from photosearch.db import PhotoDB  # noqa: E402


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def find_pairs(photo_root: Path, bad: str, good: str) -> list[tuple[Path, Path]]:
    """Every `*_<bad>` dir under the library, with its `*_<good>` counterpart.

    Matched by exact directory-name suffix — NOT a LIKE/glob on the label. A
    SQL `LIKE '%_LCE-7RM6/%'` also matches `ILCE-7RM6` (underscore is a
    single-char wildcard), which would target the folders we mean to keep.
    """
    pairs = []
    for parent in sorted(photo_root.iterdir()):          # year dirs + _undated
        if not parent.is_dir():
            continue
        for d in sorted(parent.iterdir()):
            if d.is_dir() and d.name.endswith(f"_{bad}"):
                twin = d.parent / (d.name[: -len(bad)] + good)
                pairs.append((d, twin))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.environ.get("PHOTOSEARCH_DB", "photo_index.db"))
    ap.add_argument("--photo-root", default=os.environ.get("PHOTO_ROOT", "/photos"))
    ap.add_argument("--bad", required=True, help="Mangled label, e.g. LCE-7RM6")
    ap.add_argument("--good", required=True, help="Correct label, e.g. ILCE-7RM6")
    ap.add_argument("--audit", default=None, metavar="PATH",
                    help="CSV of every action taken. Required with --apply.")
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete/move. Default: dry-run.")
    args = ap.parse_args()

    if args.apply and not args.audit:
        ap.error("--apply requires --audit PATH (deletes are only undoable by re-import)")

    photo_root = Path(args.photo_root).resolve()
    pairs = find_pairs(photo_root, args.bad, args.good)
    if not pairs:
        print(f"No '*_{args.bad}' folders under {photo_root} — nothing to do.")
        return 0

    actions: list[dict] = []
    with PhotoDB(args.db) as db:
        c = db.conn
        for bad_dir, good_dir in pairs:
            files = sorted(p for p in bad_dir.iterdir() if p.is_file())
            print(f"\n{bad_dir.relative_to(photo_root)}  ({len(files)} files)  "
                  f"-> {good_dir.relative_to(photo_root)}"
                  f"{'' if good_dir.is_dir() else '  [twin does not exist]'}")

            for src in files:
                rel_bad = str(src.relative_to(photo_root))
                # instr() is a literal substring match — no wildcard traps.
                row = c.execute(
                    "SELECT id FROM photos WHERE filepath = ? OR filepath = ?",
                    (rel_bad, str(src)),
                ).fetchone()
                photo_id = row["id"] if row else None

                twin = good_dir / src.name
                identical = False
                if twin.exists():
                    try:
                        identical = sha256(src) == sha256(twin)
                    except OSError as exc:
                        print(f"  ! {src.name}: unreadable ({exc}) — skipped")
                        continue

                if identical:
                    actions.append({"action": "delete-duplicate", "src": rel_bad,
                                    "dst": str(twin.relative_to(photo_root)),
                                    "photo_id": photo_id or ""})
                    if args.apply:
                        src.unlink()
                        if photo_id is not None:
                            c.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
                else:
                    # No twin, or a same-named but different file: keep it, but
                    # under the correct label. _unique suffix avoids clobbering.
                    dst = good_dir / src.name
                    n = 1
                    while dst.exists():
                        dst = good_dir / f"{src.stem}_{n}{src.suffix}"
                        n += 1
                    rel_dst = str(dst.relative_to(photo_root))
                    actions.append({"action": "move", "src": rel_bad, "dst": rel_dst,
                                    "photo_id": photo_id or ""})
                    if args.apply:
                        good_dir.mkdir(parents=True, exist_ok=True)
                        src.rename(dst)
                        if photo_id is not None:
                            c.execute(
                                "UPDATE photos SET filepath = ?, folder = ? WHERE id = ?",
                                (rel_dst, str(dst.parent.relative_to(photo_root)), photo_id),
                            )

            if args.apply:
                try:
                    bad_dir.rmdir()          # only succeeds if now empty
                    actions.append({"action": "rmdir", "src": str(bad_dir.relative_to(photo_root)),
                                    "dst": "", "photo_id": ""})
                except OSError:
                    pass                     # non-media leftovers — leave it alone

        if args.apply:
            db.conn.commit()

    n_del = sum(1 for a in actions if a["action"] == "delete-duplicate")
    n_mv = sum(1 for a in actions if a["action"] == "move")
    n_rows = sum(1 for a in actions if a["photo_id"] != "")
    verb = "Deleted" if args.apply else "Would delete"
    verb2 = "moved" if args.apply else "would move"
    print(f"\n{verb} {n_del:,} duplicate file(s); {verb2} {n_mv:,} unique file(s) "
          f"into '_{args.good}' folders. {n_rows:,} of them had a DB row.")

    if args.audit:
        with open(args.audit, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["action", "src", "dst", "photo_id"])
            w.writeheader()
            w.writerows(actions)
        print(f"Audit: {args.audit}")

    if not args.apply:
        print("\nDry run — nothing changed. Re-run with --audit PATH --apply.")
    elif n_rows:
        print("DB rows changed — run `cleanup-orphans` to drop dangling vec0 rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
