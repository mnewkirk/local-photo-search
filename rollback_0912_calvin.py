"""Reverse unmatch_0912_calvin.py — restore the 170 Calvin temporal matches
on 2026-09-12 to exactly what they were.

Do NOT use `photosearch restore-unmatched-faces` for this. face_dedupe_undo
also holds ~9.6k rows from the June 2026 over-matching cleanup, and that
command restores every staged row, which would undo that earlier work too.

Only restores faces that are STILL unmatched and still carry the
'dedupe_unmatched' marker — if you have since named one of these faces
correctly by hand, this leaves your label alone.

  cat rollback_0912_calvin.py | ssh <nas-user>@<nas-host> \\
    'cd /volume1/docker/photosearch && docker compose -f docker-compose.nas.yml \\
     run --rm -T -e APPLY=1 --entrypoint python photosearch -'
"""
import os
from photosearch.db import PhotoDB

APPLY = os.environ.get("APPLY") == "1"

FACE_IDS = [
    502779, 502810, 502819, 502828, 502835, 502857, 502874, 502879,
    502884, 502888, 502906, 502918, 502948, 502953, 502986, 503033,
    503069, 503103, 503127, 503140, 503149, 503165, 503180, 503194,
    503198, 503204, 503233, 503262, 503272, 503292, 503293, 503339,
    503344, 503352, 503359, 503365, 503394, 503412, 503432, 503467,
    503470, 503476, 503510, 503516, 503522, 503527, 503533, 503536,
    503560, 503566, 503571, 503593, 503641, 503644, 503646, 503648,
    503698, 503704, 503754, 503756, 503776, 503779, 503791, 503808,
    503815, 503834, 503845, 503854, 503876, 503901, 503917, 503941,
    503952, 503953, 503954, 503973, 503995, 503998, 504018, 504033,
    504036, 504041, 504050, 504055, 504063, 504066, 504079, 504082,
    504091, 504125, 504129, 504157, 504163, 504175, 504191, 504198,
    504248, 504252, 504255, 504282, 504297, 504316, 504320, 504324,
    504350, 504403, 504416, 504463, 504478, 504482, 504487, 504513,
    504519, 504521, 504593, 504616, 504618, 504619, 504629, 504660,
    504668, 504676, 504681, 504703, 504753, 504756, 504760, 504780,
    504784, 504788, 504819, 504836, 504839, 504844, 504848, 504888,
    504898, 504952, 504958, 504960, 504965, 504979, 504999, 505002,
    505007, 505025, 505038, 505058, 505084, 505110, 505157, 505173,
    505187, 505193, 505220, 505247, 505254, 505258, 505266, 505310,
    505314, 505319, 505322, 505329, 505334, 505375, 505379, 505382,
    505436, 505440
]

with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
    c = db.conn
    ph = ",".join("?" * len(FACE_IDS))
    todo = c.execute(
        "SELECT u.face_id, u.person_id, u.match_source "
        "FROM face_dedupe_undo u JOIN faces f ON f.id = u.face_id "
        f"WHERE f.person_id IS NULL "
        f"  AND IFNULL(f.match_source,'') = 'dedupe_unmatched' "
        f"  AND u.face_id IN ({ph})",
        FACE_IDS).fetchall()
    print(f"{len(todo)} of {len(FACE_IDS)} pinned faces are restorable")
    if not APPLY:
        print("DRY RUN - nothing written. Set APPLY=1 to write.")
        raise SystemExit(0)
    for r in todo:
        c.execute("UPDATE faces SET person_id = ?, match_source = ? WHERE id = ?",
                  (r["person_id"], r["match_source"], r["face_id"]))
    db.conn.commit()
    print(f"Restored {len(todo)} face(s).")
