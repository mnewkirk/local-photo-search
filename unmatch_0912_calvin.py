"""TARGETED unmatch of the 170 Calvin TEMPORAL matches on 2026-09-12.

Why pinned ids and not a filter: a filter re-evaluated later catches whatever
is temporal-Calvin-on-that-date *at the time it runs*, which is not the set
that was reviewed. The same trap rollback_calvin.py documents in reverse — a
"Calvin + temporal + Aug 29" filter would have restored 262 faces instead of
the 223 actually changed. These 170 ids were read from the NAS after
resolve_dups, so they are exactly the set in question.

Why at all: temporal matching measured ~4 percent accurate on the near-identical
2026-08-29 soccer shoot (strict 14/14 correct, temporal 1/26). The 26 STRICT
Calvin faces on 2026-09-12 are deliberately left alone.

Reversible: every (face_id, person_id, match_source) is snapshotted into
face_dedupe_undo first. Do NOT reverse with `restore-unmatched-faces` — that
restores every staged row, including ~9.6k from the June 2026 cleanup. Use
rollback_0912_calvin.py, which pins the same ids.

match_source is set to 'dedupe_unmatched' rather than NULL. That is what makes
the unmatch survive a replica face-state push (face_state.apply_face_state
skips those faces on an additive fill). NOTE it does NOT survive a direct
`match-faces --temporal` run: that selects on `person_id IS NULL` with no
marker filter, and would re-tag these faces.

  cat unmatch_0912_calvin.py | ssh cantimatt@192.168.1.237 \\
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
        "SELECT f.id, f.person_id, f.match_source, pe.name "
        "FROM faces f JOIN persons pe ON pe.id = f.person_id "
        f"WHERE f.id IN ({ph}) AND f.person_id IS NOT NULL",
        FACE_IDS).fetchall()

    print(f"{len(todo)} of {len(FACE_IDS)} pinned faces are still matched")
    by_src = {}
    for r in todo:
        k = (r["name"], r["match_source"])
        by_src[k] = by_src.get(k, 0) + 1
    for (name, src), n in sorted(by_src.items()):
        print(f"   {name} / {src}: {n}")

    # Refuse to touch anything that isn't what was reviewed. If a face has
    # since been manually corrected, it is no longer ours to unmatch.
    wrong = [r["id"] for r in todo
             if r["name"] != "Calvin" or r["match_source"] != "temporal"]
    if wrong:
        raise SystemExit(f"ABORT: {len(wrong)} pinned face(s) are no longer "
                         f"Calvin/temporal (e.g. {wrong[:5]}). Re-derive the ids.")

    if not APPLY:
        print("DRY RUN - nothing written. Set APPLY=1 to write.")
        raise SystemExit(0)

    c.execute("CREATE TABLE IF NOT EXISTS face_dedupe_undo ("
              "face_id INTEGER PRIMARY KEY, person_id INTEGER, match_source TEXT, "
              "unmatched_at TEXT DEFAULT (datetime('now')))")
    for r in todo:
        c.execute("INSERT OR REPLACE INTO face_dedupe_undo"
                  "(face_id, person_id, match_source) VALUES (?, ?, ?)",
                  (r["id"], r["person_id"], r["match_source"]))
    ids = [r["id"] for r in todo]
    ph2 = ",".join("?" * len(ids))
    c.execute(f"UPDATE faces SET person_id = NULL, match_source = 'dedupe_unmatched' "
              f"WHERE id IN ({ph2})", ids)
    db.conn.commit()
    print(f"Applied: unmatched {len(ids)} face(s). Reversible via "
          f"rollback_0912_calvin.py (NOT restore-unmatched-faces).")
