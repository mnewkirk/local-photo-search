# Handoff — re-run the 2026-09-12 soccer "Best 50 / Next 200"

Written to resume cleanly in a fresh session. Everything below is **verified on
2026-09-18** by running it, not recalled. Commands are the ones I actually ran.

---

## The ask

Re-do the Best 50 / Next 200 / All 250 curation for the **2026-09-12** soccer
shoot, now that the face labels behind it are trustworthy.

## Two inputs changed, not one

### 1. The face labels were largely noise before

`scripts/rank_shoot.py` uses face labels for its per-person coverage rules
(`--min-per-person`, `--max-per-person`) and for its `crowd` action proxy. It
reads `person_id` **with no `match_source` filter** (`rank_shoot.py:179`), so
whatever was labelled is what it used.

When collections 32–37 were built, this date carried **162 `temporal` Calvin
matches** — a source measured **~4% accurate** on these shoots. So "at least 3
frames of every person" was partly guaranteeing coverage of people who were not
in the frame.

Authoritative NAS state today (`441` labelled faces on the date):

| source | count |
|---|---|
| `manual` | 395 |
| `strict` | 26 |
| `temporal` | **20** (was 162 for Calvin alone) |
| `rejected` — a human "no" | 136 |

Trusted labels now spread evenly across 15 players: Carson Jones 49, Finn Coupar
45, Adrian Islas 45, Lucas Alaniz 42, Robert O'Brien 39, Beckham Tinnel 39, Asa
Goldman 32, **Calvin 29**, Zack Azerad 21, William Daly 20, Oliver Munoz 18,
Levon Cirillo 13, Franklin Martinez 13, Jack O'Rourke 11, Justin Tinnel 5.

### 2. VLM aesthetics is now fully scored

`rank_shoot` refuses `aes_overall` below `--min-aes-coverage` (0.80): a partial
pass covers whichever photos the fleet claimed first, so ranking on it would
silently favour that arbitrary subset. The old run fell back to the compressed
LAION score. **Coverage is now 1020/1020**, and the run confirms the switch in
its first line. It is 10% of the score (`sharp .50, size .22, det .13, aes .10,
crowd .05`, `rank_shoot.py:227`), so expect a nudge, not a reshuffle.

---

## The result that matters — I ran the dry run

```
919 scored photos in 330 bursts (gap 1.0s); aesthetics coverage 100% -> VLM aes_overall

Best 50 — person coverage:
    Adrian Islas 8 · Finn Coupar 7 · Calvin 7 · Lucas Alaniz 6 · Asa Goldman 5
    Robert O'Brien 4 · Franklin Martinez 4 · Zack Azerad 4
    Justin Tinnel / Oliver Munoz / Jack O'Rourke / William Daly /
    Carson Jones / Levon Cirillo / Beckham Tinnel — 3 each
```

**Uncapped, Calvin now takes 7 of 50 — he took 16 before.** That is the entire
reason collections 35–37 exist (`--max-per-person 6`). The distortion was the
bad labels, not the ranking, so **the cap is probably no longer needed** — every
one of the 15 players clears the ≥3 floor and the spread is 3–8. Decide by
looking at both runs, but the uncapped one is now a reasonable default.

Top of the Best tier, for orientation:

```
DSC08503.JPG  0.914  sharp .87 size .94 det .97 aes .99 crowd .97  [Adrian Islas, Finn Coupar]
DSC08572.JPG  0.878  sharp .88 size .99 det .79 aes .68 crowd .98  [Calvin, Robert O'Brien]
DSC07959.JPG  0.874  sharp .88 size .92 det .98 aes .58 crowd .94  [Adrian Islas, Justin Tinnel]
```

---

## Verified state

```
photos on 2026-09-12 ............ 1,020   (919 of them have faces)
faces ........................... 2,672
aes_overall / laion / description  1,020 each (100%)
face crops warmed ............... 2,663 generated 2026-09-16, 0 errors
sharpness cache ................. /data/rank_shoot_2026-09-12.json
                                  242,962 bytes · 919 photos · Sep 14
bursts at --burst-gap 1.0 ....... 330   (250 slots need >=250, so it fits)
```

**The expensive phase is done and still valid.** The cache is keyed
`photo_id -> {face_id: {lap, edge, area_frac}}` and measures *pixels*, not
labels — unmatching faces changed `person_id`, never `face_id` or the image.
Verified: `--measure` reports `0 left to measure`. Do **not** delete it to
"start clean"; that is ~6 min of full-res decodes on the N100 for an identical
answer.

### Existing collections — do not overwrite

| id | name | n |
|---|---|---|
| 32–34 | `Soccer Game - 2026-09-12 - Best 50 / Next 200 / All 250` | 50 / 200 / 250 |
| 35–37 | `Soccer Game - 2026-09-12 (capped) - …` (`--max-per-person 6`) | 50 / 200 / 250 |

Use a fresh `--label` (e.g. `"Soccer Game - 2026-09-12 (v2)"`) so the new set
sits beside them and can be compared before anything is deleted.

---

## How to run it

The DB is **NAS-authoritative** and `--apply` writes collections, so run it
there. `scripts/` is **not** in the Docker image — the repo is bind-mounted at
`/repo`, so the script path is `/repo/scripts/rank_shoot.py`. (`--db` defaults
to the container's `PHOTOSEARCH_DB`.)

```bash
ssh cantimatt@192.168.1.237
cd /volume1/docker/photosearch
DC="docker compose -f docker-compose.nas.yml run --rm --entrypoint python photosearch"

# 0. confirm the cache (expect "0 left to measure")
$DC /repo/scripts/rank_shoot.py --date 2026-09-12 --measure

# 1. preview uncapped — this is the run quoted above
$DC /repo/scripts/rank_shoot.py --date 2026-09-12

# 2. preview capped, to compare the histograms
$DC /repo/scripts/rank_shoot.py --date 2026-09-12 --max-per-person 6

# 3. create the collections
$DC /repo/scripts/rank_shoot.py --date 2026-09-12 \
      --label "Soccer Game - 2026-09-12 (v2)" --apply
```

**Review on the replica** (`localhost:8001/collections`) — it has the warmed
crops and previews. Sync first, or you review yesterday's labels:

```bash
./sync-replica.sh          # ~200 s, 2.0 GB
```

At the time of writing the replica was ~2.8 h behind. Check with
`GET /api/admin/replica-status`.

---

## What to decide

1. **Drop the cap?** Calvin is at 7/50 uncapped against a 3–8 spread. Run both
   and compare, but the reason for the cap has gone away.
2. **Is `--min-per-person 3` meaningful now?** It always ran; on the old labels
   it partly guaranteed coverage of mis-tagged faces. With 15 players at 5–49
   trusted faces each it should now do what it says.
3. **Keep `--burst-gap 1.0`.** The data forces it: 1 s gives 330 bursts, 2 s
   gives ~200 — and "no two frames from one burst" makes 250 photos impossible
   below 250 bursts. The script warns if you starve it.

---

## Traps — verified, do not rediscover

- **`rank_shoot` does not filter `match_source`.** 20 `temporal` labels remain
  on this date — few enough not to matter now, but if a `match-faces` sweep
  re-inflates them the per-person rules degrade *silently*. Check
  `GET /api/faces/label-health` before any re-run.
- **Face-crop Laplacian must come from the ORIGINAL pixels.** A preview or a
  cached 200 px crop has already discarded the high-frequency signal that is the
  entire ranking. That is why measuring is slow and why the cache matters.
- **A partial VLM aesthetics pass is worse than none.** 100% today; if new
  photos land on the date it will fall back to LAION rather than rank on a
  biased subset. Trust the log line, not memory.
- **Bursts come from timestamps, not `photo_stacks`.** Scoped stacking has twice
  wiped the library's stacks; `rank_shoot` deliberately does not depend on them.
- **`crowd` is a proxy for action, not a detector** — 0.05 weight on purpose.
  There is no real action signal without the LLM passes.
- **Never `--apply` on the replica.** `sync-replica.sh` swaps `PHOTOSEARCH_DB`
  wholesale and would destroy the collections on the next sync.
- **Don't `pgrep -f` a long job to watch it** — the watcher matches itself. Use
  a marker file or the PID (see CLAUDE.md).

---

## Why the labels got better (context, not to-do)

Shipped and deployed to both machines, `ef921c9`. Full write-up in CLAUDE.md
under "Face-label integrity".

- **The rival test** — another face in the same photo claims a label better, so
  the label is on the wrong face. Catches what the margin test cannot, because
  the rival may be *unlabelled*. `--rivals-only` plus a one-click **Swap**.
- **`verify-face-labels` now defaults to trusted sources** — `temporal` was
  burying the real findings 26:1 (108 findings → 4).
- **`unmatch-person` CLI + `✂ Bulk unmatch` panel** — clear a bad
  `match_source` for one person from a farthest-first grid with multi-select.
  Writes `rejected` so auto-matching cannot undo it; `--snapshot` is the durable
  undo.
- **`📊 Label health`** — library-wide view of who carries a suspect source.
- **`Warm face crops`** card on `/admin/maintenance`, plus `--date-from/--date-to`.

### Still open in the same area

- **Ellie (5,849 temporal) and Nicole (1,829)** are not cleaned. Do **not** copy
  Calvin's approach: Ellie's is 60% beyond her bar and the crops show many
  genuinely her; Nicole's 99.8% rests on only **18** manual references. Use the
  per-face grid.
- **Matt (794 temporal, 0 manual labels)** cannot be calibrated at all.
  Hand-label a handful first, then measure.
- **2026-08-29** has the same shape (223 Calvin temporal; crops warmed
  2026-09-16) and its collections 29–31 were built on the same bad labels —
  the same re-run applies there once its labels are cleaned.
