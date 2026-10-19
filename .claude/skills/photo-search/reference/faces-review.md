# Reviewing faces in the UI

Read when helping review a shoot on `/faces`: person inspector, team review, More of this kid, Label health.

<!-- moved verbatim from CLAUDE.md (line 2869) on 2026-10-19 -->
## Reviewing a shoot in the UI: person inspector + team review

Two panels on `/faces`, both scoped to one day via the filter bar. `/faces`
round-trips the content filters (`date_from`/`date_to`/`location`/`q`/`camera`)
through the **URL**, so a shoot review survives a reload and can be bookmarked:

```
/faces?date_from=2026-09-12&date_to=2026-09-12
```

**Person inspector** — "X is tagged on other kids at Saturday's match". Opens
from a person group; sub-clusters that person's faces, scores each by L2
distance to their core, and bulk unassigns/reassigns. `GET
/api/faces/person/{id}/inspect` takes `date_from`/`date_to`. Two things that
are easy to get backwards:

- DBSCAN runs on the **scoped** faces (the groups are "the different kids that
  day"), but the reference core comes from the person's **full trusted set**.
  Scope the core too and you measure the suspect faces against themselves —
  when every trusted face is outside the window, the fallback makes the
  impostors the core.
- **The library's `eps=0.50` groups nothing on one day.** Measured on Calvin's
  212 faces from 2026-09-12: 0.50 → 0 clusters/212 noise, 0.65 → 16, 0.80 → 21,
  0.90 → 20 clusters/100 grouped, 1.00 → 14/148. A scoped call defaults to
  **0.90 / min_samples=2**; unscoped stays 0.50/3. The panel exposes the radius.

The panel states its scope ("showing 196 of 20,575") because it bulk-unassigns.

**Team review** (`⚽ Review team faces`) — the `review-faces` CLI as a panel,
offered only when `date_from == date_to`, since it learns **one** team colour
and a range would average two kits. `POST /api/faces/review-team` (SSE) runs
the same `photosearch/face_review.py` pipeline: sample the jersey colour below
each face, keep the ones matching the team, DBSCAN the survivors. Name a group
and "Assign all" → `POST /api/faces/bulk-assign`.

Measured on the real 2026-09-12 shoot: 1,767 faces over 761 photos, **267 s**
first run, learned hue 225°, 734 team / 490 other-team / 543 unreadable, **82
groups** (top ones 32, 27, 26, 23, 23) + 237 ungrouped.

- Previews are fetched via **`web._preview_bytes`**, NOT over HTTP. The NAS
  would otherwise fire ~800 requests at itself with six workers; and going
  through `_get_or_create_preview` means the replica reuses its on-disk preview
  cache, so a re-run is fast. It falls back to pulling from the NAS, which is
  the only reason this works on the replica at all (it holds no originals).
- **The hue is learned from faces already named that day** — so if those are
  wrong (a bad temporal pass), the filter is learned from garbage. Clean the
  day's labels first, or set the hue manually in the panel.
- **The "Ungrouped" bucket is not assignable.** It is DBSCAN noise — many
  different people — and naming it wholesale would be catastrophic.

<!-- moved verbatim from CLAUDE.md (line 3140) on 2026-10-19 -->
### Finding more of one person: `➕ More of this kid` on `/faces`

**Hand labels do not teach the strict matcher anything.** `match_faces_to_persons`
compares only against `face_references` (from `add-person --photo`); assigning
faces in `/faces` adds none. On 2026-09-26, after six kids were hand-labelled,
re-running `match_faces` matched **0** — correctly, as far as its inputs went.

Feeding the hand labels to the matcher's global 1.15 tolerance was measured and
rejected: on that shoot it would have written 793 labels, with a kid who had
**one** label absorbing 220. So `photosearch/face_suggest.py` +
`GET /api/faces/suggest-person` (date scope REQUIRED) + the panel rank the
scope's unmatched faces by distance to the person's trusted labels
(`manual`/`merge_review`, **library-wide** — Beckham had 94 from earlier shoots,
not today's 6) and **never write**: a cutoff slider dims tiles beyond it,
"Select ≤ cutoff" takes the rest, Assign goes through `bulk-assign`. Two flags,
both excluded from the cutoff selection: **nearer another kid** (distance to
every other person's trusted labels, library-wide) and **in photo** (the person
is already tagged in that photo). On the replica copy, 0.8–0.9 was the clean
band (Beckham 56→109, 0–2 rival flags); 1.0 starts collecting rival flags.
Tests: `tests/test_face_suggest.py`.

<!-- moved verbatim from CLAUDE.md (line 3161) on 2026-10-19 -->
### The high-level view: `📊 Label health` on `/faces`

Over-matching is invisible one date at a time — you only notice it when a grid
you happened to open is full of other people. `GET /api/faces/label-health`
ranks everyone by how much of a `match_source` they carry and names the days it
is concentrated on; the panel turns each day into a button that opens the
Bulk-unmatch grid for that person on that date. **Not gated on a date filter**,
unlike the other panels — being able to see it without picking a day first is
the entire point.

Measured 2026-09-16: **only 4 people carry any `temporal` labels at all** —
Calvin 6,331, Ellie 5,849, Nicole 1,829, Matt 794 (14,803 total). So "who else
needs this?" has a short, checkable answer.

| person | temporal | strict p50 | temporal p50 | bar | beyond |
|---|---|---|---|---|---|
| Calvin | 6,331 | 1.053 | 1.248 | 1.145 | 5,421 (86%) |
| Ellie | 5,849 | 1.136 | 1.256 | 1.243 | 3,517 (60%) |
| Nicole | 1,829 | 0.827 | 1.227 | 1.045 | 1,826 (99.8%) |
| Matt | 794 | — | — | — | **cannot calibrate: 0 manual refs** |

**The bar is the p90 of the person's own STRICT distances, not their manual
ones.** This is the trap: hand-made labels cluster in the sessions you happened
to label, so manual-to-manual distance is artificially tight, and the first
version of this used its p90 — which called **11,562 of Calvin's 11,669 strict
faces suspect**, flatly contradicting the crops. `strict` is the largest
verified-looking population a person has, so it is the honest yardstick for
"how far does this person legitimately land from their references".

**`calibratable: false` is a finding, not an omission.** Matt has 794 temporal
faces and zero manual labels, so nothing can judge him automatically; the fix is
to hand-label a few, and the panel says so in amber rather than dropping him.

Split by cost on purpose: the ranked list is **pure SQL over counts** so it
opens instantly, and the distance pass is **per person, on click**. Doing both
eagerly made it a minute-long request nobody opens twice.

**Do not generalise Calvin's result to the others.** His temporal was ~uniformly
wrong and a sweep was safe. Ellie's is 60% — the crops show several genuinely
her — and Nicole's 99.8% rests on only **18** manual references, so the bar is
weak. Those two want the per-face grid, not a blanket clear.
