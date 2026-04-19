# Google Photos import (M20)

Bring ~200K photos from Google Photos (Matt's phone + wife's phone) onto
the NAS, slotted into the existing `YYYY/YYYY-MM-DD/` layout, skipping
anything already present.

## API vs Takeout — decision

**Use Google Takeout. The API path is closed.**

As of **March 31, 2025** Google deprecated the third-party read scopes of
the Library API (`photoslibrary.readonly`, `.sharing`, `.edit.appcreateddata`).
Third-party apps can now only see media **they themselves uploaded**. The
existing `photoslibrary.appendonly` scope wired up in `google_photos.py` is
still fine for the upload direction, but it cannot enumerate or download the
user's library.

Implications:
- No API path can list, read, or download the 200K photos.
- Takeout (`takeout.google.com`) is the only sanctioned bulk-export route.
- Takeout also gives us the **JSON sidecars** (Google-authoritative metadata)
  that are richer than the EXIF in the exported files.

## Takeout mechanics worth knowing

Takeout for Google Photos produces:

- **Original files** (JPG/HEIC/MOV/MP4/RAW), but Google sometimes re-encodes
  on upload — the bytes you get back aren't always the bytes you uploaded.
- **`<filename>.json`** sidecars, one per photo, with the authoritative
  `photoTakenTime`, `geoData` (lat/lon/alt), `description`, `people`,
  `albums`, and `url`. The *EXIF* in the exported file can be missing
  or wrong; the JSON is the truth.
- **Album folder structure** in the archive: `Photos from YYYY/` for the
  firehose view plus per-album folders. We ignore the album folders for
  M20 import (future milestone could mirror Google albums into
  `collections`).
- **Known gotchas**:
  - Long filenames get truncated in archives. A file may be `IMG_12345.jpg`
    but its sidecar is `IMG_12345.json` *or* `IMG_12345.jpg.json` *or*
    `IMG_1234(1).jpg.json` — community Python tool `google-photos-takeout-helper`
    maps them correctly; worth borrowing its pairing logic rather than
    re-solving it.
  - Live photos land as separate `.HEIC` + `.MOV` pairs (no linkage).
  - Edited photos get both `IMG_XXXX.jpg` and `IMG_XXXX-edited.jpg` — we
    probably want the original, but it's per-taste.
  - Archives are split into 50GB chunks. For 200K photos, expect
    **15–25 chunks totaling 500GB–1TB**.
- **Delivery**: Takeout can ship to Google Drive or Dropbox, which is faster
  than downloading to a Mac and then rsync-ing. If the NAS has Drive mounted
  (or we use `rclone` on the NAS), the chunks can pull directly.

## Incremental Takeout is strongly recommended

Don't do one monolithic 1TB export — do it per year.

- Takeout lets you filter Google Photos by date before exporting
  (advanced settings → "All photo albums included" → pick per year).
- Per-year exports → per-year imports → per-year indexing. Each round
  is tractable and you can verify dedup is behaving before committing
  to the next batch.
- A reasonable order: newest year first (most likely to contain
  GPS-bearing phone photos that help **M19 inference** for older
  camera-only shots), then work backward.

## Dedup strategy

The existing SHA-256 `file_hash` (`index.py:25`) is **unreliable** against
Takeout output because Google re-encodes on upload. Comparing by byte hash
would force re-importing photos the user already has.

Composite signal instead — any two of these matching is a strong
"already imported" signal:

1. **photoTakenTime** (from JSON sidecar, falls back to EXIF) rounded to
   the second. Collisions are rare outside of bursts.
2. **Device/camera model** (EXIF `Make`+`Model`). Rules out the case where
   two family members happened to shoot at the same second.
3. **Filename stem** (`IMG_12345` from both sources after stripping
   `-edited`, `(1)`, etc.). Phones tend to preserve filename across uploads.
4. **Approximate GPS** — lat/lon within ~100m when both have it.
5. **File size** within ±10% (loose, since re-encoding changes size).

Implementation: before importing a Takeout photo, query:

```sql
SELECT id FROM photos
WHERE date_taken = ?
  AND (camera_model = ? OR filename LIKE ?)
LIMIT 1
```

If a hit, skip. Otherwise import. This is O(log N) with the existing
`idx_photos_date` index.

**Perceptual hash as a tiebreaker** — add a `photos.phash` column
(16-char blockhash or dHash) populated at index time. O(ms) per photo.
Lets us catch the edge case where metadata differs but the image is
visually identical (Google's HEIC→JPG re-encode with timezone drift on
the EXIF date). Probably worth shipping in the same schema bump.

## Folder layout

Land imported photos under:

```
/photos/YYYY/YYYY-MM-DD_gphotos/<original_filename>
```

The `_gphotos` suffix on the date folder keeps the origin clear and makes
rollback trivial (`rm -rf` a day's import). `find_photos()` already globs
`YYYY/YYYY-MM-DD*/`, so no changes needed to discovery.

Rationale for not mixing into the plain `YYYY-MM-DD/` folder the user's
camera photos live in: two devices shooting the same day is already a
collection-management nuisance; keeping sources physically separate means
stacking (M14) doesn't have to deduplicate across sources.

## Pipeline

```
Takeout zip chunks
    │
    ▼
[1] unpack  → staging dir `/data/takeout_staging/<YYYY>/`
[2] pair    → walk tree, match each media file to its JSON sidecar
              (borrow pairing logic from google-photos-takeout-helper)
[3] plan    → for each pair, compute (date, camera, stem, lat, lon),
              check composite dedup against photos table,
              write a per-file `plan.ndjson` with action=import|skip|conflict
[4] review  → CLI prints summary (N new, N skipped, N conflicts).
              User eyeballs a few conflicts before committing.
[5] commit  → move files to `/photos/YYYY/YYYY-MM-DD_gphotos/`,
              also write the JSON sidecar next to it (future-proof —
              `people` and `description` from Google could be useful).
[6] index   → normal `photosearch index /photos/YYYY --full` pass
              (or worker fleet for scale).
```

Step 3 writes an ndjson plan rather than acting immediately so it's
reviewable, resumable, and cancel-safe — matches the ledger pattern the
project already uses for uploads and faces.

## Scale (200K photos)

- **Storage**: ~1TB. Confirm NAS disk headroom before starting.
- **Indexing time**: at ~1000 photos/hr/worker for CLIP on N100 CPU,
  200K photos = **200 worker-hours**. Use the Docker worker fleet
  (`./run-workers.sh -n 4`) = ~50 hours of wall time. Faces: similar.
  Descriptions/tags with LLaVA: *much* longer (30–200s/photo) — consider
  leaving descriptions off for this batch and running them opportunistically.
- **JSON sidecar parse**: cheap, ~30 min total.
- **Dedup SQL queries**: ~200K queries × ~1ms each = 3–5 min. Fine.

A realistic rollout:
1. Per-year Takeout → transfer to NAS (hours to days depending on pipe).
2. Import + dedup pass (minutes to hours per year).
3. Launch indexing (CLIP + faces + quality) with worker fleet, run
   overnight. Skip descriptions initially.
4. Re-run **M19 inference** after each year — newly-imported GPS-bearing
   phone photos give the camera photos nearby-in-time neighbors they
   didn't have before.

## CLI sketch

```
photosearch takeout-import /path/to/takeout-unzipped \
    [--year 2024] \
    [--dry-run] \
    [--plan-out /data/takeout_plan_2024.ndjson] \
    [--apply /data/takeout_plan_2024.ndjson] \
    [--dest-root /photos]

# Typical flow:
photosearch takeout-import /volume1/takeout/2024 --year 2024 --dry-run \
    --plan-out /data/takeout_plan_2024.ndjson
# review summary + spot-check a few conflicts
photosearch takeout-import --apply /data/takeout_plan_2024.ndjson
```

Separate from (but pairs with) the existing `google-photos` command family
in `cli.py` that handles the **upload** direction. Import is a new command
to keep intent explicit.

## Schema changes

Small additions to `photos`:

```sql
ALTER TABLE photos ADD COLUMN phash TEXT;               -- perceptual hash
ALTER TABLE photos ADD COLUMN import_source TEXT;       -- NULL | 'takeout' | 'upload'
ALTER TABLE photos ADD COLUMN google_photo_id TEXT;     -- from JSON sidecar, if available
CREATE INDEX IF NOT EXISTS idx_photos_phash ON photos(phash);
```

`google_photo_id` lets us short-circuit dedup if the same Takeout is
re-imported, and (future) could let us link back to the Google Photos
UI from the web frontend.

## Open questions

- **Keep or drop edited versions?** Takeout includes both `IMG_1234.jpg`
  and `IMG_1234-edited.jpg`. Defaulting to "import both, stack them via
  M14 extension" gives the user both views. Simpler: "import only the
  edited version, discard original" mirrors what they see in Google Photos.
- **Live photos**: import the `.MOV` component or drop it? Current library
  already ingests MOV files but treats them as independent. A future
  refinement could pair HEIC + MOV into a new "live photo" stack type.
- **Duplicates across the two phones.** Matt and wife at the same event
  often produce near-identical shots from both phones. The composite
  dedup signal (device model) intentionally **keeps both** because they
  aren't the same photo. This is probably right but worth confirming.
- **Google's `description` and `people` fields** from the sidecar — worth
  importing? `description` could seed `photos.description` so we skip
  LLaVA on those. `people` names don't map cleanly to our `persons` table
  without the user's say-so, so probably defer.
- **Resumability.** If an import crashes mid-batch, we need a ledger so
  the next run picks up where the last left off. The per-file plan
  ndjson + a `takeout_imports` table keyed on `google_photo_id` should
  cover it.

## Related

- Enables **M19 inference** (geotag backfill) at scale — phone photos have
  GPS; camera photos don't. Importing phones first gives inference the
  neighbors it needs.
- Feeds into the **map view** downstream work in the location milestone —
  200K GPS-bearing phone photos turn the map from sparse to dense.
- Complements the existing **upload** direction (`google_photos.py`): Google
  becomes both a source and a destination, with this project as the
  canonical local store.
