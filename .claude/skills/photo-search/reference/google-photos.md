# Google Photos

Read before uploading to Google Photos.

<!-- moved verbatim from SKILL.md (line 677) on 2026-10-19 -->
## Google Photos Integration

<!-- moved verbatim from SKILL.md (line 679) on 2026-10-19 -->
### Scope & Limitations
- Only `photoslibrary.appendonly` scope is available (read/sharing deprecated March 2025)
- This is write-only: cannot list album contents, cannot read media items
- `batchAddMediaItems` returns 400 "invalid media item id" for photos removed from albums
  via Google Photos UI — the media_item_id becomes invalid for album operations
- Only option for re-adding removed photos: full re-upload of bytes
- Token stored in `google_photos_token.json` (not in the database)
- Client credentials in `client_secret.json` alongside DB

<!-- moved verbatim from SKILL.md (line 688) on 2026-10-19 -->
### Upload Flow
1. Raw bytes POSTed to Google → returns uploadToken
2. `batchCreate` with uploadTokens (batch size 50) → creates mediaItems with album assignment
3. Per-file SSE events: `start` → `begin` → `bytes_sent` → `progress` → `done`
4. Upload ledger tracks (album_id, filepath, media_item_id) per file
5. Selective re-upload: `force_reupload_ids` param targets specific photos

<!-- moved verbatim from SKILL.md (line 695) on 2026-10-19 -->
### Album ID Note
The API album ID (from `albums.create`) differs from the ID visible in Google Photos URLs.
Always use the API ID stored in the database.

---
