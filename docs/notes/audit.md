# Audit — per-photo change log

Condensed rule: `.claude/rules/search.md` ("`photo_audit` is the per-photo change log"). Code: `photosearch/audit.py`; table `photo_audit` (schema v40). Append new history under a new `###` heading; never rename an existing heading.

### Per-photo audit log (2026-10-30)

**Why.** A photo (250154) was removed from the live "Soccer Fall 2026 - All Players" collection by hand because its face was Calvin, not Koa. The collection was then rebuilt from a stale cached list and the photo went back in silently. The collection stores only current members, and `request_log.jsonl` records timing and paths but never bodies or photo ids, so nothing could say which photo had been removed.

**Design.**
- One append-only row per change in `photo_audit`: `ts` (UTC), `photo_id`, `action`, `field`, `old_value`, `new_value`, `collection_id`, `collection_name`, `face_id`, `detail`, `source`, `intent`. No foreign keys, so the history of a deleted collection or photo survives. Indexes: `(photo_id, id)` and a partial `(collection_id, id)`.
- Actions: `photo_added`, `photo_deleted`, `filepath_changed`, `field_changed` (`date_taken`, `gps_lat`, `gps_lon`, `place_name` via `update_photo`), `location_set`, `tags_set`, `collection_added`, `collection_removed`, `face_assigned`, `face_cleared` (a human "no", `match_source='rejected'`), `face_unmatched` (duplicate resolver, bulk-unmatch undo).
- Writes happen inside the existing db-layer writers, on the same connection and before `_maybe_commit()`, so a rolled-back change leaves no row and a committed one cannot lack it. Callers did not change except where they used raw SQL: the single and bulk face clear in `web.py`, `bulk_unmatch.apply/restore` and the merge-review accept now go through `db.set_face_labels`, which `assign_face_to_person` also uses.
- `delete_collection` logs a `collection_removed` (`field='collection_deleted'`) for every member before the cascade. A `BEFORE DELETE ON photos` trigger (`trg_photo_audit_delete`) logs `photo_deleted` plus a `collection_removed` (`field='photo_deleted'`) per collection the photo was in, because the `collection_photos` cascade, six CLI commands and the dedup sweep delete photos without calling a method. The trigger cannot see the audit context, so those rows say `source='db-trigger'`.
- **Who asked** comes from a context variable (`audit.context`), not arguments. The web middleware sets `source` from `request_intent.classify_source` (`ui`, `claude`, `worker`, `script`, `replica`, `other`) and the stated (`X-Photosearch-Intent`) or inferred intent. MCP calls set `claude-mcp` with the tool's `intent`; write tools add `tool <name>:` to the intent (`tools.call_tool`, also covers the Ask agent). The CLI group sets `cli` with the command name. Maintenance sweep stages set `maintenance`. Anything else (a thread that lost the context) records `internal`.

**Read path.** `GET /api/photos/{id}/audit`, `GET /api/collections/{id}/audit` (events plus `removed`: photos whose last event there is a removal and that are not members now; works after the collection is deleted), `GET /api/audit` (filters `photo_id`, `collection_id`, `collection_name`, `action`, `source`, `since`). A collection deleted and rebuilt gets a new id: query it by `collection_name` to see both lives. The photo modal has a lazy "Change history" section.

**Replica.** The NAS is the only writer and `sync-replica.sh` swaps the whole DB, so the table arrives with every nightly snapshot. Replica writes are mirrors of what the NAS already logged and would vanish at the next sync, so `audit.record` is a no-op when `PHOTOSEARCH_NAS_URL` is set or the request ran in replica mode (the middleware suppresses it). The replica's audit endpoints proxy to the NAS for fresh data and fall back to the local snapshot with `stale: true` if the NAS is unreachable.

**What is covered.** Collection add/remove/delete (all callers go through `PhotoDB`), photo add (`add_photo`) and delete (trigger), `set_photo_filepath` and `update_photo(filepath/date_taken/gps/place_name)`, `set_photo_location`, `set_photo_tags`, face labels via `set_face_labels` / `assign_face_to_person` / `unmatch_faces_as_duplicates`, and the maintenance inferred-GPS stage.

**What is NOT covered (known gaps).**
- Nothing before deploy: history starts empty and is not reconstructed.
- Machine output from the index and worker passes (`description`, scores, `categories`/`visual_tags`/`keywords` written by passes): `generations` already holds LLM provenance.
- Raw-SQL writers outside `PhotoDB`: `face_state` import/restore, `rerun` match_source stamps, `face_edit` `manual_box`, the cluster-only merge-review branch (cluster ids are ephemeral), `maintenance.repair_data` (date_taken/GPS repairs), recluster. Add a `photosearch.audit` call when touching them.
- Rename, description and cover changes of a collection (collection-level, not photo-level).
- Photo deletions report `source='db-trigger'` with no intent.
- The Ask/MCP tool surface was left unchanged (no read tool); Claude reads the history through the HTTP endpoints.
