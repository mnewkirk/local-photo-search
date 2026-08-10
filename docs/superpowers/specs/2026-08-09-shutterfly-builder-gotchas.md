# Shutterfly builder — reverse-engineering & driving gotchas

Hard-won notes for anyone driving the "sfly3 / Aurora" builder via a browser
automation tool (Chrome MCP `javascript_tool` + `computer`). These cost real
trial-and-error in the Phase 1/S2 work; read before the Phase 2 Task 1 spike.

## Capturing the project doc / save payload

- **The project loads via XHR (jQuery), saves via `fetch`.** Hook **both**, and
  match the URL loosely (`/projects/i` + `shutterfly|sfly`), not just
  `projects-api3`. A fetch-only hook misses the load; an XHR-only hook misses the
  save.
- **The Save PUT request body is ~20 bytes and the response is the full ~395 KB
  doc.** Content syncs incrementally; the PUT is a lightweight commit that returns
  the current doc. Capture the **response**, not the request. (This is why Phase 2
  drives `builderApp`, not a hand-built PUT.)
- **Direct replay of the project GET fails.** It's Apigee-fronted (needs
  `X-API-Key` + bearer); a cookie-only `fetch` returns 401. The builder's own
  auth is attached per-request in its data layer — easiest is to let the app make
  the call and capture the response, or drive `builderApp` in-page.
- **Trigger a save to capture:** the app only autosaves after an interaction.
  Passive waiting yields nothing. On a **throwaway** book, make a trivial edit
  (add a text box, type) then Save.

## Chrome MCP `javascript_tool` quirks

- **Async IIFEs frequently serialize back as `{}`.** Don't return the object from
  an `async (()=>{})()`. Instead: run the async work storing results to
  `window.__x`, return a sentinel string, then read `window.__x` in a **separate
  synchronous** call.
- **The tool redacts sensitive-looking output.** Auth header names/values and even
  key arrays come back `[BLOCKED: Sensitive key]`; any string containing a URL
  with a token/`?`/`=` can redact the **whole** result. Extract *structure*
  (keys, numeric geometry) and strip/omit URLs (`/https?:|token|\?|=/` → `<str:N>`)
  or the output vanishes.

## Editor UI / session gotchas

- **Sessions expire after ~30 min of inactivity** ("Your editing session has
  ended…" → Reload Project). Long reverse-engineering pauses will trip this.
- **Opening a real book flags "unsaved changes"** (on-open normalization) and the
  builder **autosaves**. To inspect a real book without mutating it: set
  `window.onbeforeunload = null` + swallow `beforeunload`, and **never click
  Save**. For anything that needs a save, use a **throwaway copy** and delete it
  after.
- **The Save button is a dropdown** (Save / Save as…) — clicking it opens a menu;
  click "Save" inside.
- **Project names can't contain special characters** — no `-`, `()`, etc. Use
  letters/digits/spaces (e.g. "ZZ SCRATCH schema capture delete me").
- **Copy → edit → Delete** is the safe throwaway loop; deletes go to "Recently
  Deleted" for 30 days.

## Photo identity across Google Photos → Shutterfly

- **The `sfly-<id>` upload rename does NOT survive Google Photos.** Google keeps
  the raw-upload `X-Goog-Upload-File-Name` (original basename) over
  `simpleMediaItem.fileName`. Map by **original filename** (unique per book).
- **Shutterfly's Google Photos import uses a "search"** and can pull in extra
  photos from *other* books' albums (observed: an album imported 192 vs 136).
  Always key off the target book's **known** filenames; ignore extras.
- **`photoWell` entries carry only an internal `locationSpec`, not the original
  filename.** Bridge original filename → `mediaId` via the per-book
  `photos3.shutterfly.com` **album** API, then `mediaId → photoWell → assetId`.

## Access boundaries (must be the user)

- Connecting Shutterfly to Google Photos is an **OAuth consent + Google sign-in** —
  the user does this, not the agent.
- Google Photos auth for uploads lives on the **NAS** (`google_photos_token.json`);
  the dev Mac's local token may be expired ("Google authorization has expired").
