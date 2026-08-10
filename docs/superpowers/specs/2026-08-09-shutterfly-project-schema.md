# Shutterfly project-doc schema — Spike S2 capture (2026-08-09)

Captured live from the "sfly3 / Aurora" builder by copying a real book to a
throwaway project, adding a text box, and Saving while hooking `fetch`/XHR.
This is the **read/serialized** schema Phase 2 targets. (The throwaway was
deleted afterward.)

## Transport

- **Save fires** `PUT https://projects-api3.shutterfly.com/projects/v2/project/{projectId}`
  via `fetch`. Apigee-fronted: needs `X-API-Key` + OAuth bearer (present only in
  the logged-in page).
- **Observed caveat (important for the WRITE path):** the Save PUT's *request
  body was ~20 bytes*, yet it persisted the edit and its *response was the full
  ~395 KB project document*. So the builder does **not** save by PUTting the whole
  doc — content is synced to the server incrementally by other calls, and this
  PUT is a lightweight commit that returns the current doc. **Phase 2 should
  drive `builderApp`'s in-page data layer** (add photo to cell, set crop, add
  text via the app's own methods) and let it save — not hand-craft a doc and PUT
  it. The schema below is the target state to *construct and verify*, not
  necessarily a body to POST directly.

## Document shape

```
{
  partner, dataCenter, environment, accountHash, shardKey, accountId, sflyGuid,
  projectType, projectMetadata:[{name,metadataType,value}...], commerceSku,
  projectName, version, recordSeq, _id, state, ...,
  photoWell: [ … ],              // the project's photo library (see below)
  surfaceCategories: [
    { surfaces: [                 // grouped by category (cover set vs page spreads)
        {
          surfaceNumber, version, surfaceMetadata, renderingOutput,
          surfaceData: {
            pageDetails: { width, height, dpi, minDpi },   // surface size in PIXELS @ dpi (300)
            layeredItems: [ …elements… ]
          }
        }
      ] }
  ]
}
```

### Photo element (`layeredItems[i]`)

```
{
  type: "photo",
  container: { x, y, w, h, rot },        // the CELL FRAME on the surface (px @ dpi)
  content: {
    contentType: "UserPhoto",
    userData: {
      w, h, x, y, rot,                   // the IMAGE's placement inside the frame = the CROP
      assetId,                           // 156-char asset ref (== photoWell.assetRef)
      mediaId,                           // e.g. "1857799145358292" (== photoWell.id)
      journalCore, locationSpec          // locationSpec = Shutterfly-internal name (NOT original filename)
    }
  },
  layerMetadata: [ … ], date
}
```

- **container** = where the cell sits on the page. **content.userData {w,h,x,y}** =
  how the image is scaled/offset within that cell (the crop). Our photo-search
  `cell {x,y,w,h}` → `container`; our `crop_cx/cy/zoom` (via `book_export.plan_cell`)
  → `content.userData {w,h,x,y}`. All in px at the surface dpi (scale our 28×11-inch
  coords by dpi).

### Text element (`layeredItems[i]`)

```
{
  container: { x, y, w, h, rot },
  content: { userData: {
    assetId: "DA_46461",                 // text DESIGN/style asset (not a photo)
    fontSize, fontColor, alignmentAnchor, // e.g. 17, "#000000", "5"
    linesOfText: [ … ], userEditedText: true, isCustomDesigned: true,
    fullText: "SCHEMA TEST 12345"        // the caption text
  } }
}
```

### photoWell (project photo library)

```
photoWell: [
  { source:"shutterfly", url, thumbnailUrl, assetRef, height, width,
    id,                                  // mediaId
    ownerId, locationSpec }              // locationSpec = accountId+uploadTimestamp+ext, NOT original filename
]
```

## Mapping (photo-search photo_id → placed Shutterfly photo)

`photoWell` entries do **not** carry the original filename (only an internal
`locationSpec`). So the mapping is two hops:

1. **original filename → mediaId**: query the Shutterfly Photos *album* API
   (`photos3.shutterfly.com`, the per-book imported album) — the library exposes
   original filenames per photo, keyed by the same `id`/mediaId used in
   `photoWell`.
2. **mediaId → photoWell entry → assetRef**: then place a photo element
   referencing that `assetId`/`mediaId` with the computed `container` + crop.

(Recall: original filenames are unique within a book — see the Phase 1 plan's
Spike S1 findings — so step 1 is collision-free.)

## Open questions for the Phase 2 plan

- **Write path:** confirm how to programmatically apply changes — enumerate the
  incremental content-sync call(s), or (preferred) the `builderApp` data-layer
  methods to add/position photos + text, then trigger save.
- **Coordinate origin & bleed:** confirm origin corner and how bleed is included
  in `container` coords vs `pageDetails`.
- **surfaceCategories layout:** which category holds cover (front/spine/back) vs
  the page spreads, and how spread ↔ two-page surfaces map for layflat.
- **Text style asset:** how to choose the `DA_*` text design assetId + font to
  match the book's caption styling.
