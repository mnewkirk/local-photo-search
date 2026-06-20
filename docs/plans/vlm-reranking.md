# VLM Re-ranking — "find THE photo" (M27)

**Status:** Planned (design doc). Not started. Next milestone after the
`summarize` faceting tool. Captured 2026-06-20.

## 1. Motivation

The retrieval stack (CLIP + structured filters + RRF) has good **recall** but
weak **precision** for "find THE specific photo" requests:

> "the one where Ellie is blowing out her birthday candles"
> "the photo where Calvin scored the goal"
> "the group shot where everyone's actually looking at the camera"

CLIP ranks by coarse visual/semantic similarity and the filters narrow by
metadata, but neither *looks* at the image to verify the specific moment. The
agent bake-off showed this: `search_photos(people=['Ellie'], keyword='birthday')`
returns ~20 plausible candidates, but the *exact* candle-blowing frame is buried
and there's no way to pick it out.

**Fix:** re-rank the top-K candidates with a vision model that actually looks at
each image and scores it against the request. This is the `get_photo_image`
path turned into a deliberate re-ranking stage — cheap retrieval for recall,
expensive vision pass for precision, only on the final short list.

## 2. Where it fits

Pipeline:

```
search_photos / summarize        →  top-K candidate ids (K≈20-40)
        ↓
rerank_photos(ids, criteria)     →  per-photo {score 0-1, reason}, re-sorted
        ↓
agent answers with the top few   (UI shows them)
```

**Shape decision — a separate tool, `rerank_photos(photo_ids, criteria)`**
(recommended), not a flag on `search_photos` nor an automatic loop stage:

- (a) **Separate tool** ✅ — explicit and composable: the agent runs a normal
  search, *sees the count*, and chooses to rerank only when the user wants a
  specific shot. Naturally MCP-exposable. Keeps the cheap path cheap. The agent
  already decides when precision matters (it has the prompt).
- (b) Flag on `search_photos` — couples a slow vision pass to every search;
  hard to scope K and criteria independently; muddies the cheap path.
- (c) Automatic agent-loop stage — fires even when unwanted, burns latency on
  broad-browse queries, and removes the agent's judgment.

So: `rerank_photos` takes an explicit candidate id list (from a prior search the
agent already ran) plus a free-text `criteria` describing what to look for,
returns the ids re-scored and re-sorted. The system prompt teaches *when* to
reach for it.

## 3. The vision-model call

**Per-image, not batched.** Each candidate is one vision request: the
thumbnail + a focused prompt asking "does this photo match: <criteria>? Score
0.0-1.0 and give a one-line reason." Rationale:

- Batching many images into one request blows the context window (each 600px
  data-URI is large) and degrades attention/accuracy per image.
- Per-image is embarrassingly parallel against the local GPU and gives clean,
  independent scores. Cap concurrency to the LM Studio slot count.

**Building the request:** reuse the existing multimodal plumbing in
`describe.py` — `_image_ref_to_b64()` (resize + base64) and `_to_openai_message()`
(wraps as OpenAI `image_url` data-URI content). The rerank client is an OpenAI
`/chat/completions` call to `PHOTOSEARCH_TEXT_LLM_URL` with a vision message.

**Model — needs a VISION model, by role.** The agent's pick (qwen3.5-9b) is the
text planner; reranking needs a vision-capable model. Route by the existing
`visual` role: `PHOTOSEARCH_LLM_VISUAL_MODEL` (mirrors describe.py's
`_resolve_openai_model(role="visual")`). Requires a vision model loaded in LM
Studio (qwen-VL / gemma-vision / llava-class). If none is configured, the tool
returns a clear "no vision model" error and the agent falls back to the
CLIP/RRF order.

**Structured output:** force a tiny JSON object per image — `{"score": 0.0-1.0,
"reason": "..."}` — via a strict prompt (and JSON mode where the backend
supports it). Parse defensively; on unparseable output, score it 0 with a note
(don't crash the batch).

**Prompt design:** keep it discriminating, not generous — "Score how well THIS
photo matches the request. 1.0 = clearly the described moment; 0.0 = unrelated.
Be strict; most candidates are near-misses." Include the criteria verbatim.

## 4. Cost / latency + gating

K images × one vision call each is the expensive part (≈1-4 s/image on the local
GPU). Controls:

- **Opt-in by design** — only invoked when the agent judges the user wants a
  specific shot (the tool's existence + a prompt rule). Broad-browse queries
  never pay for it.
- **Cap K** (default ~24, hard max ~40). The agent passes a pre-narrowed list;
  rerank is the *last* stage, not a scan.
- **Early-exit / threshold** — optionally stop once N photos clear a high score,
  or return only those above a floor.
- **Concurrency** bounded to the LM Studio loaded-slot count.
- **Privacy:** fully local (vision model on the GPU box, thumbnails from the
  local replica or NAS proxy — nothing leaves the network). The MCP path still
  honors `PHOTOSEARCH_MCP_ALLOW_IMAGES`: if image returns are disabled, the
  text-result `rerank_photos` (scores + reasons, no pixels out) is fine, but any
  variant that would emit pixels stays gated.

## 5. Integration points (minimal code surface)

- **`photosearch/tools.py`** — new `rerank_photos` ToolSpec + handler: resolve
  each id → thumbnail path (the existing `_thumb_path` helper / NAS proxy on a
  replica) → vision call → return `[{id, score, reason}]` sorted desc, plus the
  compact hit fields so the UI can render. Schema: `photo_ids: int[]`,
  `criteria: string`, optional `top_n`, `min_score`.
- **`photosearch/agent.py`** — one system-prompt rule: "When the user wants a
  *specific* photo ('the one where…', 'the exact shot of…'), first
  `search_photos` to get candidates, then `rerank_photos(those ids, criteria)`
  and answer with the top results." Keep it off for broad/browse requests.
- **Vision client** — a small helper (in `tools.py` or a new `rerank.py`)
  reusing describe.py's `_image_ref_to_b64` / OpenAI message shape; or factor
  describe's `_openai_chat_with_retry` so both share it. Model via the `visual`
  role env.
- **Thumbnail access** — on the NAS, `_thumb_path` reads local files; on a
  replica, reuse the `_fetch_from_nas` proxy so rerank works there too.
- **MCP** — `rerank_photos` is a normal tool, exposed automatically via
  `mcp_tools()`; gating note in §4.

## 6. Eval plan

Extend the bake-off harness (`evals/bakeoff.py`, or a sibling
`evals/rerank_eval.py`) with **precision-oriented prompts** that have a known
single correct photo or a tiny correct set:

- "find the photo of <person> blowing out birthday candles"
- "the shot where <person> is mid-jump"
- "the one group photo where everyone is looking at the camera"

Metric: does the correct shot land in the **top 1 / top 3** *after* rerank vs.
the CLIP/RRF baseline order (run both, compare). Reuse the HTML-snapshot report
so the win is visible — baseline strip vs reranked strip side by side. Seed a
few ground-truth ids by hand from the library.

## 7. Risks / open questions

- **Vision model availability + quality in LM Studio** — the single biggest
  dependency. Which vision model is loaded, and is it good enough at
  fine-grained "is this the candle moment" judgments? Needs a real bakeoff of
  vision models (separate from the text-agent one).
- **Latency budget** — K×vision is seconds-to-tens-of-seconds; is that
  acceptable interactively, or should rerank stream partial results / show a
  progress affordance in the Ask UI?
- **Thumbnail (600px) vs preview (1920px)** for reranking — 600px is cheaper and
  usually enough; fine detail (jersey numbers, faces in a crowd) may need the
  preview. Make the source size configurable; default thumbnail.
- **Score calibration** — vision scores may bunch up; consider relative ranking
  within the candidate set rather than absolute thresholds, and how rerank
  scores compose with (override? blend with?) the existing RRF order.
- **Whether to also feed the description** — giving the model the stored LLaVA
  description alongside the image may help or may bias it; test both.

## 8. Sequencing

1. Vision-model bakeoff: pick `PHOTOSEARCH_LLM_VISUAL_MODEL` (reuse the eval
   harness pattern with image prompts).
2. `rerank_photos` tool + vision client (per-image, structured score).
3. Agent system-prompt rule + MCP exposure (free via the shared layer).
4. Precision eval (top-1/top-3 vs baseline) + HTML report.
5. UI: surface a "find the exact one" affordance / progress in the Ask flow.

## Related plans
- `docs/plans/search-improvement-backlog.md` — this is backlog item #2.
- `docs/plans/llm-driven-search.md` — M24 agent + shared tool layer (the base).
- `docs/plans/local-replica-and-writes.md` — M26 replica/image-proxy (thumbnail
  access for rerank on the GPU box).
