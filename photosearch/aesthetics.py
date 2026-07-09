"""VLM-based aesthetic scoring — the rich, per-attribute replacement for the
LAION CLIP quality predictor (`quality.py`).

The LAION predictor writes a single `aesthetic_score` whose 1–10 scale is
inherently compressed (best library photo ~6.2; a 25k curated-Unsplash sample
spanned only 3.58–6.92). This module instead asks a vision LLM (qwen*-VL via the
existing Ollama / LM Studio routing in `describe.py`) for a structured critique:

  * three top-level dimensions — Technical Excellence, Composition,
    Impact & Storytelling — each the mean of its sub-attributes,
  * a short per-dimension critique (reasoning-before-number, which improves
    score calibration),
  * a free-text style critique (lighting / mood / tonal character / color /
    composition notes) that is searchable,
  * a small set of controlled style tags for faceted filtering.

`aes_overall` is the weighted mean of the three dimensions. The compressed-scale
problem is fixed downstream by percentile-normalizing `aes_overall` across the
library (`normalize-aesthetics` CLI → `aes_overall_pct`), so the best photo
reads as top-tier regardless of the raw model's central tendency.

The VLM number is not fully trusted for the technical axis; the plan allows an
optional objective IQA anchor (MUSIQ/TOPIQ via `iqa.py`) to be blended in — see
`compute_overall`'s `technical_override` hook.

Nothing here touches `quality.py`; the LAION score is kept as a cheap prior.
"""

import json
import re
from pathlib import Path
from typing import Optional

# The scoring taxonomy. Each dimension is the (equal-weight) mean of its
# sub-attributes; the overall is the DIMENSION_WEIGHTS-weighted mean of the
# three dimensions. Weights live here so they can be retuned and re-applied to
# already-scored photos via `recompute-aesthetic-overall` — no VLM re-run.
DIMENSION_SUBATTRS: dict[str, list[str]] = {
    "technical": ["sharpness", "exposure", "depth_of_field", "white_balance"],
    "composition": ["framing", "leading_lines", "rule_of_thirds", "balance"],
    "impact": ["emotion", "originality", "wow"],
}
DIMENSIONS: list[str] = list(DIMENSION_SUBATTRS.keys())
ALL_SUBATTRS: list[str] = [s for subs in DIMENSION_SUBATTRS.values() for s in subs]

# Impact / "wow" is weighted highest — a technically perfect but forgettable
# frame should not outrank a striking one. Retunable; must sum to 1.0.
DIMENSION_WEIGHTS: dict[str, float] = {
    "technical": 0.30,
    "composition": 0.30,
    "impact": 0.40,
}

# Free-text style facets (searchable prose, one short phrase each).
STYLE_FACETS: list[str] = [
    "lighting", "mood", "tonal_character", "color", "composition_notes",
]

# The role name for `describe._ollama_chat_with_retry(role=...)` — resolves the
# LM Studio model via PHOTOSEARCH_LLM_AESTHETICS_MODEL.
AESTHETICS_ROLE = "aesthetics"

# Default vision model when routing through Ollama (LM Studio overrides by role).
MODEL = "qwen2.5-vl"

# Human-readable one-liners describing each sub-attribute, embedded in the
# prompt so the model scores the intended thing.
_SUBATTR_HELP: dict[str, str] = {
    "sharpness": "critical focus on the subject; micro-contrast; absence of unwanted blur",
    "exposure": "correct tonal range; highlights and shadows retained, not clipped",
    "depth_of_field": "intentional, well-controlled focus plane / subject separation",
    "white_balance": "accurate, pleasing color temperature; no unwanted cast",
    "framing": "deliberate crop and edges; nothing important cut; clean margins",
    "leading_lines": "lines/shapes that guide the eye toward the subject",
    "rule_of_thirds": "strong placement of the subject on thirds / dynamic points",
    "balance": "visual weight distributed well; no awkward empty or crowded areas",
    "emotion": "emotional resonance; does it make the viewer feel something",
    "originality": "a fresh perspective, moment, or treatment vs a generic snapshot",
    "wow": "the immediate 'wow' / gallery-worthy factor; overall striking impact",
}

# Rubric anchors — the single most important lever against VLM central-tendency
# (the VLM's own version of the LAION compression). Every score band is anchored
# so the model uses the full range instead of clustering at 6–8.
_RUBRIC = """\
Score each attribute as an INTEGER from 1 to 10 using this rubric:
  1-2  = poor / clearly failed (out of focus, badly exposed, no thought)
  3-4  = below average / snapshot with obvious weaknesses
  5    = average — an ordinary competent photo (MOST photos are here)
  6-7  = good — well executed, a clear strength on this attribute
  8-9  = excellent — professional, striking, near-flawless
  10   = exceptional — the best you would ever expect to see
Most everyday photos are a 5. Reserve 9-10 for genuinely exceptional work and
1-2 for genuine failures. Do NOT cluster every score at 6-8 — spread them."""


def build_aesthetics_prompt(style_vocab: Optional[list[str]] = None) -> str:
    """Assemble the aesthetics prompt. Asks for a compact JSON object: a short
    critique then integer sub-scores per dimension, plus style facets + tags.

    Kept terse on purpose — the OpenAI route caps output at ~768 tokens, so
    critiques must be one brief phrase each to avoid truncating the JSON.
    """
    lines = []
    for dim in DIMENSIONS:
        subs = DIMENSION_SUBATTRS[dim]
        pretty = {
            "technical": "TECHNICAL EXCELLENCE",
            "composition": "COMPOSITION",
            "impact": "IMPACT & STORYTELLING",
        }[dim]
        lines.append(f"- {pretty}: " + "; ".join(
            f"{s} ({_SUBATTR_HELP[s]})" for s in subs))
    attrs_block = "\n".join(lines)

    # Build the JSON skeleton the model must fill in.
    def _dim_obj(dim: str) -> str:
        subs = ", ".join(f'"{s}": <1-10>' for s in DIMENSION_SUBATTRS[dim])
        return f'"{dim}": {{"critique": "<one brief phrase>", {subs}}}'

    skeleton = (
        "{"
        + ", ".join(_dim_obj(d) for d in DIMENSIONS)
        + ', "style": {'
        + ", ".join(f'"{f}": "<one brief phrase>"' for f in STYLE_FACETS)
        + '}, "style_tags": ["<tag>", ...]}'
    )

    vocab_line = ""
    if style_vocab:
        vocab_line = (
            "\nFor style_tags, choose only from this list (include every tag "
            "that clearly applies, omit the rest):\n" + ", ".join(style_vocab))

    return (
        "You are an expert photography critic. Evaluate this photograph across "
        "the attributes below.\n\n"
        f"{attrs_block}\n\n"
        f"{_RUBRIC}\n\n"
        "Then describe its visual style: lighting, mood, tonal character, color, "
        "and composition notes — one brief phrase each."
        f"{vocab_line}\n\n"
        "Respond with ONLY a single JSON object, no prose before or after, in "
        "exactly this shape (integers for scores):\n"
        f"{skeleton}"
    )


def _extract_json(raw: str) -> Optional[dict]:
    """Pull the first balanced JSON object out of a model response.

    VLMs frequently wrap JSON in ```json fences or add a sentence before/after.
    We scan for the first '{' and walk braces to the matching '}', ignoring
    braces inside strings, then json.loads that slice.
    """
    if not raw:
        return None
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except (ValueError, json.JSONDecodeError):
                    return None
    return None


def _coerce_score(v) -> Optional[float]:
    """Coerce a model-supplied score to a float clamped to [1, 10]; None if not
    a usable number (e.g. "N/A", empty, out-of-range junk)."""
    if isinstance(v, bool):  # bool is an int subclass — reject it
        return None
    if isinstance(v, (int, float)):
        f = float(v)
    elif isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        if not m:
            return None
        f = float(m.group())
    else:
        return None
    if f != f:  # NaN
        return None
    return max(1.0, min(10.0, f))


def parse_aesthetics_response(raw: str) -> Optional[dict]:
    """Parse a raw VLM response into the normalized aesthetics dict, or None if
    it can't be trusted (missing/invalid scores) so the caller can retry/defer.

    Returns:
        {
          "sub_scores": {<subattr>: float, ...},   # all 11, in [1,10]
          "dim_scores": {<dim>: float},            # equal-weight mean of subs
          "overall": float,                        # DIMENSION_WEIGHTS weighted
          "critiques": {<dim>: str},
          "style": {<facet>: str},
          "style_tags": [str, ...],
        }
    """
    data = _extract_json(raw)
    if not isinstance(data, dict):
        return None

    sub_scores: dict[str, float] = {}
    critiques: dict[str, str] = {}
    for dim in DIMENSIONS:
        block = data.get(dim)
        if not isinstance(block, dict):
            return None
        for sub in DIMENSION_SUBATTRS[dim]:
            score = _coerce_score(block.get(sub))
            if score is None:
                return None  # every sub-attribute must be present & valid
            sub_scores[sub] = score
        crit = block.get("critique")
        if isinstance(crit, str) and crit.strip():
            critiques[dim] = crit.strip()

    style_raw = data.get("style")
    style: dict[str, str] = {}
    if isinstance(style_raw, dict):
        for facet in STYLE_FACETS:
            val = style_raw.get(facet)
            if isinstance(val, str) and val.strip():
                style[facet] = val.strip()

    tags_raw = data.get("style_tags")
    style_tags: list[str] = []
    if isinstance(tags_raw, list):
        seen = set()
        for t in tags_raw:
            if isinstance(t, str):
                tag = t.strip().lower()
                if tag and tag not in seen:
                    seen.add(tag)
                    style_tags.append(tag)

    dim_scores, overall = compute_overall(sub_scores)
    return {
        "sub_scores": sub_scores,
        "dim_scores": dim_scores,
        "overall": overall,
        "critiques": critiques,
        "style": style,
        "style_tags": style_tags,
    }


def compute_overall(
    sub_scores: dict[str, float],
    weights: Optional[dict[str, float]] = None,
    technical_override: Optional[float] = None,
) -> tuple[dict[str, float], float]:
    """Pure: derive per-dimension scores and the weighted overall from the 11
    sub-attribute scores. No VLM. Used at submit time and by the reweight CLI.

    Each dimension = equal-weight mean of its sub-attributes. `overall` =
    `weights`-weighted mean of the dimensions (weights renormalized over the
    dimensions actually present). `technical_override`, when given (an objective
    IQA technical score in [1,10]), replaces the VLM's technical dimension —
    the optional hybrid-anchor hook.

    Returns (dim_scores, overall), both rounded to 3 decimals. Missing
    sub-attributes are skipped; a dimension with no scores is omitted.
    """
    w = weights or DIMENSION_WEIGHTS
    dim_scores: dict[str, float] = {}
    for dim, subs in DIMENSION_SUBATTRS.items():
        vals = [sub_scores[s] for s in subs if s in sub_scores]
        if vals:
            dim_scores[dim] = round(sum(vals) / len(vals), 3)
    if technical_override is not None and "technical" in dim_scores:
        dim_scores["technical"] = round(float(technical_override), 3)

    total_w = sum(w.get(d, 0.0) for d in dim_scores)
    if total_w <= 0:
        overall = round(sum(dim_scores.values()) / len(dim_scores), 3) if dim_scores else 0.0
    else:
        overall = round(
            sum(dim_scores[d] * w.get(d, 0.0) for d in dim_scores) / total_w, 3)
    return dim_scores, overall


def percentile_ranks(values: list[float]) -> list[float]:
    """Library-relative percentile (0–100) for each value — the fix for the
    compressed raw scale. Ties share the average rank. Pure helper mirrored by
    the SQL `normalize-aesthetics` CLI; kept here for unit testing.

    Uses the standard "average rank" definition: pct = 100 * (rank - 0.5) / n
    over the sorted order, so the median lands near 50 and the top value near
    100 without ever hitting exactly 0 or 100 for a single extreme.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [50.0]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # positions i..j are tied; average 0-based rank is (i + j) / 2
        avg_pos = (i + j) / 2.0
        pct = 100.0 * (avg_pos + 0.5) / n
        for k in range(i, j + 1):
            ranks[order[k]] = round(pct, 3)
        i = j + 1
    return ranks


def aesthetics_from_row(photo: dict) -> Optional[dict]:
    """Assemble the structured aesthetics breakdown for an API response from a
    photo row's aes_* columns, or None if the photo hasn't been scored.

    Shape:
      {"overall": float, "overall_pct": float,
       "dimensions": {<dim>: {"score": float, "critique": str,
                              "subs": {<sub>: float}}},
       "style": {<facet>: str}, "style_tags": [str],
       "iqa": {"technical": float|None, "overall": float|None},
       "model": str}
    """
    if photo.get("aes_overall") is None:
        return None
    style_blob = {}
    if photo.get("aes_style"):
        try:
            style_blob = json.loads(photo["aes_style"])
        except (ValueError, TypeError):
            style_blob = {}
    critiques = style_blob.get("critiques", {}) if isinstance(style_blob, dict) else {}
    facets = style_blob.get("facets", {}) if isinstance(style_blob, dict) else {}
    tags = []
    if photo.get("aes_style_tags"):
        try:
            tags = json.loads(photo["aes_style_tags"]) or []
        except (ValueError, TypeError):
            tags = []
    dims = {}
    for dim, subs in DIMENSION_SUBATTRS.items():
        dims[dim] = {
            "score": photo.get(f"aes_{dim}"),
            "critique": critiques.get(dim) if isinstance(critiques, dict) else None,
            "subs": {s: photo.get(f"aes_{s}") for s in subs},
        }
    return {
        "overall": photo.get("aes_overall"),
        "overall_pct": photo.get("aes_overall_pct"),
        "dimensions": dims,
        "style": facets if isinstance(facets, dict) else {},
        "style_tags": tags,
        "iqa": {"technical": photo.get("aes_technical_iqa"),
                "overall": photo.get("aes_overall_iqa")},
        "model": photo.get("aes_model"),
        "scored_at": photo.get("aes_scored_at"),
    }


# ---------------------------------------------------------------------------
# DB-facing backfills (reused by the CLI and the maintenance sweep). Pure-SQL
# where possible; no VLM.
# ---------------------------------------------------------------------------

def recompute_overall_scores(db, weights: Optional[dict[str, float]] = None,
                             apply: bool = True) -> int:
    """Re-derive aes_technical/composition/impact + aes_overall from the stored
    sub-attribute columns using `weights` (default DIMENSION_WEIGHTS). No VLM —
    use this after retuning weights. Returns the number of rows affected.

    Operates on every scored row (those with the sub-attribute columns filled).
    Percentiles go stale after a reweight, so callers should follow with
    `normalize_overall`.
    """
    sub_cols = [f"aes_{s}" for s in ALL_SUBATTRS]
    where = " AND ".join(f"{c} IS NOT NULL" for c in sub_cols)
    rows = db.conn.execute(
        f"SELECT id, {', '.join(sub_cols)} FROM photos WHERE {where}"
    ).fetchall()
    if not apply:
        return len(rows)
    for row in rows:
        subs = {s: row[f"aes_{s}"] for s in ALL_SUBATTRS}
        dims, overall = compute_overall(subs, weights=weights)
        db.conn.execute(
            "UPDATE photos SET aes_technical=?, aes_composition=?, aes_impact=?, "
            "aes_overall=? WHERE id=?",
            (dims.get("technical"), dims.get("composition"), dims.get("impact"),
             overall, row["id"]),
        )
    db.conn.commit()
    return len(rows)


def normalize_overall(db, apply: bool = True) -> int:
    """Compute aes_overall_pct as the library-relative percentile (0–100) of
    aes_overall across every scored photo. This is the fix for the compressed
    raw scale — the UI/search rank on the percentile, so the best photo reads
    ~100 regardless of the model's central tendency. Returns rows updated.
    """
    rows = db.conn.execute(
        "SELECT id, aes_overall FROM photos WHERE aes_overall IS NOT NULL"
    ).fetchall()
    if not rows or not apply:
        return len(rows)
    ids = [r["id"] for r in rows]
    pcts = percentile_ranks([r["aes_overall"] for r in rows])
    db.conn.executemany(
        "UPDATE photos SET aes_overall_pct=? WHERE id=?",
        [(p, i) for p, i in zip(pcts, ids)],
    )
    db.conn.commit()
    return len(rows)


def normalize_subject_overall(db, apply: bool = True) -> int:
    """Compute aes_subject_overall_pct as the library-relative percentile (0–100)
    of aes_subject_overall across every subject-scored photo — the subject-crop
    analogue of `normalize_overall`. Returns rows updated. See
    photosearch/subjects.py + docs/plans/subject-aware-quality.md.
    """
    rows = db.conn.execute(
        "SELECT id, aes_subject_overall FROM photos WHERE aes_subject_overall IS NOT NULL"
    ).fetchall()
    if not rows or not apply:
        return len(rows)
    ids = [r["id"] for r in rows]
    pcts = percentile_ranks([r["aes_subject_overall"] for r in rows])
    db.conn.executemany(
        "UPDATE photos SET aes_subject_overall_pct=? WHERE id=?",
        [(p, i) for p, i in zip(pcts, ids)],
    )
    db.conn.commit()
    return len(rows)


def score_photo_aesthetics(
    image_path: str,
    model: str = MODEL,
    style_vocab: Optional[list[str]] = None,
) -> Optional[dict]:
    """Score one photo. Returns the normalized aesthetics dict (see
    `parse_aesthetics_response`) or None on unreachable model / unparseable
    response so the worker defers the photo for retry.

    Reuses `describe`'s image encoder + chat router, so this works against both
    Ollama and an OpenAI-compatible backend (LM Studio) with no extra code.
    """
    from .describe import _encode_image_for_ollama, _ollama_chat_with_retry

    path = Path(image_path)
    if not path.exists():
        return None
    if style_vocab is None:
        try:
            from .vocab_aesthetic_style import AESTHETIC_STYLE_VOCABULARY
            style_vocab = AESTHETIC_STYLE_VOCABULARY
        except Exception:
            style_vocab = None

    prompt = build_aesthetics_prompt(style_vocab)
    encoded = _encode_image_for_ollama(str(path))
    image_ref = encoded if encoded is not None else str(path)
    messages = [{"role": "user", "content": prompt, "images": [image_ref]}]

    # A little temperature helps the model spread scores instead of anchoring;
    # num_predict is generous for the JSON (terse critiques keep it well under
    # the OpenAI route's 768-token cap). num_ctx matches the vision default.
    options = {"temperature": 0.3, "num_ctx": 8192, "num_predict": 700}

    for attempt in range(2):  # one retry with a temperature bump on parse fail
        try:
            raw = _ollama_chat_with_retry(
                model=model, messages=messages, options=options,
                role=AESTHETICS_ROLE)
        except Exception:
            return None
        parsed = parse_aesthetics_response(raw or "")
        if parsed is not None:
            return parsed
        options = dict(options, temperature=0.5)
    return None
