"""Hallucination detection and verification for photo descriptions and tags.

Two-pass approach:
  Pass 1 (CLIP): Fast check — extract key nouns from the description, embed
    each as text via CLIP, and compare against the photo's CLIP embedding.
    Nouns with very low similarity are flagged as potential hallucinations.

  Pass 2 (LLM): Slower, more accurate — send the photo back to the vision
    model with a verification prompt asking it to confirm or deny specific
    claims from the description. Only runs on photos flagged by Pass 1.

When hallucinations are confirmed, the description and/or tags are
automatically regenerated.
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("photosearch.verify")

# ---------------------------------------------------------------------------
# Pass 1: CLIP-based fast check
# ---------------------------------------------------------------------------

# Common words that shouldn't be checked as visual nouns
_STOP_WORDS = {
    # Determiners, conjunctions, prepositions, pronouns
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "there", "here",
    "what", "which", "who", "whom", "where", "when", "why", "how",
    "not", "no", "nor", "as", "if", "then", "than", "too", "very",
    "just", "about", "above", "below", "between", "into", "through",
    "during", "before", "after", "while", "also", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "so", "up", "out", "off", "over", "under", "again",
    "further", "once", "all", "any", "many", "much", "one", "two",
    "three", "four", "five", "several", "around", "near", "next",
    "their", "they", "them", "he", "she", "him", "her", "his",
    # Spatial/compositional terms
    "background", "foreground", "left", "right", "center", "side",
    "top", "bottom", "front", "back", "scene", "image", "photo",
    "picture", "view", "area", "part", "way", "time", "day",
    # Verbs / participles (not visual objects)
    "appears", "seems", "look", "looks", "looking", "seen", "visible",
    "showing", "shown", "depicted", "display", "displays",
    "wearing", "dressed", "holding", "standing", "sitting", "facing",
    "playing", "running", "walking", "moving", "engaged", "positioned",
    "placed", "located", "surrounded", "captured", "taken", "set",
    "features", "featuring", "including", "includes", "suggests",
    # Adjectives (not objects you can point to)
    "possibly", "likely", "perhaps", "approximately", "slightly",
    "multiple", "various", "different", "large", "small", "young",
    "old", "new", "open", "dark", "light", "bright", "white", "black",
    "red", "blue", "green", "yellow", "pink", "orange", "brown", "gray",
    "grey", "tall", "short", "long", "wide", "number", "another",
    "clear", "warm", "cool", "soft", "hard", "dry", "wet",
    "casual", "formal", "traditional", "modern", "natural", "typical",
    "general", "overall", "main", "primary", "entire", "whole",
    # Abstract / non-visual nouns
    "activity", "activities", "action", "actions", "event", "events",
    "moment", "occasion", "setting", "environment", "surroundings",
    "attire", "clothing", "outfit", "apparel", "garment", "garments",
    "individual", "individuals", "figure", "figures", "subject", "subjects",
    "manner", "fashion", "appearance", "expression", "pose",
}

# Phrases that describe style/composition, not objects — skip these
_ABSTRACT_PHRASES = {
    "composition", "lighting", "exposure", "focus", "blur", "depth",
    "atmosphere", "mood", "tone", "contrast", "shadow", "shadows",
    "highlights", "angle", "perspective", "style",
}


def _extract_nouns(text: str) -> list[str]:
    """Extract candidate visual nouns from a description.

    Simple heuristic: split on non-alpha, keep words > 2 chars that aren't
    in the stop list. Groups multi-word phrases where possible.
    """
    # Lowercase and split into words
    words = re.findall(r"[a-z]+(?:[-][a-z]+)*", text.lower())
    nouns = []
    seen = set()
    for w in words:
        if w in _STOP_WORDS or w in _ABSTRACT_PHRASES:
            continue
        if len(w) <= 2:
            continue
        if w in seen:
            continue
        seen.add(w)
        nouns.append(w)
    return nouns


def clip_score_description(
    photo_embedding: list[float],
    description: str,
) -> list[dict]:
    """Score description nouns against photo CLIP embedding.

    Args:
        photo_embedding: Pre-computed CLIP embedding for the photo (512-dim).
        description: The text description to score.

    Returns:
        List of scored nouns: [{"noun": str, "similarity": float}, ...]
        Sorted by similarity ascending (most suspicious first).
    """
    from .clip_embed import embed_text
    import numpy as np

    nouns = _extract_nouns(description)
    if not nouns:
        logger.info("  No visual nouns extracted from description")
        return []

    logger.info("  Extracted nouns: %s", ", ".join(nouns))
    photo_vec = np.array(photo_embedding, dtype=np.float32)
    scored = []

    for noun in nouns:
        # Embed the noun as "a photo of {noun}" for better CLIP alignment
        text_emb = embed_text(f"a photo of {noun}")
        if text_emb is None:
            continue
        text_vec = np.array(text_emb, dtype=np.float32)
        # Cosine similarity (both are already unit-normalized)
        sim = float(np.dot(photo_vec, text_vec))
        scored.append({"noun": noun, "similarity": round(sim, 4)})

    scored.sort(key=lambda x: x["similarity"])
    return scored


def clip_score_tags(
    photo_embedding: list[float],
    tags: list[str],
) -> list[dict]:
    """Score tags against photo CLIP embedding.

    Args:
        photo_embedding: Pre-computed CLIP embedding for the photo.
        tags: List of tag strings to score.

    Returns:
        List of scored tags: [{"tag": str, "similarity": float}, ...]
        Sorted by similarity ascending (most suspicious first).
    """
    from .clip_embed import embed_text
    import numpy as np

    if not tags:
        return []

    photo_vec = np.array(photo_embedding, dtype=np.float32)
    scored = []

    for tag in tags:
        text_emb = embed_text(f"a photo of {tag}")
        if text_emb is None:
            continue
        text_vec = np.array(text_emb, dtype=np.float32)
        sim = float(np.dot(photo_vec, text_vec))
        scored.append({"tag": tag, "similarity": round(sim, 4)})

    scored.sort(key=lambda x: x["similarity"])
    return scored


# ---------------------------------------------------------------------------
# Pass 2: LLM verification
# ---------------------------------------------------------------------------

VERIFY_PROMPT = """\
Look at this photo carefully. A previous AI generated this description:

"{description}"

Check whether the description accurately matches the photo. Only flag \
something as WRONG if you are confident it is NOT in the photo. \
Minor wording differences or subjective interpretations are fine — only \
flag concrete objects or subjects that are clearly absent or clearly \
misidentified.

Use this exact format for each error, one per line:

WRONG: <the specific object or subject that is not in the photo>

If the description is accurate, respond with exactly:
ALL CORRECT

Do not explain. Do not add anything else.\
"""


def llm_verify_description(
    image_path: str,
    description: str,
    tags: list[str],
    model: str = "llava",
) -> list[dict]:
    """Use LLM to find hallucinations in a photo description.

    Instead of asking the LLM to verify a checklist of nouns (which requires
    good noun extraction), this gives the LLM the full description and asks
    it to identify errors. This produces much better results because:
      - The LLM sees the full context, not isolated words
      - No noun extraction needed — the LLM identifies the claims itself
      - The skeptical framing ("find ERRORS") counters confirmation bias

    Args:
        image_path: Path to the image.
        description: The original description to verify.
        tags: The original tags to verify.
        model: Ollama model name.

    Returns:
        List of confirmed hallucinations:
          [{"noun": str, "llm_says": "NO"}, ...]
    """
    from .describe import _ollama_chat_with_retry

    if not description and not tags:
        return []

    # Build the text to verify — combine description and tags
    verify_text = description or ""
    if tags:
        verify_text += f"\n\nTags: {', '.join(tags)}"

    prompt = VERIFY_PROMPT.format(description=verify_text)

    logger.info("  LLM verify prompt:\n%s", prompt)

    try:
        response = _ollama_chat_with_retry(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }],
        )
    except Exception as e:
        logger.warning("LLM verify failed for %s: %s", image_path, e)
        return []

    if not response:
        return []

    logger.info("  LLM raw response:\n%s", response.strip())

    # Parse response: look for "WRONG:" lines
    confirmed = []
    if "ALL CORRECT" in response.upper():
        logger.info("  LLM says: ALL CORRECT")
        return []

    for line in response.strip().split("\n"):
        match = re.match(r"^\s*WRONG:\s*(.+)", line, re.IGNORECASE)
        if match:
            wrong_item = match.group(1).strip().rstrip(".")
            confirmed.append({"noun": wrong_item, "llm_says": "NO"})
            logger.info("    WRONG: %s", wrong_item)

    return confirmed


# ---------------------------------------------------------------------------
# Full verification pipeline
# ---------------------------------------------------------------------------

def _flag_by_clip(desc_scores, tag_scores, clip_threshold):
    """Flag nouns/tags using both absolute threshold and relative scoring.

    Uses two strategies:
      1. Absolute: anything below clip_threshold is flagged.
      2. Relative: anything more than 1.5 std deviations below the median is
         flagged, even if above the absolute threshold.

    Returns (desc_flagged, tag_flagged, all_clip_items) where the first two
    are lists of flagged items and all_clip_items is the full scored list
    for logging purposes.
    """
    import numpy as np

    # Collect all similarity scores for computing median/std
    all_sims = [s["similarity"] for s in desc_scores] + [s["similarity"] for s in tag_scores]
    median_sim = float(np.median(all_sims)) if all_sims else 0.0
    std_sim = float(np.std(all_sims)) if len(all_sims) > 1 else 0.0
    relative_threshold = median_sim - 1.5 * std_sim if std_sim > 0 else 0.0

    effective_threshold = max(clip_threshold, relative_threshold)
    logger.info("  CLIP thresholds: absolute=%.4f, relative=%.4f (median=%.4f, std=%.4f) → effective=%.4f",
                clip_threshold, relative_threshold, median_sim, std_sim, effective_threshold)

    desc_flagged = []
    tag_flagged = []
    all_clip_items = []

    for s in desc_scores:
        flagged = s["similarity"] < effective_threshold
        status = "FLAGGED" if flagged else "ok"
        logger.info("    %-20s  sim=%.4f  %s", s["noun"], s["similarity"], status)
        item = {"type": "description", **s}
        all_clip_items.append(item)
        if flagged:
            desc_flagged.append(s)

    for s in tag_scores:
        flagged = s["similarity"] < effective_threshold
        status = "FLAGGED" if flagged else "ok"
        logger.info("    tag %-16s  sim=%.4f  %s", s["tag"], s["similarity"], status)
        item = {"type": "tag", **s}
        all_clip_items.append(item)
        if flagged:
            tag_flagged.append(s)

    return desc_flagged, tag_flagged, all_clip_items


def verify_photo(
    db,
    photo: dict,
    photo_embedding: list[float] | None = None,
    clip_threshold: float = 0.18,
    verify_model: str = "minicpm-v",
    regen_model: str = "llava",
    auto_regenerate: bool = True,
    llm_all: bool = False,
) -> dict:
    """Run full two-pass verification on a single photo.

    Args:
        db: PhotoDB instance.
        photo: Photo dict from database (must have id, filepath, description, tags).
        photo_embedding: Pre-computed CLIP embedding. If None, will be loaded from DB.
        clip_threshold: CLIP similarity threshold for flagging.
        verify_model: Ollama vision model for verification (should differ from regen_model).
        regen_model: Ollama model for regenerating descriptions/tags on failure.
        auto_regenerate: If True, regenerate description/tags when hallucinations found.
        llm_all: If True, skip CLIP gate and always run LLM verification.

    Returns:
        dict with keys:
          status: 'pass' | 'fail' | 'regenerated'
          clip_flags: list of CLIP-flagged items
          llm_confirmed: list of LLM-confirmed hallucinations
          regenerated: bool
    """
    description = photo.get("description") or ""
    tags_raw = photo.get("tags")
    tags = json.loads(tags_raw) if tags_raw and isinstance(tags_raw, str) else (tags_raw or [])
    filepath = db.resolve_filepath(photo.get("filepath", ""))
    photo_id = photo["id"]

    result = {
        "status": "pass",
        "clip_flags": [],
        "llm_confirmed": [],
        "regenerated": False,
    }

    if not description and not tags:
        # Nothing to verify
        _save_verification(db, photo_id, "pass", [])
        return result

    # Get photo CLIP embedding
    if photo_embedding is None:
        from .db import _serialize_float_list, _deserialize_float_list, CLIP_DIMENSIONS
        row = db.conn.execute(
            "SELECT embedding FROM clip_embeddings WHERE photo_id = ?", (photo_id,)
        ).fetchone()
        if row:
            photo_embedding = _deserialize_float_list(row["embedding"], CLIP_DIMENSIONS)

    if photo_embedding is None:
        # Can't do CLIP check without embedding — skip to LLM only
        logger.debug("No CLIP embedding for photo %d, skipping CLIP check", photo_id)
        _save_verification(db, photo_id, "pass", [])
        return result

    # --- Pass 1: CLIP scoring (informational for batch filtering) ---
    desc_scores = clip_score_description(photo_embedding, description) if description else []
    tag_scores = clip_score_tags(photo_embedding, tags) if tags else []

    desc_flagged, tag_flagged, all_clip_items = _flag_by_clip(desc_scores, tag_scores, clip_threshold)
    result["clip_flags"] = [item for item in all_clip_items
                            if any(f.get("noun") == item.get("noun") and f.get("similarity") == item.get("similarity")
                                   for f in desc_flagged)
                            or any(f.get("tag") == item.get("tag") and f.get("similarity") == item.get("similarity")
                                   for f in tag_flagged)]

    # Decide whether to run LLM verification
    needs_llm = llm_all or desc_flagged or tag_flagged
    if not needs_llm:
        logger.info("  CLIP pass: no nouns below effective threshold")
        _save_verification(db, photo_id, "pass", [])
        return result

    # --- Pass 2: LLM verification ---
    # Instead of checking individual nouns, give the LLM the full description
    # and ask it to identify errors. This avoids the noun extraction problem.
    if filepath and Path(filepath).exists():
        confirmed = llm_verify_description(filepath, description, tags, model=verify_model)
        logger.info("  LLM found %d error(s): %s",
                     len(confirmed),
                     ", ".join(c["noun"] for c in confirmed) if confirmed else "(none)")
    else:
        logger.warning("Photo file not found for LLM verify: %s", filepath)
        confirmed = []

    if not confirmed:
        # LLM says description is correct — pass
        _save_verification(db, photo_id, "pass", result["clip_flags"])
        return result

    # --- Pass 3: CLIP cross-check on LLM findings ---
    # The LLM may over-flag. Check each flagged item against CLIP: if CLIP
    # gives it a reasonable similarity score, the object is probably there
    # and the LLM is wrong. Both must agree to confirm a hallucination.
    from .clip_embed import embed_text
    import numpy as np

    photo_vec = np.array(photo_embedding, dtype=np.float32)
    all_sims = [s["similarity"] for s in desc_scores] + [s["similarity"] for s in tag_scores]
    median_sim = float(np.median(all_sims)) if all_sims else 0.0

    verified_confirmed = []
    for item in confirmed:
        noun = item["noun"]
        # Extract key words from the LLM's finding for CLIP check
        # e.g. "adult crouching holding a frisbee" → check "frisbee"
        # Use the whole phrase as CLIP handles it well
        text_emb = embed_text(f"a photo of {noun}")
        if text_emb is not None:
            text_vec = np.array(text_emb, dtype=np.float32)
            sim = float(np.dot(photo_vec, text_vec))
            # If CLIP gives it above-median similarity, override the LLM
            if sim >= median_sim:
                logger.info("    CLIP override: '%s' sim=%.4f >= median=%.4f — keeping (LLM was wrong)",
                            noun, sim, median_sim)
                continue
            else:
                logger.info("    CLIP confirms: '%s' sim=%.4f < median=%.4f — hallucination",
                            noun, sim, median_sim)
        verified_confirmed.append(item)

    result["llm_confirmed"] = verified_confirmed

    if not verified_confirmed:
        logger.info("  All LLM findings overridden by CLIP — pass")
        _save_verification(db, photo_id, "pass", result["clip_flags"])
        return result

    logger.info("  Confirmed %d hallucination(s) after CLIP cross-check: %s",
                len(verified_confirmed),
                ", ".join(c["noun"] for c in verified_confirmed))

    # --- Hallucinations confirmed by both LLM and CLIP ---
    result["status"] = "fail"
    confirmed_nouns = {c["noun"] for c in verified_confirmed}

    if auto_regenerate and filepath and Path(filepath).exists():
        regenerated = _regenerate(db, photo_id, filepath, confirmed_nouns, tags, regen_model)
        if regenerated:
            result["status"] = "regenerated"
            result["regenerated"] = True

    _save_verification(db, photo_id, result["status"], result["clip_flags"])
    return result


def _regenerate(
    db,
    photo_id: int,
    filepath: str,
    confirmed_nouns: set,
    old_tags: list[str],
    model: str,
) -> bool:
    """Regenerate description and/or tags for a photo with confirmed hallucinations."""
    from .describe import describe_photo, tag_photo, DESCRIBE_PROMPT

    regenerated = False

    # Regenerate description with a stricter prompt
    strict_prompt = DESCRIBE_PROMPT + (
        "\n\nIMPORTANT: A previous description of this photo was found to contain "
        "hallucinated objects that are NOT in the image. Be extra careful to ONLY "
        "describe what you can clearly see. Do NOT mention: "
        + ", ".join(sorted(confirmed_nouns)) + "."
    )

    new_desc = describe_photo(filepath, model=model, prompt=strict_prompt)
    if new_desc:
        db.conn.execute(
            "UPDATE photos SET description = ? WHERE id = ?",
            (new_desc, photo_id),
        )
        regenerated = True
        logger.info("  Regenerated description for photo %d", photo_id)

    # Always regenerate tags when description had hallucinations — they were
    # generated by the same model looking at the same image, so they're suspect.
    if regenerated:
        new_tags = tag_photo(filepath, model=model)
        if new_tags is not None:
            db.conn.execute(
                "UPDATE photos SET tags = ? WHERE id = ?",
                (json.dumps(new_tags), photo_id),
            )
            regenerated = True
            logger.info("  Regenerated tags for photo %d", photo_id)

    if regenerated:
        db.conn.commit()

    return regenerated


def _save_verification(db, photo_id: int, status: str, flags: list):
    """Write verification results to the database."""
    now = datetime.now().isoformat()
    flags_json = json.dumps(flags) if flags else None
    db.conn.execute(
        """UPDATE photos
           SET verified_at = ?, verification_status = ?, hallucination_flags = ?
           WHERE id = ?""",
        (now, status, flags_json, photo_id),
    )
    db.conn.commit()


# ---------------------------------------------------------------------------
# Batch verification
# ---------------------------------------------------------------------------

def verify_photos(
    db,
    photos: list[dict] | None = None,
    clip_threshold: float = 0.18,
    verify_model: str = "minicpm-v",
    regen_model: str = "llava",
    auto_regenerate: bool = True,
    force: bool = False,
    llm_all: bool = False,
) -> dict:
    """Verify descriptions and tags for multiple photos.

    Args:
        db: PhotoDB instance.
        photos: List of photo dicts. If None, queries all photos with descriptions.
        clip_threshold: CLIP similarity threshold for flagging.
        verify_model: Ollama vision model for the verification pass.
        regen_model: Ollama model for regenerating descriptions on failure.
        auto_regenerate: Auto-regenerate on confirmed hallucinations.
        force: Re-verify even previously verified photos.

    Returns:
        Summary dict: {total, checked, passed, failed, regenerated}
    """
    from .db import _deserialize_float_list, CLIP_DIMENSIONS

    if photos is None:
        if force:
            rows = db.conn.execute(
                "SELECT * FROM photos WHERE description IS NOT NULL OR tags IS NOT NULL"
            ).fetchall()
        else:
            rows = db.conn.execute(
                """SELECT * FROM photos
                   WHERE (description IS NOT NULL OR tags IS NOT NULL)
                   AND verified_at IS NULL"""
            ).fetchall()
        photos = [dict(r) for r in rows]

    total = len(photos)
    if total == 0:
        print("  No photos to verify.")
        return {"total": 0, "checked": 0, "passed": 0, "failed": 0, "regenerated": 0}

    # Bulk-load CLIP embeddings for efficiency
    photo_ids = [p["id"] for p in photos]
    embeddings = {}
    batch_size = 500
    for i in range(0, len(photo_ids), batch_size):
        batch = photo_ids[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        rows = db.conn.execute(
            f"SELECT photo_id, embedding FROM clip_embeddings WHERE photo_id IN ({placeholders})",
            batch,
        ).fetchall()
        for r in rows:
            embeddings[r["photo_id"]] = _deserialize_float_list(r["embedding"], CLIP_DIMENSIONS)

    stats = {"total": total, "checked": 0, "passed": 0, "failed": 0, "regenerated": 0}

    for i, photo in enumerate(photos):
        pid = photo["id"]
        fname = photo.get("filename", f"id={pid}")
        print(f"  [{i+1}/{total}] {fname} ... ", end="", flush=True)

        emb = embeddings.get(pid)
        result = verify_photo(
            db, photo,
            photo_embedding=emb,
            clip_threshold=clip_threshold,
            verify_model=verify_model,
            regen_model=regen_model,
            auto_regenerate=auto_regenerate,
            llm_all=llm_all,
        )

        stats["checked"] += 1
        if result["status"] == "pass":
            stats["passed"] += 1
            if result["clip_flags"]:
                flagged_items = ", ".join(
                    f"{f.get('noun', f.get('tag'))}={f['similarity']:.4f}"
                    for f in result["clip_flags"]
                )
                print(f"pass (CLIP flagged [{flagged_items}], LLM cleared)")
            else:
                print("pass")
        elif result["status"] == "regenerated":
            stats["regenerated"] += 1
            confirmed = [c["noun"] for c in result["llm_confirmed"]]
            print(f"REGENERATED — hallucinations: {', '.join(confirmed)}")
        else:
            stats["failed"] += 1
            confirmed = [c["noun"] for c in result["llm_confirmed"]]
            print(f"FAIL — hallucinations: {', '.join(confirmed)}")

    return stats
