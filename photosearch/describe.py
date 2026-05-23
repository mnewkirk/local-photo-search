"""Generate natural language photo descriptions using a local vision model via Ollama.

Sends each photo to a multimodal LLM (default: llava) running on Ollama and gets
back a concise, search-friendly description. Descriptions are stored in the
photos.description column and searched via text matching.

Requirements:
  - Ollama running locally (default http://localhost:11434)
  - A vision-capable model pulled: ollama pull llava

The description prompt is tuned for photo search: it asks the model to describe
who/what is in the photo, what they're doing, the setting, and notable visual
details — all in a few sentences that will match natural language queries.
"""

import base64
import io
import time
from collections import Counter
from pathlib import Path
from typing import Optional

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

try:
    from PIL import Image, ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Default describe model. A 100-image bakeoff (2026-05-14) had llama3.2-vision
# beat llava 68-26 on free-form description quality, especially on text/OCR-heavy
# images. It needs the tuned options + detect-retry below to suppress its
# repetition-loop failure mode. Other options: llava, llava:13b, moondream
# (moondream is ~1.6B — much faster on CPU, lower quality).
MODEL = "llama3.2-vision"

# Default tags model — kept separate from MODEL. The tags pass is a constrained
# vocabulary-selection task, not free-form prose: llama3.2-vision degenerates
# and hallucinates out-of-vocab tags on it, while llava follows the TAG_PROMPT
# cleanly. Decoupled so describe and tags each use the model good at their task.
TAGS_MODEL = "llava"

# Text-only model used by the v23 category-content and keywords passes.
# Decided in Phase 0 bakeoff 2026-05-18 (vs qwen2.5:3b head-to-head): cleaner
# multi-word phrase output, lower hallucination rate, comparable latency.
CATEGORY_CONTENT_MODEL = "llama3.2:3b"
KEYWORDS_MODEL = "llama3.2:3b"

# Maximum long-edge size (pixels) when resizing images before sending to Ollama.
# Vision models resize internally anyway — sending a smaller image cuts I/O and
# model preprocessing time with no meaningful quality loss for descriptions.
_MAX_IMAGE_PX = 1024

# Generation options passed to Ollama chat calls. Model-aware: llama3.2-vision
# is prone to repetition-loop degeneration under greedy decoding (temp=0), so it
# gets repeat_penalty + a small temperature. A 100-image bakeoff showed temp=0
# alone left 19% of llama3.2-vision outputs degenerate; these params + the
# detect-and-retry below brought that to ~0%. llava/moondream are not loop-prone
# under greedy decoding, so they keep deterministic temp=0.
# num_ctx caps the KV-cache size per request. Our prompts (text + ~576 llava
# image tokens + ~200 output) fit well under 8192. Pinning it here keeps the
# context from auto-sizing huge on a big-VRAM GPU — at NUM_PARALLEL=3 an
# auto-sized 32k context produced a ~16 GiB KV cache that spilled model layers
# to CPU. Travels with the code, so it also applies to the NAS Docker Ollama.
_NUM_CTX = 8192
# The text-only passes (category-content, keywords) carry NO image tokens —
# just a short description (~250 tok worst case) + the ~80-term vocab + scaffold
# (~700 tok total). They inherited the vision _NUM_CTX=8192, which forces a 4x-
# oversized KV cache; under NUM_PARALLEL>1 that allocation churn is a suspected
# feeder for the Ollama runner stall (see project-mac-ollama-concurrency-limit).
# 2048 leaves comfortable headroom for the largest real descriptions.
_TEXT_NUM_CTX = 2048
_DEFAULT_OLLAMA_OPTIONS = {
    "num_predict": 150,
    "temperature": 0,
    "num_ctx": _NUM_CTX,
}
_LLAMA_VISION_OPTIONS = {
    "num_predict": 200,
    "temperature": 0.4,
    "repeat_penalty": 1.5,
    "repeat_last_n": 320,
    "num_ctx": _NUM_CTX,
}


def _options_for_model(model: str) -> dict:
    """Return Ollama generation options tuned for the given model."""
    if model.startswith("llama3.2-vision"):
        return _LLAMA_VISION_OPTIONS
    return _DEFAULT_OLLAMA_OPTIONS

# Ollama API host — override with OLLAMA_HOST env var if non-default.
# The ollama Python client reads OLLAMA_HOST automatically.

# Prompt designed for search-friendly descriptions. Key goals:
#   1. Mention people (count, approximate age, what they're wearing/doing)
#   2. Describe the setting and environment
#   3. Note prominent objects, animals, or activities
#   4. Keep it concise (2-4 sentences) so text search works well
#   5. No flowery language — factual and specific
DESCRIBE_PROMPT = """\
Describe this photo in 2-4 concise sentences for a search index. Include:
- Who or what is in the photo (number of people, approximate ages, clothing)
- What they are doing (actions, poses)
- The setting (indoor/outdoor, location type such as beach/park/street/home)
- Whether people are present or absent
Be factual. Only describe what you can clearly see — do not guess at objects you \
are unsure about. Do not start with "The image shows" or similar preamble.\
"""

# Moondream is a VQA model trained on direct questions, not structured prompts.
# Two failure modes to avoid:
#   1. Empty response — caused by complex bullet-point prompts
#   2. Bounding-box dump — caused by prompts that resemble detection tasks
#      (e.g. "describe objects", "list what you see")
# A conversational question avoids both: moondream answers it with plain text.
MOONDREAM_DESCRIBE_PROMPT = (
    "Write 2-3 sentences describing this photo. "
    "Include who or what is in the scene, what is happening, and where it was taken. "
    "Use plain sentences, not lists."
)

def _get_describe_prompt(model: str) -> str:
    """Return the appropriate describe prompt for the given model."""
    if model.startswith("moondream"):
        return MOONDREAM_DESCRIBE_PROMPT
    return DESCRIBE_PROMPT


def _encode_image_for_ollama(image_path: str) -> Optional[str]:
    """Resize the image to _MAX_IMAGE_PX long edge and return a base64 JPEG string.

    LLaVA and moondream both resize images internally before inference, so
    sending a smaller image costs nothing in quality for description purposes
    but meaningfully reduces disk I/O and model preprocessing time — especially
    for large Sony RAW-to-JPEG files (10-25 MB).

    Returns a base64 string on success, or None if PIL is unavailable (in which
    case the caller falls back to passing the file path directly).
    """
    if not HAS_PIL:
        return None
    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        if max(img.size) > _MAX_IMAGE_PX:
            img.thumbnail((_MAX_IMAGE_PX, _MAX_IMAGE_PX), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def check_available(model: str = MODEL) -> None:
    """Raise a clear error if Ollama is not reachable or the model isn't pulled."""
    if not HAS_OLLAMA:
        raise RuntimeError(
            "ollama Python package is not installed.\n"
            "Run: pip install ollama"
        )
    try:
        # List local models to verify Ollama is running
        models = ollama.list()
        model_names = [m.model for m in models.models] if hasattr(models, 'models') else []
        # Normalize names: "llava:latest" should match "llava"
        normalized = {n.split(":")[0] for n in model_names}
        if model.split(":")[0] not in normalized:
            available = ", ".join(model_names) if model_names else "(none)"
            raise RuntimeError(
                f"Model '{model}' not found in Ollama.\n"
                f"Available models: {available}\n"
                f"Run: ollama pull {model}"
            )
    except Exception as e:
        if "Connection" in str(type(e).__name__) or "refused" in str(e).lower():
            raise RuntimeError(
                "Cannot connect to Ollama. Make sure it's running:\n"
                "  ollama serve"
            ) from e
        raise


_MAX_RETRIES = 3
_RETRY_DELAY = 5  # seconds between retries

# Per-call wall-clock cap for ollama.chat. Without this, a wedged Ollama
# runner can hang an `ollama.chat()` call indefinitely — observed in the
# Phase 0 bakeoff (one call ran 1h 23m before 500-ing) and again in the
# Phase 5 backfill (all 4 workers simultaneously stuck at photo 16/16
# with no progress). 120s is generous for vision-on-CPU (~60-90s typical)
# and 30-60× the normal text-only call. A timeout fires a retry; after
# _MAX_RETRIES the caller's existing exception-handling catches and
# either marks the photo failed or returns []. Override per-call by
# passing `timeout=...` to `_ollama_chat_with_retry`.
_DEFAULT_OLLAMA_TIMEOUT_S = 120

# Tighter timeout for the text-only category/keyword passes. These calls are
# normally ~1-3s (small llama3.2:3b, ~20-token output), but Ollama intermittently
# stalls under sustained NUM_PARALLEL=1 text load — a freeze that reproduces on
# BOTH the Windows GPU and a CPU-only WSL2 Ollama, and is NOT input-, model-, or
# num_predict-related (the stuck photos replay in ~1s in isolation). Root cause is
# still open; this just bounds the blast radius: a stall aborts in 10s and the
# retry/next-pass picks the photo up later, instead of blocking the single Ollama
# slot for the full 120s. Vision passes keep the 120s default (they're slower and
# have not shown this stall).
_TEXT_OLLAMA_TIMEOUT_S = 10

# Substrings that signal the llama runner was OOM-killed rather than a genuine
# transient network error. Surfaces in Ollama errors as:
#   "llama runner process has terminated: %!w(<nil>) (status code: 500)"
# On Mac with managed Docker Ollama + worker fleet, this almost always means
# Docker Desktop's VM ran out of memory loading the vision model — see
# run-workers.sh header for the full explanation and fixes.
_RUNNER_OOM_PATTERNS = ("llama runner process has terminated", "runner has terminated")
_RUNNER_OOM_HINT_SHOWN = False


def _maybe_print_runner_oom_hint() -> None:
    """Print a one-time diagnostic hint when the llama-runner OOM pattern appears.

    Retries will still run, but the user sees the likely root cause immediately
    instead of watching identical 500 errors scroll by.
    """
    global _RUNNER_OOM_HINT_SHOWN
    if _RUNNER_OOM_HINT_SHOWN:
        return
    _RUNNER_OOM_HINT_SHOWN = True
    import sys
    sys.stderr.write(
        "\n"
        "  ⚠ Ollama llama runner terminated — this is almost always the model\n"
        "    being OOM-killed by the Docker Desktop VM, not a network blip.\n"
        "    Likely cause: managed Docker Ollama + worker fleet contending for\n"
        "    Docker's VM memory. Fixes (easiest first):\n"
        "      1. Prefer native 'ollama serve' on the host (./run-workers.sh --stop,\n"
        "         then run 'ollama serve' + 'ollama pull <model>', then relaunch workers)\n"
        "      2. Raise Docker Desktop memory (Settings → Resources → Memory → 16 GiB+)\n"
        "         and restart Docker Desktop\n"
        "      3. Reduce fleet pressure: ./run-workers.sh -n 2 -m 2g ...\n"
        "\n"
    )
    sys.stderr.flush()

import re as _re
_BBOX_PATTERN = _re.compile(r'\[\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*\]')

def _is_valid_description(text: str) -> bool:
    """Return False for moondream's garbage outputs (bounding-box arrays, etc.).

    Moondream sometimes fires its object-detection head instead of its text
    generation head and outputs coordinate arrays like:
        ids = [0.39, 0.72, 0.46, 0.80]
        idskfjf [0.12, 0.13, 0.87, 0.35]
    These are useless as search descriptions and must be filtered out.
    """
    if not text:
        return False
    # Reject responses that are primarily a bounding-box coordinate dump
    if _BBOX_PATTERN.search(text):
        # Allow if there is substantial surrounding text (real description with coords)
        non_coord = _BBOX_PATTERN.sub('', text).strip()
        word_count = len(non_coord.split())
        if word_count < 6:
            return False
    # Reject very short responses that are unlikely to be useful descriptions
    if len(text.split()) < 4:
        return False
    return True


def _is_degenerate(text: str) -> bool:
    """Detect repetition-loop / runaway degeneration in a model output.

    llama3.2-vision occasionally falls into repetition loops — the same phrase
    repeated many times, ellipsis spam, or runaway length. A 100-image bakeoff
    found this in 19% of temp=0 outputs. This heuristic caught all 19/19 with
    no false positives on clean outputs; describe_photo uses it to retry or
    fall back. Returns False for short outputs (e.g. tag lists), which are not
    loop-prone and would be mis-flagged by the length-based checks.
    """
    if not text:
        return False  # emptiness is handled by _is_valid_description
    words = text.split()
    n = len(words)
    if n == 0:
        return False
    uniq_ratio = len(set(w.lower() for w in words)) / n
    sixgrams = [tuple(words[i:i + 6]) for i in range(max(0, n - 5))]
    max_rep6 = max(Counter(sixgrams).values()) if sixgrams else 0
    ellipsis = text.count("…") + text.count("...")
    return (uniq_ratio < 0.45 and n > 60) or max_rep6 >= 3 or ellipsis > 8 or n > 350


def _ollama_chat_with_retry(
    model: str,
    messages: list,
    retries: int = _MAX_RETRIES,
    options: Optional[dict] = None,
    timeout: Optional[float] = None,
) -> Optional[str]:
    """Call ollama.chat with retry logic for transient failures.

    Each attempt is bounded by `timeout` seconds (default
    _DEFAULT_OLLAMA_TIMEOUT_S). Wedged Ollama runners that would otherwise
    block forever are aborted, the call is treated as a transient error,
    and the retry loop kicks in. The underlying thread is daemon, so it
    dies with the process if it can't be cleanly interrupted.

    Retries on timeouts, connection errors, and server errors.
    Returns the response text or None.
    """
    import queue as _queue
    import threading as _threading

    if timeout is None:
        timeout = _DEFAULT_OLLAMA_TIMEOUT_S

    call_kwargs: dict = {"model": model, "messages": messages}
    if options:
        call_kwargs["options"] = options

    for attempt in range(1, retries + 1):
        result_q: _queue.Queue = _queue.Queue(maxsize=1)

        def _worker():
            try:
                response = ollama.chat(**call_kwargs)
                text = response.message.content.strip()
                result_q.put(("ok", text if text else None))
            except Exception as ex:
                result_q.put(("err", ex))

        _threading.Thread(target=_worker, daemon=True).start()
        try:
            kind, val = result_q.get(timeout=timeout)
        except _queue.Empty:
            kind = "err"
            val = TimeoutError(
                f"ollama.chat exceeded {timeout}s timeout on attempt {attempt}/{retries}"
            )

        if kind == "ok":
            return val

        e = val
        err_str = str(e).lower()
        if any(p in err_str for p in _RUNNER_OOM_PATTERNS):
            _maybe_print_runner_oom_hint()
        is_transient = any(kw in err_str for kw in [
            "timeout", "connection", "refused", "reset", "broken pipe",
            "503", "502", "500", "unavailable",
        ])
        if is_transient and attempt < retries:
            print(f" [retry {attempt}/{retries} in {_RETRY_DELAY}s: {e}]", end="", flush=True)
            time.sleep(_RETRY_DELAY)
        else:
            raise e
    return None


def describe_photo(
    image_path: str,
    model: str = MODEL,
    prompt: Optional[str] = None,
) -> Optional[str]:
    """Generate a description for a single photo.

    Args:
        image_path: Path to the image file (JPEG, PNG, etc.)
        model: Ollama model name (must be multimodal).
        prompt: The prompt to send alongside the image. Defaults to the
                model-appropriate prompt (moondream gets a simpler question-style
                prompt; all others get the structured DESCRIBE_PROMPT).

    Returns:
        A text description string, or None if generation failed.
    """
    if not HAS_OLLAMA:
        return None

    path = Path(image_path)
    if not path.exists():
        print(f"  Warning: image not found: {image_path}")
        return None

    if prompt is None:
        prompt = _get_describe_prompt(model)

    # Resize to _MAX_IMAGE_PX before sending — vision models resize internally
    # anyway, so this is free quality-wise but cuts I/O and preprocessing time.
    encoded = _encode_image_for_ollama(str(path))
    image_ref = encoded if encoded is not None else str(path)

    messages = [{"role": "user", "content": prompt, "images": [image_ref]}]
    options = _options_for_model(model)

    try:
        result = _ollama_chat_with_retry(
            model=model,
            messages=messages,
            options=options,
        )
        # Filter out moondream's bounding-box coordinate dumps and other garbage
        if result is not None and not _is_valid_description(result):
            result = None

        # Moondream sometimes returns empty or coordinate garbage on the first
        # attempt. Retry once with a bare fallback question — different enough
        # to avoid the same failure mode.
        if result is None and model.startswith("moondream"):
            result = _ollama_chat_with_retry(
                model=model,
                messages=[{
                    "role": "user",
                    "content": "Describe what you see in this photo in 2 sentences.",
                    "images": [image_ref],
                }],
                options=options,
            )
            if result is not None and not _is_valid_description(result):
                result = None

        # Degeneration recovery: llama3.2-vision occasionally falls into
        # repetition loops. Its options carry temperature>0 so each retry is
        # independent — a 100-image bakeoff showed up to 2 retries clears every
        # loop. If retries still fail, fall back to llava (not loop-prone under
        # greedy decoding) before giving up.
        if result is not None and _is_degenerate(result):
            for _ in range(2):
                retry = _ollama_chat_with_retry(model=model, messages=messages, options=options)
                if retry and _is_valid_description(retry) and not _is_degenerate(retry):
                    result = retry
                    break
            else:
                if not model.startswith("llava"):
                    fallback = _ollama_chat_with_retry(
                        model="llava", messages=messages,
                        options=_options_for_model("llava"),
                    )
                    if fallback and _is_valid_description(fallback) and not _is_degenerate(fallback):
                        result = fallback

        return result
    except Exception as e:
        print(f"  Warning: description failed for {path.name}: {e}")
        return None


# Prompt for aesthetic critique — complements the numeric aesthetic score
# with a human-readable explanation of what works and what doesn't.
AESTHETIC_CRITIQUE_PROMPT = """\
Rate this photo's quality in exactly 2 short sentences. First sentence: what \
works (be specific — name the compositional technique, lighting quality, or \
subject). Second sentence: what doesn't work or could be better (be honest — \
mention blur, exposure problems, distracting elements, weak composition, or \
lack of a clear subject). If the photo has real problems, say so directly. \
Do not start with "This photo" or "The image". Do not be generous. \
Do not comment on resolution or image quality artifacts — you are seeing a \
downscaled preview and cannot judge the original resolution.\
"""


def critique_photo(
    image_path: str,
    model: str = MODEL,
) -> Optional[str]:
    """Generate an aesthetic critique for a single photo via Ollama.

    Returns a short text evaluating composition, lighting, and subject
    clarity — or None if generation failed.
    """
    return describe_photo(image_path, model=model, prompt=AESTHETIC_CRITIQUE_PROMPT)


# ---------------------------------------------------------------------------
# Visual tagging (replaces old M9 TAG_VOCABULARY / tag_photo)
# ---------------------------------------------------------------------------
# A focused visual-quality vocabulary (36 terms) assigned by Ollama vision.
# Mood / light / composition only — content is handled by extract_categories.

_VISUAL_MAX_PLAUSIBLE_TAGS = 12  # tighter than the old 16; smaller vocab.


def _build_category_prompt(description: str, vocab: list[str]) -> str:
    vocab_str = ", ".join(vocab)
    return (
        "From the vocabulary below, return ONLY tags that apply to this photo description.\n"
        "Rules:\n"
        "- Return a comma-separated list with no commentary.\n"
        "- Only use tags from the vocabulary; ignore anything else.\n\n"
        f"Vocabulary: {vocab_str}\n\n"
        f"Description: {description.strip()}\n"
    )


def extract_categories_from_description(
    description: Optional[str],
    model: str = CATEGORY_CONTENT_MODEL,
) -> Optional[list[str]]:
    """Map a description → list of in-vocab categories via a text-only LLM.

    Returns a list (possibly empty = genuinely no in-vocab categories) on
    success, or None if the Ollama call timed out / errored so the caller can
    defer the photo for retry instead of recording an empty result.
    """
    if not description or not description.strip():
        return []
    from .vocab_content import CONTENT_VOCABULARY
    if not HAS_OLLAMA:
        return []
    vocab_set = set(CONTENT_VOCABULARY)
    prompt = _build_category_prompt(description, CONTENT_VOCABULARY)
    try:
        raw = _ollama_chat_with_retry(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_ctx": _TEXT_NUM_CTX},
            timeout=_TEXT_OLLAMA_TIMEOUT_S,
        )
    except Exception:
        # Timeout / connection error — return None (NOT []) so the caller can
        # tell "Ollama stalled, retry later" apart from "ran fine, no categories".
        # Returning [] here would let the worker mark the photo permanently done
        # with empty categories on a mere stall.
        return None
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for token in raw.split(","):
        t = token.strip().lower().rstrip(".")
        if t in vocab_set and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def extract_keywords_from_description(
    description: Optional[str],
    model: str = KEYWORDS_MODEL,
) -> Optional[list[str]]:
    """Extract 5-15 free-form lowercased keywords from a description.

    Returns a list on success (possibly empty), or None if the Ollama call
    timed out / errored, so the caller can defer the photo for retry.
    """
    if not description or not description.strip():
        return []
    if not HAS_OLLAMA:
        return []
    from .bakeoff import build_keyword_prompt, parse_keywords_response
    prompt = build_keyword_prompt(description)
    try:
        raw = _ollama_chat_with_retry(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_ctx": _TEXT_NUM_CTX},
            timeout=_TEXT_OLLAMA_TIMEOUT_S,
        )
    except Exception:
        # None (not []) signals "stalled, retry later" vs "ran fine, no keywords".
        return None
    return parse_keywords_response(raw)


def _build_visual_prompt(vocab: list[str]) -> str:
    return (
        "Pick visual-quality tags for this photo from this list: "
        + ", ".join(vocab)
        + "\n\nRules:\n"
        "- Return ONLY a comma-separated list of tags from the list above.\n"
        "- Mood / light / composition only. Don't describe content.\n"
        "- Include every tag that clearly applies.\n"
    )


def _parse_visual_response(raw: str, vocab_set: set[str]) -> list[str]:
    out = []
    seen: set[str] = set()
    for token in (raw or "").split(","):
        t = token.strip().lower().rstrip(".")
        if t in vocab_set and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def tag_visual_photo(
    image_path: str,
    model: str = TAGS_MODEL,
) -> Optional[list[str]]:
    """Generate visual-quality tags for a single photo via Ollama (vision).

    Calls `_ollama_chat_with_retry` directly (not via describe_photo) so the
    test surface is uniform — same mock point as extract_categories/keywords.
    Mirrors the regurgitation guard from the old `tag_photo` at threshold 12.
    """
    from .vocab_visual import VISUAL_VOCABULARY
    if not HAS_OLLAMA:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    vocab_set = set(VISUAL_VOCABULARY)
    prompt = _build_visual_prompt(VISUAL_VOCABULARY)
    encoded = _encode_image_for_ollama(str(path))
    image_ref = encoded if encoded is not None else str(path)
    options = _options_for_model(model)

    try:
        raw = _ollama_chat_with_retry(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [image_ref]}],
            options=options,
        )
    except Exception:
        return None
    if not raw:
        return None
    tags = _parse_visual_response(raw, vocab_set)
    if len(tags) >= _VISUAL_MAX_PLAUSIBLE_TAGS:
        # Retry with temp bump (regurgitation guard — same shape as old tag_photo).
        retry_opts = dict(options)
        retry_opts["temperature"] = 0.4
        retry_opts.setdefault("repeat_penalty", 1.3)
        try:
            raw2 = _ollama_chat_with_retry(
                model=model,
                messages=[{"role": "user", "content": prompt, "images": [image_ref]}],
                options=retry_opts,
            )
        except Exception:
            raw2 = None
        if not raw2:
            return None
        tags2 = _parse_visual_response(raw2, vocab_set)
        if len(tags2) >= _VISUAL_MAX_PLAUSIBLE_TAGS or not tags2:
            return None
        tags = tags2
    return tags if tags else None


def describe_photos_batch(
    image_paths: list[str],
    model: str = MODEL,
    prompt: Optional[str] = None,
) -> list[Optional[str]]:
    """Generate descriptions for multiple photos sequentially.

    LLaVA processes one image at a time (no true batching), so this is
    a simple loop with progress reporting. Returns a list parallel to
    image_paths — each entry is a description string or None.
    """
    results: list[Optional[str]] = []
    total = len(image_paths)

    for i, path in enumerate(image_paths, 1):
        fname = Path(path).name
        print(f"  [{i}/{total}] {fname} ...", end="", flush=True)
        t0 = time.time()
        desc = describe_photo(path, model=model, prompt=prompt)
        elapsed = time.time() - t0

        if desc:
            # Show a preview of the description
            preview = desc[:80] + "..." if len(desc) > 80 else desc
            print(f" ({elapsed:.1f}s) {preview}")
        else:
            print(f" ({elapsed:.1f}s) no description generated")

        results.append(desc)

    return results
