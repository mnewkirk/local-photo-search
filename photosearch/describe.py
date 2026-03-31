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
import time
from pathlib import Path
from typing import Optional

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

# Default model — llava is the most widely available multimodal model on Ollama.
# Other options: llava:13b (better quality, slower), llava-llama3, moondream.
MODEL = "llava"

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


def describe_photo(
    image_path: str,
    model: str = MODEL,
    prompt: str = DESCRIBE_PROMPT,
) -> Optional[str]:
    """Generate a description for a single photo.

    Args:
        image_path: Path to the image file (JPEG, PNG, etc.)
        model: Ollama model name (must be multimodal).
        prompt: The prompt to send alongside the image.

    Returns:
        A text description string, or None if generation failed.
    """
    if not HAS_OLLAMA:
        return None

    path = Path(image_path)
    if not path.exists():
        print(f"  Warning: image not found: {image_path}")
        return None

    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [str(path)],
            }],
        )
        text = response.message.content.strip()
        return text if text else None

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
# Semantic tagging — M9
# ---------------------------------------------------------------------------
# A fixed vocabulary of category tags that the LLM assigns to each photo.
# These replace the hand-curated _TERM_EXPANSIONS dictionary in search.py
# with LLM-quality semantic understanding at zero search-time cost.

TAG_VOCABULARY = [
    # Living things
    "animal", "bird", "fish", "insect", "pet", "wildlife",
    # People
    "person", "child", "group", "crowd", "portrait",
    # Activities
    "action", "sports", "playing", "walking", "swimming", "surfing",
    "running", "climbing", "dancing", "eating", "cooking", "working",
    # Scenes & settings
    "landscape", "seascape", "cityscape", "mountain", "forest", "desert",
    "beach", "ocean", "lake", "river", "waterfall", "sky", "sunset",
    "sunrise", "night",
    # Built environment
    "building", "architecture", "street", "road", "bridge", "vehicle",
    "car", "boat", "airplane",
    # Nature
    "flower", "plant", "tree", "garden", "rock", "snow", "rain",
    # Indoor
    "indoor", "home", "kitchen", "room", "office",
    # Mood / style
    "dramatic", "peaceful", "colorful", "dark", "bright", "foggy",
    "silhouette", "reflection",
    # Objects
    "food", "drink", "sign", "art", "sculpture", "flag",
    # Photography qualities
    "close-up", "wide-angle", "aerial", "underwater",
]

TAG_PROMPT = """\
Tag this photo using ONLY tags from this list: {tags}

Rules:
- Return ONLY a comma-separated list of tags that apply. Nothing else.
- Include every tag that clearly applies. Omit any that don't.
- If you see a specific animal (e.g. hawk, deer), include "animal" AND "bird" or "wildlife" as appropriate.
- If people are present, include "person" and relevant activity tags.
- Include setting tags (beach, ocean, forest, etc.) and mood tags (dramatic, peaceful, etc.) when they apply.
- Do NOT add tags that are not in the list above.
""".format(tags=", ".join(TAG_VOCABULARY))


def tag_photo(
    image_path: str,
    model: str = MODEL,
) -> Optional[list[str]]:
    """Generate semantic tags for a single photo via Ollama.

    Returns a list of tags from TAG_VOCABULARY, or None if generation failed.
    """
    raw = describe_photo(image_path, model=model, prompt=TAG_PROMPT)
    if not raw:
        return None

    # Parse comma-separated response, validate against vocabulary
    vocab_set = set(TAG_VOCABULARY)
    tags = []
    for token in raw.split(","):
        tag = token.strip().lower().rstrip(".")
        if tag in vocab_set:
            tags.append(tag)

    return tags if tags else None


def describe_photos_batch(
    image_paths: list[str],
    model: str = MODEL,
    prompt: str = DESCRIBE_PROMPT,
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
