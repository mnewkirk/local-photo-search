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
import json
import os
import re
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


_REACHABILITY_BACKOFF_S = (5, 10, 20, 40, 60)


def check_available(model: str = MODEL) -> None:
    """Raise a clear error if the LLM backend isn't reachable or the model isn't pulled."""
    base = os.environ.get("PHOTOSEARCH_TEXT_LLM_URL")
    if base:
        # OpenAI-compatible backend (LM Studio / llama-server) for the text
        # passes — verify it instead of Ollama. Scoped to text-pass workers;
        # don't set PHOTOSEARCH_TEXT_LLM_URL on a worker running vision passes.
        # Retried with backoff (~2.5 min in all): LM Studio stops answering
        # while it JIT-loads a model, and a pass that starts right after its
        # model was ejected hits exactly that window. This check runs once per
        # batch OUTSIDE the per-photo retry wrapper, so a single timeout here
        # used to kill the whole worker process mid-fleet (2026-09-26).
        import urllib.request
        last: Exception | None = None
        for delay in (0,) + _REACHABILITY_BACKOFF_S:
            if delay:
                time.sleep(delay)
            try:
                urllib.request.urlopen(base.rstrip("/") + "/models", timeout=10).read()
                return
            except Exception as e:  # noqa: BLE001 — retried, then re-raised
                last = e
        raise RuntimeError(
            f"Cannot reach OpenAI-compatible LLM at {base} "
            f"(PHOTOSEARCH_TEXT_LLM_URL):\n  {last}"
        ) from last
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

# --- LLM call instrumentation (diagnosing the freeze-then-bulk-release stall) ---
# Always-on slow-call warning + an in-flight watchdog so a frozen slot is visible
# live in the worker log (with wall-clock timestamps + token counts), instead of
# only showing up as one big per-photo number.
#   PHOTOSEARCH_LLM_TRACE=1        -> log every call's timing + tokens (not just slow ones)
#   PHOTOSEARCH_LLM_SLOW_WARN_S=60 -> warn when one call exceeds this many seconds
#   PHOTOSEARCH_LLM_WATCHDOG_S=30  -> "still waiting Ns" heartbeat cadence during a call
_LLM_TRACE = bool(os.environ.get("PHOTOSEARCH_LLM_TRACE"))
_LLM_SLOW_WARN_S = float(os.environ.get("PHOTOSEARCH_LLM_SLOW_WARN_S", "60"))
_LLM_WATCHDOG_S = float(os.environ.get("PHOTOSEARCH_LLM_WATCHDOG_S", "30"))


# Per-ATTEMPT observer for the eval harnesses (photosearch/model_eval.py).
# The retry loop below hides what an eval needs to count: a timeout that
# recovered on attempt 2 never surfaces, and `finish_reason` (truncation at
# max_tokens) is dropped. When set, it is called once per attempt as
# hook(role=, model=, attempt=, outcome="ok"|"timeout"|"error", elapsed=,
# completion_tokens=, finish_reason=). None in production; its own exceptions
# are swallowed so an observer can never change a call's result.
_ATTEMPT_HOOK = None


class UnusableAnswer(ValueError):
    """The model ANSWERED, but the answer must not be stored.

    Cut off at the token limit, a refusal, a vocabulary recitation, a keyword
    list that is not about the description. Every pass used to store these as
    a success — 4,350 descriptions end mid-sentence on the live library — so
    the bad output was invisible until a later audit.

    Deliberately NOT a TimeoutError / ConnectionError: it is the PHOTO's
    attempt, not the worker's transport problem. The worker reports it as a
    failure row, which spends one of MAX_PROCESS_ATTEMPTS and logs to
    `index_errors` — so a photo that always produces garbage is retired by the
    cap instead of being deferred (text passes) or re-claimed forever.
    """


class TruncatedOutput(UnusableAnswer):
    """The model stopped because it hit its output-token limit
    (`finish_reason` / `done_reason` == "length") — even after one retry with
    double the budget. `text` is the partial output, for diagnostics only."""

    def __init__(self, message: str, text: Optional[str] = None):
        super().__init__(message)
        self.text = text


class ContextOverflow(RuntimeError):
    """The request does not fit the model's context window.

    LM Studio splits a model's context across its parallel slots, so a
    multi-image request (a video's K keyframes) can 400 with "Context size has
    been exceeded" -- Risk 8 of docs/plans/video-support.md. That is a
    CONFIGURATION failure (raise the model's per-model context default, or
    send fewer / smaller frames), never an answer: it is deliberately neither
    an UnusableAnswer (which would spend the video's attempts) nor a
    TimeoutError / ConnectionError (which would defer and retry forever).

    The message deliberately avoids the words the timeout classifiers match
    ("exceeded", "timeout"); the backend's own error text is on `.detail`.
    """

    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(message)
        self.detail = detail


class VideoTransportError(ConnectionError):
    """describe_video(raise_unusable=True) could not reach the model at all
    (timeout, refused connection, backend error). A ConnectionError on
    purpose: the worker treats it as transient and defers the video without
    spending an attempt — distinct from an answer that was unusable."""


class RoleModelNotConfigured(RuntimeError):
    """A strict role (see `_STRICT_ROLES`) has no model configured on the
    OpenAI-compatible route. Raised instead of silently falling back to
    another role's model -- the fleet should fail at startup
    (`require_role_model`), not log a video description under the wrong
    model."""


# Substrings a backend uses for "the prompt does not fit the context".
# LM Studio: "Context size has been exceeded"; llama.cpp: "exceeds the
# available context size" / n_ctx; OpenAI-style: "maximum context length";
# Ollama: "input length exceeds context length".
_CONTEXT_OVERFLOW_PATTERNS = (
    "context size", "context length", "context window", "n_ctx",
    "maximum context", "available context", "too many tokens",
)


def _context_overflow_detail(e: BaseException) -> Optional[str]:
    """The backend's error text when `e` is a context overflow, else None.

    An HTTP 400/413 carries the reason only in its body, which `_do_request`
    stashes on the exception as `photosearch_body`; Ollama puts it in the
    message itself."""
    if isinstance(e, ContextOverflow):
        return e.detail or str(e)
    body = getattr(e, "photosearch_body", None)
    code = getattr(e, "code", None)
    candidates = []
    if body and code in (400, 413):
        candidates.append(str(body))
    candidates.append(str(e))
    for text in candidates:
        low = text.lower()
        if any(p in low for p in _CONTEXT_OVERFLOW_PATTERNS):
            return text[:500]
    return None


def _context_overflow(role, model, detail) -> ContextOverflow:
    return ContextOverflow(
        f"LLM {role or 'call'} request does not fit the context window of "
        f"model={model} -- raise that model's context length (LM Studio "
        f"per-model default, 16384+ for video) or send fewer/smaller images",
        detail)


# Output-token budget per role on the OpenAI-compatible route. It used to be a
# flat 768 for every call, which let a reasoning model that was left thinking
# spend 700 tokens before answering a 10-token tag question. With reasoning off
# (PHOTOSEARCH_LLM_REASONING_EFFORT=none) the real answers are far smaller —
# describe ~100 tokens, a visual tag line ~10, a category list <= ~200 (gemma
# lists 25 tags on average, 52 at most). A truncation retries ONCE at double
# the budget before it is treated as unusable, so these are bars to clear,
# not hard ceilings. Unlisted roles (aesthetics: a JSON critique) keep 768.
_OPENAI_MAX_TOKENS = {
    "describe": 512,
    "verify": 512,
    "visual": 128,
    "text": 256,
    # Video roles (docs/plans/video-support.md steps 3b / 5). `subject` is a
    # short grounding JSON array; `video` is a whole-clip description, longer
    # than a photo's because it narrates a sequence over K frames.
    "subject": 256,
    "video": 768,
}
_DEFAULT_OPENAI_MAX_TOKENS = 768

# The stop reason both backends report when generation ran out of tokens.
_LENGTH = "length"


def _is_timeout(e: BaseException) -> bool:
    es = str(e).lower()
    return isinstance(e, TimeoutError) or "timed out" in es or "exceeded" in es \
        or "timeout" in es


def _expanded_retry(call, budget, role, model, partial):
    """Run the double-budget retry after a confirmed truncation.

    A TIMEOUT here is not a transport stall: the first call already proved
    the answer does not fit the budget, and the bigger one could not finish
    inside the wall-clock cap. Treating it as a stall deferred the photo
    without spending an attempt, so it was claimed again at once — forever.
    IMAG2074 (2026-10-06) was claimed ~90 times in 1.5 h: gemma loops
    `railing, railing, …` at temperature 0 until any limit it is given.
    Connection errors are still re-raised as transport failures.
    """
    try:
        return call()
    except ContextOverflow:
        raise
    except Exception as e:
        if _is_timeout(e):
            raise TruncatedOutput(
                f"LLM {role or 'call'} output overran {budget} tokens and the "
                f"{budget * 2}-token retry timed out (model={model})", partial) from e
        raise


def _notify_attempt(**kw):
    hook = _ATTEMPT_HOOK
    if hook is None:
        return
    try:
        hook(**kw)
    except Exception:
        pass


def _llm_trace(role, model, elapsed, completion_tokens=None, prompt_tokens=None):
    """Emit a per-call timing line. Always fires for slow calls (>=
    _LLM_SLOW_WARN_S); fires for every call when PHOTOSEARCH_LLM_TRACE is set.
    The computed t/s is the headline signal — a healthy decode is tens of t/s;
    the stall drops it below ~1 t/s."""
    slow = elapsed >= _LLM_SLOW_WARN_S
    if not (_LLM_TRACE or slow):
        return
    parts = [f"role={role or '?'}", f"model={model}", f"elapsed={elapsed:.1f}s"]
    if completion_tokens:
        parts.append(f"out_tok={completion_tokens}")
        if elapsed > 0:
            parts.append(f"t/s={completion_tokens / elapsed:.2f}")
    if prompt_tokens:
        parts.append(f"in_tok={prompt_tokens}")
    tag = "⚠ SLOW-LLM" if slow else "llm-trace"
    print(f"\n  [{tag} {time.strftime('%H:%M:%S')}] " + " ".join(parts), flush=True)


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
    role: Optional[str] = None,
) -> Optional[str]:
    """Call ollama.chat with retry logic for transient failures.

    Each attempt is bounded by `timeout` seconds (default
    _DEFAULT_OLLAMA_TIMEOUT_S). Wedged Ollama runners that would otherwise
    block forever are aborted, the call is treated as a transient error,
    and the retry loop kicks in. The underlying thread is daemon, so it
    dies with the process if it can't be cleanly interrupted.

    Retries on timeouts, connection errors, and server errors.
    Returns the response text or None.

    If PHOTOSEARCH_TEXT_LLM_URL is set, ALL calls (text and vision) route to that
    OpenAI-compatible endpoint instead of Ollama — model names mapped via
    PHOTOSEARCH_LLM_MODEL_MAP, images converted to the OpenAI vision format.
    """
    base = os.environ.get("PHOTOSEARCH_TEXT_LLM_URL")
    if base:
        return _openai_route(base, model, messages, retries, timeout, options, role)

    text, done_reason = _ollama_chat_attempts(model, messages, retries, options, timeout)
    if done_reason != _LENGTH:
        return text
    # Ran out of num_predict. Answering with a cut-off description (llava's
    # ~680-char descriptions were num_predict=150) is what this guards.
    budget = (options or {}).get("num_predict")
    if not budget or budget < 0:
        # No per-call cap to raise — the context window itself filled.
        raise TruncatedOutput(
            f"LLM {role or 'call'} output was cut off (done_reason=length, "
            f"model={model})", text)
    bigger = dict(options, num_predict=budget * 2)
    print(f" [output cut off at {budget} tokens; retrying at {budget * 2}]",
          end="", flush=True)
    text, done_reason = _expanded_retry(
        lambda: _ollama_chat_attempts(model, messages, retries, bigger, timeout),
        budget, role, model, text)
    if done_reason == _LENGTH:
        raise TruncatedOutput(
            f"LLM {role or 'call'} output was cut off at {budget * 2} tokens "
            f"twice (done_reason=length, model={model})", text)
    return text


def _ollama_chat_attempts(model, messages, retries, options, timeout):
    """The transport-retry loop for one Ollama call. Returns (text, done_reason)."""
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
                reason = getattr(response, "done_reason", None)
                result_q.put(("ok", (text if text else None,
                                     reason if isinstance(reason, str) else None)))
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
        detail = _context_overflow_detail(e)
        if detail is not None:
            raise _context_overflow(None, model, detail) from e
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
    return None, None


def _openai_chat_with_retry(
    base_url: str,
    model: str,
    messages: list,
    retries: int = _MAX_RETRIES,
    timeout: Optional[float] = None,
    temperature: float = 0.0,
    max_tokens: int = _DEFAULT_OPENAI_MAX_TOKENS,
    role: Optional[str] = None,
) -> Optional[str]:
    """Call an OpenAI-compatible /chat/completions endpoint (LM Studio,
    llama-server, ...). Same return contract as _ollama_chat_with_retry: text on
    success, None on empty, raises on persistent timeout/connection error so the
    caller's defer-on-error path still works.

    A response that stopped at `max_tokens` (`finish_reason == "length"`) is
    retried once at double the budget and then raises TruncatedOutput — it is
    never returned as if it were a whole answer. That includes an EMPTY
    response cut off mid-reasoning: a thinking model with reasoning left on
    spends the whole budget thinking and returns '' (the 2026-10-03 batch).
    """
    text, finish = _openai_chat_attempts(base_url, model, messages, retries,
                                         timeout, temperature, max_tokens, role)
    if finish != _LENGTH:
        return text
    print(f" [output cut off at {max_tokens} tokens; retrying at {max_tokens * 2}]",
          end="", flush=True)
    text, finish = _expanded_retry(
        lambda: _openai_chat_attempts(base_url, model, messages, retries,
                                      timeout, temperature, max_tokens * 2, role),
        max_tokens, role, model, text)
    if finish == _LENGTH:
        raise TruncatedOutput(
            f"LLM {role or 'call'} output was cut off at {max_tokens * 2} tokens "
            f"twice (finish_reason=length, model={model})", text)
    return text


def _openai_chat_attempts(
    base_url: str,
    model: str,
    messages: list,
    retries: int,
    timeout: Optional[float],
    temperature: float,
    max_tokens: int,
    role: Optional[str],
):
    """The transport-retry loop for one OpenAI-compatible call.
    Returns (text, finish_reason).

    `timeout` is a true WALL-CLOCK cap per attempt: the request runs in a daemon
    thread and is abandoned via queue.get(timeout=) if it overruns, mirroring the
    Ollama route. urllib's own timeout is only a socket-*idle* timeout — a server
    that trickles tokens (a frozen llama.cpp slot at <1 t/s) never trips it and
    would otherwise run unbounded (a 966s LM Studio call was observed).
    """
    import queue as _queue
    import threading as _threading
    import urllib.request
    if timeout is None:
        timeout = _DEFAULT_OLLAMA_TIMEOUT_S
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    # Reasoning models (gemma-4, qwen3.x) think by default and, on a short
    # tag/JSON task, spend the ENTIRE max_tokens budget on hidden reasoning
    # and return an empty content string — measured on gemma-4-26b: 765 of 768
    # tokens reasoning, '' back, ~80 s/photo, every photo UNANSWERED.
    # `reasoning_effort: "none"` is the OpenAI-style switch LM Studio honours
    # (3.6 s and a real answer); `chat_template_kwargs.enable_thinking` did
    # nothing. Opt-in via env so non-reasoning backends never see the field.
    effort = os.environ.get("PHOTOSEARCH_LLM_REASONING_EFFORT", "").strip()
    if effort:
        body["reasoning_effort"] = effort
    payload = json.dumps(body).encode("utf-8")
    for attempt in range(1, retries + 1):
        t0 = time.time()
        wd_stop = _threading.Event()
        result_q: _queue.Queue = _queue.Queue(maxsize=1)

        def _watchdog(start=t0, att=attempt):
            while not wd_stop.wait(_LLM_WATCHDOG_S):
                print(
                    f"\n  ⏳ [{time.strftime('%H:%M:%S')}] LLM {role or 'call'} "
                    f"still waiting {time.time() - start:.0f}s "
                    f"(model={model}, attempt {att}/{retries})",
                    flush=True,
                )

        def _do_request():
            try:
                req = urllib.request.Request(
                    url, data=payload, headers={"Content-Type": "application/json"})
                # urllib's timeout is socket-idle only; kept as a backstop so an
                # abandoned thread eventually dies.
                resp = urllib.request.urlopen(req, timeout=timeout)
                result_q.put(("ok", json.loads(resp.read())))
            except Exception as ex:
                # An HTTP 400's reason (e.g. LM Studio's "Context size has
                # been exceeded") is only in the body; keep it for
                # _context_overflow_detail.
                if getattr(ex, "code", None) in (400, 413) and hasattr(ex, "read"):
                    try:
                        ex.photosearch_body = ex.read().decode("utf-8", "replace")
                    except Exception:
                        pass
                result_q.put(("err", ex))

        _threading.Thread(target=_watchdog, daemon=True).start()
        _threading.Thread(target=_do_request, daemon=True).start()
        try:
            kind, val = result_q.get(timeout=timeout)  # hard wall-clock cap
        except _queue.Empty:
            kind, val = "err", TimeoutError(
                f"openai chat exceeded {timeout}s wall-clock on attempt {attempt}/{retries}")
        finally:
            wd_stop.set()

        if kind == "ok":
            data = val
            usage = data.get("usage") or {}
            _llm_trace(role, model, time.time() - t0,
                       usage.get("completion_tokens"), usage.get("prompt_tokens"))
            choice = data["choices"][0]
            _notify_attempt(role=role, model=model, attempt=attempt, outcome="ok",
                            elapsed=time.time() - t0,
                            completion_tokens=usage.get("completion_tokens"),
                            finish_reason=choice.get("finish_reason"))
            text = (choice["message"]["content"] or "").strip()
            return (text if text else None), choice.get("finish_reason")

        e = val
        es = str(e).lower()
        _notify_attempt(role=role, model=model, attempt=attempt,
                        outcome="timeout" if isinstance(e, TimeoutError)
                        or "timed out" in es or "timeout" in es else "error",
                        elapsed=time.time() - t0, completion_tokens=None,
                        finish_reason=None)
        detail = _context_overflow_detail(e)
        if detail is not None:
            raise _context_overflow(role, model, detail) from e
        transient = any(k in es for k in [
            "timeout", "timed out", "exceeded", "connection", "refused", "reset",
            "broken pipe", "502", "503", "500", "unavailable",
        ])
        if transient and attempt < retries:
            print(f" [retry {attempt}/{retries} after {time.time() - t0:.0f}s, "
                  f"in {_RETRY_DELAY}s: {e}]", end="", flush=True)
            time.sleep(_RETRY_DELAY)
        else:
            raise e
    return None, None


def _resolve_openai_model(model: str, role: Optional[str] = None) -> str:
    """Resolve the OpenAI / LM Studio model id for a call by its descriptive
    ROLE, not by model name. Roles name the *pass* the call serves:

        text       -> category-content + keywords    PHOTOSEARCH_LLM_TEXT_MODEL
        describe   -> describe pass + verify regen    PHOTOSEARCH_LLM_DESCRIBE_MODEL
        verify     -> verify's independent verifier   PHOTOSEARCH_LLM_VERIFY_MODEL
        visual     -> category-visual tags            PHOTOSEARCH_LLM_VISUAL_MODEL
        aesthetics -> VLM aesthetic scoring           PHOTOSEARCH_LLM_AESTHETICS_MODEL
        subject    -> video-subjects grounding        PHOTOSEARCH_LLM_SUBJECT_MODEL
        video      -> video-describe                  PHOTOSEARCH_LLM_VIDEO_MODEL

    So you point each role at whatever you actually loaded in LM Studio (Gemma,
    Qwen-VL, llava, ...) without the config keys pretending to be model names.
    Falls back to PHOTOSEARCH_TEXT_LLM_MODEL (legacy single-model) then the raw
    name.

    `subject` falls back to the AESTHETICS model first: the photo subject pass
    (`subjects.ground_photo_subjects`) grounds under role `aesthetics`, so an
    unset SUBJECT var grounds video keyframes with exactly the model photos
    use today. `video` is STRICT (`_STRICT_ROLES`): with its env var unset it
    raises RoleModelNotConfigured rather than falling back to anything (D8 /
    step 5 of docs/plans/video-support.md).
    """
    if role:
        v = os.environ.get(f"PHOTOSEARCH_LLM_{role.upper()}_MODEL")
        if v:
            return v
        if role in _STRICT_ROLES:
            raise RoleModelNotConfigured(
                f"no model configured for LLM role {role!r}: set "
                f"PHOTOSEARCH_LLM_{role.upper()}_MODEL (this role deliberately "
                f"does not fall back to another role's model)")
        if role == "subject":
            aes = os.environ.get("PHOTOSEARCH_LLM_AESTHETICS_MODEL")
            if aes:
                return aes
        # Vision roles reuse the configured vision model if their own role env
        # var is unset — the raw Ollama defaults (e.g. "qwen2.5-vl",
        # "llama3.2-vision") are NOT valid LM Studio ids, so falling through to
        # `model` would 400 with "Invalid model identifier". Mirrors
        # rerun._resolve_model. `aesthetics` is the pass that surfaced this.
        if role in ("describe", "verify", "visual", "aesthetics", "subject"):
            visual = os.environ.get("PHOTOSEARCH_LLM_VISUAL_MODEL")
            if visual:
                return visual
    return os.environ.get("PHOTOSEARCH_TEXT_LLM_MODEL") or model


# pass -> LLM role. The role, not the model name, is what selects the model on
# the OpenAI-compatible (LM Studio) route, so it is also what provenance has to
# resolve through. `rerun._PASS_LLM` carries the same roles alongside its Ollama
# defaults; a test pins the two together.
PASS_ROLES = {
    "describe":         "describe",
    "verify":           "verify",
    "category-content": "text",
    "keywords":         "text",
    "category-visual":  "visual",
    "aesthetics":       "aesthetics",
    # docs/plans/video-support.md step 3b: per-segment keyframe grounding.
    "video-subjects":   "subject",
    # step 5: one multi-image call per clip, strict role (no fallback, D8).
    "video-describe":   "video",
}

# Roles that must be configured explicitly on the OpenAI-compatible route —
# no fallback to another role's model, no legacy single-model var.
_STRICT_ROLES = frozenset({"video"})

# The video passes' roles. Both are wired now (video-subjects in step 3b,
# video-describe in step 5) and live in PASS_ROLES, which
# tests/test_llm_provenance.py pins to rerun._PASS_LLM; this view is kept for
# callers that want only the video passes.
VIDEO_PASS_ROLES = {p: r for p, r in PASS_ROLES.items() if p.startswith("video-")}


def require_role_model(role: str, model: Optional[str] = None) -> Optional[str]:
    """Fail fast when `role` cannot resolve to a model — for fleet startup.

    OpenAI-compatible route (PHOTOSEARCH_TEXT_LLM_URL set): returns the model
    id the role resolves to (`_resolve_openai_model`, the same function the
    request uses). Raises RoleModelNotConfigured for a strict role (`video`)
    whose env var is unset, or for any role that resolves to nothing.

    Ollama route: models are chosen by NAME, not role, so this returns
    `model` unchanged (None when not given) and never raises — the passed
    name (the pass's CLI default or --*-model flag) is what runs.
    """
    if not os.environ.get("PHOTOSEARCH_TEXT_LLM_URL"):
        return model
    resolved = _resolve_openai_model(model or "", role)
    if not resolved:
        raise RoleModelNotConfigured(
            f"no model configured for LLM role {role!r}: set "
            f"PHOTOSEARCH_LLM_{role.upper()}_MODEL")
    return resolved


def effective_model(model: str, role: Optional[str] = None) -> str:
    """The model id a call with (`model`, `role`) will ACTUALLY hit.

    This is what belongs in `generations.model_used`. Logging the configured
    name instead is why 159,647 of 159,650 `category-visual` rows on the live
    library claim `llava`: on the LM Studio route the name is ignored and the
    model is chosen by role, so the log recorded a model that never ran.

    Deliberately the SAME function the request uses (`_resolve_openai_model`),
    not a reimplementation — a second copy would drift and be undetectable,
    since a wrong provenance string still looks like a string.
    """
    if os.environ.get("PHOTOSEARCH_TEXT_LLM_URL"):
        return _resolve_openai_model(model, role)
    return model


def unload_openai_models(models) -> list[str]:
    """Unload `models` (effective ids) from LM Studio; return the ids unloaded.

    LM Studio keeps every JIT-loaded model resident until told otherwise, so a
    fleet that walks describe -> visual -> text -> verify ends with all of them
    in VRAM at once (2026-10-04: four 16k-context models on one 24 GB card).
    Ollama evicts on its own; this is the LM Studio half. Uses LM Studio's
    native REST API (`/api/v1/models`, `/api/v1/models/unload`), which a plain
    OpenAI-compatible server lacks — so it is best-effort: any failure, or a
    backend without that API, unloads nothing and never raises. JIT loading
    brings a model back on the next request that needs it.
    """
    import urllib.request
    base = os.environ.get("PHOTOSEARCH_TEXT_LLM_URL")
    wanted = {m for m in models if m}
    if not base or not wanted:
        return []
    root = base.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        with urllib.request.urlopen(f"{root}/api/v1/models", timeout=10) as r:
            listing = json.load(r).get("models", [])
    except Exception as e:
        print(f"  (model unload skipped: cannot list LM Studio models: {e})", flush=True)
        return []
    done = []
    for m in listing:
        for inst in m.get("loaded_instances") or []:
            iid = inst.get("id")
            if m.get("key") not in wanted and iid not in wanted:
                continue
            try:
                req = urllib.request.Request(
                    f"{root}/api/v1/models/unload",
                    data=json.dumps({"instance_id": iid}).encode("utf-8"),
                    headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=60).read()
                done.append(iid)
            except Exception as e:
                print(f"  (could not unload {iid}: {e})", flush=True)
    return done


def effective_model_version(model: str) -> Optional[str]:
    """Provenance digest for `generations.model_version`.

    On the OpenAI-compatible route there is no Ollama to query and asking
    anyway blocks ~80 s retrying localhost:11434, so use a static marker.
    Otherwise it is the short Ollama digest.
    """
    if os.environ.get("PHOTOSEARCH_TEXT_LLM_URL"):
        return "lmstudio"
    from .worker import _model_version  # lazy: worker imports describe
    return _model_version(model)


def _image_ref_to_b64(ref):
    """Normalize an Ollama image ref (file path OR base64) to bare base64."""
    try:
        if isinstance(ref, str) and len(ref) < 1024 and os.path.exists(ref):
            return _encode_image_for_ollama(ref)  # resize + base64
        return ref  # already base64
    except Exception:
        return None


def _to_openai_message(m: dict) -> dict:
    """Convert an Ollama-style message ({content, images:[...]}) to OpenAI shape.
    Image messages become a content list with image_url data URIs (vision)."""
    role = m.get("role", "user")
    imgs = m.get("images")
    if not imgs:
        return {"role": role, "content": m.get("content", "")}
    content = [{"type": "text", "text": m.get("content", "")}]
    for ref in imgs:
        b64 = _image_ref_to_b64(ref)
        if b64:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return {"role": role, "content": content}


def _openai_route(base, model, messages, retries, timeout, options, role=None):
    """Route an Ollama-style chat call (text or vision) to an OpenAI endpoint."""
    conv = [_to_openai_message(m) for m in messages]
    temp = (options or {}).get("temperature", 0.0)
    return _openai_chat_with_retry(
        base, _resolve_openai_model(model, role), conv,
        retries=retries, timeout=timeout, temperature=temp,
        max_tokens=_OPENAI_MAX_TOKENS.get(role, _DEFAULT_OPENAI_MAX_TOKENS),
        role=role)


def _text_chat_with_retry(model, messages, options=None, timeout=None):
    # Backend routing (Ollama vs OpenAI endpoint) lives in _ollama_chat_with_retry;
    # this remains the text-pass entry point (role 'text').
    return _ollama_chat_with_retry(model=model, messages=messages, options=options,
                                   timeout=timeout, role="text")


def ends_mid_sentence(text: str) -> bool:
    """True when a description stops mid-sentence — no closing punctuation.

    The write-time twin of `stale_descriptions.is_truncated` (which audits
    what is already stored). It catches the cut-offs no stop reason reports:
    llama3.2-vision's 20-60-token stops ended on a normal EOS, so neither
    `done_reason` nor `finish_reason` said anything was wrong.
    """
    from .stale_descriptions import is_truncated
    return is_truncated(text)


def _description_problem(text: Optional[str]) -> Optional[str]:
    """Why a describe answer must not be stored, or None if it is fine."""
    if text is None:
        return "empty response"
    if not _is_valid_description(text):
        return "not a description"
    if _is_degenerate(text):
        return "repetition loop"
    if ends_mid_sentence(text):
        return "ends mid-sentence"
    return None


def describe_photo(
    image_path: str,
    model: str = MODEL,
    prompt: Optional[str] = None,
    raise_unusable: bool = False,
) -> Optional[str]:
    """Generate a description for a single photo.

    Args:
        image_path: Path to the image file (JPEG, PNG, etc.)
        model: Ollama model name (must be multimodal).
        prompt: The prompt to send alongside the image. Defaults to the
                model-appropriate prompt (moondream gets a simpler question-style
                prompt; all others get the structured DESCRIBE_PROMPT).
        raise_unusable: when the model answered but every answer was unusable
                (cut off, a repetition loop, ends mid-sentence), raise
                UnusableAnswer instead of returning None. The worker sets it,
                so the photo becomes a failure row — one attempt spent AND an
                `index_errors` entry — instead of a silent "no description".

    Returns:
        A text description string, or None if generation failed.

    A bad answer is never returned as a description. This used to hand back
    whatever the last retry produced, even a degenerate one, and stored cut-off
    text as a finished description — 4,350 of them on the live library.
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

    def _ask(model_, messages_, options_):
        """(text, problem). A truncation the chat helper could not recover
        from is a problem like any other, not an exception."""
        try:
            text = _ollama_chat_with_retry(model=model_, messages=messages_,
                                           options=options_, role="describe")
        except TruncatedOutput as e:
            return None, str(e)
        return text, _description_problem(text)

    try:
        result, problem = _ask(model, messages, options)

        # Moondream sometimes returns empty or coordinate garbage on the first
        # attempt. Retry once with a bare fallback question — different enough
        # to avoid the same failure mode.
        if problem and model.startswith("moondream"):
            result, problem = _ask(model, [{
                "role": "user",
                "content": "Describe what you see in this photo in 2 sentences.",
                "images": [image_ref],
            }], options)

        # Recovery: llama3.2-vision occasionally falls into repetition loops.
        # Its options carry temperature>0 so each retry is independent — a
        # 100-image bakeoff showed up to 2 retries clears every loop. If
        # retries still fail, fall back to llava (not loop-prone under greedy
        # decoding). A cut-off or mid-sentence answer takes the same path.
        if problem and problem not in ("empty response", "not a description"):
            for _ in range(2):
                result, problem = _ask(model, messages, options)
                if not problem:
                    break
            if problem and not model.startswith("llava"):
                result, problem = _ask("llava", messages, _options_for_model("llava"))
    except Exception as e:
        print(f"  Warning: description failed for {path.name}: {e}")
        return None

    if problem is None:
        return result
    if problem in ("empty response", "not a description"):
        return None        # unchanged contract: "no description" is not news
    if raise_unusable:
        raise UnusableAnswer(f"describe: {problem}")
    print(f"  Warning: no usable description for {path.name}: {problem}")
    return None


# ---------------------------------------------------------------------------
# Video describe (docs/plans/video-support.md step 5)
# ---------------------------------------------------------------------------

# The video-describe prompt. A module constant so the eval harness can swap it
# (pass `prompt=` to describe_video). Placeholders are replaced literally, not
# via str.format, so a swapped prompt may contain other braces freely:
#   {n_frames}   number of frames sent
#   {duration}   clip running time, e.g. "42.0 s" or "12:05 (725.0 s)"
#   {frame_list} one line per frame: "frame 1 at 0.0 s"
#   {subjects}   the subjects-over-time block (with trailing newline) or ""
VIDEO_DESCRIBE_PROMPT = """\
These {n_frames} images are frames sampled in order from ONE video clip that \
runs {duration}. Frame times:
{frame_list}
{subjects}Describe what HAPPENS over the course of the clip in 3 to 5 \
sentences: who or what is in it, what they do, and how things change from the \
start to the end (actions, the sequence of events, camera movement). Mention \
the setting and any clearly readable text. Describe it as one continuous \
event: do not describe the frames one by one and do not mention frames, \
images or timestamps. Be factual; only use a person's name if it is given \
above. End with a complete sentence.
"""

# How the optional subjects-over-time summary is introduced in the prompt.
VIDEO_SUBJECTS_PREFIX = (
    "A separate detector found these main subjects over time (it may be "
    "imperfect): "
)

# Ollama options for a multi-image request. K keyframes at ~1024 px are
# ~1k tokens each on qwen-VL, so the 8k photo context is too small.
_VIDEO_NUM_CTX = 16384
_VIDEO_NUM_PREDICT = 768


def _fmt_seconds(s: float) -> str:
    s = max(0.0, float(s))
    if s < 60:
        return f"{s:.1f} s"
    m, sec = divmod(int(round(s)), 60)
    return f"{m}:{sec:02d} ({s:.1f} s)"


def build_video_prompt(timestamps, duration_s: float,
                       subject_summary: Optional[str] = None,
                       template: Optional[str] = None) -> str:
    """Fill the video-describe template (VIDEO_DESCRIBE_PROMPT by default)."""
    tpl = VIDEO_DESCRIBE_PROMPT if template is None else template
    frame_list = "\n".join(f"frame {i} at {float(t):.1f} s"
                           for i, t in enumerate(timestamps, 1))
    subjects = ""
    if subject_summary and str(subject_summary).strip():
        subjects = VIDEO_SUBJECTS_PREFIX + str(subject_summary).strip() + "\n"
    return (tpl.replace("{n_frames}", str(len(timestamps)))
               .replace("{duration}", _fmt_seconds(duration_s))
               .replace("{frame_list}", frame_list)
               .replace("{subjects}", subjects))


def _frame_to_b64(frame) -> str:
    """A keyframe as bare base64 JPEG. Accepts raw bytes (already-encoded
    JPEG/PNG from video-scan's upload, sent as-is) or a file path (resized to
    _MAX_IMAGE_PX like a photo)."""
    if isinstance(frame, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(frame)).decode("utf-8")
    path = Path(frame)
    if not path.exists():
        raise FileNotFoundError(f"keyframe not found: {frame}")
    b64 = _encode_image_for_ollama(str(path))
    if b64 is None:
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return b64


def describe_video(
    frames,
    timestamps,
    duration_s: float,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
    raise_unusable: bool = False,
    subject_summary: Optional[str] = None,
) -> Optional[str]:
    """Describe what happens over a video clip from K keyframes in ONE call.

    Args:
        frames: K keyframes, each a file path or encoded image bytes, in time
            order.
        timestamps: K frame times in seconds (same order as `frames`).
        duration_s: the clip's full running time in seconds.
        model: the model name for the Ollama route (default: the describe
            MODEL). On the OpenAI-compatible route the model is chosen by role
            `video` (PHOTOSEARCH_LLM_VIDEO_MODEL) and the name is ignored; that
            role has NO fallback, so an unset var raises
            RoleModelNotConfigured.
        prompt: a template replacing VIDEO_DESCRIBE_PROMPT (same
            placeholders; see build_video_prompt). For the eval.
        raise_unusable: raise UnusableAnswer (TruncatedOutput for a cut-off)
            instead of returning None when the model answered but every answer
            was unusable. The worker sets it so the video becomes a failure
            row.
        subject_summary: optional "0-12 s dog, 12-40 s Calvin" line from the
            video-subjects pass, included in the prompt.

    Returns the description, or None for: no Ollama client, or (only with
    raise_unusable=False) an empty / non-description answer, a transport
    failure, or an unusable answer. With raise_unusable=True every bad answer
    raises UnusableAnswer (after one retry at temperature 0.4) and a
    transport failure raises VideoTransportError (a ConnectionError).

    Raises ContextOverflow when the K-image request does not fit the model's
    context (a config failure, never an answer) and RoleModelNotConfigured
    for a missing video model on the OpenAI route — neither is swallowed.

    Guards are describe's (`_description_problem`): a truncation is retried
    at double the budget inside the chat helper; a cut-off, repetition loop or
    mid-sentence answer gets ONE retry at temperature 0.4, then is unusable.
    """
    frames = list(frames)
    timestamps = [float(t) for t in timestamps]
    if not frames:
        raise ValueError("describe_video needs at least one frame")
    if len(frames) != len(timestamps):
        raise ValueError(f"{len(frames)} frames but {len(timestamps)} timestamps")
    if not HAS_OLLAMA and not os.environ.get("PHOTOSEARCH_TEXT_LLM_URL"):
        return None
    model = model or MODEL
    images = [_frame_to_b64(f) for f in frames]
    text_prompt = build_video_prompt(timestamps, duration_s,
                                     subject_summary=subject_summary,
                                     template=prompt)
    messages = [{"role": "user", "content": text_prompt, "images": images}]

    def _ask(temperature):
        options = {"temperature": temperature, "num_ctx": _VIDEO_NUM_CTX,
                   "num_predict": _VIDEO_NUM_PREDICT}
        try:
            text = _ollama_chat_with_retry(model=model, messages=messages,
                                           options=options, role="video")
        except TruncatedOutput as e:
            return None, f"cut off ({e})", e
        return text, _description_problem(text), None

    empty = ("empty response", "not a description")
    try:
        result, problem, trunc = _ask(0.0)
        # The worker (raise_unusable) gives an empty / non-description answer
        # the same one retry as any other problem, then calls it unusable: a
        # None there used to be deferred, i.e. re-claimed forever.
        if problem and (raise_unusable or problem not in empty):
            result, problem, trunc = _ask(0.4)
    except (ContextOverflow, RoleModelNotConfigured):
        raise
    except UnusableAnswer:
        if raise_unusable:
            raise
        return None
    except Exception as e:
        if raise_unusable:
            # Transport (timeout, refused connection, backend 5xx): the
            # worker's problem, not the video's — a ConnectionError, so the
            # worker's _TRANSIENT handler defers it with no attempt spent.
            raise VideoTransportError(f"video-describe transport failure: "
                                      f"{type(e).__name__}: {e}") from e
        print(f"  Warning: video description failed: {e}")
        return None

    if problem is None:
        return result
    if problem in empty and not raise_unusable:
        return None
    if raise_unusable:
        if trunc is not None:
            raise trunc
        raise UnusableAnswer(f"video-describe: {problem}")
    print(f"  Warning: no usable video description: {problem}")
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

# Hard cap on PERCEIVED tags in one category-visual answer. The first prompt
# had none — its only rule was "Include every tag that clearly applies". Five
# is the cap the prompt states AND what the guard enforces; the prompt also
# tells the model that 1-3 is normal, which is what actually moves the average
# (measured 1.8). The guard does NOT limit how many tags come from one section:
# `backlit, silhouette` describe one photo honestly.
_VISUAL_MAX_TAGS = 5

# Over-selection / regurgitation threshold for the category-visual pass, as a
# `>=` bound — i.e. reject at MORE than the cap the prompt states. It was 12,
# inherited from the 78-term `tags` vocabulary, and the real failure never came
# near it: the observed maximum was 11 tags and the median 5, so the guard
# never fired on the thing that was actually wrong. Defined from
# `_VISUAL_MAX_TAGS` so the prompt and the guard cannot drift apart.
_VISUAL_MAX_PLAUSIBLE_TAGS = _VISUAL_MAX_TAGS + 1


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


# Count BACKSTOP for one category-content answer, judged on the GROUNDED list
# (docs/plans/category-grounding.md). It used to be the only guard, at 60, set
# for llama3.2:3b's recitations (53 to 232 alphabetical terms) — but a count
# cap is blind to a 20-52-term partial recitation and fired on gemma's honest
# 68-tag answer for photo 251357, throwing away ~44 good tags. Recitation is
# now judged by `category_grounding.is_recitation` (vocabulary order AND
# unsupported concrete nouns); this cap only guards a runaway answer. gemma's
# worst observed answer was 84 and the 256-token text budget makes > 100
# nearly impossible on LM Studio, so it mainly guards the Ollama route.
# Alias of `category_grounding.MAX_PLAUSIBLE`, which is the source of truth.
from .category_grounding import MAX_PLAUSIBLE as _CATEGORY_MAX_PLAUSIBLE  # noqa: E402

# Keywords: the prompt asks for 5-15 and no stored row has more than 24.
_KEYWORDS_MAX_PLAUSIBLE = 30
# One "keyword" this long is a whole list the model forgot to comma-separate
# (847 stored rows, e.g. "bride groom wedding ceremony reception lace veil").
_KEYWORD_MAX_WORDS = 5

# Sentinel: the call stalled or errored, so the photo is DEFERRED (re-claimed
# later, no attempt spent) — distinct from a bad answer, which raises.
_DEFER = object()


def _text_pass_answer(pass_name, model, prompt, parse, problem_of,
                      soft_problem_of=None):
    """Ask, judge the answer, retry once (temperature 0.4) if it is bad.

    Returns the parsed list, None to defer (transport stall/error), or raises
    UnusableAnswer when both answers were bad — including a truncation.

    `problem_of(raw, parsed)` judges the RAW parse; a problem makes the answer
    unstorable. `soft_problem_of(parsed)` (optional) names a problem the caller
    can repair but would rather resample once: a storable first answer that
    has one gets ONE retry at 0.4, and the retry wins only when it is clean of
    both kinds. Otherwise — the retry is bad, stalled, or has the same soft
    problem — the first answer is returned, so a soft retry can never make the
    result worse or turn a usable answer into a deferral."""
    def ask(temperature):
        try:
            raw = _text_chat_with_retry(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature, "num_ctx": _TEXT_NUM_CTX},
                timeout=_TEXT_OLLAMA_TIMEOUT_S,
            )
        except TruncatedOutput as e:
            return None, f"cut off ({e})"
        except Exception:
            return _DEFER, None
        if _is_explicit_empty_answer(raw) and (raw or "").strip():
            return [], None               # the model answered: "none"
        if not raw:
            return None, "empty response"
        out = parse(raw)
        return out, problem_of(raw, out)

    out, problem = ask(0)
    if out is _DEFER:
        return None
    if problem is None:
        if soft_problem_of is None or not out or soft_problem_of(out) is None:
            return out
        out2, problem2 = ask(0.4)
        if (out2 is _DEFER or out2 is None or problem2 is not None
                or (out2 and soft_problem_of(out2) is not None)):
            return out
        return out2
    out2, problem2 = ask(0.4)
    if out2 is _DEFER:
        return None
    if problem2 is None:
        return out2
    raise UnusableAnswer(f"{pass_name}: {problem2}")


def extract_categories_from_description(
    description: Optional[str],
    model: str = CATEGORY_CONTENT_MODEL,
    report: Optional[dict] = None,
) -> Optional[list[str]]:
    """Map a description → list of in-vocab categories via a text-only LLM.

    Returns a list (empty only when the model explicitly answered "none") on
    success, or None if the call timed out / errored so the caller can defer
    the photo for retry instead of recording an empty result.

    The returned list is GROUNDED (`category_grounding.ground`): terms the
    description contradicts — `basketball` on a soccer description,
    `artificial lighting` on "outdoors under trees" — are dropped. Grounding
    never adds a term. When the description decides the contradiction there
    is NO retry: the description is the evidence, and a resample adds nothing
    (the 0.4 retry on photo 251357 was worse, 84 tags). Only when two sides of
    a group were emitted and the text cues neither is the answer resampled
    once; if the retry still cannot be decided, both sides are dropped.

    Raises UnusableAnswer when the model answered but, twice, the answer could
    not be stored, judged in this order: cut off at the token limit, empty or
    nothing from the vocabulary, a runaway list (more than
    `category_grounding.MAX_PLAUSIBLE` after grounding), a vocabulary
    recitation (`category_grounding.is_recitation`), or an answer grounding
    would empty ("no grounded categories"). An empty or off-vocabulary answer
    used to be stored as '[]' — done, for good, with no categories.

    `report`, when given, is filled with `raw` (terms before grounding) and
    `dropped` ({term: reason}) for the worker's log line.
    """
    if report is not None:
        report.update(raw=0, dropped={})
    if not description or not description.strip():
        return []
    from .vocab_content import CONTENT_VOCABULARY
    if not HAS_OLLAMA:
        return []
    from . import category_grounding as cg
    vocab_set = set(CONTENT_VOCABULARY)

    def parse(raw):
        out: list[str] = []
        seen: set[str] = set()
        for token in raw.split(","):
            t = token.strip().lower().rstrip(".")
            if t in vocab_set and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def problem_of(raw, cats):
        if not cats:
            return "no in-vocabulary categories"
        g = cg.ground(description, cats)
        if len(g.kept) > cg.MAX_PLAUSIBLE:
            return (f"runaway answer ({len(g.kept)} categories, "
                    f"more than {cg.MAX_PLAUSIBLE})")
        if g.recitation:
            s = cg.recitation_stats(description, cats)
            return (f"vocabulary recitation ({len(cats)} categories, "
                    f"{s['unsupported_obj']} unsupported)")
        if not g.kept:
            return "no grounded categories"
        return None

    def undecided(cats):
        # Two or more sides of a group emitted and the text cues none: the
        # one case where the description cannot pick, so resample once.
        reasons = {r for r in cg.ground(description, cats).dropped.values()
                   if _CATEGORY_UNDECIDED in r}
        return "; ".join(sorted(reasons)) or None

    out = _text_pass_answer("category-content", model,
                            _build_category_prompt(description, CONTENT_VOCABULARY),
                            parse, problem_of, soft_problem_of=undecided)
    if not out:
        return out
    g = cg.ground(description, out)
    if report is not None:
        report.update(raw=len(out), dropped=dict(g.dropped))
    return g.kept


# `category_grounding.group_drops` words the reason for a group whose emitted
# sides are ALL uncued as "<group>: <sides> all emitted, none cued".
# tests/test_text_pass_safeguards.py pins the retry on it, so a rewording
# there fails a test instead of silently dropping the retry.
_CATEGORY_UNDECIDED = "none cued"


def extract_keywords_from_description(
    description: Optional[str],
    model: str = KEYWORDS_MODEL,
) -> Optional[list[str]]:
    """Extract 5-15 free-form lowercased keywords from a description.

    Returns a list on success, or None if the call timed out / errored, so the
    caller can defer the photo for retry.

    Raises UnusableAnswer when the model answered but, twice, the answer could
    not be stored: cut off, empty, a refusal ("i couldn't find any text…"),
    a whole list as one keyword, implausibly many, or keywords that are not
    about the description (fewer than KEYWORD_MATCH_MIN of them use its
    words). Each of those used to be stored as the photo's keywords.
    """
    if not description or not description.strip():
        return []
    if not HAS_OLLAMA:
        return []
    from .bakeoff import build_keyword_prompt, parse_keywords_response
    from .stale_descriptions import (KEYWORD_MATCH_MIN, _REFUSAL,
                                     keyword_match_ratio)

    def problem_of(raw, kws):
        if not kws:
            return "no keywords"
        if any(_REFUSAL.search(k) for k in kws):
            return "refusal stored as keywords"
        if len(kws) == 1 and len(kws[0].split()) > _KEYWORD_MAX_WORDS:
            return "whole list as one keyword"
        if len(kws) > _KEYWORDS_MAX_PLAUSIBLE:
            return f"implausibly many keywords ({len(kws)})"
        ratio = keyword_match_ratio(description, kws)
        if ratio is not None and ratio < KEYWORD_MATCH_MIN:
            return f"keywords not in the description ({ratio:.0%} match)"
        return None

    return _text_pass_answer("keywords", model, build_keyword_prompt(description),
                             parse_keywords_response, problem_of)


def _build_visual_prompt(vocab: list[str]) -> str:
    """Build the category-visual prompt from the PERCEIVED vocabulary.

    TWO failures shaped this, both measured on live output. DON'T UNDO EITHER
    without re-running the A/B.

    1. A flat checklist whose only rule was "Include every tag that clearly
       applies" collapsed the output: 1,373 photos of one shoot produced 163
       distinct tag sets, one verbatim 8-tag set on 101 of them.
    2. Its replacement grouped the terms into AXES and said "Pick at most ONE
       tag from each group". The model read that as pick one FROM EACH group,
       so every photo collected a viewpoint tag and a composition tag whether
       or not one applied — `close-up, sunny, symmetrical` on a person walking
       down stairs, `close-up ... wide-angle` on an artichoke.

    So there are NO axes to fill: the over-applied terms sit in a RARE section
    that spells out the exact situation each one needs, including what it is
    NOT. And there are NO worked examples — a variant carrying two of them
    leaked `close-up` out of the example onto an unrelated baseball photo.

    Measured on 11 hand-labelled photos (qwen2.5-vl-7b-instruct, temperature 0):
    axes prompt 16 right / 5 wrong / 3.3 tags per photo; this one 14 right /
    0 wrong / 1.8 tags; this one plus two examples 14 right / 1 wrong.

    3. (2026-10-09) minicpm-v-4_5, the fleet's visual model, settled on
       `sunny, colorful, vibrant` for whole shoots, indoor gym futsal
       included. So `sunny`/`overcast` now say "never indoors", the LIGHT
       header rules weather out under lamps or flash, and colorful / vibrant
       / muted got definitions. 12 gym frames: 12 `sunny` -> 0. 60 owner
       labels: 82 -> 58 false positives, precision 0.54 -> 0.63, recall
       unchanged at 0.56 (`evals/prompts/visual_attractor_d.txt` is this
       prompt). A blanket "most photos have NO colour tag" rule cut recall
       to 0.49; defining harsh-light as "...is just sunny" brought indoor
       `sunny` back.

    The capture-fact and frozen terms are deliberately absent — see
    photosearch/visual_tags_derive.py.
    """
    from .visual_tags_derive import (PERCEIVED_GLOSS, PROMPT_SECTIONS,
                                     _INLINE_SECTIONS)

    offered = set(vocab)

    def _render(terms):
        return ", ".join(
            f"{t} ({PERCEIVED_GLOSS[t]})" if t in PERCEIVED_GLOSS else t
            for t in terms if t in offered
        )

    lines = []
    listed = set()
    for header, groups in PROMPT_SECTIONS:
        rendered = [_render(g) for g in groups]
        rendered = [r for r in rendered if r]
        if not rendered:
            continue
        listed.update(t for g in groups for t in g)
        if header in _INLINE_SECTIONS and len(rendered) == 1:
            lines.append(f"{header} {rendered[0]}")
        else:
            lines.append(header)
            lines.extend(f"  {r}" for r in rendered)
    # A term the caller offered that no section claims (a regenerated
    # vocabulary with a new word) still has to be reachable.
    extra = _render([t for t in vocab if t not in listed])
    if extra:
        lines.append("OTHER:")
        lines.append(f"  {extra}")

    return (
        "Tag how this photo LOOKS, using only tags from the list below.\n\n"
        "Most photos deserve 1 to 3 tags, and many deserve only one. Do NOT "
        "try to use every section: most photos have NO viewpoint tag and NO "
        "composition tag at all.\n\n"
        + "\n".join(lines)
        + "\n\nAnswer with ONLY a comma-separated list of tags spelled exactly "
        f"as above, at most {_VISUAL_MAX_TAGS}. If nothing clearly applies, "
        "answer: none\n"
    )


# A leading "Tags:" / "RIGHT:" / "Answer:" style label on the answer line.
_VISUAL_LABEL_RE = re.compile(r"^\w[\w \-]*:\s*")
# A bullet or list-number at the start of a line.
_VISUAL_BULLET_RE = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s+")
# Everything a model wraps a token in: quotes, brackets, backticks, emphasis.
_VISUAL_WRAPPERS = "\"'`*_[]{}()<> \t.,;:"
# Answers that genuinely mean "no tags". Anything else that parses to nothing
# is an unparseable FAILURE, not an empty result — see `tag_visual_photo`.
_VISUAL_EMPTY_ANSWERS = {
    "", "none", "no tags", "no tag", "n/a", "na", "nothing", "-", "[]",
    "none of the above", "no visual tags",
}


# Off-vocabulary words a model uses for an on-vocabulary tag. Applied only when
# the target is in the vocabulary being parsed. gemma-4 answered `misty` on a
# photographer-keyworded fog shot and the tag was silently dropped (Unsplash
# 26-lAP0XprM, 2026-09-26). Keep this to true synonyms: a near-miss mapping
# (say `cloudy` -> `overcast`) would manufacture a tag the model did not mean.
_VISUAL_SYNONYMS = {
    "misty": "foggy",
}


def _is_explicit_empty_answer(raw: Optional[str]) -> bool:
    """True when the model said "no tags" rather than saying nothing usable."""
    s = (raw or "").strip().strip("`")
    s = _VISUAL_BULLET_RE.sub("", s)
    s = _VISUAL_LABEL_RE.sub("", s)
    s = s.strip(_VISUAL_WRAPPERS).lower()
    return s in _VISUAL_EMPTY_ANSWERS


def _parse_visual_response(raw: str, vocab_set: set[str]) -> list[str]:
    """Read a tag list out of whatever shape the model actually emitted.

    The first version split on commas and nothing else, so `RIGHT: sunny` ->
    `[]`, `Tags: sunny, peaceful` lost `sunny`, and a bullet list parsed to
    nothing at all. An empty parse then looked exactly like "this photo has no
    visual qualities" and was persisted as `'[]'` — NOT NULL, so the photo left
    the queue permanently, un-retried, looking done.

    DON'T SIMPLIFY THIS BACK to a comma split. The prompt asks for one shape;
    small vision models supply six, and the cost of not reading one is a
    silently mistagged photo that never comes back.
    """
    text = (raw or "").strip()
    if not text:
        return []
    # Strip a ``` fence, with or without a language tag.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    # A JSON array is a common answer shape and the cleanest to read.
    if text.startswith("["):
        try:
            parsed = json.loads(text.replace("'", '"'))
        except ValueError:
            parsed = None
        if isinstance(parsed, list):
            text = ", ".join(str(x) for x in parsed)

    out: list[str] = []
    seen: set[str] = set()
    for line in re.split(r"[\r\n]+", text):
        line = _VISUAL_BULLET_RE.sub("", line)
        line = _VISUAL_LABEL_RE.sub("", line)
        for token in re.split(r"[,;]", line):
            t = token.strip().strip(_VISUAL_WRAPPERS).lower()
            if t not in vocab_set and _VISUAL_SYNONYMS.get(t) in vocab_set:
                t = _VISUAL_SYNONYMS[t]
            if t in vocab_set and t not in seen:
                seen.add(t)
                out.append(t)
    return out


_OVER_SELECTION = "over-selection"
_CONTRADICTION = "contradiction"
_UNPARSEABLE = "unparseable"


def _visual_contradictions(tags) -> list[tuple[str, str]]:
    """The mutually-exclusive pairs both present in `tags`."""
    from .visual_tags_derive import CONTRADICTORY_PAIRS

    have = set(tags)
    return [p for p in CONTRADICTORY_PAIRS if p[0] in have and p[1] in have]


def _visual_answer_problem(tags) -> Optional[str]:
    """Why this answer is not believable, or None if it is fine."""
    if len(tags) >= _VISUAL_MAX_PLAUSIBLE_TAGS:
        return _OVER_SELECTION
    if _visual_contradictions(tags):
        return _CONTRADICTION
    return None


def _drop_visual_contradictions(tags) -> list[str]:
    """Remove BOTH members of every contradictory pair, keeping the rest.

    Both go, not the "better" one: when a model says a photo is at once
    peaceful and dramatic, nothing in the answer tells you which half it meant,
    so neither is evidence. The tags that were never in question are still
    good, which is why this repairs rather than rejecting the whole response.
    """
    doomed: set[str] = set()
    for a, b in _visual_contradictions(tags):
        doomed.add(a)
        doomed.add(b)
    return [t for t in tags if t not in doomed]


def tag_visual_photo(
    image_path: str,
    model: str = TAGS_MODEL,
    raise_unusable: bool = False,
) -> Optional[list[str]]:
    """Generate PERCEIVED visual tags for a single photo via Ollama (vision).

    Calls `_ollama_chat_with_retry` directly (not via describe_photo) so the
    test surface is uniform — same mock point as extract_categories/keywords.

    Two guards on the answer, both resolved with at most one retry:

      * OVER-SELECTION (>= `_VISUAL_MAX_PLAUSIBLE_TAGS`) — a retry that still
        over-selects is the regurgitation signature (the model is reciting the
        vocabulary, not looking at the photo) and the whole response is
        dropped. Unchanged in shape from the old guard; only the threshold
        moved, from 12 — which the real failure never reached, its maximum
        being 11 — down to the cap the prompt now states.
      * CONTRADICTION — both halves of a mutually exclusive pair. A retry that
        is still contradictory gets REPAIRED, not rejected: both members go and
        the untouched tags survive.
      * UNPARSEABLE — a non-empty response that yields no tag at all. Retried
        once, then treated as a FAILED GENERATION.

    Two different empties, and conflating them is what made the original bug
    permanent:

      ``[]``    the model answered, and the answer was "no tags". A real
                result: the server writes '[]' and the photo is done.
      ``None``  no usable answer (no response, unparseable prose,
                regurgitation, or a guard that removed everything). The server
                leaves `visual_tags` NULL — so the photo stays claimable — but
                still marks it processed, so a repeatable failure is bounded by
                MAX_PROCESS_ATTEMPTS instead of being re-claimed forever. Same
                shape as the aesthetics pass's empty-scores row.

    With `raise_unusable=True` (the worker) a no-usable-answer case raises
    UnusableAnswer instead of returning None, so it is reported as a failure
    row: the same one attempt, plus an `index_errors` entry. 307 photos were
    retired `blocked` on 2026-10-03 with no error recorded anywhere.
    Transport failures still return None either way.
    """
    from .visual_tags_derive import PERCEIVED_VOCABULARY
    if not HAS_OLLAMA:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    # PERCEIVED only — the capture facts are derived from EXIF server-side and
    # would be stripped from this answer anyway. Asking for them just wastes
    # tokens and invites the model to fill its quota with guesses.
    vocab_set = set(PERCEIVED_VOCABULARY)
    prompt = _build_visual_prompt(PERCEIVED_VOCABULARY)
    encoded = _encode_image_for_ollama(str(path))
    image_ref = encoded if encoded is not None else str(path)
    options = _options_for_model(model)

    def _no_answer(reason):
        if raise_unusable:
            raise UnusableAnswer(f"category-visual: {reason}")
        return None

    try:
        raw = _ollama_chat_with_retry(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [image_ref]}],
            options=options,
            role="visual",
        )
    except TruncatedOutput as e:
        return _no_answer(f"cut off ({e})")
    except Exception:
        return None
    if not raw:
        return _no_answer("empty response")   # no response at all — a failure
    if _is_explicit_empty_answer(raw):
        return []                         # the model answered: "no tags"
    tags = _parse_visual_response(raw, vocab_set)
    problem = _UNPARSEABLE if not tags else _visual_answer_problem(tags)
    if problem is not None:
        # One retry with a temperature bump — same shape as the old guard.
        retry_opts = dict(options)
        retry_opts["temperature"] = 0.4
        retry_opts.setdefault("repeat_penalty", 1.3)
        try:
            raw2 = _ollama_chat_with_retry(
                model=model,
                messages=[{"role": "user", "content": prompt, "images": [image_ref]}],
                options=retry_opts,
                role="visual",
            )
        except Exception:
            raw2 = None
        if raw2 and _is_explicit_empty_answer(raw2):
            return []                     # the retry answered: "no tags"
        tags2 = _parse_visual_response(raw2, vocab_set) if raw2 else []
        if tags2:
            problem2 = _visual_answer_problem(tags2)
        else:
            problem2 = _UNPARSEABLE if raw2 else None

        if tags2 and problem2 is None:
            tags = tags2                  # the retry produced a clean answer
        elif problem2 == _CONTRADICTION:
            # Repair rather than reject: both members of each bad pair go, the
            # tags that were never in question survive.
            tags = _drop_visual_contradictions(tags2)
        elif problem2 is None and not tags2 and problem == _CONTRADICTION:
            # The retry produced nothing at all — repair the first answer.
            tags = _drop_visual_contradictions(tags)
        else:
            # Regurgitation, or still unparseable: no usable answer. None (not
            # []) so the column stays NULL and the photo is re-claimed, bounded
            # by MAX_PROCESS_ATTEMPTS.
            return _no_answer(problem2 or problem)
    return tags if tags else _no_answer("contradictions left no tags")


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
