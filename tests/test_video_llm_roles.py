"""Video LLM layer: `subject` / `video` roles, describe_video, category
grounding (docs/plans/video-support.md steps 3b and 5).

The LLM transport is mocked the way tests/test_text_pass_safeguards.py does
it: urlopen for the OpenAI route, `ollama.chat` for Ollama, or
`_ollama_chat_with_retry` itself for the guard logic.
"""
import io
import json
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest
from PIL import Image

from photosearch import describe as d
from photosearch import subjects as s

_ROLE_VARS = ("PHOTOSEARCH_LLM_SUBJECT_MODEL", "PHOTOSEARCH_LLM_VIDEO_MODEL",
              "PHOTOSEARCH_LLM_AESTHETICS_MODEL", "PHOTOSEARCH_LLM_VISUAL_MODEL",
              "PHOTOSEARCH_TEXT_LLM_MODEL", "PHOTOSEARCH_TEXT_LLM_URL",
              "PHOTOSEARCH_LLM_REASONING_EFFORT")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ROLE_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(d, "_RETRY_DELAY", 0)


class _Resp:
    def __init__(self, payload): self.payload = payload
    def read(self): return json.dumps(self.payload).encode()


def _openai(monkeypatch, answers):
    """Stub urlopen with (content, finish_reason) answers or exceptions."""
    bodies = []
    q = list(answers)

    def fake_urlopen(req, timeout=None):
        bodies.append(json.loads(req.data))
        a = q.pop(0)
        if isinstance(a, BaseException):
            raise a
        content, finish = a
        return _Resp({"choices": [{"message": {"content": content},
                                   "finish_reason": finish}], "usage": {}})
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    return bodies


def _jpeg_bytes(w=64, h=48, color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Role resolution
# ---------------------------------------------------------------------------

def test_subject_role_uses_its_own_env(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_LLM_SUBJECT_MODEL", "subj-m")
    monkeypatch.setenv("PHOTOSEARCH_LLM_AESTHETICS_MODEL", "aes-m")
    assert d._resolve_openai_model("x", "subject") == "subj-m"


def test_subject_role_falls_back_to_the_photo_grounding_model(monkeypatch):
    # Photos ground under role `aesthetics`; an unset SUBJECT var must give
    # the same model photos use.
    monkeypatch.setenv("PHOTOSEARCH_LLM_AESTHETICS_MODEL", "aes-m")
    monkeypatch.setenv("PHOTOSEARCH_LLM_VISUAL_MODEL", "vis-m")
    assert d._resolve_openai_model("x", "subject") == \
        d._resolve_openai_model("x", "aesthetics") == "aes-m"
    monkeypatch.delenv("PHOTOSEARCH_LLM_AESTHETICS_MODEL")
    assert d._resolve_openai_model("x", "subject") == \
        d._resolve_openai_model("x", "aesthetics") == "vis-m"


def test_video_role_resolves_its_env(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_LLM_VIDEO_MODEL", "vid-m")
    assert d._resolve_openai_model("x", "video") == "vid-m"


def test_video_role_never_falls_back(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_LLM_VISUAL_MODEL", "vis-m")
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_MODEL", "legacy-m")
    with pytest.raises(d.RoleModelNotConfigured, match="PHOTOSEARCH_LLM_VIDEO_MODEL"):
        d._resolve_openai_model("x", "video")


def test_require_role_model(monkeypatch):
    # Ollama route: the passed name is what runs; nothing to check.
    assert d.require_role_model("video", "llava") == "llava"
    assert d.require_role_model("video") is None
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    with pytest.raises(d.RoleModelNotConfigured):
        d.require_role_model("video")
    monkeypatch.setenv("PHOTOSEARCH_LLM_VIDEO_MODEL", "vid-m")
    assert d.require_role_model("video") == "vid-m"
    monkeypatch.setenv("PHOTOSEARCH_LLM_AESTHETICS_MODEL", "aes-m")
    assert d.require_role_model("subject") == "aes-m"


def test_effective_model_for_the_video_roles(monkeypatch):
    assert d.effective_model("llava", "video") == "llava"      # Ollama route
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    monkeypatch.setenv("PHOTOSEARCH_LLM_VIDEO_MODEL", "vid-m")
    assert d.effective_model("llava", "video") == "vid-m"


def test_existing_roles_are_unchanged(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_LLM_VISUAL_MODEL", "vis-m")
    for role in ("describe", "verify", "visual", "aesthetics"):
        assert d._resolve_openai_model("x", role) == "vis-m"
    assert d._resolve_openai_model("x", "text") == "x"


def test_video_pass_roles_are_wired_into_pass_roles():
    # video-subjects (step 3b) and video-describe (step 5) are both wired.
    assert d.VIDEO_PASS_ROLES == {"video-subjects": "subject",
                                  "video-describe": "video"}
    assert d.PASS_ROLES["video-subjects"] == "subject"
    assert d.PASS_ROLES["video-describe"] == "video"


@pytest.mark.parametrize("role,budget", [("subject", 256), ("video", 768)])
def test_video_roles_send_their_token_budget(monkeypatch, role, budget):
    monkeypatch.setenv(f"PHOTOSEARCH_LLM_{role.upper()}_MODEL", "m")
    bodies = _openai(monkeypatch, [("ok", "stop")])
    d._ollama_chat_with_retry(model="m", messages=[{"role": "user", "content": "hi"}],
                              timeout=5, role=role)
    assert bodies[0]["max_tokens"] == budget


# ---------------------------------------------------------------------------
# Context overflow
# ---------------------------------------------------------------------------

def _http_400(body):
    return urllib.error.HTTPError("http://x/v1/chat/completions", 400,
                                  "Bad Request", {}, io.BytesIO(body.encode()))


def test_a_context_overflow_400_is_its_own_exception(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_LLM_VIDEO_MODEL", "vid-m")
    bodies = _openai(monkeypatch, [_http_400('{"error":"Context size has been exceeded."}')])
    with pytest.raises(d.ContextOverflow) as ei:
        d._ollama_chat_with_retry(model="m", messages=[{"role": "user", "content": "x"}],
                                  timeout=5, role="video")
    assert len(bodies) == 1                       # not retried as transient
    assert "Context size" in ei.value.detail
    assert not isinstance(ei.value, (d.UnusableAnswer, TimeoutError, ConnectionError))
    assert not d._is_timeout(ei.value)


def test_an_unrelated_400_is_not_an_overflow(monkeypatch):
    _openai(monkeypatch, [_http_400('{"error":"Invalid model identifier"}')])
    with pytest.raises(urllib.error.HTTPError):
        d._ollama_chat_with_retry(model="m", messages=[{"role": "user", "content": "x"}],
                                  timeout=5, role="text")


def test_ollama_context_error_is_an_overflow(monkeypatch):
    def chat(**kw):
        raise RuntimeError("input length exceeds context length (status code: 400)")
    monkeypatch.setattr(d, "ollama", SimpleNamespace(chat=chat))
    with pytest.raises(d.ContextOverflow):
        d._ollama_chat_with_retry(model="m", messages=[{"role": "user", "content": "x"}],
                                  timeout=5, role="video")


# ---------------------------------------------------------------------------
# describe_video
# ---------------------------------------------------------------------------

GOOD = ("A brown dog runs across a grassy park toward the camera. It picks up "
        "a red ball and carries it back to a woman standing by a bench.")
CUT = "A brown dog runs across a grassy park toward the camera and then"


def _script(monkeypatch, answers):
    """Script `_ollama_chat_with_retry`; record each call's kwargs."""
    calls = []
    q = list(answers)

    def chat(**kw):
        calls.append(kw)
        a = q.pop(0)
        if isinstance(a, BaseException):
            raise a
        return a
    monkeypatch.setattr(d, "HAS_OLLAMA", True)
    monkeypatch.setattr(d, "_ollama_chat_with_retry", chat)
    return calls


FRAMES = [b"\xff\xd8one", b"\xff\xd8two", b"\xff\xd8three"]
TS = [0.0, 12.5, 40.0]


def test_describe_video_sends_k_images_and_the_timeline(monkeypatch):
    calls = _script(monkeypatch, [GOOD])
    out = d.describe_video(FRAMES, TS, 42.0, model="m",
                           subject_summary="0-12 s dog, 12-40 s Calvin")
    assert out == GOOD
    (call,) = calls
    assert call["role"] == "video"
    msg = call["messages"][0]
    assert len(msg["images"]) == 3
    p = msg["content"]
    assert "These 3 images" in p and "42.0 s" in p
    for i, t in enumerate(TS, 1):
        assert f"frame {i} at {t:.1f} s" in p
    assert "0-12 s dog, 12-40 s Calvin" in p
    assert "HAPPENS" in p


def test_describe_video_openai_message_has_k_image_parts(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_LLM_VIDEO_MODEL", "vid-m")
    bodies = _openai(monkeypatch, [(GOOD, "stop")])
    assert d.describe_video(FRAMES, TS, 42.0) == GOOD
    body = bodies[0]
    assert body["model"] == "vid-m" and body["max_tokens"] == 768
    parts = body["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text"] + ["image_url"] * 3
    assert "frame 3 at 40.0 s" in parts[0]["text"]


def test_describe_video_without_a_video_model_raises(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_LLM_VISUAL_MODEL", "vis-m")
    _openai(monkeypatch, [])
    with pytest.raises(d.RoleModelNotConfigured):
        d.describe_video(FRAMES, TS, 42.0)


def test_describe_video_prompt_is_swappable_and_has_no_subjects_by_default(monkeypatch):
    calls = _script(monkeypatch, [GOOD, GOOD])
    d.describe_video(FRAMES, TS, 42.0)
    assert d.VIDEO_SUBJECTS_PREFIX not in calls[0]["messages"][0]["content"]
    d.describe_video(FRAMES, TS, 725.0, prompt='{n_frames} frames, {duration}; {"x": 1}')
    assert calls[1]["messages"][0]["content"] == '3 frames, 12:05 (725.0 s); {"x": 1}'


def test_describe_video_frame_paths_are_encoded(monkeypatch, tmp_path):
    p = tmp_path / "f.jpg"
    p.write_bytes(_jpeg_bytes())
    calls = _script(monkeypatch, [GOOD])
    d.describe_video([str(p)], [1.0], 2.0)
    img = calls[0]["messages"][0]["images"][0]
    assert isinstance(img, str) and len(img) > 20


def test_describe_video_mismatched_inputs():
    with pytest.raises(ValueError):
        d.describe_video([b"a"], [1.0, 2.0], 3.0)
    with pytest.raises(ValueError):
        d.describe_video([], [], 3.0)


def test_describe_video_mid_sentence_retries_once_at_0_4(monkeypatch):
    calls = _script(monkeypatch, [CUT, GOOD])
    assert d.describe_video(FRAMES, TS, 42.0) == GOOD
    assert [c["options"]["temperature"] for c in calls] == [0.0, 0.4]


def test_describe_video_unusable_twice_raises_for_the_worker(monkeypatch):
    calls = _script(monkeypatch, [CUT, CUT])
    with pytest.raises(d.UnusableAnswer, match="mid-sentence"):
        d.describe_video(FRAMES, TS, 42.0, raise_unusable=True)
    assert len(calls) == 2
    _script(monkeypatch, [CUT, CUT])
    assert d.describe_video(FRAMES, TS, 42.0) is None


def test_describe_video_truncated_twice_raises_truncated(monkeypatch):
    _script(monkeypatch, [d.TruncatedOutput("cut", "A dog"),
                          d.TruncatedOutput("cut", "A dog on")])
    with pytest.raises(d.TruncatedOutput):
        d.describe_video(FRAMES, TS, 42.0, raise_unusable=True)


def test_describe_video_truncation_end_to_end_doubles_the_budget(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_LLM_VIDEO_MODEL", "vid-m")
    bodies = _openai(monkeypatch, [("A dog", "length"), ("A dog on", "length"),
                                   ("A dog", "length"), ("A dog on", "length")])
    with pytest.raises(d.TruncatedOutput):
        d.describe_video(FRAMES, TS, 42.0, raise_unusable=True)
    assert [b["max_tokens"] for b in bodies] == [768, 1536, 768, 1536]


def test_describe_video_repetition_loop_is_unusable(monkeypatch):
    loop = "The dog runs and the dog runs and " * 4 + "the dog runs."
    _script(monkeypatch, [loop, loop])
    with pytest.raises(d.UnusableAnswer, match="repetition"):
        d.describe_video(FRAMES, TS, 42.0, raise_unusable=True)


def test_describe_video_context_overflow_propagates(monkeypatch):
    _script(monkeypatch, [d.ContextOverflow("too big", "Context size has been exceeded")])
    with pytest.raises(d.ContextOverflow):
        d.describe_video(FRAMES, TS, 42.0, raise_unusable=True)


def test_describe_video_transport_error_is_none(monkeypatch):
    _script(monkeypatch, [ConnectionError("refused")])
    assert d.describe_video(FRAMES, TS, 42.0, raise_unusable=True) is None


def test_describe_video_empty_answer_is_none_even_for_the_worker(monkeypatch):
    _script(monkeypatch, [None])
    assert d.describe_video(FRAMES, TS, 42.0, raise_unusable=True) is None


# ---------------------------------------------------------------------------
# Photo grounding is unchanged
# ---------------------------------------------------------------------------

_PINNED_PHOTO_PROMPT = (
    "Identify the main subject(s) of this photograph — the primary animal(s), "
    "person(s), or object(s) it is about. Ignore background scenery. The image "
    "is {w}x{h} pixels with the origin at the top-left. Respond with ONLY a "
    "minified JSON array, one object per distinct main subject, each "
    '{{"label":"<what it is>","bbox_2d":[x1,y1,x2,y2]}} using absolute pixel '
    "coordinates. If there is no clear subject (e.g. a pure landscape), respond "
    "with an empty array []."
)


def test_photo_grounding_prompt_is_byte_identical():
    assert s.GROUNDING_PROMPT == _PINNED_PHOTO_PROMPT


def test_photo_grounding_call_is_unchanged(monkeypatch, tmp_path):
    p = tmp_path / "p.jpg"
    Image.new("RGB", (200, 100)).save(p)
    calls = _script(monkeypatch, ['[{"label":"dog","bbox_2d":[10,10,60,60]}]'])
    out = s.ground_photo_subjects(str(p))
    (c,) = calls
    assert c["role"] == "aesthetics"
    assert c["model"] == s.AES_MODEL
    assert c["options"] == {"temperature": 0.0, "num_ctx": 8192, "num_predict": 400}
    assert c["messages"][0]["content"] == _PINNED_PHOTO_PROMPT.format(w=200, h=100)
    assert out == [{"label": "dog", "bbox": [0.05, 0.1, 0.3, 0.6], "area_frac": 0.125}]
    assert "category" not in out[0]


# ---------------------------------------------------------------------------
# Category grounding
# ---------------------------------------------------------------------------

W, H = 200, 100


@pytest.mark.parametrize("raw", [
    '[{"label":"dog","category":"animal","bbox_2d":[20,10,120,90]}]',
    '```json\n[{"label":"dog","category":"animal","bbox_2d":[20,10,120,90]}]\n```',
    'Here you go: [{"label":"dog","category":"animal","bbox_2d":[20,10,120,90]}] '
    'The dog is clearly the subject.',
    '{"label":"dog","category":"animal","bbox_2d":[20,10,120,90]}',
    '{"subjects":[{"label":"dog","category":"Animal","bbox":[20,10,120,90]}]}',
    '[{"label":"dog","category":"animal","bbox_2d":[0.1,0.1,0.6,0.9]}]',
    '[{"label":"dog","category":"animal","bbox_2d":[100,100,600,900]}]',
])
def test_category_parse_variants(raw):
    (b,) = s.parse_category_subjects(raw, W, H)
    assert b["label"] == "dog" and b["category"] == "animal"
    assert b["bbox"] == [0.1, 0.1, 0.6, 0.9]


def test_category_parse_empty_is_a_real_answer():
    assert s.parse_category_subjects("[]", W, H) == []
    assert s.parse_category_subjects("```json\n[]\n```", W, H) == []


@pytest.mark.parametrize("raw", [None, "", "I cannot tell.", "[{broken",
                                 '[{"label":"dog","category":"animal"}]'])
def test_category_parse_unusable_is_none(raw):
    assert s.parse_category_subjects(raw, W, H) is None


def test_landscape_gets_a_label_and_the_whole_frame():
    out = s.parse_category_subjects(
        '[{"label":"mountain lake","category":"landscape"}]', W, H)
    assert out == [{"label": "mountain lake", "category": "landscape",
                    "bbox": s.WHOLE_FRAME, "area_frac": 1.0}]
    out = s.parse_category_subjects(
        '[{"label":"mountain lake","category":"landscape","bbox_2d":[0,0,200,100]}]', W, H)
    assert out[0]["bbox"] == s.WHOLE_FRAME


def test_categories_are_normalised():
    raw = ('[{"label":"a small boy","bbox_2d":[0,0,50,50]},'
           '{"label":"kite","category":"thing","bbox_2d":[100,0,120,20]},'
           '{"label":"soccer","category":"sports","bbox_2d":[0,0,200,100]}]')
    by = {b["label"]: b["category"] for b in s.parse_category_subjects(raw, W, H)}
    assert by == {"a small boy": "person", "kite": "object", "soccer": "activity"}


def test_category_output_feeds_score_subjects():
    vs = pytest.importorskip("photosearch.video_subjects")
    segs = [(0.0, 3.0), (3.0, 6.0)]
    g = [s.parse_category_subjects(
            '[{"label":"mountain lake","category":"landscape"}]', W, H),
         s.parse_category_subjects('[]', W, H)]
    res = vs.score_subjects(segs, g, W=W, H=H)
    assert res.primaries[0] is not None and res.primaries[1] is None


def test_ground_frame_subjects_role_prompt_and_bytes(monkeypatch):
    calls = _script(monkeypatch, ['[{"label":"dog","category":"animal","bbox_2d":[20,10,120,90]}]'])
    out = s.ground_frame_subjects(_jpeg_bytes(W, H))
    (c,) = calls
    assert c["role"] == "subject"
    assert c["options"]["num_predict"] == 256
    content = c["messages"][0]["content"]
    assert "200x100" in content and '"category"' in content and "{{" not in content
    assert out[0]["category"] == "animal"


def test_ground_frame_subjects_retries_then_raises(monkeypatch):
    calls = _script(monkeypatch, ["no idea", "still no idea"])
    with pytest.raises(d.UnusableAnswer):
        s.ground_frame_subjects(_jpeg_bytes(W, H), raise_unusable=True)
    assert [c["options"]["temperature"] for c in calls] == [0.0, 0.4]
    _script(monkeypatch, ["no idea", "still no idea"])
    assert s.ground_frame_subjects(_jpeg_bytes(W, H)) is None


def test_ground_frame_subjects_recovers_on_retry(monkeypatch):
    _script(monkeypatch, ["no idea", "[]"])
    assert s.ground_frame_subjects(_jpeg_bytes(W, H), raise_unusable=True) == []


def test_ground_frame_subjects_transport_defers(monkeypatch):
    calls = _script(monkeypatch, [ConnectionError("refused")])
    assert s.ground_frame_subjects(_jpeg_bytes(W, H), raise_unusable=True) is None
    assert len(calls) == 1


def test_ground_frame_subjects_overflow_propagates(monkeypatch):
    _script(monkeypatch, [d.ContextOverflow("too big")])
    with pytest.raises(d.ContextOverflow):
        s.ground_frame_subjects(_jpeg_bytes(W, H))


def test_ground_frame_subjects_without_category_uses_the_photo_prompt(monkeypatch):
    calls = _script(monkeypatch, ['[{"label":"boy","bbox_2d":[0,0,50,50]}]'])
    out = s.ground_frame_subjects(_jpeg_bytes(W, H), with_category=False)
    assert calls[0]["messages"][0]["content"] == s.GROUNDING_PROMPT.format(w=W, h=H)
    assert out[0]["category"] == "person"
