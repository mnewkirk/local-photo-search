"""Subject-layer scoring (docs/plans/video-support.md step 3b): primary choice,
label linking, face join, S1/S2 credit, whole-clip denominator."""

import dataclasses
import json

import numpy as np
import pytest

from photosearch import video_analysis as va
from photosearch import video_subjects as vs

W, H = 1920, 1080


def _emb(person, frame):
    v = np.random.default_rng(person).normal(size=512)
    v /= np.linalg.norm(v)
    n = np.random.default_rng(person * 1000 + frame).normal(size=512)
    v = v + 0.1 * n / np.linalg.norm(n)
    return (v / np.linalg.norm(v)).tolist()


def _face_frames(duration, present, cx=960, cy=400, size=160, person=1):
    """present(t) -> bool; one face at (cx, cy)."""
    out = []
    for i, t in enumerate(va.sample_times(duration, 4.0)):
        faces = []
        if present(t):
            faces.append({"bbox": (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2),
                          "det_score": 0.9, "embedding": _emb(person, i)})
        out.append({"t": t, "faces": faces})
    return out


DOG = {"label": "dog", "category": "animal", "bbox": [0.3, 0.3, 0.7, 0.9]}
# person box around the face at (960, 400) ± 80 px → x 0.4-0.6, y 0.2-1.0
PERSON = {"label": "boy", "category": "person", "bbox": [0.4, 0.2, 0.6, 1.0]}


def test_normalize_label():
    for s in ("dog", "the dog", "The Dog", "our dog", "puppy", "our dogs", "a puppy",
              "puppies", "the dog's", "Dog."):
        assert vs.normalize_label(s) == "dog", s
    assert vs.normalize_label("mountain lake") == "mountain lake"
    assert vs.normalize_label("the mountains") == "mountain"
    assert vs.normalize_label("grass") == "grass"
    assert vs.normalize_label("kids") == "person"
    assert vs.normalize_label(None) == ""


def test_dog_the_dog_puppy_link_to_one_subject():
    segs = va.segment_schedule(12)
    g = [[dict(DOG, label="dog")], [dict(DOG, label="the dog")],
         [dict(DOG, label="puppy")], [dict(DOG, label="our dogs")]]
    sa = vs.score_subjects(segs, g, W=W, H=H)
    assert len(sa.subjects) == 1
    s = sa.subjects[0]
    assert s.label == "dog" and s.category == "animal"
    assert s.share == pytest.approx(1.0) and s.focus_s == pytest.approx(12.0)
    assert s.n_segments == 4 and s.focus_intervals == [[0.0, 12.0]]


def test_empty_segments_count_in_the_denominator():
    segs = va.segment_schedule(12)
    sa = vs.score_subjects(segs, [[DOG], [], [DOG], None], W=W, H=H)
    assert sa.duration == pytest.approx(12.0)
    assert sa.subjects[0].share == pytest.approx(0.5)
    assert sa.primaries == [0, None, 0, None]
    assert sa.n_empty == 1 and sa.n_ungrounded == 1
    assert sa.summary_text() == "0–3 s dog, 3–6 s no subject, 6–9 s dog, 9–12 s no subject"
    json.dumps(sa.to_json())


def test_primary_is_highest_salience_box():
    small_centre = {"label": "ball", "category": "object", "bbox": [0.47, 0.47, 0.53, 0.53]}
    big_edge = {"label": "tree", "category": "landscape", "bbox": [0.0, 0.0, 0.2, 1.0]}
    big_centre = {"label": "dog", "category": "animal", "bbox": [0.3, 0.2, 0.7, 0.8]}
    p = vs.choose_primary([small_centre, big_edge, big_centre], W, H)
    assert p["label"] == "dog"
    assert vs.choose_primary([], W, H) is None


def test_person_box_without_a_face_stays_unidentified():
    segs = va.segment_schedule(6)
    fa = va.analyze_faces(_face_frames(6, lambda t: False), W, H, 6)
    sa = vs.score_subjects(segs, [[PERSON], [PERSON]], faces=fa)
    assert len(sa.subjects) == 1
    s = sa.subjects[0]
    assert s.face_ix is None and s.category == "person" and s.label == "person"
    # a face OUTSIDE the person box doesn't identify it either
    fa2 = va.analyze_faces(_face_frames(6, lambda t: True, cx=200), W, H, 6)
    sa2 = vs.score_subjects(segs, [[PERSON], [PERSON]], faces=fa2)
    assert sa2.subjects[0].face_ix is None


def test_person_box_takes_the_face_identity():
    segs = va.segment_schedule(6)
    fa = va.analyze_faces(_face_frames(6, lambda t: True), W, H, 6)
    sa = vs.score_subjects(segs, [[PERSON], [dict(PERSON, label="a child")]], faces=fa)
    assert len(sa.subjects) == 1
    assert sa.subjects[0].face_ix == 0
    assert sa.summary_text({0: "Calvin"}) == "0–6 s Calvin (face_ix 0)"


def test_face_join_through_coasting_window():
    # keyframe at 1.5 s; the face is missing 1.1-1.7 s (a 1.0 s coasted gap)
    segs = va.segment_schedule(3)
    fa = va.analyze_faces(_face_frames(3, lambda t: not (1.1 < t < 1.7)), W, H, 3)
    assert len(fa.identities) == 1
    assert fa.identities[0].presence_share == pytest.approx(1.0)   # coasted
    sa = vs.score_subjects(segs, [[PERSON]], faces=fa)
    assert sa.subjects[0].face_ix == 0
    strict = dataclasses.replace(vs.DEFAULT_SUBJECT_PARAMS, join_window_s=0.0)
    sa0 = vs.score_subjects(segs, [[PERSON]], faces=fa, params=strict)
    assert sa0.subjects[0].face_ix is None


def test_s1_whole_segment_vs_s2_tracked_portion():
    # the person's face is present only for the first half of each 3 s segment
    segs = va.segment_schedule(6)
    fa = va.analyze_faces(_face_frames(6, lambda t: (t % 3) < 1.5), W, H, 6)
    kts = [0.625, 3.625]   # keyframes inside the tracked halves
    s1 = vs.score_subjects(segs, [[PERSON], [PERSON]], faces=fa, keyframe_times=kts)
    s2 = vs.score_subjects(segs, [[PERSON], [PERSON]], faces=fa, keyframe_times=kts,
                           params=dataclasses.replace(vs.DEFAULT_SUBJECT_PARAMS, credit="S2"))
    assert s1.subjects[0].face_ix is not None
    assert s1.subjects[0].share == pytest.approx(1.0)
    assert s2.subjects[0].share == pytest.approx(0.5)
    assert s2.subjects[0].focus_intervals == [[0.0, 1.5], [3.0, 4.5]]
    # non-person subjects are credited the whole segment under S2 too
    d2 = vs.score_subjects(segs, [[DOG], []], faces=fa,
                           params=dataclasses.replace(vs.DEFAULT_SUBJECT_PARAMS, credit="S2"))
    assert d2.subjects[0].share == pytest.approx(0.5)


def test_similarity_callback_links_different_labels():
    segs = va.segment_schedule(9)
    g = [[dict(DOG, label="dog")], [dict(DOG, label="golden retriever")],
         [{"label": "mountain lake", "category": "landscape", "bbox": [0, 0, 1, 1]}]]
    assert len(vs.score_subjects(segs, g, W=W, H=H).subjects) == 3
    sim = lambda a, b: 0.95 if {a["norm"], b["norm"]} == {"dog", "golden retriever"} else 0.1
    sa = vs.score_subjects(segs, g, W=W, H=H, similarity=sim)
    assert len(sa.subjects) == 2
    assert sa.subjects[0].share == pytest.approx(2 / 3, abs=1e-3)


def test_forty_segment_cap_on_a_29_minute_clip():
    segs = va.segment_schedule(29 * 60)
    assert len(segs) == 40
    g = [[DOG] if i < 10 else [] for i in range(40)]
    sa = vs.score_subjects(segs, g, W=W, H=H)
    assert sa.subjects[0].share == pytest.approx(0.25)
    assert sa.subjects[0].focus_s == pytest.approx(435.0)
    with pytest.raises(ValueError):
        vs.score_subjects(segs, g[:-1], W=W, H=H)
