"""Worked examples for the pure-numpy video core (docs/plans/video-support.md
step 3): schedules, salience, tracker, identity merge, focus, shares, quality."""

import dataclasses
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from photosearch import video_analysis as va

W, H = 1920, 1080
FPS = 4.0


def _emb(seed, noise=0.0, base_seed=None):
    rng = np.random.default_rng(seed if base_seed is None else base_seed)
    v = rng.normal(size=512)
    v /= np.linalg.norm(v)
    if noise:
        n = np.random.default_rng(seed + 10_000).normal(size=512)
        v = v + noise * n / np.linalg.norm(n)
        v /= np.linalg.norm(v)
    return v.tolist()


def _face(cx, cy, size, person, frame_ix=0, score=0.9, sharp=None, emb=True):
    d = {"bbox": (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2),
         "det_score": score,
         "embedding": _emb(frame_ix * 97 + hash(person) % 997, 0.1, base_seed=person) if emb else None}
    if sharp is not None:
        d["sharpness"] = sharp
    return d


def _frames(duration, faces_at):
    """faces_at(i, t) -> list of face dicts."""
    times = va.sample_times(duration, FPS)
    return [{"t": t, "faces": faces_at(i, t)} for i, t in enumerate(times)]


# ---------------------------------------------------------------- schedules


def test_sample_fps_by_duration():
    assert va.sample_fps(5) == 4.0
    assert va.sample_fps(120) == 3.0
    assert va.sample_fps(1740) == 2.0
    assert len(va.sample_times(10, 4)) == 40


def test_keyframe_count_schedule():
    assert va.keyframe_count(1.5) == 1
    assert va.keyframe_count(2.0) == 1
    assert va.keyframe_count(5) == 3
    assert va.keyframe_count(10) == 3
    assert 3 < va.keyframe_count(35) < 8
    assert va.keyframe_count(60) == 8
    assert va.keyframe_count(1740) == 8
    ks = [va.keyframe_count(d) for d in range(0, 120)]
    assert ks == sorted(ks)
    assert va.keyframe_centres(12, 3) == pytest.approx([2.0, 6.0, 10.0])


def test_keyframe_nudged_to_sharpest_within_half_second():
    times = va.sample_times(12, FPS)
    sharp = [1.0] * len(times)
    # centre 6.0: a sharper frame 0.375 s away wins; a sharper one 1 s away doesn't
    i_near = min(range(len(times)), key=lambda i: abs(times[i] - 6.375))
    i_far = min(range(len(times)), key=lambda i: abs(times[i] - 7.125))
    sharp[i_near] = 5.0
    sharp[i_far] = 50.0
    kt = va.keyframe_times(12, times, sharp, k=3)
    assert kt[1] == pytest.approx(times[i_near])
    # no sharpness → plain centres
    assert va.keyframe_times(12, k=3) == pytest.approx([2.0, 6.0, 10.0])


def test_segment_schedule_three_seconds_and_cap_40():
    segs = va.segment_schedule(12)
    assert segs == [(0, 3), (3, 6), (6, 9), (9, 12)]
    segs = va.segment_schedule(10)          # 1 s remainder folds into the last
    assert len(segs) == 3 and segs[-1] == (6, 10)
    segs = va.segment_schedule(10.6)        # 1.6 s remainder is its own segment
    assert len(segs) == 4
    long = va.segment_schedule(29 * 60)
    assert len(long) == 40
    assert long[0][1] - long[0][0] == pytest.approx(43.5)
    assert long[-1][1] == pytest.approx(1740)
    for d in (0.4, 3, 7.7, 119, 1740):
        s = va.segment_schedule(d)
        assert sum(b - a for a, b in s) == pytest.approx(d)
    assert len(va.coarsen_segments(long, 12)) == 12
    assert va.coarsen_segments(long, 12)[-1][1] == pytest.approx(1740)


# ---------------------------------------------------------------- salience


def test_salience_formula():
    A, C, S = va.face_salience((W / 2 - 54, H / 2 - 54, W / 2 + 54, H / 2 + 54), W, H)
    assert A == pytest.approx(108 / 1080)
    assert C == pytest.approx(1.0)
    # at the right edge (dx = W/2): r^2 = 1 → C = exp(-1/(2*0.35^2))
    _, C2, _ = va.face_salience((W - 54, H / 2 - 54, W + 54, H / 2 + 54), W, H)
    assert C2 == pytest.approx(np.exp(-1 / (2 * 0.35 ** 2)))
    # vertical offsets count half
    _, C3, _ = va.face_salience((W / 2 - 54, H - 54, W / 2 + 54, H + 54), W, H)
    assert C3 == pytest.approx(np.exp(-0.5 / (2 * 0.35 ** 2)))
    # sigma off
    assert va.face_salience((0, 0, 108, 108), W, H, sigma=None)[1] == 1.0


def test_presence_gates():
    p = va.DEFAULT_PARAMS
    assert va.presence_eligible({"bbox": (0, 0, 20, 20), "det_score": 0.6}, p)
    assert not va.presence_eligible({"bbox": (0, 0, 20, 20), "det_score": 0.4}, p)
    assert not va.presence_eligible({"bbox": (0, 0, 15, 40), "det_score": 0.9}, p)


# ---------------------------------------------------------------- shares


def test_one_dominant_face_share_is_one():
    fr = _frames(10, lambda i, t: [_face(W / 2, H / 2, 300, 1, i)])
    fa = va.analyze_faces(fr, W, H, 10)
    assert len(fa.identities) == 1
    ident = fa.identities[0]
    assert ident.focus_share == pytest.approx(1.0)
    assert ident.presence_share == pytest.approx(1.0)
    assert ident.focus_intervals == [[0.0, 10.0]]
    assert ident.track_count == 1
    assert len(ident.encoding) == 512


def test_two_equal_people_half_each():
    # both present throughout; A is the bigger face for the first half, B after
    def faces(i, t):
        big_a = t < 5
        return [_face(W * 0.4, H / 2, 300 if big_a else 150, 1, i),
                _face(W * 0.6, H / 2, 150 if big_a else 300, 2, i)]
    fa = va.analyze_faces(_frames(10, faces), W, H, 10)
    assert len(fa.identities) == 2
    for ident in fa.identities:
        assert ident.focus_share == pytest.approx(0.5)
        assert ident.presence_share == pytest.approx(1.0)


def test_subject_leaves_at_half_time():
    fr = _frames(10, lambda i, t: [_face(W / 2, H / 2, 300, 1, i)] if t < 5 else [])
    fa = va.analyze_faces(fr, W, H, 10)
    ident = fa.identities[0]
    assert ident.focus_share == pytest.approx(0.5)
    assert ident.presence_share == pytest.approx(0.5)
    assert ident.focus_s == pytest.approx(5.0)
    assert fa.duration == pytest.approx(10.0)


def test_one_frame_miss_does_not_flip_focus():
    def faces(i, t, miss):
        out = [_face(W * 0.7, H / 2, 120, 2, i)]
        if i not in miss:
            out.append(_face(W / 2, H / 2, 300, 1, i))
        return out
    fr = _frames(10, lambda i, t: faces(i, t, {20}))
    f1 = va.analyze_faces(fr, W, H, 10)
    a = f1.identities[0]
    assert a.focus_share == pytest.approx(1.0)
    assert f1.raw_focus[20] != f1.focus[20]          # smoothing did the work
    f0 = va.analyze_faces_f0(fr, W, H, 10)           # baseline flips
    assert f0.focus[20] != f0.focus[19]
    # a two-frame miss is a real change at the default tolerance
    fr2 = _frames(10, lambda i, t: faces(i, t, {20, 21}))
    f1b = va.analyze_faces(fr2, W, H, 10)
    assert f1b.focus[20] != f1b.focus[19]
    wider = dataclasses.replace(va.DEFAULT_PARAMS, focus_miss_frames=2)
    assert va.analyze_faces(fr2, W, H, 10, wider).identities[0].focus_share == pytest.approx(1.0)


def test_overlapping_tracks_never_merge_even_with_identical_embeddings():
    # the same embedding on two simultaneous faces (one detected twice, twins):
    # a person is in a frame at most once, so these stay two identities
    same = _emb(0, base_seed=5)

    def faces(i, t):
        return [{"bbox": (100, 100, 400, 400), "det_score": 0.9, "embedding": same},
                {"bbox": (1400, 100, 1700, 400), "det_score": 0.9, "embedding": same}]
    fa = va.analyze_faces(_frames(6, faces), W, H, 6)
    assert len(fa.identities) == 2
    assert all(i.presence_share == pytest.approx(1.0) for i in fa.identities)


def test_tracks_of_one_person_separated_in_time_merge():
    # same person, left side 0-3 s, absent 3-7 s, right side 7-10 s (Risk 6)
    def faces(i, t):
        if t < 3:
            return [_face(400, H / 2, 250, 7, i)]
        if t >= 7:
            return [_face(1500, H / 2, 250, 7, i)]
        return []
    fa = va.analyze_faces(_frames(10, faces), W, H, 10)
    assert len(fa.identities) == 1
    ident = fa.identities[0]
    assert ident.track_count == 2
    assert ident.presence_s == pytest.approx(6.0)
    assert len(ident.presence_intervals) == 2
    # a different person in the second half is NOT merged
    def faces2(i, t):
        if t < 3:
            return [_face(400, H / 2, 250, 7, i)]
        if t >= 7:
            return [_face(1500, H / 2, 250, 8, i)]
        return []
    assert len(va.analyze_faces(_frames(10, faces2), W, H, 10).identities) == 2


def test_coasting_across_a_short_gap():
    # no embeddings: only IoU can link. Missing 0.75 s (3 frames; dets 1.0 s apart)
    def faces(i, t):
        if 4.0 < t < 4.7:
            return []
        return [_face(W / 2, H / 2, 300, 1, i, emb=False)]
    fa = va.analyze_faces(_frames(10, faces), W, H, 10)
    assert len(fa.identities) == 1
    assert fa.identities[0].presence_share == pytest.approx(1.0)
    # a 2 s gap is not coasted: two tracks, and without embeddings no merge
    def faces2(i, t):
        if 4.0 < t < 6.0:
            return []
        return [_face(W / 2, H / 2, 300, 1, i, emb=False)]
    fb = va.analyze_faces(_frames(10, faces2), W, H, 10)
    assert len(fb.identities) == 2
    assert sum(i.presence_s for i in fb.identities) == pytest.approx(8.0)


def test_embedding_relinks_within_window_after_a_jump():
    # the face jumps across the frame (IoU 0) after 1.25 s: embedding links it
    def faces(i, t):
        if 4.0 < t < 5.2:
            return []
        x = 400 if t < 5 else 1500
        return [_face(x, H / 2, 250, 3, i)]
    fa = va.analyze_faces(_frames(10, faces), W, H, 10)
    assert len(fa.identities) == 1
    assert fa.identities[0].track_count == 1


def test_small_faces_are_present_but_never_focus():
    fr = _frames(4, lambda i, t: [_face(W / 2, H / 2, 20, 1, i)])   # A = 20/1080 < 0.03
    fa = va.analyze_faces(fr, W, H, 4)
    assert fa.identities[0].presence_share == pytest.approx(1.0)
    assert fa.identities[0].focus_share == 0.0


def test_segment_focus_and_summary():
    fr = _frames(12, lambda i, t: [_face(W / 2, H / 2, 300, 1, i)] if t < 6 else
                 [_face(W / 2, H / 2, 300, 2, i)])
    fa = va.analyze_faces(fr, W, H, 12)
    segs = va.segment_schedule(12)
    sf = va.segment_focus(fa, segs)
    assert sf[0] == sf[1] and sf[2] == sf[3] and sf[0] != sf[2]
    s = va.summarize_face_segments(fa, segs, names={sf[0]: "Calvin"})
    assert s["text"] == f"0–6 s Calvin (face_ix {sf[0]}), 6–12 s person (face_ix {sf[2]})"
    json.dumps(s)
    json.dumps(fa.to_json())


# ---------------------------------------------------------------- quality


def test_steady_pan_has_no_shake_and_jitter_does():
    n = 120
    pan = va.shake_score([15.0] * n, [0.0] * n, FPS, W)
    assert pan == pytest.approx(0.0, abs=1e-9)
    rng = np.random.default_rng(0)
    jitter = va.shake_score(15.0 + rng.normal(0, 20, n), rng.normal(0, 20, n), FPS, W)
    assert jitter > 0.5
    still = va.shake_score([0.0] * n, [0.0] * n, FPS, W)
    assert still == 0.0
    # shorter than the window: linear-fit fallback, pan still ~0
    assert va.shake_score([7.0] * 3, [1.0] * 3, FPS, W) == pytest.approx(0.0, abs=1e-9)


def test_immerkaer_sigma_recovers_noise_and_masks_edges():
    rng = np.random.default_rng(1)
    base = np.full((200, 300), 120.0)
    noisy = base + rng.normal(0, 5, base.shape)
    assert va.immerkaer_sigma(noisy) == pytest.approx(5.0, rel=0.2)
    assert va.immerkaer_sigma(noisy, flat_percentile=None) == pytest.approx(5.0, rel=0.1)
    # a strong texture inflates the unmasked estimate; the mask holds it down
    tex = base.copy()
    tex[:, 150:] += rng.normal(0, 40, (200, 150))      # strong fine detail on one half
    tex_noisy = tex + rng.normal(0, 5, base.shape)
    unmasked = va.immerkaer_sigma(tex_noisy, flat_percentile=None)
    masked = va.immerkaer_sigma(tex_noisy)
    assert unmasked > 3 * masked
    assert masked == pytest.approx(5.0, rel=0.35)
    assert va.noise_score([4.0, None, 6.0, 5.0, float("nan")]) == 5.0


def test_focus_consistency_is_relative_to_the_video():
    assert va.focus_consistency([10, 10, 10, 10, 2]) == pytest.approx(0.8)
    # scale-free: the same shape at 100x the sharpness reads the same
    assert va.focus_consistency([1000, 1000, 1000, 1000, 200]) == pytest.approx(0.8)
    assert va.focus_consistency([]) is None
    fr = _frames(2, lambda i, t: [_face(W / 2, H / 2, 300, 1, i, sharp=10.0 if i else 1.0)])
    fa = va.analyze_faces(fr, W, H, 2)
    vals = va.main_focus_sharpness(fa)
    assert len(vals) == len(fr)
    assert va.focus_consistency(vals) == pytest.approx(7 / 8)


def test_exposure_clipping():
    a = np.zeros((10, 10))
    a[:2] = 255
    a[2:5] = 128
    hi, lo = va.exposure_clipping(a)
    assert hi == pytest.approx(0.2) and lo == pytest.approx(0.5)
    agg = va.aggregate_exposure([(0.2, 0.5), (0.0, 0.1)])
    assert agg["highlight_frac"] == pytest.approx(0.1)


# ---------------------------------------------------------------- isolation


def test_imports_with_numpy_only():
    """CI mocks cv2/scipy/torch; the module must not need them at all."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import sys\n"
        "import photosearch.video_analysis, photosearch.video_subjects\n"
        "bad = sorted(m for m in ('cv2','scipy','torch','insightface','onnxruntime','sklearn')"
        " if m in sys.modules)\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == ""
