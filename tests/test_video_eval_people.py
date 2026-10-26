"""Faster video-eval labelling: proxies, cross-sample face clusters, run-grouped
segment labelling (photosearch/video_eval_people.py, docs/plans/video-support.md
step 4)."""

import json
import os

import numpy as np
import pytest

from photosearch import model_eval as me
from photosearch import video_eval_people as vp

from tests.test_video_eval import (API, SECRETS, VIDEO_BYTES, _cache_original, _sample,  # noqa: F401
                                   _eval_dir, ve)


def _unit(seed, dim=512):
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _near(base, seed, scale=0.005):
    v = base + np.random.default_rng(seed).normal(size=base.shape).astype(np.float32) * scale
    return v / np.linalg.norm(v)


def _clusters(pid, faces_by_seg, seg_version=None):
    """faces_by_seg: {seg_ix: [cluster_key or None(noise)]} -> a cluster file."""
    seg_version = seg_version or me.video_seg_version()
    faces, clusters = {}, {}
    noise = []
    for ix, keys in faces_by_seg.items():
        for k, key in enumerate(keys):
            fid = f"{pid}-{ix * 1000 + 500}-{10 * k}-0"
            faces[fid] = {"photo_id": pid, "t": ix + 0.5, "seg_ix": ix,
                          "seg_version": seg_version, "bbox": [10 * k, 0, 10 * k + 50, 50],
                          "det_score": 0.9}
            if key is None:
                noise.append(fid)
            else:
                clusters.setdefault(key, []).append(fid)
    return {"version": "v1", "faces": faces, "noise": noise,
            "clusters": [{"key": k, "faces": f, "suggestion": None} for k, f in clusters.items()]}


# --------------------------------------------------------------------------
# Run grouping (pure)
# --------------------------------------------------------------------------

def test_group_runs_merges_only_consecutive_equal_signatures():
    a, b, none = ((("p", 1),), "faces"), ((("p", 2),), "faces"), ((), "none")
    assert vp.group_runs([a, a, b, b, a, none, none]) == [[0, 1], [2, 3], [4], [5, 6]]
    assert vp.group_runs([]) == []


def test_signatures_follow_names_and_face_state():
    v = me.video_seg_version()
    cl = _clusters(7, {0: ["c1"], 1: ["c1", None], 2: [None], 4: ["c2"]})
    names = {f: {"kind": "person", "person_id": 3} for f in cl["faces"]
             if f in cl["clusters"][0]["faces"]}
    sigs = vp.segment_signatures(cl, names, 7, v, 5)
    assert sigs[0] == ((("p", 3),), "faces") == sigs[1]     # a stranger beside Calvin: same run
    assert sigs[2] == ((), "others")                          # faces, nobody named or clustered
    assert sigs[3] == ((), "none")                            # no face at all
    assert sigs[4] == ((), "others")                          # a cluster not named yet = others
    assert vp.group_runs(sigs) == [[0, 1], [2], [3], [4]]
    sigs2 = vp.segment_signatures(cl, names, 7, v, 6)
    assert vp.group_runs(sigs2) == [[0, 1], [2], [3], [4], [5]]
    # Junk is not a face; someone_else / ejected count as "others".
    junk = {f: {"kind": "junk"} for f in cl["faces"] if f in cl["clusters"][1]["faces"]}
    assert vp.segment_signatures(cl, junk, 7, v, 5)[4] == ((), "none")
    # Faces from a stale segmentation are ignored.
    assert vp.segment_signatures(cl, names, 7, v + "x", 5) == [((), "none")] * 5


def test_cluster_status_majority_and_ejected():
    c = {"key": "k", "faces": ["1-0-0-0", "1-1-0-0", "1-2-0-0"]}
    assert vp.cluster_status(c, {})["answer"] is None
    names = {"1-0-0-0": {"kind": "person", "person_id": 4},
             "1-1-0-0": {"kind": "person", "person_id": 4},
             "1-2-0-0": {"kind": "person", "person_id": 9, "ejected": True}}
    st = vp.cluster_status(c, names)
    assert st["answer"] == "p4" and st["done"]


# --------------------------------------------------------------------------
# Cluster file + names store
# --------------------------------------------------------------------------

def test_cluster_file_format_is_validated():
    cl = _clusters(7, {0: ["c1"]})
    saved = vp.save_clusters(cl)
    assert vp.load_clusters()["version"] == "v1" and saved["created"]
    with pytest.raises(ValueError):
        vp.save_clusters(dict(cl, version=""))
    bad = dict(cl, clusters=[{"key": "x", "faces": ["7-1-1-1"], "suggestion": None}])
    with pytest.raises(ValueError):
        vp.save_clusters(bad)
    with pytest.raises(ValueError):
        vp.save_clusters(dict(cl, faces={"../etc": {}}))


def test_face_id_is_a_function_of_frame_and_box():
    assert vp.face_id(12, 1.2345, (10.4, 20.6, 50, 60)) == "12-1234-10-21"
    with pytest.raises(ValueError):
        vp.crop_path("../../x")


def test_save_names_keeps_ejected_mark_only_from_the_pool():
    vp.save_names(["1-0-0-0"], "ejected")
    vp.save_names(["1-0-0-0"], "person", 5, keep_ejected=True)
    assert vp.load_names()["1-0-0-0"] == {**vp.load_names()["1-0-0-0"], "kind": "person",
                                          "person_id": 5, "ejected": True}
    vp.save_names(["1-0-0-0"], "person", 5)          # put back from the card: no mark
    assert "ejected" not in vp.load_names()["1-0-0-0"]
    vp.save_names(["1-0-0-0"], None)
    assert vp.load_names() == {}
    with pytest.raises(ValueError):
        vp.save_names(["1-0-0-0"], "person")          # no person_id
    with pytest.raises(ValueError):
        vp.save_names(["1-0-0-0"], "maybe")


def test_suggestion_is_conservative():
    a, b = _unit(1), _unit(2)
    refs_p = [1] * 5 + [2] * 5
    refs = np.stack([_near(a, i) for i in range(5)] + [_near(b, 10 + i) for i in range(5)])
    sug = vp.suggest_person(np.stack([_near(a, 50 + i) for i in range(4)]), refs_p, refs)
    assert sug["person_id"] == 1 and sug["dist"] < 0.2
    # Far from everyone: no suggestion.
    assert vp.suggest_person(np.stack([_unit(99)]), refs_p, refs) is None
    # Equidistant between two people: no suggestion (margin).
    mid = (a + b) / np.linalg.norm(a + b)
    assert vp.suggest_person(np.stack([mid]), [1, 2], np.stack([a, b]), max_dist=2.0) is None


def test_cluster_faces_end_to_end_with_stub_detector(monkeypatch):
    _sample([21, 22], durations={21: 9.0, 22: 6.0})
    for pid in (21, 22):
        _cache_original(pid)
    calvin, ellie = _unit(3), _unit(4)
    seen = {"n": 0}

    def frames(path, times):
        for t in times:
            yield t, np.zeros((120, 160, 3), np.uint8)

    def detect(fr):
        seen["n"] += 1
        k = seen["n"]
        return [{"bbox": (10, 70, 70, 10), "det_score": 0.9, "encoding": _near(calvin, k).tolist()},
                {"bbox": (10, 150, 70, 90), "det_score": 0.9, "encoding": _near(ellie, 1000 + k).tolist()},
                {"bbox": (0, 5, 5, 0), "det_score": 0.9, "encoding": _unit(k + 5000).tolist()}]  # too small

    refs = ([1] * 3, np.stack([_near(calvin, 7000 + i) for i in range(3)]), {1: "Calvin"})
    data = ve.cluster_faces(detect=detect, frames=frames, refs=refs, log=lambda *a: None)
    assert data["stats"]["clusters"] == 2 and data["stats"]["noise"] == 0
    assert data["stats"]["faces"] == 2 * len(ve.segment_frame_times(me.video_label_segments(9.0))) + \
        2 * len(ve.segment_frame_times(me.video_label_segments(6.0)))
    sug = [c["suggestion"] for c in data["clusters"]]
    assert sum(1 for s in sug if s) == 1 and next(s for s in sug if s)["person_id"] == 1
    fid = next(iter(data["faces"]))
    assert vp.crop_path(fid).exists()
    # The midpoint keyframes are pre-written at the strip size, under the
    # name the frame endpoint's cache reads.
    mid = me.video_label_segments(9.0)[0]
    assert vp.frame_path(21, (mid[0] + mid[1]) / 2, ve.STRIP_PX).exists()
    # Re-clustering reuses the detection cache.
    n = seen["n"]
    ve.cluster_faces(detect=detect, frames=frames, refs=refs, log=lambda *a: None)
    assert seen["n"] == n


def test_fetch_proxy_records_playback_and_streams(monkeypatch):
    _sample([31])
    calls = []

    def get_json(url, headers):
        calls.append(headers)
        return {"video": {"playback": {"codec": "hevc", "container": "mov"}, "needs_proxy": True,
                          "proxy_url": "/api/videos/31/proxy?v=2", "proxy": {"version": 2}}}

    def stream(url, dest, headers):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x" * 10)
        return 10

    e = vp.fetch_proxy(31, "http://nas:8000", intent="Fetch playable proxies for the video eval",
                       get_json=get_json, stream=stream)
    assert e["proxy_bytes"] == 10 and e["needs_proxy"] is True
    assert calls[0]["X-Photosearch-Source"] == "claude" and calls[0]["X-Photosearch-Intent"]
    info = vp.page_video_info(31, API)
    assert info["proxy_url"] == f"{API}/video/proxy/31" and info["playback"]["codec"] == "hevc"
    # Same version again: not re-streamed.
    vp.fetch_proxy(31, "http://nas:8000", intent="x", get_json=get_json,
                   stream=lambda *a: (_ for _ in ()).throw(AssertionError("re-fetched")))


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

@pytest.fixture
def people_env(client, db):
    ids = [r[0] for r in db.conn.execute("SELECT id FROM photos ORDER BY id LIMIT 3")]
    person = db.conn.execute("SELECT id FROM persons ORDER BY id LIMIT 1").fetchone()[0]
    _sample(ids, strata={ids[2]: "live-paired"})
    for pid in ids:
        _cache_original(pid)
    v = me.video_seg_version()
    for variant in ("secretvariantA", "secretvariantB"):
        run = me.open_run(ve.PASS_SUBJECT, variant, approach="S1",
                          grounding_model="vendor/secret-model")
        run["items"][str(ids[0])] = {"seg_version": v, "pred": [["p", person], ["p", person], ["none"]],
                                     "label_segments": [[0, 3], [3, 6], [6, 9]], "present": [person]}
        me.save_run(ve.PASS_SUBJECT, variant, run)
    cl = _clusters(ids[0], {0: ["c1"], 1: ["c1"], 2: []})
    cl["clusters"][0]["suggestion"] = {"person_id": person, "dist": 0.5, "margin": 0.2}
    vp.save_clusters(cl)
    for f in cl["faces"]:
        p = vp.crop_path(f)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xd8\xff\xd9")
    return client, ids, person, cl


def test_people_tab_lists_clusters_and_names_faces(people_env):
    client, ids, person, cl = people_env
    d = client.get(API + "/video/people").json()
    assert len(d["clusters"]) == 1 and d["clusters"][0]["size"] == 2
    assert d["clusters"][0]["suggestion"]["person_id"] == person
    faces = [f["id"] for f in d["clusters"][0]["faces"]]
    r = client.put(API + "/video/people/names", json={"faces": faces, "kind": "person",
                                                      "person_id": person})
    assert r.status_code == 200, r.text
    on_disk = json.loads((vp.video_dir() / vp.NAMES_FILE).read_text())
    assert {on_disk[f]["person_id"] for f in faces} == {person}
    assert not [p for p in os.listdir(vp.video_dir()) if p.endswith(".tmp")]   # atomic
    assert client.put(API + "/video/people/names", json={"faces": ["9-9-9-9"], "kind": "junk"}).status_code == 404
    assert client.put(API + "/video/people/names", json={"faces": faces, "kind": "person",
                                                         "person_id": 999999}).status_code == 400
    assert client.put(API + "/video/people/names", json={"faces": faces, "kind": "bogus"}).status_code == 400
    assert client.get(API + "/video/face-crop/" + faces[0]).status_code == 200
    assert client.get(API + "/video/face-crop/x.y").status_code == 400
    done = client.get(API + "/video/people").json()
    assert done["progress"] == {"done": 1, "total": 1}


def test_subject_runs_are_blind_and_preselect_nothing(people_env):
    client, ids, person, cl = people_env
    client.put(API + "/video/people/names", json={"faces": cl["clusters"][0]["faces"],
                                                  "kind": "person", "person_id": person})
    body = client.get(API + "/video/subject").text
    for s in SECRETS:
        assert s not in body, s
    for s in ("approach", "pred", "\"S1\"", "\"F1\"", "grounding"):
        assert s not in body, s
    d = json.loads(body)
    first = next(v for v in d["videos"] if v["photo_id"] == ids[0])
    assert first["runs"] == [{"segs": [0, 1], "people": [person], "faces": "faces"},
                             {"segs": [2], "people": [], "faces": "none"}]
    # Nothing answered for the owner: the runs suggest WHO is visible, never a label.
    assert all(s["label"] is None for s in first["segments"]) and first["who"] is None
    assert first["segments"][0]["frame_url"].startswith(API + "/video/frame/")
    assert d["clusters_ready"] is True
    # Labelling a run still stores one label per segment (same keys as before).
    v = me.video_seg_version()
    r = client.put(f"{API}/video/subject/{ids[0]}", json={"seg_version": v, "people": [person], "segments": {
        "0": {"kind": "person", "person_id": person}, "1": {"kind": "person", "person_id": person},
        "2": {"kind": "none"}}})
    assert r.status_code == 200
    segs = me.load_video_subject_labels()["segments"]
    assert {f"{ids[0]}:{v}:{i}" for i in range(3)} <= set(segs)


def test_proxy_is_range_served_and_info_points_at_it(people_env):
    client, ids, person, cl = people_env
    assert client.get(f"{API}/video/proxy/{ids[0]}").status_code == 404
    p = vp.proxy_path(ids[0])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(VIDEO_BYTES)
    r = client.get(f"{API}/video/proxy/{ids[0]}", headers={"Range": "bytes=5-14"})
    assert r.status_code == 206 and r.content == VIDEO_BYTES[5:15]
    assert r.headers["content-type"].startswith("video/mp4")
    info = client.get(f"{API}/video/info/{ids[0]}").json()
    assert info["proxy_url"] == f"{API}/video/proxy/{ids[0]}"
    assert client.get(f"{API}/video/info/999999").status_code == 404
    subj = client.get(API + "/video/subject").json()
    first = next(v for v in subj["videos"] if v["photo_id"] == ids[0])
    assert first["video"]["proxy_url"] == f"{API}/video/proxy/{ids[0]}"


def test_downscaled_frames_come_from_the_disk_cache(people_env, monkeypatch, tmp_path):
    from PIL import Image
    client, ids, person, cl = people_env
    jpg = tmp_path / "k.jpg"
    Image.new("RGB", (8, 6)).save(jpg)
    got = []
    monkeypatch.setattr(me, "cached_frame_jpeg", lambda pid, t, px=768, grab=None: got.append((pid, t, px)) or jpg)
    r = client.get(f"{API}/video/frame/{ids[0]}?t=1.5&max_px=320")
    assert r.status_code == 200 and got == [(ids[0], 1.5, 320)]


# --------------------------------------------------------------------------
# Repair: drop a video's faces, incremental clustering, video identification
# --------------------------------------------------------------------------

def test_drop_video_faces_keeps_every_other_name_and_key():
    a = _clusters(41, {0: ["c1", "c1"], 1: ["c2", None]})
    b = _clusters(42, {0: ["c1"], 1: ["c3", "c3", None]})
    cl = {"version": "v1", "faces": {**a["faces"], **b["faces"]}, "noise": a["noise"] + b["noise"],
          "clusters": [{"key": k, "suggestion": None,
                        "faces": [f for x in (a, b) for c in x["clusters"] if c["key"] == k for f in c["faces"]]}
                       for k in ("c1", "c2", "c3")]}
    names = {f: {"kind": "person", "person_id": 1} for f in cl["faces"]}
    out, new_names, rep = vp.drop_video_faces(cl, names, [42])
    assert all(m["photo_id"] == 41 for m in out["faces"].values())
    assert [c["key"] for c in out["clusters"]] == ["c1", "c2"]          # c3 emptied, keys kept
    assert rep["clusters_emptied"] == ["c3"] and rep["faces_removed"] == len(b["faces"])
    assert new_names == {f: n for f, n in names.items() if f.startswith("41-")}
    assert all(f.startswith("41-") for f in out["noise"])
    assert len(cl["faces"]) == len(a["faces"]) + len(b["faces"])        # input not mutated


def test_incremental_clustering_adds_faces_without_renumbering(monkeypatch):
    _sample([51, 52], durations={51: 9.0, 52: 9.0})
    for pid in (51, 52):
        _cache_original(pid)
    calvin, ellie, stranger = _unit(11), _unit(12), _unit(13)
    phase = {"v": "old", "n": 0}

    def frames(path, times):
        for t in times:
            yield t, np.zeros((120, 160, 3), np.uint8)

    def detect(fr):
        phase["n"] += 1
        k = phase["n"]
        if phase["v"] == "old":
            return [{"bbox": (10, 70, 70, 10), "det_score": 0.9, "encoding": _near(calvin, k).tolist()},
                    {"bbox": (10, 150, 70, 90), "det_score": 0.9, "encoding": _near(ellie, 900 + k).tolist()}]
        return [{"bbox": (20, 75, 80, 15), "det_score": 0.9, "encoding": _near(calvin, 2000 + k).tolist()},
                {"bbox": (90, 75, 150, 15), "det_score": 0.9, "encoding": _near(stranger, 3000 + k).tolist()}]

    no_refs = ([], np.zeros((0, 512), np.float32), {})
    _sample([51])
    ve.cluster_faces(detect=detect, frames=frames, refs=no_refs, log=lambda *a: None)
    _sample([51, 52], durations={51: 9.0, 52: 9.0})
    old = vp.load_clusters()
    # bbox is stored [left, top, right, bottom]: the old Calvin faces sit at left 10.
    calvin_key = next(c["key"] for c in old["clusters"] if old["faces"][c["faces"][0]]["bbox"][0] == 10)
    vp.save_names(next(c for c in old["clusters"] if c["key"] == calvin_key)["faces"], "person", 7)
    names_before = vp.load_names()

    phase["v"] = "new"
    data = ve.cluster_faces_incremental(ids=[52], detect=detect, frames=frames, refs=no_refs,
                                        log=lambda *a: None)
    # Old clusters: same keys, same order, old faces still there.
    assert [c["key"] for c in data["clusters"][:len(old["clusters"])]] == [c["key"] for c in old["clusters"]]
    for c_old, c_new in zip(old["clusters"], data["clusters"]):
        assert c_new["faces"][:len(c_old["faces"])] == c_old["faces"]
    assert vp.load_names() == names_before                    # names untouched
    calvin_c = next(c for c in data["clusters"] if c["key"] == calvin_key)
    added = [f for f in calvin_c["faces"] if f.startswith("52-")]
    assert added and all(data["faces"][f]["bbox"][1] == 20 for f in added)
    # The stranger is not forced into an old cluster: a new one is appended.
    new = data["clusters"][len(old["clusters"]):]
    assert len(new) == 1 and all(data["faces"][f]["bbox"][1] == 90 for f in new[0]["faces"])
    assert data["params"]["incremental"][-1]["new_clusters"] == 1
    # Running it again adds nothing.
    again = ve.cluster_faces_incremental(ids=[52], detect=detect, frames=frames, refs=no_refs,
                                         log=lambda *a: None)
    assert again["params"]["incremental"][-1]["faces_added"] == 0
    assert [c["faces"] for c in again["clusters"]] == [c["faces"] for c in data["clusters"]]


def test_labelling_pages_identify_videos_and_stay_blind(people_env):
    client, ids, person, cl = people_env
    body = client.get(API + "/video/subject").text
    for s in SECRETS:
        assert s not in body, s
    d = json.loads(body)
    n = len(d["videos"])
    for i, v in enumerate(d["videos"]):
        idt = v["ident"]
        assert idt["photo_id"] == v["photo_id"] and idt["position"] == i + 1 and idt["of"] == n
        assert idt["filename"] == f"V{v['photo_id']}.MP4"
        assert idt["app_url"] == f"/?photo={v['photo_id']}"
        assert set(idt) == {"photo_id", "filename", "date", "position", "of", "app_url"}
    body = client.get(API + "/video/people").text
    for s in SECRETS:
        assert s not in body, s
    card = json.loads(body)["clusters"][0]
    assert card["videos"] == 1
    assert [(x["photo_id"], x["faces"], x["position"]) for x in card["video_list"]] == [(ids[0], 2, 1)]
    assert card["video_list"][0]["app_url"] == f"/?photo={ids[0]}"
