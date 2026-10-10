"""Concurrent label saves must not lose each other.

The /eval/models page saves a photo's labels as parallel PUTs, and FastAPI runs
sync handlers in a thread pool. Each save used to be read-modify-write of one
JSON file with no lock, so parallel saves overwrote each other: on 2026-09-28
the owner's 126 Differences answers left 38 on disk, and every one of 30
'All claims' photos lost labels the same way."""

import threading

import pytest

from photosearch import model_eval as me


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_MODEL_EVAL_DIR", str(tmp_path / "me"))


def _hammer(fn, n=40):
    start = threading.Barrier(n)

    def run(i):
        start.wait()
        fn(i)
    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_parallel_dispute_labels_all_survive():
    _hammer(lambda i: me.save_dispute_label(f"k{i}", [f"a{i}"]))
    assert len(me.load_dispute_labels()) == 40


def test_parallel_claims_all_survive():
    _hammer(lambda i: me.save_claim(f"sha{i}", i, 3, [0]))
    assert len(me.load_claims()) == 40


def test_parallel_text_labels_all_survive():
    _hammer(lambda i: me.save_keyword_label(f"{i}:s", ["a", "b"], ["a"]))
    _hammer(lambda i: me.save_text_truth(i, "TEXT"))
    _hammer(lambda i: me.save_pref(f"p{i}", None))
    assert len(me.load_keyword_labels()) == 40
    assert len(me.load_text_truth()) == 40
    assert len(me.load_prefs()) == 40
