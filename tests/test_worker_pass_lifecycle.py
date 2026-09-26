"""Worker pass retirement + sequential draining.

Two behaviours, both about NOT idling:

- A pass is retired the first time its queue reports EMPTY, and the worker
  exits once every pass has retired. `--stay-alive` restores the old
  poll-forever loop. This matters because an idle claim still opens
  BEGIN IMMEDIATE on the NAS's SQLite file and takes the single write lock —
  two workers idling for 13 days starved face assignments (500s), collection
  writes (30 min for 250 rows) and an entire overnight camera ingest (2,040
  files moved, zero rows written).

- Sequential (the default since 2026-09-26) drains one pass fully before
  starting the next instead of round-robining a batch at a time, so every
  photo gets CLIP before the slow LLM passes start and model weights aren't
  swapped in and out between every batch. `--round-robin` opts back in.

The claim call is stubbed: these pin the LOOP's control flow, which is where
the subtle bugs live, without needing torch or a NAS.
"""

import pytest

from photosearch import worker as W


class FakeClient:
    """Serves a scripted number of batches per pass, then reports empty."""

    worker_id = "test-worker"

    def __init__(self, batches: dict[str, int], fail_first: set[str] | None = None):
        self.remaining = dict(batches)
        self.calls: list[str] = []          # every claim, in order
        self.fail_first = set(fail_first or ())
        self._n = 0

    def claim_batch(self, pass_type, **kw):
        self.calls.append(pass_type)
        if pass_type in self.fail_first:
            self.fail_first.discard(pass_type)
            raise W._TRANSIENT[0]("simulated transport failure")
        if self.remaining.get(pass_type, 0) > 0:
            self.remaining[pass_type] -= 1
            self._n += 1
            return {"batch_id": f"b{self._n:04d}", "photos": [{"id": self._n}],
                    "remaining": self.remaining[pass_type]}
        return {"batch_id": None, "photos": []}

    def get_status(self, **kw):
        return {"queue_depth": dict(self.remaining), "active_claims": []}

    def renew_claim(self, *a, **k):
        return {}

    def submit_results(self, batch_id, pass_type, **kw):
        return {"written": 1, "processed": 1}


def run(batches, monkeypatch, tmp_path, fail_first=None, **kwargs):
    client = FakeClient(batches, fail_first=fail_first)
    monkeypatch.setattr(W, "WorkerClient", lambda *a, **k: client)
    monkeypatch.setattr(W, "_download_batch", lambda c, photos, d: {p["id"]: "x" for p in photos})
    for name in ("_process_clip", "_process_quality", "_process_faces"):
        monkeypatch.setattr(W, name, lambda d, **k: [{"photo_id": 1}])
    monkeypatch.setattr(W, "_unload_pass_models", lambda p: None)
    monkeypatch.setattr(W, "_flush_caches", lambda: None)
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    monkeypatch.setattr(W.tempfile, "mkdtemp", lambda **k: str(tmp_path))
    monkeypatch.setattr(W.shutil, "rmtree", lambda *a, **k: None)

    class _HB:
        def __init__(self, *a, **k): pass
        def start(self): pass
        def stop(self): pass
    monkeypatch.setattr(W, "_ClaimHeartbeat", _HB)

    W.run_worker(server="http://fake", passes=list(batches), **kwargs)
    return client


def test_exits_once_every_pass_is_drained(monkeypatch, tmp_path):
    """The headline: no --stay-alive means the fleet shuts itself down."""
    c = run({"clip": 2, "quality": 1}, monkeypatch, tmp_path)
    assert c.remaining == {"clip": 0, "quality": 0}
    # returning at all is the assertion — the old loop would spin forever


def test_retired_pass_is_not_polled_again(monkeypatch, tmp_path):
    """Once quality reports empty it must never be claimed again, even while
    clip still has work — that polling is what holds the NAS write lock."""
    c = run({"clip": 3, "quality": 0}, monkeypatch, tmp_path)
    assert c.calls.count("quality") == 1, c.calls
    assert c.calls.count("clip") >= 3


def test_stay_alive_keeps_polling_a_dry_queue(monkeypatch, tmp_path):
    """With --stay-alive an empty pass is retried rather than retired."""
    calls = {"n": 0}

    class Looping(FakeClient):
        def claim_batch(self, pass_type, **kw):
            calls["n"] += 1
            if calls["n"] > 6:               # break out of the infinite loop
                raise KeyboardInterrupt
            return super().claim_batch(pass_type, **kw)

    client = Looping({"clip": 0})
    monkeypatch.setattr(W, "WorkerClient", lambda *a, **k: client)
    monkeypatch.setattr(W, "_unload_pass_models", lambda p: None)
    monkeypatch.setattr(W, "_flush_caches", lambda: None)
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    monkeypatch.setattr(W.tempfile, "mkdtemp", lambda **k: str(tmp_path))
    W.run_worker(server="http://fake", passes=["clip"], stay_alive=True)
    assert client.calls.count("clip") > 1, "stay-alive must keep retrying"


def test_transient_failure_does_not_retire_a_pass(monkeypatch, tmp_path):
    """A lock/timeout/503 must NOT look like an empty queue — otherwise one
    NAS hiccup silently kills the fleet, which is worse than idling."""
    c = run({"clip": 2}, monkeypatch, tmp_path, fail_first={"clip"})
    # first claim raised, yet clip stayed active and still drained its work
    assert c.remaining["clip"] == 0
    assert c.calls.count("clip") >= 3        # failure + 2 batches + final empty


def test_sequential_drains_in_order(monkeypatch, tmp_path):
    """-p clip,quality --sequential must finish every clip batch before the
    first quality claim."""
    c = run({"clip": 3, "quality": 2}, monkeypatch, tmp_path, sequential=True)
    first_quality = c.calls.index("quality")
    assert set(c.calls[:first_quality]) == {"clip"}, c.calls
    assert c.remaining == {"clip": 0, "quality": 0}


def test_sequential_is_the_default(monkeypatch, tmp_path):
    """With no flag, clip drains completely before quality is ever claimed."""
    c = run({"clip": 3, "quality": 2}, monkeypatch, tmp_path)
    first_quality = c.calls.index("quality")
    assert set(c.calls[:first_quality]) == {"clip"}, c.calls
    assert c.remaining == {"clip": 0, "quality": 0}


def test_roundrobin_interleaves_when_asked(monkeypatch, tmp_path):
    """sequential=False: quality is claimed before clip is drained."""
    c = run({"clip": 3, "quality": 2}, monkeypatch, tmp_path, sequential=False)
    first_quality = c.calls.index("quality")
    assert c.calls[:first_quality].count("clip") == 1, c.calls


def test_cli_worker_defaults_to_sequential():
    """`cli.py worker` with neither flag must pass sequential=True."""
    import cli
    opt = {o.name: o for o in cli.worker.params}["sequential"]
    assert opt.default is True
    assert "--round-robin" in opt.secondary_opts


# --- the /admin/maintenance fleet launcher must expose both flags too -------

def test_workers_start_passes_flags_to_run_workers(monkeypatch):
    """The UI launcher builds a run-workers.sh command line; the flags have to
    reach it or the checkboxes are decorative."""
    import photosearch.admin_api as A

    seen = {}

    class R:
        returncode = 0; stdout = "ok"; stderr = ""

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return R()

    monkeypatch.setattr(A.subprocess, "run", fake_run)
    monkeypatch.setattr(A, "_run_workers_script", lambda: __file__)  # any existing path
    monkeypatch.setattr(A, "_fleet_server_url", lambda: "http://nas:8000")
    monkeypatch.setattr(A, "_native_repo_dir", lambda: ".")
    monkeypatch.setattr(A, "_fleet_env", lambda: {})

    req = A.WorkersStartRequest(passes=["clip", "quality"], count=2,
                                sequential=True, stay_alive=True)
    out = A.admin_workers_start(req)
    assert "--sequential" in seen["cmd"]
    assert "--stay-alive" in seen["cmd"]
    assert out["sequential"] is True and out["stay_alive"] is True


def test_workers_start_defaults_sequential_exit_when_drained(monkeypatch):
    """Default must be exit-when-drained and sequential."""
    import photosearch.admin_api as A

    seen = {}

    class R:
        returncode = 0; stdout = "ok"; stderr = ""

    monkeypatch.setattr(A.subprocess, "run", lambda cmd, **kw: (seen.__setitem__("cmd", cmd), R())[1])
    monkeypatch.setattr(A, "_run_workers_script", lambda: __file__)
    monkeypatch.setattr(A, "_fleet_server_url", lambda: "http://nas:8000")
    monkeypatch.setattr(A, "_native_repo_dir", lambda: ".")
    monkeypatch.setattr(A, "_fleet_env", lambda: {})

    A.admin_workers_start(A.WorkersStartRequest(passes=["clip"], count=1))
    assert "--sequential" in seen["cmd"]
    assert "--round-robin" not in seen["cmd"]
    assert "--stay-alive" not in seen["cmd"]

    A.admin_workers_start(A.WorkersStartRequest(passes=["clip"], count=1, sequential=False))
    assert "--round-robin" in seen["cmd"]
    assert "--sequential" not in seen["cmd"]


def test_contended_response_does_not_retire_a_pass(monkeypatch, tmp_path):
    """claim-batch returns contended=True when it found work but lost the race.

    That must not look like an empty queue: retiring on contention would shrink
    a fleet exactly when it is busiest, and the two changes (unlocked scan +
    drain-and-exit) would otherwise combine into that bug.
    """
    class Contending(FakeClient):
        def __init__(self):
            super().__init__({"clip": 1})
            self.n = 0

        def claim_batch(self, pass_type, **kw):
            self.calls.append(pass_type)
            self.n += 1
            if self.n == 1:
                return {"batch_id": None, "photos": [], "contended": True}
            return super().claim_batch(pass_type, **kw)

    client = Contending()
    monkeypatch.setattr(W, "WorkerClient", lambda *a, **k: client)
    monkeypatch.setattr(W, "_download_batch", lambda c, photos, d: {p["id"]: "x" for p in photos})
    monkeypatch.setattr(W, "_process_clip", lambda d, **k: [{"photo_id": 1}])
    monkeypatch.setattr(W, "_unload_pass_models", lambda p: None)
    monkeypatch.setattr(W, "_flush_caches", lambda: None)
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    monkeypatch.setattr(W.tempfile, "mkdtemp", lambda **k: str(tmp_path))
    monkeypatch.setattr(W.shutil, "rmtree", lambda *a, **k: None)

    class _HB:
        def __init__(self, *a, **k): pass
        def start(self): pass
        def stop(self): pass
    monkeypatch.setattr(W, "_ClaimHeartbeat", _HB)

    W.run_worker(server="http://fake", passes=["clip"])
    # survived the contended reply, then did the real batch
    assert client.remaining["clip"] == 0
    assert client.calls.count("clip") >= 3
