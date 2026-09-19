"""The /batches page and the pure module it depends on must both be served.

`batch-flow.js` is a separate file precisely so it can be unit-tested (see
frontend/__tests__/batch-flow.test.js). That split only works if the page can
actually load it, and it goes out through web.py's static catch-all rather than
a route of its own — so a change to that catch-all would break the page
silently. Hence the second test.
"""


def test_batches_page_serves(client):
    r = client.get("/batches")
    assert r.status_code == 200
    assert "Ingest batches" in r.text
    # It must pull in the module, not inline a second copy of the logic.
    assert '<script src="/batch-flow.js">' in r.text
    assert r.headers["cache-control"] == "no-cache"


def test_batch_flow_module_is_served(client):
    r = client.get("/batch-flow.js")
    assert r.status_code == 200
    assert "PS.BatchFlow" in r.text
    # Stale JS on a status page would show a stale pipeline.
    assert r.headers["cache-control"] == "no-cache"


def test_maintenance_page_links_to_the_per_batch_view(client):
    assert 'href: \'/batches\'' in client.get("/admin/maintenance").text
