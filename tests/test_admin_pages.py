"""The admin pages split out of status.html (deploy + maintenance) must serve,
and status.html must keep serving as the trimmed monitoring dashboard."""

import pytest


@pytest.mark.parametrize("path,marker", [
    ("/status", "Indexing Status"),
    ("/admin/deploy", "Deployment"),
    ("/admin/maintenance", "Maintenance"),
])
def test_page_serves(client, path, marker):
    r = client.get(path)
    assert r.status_code == 200
    assert marker in r.text


def test_status_no_longer_has_moved_panels(client):
    """The moved/cut panels should be gone from the status page body."""
    body = client.get("/status").text
    assert "Run commands" not in body
    # The moved components' h2 headings live on the admin pages now.
    assert "Maintenance sweep" not in body
    assert "project docs on GitHub" in body  # the replacement docs link


def test_admin_pages_have_their_panels(client):
    assert "docker compose build" in client.get("/admin/deploy").text or \
           "Build" in client.get("/admin/deploy").text
    assert "Maintenance sweep" in client.get("/admin/maintenance").text
