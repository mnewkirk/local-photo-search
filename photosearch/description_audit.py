"""The one seam between people-bind and the per-photo audit log.

docs/plans/description-names.md. Relabels (face assign / clear / reject) are
the audit log's own events, so the rendered text is NOT recorded: it is a pure
function of state that is already audited, and the audit view can compute
"text now" by calling :func:`description_render.render`.

What this module records is the people-bind pass's own writes, as kinds:

* ``bound``    a row was written (or replaced) for a photo;
* ``unbound``  a row was dropped (the description changed, or ``clear-pass``);
* ``rebound``  the maintenance sweep re-paired a re-detected face, or dropped a
               row whose face is gone and re-queued the photo.

Today :func:`record` only logs. When the per-photo audit branch lands
(``photosearch/audit.py``, ``photo_audit``), ONLY the body of this function
changes: call ``audit.record(conn, photo_id, <action>, detail=json.dumps(details))``
there. Every call site already passes the connection. It must never raise: an
audit failure must not fail a submit that already landed.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

BOUND = "bound"
UNBOUND = "unbound"
REBOUND = "rebound"
KINDS = (BOUND, UNBOUND, REBOUND)


def record(conn, photo_id: int, kind: str, details: dict | None = None) -> None:
    try:
        if kind not in KINDS:
            logger.warning("description_audit: unknown kind %r", kind)
            return
        logger.info("description_audit %s photo=%s %s", kind, photo_id,
                    json.dumps(details or {}, sort_keys=True))
    except Exception:                                  # noqa: BLE001 — never fail a write
        logger.exception("description_audit.record failed")
