"""Feature flags read from the environment, per call (so a test or a restart
with a new env flips them without a code change).

``PHOTOSEARCH_VIDEO_SUBJECT_SHARE`` — docs/plans/video-support.md step 6, D16:
the video main-subject share is COMPUTED (the video-subjects pass) but its
search surface stays hidden until the step 4 holdout eval passes. Off by
default. When off:

* ``GET /api/search`` rejects ``min_focus_share`` / ``min_focus_seconds`` /
  ``subject`` / ``min_subject_share`` and the ``focus_desc`` /
  ``subject_desc`` sorts with a 400 that names this variable (a silently
  ignored filter would return a broader answer than was asked for);
* the Ask/MCP tool schemas omit those properties and sort values, and a
  tool call that sends them anyway has them ignored (the model cannot know
  about them, and tool calls degrade rather than fail);
* the agent's locked filters drop them;
* ``GET /api/videos/subject-labels`` (the Subject typeahead) 404s;
* the search page hides the controls (it reads the flag from ``/api/stats``).

The PhotoModal subject timeline is an AUDIT tool and is shown regardless.
"""

from __future__ import annotations

import os

VIDEO_SUBJECT_SHARE_ENV = "PHOTOSEARCH_VIDEO_SUBJECT_SHARE"

# search_combined / tool arg names the flag gates, and the sorts.
VIDEO_SHARE_PARAMS = ("min_focus_share", "min_focus_seconds", "subject",
                      "min_subject_share")
VIDEO_SHARE_SORTS = ("focus_desc", "subject_desc")


def video_subject_share_enabled() -> bool:
    v = os.environ.get(VIDEO_SUBJECT_SHARE_ENV, "")
    return v.strip().lower() in ("1", "true", "yes", "on")
