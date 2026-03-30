#!/usr/bin/env python3
"""
Diagnostic: print CLIP scores for test queries to find optimal CLIP_MIN_SCORE.
Run with: python scripts/diagnose_clip_threshold.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from photosearch.db import PhotoDB
from photosearch.search import search_semantic

SAMPLE_PHOTOS = {
    "DSC04878.JPG", "DSC04880.JPG", "DSC04894.JPG",
    "DSC04895.JPG", "DSC04899.JPG", "DSC04907.JPG", "DSC04922.JPG",
}

QUERIES = [
    ("people outdoors", {"include": {"DSC04894.JPG", "DSC04895.JPG", "DSC04907.JPG", "DSC04922.JPG"},
                         "exclude": {"DSC04878.JPG", "DSC04880.JPG", "DSC04899.JPG"}}),
    ("ships", {"include": set(), "exclude": SAMPLE_PHOTOS}),
]

def main():
    db = PhotoDB("photo_index.db")

    for query, ground_truth in QUERIES:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print(f"{'='*60}")

        # Search with a very low threshold to get ALL scores
        results = search_semantic(db, query, limit=20, min_score=-999.0)

        # Filter to just sample photos and sort by score desc
        sample_results = [(r["filename"], r["score"]) for r in results
                          if r["filename"] in SAMPLE_PHOTOS]
        sample_results.sort(key=lambda x: -x[1])

        if sample_results:
            print(f"{'Filename':<20} {'Score':>8}  {'Expected?'}")
            print("-" * 45)
            for fname, score in sample_results:
                if fname in ground_truth.get("include", set()):
                    label = "✓ WANT"
                elif fname in ground_truth.get("exclude", set()):
                    label = "✗ DON'T WANT"
                else:
                    label = "  (neutral)"
                print(f"{fname:<20} {score:>8.4f}  {label}")
        else:
            print("  (no results even at min_score=-999)")

        # Show score gaps
        want_scores = [s for f, s in sample_results if f in ground_truth.get("include", set())]
        dont_scores = [s for f, s in sample_results if f in ground_truth.get("exclude", set())]
        if want_scores and dont_scores:
            print(f"\n  WANT range:      {min(want_scores):.4f} to {max(want_scores):.4f}")
            print(f"  DON'T WANT range:{min(dont_scores):.4f} to {max(dont_scores):.4f}")
            gap = min(want_scores) - max(dont_scores)
            if gap > 0:
                midpoint = max(dont_scores) + gap / 2
                print(f"  Gap: {gap:.4f}  →  suggested threshold: {midpoint:.4f}")
            else:
                print(f"  ⚠️  Scores OVERLAP by {-gap:.4f} — threshold alone won't separate them")

if __name__ == "__main__":
    main()
