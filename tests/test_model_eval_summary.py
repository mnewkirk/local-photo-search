"""evals/model_eval_summary.py — Pareto marking, the fits-alone check and the
sequential fleet-time estimate (one model loaded at a time)."""

import importlib.util
import os

import pytest


def _load():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "evals", "model_eval_summary.py")
    spec = importlib.util.spec_from_file_location("model_eval_summary_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load()


def _row(model, head, sp, **kw):
    return {"model": model, "variant": model, "headline": head, "s_per_photo": sp,
            "latency": "solo", "detail": "", "flags": [], **kw}


def test_pareto_keeps_only_undominated_rows():
    a, b, c, d = _row("a", 0.9, 3.0), _row("b", 0.8, 1.0), _row("c", 0.7, 2.0), _row("d", None, 1)
    front = S.pareto([a, b, c, d])
    assert front == [a, b]                   # c is worse AND slower than b


def test_fits_alone():
    models = {"big": {"vram_gb": 23.0}, "ok": {"vram_gb": 9.0}}
    assert S.fits_alone("big", models, 1.5) is False
    assert S.fits_alone("ok", models, 1.5) is True
    assert S.fits_alone("unknown", models, 1.5) is None


def test_schedule_counts_one_load_per_model_change():
    roles = {"describe": [_row("q", 1, 2.0)], "category-content": [_row("t", 1, 0.5)],
             "keywords": [_row("t", 1, 0.5)], "verify": [_row("g", 1, 1.0)],
             "visual": [_row("g", 1, 1.0)], "aesthetics": [_row("g", 1, 3.0)]}
    models = {"q": {"swap_s": 20}, "t": {"swap_s": 5}, "g": {"swap_s": 10}}
    assign = {"describe": "q", "category-content": "t", "keywords": "t", "verify": "g",
              "visual": "g", "aesthetics": "g"}
    total, lines, problems = S.schedule(assign, roles, models, photos=100)
    # 100*(2 + .5 + .5 + 1 + 1 + 3) + loads q(20) + t(5) + g(10) — t and g load once each.
    assert total == pytest.approx(800 + 35)
    assert problems == []
    assign["verify"] = assign["describe"] = "q"
    _, _, problems = S.schedule(assign, {**roles, "verify": [_row("q", 1, 1.0)]}, models, 100)
    assert any("not independent" in p for p in problems)


def test_render_marks_same_model_verify_and_unknown_vram():
    roles = {"verify": [_row("q", 0.5, 1.0, describe_source="q"), _row("g", 0.4, 1.0,
                                                                     describe_source="q")]}
    text = S.render(roles, {"g": {"vram_gb": 5, "swap_s": 3}})
    assert "SAME AS DESCRIBE" in text
    assert "no runs yet" in text             # other roles
    # The example assignment must not pick the same-model verifier.
    ex = [l for l in text.splitlines() if l.strip().startswith("verify ")]
    assert ex and " g " in ex[0] + " "


def test_no_pareto_star_for_a_lead_inside_the_ci():
    fast = _row("fast", 0.56, 0.26, ci=(0.54, 0.59))
    slow = _row("slow", 0.57, 0.46, ci=(0.54, 0.59))       # "better" by noise, and slower
    best = _row("best", 0.74, 0.73, ci=(0.71, 0.76))       # really better
    front = S.pareto([fast, slow, best])
    assert fast in front and best in front and slow not in front
    assert S.ties_with_best([fast, slow, best]) == set()
    near = _row("near", 0.72, 0.9, ci=(0.69, 0.75))
    assert S.ties_with_best([best, near, fast]) == {id(near)}
    text = S.render({"category-content": [fast, slow, best, near]}, {})
    assert "≈0.720" in text


def _fake(monkeypatch, name, **attrs):
    mod = S._script(name)
    for k, v in attrs.items():
        monkeypatch.setattr(mod, k, v)
    real = S._script
    monkeypatch.setattr(S, "_script", lambda n: mod if n == name else real(n))
    return mod


def _drow(variant, records, **kw):
    r = {"variant": variant, "effective_model": variant, "n": len(records),
         "answered": sum(x["answered"] for x in records), "degenerate_final": 0,
         "truncated": 0, "format": 0, "bad": 0, "screen_out": False, "replay_bad": 0,
         "replay_screen_out": False, "lat_p50": 1.0, "latency_label": "solo",
         "disp_records": records, "spec": {"stale": True, "why": "no spec hash"}}
    r["disp_orig_right"] = sum(x["orig_right"] for x in records)
    r["disp_orig_wrong"] = sum(x["orig_wrong"] for x in records)
    r.update(kw)
    return r


def _rec(answered, orig_right, orig_wrong, right=None, wrong=None, orig_q=2):
    return {"answered": answered, "orig_right": orig_right, "orig_wrong": orig_wrong,
            "right": orig_right if right is None else right,
            "wrong": orig_wrong if wrong is None else wrong, "orig_q": orig_q, "all_q": orig_q}


def test_collect_describe_scores_original_questions_over_the_full_sample(monkeypatch):
    # "added" answers its own extra questions right (all-points 0.75) but is
    # 0.5 on the questions everyone faced; "gaps" skips a photo.
    added = _drow("added", [_rec(True, 1, 1, right=3, wrong=1)] * 4)
    gaps = _drow("gaps", [_rec(True, 2, 0)] * 3 + [_rec(False, 0, 0, orig_q=2)],
                 replay_screen_out=True, replay_bad=1)
    _fake(monkeypatch, "describe_eval", build_report=lambda **kw: [added, gaps])
    rows = {r["variant"]: r for r in S.collect_describe(None)}
    assert rows["added"]["headline"] == 0.5 and "all pts 0.75" in rows["added"]["detail"]
    assert rows["gaps"]["headline"] == pytest.approx(6 / 8)     # 2 unanswered qs = wrong
    assert (rows["gaps"]["n_answered"], rows["gaps"]["n_full"]) == (3, 4)
    assert rows["gaps"]["ci"] is not None
    assert any("replayed" in f for f in rows["gaps"]["flags"])
    assert any(f.startswith("STALE") for f in rows["gaps"]["flags"])


def test_collect_text_uses_f1_not_precision(monkeypatch):
    """llama's precision edged gemma's (0.828 vs 0.826) while its recall was
    0.25 vs 0.66; the precision headline picked the model rejected on recall."""
    def recs(tp, fp, fn, n=20):
        return [{"answered": True, "tp": tp, "fp": fp, "fn": fn, "unjudged": 0}] * n

    def row(v, rs):
        return {"pass": "category-content", "variant": v, "effective_model": v, "n": 20,
                "n_full": 20, "deferred": 0, "attempts": 20, "timeouts": 0, "lat_p50": 0.5,
                "latency_label": "solo", "labelled": 20, "answered_labelled": 20,
                "tags": 1, "unsupported_lenient": 0, "records": rs,
                "unjudged": 0, "spec": {"stale": False, "why": ""}}
    llama = row("llama", recs(5, 1, 15))          # P 0.83 R 0.25
    gemma = row("gemma", recs(12, 3, 6))          # P 0.80 R 0.67
    _fake(monkeypatch, "text_passes_eval", build_report=lambda **kw: [llama, gemma])
    got = {r["variant"]: r for r in S.collect_text(None)["category-content"]}
    assert got["gemma"]["headline"] > got["llama"]["headline"]
    assert "P 0.8" in got["gemma"]["detail"] and "R 0.67" in got["gemma"]["detail"]
    text = S.render({"category-content": list(got.values())}, {})
    ex = [l for l in text.splitlines() if l.strip().startswith("category-content ")]
    assert ex and "gemma" in ex[0]


def test_collect_verify_counts_failures_and_stratifies_the_ci(monkeypatch):
    recs = ([{"kind": "planted", "matched": True, "flagged": True, "failed": False,
              "missing": False}] * 8
            + [{"kind": "planted", "matched": False, "flagged": False, "failed": True,
                "missing": False}] * 2
            + [{"kind": "clean", "matched": False, "flagged": False, "failed": False,
                "missing": False}] * 9
            + [{"kind": "clean", "matched": False, "flagged": False, "failed": False,
                "missing": True}])
    row = {"variant": "g", "effective_model": "g", "mode": "llm", "planted": [10, 8, 8],
           "clean": [10, 0], "records": recs, "failed": {"planted": 2}, "missing": {"clean": 1},
           "clean_flags": 0, "lat_p50": 0.7, "latency_label": "solo",
           "spec": {"stale": False, "why": ""}}
    _fake(monkeypatch, "verify_eval",
          build_report=lambda: ({"source_effective_model": "q"}, [row]))
    r = S.collect_verify()[0]
    assert r["headline"] == pytest.approx(0.8 - 0.1)   # 8/10 caught − 1/10 clean not run
    assert (r["n_answered"], r["n_full"]) == (17, 20)
    assert r["ci"] and r["ci"][0] <= r["headline"] <= r["ci"][1]
    assert "3 failed/not run" in r["flags"]
