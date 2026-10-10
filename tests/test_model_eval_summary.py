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
