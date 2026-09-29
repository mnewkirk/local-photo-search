#!/usr/bin/env python
"""Describe eval — measure the ways a description goes wrong.

Descriptions feed everything downstream (category-content and keywords only
ever see the text; the sailboat that got tagged "soccer" started as a
truncated description), and there is no single right description to score
against. So this measures failure modes, cheapest first:

  AUTOMATIC (no owner time — screen models before spending any):
    answered, degenerate (first attempt and final), truncated (hit the
    token cap), retried, fell back to `llava`, length, latency, and nouns the
    photo's own CLIP embedding does not support (verify.py's first pass).
  OWNER (on /eval/models, step 4 of the plan):
    wrong claims per description, blind pairwise preference, and the
    visible-text ground truth that `text accuracy` scores against.

Subcommands:

  sample           the visual-tags sample (same photos, same strata, so
                   describe shares a set with category-visual) plus ~10
                   photos with visible text:
                       python evals/describe_eval.py sample --db photo_index.db.local
  fetch-originals  pull every sample photo ONCE into the local cache, while
                   the NAS is up:
                       python evals/describe_eval.py fetch-originals
  run              the PRODUCTION `describe.describe_photo` (real prompt,
                   retry, degeneration recovery, fallback), one model per run:
                       export PHOTOSEARCH_TEXT_LLM_URL=http://localhost:1234/v1
                       python evals/describe_eval.py run --variant qwen9b --model qwen/qwen3.5-9b
  report           every cached variant side by side, plus `stored` (what
                   photos.description holds today):
                       python evals/describe_eval.py report --db photo_index.db.local

Storage: photosearch/model_eval.py, pass `describe`, under
PHOTOSEARCH_MODEL_EVAL_DIR (default ./evals/model-evals, git-ignored). The DB
is only ever opened read-only. Plan: docs/plans/model-eval-harnesses.md.
"""
import argparse
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

from photosearch import model_eval as me  # noqa: E402

PASS = "describe"
TEXT_STRATUM = "text"
# The describe call's max_tokens on the OpenAI route (_openai_chat_with_retry).
MAX_TOKENS = 768
# A model above this share of unanswered / degenerate / truncated output is
# not worth an owner's labelling time.
SCREEN_MAX_BAD = 0.10
CLIP_THRESHOLD = 0.18        # verify.verify_photo's default

# Stored descriptions that mention visible text. A candidate list for the
# `text` stratum, not a detector — the owner vetoes with --exclude.
_TEXT_HINT = re.compile(
    r"\b(sign|signs|signage|reads|written|menu|label|poster|banner|placard|"
    r"plaque|headline|lettering|inscription|the words?)\b|\"[^\"]{3,}\"",
    re.I)
# The describe prompt asks about text, so most descriptions carry a stock
# "There is no text visible in the image." Those sentences are dropped first.
_NO_TEXT_SENTENCE = re.compile(r"[^.]*\bno (visible )?(text|signs?|writing)\b[^.]*\.?", re.I)


# --------------------------------------------------------------------------
# Sample
# --------------------------------------------------------------------------

def text_candidates(conn, exclude=(), limit=200):
    """(photo_id, snippet) of photos whose stored description mentions text."""
    out = []
    ex = set(int(i) for i in exclude)
    for row in conn.execute(
            "SELECT id, description FROM photos WHERE description IS NOT NULL "
            "ORDER BY id"):
        if row["id"] in ex:
            continue
        desc = _NO_TEXT_SENTENCE.sub(" ", row["description"])
        m = _TEXT_HINT.search(desc)
        if m:
            s = max(0, m.start() - 60)
            out.append((row["id"], desc[s:m.end() + 60].replace("\n", " ")))
    return out[:limit] if limit else out


def draw_sample(conn, *, text_n=10, seed=11, exclude=(), text_ids=(), visual_sample=None):
    from photosearch import visual_tag_eval
    base = visual_sample if visual_sample is not None else visual_tag_eval.load_sample()
    photos = [dict(p) for p in base["photos"]]
    if not photos:
        raise SystemExit("The visual-tags sample is empty — draw it first "
                         "(python evals/visual_tags_eval.py sample).")
    have = {p["photo_id"] for p in photos}
    ex = set(int(i) for i in exclude)
    chosen = [int(i) for i in text_ids if int(i) not in have]
    if len(chosen) < text_n:
        pool = [pid for pid, _ in text_candidates(conn, exclude=have | ex | set(chosen),
                                                  limit=0)]
        random.Random(seed).shuffle(pool)
        chosen += pool[:text_n - len(chosen)]
    photos += [{"photo_id": pid, "stratum": TEXT_STRATUM} for pid in chosen]
    return photos


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def production_describer(path, model, prompt=None):
    from photosearch import describe
    return describe.describe_photo(path, model=model, prompt=prompt)


def describe_one(path, nominal, prompt=None, describer=None):
    """(text, item) for one photo. Raises TransportError when no answer came
    back, so the caller does not cache it."""
    from photosearch import describe
    describer = describer or production_describer
    rec = me.Recorder()
    t0 = time.time()
    with rec.active():
        text = describer(path, nominal, prompt)
    latency = time.time() - t0
    rec.check(text is None, "describe_photo")
    describe_calls = [c for c in rec.calls if c.get("model") != "llava"]
    first_raw = rec.calls[0].get("raw") if rec.calls else None
    item = {
        "text": text,
        "text_sha": me.text_sha(text),
        "latency_s": round(latency, 3),
        "calls": len(rec.calls),
        "attempts": rec.attempts(),
        "timeouts": rec.count("timeout"),
        "retried": len(describe_calls) > 1,
        # On LM Studio the `llava` fallback resolves by ROLE to the describe
        # model itself (describe.py), so it is really a third retry. Counted by
        # the nominal name the production code asked for.
        "fallback_called": any(c.get("model") == "llava" for c in rec.calls),
        "degenerate_first": bool(first_raw) and describe._is_degenerate(first_raw),
        "truncated": rec.truncated(MAX_TOKENS),
    }
    return text, item


def run_variant(variant, *, model=None, prompt_file=None, server=me.DEFAULT_SERVER,
                limit=None, force=False, solo=False, describer=None, fetch=None,
                log=print):
    from photosearch import describe
    me.pin_role_model("describe", model)
    nominal = describe.MODEL
    effective = describe.effective_model(model or nominal, "describe")
    prompt = None
    prompt_sha = None
    if prompt_file:
        with open(prompt_file, encoding="utf-8") as f:
            prompt = f.read()
        prompt_sha = me.text_sha(prompt)
    run = me.open_run(PASS, variant, force=force, effective_model=effective,
                      prompt_sha=prompt_sha)
    run.setdefault("model", model or nominal)
    run.setdefault("prompt_file", prompt_file)

    todo = [pid for pid in me.sample_ids(PASS) if str(pid) not in run["items"]]
    if limit is not None:
        todo = todo[:limit]
    loaded_start = me.lmstudio_loaded()
    log(f"[run] {variant}: model={effective} prompt={prompt_file or 'production'} — "
        f"{len(todo)} to do, {len(run['items'])} cached  (loaded: {loaded_start})")

    fc = me.FailureCounter(log)
    for i, pid in enumerate(todo, 1):
        try:
            path = me.original_path(pid, server, fetch)
            text, item = describe_one(str(path), nominal, prompt, describer)
        except Exception as e:
            fc.fail(pid, e)
            continue
        fc.ok()
        item["effective_model"] = effective
        run["items"][str(pid)] = item
        me.save_run(PASS, variant, run)
        shown = "UNANSWERED" if text is None else f"{len(text.split())} words"
        flags = " ".join(k for k in ("retried", "fallback_called", "truncated",
                                     "degenerate_first") if item[k])
        log(f"  [{i}/{len(todo)}] {pid}: {shown} ({item['latency_s']:.1f}s) {flags}")
    fc.summary()
    loaded_end = me.lmstudio_loaded()
    run["loaded_models_start"], run["loaded_models_end"] = loaded_start, loaded_end
    run["latency_label"] = me.latency_label(effective, loaded_start, loaded_end, solo)
    me.save_run(PASS, variant, run)
    return run


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def stored_run(conn):
    """Pseudo-variant: the description production holds today, with the model
    of the latest describe/verify generation that wrote it."""
    items = {}
    ids = me.sample_ids(PASS)
    for pid in ids:
        row = conn.execute("SELECT description FROM photos WHERE id = ?", (pid,)).fetchone()
        if row is None:
            continue
        gen = None
        try:
            gen = conn.execute(
                "SELECT model_used FROM generations WHERE photo_id = ? AND "
                "text_type IN ('describe', 'verify') ORDER BY id DESC LIMIT 1",
                (pid,)).fetchone()
        except Exception:
            pass
        text = row["description"]
        items[str(pid)] = {"text": text, "text_sha": me.text_sha(text),
                           "stored_model": gen["model_used"] if gen else None}
    return {"variant": me.STORED, "effective_model": "(production, mixed)",
            "items": items, "latency_label": "—"}


def _tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").casefold()) if len(t) >= 3]


def text_accuracy(text, truth):
    """(recall of the owner's visible-text tokens in the description,
    invented quoted strings). A quote counts as invented when none of its
    tokens appear in the truth."""
    truth_toks = set(_tokens(truth))
    if not truth_toks:
        return None, 0
    desc_toks = set(_tokens(text))
    recall = len(truth_toks & desc_toks) / len(truth_toks)
    invented = 0
    for q in re.findall(r"[\"“]([^\"”]{2,})[\"”]", text or ""):
        qt = set(_tokens(q))
        if qt and not (qt & truth_toks):
            invented += 1
    return recall, invented


def clip_flags(conn, pid, text, cache):
    """Nouns in `text` the photo's CLIP embedding does not support — verify's
    first pass, reused. Cached by (photo, text) since both are fixed. None
    when CLIP or the embedding is unavailable."""
    key = f"{pid}:{me.text_sha(text)}"
    if key in cache:
        return cache[key]
    emb = me.clip_embedding(conn, pid) if conn is not None else None
    if emb is None or not text:
        return None
    try:
        from photosearch import verify
        scores = verify.clip_score_description(emb, text)
        flagged, _, _ = verify._flag_by_clip(scores, [], CLIP_THRESHOLD)
    except Exception:
        return None
    cache[key] = {"nouns": len(scores), "flagged": [s["noun"] for s in flagged]}
    return cache[key]


def summarize(run, *, strata=None, truth=None, conn=None, clip_cache=None, use_clip=True):
    from photosearch import describe
    items = run["items"]
    n = len(items)
    texts = {pid: it.get("text") for pid, it in items.items()}
    answered = [t for t in texts.values() if t]
    words = [len(t.split()) for t in answered]
    row = {
        "variant": run.get("variant"), "effective_model": run.get("effective_model"),
        "n": n, "answered": len(answered),
        "degenerate_first": sum(1 for it in items.values() if it.get("degenerate_first")),
        "degenerate_final": sum(1 for t in answered if describe._is_degenerate(t)),
        "truncated": sum(1 for it in items.values() if it.get("truncated")),
        "retried": sum(1 for it in items.values() if it.get("retried")),
        "fallback": sum(1 for it in items.values() if it.get("fallback_called")),
        "timeouts": sum(it.get("timeouts") or 0 for it in items.values()),
        "words_p50": me.median(words), "words_p90": me.percentile(words, 0.9),
        "lat_p50": me.median(me.latencies(items)),
        "lat_p90": me.percentile(me.latencies(items), 0.9),
        "latency_label": run.get("latency_label", "unknown"),
    }
    row["format"] = sum(1 for t in answered if format_problem(t))
    bad = (n - row["answered"]) + row["degenerate_final"] + row["truncated"] + row["format"]
    row["screen_out"] = n >= me.MIN_N and bad / n > SCREEN_MAX_BAD

    # Disputed facts: each description is right / wrong / silent on each point
    # its sibling descriptions disagreed on (describe_eval.py disputes).
    disputes, dlabels = me.load_disputes(), me.load_dispute_labels()
    tally = {"right": 0, "wrong": 0, "silent": 0}
    for pid, it in items.items():
        d = disputes.get(str(pid))
        sha = it.get("text_sha") or me.text_sha(it.get("text"))
        if not d or sha not in d.get("shas", []):
            continue
        for q in d["questions"]:
            v = me.dispute_verdict(q["answers"].get(sha), dlabels.get(q["key"]))
            if v:
                tally[v] += 1
    row.update({f"disp_{k}": v for k, v in tally.items()})

    claims = me.load_claims()
    errs = [me.claim_errors(claims.get(it.get("text_sha") or me.text_sha(t)))
            for (pid, t), it in zip(texts.items(), items.values()) if t]
    errs = [x for x in errs if x is not None]
    row["claims_n"] = len(errs)
    row["clean"] = sum(1 for x in errs if x == 0)
    row["wrong_per_desc"] = sum(errs) / len(errs) if errs else None
    row["wrong_ci"] = bootstrap_mean_ci(errs) if errs else None

    if use_clip and conn is not None:
        per = [clip_flags(conn, pid, t, clip_cache if clip_cache is not None else {})
               for pid, t in texts.items() if t]
        per = [p for p in per if p is not None]
        row["clip_scored"] = len(per)
        row["clip_nouns"] = sum(p["nouns"] for p in per)
        row["clip_flagged"] = sum(len(p["flagged"]) for p in per)
    if truth:
        recs, invented = [], 0
        for pid, t in texts.items():
            tr = truth.get(str(pid))
            if not tr or not tr.get("done") or (strata and strata.get(int(pid)) != TEXT_STRATUM):
                continue
            r, inv = text_accuracy(t or "", tr.get("text", ""))
            if r is not None:
                recs.append(r)
                invented += inv
        row["text_n"] = len(recs)
        row["text_recall"] = sum(recs) / len(recs) if recs else None
        row["text_invented"] = invented
    return row


def bootstrap_mean_ci(xs, n_boot=2000, seed=3):
    """95% percentile-bootstrap CI of a mean (seeded, so reports are stable)."""
    if not xs:
        return None
    rnd = random.Random(seed)
    k = len(xs)
    means = sorted(sum(rnd.choice(xs) for _ in range(k)) / k for _ in range(n_boot))
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])


def sign_test_p(wins, losses):
    """Two-sided exact binomial sign test, ties excluded."""
    from math import comb
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def build_pairs(baseline, variants, *, n=40, seed=5):
    """Blind side-by-side pairs: `baseline` against each other variant, on n
    photos where every compared variant answered and the texts differ.
    Only finalists belong here — this is the expensive owner task."""
    runs = {v: me.load_run(PASS, v) for v in [baseline] + list(variants)}
    missing = [v for v, r in runs.items() if r is None]
    if missing:
        raise SystemExit(f"no cached run for: {missing}")
    others = [v for v in variants if v != baseline]
    if not others:
        raise SystemExit("--variants must name at least one variant besides the baseline")
    eligible = []
    for pid in me.sample_ids(PASS):
        texts = {v: (r["items"].get(str(pid)) or {}).get("text") for v, r in runs.items()}
        if all(texts.values()):
            eligible.append(pid)
    rnd = random.Random(seed)
    chosen = eligible if len(eligible) <= n else sorted(rnd.sample(eligible, n))
    pairs = []
    for pid in chosen:
        a = runs[baseline]["items"][str(pid)]
        for v in others:
            b = runs[v]["items"][str(pid)]
            a_sha = a.get("text_sha") or me.text_sha(a["text"])
            b_sha = b.get("text_sha") or me.text_sha(b["text"])
            if a_sha == b_sha:
                continue            # identical text: nothing to prefer
            pairs.append({"key": me.pair_key(a_sha, b_sha), "photo_id": pid,
                          "a_sha": a_sha, "b_sha": b_sha,
                          "a_variant": baseline, "b_variant": v})
    data = {"created": me.now_iso(), "seed": seed, "baseline": baseline, "pairs": pairs}
    me.write_pass_file(PASS, "pairs.json", data)
    return data


def pairwise_table():
    """[(challenger, baseline, wins, ties, losses, win_rate, p)] from the
    owner's preferences; ties count half toward the win rate."""
    prefs = me.load_prefs()
    agg = {}
    for p in me.load_pairs()["pairs"]:
        pref = prefs.get(p["key"])
        if pref is None:
            continue
        key = (p["b_variant"], p["a_variant"])
        w, t, l = agg.get(key, (0, 0, 0))
        winner = pref.get("winner_sha")
        if winner is None:
            t += 1
        elif winner == p["b_sha"]:
            w += 1
        else:
            l += 1
        agg[key] = (w, t, l)
    rows = []
    for (b, a), (w, t, l) in sorted(agg.items()):
        n = w + t + l
        rows.append((b, a, w, t, l, (w + t / 2) / n if n else None, sign_test_p(w, l)))
    return rows


_FORMAT_RE = re.compile(r"\*\*|^\s*#{1,6}\s|^\s*[-*•]\s+\S|^\s*\d+[.)]\s+\S", re.M)


def format_problem(text):
    """Markdown/list structure in what should be plain prose. It lands verbatim
    in the search blob and the photo modal — minicpm-v-4_5 appended a
    '**Search Index Description:**' bullet list on 2026-09-26."""
    return bool(text and _FORMAT_RE.search(text))


DISPUTE_PROMPT = """You are comparing several descriptions of the SAME photo, written independently. You cannot see the photo.

List the FACTUAL points on which the descriptions disagree AND that a person looking at the photo could settle: how many people or things there are, what is present or absent, colours, what people are doing, the activity or sport, ages or genders, and any text they quote. One question per disagreement.

Ignore differences of wording, style, mood or amount of detail when the facts agree. Also list a notable object or detail that only ONE description claims, as a question about whether it is there. A description that says nothing about a point answers "not mentioned".

{blocks}

Answer with ONLY a JSON array of at most {max_q} objects, each {{"question": "...", "answers": {{{keys}}}}}, where every answer is a short phrase. If the descriptions agree on every checkable fact, answer [].
"""


def _parse_disputes(raw, labels):
    import json as _json
    if not raw:
        return None
    s = raw.strip()
    a, b = s.find("["), s.rfind("]")
    if a < 0 or b < a:
        return None
    try:
        data = _json.loads(s[a:b + 1])
    except ValueError:
        return None
    if not isinstance(data, list):
        return None
    out = []
    for q in data:
        if not isinstance(q, dict) or not isinstance(q.get("question"), str):
            continue
        ans = q.get("answers")
        if not isinstance(ans, dict):
            continue
        norm = {lab: me.norm_answer(ans.get(lab)) for lab in labels}
        given = {v for v in norm.values() if v is not None}
        silent = any(v is None for v in norm.values())
        # A real dispute: two different answers, or one claim the rest are
        # silent on. Unanimous answers are not a disagreement.
        if len(given) >= 2 or (given and silent):
            out.append({"question": q["question"].strip(), "answers": norm})
    return out


def build_disputes(*, model=None, variants=None, limit=None, force=False, max_q=6,
                   chat=None, log=print):
    """Ask a text model where each photo's descriptions disagree. Resumable:
    a photo whose set of descriptions is unchanged is skipped."""
    from photosearch import describe
    import random as _random
    me.pin_role_model("text", model)
    effective = describe.effective_model(model or "llama3.2:3b", "text")
    chat = chat or describe._ollama_chat_with_retry
    variants = variants or me.list_variants(PASS)
    runs = {v: me.load_run(PASS, v) for v in variants}
    data = {} if force else me.load_disputes()
    todo = []
    for pid in me.sample_ids(PASS):
        texts = {}
        for r in runs.values():
            it = (r or {"items": {}})["items"].get(str(pid)) or {}
            if it.get("text"):
                texts.setdefault(it.get("text_sha") or me.text_sha(it["text"]), it["text"])
        if len(texts) < 2:
            continue
        shas = sorted(texts)
        prev = data.get(str(pid)) or {}
        if set(shas) <= set(prev.get("shas") or []):
            continue
        if prev and not force and any(q["key"] in me.load_dispute_labels()
                                      for q in prev.get("questions", [])):
            log(f"  - {pid}: labelled; its descriptions changed — use --add-variant, "
                "not a re-extraction (that would orphan the labels)")
            continue
        todo.append((pid, texts, shas))
    if limit is not None:
        todo = todo[:limit]
    log(f"[disputes] model={effective} — {len(todo)} photo(s) to compare")
    fc = me.FailureCounter(log)
    for i, (pid, texts, shas) in enumerate(todo, 1):
        order = list(shas)
        _random.Random(f"disputes-{pid}").shuffle(order)
        labels = [f"D{k + 1}" for k in range(len(order))]
        blocks = "\n\n".join(f"{lab}: {texts[sha]}" for lab, sha in zip(labels, order))
        prompt = DISPUTE_PROMPT.format(blocks=blocks, max_q=max_q,
                                       keys=", ".join(f'"{l}": "..."' for l in labels))
        try:
            raw = chat(model=model or "llama3.2:3b",
                       messages=[{"role": "user", "content": prompt}],
                       options={"temperature": 0}, role="text", timeout=180)
        except Exception as e:
            fc.fail(pid, e)
            continue
        qs = _parse_disputes(raw, labels)
        if qs is None:
            fc.fail(pid, f"unparseable: {(raw or '')[:80]!r}")
            continue
        fc.ok()
        by_label = dict(zip(labels, order))
        questions = [{"key": me.dispute_key(pid, q["question"], shas), "question": q["question"],
                      "answers": {by_label[l]: a for l, a in q["answers"].items()}}
                     for q in qs[:max_q]]
        data[str(pid)] = {"shas": shas, "model": effective, "created": me.now_iso(),
                          "questions": questions}
        me.save_disputes(data)
        log(f"  [{i}/{len(todo)}] {pid}: {len(questions)} disputed point(s)")
    fc.summary()
    return data


ADD_PROMPT = """A photo was described independently by several people. Where they disagreed, the facts were turned into questions, each listed below with the answers already given. You cannot see the photo.

Read ONE more description, NEW, and do two things:
1. For every question, give NEW's answer. If NEW says the same thing as one of the listed answers, copy that answer exactly. If NEW says something different, write NEW's own short answer. If NEW does not address the question, write "not mentioned".
2. List up to {max_new} specific, checkable detail(s) that NEW states and that none of the questions already cover, most notable first: an object, a count, a colour, an action, or text it quotes. Write each as a question with NEW's answer. Skip mood, style and general setting.

Questions:
{questions}

NEW: {text}

Answer with ONLY a JSON object: {{"answers": {{{keys}}}, "new": [{{"question": "...", "answer": "..."}}]}}
"""


def _parse_add(raw, qids):
    import json as _json
    if not raw:
        return None
    s = raw.strip()
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b < a:
        return None
    try:
        data = _json.loads(s[a:b + 1])
    except ValueError:
        return None
    ans = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(ans, dict):
        return None
    answers = {q: me.norm_answer(ans.get(q)) for q in qids}
    new = []
    for item in data.get("new") or []:
        if isinstance(item, dict) and isinstance(item.get("question"), str):
            a2 = me.norm_answer(item.get("answer"))
            if a2 is not None:
                new.append({"question": item["question"].strip(), "answer": a2})
    return answers, new


# In the original extraction each model faced at most ~0.7 "only this one says
# X" questions per photo (qwen3.5-9b 0.61, minicpm 0.70). Three per photo for an
# added model (the first run of this) is ~5x the scrutiny and ~200 extra labels;
# one keeps it close to parity.
ADD_MAX_NEW = 1


def add_variant_to_disputes(variant, *, model=None, chat=None, max_new=ADD_MAX_NEW, log=print):
    """Score one more describe variant against the ALREADY-LABELLED disputes,
    without re-extracting (which would orphan every label).

    The text model maps the new description onto each existing question —
    copying an existing answer when it agrees — so most of it is scored by the
    owner's labels as they stand. An answer the owner never saw is left
    unjudged (the label records `options_seen`) and shows on the Differences
    tab as new. Up to `max_new` (default 1) details only the new description states
    become new questions, so its own inventions are not invisible."""
    from photosearch import describe
    me.pin_role_model("text", model)
    effective = describe.effective_model(model or "llama3.2:3b", "text")
    chat = chat or describe._ollama_chat_with_retry
    run = me.load_run(PASS, variant)
    if run is None:
        raise SystemExit(f"no describe run {variant!r}")
    data = me.load_disputes()
    labels = me.load_dispute_labels()
    # Stamp every existing label with the answers it was made against, before
    # any new answer appears — or "None right" would silently convict it.
    for d in data.values():
        for q in d["questions"]:
            lab = labels.get(q["key"])
            if lab is not None and "options_seen" not in lab and not lab.get("cant_tell"):
                me.save_dispute_label(q["key"], lab.get("correct") or [], False,
                                      options_seen=me.dispute_options(q))
    todo = []
    for pid, d in data.items():
        it = run["items"].get(pid) or {}
        sha = it.get("text_sha") or me.text_sha(it.get("text"))
        if it.get("text") and sha not in d["shas"]:
            todo.append((pid, d, it["text"], sha))
    log(f"[add] {variant} into disputes via {effective} — {len(todo)} photo(s)")
    fc = me.FailureCounter(log)
    added_new = 0
    for i, (pid, d, text, sha) in enumerate(todo, 1):
        qids = [f"Q{k + 1}" for k in range(len(d["questions"]))]
        listing = "\n".join(
            f"{qid}: {q['question']} | answers: " +
            "; ".join(f'"{o}"' for o in me.dispute_options(q))
            for qid, q in zip(qids, d["questions"]))
        prompt = ADD_PROMPT.format(questions=listing, text=text, max_new=max_new,
                                   keys=", ".join(f'"{q}": "..."' for q in qids))
        try:
            raw = chat(model=model or "llama3.2:3b",
                       messages=[{"role": "user", "content": prompt}],
                       options={"temperature": 0}, role="text", timeout=180)
        except Exception as e:
            fc.fail(pid, e)
            continue
        parsed = _parse_add(raw, qids)
        if parsed is None:
            fc.fail(pid, f"unparseable: {(raw or '')[:80]!r}")
            continue
        fc.ok()
        answers, new = parsed
        new = new[:max_new]
        for qid, q in zip(qids, d["questions"]):
            a = answers[qid]
            # Snap to an existing answer when it matches after normalisation.
            for o in me.dispute_options(q):
                if a is not None and a == o:
                    a = o
            q["answers"][sha] = a
        for n in new:
            q = {"key": me.dispute_key(pid, n["question"], [sha]), "question": n["question"],
                 "answers": {s2: None for s2 in d["shas"]}, "added_by": variant}
            q["answers"][sha] = n["answer"]
            d["questions"].append(q)
            added_new += 1
        d["shas"] = sorted(d["shas"] + [sha])
        d.setdefault("added", []).append({"variant": variant, "model": effective,
                                          "at": me.now_iso()})
        me.save_disputes(data)
    fc.summary()
    labels = me.load_dispute_labels()
    unjudged = sum(1 for d in data.values() for q in d["questions"]
                   if not me.dispute_label_current(q, labels.get(q["key"])))
    log(f"[add] done — {added_new} new question(s); {unjudged} point(s) now need a label")
    return data


def build_report(*, db=None, use_clip=True):
    """Rows for every cached variant (+ `stored` when a DB is given). Also the
    entry point step 7's cross-pass summary calls."""
    conn = me.open_db_readonly(db) if db else None
    strata = {p["photo_id"]: p["stratum"] for p in me.load_sample(PASS)["photos"]}
    truth = me.read_pass_file(PASS, "text_truth.json", {})
    clip_cache = me.read_pass_file(PASS, "clip_flags.json", {})
    runs = [me.load_run(PASS, v) for v in me.list_variants(PASS)]
    if conn is not None:
        runs.append(stored_run(conn))
    rows = [summarize(r, strata=strata, truth=truth, conn=conn, clip_cache=clip_cache,
                      use_clip=use_clip) for r in runs]
    if clip_cache:
        me.write_pass_file(PASS, "clip_flags.json", clip_cache)
    return rows


def _f(v, d=1):
    return "—" if v is None else f"{v:.{d}f}"


def render(rows):
    out = []
    hdr = (f"{'variant':<18} {'effective model':<26} {'n':>3} {'ans':>5} {'degen1':>6} "
           f"{'degen':>5} {'trunc':>5} {'retry':>5} {'fallbk':>6} {'t/o':>4} "
           f"{'fmt':>4} {'words':>9} {'s/photo':>11} {'CLIP✗/noun':>10} {'text rec':>8}  latency")
    out.append(hdr)
    for r in rows:
        n = r["n"]
        clip = (me.fmt_ratio(r["clip_flagged"], r["clip_nouns"])
                if r.get("clip_nouns") else "—")
        tr = (f"{r['text_recall']:.2f}" if r.get("text_recall") is not None else "—")
        out.append(
            f"{str(r['variant']):<18} {str(r['effective_model'])[:26]:<26} {n:>3} "
            f"{me.fmt_ratio(r['answered'], n):>5} {me.fmt_ratio(r['degenerate_first'], n):>6} "
            f"{me.fmt_ratio(r['degenerate_final'], n):>5} {me.fmt_ratio(r['truncated'], n):>5} "
            f"{me.fmt_ratio(r['retried'], n):>5} {me.fmt_ratio(r['fallback'], n):>6} "
            f"{r['timeouts']:>4} {me.fmt_ratio(r['format'], n):>4} "
            f"{_f(r['words_p50'], 0) + '/' + _f(r['words_p90'], 0):>9} "
            f"{_f(r['lat_p50']) + '/' + _f(r['lat_p90']):>11} {clip:>10} {tr:>8}  "
            f"{r['latency_label']}{'   ✗ SCREEN OUT' if r['screen_out'] else ''}")
    disp = [r for r in rows if r.get("disp_right") or r.get("disp_wrong")]
    if disp:
        out.append("")
        out.append(f"{'variant':<18} {'disputed pts':>12} {'right':>6} {'wrong':>6} {'silent':>6} "
                   f"{'right when it answered':>22}")
        for r in disp:
            ans = r["disp_right"] + r["disp_wrong"]
            out.append(f"{str(r['variant']):<18} {ans + r['disp_silent']:>12} {r['disp_right']:>6} "
                       f"{r['disp_wrong']:>6} {r['disp_silent']:>6} "
                       f"{me.fmt_ratio(r['disp_right'], ans):>22}")
        out.append("disputed pts = facts the descriptions disagreed on, labelled on the "
                   "Differences tab; wrong = a claim the photo contradicts.")
    labelled = [r for r in rows if r.get("claims_n")]
    if labelled:
        out.append("")
        out.append(f"{'variant':<18} {'labelled':>8} {'clean':>6} {'wrong/desc':>10} {'95% CI':>13}")
        for r in labelled:
            ci = r["wrong_ci"]
            out.append(f"{str(r['variant']):<18} {r['claims_n']:>8} "
                       f"{me.fmt_ratio(r['clean'], r['claims_n']):>6} "
                       f"{_f(r['wrong_per_desc'], 2):>10} "
                       f"{('[' + _f(ci[0], 2) + ',' + _f(ci[1], 2) + ']') if ci else '—':>13}")
        out.append("clean = share of labelled descriptions with no wrong claim.")
    pw = pairwise_table()
    if pw:
        out.append("")
        out.append(f"{'challenger':<18} {'vs':<18} {'win':>4} {'tie':>4} {'loss':>4} "
                   f"{'win rate':>8} {'sign p':>7}")
        for b, a, w, t, l, rate, p in pw:
            out.append(f"{b:<18} {a:<18} {w:>4} {t:>4} {l:>4} {_f(rate, 2):>8} {_f(p, 3):>7}")
        out.append("win rate counts a tie as half; sign test excludes ties.")
    out.append("")
    out.append(f"ratios are shares of n; words and s/photo are p50/p90; 'n<3' = too few. "
               f"fmt = markdown/list structure in the prose. SCREEN OUT = more than "
               f"{SCREEN_MAX_BAD:.0%} unanswered/degenerate/truncated/badly formatted.")
    return "\n".join(out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    db_default = os.environ.get("PHOTOSEARCH_DB")

    s = sub.add_parser("sample")
    s.add_argument("--db", default=db_default)
    s.add_argument("--text-n", type=int, default=10)
    s.add_argument("--text-ids", default="", help="Comma-separated ids to use first.")
    s.add_argument("--exclude", default="", help="Comma-separated ids to veto.")
    s.add_argument("--seed", type=int, default=11)
    s.add_argument("--list", action="store_true",
                   help="Print text-stratum candidates and exit.")
    s.add_argument("--force", action="store_true")

    f = sub.add_parser("fetch-originals")
    f.add_argument("--server", default=me.DEFAULT_SERVER)

    r = sub.add_parser("run")
    r.add_argument("--variant", required=True)
    r.add_argument("--model", help="LM Studio id; pins PHOTOSEARCH_LLM_DESCRIBE_MODEL.")
    r.add_argument("--prompt-file")
    r.add_argument("--server", default=me.DEFAULT_SERVER)
    r.add_argument("--limit", type=int)
    r.add_argument("--force", action="store_true")
    r.add_argument("--solo", action="store_true")

    dp = sub.add_parser("disputes", help="find the facts each photo's descriptions disagree on")
    dp.add_argument("--model", help="LM Studio id of the text model that compares them")
    dp.add_argument("--variants", help="comma-separated; default every cached variant")
    dp.add_argument("--limit", type=int)
    dp.add_argument("--force", action="store_true")
    dp.add_argument("--add-variant", help="score one more variant against the existing, "
                                         "already-labelled disputes")

    pr = sub.add_parser("pairs", help="build blind pairwise comparisons (finalists only)")
    pr.add_argument("--baseline", required=True)
    pr.add_argument("--variants", required=True, help="comma-separated challengers")
    pr.add_argument("--n", type=int, default=40)
    pr.add_argument("--seed", type=int, default=5)

    p = sub.add_parser("report")
    p.add_argument("--db", default=db_default)
    p.add_argument("--no-clip", action="store_true")

    args = ap.parse_args(argv)
    ids = lambda s: [int(x) for x in s.split(",") if x.strip()]  # noqa: E731

    if args.cmd == "sample":
        conn = me.open_db_readonly(args.db)
        if args.list:
            for pid, snip in text_candidates(conn, exclude=ids(args.exclude), limit=80):
                print(f"{pid:>8}  …{snip}…")
            return
        if me.load_sample(PASS)["photos"] and not args.force:
            raise SystemExit("describe sample exists; labels are keyed to it. "
                             "--force to redraw.")
        photos = draw_sample(conn, text_n=args.text_n, seed=args.seed,
                             exclude=ids(args.exclude), text_ids=ids(args.text_ids))
        me.save_sample(PASS, photos, seed=args.seed, source="visual-tags+text")
        n_text = sum(1 for p in photos if p["stratum"] == TEXT_STRATUM)
        print(f"[sample] {len(photos)} photos ({n_text} in the text stratum)")
    elif args.cmd == "fetch-originals":
        me.prefetch(me.sample_ids(PASS), args.server)
    elif args.cmd == "run":
        run_variant(args.variant, model=args.model, prompt_file=args.prompt_file,
                    server=args.server, limit=args.limit, force=args.force,
                    solo=args.solo)
    elif args.cmd == "disputes" and args.add_variant:
        add_variant_to_disputes(args.add_variant, model=args.model)
    elif args.cmd == "disputes":
        build_disputes(model=args.model, limit=args.limit, force=args.force,
                       variants=[v.strip() for v in args.variants.split(",")] if args.variants else None)
    elif args.cmd == "pairs":
        data = build_pairs(args.baseline, [v.strip() for v in args.variants.split(",") if v.strip()],
                           n=args.n, seed=args.seed)
        print(f"[pairs] {len(data['pairs'])} pairs written — label them on /eval/models?tab=pairs")
    elif args.cmd == "report":
        print(render(build_report(db=args.db, use_clip=not args.no_clip)))


if __name__ == "__main__":
    main()
