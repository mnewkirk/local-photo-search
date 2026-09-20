/**
 * Batch flow — presentation logic for the /batches diagram.
 *
 * A pure, framework-free module: no DOM, no React, no state. Everything the
 * page decides from data alone lives here so it can be unit-tested without a
 * browser (see __tests__/batch-flow.test.js) — which row a step belongs in,
 * what its box says, and the one sentence at the top that tells the owner what
 * to do next.
 *
 * The vocabulary is frozen by photosearch/batch_state.py: STEP_ORDER, the six
 * STATES, and the `{step, kind, state, total, eligible, done, remaining,
 * failed, waiting_on, detail}` row shape. Nothing here re-derives state — the
 * server already did that, including the two traps that make `remaining == 0`
 * mean three different things.
 *
 * Loads as a browser global (window.PS.BatchFlow) or a CommonJS module.
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.PS = root.PS || {};
  root.PS.BatchFlow = api;
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // Frozen in batch_state.py. Also the legend's order: the two states that
  // mean "you are not finished and nothing is happening" (needs_queue,
  // blocked) sit at the end where they read as the exceptions they are.
  var STATES = ['completed', 'running', 'queued', 'needs_queue', 'waiting', 'blocked'];

  // Colour is NOT the signal — this is a status display, and a reader who
  // cannot tell green from amber still has to be able to run the pipeline. So
  // every box carries a glyph AND the state's word, and the glyphs are picked
  // to be distinguishable from each other at a glance (the test pins that they
  // are at least distinct).
  var STATE_META = {
    completed:   { label: 'Completed',          cls: 'st-completed',   glyph: '✓' },  // check
    running:     { label: 'Running',            cls: 'st-running',     glyph: '▶' },  // play
    queued:      { label: 'Queued',             cls: 'st-queued',      glyph: '⧗' },  // hourglass
    needs_queue: { label: 'Needs to be queued', cls: 'st-needs-queue', glyph: '⚑' },  // flag
    waiting:     { label: 'Waiting',            cls: 'st-waiting',     glyph: '…' },  // ellipsis
    blocked:     { label: 'Blocked',            cls: 'st-blocked',     glyph: '✕' },  // cross
  };

  // The diagram's shape. Each row is one horizontal band of boxes; the bands
  // run top to bottom in dependency order, so an arrow between two rows always
  // means "this cannot start until that is done".
  //
  // Row 3 is the three description-gated passes — they sit under `describe`
  // because that is literally what gates them (DEPENDS_ON in batch_state.py),
  // and because a reader who does not know that will otherwise read their
  // "Waiting" as a fault.
  var ROWS = [
    { key: 'ingest',  label: 'Ingest',
      steps: ['ingest'] },
    { key: 'clip',    label: 'Embed',
      steps: ['clip'] },
    { key: 'enrich',  label: 'Worker passes — run in parallel',
      steps: ['faces', 'quality', 'aesthetics', 'describe', 'category-visual'] },
    { key: 'text',    label: 'From the description',
      steps: ['category-content', 'keywords', 'verify'] },
    { key: 'nas',     label: 'NAS stages',
      steps: ['stacking', 'normalize_aesthetics', 'match_faces', 'resolve_dups',
              'warm_crops'] },
    { key: 'desktop', label: 'Desktop',
      steps: ['rank_measure'] },
  ];

  var OTHER_ROW_LABEL = 'Other steps';

  // ---------------------------------------------------------------------
  // formatting helpers
  // ---------------------------------------------------------------------

  // Deliberately not toLocaleString: this runs in tests and in a browser whose
  // locale we do not control, and "1 143" vs "1,143" is not worth a flaky test.
  function fmt(n) {
    var v = Math.round(Number(n) || 0);
    return String(v).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  function rate(n) {
    var v = Number(n) || 0;
    return v >= 10 ? String(Math.round(v)) : String(Math.round(v * 10) / 10);
  }

  function plural(n, one, many) { return n === 1 ? one : many; }

  /** Human form of a step name. The worker passes are the CLI's own pass
   *  names and stay verbatim (they are what you would type); the NAS/desktop
   *  steps are snake_case internals and read better as words. */
  function stepLabel(name) {
    return String(name || '').replace(/_/g, ' ');
  }

  /** SQLite stamps `YYYY-MM-DD HH:MM:SS` in UTC with no zone marker. Date
   *  parses that as LOCAL time in most browsers, which would make a heartbeat
   *  minutes-to-hours off, so the zone is spelled out here. */
  function parseTs(ts) {
    if (!ts) return NaN;
    return Date.parse(String(ts).replace(' ', 'T').replace(/Z?$/, 'Z'));
  }

  function minutesSince(ts, now) {
    var t = parseTs(ts);
    if (isNaN(t)) return null;
    var ms = (now == null ? Date.now() : now) - t;
    return ms / 60000;
  }

  /** The `YYYY-MM-DD` a batch's dated folder is named for, or null.
   *  `2091/2091-09-19_ILCE-7RM6` -> `2091-09-19`. Undated ingests land in
   *  `_undated/<suffix>` and legitimately have no day. */
  function dayOf(directory) {
    if (!directory) return null;
    var base = String(directory).split('/').pop();
    var m = base.match(/^(\d{4}-\d{2}-\d{2})/);
    return m ? m[1] : null;
  }

  // ---------------------------------------------------------------------
  // layout
  // ---------------------------------------------------------------------

  /**
   * Group the server's step rows into the diagram's bands.
   *
   * Unknown steps are kept in a trailing row rather than dropped: a newer
   * server can emit a step this page has never heard of, and silently hiding
   * it would under-report the work left — the one thing this page exists not
   * to do. Rows with nothing in them are dropped, so a partial `steps` array
   * (the minimal `computing` body sends none) renders cleanly.
   */
  function layout(steps) {
    var list = steps || [];
    var byName = {};
    var i;
    for (i = 0; i < list.length; i++) {
      if (list[i] && list[i].step) byName[list[i].step] = list[i];
    }

    var placed = {};
    var out = [];
    ROWS.forEach(function (row) {
      var found = [];
      row.steps.forEach(function (name) {
        if (byName[name]) { found.push(byName[name]); placed[name] = true; }
      });
      if (found.length) out.push({ key: row.key, label: row.label, steps: found });
    });

    var leftover = list.filter(function (s) {
      return s && s.step && !placed[s.step];
    });
    if (leftover.length) {
      out.push({ key: 'other', label: OTHER_ROW_LABEL, steps: leftover });
    }
    return out;
  }

  // ---------------------------------------------------------------------
  // stepCaption
  // ---------------------------------------------------------------------

  /**
   * The small grey line under a box's name — the number that makes the state
   * actionable.
   *
   * `done / eligible`, not `done / total`: the three description-gated passes
   * can only ever act on the photos that have a description, and showing them
   * against the whole batch would read as permanently behind.
   */
  function stepCaption(step) {
    if (!step) return '';
    var st = step.state;

    if (st === 'waiting') {
      return step.waiting_on ? 'waiting on ' + stepLabel(step.waiting_on) : 'waiting';
    }
    if (st === 'blocked') {
      if (step.failed > 0) return fmt(step.failed) + ' failed';
      return step.detail || 'blocked';
    }
    if (st === 'completed') {
      var doneText = step.done > 0 ? fmt(step.done) + ' done' : 'done';
      // e.g. faces: "113 with no detectable face" — finished, but worth seeing.
      return step.detail ? doneText + ' · ' + step.detail : doneText;
    }

    var base = fmt(step.done) + ' / ' + fmt(step.eligible);
    var extra = [];
    if (step.detail === 'stalled') extra.push('stalled');
    if (step.failed > 0) extra.push(fmt(step.failed) + ' failed');
    return extra.length ? base + ' · ' + extra.join(' · ') : base;
  }

  // ---------------------------------------------------------------------
  // summarize
  // ---------------------------------------------------------------------

  function byState(steps, st) {
    return (steps || []).filter(function (s) { return s && s.state === st; });
  }

  // ---------------------------------------------------------------------
  // fleetLaunchPasses — what ONE launch covers
  // ---------------------------------------------------------------------

  // Frozen in batch_state.py: WORKER_PASSES and DEPENDS_ON's worker half.
  var WORKER_PASSES = ['clip', 'faces', 'quality', 'aesthetics', 'describe',
    'category-visual', 'category-content', 'keywords', 'verify'];
  var WORKER_DEPENDS_ON = {
    'category-content': 'describe', keywords: 'describe', verify: 'describe',
  };
  // Dependency states that mean "satisfied without another launch".
  var UNDERWAY = ['completed', 'running', 'queued'];

  /**
   * The worker passes one fleet launch covers, in WORKER_PASSES order.
   *
   * **This is a MIRROR of `batch_state.fleet_launch_passes` (Python).** The
   * button says "N passes" and the server decides which N; if the two drift,
   * the page lies about what the click will do. They are pinned to the same
   * cases — `frontend/__tests__/batch-flow.test.js` and
   * `tests/test_batch_state.py` carry the same five scenarios (fresh batch,
   * dependency already completed, dependency blocked, dependency running,
   * nothing to launch), so changing one without the other fails a test on
   * both sides.
   *
   * Not just the `needs_queue` ones: the fleet runs sequentially through a
   * dependency order, and a second launch mid-run is refused (it would kill
   * the running fleet), so the description-gated passes have to ride along
   * even though they are `waiting` at click time.
   */
  function fleetLaunchPasses(state) {
    var rows = {};
    ((state && state.steps) || []).forEach(function (s) {
      if (s && s.step) rows[s.step] = s;
    });
    var chosen = [];
    var chosenSet = {};
    WORKER_PASSES.forEach(function (name) {
      var row = rows[name];
      if (!row) return;
      if (row.state === 'needs_queue') {
        chosen.push(name); chosenSet[name] = true;
        return;
      }
      if (row.state !== 'waiting') return;
      var dep = row.waiting_on || WORKER_DEPENDS_ON[name];
      if (!dep) return;
      var depState = rows[dep] ? rows[dep].state : null;
      // A `blocked` dependency does NOT admit its dependents — there will be
      // no output for them to read.
      if (chosenSet[dep] || UNDERWAY.indexOf(depState) !== -1) {
        chosen.push(name); chosenSet[name] = true;
      }
    });
    return chosen;
  }

  function sweepLine(sweep, now) {
    if (!sweep) return 'Ingest is running';
    if (sweep.stalled) {
      var mins = minutesSince(sweep.heartbeat_at, now);
      if (mins == null) return 'Stalled: no file moved recently';
      return 'Stalled: no file moved for ' + Math.max(1, Math.round(mins)) + ' min';
    }
    return 'Ingest is running — ' + fmt(sweep.files_moved) + ' files moved, '
      + rate(sweep.files_per_min) + '/min';
  }

  /**
   * `{done, total, headline}` for the top of the diagram.
   *
   * `done`/`total` count STEPS, not photos — the question the page answers is
   * "how far through the pipeline is this batch", and a batch can be 99% of
   * the way through `describe` with five steps not yet started.
   *
   * The headline is one sentence per `next_action`. Note what is NOT done
   * here: a missing `ready` is never read as false. The detail endpoint
   * returns a minimal `{batch, steps: [], stale, computing}` body while the
   * server is mid-derivation, and treating that as "not ready" would flip a
   * finished batch back to unfinished on every slow poll.
   */
  function summarize(state, now) {
    var steps = (state && state.steps) || [];
    var done = byState(steps, 'completed').length;
    var out = { done: done, total: steps.length, headline: '' };

    if (!state || !steps.length) {
      out.headline = 'Computing…';
      return out;
    }

    if (state.ready) { out.headline = 'Ready to review'; return out; }

    switch (state.next_action) {
      case 'wait_ingest':
        out.headline = sweepLine(state.sweep, now);
        return out;

      case 'launch_fleet': {
        // The count is the LAUNCH SET, not the `needs_queue` passes: one
        // sequential fleet drains the description-gated passes too, and a
        // headline that undercounted would promise less than the click does.
        var n = fleetLaunchPasses(state).length;
        out.headline = n
          ? 'Launch the worker fleet for ' + n + ' ' + plural(n, 'pass', 'passes')
          : 'Launch the worker fleet';
        return out;
      }

      case 'advance_nas': {
        var nas = byState(steps, 'needs_queue').filter(function (s) {
          return s.kind === 'nas';
        });
        if (!nas.length) { out.headline = 'Advance the NAS steps'; return out; }
        out.headline = 'Run ' + nas.length + ' NAS '
          + plural(nas.length, 'step', 'steps') + ': '
          + nas.map(function (s) { return stepLabel(s.step); }).join(', ');
        return out;
      }

      case 'review_blocked': {
        var blocked = byState(steps, 'blocked');
        var failed = blocked.reduce(function (a, s) { return a + (s.failed || 0); }, 0);
        var names = blocked.map(function (s) { return stepLabel(s.step); }).join(', ');
        out.headline = 'Blocked: ' + (names || 'a step') + ' can’t finish'
          + (failed ? ' (' + fmt(failed) + ' failed)' : '');
        return out;
      }

      case 'wait':
        out.headline = 'Everything is queued or running — nothing to launch';
        return out;

      default: {
        // next_action is null but `ready` is false. batch_state does that for
        // exactly one case: a batch whose photos are gone (deleted, pruned,
        // re-foldered). There is genuinely nothing to launch — scoping a run
        // to an empty directory is a no-op — so say that instead of offering
        // an action that would do nothing.
        var ingest = steps.filter(function (s) { return s.step === 'ingest'; })[0];
        var photos = ingest ? ingest.total
          : ((state.batch && state.batch.photo_count) || 0);
        out.headline = photos ? 'Nothing to do' : 'This batch has no photos';
        return out;
      }
    }
  }

  // ---------------------------------------------------------------------
  // advanceButton — the ONE action
  // ---------------------------------------------------------------------

  /**
   * What the "Advance batch" button should say and do, from the state alone.
   *
   * Returns `{label, enabled, action, reason}` where `action` is
   * 'launch_fleet' | 'advance_nas' | null. The two live actions hit different
   * endpoints on different machines — the fleet launches where the GPU is,
   * the NAS steps run where the DB and files are — so the button has to name
   * which one it means, not just say "Advance".
   *
   * Every disabled case carries a `reason`. A greyed button with no
   * explanation is the failure mode this whole page exists to avoid: the
   * owner is left guessing whether the pipeline is stuck or merely busy.
   */
  function advanceButton(state, opts) {
    var o = opts || {};
    var steps = (state && state.steps) || [];

    function out(label, enabled, action, reason) {
      return {
        label: label,
        // `busy` never changes the label — the button must not appear to
        // offer a different action just because a request is in flight.
        enabled: enabled && !o.busy,
        action: action,
        reason: o.busy && enabled ? 'Working…' : reason,
      };
    }

    if (!state || !steps.length) {
      return out('Advance batch', false, null,
        'Still working out what this batch needs.');
    }
    if (state.ready) {
      return out('Advance batch', false, null,
        'Ready to review — nothing left to advance.');
    }

    switch (state.next_action) {
      case 'launch_fleet': {
        // Same set the server will launch — see fleetLaunchPasses.
        var n = fleetLaunchPasses(state).length;
        return out(n ? 'Launch fleet — ' + n + ' ' + plural(n, 'pass', 'passes')
          : 'Launch worker fleet', true, 'launch_fleet', '');
      }
      case 'advance_nas': {
        var nas = byState(steps, 'needs_queue').filter(function (s) {
          return s.kind === 'nas';
        }).length;
        return out(nas ? 'Advance batch — ' + nas + ' NAS '
          + plural(nas, 'step', 'steps') : 'Advance batch',
          true, 'advance_nas', '');
      }
      case 'wait_ingest':
        return out('Advance batch', false, null,
          'Ingest is still running — wait for it to finish.');
      case 'wait':
        return out('Advance batch', false, null,
          'Everything is queued or running — nothing to launch.');
      case 'review_blocked':
        return out('Advance batch', false, null,
          'A step is blocked — review it before advancing.');
      default:
        // next_action null with ready false: batch_state does that for a
        // batch whose photos are gone. Nothing to launch, by design.
        return out('Advance batch', false, null, 'Nothing to advance.');
    }
  }

  // ---------------------------------------------------------------------
  // advanceLogLine — one SSE frame -> one log line
  // ---------------------------------------------------------------------

  /**
   * Turn one parsed SSE frame (`{event, data}` from PS.parseSSEFrame) into
   * `{text, cls}` for the advance log, or null when there is nothing to show.
   *
   * This is a pure function on purpose. The first version of it lived inline
   * in batches.html and was **completely dead** — it fed frames to a parser
   * that returned null for every `event:`-led frame, so the log rendered one
   * client-side line and nothing else, including swallowing the terminal
   * `fatal`. A dead log on the page whose job is to report what the pipeline
   * is doing is invisible until someone needs it most, so it is tested.
   *
   * The server's payloads carry no discriminator field, so shape decides —
   * but `event:` wins where it disagrees, because `fatal` is the one frame
   * that must never be mistaken for progress.
   */
  function advanceLogLine(frame) {
    if (!frame) return null;
    var d = frame.data || {};
    var name = frame.event;

    if (name === 'fatal' || d.error) {
      return { text: '! ' + (d.error || 'stream failed'), cls: 'l-err' };
    }
    if (name === 'cancelled') return { text: '— cancelled —', cls: 'l-err' };
    if (d.line !== undefined && d.line !== null) {
      return { text: String(d.line), cls: '' };
    }
    if (d.cmd) return { text: '$ ' + d.cmd, cls: '' };
    if (d.returncode !== undefined && d.returncode !== null) {
      return d.returncode === 0
        ? { text: '— finished —', cls: 'l-ok' }
        : { text: '— exited ' + d.returncode + ' —', cls: 'l-err' };
    }
    return null;
  }

  // ---------------------------------------------------------------------
  // placeholder — what the diagram area shows when it is not a diagram
  // ---------------------------------------------------------------------

  /**
   * Decide what the diagram area should show, from everything the page knows:
   * `{selected, listLoaded, hasBatches, state, err, computing}`.
   *
   * Returns `{kind, text, error}`:
   *   kind   'diagram' | 'error' | 'empty' | 'loading' | 'computing'
   *   text   the placeholder line for the non-diagram kinds, '' otherwise
   *   error  the error sentence to show, or ''. Set whenever `err` is — INCLUDING
   *          alongside a live diagram — so a failing poll is never silent.
   *
   * This is a pure function because getting it wrong is invisible: the first
   * version rendered the error only inside the branch that already had a
   * `state`, so a batch whose very first fetch 404'd sat on "Loading…"
   * forever, polling, saying nothing. On a page whose whole job is to not
   * under-report problems, that is the worst failure available — hence the
   * two rules below, both pinned by tests:
   *
   *   - an error with nothing to show is its OWN kind, never "Loading…"
   *     alongside it;
   *   - an error with a good diagram still surfaces, over the diagram.
   */
  function placeholder(view) {
    var v = view || {};
    var error = v.err ? 'Couldn’t load this batch — ' + v.err : '';
    function out(kind, text) { return { kind: kind, text: text, error: error }; }

    if (error && !v.state) return out('error', '');
    if (v.state) return out('diagram', '');
    if (v.selected == null) {
      if (v.listLoaded && !v.hasBatches) {
        return out('empty', 'No ingest batches yet. They appear after an '
          + 'ingest sweep registers one.');
      }
      return out('loading', 'Loading…');
    }
    return v.computing
      ? out('computing', 'Computing batch state…')
      : out('loading', 'Loading…');
  }

  return {
    STATES: STATES,
    STATE_META: STATE_META,
    ROWS: ROWS,
    WORKER_PASSES: WORKER_PASSES,
    advanceButton: advanceButton,
    advanceLogLine: advanceLogLine,
    fleetLaunchPasses: fleetLaunchPasses,
    layout: layout,
    placeholder: placeholder,
    summarize: summarize,
    sweepLine: sweepLine,
    stepCaption: stepCaption,
    stepLabel: stepLabel,
    dayOf: dayOf,
    fmt: fmt,
    rate: rate,
    minutesSince: minutesSince,
  };
}));
