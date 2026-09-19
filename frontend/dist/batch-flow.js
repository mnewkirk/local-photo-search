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
      return step.done > 0 ? fmt(step.done) + ' done' : 'done';
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
        // Worker passes only: a NAS stage in needs_queue is not something the
        // fleet can pick up, and counting it would send the owner to the
        // wrong button.
        var n = byState(steps, 'needs_queue').filter(function (s) {
          return s.kind === 'worker';
        }).length;
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

  return {
    STATES: STATES,
    STATE_META: STATE_META,
    ROWS: ROWS,
    layout: layout,
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
