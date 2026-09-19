/**
 * Tests for batch-flow.js — the /batches flow diagram's presentation logic.
 *
 * The page itself is untestable without a browser, so every decision that can
 * be made from data alone lives in this pure module: which box goes in which
 * row, what the box says, and the one-sentence headline that tells the owner
 * what to do next. That is the whole point of the split.
 *
 * STEP_ORDER / STATES here mirror photosearch/batch_state.py. If those change,
 * `every STEP_ORDER step appears exactly once` below is the test that fails.
 */

const BF = require('../dist/batch-flow.js');

// Mirrors batch_state.STEP_ORDER exactly (globals.md, frozen).
const STEP_ORDER = ['ingest', 'clip', 'faces', 'quality', 'aesthetics', 'describe',
  'category-visual', 'category-content', 'keywords', 'verify',
  'stacking', 'normalize_aesthetics', 'match_faces', 'resolve_dups', 'warm_crops',
  'rank_measure'];

const STATES = ['completed', 'running', 'queued', 'needs_queue', 'waiting', 'blocked'];

const KIND = {
  ingest: 'ingest', stacking: 'nas', normalize_aesthetics: 'nas', match_faces: 'nas',
  resolve_dups: 'nas', warm_crops: 'nas', rank_measure: 'desktop',
};

/** One step row in the shape batch_state emits. */
function step(name, state, over) {
  return Object.assign({
    step: name,
    kind: KIND[name] || 'worker',
    state: state || 'needs_queue',
    total: 100, eligible: 100, done: 0, remaining: 100, failed: 0,
    waiting_on: null, detail: null,
  }, over || {});
}

/** A whole batch_state() response with every step in one state. */
function state(over, stepState) {
  return Object.assign({
    batch: { id: 1, directory: '2091/2091-09-19_ILCE-7RM6', photo_count: 100 },
    sweep: null,
    ready: false,
    next_action: 'launch_fleet',
    steps: STEP_ORDER.map((s) => step(s, stepState || 'needs_queue')),
  }, over || {});
}

// =========================================================================
// STATE_META — six states, each legible without colour
// =========================================================================

describe('STATE_META', () => {
  test('covers exactly the six frozen states', () => {
    expect(Object.keys(BF.STATE_META).sort()).toEqual([...STATES].sort());
    expect(BF.STATES).toEqual(STATES);
  });

  test('labels are the ones the brief specifies', () => {
    expect(BF.STATE_META.completed.label).toBe('Completed');
    expect(BF.STATE_META.running.label).toBe('Running');
    expect(BF.STATE_META.queued.label).toBe('Queued');
    expect(BF.STATE_META.needs_queue.label).toBe('Needs to be queued');
    expect(BF.STATE_META.waiting.label).toBe('Waiting');
    expect(BF.STATE_META.blocked.label).toBe('Blocked');
  });

  test('every state carries a class and a glyph, and no two glyphs collide', () => {
    // This is a status display: colour alone is not a signal. Every box shows
    // a glyph too, so the glyphs have to be distinguishable from each other.
    const glyphs = STATES.map((s) => {
      const m = BF.STATE_META[s];
      expect(typeof m.cls).toBe('string');
      expect(m.cls.length).toBeGreaterThan(0);
      expect(typeof m.glyph).toBe('string');
      expect(m.glyph.length).toBeGreaterThan(0);
      return m.glyph;
    });
    expect(new Set(glyphs).size).toBe(STATES.length);
  });
});

// =========================================================================
// layout
// =========================================================================

describe('layout', () => {
  const rows = BF.layout(STEP_ORDER.map((s) => step(s)));
  const flat = rows.reduce((a, r) => a.concat(r.steps.map((s) => s.step)), []);

  test('places every STEP_ORDER step exactly once', () => {
    expect([...flat].sort()).toEqual([...STEP_ORDER].sort());
    expect(flat.length).toBe(STEP_ORDER.length);
  });

  test('rows are the pipeline shape from the brief', () => {
    expect(rows.map((r) => r.steps.map((s) => s.step))).toEqual([
      ['ingest'],
      ['clip'],
      ['faces', 'quality', 'aesthetics', 'describe', 'category-visual'],
      ['category-content', 'keywords', 'verify'],
      ['stacking', 'normalize_aesthetics', 'match_faces', 'resolve_dups', 'warm_crops'],
      ['rank_measure'],
    ]);
  });

  test('every row has a key and a label', () => {
    rows.forEach((r) => {
      expect(typeof r.key).toBe('string');
      expect(r.label.length).toBeGreaterThan(0);
    });
  });

  test('keeps an unknown step in a trailing row rather than dropping it', () => {
    // A newer server can emit a step this page has never heard of. Silently
    // dropping it would under-report the work left, which is the one thing
    // this page exists not to do.
    const out = BF.layout(STEP_ORDER.map((s) => step(s)).concat([step('brand_new')]));
    const last = out[out.length - 1];
    expect(last.steps.map((s) => s.step)).toEqual(['brand_new']);
    const all = out.reduce((a, r) => a.concat(r.steps.map((s) => s.step)), []);
    expect(all.length).toBe(STEP_ORDER.length + 1);
  });

  test('drops rows with nothing in them', () => {
    const out = BF.layout([step('ingest'), step('clip')]);
    expect(out.map((r) => r.steps.map((s) => s.step))).toEqual([['ingest'], ['clip']]);
  });

  test('tolerates no steps at all', () => {
    expect(BF.layout([])).toEqual([]);
    expect(BF.layout(null)).toEqual([]);
  });

  test('carries the step objects through, not just their names', () => {
    const out = BF.layout([step('describe', 'running', { done: 7 })]);
    expect(out[0].steps[0].state).toBe('running');
    expect(out[0].steps[0].done).toBe(7);
  });
});

// =========================================================================
// summarize
// =========================================================================

describe('summarize — done/total', () => {
  test('counts completed steps against all steps', () => {
    const s = state({ steps: [step('ingest', 'completed'), step('clip', 'completed'),
      step('faces', 'needs_queue'), step('quality', 'running')] });
    const out = BF.summarize(s);
    expect(out.done).toBe(2);
    expect(out.total).toBe(4);
  });
});

describe('summarize — headline per next_action', () => {
  test('ready to review', () => {
    const s = state({ ready: true, next_action: null }, 'completed');
    expect(BF.summarize(s).headline).toBe('Ready to review');
  });

  test('ingest running quotes the sweep progress', () => {
    const s = state({
      next_action: 'wait_ingest',
      sweep: { status: 'moving', files_moved: 412, files_seen: 500,
               files_per_min: 38.4, stalled: false,
               heartbeat_at: '2091-09-19 10:00:00' },
    });
    expect(BF.summarize(s).headline)
      .toBe('Ingest is running — 412 files moved, 38/min');
  });

  test('a slow sweep keeps one decimal', () => {
    const s = state({
      next_action: 'wait_ingest',
      sweep: { status: 'moving', files_moved: 7, files_per_min: 2.34, stalled: false },
    });
    expect(BF.summarize(s).headline).toBe('Ingest is running — 7 files moved, 2.3/min');
  });

  test('a stalled sweep says how long it has been stuck', () => {
    const s = state({
      next_action: 'wait_ingest',
      sweep: { status: 'moving', files_moved: 412, files_per_min: 38.4, stalled: true,
               heartbeat_at: '2091-09-19 10:00:00' },
    });
    const now = Date.parse('2091-09-19T10:06:00Z');
    expect(BF.summarize(s, now).headline).toBe('Stalled: no file moved for 6 min');
  });

  test('a stalled sweep with no readable heartbeat still says stalled', () => {
    const s = state({
      next_action: 'wait_ingest',
      sweep: { status: 'moving', files_moved: 1, files_per_min: 0, stalled: true },
    });
    expect(BF.summarize(s).headline).toBe('Stalled: no file moved recently');
  });

  test('launch the fleet counts the worker passes that need queueing', () => {
    const s = state({
      next_action: 'launch_fleet',
      steps: [
        step('ingest', 'completed'),
        step('clip', 'needs_queue'), step('faces', 'needs_queue'),
        step('quality', 'needs_queue'), step('aesthetics', 'needs_queue'),
        step('describe', 'needs_queue'),
        step('verify', 'waiting', { waiting_on: 'describe' }),
        // A NAS step in needs_queue must NOT be counted as a fleet pass.
        step('stacking', 'needs_queue'),
      ],
    });
    expect(BF.summarize(s).headline).toBe('Launch the worker fleet for 5 passes');
  });

  test('one pass is singular', () => {
    const s = state({
      next_action: 'launch_fleet',
      steps: [step('ingest', 'completed'), step('clip', 'needs_queue')],
    });
    expect(BF.summarize(s).headline).toBe('Launch the worker fleet for 1 pass');
  });

  test('advancing the NAS names the stages', () => {
    const s = state({
      next_action: 'advance_nas',
      steps: [step('stacking', 'needs_queue'), step('match_faces', 'needs_queue'),
        step('warm_crops', 'waiting', { waiting_on: 'faces' })],
    });
    expect(BF.summarize(s).headline).toBe('Run 2 NAS steps: stacking, match faces');
  });

  test('one NAS stage is singular', () => {
    const s = state({ next_action: 'advance_nas', steps: [step('stacking', 'needs_queue')] });
    expect(BF.summarize(s).headline).toBe('Run 1 NAS step: stacking');
  });

  test('blocked names the steps and the photos they gave up on', () => {
    const s = state({
      next_action: 'review_blocked',
      steps: [step('describe', 'blocked', { failed: 12 }),
        step('keywords', 'blocked', { failed: 2 }),
        step('clip', 'completed')],
    });
    expect(BF.summarize(s).headline)
      .toBe('Blocked: describe, keywords can’t finish (14 failed)');
  });

  test('blocked with no failure count still reads', () => {
    const s = state({
      next_action: 'review_blocked',
      steps: [step('ingest', 'blocked', { detail: 'disk full' })],
    });
    expect(BF.summarize(s).headline).toBe('Blocked: ingest can’t finish');
  });

  test('nothing to launch while work is in flight', () => {
    const s = state({ next_action: 'wait' }, 'running');
    expect(BF.summarize(s).headline)
      .toBe('Everything is queued or running — nothing to launch');
  });

  test('an emptied batch says so instead of offering an action', () => {
    // batch_state returns next_action=None for a batch whose photos are gone.
    const s = state({
      next_action: null, ready: false,
      batch: { id: 1, directory: '2091/x', photo_count: 0 },
      steps: STEP_ORDER.map((n) => step(n, 'needs_queue',
        { total: 0, eligible: 0, remaining: 0 })),
    });
    expect(BF.summarize(s).headline).toBe('This batch has no photos');
  });

  test('the minimal computing body does not pretend to know anything', () => {
    // GET /api/batches/{id} returns {batch, steps: [], stale, computing} with
    // NO `ready` / `next_action` when the server is mid-derivation. Reading a
    // missing `ready` as false would show "not ready" for a ready batch.
    const out = BF.summarize({ batch: { id: 1 }, steps: [], stale: true, computing: true });
    expect(out.headline).toBe('Computing…');
    expect(out.done).toBe(0);
    expect(out.total).toBe(0);
  });

  test('no state at all is survivable', () => {
    expect(BF.summarize(null).headline).toBe('Computing…');
  });
});

// =========================================================================
// stepCaption
// =========================================================================

describe('stepCaption', () => {
  test('completed says how many landed, with thousands separators', () => {
    expect(BF.stepCaption(step('describe', 'completed',
      { total: 1373, eligible: 1373, done: 1373, remaining: 0 }))).toBe('1,373 done');
  });

  test('a completed step with nothing to do just says done', () => {
    expect(BF.stepCaption(step('stacking', 'completed',
      { total: 0, eligible: 0, done: 0, remaining: 0 }))).toBe('done');
  });

  test('in-progress shows done over eligible', () => {
    expect(BF.stepCaption(step('describe', 'running',
      { total: 1373, eligible: 1373, done: 1143, remaining: 230 })))
      .toBe('1,143 / 1,373');
  });

  test('needs_queue shows the same progress shape', () => {
    expect(BF.stepCaption(step('faces', 'needs_queue',
      { total: 300, eligible: 300, done: 0, remaining: 300 }))).toBe('0 / 300');
  });

  test('waiting names what it is waiting on', () => {
    expect(BF.stepCaption(step('keywords', 'waiting', { waiting_on: 'describe' })))
      .toBe('waiting on describe');
    expect(BF.stepCaption(step('warm_crops', 'waiting', { waiting_on: 'match_faces' })))
      .toBe('waiting on match faces');
  });

  test('waiting with no named dependency still reads', () => {
    expect(BF.stepCaption(step('keywords', 'waiting'))).toBe('waiting');
  });

  test('blocked leads with the failure count', () => {
    expect(BF.stepCaption(step('describe', 'blocked',
      { done: 10, eligible: 22, failed: 12 }))).toBe('12 failed');
  });

  test('blocked with no count falls back to the detail', () => {
    expect(BF.stepCaption(step('ingest', 'blocked', { detail: 'disk full' })))
      .toBe('disk full');
    expect(BF.stepCaption(step('ingest', 'blocked'))).toBe('blocked');
  });

  test('a stalled ingest says so next to its progress', () => {
    expect(BF.stepCaption(step('ingest', 'running',
      { total: 412, eligible: 412, done: 0, detail: 'stalled' })))
      .toBe('0 / 412 · stalled');
  });

  test('failures are surfaced even when the step is still claimable', () => {
    expect(BF.stepCaption(step('quality', 'needs_queue',
      { total: 500, eligible: 500, done: 470, remaining: 30, failed: 8 })))
      .toBe('470 / 500 · 8 failed');
  });

  test('a missing step is not a crash', () => {
    expect(BF.stepCaption(null)).toBe('');
  });
});

// =========================================================================
// placeholder — what the diagram area shows when it is not a diagram
// =========================================================================

describe('placeholder', () => {
  const view = (over) => Object.assign({
    selected: 1, listLoaded: true, hasBatches: true,
    state: null, err: null, computing: false,
  }, over || {});

  test('an error before any state has loaded is SHOWN, not swallowed', () => {
    // The regression this function exists for: the error used to be rendered
    // only inside the branch that already had a `state`, so a batch whose
    // first fetch 404'd polled forever showing "Loading...", silently.
    const out = BF.placeholder(view({ err: 'HTTP 404' }));
    expect(out.kind).toBe('error');
    expect(out.error).toContain('HTTP 404');
  });

  test('a hard error never shows a loading line beside it', () => {
    const out = BF.placeholder(view({ err: 'HTTP 500', computing: true }));
    expect(out.kind).toBe('error');
    expect(out.text).toBe('');
  });

  test('an error alongside a good diagram still surfaces', () => {
    // A poll that starts failing must not be hidden by the last good picture.
    const out = BF.placeholder(view({ state: state(), err: 'NetworkError' }));
    expect(out.kind).toBe('diagram');
    expect(out.error).toContain('NetworkError');
  });

  test('a healthy diagram carries no error', () => {
    const out = BF.placeholder(view({ state: state() }));
    expect(out.kind).toBe('diagram');
    expect(out.error).toBe('');
    expect(out.text).toBe('');
  });

  test('mid-derivation with nothing cached says so', () => {
    expect(BF.placeholder(view({ computing: true })).kind).toBe('computing');
    expect(BF.placeholder(view({ computing: true })).text)
      .toBe('Computing batch state…');
  });

  test('a selected batch with nothing yet is loading', () => {
    expect(BF.placeholder(view()).kind).toBe('loading');
  });

  test('a loaded, empty library explains itself', () => {
    const out = BF.placeholder(view({ selected: null, hasBatches: false }));
    expect(out.kind).toBe('empty');
    expect(out.text).toContain('No ingest batches yet');
  });

  test('no selection before the list arrives is still just loading', () => {
    const out = BF.placeholder(view({ selected: null, listLoaded: false, hasBatches: false }));
    expect(out.kind).toBe('loading');
  });

  test('no view at all is survivable', () => {
    expect(BF.placeholder(null).kind).toBe('loading');
  });
});

// =========================================================================
// small helpers the page shares
// =========================================================================

describe('stepLabel', () => {
  test('underscored NAS steps read as words, pass names stay verbatim', () => {
    expect(BF.stepLabel('normalize_aesthetics')).toBe('normalize aesthetics');
    expect(BF.stepLabel('rank_measure')).toBe('rank measure');
    expect(BF.stepLabel('category-content')).toBe('category-content');
  });
});

describe('dayOf', () => {
  test('pulls the date off a dated ingest folder', () => {
    expect(BF.dayOf('2091/2091-09-19_ILCE-7RM6')).toBe('2091-09-19');
    expect(BF.dayOf('2091/2091-09-19_phone-matt')).toBe('2091-09-19');
  });

  test('is null when the folder is not dated', () => {
    expect(BF.dayOf('_undated/phone-matt')).toBe(null);
    expect(BF.dayOf('2091/holiday')).toBe(null);
    expect(BF.dayOf(null)).toBe(null);
  });
});

describe('fmt', () => {
  test('groups thousands and survives nothing', () => {
    expect(BF.fmt(1143)).toBe('1,143');
    expect(BF.fmt(0)).toBe('0');
    expect(BF.fmt(null)).toBe('0');
    expect(BF.fmt(1234567)).toBe('1,234,567');
  });
});

describe('module shape', () => {
  test('publishes itself on window.PS as well as CommonJS', () => {
    // The pages load it as a plain <script> after shared.js.
    expect(window.PS.BatchFlow).toBe(BF);
  });
});
