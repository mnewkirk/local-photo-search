/**
 * Tests for split-geometry.js — the multi-panel print math.
 *
 * The handoff spec supplies a worked example that doubles as the regression
 * test, so it anchors most of what follows: source 6528 x 4352 (3:2), six
 * 13x19" portrait sheets as 3 cols x 2 rows, gap 0.5", window mode ->
 * 40" x 38.5", 113 dpi, 31% crop, and 17,326 x 11,552 px to reach 300 dpi.
 */

const G = require('../dist/split-geometry.js');

// The worked example's source image.
const IMG = { w: 6528, h: 4352 };
const CAND = { pw: 13, ph: 19, cols: 3, rows: 2 };
const OPTS = { gutter: 0.5, mode: 'window' };

describe('plan — the worked example', () => {
  const P = G.plan(IMG, CAND, OPTS);

  test('wall footprint is edge to edge including gaps', () => {
    expect(P.outerW).toBeCloseTo(40, 6);      // 3*13 + 2*0.5
    expect(P.outerH).toBeCloseTo(38.5, 6);    // 2*19 + 1*0.5
  });

  test('window mode puts the gaps inside the image field', () => {
    expect(P.cW).toBeCloseTo(40, 6);
    expect(P.cH).toBeCloseTo(38.5, 6);
  });

  test('delivers 113 dpi', () => {
    expect(Math.round(P.dpi)).toBe(113);
  });

  test('costs 31% of the frame', () => {
    expect(Math.round((1 - P.keep) * 100)).toBe(31);
  });

  test('height binds, and the crop uses it in full', () => {
    expect(P.bindingAxis).toBe('height');
    expect(P.ch).toBeCloseTo(4352, 6);
    expect(Math.round(P.cw)).toBe(4522);
  });
});

describe('plan — invariants', () => {
  test('at least one slack is zero, by construction', () => {
    // The handoff prose says "exactly one", but when the source aspect equals
    // the field aspect both are zero — nothing is trimmed at all. So the real
    // invariant is "at least one", covered exactly by the next test.
    G.SIZES.forEach(([a, b]) => {
      G.GRIDS.forEach(([cols, rows]) => {
        const P = G.plan(IMG, { pw: a, ph: b, cols, rows }, OPTS);
        const zeroes = [P.slackX, P.slackY].filter((s) => Math.abs(s) < 1e-9);
        expect(zeroes.length).toBeGreaterThanOrEqual(1);
      });
    });
  });

  test('a source matching the field aspect exactly is not cropped at all', () => {
    // 2 x 1 sheets of 10x8 landscape, no gap -> a 20:8 field. A 20:8 source
    // fits it perfectly, so both slacks vanish and keep is 1.
    const P = G.plan({ w: 2000, h: 800 }, { pw: 10, ph: 8, cols: 2, rows: 1 },
                     { gutter: 0, mode: 'window' });
    expect(P.slackX).toBeCloseTo(0, 9);
    expect(P.slackY).toBeCloseTo(0, 9);
    expect(P.keep).toBeCloseTo(1, 9);
  });

  test('the binding axis agrees with which slack is zero', () => {
    // These two must never drift apart — an inverted comparison here once told
    // users their width was the constraint when it was the height.
    G.SIZES.forEach(([a, b]) => {
      G.GRIDS.forEach(([cols, rows]) => {
        const P = G.plan(IMG, { pw: a, ph: b, cols, rows }, OPTS);
        if (P.bindingAxis === 'height') expect(P.slackY).toBeCloseTo(0, 9);
        else expect(P.slackX).toBeCloseTo(0, 9);
      });
    });
  });

  test('keep is scale-free — it depends only on the two aspect ratios', () => {
    // Same aspect at three very different pixel counts. (Exact multiples: an
    // approximate one like 653x435 is a *different* aspect and does differ.)
    const small = G.plan({ w: 3264, h: 2176 }, CAND, OPTS);
    const huge = G.plan({ w: 65280, h: 43520 }, CAND, OPTS);
    const base = G.plan(IMG, CAND, OPTS);
    expect(small.keep).toBeCloseTo(base.keep, 9);
    expect(huge.keep).toBeCloseTo(base.keep, 9);
  });

  test('crop cost is a shape mismatch — resolution never fixes it', () => {
    const base = G.plan(IMG, CAND, OPTS);
    const upscaled = G.plan({ w: IMG.w * 4, h: IMG.h * 4 }, CAND, OPTS);
    expect(upscaled.keep).toBeCloseTo(base.keep, 9);   // unchanged
    expect(upscaled.dpi).toBeCloseTo(base.dpi * 4, 6); // but dpi scales
  });

  test('keep collapses to the aspect ratio quotient', () => {
    const P = G.plan(IMG, CAND, OPTS);
    const ta = P.cW / P.cH, sa = IMG.w / IMG.h;
    expect(P.keep).toBeCloseTo(sa > ta ? ta / sa : sa / ta, 9);
  });

  test('continue mode excludes the gaps from the image field', () => {
    const P = G.plan(IMG, CAND, { gutter: 0.5, mode: 'continue' });
    expect(P.cW).toBeCloseTo(39, 6);     // 3*13, no gaps
    expect(P.cH).toBeCloseTo(38, 6);     // 2*19, no gaps
    expect(P.outerW).toBeCloseTo(40, 6); // footprint still includes them
  });

  test('a zero gap makes the two modes identical', () => {
    const w = G.plan(IMG, CAND, { gutter: 0, mode: 'window' });
    const c = G.plan(IMG, CAND, { gutter: 0, mode: 'continue' });
    expect(w.dpi).toBeCloseTo(c.dpi, 9);
    expect(w.keep).toBeCloseTo(c.keep, 9);
  });
});

describe('panelRects', () => {
  test('emits one rect per panel, in row-major order', () => {
    const { rects } = G.panelRects(IMG, CAND, OPTS);
    expect(rects.length).toBe(6);
    expect(rects.map((r) => [r.c, r.r])).toEqual([
      [0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1],
    ]);
  });

  test('panels tile the crop without overlapping', () => {
    const { rects } = G.panelRects(IMG, CAND, OPTS);
    const P = G.plan(IMG, CAND, OPTS);
    // Adjacent columns are one stride apart, and the stride exceeds the panel
    // width by exactly the gap's worth of pixels (window mode).
    const stride = rects[1].sx - rects[0].sx;
    expect(stride).toBeCloseTo(P.stepX * P.cw, 6);
    expect(stride).toBeGreaterThan(rects[0].sw);
  });

  test('every panel rect stays inside the source image', () => {
    const { rects } = G.panelRects(IMG, CAND, OPTS);
    rects.forEach((r) => {
      expect(r.sx).toBeGreaterThanOrEqual(-1e-6);
      expect(r.sy).toBeGreaterThanOrEqual(-1e-6);
      expect(r.sx + r.sw).toBeLessThanOrEqual(IMG.w + 1e-6);
      expect(r.sy + r.sh).toBeLessThanOrEqual(IMG.h + 1e-6);
    });
  });

  test('panels are all the same size', () => {
    const { rects } = G.panelRects(IMG, CAND, OPTS);
    rects.forEach((r) => {
      expect(r.sw).toBeCloseTo(rects[0].sw, 6);
      expect(r.sh).toBeCloseTo(rects[0].sh, 6);
    });
  });

  test('pan shifts the crop along the slack axis only', () => {
    const left = G.panelRects(IMG, CAND, { ...OPTS, sx: -1 }).rects[0];
    const right = G.panelRects(IMG, CAND, { ...OPTS, sx: 1 }).rects[0];
    const P = G.plan(IMG, CAND, OPTS);
    expect(right.sx - left.sx).toBeCloseTo(P.slackX, 6);
    expect(right.sy).toBeCloseTo(left.sy, 6);   // height binds: no vertical play
  });
});

describe('needPx — the worked example', () => {
  const P = G.plan(IMG, CAND, OPTS);
  const n = G.needPx(IMG, P, 300);

  test('requires 17,326 x 11,552 px', () => {
    expect(n.w).toBe(17326);
    expect(n.h).toBe(11552);
  });

  test('reports 200.1 MP and a 2.7x factor', () => {
    expect(n.mp).toBe(200.1);
    expect(n.k).toBe(2.7);
  });

  test('always rounds up to even pixels', () => {
    expect(n.w % 2).toBe(0);
    expect(n.h % 2).toBe(0);
  });

  test('asking for the dpi already achieved is a no-op scale', () => {
    const same = G.needPx(IMG, P, P.dpi);
    expect(same.k).toBe(1);
    expect(same.w).toBeGreaterThanOrEqual(IMG.w);
  });
});

describe('candidates', () => {
  test('returns nothing without an image', () => {
    expect(G.candidates(null, {})).toEqual([]);
  });

  test('every survivor clears both quality gates', () => {
    const o = { minDpi: 150, maxCrop: 0.3, wall: 65 };
    G.candidates(IMG, o).forEach((r) => {
      expect(r.P.dpi).toBeGreaterThanOrEqual(150);
      expect(1 - r.P.keep).toBeLessThanOrEqual(0.3 + 1e-4);
    });
  });

  test('honors the wall limit when wallOnly is on', () => {
    const o = { minDpi: 100, maxCrop: 1, wall: 30, wallOnly: true };
    G.candidates(IMG, o).forEach((r) => {
      expect(r.P.outerW).toBeLessThanOrEqual(30);
      expect(r.P.outerH).toBeLessThanOrEqual(30);
    });
  });

  test('honors the panel-count filter', () => {
    const o = { minDpi: 100, maxCrop: 1, count: 4, wallOnly: false };
    const got = G.candidates(IMG, o);
    expect(got.length).toBeGreaterThan(0);
    got.forEach((r) => expect(r.cand.cols * r.cand.rows).toBe(4));
  });

  test('honors the sheet-size filter', () => {
    const o = { minDpi: 100, maxCrop: 1, sizes: ['13×19'], wallOnly: false };
    G.candidates(IMG, o).forEach((r) => {
      expect([r.P.pw, r.P.ph].sort((a, b) => a - b)).toEqual([13, 19]);
    });
  });

  test('is ranked by score, highest first', () => {
    const got = G.candidates(IMG, { minDpi: 100, maxCrop: 0.5, wallOnly: false });
    const scores = got.map((r) => r.score);
    expect([...scores].sort((a, b) => b - a)).toEqual(scores);
  });

  test('the top pick is Pareto-optimal on reach and crop', () => {
    // The goal is filling a wall, so reach leads — but crop is still weighed
    // (1000 vs 600), which means the largest piece does not always win. What
    // must hold is that nothing beats the top on BOTH axes at once.
    const got = G.candidates(IMG, { minDpi: 100, maxCrop: 0.5, wall: 65 });
    const top = got[0];
    const span = (r) => Math.max(r.P.outerW, r.P.outerH);
    got.slice(1).forEach((r) => {
      const better = span(r) > span(top) + 1e-9 && r.P.keep > top.P.keep + 1e-9;
      expect(better).toBe(false);
    });
  });

  test('with crop held equal, the larger piece ranks higher', () => {
    const got = G.candidates(IMG, { minDpi: 100, maxCrop: 0.5, wall: 65 });
    for (let i = 0; i < got.length - 1; i++) {
      for (let j = i + 1; j < got.length; j++) {
        if (Math.abs(got[i].P.keep - got[j].P.keep) < 1e-9) {
          expect(Math.max(got[i].P.outerW, got[i].P.outerH))
            .toBeGreaterThanOrEqual(Math.max(got[j].P.outerW, got[j].P.outerH) - 1e-9);
        }
      }
    }
  });

  test('caps the list at 16', () => {
    const got = G.candidates(IMG, { minDpi: 100, maxCrop: 1, wallOnly: false });
    expect(got.length).toBeLessThanOrEqual(16);
  });

  test('nothing ever exceeds the 10-foot ceiling', () => {
    G.candidates(IMG, { minDpi: 100, maxCrop: 1, wallOnly: false }).forEach((r) => {
      expect(r.P.outerW).toBeLessThanOrEqual(G.MAX_SPAN_IN);
      expect(r.P.outerH).toBeLessThanOrEqual(G.MAX_SPAN_IN);
    });
  });
});

describe('reachable', () => {
  test('ignores the dpi and crop gates but keeps the physical ones', () => {
    const o = { minDpi: 300, maxCrop: 0.05, wall: 65, wallOnly: true };
    const reach = G.reachable(IMG, o);
    const cands = G.candidates(IMG, o);
    expect(reach.length).toBeGreaterThan(cands.length);
    reach.forEach((r) => {
      expect(r.P.outerW).toBeLessThanOrEqual(65);
      expect(r.P.outerH).toBeLessThanOrEqual(65);
    });
  });

  test('is empty when no sheet size is enabled', () => {
    expect(G.reachable(IMG, { sizes: [] })).toEqual([]);
  });
});

describe('forGrid — trying an arbitrary grid', () => {
  test('serves grids the curated GRIDS list does not contain', () => {
    // 3x3 is not in GRIDS, so candidates/reachable can never surface it —
    // which is exactly why trying a grid needs its own enumerator.
    expect(G.GRIDS.some(([c, r]) => c === 3 && r === 3)).toBe(false);
    expect(G.reachable(IMG, { wallOnly: false }).some(
      (x) => x.cand.cols === 3 && x.cand.rows === 3)).toBe(false);

    const got = G.forGrid(IMG, 3, 3, { wallOnly: false });
    expect(got.length).toBeGreaterThan(0);
    got.forEach((x) => {
      expect(x.cand.cols).toBe(3);
      expect(x.cand.rows).toBe(3);
    });
  });

  test('ignores the dpi and crop floors — seeing why is the point', () => {
    const got = G.forGrid(IMG, 6, 6, { minDpi: 300, maxCrop: 0.01 });
    // Any survivor is there on physical grounds alone.
    got.forEach((x) => {
      expect(x.P.outerW).toBeLessThanOrEqual(G.MAX_SPAN_IN);
      expect(x.P.outerH).toBeLessThanOrEqual(G.MAX_SPAN_IN);
    });
  });

  test('still honors the 10-foot ceiling, so absurd grids return nothing', () => {
    expect(G.forGrid(IMG, 20, 20, {})).toEqual([]);
  });

  test('prefers a wall-fitting option, then the least crop', () => {
    const got = G.forGrid(IMG, 2, 2, { wall: 65 });
    expect(got.length).toBeGreaterThan(1);
    const fits = got.map((x) => x.fits);
    // All the fitting ones come first.
    expect(fits.slice(0, fits.filter(Boolean).length).every(Boolean)).toBe(true);
    const fitting = got.filter((x) => x.fits);
    for (let i = 0; i < fitting.length - 1; i++) {
      expect(fitting[i].P.keep).toBeGreaterThanOrEqual(fitting[i + 1].P.keep - 1e-9);
    }
  });

  test('returns nothing for a nonsense grid or missing image', () => {
    expect(G.forGrid(IMG, 0, 3, {})).toEqual([]);
    expect(G.forGrid(null, 3, 3, {})).toEqual([]);
  });
});

describe('diagnose', () => {
  test('branch 1 — nothing selected at all', () => {
    const d = G.diagnose(IMG, { sizes: [] });
    expect(d.branch).toBe('empty');
    expect(d.title).toBe('Nothing to arrange');
    expect(d.fix).toBeNull();
  });

  test('branch 2 — crop passes, resolution blocks', () => {
    // The spec's own example: 300 dpi floor, 6 panels, 50% crop allowance.
    const d = G.diagnose(IMG, {
      minDpi: 300, maxCrop: 0.5, count: 6, sizes: ['13×19'], wallOnly: false,
    });
    expect(d.branch).toBe('dpi');
    expect(d.title).toBe('Nothing clears 300 dpi');
    expect(d.body).toContain('Crop is not the problem');
    expect(d.body).toContain('height');            // names the binding axis
    expect(d.body).toContain('4,522 × 4,352 px');  // cites the CROP dims
    expect(d.body).toContain('17,326 × 11,552 px'); // states required source
    expect(d.fix.kind).toBe('minDpi');
  });

  test('branch 2 fix names a preset at or below what was achieved', () => {
    const d = G.diagnose(IMG, {
      minDpi: 300, maxCrop: 0.5, count: 6, sizes: ['13×19'], wallOnly: false,
    });
    expect(d.fix.minDpi).toBe(100);   // achieved 113 -> highest preset <= 113
  });

  test('branch 3 — crop blocks', () => {
    const d = G.diagnose(IMG, {
      minDpi: 100, maxCrop: 0.05, count: 6, sizes: ['13×19'], wallOnly: false,
    });
    expect(d.branch).toBe('crop');
    expect(d.title).toContain('Nothing fits inside a 5% crop');
    expect(d.body).toContain('cannot agree');
    expect(d.fix.kind).toBe('cropDpi');
    expect(d.fix.maxCrop).toBeGreaterThanOrEqual(0.3);
  });

  test('branch 3 quotes the crop the shape needs, not the user setting', () => {
    // Quoting the user's allowance yields "with up to 30% crop" immediately
    // after "costing 31%" — a self-contradicting sentence.
    const d = G.diagnose(IMG, {
      minDpi: 300, maxCrop: 0.3, count: 6, sizes: ['13×19'], wallOnly: false,
    });
    expect(d.branch).toBe('crop');
    expect(d.body).toContain('at the 31% crop this shape needs');
    expect(d.body).not.toContain('with up to 30% crop');
  });

  test('branch 3 names the arrangement instead of "undefined × undefined"', () => {
    // reachable() rows carry the grid on .cand only; reading cols/rows off the
    // top level (which only candidates() sets) printed undefined here.
    const d = G.diagnose(IMG, {
      minDpi: 300, maxCrop: 0.3, count: 6, sizes: ['13×19'], wallOnly: false,
    });
    expect(d.body).not.toContain('undefined');
    expect(d.body).toMatch(/The closest is \d+ × \d+ (portrait|landscape)/);
  });

  test('branch 3 quotes the field aspect to two decimals', () => {
    // 40/38.5 is 1.04:1; one decimal rounds it to a meaningless "1:1" right
    // next to the source's "1.5:1", which reads as though they agree.
    const d = G.diagnose(IMG, {
      minDpi: 300, maxCrop: 0.3, count: 6, sizes: ['13×19'], wallOnly: false,
    });
    expect(d.body).toContain('1.04:1 field');
    expect(d.body).toContain('1.5:1');
  });

  test('crop presets match the prototype', () => {
    // A 31% requirement must land on 50%, per the spec's worked example.
    expect(G.CROPS).toEqual([0.05, 0.15, 0.3, 0.5, 1]);
    const d = G.diagnose(IMG, {
      minDpi: 300, maxCrop: 0.3, count: 6, sizes: ['13×19'], wallOnly: false,
    });
    expect(d.fix.maxCrop).toBe(0.5);
    expect(d.body).toContain('set the crop allowance to 50%');
  });

  test('branch 3 stays silent about resolution when crop alone is at fault', () => {
    const d = G.diagnose(IMG, {
      minDpi: 100, maxCrop: 0.05, count: 6, sizes: ['13×19'], wallOnly: false,
    });
    expect(d.body).not.toContain('would have to be about');
  });

  test('a dead end is named as one and gets no hopeful button', () => {
    // A tiny source can't clear even the lowest floor, so there is no dpi
    // preset to offer — the app must say so instead of dangling a fix.
    const d = G.diagnose({ w: 640, h: 427 }, {
      minDpi: 300, maxCrop: 0.6, count: 6, sizes: ['13×19'], wallOnly: false,
    });
    expect(d.branch).toBe('dpi-deadend');
    expect(d.body).toContain('cannot work with this file');
    expect(d.fix === null || d.fix.kind === 'select').toBe(true);
  });

  test('every fix actually produces results when applied', () => {
    // The governing rule: a fix button that yields nothing is worse than none.
    const cases = [
      { minDpi: 300, maxCrop: 0.5, count: 6, sizes: ['13×19'], wallOnly: false },
      { minDpi: 100, maxCrop: 0.05, count: 6, sizes: ['13×19'], wallOnly: false },
      { minDpi: 240, maxCrop: 0.1, wallOnly: true, wall: 65 },
      { minDpi: 300, maxCrop: 0.2, count: 4, wallOnly: false },
    ];
    const empty = cases.filter((o) => G.candidates(IMG, o).length === 0);
    expect(empty.length).toBeGreaterThan(0);   // the fixture must exercise it
    empty.forEach((o) => {
      const d = G.diagnose(IMG, o);
      if (!d.fix) return;
      const next = { ...o };
      if (d.fix.kind === 'minDpi') next.minDpi = d.fix.minDpi;
      if (d.fix.kind === 'cropDpi') {
        next.maxCrop = d.fix.maxCrop;
        if (d.fix.minDpi != null) next.minDpi = d.fix.minDpi;
      }
      if (d.fix.kind === 'select') return;   // offers an arrangement, not a setting
      expect(G.candidates(IMG, next).length).toBeGreaterThan(0);
    });
  });
});

describe('upscale integration', () => {
  test('asks for the factor actually needed, not the next rung', () => {
    // Topaz takes arbitrary decimals, so a plan needing 1.3x should ask for
    // 1.3x rather than 2x — the latter is 2.4x the pixels, all of them
    // carrying synthesized texture the plan never wanted.
    expect(G.upscaleFor({ k: 1.3 })).toBe(1.3);
    expect(G.upscaleFor({ k: 2 })).toBe(2);
    expect(G.upscaleFor({ k: 2.7 })).toBe(2.7);
  });

  test('rounds up to a tenth so the result clears the floor', () => {
    expect(G.upscaleFor({ k: 1.21 })).toBe(1.3);
    expect(G.upscaleFor({ k: 1.999 })).toBe(2);
  });

  test('returns null when the source is already big enough', () => {
    expect(G.upscaleFor({ k: 1 })).toBeNull();
    expect(G.upscaleFor({ k: 0.6 })).toBeNull();
  });

  test('returns null past the ceiling, rather than a useless offer', () => {
    expect(G.upscaleFor({ k: 9 })).toBeNull();
    expect(G.upscaleFor({ k: G.MAX_UPSCALE })).toBe(G.MAX_UPSCALE);
  });

  test('the worked example needs 2.7x to reach 300 dpi', () => {
    const P = G.plan(IMG, CAND, OPTS);
    expect(G.upscaleFor(G.needPx(IMG, P, 300))).toBe(2.7);
  });

  test('an upscaled source actually clears the floor that blocked it', () => {
    const o = { minDpi: 300, maxCrop: 0.5, count: 6, sizes: ['13×19'], wallOnly: false };
    expect(G.candidates(IMG, o).length).toBe(0);
    const d = G.diagnose(IMG, o);
    const scale = G.upscaleFor(d.upscale);
    const bigger = { w: Math.round(IMG.w * scale), h: Math.round(IMG.h * scale) };
    expect(G.candidates(bigger, o).length).toBeGreaterThan(0);
    // ...and not wastefully: one tenth less would not have been enough.
    const short = { w: Math.round(IMG.w * (scale - 0.1)),
                    h: Math.round(IMG.h * (scale - 0.1)) };
    expect(G.candidates(short, o).length).toBe(0);
  });
});

describe('reachAtDpi', () => {
  test('reports the long-edge reach, rounded down to the half inch', () => {
    expect(G.reachAtDpi(IMG, 240)).toBe(27);
    expect(G.reachAtDpi(IMG, 120)).toBe(54);
  });
});
