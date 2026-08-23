/**
 * Split — multi-panel print geometry.
 *
 * A pure, framework-free port of the planner prototype's math. No DOM, no
 * React, no state: every function takes the image and the settings explicitly
 * so the whole thing is unit-testable (see __tests__/split-geometry.test.js).
 *
 * The governing constraint: every panel is one whole borderless sheet. Nothing
 * is ever cut, so panel sizes are the printer's stock sizes and the piece's
 * dimensions are whatever those sheets add up to, plus the wall gaps.
 *
 * Loads as a browser global (window.SplitGeom) or a CommonJS module.
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.SplitGeom = api;
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // Paper stock in inches, short edge first. Borderless sizes for an
  // Epson XP-15000; a different printer means a different table.
  var SIZES = [[4, 6], [5, 7], [8, 10], [8.5, 11], [11, 14], [11, 17], [13, 19]];

  // [cols, rows]. Both orientations of each shape are listed explicitly.
  var GRIDS = [[2, 1], [1, 2], [3, 1], [1, 3], [4, 1], [1, 4], [5, 1], [1, 5],
               [6, 1], [2, 2], [3, 2], [2, 3], [4, 3], [3, 4]];

  // Panel counts the curated GRIDS list can actually produce. Derived, not
  // hand-written: the UI used to offer 8, which NO grid makes, so picking it
  // was a guaranteed dead end.
  var PANEL_COUNTS = GRIDS.map(function (g) { return g[0] * g[1]; })
    .filter(function (n, i, a) { return a.indexOf(n) === i; })
    .sort(function (a, b) { return a - b; });

  var DPIS = [100, 120, 150, 200, 240, 300];
  var CROPS = [0.05, 0.15, 0.3, 0.5, 1];

  // A 10-foot ceiling. Past this the arrangement is not a wall piece.
  var MAX_SPAN_IN = 120;
  // Topaz's ceiling for the Autopilot factor.
  var MAX_UPSCALE = 6;
  // Float slop when comparing a computed crop against the user's allowance.
  var CROP_EPS = 0.0001;

  var DEFAULTS = {
    gutter: 0.5,        // wall gap between sheets, inches
    mode: 'window',     // 'window' = gaps are part of the image, 'continue' = not
    minDpi: 240,
    maxCrop: 0.3,
    wall: 65,           // display wall, inches square
    wallOnly: true,     // drop arrangements that overhang the wall
    sizes: null,        // null = all stock sizes; else array of labels
    count: 'any',       // 'any' or a panel count
    sx: 0, sy: 0,       // pan, each in [-1, 1]
  };

  function opts(o) {
    var out = {}, k;
    for (k in DEFAULTS) out[k] = DEFAULTS[k];
    for (k in (o || {})) if (o[k] !== undefined) out[k] = o[k];
    return out;
  }

  function sizeLabel(a, b) { return a + '×' + b; }

  /**
   * The core. Everything else is built on this.
   *
   * `cand` is {pw, ph, cols, rows} in inches; the image is {w, h} in pixels.
   * Returns the full geometry of one arrangement.
   */
  function plan(img, cand, o) {
    var s = opts(o), g = s.gutter;
    var pw = cand.pw, ph = cand.ph, cols = cand.cols, rows = cand.rows;

    var outerW = cols * pw + (cols - 1) * g;   // wall footprint, edge to edge
    var outerH = rows * ph + (rows - 1) * g;

    // In 'window' mode the wall gaps are part of the image — the picture
    // continues behind them, as if seen through a window. In 'continue' mode
    // the gaps are not, so only the inked area carries image.
    var cW = s.mode === 'window' ? outerW : cols * pw;
    var cH = s.mode === 'window' ? outerH : rows * ph;

    var ta = cW / cH;            // target aspect of the image field
    var sa = img.w / img.h;      // source aspect

    // Exactly one axis is used in full; the other is trimmed. Derive the
    // binding axis from THIS comparison, never by comparing the slacks —
    // letting those two drift apart was a real bug in the prototype.
    var cw, ch, bindingAxis;
    if (sa > ta) { ch = img.h; cw = img.h * ta; bindingAxis = 'height'; }
    else { cw = img.w; ch = img.w / ta; bindingAxis = 'width'; }

    var slackX = img.w - cw, slackY = img.h - ch;

    return {
      outerW: outerW, outerH: outerH,
      cW: cW, cH: cH, cw: cw, ch: ch,
      ox: slackX * (s.sx + 1) / 2,      // crop origin from the pan control
      oy: slackY * (s.sy + 1) / 2,
      slackX: slackX, slackY: slackY, bindingAxis: bindingAxis,
      stepX: s.mode === 'window' ? (pw + g) / cW : pw / cW,
      stepY: s.mode === 'window' ? (ph + g) / cH : ph / cH,
      inkW: cols * pw, inkH: rows * ph,
      dpi: cw / cW,                     // pixels per inch actually delivered
      keep: (cw * ch) / (img.w * img.h),  // AREA fraction of the source retained
      pw: pw, ph: ph, cols: cols, rows: rows, g: g, mode: s.mode,
    };
  }

  /** Source-pixel rectangle per panel — the numbers an export writes. */
  function panelRects(img, cand, o) {
    var P = plan(img, cand, o), rects = [], r, c;
    for (r = 0; r < cand.rows; r++) {
      for (c = 0; c < cand.cols; c++) {
        rects.push({
          sx: P.ox + c * P.stepX * P.cw,
          sy: P.oy + r * P.stepY * P.ch,
          sw: (cand.pw / P.cW) * P.cw,
          sh: (cand.ph / P.cH) * P.ch,
          r: r, c: c,
        });
      }
    }
    return { rects: rects, dpi: P.dpi };
  }

  /**
   * What the source would have to be to hit `dpi` on this arrangement.
   *
   * `keep` is scale-free — it depends only on the two aspect ratios, never on
   * the pixel count — so the requirement is a uniform scale-up of the current
   * file by (target dpi / achieved dpi). This is what makes a Topaz upscale a
   * real fix rather than a guess.
   */
  function needPx(img, P, dpi) {
    var k = dpi / P.dpi;
    var w = Math.ceil(img.w * k / 2) * 2;   // even pixels
    var h = Math.ceil(img.h * k / 2) * 2;
    return { w: w, h: h, mp: Math.round(w * h / 1e5) / 10, k: Math.round(k * 10) / 10 };
  }

  /** Walk every stock size x orientation x grid, applying `filter`. */
  function enumerate(img, o, filter) {
    var s = opts(o), out = [];
    SIZES.forEach(function (pair) {
      var lab = sizeLabel(pair[0], pair[1]);
      if (s.sizes && s.sizes.indexOf(lab) < 0) return;
      var orients = pair[0] === pair[1]
        ? [pair] : [[pair[0], pair[1]], [pair[1], pair[0]]];
      orients.forEach(function (or) {
        GRIDS.forEach(function (gr) {
          var cols = gr[0], rows = gr[1];
          if (s.count !== 'any' && cols * rows !== Number(s.count)) return;
          var cand = { pw: or[0], ph: or[1], cols: cols, rows: rows };
          var P = plan(img, cand, s);
          if (s.wallOnly && (P.outerW > s.wall || P.outerH > s.wall)) return;
          if (P.outerW > MAX_SPAN_IN || P.outerH > MAX_SPAN_IN) return;
          var row = { cand: cand, P: P, label: lab };
          if (filter(row, s)) out.push(row);
        });
      });
    });
    return out;
  }

  function passesQuality(row, s) {
    return row.P.dpi >= s.minDpi && (1 - row.P.keep) <= s.maxCrop + CROP_EPS;
  }

  /**
   * Ranked arrangements that clear every gate.
   *
   * Ranking is deliberate: reach across the wall leads, because the user's goal
   * is to fill a wall. Resolution is pass/fail against their floor — pixels
   * beyond it earn nothing — and crop is a tolerance with the first 3% free.
   */
  function candidates(img, o) {
    if (!img) return [];
    var s = opts(o);
    var rows = enumerate(img, s, passesQuality);
    rows.forEach(function (row) {
      var P = row.P;
      var fits = P.outerW <= s.wall && P.outerH <= s.wall;
      var reach = Math.min(1, Math.max(P.outerW, P.outerH) / s.wall);
      var crop = Math.max(0, (1 - P.keep) - 0.03);
      row.fits = fits;
      row.score = reach * 1000 - crop * 600 + (fits ? 60 : -500);
      row.pw = P.pw; row.ph = P.ph; row.cols = P.cols; row.rows = P.rows;
    });
    rows.sort(function (x, y) { return y.score - x.score; });
    return rows.slice(0, 16);
  }

  /**
   * Everything physically buildable, ignoring the dpi and crop gates. Exists
   * solely to power the diagnostics: it is the set of arrangements that are
   * quality-blocked rather than impossible.
   */
  function reachable(img, o) {
    if (!img) return [];
    return enumerate(img, o, function () { return true; });
  }

  /**
   * Every sheet size that can build ONE given grid, best first.
   *
   * `candidates`/`reachable` only walk the curated GRIDS list, so a grid that
   * is not on it (3x3, say) is invisible to them. Trying an arbitrary grid is
   * exactly the case that list cannot serve, so this enumerates sheet sizes
   * directly for the cols/rows asked for.
   *
   * The dpi and crop floors are deliberately NOT enforced — seeing *why* a
   * grid falls short is the point of asking for it, and such rows are labelled
   * rather than hidden.
   *
   * **The sheet-size selection IS enforced.** Those two are different kinds of
   * constraint and must not be conflated: a floor is a preference you might
   * want to see violated, while "sheet sizes in play" is a hard fact about
   * which paper you own. Ignoring it once produced a suggested 4x2 on 8.5x11"
   * for someone who had only 13x19" switched on.
   */
  function forGrid(img, cols, rows, o) {
    if (!img || cols < 1 || rows < 1) return [];
    var s = opts(o), out = [];
    SIZES.forEach(function (pair) {
      if (s.sizes && s.sizes.indexOf(sizeLabel(pair[0], pair[1])) < 0) return;
      var orients = pair[0] === pair[1]
        ? [pair] : [[pair[0], pair[1]], [pair[1], pair[0]]];
      orients.forEach(function (or) {
        var cand = { pw: or[0], ph: or[1], cols: cols, rows: rows };
        var P = plan(img, cand, s);
        if (P.outerW > MAX_SPAN_IN || P.outerH > MAX_SPAN_IN) return;
        out.push({ cand: cand, P: P, label: sizeLabel(pair[0], pair[1]),
                   fits: P.outerW <= s.wall && P.outerH <= s.wall });
      });
    });
    // Prefer something that fits the wall, then the least crop, then the
    // largest piece — the same instincts as the main ranking.
    out.sort(function (a, b) {
      if (a.fits !== b.fits) return a.fits ? -1 : 1;
      if (Math.abs(a.P.keep - b.P.keep) > 1e-9) return b.P.keep - a.P.keep;
      return Math.max(b.P.outerW, b.P.outerH) - Math.max(a.P.outerW, a.P.outerH);
    });
    return out;
  }

  /**
   * Largest arrangement that genuinely clears the current floors, looking past
   * the panel-count and sheet-size filters — the "try this instead" answer.
   */
  function bestAlternative(img, o) {
    if (!img) return null;
    var s = opts(o);
    s.sizes = null; s.count = 'any';
    var rows = enumerate(img, s, passesQuality);
    rows.forEach(function (r) { r.reach = Math.max(r.P.outerW, r.P.outerH); });
    rows.sort(function (a, b) { return b.reach - a.reach; });
    return rows[0] || null;
  }

  // --- formatting helpers, shared by the UI and the diagnostics -------------

  function fmt(n) {
    return (Math.round(n * 10) / 10).toString().replace(/\.0$/, '');
  }
  function pct(x) { return Math.round(x * 100) + '%'; }
  function inches(x) { return fmt(x) + '″'; }
  function commas(n) { return Math.round(n).toLocaleString('en-US'); }

  // Reads the grid off `cand`, which every row carries — `candidates()` also
  // copies cols/rows to the top level, but `reachable()` rows do not, and
  // reaching for those produced "undefined × undefined" in the diagnostics.
  function describe(row) {
    var c = row.cand || row;
    return c.cols + ' × ' + c.rows + ' ' +
      (row.P.pw < row.P.ph ? 'portrait' : 'landscape');
  }

  /** Aspect ratio as "1.04:1" — one decimal rounds 40/38.5 to a bogus "1:1". */
  function ratio(x) { return (Math.round(x * 100) / 100) + ':1'; }

  /**
   * `crop` is the allowance to quote: pass null to quote the user's setting
   * (correct when crop already passes), or the crop this shape actually demands
   * (correct when crop is the blocker). Quoting the user's allowance in the
   * crop-blocked branch produces a self-contradicting sentence — "with up to
   * 30% crop" right after "costing 31%".
   */
  function needSentence(img, P, dpi, maxCrop, crop) {
    var n = needPx(img, P, dpi);
    var c = crop == null ? maxCrop : crop;
    var q = crop == null
      ? 'with up to ' + (c === 1 ? 'any' : pct(c)) + ' crop'
      : 'at the ' + pct(c) + ' crop this shape needs';
    return 'To hold ' + dpi + ' dpi here ' + q +
      ', the original would have to be about ' + commas(n.w) + ' × ' +
      commas(n.h) + ' px (' + n.mp + ' MP) — ' + fmt(n.k) +
      '× this file in each direction.';
  }

  /**
   * The empty state.
   *
   * Never a bare "no results": always name the single blocking constraint,
   * quantify it, state what the file would have to be, and offer one action
   * that actually resolves it. A fix button that produces nothing when pressed
   * is worse than no button.
   */
  function diagnose(img, o) {
    var s = opts(o);
    var reach = reachable(img, s);

    if (!reach.length) {
      // Empty has several causes and they need different answers. Blaming the
      // sheet sizes unconditionally was wrong in every case but the first —
      // it told people to turn on sizes that were already on.
      if (s.sizes && !s.sizes.length) {
        return {
          branch: 'empty-sizes',
          title: 'Nothing to arrange',
          body: 'No sheet size is selected, so there is nothing to arrange. ' +
                'Turn at least one back on.',
          fix: null,
        };
      }

      // Is that panel count achievable by any grid at all?
      if (s.count !== 'any' && PANEL_COUNTS.indexOf(Number(s.count)) < 0) {
        return {
          branch: 'empty-count',
          title: 'No arrangement uses exactly ' + s.count + ' panels',
          body: 'The grids on offer come to ' + PANEL_COUNTS.join(', ') +
                ' panels. ' + s.count + ' is not among them, so nothing can ' +
                'match. Clear the panel-count filter, or use the ' +
                'cols × rows boxes to try that grid directly.',
          fix: { kind: 'count', label: 'Clear the panel-count filter', count: 'any' },
        };
      }

      // The count is buildable in principle, so the sheet/wall filters are
      // what emptied it. Say which, and offer the one that actually helps.
      var loose = enumerate(img, Object.assign({}, s, {
        sizes: null, wallOnly: false }), function () { return true; });
      if (loose.length) {
        var withSizes = enumerate(img, Object.assign({}, s, { wallOnly: false }),
                                  function () { return true; });
        if (!withSizes.length) {
          return {
            branch: 'empty-sizes-filter',
            title: 'Nothing to arrange',
            body: 'No arrangement of ' + s.count + ' panels uses the sheet ' +
                  'sizes you have left on. Turn more sizes back on.',
            fix: { kind: 'sizes', label: 'Use every sheet size', sizes: null },
          };
        }
        var biggest = withSizes.slice().sort(function (a, b) {
          return Math.min(a.P.outerW, a.P.outerH) - Math.min(b.P.outerW, b.P.outerH);
        })[0];
        return {
          branch: 'empty-wall',
          title: 'Nothing fits your ' + fmt(s.wall) + '″ wall',
          body: 'These sheets can make ' + s.count + ' panels, but the ' +
                'smallest arrangement is ' + inches(biggest.P.outerW) + ' × ' +
                inches(biggest.P.outerH) + ' — larger than your wall. Allow ' +
                'pieces that overhang, or use smaller sheets.',
          fix: { kind: 'wallOnly', label: 'Show pieces bigger than my wall',
                 wallOnly: false },
        };
      }

      return {
        branch: 'empty-huge',
        title: 'Nothing to arrange',
        body: 'No sheet size builds ' + s.count + ' panels under ' +
              MAX_SPAN_IN + '″. Try a smaller panel count.',
        fix: { kind: 'count', label: 'Clear the panel-count filter', count: 'any' },
      };
    }

    var cropOk = reach.filter(function (r) {
      return (1 - r.P.keep) <= s.maxCrop + CROP_EPS;
    });

    if (cropOk.length) {
      // Crop passes; resolution is what blocks. Quote the highest-dpi
      // crop-passing arrangement — the closest thing to working.
      var best = cropOk.slice().sort(function (a, b) { return b.P.dpi - a.P.dpi; })[0];
      var P = best.P;
      var n = best.cand.cols * best.cand.rows;
      var preset = null, i;
      for (i = DPIS.length - 1; i >= 0; i--) {
        if (DPIS[i] <= Math.floor(P.dpi)) { preset = DPIS[i]; break; }
      }
      var head = n + ' × ' + sizeLabel(P.pw, P.ph) + '″ ' +
        (P.pw < P.ph ? 'portrait' : 'landscape') + ' spans ' +
        inches(P.outerW) + ' × ' + inches(P.outerH) + '. Its shape leaves you ' +
        commas(P.cw) + ' × ' + commas(P.ch) + ' px to spread across that (your ' +
        P.bindingAxis + ' is what binds, not the full ' + commas(img.w) + ' × ' +
        commas(img.h) + '), so it lands at ' + Math.round(P.dpi) + ' dpi. ';

      if (preset == null) {
        // A dead end is named as a dead end — it never gets a hopeful button.
        var alt = bestAlternative(img, s);
        return {
          branch: 'dpi-deadend',
          title: 'Nothing clears ' + s.minDpi + ' dpi',
          body: 'Crop is not the problem — resolution is. ' + head +
                needSentence(img, P, s.minDpi, s.maxCrop, null) +
                ' That is below the lowest floor on offer, so this combination ' +
                'cannot work with this file.',
          alternative: alt,
          fix: alt ? {
            kind: 'select',
            label: 'Show ' + describe(alt) + ' instead',
            cand: alt.cand,
          } : null,
          upscale: needPx(img, P, s.minDpi),
        };
      }

      return {
        branch: 'dpi',
        title: 'Nothing clears ' + s.minDpi + ' dpi',
        body: 'Crop is not the problem — resolution is. ' + head +
              needSentence(img, P, s.minDpi, s.maxCrop, null) +
              ' Set the floor to ' + preset + ' dpi or lower and it appears.',
        fix: { kind: 'minDpi', label: 'Set floor to ' + preset + ' dpi', minDpi: preset },
        upscale: needPx(img, P, s.minDpi),
      };
    }

    // Crop is the blocker: nothing is inside the allowance. Quote the least-crop
    // arrangement, tie-broken by dpi.
    var closest = reach.slice().sort(function (a, b) {
      var d = (1 - b.P.keep) - (1 - a.P.keep);
      return d !== 0 ? -d : b.P.dpi - a.P.dpi;
    })[0];
    var CP = closest.P;
    var need = 1 - CP.keep;
    var cropPreset = null;
    for (var j = 0; j < CROPS.length; j++) {
      if (CROPS[j] >= need - CROP_EPS) { cropPreset = CROPS[j]; break; }
    }
    var dpiPreset = null;
    for (var m = DPIS.length - 1; m >= 0; m--) {
      if (DPIS[m] <= Math.floor(CP.dpi)) { dpiPreset = DPIS[m]; break; }
    }
    var nPanels = closest.cand.cols * closest.cand.rows;
    var body = nPanels + ' sheets of ' + sizeLabel(CP.pw, CP.ph) + '″ make a ' +
      ratio(CP.cW / CP.cH) + ' field, and your photograph is ' +
      ratio(img.w / img.h) + ' — they cannot agree. The closest is ' +
      describe(closest) + ' at ' + inches(CP.outerW) + ' × ' + inches(CP.outerH) +
      ', costing ' + pct(need) + ' of the frame and landing at ' +
      Math.round(CP.dpi) + ' dpi. ';

    // The required-source sentence appears ONLY when resolution co-blocks.
    // When crop alone is at fault, resolution is not mentioned.
    var dpiAlsoBlocks = CP.dpi < s.minDpi;
    if (dpiAlsoBlocks) body += needSentence(img, CP, s.minDpi, s.maxCrop, need) + ' ';

    var parts = [];
    if (cropPreset != null) parts.push('set the crop allowance to ' + pct(cropPreset));
    if (dpiAlsoBlocks && dpiPreset != null) parts.push('the resolution floor to ' + dpiPreset + ' dpi');
    if (parts.length) body += 'To allow it, ' + parts.join(' and ') + '.';

    return {
      branch: 'crop',
      title: 'Nothing fits inside a ' + pct(s.maxCrop) + ' crop',
      body: body,
      fix: cropPreset != null ? {
        kind: 'cropDpi',
        label: 'Allow ' + pct(cropPreset) + ' crop' +
               (dpiAlsoBlocks && dpiPreset != null ? ' · ' + dpiPreset + ' dpi' : ''),
        maxCrop: cropPreset,
        minDpi: dpiAlsoBlocks && dpiPreset != null ? dpiPreset : null,
      } : null,
      upscale: dpiAlsoBlocks ? needPx(img, CP, s.minDpi) : null,
    };
  }

  /** "At 240 dpi this file reaches 27 inches along its long edge." */
  function reachAtDpi(img, minDpi) {
    return Math.floor((Math.max(img.w, img.h) / minDpi) * 2) / 2;
  }

  /**
   * The upscale factor this plan actually needs.
   *
   * Topaz takes arbitrary decimal factors (verified: 1.3 on a 3660x2997 source
   * gives exactly 4758x3896), so ask for what is required rather than rounding
   * up to the next rung of a 2/4/6 ladder. A plan that needs 1.3x should not
   * pay for 2x — that is 2.4x the pixels, and every one of them carries
   * synthesized texture the plan never asked for.
   *
   * Rounded UP to a tenth so the result clears the floor rather than landing
   * a hair under it. Returns null past MAX_UPSCALE, so the UI can say a plan is
   * out of reach instead of offering an upscale that would not fix it.
   */
  function upscaleFor(need) {
    var k = Math.ceil(need.k * 10) / 10;
    if (k <= 1) return null;              // already enough resolution
    return k <= MAX_UPSCALE ? k : null;
  }

  return {
    SIZES: SIZES, GRIDS: GRIDS, DPIS: DPIS, CROPS: CROPS,
    PANEL_COUNTS: PANEL_COUNTS,
    MAX_SPAN_IN: MAX_SPAN_IN, MAX_UPSCALE: MAX_UPSCALE, DEFAULTS: DEFAULTS,
    plan: plan, panelRects: panelRects, needPx: needPx,
    candidates: candidates, reachable: reachable, bestAlternative: bestAlternative,
    forGrid: forGrid,
    diagnose: diagnose, needSentence: needSentence, reachAtDpi: reachAtDpi,
    upscaleFor: upscaleFor, sizeLabel: sizeLabel, describe: describe,
    fmt: fmt, pct: pct, inches: inches, commas: commas, ratio: ratio,
  };
}));
