/**
 * PhotoSearch Shared Components (M15 + M16)
 *
 * Provides:
 *   PS.SharedHeader   — consistent nav header across all pages
 *   PS.PhotoModal      — unified photo detail modal with sidebar
 *   PS.formatFocalLength / PS.formatFNumber — camera helpers
 *
 * Load AFTER React CDN, BEFORE page-specific inline scripts:
 *   <script src="/shared.js"></script>
 */
(function () {
  'use strict';

  var e = React.createElement;
  var useState = React.useState;
  var useEffect = React.useEffect;
  var useCallback = React.useCallback;
  var useRef = React.useRef;
  var useMemo = React.useMemo;

  var API = '';  // same origin

  // =========================================================================
  // Namespace
  // =========================================================================
  var PS = window.PS = window.PS || {};

  // =========================================================================
  // Shared responsive baseline
  // =========================================================================
  // Injected once per page (every page loads shared.js). The page nav header
  // (.header-inner) is `display:flex` with NO flex-wrap on any page, so on a
  // phone the nav links + page controls (e.g. review's folder picker) overflow
  // horizontally off-screen. These rules are ADDITIVE — no page sets flex-wrap
  // or these mobile caps, so nothing existing is overridden. Appended to <head>
  // after the page's own <style>, so it wins ties.
  (function injectResponsiveBaseline() {
    if (document.getElementById('ps-responsive-baseline')) return;
    var css = [
      '.header-inner { flex-wrap: wrap; }',
      '@media (max-width: 640px) {',
      '  .header { padding-left: 12px; padding-right: 12px; }',
      '  .header-inner { gap: 8px 10px; }',
      '  .header-inner input[type="text"], .header-inner input[type="search"],',
      '  .header-inner select, .header-inner .ctrl-input { max-width: 78vw; }',
      '}',
    ].join('\n');
    var el = document.createElement('style');
    el.id = 'ps-responsive-baseline';
    el.textContent = css;
    (document.head || document.documentElement).appendChild(el);
  })();

  // =========================================================================
  // Helpers
  // =========================================================================
  PS.formatFocalLength = function formatFocalLength(fl) {
    if (fl && fl.includes('/')) {
      var parts = fl.split('/');
      return (parseInt(parts[0]) / parseInt(parts[1])).toFixed(0) + 'mm';
    }
    return fl + 'mm';
  };

  PS.formatFNumber = function formatFNumber(fn) {
    if (fn && fn.includes('/')) {
      var parts = fn.split('/');
      return (parseInt(parts[0]) / parseInt(parts[1])).toFixed(1);
    }
    return fn;
  };

  // =========================================================================
  // Label chip — the one toggle the eval labelling pages share
  // =========================================================================
  // Presentational only: the page decides what a click means (the visual-tag
  // page cycles no → yes → debatable; /eval/models toggles a claim wrong).
  // Styling stays with each page's `.chip` / `.chip.<state>` CSS.
  // Props: label, state (CSS class, e.g. 'no'|'yes'|'debatable'|'wrong'),
  //        mark (glyph), gloss, note, wide, title, disabled,
  //        onClick(ev), onContextMenu(ev).
  PS.Chip = function Chip(props) {
    var h = React.createElement;
    var wide = props.wide != null ? props.wide : !!(props.gloss || props.note);
    return h('button', {
      type: 'button', className: 'chip ' + (props.state || 'no') + (wide ? ' wide' : ''),
      onClick: props.onClick, onContextMenu: props.onContextMenu,
      title: props.title, disabled: props.disabled,
    },
      h('span', { className: 'mark' }, props.mark || ''),
      h('span', { className: 'tag' }, props.label),
      props.gloss ? h('span', { className: 'gloss' }, props.gloss) : null,
      props.note ? h('span', { className: 'gloss note' }, props.note) : null);
  };

  // =========================================================================
  // Loupe — the ORIGINAL at actual pixels, pannable
  // =========================================================================
  // Sharpness has to be judged at native resolution: a 1920 px /preview has
  // already averaged away the focus error it would reveal (the same mistake
  // that made a 336 px VLM tile useless as a blur judge). So the loupe loads
  // /api/photos/{id}/full and draws it at 1 image pixel = 1 DEVICE pixel —
  // CSS size divided by devicePixelRatio, or a HiDPI screen would upsample it
  // 2x and fake a softness that is not in the file.
  //
  // Pure geometry, exported for tests: the scroll offset that puts the
  // fraction `focus` ({x, y} in 0..1 of the image) at the viewport centre,
  // clamped to the scrollable range.
  PS.loupeScroll = function loupeScroll(focus, imgW, imgH, viewW, viewH) {
    var fx = focus && focus.x != null ? focus.x : 0.5;
    var fy = focus && focus.y != null ? focus.y : 0.5;
    function clamp(v, max) { return Math.max(0, Math.min(Math.round(v), Math.max(0, max))); }
    return { left: clamp(fx * imgW - viewW / 2, imgW - viewW),
             top: clamp(fy * imgH - viewH / 2, imgH - viewH) };
  };

  // The CSS size that shows `natural` image pixels at `zoom` x actual pixels.
  PS.loupeCssSize = function loupeCssSize(naturalW, naturalH, zoom, dpr) {
    var s = (zoom || 1) / (dpr || 1);
    return { width: Math.round(naturalW * s), height: Math.round(naturalH * s) };
  };

  // Props: src (the original's URL), focus ({x, y} fractions — where the user
  //        clicked on the fitted image; default the centre), onClose().
  // Drag or scroll to pan, arrow keys pan, `z` toggles 100% / 200%, Esc or the
  // × closes. While open it owns the keyboard (capture phase + stopPropagation),
  // so a page's own arrow-key navigation does not fire underneath it.
  PS.Loupe = function Loupe(props) {
    var h = React.createElement;
    var zoomS = useState(1); var zoom = zoomS[0]; var setZoom = zoomS[1];
    var natS = useState(null); var nat = natS[0]; var setNat = natS[1];
    var errS = useState(null); var err = errS[0]; var setErr = errS[1];
    var boxRef = useRef(null);
    var drag = useRef(null);
    var focusRef = useRef(props.focus || { x: 0.5, y: 0.5 });
    var dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
    var size = nat ? PS.loupeCssSize(nat.w, nat.h, zoom, dpr) : null;

    // Re-centre on the focus point whenever the drawn size changes (load,
    // zoom). On a zoom toggle, the focus is the current view centre.
    useEffect(function () {
      var box = boxRef.current;
      if (!box || !size) return;
      var p = PS.loupeScroll(focusRef.current, size.width, size.height,
                             box.clientWidth, box.clientHeight);
      box.scrollLeft = p.left; box.scrollTop = p.top;
    }, [size && size.width, size && size.height]);

    function viewCentre() {
      var box = boxRef.current;
      if (!box || !size || !size.width || !size.height) return focusRef.current;
      return { x: (box.scrollLeft + box.clientWidth / 2) / size.width,
               y: (box.scrollTop + box.clientHeight / 2) / size.height };
    }

    function toggleZoom() {
      focusRef.current = viewCentre();
      setZoom(function (z) { return z === 1 ? 2 : 1; });
    }

    useEffect(function () {
      function onKey(ev) {
        var box = boxRef.current;
        var step = 200;
        var handled = true;
        if (ev.key === 'Escape') props.onClose();
        else if (ev.key === 'z') toggleZoom();
        else if (box && ev.key === 'ArrowLeft') box.scrollLeft -= step;
        else if (box && ev.key === 'ArrowRight') box.scrollLeft += step;
        else if (box && ev.key === 'ArrowUp') box.scrollTop -= step;
        else if (box && ev.key === 'ArrowDown') box.scrollTop += step;
        else handled = false;
        if (handled) { ev.preventDefault(); ev.stopPropagation(); }
        else if (ev.key === 'Enter' || ev.key === 's' || ev.key === 'u') {
          // Never let a save / jump fire on a photo the labeller cannot see.
          ev.preventDefault(); ev.stopPropagation();
        }
      }
      window.addEventListener('keydown', onKey, true);
      return function () { window.removeEventListener('keydown', onKey, true); };
    }, [size && size.width]);

    function down(ev) {
      var box = boxRef.current;
      if (!box || ev.button !== 0) return;
      drag.current = { x: ev.clientX, y: ev.clientY, l: box.scrollLeft, t: box.scrollTop };
      if (ev.currentTarget.setPointerCapture) ev.currentTarget.setPointerCapture(ev.pointerId);
      ev.preventDefault();
    }
    function move(ev) {
      var d = drag.current; var box = boxRef.current;
      if (!d || !box) return;
      box.scrollLeft = d.l - (ev.clientX - d.x);
      box.scrollTop = d.t - (ev.clientY - d.y);
    }
    function up() { drag.current = null; }

    var overlay = { position: 'fixed', inset: 0, zIndex: 2000, background: 'rgba(0,0,0,0.94)',
                    display: 'flex', flexDirection: 'column' };
    var bar = { display: 'flex', gap: 12, alignItems: 'center', padding: '8px 14px',
                color: '#ccc', fontSize: 13, borderBottom: '1px solid #333' };
    var btn = { background: '#242424', border: '1px solid #444', color: '#eee',
                borderRadius: 6, padding: '4px 12px', cursor: 'pointer', fontFamily: 'inherit' };

    return h('div', { style: overlay, role: 'dialog', 'aria-label': 'Loupe: original at actual pixels' },
      h('div', { style: bar },
        h('b', null, zoom === 1 ? '100% — actual pixels' : '200%'),
        h('span', null, nat ? nat.w + ' × ' + nat.h + ' px original' : err ? '' : 'loading the original…'),
        h('span', { style: { color: '#888' } }, 'drag / scroll / arrow keys to pan · z zoom · Esc close'),
        h('span', { style: { flex: 1 } }),
        h('button', { type: 'button', style: btn, onClick: toggleZoom }, zoom === 1 ? '200%' : '100%'),
        h('button', { type: 'button', style: btn, onClick: props.onClose, 'aria-label': 'Close loupe' }, '× Close')),
      err ? h('div', { style: { color: '#f87171', padding: 20 } }, err) : null,
      h('div', {
        ref: boxRef,
        style: { flex: 1, overflow: 'auto', cursor: drag.current ? 'grabbing' : 'grab',
                 touchAction: 'none', userSelect: 'none' },
        onPointerDown: down, onPointerMove: move, onPointerUp: up, onPointerCancel: up,
      },
        h('img', {
          src: props.src, alt: 'original', draggable: false,
          // Until the natural size is known, keep it off-screen at its natural
          // size so the browser decodes it once; never CSS-scaled to fit.
          style: size
            ? { display: 'block', width: size.width, height: size.height, maxWidth: 'none',
                imageRendering: zoom > 1 ? 'pixelated' : 'auto' }
            : { position: 'absolute', visibility: 'hidden', maxWidth: 'none' },
          onLoad: function (ev) {
            setNat({ w: ev.target.naturalWidth, h: ev.target.naturalHeight });
          },
          onError: function () {
            setErr('The browser could not display this original (HEIC or RAW, or the '
                   + 'NAS is unreachable). Judge it in another viewer at 100%.');
          },
        })));
  };

  // =========================================================================
  // M15 — SharedHeader
  // =========================================================================
  // Props:
  //   activePage  — 'search' | 'review' | 'faces' | 'merges' | 'collections' | 'book' | 'split' | 'map' | 'geotag' | 'batches' | 'status' | 'admin'
  //   children    — optional React nodes (search form, review controls, etc.)
  PS.SharedHeader = function SharedHeader(props) {
    var activePage = props.activePage || 'search';
    var children = props.children;

    var navLinks = [
      { href: '/',            label: 'Search',      id: 'search' },
      { href: '/review',      label: 'Review',      id: 'review' },
      { href: '/faces',       label: 'Faces',       id: 'faces' },
      { href: '/merges',      label: 'Merges',      id: 'merges' },
      { href: '/collections', label: 'Collections', id: 'collections' },
      { href: '/book',        label: 'Books',       id: 'book' },
      // Needs a photo, so a bare visit lands on its empty state — but it
      // was otherwise reachable only from a photo's detail panel.
      { href: '/split',       label: 'Split',       id: 'split' },
      { href: '/map',         label: 'Map',         id: 'map' },
      { href: '/geotag',      label: 'Geotag',      id: 'geotag' },
      { href: '/batches',     label: 'Batches',     id: 'batches' },
      { href: '/status',      label: 'Status',      id: 'status' },
      { href: '/admin/deploy',      label: 'Deploy', id: 'deploy' },
      { href: '/admin/maintenance', label: 'Maint',  id: 'maint' },
      { href: '/admin/vocab', label: 'Vocab',       id: 'admin' },
      { href: '/logs',        label: 'Logs',        id: 'logs' },
    ];

    return e('div', { className: 'header' },
      e('div', { className: 'header-inner' },
        e('div', { className: 'logo' },
          e('span', null, 'photo'), 'search',
        ),
        // Nav links
        navLinks.map(function (link) {
          return e('a', {
            key: link.id,
            href: link.href,
            className: 'nav-link' + (activePage === link.id ? ' active' : ''),
          }, link.label);
        }),
        // Page-specific controls (search bar, review folder selector, etc.)
        children,
      ),
    );
  };

  // =========================================================================
  // M16 — PhotoModal  (shared photo detail sidebar)
  // =========================================================================
  // Props:
  //   photo            — { id, filename, date_taken, score, clip_score, aesthetic_score, colors, ... }
  //   detail           — full detail object (if null & fetchDetail=true, modal fetches it)
  //   fetchDetail      — bool: if true, fetch /api/photos/:id when photo changes (default false)
  //   onClose          — close callback
  //   onPrev / onNext  — navigation callbacks (may be null)
  //   index / total    — current position in list
  //   persons          — array of { id, name } for face editing dropdown
  //   onFaceAssigned   — (faceId, personName|null) callback when face name changes
  //   showFaces        — show face editing section (default true)
  //   showCollections  — show collections section (default false)
  //   showLocation     — show location section (default true)
  //   showSearchScore  — show search score section (default false)
  //   showAesthetics   — show full aesthetic quality section (default true)
  //   headerChildren   — React nodes to render at the TOP of the sidebar (before photo info)
  //   footerChildren   — React nodes to render at the BOTTOM of the sidebar (after colors)
  //   onDetailLoaded   — callback when detail is fetched (for pages that need it)
  //   onDetailChanged  — callback when detail changes locally (face edits)

  // =========================================================================
  // Video subject strip (video-support step 6) — the owner's audit view of
  // the main-subject metric. Always shown on a video (not flag-gated, D16):
  // a timeline of segments coloured by primary subject, a legend, and per-
  // person presence + focus bars. Click anywhere to seek the <video>.
  // Data: GET /api/videos/{id}/subjects (names read through faces.person_id).
  // =========================================================================
  var SUBJECT_COLORS = ['#60a5fa', '#f472b6', '#34d399', '#fbbf24', '#a78bfa',
                        '#f87171', '#22d3ee', '#a3e635', '#fb923c', '#e879f9'];
  var NO_SUBJECT_COLOR = 'rgba(148,163,184,0.25)';

  PS.subjectColor = function (ix) {
    if (ix == null) return NO_SUBJECT_COLOR;
    return SUBJECT_COLORS[Math.abs(ix) % SUBJECT_COLORS.length];
  };

  PS.VideoSubjectStrip = function VideoSubjectStrip(props) {
    var photoId = props.photoId;
    var videoRef = props.videoRef;
    var _d = useState(null); var data = _d[0]; var setData = _d[1];
    var _err = useState(''); var err = _err[0]; var setErr = _err[1];
    var _t = useState(null); var now = _t[0]; var setNow = _t[1];

    useEffect(function () {
      setData(null); setErr('');
      if (!photoId) return;
      var dead = false;
      fetch(API + '/api/videos/' + photoId + '/subjects')
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function (d) { if (!dead) setData(d); })
        .catch(function (ex) { if (!dead) setErr(ex.message || String(ex)); });
      return function () { dead = true; };
    }, [photoId]);

    // Playhead: follow the <video>'s clock.
    useEffect(function () {
      var v = videoRef && videoRef.current;
      if (!v) return;
      var tick = function () { setNow(v.currentTime); };
      v.addEventListener('timeupdate', tick);
      v.addEventListener('seeked', tick);
      return function () {
        v.removeEventListener('timeupdate', tick);
        v.removeEventListener('seeked', tick);
      };
    }, [videoRef, data]);

    var heading = e('h3', null, 'Main subject ',
      e('span', { style: { fontSize: 10, padding: '1px 6px', borderRadius: 8, marginLeft: 4,
                           background: 'var(--surface2)', color: 'var(--text-muted)',
                           border: '1px solid var(--border)', verticalAlign: 'middle' } }, 'beta'));
    if (err) return e('div', { className: 'detail-section' }, heading,
      e('p', { style: { color: 'var(--text-muted)', fontSize: 12 } }, 'Could not load subjects: ' + err));
    if (!data) return e('div', { className: 'detail-section' }, heading,
      e('p', { style: { color: 'var(--text-muted)', fontSize: 12 } }, 'Loading…'));
    // Who is present: named people with >= presence_min_s of tracked face
    // (owner decision 2026-10-28; server-side, from the scan's tracks).
    var minS = data.presence_min_s != null ? data.presence_min_s : 2;
    var presentLine = (data.present && data.present.length)
      ? e('div', { style: { fontSize: 12, marginTop: 4 } },
          e('span', { style: { color: 'var(--text-muted)' } }, 'Present (\u2265 ' + minS + ' s): '),
          data.present.map(function (p) { return p.name + ' ' + Math.round(p.presence_s) + ' s'; }).join(', '))
      : null;
    if (data.subjects_version == null) return e('div', { className: 'detail-section' }, heading,
      presentLine,
      e('p', { style: { color: 'var(--text-muted)', fontSize: 12 } },
        'Not analysed yet — the video-subjects pass has not run on this clip.'));

    var dur = data.duration || 0;
    if (!dur && data.segments && data.segments.length) dur = data.segments[data.segments.length - 1].end;
    var pct = function (t) { return dur ? Math.max(0, Math.min(100, 100 * t / dur)) : 0; };
    var seek = function (t) {
      var v = videoRef && videoRef.current;
      if (!v || t == null || isNaN(t)) return;
      try { v.currentTime = Math.max(0, t); } catch (_) {}
    };
    var seekFromEvent = function (ev) {
      var r = ev.currentTarget.getBoundingClientRect();
      if (!r.width || !dur) return;
      seek(dur * (ev.clientX - r.left) / r.width);
    };
    var fmt = function (t) { return PS.formatDuration(t) || '0:00'; };
    var bySix = {};
    (data.subjects || []).forEach(function (s) { bySix[s.ix] = s; });

    var playhead = (now != null && dur) && e('div', {
      style: { position: 'absolute', top: -2, bottom: -2, width: 2, left: 'calc(' + pct(now) + '% - 1px)',
               background: '#fff', boxShadow: '0 0 2px #000', pointerEvents: 'none' } });

    var track = function (intervals, color, height, title) {
      return e('div', {
        title: title, onClick: seekFromEvent,
        style: { position: 'relative', height: height, borderRadius: 3, cursor: 'pointer',
                 background: 'rgba(148,163,184,0.12)', overflow: 'hidden' },
      }, (intervals || []).map(function (iv, i) {
        var a = iv[0], b = iv[1];
        return e('div', { key: i, style: { position: 'absolute', top: 0, bottom: 0,
          left: pct(a) + '%', width: Math.max(0.5, pct(b) - pct(a)) + '%', background: color } });
      }));
    };

    // Timeline of primaries (merged runs).
    var runs = data.timeline && data.timeline.length ? data.timeline
      : (data.segments || []).map(function (sg) {
          var s2 = sg.subject_ix != null ? bySix[sg.subject_ix] : null;
          return { start: sg.start, end: sg.end, subject_ix: sg.subject_ix, display: s2 ? s2.display : null };
        });
    var timeline = e('div', {
      onClick: seekFromEvent,
      style: { position: 'relative', height: 22, borderRadius: 4, cursor: 'pointer',
               background: NO_SUBJECT_COLOR, marginTop: 4 },
    },
      runs.map(function (r, i) {
        var lab = r.display || 'no subject';
        return e('div', {
          key: i,
          title: fmt(r.start) + '–' + fmt(r.end) + '  ' + lab,
          style: { position: 'absolute', top: 0, bottom: 0, left: pct(r.start) + '%',
                   width: Math.max(0.5, pct(r.end) - pct(r.start)) + '%',
                   background: PS.subjectColor(r.subject_ix),
                   borderRight: '1px solid rgba(0,0,0,0.35)' },
        });
      }),
      playhead,
    );

    // Legend: every subject (biggest share first) plus "no subject".
    var subs = (data.subjects || []).slice().sort(function (a, b) { return (b.share || 0) - (a.share || 0); });
    var legend = e('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '4px 10px', marginTop: 6, fontSize: 12 } },
      subs.map(function (s2) {
        return e('span', {
          key: s2.ix, style: { display: 'inline-flex', alignItems: 'center', gap: 4, cursor: s2.rep_t != null ? 'pointer' : 'default' },
          title: (s2.category || '') + (s2.n_segments != null ? ', ' + s2.n_segments + ' segment(s)' : '')
                 + (s2.rep_t != null ? ' — click to see it' : ''),
          onClick: function () { if (s2.rep_t != null) seek(s2.rep_t); },
        },
          e('span', { style: { width: 10, height: 10, borderRadius: 2, background: PS.subjectColor(s2.ix), display: 'inline-block' } }),
          (s2.display || s2.label || '?') + ' ' + Math.round((s2.share || 0) * 100) + '%'
            + (s2.focus_s != null ? ' (' + Math.round(s2.focus_s) + ' s)' : ''));
      }),
      e('span', { style: { display: 'inline-flex', alignItems: 'center', gap: 4, color: 'var(--text-muted)' } },
        e('span', { style: { width: 10, height: 10, borderRadius: 2, background: NO_SUBJECT_COLOR, display: 'inline-block' } }),
        'no subject'),
    );

    // Per-person rows: dense face-track presence above, main-subject focus below.
    var personRows = subs.filter(function (s2) { return s2.face_id != null && s2.present !== false; }).map(function (s2) {
      var col = PS.subjectColor(s2.ix);
      return e('div', { key: 'p' + s2.ix, style: { marginTop: 6 } },
        e('div', { style: { fontSize: 12, display: 'flex', justifyContent: 'space-between', gap: 6 } },
          e('span', { style: { fontWeight: 600 } }, s2.display),
          e('span', { style: { color: 'var(--text-muted)' } },
            'subject ' + Math.round((s2.share || 0) * 100) + '% · '
            + Math.round(s2.focus_s || 0) + ' s'
            + (s2.presence_s != null ? ' · present ' + Math.round(s2.presence_s) + ' s' : ''))),
        track(s2.presence_intervals, col + '66', 6, 'Present (face tracked) — click to seek'),
        e('div', { style: { height: 2 } }),
        track(s2.focus_intervals, col, 6, 'Main subject — click to seek'));
    });

    return e('div', { className: 'detail-section' },
      heading,
      e('div', { style: { fontSize: 11, color: 'var(--text-muted)' } },
        'Share of the whole clip (' + fmt(dur) + ')' + (data.model ? ' · ' + data.model : '')),
      presentLine,
      timeline,
      e('div', { style: { display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)' } },
        e('span', null, '0:00'), e('span', null, fmt(dur))),
      legend,
      personRows.length > 0 && e('div', { style: { marginTop: 4 } }, personRows),
    );
  };

  // =========================================================================
  // Video quality numbers (video-support step 7, D11). Always shown as plain
  // NUMBERS with a display-only plain-language band — never written as a tag
  // anywhere (that needs a holdout eval that hasn't run; see D11 in
  // docs/plans/video-support.md). Reads `detail.video.quality`, already part
  // of the PhotoModal's own /api/photos/{id} fetch — no extra network call.
  // =========================================================================
  // Bands are a rough, undocumented-elsewhere display heuristic (not derived
  // from an eval) — purely to help a human read the raw number at a glance.
  function _shakeBand(v) {
    if (v == null) return null;
    return v < 0.5 ? 'steady' : v < 2.0 ? 'some movement' : 'shaky';
  }
  function _noiseBand(v) {
    if (v == null) return null;
    return v < 3 ? 'clean' : v < 8 ? 'some noise' : 'noisy';
  }
  function _focusBand(v) {
    if (v == null) return null;
    return v >= 0.8 ? 'consistently sharp' : v >= 0.5 ? 'sometimes soft' : 'often soft';
  }
  function _qualityRow(label, value, decimals, band) {
    return e('div', {
      key: label,
      style: { display: 'flex', justifyContent: 'space-between', fontSize: 12, marginTop: 2 },
    },
      e('span', { style: { color: 'var(--text-muted)' } }, label),
      e('span', null,
        value == null ? '—' : value.toFixed(decimals) + (band ? '  (' + band + ')' : '')),
    );
  }

  PS.VideoQualityBlock = function VideoQualityBlock(props) {
    var q = props.quality;
    if (!q) return null;
    if (q.scan_error) {
      return e('div', { className: 'detail-section' },
        e('h3', null, 'Video quality'),
        e('p', { style: { color: 'var(--text-muted)', fontSize: 12 } },
          'Could not be measured: ' + q.scan_error));
    }
    if (q.shake == null && q.noise == null && q.focus_consistency == null) {
      return e('div', { className: 'detail-section' },
        e('h3', null, 'Video quality'),
        e('p', { style: { color: 'var(--text-muted)', fontSize: 12 } },
          'Not measured yet — the video-scan pass has not run on this clip.'));
    }
    var exp = q.exposure || {};
    var hi = exp.highlight_frac, lo = exp.shadow_frac;
    var expBand = (hi != null || lo != null)
      ? (((hi || 0) > 0.05 || (lo || 0) > 0.05) ? 'some clipping' : 'well exposed')
      : null;
    return e('div', { className: 'detail-section' },
      e('h3', null, 'Video quality'),
      _qualityRow('Shake', q.shake, 2, _shakeBand(q.shake)),
      _qualityRow('Noise', q.noise, 1, _noiseBand(q.noise)),
      _qualityRow('Focus consistency',
        q.focus_consistency != null ? q.focus_consistency * 100 : null, 0,
        _focusBand(q.focus_consistency)),
      (hi != null || lo != null) && e('div', {
        style: { display: 'flex', justifyContent: 'space-between', fontSize: 12, marginTop: 2 },
      },
        e('span', { style: { color: 'var(--text-muted)' } }, 'Exposure clipping'),
        e('span', null,
          'highlights ' + Math.round((hi || 0) * 100) + '%, shadows ' + Math.round((lo || 0) * 100) + '%'
          + (expBand ? '  (' + expBand + ')' : ''))),
    );
  };

  PS.PhotoModal = function PhotoModal(props) {
    var photo = props.photo;
    var externalDetail = props.detail || null;
    var fetchDetailProp = props.fetchDetail || false;
    var onClose = props.onClose;
    var onPrev = props.onPrev;
    var onNext = props.onNext;
    var index = props.index || 0;
    var total = props.total || 0;
    var persons = props.persons || [];
    var onFaceAssigned = props.onFaceAssigned;
    var showFaces = props.showFaces !== false;
    var showCollections = props.showCollections || false;
    var showLocation = props.showLocation !== false;
    var showSearchScore = props.showSearchScore || false;
    var showAesthetics = props.showAesthetics !== false;
    var headerChildren = props.headerChildren || null;
    var footerChildren = props.footerChildren || null;
    var onDetailLoaded = props.onDetailLoaded;
    var onDetailChanged = props.onDetailChanged;

    // ---- Internal detail state (when fetching ourselves) ----
    var _internalDetail = useState(null);
    var internalDetail = _internalDetail[0];
    var setInternalDetail = _internalDetail[1];

    var detail = fetchDetailProp ? internalDetail : externalDetail;
    // Videos (video-support step 2): played with <video>, no face overlays.
    var isVideo = PS.isVideo(photo) || PS.isVideo(detail);
    // The subject strip seeks this element (video-support step 6).
    var videoRef = useRef(null);
    // Original vs the H.264 proxy (video-proxy pass). `videoErrored` flips
    // when the ORIGINAL fails to play, which switches to the proxy if there
    // is one, else to the "preparing / download" panel.
    var _videoErr = useState(false);
    var videoErrored = _videoErr[0];     var setVideoErrored = _videoErr[1];
    useEffect(function () { setVideoErrored(false); }, [photo && photo.id]);
    var videoInfo = detail && detail.video;
    var videoChoice = isVideo && photo
      ? PS.chooseVideoSource(videoInfo, API + '/api/photos/' + photo.id + '/full',
                             PS.canPlayVideo(videoInfo && videoInfo.playback), API,
                             (detail || photo).filename)
      : null;
    if (videoChoice && videoChoice.mode === 'original' && videoErrored) {
      videoChoice = videoChoice.fallback
        ? { mode: 'proxy', src: videoChoice.fallback, fallback: null }
        : { mode: 'preparing', src: null, fallback: null };
    }
    var videoPlayable = !isVideo || !!(videoChoice && videoChoice.mode !== 'preparing');

    useEffect(function () {
      if (!fetchDetailProp || !photo) return;
      setInternalDetail(null);
      fetch(API + '/api/photos/' + photo.id)
        .then(function (r) { return r.json(); })
        .then(function (d) {
          setInternalDetail(d);
          if (onDetailLoaded) onDetailLoaded(d);
        })
        .catch(function () {});
    }, [photo && photo.id, fetchDetailProp]);

    // ---- Generation-history state (lazy-fetched on expand) ----
    var _gens = useState(null);          // {generations: [...]} once loaded
    var gens = _gens[0];                 var setGens = _gens[1];
    var _gensOpen = useState(false);     // expansion toggle
    var gensOpen = _gensOpen[0];         var setGensOpen = _gensOpen[1];
    var _gensLoading = useState(false);
    var gensLoading = _gensLoading[0];   var setGensLoading = _gensLoading[1];

    // ---- Change history (photo_audit), lazy-fetched on expand ----
    var _aud = useState(null);           // {events: [...]} once loaded
    var aud = _aud[0];                   var setAud = _aud[1];
    var _audOpen = useState(false);
    var audOpen = _audOpen[0];           var setAudOpen = _audOpen[1];
    var _audLoading = useState(false);
    var audLoading = _audLoading[0];     var setAudLoading = _audLoading[1];

    // ---- Re-run-passes state (M28) ----
    // A video only has its own passes (db.PASS_MEDIA): the image passes
    // would never claim it, and a still has no video-probe. category-content
    // and keywords read any description, a video's included (step 5).
    var RERUN_PASSES = PS.isVideo(photo)
      ? ['video-probe', 'video-proxy', 'video-scan', 'video-subjects', 'video-describe',
         'category-content', 'keywords']
      : ['describe', 'verify', 'category-content', 'keywords',
         'category-visual', 'clip', 'faces', 'quality', 'aesthetics'];
    var _rerunOpen = useState(false);
    var rerunOpen = _rerunOpen[0];       var setRerunOpen = _rerunOpen[1];
    var _rerunSel = useState({});         // { pass: true }
    var rerunSel = _rerunSel[0];          var setRerunSel = _rerunSel[1];
    var _rerunMode = useState('sync');    // 'sync' | 'queue'
    var rerunMode = _rerunMode[0];        var setRerunMode = _rerunMode[1];
    var _rerunBusy = useState(false);
    var rerunBusy = _rerunBusy[0];        var setRerunBusy = _rerunBusy[1];
    var _rerunMsg = useState(null);
    var rerunMsg = _rerunMsg[0];          var setRerunMsg = _rerunMsg[1];

    // ---- Topaz upscale state (M30) ----
    // Runs the local Topaz Photo AI CLI and writes to the export tree; unlike
    // re-run passes this touches neither the DB nor the library, so there is
    // nothing to mirror or refresh afterward.
    var UPSCALE_ENHANCEMENTS = ['upscale', 'noise', 'sharpen', 'lighting', 'color'];
    // Verified against the CLI. Redefine/Recovery need downloading in the
    // Topaz GUI first, so they are not offered until they exist.
    var UPSCALE_MODELS = ['Standard', 'Standard V2', 'High Fidelity',
                          'High Fidelity V2', 'Low Resolution', 'CGI'];
    var _upOpen = useState(false);
    var upOpen = _upOpen[0];             var setUpOpen = _upOpen[1];
    var _upScale = useState('auto');     // 'auto' | '2' | '4' | '6'
    var upScale = _upScale[0];           var setUpScale = _upScale[1];
    var _upEnh = useState({ upscale: true });
    var upEnh = _upEnh[0];               var setUpEnh = _upEnh[1];
    var _upBusy = useState(false);
    var upBusy = _upBusy[0];             var setUpBusy = _upBusy[1];
    var _upMsg = useState(null);
    var upMsg = _upMsg[0];               var setUpMsg = _upMsg[1];
    var _upModel = useState('');         // '' = let Autopilot pick
    var upModel = _upModel[0];           var setUpModel = _upModel[1];
    // Autopilot merges its own picks into every run — in practice a
    // "Sharpen Strong" pass that haloes an already-upscaled image. On by
    // default because that is usually what you want for a clean upscale.
    // 25% is the default: full strength visibly over-synthesizes texture on
    // detailed natural subjects. Topaz itself has no working dial.
    var _upStrength = useState('25');
    var upStrength = _upStrength[0];     var setUpStrength = _upStrength[1];
    var _upOnly = useState(true);
    var upOnly = _upOnly[0];             var setUpOnly = _upOnly[1];
    var _upPrior = useState(null);       // null = not loaded yet, [] = none
    var upPrior = _upPrior[0];           var setUpPrior = _upPrior[1];

    // Prior exports live only as files on disk, so re-read them whenever the
    // section is opened or the photo changes — otherwise a page refresh loses
    // every path until you upscale again.
    var loadUpPrior = React.useCallback(function () {
      if (!photo || !photo.id) return;
      fetch(API + '/api/admin/upscaled-list?photo_id=' + photo.id)
        .then(function (r) { return r.ok ? r.json() : { exports: [] }; })
        .then(function (d) { setUpPrior(d.exports || []); })
        .catch(function () { setUpPrior([]); });
    }, [photo && photo.id]);

    useEffect(function () {
      setUpPrior(null);
      setUpMsg(null);
      if (upOpen) loadUpPrior();
    }, [photo && photo.id]);

    useEffect(function () {
      if (upOpen && upPrior === null) loadUpPrior();
    }, [upOpen]);

    // Reset gen + re-run + upscale state when the focused photo changes
    useEffect(function () {
      setGens(null); setGensOpen(false); setGensLoading(false);
      setAud(null); setAudOpen(false); setAudLoading(false);
      setRerunSel({}); setRerunMsg(null); setRerunBusy(false);
      setUpMsg(null); setUpBusy(false);
    }, [photo && photo.id]);

    function submitUpscale() {
      var enh = UPSCALE_ENHANCEMENTS.filter(function (x) { return upEnh[x]; });
      if (!photo || enh.length === 0) return;
      setUpBusy(true);
      setUpMsg({ kind: 'info', text: 'Upscaling with Topaz… (~20-30s per photo)' });
      fetch(API + '/api/admin/upscale-photo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          photo_ids: [photo.id],
          scale: upScale === 'auto' ? null : parseFloat(upScale),
          enhancements: enh,
          model: upModel || null,
          override: upOnly,
          strength: parseInt(upStrength, 10) / 100,
        }),
      }).then(function (r) {
        return r.json().then(function (d) { return { ok: r.ok, d: d }; });
      }).then(function (res) {
        if (!res.ok) {
          setUpMsg({ kind: 'err', text: (res.d && res.d.detail) || 'Upscale failed' });
          setUpBusy(false);
          return;
        }
        var errs = (res.d && res.d.errors) || [];
        var done = (res.d && res.d.results) || [];
        if (errs.length) {
          setUpMsg({ kind: 'err', text: 'Upscale failed: ' + errs[0].error });
        } else if (done.length) {
          var r0 = done[0];
          setUpMsg({
            kind: 'ok',
            text: r0.skipped
              ? 'Already exported'
              : 'Saved ' + (r0.bytes / 1e6).toFixed(1) + ' MB',
            // Windows path for humans; the POSIX one is only meaningful to the
            // server process. view is same-origin because Chrome refuses to
            // follow a file:// link from an http page.
            path: r0.windows_path || r0.output,
            view: API + '/api/admin/upscaled-file?path=' + encodeURIComponent(r0.output),
          });
          loadUpPrior();   // the new variant joins the list
        }
        setUpBusy(false);
      }).catch(function (e) {
        setUpMsg({ kind: 'err', text: 'Upscale error: ' + e });
        setUpBusy(false);
      });
    }

    // Toggle one pass. Selecting describe also selects the text passes derived
    // from the description (category-content + keywords) — re-describing alone
    // leaves those stale (the exact failure that mislabels e.g. soccer photos).
    function toggleRerunPass(p) {
      setRerunSel(function (prev) {
        var next = Object.assign({}, prev);
        if (next[p]) { delete next[p]; }
        else {
          next[p] = true;
          // Both can rewrite the description, which re-queues the text
          // passes server-side — so re-run them in the same go.
          if (p === 'describe' || p === 'verify' || p === 'video-describe') { next['category-content'] = true; next['keywords'] = true; }
        }
        return next;
      });
    }

    function submitRerun() {
      var passes = RERUN_PASSES.filter(function (p) { return rerunSel[p]; });
      if (!photo || passes.length === 0) return;
      setRerunBusy(true);
      setRerunMsg({ kind: 'info', text: (rerunMode === 'sync' ? 'Running ' : 'Queuing ') + passes.join(', ') + '…' });
      fetch(API + '/api/admin/rerun-passes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ photo_ids: [photo.id], passes: passes, mode: rerunMode }),
      }).then(function (r) {
        return r.json().then(function (d) { return { ok: r.ok, d: d }; });
      }).then(function (res) {
        if (!res.ok) {
          setRerunMsg({ kind: 'err', text: (res.d && res.d.detail) || 'Re-run failed' });
          setRerunBusy(false);
          return;
        }
        if (rerunMode === 'queue') {
          setRerunMsg({ kind: 'ok', text: 'Queued — a worker will process it; click Refresh once done.' });
          setRerunBusy(false);
          return;
        }
        var errs = (res.d && res.d.errors) || [];
        var done = (res.d && res.d.results) || [];
        setRerunMsg({ kind: errs.length ? 'err' : 'ok',
          text: 'Re-ran ' + done.length + ' pass(es)' + (errs.length ? ', ' + errs.length + ' failed: ' + errs.map(function (x) { return x.pass; }).join(', ') : '') + '.' });
        refreshDetail();
        setRerunBusy(false);
      }).catch(function (e) {
        setRerunMsg({ kind: 'err', text: 'Re-run error: ' + e });
        setRerunBusy(false);
      });
    }

    // Pull the freshly-mirrored row and update the visible detail. For queue
    // mode this also serves as the "Refresh" action (mirror first, then re-read).
    function refreshDetail(mirrorFirst) {
      if (!photo) return Promise.resolve();
      var pre = mirrorFirst
        ? fetch(API + '/api/admin/mirror-photos', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ photo_ids: [photo.id] }),
          }).catch(function () {})
        : Promise.resolve();
      return pre.then(function () {
        return fetch(API + '/api/photos/' + photo.id).then(function (r) { return r.json(); });
      }).then(function (d) {
        if (fetchDetailProp) setInternalDetail(d);
        if (onDetailChanged) onDetailChanged(d);
        if (onDetailLoaded) onDetailLoaded(d);
        setGens(null);  // force re-fetch of generation history on next expand
      }).catch(function () {});
    }

    function toggleGens() {
      if (!photo) return;
      if (gensOpen) { setGensOpen(false); return; }
      setGensOpen(true);
      if (gens || gensLoading) return; // already fetched
      setGensLoading(true);
      fetch(API + '/api/photos/' + photo.id + '/generations')
        .then(function (r) { return r.json(); })
        .then(function (d) { setGens(d); })
        .catch(function () { setGens({ generations: [], error: true }); })
        .then(function () { setGensLoading(false); });
    }

    function toggleAudit() {
      if (!photo) return;
      if (audOpen) { setAudOpen(false); return; }
      setAudOpen(true);
      setAudLoading(true);   // refetch each expand: collection edits happen in this modal
      fetch(API + '/api/photos/' + photo.id + '/audit?limit=100')
        .then(function (r) { return r.json(); })
        .then(function (d) { setAud(d); })
        .catch(function () { setAud({ events: [], error: true }); })
        .then(function () { setAudLoading(false); });
    }

    // One audit row as a sentence: what changed, with old -> new where known.
    function describeAudit(ev) {
      var coll = ev.collection_name ? '"' + ev.collection_name + '"' : 'collection #' + ev.collection_id;
      switch (ev.action) {
        case 'collection_added': return 'Added to ' + coll;
        case 'collection_removed':
          return (ev.field === 'collection_deleted' ? 'Removed (collection deleted) from '
            : ev.field === 'photo_deleted' ? 'Removed (photo deleted) from ' : 'Removed from ') + coll;
        case 'face_assigned': return 'Face ' + ev.face_id + ' named ' + (ev.new_value || '?') +
          (ev.old_value ? ' (was ' + ev.old_value + ')' : '');
        case 'face_cleared': return 'Face ' + ev.face_id + ' cleared' +
          (ev.old_value ? ' (was ' + ev.old_value + ')' : '') + ' and marked rejected';
        case 'face_unmatched': return 'Face ' + ev.face_id + ' unmatched' +
          (ev.old_value ? ' (was ' + ev.old_value + ')' : '');
        case 'filepath_changed': return 'Moved: ' + ev.old_value + ' → ' + ev.new_value;
        case 'tags_set': return 'Tags (' + ev.field + ') changed';
        case 'location_set': return 'Location set';
        case 'field_changed': return ev.field + ': ' + (ev.old_value || '∅') + ' → ' + (ev.new_value || '∅');
        case 'photo_added': return 'Added to the library';
        case 'photo_deleted': return 'Photo deleted';
        default: return ev.action;
      }
    }

    // Inline "model: X" chip rendered next to a text section header. Takes a
    // text_type; an explicit `note` overrides the default "generated <date>"
    // tooltip (used for description-vs-verify to flag a regeneration).
    function renderChip(cm, note) {
      if (!cm) return null;
      var label = cm.model
        ? cm.model + (cm.version ? '@' + String(cm.version).slice(0, 7) : '')
        : 'model unknown';
      return e('span', {
        style: {
          fontSize: 11, fontWeight: 'normal', marginLeft: 8,
          padding: '1px 8px', borderRadius: 10,
          background: cm.model ? 'var(--surface2)' : 'transparent',
          color: cm.model ? 'var(--text-muted)' : '#888',
          border: cm.model ? '1px solid var(--border)' : '1px dashed var(--border)',
          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
        },
        title: note || (cm.created_at ? 'generated ' + cm.created_at : 'producing model not recorded'),
      }, label);
    }

    function modelChip(textType) {
      var cm = detail && detail.current_models && detail.current_models[textType];
      return renderChip(cm);
    }

    // The displayed `description` text comes from whichever of describe/verify
    // is most recent (verify regenerates the description on hallucination).
    // Show the chip for that one so the badge matches the visible text.
    function descriptionChip() {
      var cm = detail && detail.current_models;
      if (!cm) return null;
      var d = cm.describe, v = cm.verify;
      if (v && d) {
        var picked = (v.created_at || '') > (d.created_at || '') ? v : d;
        var note = (picked === v)
          ? 'regenerated by verify pass on ' + v.created_at
          : 'generated ' + d.created_at;
        return renderChip(picked, note);
      }
      return renderChip(v || d);
    }

    // ---- Face editing state ----
    var _hoveredFace = useState(null);
    var hoveredFace = _hoveredFace[0];
    var setHoveredFace = _hoveredFace[1];

    var _editingFace = useState(null);
    var editingFace = _editingFace[0];
    var setEditingFace = _editingFace[1];

    var _editMode = useState('pick');
    var editMode = _editMode[0];
    var setEditMode = _editMode[1];

    var _editName = useState('');
    var editName = _editName[0];
    var setEditName = _editName[1];

    var _editPick = useState('');
    var editPick = _editPick[0];
    var setEditPick = _editPick[1];

    var editInputRef = useRef(null);

    // ---- Manual face detection / box editing (features 4-6) ----
    var _faceBusy = useState(null);      // status message string | null
    var faceBusy = _faceBusy[0];
    var setFaceBusy = _faceBusy[1];

    var _addingBox = useState(false);    // "Add face box" drag mode armed
    var addingBox = _addingBox[0];
    var setAddingBox = _addingBox[1];

    var _dragBox = useState(null);       // live drag rect in normalized 0-1 coords
    var dragBox = _dragBox[0];
    var setDragBox = _dragBox[1];
    var dragStartRef = useRef(null);

    // Local face patches applied on top of `detail.faces` so add/remove show
    // instantly — the modal's `detail` may be an external prop we can't re-fetch
    // into. Dedup against detail.faces means pages that DO re-fetch converge
    // (the real row supersedes the optimistic one). Reset on photo change.
    var _removedFaceIds = useState(function () { return {}; });
    var removedFaceIds = _removedFaceIds[0];
    var setRemovedFaceIds = _removedFaceIds[1];

    var _addedFaces = useState(function () { return []; });
    var addedFaces = _addedFaces[0];
    var setAddedFaces = _addedFaces[1];

    // Server-truth face list re-fetched after a mutation, overriding a stale
    // external `detail.faces`. null = fall back to detail.faces. Reset per photo.
    var _liveFaces = useState(null);
    var liveFaces = _liveFaces[0];
    var setLiveFaces = _liveFaces[1];

    // The named description follows the face labels (docs/plans/description-
    // names.md). syncFaces re-fetches the photo after a face edit and keeps
    // the server's rendered text here, so a relabel changes it without
    // reopening the modal. null = use `detail`. Reset per photo.
    var _liveNames = useState(null);
    var liveNames = _liveNames[0];
    var setLiveNames = _liveNames[1];
    // Per-viewer convenience only: show the raw LLM text instead of the named
    // one. localStorage can throw (private window), so wrap every access.
    var _showRaw = useState(function () {
      try { return window.localStorage.getItem('ps.descriptionRaw') === '1'; }
      catch (e) { return false; }
    });
    var showRawDesc = _showRaw[0];
    var setShowRawDesc = _showRaw[1];

    // ---- Image rect for face bounding boxes ----
    var _imgRect = useState(null);
    var imgRect = _imgRect[0];
    var setImgRect = _imgRect[1];
    var imgRef = useRef(null);

    // ---- Collection state ----
    var _photoColls = useState([]);
    var photoColls = _photoColls[0];
    var setPhotoColls = _photoColls[1];

    var _allColls = useState([]);
    var allColls = _allColls[0];
    var setAllColls = _allColls[1];

    var _showCollAdd = useState(false);
    var showCollAdd = _showCollAdd[0];
    var setShowCollAdd = _showCollAdd[1];

    var _newCollName = useState('');
    var newCollName = _newCollName[0];
    var setNewCollName = _newCollName[1];

    var _collBusy = useState(false);
    var collBusy = _collBusy[0];
    var setCollBusy = _collBusy[1];

    // ---- Stack state ----
    var _stackData = useState(null);
    var stackData = _stackData[0];
    var setStackData = _stackData[1];

    var _stackExpanded = useState(false);
    var stackExpanded = _stackExpanded[0];
    var setStackExpanded = _stackExpanded[1];

    var _nearbyStacks = useState(null);
    var nearbyStacks = _nearbyStacks[0];
    var setNearbyStacks = _nearbyStacks[1];

    var _showAddToStack = useState(false);
    var showAddToStack = _showAddToStack[0];
    var setShowAddToStack = _showAddToStack[1];

    var _addingToStack = useState(false);
    var addingToStack = _addingToStack[0];
    var setAddingToStack = _addingToStack[1];

    // Fetch stack members when photo changes
    useEffect(function () {
      setStackData(null);
      setStackExpanded(false);
      setNearbyStacks(null);
      setShowAddToStack(false);
      if (!photo) return;
      // Check if detail or photo has stack info
      var stackInfo = (detail && detail.stack) || null;
      var stackId = stackInfo ? stackInfo.stack_id : (photo.stack_id || null);
      if (!stackId) return;
      fetch(API + '/api/stacks/' + stackId)
        .then(function (r) { return r.json(); })
        .then(function (d) { setStackData(d); })
        .catch(function () {});
    }, [photo && photo.id, detail && detail.stack]);

    // Fetch collections when photo changes (only if showCollections)
    useEffect(function () {
      if (!showCollections || !photo) return;
      fetch(API + '/api/photos/' + photo.id + '/collections')
        .then(function (r) { return r.json(); })
        .then(function (d) { setPhotoColls(d.collections || []); })
        .catch(function () {});
      fetch(API + '/api/collections')
        .then(function (r) { return r.json(); })
        .then(function (d) { setAllColls(d.collections || []); })
        .catch(function () {});
    }, [photo && photo.id, showCollections]);

    // ---- Face editing helpers ----
    useEffect(function () {
      if (editingFace && editMode === 'new' && editInputRef.current) editInputRef.current.focus();
    }, [editingFace, editMode]);

    var cancelEdit = useCallback(function () {
      setEditingFace(null);
      setEditMode('pick');
      setEditName('');
      setEditPick('');
    }, []);

    var updateDetailFace = useCallback(function (faceId, personName) {
      var updater = function (prev) {
        if (!prev || !prev.faces) return prev;
        return Object.assign({}, prev, {
          faces: prev.faces.map(function (f) {
            return f.id === faceId
              ? Object.assign({}, f, { person_name: personName, match_source: personName ? 'manual' : null })
              : f;
          }),
        });
      };
      if (fetchDetailProp) {
        setInternalDetail(updater);
      }
      if (onDetailChanged) onDetailChanged(updater);
      // Keep the local face patches' labels in sync when a face is renamed/unset
      // before a re-fetch replaces them (added boxes + the liveFaces override).
      var relabel = function (f) {
        return f.id === faceId
          ? Object.assign({}, f, { person_name: personName,
                                   match_source: personName ? 'manual'
                                     : (f.match_source === 'manual_box' ? 'manual_box' : null) })
          : f;
      };
      setAddedFaces(function (prev) { return prev.map(relabel); });
      setLiveFaces(function (prev) { return prev ? prev.map(relabel) : prev; });
    }, [fetchDetailProp, onDetailChanged]);

    var saveExistingName = useCallback(function (faceId) {
      if (!editPick) return;
      fetch(API + '/api/faces/' + faceId + '/assign?name=' + encodeURIComponent(editPick), { method: 'POST' })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data.ok) {
            if (onFaceAssigned) onFaceAssigned(faceId, data.person_name);
            updateDetailFace(faceId, data.person_name);
          }
          cancelEdit();
        })
        .catch(function () { cancelEdit(); });
    }, [editPick, onFaceAssigned, cancelEdit, updateDetailFace]);

    var saveNewName = useCallback(function (faceId) {
      var name = editName.trim();
      if (!name) return;
      fetch(API + '/api/faces/' + faceId + '/assign?name=' + encodeURIComponent(name), { method: 'POST' })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data.ok) {
            if (onFaceAssigned) onFaceAssigned(faceId, data.person_name);
            updateDetailFace(faceId, data.person_name);
          }
          cancelEdit();
        })
        .catch(function () { cancelEdit(); });
    }, [editName, onFaceAssigned, cancelEdit, updateDetailFace]);

    var clearFaceName = useCallback(function (faceId) {
      fetch(API + '/api/faces/' + faceId + '/clear', { method: 'POST' })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data.ok) {
            if (onFaceAssigned) onFaceAssigned(faceId, null);
            updateDetailFace(faceId, null);
          }
          cancelEdit();
        })
        .catch(function () { cancelEdit(); });
    }, [onFaceAssigned, cancelEdit, updateDetailFace]);

    // Re-fetch the authoritative face list after a mutation (the endpoints
    // already mirrored to the local DB in replica mode), so re-detect's new
    // faces surface and add/remove reconcile — without a page reload. Overrides
    // a stale external `detail.faces` via `liveFaces`.
    var syncFaces = useCallback(function () {
      if (!photo) return;
      fetch(API + '/api/photos/' + photo.id)
        .then(function (r) { return r.json(); })
        .then(function (d) {
          if (d && d.faces) setLiveFaces(d.faces);
          // description_display is absent when nothing is named (or the flag
          // is off): that is a real answer, so the old named text must go.
          if (d) setLiveNames({
            display: d.description_display || null,
            subjects: d.description_subjects || null });
        })
        .catch(function () {});
    }, [photo]);

    // Delete a face detection box entirely (feature 4). Distinct from Unset,
    // which only removes the person name; this drops the row + encoding.
    var deleteFaceBox = useCallback(function (faceId) {
      setFaceBusy('Removing box…');
      fetch(API + '/api/faces/' + faceId, { method: 'DELETE' })
        .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
        .then(function (res) {
          if (res.ok) {
            // Hide it immediately (works even when `detail` is an external prop);
            // refreshDetail() re-syncs pages that own their detail.
            setRemovedFaceIds(function (prev) {
              var n = Object.assign({}, prev); n[faceId] = true; return n;
            });
            setAddedFaces(function (prev) {
              return prev.filter(function (f) { return f.id !== faceId; });
            });
            setFaceBusy(null);
            cancelEdit();
            syncFaces();
          } else {
            setFaceBusy((res.d && res.d.detail) || 'Remove failed');
          }
        })
        .catch(function (e) { setFaceBusy('Remove error: ' + e); });
    }, [cancelEdit, syncFaces]);

    // Re-run InsightFace on this photo (feature 5). Augment keeps tagged faces
    // and adds only non-overlapping detections; replace wipes first.
    var redetectFaces = useCallback(function (replace) {
      if (!photo) return;
      setFaceBusy(replace ? 'Re-detecting (replace)…' : 'Detecting faces…');
      fetch(API + '/api/photos/' + photo.id + '/detect-faces', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ replace: !!replace }),
      })
        .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
        .then(function (res) {
          if (res.ok) {
            var d = res.d || {};
            setFaceBusy('Added ' + (d.added || 0) + ', kept ' + (d.kept || 0)
              + (d.removed ? ', removed ' + d.removed : '') + '.');
            setRemovedFaceIds({});
            setAddedFaces([]);
            syncFaces();
          } else {
            setFaceBusy((res.d && res.d.detail) || 'Detection failed');
          }
        })
        .catch(function (e) { setFaceBusy('Detect error: ' + e); });
    }, [photo, syncFaces]);

    // Persist a hand-drawn face box (feature 6). bbox is normalized 0-1.
    var addFaceBox = useCallback(function (bbox) {
      if (!photo) return;
      setFaceBusy('Adding box…');
      fetch(API + '/api/photos/' + photo.id + '/add-face-box', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bbox: bbox }),
      })
        .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
        .then(function (res) {
          if (res.ok) {
            var d = res.d || {};
            // Surface the new box immediately with the coords the user drew
            // (the response has no bbox); dedup drops it once a re-fetch lands.
            var iw = (detail && detail.image_width) || 0;
            var ih = (detail && detail.image_height) || 0;
            if (d.face_id && iw && ih) {
              var nf = {
                id: d.face_id,
                person_name: d.person_name || null,
                cluster_id: null,
                match_source: d.person_id ? 'manual' : 'manual_box',
                det_score: d.det_score != null ? d.det_score : null,
                bbox: {
                  top: Math.round(bbox.top * ih),
                  right: Math.round(bbox.right * iw),
                  bottom: Math.round(bbox.bottom * ih),
                  left: Math.round(bbox.left * iw),
                },
              };
              setAddedFaces(function (prev) { return prev.concat([nf]); });
            }
            setFaceBusy(d.encoded ? 'Face added (encoded — matchable).'
              : 'Box added (no face found — assign a name manually).');
            syncFaces();
          } else {
            setFaceBusy((res.d && res.d.detail) || 'Add box failed');
          }
        })
        .catch(function (e) { setFaceBusy('Add box error: ' + e); });
    }, [photo, detail, syncFaces]);

    // ---- Collection helpers ----
    var refreshPhotoColls = useCallback(function () {
      if (!photo) return;
      fetch(API + '/api/photos/' + photo.id + '/collections')
        .then(function (r) { return r.json(); })
        .then(function (d) { setPhotoColls(d.collections || []); })
        .catch(function () {});
    }, [photo]);

    var addToCollection = useCallback(function (collId) {
      setCollBusy(true);
      fetch(API + '/api/collections/' + collId + '/photos', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ photo_ids: [photo.id] }),
      })
        .then(function (r) { return r.json(); })
        .then(function () {
          refreshPhotoColls();
          setShowCollAdd(false);
          setCollBusy(false);
        })
        .catch(function () { setCollBusy(false); });
    }, [photo, refreshPhotoColls]);

    var createAndAdd = useCallback(function () {
      var name = newCollName.trim();
      if (!name) return;
      setCollBusy(true);
      fetch(API + '/api/collections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name, photo_ids: [photo.id] }),
      })
        .then(function (r) { return r.json(); })
        .then(function () {
          refreshPhotoColls();
          fetch(API + '/api/collections')
            .then(function (r) { return r.json(); })
            .then(function (d) { setAllColls(d.collections || []); })
            .catch(function () {});
          setNewCollName('');
          setShowCollAdd(false);
          setCollBusy(false);
        })
        .catch(function () { setCollBusy(false); });
    }, [photo, newCollName, refreshPhotoColls]);

    var removeFromCollection = useCallback(function (collId) {
      setCollBusy(true);
      fetch(API + '/api/collections/' + collId + '/photos/remove', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ photo_ids: [photo.id] }),
      })
        .then(function (r) { return r.json(); })
        .then(function () {
          refreshPhotoColls();
          setCollBusy(false);
        })
        .catch(function () { setCollBusy(false); });
    }, [photo, refreshPhotoColls]);

    // ---- Image rect calculation for face bounding boxes ----
    var updateImgRect = useCallback(function () {
      var img = imgRef.current;
      if (!img || !img.naturalWidth) return;
      var container = img.parentElement;
      var contRect = container.getBoundingClientRect();
      var imgElRect = img.getBoundingClientRect();
      var elOx = imgElRect.left - contRect.left;
      var elOy = imgElRect.top - contRect.top;
      var cw = img.clientWidth, ch = img.clientHeight;
      var nw = img.naturalWidth, nh = img.naturalHeight;
      var scale = Math.min(cw / nw, ch / nh);
      var dw = nw * scale, dh = nh * scale;
      var fitOx = (cw - dw) / 2, fitOy = (ch - dh) / 2;
      setImgRect({ ox: elOx + fitOx, oy: elOy + fitOy, dw: dw, dh: dh, nw: nw, nh: nh });
    }, []);

    useEffect(function () {
      window.addEventListener('resize', updateImgRect);
      return function () { window.removeEventListener('resize', updateImgRect); };
    }, [updateImgRect]);

    // Clear stale imgRect when photo changes so face overlays don't flash
    useEffect(function () {
      setImgRect(null);
      setAddingBox(false);
      setDragBox(null);
      setFaceBusy(null);
      setRemovedFaceIds({});
      setAddedFaces([]);
      setLiveFaces(null);
      setLiveNames(null);
      dragStartRef.current = null;
    }, [photo && photo.id]);

    // ---- Drag-to-add-face-box handlers (feature 6) ----
    function _evToNorm(ev) {
      if (!imgRect) return null;
      var cont = ev.currentTarget.getBoundingClientRect();
      var x = (ev.clientX - cont.left - imgRect.ox) / imgRect.dw;
      var y = (ev.clientY - cont.top - imgRect.oy) / imgRect.dh;
      return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
    }
    function onImgMouseDown(ev) {
      if (!addingBox || !imgRect) return;
      ev.preventDefault();
      var p = _evToNorm(ev);
      dragStartRef.current = p;
      setDragBox({ left: p.x, top: p.y, right: p.x, bottom: p.y });
    }
    function onImgMouseMove(ev) {
      if (!addingBox || !dragStartRef.current) return;
      var s = dragStartRef.current, p = _evToNorm(ev);
      setDragBox({ left: Math.min(s.x, p.x), top: Math.min(s.y, p.y),
                   right: Math.max(s.x, p.x), bottom: Math.max(s.y, p.y) });
    }
    function onImgMouseUp(ev) {
      if (!addingBox || !dragStartRef.current) return;
      var s = dragStartRef.current, p = _evToNorm(ev);
      dragStartRef.current = null;
      setDragBox(null);
      setAddingBox(false);
      var box = { left: Math.min(s.x, p.x), top: Math.min(s.y, p.y),
                  right: Math.max(s.x, p.x), bottom: Math.max(s.y, p.y) };
      if ((box.right - box.left) > 0.02 && (box.bottom - box.top) > 0.02) {
        addFaceBox({ top: box.top, right: box.right, bottom: box.bottom, left: box.left });
      } else {
        setFaceBusy('Box too small — cancelled.');
      }
    }

    // detail.faces with local add/remove patches applied (see the patch state
    // above). Added faces not yet present in detail are appended; deduped by id
    // so a page that re-fetches converges without doubling the row.
    var visibleFaces = useMemo(function () {
      var src = liveFaces != null ? liveFaces : ((detail && detail.faces) || []);
      var base = src.filter(function (f) { return !removedFaceIds[f.id]; });
      var seen = {};
      base.forEach(function (f) { seen[f.id] = 1; });
      var extra = addedFaces.filter(function (f) {
        return !seen[f.id] && !removedFaceIds[f.id];
      });
      return base.concat(extra);
    }, [detail, liveFaces, removedFaceIds, addedFaces]);

    // Build face bounding box overlay data
    var faceOverlays = useMemo(function () {
      if (!detail || !imgRect) return [];
      // Use original image dimensions for bbox scaling.
      // imgRect.nw/nh is the PREVIEW image size (≤1920px), but bbox coords are
      // stored in original image space (e.g. 4608px). Using detail.image_width
      // gives the correct scale factor regardless of which image tier is served.
      var origW = (detail && detail.image_width) ? detail.image_width : imgRect.nw;
      var origH = (detail && detail.image_height) ? detail.image_height : imgRect.nh;
      return visibleFaces.filter(function (f) { return f.bbox; }).map(function (f) {
        var bbox = f.bbox;
        var sx = imgRect.dw / origW;
        var sy = imgRect.dh / origH;
        return {
          id: f.id,
          label: f.person_name || ('Unknown #' + (f.cluster_id || '?')),
          style: {
            left: imgRect.ox + bbox.left * sx,
            top: imgRect.oy + bbox.top * sy,
            width: (bbox.right - bbox.left) * sx,
            height: (bbox.bottom - bbox.top) * sy,
          },
        };
      });
    }, [detail, imgRect, visibleFaces]);

    // ---- Keyboard navigation ----
    useEffect(function () {
      var handler = function (ev) {
        if (ev.key === 'Escape') onClose();
        if (ev.key === 'ArrowLeft' && onPrev) { ev.preventDefault(); onPrev(); }
        if (ev.key === 'ArrowRight' && onNext) { ev.preventDefault(); onNext(); }
      };
      window.addEventListener('keydown', handler);
      return function () { window.removeEventListener('keydown', handler); };
    }, [onClose, onPrev, onNext]);

    // ---- Touch swipe navigation (mobile) ----
    // Attach to the modal overlay element (via ref) so swipes are scoped to the
    // modal and don't compete with Safari's edge-swipe back gesture.
    var overlayRef = useRef(null);
    var touchRef = useRef(null);
    useEffect(function () {
      var el = overlayRef.current;
      if (!el) return;
      var onTouchStart = function (ev) {
        var t = ev.touches[0];
        touchRef.current = { x: t.clientX, y: t.clientY };
      };
      var onTouchMove = function (ev) {
        // Once we detect a horizontal swipe, prevent vertical scroll so the
        // gesture feels intentional and the page doesn't bounce.
        if (!touchRef.current) return;
        var t = ev.touches[0];
        var dx = Math.abs(t.clientX - touchRef.current.x);
        var dy = Math.abs(t.clientY - touchRef.current.y);
        if (dx > 10 && dx > dy) ev.preventDefault();
      };
      var onTouchEnd = function (ev) {
        if (!touchRef.current) return;
        var t = ev.changedTouches[0];
        var dx = t.clientX - touchRef.current.x;
        var dy = t.clientY - touchRef.current.y;
        touchRef.current = null;
        // Require horizontal swipe > 50px and more horizontal than vertical
        if (Math.abs(dx) < 50 || Math.abs(dy) > Math.abs(dx)) return;
        if (dx < 0 && onNext) onNext();   // swipe left → next
        if (dx > 0 && onPrev) onPrev();   // swipe right → prev
      };
      el.addEventListener('touchstart', onTouchStart, { passive: true });
      el.addEventListener('touchmove', onTouchMove, { passive: false });
      el.addEventListener('touchend', onTouchEnd, { passive: true });
      return function () {
        el.removeEventListener('touchstart', onTouchStart);
        el.removeEventListener('touchmove', onTouchMove);
        el.removeEventListener('touchend', onTouchEnd);
      };
    }, [onPrev, onNext]);

    if (!photo) return null;

    // ---- Render helpers ----
    function renderFaceTag(f) {
      if (editingFace === f.id) {
        return e('div', { key: f.id, className: 'face-edit-panel' },
          e('div', { style: { fontWeight: 600, marginBottom: 2 } },
            f.person_name || ('Unknown #' + (f.cluster_id || '?'))),
          f.det_score != null && e('div', { style: { fontSize: 10,
            color: f.det_score < 0.65 ? '#fb923c' : 'var(--text-muted)', marginBottom: 4 } },
            'detection ' + f.det_score.toFixed(2)
              + (f.det_score < 0.65 ? ' · low confidence' : '')
              + (f.match_source === 'manual_box' ? ' · hand-drawn' : '')),

          // Pick existing person
          editMode === 'pick' && e('select', {
            value: editPick,
            onChange: function (ev) { setEditPick(ev.target.value); },
            autoFocus: true,
          },
            e('option', { value: '' }, 'Choose existing person\u2026'),
            persons.map(function (p) {
              return e('option', { key: p.id, value: p.name }, p.name);
            }),
          ),
          editMode === 'pick' && e('div', { className: 'face-edit-buttons' },
            e('button', {
              className: 'primary',
              onClick: function () { saveExistingName(f.id); },
              disabled: !editPick,
              style: !editPick ? { opacity: 0.5, cursor: 'default' } : {},
            }, 'Apply'),
            e('button', { onClick: function () { setEditMode('new'); setEditName(''); } }, 'New Name'),
            f.person_name && e('button', {
              className: 'danger',
              onClick: function () { clearFaceName(f.id); },
            }, 'Unset'),
            e('button', {
              className: 'danger',
              title: 'Delete this detection box entirely',
              onClick: function () { deleteFaceBox(f.id); },
            }, 'Remove box'),
            e('button', { onClick: cancelEdit }, 'Cancel'),
          ),

          // New name text input
          editMode === 'new' && e('input', {
            ref: editInputRef,
            type: 'text',
            value: editName,
            placeholder: 'Enter new name\u2026',
            onChange: function (ev) { setEditName(ev.target.value); },
            onKeyDown: function (ev) {
              if (ev.key === 'Enter') saveNewName(f.id);
              if (ev.key === 'Escape') cancelEdit();
            },
          }),
          editMode === 'new' && e('div', { className: 'face-edit-buttons' },
            e('button', {
              className: 'primary',
              onClick: function () { saveNewName(f.id); },
              disabled: !editName.trim(),
              style: !editName.trim() ? { opacity: 0.5, cursor: 'default' } : {},
            }, 'Save'),
            e('button', { onClick: function () { setEditMode('pick'); } }, 'Back'),
            e('button', { onClick: cancelEdit }, 'Cancel'),
          ),
        );
      }
      // Normal face tag
      return e('span', {
        key: f.id,
        className: 'face-tag' + (hoveredFace === f.id ? ' active' : ''),
        onMouseEnter: function () { setHoveredFace(f.id); },
        onMouseLeave: function () { setHoveredFace(null); },
        onClick: function () {
          setEditingFace(f.id);
          setEditMode('pick');
          setEditPick('');
          setEditName('');
        },
        title: 'Click to change',
      },
        f.person_name || ('Unknown #' + (f.cluster_id || '?')),
        f.match_source && e('span', { className: 'match-badge' },
          f.match_source === 'temporal' ? '\u23F1' : f.match_source === 'strict' ? '\u2713'
          : f.match_source === 'manual' ? '\u270E' : ''),
      );
    }

    function renderAesthetics() {
      var s = (detail && detail.aesthetic_score != null) ? detail.aesthetic_score
            : (photo.aesthetic_score != null ? photo.aesthetic_score : null);
      if (s == null) return null;

      var color = s >= 6 ? '#4ade80' : s >= 4.5 ? '#facc15' : '#f87171';
      var label = s >= 7 ? 'Excellent' : s >= 6 ? 'Good' : s >= 4.5 ? 'Average' : 'Below average';
      var concepts = detail && detail.aesthetic_concepts;
      var critique = detail && detail.aesthetic_critique;

      return e('div', { className: 'detail-section' },
        e('h3', null, 'Aesthetic Quality'),
        e('div', null,
          e('div', { style: { display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 } },
            e('span', { style: { fontSize: 24, fontWeight: 600, color: color } }, s.toFixed(1)),
            e('span', { style: { fontSize: 13, color: 'var(--text-muted)' } }, '/ 10 \u00B7 ' + label),
          ),
          e('div', { style: {
            height: 6, background: 'var(--surface2)', borderRadius: 3, overflow: 'hidden',
            marginBottom: (concepts || critique) ? 10 : 0,
          } },
            e('div', { style: {
              width: (s * 10) + '%', height: '100%', background: color, borderRadius: 3,
            } }),
          ),
          concepts && concepts.strengths && concepts.strengths.length > 0 &&
            e('div', { style: { fontSize: 13, marginBottom: 4 } },
              e('span', { style: { color: '#4ade80' } }, '\u25B2 '),
              e('span', { style: { color: 'var(--text-muted)' } }, 'Strengths: '),
              concepts.strengths.join(', '),
            ),
          concepts && concepts.weaknesses && concepts.weaknesses.length > 0 &&
            e('div', { style: { fontSize: 13, marginBottom: 4 } },
              e('span', { style: { color: '#f87171' } }, '\u25BC '),
              e('span', { style: { color: 'var(--text-muted)' } }, 'Weaknesses: '),
              concepts.weaknesses.join(', '),
            ),
          critique &&
            e('p', { style: { fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.5, marginTop: 6,
                               fontStyle: 'italic' } },
              critique,
            ),
        ),
      );
    }

    // M-aesthetics: rich VLM per-attribute breakdown (detail.aesthetics).
    var DIM_LABELS = { technical: 'Technical Excellence', composition: 'Composition',
                       impact: 'Impact & Storytelling' };
    function _scoreColor(v) {
      return v == null ? 'var(--text-muted)' : v >= 7.5 ? '#4ade80'
           : v >= 5.5 ? '#facc15' : v >= 3.5 ? '#fb923c' : '#f87171';
    }
    function renderVlmAesthetics() {
      var a = detail && detail.aesthetics;
      if (!a) return null;
      var pct = a.overall_pct;
      var overall = a.overall;
      var bars = Object.keys(DIM_LABELS).map(function (dim) {
        var d = (a.dimensions && a.dimensions[dim]) || {};
        var sc = d.score;
        var subs = d.subs || {};
        return e('div', { key: dim, style: { marginBottom: 8 } },
          e('div', { style: { display: 'flex', justifyContent: 'space-between',
                              alignItems: 'baseline', fontSize: 13, marginBottom: 3 } },
            e('span', null, DIM_LABELS[dim]),
            e('span', { style: { fontWeight: 600, color: _scoreColor(sc) } },
              sc == null ? '–' : sc.toFixed(1)),
          ),
          e('div', { style: { height: 5, background: 'var(--surface2)', borderRadius: 3,
                              overflow: 'hidden' } },
            e('div', { style: { width: ((sc || 0) * 10) + '%', height: '100%',
                                background: _scoreColor(sc), borderRadius: 3 } })),
          d.critique && e('div', { style: { fontSize: 11, color: 'var(--text-muted)',
                                            fontStyle: 'italic', marginTop: 2 } }, d.critique),
          e('div', { style: { fontSize: 11, color: 'var(--text-muted)', marginTop: 2 } },
            Object.keys(subs).map(function (k) {
              return k.replace(/_/g, ' ') + ' ' + (subs[k] == null ? '–' : subs[k].toFixed(0));
            }).join(' · ')),
        );
      });
      var facets = a.style || {};
      var facetRows = Object.keys(facets).filter(function (k) { return facets[k]; }).map(function (k) {
        return e('div', { key: k, style: { fontSize: 12, marginBottom: 2 } },
          e('span', { style: { color: 'var(--text-muted)' } }, k.replace(/_/g, ' ') + ': '),
          facets[k]);
      });
      return e('div', { className: 'detail-section' },
        e('h3', null, 'Aesthetic Evaluation'),
        (function () {
          // Full-frame vs subject scores, side by side (v27). The subject-crop
          // score judges the primary subject rather than the background; search
          // ranks on the subject score when present, else the full-frame score
          // (COALESCE(aes_subject_overall_pct, aes_overall_pct)) — the RANKED
          // badge marks whichever one that is.
          var raw = (detail && detail.aes_raw) || {};
          var so = raw.aes_subject_overall, sp = raw.aes_subject_overall_pct;
          var boxes = (detail && detail.subject_boxes) || null;
          var subjScored = so != null;
          var subjLabels = (boxes && boxes.length)
            ? boxes.map(function (bx) { return bx.label || 'subject'; }).join(', ')
            : null;
          var LBL = { fontSize: 11, color: 'var(--text-muted)', width: 74,
                      textTransform: 'uppercase', letterSpacing: '0.04em' };
          function rankBadge() {
            return e('span', { title: 'Score search ranks this photo on',
              style: { fontSize: 10, fontWeight: 700, marginLeft: 6, padding: '1px 6px',
                       borderRadius: 8, background: '#2563eb', color: '#fff',
                       verticalAlign: 'middle', letterSpacing: '0.04em' } }, 'RANKED');
          }
          return e('div', null,
            // Full-frame headline
            e('div', { style: { display: 'flex', alignItems: 'baseline', gap: 8,
                                marginBottom: (subjScored || subjLabels) ? 6 : 10 } },
              e('span', { style: LBL }, 'Full frame'),
              e('span', { style: { fontSize: 26, fontWeight: 700, color: _scoreColor(overall) } },
                pct == null ? (overall == null ? '–' : overall.toFixed(1)) : Math.round(pct)),
              e('span', { style: { fontSize: 12, color: 'var(--text-muted)' } },
                pct == null ? '/ 10'
                  : 'pct' + (overall != null ? ' · ' + overall.toFixed(1) + '/10' : '')),
              !subjScored && rankBadge(),
            ),
            // Subject score (only when the subject was separately crop-scored)
            subjScored && e('div', { style: { display: 'flex', alignItems: 'baseline',
                                              gap: 8, marginBottom: 10 } },
              e('span', { style: LBL }, 'Subject'),
              e('span', { style: { fontSize: 26, fontWeight: 700, color: _scoreColor(so) } },
                sp != null ? Math.round(sp) : so.toFixed(1)),
              e('span', { style: { fontSize: 12, color: 'var(--text-muted)' } },
                (sp != null ? 'pct · ' : '') + so.toFixed(1) + '/10'
                  + (subjLabels ? ' · ' + subjLabels : '')),
              rankBadge(),
            ),
            // Grounded, but the subject fills the frame → no separate crop score.
            !subjScored && subjLabels && e('div', { style: { fontSize: 12,
              color: 'var(--text-muted)', marginBottom: 10 } },
              'Subject: ' + subjLabels + ' — fills the frame, scored on the full frame.'),
            // Grounded, no distinct subject (landscape).
            !subjScored && boxes && boxes.length === 0 && e('div', { style: { fontSize: 12,
              color: 'var(--text-muted)', marginBottom: 10 } },
              'No distinct subject — scored on the full frame.'),
          );
        })(),
        // Per-day rank (v28): how this photo ranks among others taken the same
        // day — the ranking the "Best of day" sort uses.
        (function () {
          var raw = (detail && detail.aes_raw) || {};
          var d = raw.aes_subject_overall_day_pct != null ? raw.aes_subject_overall_day_pct
                : raw.aes_overall_day_pct;
          if (d == null) return null;
          return e('div', { style: { fontSize: 12, color: 'var(--text-muted)', marginBottom: 10 } },
            e('span', { style: { textTransform: 'uppercase', letterSpacing: '0.04em' } }, 'Best of day'),
            ': ' + Math.round(d) + ' percentile');
        })(),
        bars,
        facetRows.length > 0 && e('div', { style: { marginTop: 8 } },
          e('div', { style: { fontSize: 12, fontWeight: 600, marginBottom: 3 } }, 'Style'),
          facetRows),
        a.style_tags && a.style_tags.length > 0 && e('div', {
          style: { marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 4 } },
          a.style_tags.map(function (t, i) {
            return e('span', { key: i, style: {
              fontSize: 11, padding: '2px 7px', borderRadius: 10,
              background: 'var(--surface2)', color: 'var(--text-muted)' } }, t);
          })),
        a.model && e('div', { style: { fontSize: 10, color: 'var(--text-muted)', marginTop: 8 } },
          'model: ' + a.model),
      );
    }

    function renderCamera() {
      var d = detail;
      if (!d) return null;
      var hasCamera = d.camera_model || d.focal_length || d.f_number || d.iso || d.image_width;
      if (!hasCamera) return null;

      return e('div', { className: 'detail-section' },
        e('h3', null, 'Camera'),
        d.camera_model && e('div', { className: 'detail-row' },
          e('span', { className: 'label' }, 'Model'),
          e('span', null, d.camera_model),
        ),
        d.focal_length && e('div', { className: 'detail-row' },
          e('span', { className: 'label' }, 'Focal Length'),
          e('span', null, PS.formatFocalLength(d.focal_length)),
        ),
        d.exposure_time && e('div', { className: 'detail-row' },
          e('span', { className: 'label' }, 'Exposure'),
          e('span', null, d.exposure_time),
        ),
        d.f_number && e('div', { className: 'detail-row' },
          e('span', { className: 'label' }, 'Aperture'),
          e('span', null, 'f/' + PS.formatFNumber(d.f_number)),
        ),
        d.iso && e('div', { className: 'detail-row' },
          e('span', { className: 'label' }, 'ISO'),
          e('span', null, d.iso),
        ),
        d.image_width && e('div', { className: 'detail-row' },
          e('span', { className: 'label' }, 'Resolution'),
          e('span', null, d.image_width + ' \u00D7 ' + d.image_height),
        ),
      );
    }

    function renderColors() {
      var colors = (detail && detail.colors) || photo.colors;
      if (!colors || colors.length === 0) return null;

      return e('div', { className: 'detail-section' },
        e('h3', null, 'Dominant Colors'),
        e('div', { className: 'color-dots', style: { gap: 6 } },
          colors.map(function (c, i) {
            return e('span', {
              key: i, className: 'color-dot',
              style: { background: c, width: 20, height: 20 },
              title: c,
            });
          }),
        ),
      );
    }

    function renderCollections() {
      if (!showCollections) return null;

      return e('div', { className: 'detail-section' },
        e('h3', null, 'Collections'),
        photoColls.length > 0
          ? e('div', { style: { display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 } },
              photoColls.map(function (c) {
                return e('span', {
                  key: c.id,
                  style: {
                    display: 'inline-flex', alignItems: 'center', gap: 4,
                    background: 'var(--surface2)', borderRadius: 4, padding: '2px 8px',
                    fontSize: 13, cursor: 'pointer',
                  },
                  title: 'Click to remove from collection',
                  onClick: function () { removeFromCollection(c.id); },
                },
                  c.name,
                  e('span', { style: { opacity: 0.5, marginLeft: 2 } }, '\u00D7'),
                );
              }),
            )
          : e('div', { style: { color: 'var(--text-muted)', fontSize: 13, marginBottom: 8 } }, 'Not in any collection'),

        !showCollAdd
          ? e('button', {
              style: { fontSize: 13, padding: '4px 10px', cursor: 'pointer',
                       background: 'var(--surface2)', border: '1px solid var(--border)',
                       borderRadius: 4, color: 'var(--text)' },
              onClick: function () { setShowCollAdd(true); },
            }, '+ Add to Collection')
          : e('div', { style: { display: 'flex', flexDirection: 'column', gap: 6 } },
              // Existing collections not already assigned
              allColls.filter(function (c) { return !photoColls.find(function (pc) { return pc.id === c.id; }); }).length > 0 &&
                e('select', {
                  style: { fontSize: 13, padding: '4px 6px', background: 'var(--bg)',
                           color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 4 },
                  value: '',
                  disabled: collBusy,
                  onChange: function (ev) { if (ev.target.value) addToCollection(parseInt(ev.target.value)); },
                },
                  e('option', { value: '' }, 'Add to existing\u2026'),
                  allColls
                    .filter(function (c) { return !photoColls.find(function (pc) { return pc.id === c.id; }); })
                    .map(function (c) { return e('option', { key: c.id, value: c.id }, c.name); }),
                ),
              // Create new
              e('div', { style: { display: 'flex', gap: 4 } },
                e('input', {
                  type: 'text',
                  placeholder: 'New collection\u2026',
                  value: newCollName,
                  disabled: collBusy,
                  style: { flex: 1, fontSize: 13, padding: '4px 6px', background: 'var(--bg)',
                           color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 4 },
                  onChange: function (ev) { setNewCollName(ev.target.value); },
                  onKeyDown: function (ev) {
                    if (ev.key === 'Enter') createAndAdd();
                    if (ev.key === 'Escape') setShowCollAdd(false);
                  },
                }),
                e('button', {
                  style: { fontSize: 13, padding: '4px 8px', cursor: 'pointer',
                           background: 'var(--accent)', border: 'none', borderRadius: 4, color: '#fff' },
                  disabled: collBusy || !newCollName.trim(),
                  onClick: createAndAdd,
                }, 'Create'),
              ),
              e('button', {
                style: { fontSize: 13, padding: '2px 8px', cursor: 'pointer', alignSelf: 'flex-start',
                         background: 'transparent', border: 'none', color: 'var(--text-muted)' },
                onClick: function () { setShowCollAdd(false); setNewCollName(''); },
              }, 'Cancel'),
            ),
      );
    }

    // ===== Main render =====
    return e('div', { className: 'modal-overlay', ref: overlayRef, onClick: function (ev) {
      if (ev.target === ev.currentTarget) onClose();
    } },
      e('button', { className: 'modal-close', onClick: onClose }, '\u00D7'),
      total > 1 && onPrev && e('button', { className: 'modal-nav prev', onClick: function (ev) { ev.stopPropagation(); onPrev(); }, title: 'Previous (\u2190)' }, '\u2039'),
      total > 1 && onNext && e('button', { className: 'modal-nav next', onClick: function (ev) { ev.stopPropagation(); onNext(); }, title: 'Next (\u2192)' }, '\u203A'),
      e('div', { className: 'modal' },
        // Image pane
        e('div', { className: 'modal-image',
                   style: { position: 'relative', cursor: addingBox ? 'crosshair' : 'default' },
                   onMouseDown: onImgMouseDown,
                   onMouseMove: onImgMouseMove,
                   onMouseUp: onImgMouseUp,
                   onMouseLeave: function () { if (addingBox) { dragStartRef.current = null; setDragBox(null); } } },
          isVideo && videoPlayable
            ? e('video', {
                key: 'video-' + photo.id + '-' + videoChoice.mode,
                ref: videoRef,
                src: videoChoice.src,
                poster: API + '/api/photos/' + photo.id + '/preview',
                controls: true,
                preload: 'metadata',
                playsInline: true,
                onError: videoChoice.mode === 'original'
                  ? function () { setVideoErrored(true); } : undefined,
                style: { maxWidth: '100%', maxHeight: '100%', background: '#000' },
              })
            : e('img', {
                key: 'photo-' + photo.id,
                ref: imgRef,
                src: API + '/api/photos/' + photo.id + '/preview',
                alt: photo.filename,
                onLoad: updateImgRect,
                draggable: false,
              }),
          // Nothing playable yet (HEVC / avi / mts and no proxy made): say
          // so and offer the original instead of a broken player.
          isVideo && !videoPlayable && e('div', {
            style: { position: 'absolute', left: 0, right: 0, bottom: 16,
                     textAlign: 'center' },
          },
            e('a', {
              href: API + '/api/photos/' + photo.id + '/full?download=1',
              download: photo.filename || '',
              style: { padding: '6px 14px', borderRadius: 6, background: 'var(--surface2)',
                       color: 'var(--text)', border: '1px solid var(--border)',
                       textDecoration: 'none', fontSize: 13 },
            }, (videoInfo && videoInfo.needs_proxy === false
                  ? '\u2193 This browser cannot play this video \u2014 download '
                  : 'Preparing a playable copy \u2014 or \u2193 download ') +
               (photo.filename || 'the original')),
          ),
          // Playing the 720p copy: the original stays one click away.
          isVideo && videoPlayable && videoChoice.mode === 'proxy' && e('a', {
            href: API + '/api/photos/' + photo.id + '/full?download=1',
            download: photo.filename || '',
            title: 'Playing a 720p H.264 copy; this downloads the original file',
            style: { position: 'absolute', right: 8, top: 8, padding: '2px 8px',
                     borderRadius: 4, background: 'rgba(0,0,0,0.55)', color: '#ddd',
                     textDecoration: 'none', fontSize: 11 },
          }, '720p copy \u00b7 \u2193 original'),
          // Face bounding box overlays (hovered face, or ALL while arming a box
          // so the user sees what already exists before drawing a new one).
          // None on a video: the boxes belong to one frame, not the clip.
          (isVideo ? [] : faceOverlays).filter(function (ov) { return addingBox || hoveredFace === ov.id; }).map(function (ov) {
            var active = hoveredFace === ov.id;
            return e('div', {
              key: 'bbox-' + ov.id,
              className: 'face-bbox-overlay',
              style: Object.assign({}, ov.style,
                { borderColor: '#4ade80', opacity: (addingBox && !active) ? 0.4 : 1 }),
            },
              e('span', {
                className: 'face-bbox-label',
                style: { background: '#4ade80', color: '#000' },
              }, ov.label),
            );
          }),
          // Live drag rectangle while drawing a new box.
          addingBox && dragBox && imgRect && e('div', {
            className: 'face-bbox-overlay',
            style: {
              position: 'absolute',
              left: imgRect.ox + dragBox.left * imgRect.dw,
              top: imgRect.oy + dragBox.top * imgRect.dh,
              width: (dragBox.right - dragBox.left) * imgRect.dw,
              height: (dragBox.bottom - dragBox.top) * imgRect.dh,
              border: '2px dashed #60a5fa', background: 'rgba(96,165,250,0.15)',
            },
          }),
        ),

        // Sidebar
        e('div', { className: 'modal-sidebar' },
          // Header children slot (e.g. review's select/deselect button)
          headerChildren,

          // Photo info
          e('div', { className: 'detail-section' },
            e('h3', null, 'Photo'),
            e('p', { style: { fontWeight: 600, fontSize: 16 } }, photo.filename),
            detail && detail.filepath && e('p', { style: { color: 'var(--text-muted)', fontSize: 11, wordBreak: 'break-all', marginTop: 2 } }, detail.filepath),
            // Download the original file. ?download=1 makes the server send it as
            // an attachment, so this works even when the `download` attr is
            // ignored (e.g. a cross-origin replica host).
            e('a', {
              href: API + '/api/photos/' + photo.id + '/full?download=1',
              download: photo.filename || '',
              title: 'Download the original file',
              style: {
                display: 'inline-flex', alignItems: 'center', gap: 6, marginTop: 8,
                padding: '5px 12px', fontSize: 13, borderRadius: 6, width: 'fit-content',
                background: 'var(--surface2)', color: 'var(--text)',
                border: '1px solid var(--border)', textDecoration: 'none',
              },
            }, '↓ Download original'),
            e('div', { style: { display: 'flex', gap: 12, alignItems: 'center' } },
              photo.date_taken && e('span', { style: { color: 'var(--text-muted)', fontSize: 13 } }, photo.date_taken),
              total > 1 && e('span', { style: { color: 'var(--text-muted)', fontSize: 13 } }, (index + 1) + ' / ' + total),
            ),
          ),

          // Video main-subject timeline (step 6) — an audit tool, always on.
          isVideo && e(PS.VideoSubjectStrip, { key: 'subj-' + photo.id, photoId: photo.id, videoRef: videoRef }),

          // Video quality numbers (step 7) — numbers only, no tags.
          isVideo && e(PS.VideoQualityBlock, { key: 'vq-' + photo.id,
                                               quality: detail && detail.video && detail.video.quality }),

          // Stack
          stackData && stackData.members && stackData.members.length > 1 && e('div', { className: 'detail-section' },
            e('h3', {
              style: { cursor: 'pointer', userSelect: 'none' },
              onClick: function () { setStackExpanded(!stackExpanded); },
            }, 'Stack (\u00d7' + stackData.members.length + ') ', e('span', { style: { fontSize: 11 } }, stackExpanded ? '\u25B2' : '\u25BC')),
            stackExpanded && e('div', { style: { display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 6 } },
              stackData.members.slice().sort(function (a, b) {
                var da = a.date_taken || a.filename;
                var db = b.date_taken || b.filename;
                return da < db ? -1 : da > db ? 1 : 0;
              }).map(function (m) {
                var isCurrent = m.id === photo.id;
                var isTop = m.is_top;
                return e('div', {
                  key: m.id,
                  style: {
                    width: 64, height: 64, borderRadius: 4, overflow: 'hidden',
                    cursor: isCurrent ? 'default' : 'pointer',
                    border: isCurrent ? '2px solid var(--accent)' : isTop ? '2px solid #4ade80' : '2px solid transparent',
                    position: 'relative', opacity: isCurrent ? 1 : 0.8,
                  },
                  title: m.filename + (isTop ? ' (top)' : '') + (m.aesthetic_score ? ' \u2605' + m.aesthetic_score.toFixed(1) : ''),
                  onClick: function () {
                    if (isCurrent) return;
                    // Navigate within the modal by calling onStackNavigate if provided,
                    // otherwise swap the photo in-place by triggering detail fetch
                    if (props.onStackNavigate) {
                      props.onStackNavigate(m);
                    }
                  },
                },
                  e('img', {
                    src: API + '/api/photos/' + m.id + '/thumbnail?v=' + encodeURIComponent(m.filename),
                    style: { width: '100%', height: '100%', objectFit: 'cover' },
                  }),
                  isTop && e('div', {
                    style: { position: 'absolute', bottom: 0, left: 0, right: 0, background: 'rgba(0,0,0,0.6)', color: '#4ade80', fontSize: 9, textAlign: 'center', padding: '1px 0' },
                  }, 'TOP'),
                );
              }),
            ),
            stackExpanded && e('div', { style: { marginTop: 6, display: 'flex', gap: 6, flexWrap: 'wrap' } },
              // Show "Make this top" only if current photo is NOT already the top
              !stackData.members.some(function (m) { return m.id === photo.id && m.is_top; }) &&
              e('button', {
                style: { fontSize: 11, padding: '3px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--surface2)', color: 'var(--text)', cursor: 'pointer' },
                onClick: function () {
                  fetch(API + '/api/stacks/' + stackData.id + '/top', {
                    method: 'PUT', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ photo_id: photo.id }),
                  }).then(function () {
                    fetch(API + '/api/stacks/' + stackData.id).then(function (r) { return r.json(); }).then(setStackData);
                  });
                },
              }, 'Make this top'),
              e('button', {
                style: { fontSize: 11, padding: '3px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--surface2)', color: 'var(--text)', cursor: 'pointer' },
                onClick: function () {
                  if (!confirm('Remove this photo from the stack?')) return;
                  fetch(API + '/api/photos/' + photo.id + '/unstack', { method: 'POST' })
                    .then(function () { setStackData(null); });
                },
              }, 'Unstack this'),
            ),
          ),

          // Add to Stack (shown for unstacked photos)
          !stackData && photo && e('div', { className: 'detail-section' },
            !showAddToStack && e('button', {
              style: { fontSize: 12, padding: '4px 10px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--surface2)', color: 'var(--text)', cursor: 'pointer' },
              onClick: function () {
                setShowAddToStack(true);
                setNearbyStacks(null);
                fetch(API + '/api/photos/' + photo.id + '/nearby-stacks')
                  .then(function (r) { return r.json(); })
                  .then(function (d) { setNearbyStacks(d.stacks || []); })
                  .catch(function () { setNearbyStacks([]); });
              },
            }, 'Add to Stack'),
            showAddToStack && e('div', null,
              e('h3', null, 'Add to Stack'),
              nearbyStacks === null && e('div', { style: { fontSize: 12, color: 'var(--text-muted)', padding: '4px 0' } }, 'Finding nearby stacks\u2026'),
              nearbyStacks && nearbyStacks.length === 0 && e('div', { style: { fontSize: 12, color: 'var(--text-muted)', padding: '4px 0' } }, 'No nearby stacks found.'),
              nearbyStacks && nearbyStacks.length > 0 && e('div', { style: { display: 'flex', flexDirection: 'column', gap: 4 } },
                nearbyStacks.map(function (ns) {
                  return e('div', {
                    key: ns.stack_id,
                    style: {
                      display: 'flex', alignItems: 'center', gap: 8, padding: '4px 6px',
                      borderRadius: 4, border: '1px solid var(--border)', background: 'var(--surface2)',
                      cursor: addingToStack ? 'default' : 'pointer', opacity: addingToStack ? 0.5 : 1,
                    },
                    onClick: function () {
                      if (addingToStack) return;
                      setAddingToStack(true);
                      fetch(API + '/api/stacks/' + ns.stack_id + '/add', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ photo_id: photo.id }),
                      })
                        .then(function (r) { return r.json(); })
                        .then(function () {
                          // Refresh — fetch the stack we just joined
                          return fetch(API + '/api/stacks/' + ns.stack_id).then(function (r) { return r.json(); });
                        })
                        .then(function (d) {
                          setStackData(d);
                          setShowAddToStack(false);
                          setAddingToStack(false);
                        })
                        .catch(function () { setAddingToStack(false); });
                    },
                  },
                    ns.top_photo_id && e('img', {
                      src: API + '/api/photos/' + ns.top_photo_id + '/thumbnail?v=t',
                      style: { width: 40, height: 40, borderRadius: 3, objectFit: 'cover', flexShrink: 0 },
                    }),
                    e('div', { style: { fontSize: 12, lineHeight: 1.3 } },
                      e('div', { style: { fontWeight: 500 } }, ns.top_filename || ('Stack #' + ns.stack_id)),
                      e('div', { style: { color: 'var(--text-muted)' } }, '\u00d7' + ns.member_count + ' \u00B7 ' + ns.distance_sec + 's away'),
                    ),
                  );
                }),
              ),
              e('button', {
                style: { fontSize: 11, padding: '3px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--surface2)', color: 'var(--text)', cursor: 'pointer', marginTop: 6 },
                onClick: function () { setShowAddToStack(false); },
              }, 'Cancel'),
            ),
          ),

          // Description — the RAW LLM text, or (flag on, a face is labelled)
          // the same text with the child's name filled in from the CURRENT
          // face label. Never edited text: spans of the raw one, swapped.
          ((detail && detail.description) || photo.description) && e('div', { className: 'detail-section' },
            e('h3', null, 'Description', descriptionChip()),
            (function () {
              var raw = (detail && detail.description) || photo.description;
              var named = liveNames ? liveNames.display
                : (detail && detail.description_display) || null;
              var subjects = liveNames ? liveNames.subjects
                : (detail && detail.description_subjects) || null;
              var shown = (named && !showRawDesc) ? named : raw;
              var notNamed = (subjects || []).filter(function (s) { return s.state !== 'named'; });
              return e('div', null,
                e('p', null, shown),
                named && e('div', { style: { fontSize: 11, color: 'var(--text-muted)', marginTop: 2 } },
                  showRawDesc ? 'Original text. ' : 'Names from face labels. ',
                  e('a', { href: '#', style: { color: 'var(--text-muted)' },
                    onClick: function (ev) {
                      ev.preventDefault();
                      var next = !showRawDesc;
                      setShowRawDesc(next);
                      try { window.localStorage.setItem('ps.descriptionRaw', next ? '1' : '0'); }
                      catch (err) { /* private window: the toggle just does not persist */ }
                    } }, showRawDesc ? 'Show names' : 'Show original')),
                notNamed.length > 0 && e('div', { style: { fontSize: 11, color: 'var(--text-muted)', marginTop: 2 } },
                  notNamed.map(function (s) {
                    return '\u201c' + s.generic + '\u201d not named: ' + ({
                      unlabelled: 'no face label', untrusted_source: 'label is not trusted',
                      rejected: 'label rejected', stale: 'face changed',
                      ambiguous_name: 'name is ambiguous in this photo' }[s.state] || s.state);
                  }).join(' \u00b7 ')));
            })(),
          ),

          // Faces
          showFaces && detail && detail.faces && e('div', { className: 'detail-section' },
            e('h3', null, 'People'),
            visibleFaces.length > 0
              ? visibleFaces.map(renderFaceTag)
              : e('div', { style: { fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 } },
                  'No faces detected.'),
            // Detection tools (features 5 + 6). Compute runs on the NAS (originals
            // + InsightFace); replica mode proxies + mirrors, so it works here too.
            (function () {
              var btn = { fontSize: 11, padding: '3px 9px', borderRadius: 4,
                          border: '1px solid var(--border)', background: 'var(--surface2)',
                          color: 'var(--text)', cursor: 'pointer' };
              var btnActive = Object.assign({}, btn, { background: '#2563eb',
                                                       borderColor: '#2563eb', color: '#fff' });
              return e('div', { style: { display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 } },
                e('button', { style: btn, disabled: !!addingBox,
                  onClick: function () { redetectFaces(false); } }, 'Detect faces'),
                visibleFaces.length > 0 && e('button', { style: btn,
                  title: 'Delete all faces on this photo and re-detect from scratch',
                  onClick: function () {
                    if (window.confirm('Replace all faces on this photo? This wipes any names you assigned.'))
                      redetectFaces(true);
                  } }, 'Re-detect (replace)'),
                e('button', { style: addingBox ? btnActive : btn,
                  onClick: function () {
                    var next = !addingBox;
                    setAddingBox(next);
                    setFaceBusy(next ? 'Draw a box around the face on the image →' : null);
                  } }, addingBox ? 'Cancel box' : 'Add face box'),
              );
            })(),
            faceBusy && e('div', { style: { fontSize: 11, color: 'var(--text-muted)', marginTop: 6 } },
              faceBusy),
          ),

          // Aesthetic quality
          showAesthetics && renderVlmAesthetics(),
          // Old LAION quality section is a FALLBACK only — shown when the photo
          // hasn't been VLM-scored yet (so we never show both scores at once).
          showAesthetics && !(detail && detail.aesthetics) && renderAesthetics(),

          // Tags
          detail && detail.tags && detail.tags.length > 0 && e('div', { className: 'detail-section' },
            e('h3', null, 'Tags', modelChip('tags')),
            e('div', { style: { display: 'flex', flexWrap: 'wrap', gap: 4 } },
              detail.tags.map(function (tag) {
                return e('span', { key: tag, style: {
                  fontSize: 12, padding: '2px 8px', borderRadius: 12,
                  background: 'var(--surface2)', color: 'var(--text)',
                  border: '1px solid var(--border)',
                } }, tag);
              }),
            ),
          ),

          // Prior generations — lazy-fetched on expand. Shows full describe/
          // tags/verify history (different models, regenerations, backfilled
          // unknowns). Most modals never expand this, so we don't pre-load it
          // with the photo detail fetch.
          detail && ((detail.description && detail.description.length > 0) ||
                     (detail.tags && detail.tags.length > 0)) &&
          e('div', { className: 'detail-section' },
            e('h3', {
              onClick: toggleGens,
              style: { cursor: 'pointer', userSelect: 'none' },
              title: 'Show describe/tags/verify generation history',
            }, (gensOpen ? '▾ ' : '▸ ') + 'Prior generations'),
            gensOpen && e('div', null,
              gensLoading && e('p', { style: { fontSize: 12, color: 'var(--text-muted)' } }, 'Loading…'),
              gens && (gens.generations || []).length === 0 &&
                e('p', { style: { fontSize: 12, color: 'var(--text-muted)' } }, 'No generation history recorded.'),
              gens && (gens.generations || []).map(function (g) {
                var version = g.model_version ? '@' + String(g.model_version).slice(0, 7) : '';
                return e('div', {
                  key: g.id,
                  style: {
                    padding: '8px 10px', marginBottom: 6, borderRadius: 4,
                    background: 'var(--surface2)', border: '1px solid var(--border)',
                    fontSize: 12,
                  },
                }, e('div', {
                  style: {
                    display: 'flex', justifyContent: 'space-between',
                    color: 'var(--text-muted)', marginBottom: 4,
                    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                  },
                }, e('span', null,
                    e('strong', { style: { color: 'var(--text)' } }, g.text_type),
                    ' · ', (g.model_used || '(unknown)'), version
                  ),
                  e('span', null, g.created_at)
                ),
                e('div', { style: { whiteSpace: 'pre-wrap', wordBreak: 'break-word' } },
                  g.generated_text
                ));
              })
            )
          ),

          // Change history — collection adds/removals, face labels, tags,
          // location, moves; who asked and why. Starts when the audit log was
          // deployed, so an empty list does not mean nothing ever changed.
          e('div', { className: 'detail-section' },
            e('h3', {
              onClick: toggleAudit,
              style: { cursor: 'pointer', userSelect: 'none' },
              title: 'What changed on this photo, when, and who asked',
            }, (audOpen ? '▾ ' : '▸ ') + 'Change history'),
            audOpen && e('div', null,
              audLoading && e('p', { style: { fontSize: 12, color: 'var(--text-muted)' } }, 'Loading…'),
              aud && aud.stale && e('p', { style: { fontSize: 12, color: 'var(--text-muted)' } }, aud.stale_reason),
              aud && !audLoading && (aud.events || []).length === 0 &&
                e('p', { style: { fontSize: 12, color: 'var(--text-muted)' } },
                  aud.error ? 'Could not load the history.' : 'No changes recorded since the audit log started.'),
              aud && (aud.events || []).map(function (ev) {
                return e('div', {
                  key: ev.id,
                  style: {
                    padding: '6px 10px', marginBottom: 4, borderRadius: 4, fontSize: 12,
                    background: 'var(--surface2)', border: '1px solid var(--border)',
                  },
                }, e('div', null, describeAudit(ev)),
                e('div', { style: { color: 'var(--text-muted)', fontSize: 11 } },
                  ev.ts + ' UTC · ' + (ev.source || '?') + (ev.intent ? ' · ' + ev.intent : '')));
              })
            )
          ),

          // Re-run passes (M28) — recompute any index pass on this photo.
          // Synchronous (compute now via the local models) or queued for the
          // worker fleet; writes to the NAS and mirrors back to the replica.
          e('div', { className: 'detail-section' },
            e('h3', {
              onClick: function () { setRerunOpen(!rerunOpen); },
              style: { cursor: 'pointer', userSelect: 'none' },
              title: 'Recompute index passes on this photo',
            }, (rerunOpen ? '▾ ' : '▸ ') + 'Re-run passes'),
            rerunOpen && e('div', null,
              e('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '4px 12px', margin: '6px 0 10px' } },
                RERUN_PASSES.map(function (p) {
                  return e('label', { key: p, style: { fontSize: 12, display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' } },
                    e('input', { type: 'checkbox', checked: !!rerunSel[p],
                      onChange: function () { toggleRerunPass(p); } }),
                    p);
                })
              ),
              e('div', { style: { display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8, fontSize: 12 } },
                e('label', { style: { display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' } },
                  e('input', { type: 'radio', name: 'rerunmode-' + photo.id, checked: rerunMode === 'sync',
                    onChange: function () { setRerunMode('sync'); } }), 'Run now'),
                e('label', { style: { display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' } },
                  e('input', { type: 'radio', name: 'rerunmode-' + photo.id, checked: rerunMode === 'queue',
                    onChange: function () { setRerunMode('queue'); } }), 'Queue for fleet')
              ),
              e('div', { style: { display: 'flex', gap: 8, alignItems: 'center' } },
                e('button', {
                  onClick: submitRerun,
                  disabled: rerunBusy || RERUN_PASSES.filter(function (p) { return rerunSel[p]; }).length === 0,
                  style: { fontSize: 12, padding: '4px 12px', cursor: rerunBusy ? 'wait' : 'pointer' },
                }, rerunBusy ? 'Working…' : (rerunMode === 'sync' ? 'Run now' : 'Queue')),
                e('button', {
                  onClick: function () { setRerunMsg({ kind: 'info', text: 'Refreshing…' }); refreshDetail(true).then(function () { setRerunMsg({ kind: 'ok', text: 'Refreshed.' }); }); },
                  disabled: rerunBusy,
                  title: 'Pull the latest values from the NAS into this view',
                  style: { fontSize: 12, padding: '4px 12px', cursor: 'pointer' },
                }, 'Refresh')
              ),
              rerunMsg && e('p', {
                style: { fontSize: 12, marginTop: 8,
                  color: rerunMsg.kind === 'err' ? 'var(--danger, #c0392b)'
                       : rerunMsg.kind === 'ok' ? 'var(--success, #2e7d32)' : 'var(--text-muted)' },
              }, rerunMsg.text)
            )
          ),

          // Split print (M31) — plan this photo across several borderless
          // sheets. Lives next to Upscale because the planner leans on it: a
          // wall piece spreads one file over many sheets, so dpi is the usual
          // blocker and a bigger source is the only real fix.
          e('div', { className: 'detail-section' },
            e('h3', null,
              e('a', {
                href: '/split?photo=' + photo.id,
                style: { textDecoration: 'none' },
                title: 'Plan this photo as a diptych, triptych, or panel grid',
              }, '⊞ Split print →'))
          ),

          // Upscale with Topaz (M30) — runs the local Topaz Photo AI CLI and
          // writes to the export tree. The library and DB are untouched, so
          // there is no mirror/refresh step as there is for re-run passes.
          e('div', { className: 'detail-section' },
            e('h3', {
              onClick: function () { setUpOpen(!upOpen); },
              style: { cursor: 'pointer', userSelect: 'none' },
              title: 'Upscale this photo locally with Topaz Photo AI',
            }, (upOpen ? '▾ ' : '▸ ') + 'Upscale (Topaz)'),
            upOpen && e('div', null,
              e('div', { style: { display: 'flex', alignItems: 'center', gap: 8, margin: '6px 0 8px', fontSize: 12 } },
                e('span', { style: { color: 'var(--text-muted)' } }, 'Scale'),
                e('select', {
                  value: upScale,
                  onChange: function (ev) { setUpScale(ev.target.value); },
                  style: { fontSize: 12 },
                  title: 'Auto uses your Topaz Autopilot preference and leaves '
                       + 'your Topaz settings untouched',
                },
                  // Topaz takes arbitrary decimals, so modest factors are
                  // offered too — a print that needs 1.3x should not pay
                  // for 2x.
                  e('option', { value: 'auto' }, 'Auto'),
                  ['1.25', '1.5', '2', '3', '4', '6'].map(function (v) {
                    return e('option', { key: v, value: v }, v + 'x');
                  })
                ),
                e('span', { style: { color: 'var(--text-muted)', marginLeft: 6 } }, 'Strength'),
                e('select', {
                  value: upStrength,
                  onChange: function (ev) { setUpStrength(ev.target.value); },
                  style: { fontSize: 12 },
                  title: 'Blends the result toward a plain resize. Topaz has no '
                       + 'working strength control, so this is applied afterward.',
                },
                  ['100', '75', '50', '25'].map(function (v) {
                    return e('option', { key: v, value: v }, v + '%');
                  })
                ),
                e('span', { style: { color: 'var(--text-muted)', marginLeft: 6 } }, 'Model'),
                e('select', {
                  value: upModel,
                  onChange: function (ev) { setUpModel(ev.target.value); },
                  style: { fontSize: 12 },
                  title: 'Autopilot usually picks High Fidelity V2. Standard is '
                       + 'milder if that looks over-processed.',
                },
                  e('option', { value: '' }, 'Auto'),
                  UPSCALE_MODELS.map(function (m) {
                    return e('option', { key: m, value: m }, m);
                  })
                )
              ),
              e('label', {
                style: { display: 'flex', alignItems: 'center', gap: 6,
                         fontSize: 12, marginBottom: 8 },
                title: 'Without this, Topaz also applies whatever else Autopilot '
                     + 'decided on — usually a Sharpen Strong pass, which haloes '
                     + 'an already-upscaled image.',
              },
                e('input', {
                  type: 'checkbox', checked: upOnly,
                  onChange: function (ev) { setUpOnly(ev.target.checked); },
                }),
                e('span', { style: { color: 'var(--text-muted)' } },
                  'Only the checked enhancements (ignore Autopilot extras)')
              ),
              e('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '4px 12px', marginBottom: 8 } },
                UPSCALE_ENHANCEMENTS.map(function (x) {
                  return e('label', { key: x, style: { fontSize: 12, display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' } },
                    e('input', { type: 'checkbox', checked: !!upEnh[x],
                      onChange: function () {
                        setUpEnh(function (prev) {
                          var next = Object.assign({}, prev);
                          if (next[x]) { delete next[x]; } else { next[x] = true; }
                          return next;
                        });
                      } }),
                    x);
                })
              ),
              e('button', {
                onClick: submitUpscale,
                disabled: upBusy || UPSCALE_ENHANCEMENTS.filter(function (x) { return upEnh[x]; }).length === 0,
                style: { fontSize: 12, padding: '4px 12px', cursor: upBusy ? 'wait' : 'pointer' },
              }, upBusy ? 'Upscaling…' : 'Upscale'),
              upMsg && e('div', { style: { fontSize: 12, marginTop: 8 } },
                e('span', {
                  style: { color: upMsg.kind === 'err' ? 'var(--danger, #c0392b)'
                                : upMsg.kind === 'ok' ? 'var(--success, #2e7d32)'
                                : 'var(--text-muted)' },
                }, upMsg.text),
                // Result path: shown Windows-style, one click to view it in a
                // new tab, one click to copy the path for Explorer.
                upMsg.path && e('div', { style: { marginTop: 6 } },
                  e('code', {
                    style: { wordBreak: 'break-all', userSelect: 'all',
                             color: 'var(--text-muted)' },
                  }, upMsg.path),
                  e('div', { style: { display: 'flex', gap: 10, marginTop: 6 } },
                    e('a', {
                      href: upMsg.view, target: '_blank', rel: 'noopener',
                      title: 'Open the upscaled image in a new tab',
                    }, 'View'),
                    e('a', {
                      href: '#',
                      title: 'Copy the Windows path to the clipboard',
                      onClick: function (ev) {
                        ev.preventDefault();
                        navigator.clipboard.writeText(upMsg.path).then(function () {
                          setUpMsg(function (m) {
                            return Object.assign({}, m, { text: 'Path copied.' });
                          });
                        });
                      },
                    }, 'Copy path')
                  )
                )
              ),
              // Everything exported for this photo so far. Each option set is
              // its own file, so re-running with different settings adds a row
              // rather than being skipped as "already exported".
              upPrior && upPrior.length > 0 && e('div',
                { style: { marginTop: 12, borderTop: '1px solid var(--border)',
                           paddingTop: 8 } },
                e('div', { style: { fontSize: 11, color: 'var(--text-muted)',
                                    marginBottom: 6 } },
                  'Previous exports (' + upPrior.length + ')'),
                upPrior.map(function (x) {
                  return e('div', { key: x.output,
                    style: { display: 'flex', alignItems: 'baseline', gap: 8,
                             fontSize: 12, padding: '3px 0' } },
                    e('a', { href: API + '/api/admin/upscaled-file?path=' +
                                   encodeURIComponent(x.output),
                             target: '_blank', rel: 'noopener',
                             style: { flex: '0 0 auto' } }, x.label),
                    e('span', { style: { color: 'var(--text-muted)', flex: '1 1 auto',
                                         whiteSpace: 'nowrap' } },
                      (x.bytes / 1e6).toFixed(1) + ' MB · ' +
                      String(x.modified).replace('T', ' ').slice(0, 16)),
                    e('a', {
                      href: '#', style: { flex: '0 0 auto', fontSize: 11 },
                      title: x.windows_path || x.output,
                      onClick: function (ev) {
                        ev.preventDefault();
                        navigator.clipboard.writeText(x.windows_path || x.output);
                      },
                    }, 'copy')
                  );
                })
              ),
              upPrior && upPrior.length === 0 && e('div',
                { style: { marginTop: 10, fontSize: 11, color: 'var(--text-muted)' } },
                'No exports yet for this photo.')
            )
          ),

          // Search score (index.html only)
          showSearchScore && photo.score != null && e('div', { className: 'detail-section' },
            e('h3', null, 'Search Score'),
            e('div', { className: 'detail-row' },
              e('span', { className: 'label' }, 'Combined'),
              e('span', null, photo.score.toFixed(4)),
            ),
            photo.clip_score != null && e('div', { className: 'detail-row' },
              e('span', { className: 'label' }, 'CLIP'),
              e('span', null, photo.clip_score.toFixed(4)),
            ),
          ),

          // Location
          showLocation && detail && detail.place_name && e('div', { className: 'detail-section' },
            e('h3', null, 'Location'),
            e('div', { className: 'detail-row' },
              e('span', { className: 'label' }, 'Place'),
              e('span', null, detail.place_name),
            ),
          ),

          // Camera
          renderCamera(),

          // Colors
          renderColors(),

          // Collections
          renderCollections(),

          // Google Photos upload (single-photo)
          e(PS.GooglePhotosButton, { photo: photo }),

          // Footer children slot (e.g. review's cluster info, raw file)
          footerChildren,
        ),
      ),
    );
  };

  // =========================================================================
  // GooglePhotosButton — upload a single photo from the modal sidebar
  // =========================================================================
  // Props:
  //   photo  — { id, filename, description, ... }

  PS.GooglePhotosButton = function GooglePhotosButton(props) {
    var photo = props.photo;

    var _status = useState(null);
    var status = _status[0];   // null | 'uploading' | 'done' | 'error'
    var setStatus = _status[1];

    var _msg = useState('');
    var msg = _msg[0];
    var setMsg = _msg[1];

    var _googleStatus = useState(null);
    var googleStatus = _googleStatus[0];
    var setGoogleStatus = _googleStatus[1];

    var _expanded = useState(false);
    var expanded = _expanded[0];
    var setExpanded = _expanded[1];

    var _showManual = useState(false);
    var showManual = _showManual[0];
    var setShowManual = _showManual[1];

    var _manualCode = useState('');
    var manualCode = _manualCode[0];
    var setManualCode = _manualCode[1];

    var _manualRedirectUri = useState('');
    var manualRedirectUri = _manualRedirectUri[0];
    var setManualRedirectUri = _manualRedirectUri[1];

    var _manualError = useState('');
    var manualError = _manualError[0];
    var setManualError = _manualError[1];

    var _manualBusy = useState(false);
    var manualBusy = _manualBusy[0];
    var setManualBusy = _manualBusy[1];

    // Fetch Google connection status when expanded
    useEffect(function () {
      if (!expanded) return;
      fetch(API + '/api/google/status')
        .then(function (r) { return r.json(); })
        .then(setGoogleStatus)
        .catch(function () {});
    }, [expanded]);

    var doUpload = function () {
      setStatus('uploading');
      setMsg('');
      fetch(API + '/api/google/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ photo_ids: [photo.id], include_description: true }),
      })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data.detail) { setStatus('error'); setMsg(data.detail); return; }
          if (data.uploaded > 0) {
            setStatus('done');
            setMsg('\u2713 Uploaded to Google Photos');
          } else {
            var firstErr = (data.results || []).find(function (r) { return r.error; });
            setStatus('error');
            setMsg((firstErr && firstErr.error) || 'Upload failed');
          }
        })
        .catch(function (err) { setStatus('error'); setMsg(err.message); });
    };

    var doConnect = function () {
      var port = window.location.port || '8000';
      var redirectUri = 'http://localhost:' + port + '/api/google/callback';
      setManualRedirectUri(redirectUri);
      setShowManual(false);
      setManualCode('');
      setManualError('');
      fetch(API + '/api/google/authorize?port=' + encodeURIComponent(port))
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data.auth_url) {
            window.open(data.auth_url, '_blank', 'width=600,height=700');
            setShowManual(true);
            var listener = function (ev) {
              if (ev.data === 'google_photos_connected') {
                window.removeEventListener('message', listener);
                fetch(API + '/api/google/status').then(function (r) { return r.json(); }).then(setGoogleStatus).catch(function () {});
                setShowManual(false);
              }
            };
            window.addEventListener('message', listener);
          }
        })
        .catch(function () {});
    };

    var doManualCode = function () {
      var code = manualCode.trim();
      if (!code) return;
      setManualBusy(true);
      setManualError('');
      fetch(API + '/api/google/exchange-code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: code, redirect_uri: manualRedirectUri }),
      })
        .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
        .then(function (res) {
          if (!res.ok) throw new Error(res.d.detail || 'Exchange failed');
          fetch(API + '/api/google/status').then(function (r) { return r.json(); }).then(setGoogleStatus).catch(function () {});
          setShowManual(false);
          setManualCode('');
        })
        .catch(function (err) { setManualError(err.message); })
        .finally(function () { setManualBusy(false); });
    };

    if (!expanded) {
      return e('div', { className: 'detail-section' },
        e('button', {
          style: {
            fontSize: 12, padding: '4px 10px', borderRadius: 4,
            border: '1px solid var(--border)', background: 'var(--surface2)',
            color: 'var(--accent)', cursor: 'pointer', width: '100%', textAlign: 'left',
          },
          onClick: function () { setExpanded(true); setStatus(null); setMsg(''); },
        }, '\uD83D\uDDBC\uFE0F Upload to Google Photos'),
      );
    }

    var notConfigured = googleStatus && !googleStatus.configured;
    var notAuthenticated = googleStatus && googleStatus.configured && !googleStatus.authenticated;
    var ready = googleStatus && googleStatus.configured && googleStatus.authenticated;

    return e('div', { className: 'detail-section' },
      e('h3', null, '\uD83D\uDDBC Upload to Google Photos'),

      status === 'done' && e('p', { style: { color: 'var(--green)', fontSize: 13, marginBottom: 8 } }, msg),
      status === 'error' && e('p', { style: { color: 'var(--red)', fontSize: 13, marginBottom: 8 } }, msg),
      status === 'uploading' && e('p', { style: { color: 'var(--text-muted)', fontSize: 13, marginBottom: 8 } }, 'Uploading\u2026'),

      notConfigured && e('div', null,
        e('p', { style: { fontSize: 12, color: 'var(--red)', marginBottom: 4 } },
          'Setup required: place ', e('code', null, 'google_client_secret.json'), ' alongside the database, then restart.',
        ),
        e('p', { style: { fontSize: 12, color: 'var(--text-muted)' } },
          'Open a collection and click \u201cUpload to Google Photos\u201d for step-by-step instructions.',
        ),
      ),

      notAuthenticated && e('div', null,
        !showManual && e('div', null,
          e('p', { style: { fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 } },
            'Connect your Google account to upload.',
          ),
          e('button', {
            style: { fontSize: 12, padding: '4px 10px', borderRadius: 4, border: '1px solid var(--accent)', background: 'var(--accent)', color: '#fff', cursor: 'pointer' },
            onClick: doConnect,
          }, 'Connect Google Photos'),
        ),
        showManual && e('div', null,
          e('p', { style: { fontSize: 12, marginBottom: 6 } }, 'Sign-in window opened.'),
          e('details', null,
            e('summary', { style: { fontSize: 12, color: 'var(--text-muted)', cursor: 'pointer', marginBottom: 6 } }, 'Got an error page? (NAS users)'),
            e('p', { style: { fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 } },
              'Copy the ', e('code', null, 'code=XXXX'), ' value from the error URL and paste it here:',
            ),
            e('input', {
              type: 'text',
              placeholder: 'Authorization code',
              value: manualCode,
              onChange: function (ev) { setManualCode(ev.target.value); },
              onKeyDown: function (ev) { if (ev.key === 'Enter') doManualCode(); },
              style: { width: '100%', fontSize: 12, padding: '4px 8px', background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text)', marginBottom: 4 },
            }),
            manualError && e('p', { style: { fontSize: 11, color: 'var(--red)', marginBottom: 4 } }, manualError),
            e('button', {
              style: { fontSize: 11, padding: '3px 8px', borderRadius: 4, border: '1px solid var(--accent)', background: 'var(--accent)', color: '#fff', cursor: 'pointer', opacity: manualBusy ? 0.6 : 1 },
              onClick: doManualCode,
              disabled: manualBusy || !manualCode.trim(),
            }, manualBusy ? 'Connecting\u2026' : 'Submit code'),
          ),
          e('div', { style: { marginTop: 8, display: 'flex', gap: 6 } },
            e('button', {
              style: { fontSize: 11, padding: '3px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--surface2)', color: 'var(--text)', cursor: 'pointer' },
              onClick: function () { fetch(API + '/api/google/status').then(function (r) { return r.json(); }).then(function (d) { setGoogleStatus(d); if (d.authenticated) setShowManual(false); }).catch(function () {}); },
            }, 'Check status'),
            e('button', {
              style: { fontSize: 11, padding: '3px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--surface2)', color: 'var(--text)', cursor: 'pointer' },
              onClick: doConnect,
            }, 'Re-open'),
          ),
        ),
      ),

      ready && status !== 'done' && e('div', { style: { display: 'flex', gap: 6, flexWrap: 'wrap' } },
        e('button', {
          style: { fontSize: 12, padding: '4px 10px', borderRadius: 4, border: '1px solid var(--accent)', background: 'var(--accent)', color: '#fff', cursor: 'pointer', opacity: status === 'uploading' ? 0.6 : 1 },
          onClick: doUpload,
          disabled: status === 'uploading',
        }, status === 'uploading' ? 'Uploading\u2026' : 'Upload this photo'),
        e('button', {
          style: { fontSize: 12, padding: '4px 10px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--surface2)', color: 'var(--text)', cursor: 'pointer' },
          onClick: function () { setExpanded(false); setStatus(null); setMsg(''); },
        }, 'Cancel'),
      ),

      googleStatus === null && e('p', { style: { fontSize: 12, color: 'var(--text-muted)' } }, 'Checking status\u2026'),
    );
  };

  // =========================================================================
  // Sort Control — shared sort-by toggle for all pages
  // =========================================================================

  /**
   * Sort options available across all pages.
   * Each page may use a subset — pass `options` to restrict.
   */
  var SORT_OPTIONS = {
    relevance:    { label: 'Relevance',      icon: '\u2605' },  // ★
    date_desc:    { label: 'Newest first',   icon: '\u2193' },  // ↓
    date_asc:     { label: 'Oldest first',   icon: '\u2191' },  // ↑
    quality_desc: { label: 'Best quality',   icon: '\u2b50' },  // ⭐ (using ★ variant)
    aesthetic_desc: { label: 'Best aesthetics', icon: '\u2728' },  // sparkle - VLM percentile
    subject_aesthetic_desc: { label: 'Best subject', icon: '\ud83d\udfe2' },  // \ud83d\udfe2 - subject-crop percentile
    day_quality_desc: { label: 'Best of day', icon: '\ud83d\udcc5' },  // \ud83d\udcc5 - per-day percentile
    focus_desc:   { label: 'Main subject',   icon: '\u25B6' },  // video: person's main-subject share
    subject_desc: { label: 'Subject share',  icon: '\u25B6' },  // video: subject label's share
    steadiest:    { label: 'Steadiest',      icon: '\uD83C\uDFA5' },  // video: ascending shake
    least_noisy:  { label: 'Least noisy',    icon: '\uD83C\uDFA5' },  // video: ascending noise
    name_asc:     { label: 'Name A\u2013Z',  icon: 'Az' },
  };

  /**
   * PS.SortControl({ value, onChange, options })
   *
   * A compact inline sort toggle rendered as a row of clickable pills.
   *
   *   value:   current sort key (string)
   *   onChange: (newKey) => void
   *   options: array of sort keys to show (defaults to all)
   */
  // ---- Aesthetic score: single source of truth for every card + sort --------
  // The new VLM score is a library-relative percentile (0-100). The effective
  // score is the subject-crop percentile when present (what search ranks on),
  // else the full-frame percentile. Only ~a fraction of the library is VLM-
  // scored, so when neither exists we fall back to the OLD LAION aesthetic_score
  // (/10), shown faded + marked provisional.
  PS.effectiveAestheticPct = function (photo) {
    if (!photo) return null;
    if (photo.aes_subject_overall_pct != null) return photo.aes_subject_overall_pct;
    if (photo.aes_overall_pct != null) return photo.aes_overall_pct;
    return null;
  };

  // Per-day percentile (v28): the photo's rank within its own capture day.
  // Returns -1 (sorts last) when unscored/undated. Used by the "Best of day"
  // sort; the whole-library effective pct above still drives the card badge.
  PS.effectiveAestheticDayPct = function (photo) {
    if (!photo) return -1;
    if (photo.aes_subject_overall_day_pct != null) return photo.aes_subject_overall_day_pct;
    if (photo.aes_overall_day_pct != null) return photo.aes_overall_day_pct;
    return -1;
  };

  // Monotonic sort value so "Best quality" ranks VLM-scored photos (by
  // percentile) above old-scored ones (by /10) above unscored — one comparable
  // scale. Higher = better.
  PS.aestheticSortValue = function (photo) {
    var pct = PS.effectiveAestheticPct(photo);
    if (pct != null) return 1000 + pct;                       // 1000..1100
    if (photo && photo.aesthetic_score != null) return photo.aesthetic_score; // 0..~10
    return -1;
  };

  // The badge every card renders. Green/yellow/orange by percentile when VLM-
  // scored; a faded ★/10 (provisional) when only the old model has run; null
  // when neither exists. opts.style merges onto the span.
  PS.aestheticBadge = function (photo, opts) {
    opts = opts || {};
    var pct = PS.effectiveAestheticPct(photo);
    if (pct != null) {
      var v = Math.round(pct);
      var col = pct >= 75 ? '#4ade80' : pct >= 40 ? '#facc15' : '#fb923c';
      var subj = photo.aes_subject_overall_pct != null;
      return e('span', {
        className: 'aes-badge',
        title: 'Aesthetic percentile ' + v + (subj ? ' (subject-aware)' : ' (full frame)'),
        style: Object.assign({ color: col, fontWeight: 600, whiteSpace: 'nowrap' }, opts.style || {}),
      }, '✨' + v);
    }
    if (photo && photo.aesthetic_score != null) {
      return e('span', {
        className: 'aes-badge aes-badge-old',
        title: 'Provisional — old quality model (not yet VLM-scored)',
        style: Object.assign({ color: 'var(--text-muted, #999)', opacity: 0.7,
                               fontStyle: 'italic', whiteSpace: 'nowrap' }, opts.style || {}),
      }, '★' + photo.aesthetic_score.toFixed(1));
    }
    return null;
  };

  // ---- Camera: shared badge + filter dropdown, used on every grid ----------
  // ---- Video: shared helpers (docs/plans/video-support.md step 2) --------
  PS.isVideo = function (photo) {
    return !!(photo && photo.media_type === 'video');
  };

  // "0:42", "12:05", "1:02:03" — null when the duration is unknown.
  PS.formatDuration = function (seconds) {
    if (seconds == null || isNaN(seconds)) return null;
    var t = Math.max(0, Math.round(seconds));
    var h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), sec = t % 60;
    var ss = (sec < 10 ? '0' : '') + sec;
    if (h) return h + ':' + (m < 10 ? '0' : '') + m + ':' + ss;
    return m + ':' + ss;
  };

  // .avi / .mts / .m2ts play in no mainstream browser.
  PS.videoBrowserPlayable = function (photo) {
    if (photo && photo.video && photo.video.browser_playable === false) return false;
    var name = ((photo && photo.filename) || '').toLowerCase();
    return !/\.(avi|mts|m2ts)$/.test(name);
  };

  // ---- Video playback: original vs the H.264 proxy (video-proxy pass) ----
  // The owner's Windows Chrome/Edge decode NO HEVC (measured 2026-10-18), so
  // most clips need the desktop-made H.264 copy (/api/videos/{id}/proxy).
  // These three are pure (environment injectable) so jest pins them.

  // RFC 6381 codec string for MediaSource.isTypeSupported, from the probe's
  // playback block (detail.video.playback: codec / profile / level). null
  // when it cannot be built — "unknown", never a guess.
  var H264_PROFILE_IDC = { 'baseline': '42', 'constrained baseline': '42', 'main': '4d',
                           'extended': '58', 'high': '64', 'high 10': '6e',
                           'high 4:2:2': '7a', 'high 4:4:4 predictive': 'f4' };
  PS.videoCodecString = function (pb) {
    if (!pb || !pb.codec) return null;
    var codec = String(pb.codec).toLowerCase();
    var level = parseInt(pb.level, 10);
    if (codec === 'h264') {
      var idc = H264_PROFILE_IDC[String(pb.profile || '').toLowerCase()];
      if (!idc || !(level > 0)) return null;
      var constraint = /^constrained baseline$/i.test(pb.profile || '') ? 'e0' : '00';
      var hex = level.toString(16);
      return 'avc1.' + idc + constraint + (hex.length < 2 ? '0' : '') + hex;
    }
    if (codec === 'hevc') {
      if (!(level > 0)) return null;
      var main10 = /10/.test(pb.profile || '') || (pb.bit_depth || 8) > 8;
      return main10 ? 'hvc1.2.4.L' + level + '.B0' : 'hvc1.1.6.L' + level + '.B0';
    }
    if (codec === 'av1') return 'av01.0.08M.' + ((pb.bit_depth || 8) > 8 ? '10' : '08');
    if (codec === 'vp9') return 'vp09.00.10.' + ((pb.bit_depth || 8) > 8 ? '10' : '08');
    return null;
  };

  var MP4_CONTAINERS = ['mp4', 'mov', 'm4v'];

  // Can THIS browser play the original? true / false / null (unknown).
  // env = { isTypeSupported(mime), canPlayType(mime) } for tests; defaults to
  // window.MediaSource and a scratch <video>. A non-mp4/mov container (avi,
  // mpegts) is false outright; so is a long edge over 3840 (the proxy rule).
  PS.canPlayVideo = function (pb, env) {
    if (!pb) return null;
    var container = String(pb.container || '').toLowerCase();
    if (container && MP4_CONTAINERS.indexOf(container) < 0) return false;
    if (pb.long_edge && pb.long_edge > 3840) return false;
    var codecs = PS.videoCodecString(pb);
    if (!codecs) return null;
    var mime = 'video/mp4; codecs="' + codecs + '"';
    env = env || {};
    var its = env.isTypeSupported;
    var cpt = env.canPlayType;
    if (!its && !cpt && typeof window !== 'undefined') {
      if (window.MediaSource && typeof window.MediaSource.isTypeSupported === 'function') {
        its = function (m) { return window.MediaSource.isTypeSupported(m); };
      } else if (typeof document !== 'undefined') {
        var probe = document.createElement('video');
        if (probe && typeof probe.canPlayType === 'function') {
          cpt = function (m) { return probe.canPlayType(m); };
        }
      }
    }
    try {
      if (its) return !!its(mime);
      if (cpt) { var r = cpt(mime); return r === 'probably' || r === 'maybe'; }
    } catch (err) { return null; }
    return null;
  };

  // Which source the modal plays. Returns {mode, src, fallback}:
  //   'proxy'     — the H.264 copy (the original can't play here, or its
  //                 container is not mp4/mov, or unknown + the server says
  //                 it needs one);
  //   'original'  — the file itself; `fallback` is the proxy to switch to
  //                 if the <video> errors;
  //   'preparing' — nothing playable yet: say so and offer the download.
  // `video` is detail.video (may be null before the detail loads);
  // `canPlay` is PS.canPlayVideo's answer; `api` prefixes the proxy URL.
  PS.chooseVideoSource = function (video, originalUrl, canPlay, api, filename) {
    var pb = video && video.playback;
    var proxy = video && video.proxy_url ? (api || '') + video.proxy_url : null;
    var containerOk = pb && pb.container
      ? MP4_CONTAINERS.indexOf(String(pb.container).toLowerCase()) >= 0
      : PS.videoBrowserPlayable({ video: video, filename: filename || '' });
    var needs = !!(video && video.needs_proxy);
    if (proxy && (canPlay === false || !containerOk || (canPlay == null && needs))) {
      return { mode: 'proxy', src: proxy, fallback: null };
    }
    if (!proxy && (canPlay === false || !containerOk)) {
      return { mode: 'preparing', src: null, fallback: null };
    }
    return { mode: 'original', src: originalUrl, fallback: proxy };
  };

  // The grid-card badge: "▶ 0:42" once the probe has filled the duration,
  // a plain "▶" before. null for a still. opts.style merges onto the span.
  PS.videoBadge = function (photo, opts) {
    if (!PS.isVideo(photo)) return null;
    opts = opts || {};
    var d = PS.formatDuration(photo.duration_s);
    return e('span', {
      className: 'video-badge',
      title: d ? 'Video, ' + d : 'Video',
      style: Object.assign({
        position: 'absolute', top: 6, right: 6, zIndex: 2,
        padding: '1px 7px', borderRadius: 10, fontSize: 12, fontWeight: 600,
        background: 'rgba(0,0,0,0.65)', color: '#fff', whiteSpace: 'nowrap',
        pointerEvents: 'none',
      }, opts.style || {}),
    }, d ? '\u25B6 ' + d : '\u25B6');
  };

  // "Calvin 63%" / "dog 90%" on a search card, when the video subject-share
  // filter or sort annotated the row (video-support step 6). null otherwise.
  PS.subjectShareBadge = function (photo, opts) {
    if (!photo) return null;
    opts = opts || {};
    var bits = [];
    if (photo.focus_share != null) {
      var people = photo.focus_people || [];
      if (people.length) {
        people.forEach(function (p) {
          bits.push((p.name || 'person') + ' ' + Math.round((p.share || 0) * 100) + '%');
        });
      } else {
        bits.push(Math.round(photo.focus_share * 100) + '%');
      }
    }
    if (photo.subject_share != null) {
      bits.push((photo.subject_label || 'subject') + ' ' + Math.round(photo.subject_share * 100) + '%');
    }
    if (!bits.length) return null;
    var secs = photo.focus_seconds != null ? photo.focus_seconds : photo.subject_seconds;
    return e('span', {
      className: 'subject-share-badge',
      title: 'Main subject of the video' + (secs != null ? ' for ' + Math.round(secs) + ' s' : '')
             + ' (share of the whole clip; beta)',
      style: Object.assign({ color: '#facc15', whiteSpace: 'nowrap', fontSize: 11 }, opts.style || {}),
    }, '\u25B6 ' + bits.join(' \u00b7 '));
  };

  PS.cameraLabel = function (photo) {
    if (!photo) return null;
    return photo.camera_model || photo.camera_make || null;
  };

  // The camera badge every card renders (📷 model). opts.style merges on.
  PS.cameraBadge = function (photo, opts) {
    opts = opts || {};
    var label = PS.cameraLabel(photo);
    if (!label) return null;
    return e('span', {
      className: 'camera-badge',
      title: 'Camera: ' + label + (photo.camera_make && photo.camera_make !== label
                                   ? ' (' + photo.camera_make + ')' : ''),
      style: Object.assign({ color: 'var(--text-muted, #999)', whiteSpace: 'nowrap' },
                           opts.style || {}),
    }, '📷 ' + label);
  };

  // Reusable camera filter <select>. Fetches /api/cameras once; props:
  //   value (model | ''), onChange(model), style?, allLabel?
  PS.CameraFilter = function CameraFilter(props) {
    var _c = useState(null);
    var cams = _c[0];
    var setCams = _c[1];
    useEffect(function () {
      fetch(API + '/api/cameras')
        .then(function (r) { return r.json(); })
        .then(function (d) { setCams(d.cameras || []); })
        .catch(function () { setCams([]); });
    }, []);
    return e('select', {
      value: props.value || '',
      onChange: function (ev) { props.onChange(ev.target.value); },
      title: 'Filter by camera',
      style: props.style || {},
    },
      e('option', { value: '' }, props.allLabel || 'All cameras'),
      (cams || []).map(function (c) {
        return e('option', { key: c.model, value: c.model },
          (c.model || c.make) + ' (' + c.count + ')');
      }),
    );
  };

  PS.SortControl = function SortControl(props) {
    var value = props.value;
    var onChange = props.onChange;
    var options = props.options || ['date_desc', 'date_asc', 'quality_desc'];

    return e('div', {
      style: {
        display: 'inline-flex',
        gap: 0,
        borderRadius: 4,
        overflow: 'hidden',
        border: '1px solid var(--border, #444)',
        fontSize: 12,
        flexShrink: 0,
      },
    },
      options.map(function (key) {
        var opt = SORT_OPTIONS[key];
        if (!opt) return null;
        var active = value === key;
        return e('button', {
          key: key,
          onClick: function () { onChange(key); },
          title: opt.label,
          style: {
            padding: '3px 10px',
            cursor: 'pointer',
            border: 'none',
            background: active ? 'var(--accent, #4a9eff)' : 'transparent',
            color: active ? '#fff' : 'var(--text-muted, #999)',
            fontWeight: active ? 600 : 400,
            fontSize: 12,
            whiteSpace: 'nowrap',
            transition: 'background 0.15s, color 0.15s',
          },
        }, opt.label);
      }),
    );
  };

  /**
   * PS.applySortOrder(items, sortKey, options)
   *
   * Returns a new sorted array. Does not mutate the input.
   *
   *   items:   array of photo objects (with date_taken, aesthetic_score, filename, score)
   *   sortKey: one of the SORT_OPTIONS keys
   *   options: { idField: 'id' } — optional overrides
   */
  PS.applySortOrder = function applySortOrder(items, sortKey) {
    if (!items || !items.length) return items;
    var sorted = items.slice();  // shallow copy

    // Photos from old imports without EXIF have date_taken === null but
    // their parent folder is named YYYY-MM-DD, so the backend includes an
    // `effective_date` that falls back to the folder date. Prefer it when
    // present; fall back to raw date_taken for endpoints that don't set it
    // yet (search results, review, collections).
    var dateKey = function (p) { return p.effective_date || p.date_taken || ''; };
    // Push empty date keys to the TAIL in both directions. localeCompare
    // sorts '' before anything in ASC, which is why "oldest first" used
    // to surface undated photos at the top instead of the actually-old
    // ones. Matches the backend's `<expr> IS NULL, <expr> DIR` ordering.
    var cmp = function (a, b, dir) {
      var da = dateKey(a), db = dateKey(b);
      if (!da && !db) return 0;
      if (!da) return 1;
      if (!db) return -1;
      return dir === 'asc' ? da.localeCompare(db) : db.localeCompare(da);
    };

    switch (sortKey) {
      case 'date_desc':
        sorted.sort(function (a, b) { return cmp(a, b, 'desc'); });
        break;
      case 'date_asc':
        sorted.sort(function (a, b) { return cmp(a, b, 'asc'); });
        break;
      case 'quality_desc':
      case 'aesthetic_desc':
        // "Best quality" now ranks by the new VLM score (subject-aware, then
        // full-frame percentile), falling back to the old model for photos the
        // VLM hasn't scored yet. Single comparable scale via aestheticSortValue.
        sorted.sort(function (a, b) {
          return PS.aestheticSortValue(b) - PS.aestheticSortValue(a);
        });
        break;
      case 'subject_aesthetic_desc':
        sorted.sort(function (a, b) {
          var av = a.aes_subject_overall_pct != null ? a.aes_subject_overall_pct
                 : (a.aes_overall_pct != null ? a.aes_overall_pct : -1);
          var bv = b.aes_subject_overall_pct != null ? b.aes_subject_overall_pct
                 : (b.aes_overall_pct != null ? b.aes_overall_pct : -1);
          return bv - av;
        });
        break;
      case 'day_quality_desc':
        // "Best of day": per-day percentile (subject-aware, then full-frame),
        // so each day's strongest photos lead across a multi-day range.
        sorted.sort(function (a, b) {
          return PS.effectiveAestheticDayPct(b) - PS.effectiveAestheticDayPct(a);
        });
        break;
      case 'name_asc':
        sorted.sort(function (a, b) {
          return (a.filename || '').localeCompare(b.filename || '');
        });
        break;
      case 'focus_desc':
      case 'subject_desc':
        // Video main-subject share (server-annotated); rows without one last.
        var shareKey = sortKey === 'focus_desc' ? 'focus_share' : 'subject_share';
        sorted.sort(function (a, b) {
          var av = a[shareKey] != null ? a[shareKey] : -1;
          var bv = b[shareKey] != null ? b[shareKey] : -1;
          return bv - av;
        });
        break;
      case 'steadiest':
      case 'least_noisy':
        // Video quality (server-annotated shake/noise); ascending, rows
        // without a measured value last (never NULL-first — unmeasured is
        // not "steadiest").
        var qKey = sortKey === 'steadiest' ? 'shake' : 'noise';
        sorted.sort(function (a, b) {
          var av = a[qKey], bv = b[qKey];
          if (av == null && bv == null) return 0;
          if (av == null) return 1;
          if (bv == null) return -1;
          return av - bv;
        });
        break;
      case 'relevance':
      default:
        // Return original order (relevance from search, or manual order)
        return items;
    }
    return sorted;
  };

  // =========================================================================
  // Maintenance NAS-vs-replica drift panel (M26b follow-up)
  // =========================================================================

  // Push-status banner. Distinguishes a PARTIAL push (some stages landed on the
  // NAS, some failed) from a total failure — a blanket "failed" would read as
  // "nothing pushed" and prompt a needless re-push. Names the stages that
  // actually failed so the message is honest about what did and didn't land.
  function pushBanner(push) {
    if (!push || (push.state !== 'failed' && push.state !== 'partial')) return null;
    var stages = push.stages || {};
    var okSet = { applied: 1, triggered: 1, skipped: 1 };
    var failedNames = Object.keys(stages).filter(function (k) {
      return !okSet[(stages[k] || {}).status];
    });
    var detail = failedNames.length ? ' (failed: ' + failedNames.join(', ') + ')' : '';
    if (push.state === 'partial') {
      return e('div', { className: 'ms-info' },
        'Last push partially applied — some stages landed, some failed: '
        + (push.error || 'unknown') + detail);
    }
    return e('div', { className: 'ms-warn' },
      'Last push failed: ' + (push.error || 'unknown') + detail);
  }

  // Per-stage NAS-vs-replica drift. "Replica ahead — unpushed" is the state
  // that matters: it means locally-computed maintenance work will be
  // destroyed by the next replica sync unless it's pushed first. Renders
  // null outside replica mode (i.e. on the NAS itself, or a non-replica
  // single-machine deployment) — there's no drift to show.
  PS.MaintenanceSyncPanel = function MaintenanceSyncPanel(props) {
    var compact = props && props.compact;
    var st = useState(null), data = st[0], setData = st[1];
    var ps = useState(null), push = ps[0], setPush = ps[1];

    // Report replica_mode up to a parent (admin_maintenance's MaintenanceSweep
    // uses it to grey out replica-impossible checkboxes) so this panel stays the
    // SINGLE fetcher of /maintenance-fingerprint — the parent no longer fetches
    // it separately. onReplicaMode is expected to be a stable setter.
    var onReplicaMode = props && props.onReplicaMode;
    var reportMode = useCallback(function (m) {
      if (onReplicaMode) onReplicaMode(!!m);
    }, [onReplicaMode]);

    // Returns a promise covering ALL of its fetches (the nested one is returned,
    // not fired-and-forgotten) — PS.poll re-arms on it, so a round is not over
    // until every request in it has come back.
    var load = useCallback(function (signal) {
      var opts = signal ? { signal: signal } : undefined;
      var fp = fetch('/api/admin/maintenance-fingerprint', opts)
        .then(function (r) { return r.json(); })
        .then(function (l) {
          reportMode(l.replica_mode);
          if (!l.replica_mode) { setData({ replica_mode: false }); return; }
          return fetch('/api/admin/maintenance-nas-fingerprint', opts)
            .then(function (r) { return r.json(); })
            .then(function (n) { setData({ replica_mode: true, local: l, nas: n }); })
            .catch(function () { setData({ replica_mode: true, local: l, nas: null }); });
        })
        .catch(function () {});  // endpoint absent (old build) → panel stays hidden
      var ps = fetch('/api/admin/maintenance-push-status', opts)
        .then(function (r) { return r.json(); }).then(setPush).catch(function () {});
      return Promise.all([fp, ps]);
    }, [reportMode]);

    // Poll so the panel reflects the current sync state after a sweep/push —
    // the whole point of the panel is to make drift visible, and a load-once
    // effect would freeze on the pre-sweep state until a manual page reload.
    // PS.poll, not setInterval: see its comment for the 2026-09-19 pile-up.
    useEffect(function () {
      return PS.poll(load, 5000);
    }, [load]);

    if (!data || !data.replica_mode) return null;
    var local = data.local, nas = data.nas;
    // The proxy (admin_maintenance_nas_fingerprint) returns a normal 200 with
    // an `error` field on failure rather than a non-2xx status — same
    // convention as /api/admin/workers/queue-status — so `nas` is always a
    // JSON object here; check for `.stages` rather than fetch rejection.
    if (!nas || nas.error || !nas.stages) {
      var unreachableMsg = 'NAS unreachable — drift unknown.'
        + (nas && nas.error ? ' (' + nas.error + ')' : '');
      return e('div', { className: 'deploy-meta' }, unreachableMsg);
    }

    var drift = local.photo_count !== nas.photo_count || local.photo_max_id !== nas.photo_max_id;
    var names = Object.keys(local.stages).concat(Object.keys(nas.stages))
      .filter(function (v, i, a) { return a.indexOf(v) === i; }).sort();

    var rows = names.map(function (name) {
      var l = local.stages[name], n = nas.stages[name];
      var state, cls;
      if (!l && !n) { state = 'Never run'; cls = 'ms-none'; }
      else if (l && (!n || l.last_run_at > n.last_run_at)) {
        state = 'Replica ahead — unpushed'; cls = 'ms-warn';
      } else if (n && (!l || n.last_run_at > l.last_run_at)) {
        state = 'NAS ahead'; cls = 'ms-info';
      } else { state = 'In sync'; cls = 'ms-ok'; }
      return e('tr', { key: name, className: cls },
        e('td', null, name),
        e('td', null, n ? n.last_run_at : '—'),
        e('td', null, l ? l.last_run_at : '—'),
        e('td', null, state));
    });

    var unpushed = names.filter(function (name) {
      var l = local.stages[name], n = nas.stages[name];
      return l && (!n || l.last_run_at > n.last_run_at);
    });

    if (compact) {
      return e('div', { className: 'deploy-meta' },
        drift ? e('span', { className: 'ms-warn' }, 'Photo index drift — sync needed. ') : null,
        unpushed.length
          ? e('a', { href: '/admin_maintenance' },
              unpushed.length + ' stage(s) unpushed — will be lost on next sync →')
          : e('span', { className: 'ms-ok' }, 'Maintenance in sync'));
    }

    return e('div', { className: 'deploy-wrap' },
      e('h3', null, 'Maintenance: NAS vs replica'),
      drift ? e('div', { className: 'ms-warn' },
        'Photo index drift: NAS ' + nas.photo_count + ' photos / max id '
        + nas.photo_max_id + ', replica ' + local.photo_count + ' / '
        + local.photo_max_id + '. A push will auto-sync first.') : null,
      pushBanner(push),
      e('table', { className: 'ms-table' },
        e('thead', null, e('tr', null,
          e('th', null, 'Stage'), e('th', null, 'NAS'),
          e('th', null, 'Replica'), e('th', null, 'State'))),
        e('tbody', null, rows)));
  };

  // Parse one SSE frame into {event, data, raw}, or null when there is
  // nothing to act on (a keepalive comment, a blank frame, a frame with no
  // `data:` line, or a payload that isn't JSON).
  //
  // **It must find the `data:` line ANYWHERE in the frame.** The first
  // version anchored on `^data:` after trimming the whole chunk, so it
  // returned null for every frame that led with an `event:` line — which is
  // what the admin SSE endpoints emit (`_stream_subprocess` in
  // photosearch/admin_api.py writes `event: line\ndata: {...}`). The result
  // was a silently dead log: no output, and a terminal `fatal` swallowed.
  // Two pages had already worked around it with their own `data:`-line
  // filter, which is exactly the duplication this helper exists to prevent.
  //
  // Per the SSE spec, repeated `data:` lines in one frame join with "\n",
  // and one optional space after the colon is part of the delimiter.
  PS.parseSSEFrame = function parseSSEFrame(chunk) {
    var text = String(chunk == null ? '' : chunk);
    if (!text.trim()) return null;
    var lines = text.split('\n');
    var event = null, dataParts = [], sawData = false;
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i].trim();
      if (!line || line.charAt(0) === ':') continue;   // blank / keepalive comment
      var colon = line.indexOf(':');
      var field = colon === -1 ? line : line.slice(0, colon);
      var value = colon === -1 ? '' : line.slice(colon + 1);
      if (value.charAt(0) === ' ') value = value.slice(1);
      if (field === 'event') event = value;
      else if (field === 'data') { sawData = true; dataParts.push(value); }
    }
    if (!sawData) return null;
    var raw = dataParts.join('\n');
    try {
      return { event: event, data: JSON.parse(raw), raw: raw };
    } catch (_) {
      return null;
    }
  };

  // The payload alone — what most callers want. Kept as the original name and
  // shape so the existing consumers (faces.html, admin_maintenance.html) need
  // no change; reach for parseSSEFrame when you need the `event:` name to
  // tell `done` from `fatal`.
  PS.parseSSEChunk = function parseSSEChunk(chunk) {
    var frame = PS.parseSSEFrame(chunk);
    return frame ? frame.data : null;
  };

  // Poll `fn` every `ms` — but re-arm only AFTER the previous call settles.
  // Use this instead of setInterval for anything that hits the server.
  //
  // setInterval fires whether or not the last request came back. On 2026-09-19
  // the NAS's disks were saturated, each status poll took minutes, and two open
  // tabs kept issuing more until all 40 request threads were stuck — the status
  // page wedged the box it was reporting on. With this, a slow server is polled
  // SLOWER, never harder: at most one call is in flight per poller.
  //
  // `fn(signal)` should return the promise for its work (return the fetch chain;
  // use Promise.all for several). A non-promise return or a throw is tolerated.
  // Polling pauses while the tab is hidden and fires at once when it is shown.
  // Returns stop(), which also aborts the in-flight call via `signal`.
  PS.poll = function poll(fn, ms) {
    var stopped = false, inFlight = false, timer = null, controller = null;

    function run() {
      timer = null;
      if (stopped || inFlight) return;
      if (document.hidden) return;          // visibilitychange re-runs us
      inFlight = true;
      controller = (typeof AbortController !== 'undefined') ? new AbortController() : null;
      var result;
      try { result = fn(controller ? controller.signal : undefined); }
      catch (_) { result = null; }
      Promise.resolve(result).catch(function () {}).then(function () {
        inFlight = false;
        controller = null;
        if (!stopped) timer = setTimeout(run, ms);
      });
    }

    function onVisible() {
      if (stopped || document.hidden || inFlight) return;
      if (timer) { clearTimeout(timer); timer = null; }
      run();
    }

    document.addEventListener('visibilitychange', onVisible);
    run();

    return function stop() {
      stopped = true;
      if (timer) { clearTimeout(timer); timer = null; }
      if (controller) controller.abort();
      document.removeEventListener('visibilitychange', onVisible);
    };
  };

})();
