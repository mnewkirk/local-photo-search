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
  // M15 — SharedHeader
  // =========================================================================
  // Props:
  //   activePage  — 'search' | 'review' | 'faces' | 'merges' | 'collections' | 'map' | 'geotag' | 'status' | 'admin'
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
      { href: '/map',         label: 'Map',         id: 'map' },
      { href: '/geotag',      label: 'Geotag',      id: 'geotag' },
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

    // ---- Re-run-passes state (M28) ----
    var RERUN_PASSES = ['describe', 'category-content', 'category-visual',
                        'keywords', 'verify', 'clip', 'faces', 'quality', 'aesthetics'];
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

    // Reset gen + re-run state when the focused photo changes
    useEffect(function () {
      setGens(null); setGensOpen(false); setGensLoading(false);
      setRerunSel({}); setRerunMsg(null); setRerunBusy(false);
    }, [photo && photo.id]);

    // Toggle one pass. Selecting describe also selects the text passes derived
    // from the description (category-content + keywords) — re-describing alone
    // leaves those stale (the exact failure that mislabels e.g. soccer photos).
    function toggleRerunPass(p) {
      setRerunSel(function (prev) {
        var next = Object.assign({}, prev);
        if (next[p]) { delete next[p]; }
        else {
          next[p] = true;
          if (p === 'describe') { next['category-content'] = true; next['keywords'] = true; }
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
    }, [photo && photo.id]);

    // Build face bounding box overlay data
    var faceOverlays = useMemo(function () {
      if (!detail || !detail.faces || !imgRect) return [];
      // Use original image dimensions for bbox scaling.
      // imgRect.nw/nh is the PREVIEW image size (≤1920px), but bbox coords are
      // stored in original image space (e.g. 4608px). Using detail.image_width
      // gives the correct scale factor regardless of which image tier is served.
      var origW = (detail && detail.image_width) ? detail.image_width : imgRect.nw;
      var origH = (detail && detail.image_height) ? detail.image_height : imgRect.nh;
      return detail.faces.filter(function (f) { return f.bbox; }).map(function (f) {
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
    }, [detail, imgRect]);

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
        e('div', { style: { display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 10 } },
          e('span', { style: { fontSize: 28, fontWeight: 700,
                               color: _scoreColor(overall) } },
            pct == null ? (overall == null ? '–' : overall.toFixed(1)) : Math.round(pct)),
          e('span', { style: { fontSize: 13, color: 'var(--text-muted)' } },
            pct == null ? '/ 10 overall'
              : 'percentile · ' + (overall != null ? overall.toFixed(1) + '/10 raw' : '')),
        ),
        (function () {
          // Subject-aware quality (v27): the primary subject's crop score, which
          // judges the subject rather than the background. From aes_raw.
          var raw = (detail && detail.aes_raw) || {};
          var so = raw.aes_subject_overall, sp = raw.aes_subject_overall_pct;
          var boxes = (detail && detail.subject_boxes) || null;
          if (so != null) {
            var lbl = boxes && boxes.length ? (boxes[0].label || 'subject') : 'subject';
            return e('div', { style: { fontSize: 13, marginBottom: 10, padding: '5px 9px',
                                       background: 'var(--surface2)', borderRadius: 6 } },
              e('span', { style: { color: 'var(--text-muted)' } }, 'Subject (' + lbl + '): '),
              e('span', { style: { fontWeight: 700, color: _scoreColor(so) } },
                (sp != null ? Math.round(sp) + ' pct' : '') + ' · ' + so.toFixed(1) + '/10'));
          }
          if (boxes && boxes.length === 0) {
            return e('div', { style: { fontSize: 12, color: 'var(--text-muted)', marginBottom: 10 } },
              'No distinct subject — scored on the full frame.');
          }
          return null;
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
        e('div', { className: 'modal-image', style: { position: 'relative' } },
          e('img', {
            key: 'photo-' + photo.id,
            ref: imgRef,
            src: API + '/api/photos/' + photo.id + '/preview',
            alt: photo.filename,
            onLoad: updateImgRect,
          }),
          // Face bounding box overlays (only for hovered face)
          faceOverlays.filter(function (ov) { return hoveredFace === ov.id; }).map(function (ov) {
            return e('div', {
              key: 'bbox-' + ov.id,
              className: 'face-bbox-overlay',
              style: Object.assign({}, ov.style, { borderColor: '#4ade80' }),
            },
              e('span', {
                className: 'face-bbox-label',
                style: { background: '#4ade80', color: '#000' },
              }, ov.label),
            );
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

          // Description
          ((detail && detail.description) || photo.description) && e('div', { className: 'detail-section' },
            e('h3', null, 'Description', descriptionChip()),
            e('p', null, (detail && detail.description) || photo.description),
          ),

          // Faces
          showFaces && detail && detail.faces && detail.faces.length > 0 && e('div', { className: 'detail-section' },
            e('h3', null, 'People'),
            detail.faces.map(renderFaceTag),
          ),

          // Aesthetic quality
          showAesthetics && renderVlmAesthetics(),
          showAesthetics && renderAesthetics(),

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
        sorted.sort(function (a, b) {
          return (b.aesthetic_score || 0) - (a.aesthetic_score || 0);
        });
        break;
      case 'aesthetic_desc':
        sorted.sort(function (a, b) {
          return (b.aes_overall_pct || 0) - (a.aes_overall_pct || 0);
        });
        break;
      case 'name_asc':
        sorted.sort(function (a, b) {
          return (a.filename || '').localeCompare(b.filename || '');
        });
        break;
      case 'relevance':
      default:
        // Return original order (relevance from search, or manual order)
        return items;
    }
    return sorted;
  };

})();
