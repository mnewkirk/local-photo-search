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
  //   activePage  — 'search' | 'review' | 'faces' | 'collections'
  //   children    — optional React nodes (search form, review controls, etc.)
  PS.SharedHeader = function SharedHeader(props) {
    var activePage = props.activePage || 'search';
    var children = props.children;

    var navLinks = [
      { href: '/',            label: 'Search',      id: 'search' },
      { href: '/review',      label: 'Review',      id: 'review' },
      { href: '/faces',       label: 'Faces',       id: 'faces' },
      { href: '/collections', label: 'Collections', id: 'collections' },
      { href: '/status',      label: 'Status',      id: 'status' },
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
      return detail.faces.filter(function (f) { return f.bbox; }).map(function (f) {
        var bbox = f.bbox;
        var sx = imgRect.dw / imgRect.nw;
        var sy = imgRect.dh / imgRect.nh;
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
    return e('div', { className: 'modal-overlay', onClick: function (ev) {
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
            e('h3', null, 'Description'),
            e('p', null, (detail && detail.description) || photo.description),
          ),

          // Faces
          showFaces && detail && detail.faces && detail.faces.length > 0 && e('div', { className: 'detail-section' },
            e('h3', null, 'People'),
            detail.faces.map(renderFaceTag),
          ),

          // Aesthetic quality
          showAesthetics && renderAesthetics(),

          // Tags
          detail && detail.tags && detail.tags.length > 0 && e('div', { className: 'detail-section' },
            e('h3', null, 'Tags'),
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

          // Footer children slot (e.g. review's cluster info, raw file)
          footerChildren,
        ),
      ),
    );
  };

})();
