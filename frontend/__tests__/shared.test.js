/**
 * Tests for shared.js — SharedHeader, PhotoModal, and helper functions.
 *
 * shared.js is an IIFE that attaches components to window.PS.
 * We load React as a global, then require shared.js, then test
 * using React Testing Library.
 */

const React = require('react');
const ReactDOM = require('react-dom');
const { render, screen, fireEvent, act, waitFor } = require('@testing-library/react');
require('@testing-library/jest-dom');

// Set up globals that shared.js expects
global.React = React;
global.ReactDOM = ReactDOM;
window.React = React;
window.ReactDOM = ReactDOM;

// Mock fetch globally before loading shared.js
global.fetch = jest.fn();

// Load shared.js — it attaches everything to window.PS
require('../dist/shared.js');

const PS = window.PS;
const e = React.createElement;

// Reset fetch mock between tests
beforeEach(() => {
  fetch.mockReset();
});


// =========================================================================
// Helper functions
// =========================================================================

describe('PS.formatFocalLength', () => {
  test('formats fractional focal length', () => {
    expect(PS.formatFocalLength('50/1')).toBe('50mm');
  });

  test('formats large fractional focal length', () => {
    expect(PS.formatFocalLength('200/1')).toBe('200mm');
  });

  test('formats non-fractional focal length', () => {
    expect(PS.formatFocalLength('85')).toBe('85mm');
  });

  test('handles division in fraction', () => {
    expect(PS.formatFocalLength('105/1')).toBe('105mm');
  });
});


describe('PS.formatFNumber', () => {
  test('formats fractional f-number', () => {
    expect(PS.formatFNumber('28/10')).toBe('2.8');
  });

  test('formats whole f-number', () => {
    expect(PS.formatFNumber('4')).toBe('4');
  });

  test('formats f/1.4 from fraction', () => {
    expect(PS.formatFNumber('14/10')).toBe('1.4');
  });

  test('formats f/5.6 from fraction', () => {
    expect(PS.formatFNumber('56/10')).toBe('5.6');
  });
});


// =========================================================================
// SharedHeader
// =========================================================================

describe('PS.SharedHeader', () => {
  test('renders logo', () => {
    render(e(PS.SharedHeader, { activePage: 'search' }));
    // Logo has "photo" in a span and "search" as text
    const logo = document.querySelector('.logo');
    expect(logo).toBeTruthy();
    expect(logo.textContent).toContain('photo');
    expect(logo.textContent).toContain('search');
  });

  test('renders all nav links', () => {
    render(e(PS.SharedHeader, { activePage: 'search' }));
    const links = document.querySelectorAll('.nav-link');
    expect(links.length).toBe(15);
    const labels = Array.from(links).map(l => l.textContent);
    expect(labels).toEqual(['Search', 'Review', 'Faces', 'Merges', 'Collections',
      'Books', 'Split', 'Map', 'Geotag', 'Batches', 'Status', 'Deploy', 'Maint',
      'Vocab', 'Logs']);
  });

  test('marks active page with active class', () => {
    render(e(PS.SharedHeader, { activePage: 'faces' }));
    const links = document.querySelectorAll('.nav-link');
    const facesLink = Array.from(links).find(l => l.textContent === 'Faces');
    expect(facesLink.classList.contains('active')).toBe(true);
    // Others should not be active
    const searchLink = Array.from(links).find(l => l.textContent === 'Search');
    expect(searchLink.classList.contains('active')).toBe(false);
  });

  test('renders correct href for each page', () => {
    render(e(PS.SharedHeader, { activePage: 'search' }));
    const links = document.querySelectorAll('.nav-link');
    const hrefs = Array.from(links).map(l => l.getAttribute('href'));
    expect(hrefs).toEqual(['/', '/review', '/faces', '/merges', '/collections',
      '/book', '/split', '/map', '/geotag', '/batches', '/status', '/admin/deploy',
      '/admin/maintenance', '/admin/vocab', '/logs']);
  });

  test('renders children inside header', () => {
    render(e(PS.SharedHeader, { activePage: 'search' },
      e('div', { 'data-testid': 'child' }, 'Search Form'),
    ));
    expect(screen.getByTestId('child')).toBeInTheDocument();
    expect(screen.getByText('Search Form')).toBeInTheDocument();
  });

  test('renders without children', () => {
    render(e(PS.SharedHeader, { activePage: 'collections' }));
    // Should render without error
    expect(document.querySelector('.header')).toBeTruthy();
  });

  test('defaults to search if no activePage', () => {
    render(e(PS.SharedHeader, {}));
    const links = document.querySelectorAll('.nav-link');
    const searchLink = Array.from(links).find(l => l.textContent === 'Search');
    expect(searchLink.classList.contains('active')).toBe(true);
  });

  test('each page gets correct active state', () => {
    const pages = ['search', 'review', 'faces', 'collections'];
    const labels = ['Search', 'Review', 'Faces', 'Collections'];

    pages.forEach((page, i) => {
      const { unmount } = render(e(PS.SharedHeader, { activePage: page }));
      const links = document.querySelectorAll('.nav-link');
      const activeLink = Array.from(links).find(l => l.classList.contains('active'));
      expect(activeLink.textContent).toBe(labels[i]);
      unmount();
    });
  });
});


// =========================================================================
// PhotoModal — basic rendering
// =========================================================================

describe('PS.PhotoModal', () => {
  const basePhoto = {
    id: 1,
    filename: 'DSC04878.JPG',
    date_taken: '2026-03-13T10:00:00',
    aesthetic_score: 7.5,
  };

  const baseDetail = {
    id: 1,
    filename: 'DSC04878.JPG',
    date_taken: '2026-03-13T10:00:00',
    description: 'Rocky coastline with waves',
    aesthetic_score: 7.5,
    aesthetic_concepts: ['composition', 'lighting'],
    aesthetic_critique: 'Good composition with natural light',
    camera_model: 'ILCE-7M4',
    focal_length: '70/1',
    f_number: '28/10',
    iso: 200,
    image_width: 7008,
    image_height: 4672,
    colors: ['#3b5998', '#8b9dc3'],
    tags: ['ocean', 'rocks'],
    faces: [],
    place_name: 'Morro Bay, CA',
  };

  test('renders modal overlay', () => {
    render(e(PS.PhotoModal, {
      photo: basePhoto,
      detail: baseDetail,
      onClose: jest.fn(),
    }));
    const overlay = document.querySelector('.modal-overlay');
    expect(overlay).toBeTruthy();
  });

  test('shows filename', () => {
    render(e(PS.PhotoModal, {
      photo: basePhoto,
      detail: baseDetail,
      onClose: jest.fn(),
    }));
    expect(screen.getByText('DSC04878.JPG')).toBeInTheDocument();
  });

  test('shows description', () => {
    render(e(PS.PhotoModal, {
      photo: basePhoto,
      detail: baseDetail,
      onClose: jest.fn(),
    }));
    expect(screen.getByText('Rocky coastline with waves')).toBeInTheDocument();
  });

  test('shows camera metadata', () => {
    render(e(PS.PhotoModal, {
      photo: basePhoto,
      detail: baseDetail,
      onClose: jest.fn(),
    }));
    expect(screen.getByText('ILCE-7M4')).toBeInTheDocument();
  });

  test('shows tags', () => {
    render(e(PS.PhotoModal, {
      photo: basePhoto,
      detail: baseDetail,
      onClose: jest.fn(),
    }));
    expect(screen.getByText('ocean')).toBeInTheDocument();
    expect(screen.getByText('rocks')).toBeInTheDocument();
  });

  test('shows place name when showLocation is true', () => {
    render(e(PS.PhotoModal, {
      photo: basePhoto,
      detail: baseDetail,
      onClose: jest.fn(),
      showLocation: true,
    }));
    expect(screen.getByText(/Morro Bay/)).toBeInTheDocument();
  });

  test('hides location when showLocation is false', () => {
    render(e(PS.PhotoModal, {
      photo: basePhoto,
      detail: baseDetail,
      onClose: jest.fn(),
      showLocation: false,
    }));
    // The place name should not appear
    expect(screen.queryByText(/Morro Bay/)).toBeNull();
  });

  test('shows navigation counter', () => {
    render(e(PS.PhotoModal, {
      photo: basePhoto,
      detail: baseDetail,
      onClose: jest.fn(),
      index: 2,
      total: 10,
    }));
    expect(screen.getByText('3 / 10')).toBeInTheDocument();
  });

  test('calls onClose when Escape is pressed', () => {
    const onClose = jest.fn();
    render(e(PS.PhotoModal, {
      photo: basePhoto,
      detail: baseDetail,
      onClose,
    }));
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalled();
  });

  test('calls onNext on ArrowRight', () => {
    const onNext = jest.fn();
    render(e(PS.PhotoModal, {
      photo: basePhoto,
      detail: baseDetail,
      onClose: jest.fn(),
      onNext,
    }));
    fireEvent.keyDown(document, { key: 'ArrowRight' });
    expect(onNext).toHaveBeenCalled();
  });

  test('calls onPrev on ArrowLeft', () => {
    const onPrev = jest.fn();
    render(e(PS.PhotoModal, {
      photo: basePhoto,
      detail: baseDetail,
      onClose: jest.fn(),
      onPrev,
    }));
    fireEvent.keyDown(document, { key: 'ArrowLeft' });
    expect(onPrev).toHaveBeenCalled();
  });

  test('renders headerChildren', () => {
    render(e(PS.PhotoModal, {
      photo: basePhoto,
      detail: baseDetail,
      onClose: jest.fn(),
      headerChildren: e('div', { 'data-testid': 'header-custom' }, 'Custom Header'),
    }));
    expect(screen.getByTestId('header-custom')).toBeInTheDocument();
  });

  test('renders footerChildren', () => {
    render(e(PS.PhotoModal, {
      photo: basePhoto,
      detail: baseDetail,
      onClose: jest.fn(),
      footerChildren: e('div', { 'data-testid': 'footer-custom' }, 'Custom Footer'),
    }));
    expect(screen.getByTestId('footer-custom')).toBeInTheDocument();
  });

  test('shows aesthetic score when showAesthetics is true', () => {
    render(e(PS.PhotoModal, {
      photo: basePhoto,
      detail: baseDetail,
      onClose: jest.fn(),
      showAesthetics: true,
    }));
    expect(screen.getByText('7.5')).toBeInTheDocument();
  });

  test('hides aesthetic score when showAesthetics is false', () => {
    render(e(PS.PhotoModal, {
      photo: basePhoto,
      detail: { ...baseDetail, aesthetic_score: 7.5 },
      onClose: jest.fn(),
      showAesthetics: false,
    }));
    // The score text should not be in any "Quality" section
    const qualitySections = document.querySelectorAll('.detail-section');
    const qualityLabels = Array.from(qualitySections)
      .map(s => s.textContent)
      .filter(t => t.includes('Quality'));
    expect(qualityLabels.length).toBe(0);
  });

  test('shows color palette', () => {
    render(e(PS.PhotoModal, {
      photo: basePhoto,
      detail: baseDetail,
      onClose: jest.fn(),
    }));
    const colorDots = document.querySelectorAll('.color-dot');
    expect(colorDots.length).toBe(2);
  });
});


// =========================================================================
// PhotoModal — face rendering
// =========================================================================

describe('PS.PhotoModal — faces', () => {
  const photoWithFaces = {
    id: 2,
    filename: 'DSC04894.JPG',
    date_taken: '2026-03-13T11:30:00',
  };

  const detailWithFaces = {
    id: 2,
    filename: 'DSC04894.JPG',
    date_taken: '2026-03-13T11:30:00',
    description: 'Two people on overlook',
    faces: [
      { id: 1, person_name: 'Alex', bbox: { top: 100, right: 200, bottom: 250, left: 50 } },
      { id: 2, person_name: 'Sam', bbox: { top: 100, right: 400, bottom: 250, left: 300 } },
    ],
    tags: [],
    colors: [],
  };

  test('shows face names when showFaces is true', () => {
    render(e(PS.PhotoModal, {
      photo: photoWithFaces,
      detail: detailWithFaces,
      onClose: jest.fn(),
      showFaces: true,
    }));
    expect(screen.getByText('Alex')).toBeInTheDocument();
    expect(screen.getByText('Sam')).toBeInTheDocument();
  });

  test('hides faces when showFaces is false', () => {
    render(e(PS.PhotoModal, {
      photo: photoWithFaces,
      detail: detailWithFaces,
      onClose: jest.fn(),
      showFaces: false,
    }));
    // Face tags should not appear in the sidebar
    const faceTags = document.querySelectorAll('.face-tag');
    expect(faceTags.length).toBe(0);
  });

  test('renders unknown face label', () => {
    const detailUnknown = {
      ...detailWithFaces,
      faces: [
        { id: 3, person_name: null, bbox: { top: 300, right: 400, bottom: 380, left: 320 }, cluster_id: 99 },
      ],
    };
    render(e(PS.PhotoModal, {
      photo: photoWithFaces,
      detail: detailUnknown,
      onClose: jest.fn(),
      showFaces: true,
    }));
    // Unknown faces should show some kind of label like "Unknown" or "?"
    expect(screen.getByText(/unknown/i)).toBeInTheDocument();
  });
});


// =========================================================================
// PhotoModal — fetchDetail mode
// =========================================================================

describe('PS.PhotoModal — fetchDetail', () => {
  const photo = {
    id: 1,
    filename: 'DSC04878.JPG',
    date_taken: '2026-03-13T10:00:00',
  };

  test('fetches detail on mount when fetchDetail is true', async () => {
    const mockDetail = {
      id: 1,
      filename: 'DSC04878.JPG',
      description: 'Fetched description',
      faces: [],
      tags: [],
      colors: [],
    };
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockDetail,
    });

    await act(async () => {
      render(e(PS.PhotoModal, {
        photo,
        fetchDetail: true,
        onClose: jest.fn(),
      }));
    });

    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/api/photos/1'));
  });

  test('does not fetch when detail is provided', () => {
    const detail = {
      id: 1,
      filename: 'DSC04878.JPG',
      description: 'Already have it',
      faces: [],
      tags: [],
      colors: [],
    };

    render(e(PS.PhotoModal, {
      photo,
      detail,
      fetchDetail: false,
      onClose: jest.fn(),
    }));

    expect(fetch).not.toHaveBeenCalled();
  });
});


// =========================================================================
// PhotoModal — collections (when showCollections is true)
// =========================================================================

describe('PS.PhotoModal — collections', () => {
  const photo = { id: 1, filename: 'test.jpg' };
  const detail = {
    id: 1,
    filename: 'test.jpg',
    description: 'Test',
    faces: [],
    tags: [],
    colors: [],
  };

  test('fetches collections when showCollections is true', async () => {
    fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ collections: [] }),
    });

    await act(async () => {
      render(e(PS.PhotoModal, {
        photo,
        detail,
        onClose: jest.fn(),
        showCollections: true,
      }));
    });

    // Should have fetched both photo collections and all collections
    const fetchCalls = fetch.mock.calls.map(c => c[0]);
    expect(fetchCalls.some(url => url.includes('/collections'))).toBe(true);
  });

  test('does not fetch collections when showCollections is false', () => {
    render(e(PS.PhotoModal, {
      photo,
      detail,
      onClose: jest.fn(),
      showCollections: false,
    }));

    const fetchCalls = fetch.mock.calls.map(c => c[0]);
    expect(fetchCalls.some(url => url.includes('/collections'))).toBe(false);
  });
});

// =========================================================================
// PS.parseSSEFrame / PS.parseSSEChunk
// =========================================================================
//
// These parse the frames three pages stream. The bug they are written for:
// the first version anchored on `^data:` after trimming the whole frame, so
// every frame that leads with an `event:` line — which is what
// _stream_subprocess emits for the admin endpoints — parsed as null. The
// /batches advance log was therefore completely dead, terminal `fatal`
// included. Two other pages had already worked around it with their own
// `data:`-line filter.

describe('PS.parseSSEFrame', () => {
  test('reads a frame that leads with an event: line', () => {
    const f = PS.parseSSEFrame('event: line\ndata: {"line": "hello"}');
    expect(f.event).toBe('line');
    expect(f.data).toEqual({ line: 'hello' });
  });

  test('reads a data-only frame — the two existing callers still work', () => {
    const f = PS.parseSSEFrame('data: {"type": "done"}');
    expect(f.event).toBe(null);
    expect(f.data).toEqual({ type: 'done' });
  });

  test('surfaces the terminal event names', () => {
    expect(PS.parseSSEFrame('event: fatal\ndata: {"error": "boom"}').event).toBe('fatal');
    expect(PS.parseSSEFrame('event: done\ndata: {"returncode": 0}').data.returncode).toBe(0);
  });

  test('is null for a keepalive comment, a blank frame and garbage', () => {
    expect(PS.parseSSEFrame(': keepalive')).toBe(null);
    expect(PS.parseSSEFrame('')).toBe(null);
    expect(PS.parseSSEFrame('   ')).toBe(null);
    expect(PS.parseSSEFrame(null)).toBe(null);
    expect(PS.parseSSEFrame('not an sse frame at all')).toBe(null);
    expect(PS.parseSSEFrame('event: line')).toBe(null);          // no data:
    expect(PS.parseSSEFrame('data: {not json}')).toBe(null);
  });

  test('joins repeated data: lines with a newline, per the SSE spec', () => {
    const f = PS.parseSSEFrame('event: x\ndata: {"a":\ndata: 1}');
    expect(f.data).toEqual({ a: 1 });
  });

  test('tolerates CRLF, trailing blank lines and a missing space after the colon', () => {
    expect(PS.parseSSEFrame('event: line\r\ndata: {"line":"x"}\r\n').data).toEqual({ line: 'x' });
    expect(PS.parseSSEFrame('data:{"line":"x"}').data).toEqual({ line: 'x' });
  });
});

describe('PS.parseSSEChunk', () => {
  test('returns the payload alone, for the callers that predate the frame form', () => {
    expect(PS.parseSSEChunk('data: {"type": "start"}')).toEqual({ type: 'start' });
    expect(PS.parseSSEChunk('event: line\ndata: {"line": "x"}')).toEqual({ line: 'x' });
  });

  test('still returns null where it always did', () => {
    expect(PS.parseSSEChunk(': keepalive')).toBe(null);
    expect(PS.parseSSEChunk('')).toBe(null);
    expect(PS.parseSSEChunk('data: nope')).toBe(null);
  });
});

describe('parseSSEFrame + BatchFlow.advanceLogLine, composed', () => {
  // The regression in one test: real frames as photosearch/admin_api.py's
  // _stream_subprocess writes them, through both halves, to a log line.
  const BF = require('../dist/batch-flow.js');
  const render = (frame) => BF.advanceLogLine(PS.parseSSEFrame(frame));

  test('a real subprocess stream produces a real log', () => {
    const frames = [
      'event: start\ndata: {"cmd": "python cli.py batch-advance --batch 7"}',
      'event: line\ndata: {"line": "  [stacking] done: stacks=3"}',
      'event: line\ndata: {"line": "  [warm_crops] done: ok=812"}',
      'event: done\ndata: {"returncode": 0}',
    ];
    expect(frames.map(render).map((l) => l && l.text)).toEqual([
      '$ python cli.py batch-advance --batch 7',
      '  [stacking] done: stacks=3',
      '  [warm_crops] done: ok=812',
      '— finished —',
    ]);
  });

  test('the fatal a replica emits when the NAS is unreachable is not lost', () => {
    const out = render('event: fatal\ndata: {"error": "could not reach authoritative server"}');
    expect(out.cls).toBe('l-err');
    expect(out.text).toContain('could not reach');
  });
});
