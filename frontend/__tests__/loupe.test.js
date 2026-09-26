/**
 * PS.Loupe — the ORIGINAL at actual pixels, for judging sharpness at native
 * resolution (docs/plans/sharpness-measurement.md, step 1).
 */
const React = require('react');
const ReactDOM = require('react-dom');
const { render, screen, fireEvent } = require('@testing-library/react');
require('@testing-library/jest-dom');

global.React = React;
global.ReactDOM = ReactDOM;
window.React = React;
window.ReactDOM = ReactDOM;
global.fetch = jest.fn();
require('../dist/shared.js');
const PS = window.PS;
const e = React.createElement;

describe('PS.loupeScroll', () => {
  test('centres the focus point in the viewport', () => {
    expect(PS.loupeScroll({ x: 0.5, y: 0.5 }, 6000, 4000, 1000, 800))
      .toEqual({ left: 2500, top: 1600 });
    expect(PS.loupeScroll({ x: 0.25, y: 0.75 }, 6000, 4000, 1000, 800))
      .toEqual({ left: 1000, top: 2600 });
  });
  test('clamps to the scrollable range at the edges', () => {
    expect(PS.loupeScroll({ x: 0, y: 1 }, 6000, 4000, 1000, 800))
      .toEqual({ left: 0, top: 3200 });
  });
  test('an image smaller than the viewport does not scroll', () => {
    expect(PS.loupeScroll({ x: 0.9, y: 0.9 }, 500, 400, 1000, 800))
      .toEqual({ left: 0, top: 0 });
  });
  test('defaults to the centre', () => {
    expect(PS.loupeScroll(null, 2000, 1000, 1000, 500)).toEqual({ left: 500, top: 250 });
  });
});

describe('PS.loupeCssSize', () => {
  test('100% is one image pixel per DEVICE pixel', () => {
    expect(PS.loupeCssSize(6000, 4000, 1, 1)).toEqual({ width: 6000, height: 4000 });
    // HiDPI: CSS size halves, or the browser would upsample 2x and fake softness.
    expect(PS.loupeCssSize(6000, 4000, 1, 2)).toEqual({ width: 3000, height: 2000 });
    expect(PS.loupeCssSize(6000, 4000, 2, 2)).toEqual({ width: 6000, height: 4000 });
  });
});

describe('PS.Loupe', () => {
  test('loads the original, not the preview, and closes on Esc and ×', () => {
    const onClose = jest.fn();
    render(e(PS.Loupe, { src: '/api/photos/7/full', focus: { x: 0.2, y: 0.3 }, onClose }));
    const img = screen.getByAltText('original');
    expect(img.getAttribute('src')).toBe('/api/photos/7/full');
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByLabelText('Close loupe'));
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  test('draws at natural size once loaded and never CSS-fits it', () => {
    render(e(PS.Loupe, { src: '/x', onClose: () => {} }));
    const img = screen.getByAltText('original');
    Object.defineProperty(img, 'naturalWidth', { value: 4000 });
    Object.defineProperty(img, 'naturalHeight', { value: 3000 });
    fireEvent.load(img);
    const dpr = window.devicePixelRatio || 1;
    expect(img.style.width).toBe(Math.round(4000 / dpr) + 'px');
    expect(img.style.maxWidth).toBe('none');
    expect(screen.getByText(/4000 × 3000 px original/)).toBeInTheDocument();
  });

  test('swallows the page\'s save / navigate keys while open', () => {
    const pageKey = jest.fn();
    render(e(PS.Loupe, { src: '/x', onClose: () => {} }));
    document.body.addEventListener('keydown', pageKey);
    fireEvent.keyDown(document.body, { key: 'Enter' });
    fireEvent.keyDown(document.body, { key: 'ArrowRight' });
    expect(pageKey).not.toHaveBeenCalled();
    document.body.removeEventListener('keydown', pageKey);
  });

  test('says so when the browser cannot draw the original', () => {
    render(e(PS.Loupe, { src: '/x.heic', onClose: () => {} }));
    fireEvent.error(screen.getByAltText('original'));
    expect(screen.getByText(/could not display this original/)).toBeInTheDocument();
  });
});
