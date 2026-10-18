/**
 * Video source choice (video-proxy pass): PS.videoCodecString,
 * PS.canPlayVideo and PS.chooseVideoSource — pure, environment injected.
 *
 * Measured on the owner's Windows Chrome/Edge (2026-10-18): no HEVC at all;
 * H.264 8-bit 4:2:0 in mp4/mov plays.
 */

const React = require('react');
const ReactDOM = require('react-dom');
global.React = React;
global.ReactDOM = ReactDOM;
window.React = React;
window.ReactDOM = ReactDOM;
global.fetch = jest.fn();
require('../dist/shared.js');
const PS = window.PS;

// The owner's browser: H.264 yes, HEVC no.
const chromeWin = { isTypeSupported: (m) => /avc1\./.test(m) };
// Safari-ish: both.
const safari = { isTypeSupported: (m) => /avc1\.|hvc1\./.test(m) };

const HEVC_HDR = { codec: 'hevc', container: 'mov', profile: 'Main 10', level: 120,
  bit_depth: 10, long_edge: 1920, pix_fmt: 'yuv420p10le' };
const H264 = { codec: 'h264', container: 'mov', profile: 'High', level: 40,
  bit_depth: 8, long_edge: 1920, pix_fmt: 'yuv420p' };
const AVI = { codec: 'mpeg4', container: 'avi', profile: 'Advanced Simple Profile',
  level: 5, long_edge: 1920 };
const MTS = Object.assign({}, H264, { container: 'mpegts' });

describe('PS.videoCodecString', () => {
  test('h264 profile + level -> avc1', () => {
    expect(PS.videoCodecString(H264)).toBe('avc1.640028');
    expect(PS.videoCodecString({ codec: 'h264', profile: 'Main', level: 31 })).toBe('avc1.4d001f');
    expect(PS.videoCodecString({ codec: 'h264', profile: 'Constrained Baseline', level: 30 }))
      .toBe('avc1.42e01e');
  });
  test('hevc main / main10 -> hvc1', () => {
    expect(PS.videoCodecString(HEVC_HDR)).toBe('hvc1.2.4.L120.B0');
    expect(PS.videoCodecString({ codec: 'hevc', profile: 'Main', level: 93 }))
      .toBe('hvc1.1.6.L93.B0');
  });
  test('unknown is null, never a guess', () => {
    expect(PS.videoCodecString(null)).toBeNull();
    expect(PS.videoCodecString({ codec: 'h264', profile: 'High' })).toBeNull();
    expect(PS.videoCodecString(AVI)).toBeNull();
  });
});

describe('PS.canPlayVideo', () => {
  test('the measured matrix', () => {
    expect(PS.canPlayVideo(H264, chromeWin)).toBe(true);
    expect(PS.canPlayVideo(HEVC_HDR, chromeWin)).toBe(false);
    expect(PS.canPlayVideo(HEVC_HDR, safari)).toBe(true);
  });
  test('avi / mts never play, whatever the codec', () => {
    expect(PS.canPlayVideo(AVI, safari)).toBe(false);
    expect(PS.canPlayVideo(MTS, chromeWin)).toBe(false);
  });
  test('over 3840 is not playable (the proxy rule)', () => {
    expect(PS.canPlayVideo(Object.assign({}, HEVC_HDR, { long_edge: 7680 }), safari)).toBe(false);
  });
  test('canPlayType fallback and unknowns', () => {
    expect(PS.canPlayVideo(H264, { canPlayType: () => 'probably' })).toBe(true);
    expect(PS.canPlayVideo(H264, { canPlayType: () => '' })).toBe(false);
    expect(PS.canPlayVideo(null, chromeWin)).toBeNull();
    expect(PS.canPlayVideo({ codec: 'h264', container: 'mp4' }, chromeWin)).toBeNull();
    expect(PS.canPlayVideo(H264, { isTypeSupported: () => { throw new Error('x'); } })).toBeNull();
  });
});

describe('PS.chooseVideoSource', () => {
  const ORIG = '/api/photos/5/full';
  const withProxy = (pb, extra) => Object.assign(
    { playback: pb, proxy_url: '/api/videos/5/proxy?v=1', needs_proxy: true }, extra || {});
  const noProxy = (pb, extra) => Object.assign(
    { playback: pb, proxy_url: null, needs_proxy: true }, extra || {});

  test('HEVC on Chrome with a proxy -> the proxy', () => {
    const c = PS.chooseVideoSource(withProxy(HEVC_HDR), ORIG,
      PS.canPlayVideo(HEVC_HDR, chromeWin), 'http://x');
    expect(c).toEqual({ mode: 'proxy', src: 'http://x/api/videos/5/proxy?v=1', fallback: null });
  });
  test('HEVC on Safari with a proxy -> the original, proxy as fallback', () => {
    const c = PS.chooseVideoSource(withProxy(HEVC_HDR), ORIG, true, '');
    expect(c).toEqual({ mode: 'original', src: ORIG, fallback: '/api/videos/5/proxy?v=1' });
  });
  test('avi with a proxy -> the proxy even if the codec check is unknown', () => {
    expect(PS.chooseVideoSource(withProxy(AVI), ORIG, null, '').mode).toBe('proxy');
  });
  test('unknown playability + needs_proxy + a proxy -> the proxy', () => {
    expect(PS.chooseVideoSource(withProxy(HEVC_HDR), ORIG, null, '').mode).toBe('proxy');
  });
  test('no proxy yet and unplayable -> preparing', () => {
    expect(PS.chooseVideoSource(noProxy(HEVC_HDR), ORIG, false, '').mode).toBe('preparing');
    expect(PS.chooseVideoSource(noProxy(MTS), ORIG, null, '').mode).toBe('preparing');
  });
  test('playable H.264 -> the original, no fallback', () => {
    const c = PS.chooseVideoSource({ playback: H264, proxy_url: null, needs_proxy: false },
      ORIG, true, '');
    expect(c).toEqual({ mode: 'original', src: ORIG, fallback: null });
  });
  test('before the detail loads: the filename decides', () => {
    expect(PS.chooseVideoSource(null, ORIG, null, '', 'clip.MP4').mode).toBe('original');
    expect(PS.chooseVideoSource(null, ORIG, null, '', 'old.AVI').mode).toBe('preparing');
  });
});

describe('PS.PhotoModal video source', () => {
  const { render, fireEvent } = require('@testing-library/react');
  const e = React.createElement;
  const photo = { id: 5, filename: 'IMG_0457.MOV', media_type: 'video' };
  const detail = (video) => ({ id: 5, filename: 'IMG_0457.MOV', media_type: 'video',
    faces: [], colors: [], categories: [], visual_tags: [], keywords: [], video: video });

  beforeEach(() => {
    fetch.mockReset();
    fetch.mockResolvedValue({ ok: true, json: () => Promise.resolve({}) });
    // jsdom has no MediaSource: pretend to be the owner's Chrome.
    window.MediaSource = { isTypeSupported: (m) => /avc1\./.test(m) };
  });
  afterEach(() => { delete window.MediaSource; });

  test('plays the proxy for HEVC, with the original one click away', () => {
    render(e(PS.PhotoModal, { photo: photo, onClose: jest.fn(),
      detail: detail({ playback: HEVC_HDR, needs_proxy: true,
                       proxy_url: '/api/videos/5/proxy?v=1' }) }));
    const v = document.querySelector('video');
    expect(v.getAttribute('src')).toMatch(/\/api\/videos\/5\/proxy\?v=1$/);
    expect(document.body.textContent).toMatch(/720p copy/);
  });

  test('no proxy yet: no broken player, a preparing note + download', () => {
    render(e(PS.PhotoModal, { photo: photo, onClose: jest.fn(),
      detail: detail({ playback: HEVC_HDR, needs_proxy: true, proxy_url: null }) }));
    expect(document.querySelector('video')).toBeNull();
    expect(document.body.textContent).toMatch(/Preparing a playable copy/);
    expect(document.querySelector('a[href$="/full?download=1"]')).toBeTruthy();
  });

  test('an original that errors falls back to the proxy', () => {
    render(e(PS.PhotoModal, { photo: photo, onClose: jest.fn(),
      detail: detail({ playback: H264, needs_proxy: false,
                       proxy_url: '/api/videos/5/proxy?v=1' }) }));
    let v = document.querySelector('video');
    expect(v.getAttribute('src')).toMatch(/\/api\/photos\/5\/full$/);
    fireEvent.error(v);
    v = document.querySelector('video');
    expect(v.getAttribute('src')).toMatch(/proxy\?v=1$/);
  });
});
