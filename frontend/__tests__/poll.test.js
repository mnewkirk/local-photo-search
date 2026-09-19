/**
 * Tests for PS.poll — the settle-then-rearm poller that replaces setInterval.
 *
 * setInterval fires whether or not the previous request came back. On
 * 2026-09-19 that is what filled the NAS's 40-thread request pool: against a
 * starved disk each poll took minutes, and two open tabs kept stacking more.
 * The property that matters is therefore "never more than one call in flight".
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

function deferred() {
  var d = {};
  d.promise = new Promise(function (res, rej) { d.resolve = res; d.reject = rej; });
  return d;
}

// Let resolved promises run their .then chains.
function flush() { return Promise.resolve().then(function () {}).then(function () {}); }

function setHidden(hidden) {
  Object.defineProperty(document, 'hidden', { configurable: true, get: function () { return hidden; } });
  document.dispatchEvent(new Event('visibilitychange'));
}

beforeEach(() => { jest.useFakeTimers(); setHidden(false); });
afterEach(() => { jest.useRealTimers(); });

test('runs immediately, then re-arms only after the call settles', async () => {
  var pending = [];
  var fn = jest.fn(function () { var d = deferred(); pending.push(d); return d.promise; });
  var stop = PS.poll(fn, 1000);

  expect(fn).toHaveBeenCalledTimes(1);

  // A slow server: ten intervals pass and the first call still has not
  // returned. setInterval would have fired ten more times here.
  jest.advanceTimersByTime(10000);
  expect(fn).toHaveBeenCalledTimes(1);

  pending[0].resolve();
  await flush();
  expect(fn).toHaveBeenCalledTimes(1);   // waits the interval AFTER settling
  jest.advanceTimersByTime(1000);
  expect(fn).toHaveBeenCalledTimes(2);
  stop();
});

test('a rejected call still re-arms', async () => {
  var fn = jest.fn(function () { return Promise.reject(new Error('503')); });
  var stop = PS.poll(fn, 1000);
  await flush();
  jest.advanceTimersByTime(1000);
  expect(fn).toHaveBeenCalledTimes(2);
  stop();
});

test('a synchronous throw or a non-promise return does not kill the loop', async () => {
  var n = 0;
  var fn = jest.fn(function () { n += 1; if (n === 1) throw new Error('boom'); return undefined; });
  var stop = PS.poll(fn, 1000);
  await flush();
  jest.advanceTimersByTime(1000);
  await flush();
  jest.advanceTimersByTime(1000);
  expect(fn).toHaveBeenCalledTimes(3);
  stop();
});

test('stop() cancels the next run and aborts the one in flight', async () => {
  var signals = [];
  var fn = jest.fn(function (signal) { signals.push(signal); return deferred().promise; });
  var stop = PS.poll(fn, 1000);
  expect(signals[0].aborted).toBe(false);
  stop();
  expect(signals[0].aborted).toBe(true);
  jest.advanceTimersByTime(5000);
  expect(fn).toHaveBeenCalledTimes(1);
});

test('does not poll while the tab is hidden, and resumes at once when shown', async () => {
  var fn = jest.fn(function () { return Promise.resolve(); });
  var stop = PS.poll(fn, 1000);
  await flush();
  expect(fn).toHaveBeenCalledTimes(1);

  setHidden(true);
  jest.advanceTimersByTime(60000);
  await flush();
  expect(fn).toHaveBeenCalledTimes(1);

  setHidden(false);
  expect(fn).toHaveBeenCalledTimes(2);
  stop();
});

test('becoming visible while a call is in flight does not start a second one', async () => {
  var d = deferred();
  var fn = jest.fn(function () { return d.promise; });
  var stop = PS.poll(fn, 1000);
  setHidden(true);
  setHidden(false);
  expect(fn).toHaveBeenCalledTimes(1);
  stop();
});
