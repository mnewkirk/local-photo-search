/**
 * Jest setup: make React + ReactDOM available as globals
 * before shared.js is loaded (it expects them on window).
 */
const React = require('react');
const ReactDOM = require('react-dom');

global.React = React;
global.ReactDOM = ReactDOM;
window.React = React;
window.ReactDOM = ReactDOM;
