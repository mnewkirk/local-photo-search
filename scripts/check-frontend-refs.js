#!/usr/bin/env node
/*
 * Catch bare identifiers a page CALLS but never defines.
 *
 * Why this exists: the pages are plain React (no build step, no bundler, no
 * linter), so a helper copy-pasted between pages — or referenced from the
 * wrong page — is a ReferenceError the moment that code path runs, and
 * nothing catches it before a human clicks the button. A syntax check does
 * NOT catch it: `new Function(src)` happily accepts a call to an undefined
 * name. That is exactly how `parseSSEChunk` (defined only in
 * admin_maintenance.html) shipped in faces.html.
 *
 * Deliberately conservative: it only looks at BARE `name(` call sites, and
 * treats anything bound anywhere in the same script as defined. It will miss
 * some real bugs; it should never report a false one.
 */
const fs = require('fs');
const path = require('path');

const DIR = path.join(__dirname, '..', 'frontend', 'dist');

// Blank out comments and string/template literals so prose can't look like
// code. Done as a single left-to-right scan, NOT layered regexes: strip
// comments first and an apostrophe in a comment desynchronises every string
// after it; strip strings first and `//` inside a URL eats the line. Either
// order produced phantom "undefined" names like `All(` out of `'All (' + n`.
function decomment(src) {
  let out = '';
  for (let i = 0; i < src.length; i++) {
    const c = src[i], d = src[i + 1];
    if (c === '/' && d === '/') { while (i < src.length && src[i] !== '\n') i++; out += '\n'; continue; }
    if (c === '/' && d === '*') { i += 2; while (i < src.length && !(src[i] === '*' && src[i + 1] === '/')) i++; i++; out += ' '; continue; }
    if (c === '"' || c === "'" || c === '`') {
      const q = c; i++;
      while (i < src.length && src[i] !== q) { if (src[i] === '\\') i++; i++; }
      out += q + q; continue;
    }
    out += c;
  }
  return out;
}

function inlineScript(file) {
  const src = fs.readFileSync(file, 'utf8');
  const out = [];
  const re = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g;
  let m;
  while ((m = re.exec(src))) out.push(m[1]);
  return out.join('\n');
}

const KEYWORDS = new Set(['if', 'for', 'while', 'switch', 'catch', 'return', 'function',
  'typeof', 'await', 'async', 'new', 'delete', 'void', 'in', 'of', 'do', 'else', 'var',
  'let', 'const', 'class', 'yield', 'throw', 'super', 'this', 'import', 'export']);

const GLOBALS = new Set(['fetch', 'Error', 'TypeError', 'Set', 'Map', 'WeakMap', 'Date',
  'String', 'Number', 'Boolean', 'Array', 'Object', 'JSON', 'Math', 'Promise', 'RegExp',
  'Symbol', 'BigInt', 'parseInt', 'parseFloat', 'isNaN', 'isFinite', 'encodeURIComponent',
  'decodeURIComponent', 'encodeURI', 'decodeURI', 'setTimeout', 'clearTimeout',
  'setInterval', 'clearInterval', 'requestAnimationFrame', 'cancelAnimationFrame',
  'alert', 'confirm', 'prompt', 'URL', 'URLSearchParams', 'AbortController', 'Image',
  'TextDecoder', 'TextEncoder', 'Blob', 'FormData', 'FileReader', 'IntersectionObserver',
  'ResizeObserver', 'MutationObserver', 'CustomEvent', 'Event', 'DOMParser', 'structuredClone',
  'queueMicrotask', 'atob', 'btoa', 'console', 'window', 'document', 'navigator', 'location',
  'localStorage', 'sessionStorage', 'history', 'React', 'ReactDOM', 'PS', 'L', 'e',
  'useState', 'useEffect', 'useCallback', 'useMemo', 'useRef', 'useContext', 'useReducer',
  'useLayoutEffect', 'Fragment', 'createElement', 'require', 'module', 'exports']);

// A parameter token can carry destructuring punctuation and a default:
// `{ f, selected, onToggle }` splits into `{ f` and `onToggle }`.
const cleanParam = (x) => x.split('=')[0].split(':').pop()
  .replace(/[{}\[\].]/g, '').trim();

function boundNames(src) {
  const b = new Set();
  const add = (n) => { if (n && /^[A-Za-z_$][\w$]*$/.test(n)) b.add(n); };
  // function decls + named function exprs
  for (const m of src.matchAll(/function\s*\*?\s*([A-Za-z_$][\w$]*)/g)) add(m[1]);
  // simple declarations
  for (const m of src.matchAll(/(?:const|let|var)\s+([A-Za-z_$][\w$]*)/g)) add(m[1]);
  // array destructuring: const [a, setA] = useState()
  for (const m of src.matchAll(/(?:const|let|var)\s*\[([^\]]*)\]/g))
    m[1].split(',').forEach(x => add(x.replace(/\.\.\./, '').trim()));
  // object destructuring, incl. renames and defaults
  for (const m of src.matchAll(/(?:const|let|var)\s*\{([^}]*)\}/g))
    m[1].split(',').forEach(x => add(x.split('=')[0].split(':').pop().replace(/\.\.\./, '').trim()));
  // parameter lists — function foo(a, b) / (a, b) => / catch (e)
  for (const m of src.matchAll(/(?:function\s*\*?\s*[A-Za-z_$][\w$]*\s*|function\s*\*?\s*|catch\s*)\(([^)]*)\)/g))
    m[1].split(',').forEach(x => add(cleanParam(x)));
  for (const m of src.matchAll(/\(([^()]*)\)\s*=>/g))
    m[1].split(',').forEach(x => add(cleanParam(x)));
  for (const m of src.matchAll(/([A-Za-z_$][\w$]*)\s*=>/g)) add(m[1]);
  // object-literal methods / shorthand: foo(a) { ... } and foo: function
  for (const m of src.matchAll(/([A-Za-z_$][\w$]*)\s*:\s*function/g)) add(m[1]);
  return b;
}

// shared.js is a pure IIFE — `var PS = window.PS = ...` is the ONLY thing that
// escapes it. So its inner names are NOT globals, and must not be treated as
// resolving a page's bare call. Getting this wrong made the first version of
// this script pass the very bug it was written for: `parseSSEChunk` is a named
// function EXPRESSION inside the IIFE, so it looked defined while a page
// calling it bare would still throw.
//
// PS members are collected from EVERY dist/*.js, not just shared.js: a pure
// module a page loads alongside it (batch-flow.js publishes PS.BatchFlow) is
// just as real a definition. Only the `PS.x =` assignments are taken, so a
// module's inner names still do not resolve a page's bare call.
const psMembers = new Set();
for (const js of fs.readdirSync(DIR).filter(x => x.endsWith('.js')).sort()) {
  const src = decomment(fs.readFileSync(path.join(DIR, js), 'utf8'));
  for (const m of src.matchAll(/\bPS\.([A-Za-z_$][\w$]*)\s*=/g)) psMembers.add(m[1]);
}

let bad = 0;
for (const f of fs.readdirSync(DIR).filter(x => x.endsWith('.html')).sort()) {
  const raw = inlineScript(path.join(DIR, f));
  if (!raw.trim()) continue;
  const src = decomment(raw);
  const bound = boundNames(src);
  const called = new Set(
    [...src.matchAll(/(^|[^.\w$?])([A-Za-z_$][\w$]*)\s*\(/g)].map(m => m[2]));
  const unresolved = [...called].filter(n =>
    !bound.has(n) && !KEYWORDS.has(n) && !GLOBALS.has(n));
  if (unresolved.length) {
    bad++;
    console.error(`${f}: calls undefined ${unresolved.join(', ')}`);
  }
  // A PS.* call has to name something shared.js (or this page) actually puts
  // on PS — a typo there is the same runtime failure, just via a property.
  const pageMembers = new Set([...src.matchAll(/\bPS\.([A-Za-z_$][\w$]*)\s*=/g)].map(m => m[1]));
  const psCalled = new Set([...src.matchAll(/\bPS\.([A-Za-z_$][\w$]*)\s*\(/g)].map(m => m[1]));
  const missingPs = [...psCalled].filter(n => !psMembers.has(n) && !pageMembers.has(n));
  if (missingPs.length) {
    bad++;
    console.error(`${f}: calls PS.${missingPs.join(', PS.')} — not defined in shared.js`);
  }
}
if (bad) {
  console.error(`\n${bad} page(s) call an undefined function. If the name is a `
    + `shared helper, put it on PS.* in shared.js and call it as PS.name().`);
  process.exit(1);
}
console.log('frontend refs OK — no page calls an undefined function.');
