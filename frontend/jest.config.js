module.exports = {
  testEnvironment: 'jsdom',
  testMatch: ['**/__tests__/**/*.test.js'],
  // shared.js is an IIFE that attaches to window.PS — we set up React
  // globals before loading it in the test setup file.
  setupFiles: ['<rootDir>/__tests__/setup.js'],
};
