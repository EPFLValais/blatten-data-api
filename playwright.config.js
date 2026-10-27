// Browser stories, run against the local stack (blatten-data-ui compose, local Keycloak realm):
//   npm install && npx playwright install chromium && npm run story
// STATS_STORY_URL is the origin the stack serves (http://localhost:8888 by default). STATS_API
// serves the stack's /stats from another address, through a proxy on PROXY_PORT; see
// tests/browser/stats_proxy.js. CHROMIUM_PATH runs a browser already on the machine instead of
// the one Playwright downloads.
const { defineConfig, devices } = require('@playwright/test');

const PROXY_PORT = 8897;

const launchOptions = {};
if (process.env.CHROMIUM_PATH) launchOptions.executablePath = process.env.CHROMIUM_PATH;
// Without '<-loopback>' the browser sends localhost straight past the proxy.
if (process.env.STATS_API) launchOptions.proxy = { server: `http://127.0.0.1:${PROXY_PORT}`, bypass: '<-loopback>' };

module.exports = defineConfig({
  testDir: 'tests/browser',
  testMatch: '*.spec.js',
  globalSetup: require.resolve('./tests/browser/global_setup.js'),
  timeout: 60_000,
  workers: 1,
  use: {
    baseURL: process.env.STATS_STORY_URL || 'http://localhost:8888',
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], launchOptions } }],
});
module.exports.PROXY_PORT = PROXY_PORT;
