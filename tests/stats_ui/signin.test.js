// Run with: node --test tests/stats_ui/
const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../src/stats_ui/signin.js'), 'utf8');

function memoryStorage() {
  const m = new Map();
  return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, String(v)), removeItem: (k) => m.delete(k) };
}

// One page load: a fresh script context over the session storage that survives the redirect.
function pageLoad(storage, logins) {
  const ctx = {};
  vm.runInNewContext(source, ctx);
  return ctx.signInGuard(storage, async () => { logins.push(1); });
}

test('a 401 that lasts across a fresh sign-in signs in once, then is reported', async () => {
  const storage = memoryStorage();
  const logins = [];
  assert.strictEqual(await pageLoad(storage, logins).refused(), 'signing-in');
  assert.strictEqual(await pageLoad(storage, logins).refused(), 'refused');
  assert.strictEqual(logins.length, 1);
});

test('requests refused together while signing in start one sign-in', async () => {
  const logins = [];
  const guard = pageLoad(memoryStorage(), logins);
  const results = await Promise.all([guard.refused(), guard.refused(), guard.refused()]);
  assert.deepStrictEqual(results, ['signing-in', 'signing-in', 'signing-in']);
  assert.strictEqual(logins.length, 1);
});

test('an accepted request lets a later 401 sign in again', async () => {
  const storage = memoryStorage();
  const logins = [];
  await pageLoad(storage, logins).refused();
  const next = pageLoad(storage, logins);
  next.accepted();
  assert.strictEqual(await next.refused(), 'signing-in');
  assert.strictEqual(logins.length, 2);
});
