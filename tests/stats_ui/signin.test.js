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

function helpers() {
  const ctx = { URLSearchParams };
  vm.runInNewContext(source, ctx);
  return ctx;
}

const appSource = fs.readFileSync(path.join(__dirname, '../../src/stats_ui/app.js'), 'utf8');

test('signing out forgets every key the page keeps in session storage and no other', async () => {
  const { forgetSession, signInGuard } = helpers();
  const storage = memoryStorage();
  const written = [...appSource.matchAll(/sessionStorage\.setItem\('([^']+)'/g)].map((m) => m[1]);
  assert.ok(written.length > 0);
  for (const key of written) storage.setItem(key, 'x');
  await signInGuard(storage, async () => {}).refused();
  storage.setItem('another_page', 'kept');
  forgetSession(storage);
  for (const key of [...written, 'signin_after_refusal']) assert.strictEqual(storage.getItem(key), null, key);
  assert.strictEqual(storage.getItem('another_page'), 'kept');
});

test('the sign-out address ends the realm session and returns to the page', () => {
  const { signOutUrl } = helpers();
  const cfg = { logoutUrl: 'https://sso.orchard.test/realms/orchard/protocol/openid-connect/logout', clientId: 'pears' };
  const url = new URL(signOutUrl(cfg, 'id.token.value', 'https://orchard.test/stats'));
  assert.strictEqual(url.origin + url.pathname, cfg.logoutUrl);
  assert.strictEqual(url.searchParams.get('client_id'), 'pears');
  assert.strictEqual(url.searchParams.get('id_token_hint'), 'id.token.value');
  assert.strictEqual(url.searchParams.get('post_logout_redirect_uri'), 'https://orchard.test/stats');
});

test('without an ID token the sign-out address leaves out the hint', () => {
  const { signOutUrl } = helpers();
  const url = new URL(signOutUrl({ logoutUrl: 'https://sso.orchard.test/logout', clientId: 'pears' }, null, 'https://orchard.test/stats'));
  assert.strictEqual(url.searchParams.has('id_token_hint'), false);
  assert.strictEqual(url.searchParams.get('client_id'), 'pears');
});
