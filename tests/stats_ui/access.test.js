// Run with: node --test tests/stats_ui/
const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const ui = path.join(__dirname, '../../src/stats_ui');
const html = fs.readFileSync(path.join(ui, 'index.html'), 'utf8');
const appSource = fs.readFileSync(path.join(ui, 'app.js'), 'utf8');

function helpers() {
  const ctx = { URLSearchParams };
  vm.runInNewContext(fs.readFileSync(path.join(ui, 'signin.js'), 'utf8'), ctx);
  return ctx;
}

// The opening tag of the element with this id, or of the first element with this class.
const tagWith = (attr) => (html.match(new RegExp('<[a-z]+[^>]*' + attr + '[^>]*>')) || [null])[0];

test('the page opens on the access screen with the dashboard hidden', () => {
  const access = tagWith('id="access"');
  assert.ok(access, 'an access screen');
  assert.ok(!/\shidden[\s>]/.test(access), access);
  assert.match(html, /id="access-text"[^>]*>Signing in…</);
  const dashboard = [...html.matchAll(/<[a-z]+[^>]*data-dashboard[^>]*>/g)].map((m) => m[0]);
  assert.ok(dashboard.some((t) => /class="controls"/.test(t)), 'the header controls are dashboard');
  assert.ok(dashboard.some((t) => /^<main/.test(t)), 'the main sections are dashboard');
  for (const tag of dashboard) assert.match(tag, /\shidden[\s>]/, tag);
});

test('the access screen carries a sign-out button', () => {
  const start = html.indexOf('id="access"');
  const end = html.indexOf('</section>', start);
  assert.match(html.slice(start, end), /<button[^>]*id="access-signout"[^>]*>Sign out</);
  assert.match(appSource, /\$\('access-signout'\)\.addEventListener\('click', signOut\)/);
});

test('the dashboard is shown only on an accepted request and hidden again by a refusal', () => {
  const { accessAfter } = helpers();
  assert.strictEqual(accessAfter(200), null);
  assert.strictEqual(accessAfter(403), 'This account may not view the statistics.');
  assert.strictEqual(accessAfter(401, 'refused'), 'Access refused. Try again later.');
  assert.strictEqual(accessAfter(401, 'signing-in'), undefined);
  assert.strictEqual(accessAfter(500), undefined);
});

function fakeDocument() {
  const el = (hidden) => ({ hidden, textContent: '' });
  const nodes = { access: el(false), 'access-text': el(false) };
  const dashboard = [el(true), el(true)];
  return {
    nodes, dashboard,
    getElementById: (id) => nodes[id],
    querySelectorAll: (sel) => (sel === '[data-dashboard]' ? dashboard : []),
  };
}

test('showing the access screen hides every part of the dashboard, and the reverse', () => {
  const { showAccess } = helpers();
  const doc = fakeDocument();
  showAccess(doc, null);
  assert.strictEqual(doc.nodes.access.hidden, true);
  for (const d of doc.dashboard) assert.strictEqual(d.hidden, false);
  showAccess(doc, 'This account may not view the statistics.');
  assert.strictEqual(doc.nodes.access.hidden, false);
  assert.strictEqual(doc.nodes['access-text'].textContent, 'This account may not view the statistics.');
  for (const d of doc.dashboard) assert.strictEqual(d.hidden, true);
});
