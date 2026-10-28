// Run with: node --test tests/stats_ui/
const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function load() {
  const ctx = {};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../src/stats_ui/tooltip.js'), 'utf8'), ctx);
  return ctx;
}
const plain = (o) => JSON.parse(JSON.stringify(o));
const plot = { width: 600, height: 200 };
const box = { width: 120, height: 60 };

test('the box sits right of and below the pointer', () => {
  const t = load();
  assert.deepStrictEqual(plain(t.tooltipPlace({ left: 100, top: 40 }, box, plot)), { left: 112, top: 52 });
});

test('the box flips left of the pointer at the right edge', () => {
  const t = load();
  assert.deepStrictEqual(plain(t.tooltipPlace({ left: 520, top: 40 }, box, plot)), { left: 388, top: 52 });
});

test('the box flips above the pointer at the bottom', () => {
  const t = load();
  assert.deepStrictEqual(plain(t.tooltipPlace({ left: 100, top: 180 }, box, plot)), { left: 112, top: 108 });
});

test('the box never leaves the left or top of the plot', () => {
  const t = load();
  const narrow = { width: 150, height: 70 };
  assert.deepStrictEqual(plain(t.tooltipPlace({ left: 60, top: 30 }, box, narrow)), { left: 0, top: 0 });
  const p = t.tooltipPlace({ left: 590, top: 195 }, { width: 700, height: 300 }, plot);
  assert.ok(p.left >= 0 && p.top >= 0);
});
