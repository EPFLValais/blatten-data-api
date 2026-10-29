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

test('the box sits right of and below the pointer, flipping at the right and bottom edges', () => {
  const t = load();
  for (const [pointer, expected] of [
    [{ left: 100, top: 40 }, { left: 112, top: 52 }],
    [{ left: 520, top: 40 }, { left: 388, top: 52 }],
    [{ left: 100, top: 180 }, { left: 112, top: 108 }],
  ]) {
    assert.deepStrictEqual(plain(t.tooltipPlace(pointer, box, plot)), expected, JSON.stringify(pointer));
  }
});

test('the box never leaves the left or top of the plot', () => {
  const t = load();
  const narrow = { width: 150, height: 70 };
  assert.deepStrictEqual(plain(t.tooltipPlace({ left: 60, top: 30 }, box, narrow)), { left: 0, top: 0 });
  const p = t.tooltipPlace({ left: 590, top: 195 }, { width: 700, height: 300 }, plot);
  assert.ok(p.left >= 0 && p.top >= 0);
});
