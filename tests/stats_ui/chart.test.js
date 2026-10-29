// Run with: node --test tests/stats_ui/
const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const ui = path.join(__dirname, '../../src/stats_ui');

// chart.js reads the bucket and day helpers of timeline.js, as the page loads them together.
function load() {
  const ctx = {};
  vm.runInNewContext(fs.readFileSync(path.join(ui, 'timeline.js'), 'utf8'), ctx);
  vm.runInNewContext(fs.readFileSync(path.join(ui, 'chart.js'), 'utf8'), ctx);
  return ctx;
}
const plain = (o) => JSON.parse(JSON.stringify(o));
const secs = (iso) => Date.parse(iso) / 1000;

test('bucket labels map to the timestamps of their starts, in seconds', () => {
  const c = load();
  const days = { resolution: 'day', buckets: ['2026-04-01', '2026-04-02'], groups: [] };
  assert.deepStrictEqual(plain(c.chartData(days, 'requests').data[0]), [secs('2026-04-01T00:00Z'), secs('2026-04-02T00:00Z')]);
  const hours = { resolution: 'hour', buckets: ['2026-04-01T22:00Z', '2026-04-01T23:00Z'], groups: [] };
  assert.deepStrictEqual(plain(c.chartData(hours, 'requests').data[0]), [secs('2026-04-01T22:00Z'), secs('2026-04-01T23:00Z')]);
});

test('each group gets one array of the chosen measure, aligned with the buckets', () => {
  const c = load();
  const s = {
    resolution: 'day',
    buckets: ['2026-04-01', '2026-04-02', '2026-04-03'],
    groups: [
      { name: 'Kites', requests: [1, 2, 3], bytes: [10, 20, 30], whole_files: [0.5, 1, 1.5] },
      { name: 'Sails', requests: [4, 5, 6], bytes: [40, 50, 60], whole_files: [2, 2.5, 3] },
    ],
  };
  const r = c.chartData(s, 'bytes');
  assert.deepStrictEqual(plain(r.names), ['Kites', 'Sails']);
  assert.strictEqual(r.data.length, 3);
  assert.deepStrictEqual(plain(r.data.slice(1)), [[10, 20, 30], [40, 50, 60]]);
  assert.deepStrictEqual(plain(c.chartData(s, 'whole_files').data[2]), [2, 2.5, 3]);
});

test('empty buckets give 0, never a gap', () => {
  const c = load();
  const s = {
    resolution: 'day',
    buckets: ['2026-04-01', '2026-04-02', '2026-04-03'],
    groups: [{ name: 'Kites', requests: [null, 2], bytes: [], whole_files: [] }],
  };
  assert.deepStrictEqual(plain(c.chartData(s, 'requests').data[1]), [0, 2, 0]);
  assert.deepStrictEqual(plain(c.chartData(s, 'bytes').data[1]), [0, 0, 0]);
});

test('a period with no groups still draws one flat line over its buckets', () => {
  const c = load();
  const r = c.chartData({ resolution: 'day', buckets: ['2026-04-01', '2026-04-02'], groups: [] }, 'requests');
  assert.deepStrictEqual(plain(r.names), ['All']);
  assert.deepStrictEqual(plain(r.data[1]), [0, 0]);
});

const week = ['2026-04-01', '2026-04-02', '2026-04-03', '2026-04-04', '2026-04-05'];

test('a selection maps to the whole days of the buckets it covers', () => {
  const c = load();
  const hours = ['2026-04-01T22:00Z', '2026-04-01T23:00Z', '2026-04-02T00:00Z', '2026-04-02T01:00Z'];
  const weeks = ['2026-03-30', '2026-04-06', '2026-04-13'];
  for (const [buckets, resolution, from, to, expected] of [
    [week, 'day', '2026-04-01T18:00Z', '2026-04-04T06:00Z', { from: '2026-04-02', to: '2026-04-04' }],
    // A selection between two points takes the bucket nearest its middle.
    [week, 'day', '2026-04-02T02:00Z', '2026-04-02T06:00Z', { from: '2026-04-02', to: '2026-04-02' }],
    // By week, the last week runs to its Sunday.
    [weeks, 'week', '2026-04-01T00:00Z', '2026-04-14T00:00Z', { from: '2026-04-06', to: '2026-04-19' }],
    [hours, 'hour', '2026-04-01T22:30Z', '2026-04-02T01:00Z', { from: '2026-04-01', to: '2026-04-02' }],
  ]) {
    assert.deepStrictEqual(plain(c.selectionPeriod(buckets, resolution, secs(from), secs(to))), expected, resolution);
  }
});
