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

test('the last 24 hours are summed into quarter hours, empty ones 0', () => {
  const c = load();
  const now = secs('2026-04-02T12:07Z');
  const timeline = [
    { ts: secs('2026-04-02T12:01Z'), requests: 2, bytes: 20 },
    { ts: secs('2026-04-02T12:05Z'), requests: 3, bytes: 30 },
    { ts: secs('2026-04-01T12:00Z'), requests: 7, bytes: 70 },
    { ts: secs('2026-03-31T12:00Z'), requests: 9, bytes: 90 },
  ];
  const [x, y] = c.liveData(timeline, now, 'requests');
  assert.strictEqual(x[0], secs('2026-04-01T12:00Z'));
  assert.strictEqual(x[x.length - 1], secs('2026-04-02T12:00Z'));
  assert.strictEqual(x.length, 97);
  assert.strictEqual(x[1] - x[0], 900);
  assert.strictEqual(y[0], 7);
  assert.strictEqual(y[y.length - 1], 5);
  assert.strictEqual(y.reduce((a, v) => a + v, 0), 12);
  assert.strictEqual(c.liveData(timeline, now, 'bytes')[1][96], 50);
});

const week = ['2026-04-01', '2026-04-02', '2026-04-03', '2026-04-04', '2026-04-05'];

test('a selection maps to the whole days of the buckets it covers', () => {
  const c = load();
  const p = c.selectionPeriod(week, 'day', secs('2026-04-01T18:00Z'), secs('2026-04-04T06:00Z'));
  assert.deepStrictEqual(plain(p), { from: '2026-04-02', to: '2026-04-04' });
});

test('a selection between two points takes the bucket nearest its middle', () => {
  const c = load();
  const p = c.selectionPeriod(week, 'day', secs('2026-04-02T02:00Z'), secs('2026-04-02T06:00Z'));
  assert.deepStrictEqual(plain(p), { from: '2026-04-02', to: '2026-04-02' });
});

test('a selection by week covers the last week to its Sunday', () => {
  const c = load();
  const weeks = ['2026-03-30', '2026-04-06', '2026-04-13'];
  const p = c.selectionPeriod(weeks, 'week', secs('2026-04-01T00:00Z'), secs('2026-04-14T00:00Z'));
  assert.deepStrictEqual(plain(p), { from: '2026-04-06', to: '2026-04-19' });
});

test('a selection by hour maps to the days of its hours', () => {
  const c = load();
  const hours = ['2026-04-01T22:00Z', '2026-04-01T23:00Z', '2026-04-02T00:00Z', '2026-04-02T01:00Z'];
  const p = c.selectionPeriod(hours, 'hour', secs('2026-04-01T22:30Z'), secs('2026-04-02T01:00Z'));
  assert.deepStrictEqual(plain(p), { from: '2026-04-01', to: '2026-04-02' });
});

test('a selection is in the form the statistics routes take', () => {
  const c = load();
  const p = c.selectionPeriod(week, 'day', secs('2026-04-01T00:00Z'), secs('2026-04-05T00:00Z'));
  for (const v of [p.from, p.to]) assert.match(v, /^\d{4}-\d{2}-\d{2}$/);
  assert.ok(p.from <= p.to);
  assert.deepStrictEqual(Object.keys(p).sort(), ['from', 'to']);
});
