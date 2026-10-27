// Run with: node --test tests/stats_ui/
const { test, mock } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../src/stats_ui/timeline.js'), 'utf8');

function load() {
  const ctx = { AbortController, setTimeout: (...a) => setTimeout(...a), clearTimeout: (t) => clearTimeout(t) };
  vm.runInNewContext(source, ctx);
  return ctx;
}

// A strip of ten daily bars, 10 px each, starting at x = 50.
const days = Array.from({ length: 10 }, (_, i) => '2026-03-' + String(i + 1).padStart(2, '0'));
const strip = (t) => t.stripScale(days, 'day', 50, 100);
const plain = (o) => JSON.parse(JSON.stringify(o));

test('the window is drawn over the days of the period, and over everything without one', () => {
  const t = load();
  const s = strip(t);
  assert.deepStrictEqual(plain(t.windowSpan({ from: '2026-03-03', to: '2026-03-04' }, s)), { x0: 70, x1: 90 });
  assert.deepStrictEqual(plain(t.windowSpan({}, s)), { x0: 50, x1: 150 });
});

test('brushing across the strip selects the days under it, whichever way it is dragged', () => {
  const t = load();
  const s = strip(t);
  const forward = t.dragWindow(s, { mode: 'brush', start: 72, span: null }, 104);
  assert.deepStrictEqual(plain(forward), { from: '2026-03-03', to: '2026-03-06' });
  const backward = t.dragWindow(s, { mode: 'brush', start: 104, span: null }, 72);
  assert.deepStrictEqual(plain(backward), plain(forward));
});

test('moving the window keeps its width and stops at the ends of the strip', () => {
  const t = load();
  const s = strip(t);
  const span = { from: '2026-03-03', to: '2026-03-04' };
  const moved = t.dragWindow(s, { mode: 'move', start: 75, span }, 105);
  assert.deepStrictEqual(plain(moved), { from: '2026-03-06', to: '2026-03-07' });
  const pushed = t.dragWindow(s, { mode: 'move', start: 75, span }, 400);
  assert.deepStrictEqual(plain(pushed), { from: '2026-03-09', to: '2026-03-10' });
  const pulled = t.dragWindow(s, { mode: 'move', start: 75, span }, -400);
  assert.deepStrictEqual(plain(pulled), { from: '2026-03-01', to: '2026-03-02' });
});

test('a strip selection maps to the window and back', () => {
  const t = load();
  const s = strip(t);
  const period = { from: '2026-03-03', to: '2026-03-04' };
  const select = t.stripSelect(period, s);
  assert.deepStrictEqual(plain(select), { left: 20, width: 20 });
  assert.deepStrictEqual(plain(t.selectPeriod(s, select.left, select.width)), period);
  assert.deepStrictEqual(plain(t.selectPeriod(s, 22, 33)), { from: '2026-03-03', to: '2026-03-06' });
  assert.deepStrictEqual(plain(t.stripSelect({}, s)), { left: 0, width: 100 });
});

test('a drag starting on an edge of the window resizes it, inside it moves it, elsewhere brushes', () => {
  const t = load();
  const s = strip(t);
  const period = { from: '2026-03-03', to: '2026-03-05' };
  assert.strictEqual(t.dragMode(s, period, 71), 'from');
  assert.strictEqual(t.dragMode(s, period, 99), 'to');
  assert.strictEqual(t.dragMode(s, period, 85), 'move');
  assert.strictEqual(t.dragMode(s, period, 120), 'brush');
  assert.strictEqual(t.dragMode(s, {}, 85), 'brush', 'all time has no window to move');
});

test('dragging an edge moves only that end, and never past the other', () => {
  const t = load();
  const s = strip(t);
  const span = { from: '2026-03-03', to: '2026-03-05' };
  assert.deepStrictEqual(plain(t.dragWindow(s, { mode: 'from', start: 70, span }, 52)), { from: '2026-03-01', to: '2026-03-05' });
  assert.deepStrictEqual(plain(t.dragWindow(s, { mode: 'to', start: 100, span }, 128)), { from: '2026-03-03', to: '2026-03-08' });
  assert.deepStrictEqual(plain(t.dragWindow(s, { mode: 'from', start: 70, span }, 140)), { from: '2026-03-05', to: '2026-03-05' });
  assert.deepStrictEqual(plain(t.dragWindow(s, { mode: 'to', start: 100, span }, 10)), { from: '2026-03-03', to: '2026-03-03' });
});

test('clicking a bar zooms into the days it covers', () => {
  const t = load();
  assert.deepStrictEqual(plain(t.bucketPeriod('2026-03-02', 'week')), { from: '2026-03-02', to: '2026-03-08' });
  assert.deepStrictEqual(plain(t.bucketPeriod('2026-03-02', 'day')), { from: '2026-03-02', to: '2026-03-02' });
  assert.deepStrictEqual(plain(t.bucketPeriod('2026-03-02T13:00Z', 'hour')), { from: '2026-03-02', to: '2026-03-02' });
  assert.strictEqual(strip(t).bucketAt(83), 3);
});

test('the period every request uses is the window once it was dragged', () => {
  const t = load();
  const now = new Date(Date.UTC(2026, 2, 10, 12));
  const view = { range: '7', from: '', to: '' };
  assert.deepStrictEqual(plain(t.periodOf(view, now)), { from: '2026-03-04', to: '2026-03-10' });
  Object.assign(view, t.windowView({ from: '2026-03-03', to: '2026-03-06' }));
  assert.deepStrictEqual(plain(t.periodOf(view, now)), { from: '2026-03-03', to: '2026-03-06' });
  assert.deepStrictEqual(plain(t.periodOf({ range: 'all' }, now)), {});
});

test('of several quick drags only the last is loaded and applied, and a stale load is aborted', async () => {
  mock.timers.enable({ apis: ['setTimeout'] });
  try {
    const t = load();
    const started = [];
    const applied = [];
    const loader = t.latestLoader((signal, period) => {
      const call = { period, signal };
      started.push(call);
      return new Promise((resolve) => { call.answer = resolve; })
        .then(() => { if (!signal.aborted) applied.push(period); });
    }, 200);

    loader.soon('a');
    mock.timers.tick(100);
    loader.soon('b');
    mock.timers.tick(100);
    loader.soon('c');
    assert.strictEqual(started.length, 0, 'nothing loads while the window is still moving');
    mock.timers.tick(200);
    assert.deepStrictEqual(started.map((c) => c.period), ['c']);

    loader.soon('d');
    mock.timers.tick(200);
    assert.ok(started[0].signal.aborted, 'the load a later drag superseded is aborted');
    started[1].answer();
    started[0].answer();
    await new Promise((r) => setImmediate(r));
    assert.deepStrictEqual(applied, ['d']);
  } finally {
    mock.timers.reset();
  }
});
