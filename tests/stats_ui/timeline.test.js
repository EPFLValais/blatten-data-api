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

test('a drag brushes the days under it either way, moves the window within the strip, or moves one end never past the other', () => {
  const t = load();
  const s = strip(t);
  const two = { from: '2026-03-03', to: '2026-03-04' };
  const three = { from: '2026-03-03', to: '2026-03-05' };
  for (const [drag, x, expected] of [
    [{ mode: 'brush', start: 72, span: null }, 104, { from: '2026-03-03', to: '2026-03-06' }],
    [{ mode: 'brush', start: 104, span: null }, 72, { from: '2026-03-03', to: '2026-03-06' }],
    [{ mode: 'move', start: 75, span: two }, 105, { from: '2026-03-06', to: '2026-03-07' }],
    [{ mode: 'move', start: 75, span: two }, 400, { from: '2026-03-09', to: '2026-03-10' }],
    [{ mode: 'move', start: 75, span: two }, -400, { from: '2026-03-01', to: '2026-03-02' }],
    [{ mode: 'from', start: 70, span: three }, 52, { from: '2026-03-01', to: '2026-03-05' }],
    [{ mode: 'to', start: 100, span: three }, 128, { from: '2026-03-03', to: '2026-03-08' }],
    [{ mode: 'from', start: 70, span: three }, 140, { from: '2026-03-05', to: '2026-03-05' }],
    [{ mode: 'to', start: 100, span: three }, 10, { from: '2026-03-03', to: '2026-03-03' }],
  ]) {
    assert.deepStrictEqual(plain(t.dragWindow(s, drag, x)), expected, `${drag.mode} from ${drag.start} to ${x}`);
  }
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

test('a press on or within reach of an end takes the nearer end, inside the window moves it, elsewhere brushes', () => {
  const t = load();
  const s = strip(t);
  const period = { from: '2026-03-03', to: '2026-03-05' };
  const day = { from: '2026-03-04', to: '2026-03-04' };
  for (const [window, x, expected, why] of [
    [period, 71, 'from'],
    [period, 99, 'to'],
    [period, 85, 'move'],
    [period, 120, 'brush'],
    [{}, 85, 'brush', 'all time has no window to move'],
    [{}, 54, 'from', 'all time ends at the strip, and its ends can be dragged in'],
    [{}, 146, 'to'],
    [{}, 100, 'brush', 'inside an all-time window a drag still brushes'],
    [day, 84, 'from'],
    [day, 86, 'to'],
    [day, 75, 'from', 'outside the window, the handle still reaches'],
  ]) {
    assert.strictEqual(t.dragMode(s, window, x), expected, why || `${JSON.stringify(window)} at ${x}`);
  }
});

test('the dates under the handles sit under their ends, inside the strip, and merge when they would overlap', () => {
  const t = load();
  const w = { from: 40, to: 40, both: 90 };
  assert.deepStrictEqual(plain(t.endLabels(100, 200, w, 300)), [{ end: 'from', left: 80 }, { end: 'to', left: 180 }]);
  assert.deepStrictEqual(plain(t.endLabels(0, 300, w, 300)), [{ end: 'from', left: 0 }, { end: 'to', left: 260 }]);
  assert.deepStrictEqual(plain(t.endLabels(100, 130, w, 300)), [{ end: 'both', left: 70 }]);
  assert.deepStrictEqual(plain(t.endLabels(285, 300, w, 300)), [{ end: 'both', left: 210 }]);
  assert.deepStrictEqual(plain(t.endLabels(0, 10, w, 300)), [{ end: 'both', left: 0 }]);
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

test('a preset period lights its button, and a dragged or saved custom window lights none', () => {
  const t = load();
  for (const range of ['1', '7', '30', 'all']) assert.strictEqual(t.litPreset({ range, from: '', to: '' }), range);
  const view = { range: '30', from: '', to: '' };
  Object.assign(view, t.windowView({ from: '2026-03-03', to: '2026-03-06' }));
  assert.strictEqual(t.litPreset(view), null);
  assert.strictEqual(t.litPreset({ range: 'custom', from: '', to: '' }), null);
  assert.strictEqual(t.litPreset({ range: '90' }), null);
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

test('the last 24 hours ask for yesterday and today and show the hours since this hour yesterday', () => {
  const t = load();
  const now = Date.UTC(2026, 2, 10, 13, 40);
  const w = t.last24Hours(now);
  assert.deepStrictEqual(plain({ from: w.from, to: w.to }), { from: '2026-03-09', to: '2026-03-10' });
  assert.deepStrictEqual(plain(w.x), [Date.UTC(2026, 2, 9, 13) / 1000, now / 1000]);
  assert.deepStrictEqual(plain(t.periodOf({ range: '24h' }, new Date(now))), { from: '2026-03-09', to: '2026-03-10' });
  assert.strictEqual(t.litPreset({ range: '24h' }), '24h');
  const midnight = t.last24Hours(Date.UTC(2026, 2, 1, 0, 10));
  assert.deepStrictEqual(plain({ from: midnight.from, to: midnight.to }), { from: '2026-02-28', to: '2026-03-01' });
});

test('a series counts its bars from the first bucket to the one holding the end of the window or now', () => {
  const t = load();
  const now = Date.UTC(2026, 9, 22, 13, 40);
  assert.strictEqual(t.barCount('hour', '2026-10-20', '2026-10-22', now), 62);
  assert.strictEqual(t.barCount('hour', '2026-10-01', '2026-10-05', now), 120);
  assert.strictEqual(t.barCount('day', '2026-10-01', '2026-10-05', now), 5);
  // Weeks start on Monday: 2026-09-28 and 2026-10-05.
  assert.strictEqual(t.barCount('week', '2026-10-01', '2026-10-05', now), 2);
});

test('a resolution whose bars the server would refuse is disabled and says why, and the choice falls back to auto', () => {
  const t = load();
  const now = Date.UTC(2026, 9, 22, 12);
  const year = { range: 'custom', from: '2025-10-23', to: '2026-10-22', resolution: 'hour' };
  const chips = t.resolutionChips(year, t.periodOf(year, new Date(now)), null, 2500, now, 'day');
  assert.deepStrictEqual(plain(chips.map((c) => [c.value, c.label, c.disabled])), [
    ['auto', 'Auto (day)', false], ['hour', 'Hour', true], ['day', 'Day', false], ['week', 'Week', false],
  ]);
  assert.match(chips[1].title, /8,749 bars.*2,500/);
  assert.strictEqual(t.chosenResolution(year, chips), null, 'a refused choice sends no resolution');
  assert.strictEqual(t.litResolution(year, chips), 'auto');

  const week = { range: 'custom', from: '2026-10-12', to: '2026-10-18', resolution: 'day' };
  const ok = t.resolutionChips(week, t.periodOf(week, new Date(now)), null, 2500, now, 'hour');
  assert.ok(ok.every((c) => !c.disabled));
  assert.strictEqual(t.chosenResolution(week, ok), 'day');
  assert.strictEqual(t.litResolution(week, ok), 'day');
  assert.strictEqual(t.chosenResolution({ ...week, resolution: 'auto' }, ok), null);
  assert.strictEqual(ok[0].label, 'Auto', 'with another resolution chosen, the server chose none');
  const auto = { ...week, resolution: 'auto' };
  assert.strictEqual(t.resolutionChips(auto, t.periodOf(auto, new Date(now)), null, 2500, now, 'hour')[0].label, 'Auto (hour)');
  assert.strictEqual(t.resolutionChips(auto, t.periodOf(auto, new Date(now)), null, 2500, now, null)[0].label, 'Auto');
});

test('all time counts its bars from the first day with data, and waits for it', () => {
  const t = load();
  const now = Date.UTC(2026, 9, 22, 12);
  const all = { range: 'all', resolution: 'hour' };
  const known = t.resolutionChips(all, {}, '2026-10-01', 2500, now, 'day');
  assert.strictEqual(known[1].disabled, false, '21 days by hour is 517 bars');
  assert.strictEqual(t.chosenResolution(all, known), 'hour');
  const waiting = t.resolutionChips(all, {}, null, 2500, now, 'day');
  assert.ok(waiting.slice(1).every((c) => c.disabled));
  assert.strictEqual(t.chosenResolution(all, waiting), null);
});

test('the last 24 hours are always by hour', () => {
  const t = load();
  const now = Date.UTC(2026, 9, 22, 12);
  const view = { range: '24h', resolution: 'week' };
  const chips = t.resolutionChips(view, t.periodOf(view, new Date(now)), null, 2500, now, 'hour');
  assert.deepStrictEqual(plain(chips.map((c) => c.disabled)), [false, false, true, true]);
  assert.strictEqual(t.chosenResolution(view, chips), 'hour');
  assert.strictEqual(t.chosenResolution({ range: '24h', resolution: 'auto' }, chips), 'hour');
  assert.strictEqual(t.litResolution(view, chips), 'auto');
});
