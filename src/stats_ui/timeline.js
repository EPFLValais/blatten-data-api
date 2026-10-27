// The overview strip and the period it selects. Periods are whole UTC days, `from..=to`, as the
// statistics routes take them; the strip maps them to and from x positions over its plot, which is
// uPlot's selection there.
const DAY_MS = 86400000;
const STEP_MS = { hour: 3600000, day: DAY_MS, week: 7 * DAY_MS };

const isoDay = (ms) => new Date(ms).toISOString().slice(0, 10);
const dayStart = (day) => Date.parse(day + 'T00:00Z');
// Buckets are `YYYY-MM-DD` by day and week, `YYYY-MM-DDTHH:00Z` by hour.
const bucketStart = (b) => (b.length === 10 ? dayStart(b) : Date.parse(b));

// The scale of a strip whose bars, one per bucket, fill `left..left + width`.
function stripScale(buckets, resolution, left, width) {
  const first = buckets.length ? bucketStart(buckets[0]) : dayStart(isoDay(Date.now()));
  const end = buckets.length ? bucketStart(buckets[buckets.length - 1]) + STEP_MS[resolution] : first + DAY_MS;
  const perMs = width / Math.max(1, end - first);
  return {
    left, width, first, end,
    firstDay: isoDay(first),
    lastDay: isoDay(end - 1),
    x: (ms) => left + (Math.min(end, Math.max(first, ms)) - first) * perMs,
    at: (x) => first + (Math.min(left + width, Math.max(left, x)) - left) / perMs,
    pxPerDay: DAY_MS * perMs,
    bucketAt: (x) => Math.min(buckets.length - 1, Math.max(0, Math.floor((x - left) / (width / Math.max(1, buckets.length))))),
  };
}

// Where the window over `period` sits on the strip; without a period it covers everything.
function windowSpan(period, s) {
  return {
    x0: period.from ? s.x(dayStart(period.from)) : s.left,
    x1: period.to ? s.x(dayStart(period.to) + DAY_MS) : s.left + s.width,
  };
}

// The selection, from the left of the plot, that shows the window over `period`.
function stripSelect(period, s) {
  const { x0, x1 } = windowSpan(period, s);
  return { left: x0 - s.left, width: x1 - x0 };
}

// The days under a selection `left..left + width` from the left of the plot.
function selectPeriod(s, left, width) {
  const from = isoDay(s.at(s.left + left));
  const to = isoDay(Math.max(dayStart(from), s.at(s.left + left + width) - 1));
  return { from, to };
}

// What a drag starting at `x` does: within EDGE_PX of an end of the window it moves that end,
// inside the window it moves the window, and elsewhere, or over all time, it brushes a new one.
const EDGE_PX = 4;
function dragMode(s, period, x) {
  if (!period.from && !period.to) return 'brush';
  const { x0, x1 } = windowSpan(period, s);
  if (Math.abs(x - x0) <= EDGE_PX) return 'from';
  if (Math.abs(x - x1) <= EDGE_PX) return 'to';
  return x > x0 && x < x1 ? 'move' : 'brush';
}

// The period after a drag from `drag.start` to `x`. A brush selects the days under it; a move
// shifts `drag.span` by whole days and keeps it inside the strip; dragging an end moves it to the
// day boundary nearest `x`, never past the other end.
function dragWindow(s, drag, x) {
  if (drag.mode === 'brush') {
    const a = Math.min(drag.start, x);
    return selectPeriod(s, a - s.left, Math.max(drag.start, x) - a);
  }
  const boundary = () => Math.round(s.at(x) / DAY_MS) * DAY_MS;
  if (drag.mode === 'from') {
    return { from: isoDay(Math.min(boundary(), dayStart(drag.span.to))), to: drag.span.to };
  }
  if (drag.mode === 'to') {
    return { from: drag.span.from, to: isoDay(Math.max(boundary() - DAY_MS, dayStart(drag.span.from))) };
  }
  const from = dayStart(drag.span.from);
  const to = dayStart(drag.span.to);
  const lo = dayStart(s.firstDay) - from;
  const hi = dayStart(s.lastDay) - to;
  const shift = Math.min(hi, Math.max(lo, Math.round((x - drag.start) / s.pxPerDay) * DAY_MS));
  return { from: isoDay(from + shift), to: isoDay(to + shift) };
}

// The days one bar covers, for zooming into it.
function bucketPeriod(bucket, resolution) {
  const from = bucket.slice(0, 10);
  return { from, to: resolution === 'week' ? isoDay(dayStart(from) + 6 * DAY_MS) : from };
}

// The view fields a window sets.
function windowView(period) {
  return { range: 'custom', from: period.from, to: period.to };
}

// The period of a view: a preset counted back from today, a custom window, or all time.
function periodOf(view, now) {
  if (view.range === 'all') return {};
  if (view.range === 'custom') {
    const p = {};
    if (view.from) p.from = view.from;
    if (view.to) p.to = view.to;
    return p;
  }
  const today = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  return { from: isoDay(today - (Number(view.range) - 1) * DAY_MS), to: isoDay(today) };
}

// Runs `load(signal, ...args)` for the latest call only. `soon` waits until calls have stopped
// for `wait` ms; each run aborts the one before it, so an answer that arrives late is dropped.
function latestLoader(load, wait) {
  let timer = null;
  let controller = null;
  function now(...args) {
    clearTimeout(timer);
    timer = null;
    if (controller) controller.abort();
    controller = new AbortController();
    return load(controller.signal, ...args);
  }
  return {
    now,
    soon(...args) {
      clearTimeout(timer);
      timer = setTimeout(() => now(...args), wait);
    },
  };
}
