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

// What a drag starting at `x` does: within HANDLE_PX of an end of the window, all time included,
// it moves the nearer end; inside a chosen window it moves the window, and elsewhere, or inside
// the all-time window, it brushes a new one.
const HANDLE_PX = 8;
function dragMode(s, period, x) {
  const { x0, x1 } = windowSpan(period, s);
  const d0 = Math.abs(x - x0);
  const d1 = Math.abs(x - x1);
  if (Math.min(d0, d1) <= HANDLE_PX) {
    if (d0 !== d1) return d0 < d1 ? 'from' : 'to';
    return x <= (x0 + x1) / 2 ? 'from' : 'to';
  }
  if (!period.from && !period.to) return 'brush';
  return x > x0 && x < x1 ? 'move' : 'brush';
}

// Where the dates under the window's ends go, as left edges in a strip `width` wide: each centred
// under its end `x0` or `x1` and kept inside the strip, or, where the two would overlap, one label
// centred under the window. `w` holds the width of the date of each end and of both as one label.
const LABEL_GAP_PX = 6;
function endLabels(x0, x1, w, width) {
  const place = (x, wd) => Math.max(0, Math.min(width - wd, x - wd / 2));
  const from = place(x0, w.from);
  const to = place(x1, w.to);
  if (from + w.from + LABEL_GAP_PX <= to) return [{ end: 'from', left: from }, { end: 'to', left: to }];
  return [{ end: 'both', left: place((x0 + x1) / 2, w.both) }];
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

// The last 24 hours up to `now` (ms): the series routes take whole days, so they are asked for
// yesterday and today, and `x` (seconds) shows from the start of this hour yesterday until now.
function last24Hours(now) {
  return {
    from: isoDay(now - DAY_MS),
    to: isoDay(now),
    x: [Math.floor((now - DAY_MS) / STEP_MS.hour) * STEP_MS.hour / 1000, now / 1000],
  };
}

// The period of a view: a preset counted back from today, a custom window, or all time.
function periodOf(view, now) {
  if (view.range === 'all') return {};
  if (view.range === '24h') {
    const { from, to } = last24Hours(now.getTime());
    return { from, to };
  }
  if (view.range === 'custom') {
    const p = {};
    if (view.from) p.from = view.from;
    if (view.to) p.to = view.to;
    return p;
  }
  const today = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  return { from: isoDay(today - (Number(view.range) - 1) * DAY_MS), to: isoDay(today) };
}

// The preset whose button is lit for a view; a dragged or saved custom window lights none.
const PRESETS = ['24h', '1', '7', '30', 'all'];
function litPreset(view) {
  return PRESETS.includes(view.range) ? view.range : null;
}

// The start of the bucket holding `ms`; weeks start on Monday, UTC.
function bucketFloor(resolution, ms) {
  if (resolution === 'week') {
    const day = Math.floor(ms / DAY_MS) * DAY_MS;
    return day - ((new Date(day).getUTCDay() + 6) % 7) * DAY_MS;
  }
  return Math.floor(ms / STEP_MS[resolution]) * STEP_MS[resolution];
}

// The bars a series by `resolution` has over the days `from..=to`, counted as the series route
// counts them before refusing more than its limit: from the bucket holding the first day to the
// one holding the end of the last day, or `now` (ms) when that comes first.
function barCount(resolution, from, to, now) {
  const first = bucketFloor(resolution, dayStart(from));
  const last = bucketFloor(resolution, Math.min(dayStart(to) + DAY_MS - 1000, now));
  return Math.floor((last - first) / STEP_MS[resolution]) + 1;
}

// The resolution chips for a view over `period`. While Auto is in effect it names the resolution
// the server chose, `effective`, once a series has come back. A chip whose bars would pass `maxBuckets` is disabled
// with the reason; without a start, the window opens on `firstDay`, the first day with data, and
// until that is known only Auto can be chosen. The last 24 hours are always by hour.
const RESOLUTIONS = ['hour', 'day', 'week'];
function resolutionChips(view, period, firstDay, maxBuckets, now, effective) {
  const from = period.from || firstDay;
  const to = period.to || isoDay(now);
  const chips = [{ value: 'auto', label: 'Auto (' + effective + ')', disabled: false, title: '' }];
  for (const r of RESOLUTIONS) {
    const chip = { value: r, label: r.charAt(0).toUpperCase() + r.slice(1), disabled: false, title: '' };
    if (view.range === '24h') {
      if (r !== 'hour') Object.assign(chip, { disabled: true, title: 'The last 24 hours are shown by hour' });
    } else if (!from || !maxBuckets) {
      Object.assign(chip, { disabled: true, title: 'Waiting for the first day with data' });
    } else {
      const n = barCount(r, from, to, now);
      if (n > maxBuckets) {
        const fmt = (v) => v.toLocaleString('en-GB');
        Object.assign(chip, { disabled: true, title: 'By ' + r + ' this period has ' + fmt(n) + ' bars, more than the ' + fmt(maxBuckets) + ' a chart shows' });
      }
    }
    chips.push(chip);
  }
  if (!effective || litResolution(view, chips) !== 'auto') chips[0].label = 'Auto';
  return chips;
}

// The chip lit for a view: its chosen resolution, or Auto when that one is disabled.
function litResolution(view, chips) {
  const chip = chips.find((c) => c.value === view.resolution);
  return chip && !chip.disabled ? chip.value : 'auto';
}

// The resolution a series request asks for, or null to let the server choose.
function chosenResolution(view, chips) {
  if (view.range === '24h') return 'hour';
  const lit = litResolution(view, chips);
  return lit === 'auto' ? null : lit;
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
