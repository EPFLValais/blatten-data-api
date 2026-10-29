// The series the charts draw, shaped as uPlot takes them: one array of bucket starts in seconds,
// then one array of values per group, aligned with it. Reads the bucket helpers of timeline.js.

// The chosen measure of each group of a series response, a missing or empty bucket as 0. A period
// without groups gives one flat line, so the chart still shows its buckets.
function chartData(series, metric) {
  const x = series.buckets.map((b) => bucketStart(b) / 1000);
  const groups = series.groups.length ? series.groups : [{ name: 'All' }];
  const values = (g) => x.map((_, i) => (g[metric] && g[metric][i]) || 0);
  return { names: groups.map((g) => g.name), data: [x, ...groups.map(values)] };
}

// The period a selection from `lo` to `hi` (seconds) zooms to: the whole days of the buckets whose
// points it covers, or of the bucket nearest its middle when it falls between two points.
function selectionPeriod(buckets, resolution, lo, hi) {
  const starts = buckets.map(bucketStart);
  const a = Math.min(lo, hi) * 1000;
  const b = Math.max(lo, hi) * 1000;
  let first = starts.findIndex((s) => s >= a);
  let last = starts.length - 1 - [...starts].reverse().findIndex((s) => s <= b);
  if (first < 0 || last >= starts.length || first > last) {
    const mid = (a + b) / 2;
    first = last = starts.reduce((best, s, i) => (Math.abs(s - mid) < Math.abs(starts[best] - mid) ? i : best), 0);
  }
  return {
    from: bucketPeriod(buckets[first], resolution).from,
    to: bucketPeriod(buckets[last], resolution).to,
  };
}
