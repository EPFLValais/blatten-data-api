// A box beside the pointer on a uPlot chart with the time of the bucket under it and each
// series' colour, name and value there.
const TOOLTIP_GAP = 12;

// Where a box of `box` size goes for a pointer at `at` in a plot of `plot` size, all in CSS
// pixels from the plot's top left: right of and below the pointer, on the other side where it
// would cross the right or bottom edge, and never past the left or top.
function tooltipPlace(at, box, plot) {
  let left = at.left + TOOLTIP_GAP;
  let top = at.top + TOOLTIP_GAP;
  if (left + box.width > plot.width) left = at.left - TOOLTIP_GAP - box.width;
  if (top + box.height > plot.height) top = at.top - TOOLTIP_GAP - box.height;
  return { left: Math.max(0, left), top: Math.max(0, top) };
}

// The uPlot plugin that draws the box. `title(ts)` gives the bucket's text and `value(v)` a
// series' value; names come from the series labels and colours from their strokes.
function tooltipPlugin({ title, value }) {
  let tip = null;
  const row = (colour, name, text) => {
    const r = document.createElement('div');
    r.className = 'u-tooltip-row';
    const swatch = document.createElement('span');
    swatch.className = 'u-tooltip-swatch';
    swatch.style.background = colour;
    const label = document.createElement('span');
    label.className = 'u-tooltip-name';
    label.textContent = name;
    const v = document.createElement('span');
    v.className = 'u-tooltip-value';
    v.textContent = text;
    r.append(swatch, label, v);
    return r;
  };
  return {
    hooks: {
      init: [(u) => {
        tip = document.createElement('div');
        tip.className = 'u-tooltip';
        tip.hidden = true;
        u.over.appendChild(tip);
        u.over.addEventListener('mouseleave', () => { tip.hidden = true; });
      }],
      setCursor: [(u) => {
        const { idx, left, top } = u.cursor;
        if (idx === null || idx === undefined || left < 0) { tip.hidden = true; return; }
        const head = document.createElement('div');
        head.className = 'u-tooltip-title';
        head.textContent = title(u.data[0][idx]);
        const rows = u.series.slice(1).map((s, i) => {
          const v = u.data[i + 1][idx];
          const stroke = typeof s.stroke === 'function' ? s.stroke(u, i + 1) : s.stroke;
          return row(stroke, s.label, v === null || v === undefined ? '–' : value(v));
        });
        tip.replaceChildren(head, ...rows);
        tip.hidden = false;
        const p = tooltipPlace(
          { left, top },
          { width: tip.offsetWidth, height: tip.offsetHeight },
          { width: u.over.clientWidth, height: u.over.clientHeight },
        );
        tip.style.left = p.left + 'px';
        tip.style.top = p.top + 'px';
      }],
    },
  };
}
