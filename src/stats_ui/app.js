(async () => {
  const $ = (id) => document.getElementById(id);
  const cfg = await (await fetch('/stats/config.json')).json();
  const redirect = location.origin + '/stats';
  const POLL_MS = 5000;
  const PAGE = 25;
  const FILE_PAGE = 25;

  const DIMS = {
    collection: 'Collection', provider: 'Provider', data_type: 'Data type', dataset: 'Dataset',
    sensor: 'Sensor', format: 'Format', level: 'Processing level',
  };
  const GROUPS = { ...DIMS, item: 'Item' };
  const MEASURES = { requests: 'file requests', bytes: 'data sent', whole_files: 'whole files (est.)' };

  // ---- state ----------------------------------------------------------------------------------
  const saved = (() => { try { return JSON.parse(sessionStorage.getItem('stats_view') || '{}'); } catch (e) { return {}; } })();
  const view = {
    range: '7', from: '', to: '', group: 'collection', metric: 'requests',
    filter: {}, sort: { key: 'requests', dir: 'desc' }, q: '', offset: 0, ...saved,
  };
  delete view.cumulative; // a toggle the page no longer has
  if (!GROUPS[view.group]) view.group = 'collection';
  if (!MEASURES[view.metric]) view.metric = 'requests';
  const open = new Map(); // item code -> number of file rows loaded
  const persist = () => sessionStorage.setItem('stats_view', JSON.stringify(view));

  let token = null;
  let refreshToken = null;
  let idToken = null; // kept for the sign-out, in memory only
  let refreshTimer = null;
  let clockSkew = 0; // server time minus browser time, in seconds
  let lastIngest = null;
  let paused = false;
  let signedOut = false;
  let loadId = 0;
  let facets = {};
  let overview = null;
  let series = null;
  let groupRows = [];
  let strip = null; // { series, scale }: all-time totals under the window

  // ---- formatting -----------------------------------------------------------------------------
  const nf = new Intl.NumberFormat('en-GB');
  const num = (n) => nf.format(n);
  const whole = (n) => (n < 10 ? n.toFixed(1) : num(Math.round(n)));
  function bytes(n) {
    if (n < 1024) return n + ' B';
    const units = ['KiB', 'MiB', 'GiB', 'TiB'];
    let v = n;
    let i = -1;
    do { v /= 1024; i++; } while (v >= 1024 && i < units.length - 1);
    return v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2) + ' ' + units[i];
  }
  const setText = (id, text) => { $(id).textContent = text; };

  // ---- sign-in (authorization code with PKCE) -------------------------------------------------
  const b64url = (buf) =>
    btoa(String.fromCharCode(...new Uint8Array(buf))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  const random = () => b64url(crypto.getRandomValues(new Uint8Array(32)));

  async function login() {
    const verifier = random();
    const state = random();
    sessionStorage.setItem('pkce_verifier', verifier);
    sessionStorage.setItem('pkce_state', state);
    const challenge = b64url(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier)));
    location.assign(cfg.authUrl + '?' + new URLSearchParams({
      client_id: cfg.clientId, redirect_uri: redirect, response_type: 'code', scope: 'openid',
      state, code_challenge: challenge, code_challenge_method: 'S256',
    }));
  }

  function keepTokens(body) {
    token = body.access_token;
    refreshToken = body.refresh_token || null;
    if (body.id_token) idToken = body.id_token;
    clearTimeout(refreshTimer);
    if (refreshToken) refreshTimer = setTimeout(refresh, Math.max(10, (body.expires_in || 300) - 30) * 1000);
  }

  async function tokenRequest(params) {
    const res = await fetch(cfg.tokenUrl, { method: 'POST', body: new URLSearchParams({ client_id: cfg.clientId, ...params }) });
    if (!res.ok) throw new Error('token request failed');
    keepTokens(await res.json());
  }

  async function refresh() {
    try { await tokenRequest({ grant_type: 'refresh_token', refresh_token: refreshToken }); } catch (e) { await login(); }
  }

  function signOut() {
    const url = signOutUrl(cfg, idToken, redirect);
    signedOut = paused = true;
    clearTimeout(refreshTimer);
    token = refreshToken = idToken = null;
    forgetSession(sessionStorage);
    location.assign(url);
  }
  $('signout').addEventListener('click', signOut);
  $('access-signout').addEventListener('click', signOut);

  const params = new URLSearchParams(location.search);
  if (!params.has('code')) { await login(); return; }
  {
    const code = params.get('code');
    const ok = params.get('state') && params.get('state') === sessionStorage.getItem('pkce_state');
    history.replaceState(null, '', redirect);
    if (!ok) { showAccess(document, 'Sign-in could not be verified. Reload to try again.'); return; }
    try {
      await tokenRequest({
        grant_type: 'authorization_code', code, redirect_uri: redirect,
        code_verifier: sessionStorage.getItem('pkce_verifier') || '',
      });
    } catch (e) { showAccess(document, 'Sign-in failed. Reload to try again.'); return; }
    sessionStorage.removeItem('pkce_verifier');
    sessionStorage.removeItem('pkce_state');
  }

  // ---- requests -------------------------------------------------------------------------------
  const period = () => periodOf(view, new Date());
  const query = (extra, p = period()) => {
    const active = Object.fromEntries(Object.entries(view.filter).filter(([, v]) => v));
    return new URLSearchParams({ ...p, ...active, ...extra }).toString();
  };

  const guard = signInGuard(sessionStorage, login);
  async function api(path, signal) {
    const res = await fetch(path, { headers: { Authorization: 'Bearer ' + token }, signal });
    // A request answered after signing out must not sign in again or write session storage.
    if (signedOut) throw new Error('signed out');
    const guarded = res.status === 401 ? await guard.refused() : guard.accepted();
    const message = accessAfter(res.status, guarded);
    if (message !== undefined) showAccess(document, message);
    if (res.status === 401) throw new Error(guarded === 'signing-in' ? 'signing in again' : 'refused');
    if (res.status === 403) throw new Error('forbidden');
    if (!res.ok) throw new Error('request failed');
    return res;
  }

  // ---- charts ---------------------------------------------------------------------------------
  const PALETTE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#8c564b'];
  const GREYS = { 'Documents and other': '#7f8790', 'Smaller groups': '#b9c0c7' };
  const colour = (name, i) => GREYS[name] || PALETTE[i % PALETTE.length];

  // A uPlot chart in `box`, made once the box has a width: the dashboard is hidden until access is
  // granted, and a chart made in a hidden box has no size. `options(width)` gives its options for
  // the data last set; data under the same `key` is swapped in, a new key makes the chart again.
  function lineChart(box, options) {
    const c = { plot: null, key: null, data: null };
    const build = () => {
      if (c.plot) c.plot.destroy();
      c.plot = new uPlot(options(box.clientWidth), c.data, box);
      c.built = c.key;
    };
    c.set = (key, data) => {
      c.key = key;
      c.data = data;
      if (!box.clientWidth) return;
      if (c.plot && c.built === key) c.plot.setData(data); else build();
    };
    new ResizeObserver(() => {
      if (!c.data || !box.clientWidth) return;
      if (c.plot && c.built === c.key) c.plot.setSize({ width: box.clientWidth, height: c.plot.height }); else build();
    }).observe(box);
    return c;
  }

  // Buckets are UTC, so the axes and the values on hover are read in UTC too.
  const utc = (ts) => uPlot.tzDate(new Date(ts * 1e3), 'Etc/UTC');
  const fmtOf = (m) => (m === 'bytes' ? bytes : m === 'whole_files' ? whole : num);
  const shown = (fmt) => (u, v) => (v === null || v === undefined ? '–' : fmt(v));
  const bucketText = (resolution) => (ts) => {
    const d = new Date(ts * 1000).toISOString();
    if (resolution === 'hour') return longDay(d.slice(0, 10)) + ', ' + d.slice(11, 16) + ' UTC';
    return (resolution === 'week' ? 'Week of ' : '') + longDay(d.slice(0, 10));
  };
  const lineSeries = (name, i, fmt) => ({
    label: name, stroke: colour(name, i), width: 1.5, value: shown(fmt),
    points: { size: 5, fill: colour(name, i) },
  });
  const yAxis = (fmt) => ({ size: 64, values: (u, ticks) => ticks.map(fmt) });
  const fromZero = { range: (u, min, max) => uPlot.rangeNum(0, Math.max(1, max), 0.1, true) };
  // Day and month, with the hour when ticks are less than a day apart, in UTC or local time.
  const timeAxis = (timeZone) => ({
    values: (u, ticks, axis, space, incr) => ticks.map((ts) => {
      const d = new Date(ts * 1000);
      const day = d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', timeZone });
      return incr < 86400 ? d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', timeZone }) + '\n' + day : day;
    }),
  });

  // Dragging across the main chart zooms the window to the days under the selection; a double
  // click goes back to the period shown before the first zoom.
  let zoomBack = null;
  const main = lineChart($('chart-main'), (width) => {
    const fmt = fmtOf(view.metric);
    return {
      width, height: 220, tzDate: utc,
      scales: { x: { time: true }, y: fromZero },
      series: [
        { label: 'Period', value: shown(bucketText(series.resolution)) },
        ...chartData(series, view.metric).names.map((name, i) => lineSeries(name, i, fmt)),
      ],
      axes: [timeAxis('UTC'), yAxis(fmt)],
      cursor: { drag: { x: true, y: false, setScale: false } },
      hooks: {
        setSelect: [(u) => {
          if (u.select.width < 3) return;
          const lo = u.posToVal(u.select.left, 'x');
          const hi = u.posToVal(u.select.left + u.select.width, 'x');
          u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
          if (!zoomBack) zoomBack = { range: view.range, from: view.from, to: view.to };
          setWindow(selectionPeriod(series.buckets, series.resolution, lo, hi), true);
        }],
        ready: [(u) => u.over.addEventListener('dblclick', zoomOut)],
      },
    };
  });

  function zoomOut() {
    if (!zoomBack) return;
    Object.assign(view, zoomBack);
    zoomBack = null;
    rangeChanged();
  }

  function renderMain() {
    if (!series) return;
    const caption = MEASURES[view.metric] + ' per ' + series.resolution + ', by ' + GROUPS[view.group].toLowerCase();
    setText('cap-main', caption.charAt(0).toUpperCase() + caption.slice(1));
    const { names, data } = chartData(series, view.metric);
    main.set([view.metric, series.resolution, ...names].join('\n'), data);
    windowLabel(series.resolution);
  }

  // ---- overview strip and its window ----------------------------------------------------------
  const longDay = (day) => new Date(day + 'T00:00Z').toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
  function windowLabel(resolution) {
    const p = period();
    const from = p.from || (strip && strip.scale.firstDay);
    const to = p.to || (strip && strip.scale.lastDay);
    const span = from && to ? (from === to ? longDay(from) : longDay(from) + ' – ' + longDay(to)) : 'All time';
    setText('window-label', span + (resolution ? ' · by ' + resolution : ' · loading…'));
  }

  // The strip and the live chart count requests, or bytes when data sent is the measure.
  const totalMetric = () => (view.metric === 'bytes' ? 'bytes' : 'requests');
  // Ticks on the strip are whole days or longer, so its one line of labels always shows the date.
  const DAY_INCRS = [1, 2, 7, 14, 30, 91, 182, 365].map((d) => d * 86400);
  // The strip is one line of all-time totals; its uPlot selection is the window. Its x scale runs
  // from the first bucket's start to the last one's end, so it matches `strip.scale`.
  const stripChart = lineChart($('chart-strip'), (width) => ({
    width, height: 80, tzDate: utc, legend: { show: false },
    scales: { x: { time: true, range: () => [strip.scale.first / 1e3, strip.scale.end / 1e3] }, y: fromZero },
    series: [{}, { stroke: PALETTE[0], width: 1.5, points: { show: false } }],
    axes: [{ ...timeAxis('UTC'), size: 30, incrs: DAY_INCRS }, { show: false }],
    cursor: { drag: { x: false, y: false }, points: { show: false }, y: false },
    hooks: { ready: [stripPointer], draw: [drawWindow] },
  }));

  function renderStrip() {
    if (!strip) return;
    stripChart.set(totalMetric(), chartData(strip.series, totalMetric()).data.slice(0, 2));
  }

  // Called by the strip's draw hook with the plot being drawn, which is not yet `stripChart.plot`
  // while it is being made.
  function drawWindow(u = stripChart.plot) {
    if (!strip || !u) return;
    strip.scale = stripScale(strip.series.buckets, strip.series.resolution, 0, u.over.clientWidth);
    u.setSelect({ ...stripSelect(period(), strip.scale), top: 0, height: u.over.clientHeight }, false);
  }

  // Dragging on the strip brushes a new window, moves the window or moves one of its ends, as
  // `dragMode` says for where it starts; a click zooms into the bucket under it. uPlot's own
  // dragging is kept off, so the window is only ever the period.
  function stripPointer(u) {
    const over = u.over;
    const CURSORS = { brush: 'crosshair', move: 'grab', from: 'ew-resize', to: 'ew-resize' };
    let drag = null;
    const xOf = (e) => e.clientX - over.getBoundingClientRect().left;
    over.addEventListener('pointerdown', (e) => {
      if (!strip || e.button !== 0) return;
      e.preventDefault();
      const x = xOf(e);
      const p = period();
      const sel = stripSelect(p, strip.scale);
      drag = { mode: dragMode(strip.scale, p, x), start: x, span: selectPeriod(strip.scale, sel.left, sel.width), moved: false };
      over.setPointerCapture(e.pointerId);
    });
    over.addEventListener('pointermove', (e) => {
      if (!strip) return;
      const x = xOf(e);
      if (!drag) { over.style.cursor = CURSORS[dragMode(strip.scale, period(), x)]; return; }
      if (!drag.moved && Math.abs(x - drag.start) < 3) return;
      drag.moved = true;
      setWindow(dragWindow(strip.scale, drag, x), false);
    });
    over.addEventListener('pointerup', (e) => {
      if (!drag) return;
      const d = drag;
      drag = null;
      if (d.moved) { loads.now(); return; }
      const i = strip.scale.bucketAt(xOf(e));
      setWindow(bucketPeriod(strip.series.buckets[i], strip.series.resolution), true);
    });
    over.addEventListener('pointercancel', () => { drag = null; });
  }

  // Moves the window to `p` and reloads the page's numbers for it, at once or once dragging stops.
  function setWindow(p, immediately) {
    Object.assign(view, windowView(p));
    view.offset = 0;
    open.clear();
    persist();
    $('range').value = 'custom';
    $('custom').hidden = false;
    $('from').value = view.from;
    $('to').value = view.to;
    drawWindow();
    windowLabel(null);
    if (immediately) loads.now(); else loads.soon();
  }

  async function loadStrip() {
    const s = await (await api('/stats/api/series?' + query({ dim: 'total' }, {}))).json();
    strip = { series: s, scale: stripScale(s.buckets, s.resolution, 0, 1) };
    renderStrip();
    drawWindow();
    windowLabel(series && series.resolution);
  }

  const live = lineChart($('chart-live'), (width) => {
    const fmt = fmtOf(totalMetric());
    const at = (ts) => new Date(ts * 1000).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
    return {
      width, height: 110,
      scales: { x: { time: true }, y: fromZero },
      series: [{ label: 'Quarter hour from', value: shown(at) }, lineSeries('All', 0, fmt)],
      axes: [timeAxis(undefined), yAxis(fmt)],
      cursor: { drag: { x: false, y: false } },
    };
  });

  function renderLive() {
    if (!overview) return;
    live.set(totalMetric(), liveData(overview.timeline, overview.now, totalMetric()));
  }

  // ---- numbers --------------------------------------------------------------------------------
  function renderOverview(o) {
    overview = o;
    clockSkew = o.now - Date.now() / 1000;
    lastIngest = o.last_ingest_at;
    setText('k-requests', num(o.totals.requests));
    setText('k-whole', whole(o.totals.whole_files));
    setText('k-bytes', bytes(o.totals.bytes));
    setText('k-files', num(o.files_downloaded));
    setText('k-files-sub', 'of ' + num(o.files_total));
    setText('k-items', num(o.items_with_downloads));
    setText('k-items-sub', 'of ' + num(o.items_total));
    setText('k-hour', num(o.last_hour.requests));
    setText('k-hour-sub', num(o.last_24h.requests) + ' in 24 h');
    renderLive();
    updateAge();
  }

  // ---- table ----------------------------------------------------------------------------------
  const COLUMNS = [
    { key: 'name', num: false },
    { key: 'requests', label: 'File requests', num: true },
    { key: 'whole_files', label: 'Whole files (est.)', num: true },
    { key: 'bytes', label: 'Data sent', num: true },
    { key: 'files', label: 'Files requested', num: true },
    { key: 'files_total', label: 'Files in catalog', num: true },
  ];

  function setSort(key) {
    view.sort = view.sort.key === key
      ? { key, dir: view.sort.dir === 'asc' ? 'desc' : 'asc' }
      : { key, dir: key === 'name' ? 'asc' : 'desc' };
    view.offset = 0;
    persist();
    loadTable(loadId);
  }

  function headerRow(table, label, onSort) {
    const tr = table.insertRow();
    for (const c of COLUMNS) {
      const th = document.createElement('th');
      const name = c.key === 'name' ? label : c.label;
      th.textContent = name + (view.sort.key === c.key ? (view.sort.dir === 'asc' ? ' ▲' : ' ▼') : '');
      if (c.num) th.classList.add('n');
      th.addEventListener('click', () => onSort(c.key));
      tr.appendChild(th);
    }
  }

  function cell(tr, text, isNum) {
    const td = tr.insertCell();
    td.textContent = text;
    if (isNum) td.classList.add('n');
    return td;
  }

  function valueRow(tr, r) {
    cell(tr, num(r.requests), true);
    cell(tr, whole(r.whole_files), true);
    cell(tr, bytes(r.bytes), true);
    cell(tr, num(r.files_downloaded), true);
    cell(tr, num(r.files_total), true);
  }

  function renderGroups() {
    const t = $('table');
    t.replaceChildren();
    headerRow(t, GROUPS[view.group], setSort);
    const key = view.sort.key === 'files' ? 'files_downloaded' : view.sort.key;
    const sign = view.sort.dir === 'asc' ? 1 : -1;
    const rows = [...groupRows].sort((a, b) =>
      (key === 'name' ? a.name.localeCompare(b.name) : a[key] - b[key]) * sign || a.name.localeCompare(b.name));
    for (const r of rows) {
      const tr = t.insertRow();
      tr.classList.add('row');
      if (view.filter[view.group] === r.name) tr.classList.add('active');
      cell(tr, r.name);
      valueRow(tr, r);
      tr.addEventListener('click', () => {
        view.filter[view.group] = view.filter[view.group] === r.name ? '' : r.name;
        filtersChanged();
      });
    }
  }

  async function loadFiles(code, count, box) {
    const data = await (await api('/stats/api/items/' + encodeURIComponent(code) + '/files?' + query({ limit: String(count) }))).json();
    box.replaceChildren();
    if (!data.rows.length) { box.textContent = 'No requests in this period.'; return; }
    const t = document.createElement('table');
    const head = t.insertRow();
    for (const [i, name] of ['File', 'File requests', 'Whole files (est.)', 'Data sent'].entries()) {
      const th = document.createElement('th');
      th.textContent = name;
      if (i) th.classList.add('n');
      head.appendChild(th);
    }
    for (const r of data.rows) {
      const tr = t.insertRow();
      cell(tr, r.file);
      cell(tr, num(r.requests), true);
      cell(tr, whole(r.whole_files), true);
      cell(tr, bytes(r.bytes), true);
    }
    box.appendChild(t);
    open.set(code, data.rows.length);
    if (data.total > data.rows.length) {
      const more = document.createElement('button');
      more.type = 'button';
      more.textContent = 'Show more (' + num(data.total - data.rows.length) + ' left)';
      more.addEventListener('click', (e) => { e.stopPropagation(); loadFiles(code, data.rows.length + FILE_PAGE, box); });
      box.appendChild(more);
    }
  }

  function renderItems(data) {
    const t = $('table');
    t.replaceChildren();
    headerRow(t, 'Item', setSort);
    for (const r of data.rows) {
      const tr = t.insertRow();
      tr.classList.add('row', 'item');
      const first = cell(tr, r.item);
      const sub = document.createElement('span');
      sub.className = 'title';
      sub.textContent = r.title + ' · ' + r.provider + ' · ' + r.data_type;
      first.appendChild(sub);
      valueRow(tr, r);
      let detail = null;
      const toggle = async () => {
        if (detail) { detail.remove(); detail = null; tr.classList.remove('open'); open.delete(r.item); return; }
        detail = t.insertRow(tr.rowIndex + 1);
        detail.classList.add('files');
        const td = detail.insertCell();
        td.colSpan = COLUMNS.length;
        tr.classList.add('open');
        await loadFiles(r.item, open.get(r.item) || FILE_PAGE, td);
      };
      tr.addEventListener('click', toggle);
      if (open.has(r.item)) toggle();
    }
    const from = data.total ? data.offset + 1 : 0;
    setText('count', num(from) + '–' + num(data.offset + data.rows.length) + ' of ' + num(data.total));
    $('prev').disabled = data.offset === 0;
    $('next').disabled = data.offset + data.rows.length >= data.total;
  }

  async function loadTable(id, signal) {
    const items = view.group === 'item';
    $('item-tools').hidden = !items;
    if (items) {
      const sort = view.sort.key === 'name' ? 'item' : view.sort.key;
      const data = await (await api('/stats/api/items?' + query({
        q: view.q, sort, dir: view.sort.dir, limit: String(PAGE), offset: String(view.offset),
      }), signal)).json();
      if (id === loadId) renderItems(data);
    } else {
      const data = await (await api('/stats/api/groups?' + query({ dim: view.group }), signal)).json();
      if (id !== loadId) return;
      groupRows = data.rows;
      renderGroups();
    }
  }

  // ---- loading and live updates ---------------------------------------------------------------
  // The numbers, chart and table all read one period, the window's. A later load aborts the one
  // before it, so dragging the window applies only where it was let go.
  async function loadAll(signal) {
    const id = ++loadId;
    const jobs = [
      api('/stats/api/overview?' + query(), signal).then((r) => r.json()).then((o) => { if (id === loadId) renderOverview(o); }),
      api('/stats/api/series?' + query({ dim: view.group }), signal).then((r) => r.json()).then((s) => { if (id === loadId) { series = s; renderMain(); } }),
      loadTable(id, signal),
    ];
    await Promise.allSettled(jobs);
  }
  const loads = latestLoader(loadAll, 250);

  function updateAge() {
    $('live').classList.toggle('paused', paused);
    $('pause').textContent = paused ? 'Resume' : 'Pause';
    const stale = $('stale');
    if (lastIngest === null || lastIngest === undefined) {
      setText('live-text', paused ? 'Paused' : 'Waiting for data');
      stale.hidden = false;
      stale.textContent = 'No statistics have been received yet.';
      return;
    }
    const age = Math.max(0, Math.round(Date.now() / 1000 + clockSkew - lastIngest));
    setText('live-text', (paused ? 'Paused · ' : 'Live · ') + 'updated ' + (age < 90 ? age + ' s ago' : Math.round(age / 60) + ' min ago'));
    stale.hidden = age < 900;
    stale.textContent = 'No new data for ' + Math.round(age / 60) + ' minutes. The collection of counters may have stopped.';
  }

  async function poll() {
    if (paused || document.hidden) return;
    try {
      const s = await (await api('/stats/api/status')).json();
      if (s.last_ingest_at !== lastIngest) await Promise.allSettled([loadStrip(), loads.now()]);
    } catch (e) { /* retry on the next tick */ }
  }
  setInterval(poll, POLL_MS);
  setInterval(updateAge, 1000);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });

  // ---- filters and controls -------------------------------------------------------------------
  function filtersChanged() {
    view.offset = 0;
    open.clear();
    persist();
    renderFilters();
    loadStrip().catch(() => {});
    loads.now();
  }

  function renderFilters() {
    const select = $('add-filter');
    select.replaceChildren();
    const first = document.createElement('option');
    first.value = '';
    first.textContent = 'Add filter…';
    select.appendChild(first);
    for (const [dim, label] of Object.entries(DIMS)) {
      const group = document.createElement('optgroup');
      group.label = label;
      for (const v of facets[dim] || []) {
        const o = document.createElement('option');
        o.value = dim + '|' + v.value;
        o.textContent = v.value + ' (' + num(v.items) + ')';
        group.appendChild(o);
      }
      select.appendChild(group);
    }
    select.value = '';
    const chips = $('chips');
    chips.replaceChildren();
    for (const [dim, label] of Object.entries(DIMS)) {
      if (!view.filter[dim]) continue;
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.appendChild(document.createTextNode(label + ': ' + view.filter[dim]));
      const x = document.createElement('button');
      x.type = 'button';
      x.textContent = '×';
      x.setAttribute('aria-label', 'Remove filter');
      x.addEventListener('click', () => { view.filter[dim] = ''; filtersChanged(); });
      chip.appendChild(x);
      chips.appendChild(chip);
    }
  }

  async function download(level) {
    const res = await api('/stats/api/export.csv?' + query({ level_of_detail: level }));
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement('a');
    a.href = url;
    a.download = level === 'file' ? 'download-statistics-by-file.csv' : 'download-statistics.csv';
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  }

  facets = await (await api('/stats/api/facets')).json();
  for (const [key, label] of Object.entries(GROUPS)) {
    const o = document.createElement('option');
    o.value = key;
    o.textContent = label;
    $('group').appendChild(o);
  }
  $('group').value = view.group;
  $('range').value = view.range;
  $('from').value = view.from;
  $('to').value = view.to;
  $('custom').hidden = view.range !== 'custom';
  for (const b of $('metric').children) b.classList.toggle('on', b.dataset.value === view.metric);
  renderFilters();

  $('group').addEventListener('change', () => {
    view.group = $('group').value;
    view.offset = 0;
    view.sort = { key: view.metric, dir: 'desc' };
    open.clear();
    persist();
    loads.now();
  });
  $('add-filter').addEventListener('change', () => {
    const [dim, ...rest] = $('add-filter').value.split('|');
    if (dim) { view.filter[dim] = rest.join('|'); filtersChanged(); }
  });
  function rangeChanged() {
    $('range').value = view.range;
    $('custom').hidden = view.range !== 'custom';
    $('from').value = view.from;
    $('to').value = view.to;
    view.offset = 0;
    open.clear();
    persist();
    drawWindow();
    windowLabel(null);
    loads.now();
  }
  $('range').addEventListener('change', () => { view.range = $('range').value; zoomBack = null; rangeChanged(); });
  for (const id of ['from', 'to']) {
    $(id).addEventListener('change', () => { view[id] = $(id).value; zoomBack = null; rangeChanged(); });
  }
  $('metric').addEventListener('click', (e) => {
    const value = e.target.dataset && e.target.dataset.value;
    if (!value) return;
    view.metric = value;
    view.sort = { key: value, dir: 'desc' };
    for (const b of $('metric').children) b.classList.toggle('on', b === e.target);
    persist();
    renderMain();
    renderLive();
    renderStrip();
    loadTable(loadId);
  });
  $('export').addEventListener('change', async () => {
    const level = $('export').value;
    $('export').value = '';
    if (level) await download(level);
  });
  $('pause').addEventListener('click', () => { paused = !paused; updateAge(); if (!paused) poll(); });
  let searchTimer = null;
  $('search').value = view.q;
  $('search').addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { view.q = $('search').value.trim(); view.offset = 0; persist(); loadTable(loadId); }, 250);
  });
  $('prev').addEventListener('click', () => { view.offset = Math.max(0, view.offset - PAGE); loadTable(loadId); });
  $('next').addEventListener('click', () => { view.offset += PAGE; loadTable(loadId); });

  await Promise.allSettled([loadStrip(), loads.now()]);
})();
