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
    range: '7', from: '', to: '', group: 'collection', metric: 'requests', cumulative: false,
    filter: {}, sort: { key: 'requests', dir: 'desc' }, q: '', offset: 0, ...saved,
  };
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
  let strip = null; // { series, scale, box }: all-time totals under the window

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
  const fmtMeasure = (n) => (view.metric === 'bytes' ? bytes(n) : view.metric === 'whole_files' ? whole(n) : num(n));
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
  const SVG = 'http://www.w3.org/2000/svg';
  const PALETTE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#8c564b'];
  const GREYS = { 'Documents and other': '#7f8790', 'Smaller groups': '#b9c0c7' };
  const colour = (name, i) => GREYS[name] || PALETTE[i % PALETTE.length];
  function el(name, attrs, text) {
    const n = document.createElementNS(SVG, name);
    for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
    if (text !== undefined) n.textContent = text;
    return n;
  }

  // groups: [{ name, values }] over labels, drawn as stacked bars. `onBar(i)` is called when the
  // bar of label i is clicked.
  const PLOT_LEFT = 52;
  const PLOT_RIGHT = 4;
  function stackedChart(svg, labels, groups, legend, onBar) {
    svg.replaceChildren();
    if (legend) legend.replaceChildren();
    const vb = svg.viewBox.baseVal;
    const W = vb.width, H = vb.height, left = PLOT_LEFT, bottom = 16, top = 8;
    const plotW = W - left - PLOT_RIGHT, plotH = H - bottom - top;
    const totals = labels.map((_, i) => groups.reduce((a, g) => a + g.values[i], 0));
    const max = Math.max(1, ...totals);
    for (const frac of [0, 0.5, 1]) {
      const y = top + plotH - frac * plotH;
      svg.appendChild(el('line', { class: 'grid', x1: left, y1: y, x2: W, y2: y }));
      svg.appendChild(el('text', { x: left - 4, y: y + (frac === 1 ? 7 : frac === 0 ? 0 : 3), 'text-anchor': 'end' }, fmtMeasure(max * frac)));
    }
    if (!labels.length) return;
    const slot = plotW / labels.length;
    const barW = Math.min(60, Math.max(1, slot * 0.8));
    labels.forEach((label, i) => {
      let y = top + plotH;
      groups.forEach((g, gi) => {
        const v = g.values[i];
        if (v <= 0) return;
        const h = Math.max(0.5, (v / max) * plotH);
        y -= h;
        const bar = el('rect', { x: left + i * slot + (slot - barW) / 2, y, width: barW, height: h, fill: colour(g.name, gi) });
        bar.appendChild(el('title', {}, label + ' · ' + g.name + ': ' + fmtMeasure(v)));
        if (onBar) { bar.classList.add('zoom'); bar.addEventListener('click', () => onBar(i)); }
        svg.appendChild(bar);
      });
    });
    svg.appendChild(el('text', { x: left, y: H - 3 }, labels[0]));
    svg.appendChild(el('text', { x: W - 4, y: H - 3, 'text-anchor': 'end' }, labels[labels.length - 1]));
    if (legend && groups.length > 1) {
      groups.forEach((g, gi) => {
        const item = document.createElement('span');
        const sw = el('svg', { viewBox: '0 0 10 10' });
        sw.appendChild(el('rect', { width: 10, height: 10, rx: 2, fill: colour(g.name, gi) }));
        item.appendChild(sw);
        item.appendChild(document.createTextNode(g.name));
        legend.appendChild(item);
      });
    }
  }

  function renderMain() {
    if (!series) return;
    const groups = series.groups.map((g) => {
      let running = 0;
      const raw = g[view.metric];
      return { name: g.name, values: view.cumulative ? raw.map((v) => (running += v)) : raw };
    });
    const caption = (view.cumulative ? 'Cumulative ' : '') + MEASURES[view.metric] + ' per ' + series.resolution + ', by ' + GROUPS[view.group].toLowerCase();
    setText('cap-main', caption.charAt(0).toUpperCase() + caption.slice(1));
    stackedChart($('chart-main'), series.buckets, groups, $('legend'), (i) => {
      if (series.resolution !== 'hour') setWindow(bucketPeriod(series.buckets[i], series.resolution), true);
    });
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

  function renderStrip() {
    if (!strip) return;
    const svg = $('chart-strip');
    const saveMetric = view.metric;
    view.metric = view.metric === 'bytes' ? 'bytes' : 'requests';
    stackedChart(svg, strip.series.buckets, [{ name: 'All', values: strip.series.groups.length ? strip.series.groups[0][view.metric] : strip.series.buckets.map(() => 0) }], null);
    view.metric = saveMetric;
    const vb = svg.viewBox.baseVal;
    strip.scale = stripScale(strip.series.buckets, strip.series.resolution, PLOT_LEFT, vb.width - PLOT_LEFT - PLOT_RIGHT);
    strip.box = el('rect', { class: 'window', y: 0, height: vb.height - 16 });
    svg.appendChild(strip.box);
    drawWindow();
  }

  function drawWindow() {
    if (!strip) return;
    const { x0, x1 } = windowSpan(period(), strip.scale);
    strip.box.setAttribute('x', x0);
    strip.box.setAttribute('width', Math.max(1, x1 - x0));
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
    strip = { series: s };
    renderStrip();
    windowLabel(series && series.resolution);
  }

  {
    const svg = $('chart-strip');
    let drag = null;
    const xOf = (e) => {
      const r = svg.getBoundingClientRect();
      return ((e.clientX - r.left) / r.width) * svg.viewBox.baseVal.width;
    };
    svg.addEventListener('pointerdown', (e) => {
      if (!strip) return;
      const x = xOf(e);
      const { x0, x1 } = windowSpan(period(), strip.scale);
      const span = { from: isoDay(strip.scale.at(x0)), to: isoDay(strip.scale.at(x1) - 1) };
      drag = { mode: view.range !== 'all' && x >= x0 && x <= x1 ? 'move' : 'brush', start: x, span, moved: false };
      svg.setPointerCapture(e.pointerId);
    });
    svg.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const x = xOf(e);
      if (!drag.moved && Math.abs(x - drag.start) < 3) return;
      drag.moved = true;
      setWindow(dragWindow(strip.scale, drag, x), false);
    });
    svg.addEventListener('pointerup', (e) => {
      if (!drag) return;
      const d = drag;
      drag = null;
      if (d.moved) { loads.now(); return; }
      const i = strip.scale.bucketAt(xOf(e));
      setWindow(bucketPeriod(strip.series.buckets[i], strip.series.resolution), true);
    });
    svg.addEventListener('pointercancel', () => { drag = null; });
  }

  function renderLive() {
    if (!overview) return;
    const m = view.metric === 'bytes' ? 'bytes' : 'requests';
    const bucket = 900;
    const now = overview.now;
    const start = Math.floor((now - 24 * 3600) / bucket) * bucket;
    const n = Math.floor((now - start) / bucket) + 1;
    const values = new Array(n).fill(0);
    for (const p of overview.timeline) {
      const i = Math.floor((p.ts - start) / bucket);
      if (i >= 0 && i < n) values[i] += p[m];
    }
    const labels = values.map((_, i) =>
      new Date((start + i * bucket) * 1000).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }));
    const saveMetric = view.metric;
    view.metric = m;
    stackedChart($('chart-live'), labels, [{ name: 'All', values }], null);
    view.metric = saveMetric;
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
  $('cumulative').checked = view.cumulative;
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
  $('range').addEventListener('change', () => {
    view.range = $('range').value;
    $('custom').hidden = view.range !== 'custom';
    view.offset = 0;
    open.clear();
    persist();
    drawWindow();
    windowLabel(null);
    loads.now();
  });
  for (const id of ['from', 'to']) {
    $(id).addEventListener('change', () => { view[id] = $(id).value; view.offset = 0; open.clear(); persist(); drawWindow(); windowLabel(null); loads.now(); });
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
  $('cumulative').addEventListener('change', () => { view.cumulative = $('cumulative').checked; persist(); renderMain(); });
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
