#!/usr/bin/env node
// Records what every website view reaches from a STAC API, and compares two such records.
//
//   node tests/parity/parity.js record http://localhost:3000/stac > record.json
//   node tests/parity/parity.js record-site http://localhost:3000/stac [WEBSITE_DIR] > record.json
//   node tests/parity/parity.js compare before.json after.json
//
// The record holds, per collection, its id and title, and per series-level item (an item
// without `blatten:series`) its title, `blatten:file_count`, archive, and the count and
// SHA-256 of the sorted hrefs of every file reachable from it: its own assets, plus the
// assets of every item its days link (`rel: "items"`) pages to. The archive and the
// manifest are not files. An href is recorded as its path and query, since the API writes
// its own origin into them; both records are taken with the same S3_BASE_URL path. Compare
// prints each difference and exits 1 when there is one.
//
// record-site takes the same record through the website's own loaders: listing.js lists each
// collection's series level, days.js pages a series' days from its `rel: "items"` link, and
// assets.js pages the files of a series and of each of its days, as the website does when each
// is opened. The website checkout is WEBSITE_DIR, or ../blatten-data-ui beside this repository.

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const MAX_ITEMS = 1000;
const MAX_ASSETS = 500;
// Day items come with all their assets, so their pages are kept smaller.
const DAYS_PER_PAGE = 100;
const NOT_FILES = new Set(['archive', 'manifest']);

// A URL on the API's origin with the path and query of `href`, which may be relative or
// carry another origin (the API writes its public base URL into its links).
function onApi(base, href) {
  const url = new URL(href, base + '/');
  const api = new URL(base);
  url.protocol = api.protocol;
  url.host = api.host;
  return url;
}

function capped(url, param, max) {
  const asked = Number(url.searchParams.get(param));
  if (!asked || asked > max) url.searchParams.set(param, String(max));
  return url;
}

async function getJson(fetchFn, url) {
  const res = await fetchFn(url.toString());
  if (!res.ok) throw new Error(`${res.status} from ${url}`);
  return res.json();
}

// Every feature of an items listing, following `next`, at most `max` a request.
async function allFeatures(fetchFn, base, url, max = MAX_ITEMS) {
  const features = [];
  let page = capped(url, 'limit', max);
  for (;;) {
    const data = await getJson(fetchFn, page);
    features.push(...(data.features || []));
    const next = (data.links || []).find((l) => l.rel === 'next');
    if (!next || !(data.features || []).length) return features;
    page = capped(onApi(base, next.href), 'limit', max);
  }
}

function hrefPath(href) {
  const url = new URL(href, 'http://origin');
  return url.pathname + url.search;
}

const fileHrefs = (assets) =>
  Object.entries(assets || {}).filter(([key]) => !NOT_FILES.has(key)).map(([, a]) => hrefPath(a.href));

// An item with all its assets, asked MAX_ASSETS at a time as the item route pages them.
async function fullItem(fetchFn, base, collectionId, itemId) {
  const path = `collections/${encodeURIComponent(collectionId)}/items/${encodeURIComponent(itemId)}`;
  let item;
  const hrefs = [];
  for (let offset = 0; ; offset += MAX_ASSETS) {
    const url = onApi(base, path);
    url.searchParams.set('asset_limit', String(MAX_ASSETS));
    url.searchParams.set('asset_offset', String(offset));
    const page = await getJson(fetchFn, url);
    item ??= page;
    hrefs.push(...fileHrefs(page.assets));
    const meta = page._assetsMeta;
    if (!meta || offset + MAX_ASSETS >= meta.total) break;
  }
  return { item, hrefs };
}

function digest(hrefs) {
  const sorted = [...hrefs].sort();
  return {
    count: sorted.length,
    sha256: crypto.createHash('sha256').update(sorted.join('\n')).digest('hex'),
  };
}

// The record's entry for a series-level item as listed, with the hrefs of its files.
function itemRecord(listed, hrefs) {
  const archive = (listed.assets || {}).archive;
  return {
    id: listed.id,
    title: listed.properties?.title ?? null,
    file_count: listed.properties?.['blatten:file_count'] ?? null,
    archive: archive
      ? { href: hrefPath(archive.href), size: archive['file:size'] ?? null, checksum: archive['file:checksum'] ?? null }
      : null,
    files: digest(hrefs),
  };
}

async function recordItem(fetchFn, base, collectionId, listed) {
  const { item, hrefs } = await fullItem(fetchFn, base, collectionId, listed.id);
  const days = (item.links || []).find((l) => l.rel === 'items');
  if (days) {
    for (const day of await allFeatures(fetchFn, base, onApi(base, days.href), DAYS_PER_PAGE)) {
      if (day.id !== item.id) hrefs.push(...fileHrefs(day.assets));
    }
  }
  return itemRecord(listed, hrefs);
}

// The collections of the API, each with the record of every series-level item that
// listSeries(collectionId) resolves to, through recordOne(collectionId, listed).
async function recordCollections(fetchFn, base, listSeries, recordOne) {
  base = base.replace(/\/+$/, '');
  const collections = [];
  const listed = await getJson(fetchFn, onApi(base, 'collections'));
  for (const c of listed.collections || []) {
    const items = [];
    for (const item of await listSeries(c.id)) items.push(await recordOne(c.id, item));
    items.sort((a, b) => a.id.localeCompare(b.id));
    collections.push({ id: c.id, title: c.title ?? null, items });
  }
  collections.sort((a, b) => a.id.localeCompare(b.id));
  return { collections };
}

async function record(base, fetchFn = fetch) {
  base = base.replace(/\/+$/, '');
  return recordCollections(
    fetchFn,
    base,
    async (collectionId) => {
      const url = onApi(base, `collections/${encodeURIComponent(collectionId)}/items`);
      url.searchParams.set('exclude_assets', 'true');
      return (await allFeatures(fetchFn, base, url)).filter((i) => !i.properties?.['blatten:series']);
    },
    (collectionId, listed) => recordItem(fetchFn, base, collectionId, listed),
  );
}

function websiteDir() {
  return process.env.WEBSITE_DIR || path.join(__dirname, '../../../blatten-data-ui');
}

// The website's loaders, run as the page runs them: each script in one shared global scope.
function loadWebsite(dir) {
  const ctx = { URL, URLSearchParams, formatSize: String, formatDate: String };
  for (const name of ['listing.js', 'days.js', 'assets.js']) {
    vm.runInNewContext(fs.readFileSync(path.join(dir, 'src', name), 'utf8'), ctx, { filename: name });
  }
  return ctx;
}

// Every page of a website pager, read as Show more reads them.
async function drain(pager) {
  while (!pager.done) await pager.next();
  return pager;
}

async function recordSite(base, dir = websiteDir(), fetchFn = fetch) {
  base = base.replace(/\/+$/, '');
  const site = loadWebsite(dir);
  const fetchJson = (url) => getJson(fetchFn, onApi(base, url));
  const itemsPath = (collectionId) => `${base}/collections/${encodeURIComponent(collectionId)}/items`;
  // The files of an item as its opened panel pages them, given its listed file count.
  const files = async (collectionId, itemId, fileCount) => {
    const itemPath = `${itemsPath(collectionId)}/${encodeURIComponent(itemId)}`;
    const pager = site.createAssetPager(
      (limit, offset) => fetchJson(`${itemPath}?asset_limit=${limit}&asset_offset=${offset}`),
      { total: fileCount > 0 ? fileCount : null },
    );
    return fileHrefs((await drain(pager)).assets);
  };
  return recordCollections(
    fetchFn,
    base,
    (collectionId) => {
      const params = site.seriesLevel(new URLSearchParams({ exclude_assets: 'true' }));
      return site.fetchAllPages(fetchJson, itemsPath(collectionId), params);
    },
    async (collectionId, listed) => {
      const count = (item) => Number(item.properties?.['blatten:file_count']);
      const hrefs = await files(collectionId, listed.id, count(listed));
      const link = site.daysLink(listed);
      if (link) {
        const path = itemsPath(collectionId);
        const days = await drain(site.createDayPager(fetchJson, site.daysUrl(link, path, new URLSearchParams()), path));
        for (const day of days.days) hrefs.push(...(await files(collectionId, day.id, count(day))));
      }
      return itemRecord(listed, hrefs);
    },
  );
}

const byId = (list) => new Map(list.map((x) => [x.id, x]));
const show = (v) => JSON.stringify(v);

// Each difference between two records, as one line.
function compare(before, after) {
  const diffs = [];
  const a = byId(before.collections);
  const b = byId(after.collections);
  for (const id of a.keys()) if (!b.has(id)) diffs.push(`collection ${id}: missing after`);
  for (const id of b.keys()) if (!a.has(id)) diffs.push(`collection ${id}: added after`);
  for (const [id, ca] of a) {
    const cb = b.get(id);
    if (!cb) continue;
    if (ca.title !== cb.title) diffs.push(`collection ${id}: title ${show(ca.title)} -> ${show(cb.title)}`);
    const ia = byId(ca.items);
    const ib = byId(cb.items);
    if (ia.size !== ib.size) diffs.push(`collection ${id}: ${ia.size} items -> ${ib.size}`);
    for (const iid of ia.keys()) if (!ib.has(iid)) diffs.push(`item ${id}/${iid}: missing after`);
    for (const iid of ib.keys()) if (!ia.has(iid)) diffs.push(`item ${id}/${iid}: added after`);
    for (const [iid, x] of ia) {
      const y = ib.get(iid);
      if (!y) continue;
      const at = `item ${id}/${iid}`;
      if (x.title !== y.title) diffs.push(`${at}: title ${show(x.title)} -> ${show(y.title)}`);
      if (x.file_count !== y.file_count) diffs.push(`${at}: blatten:file_count ${x.file_count} -> ${y.file_count}`);
      if (show(x.archive) !== show(y.archive)) diffs.push(`${at}: archive ${show(x.archive)} -> ${show(y.archive)}`);
      if (x.files.count !== y.files.count) {
        diffs.push(`${at}: ${x.files.count} files -> ${y.files.count}`);
      } else if (x.files.sha256 !== y.files.sha256) {
        diffs.push(`${at}: ${x.files.count} files, different hrefs`);
      }
    }
  }
  return diffs;
}

async function main([mode, ...args]) {
  if (mode === 'record' && args.length === 1) {
    process.stdout.write(JSON.stringify(await record(args[0]), null, 2) + '\n');
  } else if (mode === 'record-site' && (args.length === 1 || args.length === 2)) {
    process.stdout.write(JSON.stringify(await recordSite(args[0], args[1] || websiteDir()), null, 2) + '\n');
  } else if (mode === 'compare' && args.length === 2) {
    const [before, after] = args.map((f) => JSON.parse(fs.readFileSync(f, 'utf8')));
    const diffs = compare(before, after);
    for (const d of diffs) console.log(d);
    if (diffs.length) process.exitCode = 1;
  } else {
    console.error('usage: parity.js record STAC_API_URL | parity.js record-site STAC_API_URL [WEBSITE_DIR] | parity.js compare BEFORE.json AFTER.json');
    process.exitCode = 2;
  }
}

if (require.main === module) {
  main(process.argv.slice(2)).catch((err) => {
    console.error(err.message);
    process.exitCode = 2;
  });
}

module.exports = { record, recordSite, websiteDir, compare, MAX_ITEMS, MAX_ASSETS };
