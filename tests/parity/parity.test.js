// Run with: node --test tests/parity/
const { test } = require('node:test');
const assert = require('node:assert');
const http = require('node:http');
const fs = require('node:fs');
const { record, recordSite, websiteDir, compare, MAX_ITEMS, MAX_ASSETS } = require('./parity.js');

// The website checkout whose loaders record-site runs, when there is one.
const site = websiteDir();
const needsSite = fs.existsSync(site) ? {} : { skip: `no website checkout at ${site}; set WEBSITE_DIR` };

const asset = (href, extra = {}) => ({ href, roles: ['data'], 'file:size': 10, ...extra });
const archive = (id) => asset(`/s3/archives/${id}.zip`, { roles: ['data', 'archive'], 'file:checksum': `1220${id}` });

const item = (id, collection, assets, properties = {}, links = []) => ({
  type: 'Feature',
  id,
  collection,
  properties: { title: `Item ${id}`, ...properties },
  assets,
  links,
});

// 600 files, more than one asset page.
const plumAssets = Object.fromEntries(
  Array.from({ length: 600 }, (_, n) => [`plum-${n}`, asset(`/s3/assets/plum-01/${n}.jpg`)]),
);

const dated = {
  'pear-0401-a': asset('/s3/assets/pear-01/2024-04-01_a.jpg'),
  'pear-0401-b': asset('/s3/assets/pear-01/2024-04-01_b.jpg'),
  'pear-0402-a': asset('/s3/assets/pear-01/2024-04-02_a.jpg'),
};
const undated = { factsheet: asset('/s3/assets/pear-01/factsheet.pdf', { roles: ['metadata'] }) };

const unsplit = {
  collections: [{ id: 'orchards', title: 'Orchards' }],
  items: [
    item('pear-01', 'orchards', { archive: archive('pear-01'), ...dated, ...undated }, { 'blatten:file_count': 4 }),
    item('plum-01', 'orchards', { archive: archive('plum-01'), ...plumAssets }, { 'blatten:file_count': 600 }),
  ],
};

const days = (only) => [
  item('pear-01_2024-04-01', 'orchards', only('0401'), { 'blatten:series': 'pear-01', 'blatten:file_count': 2 }),
  item('pear-01_2024-04-02', 'orchards', only('0402'), { 'blatten:series': 'pear-01', 'blatten:file_count': 1 }),
];

function split({ drop } = {}) {
  const only = (day) => Object.fromEntries(Object.entries(dated).filter(([k]) => k.includes(day) && k !== drop));
  const daysLink = {
    rel: 'items',
    title: 'Days',
    type: 'application/geo+json',
    href: "https://public.example/stac/collections/orchards/items?filter=blatten:series='pear-01'",
  };
  return {
    collections: unsplit.collections,
    items: [
      item(
        'pear-01',
        'orchards',
        { archive: archive('pear-01'), manifest: asset('/s3/archives/pear-01.manifest.csv'), ...undated },
        { 'blatten:file_count': 4 },
        [daysLink],
      ),
      ...days(only),
      unsplit.items[1],
    ],
  };
}

// A stub of the STAC API routes the website reads, paging two items at a time and refusing
// a request over the API's caps.
async function serve(catalog) {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://stub');
    const send = (status, body) => {
      res.writeHead(status, { 'content-type': 'application/json' });
      res.end(JSON.stringify(body));
    };
    const limit = Number(url.searchParams.get('limit') || 10);
    const assetLimit = Number(url.searchParams.get('asset_limit') || 0);
    if (limit > MAX_ITEMS || assetLimit > MAX_ASSETS) return send(400, { limit, assetLimit });

    if (url.pathname === '/stac/collections') return send(200, { collections: catalog.collections });

    const items = url.pathname.match(/^\/stac\/collections\/([^/]+)\/items$/);
    if (items) {
      const filter = url.searchParams.get('filter') || '';
      const series = filter.match(/^blatten:series='(.+)'$/)?.[1];
      let matched = catalog.items.filter((i) => i.collection === items[1]);
      if (series) matched = matched.filter((i) => i.properties['blatten:series'] === series);
      if (filter === 'blatten:series IS NULL') matched = matched.filter((i) => !i.properties['blatten:series']);
      const offset = Number(url.searchParams.get('offset') || 0);
      const size = Math.min(limit, 2);
      let page = matched.slice(offset, offset + size);
      if (url.searchParams.get('exclude_assets') === 'true') {
        page = page.map((i) => ({ ...i, assets: i.assets.archive ? { archive: i.assets.archive } : {} }));
      }
      const links = [];
      if (offset + size < matched.length) {
        const next = new URL(url);
        next.searchParams.set('offset', String(offset + size));
        links.push({ rel: 'next', href: `https://public.example${next.pathname}${next.search}` });
      }
      return send(200, { type: 'FeatureCollection', features: page, links, numberMatched: matched.length });
    }

    const one = url.pathname.match(/^\/stac\/collections\/([^/]+)\/items\/([^/]+)$/);
    const found = one && catalog.items.find((i) => i.collection === one[1] && i.id === one[2]);
    if (found) {
      if (!assetLimit) return send(200, found);
      const offset = Number(url.searchParams.get('asset_offset') || 0);
      const others = Object.entries(found.assets).filter(([k]) => k !== 'archive');
      const assets = Object.fromEntries(others.slice(offset, offset + assetLimit));
      if (found.assets.archive) assets.archive = found.assets.archive;
      return send(200, { ...found, assets, _assetsMeta: { total: others.length, offset } });
    }
    return send(404, {});
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  return { base: `http://127.0.0.1:${server.address().port}/stac`, close: () => server.close() };
}

async function recordOf(catalog, recorder = record) {
  const stub = await serve(catalog);
  try {
    return await recorder(stub.base);
  } finally {
    stub.close();
  }
}

const siteRecordOf = (catalog) => recordOf(catalog, (base) => recordSite(base, site));

test('a series split into days with an undated file on the series records as the series unsplit', async () => {
  const before = await recordOf(unsplit);
  const after = await recordOf(split());
  assert.deepStrictEqual(compare(before, after), []);
  assert.deepStrictEqual(after, before);

  const [orchards] = before.collections;
  assert.deepStrictEqual(orchards.items.map((i) => i.id), ['pear-01', 'plum-01']);
  const pear = orchards.items[0];
  assert.strictEqual(pear.files.count, 4);
  assert.strictEqual(pear.file_count, 4);
  assert.deepStrictEqual(pear.archive, { href: '/s3/archives/pear-01.zip', size: 10, checksum: '1220pear-01' });
  assert.strictEqual(orchards.items[1].files.count, 600, 'every asset page of an item is read');
});

test('a file dropped from a day is reported by item and count', async () => {
  const before = await recordOf(unsplit);
  const after = await recordOf(split({ drop: 'pear-0401-b' }));
  assert.deepStrictEqual(compare(before, after), ['item orchards/pear-01: 4 files -> 3']);
});

test('a changed archive, title and missing item are each reported', () => {
  const rec = (items) => ({ collections: [{ id: 'orchards', title: 'Orchards', items }] });
  const pear = {
    id: 'pear-01',
    title: 'Pears',
    file_count: 2,
    archive: { href: '/s3/archives/pear-01.zip', size: 10, checksum: 'a' },
    files: { count: 2, sha256: 'x' },
  };
  const before = rec([pear, { ...pear, id: 'plum-01' }]);
  const after = rec([{ ...pear, title: 'Pear', archive: { ...pear.archive, size: 11 }, files: { count: 2, sha256: 'y' } }]);
  assert.deepStrictEqual(compare(before, after), [
    'collection orchards: 2 items -> 1',
    'item orchards/plum-01: missing after',
    'item orchards/pear-01: title "Pears" -> "Pear"',
    'item orchards/pear-01: archive {"href":"/s3/archives/pear-01.zip","size":10,"checksum":"a"} -> {"href":"/s3/archives/pear-01.zip","size":11,"checksum":"a"}',
    'item orchards/pear-01: 2 files, different hrefs',
  ]);
});

test('the website loaders reach the same files from a split series as the script from the unsplit one', needsSite, async () => {
  const before = await recordOf(unsplit);
  const after = await siteRecordOf(split());
  assert.deepStrictEqual(compare(before, after), []);
  assert.deepStrictEqual(after, before);
  assert.deepStrictEqual(await siteRecordOf(unsplit), before);
});

test('a file dropped from a day is reported through the website loaders', needsSite, async () => {
  const before = await recordOf(unsplit);
  const after = await siteRecordOf(split({ drop: 'pear-0401-b' }));
  assert.deepStrictEqual(compare(before, after), ['item orchards/pear-01: 4 files -> 3']);
});
