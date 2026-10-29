// The statistics charts show every group's value at the bucket under the pointer in a tooltip,
// and dragging across the main chart narrows the period the whole page reads, which the all-time
// strip shows as its window.
const { test, expect } = require('@playwright/test');

const PASSWORD = 'password';

async function signIn(page, user) {
  await page.goto('/stats');
  await page.locator('#username').fill(user);
  await page.locator('#password').fill(PASSWORD);
  await page.locator('#kc-login').click();
  await expect(page).toHaveURL(/\/stats$/);
}

const preset = (page, value) => page.locator(`#range button[data-value="${value}"]`);
// The chosen period's first and last day, as the strip's caption shows them.
async function chosen(page) {
  const span = (await page.locator('#window-label').textContent()).split(' · ')[0];
  const [from, to = from] = span.split(' – ').map((d) => new Date(d + ' UTC').toISOString().slice(0, 10));
  return [from, to];
}

test('hovering the main chart shows every group at that bucket', async ({ page }) => {
  await signIn(page, 'blatten-admin');
  await preset(page, '30').click();
  await expect(preset(page, '30')).toHaveClass(/\bon\b/);
  const over = page.locator('#chart-main .u-over');
  await expect(over).toBeVisible();
  await expect(page.locator('#window-label')).toContainText('by day');

  const box = await over.boundingBox();
  const y = box.y + box.height / 2;
  await page.mouse.move(box.x + box.width * 0.6, y);
  const tip = page.locator('#chart-main .u-tooltip');
  await expect(tip).toBeVisible();
  await expect(tip.locator('.u-tooltip-title')).toHaveText(/\d{4}/);
  const names = await tip.locator('.u-tooltip-name').allTextContents();
  expect(names.length).toBeGreaterThan(0);
  for (const v of await tip.locator('.u-tooltip-value').allTextContents()) expect(v).toMatch(/^[\d,.]+$/);
  const key = page.locator('#chart-main .u-legend .u-series .u-label');
  expect(await key.allTextContents()).toEqual(names);
  await expect(page.locator('#chart-main .u-legend .u-value')).toHaveCount(0);
  const tipBox = await tip.boundingBox();
  expect(tipBox.x).toBeGreaterThanOrEqual(box.x);
  expect(tipBox.y).toBeGreaterThanOrEqual(box.y);
});

test('dragging across the main chart narrows the period, and a double-click shows all time, from a preset and after a drag', async ({ page }) => {
  await signIn(page, 'blatten-admin');
  await preset(page, 'all').click();
  await expect(page.locator('#window-label')).toContainText(/by (hour|day|week)$/);
  const all = await chosen(page);
  expect(all[0] < all[1]).toBe(true);

  await preset(page, '7').click();
  await expect(preset(page, '7')).toHaveClass(/\bon\b/);
  await expect.poll(() => chosen(page)).not.toEqual(all);
  await expect(page.locator('#window-label')).toContainText(/by (hour|day)$/);
  const over = page.locator('#chart-main .u-over');
  await expect(over).toBeVisible();
  const box = await over.boundingBox();
  const y = box.y + box.height / 2;
  await page.mouse.dblclick(box.x + box.width * 0.5, y);
  await expect(preset(page, 'all')).toHaveClass(/\bon\b/);
  await expect(page.locator('#range button.on')).toHaveCount(1);
  await expect.poll(() => chosen(page)).toEqual(all);

  await preset(page, '30').click();
  await expect(page.locator('#window-label')).toContainText('by day');
  await page.mouse.move(box.x + box.width * 0.3, y);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.5, y, { steps: 5 });
  await page.mouse.up();
  await expect(page.locator('#range button.on')).toHaveCount(0);
  await expect.poll(() => chosen(page)).not.toEqual(all);
  const [from, to] = await chosen(page);
  expect(from <= to).toBe(true);
  await page.mouse.dblclick(box.x + box.width * 0.5, y);
  await expect(preset(page, 'all')).toHaveClass(/\bon\b/);
  await expect.poll(() => chosen(page)).toEqual(all);
  await page.reload();
  await expect(preset(page, 'all')).toHaveClass(/\bon\b/);
});

test('the strip window and the main chart zoom are the same period', async ({ page }) => {
  await signIn(page, 'blatten-admin');
  await preset(page, 'all').click();
  const strip = page.locator('#chart-strip .u-over');
  const window = page.locator('#chart-strip .u-select');
  await expect(strip).toBeVisible();
  await expect(page.locator('#chart-main .u-over')).toBeVisible();
  const changedFrom = async (before) => expect.poll(() => chosen(page)).not.toEqual(before);
  const drag = async (x0, x1, y) => {
    await page.mouse.move(x0, y);
    await page.mouse.down();
    await page.mouse.move(x1, y, { steps: 5 });
    await page.mouse.up();
  };

  const sbox = await strip.boundingBox();
  const sy = sbox.y + sbox.height / 2;
  await drag(sbox.x + sbox.width * 0.2, sbox.x + sbox.width * 0.8, sy);
  await expect(page.locator('#range button.on')).toHaveCount(0);
  const [from, to] = await chosen(page);
  expect(from <= to).toBe(true);
  await expect(page.locator('#window-label')).toContainText(/by (hour|day|week)$/);
  const brushed = await window.boundingBox();
  expect(brushed.width).toBeGreaterThan(sbox.width * 0.4);

  let before = await chosen(page);
  const mid = brushed.x + brushed.width / 2;
  await drag(mid, mid - sbox.width * 0.15, sy);
  await changedFrom(before);
  const shifted = await window.boundingBox();
  expect(shifted.x).toBeLessThan(brushed.x);
  expect(Math.abs(shifted.width - brushed.width)).toBeLessThan(sbox.width * 0.05);

  before = await chosen(page);
  const right = shifted.x + shifted.width;
  await drag(right - 1, right - shifted.width * 0.5, sy);
  await changedFrom(before);
  expect((await chosen(page))[0]).toBe(before[0]);
  const narrowed = await window.boundingBox();
  expect(narrowed.x).toBeCloseTo(shifted.x, 0);
  expect(narrowed.width).toBeLessThan(shifted.width * 0.75);

  await expect(page.locator('#window-label')).toContainText(/by (hour|day|week)$/);
  const mbox = await page.locator('#chart-main .u-over').boundingBox();
  before = await chosen(page);
  await drag(mbox.x + mbox.width * 0.02, mbox.x + mbox.width * 0.1, mbox.y + mbox.height / 2);
  await changedFrom(before);
  await expect.poll(async () => (await window.boundingBox()).width).toBeLessThan(narrowed.width);
});

test('a handle drags an end of the period in from all time, with its date beneath', async ({ page }) => {
  await signIn(page, 'blatten-admin');
  await preset(page, 'all').click();
  const handle = page.locator('#chart-strip .strip-handle[data-end="from"]');
  const dates = page.locator('#strip-dates span:not([hidden])');
  await expect(handle).toBeVisible();
  await expect(dates.first()).toHaveText(/\d{4}/);
  const sbox = await page.locator('#chart-strip .u-over').boundingBox();
  const hbox = await handle.boundingBox();
  expect(hbox.x + hbox.width / 2).toBeCloseTo(sbox.x, 0);
  const firstDate = await dates.first().textContent();

  const y = hbox.y + hbox.height / 2;
  await page.mouse.move(hbox.x + hbox.width / 2 + 2, y);
  await page.mouse.down();
  await page.mouse.move(sbox.x + sbox.width * 0.4, y, { steps: 5 });
  await page.mouse.up();
  await expect(page.locator('#range button.on')).toHaveCount(0);
  const moved = await handle.boundingBox();
  expect(moved.x + moved.width / 2).toBeGreaterThan(sbox.x + sbox.width * 0.3);
  await expect(dates.first()).not.toHaveText(firstDate);
  const wbox = await page.locator('#chart-strip .u-select').boundingBox();
  expect(wbox.x + wbox.width).toBeCloseTo(sbox.x + sbox.width, 0);
});

test('the header holds no select or input, and grouping and both exports are buttons', async ({ page }) => {
  await signIn(page, 'blatten-admin');
  await expect(page.locator('#chart-main .u-over')).toBeVisible();
  await expect(page.locator('header select, header input')).toHaveCount(0);

  await preset(page, '7').click();
  const [from, to] = await chosen(page);
  expect(Date.parse(to) - Date.parse(from)).toBe(6 * 86400000);
  await preset(page, '1').click();
  await expect.poll(() => chosen(page)).toEqual([to, to]);

  await page.locator('#group button[data-value="provider"]').click();
  await expect(page.locator('#group button.on')).toHaveText('Provider');
  await expect(page.locator('#cap-main')).toContainText('by provider');
  await expect(page.locator('#table th').first()).toHaveText(/^Provider/);

  for (const [level, name] of [['item', 'download-statistics.csv'], ['file', 'download-statistics-by-file.csv']]) {
    const download = page.waitForEvent('download');
    await page.locator(`#export button[data-value="${level}"]`).click();
    expect((await download).suggestedFilename()).toBe(name);
  }
});

// The resolution a series request for the main chart asks for, or null when the server chooses.
const seriesRequest = (page) => page.waitForRequest((r) => {
  const url = new URL(r.url());
  return url.pathname === '/stats/api/series' && url.searchParams.get('dim') !== 'total';
});
const askedResolution = async (request) => new URL((await request).url()).searchParams.get('resolution');

test('one figure holds the main chart with the all-time strip beneath it, and the Last hour tile holds its numbers only', async ({ page }) => {
  await signIn(page, 'blatten-admin');
  await expect(page.locator('#chart-main .u-over')).toBeVisible();
  await expect(page.locator('#chart-strip .u-over')).toBeVisible();
  await expect(page.locator('main figure')).toHaveCount(1);
  const figure = page.locator('main figure');
  await expect(figure.locator('#chart-main canvas')).toHaveCount(1);
  await expect(figure.locator('#chart-strip canvas')).toHaveCount(1);
  await expect(figure.locator('#metric, #range, #resolution, #window-label')).toHaveCount(4);
  const main = await page.locator('#chart-main').boundingBox();
  const strip = await page.locator('#chart-strip').boundingBox();
  expect(strip.y).toBeGreaterThanOrEqual(main.y + main.height - 1);
  await expect(page.locator('#k-hour-sub')).toHaveText(/ in 24 h$/);
  const tile = page.locator('.kpi', { has: page.locator('#k-hour') });
  await expect(tile.locator('canvas, .uplot')).toHaveCount(0);
  const charts = page.locator('.uplot');
  await expect(charts).toHaveCount(2);
  expect(await charts.evaluateAll((els) => els.map((e) => e.parentElement.id))).toEqual(['chart-main', 'chart-strip']);
});

test('24 h draws the last day by hour, and a resolution chip changes the series asked for', async ({ page }) => {
  await signIn(page, 'blatten-admin');
  await expect(page.locator('#chart-main .u-over')).toBeVisible();

  let request = seriesRequest(page);
  await preset(page, '24h').click();
  expect(await askedResolution(request)).toBe('hour');
  await expect(preset(page, '24h')).toHaveClass(/\bon\b/);
  await expect(page.locator('#window-label')).toContainText('by hour');
  await expect(page.locator('#cap-main')).toContainText('per hour');
  await expect(page.locator('#resolution button[data-value="day"]')).toBeDisabled();

  request = seriesRequest(page);
  await preset(page, '7').click();
  expect(await askedResolution(request)).toBeNull();
  await expect(page.locator('#resolution button[data-value="auto"]')).toHaveText('Auto (hour)');

  request = seriesRequest(page);
  await page.locator('#resolution button[data-value="day"]').click();
  expect(await askedResolution(request)).toBe('day');
  await expect(page.locator('#resolution button.on')).toHaveText('Day');
  await expect(page.locator('#window-label')).toContainText('by day');

  request = seriesRequest(page);
  await page.locator('#resolution button[data-value="auto"]').click();
  expect(await askedResolution(request)).toBeNull();
});

test('the Documents row lists each document, and by item each document is a series of its own', async ({ page }) => {
  await signIn(page, 'blatten-admin');
  await preset(page, '30').click();
  await page.locator('#group button[data-value="item"]').click();
  await expect(page.locator('#cap-main')).toContainText('by item');
  const extra = (name) => page.locator('#table tr.extra').filter({ has: page.locator('td:first-child', { hasText: new RegExp('^' + name + '$') }) });
  const docs = extra('Documents');
  await expect(docs).toHaveCount(1);
  await docs.click();
  const files = page.locator('#table tr.files table tr').filter({ has: page.locator('td') });
  await expect(files.first()).toBeVisible();
  const names = await files.locator('td:first-child').allTextContents();
  expect(names.length).toBeGreaterThanOrEqual(2);
  const key = page.locator('#chart-main .u-legend .u-series .u-label');
  await expect.poll(() => key.allTextContents()).toEqual(expect.arrayContaining(names));
  expect(await key.allTextContents()).not.toContain('Documents');

  await page.locator('#group button[data-value="collection"]').click();
  await expect(page.locator('#cap-main')).toContainText('by collection');
  await expect.poll(() => key.allTextContents()).toContain('Documents');
  for (const n of names) expect(await key.allTextContents()).not.toContain(n);
  await expect(extra('Documents')).toHaveCount(1);
});
