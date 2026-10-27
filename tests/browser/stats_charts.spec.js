// The statistics charts show every group's value at the bucket under the pointer, and dragging
// across the main chart narrows the period the whole page reads, which the all-time strip shows
// as its window. An account without the viewer role gets no chart at all.
const { test, expect } = require('@playwright/test');

const PASSWORD = 'password';

async function signIn(page, user) {
  await page.goto('/stats');
  await page.locator('#username').fill(user);
  await page.locator('#password').fill(PASSWORD);
  await page.locator('#kc-login').click();
  await expect(page).toHaveURL(/\/stats$/);
}

test('hovering the main chart shows every group at that bucket, and dragging narrows the period', async ({ page }) => {
  await signIn(page, 'blatten-admin');
  await page.locator('#range').selectOption('30');
  const over = page.locator('#chart-main .u-over');
  await expect(over).toBeVisible();
  await expect(page.locator('#window-label')).toContainText('by day');

  const box = await over.boundingBox();
  const y = box.y + box.height / 2;
  await page.mouse.move(box.x + box.width * 0.6, y);
  const legend = page.locator('#chart-main .u-legend');
  const groups = legend.locator('.u-series').filter({ hasNot: page.locator('.u-label', { hasText: 'Period' }) });
  await expect(legend.locator('.u-series').first().locator('.u-value')).toHaveText(/\d{4}/);
  expect(await groups.count()).toBeGreaterThan(0);
  for (const v of await groups.locator('.u-value').allTextContents()) expect(v).toMatch(/^[\d,.]+$/);

  const before = await page.locator('#window-label').textContent();
  await page.mouse.move(box.x + box.width * 0.3, y);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.5, y, { steps: 5 });
  await page.mouse.up();
  await expect(page.locator('#range')).toHaveValue('custom');
  const [from, to] = [await page.locator('#from').inputValue(), await page.locator('#to').inputValue()];
  expect(from <= to).toBe(true);
  await expect(page.locator('#window-label')).not.toHaveText(before);

  await page.mouse.dblclick(box.x + box.width * 0.5, y);
  await expect(page.locator('#range')).toHaveValue('30');
  await expect(page.locator('#window-label')).toHaveText(before);
});

test('the strip window and the main chart zoom are the same period', async ({ page }) => {
  await signIn(page, 'blatten-admin');
  await page.locator('#range').selectOption('all');
  const strip = page.locator('#chart-strip .u-over');
  const window = page.locator('#chart-strip .u-select');
  await expect(strip).toBeVisible();
  await expect(page.locator('#chart-main .u-over')).toBeVisible();
  const chosen = async () => [await page.locator('#from').inputValue(), await page.locator('#to').inputValue()];
  const changedFrom = async (before) => expect.poll(chosen).not.toEqual(before);
  const drag = async (x0, x1, y) => {
    await page.mouse.move(x0, y);
    await page.mouse.down();
    await page.mouse.move(x1, y, { steps: 5 });
    await page.mouse.up();
  };

  const sbox = await strip.boundingBox();
  const sy = sbox.y + sbox.height / 2;
  await drag(sbox.x + sbox.width * 0.2, sbox.x + sbox.width * 0.8, sy);
  await expect(page.locator('#range')).toHaveValue('custom');
  const [from, to] = await chosen();
  expect(from <= to).toBe(true);
  await expect(page.locator('#window-label')).toContainText(/by (hour|day|week)$/);
  const brushed = await window.boundingBox();
  expect(brushed.width).toBeGreaterThan(sbox.width * 0.4);

  let before = await chosen();
  const mid = brushed.x + brushed.width / 2;
  await drag(mid, mid - sbox.width * 0.15, sy);
  await changedFrom(before);
  const shifted = await window.boundingBox();
  expect(shifted.x).toBeLessThan(brushed.x);
  expect(Math.abs(shifted.width - brushed.width)).toBeLessThan(sbox.width * 0.05);

  before = await chosen();
  const right = shifted.x + shifted.width;
  await drag(right - 1, right - shifted.width * 0.5, sy);
  await changedFrom(before);
  expect((await chosen())[0]).toBe(before[0]);
  const narrowed = await window.boundingBox();
  expect(narrowed.x).toBeCloseTo(shifted.x, 0);
  expect(narrowed.width).toBeLessThan(shifted.width * 0.75);

  await expect(page.locator('#window-label')).toContainText(/by (hour|day|week)$/);
  const mbox = await page.locator('#chart-main .u-over').boundingBox();
  before = await chosen();
  await drag(mbox.x + mbox.width * 0.02, mbox.x + mbox.width * 0.1, mbox.y + mbox.height / 2);
  await changedFrom(before);
  await expect.poll(async () => (await window.boundingBox()).width).toBeLessThan(narrowed.width);
});

test('an account without the viewer role gets no chart', async ({ page }) => {
  await signIn(page, 'blatten-guest');
  await expect(page.locator('#access')).toBeVisible();
  await expect(page.locator('#chart-main canvas, #chart-live canvas')).toHaveCount(0);
});
