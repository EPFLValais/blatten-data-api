// The statistics page signs in through the local realm, shows its numbers, and signs out of the
// realm itself: the next visit asks for the password again instead of signing straight back in.
// An account without the viewer role sees only the access screen, with its own way to sign out.
const { test, expect } = require('@playwright/test');

const VIEWER = 'blatten-admin';
const NOT_VIEWER = 'blatten-guest';
const PASSWORD = 'password';

async function signIn(page, user) {
  await page.locator('#username').fill(user);
  await page.locator('#password').fill(PASSWORD);
  await page.locator('#kc-login').click();
}

async function expectNoneVisible(locator) {
  const count = await locator.count();
  for (let i = 0; i < count; i++) await expect(locator.nth(i)).toBeHidden();
}

test('signing out ends the realm session, so the next visit asks for the password', async ({ page }) => {
  await page.goto('/stats');
  await expect(page.locator('#username')).toBeVisible();
  await signIn(page, VIEWER);

  await expect(page).toHaveURL(/\/stats$/);
  await expect(page.locator('#status')).toBeHidden();
  await expect(page.locator('#k-requests')).not.toHaveText('–');
  await expect(page.locator('main')).toBeVisible();
  await expect(page.locator('#access')).toBeHidden();

  const logout = page.waitForRequest((req) => new URL(req.url()).pathname.endsWith('/protocol/openid-connect/logout'));
  const back = page.waitForRequest((req) => {
    const url = new URL(req.url());
    return req.isNavigationRequest() && url.pathname === '/stats' && !url.searchParams.has('code');
  });
  await page.locator('#signout').click();
  expect(new URL((await logout).url()).searchParams.get('id_token_hint')).toBeTruthy();
  await back;
  await expect(page.locator('#username')).toBeVisible();

  await page.goto('/stats');
  await expect(page.locator('#username')).toBeVisible();
  await expect(page.locator('#k-requests')).toHaveCount(0);
});

test('an account without the viewer role sees only the access screen, and can sign out from it', async ({ page }) => {
  await page.goto('/stats');
  await expect(page.locator('#username')).toBeVisible();
  await signIn(page, NOT_VIEWER);

  await expect(page).toHaveURL(/\/stats$/);
  await expect(page.locator('#access-text')).toHaveText('This account may not view the statistics.');
  await expect(page.locator('#access')).toBeVisible();
  await expect(page.locator('#access-signout')).toBeVisible();
  await expectNoneVisible(page.locator('.controls, .controls *, .toolbar, .toolbar *'));
  await expect(page.locator('main')).toBeHidden();

  const logout = page.waitForRequest((req) => new URL(req.url()).pathname.endsWith('/protocol/openid-connect/logout'));
  await page.locator('#access-signout').click();
  await logout;
  await expect(page.locator('#username')).toBeVisible();
});
