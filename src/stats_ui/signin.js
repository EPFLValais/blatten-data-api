// A refused token signs the page in again once. The sign-in returns a fresh token at once while
// the identity provider's session lasts, so a token the API keeps refusing would otherwise send
// the browser round the sign-in without end. The mark lives in session storage because the
// sign-in reloads the page, and is cleared by the first request the API accepts.
const GUARD_KEY = 'signin_after_refusal';

function signInGuard(storage, login) {
  let signingIn = false;
  return {
    async refused() {
      if (signingIn) return 'signing-in';
      if (storage.getItem(GUARD_KEY)) return 'refused';
      signingIn = true;
      storage.setItem(GUARD_KEY, '1');
      await login();
      return 'signing-in';
    },
    accepted() { storage.removeItem(GUARD_KEY); },
  };
}

// The session-storage keys the page writes. Signing out removes these and no others: the tab's
// session storage is shared with every other page of the origin.
const PAGE_KEYS = ['stats_view', 'pkce_verifier', 'pkce_state', GUARD_KEY];

function forgetSession(storage) {
  for (const key of PAGE_KEYS) storage.removeItem(key);
}

// OpenID Connect RP-initiated logout. It ends the session in the realm, not only on this page:
// with the realm session left open, the next visit would sign straight back in.
function signOutUrl(cfg, idToken, redirect) {
  const params = { client_id: cfg.clientId, post_logout_redirect_uri: redirect };
  if (idToken) params.id_token_hint = idToken;
  return cfg.logoutUrl + '?' + new URLSearchParams(params);
}

// The dashboard is shown only once the API has accepted the viewer: the access screen stands in
// for it until then, and again after any refusal. The message for a response's status: null shows
// the dashboard, a text shows the access screen with it, undefined leaves the page as it is.
function accessAfter(status, guarded) {
  if (status >= 200 && status < 300) return null;
  if (status === 403) return 'This account may not view the statistics.';
  if (status === 401 && guarded === 'refused') return 'Access refused. Try again later.';
  return undefined;
}

function showAccess(doc, message) {
  const refused = message !== null;
  doc.getElementById('access').hidden = !refused;
  if (refused) doc.getElementById('access-text').textContent = message;
  for (const part of doc.querySelectorAll('[data-dashboard]')) part.hidden = refused;
}
