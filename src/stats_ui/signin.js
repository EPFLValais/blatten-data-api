// A refused token signs the page in again once. The sign-in returns a fresh token at once while
// the identity provider's session lasts, so a token the API keeps refusing would otherwise send
// the browser round the sign-in without end. The mark lives in session storage because the
// sign-in reloads the page, and is cleared by the first request the API accepts.
function signInGuard(storage, login) {
  const KEY = 'signin_after_refusal';
  let signingIn = false;
  return {
    async refused() {
      if (signingIn) return 'signing-in';
      if (storage.getItem(KEY)) return 'refused';
      signingIn = true;
      storage.setItem(KEY, '1');
      await login();
      return 'signing-in';
    },
    accepted() { storage.removeItem(KEY); },
  };
}
