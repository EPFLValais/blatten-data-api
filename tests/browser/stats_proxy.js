// A forward proxy for the browser that answers the stack's /stats from STATS_API, such as a build
// of a working tree on http://localhost:3000, and passes every other request through unchanged.
// The browser stays on the stack's origin, the only one the realm's client accepts as a redirect.
const http = require('node:http');

function startStatsProxy(origin, api, port) {
  const target = new URL(api);
  const server = http.createServer((req, res) => {
    const url = new URL(req.url);
    const toApi = url.origin === origin && (url.pathname === '/stats' || url.pathname.startsWith('/stats/'));
    const to = toApi ? target : url;
    const headers = { ...req.headers, host: toApi ? target.host : url.host };
    delete headers['proxy-connection'];
    const upstream = http.request(
      { hostname: to.hostname, port: to.port, method: req.method, path: url.pathname + url.search, headers },
      (answer) => { res.writeHead(answer.statusCode, answer.headers); answer.pipe(res); },
    );
    upstream.on('error', (e) => { res.writeHead(502); res.end(String(e)); });
    req.pipe(upstream);
  });
  return new Promise((resolve) => server.listen(port, '127.0.0.1', () => resolve(server)));
}

module.exports = { startStatsProxy };
