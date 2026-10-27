const { startStatsProxy } = require('./stats_proxy');
const { PROXY_PORT } = require('../../playwright.config');

module.exports = async (config) => {
  const api = process.env.STATS_API;
  if (!api) return undefined;
  const server = await startStatsProxy(new URL(config.projects[0].use.baseURL).origin, api, PROXY_PORT);
  return () => new Promise((resolve) => server.close(resolve));
};
