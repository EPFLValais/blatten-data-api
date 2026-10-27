//! HTTP side of the download statistics.
//!
//! The API reads the file proxy's metrics text itself, from `STATS_SCRAPE_URL` every
//! `STATS_SCRAPE_INTERVAL` seconds. The page at `/stats` is a public shell without data; its data
//! routes under `/stats/api/` need a Keycloak token for the configured client and role. The poller
//! and the routes each exist only when their settings are present and non-empty.

use crate::stats::{
    breakdown, code_key, code_passes, facet_values, file_name_in_item, group_totals, parse_scrape,
    series, series_window, setting, summary_rows, to_csv, Dim, Filters, ItemInfo,
    Resolution, Totals,
};
use crate::stats_store::{database_options, Store};
use axum::{
    extract::{Path, Query, Request, State},
    http::{header, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
    Extension, Json, Router,
};
use axum_keycloak_auth::{
    instance::{KeycloakAuthInstance, KeycloakConfig},
    layer::KeycloakAuthLayer,
    KeycloakAuthStatus, PassthroughMode,
};
use axum_keycloak_auth::decode::{KeycloakToken, ProfileAndEmail};
use axum_keycloak_auth::role::KeycloakRole;
use chrono::{NaiveDate, Utc};
use sqlx::postgres::PgConnectOptions;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tracing::{error, info, warn};
use url::Url;

/// The largest metrics text accepted from the file proxy.
const INGEST_LIMIT_BYTES: usize = 32 * 1024 * 1024;

/// How long one read of the file proxy's metrics may take, body included.
const SCRAPE_TIMEOUT: Duration = Duration::from_secs(10);

/// Seconds between two reads of the file proxy's metrics when `STATS_SCRAPE_INTERVAL` is unset.
const DEFAULT_SCRAPE_INTERVAL_SECS: u64 = 60;

/// The most file paths one scrape may carry. The catalog has about 100,000 files, so a larger
/// scrape is not the proxy's own and is refused whole rather than partly applied.
const MAX_PATHS_PER_SCRAPE: usize = 200_000;

#[derive(Clone)]
pub struct KeycloakSettings {
    /// Keycloak as the API reaches it: http(s) with a host, as checked by `keycloak_server_url`.
    pub url: Url,
    /// Keycloak as the browser reaches it (differs from `url` inside a compose network).
    pub browser_url: String,
    pub realm: String,
    pub client_id: String,
    pub role: String,
}

/// Where and how often the API reads the file proxy's metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct ScrapeSettings {
    pub url: Url,
    pub interval: Duration,
}

#[derive(Clone)]
pub struct StatsState {
    pub store: Arc<Store>,
    pub items: Arc<Vec<ItemInfo>>,
    /// Size in bytes of each file, by the path the file proxy reports.
    pub sizes: Arc<HashMap<String, u64>>,
    pub keycloak: Option<Arc<KeycloakSettings>>,
}

/// The statistics settings as read, before anything is built from them.
pub enum StatsSettings {
    /// No database URL is set: statistics are not configured.
    Absent,
    /// A setting is unusable, so statistics stay off; the STAC API runs without them.
    Off(String),
    On {
        database: Box<PgConnectOptions>,
        /// `Err` names why the poller is off although `STATS_SCRAPE_URL` is set.
        scrape: Result<Option<ScrapeSettings>, String>,
        /// `Err` names why the page and data routes are left out although Keycloak is configured.
        keycloak: Result<Option<Arc<KeycloakSettings>>, String>,
    },
}

/// Check the statistics settings, read through `var`. A bad value turns statistics or their page
/// off with a reason; it never fails the process.
pub fn check_settings(var: impl Fn(&str) -> Option<String>) -> StatsSettings {
    let non_empty = |name: &str| setting(var(name));
    let Some(database_url) = non_empty("STATS_DATABASE_URL") else {
        return StatsSettings::Absent;
    };
    let database = match database_options(&database_url) {
        Ok(options) => Box::new(options),
        Err(e) => return StatsSettings::Off(e.to_string()),
    };
    let keycloak = match (non_empty("KEYCLOAK_URL"), non_empty("KEYCLOAK_REALM"), non_empty("KEYCLOAK_CLIENT_ID")) {
        (Some(raw_url), Some(realm), Some(client_id)) => keycloak_server_url(&raw_url).map(|url| {
            let role = non_empty("STATS_VIEWER_ROLE").unwrap_or_else(|| "stats-viewer".into());
            let browser_url = non_empty("KEYCLOAK_BROWSER_URL").unwrap_or_else(|| raw_url.clone());
            Some(Arc::new(KeycloakSettings { url, browser_url, realm, client_id, role }))
        }),
        _ => Ok(None),
    };
    let scrape = match non_empty("STATS_SCRAPE_URL") {
        Some(raw_url) => scrape_settings(&raw_url, non_empty("STATS_SCRAPE_INTERVAL").as_deref()).map(Some),
        None => Ok(None),
    };
    StatsSettings::On { database, scrape, keycloak }
}

/// The poller's settings: an http(s) URL with a host, and a whole number of seconds above zero.
fn scrape_settings(raw_url: &str, raw_interval: Option<&str>) -> Result<ScrapeSettings, String> {
    let url = Url::parse(raw_url).map_err(|e| format!("STATS_SCRAPE_URL {raw_url:?} is not a URL: {e}"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none_or(str::is_empty) {
        return Err(format!("STATS_SCRAPE_URL {raw_url:?} must be an http(s) URL with a host"));
    }
    let secs = match raw_interval {
        None => DEFAULT_SCRAPE_INTERVAL_SECS,
        Some(raw) => match raw.parse::<u64>() {
            Ok(secs) if secs > 0 => secs,
            _ => return Err(format!("STATS_SCRAPE_INTERVAL {raw:?} must be a whole number of seconds above 0")),
        },
    };
    Ok(ScrapeSettings { url, interval: Duration::from_secs(secs) })
}

/// Keycloak's address as the auth layer needs it: http or https with a host, so the realm path can
/// be appended. The auth crate panics on a URL that cannot take a path, such as `keycloak:8080`.
pub fn keycloak_server_url(raw: &str) -> Result<Url, String> {
    let url = Url::parse(raw).map_err(|e| format!("KEYCLOAK_URL {raw:?} is not a URL: {e}"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none_or(str::is_empty) || url.cannot_be_a_base() {
        return Err(format!("KEYCLOAK_URL {raw:?} must be an http(s) URL with a host, such as http://keycloak:8080"));
    }
    Ok(url)
}

/// Prepare the store from the settings read through `var`. Returns `None` when statistics are not
/// configured or a setting is unusable, which is logged; the STAC API starts either way.
fn settings_from(
    var: impl Fn(&str) -> Option<String>,
    items: Vec<ItemInfo>,
    sizes: HashMap<String, u64>,
) -> Option<StatsState> {
    let (database, scrape, keycloak) = match check_settings(var) {
        StatsSettings::Absent => return None,
        StatsSettings::Off(reason) => {
            error!("Statistics are off: {reason}");
            return None;
        }
        StatsSettings::On { database, scrape, keycloak } => (database, scrape, keycloak),
    };
    // The pool connects on first use, so the STAC API starts whether or not the database is up.
    let store = Store::lazy_with(*database);
    {
        let store = store.clone();
        tokio::spawn(async move { store.migrate_until_ready().await });
    }

    let keycloak = match keycloak {
        Ok(Some(kc)) => Some(kc),
        Ok(None) => {
            warn!("Keycloak settings are incomplete: the statistics page and data routes are not registered");
            None
        }
        Err(reason) => {
            error!("{reason}: the statistics page and data routes are not registered");
            None
        }
    };

    let state = StatsState { store: Arc::new(store), items: Arc::new(items), sizes: Arc::new(sizes), keycloak };
    match scrape {
        Ok(Some(settings)) => match Scraper::new(settings.url) {
            Ok(scraper) => {
                info!("Reading the file proxy's metrics from {} every {}s", scraper.url, settings.interval.as_secs());
                tokio::spawn(poll(scraper, settings.interval, state.clone()));
            }
            Err(e) => error!("Statistics are not read: the HTTP client could not be built: {e:#}"),
        },
        Ok(None) => warn!("STATS_SCRAPE_URL is not set: the file proxy's metrics are not read"),
        Err(reason) => error!("{reason}: the file proxy's metrics are not read"),
    }
    Some(state)
}

/// Add the statistics routes to `app` when the settings read through `var` allow them.
pub fn mount(
    app: Router,
    var: impl Fn(&str) -> Option<String>,
    items: Vec<ItemInfo>,
    sizes: HashMap<String, u64>,
) -> Router {
    match settings_from(var, items, sizes) {
        Some(stats) => app.merge(router(stats)),
        None => app,
    }
}

pub fn router(state: StatsState) -> Router {
    let mut router = Router::new();

    if let Some(kc) = state.keycloak.clone() {
        info!("Statistics page enabled for realm {} client {}", kc.realm, kc.client_id);
        let instance = KeycloakAuthInstance::new(
            KeycloakConfig::builder()
                .server(kc.url.clone())
                .realm(kc.realm.clone())
                .build(),
        );
        let protected = Router::new()
            .route("/stats/api/status", get(status))
            .route("/stats/api/overview", get(overview))
            .route("/stats/api/facets", get(facets))
            .route("/stats/api/groups", get(groups))
            .route("/stats/api/series", get(series_route))
            .route("/stats/api/items", get(items_list))
            .route("/stats/api/items/:code/files", get(item_files))
            .route("/stats/api/summary", get(summary))
            .route("/stats/api/export.csv", get(export_csv))
            .layer(middleware::from_fn(require_token))
            .layer(
                KeycloakAuthLayer::<String>::builder()
                    .instance(instance)
                    .passthrough_mode(PassthroughMode::Pass)
                    .persist_raw_claims(false)
                    // A shared realm puts in `aud` the clients whose roles the user holds, which
                    // for a viewer may be this client alone; a local realm puts "account".
                    .expected_audiences(vec![kc.client_id.clone(), "account".to_string()])
                    // Matches the role's name in the realm or under any client; `prepare`
                    // then accepts it only under this client.
                    .required_roles(vec![kc.role.clone()])
                    .build(),
            );
        let public = Router::new()
            .route("/stats", get(page))
            .route("/stats/signin.js", get(signin_js))
            .route("/stats/timeline.js", get(timeline_js))
            .route("/stats/chart.js", get(chart_js))
            .route("/stats/vendor/uPlot.iife.min.js", get(uplot_js))
            .route("/stats/vendor/uPlot.min.css", get(uplot_css))
            .route("/stats/app.js", get(app_js))
            .route("/stats/app.css", get(app_css))
            .route("/stats/config.json", get(browser_config));
        router = router.merge(public).merge(protected);
    }

    let csp_origin = state
        .keycloak
        .as_ref()
        .map(|kc| origin_of(&kc.browser_url))
        .unwrap_or_default();
    let csp = format!(
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self' {csp_origin}; \
         base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    );
    router
        .layer(middleware::from_fn(move |req: Request, next: Next| {
            let csp = csp.clone();
            async move { with_security_headers(req, next, &csp).await }
        }))
        .with_state(state)
}

fn origin_of(url: &str) -> String {
    match url.split_once("://") {
        Some((scheme, rest)) => format!("{scheme}://{}", rest.split('/').next().unwrap_or("")),
        None => String::new(),
    }
}

/// Answer a refused token as the Keycloak layer would, and log why at warn level. The reason names
/// the failed check only, never a claim or the user, so a lasting refusal (a rotated signing key,
/// a wrong audience) can be told apart in the log.
async fn require_token(mut req: Request, next: Next) -> Response {
    match req.extensions_mut().remove::<KeycloakAuthStatus<String, ProfileAndEmail>>() {
        Some(KeycloakAuthStatus::Success(token)) => {
            req.extensions_mut().insert(token);
            next.run(req).await
        }
        Some(KeycloakAuthStatus::Failure(err)) => {
            warn!("statistics request refused: {err}");
            err.as_ref().clone().into_response()
        }
        None => {
            error!("statistics request reached the routes without a Keycloak check");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

async fn with_security_headers(req: Request, next: Next, csp: &str) -> Response {
    let mut res = next.run(req).await;
    let h = res.headers_mut();
    if let Ok(v) = HeaderValue::from_str(csp) {
        h.insert(header::CONTENT_SECURITY_POLICY, v);
    }
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    h.insert(header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    h.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    h.insert(header::REFERRER_POLICY, HeaderValue::from_static("no-referrer"));
    res
}

// ----------------------------------------------------------------------------
// Reading the file proxy's metrics
// ----------------------------------------------------------------------------

/// One HTTP client and the address it reads the metrics from.
struct Scraper {
    client: reqwest::Client,
    url: Url,
    timeout: Duration,
}

impl Scraper {
    fn new(url: Url) -> anyhow::Result<Self> {
        Self::with_timeout(url, SCRAPE_TIMEOUT)
    }

    fn with_timeout(url: Url, timeout: Duration) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder().timeout(timeout).build()?;
        Ok(Self { client, url, timeout })
    }

    /// The metrics text, refused when the answer is not a success, takes longer than the timeout
    /// or is longer than `INGEST_LIMIT_BYTES`.
    async fn fetch(&self) -> anyhow::Result<String> {
        let read = async {
            let mut res = self.client.get(self.url.clone()).send().await?.error_for_status()?;
            if res.content_length().is_some_and(|n| n > INGEST_LIMIT_BYTES as u64) {
                anyhow::bail!("the metrics are longer than {INGEST_LIMIT_BYTES} bytes");
            }
            let mut body = Vec::new();
            while let Some(chunk) = res.chunk().await? {
                if body.len() + chunk.len() > INGEST_LIMIT_BYTES {
                    anyhow::bail!("the metrics are longer than {INGEST_LIMIT_BYTES} bytes");
                }
                body.extend_from_slice(&chunk);
            }
            Ok(String::from_utf8(body)?)
        };
        tokio::time::timeout(self.timeout, read)
            .await
            .map_err(|_| anyhow::anyhow!("no full answer within {}s", self.timeout.as_secs_f32()))?
    }
}

/// Read the metrics once and add them to the store. Reading the same counters again adds nothing,
/// so several API processes may each read them.
async fn scrape_once(scraper: &Scraper, state: &StatsState) -> anyhow::Result<()> {
    if !state.store.is_ready() {
        anyhow::bail!("the statistics database is not ready");
    }
    let scrape = parse_scrape(&scraper.fetch().await?);
    if scrape.paths.len() > MAX_PATHS_PER_SCRAPE {
        anyhow::bail!("the metrics carry {} paths, more than {MAX_PATHS_PER_SCRAPE}", scrape.paths.len());
    }
    let now = Utc::now();
    state.store.ingest(now.date_naive(), now.timestamp(), &scrape, &state.sizes).await
}

/// Read the metrics every `interval`, one read at a time; a failed read is logged and the next
/// tick tries again.
async fn poll(scraper: Scraper, interval: Duration, state: StatsState) {
    let mut ticks = tokio::time::interval(interval);
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        ticks.tick().await;
        if let Err(e) = scrape_once(&scraper, &state).await {
            warn!("reading the file proxy's metrics failed: {e:#}");
        }
    }
}

// ----------------------------------------------------------------------------
// Data routes (Keycloak)
// ----------------------------------------------------------------------------

/// The query string of a data route: a period, filters on item attributes and paging.
struct Params {
    raw: HashMap<String, String>,
}

impl Params {
    fn get(&self, key: &str) -> Option<&str> {
        self.raw.get(key).map(String::as_str).filter(|v| !v.is_empty())
    }

    fn day(&self, key: &str) -> Option<NaiveDate> {
        self.get(key).and_then(|d| NaiveDate::parse_from_str(d, "%Y-%m-%d").ok())
    }

    fn number(&self, key: &str, default: usize) -> usize {
        self.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
    }

    fn filters(&self) -> Filters {
        Filters::from_lookup(|k| self.get(k))
    }
}

/// Refuse a token issued to another client of the realm or without the role under this client,
/// and a malformed period, then read the query string.
fn prepare(
    token: &KeycloakToken<String>,
    state: &StatsState,
    raw: HashMap<String, String>,
) -> Result<Params, Response> {
    if !state
        .keycloak
        .as_ref()
        .is_some_and(|kc| token.authorized_party == kc.client_id && holds_role(&token.roles, &kc.client_id, &kc.role))
    {
        warn!("statistics request refused: the token was issued to another client");
        return Err(StatusCode::FORBIDDEN.into_response());
    }
    if !state.store.is_ready() {
        return Err(StatusCode::SERVICE_UNAVAILABLE.into_response());
    }
    let params = Params { raw };
    for key in ["from", "to"] {
        if let Some(d) = params.get(key) {
            if NaiveDate::parse_from_str(d, "%Y-%m-%d").is_err() {
                return Err(StatusCode::BAD_REQUEST.into_response());
            }
        }
    }
    Ok(params)
}

/// Whether the token holds `role` as a role of this client. The realm is shared with other
/// applications, so a realm role of the same name, or one under another client, does not count.
fn holds_role(roles: &[KeycloakRole<String>], client_id: &str, role: &str) -> bool {
    roles.iter().any(|r| match r {
        KeycloakRole::Realm { .. } => false,
        KeycloakRole::Client { client, role: name } => client == client_id && name == role,
    })
}

/// Turn a database error into a plain 500; the detail goes to the log, never to the client.
fn db<T>(result: anyhow::Result<T>) -> Result<T, Response> {
    result.map_err(|e| {
        error!("statistics query failed: {e:#}");
        StatusCode::INTERNAL_SERVER_ERROR.into_response()
    })
}

fn totals_json(t: Totals) -> serde_json::Value {
    serde_json::json!({ "requests": t.requests, "bytes": t.bytes, "whole_files": t.whole_files() })
}

/// The page polls this cheap route and reloads its data only when the value changes.
async fn status(
    State(state): State<StatsState>,
    Extension(token): Extension<KeycloakToken<String>>,
    Query(raw): Query<HashMap<String, String>>,
) -> Response {
    if let Err(r) = prepare(&token, &state, raw) {
        return r;
    }
    match db(state.store.last_ingest().await) {
        Ok(last) => Json(serde_json::json!({ "last_ingest_at": last })).into_response(),
        Err(r) => r,
    }
}

/// The values each attribute can take, with item counts, for the filter menus.
async fn facets(
    State(state): State<StatsState>,
    Extension(token): Extension<KeycloakToken<String>>,
    Query(raw): Query<HashMap<String, String>>,
) -> Response {
    if let Err(r) = prepare(&token, &state, raw) {
        return r;
    }
    let menus: serde_json::Map<String, serde_json::Value> = facet_values(&state.items)
        .into_iter()
        .map(|(key, values)| {
            let rows: Vec<_> = values
                .into_iter()
                .map(|(value, items)| serde_json::json!({ "value": value, "items": items }))
                .collect();
            (key.to_string(), serde_json::Value::Array(rows))
        })
        .collect();
    Json(menus).into_response()
}

async fn overview(
    State(state): State<StatsState>,
    Extension(token): Extension<KeycloakToken<String>>,
    Query(raw): Query<HashMap<String, String>>,
) -> Response {
    let p = match prepare(&token, &state, raw) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let filters = p.filters();
    let now = Utc::now();
    let today = now.date_naive();
    let items = &state.items;
    let f = &filters;
    let data = async {
        let store = &state.store;
        let paths = store.path_totals(p.day("from"), p.day("to")).await?;
        let recent = store.recent_rows(now.timestamp() - 24 * 3600).await?;
        let today_rows = store.code_days(Some(today), Some(today)).await?;
        let by_code: HashMap<&str, &ItemInfo> = items.iter().map(|i| (i.code.as_str(), i)).collect();

        let breakdown = breakdown(&paths, items).filtered(f);

        let mut timeline: std::collections::BTreeMap<i64, Totals> = Default::default();
        let mut hour = Totals::default();
        let mut day24 = Totals::default();
        for (ts, path, t) in &recent {
            let code = code_key(path);
            if code_passes(&code, &by_code, f).is_none() {
                continue;
            }
            timeline.entry(*ts).or_default().add(*t);
            day24.add(*t);
            if *ts >= now.timestamp() - 3600 {
                hour.add(*t);
            }
        }
        let mut today_totals = Totals::default();
        for (_, code, t) in &today_rows {
            if code_passes(code, &by_code, f).is_some() {
                today_totals.add(*t);
            }
        }
        anyhow::Ok((breakdown, timeline, hour, day24, today_totals, store.last_ingest().await?))
    }
    .await;
    let (b, timeline, hour, day24, today_totals, last) = match db(data) {
        Ok(d) => d,
        Err(r) => return r,
    };

    let mut total = b.docs;
    total.add(b.unknown);
    for it in &b.items {
        total.add(Totals { requests: it.requests, bytes: it.bytes, equiv: (it.whole_files * 1000.0).round() as u64 });
    }

    Json(serde_json::json!({
        "last_ingest_at": last,
        "now": now.timestamp(),
        "filtered": filters.is_active(),
        "totals": totals_json(total),
        "today": totals_json(today_totals),
        "last_hour": totals_json(hour),
        "last_24h": totals_json(day24),
        "files_downloaded": b.items.iter().map(|i| i.files_downloaded).sum::<usize>(),
        "files_total": b.items.iter().map(|i| i.files_total).sum::<usize>(),
        "items_with_downloads": b.items.iter().filter(|i| i.requests > 0).count(),
        "items_total": b.items.len(),
        "docs": totals_json(b.docs),
        "unknown": totals_json(b.unknown),
        "timeline": timeline.iter().map(|(ts, t)| serde_json::json!({"ts": ts, "requests": t.requests, "bytes": t.bytes})).collect::<Vec<_>>(),
    }))
    .into_response()
}

/// Items summed by an attribute, for the table under a chart.
async fn groups(
    State(state): State<StatsState>,
    Extension(token): Extension<KeycloakToken<String>>,
    Query(raw): Query<HashMap<String, String>>,
) -> Response {
    let p = match prepare(&token, &state, raw) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let Some(dim) = p.get("dim").and_then(Dim::parse).filter(|d| *d != Dim::Total) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let paths = match db(state.store.path_totals(p.day("from"), p.day("to")).await) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let b = breakdown(&paths, &state.items).filtered(&p.filters());
    Json(serde_json::json!({ "rows": group_totals(&b.items, dim) })).into_response()
}

/// Requests and bytes per group by hour, day or week, for the stacked chart. `resolution` is
/// picked from the width of the period when not given.
async fn series_route(
    State(state): State<StatsState>,
    Extension(token): Extension<KeycloakToken<String>>,
    Query(raw): Query<HashMap<String, String>>,
) -> Response {
    let p = match prepare(&token, &state, raw) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let Some(dim) = Dim::parse(p.get("dim").unwrap_or("total")) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let resolution = match p.get("resolution").map(Resolution::parse) {
        Some(None) => return StatusCode::BAD_REQUEST.into_response(),
        Some(r) => r,
        None => None,
    };
    let (from, to) = (p.day("from"), p.day("to"));
    let first_data = if from.is_none() {
        match db(state.store.first_day().await) {
            Ok(d) => d,
            Err(r) => return r,
        }
    } else {
        None
    };
    let Some(window) = series_window(resolution, from, to, first_data, Utc::now().naive_utc()) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let rows = match db(state.store.code_buckets(window.resolution, from, to).await) {
        Ok(r) => r,
        Err(r) => return r,
    };
    Json(series(&rows, &state.items, dim, &p.filters(), 7, &window)).into_response()
}

const MAX_PAGE: usize = 200;

async fn items_list(
    State(state): State<StatsState>,
    Extension(token): Extension<KeycloakToken<String>>,
    Query(raw): Query<HashMap<String, String>>,
) -> Response {
    let p = match prepare(&token, &state, raw) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let paths = match db(state.store.path_totals(p.day("from"), p.day("to")).await) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let mut rows = breakdown(&paths, &state.items).filtered(&p.filters()).items;
    if let Some(needle) = p.get("q").map(str::to_lowercase) {
        rows.retain(|r| {
            [&r.item, &r.title, &r.facets.collection, &r.facets.provider, &r.facets.data_type, &r.facets.dataset, &r.facets.sensor]
                .iter()
                .any(|v| v.to_lowercase().contains(&needle))
        });
    }
    let descending = p.get("dir") != Some("asc");
    let sort = p.get("sort").unwrap_or("requests");
    match sort {
        "item" => rows.sort_by(|a, b| a.item.cmp(&b.item)),
        "collection" => rows.sort_by(|a, b| a.facets.collection.cmp(&b.facets.collection).then(a.item.cmp(&b.item))),
        "bytes" => rows.sort_by(|a, b| a.bytes.cmp(&b.bytes).then(a.item.cmp(&b.item))),
        "whole_files" => rows.sort_by(|a, b| a.whole_files.total_cmp(&b.whole_files).then(a.item.cmp(&b.item))),
        "files" => rows.sort_by(|a, b| a.files_downloaded.cmp(&b.files_downloaded).then(a.item.cmp(&b.item))),
        "files_total" => rows.sort_by(|a, b| a.files_total.cmp(&b.files_total).then(a.item.cmp(&b.item))),
        _ => rows.sort_by(|a, b| a.requests.cmp(&b.requests).then(a.item.cmp(&b.item))),
    }
    let by_name = matches!(sort, "item" | "collection");
    // Names sort ascending unless asked otherwise; numbers sort descending unless asked otherwise.
    if (by_name && p.get("dir") == Some("desc")) || (!by_name && descending) {
        rows.reverse();
    }
    let total = rows.len();
    let limit = p.number("limit", 25).clamp(1, MAX_PAGE);
    let offset = p.number("offset", 0).min(total);
    let page: Vec<_> = rows.into_iter().skip(offset).take(limit).collect();
    Json(serde_json::json!({ "total": total, "offset": offset, "rows": page })).into_response()
}

async fn item_files(
    State(state): State<StatsState>,
    Extension(token): Extension<KeycloakToken<String>>,
    Path(code): Path<String>,
    Query(raw): Query<HashMap<String, String>>,
) -> Response {
    let p = match prepare(&token, &state, raw) {
        Ok(p) => p,
        Err(r) => return r,
    };
    if !state.items.iter().any(|i| i.code == code) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let mut rows: Vec<(String, Totals)> =
        match db(state.store.path_totals(p.day("from"), p.day("to")).await) {
            Ok(p) => p,
            Err(r) => return r,
        };
    let prefix = format!("/assets/{code}/");
    let archive = format!("/archives/{code}.zip");
    rows.retain(|(p, _)| p.starts_with(&prefix) || *p == archive);
    rows.sort_by(|a, b| b.1.requests.cmp(&a.1.requests).then(a.0.cmp(&b.0)));
    let total = rows.len();
    let limit = p.number("limit", 25).clamp(1, MAX_PAGE);
    let offset = p.number("offset", 0).min(total);
    let page: Vec<_> = rows
        .iter()
        .skip(offset)
        .take(limit)
        .map(|(path, t)| {
            let name = if *path == archive {
                format!("{code}.zip (archive)")
            } else {
                file_name_in_item(path, &code).to_string()
            };
            serde_json::json!({ "file": name, "requests": t.requests, "bytes": t.bytes, "whole_files": t.whole_files() })
        })
        .collect();
    Json(serde_json::json!({ "total": total, "offset": offset, "rows": page })).into_response()
}

/// Requests and bytes by day, collection, item and archive or file: the rows of the by-day CSV.
async fn summary(
    State(state): State<StatsState>,
    Extension(token): Extension<KeycloakToken<String>>,
    Query(raw): Query<HashMap<String, String>>,
) -> Response {
    let p = match prepare(&token, &state, raw) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let rows = match db(state.store.daily(p.day("from"), p.day("to")).await) {
        Ok(r) => r,
        Err(r) => return r,
    };
    Json(serde_json::json!({ "rows": summary_rows(rows, &state.items, &p.filters()) })).into_response()
}

async fn export_csv(
    State(state): State<StatsState>,
    Extension(token): Extension<KeycloakToken<String>>,
    Query(raw): Query<HashMap<String, String>>,
) -> Response {
    let p = match prepare(&token, &state, raw) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let level_file = p.get("level_of_detail") == Some("file");
    let filters = p.filters();
    let items = &state.items;
    let body = async {
        if level_file {
            let by_code: HashMap<&str, &ItemInfo> = items.iter().map(|i| (i.code.as_str(), i)).collect();
            let mut rows = state.store.path_totals(p.day("from"), p.day("to")).await?;
            rows.retain(|(path, _)| code_passes(&code_key(path), &by_code, &filters).is_some());
            rows.sort_by(|a, b| b.1.requests.cmp(&a.1.requests).then(a.0.cmp(&b.0)));
            anyhow::Ok(crate::stats::file_csv(&rows))
        } else {
            let rows = state.store.daily(p.day("from"), p.day("to")).await?;
            Ok(to_csv(&summary_rows(rows, items, &filters)))
        }
    }
    .await;
    let body = match db(body) {
        Ok(b) => b,
        Err(r) => return r,
    };
    (
        [
            (header::CONTENT_TYPE, "text/csv; charset=utf-8"),
            (header::CONTENT_DISPOSITION, "attachment; filename=\"download-statistics.csv\""),
        ],
        body,
    )
        .into_response()
}

// ----------------------------------------------------------------------------
// Page (public, carries no data)
// ----------------------------------------------------------------------------

async fn page() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], include_str!("stats_ui/index.html"))
}

async fn signin_js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8")], include_str!("stats_ui/signin.js"))
}

async fn timeline_js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8")], include_str!("stats_ui/timeline.js"))
}

async fn chart_js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8")], include_str!("stats_ui/chart.js"))
}

// uPlot, MIT licensed; its licence is kept beside it in `stats_ui/vendor/`.
async fn uplot_js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8")], include_str!("stats_ui/vendor/uPlot.iife.min.js"))
}

async fn uplot_css() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], include_str!("stats_ui/vendor/uPlot.min.css"))
}

async fn app_js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8")], include_str!("stats_ui/app.js"))
}

async fn app_css() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], include_str!("stats_ui/app.css"))
}

async fn browser_config(State(state): State<StatsState>) -> Response {
    let Some(kc) = state.keycloak.as_ref() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let base = format!(
        "{}/realms/{}/protocol/openid-connect",
        kc.browser_url.trim_end_matches('/'),
        kc.realm
    );
    Json(serde_json::json!({
        "authUrl": format!("{base}/auth"),
        "tokenUrl": format!("{base}/token"),
        "logoutUrl": format!("{base}/logout"),
        "clientId": kc.client_id,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum_keycloak_auth::error::AuthError;
    use axum::http::Request as HttpRequest;
    use tower::ServiceExt;

    /// A store that never connects: enough for routes that answer before touching the database.
    fn state() -> StatsState {
        StatsState {
            store: Arc::new(Store::lazy("postgres://nobody:nothing@127.0.0.1:1/none").unwrap()),
            items: Arc::new(vec![]),
            sizes: Arc::new(HashMap::new()),
            keycloak: None,
        }
    }

    const BODY: &str = "process_start_time_seconds 100\n\
        http_requests_total{host=\"h\",method=\"GET\",path=\"/assets/10Aa00/a.jpg\",server=\"business\",status_code=\"200\"} 3\n\
        http_response_size_bytes_sum{host=\"h\",method=\"GET\",path=\"/assets/10Aa00/a.jpg\",server=\"business\",status_code=\"200\"} 300\n";

    #[test]
    fn the_role_counts_under_this_client_only() {
        let realm = |r: &str| KeycloakRole::Realm { role: r.to_string() };
        let client = |c: &str, r: &str| KeycloakRole::Client { client: c.to_string(), role: r.to_string() };
        assert!(!holds_role(&[realm("viewer")], "dash", "viewer"));
        assert!(holds_role(&[client("dash", "viewer")], "dash", "viewer"));
        assert!(!holds_role(&[client("elsewhere", "viewer")], "dash", "viewer"));
        assert!(!holds_role(&[realm("other"), client("dash", "other")], "dash", "viewer"));
    }

    #[tokio::test]
    async fn without_keycloak_no_route_exists() {
        let app = router(state());
        for path in ["/stats", "/stats/api/status", "/stats/api/facets", "/stats/api/overview", "/stats/api/items", "/stats/api/items/10Aa00/files", "/stats/api/summary", "/stats/api/export.csv", "/stats/config.json"] {
            let res = app.clone().oneshot(HttpRequest::get(path).body(Body::empty()).unwrap()).await.unwrap();
            assert_eq!(res.status(), StatusCode::NOT_FOUND, "{path}");
        }
    }

    fn vars(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |name| map.get(name).cloned()
    }

    const GOOD_DB: &str = "postgres://nobody:nothing@127.0.0.1:1/none";
    // An unescaped `/` in the password ends the host part, leaving `p` as the port.
    const BAD_DB: &str = "postgres://orchard:p/ear@db:5432/harvest";

    #[test]
    fn an_unparsable_database_url_turns_statistics_off_with_a_reason() {
        match check_settings(vars(&[("STATS_DATABASE_URL", BAD_DB)])) {
            StatsSettings::Off(reason) => assert!(reason.contains("STATS_DATABASE_URL"), "{reason}"),
            _ => panic!("expected statistics off"),
        }
        assert!(matches!(check_settings(vars(&[])), StatsSettings::Absent));
    }

    #[test]
    fn a_keycloak_url_without_scheme_leaves_out_the_page_with_a_reason() {
        let settings = check_settings(vars(&[
            ("STATS_DATABASE_URL", GOOD_DB),
            ("KEYCLOAK_URL", "keycloak:8080"),
            ("KEYCLOAK_REALM", "orchard"),
            ("KEYCLOAK_CLIENT_ID", "pears"),
        ]));
        match settings {
            StatsSettings::On { keycloak: Err(reason), .. } => assert!(reason.contains("KEYCLOAK_URL"), "{reason}"),
            _ => panic!("expected the page routes left out"),
        }
    }

    #[test]
    fn keycloak_url_must_be_http_with_a_host() {
        assert!(keycloak_server_url("http://keycloak:8080").is_ok());
        assert!(keycloak_server_url("https://sso.example.ch/auth").is_ok());
        for bad in ["keycloak:8080", "localhost:8180", "ftp://sso.example.ch", "mailto:a@b.ch", "sso.example.ch", "http://"] {
            assert!(keycloak_server_url(bad).is_err(), "{bad}");
        }
    }

    #[tokio::test]
    async fn bad_statistics_settings_leave_the_rest_of_the_app_up() {
        for settings in [
            vec![("STATS_DATABASE_URL", BAD_DB), ("STATS_SCRAPE_URL", "http://proxy:9090/metrics")],
            vec![
                ("STATS_DATABASE_URL", GOOD_DB),
                ("KEYCLOAK_URL", "keycloak:8080"),
                ("KEYCLOAK_REALM", "orchard"),
                ("KEYCLOAK_CLIENT_ID", "pears"),
            ],
        ] {
            let app = mount(Router::new().route("/health", get(|| async { "ok" })), vars(&settings), vec![], HashMap::new());
            let res = app.clone().oneshot(HttpRequest::get("/health").body(Body::empty()).unwrap()).await.unwrap();
            assert_eq!(res.status(), StatusCode::OK, "{settings:?}");
            let res = app.oneshot(HttpRequest::get("/stats").body(Body::empty()).unwrap()).await.unwrap();
            assert_eq!(res.status(), StatusCode::NOT_FOUND, "{settings:?}");
        }
    }

    #[tokio::test]
    async fn no_ingest_route_exists_even_with_the_old_key_set() {
        let settings = [("STATS_DATABASE_URL", GOOD_DB), ("STATS_INGEST_KEY", "k")];
        let app = mount(Router::new(), vars(&settings), vec![], HashMap::new());
        let req = HttpRequest::post("/internal/stats/ingest").header(header::AUTHORIZATION, "Bearer k");
        let res = app.oneshot(req.body(Body::from(BODY)).unwrap()).await.unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }

    async fn behind_require_token(status: Option<KeycloakAuthStatus<String, ProfileAndEmail>>) -> StatusCode {
        let app = Router::new()
            .route("/", get(|Extension(_): Extension<KeycloakToken<String>>| async { StatusCode::OK }))
            .layer(middleware::from_fn(require_token));
        let mut req = HttpRequest::get("/").body(Body::empty()).unwrap();
        if let Some(s) = status {
            req.extensions_mut().insert(s);
        }
        app.oneshot(req).await.unwrap().status()
    }

    #[tokio::test]
    async fn a_refused_token_keeps_the_keycloak_status_and_never_reaches_the_route() {
        let expired = KeycloakAuthStatus::Failure(Arc::new(AuthError::TokenExpired));
        assert_eq!(behind_require_token(Some(expired)).await, StatusCode::UNAUTHORIZED);
        let role = KeycloakAuthStatus::Failure(Arc::new(AuthError::MissingExpectedRole { role: "viewer".into() }));
        assert_eq!(behind_require_token(Some(role)).await, StatusCode::FORBIDDEN);
        assert_eq!(behind_require_token(None).await, StatusCode::INTERNAL_SERVER_ERROR);
    }

    /// A file proxy that answers `/metrics` with `BODY`, `/broken` with a 500, `/stuck` never,
    /// and `/huge` with more text than one read takes.
    async fn fake_proxy() -> Url {
        let app = Router::new()
            .route("/metrics", get(|| async { BODY }))
            .route("/broken", get(|| async { StatusCode::INTERNAL_SERVER_ERROR }))
            .route("/stuck", get(|| std::future::pending::<&'static str>()))
            .route("/huge", get(|| async { format!("{BODY}{}", "#".repeat(INGEST_LIMIT_BYTES)) }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Url::parse(&format!("http://{addr}/")).unwrap()
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn one_read_stores_a_good_answer_and_nothing_else() {
        let store = crate::stats_store::tests::baselined_store().await;
        let state = StatsState { store: Arc::new(store.clone()), ..state() };
        let proxy = fake_proxy().await;
        let scraper = |path: &str| Scraper::with_timeout(proxy.join(path).unwrap(), Duration::from_secs(1)).unwrap();
        let stored = || async {
            store.path_totals(None, None).await.unwrap().iter().map(|(_, t)| (t.requests, t.bytes)).collect::<Vec<_>>()
        };

        for path in ["broken", "stuck", "huge"] {
            assert!(scrape_once(&scraper(path), &state).await.is_err(), "{path}");
        }
        assert_eq!(stored().await, vec![]);
        assert_eq!(store.last_ingest().await.unwrap(), Some(0));

        scrape_once(&scraper("metrics"), &state).await.unwrap();
        assert_eq!(stored().await, vec![(3, 300)]);
        scrape_once(&scraper("metrics"), &state).await.unwrap();
        assert_eq!(stored().await, vec![(3, 300)]);
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn the_first_read_into_an_empty_store_counts_nothing() {
        let store = crate::stats_store::tests::test_store().await;
        let state = StatsState { store: Arc::new(store.clone()), ..state() };
        let proxy = fake_proxy().await;
        let scraper = Scraper::with_timeout(proxy.join("metrics").unwrap(), Duration::from_secs(1)).unwrap();
        scrape_once(&scraper, &state).await.unwrap();
        assert!(store.path_totals(None, None).await.unwrap().is_empty());
        assert!(store.last_ingest().await.unwrap().is_some());
    }

    #[tokio::test]
    async fn a_read_waits_for_the_database_to_be_ready() {
        let proxy = fake_proxy().await;
        let scraper = Scraper::new(proxy.join("metrics").unwrap()).unwrap();
        let err = scrape_once(&scraper, &state()).await.unwrap_err();
        assert!(err.to_string().contains("not ready"), "{err}");
    }

    #[test]
    fn an_unusable_scrape_interval_turns_the_poller_off_with_a_reason() {
        let scrape = |pairs: &[(&str, &str)]| {
            let all: Vec<_> = [("STATS_DATABASE_URL", GOOD_DB)].iter().chain(pairs).copied().collect();
            match check_settings(vars(&all)) {
                StatsSettings::On { scrape, .. } => scrape,
                _ => panic!("expected statistics on"),
            }
        };
        let url = ("STATS_SCRAPE_URL", "http://proxy:9090/metrics");
        for bad in ["0", "abc", "-5", "1.5"] {
            match scrape(&[url, ("STATS_SCRAPE_INTERVAL", bad)]) {
                Err(reason) => assert!(reason.contains("STATS_SCRAPE_INTERVAL"), "{bad}: {reason}"),
                Ok(s) => panic!("{bad}: expected the poller off, got {s:?}"),
            }
        }
        let every = |secs| Ok(Some(ScrapeSettings { url: Url::parse(url.1).unwrap(), interval: Duration::from_secs(secs) }));
        assert_eq!(scrape(&[url]), every(60));
        assert_eq!(scrape(&[url, ("STATS_SCRAPE_INTERVAL", "15")]), every(15));
        assert_eq!(scrape(&[("STATS_SCRAPE_INTERVAL", "15")]), Ok(None));
        assert!(scrape(&[("STATS_SCRAPE_URL", "proxy:9090/metrics")]).unwrap_err().contains("STATS_SCRAPE_URL"));
    }

    async fn get_text(app: &Router, path: &str) -> (StatusCode, String) {
        let res = app.clone().oneshot(HttpRequest::get(path).body(Body::empty()).unwrap()).await.unwrap();
        let status = res.status();
        let body = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn the_page_is_a_shell_and_its_script_reads_data_only_with_a_bearer_token() {
        let mut s = state();
        s.keycloak = Some(Arc::new(KeycloakSettings {
            url: Url::parse("http://127.0.0.1:1/").unwrap(),
            browser_url: "http://127.0.0.1:1/".to_string(),
            realm: "orchard".to_string(),
            client_id: "pears".to_string(),
            role: "picker".to_string(),
        }));
        let app = router(s);

        // The store never connects, so a page that read any statistics could not answer.
        let (status, page) = get_text(&app, "/stats").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(page, include_str!("stats_ui/index.html"));
        assert!(!page.contains("/stats/api/"), "the shell names no data route");
        assert!(page.contains("Signing in"));

        let (status, script) = get_text(&app, "/stats/app.js").await;
        assert_eq!(status, StatusCode::OK);
        let fetches: Vec<&str> = script.split("fetch(").skip(1).collect();
        assert!(!fetches.is_empty());
        for call in &fetches {
            let call = &call[..call.find('\n').unwrap_or(call.len())];
            let to_api = !call.starts_with("'/stats/config.json'") && !call.starts_with("cfg.tokenUrl");
            if to_api {
                assert!(call.contains("Authorization: 'Bearer ' + token"), "fetch without a bearer token: {call}");
            }
        }
        // Every data route is reached through that one request function.
        for (i, _) in script.match_indices("'/stats/api/") {
            assert!(script[..i].ends_with("api("), "data route requested outside api(): {}", &script[i..i + 40]);
        }
        assert!(script.contains("api('/stats/api/export.csv?"));
        // The page's other scripts, the vendored chart library included, make no requests of their own.
        for helper in ["/stats/signin.js", "/stats/timeline.js", "/stats/chart.js", "/stats/vendor/uPlot.iife.min.js"] {
            assert!(page.contains(&format!("<script src=\"{helper}\"></script>")), "the page loads {helper}");
            let (status, code) = get_text(&app, helper).await;
            assert_eq!(status, StatusCode::OK);
            assert!(!code.contains("fetch("), "{helper} fetches");
        }
    }

    #[tokio::test]
    async fn the_vendored_chart_library_is_served_with_its_content_types() {
        let mut s = state();
        s.keycloak = Some(Arc::new(KeycloakSettings {
            url: Url::parse("http://127.0.0.1:1/").unwrap(),
            browser_url: "http://127.0.0.1:1/".to_string(),
            realm: "orchard".to_string(),
            client_id: "pears".to_string(),
            role: "picker".to_string(),
        }));
        let app = router(s);
        for (path, content_type, body) in [
            ("/stats/vendor/uPlot.iife.min.js", "text/javascript; charset=utf-8", include_str!("stats_ui/vendor/uPlot.iife.min.js")),
            ("/stats/vendor/uPlot.min.css", "text/css; charset=utf-8", include_str!("stats_ui/vendor/uPlot.min.css")),
            ("/stats/chart.js", "text/javascript; charset=utf-8", include_str!("stats_ui/chart.js")),
        ] {
            let res = app.clone().oneshot(HttpRequest::get(path).body(Body::empty()).unwrap()).await.unwrap();
            assert_eq!(res.status(), StatusCode::OK, "{path}");
            assert_eq!(res.headers()[header::CONTENT_TYPE], content_type, "{path}");
            let served = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
            assert_eq!(served, body.as_bytes(), "{path}");
        }
        let (_, page) = get_text(&app, "/stats").await;
        assert!(page.contains("<link rel=\"stylesheet\" href=\"/stats/vendor/uPlot.min.css\">"));
    }

    #[tokio::test]
    async fn the_browser_config_names_the_realm_endpoints_including_logout() {
        let mut s = state();
        s.keycloak = Some(Arc::new(KeycloakSettings {
            url: Url::parse("http://keycloak:8080/").unwrap(),
            browser_url: "https://sso.orchard.test/".to_string(),
            realm: "orchard".to_string(),
            client_id: "pears".to_string(),
            role: "picker".to_string(),
        }));
        let (status, body) = get_text(&router(s), "/stats/config.json").await;
        assert_eq!(status, StatusCode::OK);
        let cfg: serde_json::Value = serde_json::from_str(&body).unwrap();
        let base = "https://sso.orchard.test/realms/orchard/protocol/openid-connect";
        assert_eq!(cfg["authUrl"], format!("{base}/auth"));
        assert_eq!(cfg["tokenUrl"], format!("{base}/token"));
        assert_eq!(cfg["logoutUrl"], format!("{base}/logout"));
        assert_eq!(cfg["clientId"], "pears");
    }

    const SIGNING_SECRET: &[u8] = b"orchard-test-signing-secret-01234";
    const SIGNING_SECRET_B64URL: &str = "b3JjaGFyZC10ZXN0LXNpZ25pbmctc2VjcmV0LTAxMjM0";

    /// A realm on a local port that publishes the test signing key, as Keycloak's discovery does.
    /// The receiver fires once discovery has been asked for: the auth layer panics on a request
    /// that comes before its first discovery has started.
    async fn fake_realm() -> (Url, tokio::sync::oneshot::Receiver<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let jwks_uri = format!("{base}/jwks");
        let (asked, discovering) = tokio::sync::oneshot::channel();
        let asked = Arc::new(std::sync::Mutex::new(Some(asked)));
        let realm = Router::new()
            .route(
                "/realms/:realm/.well-known/openid-configuration",
                get(move || async move {
                    if let Some(tx) = asked.lock().unwrap().take() {
                        let _ = tx.send(());
                    }
                    Json(serde_json::json!({
                        "issuer": "http://orchard.test/realms/orchard",
                        "authorization_endpoint": "http://orchard.test/auth",
                        "jwks_uri": jwks_uri,
                        "response_types_supported": ["code"],
                        "subject_types_supported": ["public"],
                        "id_token_signing_alg_values_supported": ["HS256"],
                    }))
                }),
            )
            .route(
                "/jwks",
                get(|| async {
                    Json(serde_json::json!({
                        "keys": [{ "kty": "oct", "kid": "orchard", "alg": "HS256", "k": SIGNING_SECRET_B64URL }]
                    }))
                }),
            );
        tokio::spawn(async move { axum::serve(listener, realm).await.unwrap() });
        (Url::parse(&base).unwrap(), discovering)
    }

    #[derive(Clone, Copy)]
    struct Claims<'a> {
        expires_in: i64,
        audience: &'a [&'a str],
        client: &'a str,
        realm_roles: &'a [&'a str],
        /// The client under whose `resource_access` entry `client_roles` are written.
        roles_client: &'a str,
        client_roles: &'a [&'a str],
        secret: &'a [u8],
    }

    const GOOD: Claims<'static> =
        Claims {
        expires_in: 300,
        audience: &["account"],
        client: "pears",
        realm_roles: &[],
        roles_client: "pears",
        client_roles: &["stats-viewer"],
        secret: SIGNING_SECRET,
    };

    fn token(c: Claims) -> String {
        let now = Utc::now().timestamp();
        let claims = serde_json::json!({
            "exp": now + c.expires_in,
            "iat": now - 600,
            "jti": "orchard-token",
            "iss": "http://orchard.test/realms/orchard",
            "aud": c.audience,
            "sub": "picker",
            "typ": "Bearer",
            "azp": c.client,
            "realm_access": { "roles": c.realm_roles },
            "resource_access": { c.roles_client: { "roles": c.client_roles } },
            "preferred_username": "picker",
            "email": "picker@orchard.test",
            "email_verified": true,
        });
        let mut header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256);
        header.kid = Some("orchard".into());
        jsonwebtoken::encode(&header, &claims, &jsonwebtoken::EncodingKey::from_secret(c.secret)).unwrap()
    }

    async fn guarded(store: Store) -> Router {
        let (server, discovering) = fake_realm().await;
        let app = router(StatsState {
            store: Arc::new(store),
            keycloak: Some(Arc::new(KeycloakSettings {
                url: server,
                browser_url: "http://orchard.test".into(),
                realm: "orchard".into(),
                client_id: "pears".into(),
                role: "stats-viewer".into(),
            })),
            ..state()
        });
        discovering.await.unwrap();
        app
    }

    async fn read(app: &Router, path: &str, bearer: Option<&str>) -> StatusCode {
        let mut req = HttpRequest::get(path);
        if let Some(t) = bearer {
            req = req.header(header::AUTHORIZATION, format!("Bearer {t}"));
        }
        app.clone().oneshot(req.body(Body::empty()).unwrap()).await.unwrap().status()
    }

    const READ_ROUTES: [&str; 2] = ["/stats/api/summary", "/stats/api/export.csv"];

    #[tokio::test]
    async fn read_routes_refuse_every_token_but_a_valid_one_for_this_client_and_role() {
        let app = guarded(Store::lazy(GOOD_DB).unwrap()).await;
        let refused = [
            ("no token", None, StatusCode::UNAUTHORIZED),
            ("expired", Some(token(Claims { expires_in: -300, ..GOOD })), StatusCode::UNAUTHORIZED),
            ("another audience", Some(token(Claims { audience: &["plums"], ..GOOD })), StatusCode::UNAUTHORIZED),
            ("another signing key", Some(token(Claims { secret: b"some-other-signing-secret-567890", ..GOOD })), StatusCode::UNAUTHORIZED),
            ("without the role", Some(token(Claims { client_roles: &["offline_access"], ..GOOD })), StatusCode::FORBIDDEN),
            ("the role only in the realm", Some(token(Claims { realm_roles: &["stats-viewer"], client_roles: &[], ..GOOD })), StatusCode::FORBIDDEN),
            ("the role only under another client", Some(token(Claims { roles_client: "plums", ..GOOD })), StatusCode::FORBIDDEN),
            ("issued to another client", Some(token(Claims { client: "plums", ..GOOD })), StatusCode::FORBIDDEN),
        ];
        for path in READ_ROUTES {
            for (case, bearer, expected) in &refused {
                assert_eq!(read(&app, path, bearer.as_deref()).await, *expected, "{path}: {case}");
            }
            // A good token passes the check and reaches the handler, which waits for the database.
            assert_eq!(read(&app, path, Some(&token(GOOD))).await, StatusCode::SERVICE_UNAVAILABLE, "{path}");
        }
        assert_eq!(read(&app, "/stats", None).await, StatusCode::OK);
    }

    /// On a shared realm a token's audience lists the clients whose roles the user holds, so a
    /// viewer without the realm's default account roles carries only this client's id.
    #[tokio::test]
    async fn read_routes_accept_a_token_whose_audience_is_this_client() {
        let app = guarded(Store::lazy(GOOD_DB).unwrap()).await;
        let for_this_client = Claims { audience: &["pears"], ..GOOD };
        let beside_account = Claims { audience: &["pears", "account"], ..GOOD };
        let for_another_app = Claims { audience: &["plums"], ..GOOD };
        for path in READ_ROUTES {
            assert_eq!(read(&app, path, Some(&token(for_this_client))).await, StatusCode::SERVICE_UNAVAILABLE, "{path}");
            assert_eq!(read(&app, path, Some(&token(beside_account))).await, StatusCode::SERVICE_UNAVAILABLE, "{path}");
            assert_eq!(read(&app, path, Some(&token(for_another_app))).await, StatusCode::UNAUTHORIZED, "{path}");
        }
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn read_routes_answer_a_valid_token() {
        let store = crate::stats_store::tests::test_store().await;
        let app = guarded(store).await;
        for path in READ_ROUTES {
            assert_eq!(read(&app, path, Some(&token(GOOD))).await, StatusCode::OK, "{path}");
            assert_eq!(read(&app, path, None).await, StatusCode::UNAUTHORIZED, "{path}");
        }
    }

    #[test]
    fn origin_is_scheme_and_host() {
        assert_eq!(origin_of("http://localhost:8180/"), "http://localhost:8180");
        assert_eq!(origin_of("https://sso.example.ch/x/y"), "https://sso.example.ch");
    }

    #[test]
    fn query_days_are_parsed_never_passed_on_as_text() {
        let p = Params { raw: HashMap::from([("from".to_string(), "2026-10-01".to_string()), ("to".to_string(), "x'; DROP TABLE daily; --".to_string())]) };
        assert_eq!(p.day("from"), NaiveDate::from_ymd_opt(2026, 10, 1));
        assert_eq!(p.day("to"), None);
    }
}

