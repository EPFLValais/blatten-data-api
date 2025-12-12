//! Blatten Data API - STAC API Server
//!
//! A STAC 1.1.0 compliant API server for the Birch Glacier Collapse Dataset.
//! Uses an in-memory backend loaded from JSON catalog files.

use anyhow::{Context, Result};
use axum::{
    extract::{Path, Query, State},
    http::{header, Method, StatusCode},
    response::{IntoResponse, Json},
    routing::get,
    Router,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{env, net::SocketAddr, path::PathBuf, sync::Arc};
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod catalog;

use catalog::{StacCatalog, StacItem, DEFAULT_LIMIT, MAX_LIMIT, STAC_VERSION, VALID_PROCESSING_LEVELS};

// ============================================================================
// Application State
// ============================================================================

/// Application state shared across handlers
#[derive(Clone)]
struct AppState {
    catalog: Arc<StacCatalog>,
    base_url: String,
}

// ============================================================================
// Error Handling
// ============================================================================

/// API error type for consistent error responses
#[derive(Debug, Serialize)]
struct ApiError {
    code: String,
    description: String,
}

impl ApiError {
    fn not_found(description: impl Into<String>) -> (StatusCode, Json<ApiError>) {
        (
            StatusCode::NOT_FOUND,
            Json(ApiError {
                code: "NotFound".into(),
                description: description.into(),
            }),
        )
    }

    fn bad_request(description: impl Into<String>) -> (StatusCode, Json<ApiError>) {
        (
            StatusCode::BAD_REQUEST,
            Json(ApiError {
                code: "BadRequest".into(),
                description: description.into(),
            }),
        )
    }
}

// ============================================================================
// Query Parameters
// ============================================================================

/// Query parameters for item search (GET)
#[derive(Debug, Deserialize, Default)]
struct SearchParams {
    /// Bounding box filter [west, south, east, north]
    bbox: Option<String>,
    /// Temporal filter (RFC 3339 interval: datetime or start/end)
    datetime: Option<String>,
    /// Filter by collection IDs (comma-separated)
    collections: Option<String>,
    /// Maximum number of results (capped at MAX_LIMIT)
    limit: Option<usize>,
    /// Result offset for pagination
    offset: Option<usize>,
    /// Filter by source provider
    source: Option<String>,
    /// Filter by processing level (1-4)
    processing_level: Option<i32>,
    /// Exclude assets from response (for lightweight listing)
    #[serde(default)]
    exclude_assets: bool,
}

/// POST body for search (same fields as GET params)
#[derive(Debug, Deserialize, Default)]
struct SearchBody {
    bbox: Option<Vec<f64>>,
    datetime: Option<String>,
    collections: Option<Vec<String>>,
    limit: Option<usize>,
    offset: Option<usize>,
    source: Option<String>,
    processing_level: Option<i32>,
    #[serde(default)]
    exclude_assets: bool,
}

/// Query parameters for asset pagination
#[derive(Debug, Deserialize, Default)]
struct AssetParams {
    /// Maximum number of assets to return (excludes 'archive')
    asset_limit: Option<usize>,
    /// Offset for asset pagination
    asset_offset: Option<usize>,
}

// ============================================================================
// Validation
// ============================================================================

/// Parsed and validated bounding box
#[derive(Debug, Clone)]
struct Bbox {
    west: f64,
    south: f64,
    east: f64,
    north: f64,
}

impl Bbox {
    /// Parse bbox from comma-separated string
    fn parse(s: &str) -> Result<Self, String> {
        let coords: Vec<f64> = s
            .split(',')
            .map(|p| p.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "bbox must contain valid numbers")?;

        if coords.len() != 4 {
            return Err(format!(
                "bbox must have exactly 4 values [west,south,east,north], got {}",
                coords.len()
            ));
        }

        let bbox = Bbox {
            west: coords[0],
            south: coords[1],
            east: coords[2],
            north: coords[3],
        };

        // Validate coordinates
        if bbox.west < -180.0 || bbox.west > 180.0 {
            return Err("bbox west must be between -180 and 180".into());
        }
        if bbox.east < -180.0 || bbox.east > 180.0 {
            return Err("bbox east must be between -180 and 180".into());
        }
        if bbox.south < -90.0 || bbox.south > 90.0 {
            return Err("bbox south must be between -90 and 90".into());
        }
        if bbox.north < -90.0 || bbox.north > 90.0 {
            return Err("bbox north must be between -90 and 90".into());
        }
        if bbox.south > bbox.north {
            return Err("bbox south must be <= north".into());
        }

        Ok(bbox)
    }

    /// Parse bbox from vec (for POST)
    fn from_vec(v: &[f64]) -> Result<Self, String> {
        if v.len() != 4 {
            return Err(format!(
                "bbox must have exactly 4 values, got {}",
                v.len()
            ));
        }
        Self::parse(&format!("{},{},{},{}", v[0], v[1], v[2], v[3]))
    }

    /// Check if this bbox intersects with an item's bbox
    fn intersects(&self, item_bbox: &[f64]) -> bool {
        if item_bbox.len() < 4 {
            return false;
        }
        let (iwest, isouth, ieast, inorth) = (item_bbox[0], item_bbox[1], item_bbox[2], item_bbox[3]);
        !(ieast < self.west || iwest > self.east || inorth < self.south || isouth > self.north)
    }
}

/// Parsed datetime interval for filtering
#[derive(Debug, Clone)]
struct DatetimeInterval {
    start: Option<DateTime<Utc>>,
    end: Option<DateTime<Utc>>,
}

impl DatetimeInterval {
    /// Parse datetime from STAC format:
    /// - Single datetime: "2025-01-01T00:00:00Z"
    /// - Interval: "2025-01-01T00:00:00Z/2025-12-31T23:59:59Z"
    /// - Open start: "../2025-12-31T23:59:59Z"
    /// - Open end: "2025-01-01T00:00:00Z/.."
    fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();

        if s.contains('/') {
            let parts: Vec<&str> = s.split('/').collect();
            if parts.len() != 2 {
                return Err("datetime interval must have exactly 2 parts separated by /".into());
            }

            let start = if parts[0] == ".." {
                None
            } else {
                Some(Self::parse_single(parts[0])?)
            };

            let end = if parts[1] == ".." {
                None
            } else {
                Some(Self::parse_single(parts[1])?)
            };

            Ok(DatetimeInterval { start, end })
        } else {
            // Single datetime - matches exact moment
            let dt = Self::parse_single(s)?;
            Ok(DatetimeInterval {
                start: Some(dt),
                end: Some(dt),
            })
        }
    }

    fn parse_single(s: &str) -> Result<DateTime<Utc>, String> {
        // Try full RFC 3339 first
        if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
            return Ok(dt.with_timezone(&Utc));
        }
        // Try date-only format
        if let Ok(dt) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
            return Ok(dt.and_hms_opt(0, 0, 0).unwrap().and_utc());
        }
        Err(format!("invalid datetime format: {}", s))
    }

    /// Check if an item's temporal extent overlaps with this interval
    fn matches_item(&self, item: &StacItem) -> bool {
        // Get item's temporal bounds
        let item_start = item
            .start_datetime()
            .or_else(|| item.datetime())
            .and_then(|s| Self::parse_single(s).ok());

        let item_end = item
            .end_datetime()
            .or_else(|| item.datetime())
            .and_then(|s| Self::parse_single(s).ok());

        // If item has no temporal info, exclude from temporal searches
        if item_start.is_none() && item_end.is_none() {
            return false;
        }

        // Check overlap
        // Filter interval: [self.start, self.end]
        // Item interval: [item_start, item_end]
        // Overlap exists if NOT (filter_end < item_start OR item_end < filter_start)

        if let Some(filter_end) = self.end {
            if let Some(item_s) = item_start {
                if filter_end < item_s {
                    return false;
                }
            }
        }

        if let Some(filter_start) = self.start {
            if let Some(item_e) = item_end {
                if item_e < filter_start {
                    return false;
                }
            }
        }

        true
    }
}

/// Validated search parameters
struct ValidatedSearch {
    bbox: Option<Bbox>,
    datetime: Option<DatetimeInterval>,
    collections: Option<Vec<String>>,
    limit: usize,
    offset: usize,
    source: Option<String>,
    processing_level: Option<i32>,
    exclude_assets: bool,
}

impl ValidatedSearch {
    fn from_params(params: SearchParams) -> Result<Self, String> {
        let bbox = params.bbox.as_ref().map(|b| Bbox::parse(b)).transpose()?;

        let datetime = params
            .datetime
            .as_ref()
            .map(|d| DatetimeInterval::parse(d))
            .transpose()?;

        let collections = params
            .collections
            .map(|c| c.split(',').map(|s| s.trim().to_string()).collect());

        let limit = params.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT);
        let offset = params.offset.unwrap_or(0);

        if let Some(level) = params.processing_level {
            if !VALID_PROCESSING_LEVELS.contains(&level) {
                return Err(format!(
                    "processing_level must be between {} and {}",
                    VALID_PROCESSING_LEVELS.start(),
                    VALID_PROCESSING_LEVELS.end()
                ));
            }
        }

        Ok(ValidatedSearch {
            bbox,
            datetime,
            collections,
            limit,
            offset,
            source: params.source,
            processing_level: params.processing_level,
            exclude_assets: params.exclude_assets,
        })
    }

    fn from_body(body: SearchBody) -> Result<Self, String> {
        let bbox = body.bbox.as_ref().map(|b| Bbox::from_vec(b)).transpose()?;

        let datetime = body
            .datetime
            .as_ref()
            .map(|d| DatetimeInterval::parse(d))
            .transpose()?;

        let limit = body.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT);
        let offset = body.offset.unwrap_or(0);

        if let Some(level) = body.processing_level {
            if !VALID_PROCESSING_LEVELS.contains(&level) {
                return Err(format!(
                    "processing_level must be between {} and {}",
                    VALID_PROCESSING_LEVELS.start(),
                    VALID_PROCESSING_LEVELS.end()
                ));
            }
        }

        Ok(ValidatedSearch {
            bbox,
            datetime,
            collections: body.collections,
            limit,
            offset,
            source: body.source,
            processing_level: body.processing_level,
            exclude_assets: body.exclude_assets,
        })
    }
}

// ============================================================================
// Conformance Classes
// ============================================================================

/// STAC API conformance classes this implementation supports
const CONFORMANCE_CLASSES: &[&str] = &[
    // Core STAC API
    "https://api.stacspec.org/v1.0.0/core",
    // Collections
    "https://api.stacspec.org/v1.0.0/collections",
    // Item Search
    "https://api.stacspec.org/v1.0.0/item-search",
    // OGC API Features
    "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core",
    "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/geojson",
    "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/oas30",
];

// ============================================================================
// Helper Functions
// ============================================================================

/// Strip assets from an item, keeping only the archive asset
fn strip_non_archive_assets(item: &StacItem) -> serde_json::Value {
    let mut json = serde_json::to_value(item).unwrap_or_default();
    if let Some(obj) = json.as_object_mut() {
        if let Some(assets) = obj.get("assets").and_then(|a| a.as_object()) {
            let archive = assets.get("archive").cloned();
            let mut minimal_assets = serde_json::Map::new();
            if let Some(arch) = archive {
                minimal_assets.insert("archive".to_string(), arch);
            }
            obj.insert(
                "assets".to_string(),
                serde_json::Value::Object(minimal_assets),
            );
        }
    }
    json
}

/// Convert item to JSON, optionally stripping assets
fn item_to_json(item: &StacItem, exclude_assets: bool) -> serde_json::Value {
    if exclude_assets {
        strip_non_archive_assets(item)
    } else {
        serde_json::to_value(item).unwrap_or_default()
    }
}

/// Filter an item based on validated search parameters
fn filter_item(item: &StacItem, params: &ValidatedSearch) -> bool {
    // Bbox filter
    if let Some(ref bbox) = params.bbox {
        match &item.bbox {
            Some(item_bbox) => {
                if !bbox.intersects(item_bbox) {
                    return false;
                }
            }
            None => return false, // Item has no bbox, exclude from bbox searches
        }
    }

    // Datetime filter
    if let Some(ref datetime) = params.datetime {
        if !datetime.matches_item(item) {
            return false;
        }
    }

    // Source filter
    if let Some(ref source) = params.source {
        let item_source = item
            .properties
            .get("blatten:source")
            .and_then(|v| v.as_str());
        if item_source != Some(source.as_str()) {
            return false;
        }
    }

    // Processing level filter
    if let Some(level) = params.processing_level {
        let item_level = item
            .properties
            .get("blatten:processing_level")
            .and_then(|v| v.as_i64());
        if item_level != Some(level as i64) {
            return false;
        }
    }

    true
}

/// Build pagination links
fn build_pagination_links(
    base_url: &str,
    endpoint: &str,
    offset: usize,
    limit: usize,
    total_matched: usize,
    is_post: bool,
) -> Vec<serde_json::Value> {
    let mut links = vec![];

    // Next link
    let next_offset = offset + limit;
    if next_offset < total_matched {
        if is_post {
            links.push(serde_json::json!({
                "rel": "next",
                "href": format!("{}{}", base_url, endpoint),
                "type": "application/geo+json",
                "method": "POST",
                "merge": true,
                "body": {
                    "offset": next_offset
                }
            }));
        } else {
            links.push(serde_json::json!({
                "rel": "next",
                "href": format!("{}{}?offset={}&limit={}", base_url, endpoint, next_offset, limit),
                "type": "application/geo+json"
            }));
        }
    }

    // Prev link
    if offset > 0 {
        let prev_offset = offset.saturating_sub(limit);
        if is_post {
            links.push(serde_json::json!({
                "rel": "prev",
                "href": format!("{}{}", base_url, endpoint),
                "type": "application/geo+json",
                "method": "POST",
                "merge": true,
                "body": {
                    "offset": prev_offset
                }
            }));
        } else {
            links.push(serde_json::json!({
                "rel": "prev",
                "href": format!("{}{}?offset={}&limit={}", base_url, endpoint, prev_offset, limit),
                "type": "application/geo+json"
            }));
        }
    }

    links
}

/// Generate ETag from content
fn generate_etag(collections: usize, items: usize) -> String {
    format!("\"v{}-c{}-i{}\"", STAC_VERSION, collections, items)
}

// ============================================================================
// Main Entry Point
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            env::var("RUST_LOG").unwrap_or_else(|_| "info,tower_http=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration from environment
    dotenvy::dotenv().ok();

    let base_url = env::var("STAC_BASE_URL").unwrap_or_else(|_| "http://localhost:3000".into());
    let catalog_dir = env::var("STAC_CATALOG_DIR").unwrap_or_else(|_| "stac".into());
    let port: u16 = env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);

    info!("Loading STAC catalog from {}", catalog_dir);

    // Load catalog
    let catalog = StacCatalog::load_from_dir(&PathBuf::from(&catalog_dir), &base_url)
        .context("Failed to load STAC catalog")?;

    info!(
        "Loaded {} collections with {} total items (STAC {})",
        catalog.collections.len(),
        catalog.items.len(),
        STAC_VERSION
    );

    let state = AppState {
        catalog: Arc::new(catalog),
        base_url: base_url.clone(),
    };

    // Build router
    let app = Router::new()
        // Landing page
        .route("/", get(landing_page))
        .route("/stac", get(landing_page))
        // Conformance
        .route("/stac/conformance", get(conformance))
        // Catalog
        .route("/stac/catalog.json", get(get_catalog))
        // Collections
        .route("/stac/collections", get(list_collections))
        .route("/stac/collections/{collection_id}", get(get_collection))
        .route(
            "/stac/collections/{collection_id}/items",
            get(get_collection_items),
        )
        .route(
            "/stac/collections/{collection_id}/items/{item_id}",
            get(get_item),
        )
        // Search
        .route("/stac/search", get(search_items_get).post(search_items_post))
        // Health check
        .route("/health", get(health_check))
        .with_state(state)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers([header::CONTENT_TYPE, header::ACCEPT]),
        );

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("Starting STAC API server on {}", addr);
    info!("Base URL: {}", base_url);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ============================================================================
// Route Handlers
// ============================================================================

/// STAC API Landing Page
async fn landing_page(State(state): State<AppState>) -> impl IntoResponse {
    let landing = serde_json::json!({
        "type": "Catalog",
        "id": "birch-glacier-collapse",
        "stac_version": STAC_VERSION,
        "title": "Birch Glacier Collapse and Landslide Dataset",
        "description": "STAC API for the 2025 Birch glacier collapse and landslide dataset at Blatten, CH-VS",
        "conformsTo": CONFORMANCE_CLASSES,
        "links": [
            {
                "rel": "self",
                "href": format!("{}/stac", state.base_url),
                "type": "application/json"
            },
            {
                "rel": "root",
                "href": format!("{}/stac", state.base_url),
                "type": "application/json"
            },
            {
                "rel": "conformance",
                "href": format!("{}/stac/conformance", state.base_url),
                "type": "application/json"
            },
            {
                "rel": "data",
                "href": format!("{}/stac/collections", state.base_url),
                "type": "application/json"
            },
            {
                "rel": "search",
                "href": format!("{}/stac/search", state.base_url),
                "type": "application/geo+json",
                "method": "GET"
            },
            {
                "rel": "search",
                "href": format!("{}/stac/search", state.base_url),
                "type": "application/geo+json",
                "method": "POST"
            }
        ]
    });

    (
        [(header::CACHE_CONTROL, "public, max-age=3600")],
        Json(landing),
    )
}

/// STAC API Conformance
async fn conformance() -> impl IntoResponse {
    (
        [(header::CACHE_CONTROL, "public, max-age=86400")],
        Json(serde_json::json!({
            "conformsTo": CONFORMANCE_CLASSES
        })),
    )
}

/// Get root catalog
async fn get_catalog(State(state): State<AppState>) -> impl IntoResponse {
    let etag = generate_etag(state.catalog.collections.len(), state.catalog.items.len());
    (
        [
            (header::CACHE_CONTROL, "public, max-age=3600".to_string()),
            (header::ETAG, etag),
        ],
        Json(state.catalog.root.clone()),
    )
}

/// List all collections
async fn list_collections(State(state): State<AppState>) -> impl IntoResponse {
    let collections: Vec<_> = state.catalog.collections.values().collect();
    let total = collections.len();

    let etag = generate_etag(state.catalog.collections.len(), state.catalog.items.len());
    (
        [
            (header::CACHE_CONTROL, "public, max-age=3600".to_string()),
            (header::ETAG, etag),
        ],
        Json(serde_json::json!({
            "collections": collections,
            "numberMatched": total,
            "numberReturned": total,
            "links": [
                {
                    "rel": "self",
                    "href": format!("{}/stac/collections", state.base_url),
                    "type": "application/json"
                },
                {
                    "rel": "root",
                    "href": format!("{}/stac", state.base_url),
                    "type": "application/json"
                }
            ]
        })),
    )
}

/// Get a specific collection
async fn get_collection(
    State(state): State<AppState>,
    Path(collection_id): Path<String>,
) -> impl IntoResponse {
    match state.catalog.collections.get(&collection_id) {
        Some(collection) => {
            let etag = generate_etag(state.catalog.collections.len(), state.catalog.items.len());
            (
                StatusCode::OK,
                [
                    (header::CACHE_CONTROL, "public, max-age=3600".to_string()),
                    (header::ETAG, etag),
                ],
                Json(serde_json::to_value(collection).unwrap_or_default()),
            )
                .into_response()
        }
        None => ApiError::not_found(format!("Collection '{}' not found", collection_id)).into_response(),
    }
}

/// Get items in a collection
async fn get_collection_items(
    State(state): State<AppState>,
    Path(collection_id): Path<String>,
    Query(params): Query<SearchParams>,
) -> impl IntoResponse {
    // Validate parameters
    let validated = match ValidatedSearch::from_params(params) {
        Ok(v) => v,
        Err(e) => return ApiError::bad_request(e).into_response(),
    };

    // Check collection exists
    if !state.catalog.collections.contains_key(&collection_id) {
        return ApiError::not_found(format!("Collection '{}' not found", collection_id)).into_response();
    }

    // Get and filter items
    let collection_items = state.catalog.get_collection_items(&collection_id);

    let filtered: Vec<_> = collection_items
        .iter()
        .filter(|item| filter_item(item, &validated))
        .collect();

    let total_matched = filtered.len();

    let paginated: Vec<serde_json::Value> = filtered
        .into_iter()
        .skip(validated.offset)
        .take(validated.limit)
        .map(|item| item_to_json(item, validated.exclude_assets))
        .collect();

    let mut links = vec![
        serde_json::json!({
            "rel": "self",
            "href": format!("{}/stac/collections/{}/items", state.base_url, collection_id),
            "type": "application/geo+json"
        }),
        serde_json::json!({
            "rel": "collection",
            "href": format!("{}/stac/collections/{}", state.base_url, collection_id),
            "type": "application/json"
        }),
        serde_json::json!({
            "rel": "root",
            "href": format!("{}/stac", state.base_url),
            "type": "application/json"
        }),
    ];

    // Add pagination links
    links.extend(build_pagination_links(
        &state.base_url,
        &format!("/stac/collections/{}/items", collection_id),
        validated.offset,
        validated.limit,
        total_matched,
        false,
    ));

    (
        [(header::CACHE_CONTROL, "public, max-age=300")],
        Json(serde_json::json!({
            "type": "FeatureCollection",
            "features": paginated,
            "numberMatched": total_matched,
            "numberReturned": paginated.len(),
            "links": links
        })),
    )
        .into_response()
}

/// Get a specific item (O(1) lookup)
async fn get_item(
    State(state): State<AppState>,
    Path((collection_id, item_id)): Path<(String, String)>,
    Query(params): Query<AssetParams>,
) -> impl IntoResponse {
    let item = state.catalog.get_item(&collection_id, &item_id);

    match item {
        Some(item) => {
            let mut json = serde_json::to_value(item).unwrap_or_default();

            // Paginate assets if limit is specified
            if let Some(limit) = params.asset_limit {
                if let Some(obj) = json.as_object_mut() {
                    if let Some(assets) = obj.get("assets").and_then(|a| a.as_object()) {
                        let offset = params.asset_offset.unwrap_or(0);

                        // Separate archive from other assets
                        let archive = assets.get("archive").cloned();
                        let other_assets: Vec<_> =
                            assets.iter().filter(|(k, _)| *k != "archive").collect();

                        let total_assets = other_assets.len();

                        // Paginate non-archive assets
                        let paginated: serde_json::Map<String, serde_json::Value> = other_assets
                            .into_iter()
                            .skip(offset)
                            .take(limit)
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();

                        let has_archive = archive.is_some();
                        let mut new_assets = serde_json::Map::new();
                        if let Some(arch) = archive {
                            new_assets.insert("archive".to_string(), arch);
                        }
                        new_assets.extend(paginated);

                        let returned_count = new_assets.len() - if has_archive { 1 } else { 0 };
                        obj.insert("assets".to_string(), serde_json::Value::Object(new_assets));
                        obj.insert(
                            "_assetsMeta".to_string(),
                            serde_json::json!({
                                "total": total_assets,
                                "offset": offset,
                                "returned": returned_count
                            }),
                        );
                    }
                }
            }

            let etag = format!("\"{}:{}\"", collection_id, item_id);
            (
                StatusCode::OK,
                [
                    (header::CACHE_CONTROL, "public, max-age=3600".to_string()),
                    (header::ETAG, etag),
                ],
                Json(json),
            )
                .into_response()
        }
        None => ApiError::not_found(format!(
            "Item '{}' not found in collection '{}'",
            item_id, collection_id
        ))
        .into_response(),
    }
}

/// Search items (GET)
async fn search_items_get(
    State(state): State<AppState>,
    Query(params): Query<SearchParams>,
) -> impl IntoResponse {
    // Validate parameters
    let validated = match ValidatedSearch::from_params(params) {
        Ok(v) => v,
        Err(e) => return ApiError::bad_request(e).into_response(),
    };

    perform_search(&state, validated, false).into_response()
}

/// Search items (POST)
async fn search_items_post(
    State(state): State<AppState>,
    Json(body): Json<SearchBody>,
) -> impl IntoResponse {
    // Validate parameters
    let validated = match ValidatedSearch::from_body(body) {
        Ok(v) => v,
        Err(e) => return ApiError::bad_request(e).into_response(),
    };

    perform_search(&state, validated, true).into_response()
}

/// Perform search with validated parameters
fn perform_search(
    state: &AppState,
    params: ValidatedSearch,
    is_post: bool,
) -> impl IntoResponse {
    // Filter by collections if specified
    let items_to_search: Vec<&StacItem> = if let Some(ref colls) = params.collections {
        state
            .catalog
            .items
            .iter()
            .filter(|item| {
                item.collection
                    .as_ref()
                    .map(|c| colls.contains(c))
                    .unwrap_or(false)
            })
            .collect()
    } else {
        state.catalog.items.iter().collect()
    };

    // Apply filters and count total matches
    let filtered: Vec<_> = items_to_search
        .into_iter()
        .filter(|item| filter_item(item, &params))
        .collect();

    let total_matched = filtered.len();

    // Paginate
    let paginated: Vec<serde_json::Value> = filtered
        .into_iter()
        .skip(params.offset)
        .take(params.limit)
        .map(|item| item_to_json(item, params.exclude_assets))
        .collect();

    let mut links = vec![
        serde_json::json!({
            "rel": "self",
            "href": format!("{}/stac/search", state.base_url),
            "type": "application/geo+json"
        }),
        serde_json::json!({
            "rel": "root",
            "href": format!("{}/stac", state.base_url),
            "type": "application/json"
        }),
    ];

    // Add pagination links
    links.extend(build_pagination_links(
        &state.base_url,
        "/stac/search",
        params.offset,
        params.limit,
        total_matched,
        is_post,
    ));

    (
        [(header::CACHE_CONTROL, "private, max-age=60")],
        Json(serde_json::json!({
            "type": "FeatureCollection",
            "features": paginated,
            "numberMatched": total_matched,
            "numberReturned": paginated.len(),
            "context": {
                "limit": params.limit,
                "matched": total_matched,
                "returned": paginated.len()
            },
            "links": links
        })),
    )
}

/// Health check endpoint
async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "stac_version": STAC_VERSION,
        "collections": state.catalog.collections.len(),
        "items": state.catalog.items.len()
    }))
}
