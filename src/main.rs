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

use catalog::{ItemExt, PropertiesDatetimeExt, StacCatalog, StacItem, DEFAULT_LIMIT, MAX_LIMIT, STAC_VERSION, VALID_PROCESSING_LEVELS};

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
    /// Filter by sensor name (exact match)
    sensor: Option<String>,
    /// Full-text search across item metadata and asset names
    q: Option<String>,
    /// Exclude assets from response (for lightweight listing)
    #[serde(default)]
    exclude_assets: bool,
    /// Fields filter (STAC Fields Extension): comma-separated, prefix `-` to exclude
    fields: Option<String>,
}

/// Deserializer that accepts either a string or array of strings for the `q` parameter
fn deserialize_q_param<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct QVisitor;

    impl<'de> de::Visitor<'de> for QVisitor {
        type Value = Option<Vec<String>>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string or array of strings")
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            let terms: Vec<String> = v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
            if terms.is_empty() { Ok(None) } else { Ok(Some(terms)) }
        }

        fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
            self.visit_str(&v)
        }

        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut terms = Vec::new();
            while let Some(s) = seq.next_element::<String>()? {
                let trimmed = s.trim().to_string();
                if !trimmed.is_empty() {
                    terms.push(trimmed);
                }
            }
            if terms.is_empty() { Ok(None) } else { Ok(Some(terms)) }
        }
    }

    deserializer.deserialize_any(QVisitor)
}

/// POST body fields filter (STAC Fields Extension)
#[derive(Debug, Deserialize, Default)]
struct FieldsBody {
    #[serde(default)]
    include: Vec<String>,
    #[serde(default)]
    exclude: Vec<String>,
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
    sensor: Option<String>,
    #[serde(default, deserialize_with = "deserialize_q_param")]
    q: Option<Vec<String>>,
    #[serde(default)]
    exclude_assets: bool,
    /// Fields filter (STAC Fields Extension)
    fields: Option<FieldsBody>,
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
        // Get item's temporal bounds using the parsed datetime values from stac crate
        let item_start = item.start_datetime().or_else(|| item.datetime());
        let item_end = item.end_datetime().or_else(|| item.datetime());

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
    sensor: Option<String>,
    q: Option<Vec<String>>,
    exclude_assets: bool,
    fields: FieldsFilter,
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

        let fields = params
            .fields
            .as_deref()
            .map(FieldsFilter::parse_get)
            .unwrap_or_default();

        Ok(ValidatedSearch {
            bbox,
            datetime,
            collections,
            limit,
            offset,
            source: params.source,
            processing_level: params.processing_level,
            sensor: params.sensor,
            q: params.q.map(|q| {
                q.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
            }).filter(|v: &Vec<String>| !v.is_empty()),
            exclude_assets: params.exclude_assets,
            fields,
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

        let fields = body
            .fields
            .as_ref()
            .map(FieldsFilter::from_body)
            .unwrap_or_default();

        Ok(ValidatedSearch {
            bbox,
            datetime,
            collections: body.collections,
            limit,
            offset,
            source: body.source,
            processing_level: body.processing_level,
            sensor: body.sensor,
            q: body.q.filter(|v| !v.is_empty()),
            exclude_assets: body.exclude_assets,
            fields,
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
    // Free-Text Search
    "https://api.stacspec.org/v1.0.0-rc.1/item-search#free-text",
    "https://api.stacspec.org/v1.0.0-rc.1/ogcapi-features#free-text",
    // Fields Extension
    "https://api.stacspec.org/v1.0.0/item-search#fields",
];

// ============================================================================
// Fields Extension (STAC Fields Extension)
// ============================================================================

/// Parsed fields filter for pruning item JSON responses
#[derive(Debug, Clone, Default)]
struct FieldsFilter {
    include: Vec<String>,
    exclude: Vec<String>,
}

/// Default top-level fields per STAC Fields Extension spec.
/// Note: `properties` is NOT a default — it's only included when explicitly requested.
const FIELDS_DEFAULTS: &[&str] = &[
    "type",
    "stac_version",
    "id",
    "geometry",
    "bbox",
    "links",
    "assets",
    "collection",
];

impl FieldsFilter {
    fn is_empty(&self) -> bool {
        self.include.is_empty() && self.exclude.is_empty()
    }

    /// Parse GET format: comma-separated, prefix `-` for exclude, `+` or bare for include
    fn parse_get(s: &str) -> Self {
        let mut include = Vec::new();
        let mut exclude = Vec::new();
        for field in s.split(',').map(|f| f.trim()).filter(|f| !f.is_empty()) {
            if let Some(stripped) = field.strip_prefix('-') {
                exclude.push(stripped.to_string());
            } else {
                let field = field.strip_prefix('+').unwrap_or(field);
                include.push(field.to_string());
            }
        }
        FieldsFilter { include, exclude }
    }

    /// Build from POST body
    fn from_body(body: &FieldsBody) -> Self {
        FieldsFilter {
            include: body.include.clone(),
            exclude: body.exclude.clone(),
        }
    }

    /// Apply fields filter to a serialized item JSON value (in-place)
    fn apply(&self, json: &mut serde_json::Value) {
        if self.is_empty() {
            return;
        }

        let obj = match json.as_object_mut() {
            Some(o) => o,
            None => return,
        };

        // When includes are specified, start from defaults + explicitly included fields
        if !self.include.is_empty() {
            let mut keep_top: std::collections::HashSet<&str> =
                FIELDS_DEFAULTS.iter().copied().collect();

            let mut prop_subfields: Vec<&str> = Vec::new();
            let mut include_all_properties = false;

            for field in &self.include {
                if field == "properties" {
                    include_all_properties = true;
                } else if let Some(subfield) = field.strip_prefix("properties.") {
                    prop_subfields.push(subfield);
                } else {
                    keep_top.insert(field.as_str());
                }
            }

            // Remove top-level keys not in keep set (but preserve "properties" if sub-fields requested)
            let has_prop_includes = include_all_properties || !prop_subfields.is_empty();
            let keys_to_remove: Vec<String> = obj
                .keys()
                .filter(|k| {
                    if *k == "properties" && has_prop_includes {
                        return false;
                    }
                    !keep_top.contains(k.as_str())
                })
                .cloned()
                .collect();
            for key in keys_to_remove {
                obj.remove(&key);
            }

            // Filter properties sub-fields (unless "properties" was included as a whole)
            if !include_all_properties && !prop_subfields.is_empty() {
                if let Some(props) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) {
                    let remove: Vec<String> = props
                        .keys()
                        .filter(|k| !prop_subfields.contains(&k.as_str()))
                        .cloned()
                        .collect();
                    for key in remove {
                        props.remove(&key);
                    }
                }
            }
        }

        // Apply excludes
        for field in &self.exclude {
            if let Some(subfield) = field.strip_prefix("properties.") {
                if let Some(props) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) {
                    props.remove(subfield);
                }
            } else {
                obj.remove(field);
            }
        }
    }
}

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

/// Convert item to JSON, optionally stripping assets and applying fields filter
fn item_to_json(item: &StacItem, exclude_assets: bool, fields: &FieldsFilter) -> serde_json::Value {
    let mut json = if exclude_assets {
        strip_non_archive_assets(item)
    } else {
        serde_json::to_value(item).unwrap_or_default()
    };
    fields.apply(&mut json);
    json
}

/// Filter an item based on validated search parameters
fn filter_item(item: &StacItem, params: &ValidatedSearch) -> bool {
    // Bbox filter
    if let Some(ref bbox) = params.bbox {
        match item.bbox_array() {
            Some(item_bbox) => {
                if !bbox.intersects(&item_bbox) {
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
            .get_property("blatten:source")
            .and_then(|v| v.as_str());
        if item_source != Some(source.as_str()) {
            return false;
        }
    }

    // Processing level filter
    if let Some(level) = params.processing_level {
        let item_level = item
            .get_property("blatten:processing_level")
            .and_then(|v| v.as_i64());
        if item_level != Some(level as i64) {
            return false;
        }
    }

    // Sensor filter (exact match)
    if let Some(ref sensor) = params.sensor {
        let item_sensor = item
            .get_property("blatten:sensor")
            .and_then(|v| v.as_str());
        if item_sensor != Some(sensor.as_str()) {
            return false;
        }
    }

    // Free-text search (STAC Free-Text Search Extension)
    // Multiple terms use OR semantics: match if ANY term matches ANY field
    if let Some(ref terms) = params.q {
        let term_matches = |query: &str| -> bool {
            let check_str = |field: &str| -> bool {
                item.get_property(field)
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_lowercase().contains(query))
                    .unwrap_or(false)
            };

            let id_matches = item.id.to_lowercase().contains(query);
            let prop_matches = check_str("title")
                || check_str("description")
                || check_str("blatten:sensor")
                || check_str("blatten:source")
                || check_str("blatten:code")
                || check_str("blatten:format")
                || check_str("blatten:dataset")
                || check_str("blatten:product_type")
                || check_str("blatten:bundle");

            if id_matches || prop_matches {
                return true;
            }

            // Search asset keys, titles, and hrefs
            item.assets
                .iter()
                .any(|(key, asset)| {
                    key.to_lowercase().contains(query)
                        || asset.title.as_ref().map(|t| t.to_lowercase().contains(query)).unwrap_or(false)
                        || asset.href.to_lowercase().contains(query)
                })
        };

        let any_term_matches = terms.iter().any(|term| term_matches(&term.to_lowercase()));
        if !any_term_matches {
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
    let s3_base_url = env::var("S3_BASE_URL").unwrap_or_else(|_| "/s3".into());
    let catalog_dir = env::var("STAC_CATALOG_DIR").unwrap_or_else(|_| "stac".into());
    let port: u16 = env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);

    info!("Loading STAC catalog from {}", catalog_dir);
    info!("S3 base URL: {}", s3_base_url);

    // Load catalog
    let catalog = StacCatalog::load_from_dir(&PathBuf::from(&catalog_dir), &base_url, &s3_base_url)
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
        .route("/stac/collections/:collection_id", get(get_collection))
        .route(
            "/stac/collections/:collection_id/items",
            get(get_collection_items),
        )
        .route(
            "/stac/collections/:collection_id/items/:item_id",
            get(get_item),
        )
        // Search
        .route("/stac/search", get(search_items_get).post(search_items_post))
        // KML generation (path-based only — no query params to avoid injection/encoding issues).
        // The last segment is a param (:_kml) rather than a literal because SwissTopo
        // appends hash params to the URL path (e.g. "kml&bgLayer=...") with no "?" separator.
        .route("/stac/collections/:collection_id/:_kml", get(generate_kml_collection_path))
        .route("/stac/collections/:collection_id/items/:item_id/:_kml", get(generate_kml_item_path))
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
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shutdown complete");
    Ok(())
}

async fn shutdown_signal() {
    use tokio::signal;

    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("Received Ctrl+C, shutting down..."),
        _ = terminate => info!("Received SIGTERM, shutting down..."),
    }
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
        "description": "STAC API for the dataset collected during the 2025 Birch glacier collapse and landslide at Blatten, CH-VS. Licensed under Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). By using this API you confirm that you have read the Dataset Overview and Detailed Report. This data is provided \"as is\" without warranty of any kind, express or implied.",
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
            },
            {
                "rel": "describedby",
                "href": format!("{}/s3/docs/dataset_overview.csv", state.base_url),
                "type": "text/csv",
                "title": "Dataset Overview"
            },
            {
                "rel": "describedby",
                "href": format!("{}/s3/docs/detailed_report.pdf", state.base_url),
                "type": "application/pdf",
                "title": "Detailed Report"
            },
            {
                "rel": "license",
                "href": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                "title": "Attribution-NonCommercial-ShareAlike 4.0 International"
            },
            {
                "rel": "about",
                "href": "https://blatten-data.epfl.ch/",
                "type": "text/html",
                "title": "Blatten Data website"
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
        .map(|item| item_to_json(item, validated.exclude_assets, &validated.fields))
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
        .map(|item| item_to_json(item, params.exclude_assets, &params.fields))
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

// ============================================================================
// KML Generation Endpoint
// ============================================================================

/// Path-based KML for a single item.
/// The `:_kml` param absorbs trailing junk SwissTopo appends (e.g. "kml&bgLayer=...").
async fn generate_kml_item_path(
    State(state): State<AppState>,
    Path((collection_id, item_id, _kml)): Path<(String, String, String)>,
) -> impl IntoResponse {
    if !_kml.starts_with("kml") {
        return (
            StatusCode::NOT_FOUND,
            [(header::CONTENT_TYPE, "text/plain")],
            "Not found".to_string(),
        );
    }
    match state.catalog.get_item(&collection_id, &item_id) {
        Some(item) => {
            let name = item_display_name(item);
            match item_to_kml_placemark(&name, item) {
                Some(placemark) => (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/vnd.google-earth.kml+xml")],
                    wrap_kml_document(&name, &placemark),
                ),
                None => (
                    StatusCode::BAD_REQUEST,
                    [(header::CONTENT_TYPE, "text/plain")],
                    "Item has no geometry or bbox".to_string(),
                ),
            }
        }
        None => (
            StatusCode::NOT_FOUND,
            [(header::CONTENT_TYPE, "text/plain")],
            format!("Item not found: {}/{}", collection_id, item_id),
        ),
    }
}

/// Path-based KML for all items in a collection.
/// The `:_kml` param absorbs trailing junk SwissTopo appends (e.g. "kml&bgLayer=...").
async fn generate_kml_collection_path(
    State(state): State<AppState>,
    Path((collection_id, _kml)): Path<(String, String)>,
) -> impl IntoResponse {
    if !_kml.starts_with("kml") {
        return (
            StatusCode::NOT_FOUND,
            [(header::CONTENT_TYPE, "text/plain")],
            "Not found".to_string(),
        );
    }
    let items = state.catalog.get_collection_items(&collection_id);
    if items.is_empty() {
        return (
            StatusCode::NOT_FOUND,
            [(header::CONTENT_TYPE, "text/plain")],
            format!("Collection not found or empty: {}", collection_id),
        );
    }

    let placemarks: String = items
        .iter()
        .filter_map(|item| {
            let name = item_display_name(item);
            item_to_kml_placemark(&name, item)
        })
        .collect::<Vec<_>>()
        .join("\n");

    if placemarks.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            [(header::CONTENT_TYPE, "text/plain")],
            format!("No items with geometry in collection: {}", collection_id),
        );
    }

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/vnd.google-earth.kml+xml")],
        wrap_kml_document(&collection_id, &placemarks),
    )
}

/// Get display name for an item (title or ID)
fn item_display_name(item: &stac::Item) -> String {
    item.properties
        .additional_fields
        .get("title")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| item.id.clone())
}

/// Convert a STAC item to a KML `<Placemark>` fragment (no document wrapper).
/// Returns None if the item has no geometry or bbox.
fn item_to_kml_placemark(name: &str, item: &stac::Item) -> Option<String> {
    if let Some(ref geom) = item.geometry {
        geojson_geometry_to_kml_placemark(name, geom)
    } else if let Some(bbox) = item.bbox_array() {
        let (min_lon, min_lat, max_lon, max_lat) = (bbox[0], bbox[1], bbox[2], bbox[3]);
        Some(kml_polygon_placemark(
            name,
            &[
                (min_lon, min_lat),
                (max_lon, min_lat),
                (max_lon, max_lat),
                (min_lon, max_lat),
                (min_lon, min_lat),
            ],
        ))
    } else {
        None
    }
}

/// Wrap placemark fragment(s) in a full KML document
fn wrap_kml_document(name: &str, placemarks: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    <Style id="marker">
      <IconStyle>
        <color>ff0000ff</color>
        <scale>1.2</scale>
        <Icon><href>http://maps.google.com/mapfiles/kml/pushpin/red-pushpin.png</href></Icon>
      </IconStyle>
    </Style>
    <Style id="polygon">
      <LineStyle>
        <color>ff0000ff</color>
        <width>2</width>
      </LineStyle>
      <PolyStyle>
        <color>4d0000ff</color>
      </PolyStyle>
    </Style>
{placemarks}
  </Document>
</kml>"#
    )
}

/// Convert geojson::Geometry to a KML `<Placemark>` fragment.
/// Returns None for unsupported or invalid geometry.
fn geojson_geometry_to_kml_placemark(name: &str, geom: &geojson::Geometry) -> Option<String> {
    use geojson::Value;

    match &geom.value {
        Value::Point(coords) => {
            if coords.len() >= 2 {
                Some(kml_point_placemark(name, coords[0], coords[1]))
            } else {
                None
            }
        }
        Value::Polygon(rings) => {
            if let Some(outer_ring) = rings.first() {
                let coords: Vec<(f64, f64)> = outer_ring
                    .iter()
                    .filter_map(|c| {
                        if c.len() >= 2 {
                            Some((c[0], c[1]))
                        } else {
                            None
                        }
                    })
                    .collect();
                Some(kml_polygon_placemark(name, &coords))
            } else {
                None
            }
        }
        Value::MultiPolygon(polygons) => {
            if let Some(rings) = polygons.first() {
                if let Some(outer_ring) = rings.first() {
                    let coords: Vec<(f64, f64)> = outer_ring
                        .iter()
                        .filter_map(|c| {
                            if c.len() >= 2 {
                                Some((c[0], c[1]))
                            } else {
                                None
                            }
                        })
                        .collect();
                    Some(kml_polygon_placemark(name, &coords))
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Generate a KML `<Placemark>` fragment for a point
fn kml_point_placemark(name: &str, lon: f64, lat: f64) -> String {
    format!(
        r#"    <Placemark>
      <name>{name}</name>
      <styleUrl>#marker</styleUrl>
      <Point>
        <coordinates>{lon},{lat},0</coordinates>
      </Point>
    </Placemark>"#
    )
}

/// Generate a KML `<Placemark>` fragment for a polygon
fn kml_polygon_placemark(name: &str, coords: &[(f64, f64)]) -> String {
    let coord_str: String = coords
        .iter()
        .map(|(lon, lat)| format!("{},{},0", lon, lat))
        .collect::<Vec<_>>()
        .join(" ");

    format!(
        r#"    <Placemark>
      <name>{name}</name>
      <styleUrl>#polygon</styleUrl>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>{coord_str}</coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>"#
    )
}
