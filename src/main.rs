//! Blatten Data API - STAC API Server
//!
//! A STAC 1.1.0 compliant API server for the Birch Glacier Collapse Dataset.
//! Uses an in-memory backend loaded from JSON catalog files.

use anyhow::{Context, Result};
use axum::{
    extract::{Path, Query, RawQuery, State},
    http::{header, HeaderMap, Method, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::get,
    Router,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{env, net::SocketAddr, path::PathBuf, sync::Arc};
use strsim::levenshtein;
use tower_http::compression::CompressionLayer;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod asset_routes;
mod catalog;
mod filter;
mod stats;
mod stats_routes;
mod stats_store;

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
    /// CQL2 filter (STAC Filter Extension)
    filter: Option<String>,
    /// Language of `filter`: cql2-text (the default) or cql2-json
    #[serde(rename = "filter-lang")]
    filter_lang: Option<String>,
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

/// Parse q string into OR-groups of AND-words.
/// Commas separate OR groups; whitespace separates AND words within each group.
fn parse_q_terms(raw: &str) -> Vec<Vec<String>> {
    raw.split(',')
        .map(|group| {
            group
                .split_whitespace()
                .map(|w| w.to_lowercase())
                .collect::<Vec<_>>()
        })
        .filter(|group| !group.is_empty())
        .collect()
}

/// Check if a query word matches any word in the searchable text.
/// First tries substring match (fast path). Falls back to edit-distance
/// with conservative thresholds to catch typos without false positives.
fn fuzzy_word_match(query_word: &str, searchable: &str) -> bool {
    // Fast path: substring match
    if searchable.contains(query_word) {
        return true;
    }
    // Skip fuzzy for very short words (too many false positives)
    if query_word.len() < 4 {
        return false;
    }
    let max_dist = if query_word.len() >= 7 { 2 } else { 1 };
    // Check edit distance against individual words in the searchable text
    searchable.split_whitespace().any(|word| {
        levenshtein(query_word, word) <= max_dist
    })
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
    /// CQL2 filter (STAC Filter Extension)
    filter: Option<serde_json::Value>,
    /// Language of `filter`: cql2-json (the default) or cql2-text
    #[serde(rename = "filter-lang")]
    filter_lang: Option<String>,
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
    q: Option<Vec<Vec<String>>>,
    exclude_assets: bool,
    fields: FieldsFilter,
    filter: Option<filter::Filter>,
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

        let filter = params
            .filter
            .as_deref()
            .map(|f| filter::Filter::from_get(f, params.filter_lang.as_deref()))
            .transpose()?;

        Ok(ValidatedSearch {
            bbox,
            datetime,
            collections,
            limit,
            offset,
            source: params.source,
            processing_level: params.processing_level,
            sensor: params.sensor,
            q: params.q.map(|q| parse_q_terms(&q)).filter(|v| !v.is_empty()),
            exclude_assets: params.exclude_assets,
            fields,
            filter,
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

        let filter = body
            .filter
            .map(|f| filter::Filter::from_post(f, body.filter_lang.as_deref()))
            .transpose()?;

        Ok(ValidatedSearch {
            bbox,
            datetime,
            collections: body.collections,
            limit,
            offset,
            source: body.source,
            processing_level: body.processing_level,
            sensor: body.sensor,
            q: body.q.map(|terms| {
                terms.iter()
                    .map(|term| term.split_whitespace().map(|w| w.to_lowercase()).collect::<Vec<_>>())
                    .filter(|group| !group.is_empty())
                    .collect::<Vec<Vec<String>>>()
            }).filter(|v| !v.is_empty()),
            exclude_assets: body.exclude_assets,
            fields,
            filter,
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
    // Filter Extension, Basic CQL2 in text and JSON
    "https://api.stacspec.org/v1.0.0-rc.2/item-search#filter",
    "https://api.stacspec.org/v1.0.0-rc.2/ogcapi-features/extensions/filter",
    "http://www.opengis.net/spec/ogcapi-features-3/1.0/conf/filter",
    "http://www.opengis.net/spec/ogcapi-features-3/1.0/conf/features-filter",
    "http://www.opengis.net/spec/cql2/1.0/conf/basic-cql2",
    "http://www.opengis.net/spec/cql2/1.0/conf/cql2-text",
    "http://www.opengis.net/spec/cql2/1.0/conf/cql2-json",
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

/// The order of asset keys: by text, with each run of digits compared as a number, so
/// `dem-2` comes before `dem-10`. Keys equal that way, such as `dem-02` and `dem-2`, fall back to
/// their plain text order, so the order is total.
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    /// The next run of `s` and the rest: all digits or no digits, and whether it is digits
    fn next_run(s: &str) -> (&str, bool, &str) {
        let digits = s.starts_with(|c: char| c.is_ascii_digit());
        let end = s.find(|c: char| c.is_ascii_digit() != digits).unwrap_or(s.len());
        (&s[..end], digits, &s[end..])
    }
    let (mut x, mut y) = (a, b);
    while !x.is_empty() && !y.is_empty() {
        let (x_run, x_digits, x_rest) = next_run(x);
        let (y_run, y_digits, y_rest) = next_run(y);
        let order = if x_digits && y_digits {
            let (x_run, y_run) = (x_run.trim_start_matches('0'), y_run.trim_start_matches('0'));
            x_run.len().cmp(&y_run.len()).then_with(|| x_run.cmp(y_run))
        } else {
            x_run.cmp(y_run)
        };
        if order != Ordering::Equal {
            return order;
        }
        (x, y) = (x_rest, y_rest);
    }
    x.len().cmp(&y.len()).then_with(|| a.cmp(b))
}

/// An item as JSON, every response's one way to turn an item into JSON. Its assets are held in a
/// map whose order differs from process to process, so they are put in `natural_cmp` order here,
/// which makes asset pages and bodies the same from every process serving the same catalog.
fn item_value(item: &StacItem) -> serde_json::Value {
    let mut json = serde_json::to_value(item).unwrap_or_default();
    if let Some(assets) = json.get_mut("assets").and_then(|a| a.as_object_mut()) {
        let mut entries: Vec<_> = std::mem::take(assets).into_iter().collect();
        entries.sort_by(|(a, _), (b, _)| natural_cmp(a, b));
        assets.extend(entries);
    }
    json
}

/// Keep in an item's `assets` its archive and the `limit` other assets from `offset` on, and
/// record the page in `_assetsMeta`
fn page_assets(json: &mut serde_json::Value, limit: usize, offset: usize) {
    let Some(obj) = json.as_object_mut() else { return };
    let Some(assets) = obj.get("assets").and_then(|a| a.as_object()) else { return };
    let archive = assets.get("archive").cloned();
    let others: Vec<_> = assets.iter().filter(|(k, _)| *k != "archive").collect();
    let total = others.len();
    let mut page = serde_json::Map::new();
    if let Some(archive) = archive {
        page.insert("archive".to_string(), archive);
    }
    page.extend(others.into_iter().skip(offset).take(limit).map(|(k, v)| (k.clone(), v.clone())));
    let returned = page.len() - usize::from(page.contains_key("archive"));
    obj.insert("assets".to_string(), serde_json::Value::Object(page));
    obj.insert(
        "_assetsMeta".to_string(),
        serde_json::json!({"total": total, "offset": offset, "returned": returned}),
    );
}

/// Strip assets from an item, keeping only the archive asset
fn strip_non_archive_assets(item: &StacItem) -> serde_json::Value {
    let mut json = item_value(item);
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
        item_value(item)
    };
    fields.apply(&mut json);
    json
}

/// The items of a page, each turned into JSON only as it is written out.
///
/// A whole page held as one JSON tree takes about ten times its serialized size, which is
/// hundreds of megabytes for a 300-item search; written item by item, a response holds its
/// bytes and one item's tree at a time.
struct Features<'a> {
    items: &'a [&'a StacItem],
    exclude_assets: bool,
    fields: &'a FieldsFilter,
}

impl Serialize for Features<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(
            self.items
                .iter()
                .map(|item| item_to_json(item, self.exclude_assets, self.fields)),
        )
    }
}

/// An item search or collection items response, its members in the order clients have seen.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ItemPage<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    features: Features<'a>,
    number_matched: usize,
    number_returned: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<serde_json::Value>,
    links: Vec<serde_json::Value>,
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

    // CQL2 filter (STAC Filter Extension)
    if let Some(ref filter) = params.filter {
        if !filter.matches(item) {
            return false;
        }
    }

    // Free-text search (STAC Free-Text Search Extension)
    // Commas = OR across groups, spaces = AND within each group
    // Substring matching first, edit-distance fallback for typos
    if let Some(ref groups) = params.q {
        // Collect all searchable text into one lowercased string
        let mut searchable = String::new();
        searchable.push_str(&item.id.to_lowercase());
        searchable.push(' ');
        for field in &[
            "title",
            "description",
            "blatten:sensor",
            "blatten:source",
            "blatten:code",
            "blatten:format",
            "blatten:dataset",
            "blatten:product_type",
            "blatten:bundle",
        ] {
            if let Some(val) = item.get_property(field).and_then(|v| v.as_str()) {
                searchable.push_str(&val.to_lowercase());
                searchable.push(' ');
            }
        }
        for (key, asset) in &item.assets {
            searchable.push_str(&key.to_lowercase());
            searchable.push(' ');
            if let Some(ref title) = asset.title {
                searchable.push_str(&title.to_lowercase());
                searchable.push(' ');
            }
            searchable.push_str(&asset.href.to_lowercase());
            searchable.push(' ');
        }

        // OR across groups, AND within each group
        let matches = groups
            .iter()
            .any(|words| words.iter().all(|w| fuzzy_word_match(w, &searchable)));
        if !matches {
            return false;
        }
    }

    true
}

/// How a page was asked for, which decides the shape of its next and prev links
#[derive(Clone, Copy)]
enum PageMethod<'a> {
    /// A GET with its query string as received
    Get { query: &'a str },
    /// A POST, whose links merge the new offset into the request body
    Post,
}

/// Build the next and prev links of a page.
///
/// A GET link repeats the received query with every parameter kept in its place,
/// `offset` and `limit` replaced by the new offset and the effective limit, so that
/// following it returns the next page of the same query.
fn pagination_links(
    base_url: &str,
    endpoint: &str,
    method: PageMethod,
    offset: usize,
    limit: usize,
    total_matched: usize,
) -> Vec<serde_json::Value> {
    let link = |rel: &str, page_offset: usize| match method {
        PageMethod::Post => serde_json::json!({
            "rel": rel,
            "href": format!("{}{}", base_url, endpoint),
            "type": "application/geo+json",
            "method": "POST",
            "merge": true,
            "body": {
                "offset": page_offset
            }
        }),
        PageMethod::Get { query } => {
            let mut serializer = url::form_urlencoded::Serializer::new(String::new());
            for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
                if key != "offset" && key != "limit" {
                    serializer.append_pair(&key, &value);
                }
            }
            serializer
                .append_pair("offset", &page_offset.to_string())
                .append_pair("limit", &limit.to_string());
            serde_json::json!({
                "rel": rel,
                "href": format!("{}{}?{}", base_url, endpoint, serializer.finish()),
                "type": "application/geo+json"
            })
        }
    };

    let mut links = vec![];
    let next_offset = offset + limit;
    if next_offset < total_matched {
        links.push(link("next", next_offset));
    }
    if offset > 0 {
        links.push(link("prev", offset.saturating_sub(limit)));
    }
    links
}

/// The ETag of a response built from `catalog` and nothing else but `parts`: the path ids and
/// query parameters the body depends on. Weak, because the JSON is the same answer whether it is
/// sent compressed or not, and its member order is not fixed.
fn stac_etag(catalog: &StacCatalog, parts: &[&str]) -> String {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    parts.hash(&mut hasher);
    format!("W/\"{}-{:016x}\"", catalog.version, hasher.finish())
}

/// Whether an `If-None-Match` request header matches `etag`, by weak comparison: `*`, or any
/// listed tag whose opaque part equals the one of `etag`.
fn if_none_match(headers: &HeaderMap, etag: &str) -> bool {
    let opaque = |tag: &str| tag.trim().trim_start_matches("W/").to_string();
    let ours = opaque(etag);
    headers
        .get_all(header::IF_NONE_MATCH)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .any(|tag| tag.trim() == "*" || opaque(tag) == ours)
}

/// A 304 with no body when the request already holds `etag`, otherwise the body `build` makes.
/// Both carry the ETag and `cache_control`.
fn not_modified_or<R: IntoResponse>(
    request: &HeaderMap,
    etag: String,
    cache_control: &'static str,
    build: impl FnOnce() -> R,
) -> Response {
    let not_modified = if_none_match(request, &etag);
    let headers = [(header::CACHE_CONTROL, cache_control.to_string()), (header::ETAG, etag)];
    if not_modified {
        (StatusCode::NOT_MODIFIED, headers).into_response()
    } else {
        (headers, build()).into_response()
    }
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

    let files = file_index(&catalog);
    let stat_items = stat_items(&catalog.items, &files);

    let state = AppState {
        catalog: Arc::new(catalog),
        base_url: base_url.clone(),
    };

    let asset_limits = asset_routes::Limits::from_env(|name| env::var(name).ok());
    info!(
        "Asset-bearing requests in flight at once: {}, each holding its slot at most {}s, waiting for one at most {}s",
        asset_limits.concurrency,
        asset_limits.deadline.as_secs(),
        asset_limits.wait.as_secs()
    );

    let app = stac_router(state, asset_limits);

    // Download statistics sit outside the CORS layer above and are registered only when configured.
    let app = stats_routes::mount(app, |name| env::var(name).ok(), stat_items, files);

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

/// Every STAC route, with its CORS layer, its JSON compressed as the client's `Accept-Encoding` asks
fn stac_router(state: AppState, asset_limits: asset_routes::Limits) -> Router {
    Router::new()
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
        // Queryables (STAC Filter Extension)
        .route("/stac/queryables", get(get_queryables))
        .route("/stac/collections/:collection_id/queryables", get(get_collection_queryables))
        // Search, collection items and item: asset-bearing responses, behind one concurrency limit
        .merge(asset_routes::router(
            get(search_items_get).post(search_items_post),
            get(get_collection_items),
            get(get_item),
            asset_limits,
        ))
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
        )
        .layer(CompressionLayer::new().gzip(true).br(true))
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
        "id": "nesthorn-birchgletscher-process-cascade",
        "stac_version": STAC_VERSION,
        "title": "Blatten4Science - Observational Research Data of the 2025 Nesthorn – Birchgletscher Process Cascade",
        "description": "STAC API for the data collected during the Nesthorn – Birchgletscher process cascade at Blatten, CH-VS. Licensed under Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). By using this API you confirm that you have read the Dataset Overview and Technical Documentation. This data is provided \"as is\" without warranty of any kind, express or implied.",
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
                "rel": filter::QUERYABLES_REL,
                "href": format!("{}/stac/queryables", state.base_url),
                "type": "application/schema+json"
            },
            {
                "rel": "describedby",
                "href": format!("{}/s3/docs/dataset_overview.csv", state.base_url),
                "type": "text/csv",
                "title": "Dataset Overview"
            },
            {
                "rel": "describedby",
                "href": format!("{}/s3/docs/A50039_Blatten4Science_TechnicalDocumentation.pdf", state.base_url),
                "type": "application/pdf",
                "title": "Technical Documentation"
            },
            {
                "rel": "describedby",
                "href": format!("{}/s3/docs/A50039_Supplement1_GassnerJ_GBIR_Report.pdf", state.base_url),
                "type": "application/pdf",
                "title": "Supplement 1 - GBIR Report"
            },
            {
                "rel": "cite-as",
                "href": "https://doi.org/10.5075/epfl.20.500.14299/262163",
                "title": "Citation and permanent link"
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
                "title": "Blatten4Science website"
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
async fn get_catalog(State(state): State<AppState>, headers: HeaderMap) -> impl IntoResponse {
    let etag = stac_etag(&state.catalog, &[]);
    not_modified_or(&headers, etag, "public, max-age=3600", || Json(state.catalog.root.clone()))
}

/// List all collections
async fn list_collections(State(state): State<AppState>, headers: HeaderMap) -> impl IntoResponse {
    let etag = stac_etag(&state.catalog, &[]);
    not_modified_or(&headers, etag, "public, max-age=3600", || {
        let collections: Vec<_> = state.catalog.collections.values().collect();
        let total = collections.len();
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
        }))
    })
}

/// Get a specific collection
async fn get_collection(
    State(state): State<AppState>,
    Path(collection_id): Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    match state.catalog.collections.get(&collection_id) {
        Some(collection) => {
            let etag = stac_etag(&state.catalog, &[&collection_id]);
            not_modified_or(&headers, etag, "public, max-age=3600", || {
                Json(serde_json::to_value(collection).unwrap_or_default())
            })
        }
        None => ApiError::not_found(format!("Collection '{}' not found", collection_id)).into_response(),
    }
}

/// The queryables a filter may name, as a JSON Schema
fn queryables_response(id: String) -> Response {
    (
        [
            (header::CONTENT_TYPE, "application/schema+json"),
            (header::CACHE_CONTROL, "public, max-age=86400"),
        ],
        filter::queryables_schema(&id).to_string(),
    )
        .into_response()
}

/// Queryables of the whole catalog
async fn get_queryables(State(state): State<AppState>) -> Response {
    queryables_response(format!("{}/stac/queryables", state.base_url))
}

/// Queryables of one collection: the same list as the whole catalog's
async fn get_collection_queryables(
    State(state): State<AppState>,
    Path(collection_id): Path<String>,
) -> Response {
    if !state.catalog.collections.contains_key(&collection_id) {
        return ApiError::not_found(format!("Collection '{}' not found", collection_id)).into_response();
    }
    queryables_response(format!("{}/stac/collections/{}/queryables", state.base_url, collection_id))
}

/// Get items in a collection
async fn get_collection_items(
    State(state): State<AppState>,
    Path(collection_id): Path<String>,
    Query(params): Query<SearchParams>,
    RawQuery(query): RawQuery,
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

    let paginated: Vec<&StacItem> = filtered
        .into_iter()
        .skip(validated.offset)
        .take(validated.limit)
        .copied()
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
    links.extend(pagination_links(
        &state.base_url,
        &format!("/stac/collections/{}/items", collection_id),
        PageMethod::Get { query: query.as_deref().unwrap_or_default() },
        validated.offset,
        validated.limit,
        total_matched,
    ));

    (
        [(header::CACHE_CONTROL, "public, max-age=300")],
        Json(ItemPage {
            kind: "FeatureCollection",
            features: Features {
                items: &paginated,
                exclude_assets: validated.exclude_assets,
                fields: &validated.fields,
            },
            number_matched: total_matched,
            number_returned: paginated.len(),
            context: None,
            links,
        }),
    )
        .into_response()
}

/// Get a specific item (O(1) lookup)
async fn get_item(
    State(state): State<AppState>,
    Path((collection_id, item_id)): Path<(String, String)>,
    Query(params): Query<AssetParams>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let item = state.catalog.get_item(&collection_id, &item_id);

    match item {
        Some(item) => {
            // The asset page is the only part of the body taken from the query.
            let page = params
                .asset_limit
                .map(|limit| format!("{}+{}", limit, params.asset_offset.unwrap_or(0)))
                .unwrap_or_default();
            let etag = stac_etag(&state.catalog, &[&collection_id, &item_id, &page]);
            not_modified_or(&headers, etag, "public, max-age=3600", || {
                let mut json = item_value(item);
                if let Some(limit) = params.asset_limit {
                    page_assets(&mut json, limit, params.asset_offset.unwrap_or(0));
                }
                Json(json)
            })
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
    RawQuery(query): RawQuery,
) -> impl IntoResponse {
    // Validate parameters
    let validated = match ValidatedSearch::from_params(params) {
        Ok(v) => v,
        Err(e) => return ApiError::bad_request(e).into_response(),
    };

    perform_search(&state, validated, PageMethod::Get { query: query.as_deref().unwrap_or_default() })
        .into_response()
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

    perform_search(&state, validated, PageMethod::Post).into_response()
}

/// Perform search with validated parameters
fn perform_search(
    state: &AppState,
    params: ValidatedSearch,
    method: PageMethod,
) -> Response {
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
    let paginated: Vec<&StacItem> = filtered
        .into_iter()
        .skip(params.offset)
        .take(params.limit)
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
    links.extend(pagination_links(
        &state.base_url,
        "/stac/search",
        method,
        params.offset,
        params.limit,
        total_matched,
    ));

    (
        [(header::CACHE_CONTROL, "private, max-age=60")],
        Json(ItemPage {
            kind: "FeatureCollection",
            features: Features {
                items: &paginated,
                exclude_assets: params.exclude_assets,
                fields: &params.fields,
            },
            number_matched: total_matched,
            number_returned: paginated.len(),
            context: Some(serde_json::json!({
                "limit": params.limit,
                "matched": total_matched,
                "returned": paginated.len()
            })),
            links,
        }),
    )
        .into_response()
}

/// The file index of `catalog`: every item asset and every root catalog link (the documents).
fn file_index(catalog: &StacCatalog) -> stats::FileIndex {
    let assets = catalog
        .items
        .iter()
        .flat_map(|i| i.assets.values())
        .map(|a| (&a.href, &a.r#type, &a.additional_fields));
    let links = catalog.root.links.iter().map(|l| (&l.href, &l.r#type, &l.additional_fields));
    let mut index = stats::FileIndex::default();
    for (href, media_type, fields) in assets.chain(links) {
        let Some(path) = stats::proxy_path(href.as_ref()) else { continue };
        if let Some(t) = media_type {
            index.media_types.insert(path.clone(), t.clone());
        }
        if let Some(size) = fields.get("file:size").and_then(|v| v.as_u64()) {
            index.sizes.insert(path, size);
        }
    }
    index
}

/// The items the download statistics list: each series-level item (one without
/// `blatten:series`), its files counted with those of its days. Downloads are keyed by the
/// item code in the file path, which a series and its days share.
fn stat_items(items: &[StacItem], files: &stats::FileIndex) -> Vec<stats::ItemInfo> {
    let count = |i: &StacItem, into: &mut std::collections::BTreeMap<String, usize>| {
        for (key, asset) in &i.assets {
            if key == "archive" || key == "manifest" {
                continue;
            }
            let path = stats::proxy_path(&asset.href).unwrap_or_else(|| asset.href.clone());
            *into.entry(files.format(&path)).or_default() += 1;
        }
    };
    let series_of = |i: &StacItem| {
        i.properties
            .additional_fields
            .get("blatten:series")
            .and_then(|v| v.as_str())
            .map(str::to_string)
    };
    let mut day_files: std::collections::HashMap<String, std::collections::BTreeMap<String, usize>> = std::collections::HashMap::new();
    for i in items {
        if let Some(series) = series_of(i) {
            count(i, day_files.entry(series).or_default());
        }
    }
    items
        .iter()
        .filter(|i| series_of(i).is_none())
        .filter_map(|i| {
            let text = |key: &str| {
                i.get_property(key)
                    .map(|v| match v {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .unwrap_or_else(|| "Unknown".to_string())
            };
            Some(stats::ItemInfo {
                code: i.id.clone(),
                title: i.properties.title.clone().unwrap_or_else(|| i.id.clone()),
                files_by_format: {
                    let mut files = day_files.remove(&i.id).unwrap_or_default();
                    count(i, &mut files);
                    files
                },
                facets: stats::Facets {
                    collection: i.collection.clone()?,
                    provider: text("blatten:source"),
                    data_type: text("blatten:product_type"),
                    dataset: text("blatten:dataset"),
                    sensor: text("blatten:sensor"),
                    format: text("blatten:format"),
                    level: text("blatten:processing_level"),
                },
            })
        })
        .collect()
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

/// Path-based KML for the series-level items of a collection (those without `blatten:series`).
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

    // A series' days share its geometry, so only series-level items are placed
    let placemarks: String = items
        .iter()
        .filter(|item| item.properties.additional_fields.get("blatten:series").is_none_or(|v| v.is_null()))
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
        <Icon><href>https://maps.google.com/mapfiles/kml/pushpin/red-pushpin.png</href></Icon>
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

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str) -> StacItem {
        dated_item(id, "2024-03-01T00:00:00Z")
    }

    fn dated_item(id: &str, datetime: &str) -> StacItem {
        serde_json::from_value(serde_json::json!({
            "type": "Feature",
            "stac_version": "1.1.0",
            "id": id,
            "geometry": null,
            "properties": {"datetime": datetime, "title": format!("Orchard {id}")},
            "links": [],
            "assets": {"photo": {"href": format!("/s3/{id}/pear.jpg"), "title": "pear.jpg"}},
            "collection": "orchards"
        }))
        .unwrap()
    }

    #[test]
    fn sizes_hold_sized_assets_and_documents() {
        let root = serde_json::from_value(serde_json::json!({
            "type": "Catalog",
            "stac_version": "1.1.0",
            "id": "orchards-root",
            "description": "Fruit trees",
            "links": [
                {"rel": "describedby", "href": "/s3/docs/pruning_guide.pdf", "file:size": 2048},
                {"rel": "describedby", "href": "/s3/docs/unsized.pdf"},
                {"rel": "about", "href": "https://orchard.test/", "file:size": 7}
            ]
        }))
        .unwrap();
        let mut pear = item("pear-01");
        pear.assets.insert(
            "leaf".into(),
            serde_json::from_value(serde_json::json!({"href": "/s3/assets/pear-01/leaf.jpg", "file:size": 64})).unwrap(),
        );
        let catalog = StacCatalog::from_parts(root, Default::default(), vec![pear], "v".into());
        let sizes = file_index(&catalog).sizes;
        assert_eq!(sizes.get("/docs/pruning_guide.pdf"), Some(&2048));
        assert_eq!(sizes.get("/assets/pear-01/leaf.jpg"), Some(&64));
        assert_eq!(sizes.len(), 2);
    }

    fn orchard_app() -> Router {
        orchard_app_with_leaves((0..40).collect())
    }

    /// The orchard app, its pear's leaf assets inserted in the order of `leaves`
    fn orchard_app_with_leaves(leaves: Vec<usize>) -> Router {
        let root = serde_json::from_value(serde_json::json!({
            "type": "Catalog",
            "stac_version": "1.1.0",
            "id": "orchards-root",
            "description": "Fruit trees",
            "links": []
        }))
        .unwrap();
        let collection: catalog::StacCollection = serde_json::from_value(serde_json::json!({
            "type": "Collection",
            "stac_version": "1.1.0",
            "id": "orchards",
            "description": "Orchard surveys",
            "license": "CC0-1.0",
            "extent": {
                "spatial": {"bbox": [[-180.0, -90.0, 180.0, 90.0]]},
                "temporal": {"interval": [["2024-01-01T00:00:00Z", null]]}
            },
            "links": []
        }))
        .unwrap();
        let mut pear = item("pear-01");
        for n in leaves {
            pear.assets.insert(
                format!("leaf-{n}"),
                serde_json::from_value(serde_json::json!({"href": format!("/s3/pear-01/leaf-{n}.jpg")})).unwrap(),
            );
        }
        let in_series = |mut tree: StacItem, series: &str, level: i64| {
            tree.properties.additional_fields.insert("blatten:series".into(), series.into());
            tree.properties.additional_fields.insert("blatten:processing_level".into(), level.into());
            tree
        };
        let catalog = StacCatalog::from_parts(
            root,
            [("orchards".to_string(), collection)].into(),
            vec![
                in_series(dated_item("quince-03", "2019-09-01T00:00:00Z"), "orchard-a", 1),
                in_series(dated_item("quince-04", "2019-09-02T00:00:00Z"), "orchard-a", 2),
                pear,
                in_series(item("plum-02"), "orchard-b", 2),
            ],
            "orchard-version".to_string(),
        );
        let state = AppState { catalog: Arc::new(catalog), base_url: "http://orchard.test".into() };
        stac_router(state, asset_routes::Limits::from_env(|_| None))
    }

    /// Two kite series in the orchards collection, as the generator writes them: `kites-a`
    /// over 2019-09-01 to 2019-09-03 with one day item a day, the second day cut in two
    /// parts, and `kites-b` over 2019-10-01 with its one day.
    fn kite_series_app() -> Router {
        let ranged = |id: &str, start: &str, end: &str, series: Option<&str>| -> StacItem {
            let mut properties = serde_json::json!({"datetime": null, "start_datetime": start, "end_datetime": end, "title": id});
            if let Some(series) = series {
                properties["blatten:series"] = series.into();
            }
            serde_json::from_value(serde_json::json!({
                "type": "Feature",
                "stac_version": "1.1.0",
                "id": id,
                "geometry": {"type": "Point", "coordinates": [7.8, 46.4]},
                "bbox": [7.8, 46.4, 7.8, 46.4],
                "properties": properties,
                "links": [],
                "assets": {},
                "collection": "orchards"
            }))
            .unwrap()
        };
        let root = serde_json::from_value(serde_json::json!({
            "type": "Catalog", "stac_version": "1.1.0", "id": "kites-root", "description": "Kites", "links": []
        }))
        .unwrap();
        let collection: catalog::StacCollection = serde_json::from_value(serde_json::json!({
            "type": "Collection",
            "stac_version": "1.1.0",
            "id": "orchards",
            "description": "Kite flights over the orchards",
            "license": "CC0-1.0",
            "extent": {
                "spatial": {"bbox": [[-180.0, -90.0, 180.0, 90.0]]},
                "temporal": {"interval": [["2019-09-01T00:00:00Z", null]]}
            },
            "links": []
        }))
        .unwrap();
        let catalog = StacCatalog::from_parts(
            root,
            [("orchards".to_string(), collection)].into(),
            vec![
                ranged("kites-a", "2019-09-01T00:00:00Z", "2019-09-03T23:59:59Z", None),
                ranged("kites-a_2019-09-01", "2019-09-01T08:00:00Z", "2019-09-01T18:00:00Z", Some("kites-a")),
                ranged("kites-a_2019-09-02_1", "2019-09-02T06:00:00Z", "2019-09-02T12:00:00Z", Some("kites-a")),
                ranged("kites-a_2019-09-02_2", "2019-09-02T12:10:00Z", "2019-09-02T19:00:00Z", Some("kites-a")),
                ranged("kites-a_2019-09-03", "2019-09-03T07:00:00Z", "2019-09-03T17:00:00Z", Some("kites-a")),
                ranged("kites-b", "2019-10-01T00:00:00Z", "2019-10-01T23:59:59Z", None),
                ranged("kites-b_2019-10-01", "2019-10-01T09:00:00Z", "2019-10-01T15:00:00Z", Some("kites-b")),
            ],
            "kites-version".to_string(),
        );
        let state = AppState { catalog: Arc::new(catalog), base_url: "http://kites.test".into() };
        stac_router(state, asset_routes::Limits::from_env(|_| None))
    }

    #[test]
    fn download_statistics_list_a_series_with_every_file_of_its_days_and_no_day() {
        let with = |mut tree: StacItem, series: Option<&str>, leaves: &[&str]| {
            tree.assets.clear();
            let kept: &[&str] = match series {
                Some(series) => {
                    tree.properties.additional_fields.insert("blatten:series".into(), series.into());
                    &[]
                }
                None => &["archive", "manifest"],
            };
            for name in kept.iter().chain(leaves) {
                tree.assets.insert(
                    name.to_string(),
                    serde_json::from_value(serde_json::json!({"href": format!("/s3/{name}")})).unwrap(),
                );
            }
            tree
        };
        let items = vec![
            with(item("kites-a"), None, &["logbook"]),
            with(item("kites-a_2019-09-01"), Some("kites-a"), &["k1", "k2"]),
            with(item("kites-a_2019-09-02"), Some("kites-a"), &["k3"]),
            with(item("pear-01"), None, &["leaf"]),
        ];
        let listed: Vec<(String, usize)> =
            stat_items(&items, &Default::default()).into_iter().map(|i| (i.code.clone(), i.files_by_format.values().sum())).collect();
        assert_eq!(listed, vec![("kites-a".to_string(), 4), ("pear-01".to_string(), 1)]);
    }

    #[test]
    fn download_statistics_count_an_items_files_by_each_files_own_format() {
        let mut tree = item("kites-a");
        tree.assets.clear();
        tree.properties.additional_fields.insert("blatten:format".into(), "LAZ".into());
        for (name, file, media_type) in [
            ("archive", "archives/kites-a.zip", "application/zip"),
            ("tail", "assets/kites-a/tail.laz", "application/vnd.laszip"),
            ("wing", "assets/kites-a/wing.laz", "application/vnd.laszip"),
            ("notes", "assets/kites-a/flight_notes.pdf", "application/pdf"),
        ] {
            tree.assets.insert(
                name.into(),
                serde_json::from_value(serde_json::json!({"href": format!("/s3/{file}"), "type": media_type})).unwrap(),
            );
        }
        let root = serde_json::from_value(serde_json::json!({
            "type": "Catalog", "stac_version": "1.1.0", "id": "kites-root", "description": "Kites", "links": []
        }))
        .unwrap();
        let catalog = StacCatalog::from_parts(root, Default::default(), vec![tree], "v".into());
        let index = file_index(&catalog);
        assert_eq!(index.media_types.get("/assets/kites-a/flight_notes.pdf").map(String::as_str), Some("application/pdf"));
        let listed = stat_items(&catalog.items, &index);
        let counts: Vec<(&str, usize)> = listed[0].files_by_format.iter().map(|(f, n)| (f.as_str(), *n)).collect();
        assert_eq!(counts, vec![("LAZ", 2), ("PDF", 1)]);
        assert_eq!(listed[0].files_by_format.values().sum::<usize>(), 3);
    }

    #[tokio::test]
    async fn a_datetime_search_over_series_returns_the_days_asked_for_and_the_series_covering_them() {
        let app = kite_series_app();
        let (status, _, body) = send(
            &app,
            "/stac/collections/orchards/items?datetime=2019-09-02T00:00:00Z/2019-09-02T23:59:59Z",
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let page: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let ids: Vec<&str> = page["features"].as_array().unwrap().iter().map(|f| f["id"].as_str().unwrap()).collect();
        assert_eq!(ids, vec!["kites-a", "kites-a_2019-09-02_1", "kites-a_2019-09-02_2"]);
    }

    #[tokio::test]
    async fn a_collection_kml_places_each_series_once_and_none_of_its_days() {
        let app = kite_series_app();
        let (status, _, body) = send(&app, "/stac/collections/orchards/kml", &[]).await;
        assert_eq!(status, StatusCode::OK);
        let kml = String::from_utf8(body).unwrap();
        assert_eq!(kml.matches("<Placemark>").count(), 2, "{kml}");
        assert!(kml.contains("<name>kites-a</name>") && kml.contains("<name>kites-b</name>"), "{kml}");
    }

    async fn send(
        app: &Router,
        uri: &str,
        headers: &[(header::HeaderName, &str)],
    ) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
        use tower::ServiceExt;
        let mut req = axum::http::Request::builder().uri(uri);
        for (name, value) in headers {
            req = req.header(name, *value);
        }
        let resp = app.clone().oneshot(req.body(axum::body::Body::empty()).unwrap()).await.unwrap();
        let (parts, body) = resp.into_parts();
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        (parts.status, parts.headers, bytes.to_vec())
    }

    #[tokio::test]
    async fn a_search_asked_for_gzip_is_compressed_and_decodes_to_the_plain_answer() {
        use std::io::Read;
        let app = orchard_app();
        let (status, headers, plain) = send(&app, "/stac/search?limit=10", &[]).await;
        assert_eq!(status, StatusCode::OK);
        assert!(headers.get(header::CONTENT_ENCODING).is_none());

        let (status, headers, zipped) =
            send(&app, "/stac/search?limit=10", &[(header::ACCEPT_ENCODING, "gzip")]).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(headers[header::CONTENT_ENCODING], "gzip");
        assert!(zipped.len() < plain.len());
        let mut decoded = Vec::new();
        flate2::read::GzDecoder::new(&zipped[..]).read_to_end(&mut decoded).unwrap();
        assert_eq!(decoded, plain);
    }

    #[tokio::test]
    async fn a_get_repeated_with_its_etag_is_not_modified_and_another_query_is_answered() {
        let app = orchard_app();
        for uri in [
            "/stac/catalog.json",
            "/stac/collections",
            "/stac/collections/orchards",
            "/stac/collections/orchards/items/pear-01",
            "/stac/collections/orchards/items/pear-01?asset_limit=5",
        ] {
            let (status, headers, body) = send(&app, uri, &[]).await;
            assert_eq!(status, StatusCode::OK, "{uri}");
            assert!(!body.is_empty());
            let etag = headers[header::ETAG].to_str().unwrap().to_string();

            let (status, headers, body) = send(&app, uri, &[(header::IF_NONE_MATCH, &etag)]).await;
            assert_eq!(status, StatusCode::NOT_MODIFIED, "{uri}");
            assert!(body.is_empty(), "{uri}");
            assert_eq!(headers[header::ETAG], etag.as_str());
        }

        let (_, headers, _) = send(&app, "/stac/collections/orchards/items/pear-01?asset_limit=5", &[]).await;
        let etag = headers[header::ETAG].to_str().unwrap().to_string();
        for other in [
            "/stac/collections/orchards/items/pear-01",
            "/stac/collections/orchards/items/pear-01?asset_limit=5&asset_offset=5",
            "/stac/collections/orchards/items/pear-01?asset_limit=6",
        ] {
            let (status, _, body) = send(&app, other, &[(header::IF_NONE_MATCH, &etag)]).await;
            assert_eq!(status, StatusCode::OK, "{other}");
            assert!(!body.is_empty());
        }
    }

    fn leafy_item(leaves: impl Iterator<Item = usize>) -> StacItem {
        let mut tree = item("apple-07");
        for n in leaves {
            tree.assets.insert(
                format!("leaf-{n}"),
                serde_json::from_value(serde_json::json!({"href": format!("/s3/apple-07/leaf-{n}.jpg")})).unwrap(),
            );
        }
        tree
    }

    #[test]
    fn an_items_assets_come_out_in_one_order_whatever_order_they_were_inserted_in() {
        let forward = leafy_item(0..120);
        let backward = leafy_item((0..120).rev());
        assert_eq!(
            serde_json::to_string(&item_value(&forward)).unwrap(),
            serde_json::to_string(&item_value(&backward)).unwrap()
        );
        for offset in 0..=121 {
            let mut a = item_value(&forward);
            let mut b = item_value(&backward);
            page_assets(&mut a, 7, offset);
            page_assets(&mut b, 7, offset);
            assert_eq!(serde_json::to_string(&a).unwrap(), serde_json::to_string(&b).unwrap(), "offset {offset}");
        }
    }

    #[test]
    fn asset_keys_are_sorted_with_their_numbers_compared_as_numbers() {
        let json = item_value(&leafy_item([10, 2, 1, 0, 100, 21].into_iter()));
        let keys: Vec<_> = json["assets"].as_object().unwrap().keys().cloned().collect();
        assert_eq!(keys, ["leaf-0", "leaf-1", "leaf-2", "leaf-10", "leaf-21", "leaf-100", "photo"]);

        let mut keys = vec!["dem-2-10", "dem-2-9", "dem-10-0", "dem-02-9", "archive", "dem", "dem-2"];
        keys.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(keys, ["archive", "dem", "dem-2", "dem-02-9", "dem-2-9", "dem-2-10", "dem-10-0"]);
    }

    #[tokio::test]
    async fn two_states_built_from_the_same_catalog_answer_with_the_same_asset_pages_and_etags() {
        let forward = orchard_app_with_leaves((0..40).collect());
        let backward = orchard_app_with_leaves((0..40).rev().collect());
        for uri in [
            "/stac/collections/orchards/items/pear-01",
            "/stac/collections/orchards/items/pear-01?asset_limit=5",
            "/stac/collections/orchards/items/pear-01?asset_limit=5&asset_offset=35",
        ] {
            let (_, a_headers, a_body) = send(&forward, uri, &[]).await;
            let (_, b_headers, b_body) = send(&backward, uri, &[]).await;
            assert_eq!(a_headers[header::ETAG], b_headers[header::ETAG], "{uri}");
            assert_eq!(a_body, b_body, "{uri}");
        }
    }

    #[test]
    fn if_none_match_matches_by_weak_comparison_against_any_listed_tag() {
        let tag = "W/\"abc\"";
        let matches = |value: &str| {
            let mut headers = axum::http::HeaderMap::new();
            headers.insert(header::IF_NONE_MATCH, value.parse().unwrap());
            if_none_match(&headers, tag)
        };
        assert!(matches("W/\"abc\""));
        assert!(matches("\"abc\""));
        assert!(matches("\"x\", W/\"abc\""));
        assert!(matches("*"));
        assert!(!matches("\"abcd\""));
        assert!(!matches("W/\"ab\""));
        assert!(!if_none_match(&axum::http::HeaderMap::new(), tag));
    }

    #[test]
    fn a_page_written_item_by_item_matches_the_page_built_as_one_tree() {
        let items = [item("pear-01"), item("plum-02")];
        let refs: Vec<&StacItem> = items.iter().collect();
        let fields = FieldsFilter::parse_get("id,properties.title,assets");
        let links = vec![serde_json::json!({"rel": "self", "href": "http://orchard.test/stac/search"})];
        let context = serde_json::json!({"limit": 2, "matched": 7, "returned": 2});

        for (exclude_assets, fields) in [(false, FieldsFilter::default()), (true, fields)] {
            let streamed = serde_json::to_string(&ItemPage {
                kind: "FeatureCollection",
                features: Features { items: &refs, exclude_assets, fields: &fields },
                number_matched: 7,
                number_returned: 2,
                context: Some(context.clone()),
                links: links.clone(),
            })
            .unwrap();
            let features: Vec<_> = refs.iter().map(|i| item_to_json(i, exclude_assets, &fields)).collect();
            let tree = serde_json::to_string(&serde_json::json!({
                "type": "FeatureCollection",
                "features": features,
                "numberMatched": 7,
                "numberReturned": 2,
                "context": context,
                "links": links
            }))
            .unwrap();
            assert_eq!(streamed, tree);
        }

        let without_context = serde_json::to_value(&ItemPage {
            kind: "FeatureCollection",
            features: Features { items: &[], exclude_assets: false, fields: &FieldsFilter::default() },
            number_matched: 0,
            number_returned: 0,
            context: None,
            links: vec![],
        })
        .unwrap();
        assert!(without_context.get("context").is_none());
    }

    fn link_href<'a>(links: &'a [serde_json::Value], rel: &str) -> Option<&'a str> {
        links.iter().find(|l| l["rel"] == rel).and_then(|l| l["href"].as_str())
    }

    fn query_pairs(href: &str) -> Vec<(String, String)> {
        let query = href.split_once('?').map_or("", |(_, q)| q);
        url::form_urlencoded::parse(query.as_bytes()).into_owned().collect()
    }

    #[test]
    fn get_page_links_carry_every_received_parameter_with_only_the_offset_moved() {
        let query = "collections=orchards%2Cgroves&bbox=7.1,46.2,7.9,46.5&datetime=2024-01-01T00:00:00Z%2F..\
                     &fields=id,-assets&exclude_assets=true&offset=4&limit=50&shade=dappled";
        let links = pagination_links(
            "http://orchard.test",
            "/stac/search",
            PageMethod::Get { query },
            4,
            2,
            10,
        );
        let expected = |offset: &str| {
            vec![
                ("collections".to_string(), "orchards,groves".to_string()),
                ("bbox".to_string(), "7.1,46.2,7.9,46.5".to_string()),
                ("datetime".to_string(), "2024-01-01T00:00:00Z/..".to_string()),
                ("fields".to_string(), "id,-assets".to_string()),
                ("exclude_assets".to_string(), "true".to_string()),
                ("shade".to_string(), "dappled".to_string()),
                ("offset".to_string(), offset.to_string()),
                ("limit".to_string(), "2".to_string()),
            ]
        };
        let next = link_href(&links, "next").unwrap();
        assert!(next.starts_with("http://orchard.test/stac/search?"), "{next}");
        assert_eq!(query_pairs(next), expected("6"));
        let prev = link_href(&links, "prev").unwrap();
        assert_eq!(query_pairs(prev), expected("2"));

        let first = pagination_links("http://orchard.test", "/stac/search", PageMethod::Get { query: "" }, 0, 2, 3);
        assert_eq!(query_pairs(link_href(&first, "next").unwrap()), vec![
            ("offset".to_string(), "2".to_string()),
            ("limit".to_string(), "2".to_string()),
        ]);
        assert!(link_href(&first, "prev").is_none());

        let last = pagination_links("http://orchard.test", "/stac/search", PageMethod::Get { query: "" }, 2, 2, 3);
        assert!(link_href(&last, "next").is_none());
    }

    async fn send_json(app: &Router, uri: &str, body: Option<serde_json::Value>) -> (StatusCode, serde_json::Value) {
        use tower::ServiceExt;
        let req = match body {
            Some(body) => axum::http::Request::builder()
                .method("POST")
                .uri(uri)
                .header(header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(body.to_string())),
            None => axum::http::Request::builder().uri(uri).body(axum::body::Body::empty()),
        };
        let resp = app.clone().oneshot(req.unwrap()).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or_default())
    }

    fn feature_ids(page: &serde_json::Value) -> Vec<String> {
        let mut ids: Vec<String> = page["features"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f["id"].as_str().unwrap().to_string())
            .collect();
        ids.sort();
        ids
    }

    fn encoded(text: &str) -> String {
        url::form_urlencoded::byte_serialize(text.as_bytes()).collect()
    }

    #[tokio::test]
    async fn a_cql2_text_filter_keeps_only_the_items_it_is_true_of() {
        let app = orchard_app();
        for (filter, expected) in [
            ("blatten:series='orchard-a'", vec!["quince-03", "quince-04"]),
            ("blatten:series IS NULL", vec!["pear-01"]),
            ("blatten:series IS NOT NULL AND blatten:series <> 'orchard-a'", vec!["plum-02"]),
            ("blatten:series = 'orchard-a' AND blatten:processing_level >= 2", vec!["quince-04"]),
            ("NOT (blatten:series = 'orchard-a') OR blatten:processing_level < 2", vec!["plum-02", "quince-03"]),
            ("datetime > TIMESTAMP('2024-01-01T00:00:00Z')", vec!["pear-01", "plum-02"]),
            ("datetime <= '2019-09-01T00:00:00Z'", vec!["quince-03"]),
        ] {
            for uri in [
                format!("/stac/search?filter={}", encoded(filter)),
                format!("/stac/search?filter-lang=cql2-text&filter={}", encoded(filter)),
                format!("/stac/collections/orchards/items?filter={}", encoded(filter)),
            ] {
                let (status, page) = send_json(&app, &uri, None).await;
                assert_eq!(status, StatusCode::OK, "{uri}: {page}");
                assert_eq!(feature_ids(&page), expected, "{uri}");
                assert_eq!(page["numberMatched"], expected.len(), "{uri}");
            }
        }
    }

    #[tokio::test]
    async fn a_cql2_json_filter_posted_to_search_keeps_only_the_items_it_is_true_of() {
        let app = orchard_app();
        let series = serde_json::json!({"property": "blatten:series"});
        for (filter, expected) in [
            (serde_json::json!({"op": "=", "args": [series, "orchard-a"]}), vec!["quince-03", "quince-04"]),
            (serde_json::json!({"op": "isNull", "args": [series]}), vec!["pear-01"]),
            (
                serde_json::json!({"op": "and", "args": [
                    {"op": "not", "args": [{"op": "isNull", "args": [series]}]},
                    {"op": ">", "args": [{"property": "blatten:processing_level"}, 1]}
                ]}),
                vec!["plum-02", "quince-04"],
            ),
        ] {
            for body in [
                serde_json::json!({"filter": filter, "filter-lang": "cql2-json"}),
                serde_json::json!({"filter": filter}),
            ] {
                let (status, page) = send_json(&app, "/stac/search", Some(body.clone())).await;
                assert_eq!(status, StatusCode::OK, "{body}: {page}");
                assert_eq!(feature_ids(&page), expected, "{body}");
            }
        }
        let (status, page) = send_json(
            &app,
            "/stac/search",
            Some(serde_json::json!({"filter": "blatten:series = 'orchard-b'", "filter-lang": "cql2-text"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{page}");
        assert_eq!(feature_ids(&page), ["plum-02"]);
    }

    #[tokio::test]
    async fn a_filter_naming_an_unknown_property_or_failing_to_parse_is_a_bad_request() {
        let app = orchard_app();
        for filter in ["nope='x'", "blatten:series LIKE 'orchard%'", "blatten:series = = 'x'", "(("] {
            for path in ["/stac/search", "/stac/collections/orchards/items"] {
                let uri = format!("{path}?filter={}", encoded(filter));
                let (status, body) = send_json(&app, &uri, None).await;
                assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}");
                assert!(!body["description"].as_str().unwrap().is_empty(), "{uri}");
            }
        }
        let (status, body) = send_json(&app, "/stac/search?filter=nope%3D'x'", None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["description"].as_str().unwrap().contains("nope"), "{body}");

        let (status, _) = send_json(&app, "/stac/search?filter-lang=cql2-xml&filter=x%3D1", None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _) = send_json(
            &app,
            "/stac/search",
            Some(serde_json::json!({"filter": {"op": "=", "args": [{"property": "nope"}, "x"]}})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn following_next_from_a_filtered_page_keeps_the_filter() {
        let app = orchard_app();
        let filter = encoded("blatten:series='orchard-a'");
        for (first, ids) in [
            ("/stac/search?datetime=2024-01-01T00:00:00Z%2F..&exclude_assets=true&limit=1".to_string(), ["pear-01", "plum-02"]),
            (
                "/stac/collections/orchards/items?datetime=2024-01-01T00:00:00Z%2F..&exclude_assets=true&limit=1".to_string(),
                ["pear-01", "plum-02"],
            ),
            (format!("/stac/search?filter={filter}&filter-lang=cql2-text&limit=1"), ["quince-03", "quince-04"]),
            (format!("/stac/collections/orchards/items?filter={filter}&limit=1"), ["quince-03", "quince-04"]),
        ] {
            let excluded = first.contains("exclude_assets=true");
            let (status, page) = send_json(&app, &first, None).await;
            assert_eq!(status, StatusCode::OK, "{first}");
            assert_eq!(page["numberMatched"], 2, "{first}");
            assert_eq!(feature_ids(&page), [ids[0]], "{first}");
            let next = link_href(page["links"].as_array().unwrap(), "next").unwrap().to_string();
            let (status, page) = send_json(&app, next.strip_prefix("http://orchard.test").unwrap(), None).await;
            assert_eq!(status, StatusCode::OK, "{next}");
            assert_eq!(page["numberMatched"], 2, "{next}");
            assert_eq!(feature_ids(&page), [ids[1]], "{next}");
            assert_eq!(page["features"][0]["assets"].get("photo").is_none(), excluded, "{next}");
            assert!(link_href(page["links"].as_array().unwrap(), "next").is_none(), "{next}");
        }
    }

    #[tokio::test]
    async fn queryables_are_served_as_a_json_schema_and_named_by_the_landing_page_and_conformance() {
        let app = orchard_app();
        for uri in ["/stac/queryables", "/stac/collections/orchards/queryables"] {
            let (status, headers, body) = send(&app, uri, &[]).await;
            assert_eq!(status, StatusCode::OK, "{uri}");
            assert_eq!(headers[header::CONTENT_TYPE], "application/schema+json", "{uri}");
            let schema: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(schema["$id"], format!("http://orchard.test{uri}"));
            assert_eq!(schema["additionalProperties"], false);
            let names: Vec<_> = schema["properties"].as_object().unwrap().keys().cloned().collect();
            for name in [
                "blatten:series",
                "blatten:sensor",
                "blatten:code",
                "blatten:source",
                "blatten:processing_level",
                "datetime",
                "start_datetime",
                "end_datetime",
            ] {
                assert!(names.iter().any(|n| n == name), "{uri} lacks {name}");
            }
        }
        let (status, _) = send_json(&app, "/stac/collections/nope/queryables", None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        let (_, landing) = send_json(&app, "/stac", None).await;
        assert_eq!(
            link_href(landing["links"].as_array().unwrap(), "http://www.opengis.net/def/rel/ogc/1.0/queryables"),
            Some("http://orchard.test/stac/queryables")
        );
        let (_, conformance) = send_json(&app, "/stac/conformance", None).await;
        let classes = conformance["conformsTo"].as_array().unwrap();
        for class in [
            "https://api.stacspec.org/v1.0.0-rc.2/item-search#filter",
            "https://api.stacspec.org/v1.0.0-rc.2/ogcapi-features/extensions/filter",
            "http://www.opengis.net/spec/cql2/1.0/conf/basic-cql2",
            "http://www.opengis.net/spec/cql2/1.0/conf/cql2-text",
            "http://www.opengis.net/spec/cql2/1.0/conf/cql2-json",
        ] {
            assert!(classes.iter().any(|c| c == class), "conformance lacks {class}");
        }
    }
}
