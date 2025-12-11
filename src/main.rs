//! Blatten Data API - STAC API Server
//!
//! A STAC API server for the Birch Glacier Collapse Dataset.
//! Uses an in-memory backend loaded from JSON catalog files.

use anyhow::{Context, Result};
use axum::{
    extract::{Path, Query, State},
    http::{header, Method, StatusCode},
    response::{IntoResponse, Json},
    routing::get,
    Router,
};
use serde::Deserialize;
use std::{env, net::SocketAddr, path::PathBuf, sync::Arc};
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod catalog;

use catalog::{StacCatalog, StacItem};

/// Application state shared across handlers
#[derive(Clone)]
struct AppState {
    catalog: Arc<StacCatalog>,
    base_url: String,
}

/// Query parameters for item search
#[derive(Debug, Deserialize)]
struct SearchParams {
    /// Bounding box filter [west, south, east, north]
    bbox: Option<String>,
    /// Temporal filter (RFC 3339 interval)
    datetime: Option<String>,
    /// Filter by collection IDs (comma-separated)
    collections: Option<String>,
    /// Maximum number of results
    limit: Option<usize>,
    /// Result offset for pagination
    offset: Option<usize>,
    /// Filter by source provider
    source: Option<String>,
    /// Filter by processing level
    processing_level: Option<i32>,
    /// Exclude assets from response (for lightweight listing)
    #[serde(default)]
    exclude_assets: bool,
}

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
        "Loaded {} collections with {} total items",
        catalog.collections.len(),
        catalog.items.len()
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
        .route("/stac/search", get(search_items).post(search_items_post))
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

/// STAC API Landing Page
async fn landing_page(State(state): State<AppState>) -> impl IntoResponse {
    let landing = serde_json::json!({
        "type": "Catalog",
        "id": "birch-glacier-collapse",
        "stac_version": "1.0.0",
        "title": "Birch Glacier Collapse and Landslide Dataset",
        "description": "STAC API for the 2025 Birch glacier collapse and landslide dataset at Blatten, CH-VS",
        "conformsTo": [
            "https://api.stacspec.org/v1.0.0/core",
            "https://api.stacspec.org/v1.0.0/collections",
            "https://api.stacspec.org/v1.0.0/item-search"
        ],
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

    Json(landing)
}

/// STAC API Conformance
async fn conformance() -> impl IntoResponse {
    Json(serde_json::json!({
        "conformsTo": [
            "https://api.stacspec.org/v1.0.0/core",
            "https://api.stacspec.org/v1.0.0/collections",
            "https://api.stacspec.org/v1.0.0/item-search",
            "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core",
            "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/geojson"
        ]
    }))
}

/// Get root catalog
async fn get_catalog(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.catalog.root.clone())
}

/// List all collections
async fn list_collections(State(state): State<AppState>) -> impl IntoResponse {
    let collections: Vec<_> = state.catalog.collections.values().collect();

    Json(serde_json::json!({
        "collections": collections,
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
}

/// Get a specific collection
async fn get_collection(
    State(state): State<AppState>,
    Path(collection_id): Path<String>,
) -> impl IntoResponse {
    match state.catalog.collections.get(&collection_id) {
        Some(collection) => Json(serde_json::to_value(collection).unwrap()).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "code": "NotFound",
                "description": format!("Collection '{}' not found", collection_id)
            })),
        )
            .into_response(),
    }
}

/// Get items in a collection
async fn get_collection_items(
    State(state): State<AppState>,
    Path(collection_id): Path<String>,
    Query(params): Query<SearchParams>,
) -> impl IntoResponse {
    let items: Vec<_> = state
        .catalog
        .items
        .iter()
        .filter(|item| item.collection.as_deref() == Some(&collection_id))
        .filter(|item| filter_item(item, &params))
        .skip(params.offset.unwrap_or(0))
        .take(params.limit.unwrap_or(100))
        .collect();

    let total = state
        .catalog
        .items
        .iter()
        .filter(|item| item.collection.as_deref() == Some(&collection_id))
        .count();

    // Convert items to JSON, optionally stripping assets for lightweight response
    let features: Vec<serde_json::Value> = if params.exclude_assets {
        items
            .iter()
            .map(|item| {
                let mut json = serde_json::to_value(item).unwrap();
                if let Some(obj) = json.as_object_mut() {
                    // Keep only the archive asset if it exists
                    if let Some(assets) = obj.get("assets").and_then(|a| a.as_object()) {
                        let archive = assets.get("archive").cloned();
                        let mut minimal_assets = serde_json::Map::new();
                        if let Some(arch) = archive {
                            minimal_assets.insert("archive".to_string(), arch);
                        }
                        obj.insert("assets".to_string(), serde_json::Value::Object(minimal_assets));
                    }
                }
                json
            })
            .collect()
    } else {
        items
            .iter()
            .map(|item| serde_json::to_value(item).unwrap())
            .collect()
    };

    Json(serde_json::json!({
        "type": "FeatureCollection",
        "features": features,
        "numberMatched": total,
        "numberReturned": features.len(),
        "links": [
            {
                "rel": "self",
                "href": format!("{}/stac/collections/{}/items", state.base_url, collection_id),
                "type": "application/geo+json"
            },
            {
                "rel": "collection",
                "href": format!("{}/stac/collections/{}", state.base_url, collection_id),
                "type": "application/json"
            }
        ]
    }))
}

/// Query parameters for asset pagination
#[derive(Debug, Deserialize)]
struct AssetParams {
    /// Maximum number of assets to return (excludes 'archive')
    asset_limit: Option<usize>,
    /// Offset for asset pagination
    asset_offset: Option<usize>,
}

/// Get a specific item
async fn get_item(
    State(state): State<AppState>,
    Path((collection_id, item_id)): Path<(String, String)>,
    Query(params): Query<AssetParams>,
) -> impl IntoResponse {
    let item = state
        .catalog
        .items
        .iter()
        .find(|i| i.id == item_id && i.collection.as_deref() == Some(&collection_id));

    match item {
        Some(item) => {
            let mut json = serde_json::to_value(item).unwrap();

            // Paginate assets if limit is specified
            if let Some(limit) = params.asset_limit {
                if let Some(obj) = json.as_object_mut() {
                    if let Some(assets) = obj.get("assets").and_then(|a| a.as_object()) {
                        let offset = params.asset_offset.unwrap_or(0);

                        // Separate archive from other assets
                        let archive = assets.get("archive").cloned();
                        let other_assets: Vec<_> = assets.iter()
                            .filter(|(k, _)| *k != "archive")
                            .collect();

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
                        obj.insert("_assetsMeta".to_string(), serde_json::json!({
                            "total": total_assets,
                            "offset": offset,
                            "returned": returned_count
                        }));
                    }
                }
            }

            Json(json).into_response()
        },
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "code": "NotFound",
                "description": format!("Item '{}' not found in collection '{}'", item_id, collection_id)
            })),
        )
            .into_response(),
    }
}

/// Search items (GET)
async fn search_items(
    State(state): State<AppState>,
    Query(params): Query<SearchParams>,
) -> impl IntoResponse {
    perform_search(&state, params)
}

/// Search items (POST)
async fn search_items_post(
    State(state): State<AppState>,
    Json(params): Json<SearchParams>,
) -> impl IntoResponse {
    perform_search(&state, params)
}

fn perform_search(state: &AppState, params: SearchParams) -> impl IntoResponse {
    // Filter by collections if specified
    let collection_filter: Option<Vec<&str>> = params
        .collections
        .as_ref()
        .map(|c| c.split(',').map(|s| s.trim()).collect());

    let items: Vec<_> = state
        .catalog
        .items
        .iter()
        .filter(|item| {
            if let Some(ref colls) = collection_filter {
                item.collection
                    .as_ref()
                    .map(|c| colls.contains(&c.as_str()))
                    .unwrap_or(false)
            } else {
                true
            }
        })
        .filter(|item| filter_item(item, &params))
        .skip(params.offset.unwrap_or(0))
        .take(params.limit.unwrap_or(100))
        .collect();

    let total = state.catalog.items.len();

    // Convert items to JSON, optionally stripping assets for lightweight response
    let features: Vec<serde_json::Value> = if params.exclude_assets {
        items
            .iter()
            .map(|item| {
                let mut json = serde_json::to_value(item).unwrap();
                if let Some(obj) = json.as_object_mut() {
                    // Keep only the archive asset if it exists
                    if let Some(assets) = obj.get("assets").and_then(|a| a.as_object()) {
                        let archive = assets.get("archive").cloned();
                        let mut minimal_assets = serde_json::Map::new();
                        if let Some(arch) = archive {
                            minimal_assets.insert("archive".to_string(), arch);
                        }
                        obj.insert("assets".to_string(), serde_json::Value::Object(minimal_assets));
                    }
                }
                json
            })
            .collect()
    } else {
        items
            .iter()
            .map(|item| serde_json::to_value(item).unwrap())
            .collect()
    };

    Json(serde_json::json!({
        "type": "FeatureCollection",
        "features": features,
        "numberMatched": total,
        "numberReturned": features.len(),
        "context": {
            "limit": params.limit.unwrap_or(100),
            "matched": total,
            "returned": features.len()
        },
        "links": [
            {
                "rel": "self",
                "href": format!("{}/stac/search", state.base_url),
                "type": "application/geo+json"
            },
            {
                "rel": "root",
                "href": format!("{}/stac", state.base_url),
                "type": "application/json"
            }
        ]
    }))
}

/// Filter an item based on search parameters
fn filter_item(item: &StacItem, params: &SearchParams) -> bool {
    // Bbox filter
    if let Some(ref bbox_str) = params.bbox {
        if let Some(ref item_bbox) = item.bbox {
            let bbox: Vec<f64> = bbox_str
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();

            if bbox.len() == 4 {
                let (west, south, east, north) = (bbox[0], bbox[1], bbox[2], bbox[3]);
                let (iwest, isouth, ieast, inorth) =
                    (item_bbox[0], item_bbox[1], item_bbox[2], item_bbox[3]);

                // Check if bboxes intersect
                if ieast < west || iwest > east || inorth < south || isouth > north {
                    return false;
                }
            }
        } else {
            // Item has no bbox, exclude from bbox searches
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

    // TODO: datetime filter

    true
}

/// Health check endpoint
async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "collections": state.catalog.collections.len(),
        "items": state.catalog.items.len()
    }))
}
