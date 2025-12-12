//! STAC Catalog data structures and loading
//!
//! This module provides loading and indexing of STAC catalogs from JSON files.
//! It uses the official `stac` crate types for STAC 1.1.0 compliance.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, fs, path::Path};
use tracing::{debug, info, warn};

/// STAC specification version
pub const STAC_VERSION: &str = "1.1.0";

/// Maximum items per page (prevents memory DoS)
pub const MAX_LIMIT: usize = 1000;

/// Default items per page
pub const DEFAULT_LIMIT: usize = 100;

/// Valid processing levels for this dataset
pub const VALID_PROCESSING_LEVELS: std::ops::RangeInclusive<i32> = 1..=4;

/// Root STAC Catalog with indexed collections and items
#[derive(Debug, Clone)]
pub struct StacCatalog {
    /// Root catalog JSON (for direct serving)
    pub root: Value,
    /// Collections indexed by ID
    pub collections: HashMap<String, StacCollection>,
    /// All items (flat list for searching)
    pub items: Vec<StacItem>,
    /// Items indexed by (collection_id, item_id) for O(1) lookup
    pub items_index: HashMap<(String, String), usize>,
    /// Items grouped by collection for efficient collection queries
    pub items_by_collection: HashMap<String, Vec<usize>>,
}

/// STAC Collection (1.1.0 compliant)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StacCollection {
    #[serde(rename = "type")]
    pub type_: String,
    pub id: String,
    pub stac_version: String,
    #[serde(default)]
    pub stac_extensions: Vec<String>,
    pub title: Option<String>,
    pub description: String,
    pub license: String,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub providers: Vec<Provider>,
    pub extent: Extent,
    #[serde(default)]
    pub summaries: HashMap<String, Value>,
    #[serde(default)]
    pub links: Vec<Link>,
    #[serde(default)]
    pub assets: HashMap<String, Asset>,
    /// Item assets definition (new in STAC 1.1.0 core)
    #[serde(default)]
    pub item_assets: HashMap<String, ItemAssetDefinition>,
}

/// STAC Item (Feature) - 1.1.0 compliant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StacItem {
    #[serde(rename = "type")]
    pub type_: String,
    pub stac_version: String,
    #[serde(default)]
    pub stac_extensions: Vec<String>,
    pub id: String,
    pub geometry: Option<Value>,
    pub bbox: Option<Vec<f64>>,
    pub properties: HashMap<String, Value>,
    #[serde(default)]
    pub links: Vec<Link>,
    #[serde(default)]
    pub assets: HashMap<String, Asset>,
    pub collection: Option<String>,
}

impl StacItem {
    /// Get the datetime property as a string
    pub fn datetime(&self) -> Option<&str> {
        self.properties.get("datetime").and_then(|v| v.as_str())
    }

    /// Get start_datetime property
    pub fn start_datetime(&self) -> Option<&str> {
        self.properties
            .get("start_datetime")
            .and_then(|v| v.as_str())
    }

    /// Get end_datetime property
    pub fn end_datetime(&self) -> Option<&str> {
        self.properties.get("end_datetime").and_then(|v| v.as_str())
    }
}

/// STAC Provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provider {
    pub name: String,
    #[serde(default)]
    pub roles: Vec<String>,
    pub url: Option<String>,
    pub description: Option<String>,
}

/// STAC Extent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Extent {
    pub spatial: SpatialExtent,
    pub temporal: TemporalExtent,
}

/// Spatial Extent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialExtent {
    pub bbox: Vec<Vec<Option<f64>>>,
}

/// Temporal Extent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalExtent {
    pub interval: Vec<Vec<Option<String>>>,
}

/// STAC Link (1.1.0 compliant with method/headers/body support)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    pub rel: String,
    pub href: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub type_: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merge: Option<bool>,
}

/// STAC Asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub href: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub type_: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub roles: Vec<String>,
    #[serde(rename = "file:size", skip_serializing_if = "Option::is_none")]
    pub file_size: Option<i64>,
}

/// Item Asset Definition (new in STAC 1.1.0 core - was extension before)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemAssetDefinition {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub type_: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub roles: Vec<String>,
}

impl StacCatalog {
    /// Load a STAC catalog from a directory
    pub fn load_from_dir(dir: &Path, base_url: &str) -> Result<Self> {
        // Load root catalog
        let catalog_path = dir.join("catalog.json");
        info!("Loading catalog from {:?}", catalog_path);

        let catalog_json = fs::read_to_string(&catalog_path)
            .with_context(|| format!("Failed to read {:?}", catalog_path))?;

        // Replace base URL placeholder and upgrade version
        let catalog_json = catalog_json
            .replace("${STAC_BASE_URL}", base_url)
            .replace("\"1.0.0\"", &format!("\"{}\"", STAC_VERSION));
        let root: Value = serde_json::from_str(&catalog_json)?;

        // Load collections
        let collections_dir = dir.join("collections");
        let mut collections = HashMap::new();

        if collections_dir.exists() {
            for entry in fs::read_dir(&collections_dir)? {
                let entry = entry?;
                let path = entry.path();

                // Only load collection files (not items files)
                if path.extension().map(|e| e == "json").unwrap_or(false)
                    && !path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .contains("_items")
                {
                    debug!("Loading collection from {:?}", path);

                    let json = fs::read_to_string(&path)?;
                    let json = json
                        .replace("${STAC_BASE_URL}", base_url)
                        .replace("\"1.0.0\"", &format!("\"{}\"", STAC_VERSION));

                    match serde_json::from_str::<StacCollection>(&json) {
                        Ok(mut collection) => {
                            // Ensure STAC version is 1.1.0
                            collection.stac_version = STAC_VERSION.to_string();
                            info!("Loaded collection: {}", collection.id);
                            collections.insert(collection.id.clone(), collection);
                        }
                        Err(e) => {
                            warn!("Failed to parse collection {:?}: {}", path, e);
                        }
                    }
                }
            }
        }

        // Load all items
        let all_items_path = dir.join("all_items.json");
        let items = if all_items_path.exists() {
            info!("Loading items from {:?}", all_items_path);

            let json = fs::read_to_string(&all_items_path)?;
            let json = json
                .replace("${STAC_BASE_URL}", base_url)
                .replace("\"1.0.0\"", &format!("\"{}\"", STAC_VERSION));

            let feature_collection: Value = serde_json::from_str(&json)?;

            if let Some(features) = feature_collection.get("features").and_then(|f| f.as_array()) {
                features
                    .iter()
                    .filter_map(|f| match serde_json::from_value::<StacItem>(f.clone()) {
                        Ok(mut item) => {
                            // Ensure STAC version is 1.1.0
                            item.stac_version = STAC_VERSION.to_string();
                            Some(item)
                        }
                        Err(e) => {
                            warn!("Failed to parse item: {}", e);
                            None
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            warn!("No all_items.json found");
            Vec::new()
        };

        // Build indexes for O(1) lookup
        let mut items_index = HashMap::new();
        let mut items_by_collection: HashMap<String, Vec<usize>> = HashMap::new();

        for (idx, item) in items.iter().enumerate() {
            if let Some(ref collection_id) = item.collection {
                items_index.insert((collection_id.clone(), item.id.clone()), idx);
                items_by_collection
                    .entry(collection_id.clone())
                    .or_default()
                    .push(idx);
            }
        }

        info!(
            "Loaded {} items total, indexed {} by collection",
            items.len(),
            items_index.len()
        );

        Ok(StacCatalog {
            root,
            collections,
            items,
            items_index,
            items_by_collection,
        })
    }

    /// Get an item by collection and item ID (O(1) lookup)
    pub fn get_item(&self, collection_id: &str, item_id: &str) -> Option<&StacItem> {
        self.items_index
            .get(&(collection_id.to_string(), item_id.to_string()))
            .map(|&idx| &self.items[idx])
    }

    /// Get all items in a collection
    pub fn get_collection_items(&self, collection_id: &str) -> Vec<&StacItem> {
        self.items_by_collection
            .get(collection_id)
            .map(|indices| indices.iter().map(|&idx| &self.items[idx]).collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_item() {
        let json = r#"{
            "type": "Feature",
            "stac_version": "1.1.0",
            "id": "test-item",
            "geometry": null,
            "bbox": null,
            "properties": {
                "datetime": "2025-05-28T00:00:00Z"
            },
            "links": [],
            "assets": {},
            "collection": "test-collection"
        }"#;

        let item: StacItem = serde_json::from_str(json).unwrap();
        assert_eq!(item.id, "test-item");
        assert_eq!(item.collection, Some("test-collection".to_string()));
        assert_eq!(item.stac_version, "1.1.0");
    }

    #[test]
    fn test_item_datetime_accessors() {
        let json = r#"{
            "type": "Feature",
            "stac_version": "1.1.0",
            "id": "test-item",
            "geometry": null,
            "bbox": null,
            "properties": {
                "datetime": null,
                "start_datetime": "2025-01-01T00:00:00Z",
                "end_datetime": "2025-12-31T23:59:59Z"
            },
            "links": [],
            "assets": {},
            "collection": "test-collection"
        }"#;

        let item: StacItem = serde_json::from_str(json).unwrap();
        assert!(item.datetime().is_none());
        assert_eq!(item.start_datetime(), Some("2025-01-01T00:00:00Z"));
        assert_eq!(item.end_datetime(), Some("2025-12-31T23:59:59Z"));
    }
}
