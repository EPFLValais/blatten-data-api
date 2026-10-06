//! STAC Catalog data structures and loading
//!
//! This module uses the official `stac` crate types for STAC 1.1.0 compliance.
//! It provides loading, indexing, and validation of STAC catalogs.

use anyhow::{Context, Result};
use serde_json::Value;
use std::{collections::HashMap, fs, io::BufRead, path::Path};
use tracing::{debug, info, warn};

// Re-export stac types we use
pub use stac::item::Item as StacItem;
pub use stac::{Catalog as StacCatalogRoot, Collection as StacCollection};

/// Maximum items per page (prevents memory DoS)
pub const MAX_LIMIT: usize = 1000;

/// Default items per page
pub const DEFAULT_LIMIT: usize = 100;

/// Valid processing levels for this dataset
pub const VALID_PROCESSING_LEVELS: std::ops::RangeInclusive<i32> = 1..=4;

/// STAC version string (from stac crate)
pub const STAC_VERSION: &str = "1.1.0";

/// Asset base URL as written into hrefs. A root-relative S3 base (`/s3`) is
/// served from the API's own origin, so it is prefixed with the API base URL:
/// a relative href would resolve against the item's `self` link, and an item
/// saved outside the site would lose its downloads.
pub fn resolve_s3_base_url(base_url: &str, s3_base_url: &str) -> String {
    if s3_base_url.starts_with('/') {
        format!("{}{}", base_url.trim_end_matches('/'), s3_base_url)
    } else {
        s3_base_url.to_string()
    }
}

/// Replace the `${STAC_BASE_URL}` and `${S3_BASE_URL}` placeholders the
/// generator writes into links and asset hrefs.
fn substitute_placeholders(json: &str, base_url: &str, s3_base_url: &str) -> String {
    json.replace("${STAC_BASE_URL}", base_url)
        .replace("${S3_BASE_URL}", s3_base_url)
}

/// Root STAC Catalog with indexed collections and items
#[derive(Debug, Clone)]
pub struct StacCatalog {
    /// Root catalog (from stac crate)
    pub root: StacCatalogRoot,
    /// Collections indexed by ID
    pub collections: HashMap<String, StacCollection>,
    /// All items (flat list for searching)
    pub items: Vec<StacItem>,
    /// Items indexed by (collection_id, item_id) for O(1) lookup
    pub items_index: HashMap<(String, String), usize>,
    /// Items grouped by collection for efficient collection queries
    pub items_by_collection: HashMap<String, Vec<usize>>,
}

impl StacCatalog {
    /// Load a STAC catalog from a directory
    pub fn load_from_dir(dir: &Path, base_url: &str, s3_base_url: &str) -> Result<Self> {
        let s3_base_url = &resolve_s3_base_url(base_url, s3_base_url);

        // Load root catalog
        let catalog_path = dir.join("catalog.json");
        info!("Loading catalog from {:?}", catalog_path);

        let catalog_json = fs::read_to_string(&catalog_path)
            .with_context(|| format!("Failed to read {:?}", catalog_path))?;

        let catalog_json = substitute_placeholders(&catalog_json, base_url, s3_base_url);

        let root: StacCatalogRoot =
            serde_json::from_str(&catalog_json).with_context(|| "Failed to parse catalog.json")?;

        info!("Loaded catalog: {} (STAC {})", root.id, stac::STAC_VERSION);

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
                    let json = substitute_placeholders(&json, base_url, s3_base_url);

                    match serde_json::from_str::<StacCollection>(&json) {
                        Ok(collection) => {
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

        // Load all items from NDJSON (one item per line)
        let items_path = dir.join("items.ndjson");
        let items = if items_path.exists() {
            info!("Loading items from {:?}", items_path);

            let file = fs::File::open(&items_path)
                .with_context(|| format!("Failed to open {:?}", items_path))?;
            let reader = std::io::BufReader::new(file);

            let mut items = Vec::new();
            for (line_num, line) in reader.lines().enumerate() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }

                let line = substitute_placeholders(&line, base_url, s3_base_url);

                match serde_json::from_str::<StacItem>(&line) {
                    Ok(item) => items.push(item),
                    Err(e) => {
                        warn!("Failed to parse item at line {}: {}", line_num + 1, e);
                    }
                }
            }
            items
        } else {
            warn!("No items.ndjson found");
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

/// Extension trait for StacItem to access common properties
pub trait ItemExt {
    /// Get a custom property from additional_fields
    fn get_property(&self, key: &str) -> Option<&Value>;
    /// Get the bounding box as a 4-element array [west, south, east, north]
    fn bbox_array(&self) -> Option<[f64; 4]>;
}

impl ItemExt for StacItem {
    fn get_property(&self, key: &str) -> Option<&Value> {
        self.properties.additional_fields.get(key)
    }

    fn bbox_array(&self) -> Option<[f64; 4]> {
        self.bbox.as_ref().map(|b| match b {
            stac::Bbox::TwoDimensional(arr) => *arr,
            stac::Bbox::ThreeDimensional(arr) => [arr[0], arr[1], arr[3], arr[4]],
        })
    }
}

/// Helper trait to access parsed datetime values from Properties
pub trait PropertiesDatetimeExt {
    fn start_datetime(&self) -> Option<chrono::DateTime<chrono::Utc>>;
    fn end_datetime(&self) -> Option<chrono::DateTime<chrono::Utc>>;
    fn datetime(&self) -> Option<chrono::DateTime<chrono::Utc>>;
}

impl PropertiesDatetimeExt for StacItem {
    fn start_datetime(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        self.properties.start_datetime
    }

    fn end_datetime(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        self.properties.end_datetime
    }

    fn datetime(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        self.properties.datetime
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_dir(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("catalog-test-{}-{}", name, std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("catalog.json"),
            r#"{"type": "Catalog", "stac_version": "1.1.0", "id": "kites", "description": "Kites",
                "links": [{"rel": "root", "href": "${STAC_BASE_URL}/stac"}]}"#,
        )
        .unwrap();
        fs::write(
            dir.join("items.ndjson"),
            r#"{"type":"Feature","stac_version":"1.1.0","id":"box-kite","geometry":null,"properties":{"datetime":"2025-05-28T00:00:00Z"},"links":[{"rel":"root","href":"${STAC_BASE_URL}/stac"},{"rel":"collection","href":"${STAC_BASE_URL}/stac/collections/kites"}],"assets":{"sail":{"href":"${S3_BASE_URL}/assets/box-kite/sail.tif"}},"collection":"kites"}"#,
        )
        .unwrap();
        dir
    }

    fn link_href<'a>(links: &'a [stac::Link], rel: &str) -> &'a str {
        links
            .iter()
            .find(|l| l.rel == rel)
            .map(|l| l.href.as_str())
            .unwrap()
    }

    #[test]
    fn root_relative_s3_base_resolves_against_api_base() {
        let dir = fixture_dir("relative-s3");
        let catalog = StacCatalog::load_from_dir(&dir, "https://dev.example", "/s3").unwrap();
        let item = catalog.get_item("kites", "box-kite").unwrap();

        assert_eq!(link_href(&item.links, "root"), "https://dev.example/stac");
        assert_eq!(
            link_href(&catalog.root.links, "root"),
            "https://dev.example/stac"
        );
        assert_eq!(
            item.assets["sail"].href,
            "https://dev.example/s3/assets/box-kite/sail.tif"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn absolute_s3_base_is_used_as_given() {
        let dir = fixture_dir("absolute-s3");
        let catalog =
            StacCatalog::load_from_dir(&dir, "https://dev.example", "https://files.example/bucket")
                .unwrap();
        let item = catalog.get_item("kites", "box-kite").unwrap();

        assert_eq!(
            item.assets["sail"].href,
            "https://files.example/bucket/assets/box-kite/sail.tif"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

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
    }

    #[test]
    fn test_item_ext_trait() {
        let json = r#"{
            "type": "Feature",
            "stac_version": "1.1.0",
            "id": "test-item",
            "geometry": null,
            "bbox": [7.8, 46.3, 7.9, 46.4],
            "properties": {
                "datetime": null,
                "start_datetime": "2025-01-01T00:00:00Z",
                "end_datetime": "2025-12-31T23:59:59Z",
                "blatten:source": "Terradata"
            },
            "links": [],
            "assets": {},
            "collection": "test-collection"
        }"#;

        let item: StacItem = serde_json::from_str(json).unwrap();

        // Test parsed datetime access (the stac crate parses these into DateTime<Utc>)
        assert!(item.start_datetime().is_some());
        assert!(item.end_datetime().is_some());

        // Test additional fields access
        assert!(item.get_property("blatten:source").is_some());
        assert_eq!(
            item.get_property("blatten:source").and_then(|v| v.as_str()),
            Some("Terradata")
        );

        // Test bbox
        let bbox = item.bbox_array().unwrap();
        assert_eq!(bbox.len(), 4);
        assert!((bbox[0] - 7.8).abs() < 0.001);
    }
}
