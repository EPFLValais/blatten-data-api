//! STAC Catalog data structures and loading

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, fs, path::Path};
use tracing::{debug, info, warn};

/// Root STAC Catalog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StacCatalog {
    #[serde(flatten)]
    pub root: Value,
    #[serde(skip)]
    pub collections: HashMap<String, StacCollection>,
    #[serde(skip)]
    pub items: Vec<StacItem>,
}

/// STAC Collection
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
}

/// STAC Item (Feature)
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

/// STAC Provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provider {
    pub name: String,
    #[serde(default)]
    pub roles: Vec<String>,
    pub url: Option<String>,
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

/// STAC Link
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    pub rel: String,
    pub href: String,
    #[serde(rename = "type")]
    pub type_: Option<String>,
    pub title: Option<String>,
    #[serde(default)]
    pub method: Option<String>,
}

/// STAC Asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub href: String,
    #[serde(rename = "type")]
    pub type_: Option<String>,
    pub title: Option<String>,
    #[serde(default)]
    pub roles: Vec<String>,
    #[serde(rename = "file:size")]
    pub file_size: Option<i64>,
}

impl StacCatalog {
    /// Load a STAC catalog from a directory
    pub fn load_from_dir(dir: &Path, base_url: &str) -> Result<Self> {
        // Load root catalog
        let catalog_path = dir.join("catalog.json");
        info!("Loading catalog from {:?}", catalog_path);

        let catalog_json = fs::read_to_string(&catalog_path)
            .with_context(|| format!("Failed to read {:?}", catalog_path))?;

        // Replace base URL placeholder
        let catalog_json = catalog_json.replace("${STAC_BASE_URL}", base_url);
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
                    let json = json.replace("${STAC_BASE_URL}", base_url);

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

        // Load all items
        let all_items_path = dir.join("all_items.json");
        let items = if all_items_path.exists() {
            info!("Loading items from {:?}", all_items_path);

            let json = fs::read_to_string(&all_items_path)?;
            let json = json.replace("${STAC_BASE_URL}", base_url);

            let feature_collection: Value = serde_json::from_str(&json)?;

            if let Some(features) = feature_collection.get("features").and_then(|f| f.as_array()) {
                features
                    .iter()
                    .filter_map(|f| {
                        match serde_json::from_value::<StacItem>(f.clone()) {
                            Ok(item) => Some(item),
                            Err(e) => {
                                warn!("Failed to parse item: {}", e);
                                None
                            }
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            warn!("No all_items.json found, loading from individual collection files");
            Vec::new()
        };

        info!("Loaded {} items total", items.len());

        Ok(StacCatalog {
            root,
            collections,
            items,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_item() {
        let json = r#"{
            "type": "Feature",
            "stac_version": "1.0.0",
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
}
