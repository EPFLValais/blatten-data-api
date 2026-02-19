//! STAC Catalog Generator
//!
//! Generates STAC 1.1.0 catalog from CSV metadata file with geometry extraction.
//!
//! Usage:
//!     stac-gen -i metadata.csv --output stac/
//!     stac-gen -i metadata.csv --output stac/ --data-dir ./data/
//!     stac-gen validate --catalog-dir stac/

use anyhow::{Context, Result};
use csv::ReaderBuilder;
use chrono::{NaiveDate, Utc};
use clap::Parser;
use gdal::spatial_ref::{CoordTransform, SpatialRef};
use gdal::Dataset;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{debug, error, info, warn};
use walkdir::WalkDir;

// =============================================================================
// Configuration (loaded from YAML)
// =============================================================================

/// Top-level generator configuration loaded from config.yaml (--config flag)
#[derive(Debug, Deserialize)]
struct GenConfig {
    catalog: CatalogConfig,
    collections: HashMap<String, String>,
    #[serde(default)]
    crs_overrides: Vec<CrsOverride>,
    #[serde(default)]
    exclude_files: ExcludeFiles,
    #[serde(default)]
    sensors: HashMap<String, SensorConfig>,
    #[serde(default)]
    folder_mappings: HashMap<String, FolderMapping>,
    #[serde(default)]
    collection_titles: HashMap<String, String>,
    #[serde(default)]
    geometry_overrides: HashMap<String, GeometryOverride>,
}

/// Typed folder mapping entry — maps an item code to one or more data folders
#[derive(Debug, Deserialize)]
struct FolderMapping {
    folders: Vec<String>,
    #[serde(default)]
    file: Option<String>,
    #[serde(default)]
    pattern: Option<String>,
    #[serde(default)]
    bundle_subset: bool,
    #[serde(default)]
    collection_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CatalogConfig {
    id: String,
    title: String,
    description: String,
    license: String,
    keywords: Vec<String>,
    default_bbox: Vec<f64>,
    providers: Vec<ProviderConfig>,
    links: Vec<LinkConfig>,
}

#[derive(Debug, Deserialize)]
struct ProviderConfig {
    name: String,
    roles: Vec<String>,
    url: String,
}

#[derive(Debug, Deserialize)]
struct LinkConfig {
    rel: String,
    #[serde(default)]
    href: Option<String>,
    #[serde(default)]
    href_suffix: Option<String>,
    #[serde(default, rename = "type")]
    link_type: Option<String>,
    #[serde(default)]
    title: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CrsOverride {
    filename: String,
    crs_epsg: Option<i32>,
    reason: String,
}

/// Inherit geometry from another item (e.g. 3D models from same flight as orthophotos)
#[derive(Debug, Deserialize)]
struct GeometryOverride {
    from_item: String,
}

#[derive(Debug, Deserialize, Default)]
struct ExcludeFiles {
    #[serde(default)]
    exact: Vec<String>,
    #[serde(default)]
    prefixes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct SensorConfig {
    name: String,
    #[serde(default)]
    x: Option<f64>,
    #[serde(default)]
    y: Option<f64>,
    #[serde(default)]
    elevation_m: Option<f64>,
    #[serde(default)]
    items: Vec<String>,
}

fn load_config(path: &Path) -> Result<GenConfig> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {:?}", path))?;
    serde_yaml::from_str(&content)
        .with_context(|| format!("Failed to parse config YAML: {:?}", path))
}

/// Scan a directory tree for StationFactsheet PDFs and build sensor_number → path mapping.
/// Matches filenames like `StationFactsheet_07_WebcamGeoazimutLonza_A50039.pdf`.
fn discover_factsheets(dir: &Path) -> HashMap<String, PathBuf> {
    let mut map = HashMap::new();
    for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let fname = entry.file_name().to_string_lossy();
        if fname.starts_with("StationFactsheet_") && fname.ends_with(".pdf") {
            if let Some(nn) = fname
                .strip_prefix("StationFactsheet_")
                .and_then(|s| s.split('_').next())
            {
                map.insert(nn.to_string(), entry.path().to_path_buf());
            }
        }
    }
    map
}

const STAC_VERSION: &str = "1.1.0";

/// Strip leading "prefix - " from Excel field values.
/// Handles patterns like "14 - Helimap", "D - Orthophoto", "a - Overview", "1 - 23.05."
fn strip_field_prefix(s: &str) -> String {
    if let Some(idx) = s.find(" - ") {
        s[idx + 3..].to_string()
    } else {
        s.to_string()
    }
}

// =============================================================================
// Junk File Exclusion
// =============================================================================

/// Returns true if the file should be excluded from scanning/staging/archiving.
/// Uses patterns from the config's exclude_files section.
fn is_junk_file(path: &Path, exclude: &ExcludeFiles) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let name_lower = name.to_lowercase();
    if exclude.exact.iter().any(|e| name_lower == *e) {
        return true;
    }
    if exclude.prefixes.iter().any(|p| name_lower.starts_with(p)) {
        return true;
    }
    false
}

// File Overrides / Patches are loaded from config.crs_overrides


// =============================================================================
// CLI Definition
// =============================================================================

#[derive(Parser)]
#[command(name = "stac-gen")]
#[command(about = "Generate STAC 1.1.0 catalog from Excel metadata and geospatial data")]
#[command(long_about = "\
STAC catalog generator for the Blatten Data platform.

Processes Excel metadata, scans provider data folders,
extracts geometry from GeoTIFFs/LAZ files, and generates a STAC 1.1.0 compliant catalog.

When --data and --organised-dir are provided, runs the full staging pipeline:
  1. Stage assets    — symlink files from FINAL_Data into assets/<code>/
  2. Hash assets     — parallel SHA-256 of every file (incremental)
  3. Stage archives  — create archives from assets (skips unchanged)
  4. Generate STAC   — catalog with file:checksum on every asset

Expected working directory layout:
  ./config.yaml          Configuration (sensors, collections, overrides)
  ./*.csv               CSV metadata (auto-discovered if -i not given)
  ./input/               FINAL_Data directory (symlink OK)
  ./output/              Staging root (assets/ and archives/)
  ./stac/                Generated STAC catalog")]
#[command(after_long_help = "\
EXAMPLES:
  # Full pipeline with all defaults (stages + materializes + validates)
  stac-gen --validate -y

  # Keep symlinks (skip materialization)
  stac-gen --no-materialize -y

  # Override specific paths
  stac-gen -d /mnt/data/FINAL_Data --validate

ENVIRONMENT:
  RUST_LOG=debug  Enable debug logging")]
struct Cli {
    /// Input metadata file (CSV, semicolon-delimited). If not specified,
    /// auto-discovers a single .csv file in the current directory.
    #[arg(short = 'i', long, alias = "xlsx", short_alias = 'x')]
    input: Option<PathBuf>,

    /// Output directory for STAC files
    #[arg(short, long, default_value = "stac")]
    output: PathBuf,

    /// Base URL for STAC links
    #[arg(short, long, default_value = "https://blatten-data.epfl.ch")]
    base_url: String,

    /// S3 base URL for file assets
    #[arg(long, default_value = "/s3")]
    s3_base_url: String,

    /// Data directory containing provider folders (auto-discovered).
    /// This is the FINAL_Data directory with geospatial files for geometry extraction.
    #[arg(short = 'd', long, default_value = "input")]
    data: Option<PathBuf>,

    /// Config YAML file with sensor locations, folder mappings, CRS overrides, etc.
    #[arg(short = 'c', long, default_value = "config.yaml")]
    config: PathBuf,

    /// Number of parallel threads (for hashing, archiving, geometry extraction)
    #[arg(long, default_value_t = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))]
    threads: usize,

    /// Run validation after generation (exits with code 1 if errors found)
    #[arg(long)]
    validate: bool,

    /// Output directory for organized data. Creates assets/ and archives/ subdirectories.
    #[arg(long, default_value = "output")]
    organised_dir: Option<PathBuf>,

    /// Skip confirmation prompt (for automated/CI use)
    #[arg(short = 'y', long)]
    yes: bool,

    /// Skip materialization (keep symlinks instead of converting to real copies).
    /// By default, the pipeline materializes symlinks after staging.
    #[arg(long)]
    no_materialize: bool,

    /// Skip CRC32 integrity checking of zip archives (faster). Content validation
    /// (file presence + size comparison) always runs regardless of this flag.
    #[arg(long)]
    skip_zip_validation: bool,

    /// Directory containing StationFactsheet PDFs to inject into data products.
    /// PDFs matching StationFactsheet_<NN>_*.pdf are copied into each item's assets dir.
    #[arg(long, default_value = "Documentation")]
    factsheets_dir: PathBuf,
}

/// Auto-discover a metadata file in the current directory.
/// Prefers .csv files; falls back to .xlsx if no CSV found.
/// Returns an error if zero or more than one candidate files are found.
fn discover_input() -> Result<PathBuf> {
    let all_files: Vec<PathBuf> = fs::read_dir(".")?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .map(|e| e.path())
        .collect();

    // Prefer CSV files
    let csv_files: Vec<&PathBuf> = all_files.iter()
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("csv"))
        .collect();

    if csv_files.len() == 1 {
        return Ok(csv_files[0].clone());
    }
    if csv_files.len() > 1 {
        let names: Vec<String> = csv_files.iter().map(|p| p.display().to_string()).collect();
        anyhow::bail!("Found {} .csv files in current directory: {}. Provide the correct one with -i.", csv_files.len(), names.join(", "));
    }

    // Fall back to XLSX
    let xlsx_files: Vec<&PathBuf> = all_files.iter()
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("xlsx"))
        .collect();

    match xlsx_files.len() {
        0 => anyhow::bail!("No .csv or .xlsx file found in current directory. Provide one with -i or place a .csv file here."),
        1 => Ok(xlsx_files[0].clone()),
        n => {
            let names: Vec<String> = xlsx_files.iter().map(|p| p.display().to_string()).collect();
            anyhow::bail!("Found {} .xlsx files in current directory: {}. Provide the correct one with -i.", n, names.join(", "));
        }
    }
}

// =============================================================================
// Data Structures
// =============================================================================

/// Result of geometry extraction
#[derive(Debug, Clone)]
enum ExtractionResult {
    /// Geometry extracted from geospatial file
    Extracted { source: String, bbox: Vec<f64>, geometry: serde_json::Value, override_reason: Option<String> },
    /// Manual coordinates from sensor_locations.json
    Manual { sensor_name: String, lon: f64, lat: f64, bbox: Vec<f64>, geometry: serde_json::Value },
    /// No geometry available
    NoGeometry { reason: String },
    /// Extraction failed with error
    Failed { error: String },
}

/// File information with geometry (for assets)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileInfo {
    path: PathBuf,
    size: u64,
    /// WGS84 bbox for STAC compatibility [min_lon, min_lat, max_lon, max_lat]
    bbox: Option<Vec<f64>>,
    /// WGS84 geometry as GeoJSON
    geometry: Option<serde_json::Value>,
    /// LV95 bbox for Projection Extension [min_x, min_y, max_x, max_y]
    #[serde(default)]
    bbox_lv95: Option<Vec<f64>>,
    /// LV95 geometry as GeoJSON for Projection Extension
    #[serde(default)]
    geometry_lv95: Option<serde_json::Value>,
    /// SHA-256 hash (hex, no multihash prefix) — populated from manifest
    #[serde(default)]
    hash: Option<String>,
}

/// Scanned folder information
#[derive(Debug, Clone)]
struct ScannedFolder {
    code: String,
    path: PathBuf,
    provider: String,
    files: Vec<PathBuf>,
    archive_path: Option<PathBuf>,
    archive_size: Option<u64>,
    /// Nested child codes found within this folder (e.g., 01Aa01 inside 01Aa00)
    #[allow(dead_code)]
    nested_codes: Vec<String>,
    /// True if this is a single file (not a folder) — no archive needed
    is_single_file: bool,
}

// =============================================================================
// Data Manifest Structures (for integrated pipeline)
// =============================================================================

/// Hash algorithm for file verification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum HashAlgorithm {
    Sha256,
    Xxhash,
}

impl std::str::FromStr for HashAlgorithm {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sha256" => Ok(HashAlgorithm::Sha256),
            "xxhash" | "xxh64" => Ok(HashAlgorithm::Xxhash),
            _ => Err(format!("Unknown hash algorithm: {}", s)),
        }
    }
}

impl std::fmt::Display for HashAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HashAlgorithm::Sha256 => write!(f, "sha256"),
            HashAlgorithm::Xxhash => write!(f, "xxhash"),
        }
    }
}

/// Link mode for staging files
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinkMode {
    Symlink,
    Hardlink,
    Copy,
}

impl std::str::FromStr for LinkMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "symlink" | "sym" => Ok(LinkMode::Symlink),
            "hardlink" | "hard" => Ok(LinkMode::Hardlink),
            "copy" | "cp" => Ok(LinkMode::Copy),
            _ => Err(format!("Unknown link mode: {}. Use: symlink, hardlink, or copy", s)),
        }
    }
}

impl std::fmt::Display for LinkMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinkMode::Symlink => write!(f, "symlink"),
            LinkMode::Hardlink => write!(f, "hardlink"),
            LinkMode::Copy => write!(f, "copy"),
        }
    }
}

/// File manifest entry with hash for verification
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileManifestEntry {
    /// Path in assets dir: "assets/13Da01/orthophoto.tif"
    asset_path: String,
    /// Path in archive: "13Da01/orthophoto.tif"
    archive_path: String,
    /// Source path: "Terradata/13Da01/orthophoto.tif"
    source_path: String,
    /// File size in bytes
    size: u64,
    /// SHA-256 or xxHash checksum
    hash: String,
    /// Item code
    code: String,
    /// Provider
    provider: String,
    /// Last modified (Unix timestamp)
    mtime: i64,
}

/// Archive manifest entry with content fingerprint for incremental builds
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArchiveManifestEntry {
    path: String,
    size: u64,
    /// SHA-256 hash of the archive file itself
    hash: String,
    /// Hash of sorted (filename, size, file_hash) — changes if any file changes
    content_fingerprint: String,
    file_count: usize,
    /// True if archive uses ZIP64 extensions (individual file >= 4GB or archive >= 4GB)
    #[serde(default)]
    is_zip64: bool,
}

/// Complete data manifest with all file hashes
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DataManifest {
    /// Manifest format version
    version: u32,
    /// ISO 8601 timestamp when generated
    generated_at: String,
    /// Hash algorithm used
    hash_algorithm: HashAlgorithm,
    /// Total number of files
    total_files: usize,
    /// Total size in bytes
    total_size: u64,
    /// Files keyed by asset_path
    files: BTreeMap<String, FileManifestEntry>,
    /// Archives created (code -> archive manifest entry)
    archives: BTreeMap<String, ArchiveManifestEntry>,
}

impl DataManifest {
    fn new(algorithm: HashAlgorithm) -> Self {
        Self {
            version: 1,
            generated_at: Utc::now().to_rfc3339(),
            hash_algorithm: algorithm,
            total_files: 0,
            total_size: 0,
            files: BTreeMap::new(),
            archives: BTreeMap::new(),
        }
    }

    fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        serde_json::from_str(&content).context("Failed to parse manifest.json")
    }

    /// Save manifest to file
    fn save(&self, path: &Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Update totals from files map
    fn update_totals(&mut self) {
        self.total_files = self.files.len();
        self.total_size = self.files.values().map(|f| f.size).sum();
    }
}

// =============================================================================
// Geometry Cache
// =============================================================================

const GEOMETRY_CACHE_VERSION: u32 = 1;

/// Cached geometry result for a single file
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedFileGeometry {
    size: u64,
    bbox: Option<Vec<f64>>,
    geometry: Option<serde_json::Value>,
    bbox_lv95: Option<Vec<f64>>,
    geometry_lv95: Option<serde_json::Value>,
}

/// Geometry cache for incremental runs — avoids re-running GDAL on unchanged files
#[derive(Debug, Serialize, Deserialize)]
struct GeometryCache {
    version: u32,
    generated_at: String,
    /// Hash of CRS overrides — cache invalidated when overrides change
    crs_overrides_hash: String,
    /// Keyed by absolute file path
    entries: BTreeMap<String, CachedFileGeometry>,
}

impl GeometryCache {
    fn new(crs_overrides_hash: String) -> Self {
        Self {
            version: GEOMETRY_CACHE_VERSION,
            generated_at: Utc::now().to_rfc3339(),
            crs_overrides_hash,
            entries: BTreeMap::new(),
        }
    }

    fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        serde_json::from_str(&content).context("Failed to parse geometry_cache.json")
    }

    fn save(&self, path: &Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
}

/// Deterministic hash of CRS overrides so geometry cache auto-invalidates when overrides change
fn hash_crs_overrides(overrides: &[CrsOverride]) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    for o in overrides {
        hasher.update(o.filename.as_bytes());
        hasher.update(b"|");
        if let Some(epsg) = o.crs_epsg {
            hasher.update(epsg.to_string().as_bytes());
        }
        hasher.update(b"|");
        hasher.update(o.reason.as_bytes());
        hasher.update(b"\n");
    }
    format!("{:x}", hasher.finalize())
}

// =============================================================================
// Incremental Pipeline Helpers
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArchiveAction {
    Create,    // No archive on disk
    Adopt,     // Archive exists, no manifest fingerprint
    Rebuild,   // Archive exists, fingerprint changed
    Unchanged, // Archive exists, fingerprint matches
}

fn classify_archive_action(
    code: &str,
    archive_path: &Path,
    current_fingerprint: &str,
    existing_manifest: Option<&DataManifest>,
) -> ArchiveAction {
    if !archive_path.exists() { return ArchiveAction::Create }
    let Some(manifest) = existing_manifest else { return ArchiveAction::Adopt };
    match manifest.archives.get(code) {
        None => ArchiveAction::Adopt,
        Some(entry) if entry.content_fingerprint == current_fingerprint => ArchiveAction::Unchanged,
        Some(_) => ArchiveAction::Rebuild,
    }
}

/// Collection definition
#[derive(Debug, Clone)]
struct CollectionDef {
    id: String,
    title: String,
    description: String,
}

// ProductID letter → collection slug map is now loaded from config.collections

/// Extract collection name from product_type (part before first " - ").
/// "Radar Velocity - DTM" → "Radar Velocity"
/// "Orthophoto" → "Orthophoto"
fn collection_name_from_product_type(product_type: &str) -> &str {
    product_type.split(" - ").next().unwrap_or(product_type).trim()
}

/// Build collection definitions from parsed items using the config collection map.
fn build_collections(
    items: &[ItemMetadata],
    collection_map: &HashMap<String, String>,
    title_overrides: &HashMap<String, String>,
) -> HashMap<String, CollectionDef> {
    let mut collections: HashMap<String, CollectionDef> = HashMap::new();

    for item in items {
        if let Some(ref pid) = item.product_id {
            if let Some(coll_slug) = collection_map.get(pid.as_str()) {
                let entry = collections.entry(coll_slug.to_string()).or_insert_with(|| {
                    CollectionDef {
                        id: coll_slug.to_string(),
                        title: String::new(),
                        description: String::new(),
                    }
                });
                if entry.title.is_empty() {
                    if let Some(ref pt) = item.product_type {
                        if !pt.is_empty() {
                            entry.title = collection_name_from_product_type(pt).to_string();
                        }
                    }
                }
            }
        }
    }

    // Ensure all mapped slugs have entries even if no items matched
    for coll_slug in collection_map.values() {
        collections.entry(coll_slug.to_string()).or_insert_with(|| {
            CollectionDef {
                id: coll_slug.to_string(),
                title: coll_slug.replace('-', " "),
                description: String::new(),
            }
        });
    }

    // Apply explicit title overrides from config
    for (slug, title) in title_overrides {
        if let Some(entry) = collections.get_mut(slug) {
            entry.title = title.clone();
        }
    }

    collections
}

// get_code_to_file() removed — staging pipeline handles all archive mapping

// =============================================================================
// Sensor Locations (for items without extractable geometry)
// =============================================================================

/// Sensor location with manual coordinates (supports both LV95 and WGS84)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SensorLocation {
    name: Option<String>,
    /// LV95 X coordinate (easting)
    x: Option<f64>,
    /// LV95 Y coordinate (northing)
    y: Option<f64>,
    /// WGS84 longitude (derived from x,y or provided directly)
    #[serde(default)]
    lon: Option<f64>,
    /// WGS84 latitude (derived from x,y or provided directly)
    #[serde(default)]
    lat: Option<f64>,
    elevation_m: Option<f64>,
    #[serde(default)]
    items: Vec<String>,
}

/// Convert LV95 (EPSG:2056) coordinates to WGS84 (EPSG:4326) using GDAL
/// Returns (longitude, latitude) in GeoJSON order
fn lv95_to_wgs84(x: f64, y: f64) -> Option<(f64, f64)> {
    let source_srs = SpatialRef::from_epsg(2056).ok()?;
    let mut target_srs = SpatialRef::from_epsg(4326).ok()?;

    // Force traditional GIS axis order (lon, lat) instead of EPSG standard (lat, lon)
    // This ensures compatibility with GeoJSON which requires [longitude, latitude]
    target_srs.set_axis_mapping_strategy(gdal::spatial_ref::AxisMappingStrategy::TraditionalGisOrder);

    let transform = CoordTransform::new(&source_srs, &target_srs).ok()?;

    let mut xs = [x];
    let mut ys = [y];
    let mut zs = [0.0];

    transform.transform_coords(&mut xs, &mut ys, &mut zs).ok()?;

    // xs[0] = longitude, ys[0] = latitude (due to TraditionalGisOrder)
    Some((xs[0], ys[0]))
}

/// Build sensor location lookup from config sensors, converting LV95 to WGS84
fn load_sensor_locations_from_config(sensors: &HashMap<String, SensorConfig>) -> HashMap<String, SensorLocation> {
    let mut lookup = HashMap::new();

    for (sensor_id, cfg) in sensors {
        let mut sensor = SensorLocation {
            name: Some(cfg.name.clone()),
            x: cfg.x,
            y: cfg.y,
            lon: None,
            lat: None,
            elevation_m: cfg.elevation_m,
            items: cfg.items.clone(),
        };

        // Convert LV95 to WGS84 if x,y are available
        if let (Some(x), Some(y)) = (sensor.x, sensor.y) {
            if let Some((lon, lat)) = lv95_to_wgs84(x, y) {
                sensor.lon = Some(lon);
                sensor.lat = Some(lat);
            }
        }

        for item_code in &sensor.items {
            lookup.insert(item_code.clone(), sensor.clone());
        }
        // Also store by sensor_id for direct lookup
        lookup.insert(sensor_id.clone(), sensor);
    }

    lookup
}


// =============================================================================
// Geometry Extraction
// =============================================================================

/// Extracted geometry information
#[derive(Debug, Clone)]
struct ExtractedGeometry {
    /// WGS84 bbox [min_lon, min_lat, max_lon, max_lat]
    bbox: Vec<f64>,
    /// WGS84 geometry as GeoJSON
    geometry: serde_json::Value,
    /// Source description for debugging
    source: String,
    /// LV95 bbox [min_x, min_y, max_x, max_y] for Projection Extension
    bbox_lv95: Option<Vec<f64>>,
    /// LV95 geometry as GeoJSON for Projection Extension
    geometry_lv95: Option<serde_json::Value>,
}

/// Transform coordinates from a source EPSG to WGS84 using GDAL
/// Returns bbox in GeoJSON order: [min_lon, min_lat, max_lon, max_lat]
fn transform_to_wgs84(minx: f64, miny: f64, maxx: f64, maxy: f64, epsg: i32) -> Option<Vec<f64>> {
    let source_srs = SpatialRef::from_epsg(epsg as u32).ok()?;
    let mut target_srs = SpatialRef::from_epsg(4326).ok()?;

    // Force traditional GIS axis order (lon, lat) instead of EPSG standard (lat, lon)
    target_srs.set_axis_mapping_strategy(gdal::spatial_ref::AxisMappingStrategy::TraditionalGisOrder);

    let transform = CoordTransform::new(&source_srs, &target_srs).ok()?;

    // Transform all four corners for accuracy
    let mut xs = [minx, maxx, maxx, minx];
    let mut ys = [miny, miny, maxy, maxy];
    let mut zs = [0.0; 4];

    transform.transform_coords(&mut xs, &mut ys, &mut zs).ok()?;

    // xs = longitudes, ys = latitudes (due to TraditionalGisOrder)
    Some(vec![
        xs.iter().cloned().fold(f64::INFINITY, f64::min),   // min_lon (west)
        ys.iter().cloned().fold(f64::INFINITY, f64::min),   // min_lat (south)
        xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max), // max_lon (east)
        ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max), // max_lat (north)
    ])
}

/// Extract EPSG code from WKT or other CRS representation
fn extract_epsg_from_crs(crs: &str) -> Option<String> {
    // Look for EPSG code patterns
    let patterns = [
        r#"EPSG[",:\s]+(\d+)"#,
        r#"AUTHORITY\["EPSG","(\d+)"\]"#,
        r#"urn:ogc:def:crs:EPSG::(\d+)"#,
    ];

    for pattern in &patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            if let Some(caps) = re.captures(crs) {
                if let Some(code) = caps.get(1) {
                    return Some(format!("EPSG:{}", code.as_str()));
                }
            }
        }
    }

    // Check for common Swiss CRS
    if crs.contains("CH1903+") || crs.contains("LV95") || crs.contains("2056") {
        return Some("EPSG:2056".to_string());
    }
    if crs.contains("CH1903") && !crs.contains("CH1903+") {
        return Some("EPSG:21781".to_string());
    }

    None
}

/// Extract geometry from a GeoTIFF file using GDAL
/// Returns (ExtractedGeometry, Option<override_reason>) - the second value indicates if an override was applied
/// ExtractedGeometry includes both WGS84 (for STAC item) and LV95 (for Projection Extension) coordinates
fn extract_geotiff_geometry(path: &Path, crs_overrides: &[CrsOverride]) -> (Option<ExtractedGeometry>, Option<String>) {
    let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();

    // Check if there's an override for this file
    let file_override = crs_overrides.iter().find(|o| o.filename == filename);

    let dataset = match Dataset::open(path) {
        Ok(ds) => ds,
        Err(_) => return (None, None),
    };

    // Get geotransform: [origin_x, pixel_width, rotation_x, origin_y, rotation_y, pixel_height]
    let gt = match dataset.geo_transform() {
        Ok(gt) => gt,
        Err(_) => return (None, None),
    };
    let (width, height) = dataset.raster_size();

    // Calculate corners from geotransform (native CRS coordinates)
    let minx = gt[0];
    let maxy = gt[3];
    let maxx = minx + (width as f64 * gt[1]);
    let miny = maxy + (height as f64 * gt[5]); // gt[5] is typically negative

    // Ensure correct ordering
    let (minx, maxx) = (minx.min(maxx), minx.max(maxx));
    let (miny, maxy) = (miny.min(maxy), miny.max(maxy));

    // Determine the source EPSG code
    let source_epsg: Option<i32> = if let Some(ovr) = file_override {
        ovr.crs_epsg
    } else if let Ok(spatial_ref) = dataset.spatial_ref() {
        if let Ok(wkt) = spatial_ref.to_wkt() {
            extract_epsg_from_crs(&wkt)
                .and_then(|s| s.strip_prefix("EPSG:").and_then(|n| n.parse().ok()))
        } else {
            None
        }
    } else if looks_like_lv95(minx, miny, maxx, maxy) {
        Some(2056) // Swiss LV95
    } else {
        None
    };

    // If we have a CRS override, use that directly
    if let Some(ovr) = file_override {
        if let Some(epsg) = ovr.crs_epsg {
            if let Some(bbox_wgs84) = transform_to_wgs84(minx, miny, maxx, maxy, epsg) {
                if is_valid_wgs84_bbox(&bbox_wgs84) {
                    let geometry_wgs84 = bbox_to_polygon(&bbox_wgs84);
                    let source = format!("GeoTIFF: {} (OVERRIDE: EPSG:{})", filename, epsg);

                    // Store LV95 coordinates if source is LV95
                    let (bbox_lv95, geometry_lv95) = if epsg == 2056 {
                        let native_bbox = vec![minx, miny, maxx, maxy];
                        (Some(native_bbox.clone()), Some(bbox_to_polygon(&native_bbox)))
                    } else {
                        (None, None)
                    };

                    return (
                        Some(ExtractedGeometry {
                            bbox: bbox_wgs84,
                            geometry: geometry_wgs84,
                            source,
                            bbox_lv95,
                            geometry_lv95,
                        }),
                        Some(ovr.reason.to_string()),
                    );
                }
            }
        }
    }

    // Try to transform using embedded CRS
    let (bbox_wgs84, detected_epsg) = if let Ok(spatial_ref) = dataset.spatial_ref() {
        let mut target_srs = match SpatialRef::from_epsg(4326) {
            Ok(srs) => srs,
            Err(_) => return (None, None),
        };
        // Force traditional GIS axis order (lon, lat) for GeoJSON compatibility
        target_srs.set_axis_mapping_strategy(gdal::spatial_ref::AxisMappingStrategy::TraditionalGisOrder);

        let (maybe_bbox, epsg) = if let Ok(transform) = CoordTransform::new(&spatial_ref, &target_srs) {
            // Transform all four corners
            let mut xs = [minx, maxx, maxx, minx];
            let mut ys = [miny, miny, maxy, maxy];
            let mut zs = [0.0; 4];

            if transform.transform_coords(&mut xs, &mut ys, &mut zs).is_ok() {
                // xs = longitudes, ys = latitudes (due to TraditionalGisOrder)
                (Some(vec![
                    xs.iter().cloned().fold(f64::INFINITY, f64::min),   // min_lon
                    ys.iter().cloned().fold(f64::INFINITY, f64::min),   // min_lat
                    xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max), // max_lon
                    ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max), // max_lat
                ]), source_epsg)
            } else {
                (None, None)
            }
        } else {
            // Try extracting EPSG from WKT
            if let Ok(wkt) = spatial_ref.to_wkt() {
                if let Some(epsg_str) = extract_epsg_from_crs(&wkt) {
                    if let Ok(epsg) = epsg_str.strip_prefix("EPSG:").unwrap_or("").parse::<i32>() {
                        if epsg == 4326 {
                            (Some(vec![minx, miny, maxx, maxy]), Some(4326))
                        } else {
                            (transform_to_wgs84(minx, miny, maxx, maxy, epsg), Some(epsg))
                        }
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            }
        };

        // If transformation succeeded and looks valid, use it
        if let Some(ref b) = maybe_bbox {
            if is_valid_wgs84_bbox(b) {
                (maybe_bbox, epsg)
            } else if looks_like_lv95(minx, miny, maxx, maxy) {
                // Transformed coordinates are invalid but raw bounds look like LV95
                info!(
                    "GeoTIFF {}: transform produced invalid WGS84 (bbox: {:?}), falling back to LV95. Bounds: ({:.0}, {:.0}) - ({:.0}, {:.0})",
                    filename, b, minx, miny, maxx, maxy
                );
                (transform_to_wgs84(minx, miny, maxx, maxy, 2056), Some(2056))
            } else {
                // Transformed coordinates are invalid - CRS issue
                warn!(
                    "GeoTIFF {}: transformed coordinates are invalid (bbox: {:?}). Fix the CRS in the source file or add an override.",
                    filename, b
                );
                (None, None)
            }
        } else if looks_like_lv95(minx, miny, maxx, maxy) {
            // CRS exists but transform failed — fallback to LV95 if bounds match
            info!(
                "GeoTIFF {}: CRS transform failed, falling back to LV95. Bounds: ({:.0}, {:.0}) - ({:.0}, {:.0})",
                filename, minx, miny, maxx, maxy
            );
            (transform_to_wgs84(minx, miny, maxx, maxy, 2056), Some(2056))
        } else {
            (None, None)
        }
    } else {
        // No valid CRS — fall back to LV95 if bounds match
        if looks_like_lv95(minx, miny, maxx, maxy) {
            debug!(
                "GeoTIFF {}: no CRS, assuming LV95 (EPSG:2056). Bounds: ({:.0}, {:.0}) - ({:.0}, {:.0})",
                filename, minx, miny, maxx, maxy
            );
            (transform_to_wgs84(minx, miny, maxx, maxy, 2056), Some(2056))
        } else {
            warn!(
                "GeoTIFF {}: no valid CRS. Fix the CRS metadata in the source file. Bounds: ({:.6}, {:.6}) - ({:.6}, {:.6})",
                path.file_name().unwrap_or_default().to_string_lossy(),
                minx, miny, maxx, maxy
            );
            (None, None)
        }
    };

    // Return None if no valid bbox could be computed
    let bbox_wgs84 = match bbox_wgs84 {
        Some(b) => b,
        None => return (None, None),
    };

    // Final validation
    if !is_valid_wgs84_bbox(&bbox_wgs84) {
        return (None, None);
    }

    let geometry_wgs84 = bbox_to_polygon(&bbox_wgs84);
    let source = format!("GeoTIFF: {}", filename);

    // Store LV95 coordinates if source CRS is LV95
    let (bbox_lv95, geometry_lv95) = if detected_epsg == Some(2056) {
        let native_bbox = vec![minx, miny, maxx, maxy];
        (Some(native_bbox.clone()), Some(bbox_to_polygon(&native_bbox)))
    } else {
        (None, None)
    };

    (Some(ExtractedGeometry {
        bbox: bbox_wgs84,
        geometry: geometry_wgs84,
        source,
        bbox_lv95,
        geometry_lv95,
    }), None)
}

/// Extract geometry from a LAZ/LAS file
/// Returns ExtractedGeometry with both WGS84 and LV95 coordinates when applicable
fn extract_las_geometry(path: &Path) -> Option<ExtractedGeometry> {
    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);
    let las_reader = las::Reader::new(reader).ok()?;
    let header = las_reader.header();

    let bounds = header.bounds();
    let minx = bounds.min.x;
    let miny = bounds.min.y;
    let maxx = bounds.max.x;
    let maxy = bounds.max.y;

    // Try to get CRS from VLRs
    let crs_string = header.vlrs().iter().find_map(|vlr| {
        // Look for GeoKeyDirectoryTag or WKT VLR
        if vlr.record_id == 34735 || vlr.record_id == 2112 {
            String::from_utf8(vlr.data.clone()).ok()
        } else {
            None
        }
    });

    // Default to Swiss LV95 if no CRS found (common for this dataset)
    let epsg_str = crs_string
        .as_ref()
        .and_then(|s| extract_epsg_from_crs(s))
        .unwrap_or_else(|| "EPSG:2056".to_string());

    let epsg: i32 = epsg_str.strip_prefix("EPSG:").and_then(|s| s.parse().ok()).unwrap_or(2056);

    let bbox_wgs84 = if epsg == 4326 {
        vec![minx, miny, maxx, maxy]
    } else {
        transform_to_wgs84(minx, miny, maxx, maxy, epsg)?
    };

    // Validate the transformed coordinates are reasonable
    if !is_valid_wgs84_bbox(&bbox_wgs84) {
        return None; // CRS transformation failed or produced garbage
    }

    let geometry_wgs84 = bbox_to_polygon(&bbox_wgs84);

    // Store LV95 coordinates if source CRS is LV95
    let (bbox_lv95, geometry_lv95) = if epsg == 2056 {
        let native_bbox = vec![minx, miny, maxx, maxy];
        (Some(native_bbox.clone()), Some(bbox_to_polygon(&native_bbox)))
    } else {
        (None, None)
    };

    Some(ExtractedGeometry {
        bbox: bbox_wgs84,
        geometry: geometry_wgs84,
        source: format!("LAZ/LAS: {} (EPSG:{})", path.file_name()?.to_string_lossy(), epsg),
        bbox_lv95,
        geometry_lv95,
    })
}

/// Convert bbox to GeoJSON Polygon
fn bbox_to_polygon(bbox: &[f64]) -> serde_json::Value {
    let (west, south, east, north) = (bbox[0], bbox[1], bbox[2], bbox[3]);
    serde_json::json!({
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south]
        ]]
    })
}

/// Validate that bbox coordinates are in valid WGS84 range for Switzerland area
/// Returns true if coordinates look reasonable (roughly Europe/Switzerland)
fn is_valid_wgs84_bbox(bbox: &[f64]) -> bool {
    if bbox.len() != 4 {
        return false;
    }
    let (west, south, east, north) = (bbox[0], bbox[1], bbox[2], bbox[3]);

    // Check valid WGS84 ranges
    let valid_lon = west >= -180.0 && west <= 180.0 && east >= -180.0 && east <= 180.0;
    let valid_lat = south >= -90.0 && south <= 90.0 && north >= -90.0 && north <= 90.0;

    // Check ordering
    let valid_order = west <= east && south <= north;

    // Check roughly in European/Swiss region (lon: 5-11, lat: 45-48)
    // Use wider bounds to be safe
    let roughly_swiss = west >= -20.0 && east <= 30.0 && south >= 30.0 && north <= 60.0;

    valid_lon && valid_lat && valid_order && roughly_swiss
}

/// Check if coordinates look like Swiss LV95 (EPSG:2056)
/// LV95 has X (Easting) ~2.4M-2.9M and Y (Northing) ~1.0M-1.4M
fn looks_like_lv95(minx: f64, miny: f64, maxx: f64, maxy: f64) -> bool {
    let x_in_range = minx >= 2_400_000.0 && maxx <= 2_900_000.0;
    let y_in_range = miny >= 1_000_000.0 && maxy <= 1_400_000.0;
    x_in_range && y_in_range
}

/// Create a point geometry with a small buffer for bbox
fn point_to_geometry(lon: f64, lat: f64, elevation: Option<f64>) -> (Vec<f64>, serde_json::Value) {
    let buffer = 0.001; // ~100m buffer
    let bbox = vec![lon - buffer, lat - buffer, lon + buffer, lat + buffer];
    let coords: serde_json::Value = if let Some(elev) = elevation {
        serde_json::json!([lon, lat, elev])
    } else {
        serde_json::json!([lon, lat])
    };
    let geometry = serde_json::json!({
        "type": "Point",
        "coordinates": coords
    });
    (bbox, geometry)
}

/// Parsed item metadata from XLSX
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ItemMetadata {
    code: String,
    product_id: Option<String>,
    sensor_id: Option<String>,
    dataset_id: Option<String>,
    bundle_id: Option<String>,
    sensor: Option<String>,
    product_type: Option<String>,
    dataset: Option<String>,
    bundle: Option<String>,
    description: Option<String>,
    format: Option<String>,
    technical_info: Option<String>,
    processing_level: Option<i32>,
    phase: Option<String>,
    date_first: Option<String>,
    date_last: Option<String>,
    continued: bool,
    frequency: Option<String>,
    location: Option<String>,
    source: Option<String>,
    additional_remarks: Option<String>,
    storage_mb: Option<f64>,
    internal_commentary: Option<String>,
    // Derived
    collection_id: Option<String>,
    geometry: Option<serde_json::Value>,
    bbox: Option<Vec<f64>>,
    archive_file: Option<String>,
    archive_size: Option<u64>,
    // File tracking (for hierarchical items)
    #[serde(default)]
    files: Vec<FileInfo>,
    #[serde(default)]
    file_count: usize,
    /// Base folder path for computing relative file paths
    #[serde(skip)]
    folder_path: Option<PathBuf>,
    /// SHA-256 hash of the archive file (populated from manifest)
    #[serde(default)]
    archive_hash: Option<String>,
    /// True if archive uses ZIP64 extensions (populated from manifest)
    #[serde(default)]
    archive_is_zip64: bool,
    /// True if this item is a single standalone file (no archive needed)
    #[serde(default)]
    is_single_file: bool,
}

/// Validation issue
#[derive(Debug, Serialize)]
struct ValidationIssue {
    item_id: String,
    severity: String,
    message: String,
}

// =============================================================================
// CSV Parsing
// =============================================================================

/// Parse European date string (DD.MM.YYYY) to ISO date (YYYY-MM-DD)
fn parse_european_date(s: &str) -> Option<String> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    NaiveDate::parse_from_str(s, "%d.%m.%Y")
        .ok()
        .map(|d| d.format("%Y-%m-%d").to_string())
}

/// Extract a non-empty trimmed string from a CSV field, returning None for empty/NaN
fn csv_field(s: &str) -> Option<String> {
    let s = s.trim();
    if s.is_empty() || s == "NaN" {
        None
    } else {
        Some(s.to_string())
    }
}

/// Parse the semicolon-delimited CSV metadata file (17 columns).
///
/// Column layout (0-indexed):
///  0: Code, 1: Product ID, 2: Sensor ID, 3: Dataset ID, 4: Bundle ID,
///  5: Sensor, 6: ProductType, 7: Dataset, 8: Bundle, 9: Description,
/// 10: Format, 11: Additional information, 12: Phase,
/// 13: Date first (provided), 14: Date last (provided), 15: Frequency,
/// 16: Source / Operator
fn parse_csv(path: &Path, collection_map: &HashMap<String, String>) -> Result<Vec<ItemMetadata>> {
    // Read raw bytes and convert to UTF-8 (handles Latin-1/ISO-8859-1 encoded files)
    let raw = fs::read(path)
        .with_context(|| format!("Failed to read CSV: {:?}", path))?;
    let text = if std::str::from_utf8(&raw).is_ok() {
        String::from_utf8(raw).unwrap()
    } else {
        // Assume ISO-8859-1: every byte maps to a Unicode code point
        raw.iter().map(|&b| b as char).collect()
    };

    let mut rdr = ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(true)
        .flexible(true)
        .from_reader(text.as_bytes());

    // Validate headers
    let headers = rdr.headers()
        .with_context(|| "Failed to read CSV headers")?
        .clone();
    if headers.get(0).map(|h| h.trim()) != Some("Code") {
        anyhow::bail!(
            "Unexpected CSV header: expected first column 'Code', got '{}'",
            headers.get(0).unwrap_or("(empty)")
        );
    }

    let mut items = Vec::new();

    for (row_idx, result) in rdr.records().enumerate() {
        let record = result
            .with_context(|| format!("Failed to read CSV row {}", row_idx + 2))?;

        let code = match csv_field(record.get(0).unwrap_or("")) {
            Some(c) => c,
            None => continue,
        };

        let product_id = csv_field(record.get(1).unwrap_or(""));

        let mut item = ItemMetadata {
            code: code.clone(),
            product_id: product_id.clone(),
            sensor_id: csv_field(record.get(2).unwrap_or("")),
            dataset_id: csv_field(record.get(3).unwrap_or("")),
            bundle_id: csv_field(record.get(4).unwrap_or("")),
            sensor: csv_field(record.get(5).unwrap_or("")).map(|s| strip_field_prefix(&s)),
            product_type: csv_field(record.get(6).unwrap_or("")).map(|s| strip_field_prefix(&s)),
            dataset: csv_field(record.get(7).unwrap_or("")).map(|s| strip_field_prefix(&s)),
            bundle: csv_field(record.get(8).unwrap_or("")).map(|s| strip_field_prefix(&s)),
            description: csv_field(record.get(9).unwrap_or("")),
            format: csv_field(record.get(10).unwrap_or("")),
            technical_info: csv_field(record.get(11).unwrap_or("")),
            processing_level: None,  // Not in CSV
            phase: csv_field(record.get(12).unwrap_or("")),
            date_first: record.get(13).and_then(parse_european_date),
            date_last: record.get(14).and_then(parse_european_date),
            continued: false,  // Not in CSV
            frequency: csv_field(record.get(15).unwrap_or("")),
            location: None,
            source: csv_field(record.get(16).unwrap_or("")),
            additional_remarks: None,
            storage_mb: None,
            internal_commentary: None,  // Not in CSV
            collection_id: None,
            geometry: None,
            bbox: None,
            archive_file: None,
            archive_size: None,
            files: Vec::new(),
            file_count: 0,
            folder_path: None,
            archive_hash: None,
            archive_is_zip64: false,
            is_single_file: false,
        };

        // Derive collection from ProductID via config map
        if let Some(ref pid) = item.product_id {
            if let Some(coll_id) = collection_map.get(pid.as_str()) {
                item.collection_id = Some(coll_id.to_string());
            } else {
                warn!("Item {} has ProductID '{}' not in collection map", code, pid);
            }
        }

        items.push(item);
    }

    Ok(items)
}

// =============================================================================
// STAC Generation
// =============================================================================

/// Generate a sanitized asset key from filename
/// Removes extension and replaces non-alphanumeric characters
fn sanitize_asset_key(filename: &str) -> String {
    let name = std::path::Path::new(filename)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| filename.to_string());

    // Replace non-alphanumeric characters with hyphens, collapse multiple hyphens
    let mut key = String::new();
    let mut last_was_hyphen = false;
    for c in name.chars() {
        if c.is_alphanumeric() {
            key.push(c.to_ascii_lowercase());
            last_was_hyphen = false;
        } else if !last_was_hyphen {
            key.push('-');
            last_was_hyphen = true;
        }
    }
    key.trim_matches('-').to_string()
}

/// Extract datetime from filename based on provider-specific patterns
/// Returns ISO 8601 datetime string (UTC) if found
///
/// Patterns supported:
/// - Geopraevent ISO 8601: `YYYYMMDDTHHMMSSZ` (e.g., `20250628T154202Z.jpg`)
/// - Terradata full: `YYYYMMDD-HHMMSS` (e.g., `orthophoto_20250530-190300_...`)
/// - Geopraevent/Terradata: `YYYYMMDD-HHMM` (e.g., `GP-AIM-24_20250529-0500_...`)
/// - DNAGE webcam: `YYYY-MM-DD HHMM` (e.g., `2025-05-31 1320.jpg`)
/// - Terradata short: `_YYMMDD_` (e.g., `DTM_250711_GRID_...`)
fn extract_datetime_from_filename(filename: &str) -> Option<String> {
    // Pattern 1: Geopraevent ISO 8601 with T and Z (most specific, check first)
    // e.g., 20250628T154202Z.jpg
    let iso_z_re = regex::Regex::new(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z").ok()?;
    if let Some(caps) = iso_z_re.captures(filename) {
        let year = caps.get(1)?.as_str();
        let month = caps.get(2)?.as_str();
        let day = caps.get(3)?.as_str();
        let hour = caps.get(4)?.as_str();
        let min = caps.get(5)?.as_str();
        let sec = caps.get(6)?.as_str();
        return Some(format!("{}-{}-{}T{}:{}:{}Z", year, month, day, hour, min, sec));
    }

    // Pattern 2: YYYYMMDD-HHMMSS (Terradata style with seconds)
    // e.g., orthophoto_20250530-190300_100mmPP-0-0.tif
    let full_datetime_re = regex::Regex::new(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})").ok()?;
    if let Some(caps) = full_datetime_re.captures(filename) {
        let year = caps.get(1)?.as_str();
        let month = caps.get(2)?.as_str();
        let day = caps.get(3)?.as_str();
        let hour = caps.get(4)?.as_str();
        let min = caps.get(5)?.as_str();
        let sec = caps.get(6)?.as_str();
        return Some(format!("{}-{}-{}T{}:{}:{}Z", year, month, day, hour, min, sec));
    }

    // Pattern 3: YYYYMMDD-HHMM (Geopraevent/Terradata without seconds)
    // e.g., GP-AIM-24_20250529-0500_..., Coherence_20250603-0800.jpg
    // Note: Pattern 2 (with seconds) is checked first, so this only matches 4-digit time
    // Uses [^\d] instead of lookahead since Rust regex doesn't support lookaheads
    let datetime_no_sec_re = regex::Regex::new(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})[^\d]").ok()?;
    if let Some(caps) = datetime_no_sec_re.captures(filename) {
        let year = caps.get(1)?.as_str();
        let month = caps.get(2)?.as_str();
        let day = caps.get(3)?.as_str();
        let hour = caps.get(4)?.as_str();
        let min = caps.get(5)?.as_str();
        return Some(format!("{}-{}-{}T{}:{}:00Z", year, month, day, hour, min));
    }
    // Also try pattern at end of string (no char after HHMM)
    let datetime_no_sec_end_re = regex::Regex::new(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})$").ok()?;
    if let Some(caps) = datetime_no_sec_end_re.captures(filename) {
        let year = caps.get(1)?.as_str();
        let month = caps.get(2)?.as_str();
        let day = caps.get(3)?.as_str();
        let hour = caps.get(4)?.as_str();
        let min = caps.get(5)?.as_str();
        return Some(format!("{}-{}-{}T{}:{}:00Z", year, month, day, hour, min));
    }

    // Pattern 4: YYYY-MM-DD HHMM (DNAGE webcam style with space)
    // e.g., 2025-05-31 1320.jpg
    let dnage_re = regex::Regex::new(r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2})(\d{2})").ok()?;
    if let Some(caps) = dnage_re.captures(filename) {
        let year = caps.get(1)?.as_str();
        let month = caps.get(2)?.as_str();
        let day = caps.get(3)?.as_str();
        let hour = caps.get(4)?.as_str();
        let min = caps.get(5)?.as_str();
        return Some(format!("{}-{}-{}T{}:{}:00Z", year, month, day, hour, min));
    }

    // Pattern 5: _YYMMDD_ (Terradata short date, 2-digit year)
    // e.g., DTM_250711_GRID_20cm..., LIDAR_250523_LV95...
    // Must be surrounded by underscores to avoid false matches
    let short_date_re = regex::Regex::new(r"_(\d{2})(\d{2})(\d{2})_").ok()?;
    if let Some(caps) = short_date_re.captures(filename) {
        let yy = caps.get(1)?.as_str();
        let month = caps.get(2)?.as_str();
        let day = caps.get(3)?.as_str();
        // Validate month/day ranges to avoid false positives
        let month_num: u32 = month.parse().unwrap_or(0);
        let day_num: u32 = day.parse().unwrap_or(0);
        if month_num >= 1 && month_num <= 12 && day_num >= 1 && day_num <= 31 {
            return Some(format!("20{}-{}-{}T00:00:00Z", yy, month, day));
        }
    }

    None
}

/// Create a STAC Item from metadata with all files as assets
/// This creates STAC 1.1.0 compliant items with:
/// - Item-level geometry in WGS84 (combined extent of all files)
/// - Per-asset geometry using Projection Extension (LV95 when available)
fn create_stac_item(item: &ItemMetadata, base_url: &str, s3_base_url: &str, collection_titles: &HashMap<String, String>) -> serde_json::Value {
    let collection_id = item.collection_id.as_deref().unwrap_or("unknown");

    // Build datetime fields
    let (datetime, start_datetime, end_datetime) = if item.date_first.is_some() && item.date_last.is_some() {
        (
            serde_json::Value::Null,
            Some(format!("{}T00:00:00Z", item.date_first.as_ref().unwrap())),
            Some(format!("{}T23:59:59Z", item.date_last.as_ref().unwrap())),
        )
    } else if let Some(ref date) = item.date_first {
        (
            serde_json::json!(format!("{}T00:00:00Z", date)),
            None,
            None,
        )
    } else {
        (serde_json::Value::Null, None, None)
    };

    // Build title: strip the collection title from the product type prefix to avoid
    // redundancy, but keep any qualifier that distinguishes items within a merged
    // collection. E.g. in collection "Radar":
    //   "Radar Velocity - DTM" → "Velocity - DTM"
    //   "Radar Interferograms" → "Interferograms"
    // But in collection "Orthophoto":
    //   "Orthophoto" → "" (identical, stripped entirely)
    let coll_title = collection_titles.get(collection_id).map(|s| s.as_str()).unwrap_or("");
    let subtype_str = item.product_type.as_deref().map(|pt| {
        // Strip the collection title prefix (case-insensitive) from the product type
        let trimmed = pt.strip_prefix(coll_title)
            .or_else(|| {
                // Also try stripping the collection-level name (pre-override)
                let coll_name = collection_name_from_product_type(pt);
                if coll_name.eq_ignore_ascii_case(coll_title) {
                    Some(&pt[coll_name.len()..])
                } else {
                    None
                }
            })
            .unwrap_or(pt);
        trimmed.trim_start_matches(" - ").trim()
    }).unwrap_or("");
    let dataset_str = item.dataset.as_deref()
        .filter(|s| !s.is_empty())
        .or(item.dataset_id.as_deref())
        .unwrap_or("");
    let bundle_suffix = item.bundle.as_ref()
        .filter(|b| !b.is_empty())
        .map(|b| format!(" ({})", b))
        .unwrap_or_default();
    let title = format!(
        "{} - {} - {}{}",
        item.code,
        subtype_str,
        dataset_str,
        bundle_suffix
    )
    .trim_matches(|c| c == ' ' || c == '-')
    .replace(" -  - ", " - ")
    .replace(" - - ", " - ")
    .trim_matches(|c| c == ' ' || c == '-')
    .to_string();
    let title = if title.is_empty() { item.code.clone() } else { title };

    // Build properties
    let mut properties = serde_json::json!({
        "title": title,
        "datetime": datetime,
        "blatten:code": item.code,
    });

    if let Some(ref desc) = item.description {
        properties["description"] = serde_json::json!(desc);
    }
    if let Some(ref sd) = start_datetime {
        properties["start_datetime"] = serde_json::json!(sd);
    }
    if let Some(ref ed) = end_datetime {
        properties["end_datetime"] = serde_json::json!(ed);
    }

    // Custom properties (blatten namespace)
    if let Some(ref s) = item.sensor {
        properties["blatten:sensor"] = serde_json::json!(s);
    }
    if let Some(ref pt) = item.product_type {
        properties["blatten:product_type"] = serde_json::json!(pt);
    }
    if let Some(ref src) = item.source {
        properties["blatten:source"] = serde_json::json!(src);
    }
    if let Some(level) = item.processing_level {
        properties["blatten:processing_level"] = serde_json::json!(level);
    }
    if let Some(ref phase) = item.phase {
        properties["blatten:phase"] = serde_json::json!(phase);
    }
    if let Some(ref freq) = item.frequency {
        properties["blatten:frequency"] = serde_json::json!(freq);
    }
    if let Some(ref info) = item.technical_info {
        if !info.is_empty() {
            properties["blatten:additional_info"] = serde_json::Value::String(info.clone());
        }
    }
    if let Some(ref fmt) = item.format {
        properties["blatten:format"] = serde_json::json!(fmt);
    }
    if let Some(mb) = item.storage_mb {
        properties["blatten:storage_mb"] = serde_json::json!(mb);
    }
    // File count from folder scanning
    properties["blatten:file_count"] = serde_json::json!(item.file_count);
    // Dataset and bundle metadata
    if let Some(ref ds) = item.dataset {
        properties["blatten:dataset"] = serde_json::json!(ds);
    }
    if let Some(ref b) = item.bundle {
        properties["blatten:bundle"] = serde_json::json!(b);
    }

    // Build assets - now includes all individual files
    let mut assets = serde_json::Map::new();
    let mut has_proj_extension = false;

    // Add archive asset if available
    if let Some(ref archive) = item.archive_file {
        let media_type = if archive.ends_with(".zip") {
            "application/zip"
        } else if archive.ends_with(".7z") {
            "application/x-7z-compressed"
        } else {
            "application/octet-stream"
        };

        let mut asset = serde_json::json!({
            "href": format!("{}/archives/{}", s3_base_url, archive),
            "type": media_type,
            "title": format!("{} complete archive", item.code),
            "roles": ["data", "archive"]
        });

        // Use actual archive size if available, otherwise fall back to storage_mb
        if let Some(size) = item.archive_size {
            asset["file:size"] = serde_json::json!(size);
        } else if let Some(mb) = item.storage_mb {
            asset["file:size"] = serde_json::json!((mb * 1024.0 * 1024.0) as i64);
        }

        // Add file checksum (multihash format: 1220 prefix for SHA-256)
        if let Some(ref hash) = item.archive_hash {
            asset["file:checksum"] = serde_json::json!(format!("1220{}", hash));
        }

        // Flag ZIP64 archives for UI warning
        if item.archive_is_zip64 {
            asset["blatten:zip64"] = serde_json::json!(true);
        }

        assets.insert("archive".to_string(), asset);
    }

    // Add individual file assets with Projection Extension fields
    let mut used_keys: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for file_info in &item.files {
        let filename = file_info.path.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // Generate unique asset key
        let base_key = sanitize_asset_key(&filename);
        let key = {
            let count = used_keys.entry(base_key.clone()).or_insert(0);
            let key = if *count == 0 {
                base_key.clone()
            } else {
                format!("{}-{}", base_key, count)
            };
            *count += 1;
            key
        };

        // Build S3 path as assets/{code}/{relative_path_from_folder}
        // This preserves subdirectory structure within the item folder
        let rel_path_from_folder = if let Some(ref folder_path) = item.folder_path {
            file_info.path.strip_prefix(folder_path)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| filename.clone())
        } else {
            filename.clone()
        };
        let href = format!("{}/assets/{}/{}", s3_base_url, item.code, rel_path_from_folder);
        let mime_type = get_mime_type(&file_info.path);

        let mut asset = serde_json::json!({
            "href": href,
            "type": mime_type,
            "title": filename,
            "roles": ["data"]
        });

        // Add file size
        if file_info.size > 0 {
            asset["file:size"] = serde_json::json!(file_info.size);
        }

        // Add file checksum (multihash format: 1220 prefix for SHA-256)
        if let Some(ref hash) = file_info.hash {
            asset["file:checksum"] = serde_json::json!(format!("1220{}", hash));
        }

        // Extract and add datetime from filename
        if let Some(datetime) = extract_datetime_from_filename(&filename) {
            asset["datetime"] = serde_json::json!(datetime);
        }

        // Add Projection Extension fields if LV95 coordinates available
        if let Some(ref bbox_lv95) = file_info.bbox_lv95 {
            asset["proj:code"] = serde_json::json!("EPSG:2056");
            asset["proj:bbox"] = serde_json::json!(bbox_lv95);
            has_proj_extension = true;

            if let Some(ref geometry_lv95) = file_info.geometry_lv95 {
                asset["proj:geometry"] = geometry_lv95.clone();
            }
        }

        assets.insert(key, asset);
    }

    // Build links
    let links = vec![
        serde_json::json!({
            "rel": "self",
            "href": format!("{}/stac/collections/{}/items/{}", base_url, collection_id, item.code),
            "type": "application/geo+json"
        }),
        serde_json::json!({
            "rel": "parent",
            "href": format!("{}/stac/collections/{}", base_url, collection_id),
            "type": "application/json"
        }),
        serde_json::json!({
            "rel": "collection",
            "href": format!("{}/stac/collections/{}", base_url, collection_id),
            "type": "application/json"
        }),
        serde_json::json!({
            "rel": "root",
            "href": format!("{}/stac/catalog.json", base_url),
            "type": "application/json"
        }),
    ];

    // Build stac_extensions list
    let mut extensions = vec![
        "https://stac-extensions.github.io/timestamps/v1.1.0/schema.json".to_string(),
        "https://stac-extensions.github.io/file/v2.1.0/schema.json".to_string(),
    ];
    if has_proj_extension {
        extensions.push("https://stac-extensions.github.io/projection/v2.0.0/schema.json".to_string());
    }

    serde_json::json!({
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "stac_extensions": extensions,
        "id": item.code,
        "geometry": item.geometry,
        "bbox": item.bbox,
        "properties": properties,
        "links": links,
        "assets": assets,
        "collection": collection_id
    })
}

/// Get MIME type for a file extension
fn get_mime_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase().as_str() {
        "tif" | "tiff" => "image/tiff; application=geotiff",
        "laz" => "application/vnd.laszip",
        "las" => "application/vnd.las",
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "zip" => "application/zip",
        "csv" => "text/csv",
        "json" => "application/json",
        "obj" => "model/obj",
        "ply" => "application/ply",
        _ => "application/octet-stream",
    }
}

/// Create a STAC Collection
fn create_stac_collection(
    def: &CollectionDef,
    items: &[&ItemMetadata],
    base_url: &str,
    catalog_config: &CatalogConfig,
) -> serde_json::Value {
    // Compute spatial extent using config default_bbox as fallback
    let bboxes: Vec<&Vec<f64>> = items.iter().filter_map(|i| i.bbox.as_ref()).collect();
    let spatial_bbox: serde_json::Value = if !bboxes.is_empty() {
        serde_json::json!([[
            bboxes.iter().map(|b| b[0]).fold(f64::INFINITY, f64::min),
            bboxes.iter().map(|b| b[1]).fold(f64::INFINITY, f64::min),
            bboxes.iter().map(|b| b[2]).fold(f64::NEG_INFINITY, f64::max),
            bboxes.iter().map(|b| b[3]).fold(f64::NEG_INFINITY, f64::max),
        ]])
    } else {
        serde_json::json!([catalog_config.default_bbox])
    };

    // Compute temporal extent
    let mut dates: Vec<&str> = items
        .iter()
        .filter_map(|i| i.date_first.as_deref())
        .chain(items.iter().filter_map(|i| i.date_last.as_deref()))
        .collect();
    dates.sort();

    let temporal_interval: serde_json::Value = if !dates.is_empty() {
        serde_json::json!([[
            format!("{}T00:00:00Z", dates.first().unwrap()),
            format!("{}T23:59:59Z", dates.last().unwrap()),
        ]])
    } else {
        serde_json::json!([[null, null]])
    };

    // Build summaries
    let mut processing_levels: Vec<i32> = items
        .iter()
        .filter_map(|i| i.processing_level)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    processing_levels.sort_unstable();

    let mut sources: Vec<String> = items
        .iter()
        .filter_map(|i| i.source.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    sources.sort_unstable();

    // Build providers from config
    let providers: Vec<serde_json::Value> = catalog_config.providers.iter().map(|p| {
        serde_json::json!({
            "name": p.name,
            "roles": p.roles,
            "url": p.url
        })
    }).collect();

    // Build keywords from config + collection title
    let mut keywords: Vec<String> = catalog_config.keywords.clone();
    keywords.push(def.title.to_lowercase());

    serde_json::json!({
        "type": "Collection",
        "id": def.id,
        "stac_version": STAC_VERSION,
        "stac_extensions": [
            "https://stac-extensions.github.io/timestamps/v1.1.0/schema.json"
        ],
        "title": def.title,
        "description": def.description,
        "license": catalog_config.license,
        "keywords": keywords,
        "providers": providers,
        "extent": {
            "spatial": { "bbox": spatial_bbox },
            "temporal": { "interval": temporal_interval }
        },
        "summaries": {
            "processing_level": processing_levels,
            "source": sources
        },
        "links": [
            {
                "rel": "self",
                "href": format!("{}/stac/collections/{}", base_url, def.id),
                "type": "application/json"
            },
            {
                "rel": "root",
                "href": format!("{}/stac/catalog.json", base_url),
                "type": "application/json"
            },
            {
                "rel": "parent",
                "href": format!("{}/stac/catalog.json", base_url),
                "type": "application/json"
            },
            {
                "rel": "items",
                "href": format!("{}/stac/collections/{}/items", base_url, def.id),
                "type": "application/geo+json"
            }
        ],
        "assets": {}
    })
}

/// Generate the complete STAC catalog
/// Returns (num_collections, num_items, total_assets, issues)
fn generate_catalog(
    items: &[ItemMetadata],
    output_dir: &PathBuf,
    base_url: &str,
    s3_base_url: &str,
    multi_progress: &MultiProgress,
    config: &GenConfig,
) -> Result<(usize, usize, usize, Vec<ValidationIssue>)> {
    fs::create_dir_all(output_dir)?;
    fs::create_dir_all(output_dir.join("collections"))?;

    let collections_defs = build_collections(items, &config.collections, &config.collection_titles);
    let mut issues = Vec::new();

    // Group items by collection
    let mut items_by_collection: HashMap<String, Vec<&ItemMetadata>> = HashMap::new();
    for item in items {
        if let Some(ref coll_id) = item.collection_id {
            items_by_collection
                .entry(coll_id.clone())
                .or_default()
                .push(item);
        }
    }

    // Progress bar style
    let bar_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
        .expect("Invalid bar template")
        .progress_chars("#>-");

    // Phase 1: Create STAC items in parallel
    let item_pb = multi_progress.add(ProgressBar::new(items.len() as u64));
    item_pb.set_style(bar_style.clone());
    item_pb.set_message("Creating STAC items...");

    // Build collection title lookup for item title construction
    let coll_title_map: HashMap<String, String> = collections_defs.iter()
        .map(|(id, def)| (id.clone(), def.title.clone()))
        .collect();

    // Pre-create all STAC items in parallel (needed for both per-collection and all_items files)
    let results: Vec<_> = items
        .par_iter()
        .map(|item| {
            let stac_item = create_stac_item(item, base_url, s3_base_url, &coll_title_map);

            // Count assets for stats
            let asset_count = stac_item
                .get("assets")
                .and_then(|a| a.as_object())
                .map(|a| a.len())
                .unwrap_or(0);

            // Collect validation issues
            let mut item_issues = Vec::new();
            if item.geometry.is_none() {
                item_issues.push(ValidationIssue {
                    item_id: item.code.clone(),
                    severity: "warning".to_string(),
                    message: "Missing geometry".to_string(),
                });
            }
            if item.archive_file.is_none() && !item.is_single_file {
                item_issues.push(ValidationIssue {
                    item_id: item.code.clone(),
                    severity: "warning".to_string(),
                    message: "No archive file mapped".to_string(),
                });
            }
            if item.files.is_empty() {
                item_issues.push(ValidationIssue {
                    item_id: item.code.clone(),
                    severity: "warning".to_string(),
                    message: "No files found".to_string(),
                });
            }

            item_pb.inc(1);
            (item, stac_item, asset_count, item_issues)
        })
        .collect();

    // Aggregate results
    let mut items_with_stac: Vec<(&ItemMetadata, serde_json::Value)> = Vec::with_capacity(results.len());
    let mut total_assets: usize = 0;
    for (item, stac_item, asset_count, item_issues) in results {
        items_with_stac.push((item, stac_item));
        total_assets += asset_count;
        issues.extend(item_issues);
    }

    item_pb.finish_with_message(format!("Created {} STAC items ({} assets)", items_with_stac.len(), total_assets));

    // Phase 2: Build all collection data in memory, then write in parallel
    let coll_pb = multi_progress.add(ProgressBar::new(items_by_collection.len() as u64));
    coll_pb.set_style(bar_style.clone());
    coll_pb.set_message("Building collection files...");

    // Build all collection data in parallel
    let collection_data: Vec<_> = items_by_collection
        .par_iter()
        .map(|(coll_id, coll_items)| {
            // Find collection definition (keyed by slug = coll_id)
            let def = collections_defs
                .get(coll_id.as_str())
                .expect("Unknown collection");

            // Create collection JSON
            let collection = create_stac_collection(def, coll_items, base_url, &config.catalog);
            let collection_json = serde_json::to_string_pretty(&collection).unwrap();

            coll_pb.inc(1);
            (coll_id.clone(), collection, collection_json)
        })
        .collect();

    // Sort by collection ID for deterministic output
    let mut collection_data = collection_data;
    collection_data.sort_by(|a, b| a.0.cmp(&b.0));

    // Extract collections for later use
    let stac_collections: Vec<_> = collection_data.iter().map(|(_, c, _)| c.clone()).collect();

    // Write all collection files (fast, already serialized)
    for (coll_id, _, collection_json) in &collection_data {
        let coll_path = output_dir.join("collections").join(format!("{}.json", coll_id));
        fs::write(&coll_path, collection_json)?;
    }

    // Remove stale collection files from previous runs (e.g. after collection ID renames)
    let valid_ids: std::collections::HashSet<&str> =
        collection_data.iter().map(|(id, _, _)| id.as_str()).collect();
    let collections_path = output_dir.join("collections");
    if let Ok(entries) = fs::read_dir(&collections_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str());
            if ext == Some("json") || ext == Some("ndjson") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    // For "foo_items.ndjson", check "foo"; for "foo.json", check "foo"
                    let base_id = name.strip_suffix("_items").unwrap_or(name);
                    if !valid_ids.contains(base_id) {
                        if let Err(e) = fs::remove_file(&path) {
                            eprintln!("Warning: failed to remove stale file {:?}: {}", path, e);
                        }
                    }
                }
            }
        }
    }

    coll_pb.finish_with_message(format!("Wrote {} collection files", stac_collections.len()));

    // Collect all STAC items for the all_items file, sorted by ID for deterministic output
    let mut all_items: Vec<serde_json::Value> = items_with_stac.into_iter().map(|(_, stac)| stac).collect();
    all_items.sort_by(|a, b| {
        let a_id = a["id"].as_str().unwrap_or("");
        let b_id = b["id"].as_str().unwrap_or("");
        a_id.cmp(b_id)
    });

    // Build links for root catalog
    let mut catalog_links: Vec<serde_json::Value> = vec![
        serde_json::json!({
            "rel": "self",
            "href": format!("{}/stac/catalog.json", base_url),
            "type": "application/json"
        }),
        serde_json::json!({
            "rel": "root",
            "href": format!("{}/stac/catalog.json", base_url),
            "type": "application/json"
        }),
    ];

    // Add child links for collections
    for c in &stac_collections {
        catalog_links.push(serde_json::json!({
            "rel": "child",
            "href": format!("{}/stac/collections/{}", base_url, c["id"].as_str().unwrap()),
            "type": "application/json",
            "title": c["title"]
        }));
    }

    // Add links from config (describedby, license, about, etc.)
    for link in &config.catalog.links {
        let mut link_obj = serde_json::Map::new();
        link_obj.insert("rel".to_string(), serde_json::json!(link.rel));
        // Resolve href: use href_suffix (relative to base_url + s3_base_url) or absolute href
        if let Some(ref suffix) = link.href_suffix {
            link_obj.insert("href".to_string(), serde_json::json!(format!("{}{}{}", base_url, s3_base_url, suffix)));
        } else if let Some(ref href) = link.href {
            link_obj.insert("href".to_string(), serde_json::json!(href));
        }
        if let Some(ref t) = link.link_type {
            link_obj.insert("type".to_string(), serde_json::json!(t));
        }
        if let Some(ref title) = link.title {
            link_obj.insert("title".to_string(), serde_json::json!(title));
        }
        catalog_links.push(serde_json::Value::Object(link_obj));
    }

    // Write root catalog
    let catalog = serde_json::json!({
        "type": "Catalog",
        "id": config.catalog.id,
        "stac_version": STAC_VERSION,
        "title": config.catalog.title,
        "description": config.catalog.description,
        "links": catalog_links,
        "conformsTo": [
            "https://api.stacspec.org/v1.0.0/core",
            "https://api.stacspec.org/v1.0.0/collections",
            "https://api.stacspec.org/v1.0.0/item-search"
        ]
    });
    fs::write(
        output_dir.join("catalog.json"),
        serde_json::to_string_pretty(&catalog)?,
    )?;

    // Write all items as NDJSON (one item per line, no wrapper)
    let all_items_path = output_dir.join("items.ndjson");
    let all_items_count = all_items.len();

    // Build NDJSON in parallel in memory, then write once
    let write_pb = multi_progress.add(ProgressBar::new_spinner());
    write_pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("Invalid spinner template"),
    );
    write_pb.set_message("Building items.ndjson in memory...");

    // Parallel JSON serialization
    let ndjson_lines: Vec<String> = all_items
        .par_iter()
        .map(|item| serde_json::to_string(item).unwrap())
        .collect();

    write_pb.set_message("Writing items.ndjson...");
    fs::write(&all_items_path, ndjson_lines.join("\n"))?;

    write_pb.finish_with_message(format!("Wrote {} items to items.ndjson", all_items_count));

    // Write collections list
    let collections_list = serde_json::json!({
        "collections": stac_collections,
        "links": [
            {
                "rel": "self",
                "href": format!("{}/stac/collections", base_url)
            },
            {
                "rel": "root",
                "href": format!("{}/stac/catalog.json", base_url)
            }
        ]
    });
    fs::write(
        output_dir.join("collections.json"),
        serde_json::to_string_pretty(&collections_list)?,
    )?;

    // Compute quality metrics
    let num_items = all_items.len();
    let items_with_geometry = all_items.iter().filter(|i| !i["geometry"].is_null()).count();
    let items_missing_geometry = all_items.iter().filter(|i| i["geometry"].is_null()).count();
    let items_with_archive = all_items.iter().filter(|i| {
        i.get("assets").and_then(|a| a.get("archive")).is_some()
    }).count();
    let items_missing_archive = num_items - items_with_archive;

    // Compute issues by severity
    let error_count = issues.iter().filter(|i| i.severity == "error").count();
    let warning_count = issues.iter().filter(|i| i.severity == "warning").count();
    let info_count = issues.iter().filter(|i| i.severity == "info").count();

    // Group items by collection for per-collection stats
    let mut collection_stats: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for item in &all_items {
        let coll_id = item.get("collection").and_then(|c| c.as_str()).unwrap_or("unknown");
        let num_assets = item.get("assets").and_then(|a| a.as_object()).map(|a| a.len()).unwrap_or(0);
        let entry = collection_stats.entry(coll_id.to_string()).or_insert_with(|| {
            serde_json::json!({
                "total_items": 0,
                "total_assets": 0,
                "with_geometry": 0,
                "with_archive": 0
            })
        });
        entry["total_items"] = serde_json::json!(entry["total_items"].as_i64().unwrap_or(0) + 1);
        entry["total_assets"] = serde_json::json!(entry["total_assets"].as_i64().unwrap_or(0) + num_assets as i64);
        if !item["geometry"].is_null() {
            entry["with_geometry"] = serde_json::json!(entry["with_geometry"].as_i64().unwrap_or(0) + 1);
        }
        if item.get("assets").and_then(|a| a.get("archive")).is_some() {
            entry["with_archive"] = serde_json::json!(entry["with_archive"].as_i64().unwrap_or(0) + 1);
        }
    }

    // Write validation report
    let report = serde_json::json!({
        "generated_at": Utc::now().to_rfc3339(),
        "stac_version": STAC_VERSION,
        "summary": {
            "total_collections": stac_collections.len(),
            "total_items": num_items,
            "total_assets": total_assets,
            "items_with_geometry": items_with_geometry,
            "items_missing_geometry": items_missing_geometry,
            "items_with_archive": items_with_archive,
            "items_missing_archive": items_missing_archive,
            "geometry_coverage_pct": if num_items == 0 { 0.0 } else {
                (items_with_geometry as f64 / num_items as f64 * 100.0).round()
            },
            "archive_coverage_pct": if num_items == 0 { 0.0 } else {
                (items_with_archive as f64 / num_items as f64 * 100.0).round()
            }
        },
        "issues_summary": {
            "total": issues.len(),
            "errors": error_count,
            "warnings": warning_count,
            "info": info_count
        },
        "collection_stats": collection_stats,
        "issues": issues,
        "validation_passed": error_count == 0
    });
    fs::write(
        output_dir.join("validation_report.json"),
        serde_json::to_string_pretty(&report)?,
    )?;

    Ok((stac_collections.len(), num_items, total_assets, issues))
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    // Set GDAL environment for consistent GeoTIFF CRS interpretation
    std::env::set_var("GTIFF_SRS_SOURCE", "EPSG");

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    // Resolve input path: use provided or auto-discover
    let input_file = match cli.input {
        Some(path) => path,
        None => discover_input()?,
    };

    // Destructure CLI for convenience
    let Cli {
        input: _, output, base_url, s3_base_url, data, config: config_path,
        threads, validate,
        organised_dir,
        yes, no_materialize, skip_zip_validation,
        factsheets_dir,
    } = cli;
    let materialize = !no_materialize;

    {
            // Load config
            let config = load_config(&config_path)?;
            info!("  Loaded config from {:?}", config_path);

            // Resolve assets_dir and archives_dir from organised_dir
            let assets_dir = organised_dir.as_ref().map(|d| d.join("assets"));
            let archives_dir = organised_dir.as_ref().map(|d| d.join("archives"));

            // Always use symlink for staging
            let link_mode_parsed = LinkMode::Symlink;

            info!("=== STAC Generator (v{}) ===", STAC_VERSION);

            // Configure rayon thread pool
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .ok();

            // Set up progress bars
            let multi_progress = MultiProgress::new();

            // Check data directory exists if specified
            let data = data.filter(|p| p.exists());

            // Step 1: Parse metadata (quick operation, use simple info log instead of spinner)
            let mut items = parse_csv(&input_file, &config.collections)?;
            info!("  Parsed {} items from {}", items.len(), input_file.display());

            // Step 2: Scan folders if data provided (quick operation, use simple info log)
            let scanned_folders: HashMap<String, ScannedFolder> = if let Some(ref data_path) = data {
                let folders = scan_data_folders(data_path, &config.folder_mappings, &config.exclude_files)?;
                info!("  Found {} data folders", folders.len());
                folders
            } else {
                HashMap::new()
            };

            // Collect item codes for staging/validation
            let item_codes: Vec<String> = items.iter().map(|i| i.code.clone()).collect();

            // Always use SHA-256 for hashing
            let algorithm = HashAlgorithm::Sha256;

            // Pre-flight summary: show what was detected and prompt for confirmation
            {
                info!("");
                info!("=== Summary ===");
                info!("  Input:  {} ({} items)", input_file.display(), items.len());
                if let Some(ref d) = data {
                    info!("  Data:   {} ({} folders matched)", d.display(), scanned_folders.len());
                } else {
                    info!("  Data:   (none)");
                }
                info!("  Config: {}", config_path.display());
                info!("  Output: {}", output.display());

                if let Some(ref org) = organised_dir {
                    let assets_path = org.join("assets");
                    let archives_path = org.join("archives");
                    let has_assets = assets_path.exists();
                    let has_archives = archives_path.exists();

                    if has_assets || has_archives {
                        info!("  Staging: {} (existing output detected)", org.display());

                        if has_assets {
                            // Count code directories and total files in assets/
                            let code_dirs: Vec<String> = fs::read_dir(&assets_path)
                                .map(|rd| rd
                                    .filter_map(|e| e.ok())
                                    .filter(|e| e.path().is_dir())
                                    .map(|e| e.file_name().to_string_lossy().to_string())
                                    .collect())
                                .unwrap_or_default();
                            let total_files: usize = code_dirs.iter()
                                .map(|code| {
                                    WalkDir::new(assets_path.join(code))
                                        .into_iter()
                                        .filter_map(|e| e.ok())
                                        .filter(|e| e.path().is_file())
                                        .count()
                                })
                                .sum();
                            info!("    assets/   {} codes, {} files", code_dirs.len(), total_files);
                        }

                        if has_archives {
                            let archive_count = fs::read_dir(&archives_path)
                                .map(|rd| rd
                                    .filter_map(|e| e.ok())
                                    .filter(|e| {
                                        let ext = e.path().extension()
                                            .and_then(|e| e.to_str())
                                            .unwrap_or("")
                                            .to_lowercase();
                                        ext == "zip"
                                    })
                                    .count())
                                .unwrap_or(0);
                            let total_size: u64 = fs::read_dir(&archives_path)
                                .map(|rd| rd
                                    .filter_map(|e| e.ok())
                                    .filter(|e| e.path().extension()
                                        .and_then(|e| e.to_str())
                                        .map_or(false, |e| e.eq_ignore_ascii_case("zip")))
                                    .filter_map(|e| e.metadata().ok())
                                    .map(|m| m.len())
                                    .sum())
                                .unwrap_or(0);
                            info!("    archives/ {} zips ({:.1} GB)", archive_count,
                                total_size as f64 / 1_073_741_824.0);
                        }

                        // Build extra_expected for factsheets so preview doesn't count them as "removed"
                        let factsheet_extra: HashMap<String, u64> = if factsheets_dir.is_dir() {
                            let fm = discover_factsheets(&factsheets_dir);
                            let mut extra = HashMap::new();
                            for item in &items {
                                if let Some(ref sid) = item.sensor_id {
                                    if let Some(src) = fm.get(sid.as_str()) {
                                        let fname = src.file_name().unwrap().to_string_lossy();
                                        let asset_path = format!("assets/{}/{}", item.code, fname);
                                        let size = fs::metadata(src).map(|m| m.len()).unwrap_or(0);
                                        extra.insert(asset_path, size);
                                    }
                                }
                            }
                            extra
                        } else {
                            HashMap::new()
                        };

                        // Check manifest
                        let cache_dir = output.join(".stac-cache");
                        let manifest_path = cache_dir.join("manifest.json");
                        if manifest_path.exists() {
                            if let Ok(m) = DataManifest::load(&manifest_path) {
                                info!("    manifest  {} files, {} archives tracked",
                                    m.total_files, m.archives.len());
                                let (new, removed, modified, unchanged) =
                                    preview_staging_changes(&scanned_folders, &m, &factsheet_extra);
                                info!("");
                                info!("  Changes (vs manifest):");
                                info!("    {} new, {} removed, {} modified, {} unchanged",
                                    new, removed, modified, unchanged);
                            }
                        }
                    } else {
                        info!("  Staging: {} (empty — fresh run)", org.display());
                    }
                }

                if materialize {
                    info!("  Mode:   materialize (symlinks → real copies)");
                } else {
                    info!("  Mode:   symlinks only (--no-materialize)");
                }

                // Factsheet discovery and sensor matching
                if factsheets_dir.is_dir() {
                    let factsheet_map = discover_factsheets(&factsheets_dir);
                    if !factsheet_map.is_empty() {
                        // Collect unique sensor IDs from items
                        let sensor_ids: BTreeMap<String, usize> = items.iter()
                            .filter_map(|i| i.sensor_id.as_ref())
                            .fold(BTreeMap::new(), |mut acc, sid| {
                                *acc.entry(sid.clone()).or_insert(0) += 1;
                                acc
                            });

                        info!("");
                        info!("  Factsheets: {} found in {}", factsheet_map.len(), factsheets_dir.display());
                        let mut matched = 0;
                        let mut missing = Vec::new();
                        for (sid, item_count) in &sensor_ids {
                            if let Some(path) = factsheet_map.get(sid.as_str()) {
                                let fname = path.file_name().unwrap().to_string_lossy();
                                info!("    sensor {:>2} → {} ({} items)", sid, fname, item_count);
                                matched += 1;
                            } else {
                                missing.push((sid.clone(), *item_count));
                            }
                        }
                        if !missing.is_empty() {
                            info!("    Missing factsheets for {} sensors:", missing.len());
                            for (sid, count) in &missing {
                                // Try to find the sensor name from config
                                let sensor_name = config.sensors.values()
                                    .find(|s| s.items.iter().any(|c| c.starts_with(&format!("{:0>2}", sid))))
                                    .map(|s| s.name.as_str())
                                    .unwrap_or("unknown");
                                info!("      sensor {:>2} ({}, {} items) — no PDF", sid, sensor_name, count);
                            }
                        }
                        info!("    {} of {} sensors matched", matched, sensor_ids.len());
                    }
                } else {
                    debug!("  Factsheets: directory {:?} not found (use --factsheets-dir)", factsheets_dir);
                }

                info!("");

                if !yes {
                    use std::io::Write;
                    print!("Proceed? [y/N] ");
                    std::io::stdout().flush()?;
                    let mut answer = String::new();
                    std::io::stdin().read_line(&mut answer)?;
                    if !answer.trim().eq_ignore_ascii_case("y") {
                        info!("Aborted.");
                        return Ok(());
                    }
                }
            }

            // Data manifest — populated by staging pipeline if --assets-dir is set
            #[allow(unused_assignments)]
            let mut data_manifest: Option<DataManifest> = None;

            // Stage assets if --assets-dir is set and data is provided
            if let Some(ref assets_path) = assets_dir {
                if data.is_some() {
                    info!("");
                    info!("=== Staging Pipeline (link mode: {}) ===", link_mode_parsed);
                    info!("  Hash algorithm: {}", algorithm);

                    // Load existing manifest for incremental hashing
                    let existing_manifest = {
                        let cache_dir = output.join(".stac-cache");
                        let manifest_path = cache_dir.join("manifest.json");
                        // Also check old location for migration
                        let old_manifest_path = output.join("manifest.json");
                        let effective_path = if manifest_path.exists() {
                            manifest_path.clone()
                        } else if old_manifest_path.exists() {
                            info!("  Migrating manifest from old location (stac/manifest.json → stac/.stac-cache/manifest.json)");
                            old_manifest_path.clone()
                        } else {
                            manifest_path.clone() // won't exist, handled below
                        };
                        if effective_path.exists() {
                            match DataManifest::load(&effective_path) {
                                Ok(m) => {
                                    info!("  Loaded existing manifest ({} files, {} archives)",
                                        m.total_files, m.archives.len());
                                    Some(m)
                                }
                                Err(e) => {
                                    warn!("  Failed to load manifest (full rebuild): {}", e);
                                    None
                                }
                            }
                        } else {
                            info!("  No existing manifest found");
                            None
                        }
                    };

                    // Use folder mappings from config
                    let folder_map = &config.folder_mappings;

                    // Step 1: Stage assets (symlinks from FINAL_Data + extract input archives)
                    let pb_style = ProgressStyle::default_bar()
                        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                        .expect("Invalid bar template")
                        .progress_chars("#>-");

                    let stage_assets_pb = multi_progress.add(ProgressBar::new(0));
                    stage_assets_pb.set_style(pb_style.clone());

                    let mut stage_stats = Some(stage_assets(
                        &scanned_folders,
                        &item_codes,
                        assets_path,
                        link_mode_parsed,
                        folder_map,
                        &stage_assets_pb,
                        &config.exclude_files,
                    )?);

                    // Step 1b: Inject factsheet PDFs into staged asset directories
                    if factsheets_dir.is_dir() {
                        let factsheet_map = discover_factsheets(&factsheets_dir);
                        if !factsheet_map.is_empty() {
                            let mut injected = 0;
                            for item in &items {
                                if let Some(ref sensor_id) = item.sensor_id {
                                    if let Some(src_path) = factsheet_map.get(sensor_id.as_str()) {
                                        let target_dir = assets_path.join(&item.code);
                                        if target_dir.exists() {
                                            let filename = src_path.file_name().unwrap();
                                            let target = target_dir.join(filename);
                                            // Always register so materializer doesn't delete pre-existing factsheets
                                            if let Some(ref mut stats) = stage_stats {
                                                stats.expected_files.insert(target.clone());
                                            }
                                            if !target.exists() {
                                                link_or_copy(src_path, &target, link_mode_parsed)?;
                                                injected += 1;
                                            }
                                        }
                                    }
                                }
                            }
                            info!("  Discovered {} factsheets for {} sensors", factsheet_map.len(), factsheet_map.len());
                            if injected > 0 {
                                info!("  Injected {} factsheet files into asset directories", injected);
                            }
                        }
                    } else {
                        debug!("  Factsheets directory not found: {:?}", factsheets_dir);
                    }

                    // Step 2: Hash all staged assets in parallel
                    let hash_pb = multi_progress.add(ProgressBar::new(0));
                    hash_pb.set_style(pb_style.clone());

                    let mut manifest = hash_staged_assets(
                        assets_path,
                        algorithm,
                        existing_manifest.as_ref(),
                        &scanned_folders,
                        &hash_pb,
                        &config.exclude_files,
                    )?;

                    // Step 3: Stage archives (always real copies, with fingerprinting)
                    if let Some(ref arch_path) = archives_dir {
                        let stage_arch_pb = multi_progress.add(ProgressBar::new(0));
                        stage_arch_pb.set_style(pb_style.clone());

                        stage_archives(
                            &scanned_folders,
                            &item_codes,
                            assets_path,
                            arch_path,
                            &mut manifest,
                            existing_manifest.as_ref(),
                            &stage_arch_pb,
                            &config.exclude_files,
                            skip_zip_validation,
                        )?;
                    }

                    // Save manifest
                    {
                        let cache_dir = output.join(".stac-cache");
                        fs::create_dir_all(&cache_dir)?;
                        let manifest_path = cache_dir.join("manifest.json");
                        manifest.save(&manifest_path)?;
                        info!("  Saved manifest to {:?}", manifest_path);
                        // Clean up old location if it exists
                        let old_manifest = output.join("manifest.json");
                        if old_manifest.exists() {
                            let _ = fs::remove_file(&old_manifest);
                            info!("  Removed old manifest from {:?}", old_manifest);
                        }
                    }

                    // Step 4 (optional): Materialize symlinks and clean stale files
                    if materialize {
                        info!("");
                        info!("=== Materializing ===");

                        // Clean stale files: remove any file in assets/ not in the expected set
                        if let Some(ref stage_stats) = stage_stats {
                            if !stage_stats.expected_files.is_empty() {
                                let mut stale_removed = 0usize;
                                for entry in WalkDir::new(assets_path)
                                    .into_iter()
                                    .filter_map(|e| e.ok())
                                {
                                    let path = entry.path();
                                    let is_file = path.symlink_metadata()
                                        .map_or(false, |m| m.is_file() || m.file_type().is_symlink());
                                    if !is_file { continue; }

                                    if !stage_stats.expected_files.contains(path) {
                                        let _ = fs::remove_file(path);
                                        stale_removed += 1;
                                    }
                                }
                                // Remove empty directories left behind
                                for entry in WalkDir::new(assets_path)
                                    .contents_first(true)
                                    .into_iter()
                                    .filter_map(|e| e.ok())
                                {
                                    let path = entry.path();
                                    if path == assets_path { continue; }
                                    if path.is_dir() {
                                        let _ = fs::remove_dir(path); // only succeeds if empty
                                    }
                                }
                                if stale_removed > 0 {
                                    info!("  Removed {} stale files from assets/", stale_removed);
                                }
                            }
                        }

                        // Materialize remaining symlinks/hardlinks in assets/
                        let (mat_assets, real_assets) = materialize_links(assets_path)?;
                        info!("  Assets: {} materialized, {} already real", mat_assets, real_assets);

                        // Materialize archives too
                        if let Some(ref arch_path) = archives_dir {
                            let (mat_arch, real_arch) = materialize_links(arch_path)?;
                            info!("  Archives: {} materialized, {} already real", mat_arch, real_arch);
                        }
                    }

                    data_manifest = Some(manifest);
                } else {
                    data_manifest = None;
                }
            } else {
                data_manifest = None;
            }

            // Keep a copy for quality report (before moving to Arc)
            let scanned_folders_for_report = scanned_folders.clone();

            // Track applied overrides for reporting
            let mut applied_overrides: Vec<(String, String, String)> = Vec::new(); // (code, file, reason)

            // Load geometry cache for incremental file-level extraction
            let crs_hash = hash_crs_overrides(&config.crs_overrides);
            let geometry_cache: Option<BTreeMap<String, CachedFileGeometry>> = {
                let cache_dir = output.join(".stac-cache");
                let cache_path = cache_dir.join("geometry_cache.json");
                match GeometryCache::load(&cache_path) {
                    Ok(cache) => {
                        if cache.version != GEOMETRY_CACHE_VERSION {
                            info!("Geometry cache: discarded (version {} != {})", cache.version, GEOMETRY_CACHE_VERSION);
                            None
                        } else if cache.crs_overrides_hash != crs_hash {
                            info!("Geometry cache: discarded (CRS overrides changed)");
                            None
                        } else {
                            info!("Geometry cache: loaded {} entries", cache.entries.len());
                            Some(cache.entries)
                        }
                    }
                    Err(_) => {
                        info!("Geometry cache: none found, will extract all files");
                        None
                    }
                }
            };

            // Step 3: Extract geometry
            let data_path = data.as_deref();
            let sensors_for_fallback = if data_path.is_some() || !config.sensors.is_empty() {
                let extract_pb = multi_progress.add(ProgressBar::new(items.len() as u64));
                extract_pb.set_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                        .expect("Invalid bar template")
                        .progress_chars("#>-"),
                );
                extract_pb.set_message("Extracting geometry...");

                // Load sensor locations from config
                let sensors = load_sensor_locations_from_config(&config.sensors);

                // Extract geometry in parallel - only for items WITHOUT scanned folders
                // (items with folders will get geometry from file-level extraction later)
                let extract_pb = Arc::new(Mutex::new(extract_pb));
                let sensors = Arc::new(sensors);
                let scanned_folders = Arc::new(scanned_folders);

                // Filter to items that need item-level extraction (no scanned folder)
                let items_needing_extraction: Vec<_> = items
                    .iter()
                    .filter(|item| !scanned_folders.contains_key(&item.code))
                    .collect();

                let results: Vec<(String, ExtractionResult)> = items_needing_extraction
                    .par_iter()
                    .map(|item| {
                        let result = extract_geometry_for_item_v2(
                            item,
                            &sensors,
                            &scanned_folders,
                            &config.crs_overrides,
                        );
                        if let Ok(pb) = extract_pb.lock() {
                            pb.inc(1);
                        }
                        (item.code.clone(), result)
                    })
                    .collect();

                // Apply results and collect stats
                let mut extracted = 0;
                let mut manual = 0;
                let mut no_geometry = 0;
                let mut failed = 0;

                for (code, result) in results {
                    if let Some(item) = items.iter_mut().find(|i| i.code == code) {
                        match result {
                            ExtractionResult::Extracted { bbox, geometry, source, override_reason } => {
                                item.bbox = Some(bbox);
                                item.geometry = Some(geometry);
                                extracted += 1;
                                if let Some(ref reason) = override_reason {
                                    applied_overrides.push((code.clone(), source.clone(), reason.clone()));
                                }
                            }
                            ExtractionResult::Manual { bbox, geometry, .. } => {
                                item.bbox = Some(bbox);
                                item.geometry = Some(geometry);
                                manual += 1;
                            }
                            ExtractionResult::NoGeometry { .. } => {
                                no_geometry += 1;
                            }
                            ExtractionResult::Failed { error } => {
                                failed += 1;
                                warn!("[{}] Extraction failed: {}", code, error);
                            }
                        }
                    }
                }

                if let Ok(mutex) = Arc::try_unwrap(extract_pb) {
                    if let Ok(pb) = mutex.into_inner() {
                        pb.finish_with_message(format!(
                            "Geometry: {} extracted, {} manual, {} none, {} failed",
                            extracted, manual, no_geometry, failed
                        ));
                    }
                }

                // Unwrap sensors from Arc for reuse in sensor fallback after file-level extraction
                Arc::try_unwrap(sensors).ok()
            } else {
                None
            };

            // Populate file info from scanned folders and compute combined bbox
            // Collect all files to process in parallel
            let all_files: Vec<(String, PathBuf)> = items
                .iter()
                .filter_map(|item| {
                    scanned_folders_for_report.get(&item.code).map(|folder| {
                        folder.files.iter().map(|f| (item.code.clone(), f.clone())).collect::<Vec<_>>()
                    })
                })
                .flatten()
                .collect();

            let file_pb = multi_progress.add(ProgressBar::new(all_files.len() as u64));
            file_pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                    .expect("Invalid bar template")
                    .progress_chars("#>-"),
            );
            file_pb.set_message("Extracting file geometry...");

            // Extract geometry for all files in parallel (with caching)
            let processed_count = std::sync::atomic::AtomicU64::new(0);
            let cache_hits = std::sync::atomic::AtomicU64::new(0);
            let cache_ref = geometry_cache.as_ref();
            let file_results: Vec<(String, FileInfo, (String, CachedFileGeometry))> = all_files
                .par_iter()
                .map(|(code, file_path)| {
                    let size = file_path.metadata().map(|m| m.len()).unwrap_or(0);
                    let cache_key = file_path.to_string_lossy().to_string();

                    // Check geometry cache: hit if file size unchanged
                    let (bbox, geometry, bbox_lv95, geometry_lv95) = if let Some(cached) =
                        cache_ref.and_then(|c| c.get(&cache_key)).filter(|c| c.size == size)
                    {
                        cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        (cached.bbox.clone(), cached.geometry.clone(), cached.bbox_lv95.clone(), cached.geometry_lv95.clone())
                    } else {
                        let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                        // Extract geometry for geospatial files (both WGS84 and LV95)
                        if ext == "tif" || ext == "tiff" || ext == "asc" {
                            let (geo, _override_reason) = extract_geotiff_geometry(file_path, &config.crs_overrides);
                            geo.map(|g| (Some(g.bbox), Some(g.geometry), g.bbox_lv95, g.geometry_lv95))
                                .unwrap_or((None, None, None, None))
                        } else if ext == "laz" || ext == "las" {
                            extract_las_geometry(file_path)
                                .map(|g| (Some(g.bbox), Some(g.geometry), g.bbox_lv95, g.geometry_lv95))
                                .unwrap_or((None, None, None, None))
                        } else {
                            (None, None, None, None)
                        }
                    };

                    // Update progress every 100 files to reduce contention
                    let count = processed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if count % 100 == 0 {
                        file_pb.set_position(count);
                    }

                    let cached_entry = CachedFileGeometry {
                        size,
                        bbox: bbox.clone(),
                        geometry: geometry.clone(),
                        bbox_lv95: bbox_lv95.clone(),
                        geometry_lv95: geometry_lv95.clone(),
                    };

                    (code.clone(), FileInfo {
                        path: file_path.clone(),
                        size,
                        bbox,
                        geometry,
                        bbox_lv95,
                        geometry_lv95,
                        hash: None,
                    }, (cache_key, cached_entry))
                })
                .collect();

            let hits = cache_hits.load(std::sync::atomic::Ordering::Relaxed);
            let computed = file_results.len() as u64 - hits;
            file_pb.set_position(file_results.len() as u64);
            file_pb.finish_with_message(format!(
                "Extracted geometry from {} files ({} cached, {} computed)",
                file_results.len(), hits, computed
            ));
            info!("File geometry: {} total, {} cached, {} computed", file_results.len(), hits, computed);

            // Group results by item code and collect cache entries
            let mut files_by_code: HashMap<String, Vec<FileInfo>> = HashMap::new();
            let mut new_cache_entries: BTreeMap<String, CachedFileGeometry> = BTreeMap::new();
            for (code, file_info, (cache_key, cached_entry)) in file_results {
                files_by_code.entry(code).or_default().push(file_info);
                new_cache_entries.insert(cache_key, cached_entry);
            }

            // Save geometry cache
            {
                let cache_dir = output.join(".stac-cache");
                fs::create_dir_all(&cache_dir)?;
                let mut new_cache = GeometryCache::new(crs_hash);
                new_cache.entries = new_cache_entries;
                let cache_path = cache_dir.join("geometry_cache.json");
                new_cache.save(&cache_path)?;
                info!("Saved geometry cache ({} entries)", new_cache.entries.len());
            }

            for item in &mut items {
                if let Some(folder) = scanned_folders_for_report.get(&item.code) {
                    item.file_count = folder.files.len();
                    item.is_single_file = folder.is_single_file;
                    // Store folder path for computing relative file paths in URLs
                    // (for single files, path is already the parent directory)
                    item.folder_path = Some(folder.path.clone());
                }

                if let Some(file_infos) = files_by_code.remove(&item.code) {
                    // Collect bboxes for combined dataset bbox
                    let file_bboxes: Vec<&Vec<f64>> = file_infos
                        .iter()
                        .filter_map(|f| f.bbox.as_ref())
                        .collect();

                    // Compute combined dataset bbox from all file bboxes
                    if !file_bboxes.is_empty() {
                        let combined_bbox = vec![
                            file_bboxes.iter().map(|b| b[0]).fold(f64::INFINITY, f64::min),
                            file_bboxes.iter().map(|b| b[1]).fold(f64::INFINITY, f64::min),
                            file_bboxes.iter().map(|b| b[2]).fold(f64::NEG_INFINITY, f64::max),
                            file_bboxes.iter().map(|b| b[3]).fold(f64::NEG_INFINITY, f64::max),
                        ];
                        item.geometry = Some(bbox_to_polygon(&combined_bbox));
                        item.bbox = Some(combined_bbox);
                    }

                    item.files = file_infos;
                    item.files.sort_by(|a, b| a.path.cmp(&b.path));
                }
            }

            // Step 3a-fallback: Apply sensor_locations.json to items still missing geometry
            // This catches items with scanned folders but non-geospatial files (webcams, hydro, GNSS)
            if let Some(sensors) = sensors_for_fallback {
                let mut sensor_fallback_count = 0;
                for item in &mut items {
                    if item.geometry.is_none() {
                        if let Some(sensor) = sensors.get(&item.code) {
                            if let (Some(lon), Some(lat)) = (sensor.lon, sensor.lat) {
                                let (bbox, geometry) = point_to_geometry(lon, lat, sensor.elevation_m);
                                item.bbox = Some(bbox);
                                item.geometry = Some(geometry);
                                sensor_fallback_count += 1;
                            }
                        }
                    }
                }
                if sensor_fallback_count > 0 {
                    info!("  Applied {} sensor locations (fallback for items with non-geospatial files)", sensor_fallback_count);
                }
            }

            // Step 3b: Apply geometry overrides (inherit geometry from sibling items)
            if !config.geometry_overrides.is_empty() {
                // Collect source geometries first
                let source_geometries: HashMap<String, (Vec<f64>, serde_json::Value)> = items
                    .iter()
                    .filter_map(|item| {
                        if let (Some(bbox), Some(geom)) = (&item.bbox, &item.geometry) {
                            Some((item.code.clone(), (bbox.clone(), geom.clone())))
                        } else {
                            None
                        }
                    })
                    .collect();

                let mut override_count = 0;
                for item in &mut items {
                    if item.geometry.is_some() {
                        continue;
                    }
                    if let Some(ov) = config.geometry_overrides.get(&item.code) {
                        if let Some((bbox, geom)) = source_geometries.get(&ov.from_item) {
                            item.bbox = Some(bbox.clone());
                            item.geometry = Some(geom.clone());
                            override_count += 1;
                            debug!("  Geometry override: {} ← {}", item.code, ov.from_item);
                        } else {
                            warn!("  Geometry override for {} references {}, which has no geometry", item.code, ov.from_item);
                        }
                    }
                }
                if override_count > 0 {
                    info!("  Applied {} geometry overrides", override_count);
                }
            }

            // Step 4: Scan and map archive/data files from archives directory
            if let Some(ref arch_dir) = archives_dir {
                // Build a map of code -> file path by scanning the directory
                // Supports: archives (.zip, .7z) and standalone geospatial files (.tif, .tiff, .laz, .las)
                let mut archive_map: HashMap<String, (PathBuf, u64)> = HashMap::new();

                // Supported file extensions for archives/data files
                let supported_exts = ["zip", "7z", "tif", "tiff", "laz", "las"];

                for entry in WalkDir::new(arch_dir)
                    .min_depth(1)
                    .max_depth(1)
                    .into_iter()
                    .filter_map(|e| e.ok())
                {
                    if !entry.file_type().is_file() {
                        continue;
                    }
                    let ext = entry.path().extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                    if !supported_exts.contains(&ext.as_str()) {
                        continue;
                    }

                    let filename = entry.file_name().to_string_lossy().to_string();
                    // Extract code from filename (e.g., "14Ka03.zip" -> "14Ka03", "14Ka03.tif" -> "14Ka03")
                    if let Some(code) = extract_code_from_folder(&filename, &HashMap::new()) {
                        if let Ok(metadata) = entry.metadata() {
                            // Prefer archives over raw files if both exist
                            let dominated_by_archive = archive_map.get(&code)
                                .map(|(p, _)| {
                                    let existing_ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
                                    existing_ext == "zip" || existing_ext == "7z"
                                })
                                .unwrap_or(false);

                            if !dominated_by_archive {
                                archive_map.insert(code, (entry.path().to_path_buf(), metadata.len()));
                            }
                        }
                    }
                }

                // Map archives to items (parallel ZIP64 detection + optional CRC validation)
                let work: Vec<(usize, PathBuf, u64)> = items.iter().enumerate()
                    .filter(|(_, item)| item.archive_file.is_none())
                    .filter_map(|(idx, item)| {
                        archive_map.get(&item.code).map(|(path, size)| (idx, path.clone(), *size))
                    })
                    .collect();

                let arch_pb = multi_progress.add(ProgressBar::new(work.len() as u64));
                arch_pb.set_style(ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                    .expect("Invalid bar template")
                    .progress_chars("#>-"));
                arch_pb.set_message("Scanning archives...");

                let results: Vec<(usize, String, u64, bool)> = work.par_iter()
                    .map(|(idx, path, size)| {
                        let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                        let is_zip = path.extension().and_then(|e| e.to_str()) == Some("zip");
                        let is_zip64 = if is_zip {
                            let z64 = detect_zip64(path).unwrap_or(false);
                            if !skip_zip_validation {
                                if let Err(e) = validate_zip_crc(path) {
                                    warn!("CRC validation failed for {}: {}", filename, e);
                                }
                            }
                            z64
                        } else {
                            false
                        };
                        arch_pb.inc(1);
                        (*idx, filename, *size, is_zip64)
                    })
                    .collect();

                arch_pb.finish_with_message(format!("Scanned {} archives", results.len()));

                for (idx, filename, size, is_zip64) in &results {
                    items[*idx].archive_file = Some(filename.clone());
                    items[*idx].archive_size = Some(*size);
                    items[*idx].archive_is_zip64 = *is_zip64;
                }

                info!("  Archives/files: {} mapped from directory", results.len());
            }

            // Populate file hashes and archive hashes from manifest into items
            if let Some(ref manifest) = data_manifest {
                for item in &mut items {
                    // Populate file hashes
                    for file_info in &mut item.files {
                        let rel = if let Some(ref folder) = item.folder_path {
                            file_info.path.strip_prefix(folder)
                                .map(|p| p.to_string_lossy().to_string())
                                .unwrap_or_default()
                        } else {
                            file_info.path.file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_default()
                        };
                        let asset_path = format!("assets/{}/{}", item.code, rel);
                        if let Some(entry) = manifest.files.get(&asset_path) {
                            file_info.hash = Some(entry.hash.clone());
                        }
                    }
                    // Populate archive hash and ZIP64 flag
                    if let Some(archive_entry) = manifest.archives.get(&item.code) {
                        item.archive_hash = Some(archive_entry.hash.clone());
                        item.archive_is_zip64 = archive_entry.is_zip64;
                    }
                }
            }

            // Generate STAC catalog
            let (num_collections, num_items, num_assets, issues) = generate_catalog(&items, &output, &base_url, &s3_base_url, &multi_progress, &config)?;
            info!(
                "  Generated {} collections, {} items, {} assets",
                num_collections, num_items, num_assets
            );

            // Compute detailed quality metrics - build per-item issue map
            let excel_codes: std::collections::HashSet<_> = items.iter().map(|i| &i.code).collect();

            // Build a per-item issue map: HashMap<item_code, Vec<issue_description>>
            let mut item_issues: HashMap<String, Vec<String>> = HashMap::new();

            // Track counts for summary
            let mut count_no_data_folder = 0;
            let mut count_missing_geometry = 0;
            let mut count_no_archive = 0;

            for item in &items {
                let has_folder = scanned_folders_for_report.contains_key(&item.code);
                let has_files = scanned_folders_for_report.get(&item.code).map(|f| !f.files.is_empty()).unwrap_or(false);

                // Check: Excel item without data folder
                if !has_folder {
                    item_issues.entry(item.code.clone()).or_default().push("No data folder found (Excel entry only)".to_string());
                    count_no_data_folder += 1;
                }

                // Check: Missing geometry
                if item.geometry.is_none() {
                    let reason = if has_files {
                        "Missing geometry (files have no extractable CRS)"
                    } else {
                        "Missing geometry"
                    };
                    item_issues.entry(item.code.clone()).or_default().push(reason.to_string());
                    count_missing_geometry += 1;
                }

                // Check: Has folder but no archive mapped (skip single-file items — they don't need archives)
                let is_single = scanned_folders_for_report.get(&item.code).map_or(false, |f| f.is_single_file);
                if has_folder && item.archive_file.is_none() && !is_single {
                    item_issues.entry(item.code.clone()).or_default().push("No archive mapped".to_string());
                    count_no_archive += 1;
                }
            }

            // Add validation issues from generate_catalog (skip ones we already track)
            for issue in &issues {
                // Skip issues we already track in the quality report above
                if issue.message == "Missing geometry"
                    || issue.message == "No archive file mapped"
                    || issue.message == "No files found"
                {
                    continue;
                }
                item_issues.entry(issue.item_id.clone()).or_default().push(issue.message.clone());
            }

            // Orphan folders (not items, kept separate)
            let mut orphan_folders: Vec<_> = scanned_folders_for_report.keys()
                .filter(|code| !excel_codes.contains(code))
                .cloned()
                .collect();
            orphan_folders.sort();

            // Items with geometry (for stats)
            let items_with_geometry_count = items.iter().filter(|i| i.geometry.is_some()).count();

            // Print comprehensive report - grouped by item
            info!("");
            info!("╔══════════════════════════════════════════════════════════════╗");
            info!("║                    DATA QUALITY REPORT                       ║");
            info!("╚══════════════════════════════════════════════════════════════╝");

            // Items with issues (grouped by item ID)
            let items_with_issues_count = item_issues.len();
            if !item_issues.is_empty() {
                info!("");
                info!("=== Items With Issues ({}) ===", items_with_issues_count);
                let mut sorted_codes: Vec<_> = item_issues.keys().cloned().collect();
                sorted_codes.sort();
                for code in sorted_codes {
                    if let Some(issues_list) = item_issues.get(&code) {
                        warn!("  {}:", code);
                        for issue_msg in issues_list {
                            warn!("    - {}", issue_msg);
                        }
                    }
                }
            }

            // Orphan folders (not items, kept separate)
            if !orphan_folders.is_empty() {
                info!("");
                info!("=== Orphan Folders (not in Excel) ({}) ===", orphan_folders.len());
                for code in &orphan_folders {
                    warn!("  - {}", code);
                }
            }

            // Applied overrides section
            if !applied_overrides.is_empty() {
                info!("");
                info!("=== Applied Overrides ({}) ===", applied_overrides.len());
                for (code, source, reason) in &applied_overrides {
                    info!("  [{}] {} - {}", code, source, reason);
                }
            }

            // === SUMMARY ===
            info!("");
            info!("=== Summary ===");
            info!("  Total items:              {}", items.len());
            info!("  Items with issues:        {} ({:.1}%)",
                items_with_issues_count,
                if items.is_empty() { 0.0 } else { items_with_issues_count as f64 / items.len() as f64 * 100.0 }
            );
            info!("  Items with geometry:      {} ({:.1}%)",
                items_with_geometry_count,
                if items.is_empty() { 0.0 } else { items_with_geometry_count as f64 / items.len() as f64 * 100.0 }
            );
            info!("  Items without geometry:   {}", count_missing_geometry);
            info!("  Items without data:       {}", count_no_data_folder);
            info!("  Items without archives:   {}", count_no_archive);
            info!("  Orphan folders:           {}", orphan_folders.len());

            // Run validation after generation if requested
            if validate {
                let warning_count = issues.iter().filter(|i| i.severity == "warning").count();
                let error_count = issues.iter().filter(|i| i.severity == "error").count();

                info!("");
                if error_count > 0 {
                    error!("Validation FAILED ({} warnings, {} errors)", warning_count, error_count);
                    std::process::exit(1);
                } else {
                    info!("Validation passed ({} warnings, {} errors)", warning_count, error_count);
                }
            }
        }

    Ok(())
}

// =============================================================================
// Archive Operations
// =============================================================================

/// Create a zip archive from a directory, returning the archive size in bytes.
/// ZIP64 detection is handled separately by `detect_zip64()`.
fn create_archive(source_dir: &Path, archive_path: &Path, exclude: &ExcludeFiles) -> Result<u64> {
    use zip::write::SimpleFileOptions;

    let file = File::create(archive_path)?;
    let mut zip = zip::ZipWriter::new(file);
    let base_options = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored);
    let dir_options = base_options;

    // Walk the directory and add all files (follow symlinks for staged assets)
    for entry in WalkDir::new(source_dir).follow_links(true).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        let name = path.strip_prefix(source_dir)
            .map_err(|e| anyhow::anyhow!("Failed to strip prefix: {}", e))?;

        if path.is_file() && !is_junk_file(path, exclude) {
            let file_size = fs::metadata(path)?.len();
            // large_file() is still needed for correct zip writing of >4GB entries
            let options = base_options.large_file(file_size >= 0xFFFFFFFF);
            zip.start_file(name.to_string_lossy(), options)?;
            let mut f = File::open(path)?;
            std::io::copy(&mut f, &mut zip)?;
        } else if !name.as_os_str().is_empty() {
            // Directory entry (not the root)
            zip.add_directory(name.to_string_lossy(), dir_options)?;
        }
    }

    zip.finish()?;
    let size = fs::metadata(archive_path)?.len();
    Ok(size)
}

/// Validate a zip archive's integrity and detect whether it uses ZIP64 extensions.
///
/// Opens the archive, reads every entry through the CRC32 reader (full data integrity check),
/// and inspects the zip structure for ZIP64 markers:
/// - EOCD64 record (zip64 comment present)
/// - Per-entry ZIP64 extra fields (large_file flag)
/// - Entry count >= ZIP64_ENTRY_THR (65535)
///
/// Returns `Ok(true)` if the archive uses ZIP64, `Ok(false)` otherwise, or `Err` if corrupt.
/// Cheap ZIP64 detection: reads only the central directory, not file data.
fn detect_zip64(archive_path: &Path) -> Result<bool> {
    use zip::read::HasZipMetadata;

    let file = File::open(archive_path)
        .with_context(|| format!("Failed to open archive: {}", archive_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("Failed to read zip structure: {}", archive_path.display()))?;

    // Check archive-level ZIP64 indicators
    let has_eocd64 = archive.zip64_comment().is_some();
    let many_entries = archive.len() >= zip::ZIP64_ENTRY_THR;

    // Check per-entry ZIP64 metadata (no data read needed)
    let mut has_large_entry = false;
    for i in 0..archive.len() {
        let entry = archive.by_index_raw(i)
            .with_context(|| format!("Failed to read entry {} in {}", i, archive_path.display()))?;
        if entry.get_metadata().large_file {
            has_large_entry = true;
            break;
        }
    }

    Ok(has_eocd64 || many_entries || has_large_entry)
}

/// Expensive CRC32 validation: reads every byte of every entry.
fn validate_zip_crc(archive_path: &Path) -> Result<()> {
    let file = File::open(archive_path)
        .with_context(|| format!("Failed to open archive: {}", archive_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("Failed to read zip structure: {}", archive_path.display()))?;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)
            .with_context(|| format!("Failed to read entry {} in {}", i, archive_path.display()))?;
        // Read through CRC32 reader — validates data integrity
        std::io::copy(&mut entry, &mut std::io::sink())
            .with_context(|| format!("CRC32 validation failed for entry {} in {}", entry.name(), archive_path.display()))?;
    }

    Ok(())
}

/// Validate that a zip archive's file list matches the assets directory exactly.
/// Compares both file presence AND file sizes (uncompressed).
/// Returns Ok(()) if all files match, or an error describing mismatches.
fn validate_zip_contents(archive_path: &Path, assets_dir: &Path, code: &str, exclude: &ExcludeFiles) -> Result<()> {
    let code_dir = assets_dir.join(code);
    if !code_dir.exists() {
        anyhow::bail!("Assets directory does not exist: {}", code_dir.display());
    }

    // Collect files on disk with sizes (same walk logic as create_archive)
    let mut disk_files: HashMap<String, u64> = HashMap::new();
    for entry in WalkDir::new(&code_dir).follow_links(true).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if !path.is_file() { continue; }
        if is_junk_file(path, exclude) { continue; }
        if let Ok(rel) = path.strip_prefix(&code_dir) {
            let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            disk_files.insert(rel.to_string_lossy().to_string(), size);
        }
    }

    // Collect files in zip with uncompressed sizes
    let file = File::open(archive_path)
        .with_context(|| format!("Failed to open archive: {}", archive_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("Failed to read zip: {}", archive_path.display()))?;
    let mut zip_files: HashMap<String, u64> = HashMap::new();
    for i in 0..archive.len() {
        let entry = archive.by_index_raw(i)
            .with_context(|| format!("Failed to read entry {} in {}", i, archive_path.display()))?;
        let name = entry.name().to_string();
        if !name.ends_with('/') {
            zip_files.insert(name, entry.size());
        }
    }

    // Compare: presence and sizes
    let mut in_disk_not_zip: Vec<&String> = Vec::new();
    let mut in_zip_not_disk: Vec<&String> = Vec::new();
    let mut size_mismatches: Vec<(String, u64, u64)> = Vec::new();

    for name in disk_files.keys() {
        if !zip_files.contains_key(name) {
            in_disk_not_zip.push(name);
        }
    }
    for name in zip_files.keys() {
        if !disk_files.contains_key(name) {
            in_zip_not_disk.push(name);
        }
    }
    // Check sizes for files present in both
    for (name, disk_size) in &disk_files {
        if let Some(&zip_size) = zip_files.get(name) {
            if *disk_size != zip_size {
                size_mismatches.push((name.clone(), *disk_size, zip_size));
            }
        }
    }

    in_disk_not_zip.sort();
    in_zip_not_disk.sort();
    size_mismatches.sort_by(|a, b| a.0.cmp(&b.0));

    if !in_disk_not_zip.is_empty() || !in_zip_not_disk.is_empty() || !size_mismatches.is_empty() {
        let mut msg = format!("Zip content mismatch for {}:", code);
        for f in &in_disk_not_zip { msg += &format!("\n  in assets/ but NOT in zip: {}", f); }
        for f in &in_zip_not_disk { msg += &format!("\n  in zip but NOT in assets/: {}", f); }
        for (f, disk_sz, zip_sz) in &size_mismatches {
            msg += &format!("\n  size mismatch: {} (disk: {} vs zip: {})", f, format_size(*disk_sz), format_size(*zip_sz));
        }
        anyhow::bail!(msg);
    }
    Ok(())
}

/// Format bytes as human-readable size
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

// =============================================================================
// File Hashing
// =============================================================================

/// Compute hash of a file using the specified algorithm
fn hash_file(path: &Path, algorithm: HashAlgorithm) -> Result<String> {
    use std::io::Read;

    let mut file = File::open(path)?;
    let mut buffer = vec![0u8; 64 * 1024]; // 64KB buffer

    match algorithm {
        HashAlgorithm::Sha256 => {
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            loop {
                let bytes_read = file.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }
                hasher.update(&buffer[..bytes_read]);
            }
            Ok(format!("{:x}", hasher.finalize()))
        }
        HashAlgorithm::Xxhash => {
            use xxhash_rust::xxh64::Xxh64;
            let mut hasher = Xxh64::new(0);
            loop {
                let bytes_read = file.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }
                hasher.update(&buffer[..bytes_read]);
            }
            Ok(format!("{:016x}", hasher.digest()))
        }
    }
}

/// Get file modification time as Unix timestamp
#[cfg(unix)]
fn get_mtime(path: &Path) -> Result<i64> {
    use std::os::unix::fs::MetadataExt;
    let meta = fs::metadata(path)?;
    Ok(meta.mtime())
}

#[cfg(not(unix))]
fn get_mtime(path: &Path) -> Result<i64> {
    let meta = fs::metadata(path)?;
    let modified = meta.modified()?;
    let duration = modified.duration_since(std::time::UNIX_EPOCH)?;
    Ok(duration.as_secs() as i64)
}

/// Compute a content fingerprint for a code's files in the manifest.
/// The fingerprint changes if any file's name, size, or hash changes.
fn compute_archive_fingerprint(manifest: &DataManifest, code: &str) -> String {
    use sha2::{Sha256, Digest};
    let mut entries: Vec<_> = manifest.files.iter()
        .filter(|(_, e)| e.code == code)
        .collect();
    entries.sort_by_key(|(path, _)| *path);

    let mut hasher = Sha256::new();
    for (path, entry) in &entries {
        hasher.update(format!("{}:{}:{}\n", path, entry.size, entry.hash).as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

// =============================================================================
// Link / Copy Helper
// =============================================================================

/// Link or copy a file according to the selected link mode.
/// For symlinks: creates an absolute symlink.
/// For hardlinks: creates a hard link (same filesystem required).
/// For copy: copies the file.
fn link_or_copy(source: &Path, target: &Path, mode: LinkMode) -> Result<()> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    match mode {
        LinkMode::Symlink => {
            let abs_source = fs::canonicalize(source)
                .with_context(|| format!("Cannot resolve source: {:?}", source))?;
            #[cfg(unix)]
            std::os::unix::fs::symlink(&abs_source, target)
                .with_context(|| format!("symlink {:?} -> {:?}", target, abs_source))?;
            #[cfg(not(unix))]
            fs::copy(source, target)
                .with_context(|| format!("copy {:?} -> {:?} (symlink not supported)", source, target))
                .map(|_| ())?;
        }
        LinkMode::Hardlink => {
            fs::hard_link(source, target)
                .with_context(|| format!("hardlink {:?} -> {:?}", target, source))?;
        }
        LinkMode::Copy => {
            fs::copy(source, target)
                .with_context(|| format!("copy {:?} -> {:?}", source, target))
                .map(|_| ())?;
        }
    }
    Ok(())
}

// =============================================================================
// Assets-First Pipeline: Stage, Validate, Materialize
// =============================================================================

struct StageAssetsStats {
    staged: usize,
    extracted: usize,
    skipped: usize,
    preserved: usize,
    /// All expected target paths (for stale file cleanup)
    expected_files: HashSet<PathBuf>,
}

/// Compare scanned folders against existing manifest to preview what will change.
/// `extra_expected` includes additional files (e.g. factsheets) that will be injected
/// after staging, so they shouldn't count as "removed".
/// Returns (new, removed, modified, unchanged) file counts.
fn preview_staging_changes(
    scanned: &HashMap<String, ScannedFolder>,
    manifest: &DataManifest,
    extra_expected: &HashMap<String, u64>,
) -> (usize, usize, usize, usize) {
    // Build expected: asset_path -> source file size
    let mut expected: HashMap<String, u64> = HashMap::new();
    for (code, folder) in scanned {
        for source in &folder.files {
            let rel_path = match source.strip_prefix(&folder.path) {
                Ok(p) => p.to_path_buf(),
                Err(_) => source.file_name()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| source.clone()),
            };
            let asset_path = format!("assets/{}/{}", code, rel_path.to_string_lossy());
            let size = fs::metadata(source).map(|m| m.len()).unwrap_or(0);
            expected.insert(asset_path, size);
        }
    }

    // Merge extra expected files (factsheets etc.) so they don't appear as "removed"
    expected.extend(extra_expected.iter().map(|(k, &v)| (k.clone(), v)));

    let mut new = 0usize;
    let mut modified = 0usize;
    let mut unchanged = 0usize;

    for (asset_path, &source_size) in &expected {
        match manifest.files.get(asset_path) {
            None => new += 1,
            Some(entry) if entry.size != source_size => modified += 1,
            Some(_) => unchanged += 1,
        }
    }

    let removed = manifest.files.keys()
        .filter(|k| !expected.contains_key(k.as_str()))
        .count();

    (new, removed, modified, unchanged)
}

/// Stage assets from scanned FINAL_Data folders into `assets/<code>/` using the chosen link mode.
fn stage_assets(
    scanned: &HashMap<String, ScannedFolder>,
    _item_codes: &[String],
    assets_dir: &Path,
    link_mode: LinkMode,
    _folder_mappings: &HashMap<String, FolderMapping>,
    progress: &ProgressBar,
    _exclude: &ExcludeFiles,
) -> Result<StageAssetsStats> {
    let mut stats = StageAssetsStats {
        staged: 0, extracted: 0, skipped: 0, preserved: 0,
        expected_files: HashSet::new(),
    };

    // Count total work
    let total_files: usize = scanned.values().map(|f| f.files.len()).sum();
    progress.set_length(total_files as u64);
    progress.set_message("Staging assets...");

    // Stage from scanned folders — preserve materialized (real) files, only recreate symlinks
    for (code, folder) in scanned {
        let code_dir = assets_dir.join(code);

        for source in &folder.files {
            let rel_path = match source.strip_prefix(&folder.path) {
                Ok(p) => p.to_path_buf(),
                Err(_) => source.file_name()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| source.clone()),
            };
            let target = code_dir.join(&rel_path);
            stats.expected_files.insert(target.clone());

            if let Ok(meta) = target.symlink_metadata() {
                // Target exists — check if it's a real file (materialized) or a symlink
                if meta.file_type().is_symlink() {
                    // Symlink: recreate it (source path may have changed)
                    let _ = fs::remove_file(&target);
                    link_or_copy(source, &target, link_mode)?;
                    stats.staged += 1;
                } else {
                    // Real file (materialized copy or hardlink): preserve it
                    stats.preserved += 1;
                }
            } else {
                // New file: create link/copy
                link_or_copy(source, &target, link_mode)?;
                stats.staged += 1;
            }
            progress.inc(1);
        }
    }

    progress.finish_with_message(format!(
        "Assets: {} staged, {} preserved, {} extracted, {} skipped",
        stats.staged, stats.preserved, stats.extracted, stats.skipped
    ));

    // Report materialization status
    let total = stats.staged + stats.preserved;
    if total > 0 && stats.preserved > 0 {
        let pct_materialized = (stats.preserved as f64 / total as f64) * 100.0;
        let pct_symlinks = (stats.staged as f64 / total as f64) * 100.0;
        info!("  Materialization: {:.1}% materialized ({} files), {:.1}% new symlinks ({} files)",
            pct_materialized, stats.preserved, pct_symlinks, stats.staged);
        if stats.staged > 0 {
            warn!("  Staging directory is partially materialized — re-run without --no-materialize before upload");
        }
    } else if total > 0 {
        info!("  Staging directory is non-materialized ({} symlinks) — re-run without --no-materialize before upload", stats.staged);
    }

    Ok(stats)
}

/// Hash all files in the staged assets directory, building a DataManifest.
/// Walks `assets/<code>/<file>` in parallel, following symlinks.
/// If `existing_manifest` is provided, carries forward hashes for files with matching size (incremental).
fn hash_staged_assets(
    assets_dir: &Path,
    algorithm: HashAlgorithm,
    existing_manifest: Option<&DataManifest>,
    scanned: &HashMap<String, ScannedFolder>,
    progress: &ProgressBar,
    exclude: &ExcludeFiles,
) -> Result<DataManifest> {
    let mut manifest = DataManifest::new(algorithm);

    // Collect all files from assets/ by walking code directories
    let mut all_files: Vec<(String, PathBuf, String)> = Vec::new(); // (code, full_path, asset_path)

    for (code, folder) in scanned {
        let code_dir = assets_dir.join(code);
        if !code_dir.exists() {
            continue;
        }
        // Walk the code directory (follows symlinks by default)
        for entry in WalkDir::new(&code_dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() || is_junk_file(entry.path(), exclude) {
                continue;
            }
            let rel = entry.path().strip_prefix(&code_dir)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();
            let asset_path = format!("assets/{}/{}", code, rel);
            all_files.push((code.clone(), entry.path().to_path_buf(), asset_path));
        }

        // Ignore unused variable warning
        let _ = folder;
    }

    // Also walk code directories that came from archive extraction (not in scanned)
    if assets_dir.exists() {
        for entry in WalkDir::new(assets_dir)
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_dir() {
                continue;
            }
            let code = entry.file_name().to_string_lossy().to_string();
            if scanned.contains_key(&code) {
                continue; // already handled above
            }
            let code_dir = entry.path();
            for file_entry in WalkDir::new(code_dir)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if !file_entry.file_type().is_file() || is_junk_file(file_entry.path(), exclude) {
                    continue;
                }
                let rel = file_entry.path().strip_prefix(code_dir)
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                let asset_path = format!("assets/{}/{}", code, rel);
                all_files.push((code.clone(), file_entry.path().to_path_buf(), asset_path));
            }
        }
    }

    progress.set_length(all_files.len() as u64);
    progress.set_message("Hashing assets...");

    // Hash files in parallel
    let hashed_count = std::sync::atomic::AtomicU64::new(0);
    let results: Vec<Result<(String, FileManifestEntry)>> = all_files
        .par_iter()
        .map(|(code, full_path, asset_path)| {
            let meta = fs::metadata(full_path)
                .with_context(|| format!("metadata for {:?}", full_path))?;
            let size = meta.len();

            // Incremental: carry forward hash if file size matches existing manifest
            let hash = if let Some(existing) = existing_manifest {
                if let Some(entry) = existing.files.get(asset_path.as_str()) {
                    if entry.size == size {
                        entry.hash.clone() // carry forward
                    } else {
                        hash_file(full_path, algorithm)?
                    }
                } else {
                    hash_file(full_path, algorithm)?
                }
            } else {
                hash_file(full_path, algorithm)?
            };

            let mtime = get_mtime(full_path).unwrap_or(0);

            // Compute archive_path (relative within archive): <code>/<rel>
            let archive_path = asset_path.strip_prefix("assets/").unwrap_or(asset_path).to_string();

            let count = hashed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count % 100 == 0 {
                progress.set_position(count);
            }

            Ok((asset_path.clone(), FileManifestEntry {
                asset_path: asset_path.clone(),
                archive_path,
                source_path: full_path.to_string_lossy().to_string(),
                size,
                hash,
                code: code.clone(),
                provider: String::new(), // not critical for manifest
                mtime,
            }))
        })
        .collect();

    // Collect results
    let mut errors = 0;
    for result in results {
        match result {
            Ok((key, entry)) => { manifest.files.insert(key, entry); }
            Err(e) => { warn!("Hash error: {}", e); errors += 1; }
        }
    }

    manifest.update_totals();
    progress.set_position(manifest.total_files as u64);
    progress.finish_with_message(format!(
        "Hashed {} files ({}){}", manifest.total_files, format_size(manifest.total_size),
        if errors > 0 { format!(", {} errors", errors) } else { String::new() }
    ));

    Ok(manifest)
}

struct StageArchivesStats {
    created: usize,
    skipped: usize,
}

/// Result of processing a single archive in parallel
enum ArchiveResult {
    /// Carried forward from existing manifest (unchanged)
    Unchanged { code: String, entry: ArchiveManifestEntry },
    /// Created — has new hash and entry
    Processed { code: String, entry: ArchiveManifestEntry },
    /// Skipped (no assets, no source)
    Skipped,
    /// Error during processing
    Error { code: String, error: String },
}

/// Stage archives: create `archives/<code>.zip` from staged `assets/<code>/` for every code.
/// Skips unchanged archives via content fingerprinting.
/// Archive creation and hashing runs in parallel. Records archive hashes into the manifest.
fn stage_archives(
    scanned: &HashMap<String, ScannedFolder>,
    item_codes: &[String],
    assets_dir: &Path,
    archives_dir: &Path,
    manifest: &mut DataManifest,
    existing_manifest: Option<&DataManifest>,
    progress: &ProgressBar,
    exclude: &ExcludeFiles,
    skip_zip_validation: bool,
) -> Result<StageArchivesStats> {
    fs::create_dir_all(archives_dir)?;

    // Collect all codes that need archives
    let all_codes: Vec<String> = {
        let set: std::collections::HashSet<String> = item_codes.iter().cloned()
            .chain(scanned.keys().cloned())
            .collect();
        let mut v: Vec<String> = set.into_iter().collect();
        v.sort();
        v
    };

    progress.set_length(all_codes.len() as u64);
    progress.set_message("Creating archives...");

    // Pre-compute fingerprints and file counts (read-only on manifest)
    let work_items: Vec<(String, PathBuf, String, usize)> = all_codes.iter().map(|code| {
        let archive_path = archives_dir.join(format!("{}.zip", code));
        let fingerprint = compute_archive_fingerprint(manifest, code);
        let file_count = manifest.files.values().filter(|e| e.code == *code).count();
        (code.clone(), archive_path, fingerprint, file_count)
    }).collect();

    // Create archives in parallel from assets/<code>/
    let results: Vec<ArchiveResult> = work_items
        .par_iter()
        .map(|(code, archive_path, fingerprint, file_count)| {
            let asset_dir = assets_dir.join(code);
            if !asset_dir.exists() {
                progress.inc(1);
                return ArchiveResult::Skipped;
            }
            // Skip single-file items — no point zipping a single file
            if scanned.get(code).map_or(false, |f| f.is_single_file) {
                progress.inc(1);
                return ArchiveResult::Skipped;
            }

            let action = classify_archive_action(code, archive_path, fingerprint, existing_manifest);

            // For unchanged archives, validate content is still in sync (cheap metadata check).
            // If valid, carry forward the existing manifest entry and skip rebuild.
            if matches!(action, ArchiveAction::Unchanged) {
                if let Err(e) = validate_zip_contents(archive_path, assets_dir, code, exclude) {
                    warn!("{} content mismatch — rebuilding: {}", code, e);
                    let _ = fs::remove_file(archive_path);
                    // Fall through to rebuild below
                } else {
                    let existing_entry = existing_manifest.unwrap().archives.get(code).unwrap();
                    progress.inc(1);
                    return ArchiveResult::Unchanged { code: code.clone(), entry: existing_entry.clone() };
                }
            }

            // Rebuild: create/adopt/rebuild, or stale-unchanged that failed validation above
            match create_archive(&asset_dir, archive_path, exclude) {
                Ok(size) => {
                    // Detect ZIP64 (cheap) and optionally validate CRC (expensive)
                    let is_zip64 = match detect_zip64(archive_path) {
                        Ok(z64) => z64,
                        Err(e) => {
                            progress.inc(1);
                            return ArchiveResult::Error { code: code.clone(), error: format!("zip64 detect: {}", e) };
                        }
                    };
                    // Content validation is cheap (metadata only) — always run it
                    if let Err(e) = validate_zip_contents(archive_path, assets_dir, code, exclude) {
                        progress.inc(1);
                        return ArchiveResult::Error { code: code.clone(), error: format!("content validate: {}", e) };
                    }
                    // CRC32 validation is expensive (reads every byte) — skip with --skip-zip-validation
                    if !skip_zip_validation {
                        if let Err(e) = validate_zip_crc(archive_path) {
                            progress.inc(1);
                            return ArchiveResult::Error { code: code.clone(), error: format!("crc validate: {}", e) };
                        }
                    }
                    match hash_file(archive_path, HashAlgorithm::Sha256) {
                        Ok(archive_hash) => {
                            let sidecar_path = archives_dir.join(format!("{}.zip.sha256", code));
                            let _ = fs::write(&sidecar_path, format!("{}  {}.zip\n", archive_hash, code));
                            progress.inc(1);
                            ArchiveResult::Processed {
                                code: code.clone(),
                                entry: ArchiveManifestEntry {
                                    path: archive_path.to_string_lossy().to_string(),
                                    size,
                                    hash: archive_hash,
                                    content_fingerprint: fingerprint.clone(),
                                    file_count: *file_count,
                                    is_zip64,
                                },
                            }
                        }
                        Err(e) => {
                            progress.inc(1);
                            ArchiveResult::Error { code: code.clone(), error: format!("hash: {}", e) }
                        }
                    }
                }
                Err(e) => {
                    progress.inc(1);
                    ArchiveResult::Error { code: code.clone(), error: format!("create: {}", e) }
                }
            }
        })
        .collect();

    // Merge results into manifest sequentially
    let mut stats = StageArchivesStats { created: 0, skipped: 0 };
    for result in results {
        match result {
            ArchiveResult::Unchanged { code, entry } => {
                manifest.archives.insert(code, entry);
                stats.skipped += 1;
            }
            ArchiveResult::Processed { code, entry } => {
                manifest.archives.insert(code, entry);
                stats.created += 1;
            }
            ArchiveResult::Skipped => {}
            ArchiveResult::Error { code, error } => {
                warn!("Failed to process archive for {}: {}", code, error);
            }
        }
    }

    progress.finish_with_message(format!(
        "Archives: {} created, {} unchanged",
        stats.created, stats.skipped
    ));
    Ok(stats)
}

/// Materialize staged links (symlinks/hardlinks) into real file copies.
/// Walks the given directory and replaces symlinks with real copies,
/// and hardlinks (nlink > 1) with independent copies.
fn materialize_links(dir: &Path) -> Result<(usize, usize)> {
    use std::os::unix::fs::MetadataExt;

    let mut materialized = 0usize;
    let mut already_real = 0usize;

    for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let symlink_meta = path.symlink_metadata()?;

        if symlink_meta.file_type().is_symlink() {
            // Resolve symlink target
            let real_path = fs::canonicalize(path)?;
            fs::remove_file(path)?;
            fs::copy(&real_path, path)?;
            materialized += 1;
        } else if symlink_meta.nlink() > 1 {
            // Hardlink — copy in place via tmp
            let tmp = path.with_extension("__materialize_tmp");
            fs::copy(path, &tmp)?;
            fs::remove_file(path)?;
            fs::rename(&tmp, path)?;
            materialized += 1;
        } else {
            already_real += 1;
        }
    }

    Ok((materialized, already_real))
}

// =============================================================================
// Folder Scanning
// =============================================================================

/// Scan FINAL_Data directory for provider folders and map to item codes
/// Supports nested dataset detection where child codes exist within parent folders
fn scan_data_folders(
    data: &Path,
    folder_mappings: &HashMap<String, FolderMapping>,
    exclude: &ExcludeFiles,
) -> Result<HashMap<String, ScannedFolder>> {
    let mut folders: HashMap<String, ScannedFolder> = HashMap::new();

    // Auto-discover provider directories (all top-level directories in data path)
    let mut providers: Vec<String> = fs::read_dir(data)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map_or(false, |t| t.is_dir()))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|name| !name.starts_with('.'))
        .collect();
    providers.sort(); // deterministic order

    for provider in &providers {
        let provider_path = data.join(provider);

        for entry in WalkDir::new(&provider_path)
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_dir() {
                let folder_name = entry.file_name().to_string_lossy().to_string();

                // Extract code from folder name (e.g., "13Da01_Orthophoto_..." -> "13Da01")
                let parent_code = extract_code_from_folder(&folder_name, folder_mappings);

                if let Some(parent_code) = parent_code {
                    // Scan for nested code folders and collect files
                    let nested_results = scan_folder_with_nested(entry.path(), &parent_code, provider, exclude);

                    // Add all discovered folders (parent and children)
                    for (code, folder) in nested_results {
                        // If parent already exists, merge files and nested_codes
                        if let Some(existing) = folders.get_mut(&code) {
                            // Merge files (avoid duplicates)
                            for file in folder.files {
                                if !existing.files.contains(&file) {
                                    existing.files.push(file);
                                }
                            }
                            // Merge nested_codes
                            for nested in folder.nested_codes {
                                if !existing.nested_codes.contains(&nested) {
                                    existing.nested_codes.push(nested);
                                }
                            }
                        } else {
                            folders.insert(code, folder);
                        }
                    }
                }
            } else if entry.file_type().is_file() && !is_junk_file(entry.path(), exclude) {
                // Single file (e.g., standalone .tif in Terradata/)
                let file_name = entry.file_name().to_string_lossy().to_string();
                if let Some(code) = extract_code_from_folder(&file_name, folder_mappings) {
                    if !folders.contains_key(&code) {
                        folders.insert(code.clone(), ScannedFolder {
                            code: code.clone(),
                            // Use parent dir so strip_prefix yields the filename in stage_assets
                            path: entry.path().parent().unwrap_or(entry.path()).to_path_buf(),
                            provider: provider.to_string(),
                            files: vec![entry.path().to_path_buf()],
                            archive_path: None,
                            archive_size: None,
                            nested_codes: Vec::new(),
                            is_single_file: true,
                        });
                    }
                }
            }
        }
    }

    // Apply folder mappings for shared/non-standard folders
    for (code, mapping) in folder_mappings {
        if folders.contains_key(code.as_str()) {
            continue;
        }

        let folder_rels = &mapping.folders;
        if folder_rels.is_empty() {
            continue;
        }

        let mut all_files: Vec<PathBuf> = Vec::new();
        let mut first_path: Option<PathBuf> = None;
        for folder_rel in folder_rels {
            let folder_path = data.join(folder_rel);
            if folder_path.exists() {
                if first_path.is_none() {
                    first_path = Some(folder_path.clone());
                }
                let files: Vec<PathBuf> = WalkDir::new(&folder_path)
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_type().is_file() && !is_junk_file(e.path(), exclude))
                    .map(|e| e.path().to_path_buf())
                    .collect();
                all_files.extend(files);
            }
        }

        if let Some(path) = first_path {
            // For multi-folder mappings, derive base path from the first folder's provider component
            let base_path = if folder_rels.len() > 1 {
                folder_rels[0].split('/').next()
                    .map(|p| data.join(p))
                    .unwrap_or(path)
            } else {
                path
            };
            // Derive provider from the folder path's first component (e.g., "DNAGE/10M_11M_GNSS" → "DNAGE")
            let provider = folder_rels[0].split('/').next()
                .unwrap_or("unknown")
                .to_string();
            folders.insert(code.clone(), ScannedFolder {
                code: code.clone(),
                path: base_path,
                provider,
                files: all_files,
                archive_path: None,
                archive_size: None,
                nested_codes: Vec::new(),
                is_single_file: false,
            });
        }
    }

    Ok(folders)
}

/// Scan a folder and detect nested code subfolders
/// Returns HashMap<code, ScannedFolder> with:
/// - Parent (00 suffix): Gets ALL files in folder
/// - Children (01, 02, ...): Get files from their specific subfolder
/// - Both parent and children exist as separate entries
fn scan_folder_with_nested(
    folder_path: &Path,
    parent_code: &str,
    provider: &str,
    exclude: &ExcludeFiles,
) -> HashMap<String, ScannedFolder> {
    let mut results: HashMap<String, ScannedFolder> = HashMap::new();

    // Initialize parent with empty file/nested lists
    results.insert(parent_code.to_string(), ScannedFolder {
        code: parent_code.to_string(),
        path: folder_path.to_path_buf(),
        provider: provider.to_string(),
        files: Vec::new(),
        archive_path: None,
        archive_size: None,
        nested_codes: Vec::new(),
        is_single_file: false,
    });

    // Walk all entries in the folder
    for entry in WalkDir::new(folder_path).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();

        // Check for nested code folder (directory with a different code pattern)
        if path.is_dir() && path != folder_path {
            let dir_name = path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            // Try to extract a code from this subdirectory name
            if let Some(nested_code) = extract_nested_code(&dir_name) {
                // Only create new entry if it's different from parent
                if nested_code != parent_code {
                    // Track as nested code in parent
                    if let Some(parent) = results.get_mut(parent_code) {
                        if !parent.nested_codes.contains(&nested_code) {
                            parent.nested_codes.push(nested_code.clone());
                        }
                    }

                    // Create entry for nested code if not exists
                    results.entry(nested_code.clone()).or_insert_with(|| ScannedFolder {
                        code: nested_code,
                        path: path.to_path_buf(),
                        provider: provider.to_string(),
                        files: Vec::new(),
                        archive_path: None,
                        archive_size: None,
                        nested_codes: Vec::new(),
                        is_single_file: false,
                    });
                }
            }
        }

        // Add files to appropriate codes
        if path.is_file() && !is_junk_file(path, exclude) {
            // Always add to parent
            if let Some(parent) = results.get_mut(parent_code) {
                parent.files.push(path.to_path_buf());
            }

            // Also add to nested code if file is inside a nested code folder
            for ancestor in path.ancestors().skip(1) {
                if ancestor == folder_path {
                    break;
                }
                let ancestor_name = ancestor.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();

                if let Some(nested_code) = extract_nested_code(&ancestor_name) {
                    if nested_code != parent_code {
                        if let Some(folder) = results.get_mut(&nested_code) {
                            if !folder.files.contains(&path.to_path_buf()) {
                                folder.files.push(path.to_path_buf());
                            }
                        }
                        break;
                    }
                }
            }
        }
    }

    results
}

/// Extract a nested code from a folder/file name
/// Matches patterns like "01Aa01_Sensalpin-Camera" or just "01Aa01"
fn extract_nested_code(name: &str) -> Option<String> {
    let code_re = regex::Regex::new(r"^(\d{2}[A-Z][a-z]\d{2})").ok()?;
    code_re.captures(name).and_then(|caps| {
        caps.get(1).map(|m| m.as_str().to_string())
    })
}

/// Extract item code from folder name
fn extract_code_from_folder(folder_name: &str, folder_mappings: &HashMap<String, FolderMapping>) -> Option<String> {
    // Pattern: code at start followed by underscore (e.g., "13Da01_...")
    let code_re = regex::Regex::new(r"^(\d{2}[A-Z][a-z]\d{2})").ok()?;

    if let Some(caps) = code_re.captures(folder_name) {
        return Some(caps.get(1)?.as_str().to_string());
    }

    // Check folder mappings for non-standard folder names
    for (code, mapping) in folder_mappings {
        for folder in &mapping.folders {
            if folder.contains(folder_name) || folder_name.contains(folder.split('/').last().unwrap_or("")) {
                return Some(code.clone());
            }
        }
    }

    None
}

/// Extract geometry using new parallel-friendly interface
fn extract_geometry_for_item_v2(
    item: &ItemMetadata,
    sensors: &HashMap<String, SensorLocation>,
    scanned_folders: &HashMap<String, ScannedFolder>,
    crs_overrides: &[CrsOverride],
) -> ExtractionResult {
    let code = &item.code;
    let data_format = item.format.as_deref().unwrap_or("").to_uppercase();

    // Check if format has extractable geometry
    let is_geospatial = matches!(data_format.as_str(), "TIF" | "TIFF" | "LAZ" | "LAS" | "ASC");

    // Try to extract from scanned folder files first
    if is_geospatial {
        if let Some(folder) = scanned_folders.get(code) {
            // Find a geospatial file
            for file in &folder.files {
                let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                if ext == "tif" || ext == "tiff" || ext == "asc" {
                    let (geo_result, override_reason) = extract_geotiff_geometry(file, crs_overrides);
                    if let Some(geo) = geo_result {
                        return ExtractionResult::Extracted {
                            source: geo.source,
                            bbox: geo.bbox,
                            geometry: geo.geometry,
                            override_reason,
                        };
                    }
                } else if ext == "laz" || ext == "las" {
                    if let Some(geo) = extract_las_geometry(file) {
                        return ExtractionResult::Extracted {
                            source: format!("LAZ/LAS: {}", file.file_name().unwrap_or_default().to_string_lossy()),
                            bbox: geo.bbox,
                            geometry: geo.geometry,
                            override_reason: None,
                        };
                    }
                }
            }
        }
    }

    // Fall back to manual coordinates (LV95 already converted to WGS84 during loading)
    if let Some(sensor) = sensors.get(code) {
        if let (Some(lon), Some(lat)) = (sensor.lon, sensor.lat) {
            let (bbox, geometry) = point_to_geometry(lon, lat, sensor.elevation_m);
            return ExtractionResult::Manual {
                sensor_name: sensor.name.clone().unwrap_or_else(|| code.clone()),
                lon,
                lat,
                bbox,
                geometry,
            };
        }
        // Sensor found but no valid coordinates (x/y might be null in JSON)
        return ExtractionResult::NoGeometry {
            reason: format!("Sensor '{}' found but coordinates are null", sensor.name.as_deref().unwrap_or(code)),
        };
    }

    ExtractionResult::NoGeometry {
        reason: "No extractable geometry or manual coordinates".to_string(),
    }
}
