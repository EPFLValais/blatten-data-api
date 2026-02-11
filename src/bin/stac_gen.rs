//! STAC Catalog Generator
//!
//! Generates STAC 1.1.0 catalog from XLSX metadata file with geometry extraction.
//!
//! Usage:
//!     stac-gen generate --xlsx metadata.xlsx --output stac/
//!     stac-gen generate --xlsx metadata.xlsx --output stac/ --data-dir ./data/
//!     stac-gen validate --catalog-dir stac/

use anyhow::{Context, Result};
use calamine::{open_workbook, Data, Reader, Xlsx};
use chrono::{NaiveDate, Utc};
use clap::Parser;
use gdal::spatial_ref::{CoordTransform, SpatialRef};
use gdal::Dataset;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{debug, error, info, warn};
use walkdir::WalkDir;

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
// File Overrides / Patches
// =============================================================================
//
// Known issues with source files that require manual overrides.
// These are applied during geometry extraction and reported in the summary.
//
// Add entries here for files with:
// - Missing or incorrect CRS metadata
// - Known coordinate issues
// - Other metadata problems that need patching
//

/// Override configuration for a specific file
#[derive(Debug, Clone)]
struct FileOverride {
    /// Filename pattern to match (exact match on filename, not path)
    filename: &'static str,
    /// EPSG code to use for CRS (if file has missing/broken CRS)
    crs_epsg: Option<i32>,
    /// Description of why this override exists
    reason: &'static str,
}

/// Get the list of file overrides for known issues
fn get_file_overrides() -> Vec<FileOverride> {
    vec![
        // =====================================================================
        // Terradata DEM files with missing CRS metadata
        // =====================================================================
        FileOverride {
            filename: "DTM_250711_GRID_20cm_LV95_NF02.tif",
            crs_epsg: Some(2056), // Swiss LV95
            reason: "File has ENGCRS['unnamed'] instead of EPSG:2056. Coordinates are valid LV95.",
        },
        // Add more overrides here as needed:
        // FileOverride {
        //     filename: "some_other_file.tif",
        //     crs_epsg: Some(2056),
        //     reason: "Description of the issue",
        // },
    ]
}


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
  2. Hash assets     — parallel SHA-256/xxHash of every file (incremental)
  3. Stage archives  — create archives from assets (skips unchanged)
  4. Generate STAC   — catalog with file:checksum on every asset")]
#[command(after_long_help = "\
EXAMPLES:
  # Minimal: generate STAC catalog from Excel only
  stac-gen -x metadata.xlsx

  # With geometry extraction from FINAL_Data
  stac-gen -x metadata.xlsx -d /path/to/FINAL_Data -s data/sensor_locations.json

  # Full pipeline: stage, hash, archive, generate
  stac-gen -x metadata.xlsx -d /path/to/FINAL_Data \\
    --input-archives /path/to/provider-archives/ \\
    --organised-dir /path/to/staging \\
    -s data/sensor_locations.json \\
    --folder-mappings data/folder_mappings.json \\
    --validate -y

ENVIRONMENT:
  GTIFF_SRS_SOURCE=EPSG  Use EPFL GDAL SRS source for coordinate transforms
  RUST_LOG=debug          Enable debug logging")]
struct Cli {
    /// Input XLSX file
    #[arg(short, long)]
    xlsx: PathBuf,

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
    #[arg(short = 'd', long)]
    data: Option<PathBuf>,

    /// Archives directory containing .zip files
    #[arg(short = 'a', long)]
    archives_dir: Option<PathBuf>,

    /// Sensor locations JSON for manual coordinates (LV95).
    /// Coordinates are automatically converted from LV95 (EPSG:2056) to WGS84.
    #[arg(short = 's', long)]
    sensors_file: Option<PathBuf>,

    /// Folder mappings JSON — maps item codes to data folders with non-standard naming
    #[arg(long, alias = "dnage-mappings")]
    folder_mappings: Option<PathBuf>,

    /// Station locations CSV file (LV95 coordinates).
    #[arg(long)]
    station_locations: Option<PathBuf>,

    /// Number of parallel threads (for hashing, archiving, geometry extraction)
    #[arg(long, default_value_t = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))]
    threads: usize,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Run validation after generation (exits with code 1 if errors found)
    #[arg(long)]
    validate: bool,

    /// Output directory for organized data. Creates assets/ and archives/ subdirectories.
    #[arg(long)]
    organised_dir: Option<PathBuf>,

    /// Output directory for organized assets (copies files from FINAL_Data here).
    #[arg(long)]
    assets_dir: Option<PathBuf>,

    /// Force full rebuild (ignore existing manifest)
    #[arg(long)]
    full_rebuild: bool,

    /// Hash algorithm for file verification: sha256 (default) or xxhash
    #[arg(long, default_value = "sha256")]
    hash_algorithm: String,

    /// Skip file operations (metadata only)
    #[arg(long)]
    skip_copy: bool,

    /// Dry run - show what would be done without making changes
    #[arg(long)]
    dry_run: bool,

    /// Skip confirmation prompt (for automated/CI use)
    #[arg(short = 'y', long)]
    yes: bool,

    /// Link mode for staging files: symlink (default), hardlink, or copy
    #[arg(long, default_value = "symlink")]
    link_mode: String,

    /// Materialize staged links into real copies (run before rclone sync)
    #[arg(long)]
    materialize: bool,

    /// Input archives directory containing provider .zip/.7z files to incorporate
    #[arg(long)]
    input_archives: Option<PathBuf>,
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
    /// True if archive uses ZIP64 extensions (size > 4GB)
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
    files: HashMap<String, FileManifestEntry>,
    /// Archives created (code -> archive manifest entry)
    archives: HashMap<String, ArchiveManifestEntry>,
}

impl DataManifest {
    fn new(algorithm: HashAlgorithm) -> Self {
        Self {
            version: 1,
            generated_at: Utc::now().to_rfc3339(),
            hash_algorithm: algorithm,
            total_files: 0,
            total_size: 0,
            files: HashMap::new(),
            archives: HashMap::new(),
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
    id: &'static str,
    title: String,
    description: String,
}

/// Map product type codes to collection definitions
/// Map product type letter codes to collection IDs (URL-safe slugs).
/// Titles and descriptions are populated later from Excel data.
fn get_collection_id_map() -> HashMap<&'static str, &'static str> {
    [
        ("A", "webcam-image"),
        ("B", "deformation-analysis"),
        ("D", "orthophoto"),
        ("E", "radar-displacement"),
        ("F", "radar-amplitude"),
        ("G", "radar-coherence"),
        ("H", "radar-velocity"),
        ("I", "dsm"),
        ("J", "dem"),
        ("K", "point-cloud"),
        ("L", "3d-model"),
        ("M", "gnss-data"),
        ("N", "thermal-image"),
        ("O", "hydrology"),
        ("P", "lake-level"),
        ("U", "radar-timeseries"),
    ]
    .into_iter()
    .collect()
}

/// Build collection definitions from parsed items.
/// Derives titles from Excel ProductType values.
fn build_collections(items: &[ItemMetadata]) -> HashMap<String, CollectionDef> {
    let id_map = get_collection_id_map();
    let mut collections: HashMap<String, CollectionDef> = HashMap::new();

    // Collect product_type names per product_id letter
    for item in items {
        if let Some(ref pid) = item.product_id {
            if let Some(&coll_id) = id_map.get(pid.as_str()) {
                let entry = collections.entry(pid.clone()).or_insert_with(|| {
                    CollectionDef {
                        id: coll_id,
                        title: String::new(),
                        description: String::new(),
                    }
                });
                // Use the first non-empty product_type as the title
                if entry.title.is_empty() {
                    if let Some(ref pt) = item.product_type {
                        if !pt.is_empty() {
                            entry.title = pt.clone();
                        }
                    }
                }
            }
        }
    }

    // Ensure all mapped letters have entries (fallback for letters with no items)
    for (&letter, &coll_id) in &id_map {
        collections.entry(letter.to_string()).or_insert_with(|| {
            CollectionDef {
                id: coll_id,
                title: coll_id.replace('-', " "),
                description: String::new(),
            }
        });
    }

    collections
}

/// Item code to archive file mapping
fn get_code_to_file() -> HashMap<&'static str, &'static str> {
    [
        ("02Ah00", "02Ah00_FlexCam_Birchgletscher_BirchbachChannel_SAMPLE.zip"),
        ("04Ba00", "04Ba00_DEFOX_all.zip"),
        ("04Ba01", "04Ba01_DEFOX_2to3_per_d.zip"),
        ("04Ba02", "04Ba02_DEFOX_1_per_d.zip"),
        ("06Ha00", "06Ha00_Radar_Velocities_ROI.zip"),
        // 08Aa00 handled via folder_mappings.json (combines 08Aa01 + 08Aa02)
        ("08Aa01", "08Aa01_Webcam_Lonza_1h.zip"),
        ("08Aa02", "08Aa02_Webcam_Lonza_30min.7z"),
        ("10Ma00", "10M_11M_GNSS.zip"),
        ("11Ma00", "10M_11M_GNSS.zip"),
        ("11Mb00", "10M_11M_GNSS.zip"),
        ("11Mc00", "10M_11M_GNSS.zip"),
        ("11Md00", "10M_11M_GNSS.zip"),
        ("11Me00", "10M_11M_GNSS.zip"),
        ("11Mf00", "10M_11M_GNSS.zip"),
        ("11Mg00", "10M_11M_GNSS.zip"),
        ("11Mh00", "10M_11M_GNSS.zip"),
        ("13Db06", "13Db06.7z"),
        ("13Ib06", "13Ib06.zip"),
        ("14Ia04", "14Ia04.zip"),
        ("14Ka04", "14Ka04.zip"),
        ("14La02", "14La02.zip"),
        ("14Na01", "14Na01.zip"),
        ("15Da01", "15Da01.zip"),
        ("17Pa00", "17_LakeLevel_Geoazimut.zip"),
        ("17Pb00", "17_LakeLevel_Geoazimut.zip"),
    ]
    .into_iter()
    .collect()
}

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

/// Load sensor locations from JSON file and convert LV95 to WGS84
fn load_sensor_locations(path: &Path) -> Result<HashMap<String, SensorLocation>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }

    let content = fs::read_to_string(path)?;
    let data: serde_json::Value = serde_json::from_str(&content)?;

    let mut lookup = HashMap::new();
    if let Some(sensors) = data.get("sensors").and_then(|s| s.as_object()) {
        for (sensor_id, sensor_data) in sensors {
            let mut sensor: SensorLocation = serde_json::from_value(sensor_data.clone())?;

            // Convert LV95 to WGS84 if x,y are available but lon,lat are not
            if sensor.lon.is_none() && sensor.lat.is_none() {
                if let (Some(x), Some(y)) = (sensor.x, sensor.y) {
                    if let Some((lon, lat)) = lv95_to_wgs84(x, y) {
                        sensor.lon = Some(lon);
                        sensor.lat = Some(lat);
                    }
                }
            }

            for item_code in &sensor.items {
                lookup.insert(item_code.clone(), sensor.clone());
            }
            // Also store by sensor_id for direct lookup
            lookup.insert(sensor_id.clone(), sensor);
        }
    }

    Ok(lookup)
}

/// Load station locations from CSV file (LV95 coordinates)
/// CSV format: ID,Name,x,y (with header row)
/// Returns HashMap<sensor_id, (lon_wgs84, lat_wgs84, name)>
fn load_station_locations_csv(path: &Path) -> Result<HashMap<String, (f64, f64, String)>> {
    use std::io::BufRead;

    let mut stations = HashMap::new();

    if !path.exists() {
        warn!("Station locations CSV not found: {:?}", path);
        return Ok(stations);
    }

    let file = File::open(path)?;
    let reader = std::io::BufReader::new(file);

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();

        // Skip empty lines and header rows
        if line.is_empty() || line_num < 2 {
            continue;
        }

        // Parse CSV: ID,Name,x,y
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 4 {
            continue;
        }

        let sensor_id = parts[0].trim();
        let name = parts[1].trim();
        let x: f64 = match parts[2].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let y: f64 = match parts[3].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Convert LV95 to WGS84
        if let Some((lon, lat)) = lv95_to_wgs84(x, y) {
            // Pad sensor ID with leading zero if needed (e.g., "1" -> "01")
            let padded_id = if sensor_id.len() == 1 {
                format!("0{}", sensor_id)
            } else {
                sensor_id.to_string()
            };
            stations.insert(padded_id, (lon, lat, name.to_string()));
        }
    }

    Ok(stations)
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
fn extract_geotiff_geometry(path: &Path) -> (Option<ExtractedGeometry>, Option<String>) {
    let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
    let overrides = get_file_overrides();

    // Check if there's an override for this file
    let file_override = overrides.iter().find(|o| o.filename == filename);

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
            info!(
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
// XLSX Parsing
// =============================================================================

/// Extract string from cell
fn cell_string(cell: &Data) -> Option<String> {
    match cell {
        Data::String(s) => {
            let s = s.trim();
            if s.is_empty() || s == "NaN" {
                None
            } else {
                Some(s.to_string())
            }
        }
        Data::Float(f) => Some(f.to_string()),
        Data::Int(i) => Some(i.to_string()),
        Data::Bool(b) => Some(b.to_string()),
        Data::DateTime(d) => {
            // calamine ExcelDateTime - convert to f64 serial and then to ISO
            let serial = d.as_f64();
            if let Some(date) = excel_date_to_iso(serial) {
                Some(date)
            } else {
                None
            }
        }
        Data::DateTimeIso(s) => Some(s.clone()),
        Data::Error(_) | Data::Empty => None,
        _ => None,
    }
}

/// Extract int from cell
fn cell_int(cell: &Data) -> Option<i32> {
    match cell {
        Data::Int(i) => Some(*i as i32),
        Data::Float(f) => Some(*f as i32),
        Data::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// Extract bool from cell (checks for 'x', 'true', 'yes', '1')
fn cell_bool(cell: &Data) -> bool {
    match cell {
        Data::Bool(b) => *b,
        Data::String(s) => {
            let s = s.trim().to_lowercase();
            s == "x" || s == "true" || s == "yes" || s == "1"
        }
        Data::Int(i) => *i != 0,
        Data::Float(f) => *f != 0.0,
        _ => false,
    }
}

/// Convert Excel date serial number to ISO date string
fn excel_date_to_iso(serial: f64) -> Option<String> {
    // Excel date: days since 1899-12-30
    let days = serial as i64;
    let base = NaiveDate::from_ymd_opt(1899, 12, 30)?;
    let date = base.checked_add_signed(chrono::Duration::days(days))?;
    Some(date.format("%Y-%m-%d").to_string())
}

/// Parse the Data sheet from XLSX (prefers Data > Test_Data > All_Data > first sheet)
fn parse_xlsx(path: &PathBuf) -> Result<Vec<ItemMetadata>> {
    let mut workbook: Xlsx<_> = open_workbook(path)
        .with_context(|| format!("Failed to open XLSX: {:?}", path))?;

    // Get sheet names and find preferred sheet (like Python: Data > Test_Data > All_Data > first)
    let sheet_names = workbook.sheet_names().to_vec();
    let preferred_sheets = ["Data", "Test_Data", "All_Data"];
    let sheet_name = preferred_sheets
        .iter()
        .find(|&name| sheet_names.contains(&name.to_string()))
        .map(|s| s.to_string())
        .unwrap_or_else(|| sheet_names.first().cloned().unwrap_or_default());

    info!("Using sheet: {} (available: {:?})", sheet_name, sheet_names);

    let range = workbook
        .worksheet_range(&sheet_name)
        .with_context(|| format!("Sheet '{}' not found", sheet_name))?;

    let collection_ids = get_collection_id_map();
    let code_to_file = get_code_to_file();
    let mut items = Vec::new();

    // Data starts at row 2 (1-indexed Excel), which is row 1 (0-indexed)
    // Column layout (0-indexed) based on actual Excel structure:
    // 0: Code, 1: ProductID, 2: SensorID, 3: DatasetID, 4: BundleID, 5: Sensor,
    // 6: ProductType, 7: Dataset, 8: Bundle, 9: Description, 10: Format,
    // 11: AdditionalInfo, 12: ProcessingLevel, 13: Phase, 14: DateFirst, 15: DateLast,
    // 16: Frequency, 17: Source, 18: OnSharePoint, 19: InternalCommentary
    for (row_idx, row) in range.rows().enumerate() {
        if row_idx < 1 {
            continue; // Skip header row
        }

        // Column 0 is the code
        let code = match row.get(0).and_then(cell_string) {
            Some(c) if !c.is_empty() => c,
            _ => continue,
        };

        let product_id = row.get(1).and_then(cell_string);

        let mut item = ItemMetadata {
            code: code.clone(),
            product_id: product_id.clone(),
            sensor_id: row.get(2).and_then(cell_string),
            dataset_id: row.get(3).and_then(cell_string),
            bundle_id: row.get(4).and_then(cell_string),
            sensor: row.get(5).and_then(cell_string).map(|s| strip_field_prefix(&s)),
            product_type: row.get(6).and_then(cell_string).map(|s| strip_field_prefix(&s)),
            dataset: row.get(7).and_then(cell_string).map(|s| strip_field_prefix(&s)),
            bundle: row.get(8).and_then(cell_string).map(|s| strip_field_prefix(&s)),
            description: row.get(9).and_then(cell_string),
            format: row.get(10).and_then(cell_string),
            technical_info: row.get(11).and_then(cell_string),
            processing_level: row.get(12).and_then(cell_int),
            phase: row.get(13).and_then(cell_string),
            date_first: row.get(14).and_then(cell_string),
            date_last: row.get(15).and_then(cell_string),
            continued: row.get(18).map(cell_bool).unwrap_or(false),  // OnSharePoint column
            frequency: row.get(16).and_then(cell_string),
            location: None,  // Not in current Excel layout
            source: row.get(17).and_then(cell_string),
            additional_remarks: None,  // Not in current Excel layout
            storage_mb: None,  // Not in current Excel layout
            internal_commentary: row.get(19).and_then(cell_string),
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

        // Map product_id to collection
        if let Some(ref pid) = product_id {
            if let Some(&coll_id) = collection_ids.get(pid.as_str()) {
                item.collection_id = Some(coll_id.to_string());
            }
        }

        // Map code to archive file
        if let Some(file) = code_to_file.get(code.as_str()) {
            item.archive_file = Some(file.to_string());
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
fn create_stac_item(item: &ItemMetadata, base_url: &str, s3_base_url: &str) -> serde_json::Value {
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

    // Build title: "{code} - {product_type} - {dataset} ({bundle})"
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
        item.product_type.as_deref().unwrap_or(""),
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
) -> serde_json::Value {
    // Compute spatial extent
    // Default bbox covers the Blatten/Birch Glacier area (WGS84)
    const DEFAULT_BBOX: [f64; 4] = [7.78, 46.38, 7.88, 46.44];

    let bboxes: Vec<&Vec<f64>> = items.iter().filter_map(|i| i.bbox.as_ref()).collect();
    let spatial_bbox: serde_json::Value = if !bboxes.is_empty() {
        serde_json::json!([[
            bboxes.iter().map(|b| b[0]).fold(f64::INFINITY, f64::min),
            bboxes.iter().map(|b| b[1]).fold(f64::INFINITY, f64::min),
            bboxes.iter().map(|b| b[2]).fold(f64::NEG_INFINITY, f64::max),
            bboxes.iter().map(|b| b[3]).fold(f64::NEG_INFINITY, f64::max),
        ]])
    } else {
        // Use default Blatten area bbox when no geometry available
        serde_json::json!([[DEFAULT_BBOX[0], DEFAULT_BBOX[1], DEFAULT_BBOX[2], DEFAULT_BBOX[3]]])
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
    let processing_levels: Vec<i32> = items
        .iter()
        .filter_map(|i| i.processing_level)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let sources: Vec<String> = items
        .iter()
        .filter_map(|i| i.source.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    serde_json::json!({
        "type": "Collection",
        "id": def.id,
        "stac_version": STAC_VERSION,
        "stac_extensions": [
            "https://stac-extensions.github.io/timestamps/v1.1.0/schema.json"
        ],
        "title": def.title,
        "description": def.description,
        "license": "CC-BY-NC-SA-4.0",
        "keywords": [
            "Birch Glacier",
            "Blatten",
            "Switzerland",
            "glacier collapse",
            "landslide",
            def.title.to_lowercase()
        ],
        "providers": [
            {
                "name": "Canton du Valais",
                "roles": ["licensor", "host"],
                "url": "https://www.vs.ch/"
            },
            {
                "name": "EPFL ALPOLE",
                "roles": ["processor", "host"],
                "url": "https://www.epfl.ch/research/domains/alpole/"
            }
        ],
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
) -> Result<(usize, usize, usize, Vec<ValidationIssue>)> {
    fs::create_dir_all(output_dir)?;
    fs::create_dir_all(output_dir.join("collections"))?;

    let collections_defs = build_collections(items);
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

    // Pre-create all STAC items in parallel (needed for both per-collection and all_items files)
    let results: Vec<_> = items
        .par_iter()
        .map(|item| {
            let stac_item = create_stac_item(item, base_url, s3_base_url);

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
            // Find collection definition
            let def = collections_defs
                .values()
                .find(|d| d.id == coll_id)
                .expect("Unknown collection");

            // Get STAC items for this collection
            let coll_stac_items: Vec<&serde_json::Value> = items_with_stac
                .iter()
                .filter(|(meta, _)| meta.collection_id.as_deref() == Some(coll_id.as_str()))
                .map(|(_, stac)| stac)
                .collect();

            // Create collection JSON
            let collection = create_stac_collection(def, coll_items, base_url);
            let collection_json = serde_json::to_string_pretty(&collection).unwrap();

            // Build NDJSON content in memory
            let items_ndjson: String = coll_stac_items
                .iter()
                .map(|item| serde_json::to_string(item).unwrap())
                .collect::<Vec<_>>()
                .join("\n");

            coll_pb.inc(1);
            (coll_id.clone(), collection, collection_json, items_ndjson)
        })
        .collect();

    // Extract collections for later use
    let stac_collections: Vec<_> = collection_data.iter().map(|(_, c, _, _)| c.clone()).collect();

    // Write all collection files (fast, already serialized)
    for (coll_id, _, collection_json, items_ndjson) in &collection_data {
        let coll_path = output_dir.join("collections").join(format!("{}.json", coll_id));
        fs::write(&coll_path, collection_json)?;

        let items_path = output_dir.join("collections").join(format!("{}_items.ndjson", coll_id));
        fs::write(&items_path, items_ndjson)?;
    }

    coll_pb.finish_with_message(format!("Wrote {} collection files", stac_collections.len()));

    // Collect all STAC items for the all_items file
    let all_items: Vec<serde_json::Value> = items_with_stac.into_iter().map(|(_, stac)| stac).collect();

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

    // Add describedby, license, and about links
    catalog_links.push(serde_json::json!({
        "rel": "describedby",
        "href": format!("{}{}/docs/dataset_overview.csv", base_url, s3_base_url),
        "type": "text/csv",
        "title": "Dataset Overview"
    }));
    catalog_links.push(serde_json::json!({
        "rel": "describedby",
        "href": format!("{}{}/docs/detailed_report.pdf", base_url, s3_base_url),
        "type": "application/pdf",
        "title": "Detailed Report"
    }));
    catalog_links.push(serde_json::json!({
        "rel": "license",
        "href": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "title": "Attribution-NonCommercial-ShareAlike 4.0 International"
    }));
    catalog_links.push(serde_json::json!({
        "rel": "about",
        "href": "https://blatten-data.epfl.ch/",
        "type": "text/html",
        "title": "Blatten Data website"
    }));

    // Write root catalog
    let catalog = serde_json::json!({
        "type": "Catalog",
        "id": "birch-glacier-collapse",
        "stac_version": STAC_VERSION,
        "title": "Birch Glacier Collapse and Landslide Dataset",
        "description": "Dataset collected during the 2025 Birch glacier collapse and landslide at Blatten, CH-VS. Licensed under Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). By using this API you confirm that you have read the Dataset Overview and Detailed Report. This data is provided \"as is\" without warranty of any kind, express or implied.",
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
    let mut collection_stats: HashMap<String, serde_json::Value> = HashMap::new();
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
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    // Destructure CLI for convenience
    let Cli {
        xlsx, output, base_url, s3_base_url, data, archives_dir, sensors_file,
        folder_mappings, station_locations, threads, verbose, validate,
        organised_dir, assets_dir, full_rebuild, hash_algorithm, skip_copy,
        dry_run, yes: _, link_mode, materialize, input_archives,
    } = cli;

    {
            // Resolve assets_dir and archives_dir from organised_dir if not explicitly set
            let assets_dir = assets_dir.or_else(|| organised_dir.as_ref().map(|d| d.join("assets")));
            let archives_dir = archives_dir.or_else(|| organised_dir.as_ref().map(|d| d.join("archives")));

            // Parse link mode
            let link_mode_parsed: LinkMode = link_mode.parse().unwrap_or(LinkMode::Symlink);

            info!("=== STAC Generator (v{}) ===", STAC_VERSION);

            // Configure rayon thread pool
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .ok();

            // Set up progress bars
            let multi_progress = MultiProgress::new();

            // Step 1: Parse XLSX (quick operation, use simple info log instead of spinner)
            let mut items = parse_xlsx(&xlsx)?;
            info!("  Parsed {} items from XLSX", items.len());

            // Step 2: Scan folders if data provided (quick operation, use simple info log)
            let scanned_folders: HashMap<String, ScannedFolder> = if let Some(ref data_path) = data {
                let folders = scan_data_folders(data_path, folder_mappings.as_deref())?;
                info!("  Found {} data folders", folders.len());
                folders
            } else {
                HashMap::new()
            };

            // Handle --materialize: convert links to real copies and exit
            if materialize {
                if let Some(ref assets_path) = assets_dir {
                    info!("=== Materializing links ===");
                    let (mat_assets, real_assets) = materialize_links(assets_path, dry_run)?;
                    info!("  Assets: {} materialized, {} already real", mat_assets, real_assets);
                    if let Some(ref arch_path) = archives_dir {
                        let (mat_arch, real_arch) = materialize_links(arch_path, dry_run)?;
                        info!("  Archives: {} materialized, {} already real", mat_arch, real_arch);
                    }
                    info!("Done.");
                    return Ok(());
                } else {
                    error!("--materialize requires --assets-dir or --organised-dir");
                    std::process::exit(1);
                }
            }

            // Collect xlsx codes for staging/validation
            let xlsx_codes: Vec<String> = items.iter().map(|i| i.code.clone()).collect();

            // Parse hash algorithm early (needed by staging pipeline)
            let algorithm: HashAlgorithm = hash_algorithm.parse().unwrap_or(HashAlgorithm::Sha256);

            // Data manifest — populated by staging pipeline if --assets-dir is set
            #[allow(unused_assignments)]
            let mut data_manifest: Option<DataManifest> = None;

            // Stage assets if --assets-dir is set and data is provided
            if let Some(ref assets_path) = assets_dir {
                if data.is_some() || input_archives.is_some() {
                    info!("");
                    info!("=== Staging Pipeline (link mode: {}) ===", link_mode_parsed);
                    info!("  Hash algorithm: {}", algorithm);
                    if dry_run {
                        info!("  Mode: DRY RUN");
                    }

                    // Load existing manifest for incremental hashing
                    let existing_manifest = if full_rebuild {
                        info!("  Mode: FULL REBUILD (ignoring existing manifest)");
                        None
                    } else {
                        let manifest_path = output.join("manifest.json");
                        if manifest_path.exists() {
                            match DataManifest::load(&manifest_path) {
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

                    // Load folder mappings for archive filename matching
                    let folder_map: HashMap<String, serde_json::Value> = if let Some(ref path) = folder_mappings {
                        let content = fs::read_to_string(path).unwrap_or_default();
                        let data: serde_json::Value = serde_json::from_str(&content).unwrap_or_default();
                        data.get("mappings")
                            .and_then(|m| m.as_object())
                            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                            .unwrap_or_default()
                    } else {
                        HashMap::new()
                    };

                    // Step 1: Stage assets (symlinks from FINAL_Data + extract input archives)
                    let pb_style = ProgressStyle::default_bar()
                        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                        .expect("Invalid bar template")
                        .progress_chars("#>-");

                    let stage_assets_pb = multi_progress.add(ProgressBar::new(0));
                    stage_assets_pb.set_style(pb_style.clone());

                    if !skip_copy {
                        stage_assets(
                            &scanned_folders,
                            &xlsx_codes,
                            assets_path,
                            link_mode_parsed,
                            input_archives.as_deref(),
                            &folder_map,
                            dry_run,
                            &stage_assets_pb,
                        )?;
                    }

                    // Step 2: Hash all staged assets in parallel
                    let hash_pb = multi_progress.add(ProgressBar::new(0));
                    hash_pb.set_style(pb_style.clone());

                    let mut manifest = if dry_run {
                        let m = DataManifest::new(algorithm);
                        hash_pb.finish_with_message("Hashing: skipped (dry run)");
                        m
                    } else {
                        hash_staged_assets(
                            assets_path,
                            algorithm,
                            existing_manifest.as_ref(),
                            &scanned_folders,
                            &hash_pb,
                        )?
                    };

                    // Step 3: Stage archives (always real copies, with fingerprinting)
                    if let Some(ref arch_path) = archives_dir {
                        let stage_arch_pb = multi_progress.add(ProgressBar::new(0));
                        stage_arch_pb.set_style(pb_style.clone());

                        stage_archives(
                            &scanned_folders,
                            &xlsx_codes,
                            assets_path,
                            arch_path,
                            &mut manifest,
                            existing_manifest.as_ref(),
                            dry_run,
                            &stage_arch_pb,
                        )?;
                    }

                    // Save manifest
                    if !dry_run {
                        fs::create_dir_all(&output)?;
                        let manifest_path = output.join("manifest.json");
                        manifest.save(&manifest_path)?;
                        info!("  Saved manifest to {:?}", manifest_path);
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

            // Step 3: Extract geometry
            let data_path = data.as_deref();
            let sensors_for_fallback = if data_path.is_some() || sensors_file.is_some() {
                let extract_pb = multi_progress.add(ProgressBar::new(items.len() as u64));
                extract_pb.set_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                        .expect("Invalid bar template")
                        .progress_chars("#>-"),
                );
                extract_pb.set_message("Extracting geometry...");

                // Load sensor locations
                let sensors = if let Some(ref sensors_path) = sensors_file {
                    load_sensor_locations(sensors_path)?
                } else if let Some(ref dp) = data_path {
                    let default_path = dp.join("sensor_locations.json");
                    if default_path.exists() {
                        load_sensor_locations(&default_path)?
                    } else {
                        HashMap::new()
                    }
                } else {
                    HashMap::new()
                };

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
                                if verbose {
                                    if let Some(ref reason) = override_reason {
                                        debug!("[{}] Extracted from {} [OVERRIDE: {}]", code, source, reason);
                                    } else {
                                        debug!("[{}] Extracted from {}", code, source);
                                    }
                                }
                            }
                            ExtractionResult::Manual { bbox, geometry, sensor_name, lon, lat } => {
                                item.bbox = Some(bbox);
                                item.geometry = Some(geometry);
                                manual += 1;
                                if verbose {
                                    debug!("[{}] Manual: {} at [{:.4}, {:.4}]", code, sensor_name, lon, lat);
                                }
                            }
                            ExtractionResult::NoGeometry { reason } => {
                                no_geometry += 1;
                                if verbose {
                                    debug!("[{}] No geometry: {}", code, reason);
                                }
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

            // Extract geometry for all files in parallel
            let processed_count = std::sync::atomic::AtomicU64::new(0);
            let file_results: Vec<(String, FileInfo)> = all_files
                .par_iter()
                .map(|(code, file_path)| {
                    let size = file_path.metadata().map(|m| m.len()).unwrap_or(0);
                    let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

                    // Extract geometry for geospatial files (both WGS84 and LV95)
                    let (bbox, geometry, bbox_lv95, geometry_lv95) = if ext == "tif" || ext == "tiff" || ext == "asc" {
                        let (geo, _override_reason) = extract_geotiff_geometry(file_path);
                        geo.map(|g| (Some(g.bbox), Some(g.geometry), g.bbox_lv95, g.geometry_lv95))
                            .unwrap_or((None, None, None, None))
                    } else if ext == "laz" || ext == "las" {
                        extract_las_geometry(file_path)
                            .map(|g| (Some(g.bbox), Some(g.geometry), g.bbox_lv95, g.geometry_lv95))
                            .unwrap_or((None, None, None, None))
                    } else {
                        (None, None, None, None)
                    };

                    // Update progress every 100 files to reduce contention
                    let count = processed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if count % 100 == 0 {
                        file_pb.set_position(count);
                    }

                    (code.clone(), FileInfo {
                        path: file_path.clone(),
                        size,
                        bbox,
                        geometry,
                        bbox_lv95,
                        geometry_lv95,
                        hash: None,
                    })
                })
                .collect();

            file_pb.set_position(file_results.len() as u64);
            file_pb.finish_with_message(format!("Extracted geometry from {} files", file_results.len()));

            // Group results by item code and apply
            let mut files_by_code: HashMap<String, Vec<FileInfo>> = HashMap::new();
            for (code, file_info) in file_results {
                files_by_code.entry(code).or_default().push(file_info);
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
                                if verbose {
                                    debug!("[{}] Sensor fallback: {} at [{:.4}, {:.4}]",
                                        item.code,
                                        sensor.name.as_deref().unwrap_or(&item.code),
                                        lon, lat);
                                }
                            }
                        }
                    }
                }
                if sensor_fallback_count > 0 {
                    info!("  Applied {} sensor locations (fallback for items with non-geospatial files)", sensor_fallback_count);
                }
            }

            // Step 3b: Apply station coordinates from CSV to items still missing geometry
            if let Some(ref csv_path) = station_locations {
                let stations = load_station_locations_csv(csv_path)?;
                if !stations.is_empty() {
                    info!("  Loaded {} station locations from CSV", stations.len());

                    let mut station_applied = 0;
                    for item in &mut items {
                        if item.geometry.is_none() {
                            // Extract sensor ID from item code (first 2 digits)
                            let sensor_id = &item.code[..2.min(item.code.len())];
                            if let Some((lon, lat, name)) = stations.get(sensor_id) {
                                let (bbox, geometry) = point_to_geometry(*lon, *lat, None);
                                item.bbox = Some(bbox);
                                item.geometry = Some(geometry);
                                station_applied += 1;
                                if verbose {
                                    debug!("[{}] Station coords from CSV: {} at [{:.4}, {:.4}]", item.code, name, lon, lat);
                                }
                            }
                        }
                    }

                    if station_applied > 0 {
                        info!("  Applied {} station locations from CSV", station_applied);
                    }
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

                // Map archives to items
                let mut found = 0;
                for item in &mut items {
                    // First check hardcoded mapping, then auto-discovered
                    if item.archive_file.is_none() {
                        if let Some((path, size)) = archive_map.get(&item.code) {
                            item.archive_file = Some(path.file_name().unwrap_or_default().to_string_lossy().to_string());
                            item.archive_size = Some(*size);
                            item.archive_is_zip64 = *size > 0xFFFFFFFF;
                            found += 1;
                        }
                    } else if let Some(ref archive_name) = item.archive_file {
                        // Check if hardcoded archive exists
                        let archive_path = arch_dir.join(archive_name);
                        if archive_path.exists() {
                            if let Ok(metadata) = fs::metadata(&archive_path) {
                                item.archive_size = Some(metadata.len());
                                item.archive_is_zip64 = metadata.len() > 0xFFFFFFFF;
                                found += 1;
                            }
                        } else if verbose {
                            warn!("[{}] Archive not found: {}", item.code, archive_name);
                        }
                    }
                }

                info!("  Archives/files: {} mapped from directory", found);
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
            let (num_collections, num_items, num_assets, issues) = generate_catalog(&items, &output, &base_url, &s3_base_url, &multi_progress)?;
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

/// Extract a zip archive to a target directory
fn extract_archive(archive_path: &Path, target_dir: &Path) -> Result<()> {
    let ext = archive_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

    fs::create_dir_all(target_dir)?;

    if ext == "7z" {
        // Use 7z command-line tool for .7z files
        let status = std::process::Command::new("7z")
            .args(["x", "-y", &format!("-o{}", target_dir.display())])
            .arg(archive_path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .status()
            .with_context(|| "Failed to run 7z. Is p7zip-full installed?")?;
        if !status.success() {
            anyhow::bail!("7z extraction failed for {:?} (exit code {:?})", archive_path, status.code());
        }
    } else {
        // .zip extraction
        let file = File::open(archive_path)?;
        let mut archive = zip::ZipArchive::new(file)?;
        let total = archive.len();

        for i in 0..total {
            let mut file = archive.by_index(i)?;
            let outpath = match file.enclosed_name() {
                Some(path) => target_dir.join(path),
                None => continue,
            };

            if file.name().ends_with('/') {
                fs::create_dir_all(&outpath)?;
            } else {
                if let Some(p) = outpath.parent() {
                    if !p.exists() {
                        fs::create_dir_all(p)?;
                    }
                }
                let mut outfile = File::create(&outpath)?;
                std::io::copy(&mut file, &mut outfile)?;
            }

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if let Some(mode) = file.unix_mode() {
                    fs::set_permissions(&outpath, fs::Permissions::from_mode(mode))?;
                }
            }

            if i % 500 == 0 && i > 0 {
                info!("    ... {}/{} files extracted", i, total);
            }
        }
    }

    // Count what was extracted
    let (file_count, total_size) = count_folder_contents(target_dir);
    info!("    Extracted {} files ({})", file_count, format_size(total_size));

    Ok(())
}

/// Create a zip archive from a directory
fn create_archive(source_dir: &Path, archive_path: &Path) -> Result<u64> {
    use zip::write::SimpleFileOptions;

    let file = File::create(archive_path)?;
    let mut zip = zip::ZipWriter::new(file);
    let options = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .large_file(true);

    // Walk the directory and add all files (follow symlinks for staged assets)
    for entry in WalkDir::new(source_dir).follow_links(true).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        let name = path.strip_prefix(source_dir)
            .map_err(|e| anyhow::anyhow!("Failed to strip prefix: {}", e))?;

        if path.is_file() {
            zip.start_file(name.to_string_lossy(), options)?;
            let mut f = File::open(path)?;
            std::io::copy(&mut f, &mut zip)?;
        } else if !name.as_os_str().is_empty() {
            // Directory entry (not the root)
            zip.add_directory(name.to_string_lossy(), options)?;
        }
    }

    zip.finish()?;

    // Return the size of the created archive
    let metadata = fs::metadata(archive_path)?;
    Ok(metadata.len())
}

/// Count files and total size in a directory
fn count_folder_contents(dir: &Path) -> (usize, u64) {
    let mut count = 0;
    let mut size = 0u64;

    for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            count += 1;
            size += entry.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }

    (count, size)
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
}

/// Stage assets from scanned FINAL_Data folders into `assets/<code>/` using the chosen link mode.
/// For codes missing from scanned folders but present in `input_archives`, extracts the archive.
fn stage_assets(
    scanned: &HashMap<String, ScannedFolder>,
    xlsx_codes: &[String],
    assets_dir: &Path,
    link_mode: LinkMode,
    input_archives: Option<&Path>,
    folder_mappings: &HashMap<String, serde_json::Value>,
    dry_run: bool,
    progress: &ProgressBar,
) -> Result<StageAssetsStats> {
    let mut stats = StageAssetsStats { staged: 0, extracted: 0, skipped: 0 };

    // Count total work
    let total_files: usize = scanned.values().map(|f| f.files.len()).sum();
    progress.set_length(total_files as u64);
    progress.set_message("Staging assets...");

    // Stage from scanned folders — always recreate (symlinks are instant)
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

            if dry_run {
                stats.staged += 1;
            } else {
                // Remove existing target (stale symlink, old copy, etc.)
                if target.exists() || target.symlink_metadata().is_ok() {
                    let _ = fs::remove_file(&target);
                }
                link_or_copy(source, &target, link_mode)?;
                stats.staged += 1;
            }
            progress.inc(1);
        }
    }

    // Extract archives for codes in xlsx but missing from scanned folders
    if let Some(archives_input) = input_archives {
        if !archives_input.exists() {
            warn!("  Input archives path does not exist: {}", archives_input.display());
        }
        let scanned_codes: std::collections::HashSet<&String> = scanned.keys().collect();
        let missing_codes: Vec<&String> = xlsx_codes.iter()
            .filter(|c| !scanned_codes.contains(c))
            .collect();

        if !missing_codes.is_empty() {
            info!("  Checking {} missing codes against input archives...", missing_codes.len());

            // Build a map of code -> archive path from input_archives dir
            let mut input_archive_map: HashMap<String, PathBuf> = HashMap::new();
            for entry in WalkDir::new(archives_input)
                .min_depth(1)
                .max_depth(1)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if !entry.file_type().is_file() { continue; }
                let ext = entry.path().extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                if ext != "zip" && ext != "7z" { continue; }
                let fname = entry.file_name().to_string_lossy().to_string();
                if let Some(code) = extract_code_from_folder(&fname, folder_mappings) {
                    input_archive_map.insert(code, entry.path().to_path_buf());
                }
            }

            // Log input archives and their resolved codes
            info!("  Input archives ({}):", input_archive_map.len());
            for (code, path) in &input_archive_map {
                let fname = path.file_name().unwrap_or_default().to_string_lossy();
                let status = if missing_codes.iter().any(|c| c.as_str() == code.as_str()) {
                    "MISSING — will extract"
                } else {
                    "has FINAL_Data — skip"
                };
                info!("    {} -> {} ({})", fname, code, status);
            }

            // Log missing codes without archives
            let unmatched_missing: Vec<&&String> = missing_codes.iter()
                .filter(|c| !input_archive_map.contains_key(c.as_str()))
                .collect();
            if !unmatched_missing.is_empty() {
                info!("  Missing codes without input archives ({}):", unmatched_missing.len());
                info!("    {}", unmatched_missing.iter().map(|c| c.as_str()).collect::<Vec<_>>().join(", "));
            }

            for code in missing_codes {
                let code_dir = assets_dir.join(code);
                // For extracted archives, skip if already extracted (extraction is expensive)
                if code_dir.exists() {
                    info!("    {} — skipped (already extracted)", code);
                    stats.skipped += 1;
                    continue;
                }
                if let Some(archive_path) = input_archive_map.get(code.as_str()) {
                    if dry_run {
                        info!("  WOULD EXTRACT: {} -> assets/{}/", archive_path.display(), code);
                        stats.extracted += 1;
                    } else {
                        info!("  Extracting: {} -> assets/{}/", archive_path.display(), code);
                        extract_archive(archive_path, &code_dir)?;
                        stats.extracted += 1;
                    }
                }
            }
        }
    }

    progress.finish_with_message(format!(
        "Assets: {} staged, {} extracted, {} skipped",
        stats.staged, stats.extracted, stats.skipped
    ));
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
            if !entry.file_type().is_file() {
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
                if !file_entry.file_type().is_file() {
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
    /// Dry run placeholder
    DryRun,
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
    xlsx_codes: &[String],
    assets_dir: &Path,
    archives_dir: &Path,
    manifest: &mut DataManifest,
    existing_manifest: Option<&DataManifest>,
    dry_run: bool,
    progress: &ProgressBar,
) -> Result<StageArchivesStats> {
    if !dry_run {
        fs::create_dir_all(archives_dir)?;
    }

    // Collect all codes that need archives
    let all_codes: Vec<String> = {
        let set: std::collections::HashSet<String> = xlsx_codes.iter().cloned()
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
            match action {
                ArchiveAction::Unchanged => {
                    let existing_entry = existing_manifest.unwrap().archives.get(code).unwrap();
                    progress.inc(1);
                    ArchiveResult::Unchanged { code: code.clone(), entry: existing_entry.clone() }
                }
                _ => {
                    if dry_run {
                        progress.inc(1);
                        return ArchiveResult::DryRun;
                    }
                    match create_archive(&asset_dir, archive_path) {
                        Ok(size) => {
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
                                            is_zip64: size > 0xFFFFFFFF,
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
            ArchiveResult::DryRun => {
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
fn materialize_links(dir: &Path, dry_run: bool) -> Result<(usize, usize)> {
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
            if dry_run {
                info!("  WOULD MATERIALIZE (symlink): {} -> {}", path.display(), real_path.display());
            } else {
                fs::remove_file(path)?;
                fs::copy(&real_path, path)?;
            }
            materialized += 1;
        } else if symlink_meta.nlink() > 1 {
            // Hardlink — copy in place via tmp
            if dry_run {
                info!("  WOULD MATERIALIZE (hardlink): {}", path.display());
            } else {
                let tmp = path.with_extension("__materialize_tmp");
                fs::copy(path, &tmp)?;
                fs::remove_file(path)?;
                fs::rename(&tmp, path)?;
            }
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
    folder_mappings_path: Option<&Path>,
) -> Result<HashMap<String, ScannedFolder>> {
    let mut folders: HashMap<String, ScannedFolder> = HashMap::new();

    // Load folder mappings if provided
    let folder_mappings: HashMap<String, serde_json::Value> = if let Some(path) = folder_mappings_path {
        let content = fs::read_to_string(path)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;
        data.get("mappings")
            .and_then(|m| m.as_object())
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default()
    } else {
        HashMap::new()
    };

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
                let parent_code = extract_code_from_folder(&folder_name, &folder_mappings);

                if let Some(parent_code) = parent_code {
                    // Scan for nested code folders and collect files
                    let nested_results = scan_folder_with_nested(entry.path(), &parent_code, provider);

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
            } else if entry.file_type().is_file() {
                // Single file (e.g., standalone .tif in Terradata/)
                let file_name = entry.file_name().to_string_lossy().to_string();
                if let Some(code) = extract_code_from_folder(&file_name, &folder_mappings) {
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
    for (code, mapping) in &folder_mappings {
        if folders.contains_key(code) {
            continue;
        }

        // Support both "folder" (single) and "folders" (array) for combined datasets
        let folder_rels: Vec<String> = if let Some(arr) = mapping.get("folders").and_then(|f| f.as_array()) {
            arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()
        } else if let Some(folder_rel) = mapping.get("folder").and_then(|f| f.as_str()) {
            vec![folder_rel.to_string()]
        } else {
            continue;
        };

        let mut all_files: Vec<PathBuf> = Vec::new();
        let mut first_path: Option<PathBuf> = None;
        for folder_rel in &folder_rels {
            let folder_path = data.join(folder_rel);
            if folder_path.exists() {
                if first_path.is_none() {
                    first_path = Some(folder_path.clone());
                }
                let files: Vec<PathBuf> = WalkDir::new(&folder_path)
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_type().is_file())
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
        if path.is_file() {
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
fn extract_code_from_folder(folder_name: &str, folder_mappings: &HashMap<String, serde_json::Value>) -> Option<String> {
    // Pattern: code at start followed by underscore (e.g., "13Da01_...")
    let code_re = regex::Regex::new(r"^(\d{2}[A-Z][a-z]\d{2})").ok()?;

    if let Some(caps) = code_re.captures(folder_name) {
        return Some(caps.get(1)?.as_str().to_string());
    }

    // Check folder mappings for non-standard folder names
    for (code, mapping) in folder_mappings {
        if let Some(folder) = mapping.get("folder").and_then(|f| f.as_str()) {
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
                    let (geo_result, override_reason) = extract_geotiff_geometry(file);
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
