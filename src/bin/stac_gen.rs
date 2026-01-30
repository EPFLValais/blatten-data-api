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
use clap::{Parser, Subcommand};
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
#[command(about = "Generate STAC catalog from XLSX metadata")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate STAC catalog from XLSX file
    Generate {
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

        /// Save intermediate metadata JSON
        #[arg(long)]
        save_metadata: Option<PathBuf>,

        /// Data directory containing provider folders (DNAGE/, Geopraevent/, Terradata/).
        /// This is the FINAL_Data directory with geospatial files for geometry extraction.
        #[arg(short = 'd', long)]
        data: Option<PathBuf>,

        /// Archives directory containing .zip files
        #[arg(short = 'a', long)]
        archives_dir: Option<PathBuf>,

        /// Sensor locations JSON for manual coordinates (LV95).
        /// Required for items without extractable geometry (webcams, GNSS, hydrology, radar).
        /// Coordinates are automatically converted from LV95 (EPSG:2056) to WGS84.
        /// Default: data/sensor_locations.json
        #[arg(short = 's', long)]
        sensors_file: Option<PathBuf>,

        /// DNAGE folder mappings JSON
        #[arg(long)]
        dnage_mappings: Option<PathBuf>,

        /// Validate that files exist on S3 (requires S3 access)
        #[arg(long)]
        validate_s3: bool,

        /// Number of parallel threads for geometry extraction
        #[arg(long, default_value = "4")]
        threads: usize,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Run validation after generation (exits with code 1 if errors found)
        #[arg(long)]
        validate: bool,
    },
    /// Validate an existing STAC catalog (for CI/CD pipelines)
    Validate {
        /// Catalog directory
        #[arg(short, long, default_value = "stac")]
        catalog_dir: PathBuf,
    },

    /// Extract provider archives to code-named folders in FINAL_Data structure
    Extract {
        /// Input directory containing provider .zip/.7z archives
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory (FINAL_Data structure: output/<provider>/<code>/)
        #[arg(short, long)]
        output: PathBuf,

        /// Provider name (Terradata, Geopraevent, DNAGE)
        #[arg(short, long)]
        provider: String,

        /// Overwrite existing folders
        #[arg(long)]
        overwrite: bool,

        /// Dry run - show what would be extracted without extracting
        #[arg(long)]
        dry_run: bool,
    },

    /// Package FINAL_Data folders into clean archives for S3 upload
    Package {
        /// FINAL_Data directory containing provider folders
        #[arg(short, long)]
        data: PathBuf,

        /// Output directory for archives
        #[arg(short, long)]
        output: PathBuf,

        /// Input XLSX file for validation (only package folders with Excel match)
        #[arg(short, long)]
        xlsx: Option<PathBuf>,

        /// Overwrite existing archives
        #[arg(long)]
        overwrite: bool,

        /// Dry run - show what would be packaged without creating archives
        #[arg(long)]
        dry_run: bool,
    },
}

// =============================================================================
// Data Structures
// =============================================================================

/// Result of geometry extraction
#[allow(dead_code)]
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

/// Data quality issue for reporting
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize)]
struct QualityIssue {
    item_code: String,
    severity: IssueSeverity,
    category: IssueCategory,
    message: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum IssueSeverity {
    Error,
    Warning,
    Info,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum IssueCategory {
    MissingGeometry,
    MissingArchive,
    MissingData,
    OrphanFolder,
    InvalidData,
    ExtractionError,
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
}

/// Scanned folder information
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ScannedFolder {
    code: String,
    path: PathBuf,
    provider: String,
    files: Vec<PathBuf>,
    archive_path: Option<PathBuf>,
    archive_size: Option<u64>,
}

/// Collection definition
#[derive(Debug, Clone)]
struct CollectionDef {
    id: &'static str,
    title: &'static str,
    description: &'static str,
}

/// Map product type codes to collection definitions
fn get_collections() -> HashMap<&'static str, CollectionDef> {
    [
        ("A", CollectionDef {
            id: "webcam-image",
            title: "Webcam Images",
            description: "Time-series webcam imagery from monitoring cameras",
        }),
        ("B", CollectionDef {
            id: "deformation-analysis",
            title: "Deformation Analysis",
            description: "DEFOX deformation analysis imagery",
        }),
        ("D", CollectionDef {
            id: "orthophoto",
            title: "Orthophotos",
            description: "Orthorectified aerial and drone imagery",
        }),
        ("E", CollectionDef {
            id: "radar-displacement",
            title: "Radar Displacement",
            description: "Interferometric radar displacement measurements",
        }),
        ("F", CollectionDef {
            id: "radar-amplitude",
            title: "Radar Amplitude",
            description: "Interferometric radar amplitude data",
        }),
        ("G", CollectionDef {
            id: "radar-coherence",
            title: "Radar Coherence",
            description: "Interferometric radar coherence data",
        }),
        ("H", CollectionDef {
            id: "radar-velocity",
            title: "Radar Velocity Data",
            description: "Interferometric radar velocity measurements per ROI",
        }),
        ("I", CollectionDef {
            id: "dsm",
            title: "Digital Surface Models",
            description: "Digital Surface Models (DSM) from drone and heliborne surveys",
        }),
        ("J", CollectionDef {
            id: "dem",
            title: "Digital Elevation Models",
            description: "Digital Elevation Models (DEM) from surveys",
        }),
        ("K", CollectionDef {
            id: "point-cloud",
            title: "Point Clouds",
            description: "LiDAR and photogrammetric point cloud data",
        }),
        ("L", CollectionDef {
            id: "3d-model",
            title: "3D Models",
            description: "3D visualization models for Sketchfab and similar platforms",
        }),
        ("M", CollectionDef {
            id: "gnss-data",
            title: "GNSS Position Data",
            description: "High-precision GNSS/GPS position time series",
        }),
        ("N", CollectionDef {
            id: "thermal-image",
            title: "Thermal Imagery",
            description: "Heliborne thermal infrared imagery",
        }),
        ("O", CollectionDef {
            id: "hydrology",
            title: "Hydrology Data",
            description: "River discharge and water flow measurements",
        }),
        ("P", CollectionDef {
            id: "lake-level",
            title: "Lake Level",
            description: "Lake level and volume measurements",
        }),
        ("U", CollectionDef {
            id: "radar-timeseries",
            title: "Radar Timeseries",
            description: "Interferometric radar time series data",
        }),
    ]
    .into_iter()
    .collect()
}

/// Item code to archive file mapping
fn get_code_to_file() -> HashMap<&'static str, &'static str> {
    [
        ("02Ah00", "02Ah00_FlexCam_Birchgletscher_BirchbachChannel_SAMPLE.zip"),
        ("04Ba00", "04Ba00_DEFOX_all.zip"),
        ("04Ba01", "04Ba01_DEFOX_2to3_per_d.zip"),
        ("04Ba02", "04Ba02_DEFOX_1_per_d.zip"),
        ("06Ha00", "06Ha00_Radar_Velocities_ROI.zip"),
        ("08Aa00", "08Aa00_Webcam_Lonza_all.7z"),
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
            } else {
                // Transformed coordinates are invalid - CRS issue
                warn!(
                    "GeoTIFF {}: transformed coordinates are invalid (bbox: {:?}). Fix the CRS in the source file or add an override.",
                    filename, b
                );
                (None, None)
            }
        } else {
            (None, None)
        }
    } else {
        // No valid CRS - report the issue
        if looks_like_lv95(minx, miny, maxx, maxy) {
            warn!(
                "GeoTIFF {}: no valid CRS but coordinates look like LV95 (EPSG:2056). Fix the CRS metadata in the source file. Bounds: ({:.0}, {:.0}) - ({:.0}, {:.0})",
                path.file_name().unwrap_or_default().to_string_lossy(),
                minx, miny, maxx, maxy
            );
        } else {
            warn!(
                "GeoTIFF {}: no valid CRS. Fix the CRS metadata in the source file. Bounds: ({:.6}, {:.6}) - ({:.6}, {:.6})",
                path.file_name().unwrap_or_default().to_string_lossy(),
                minx, miny, maxx, maxy
            );
        }
        (None, None)
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

/// Find and extract geospatial file from a zip archive
fn extract_from_zip(archive_path: &Path, tmpdir: &Path) -> Option<ExtractedGeometry> {
    let file = File::open(archive_path).ok()?;
    let mut archive = zip::ZipArchive::new(file).ok()?;

    // Find first geospatial file
    let geo_file_index = (0..archive.len()).find(|&i| {
        if let Ok(file) = archive.by_index(i) {
            let name = file.name().to_lowercase();
            !name.starts_with("__macosx")
                && (name.ends_with(".tif")
                    || name.ends_with(".tiff")
                    || name.ends_with(".laz")
                    || name.ends_with(".las"))
        } else {
            false
        }
    })?;

    let mut file = archive.by_index(geo_file_index).ok()?;
    let outpath = tmpdir.join(
        Path::new(file.name())
            .file_name()
            .unwrap_or_default(),
    );

    // Extract file
    let mut outfile = File::create(&outpath).ok()?;
    std::io::copy(&mut file, &mut outfile).ok()?;
    drop(outfile);

    let name_lower = outpath.to_string_lossy().to_lowercase();
    if name_lower.ends_with(".tif") || name_lower.ends_with(".tiff") {
        extract_geotiff_geometry(&outpath).0 // Ignore override tracking for zip extraction
    } else if name_lower.ends_with(".laz") || name_lower.ends_with(".las") {
        extract_las_geometry(&outpath)
    } else {
        None
    }
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

/// Extract float from cell
fn cell_float(cell: &Data) -> Option<f64> {
    match cell {
        Data::Float(f) => Some(*f),
        Data::Int(i) => Some(*i as f64),
        Data::String(s) => s.trim().parse().ok(),
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

    let collections = get_collections();
    let code_to_file = get_code_to_file();
    let mut items = Vec::new();

    // Data starts at row 4 (0-indexed)
    // Column layout (0-indexed):
    // 0: Code, 1: ProductID, 2: SensorID, 3: DatasetID, 4: BundleID, 5: Sensor,
    // 6: ProductType, 7: Dataset, 8: Bundle, 9: Description, 10: Format,
    // 11: TechnicalInfo, 12: ProcessingLevel, 13: Phase, 14: DateFirst, 15: DateLast,
    // 16: Continued, 17: Frequency, 18: Location, 19: Source, 20: AdditionalRemarks,
    // 21: StorageMB, 22: InternalCommentary
    for (row_idx, row) in range.rows().enumerate() {
        if row_idx < 4 {
            continue; // Skip header rows
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
            sensor: row.get(5).and_then(cell_string),
            product_type: row.get(6).and_then(cell_string),
            dataset: row.get(7).and_then(cell_string),
            bundle: row.get(8).and_then(cell_string),
            description: row.get(9).and_then(cell_string),
            format: row.get(10).and_then(cell_string),
            technical_info: row.get(11).and_then(cell_string),
            processing_level: row.get(12).and_then(cell_int),
            phase: row.get(13).and_then(cell_string),
            date_first: row.get(14).and_then(cell_string),
            date_last: row.get(15).and_then(cell_string),
            continued: row.get(16).map(cell_bool).unwrap_or(false),
            frequency: row.get(17).and_then(cell_string),
            location: row.get(18).and_then(cell_string),
            source: row.get(19).and_then(cell_string),
            additional_remarks: row.get(20).and_then(cell_string),
            storage_mb: row.get(21).and_then(cell_float),
            internal_commentary: row.get(22).and_then(cell_string),
            collection_id: None,
            geometry: None,
            bbox: None,
            archive_file: None,
            archive_size: None,
            files: Vec::new(),
            file_count: 0,
        };

        // Map product_id to collection
        if let Some(ref pid) = product_id {
            if let Some(coll) = collections.get(pid.as_str()) {
                item.collection_id = Some(coll.id.to_string());
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

    // Build title (matching Python: "{sensor_id} - {sensor} - {dataset_id} - {location}")
    let title = format!(
        "{} - {} - {} - {}",
        item.sensor_id.as_deref().unwrap_or(""),
        item.sensor.as_deref().unwrap_or(""),
        item.dataset_id.as_deref().unwrap_or(""),
        item.location.as_deref().unwrap_or("")
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
    if item.continued {
        properties["blatten:continued"] = serde_json::json!(true);
    }
    if let Some(ref fmt) = item.format {
        properties["blatten:format"] = serde_json::json!(fmt);
    }
    if let Some(mb) = item.storage_mb {
        properties["blatten:storage_mb"] = serde_json::json!(mb);
    }
    // File count from folder scanning
    properties["blatten:file_count"] = serde_json::json!(item.file_count);

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

        // Build S3 path as {dataset_code}/{filename}
        let rel_path = format!("{}/{}", item.code, filename);
        let mime_type = get_mime_type(&file_info.path);

        let mut asset = serde_json::json!({
            "href": format!("{}/{}", s3_base_url, rel_path),
            "type": mime_type,
            "title": filename,
            "roles": ["data"]
        });

        // Add file size
        if file_info.size > 0 {
            asset["file:size"] = serde_json::json!(file_info.size);
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
        "https://stac-extensions.github.io/timestamps/v1.2.0/schema.json".to_string(),
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
    let bboxes: Vec<&Vec<f64>> = items.iter().filter_map(|i| i.bbox.as_ref()).collect();
    let spatial_bbox: serde_json::Value = if !bboxes.is_empty() {
        serde_json::json!([[
            bboxes.iter().map(|b| b[0]).fold(f64::INFINITY, f64::min),
            bboxes.iter().map(|b| b[1]).fold(f64::INFINITY, f64::min),
            bboxes.iter().map(|b| b[2]).fold(f64::NEG_INFINITY, f64::max),
            bboxes.iter().map(|b| b[3]).fold(f64::NEG_INFINITY, f64::max),
        ]])
    } else {
        serde_json::json!([[null, null, null, null]])
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
            "https://stac-extensions.github.io/timestamps/v1.2.0/schema.json"
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
    _data_path: Option<&Path>,
    multi_progress: &MultiProgress,
) -> Result<(usize, usize, usize, Vec<ValidationIssue>)> {
    fs::create_dir_all(output_dir)?;
    fs::create_dir_all(output_dir.join("collections"))?;

    let collections_defs = get_collections();
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
            if item.archive_file.is_none() && item.files.is_empty() {
                item_issues.push(ValidationIssue {
                    item_id: item.code.clone(),
                    severity: "warning".to_string(),
                    message: "No archive file mapped and no files found".to_string(),
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

    // Write root catalog
    let catalog = serde_json::json!({
        "type": "Catalog",
        "id": "birch-glacier-collapse",
        "stac_version": STAC_VERSION,
        "title": "Birch Glacier Collapse and Landslide Dataset",
        "description": "Dataset collected during the 2025 Birch glacier collapse and landslide at Blatten, CH-VS",
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
// Validation
// =============================================================================

fn validate_catalog(catalog_dir: &PathBuf) -> Result<Vec<ValidationIssue>> {
    let mut issues = Vec::new();

    // Check catalog.json
    let catalog_path = catalog_dir.join("catalog.json");
    if !catalog_path.exists() {
        issues.push(ValidationIssue {
            item_id: "catalog".to_string(),
            severity: "error".to_string(),
            message: "catalog.json not found".to_string(),
        });
        return Ok(issues);
    }

    let catalog: serde_json::Value = serde_json::from_str(&fs::read_to_string(&catalog_path)?)?;

    if catalog.get("stac_version").and_then(|v| v.as_str()) != Some(STAC_VERSION) {
        issues.push(ValidationIssue {
            item_id: "catalog".to_string(),
            severity: "warning".to_string(),
            message: format!("STAC version is not {}", STAC_VERSION),
        });
    }

    // Check items.ndjson
    let all_items_path = catalog_dir.join("items.ndjson");
    if all_items_path.exists() {
        use std::io::BufRead;
        let file = File::open(&all_items_path)?;
        let reader = BufReader::new(file);
        let mut item_count = 0;

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<serde_json::Value>(&line) {
                Ok(item) => {
                    item_count += 1;
                    let item_id = item.get("id").and_then(|i| i.as_str()).unwrap_or("unknown");

                    if item.get("geometry").map(|g| g.is_null()).unwrap_or(true) {
                        issues.push(ValidationIssue {
                            item_id: item_id.to_string(),
                            severity: "warning".to_string(),
                            message: "Missing geometry".to_string(),
                        });
                    }

                    if item.get("assets").and_then(|a| a.as_object()).map(|a| a.is_empty()).unwrap_or(true) {
                        issues.push(ValidationIssue {
                            item_id: item_id.to_string(),
                            severity: "warning".to_string(),
                            message: "No assets".to_string(),
                        });
                    }

                    if item.get("stac_version").and_then(|v| v.as_str()) != Some(STAC_VERSION) {
                        issues.push(ValidationIssue {
                            item_id: item_id.to_string(),
                            severity: "warning".to_string(),
                            message: format!("STAC version is not {}", STAC_VERSION),
                        });
                    }
                }
                Err(e) => {
                    issues.push(ValidationIssue {
                        item_id: format!("line_{}", line_num + 1),
                        severity: "error".to_string(),
                        message: format!("Failed to parse NDJSON line: {}", e),
                    });
                }
            }
        }

        println!("Found {} items", item_count);
    }

    Ok(issues)
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

    match cli.command {
        Commands::Generate {
            xlsx,
            output,
            base_url,
            s3_base_url,
            save_metadata,
            data,
            archives_dir,
            sensors_file,
            dnage_mappings,
            validate_s3,
            threads,
            verbose,
            validate,
        } => {
            info!("=== STAC Generator (v{}) ===", STAC_VERSION);

            // Configure rayon thread pool
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .ok();

            // Set up progress bars
            let multi_progress = MultiProgress::new();
            let spinner_style = ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .expect("Invalid spinner template");

            // Step 1: Parse XLSX
            let parse_pb = multi_progress.add(ProgressBar::new_spinner());
            parse_pb.set_style(spinner_style.clone());
            parse_pb.set_message("Parsing XLSX metadata...");

            let mut items = parse_xlsx(&xlsx)?;
            parse_pb.finish_with_message(format!("Parsed {} items from XLSX", items.len()));

            // Step 2: Scan folders if data provided
            let scanned_folders: HashMap<String, ScannedFolder> = if let Some(ref data_path) = data {
                let scan_pb = multi_progress.add(ProgressBar::new_spinner());
                scan_pb.set_style(spinner_style.clone());
                scan_pb.set_message("Scanning data folders...");

                let folders = scan_data_folders(data_path, dnage_mappings.as_deref())?;
                scan_pb.finish_with_message(format!("Found {} data folders", folders.len()));
                folders
            } else {
                HashMap::new()
            };

            // Keep a copy for quality report (before moving to Arc)
            let scanned_folders_for_report = scanned_folders.clone();

            // Track applied overrides for reporting
            let mut applied_overrides: Vec<(String, String, String)> = Vec::new(); // (code, file, reason)

            // Step 3: Extract geometry
            let data_path = data.as_deref();
            if data_path.is_some() || sensors_file.is_some() {
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

                // Create temp directory for extraction
                let tmpdir = std::env::temp_dir().join("stac-gen-extract");
                fs::create_dir_all(&tmpdir)?;

                // Extract geometry in parallel - only for items WITHOUT scanned folders
                // (items with folders will get geometry from file-level extraction later)
                let extract_pb = Arc::new(Mutex::new(extract_pb));
                let sensors = Arc::new(sensors);
                let scanned_folders = Arc::new(scanned_folders);
                let tmpdir = Arc::new(tmpdir);

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
                            data_path,
                            &sensors,
                            &scanned_folders,
                            &tmpdir,
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

                if let Ok(pb) = extract_pb.lock() {
                    pb.finish_with_message(format!(
                        "Geometry: {} extracted, {} manual, {} none, {} failed",
                        extracted, manual, no_geometry, failed
                    ));
                }

                // Cleanup
                let _ = fs::remove_dir_all(tmpdir.as_ref());
            }

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
                    let (bbox, geometry, bbox_lv95, geometry_lv95) = if ext == "tif" || ext == "tiff" {
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

            // Step 4: Check archive files
            if let Some(ref arch_dir) = archives_dir {
                let arch_pb = multi_progress.add(ProgressBar::new_spinner());
                arch_pb.set_style(spinner_style.clone());
                arch_pb.set_message("Checking archive files...");

                let mut found = 0;
                let mut missing = 0;

                for item in &mut items {
                    if let Some(ref archive_name) = item.archive_file {
                        let archive_path = arch_dir.join(archive_name);
                        if archive_path.exists() {
                            if let Ok(metadata) = fs::metadata(&archive_path) {
                                item.archive_size = Some(metadata.len());
                                found += 1;
                            }
                        } else {
                            missing += 1;
                            if verbose {
                                warn!("[{}] Archive not found: {}", item.code, archive_name);
                            }
                        }
                    }
                }

                arch_pb.finish_with_message(format!("Archives: {} found, {} missing", found, missing));
            }

            // Save intermediate metadata if requested
            if let Some(ref meta_path) = save_metadata {
                info!("Saving metadata to {:?}...", meta_path);
                if let Some(parent) = meta_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let metadata = serde_json::json!({
                    "generated_at": Utc::now().to_rfc3339(),
                    "stac_version": STAC_VERSION,
                    "items": items
                });
                fs::write(meta_path, serde_json::to_string_pretty(&metadata)?)?;
            }

            // Generate STAC catalog
            let gen_pb = multi_progress.add(ProgressBar::new_spinner());
            gen_pb.set_style(spinner_style);
            gen_pb.set_message("Generating STAC catalog...");

            let (num_collections, num_items, num_assets, issues) = generate_catalog(&items, &output, &base_url, &s3_base_url, data.as_deref(), &multi_progress)?;
            gen_pb.finish_with_message(format!(
                "Generated {} collections, {} items, {} assets",
                num_collections, num_items, num_assets
            ));

            // Compute detailed quality metrics
            let excel_codes: std::collections::HashSet<_> = items.iter().map(|i| &i.code).collect();

            // 1. Excel items without matching data folders
            let excel_without_folders: Vec<_> = items.iter()
                .filter(|i| !scanned_folders_for_report.contains_key(&i.code))
                .map(|i| i.code.clone())
                .collect();

            // 2. Data folders without matching Excel entries (orphan folders)
            let orphan_folders: Vec<_> = scanned_folders_for_report.keys()
                .filter(|code| !excel_codes.contains(code))
                .cloned()
                .collect();

            // 3. Items with files but no geometry
            let files_no_geometry: Vec<_> = items.iter()
                .filter(|i| {
                    i.geometry.is_none() && scanned_folders_for_report.get(&i.code).map(|f| !f.files.is_empty()).unwrap_or(false)
                })
                .map(|i| i.code.clone())
                .collect();

            // 4. Items with folders but no archive
            let folders_no_archive: Vec<_> = items.iter()
                .filter(|i| {
                    scanned_folders_for_report.contains_key(&i.code) && i.archive_file.is_none()
                })
                .map(|i| i.code.clone())
                .collect();

            // 5. Items with geometry
            let items_with_geometry: Vec<_> = items.iter()
                .filter(|i| i.geometry.is_some())
                .map(|i| i.code.clone())
                .collect();

            // 6. Items without geometry
            let items_without_geometry: Vec<_> = items.iter()
                .filter(|i| i.geometry.is_none())
                .map(|i| i.code.clone())
                .collect();

            // Print comprehensive report (detailed issues first, summary at the end)
            info!("");
            info!("╔══════════════════════════════════════════════════════════════╗");
            info!("║                    DATA QUALITY REPORT                       ║");
            info!("╚══════════════════════════════════════════════════════════════╝");
            info!("");

            // 1. Excel items without folders
            info!("=== Excel Items Without Data Folders ({}) ===", excel_without_folders.len());
            if excel_without_folders.is_empty() {
                info!("  (none - all Excel items have matching folders)");
            } else {
                for code in &excel_without_folders {
                    warn!("  - {}", code);
                }
            }
            info!("");

            // 2. Orphan folders
            info!("=== Orphan Folders (not in Excel) ({}) ===", orphan_folders.len());
            if orphan_folders.is_empty() {
                info!("  (none - all folders have matching Excel entries)");
            } else {
                for code in &orphan_folders {
                    warn!("  - {}", code);
                }
            }
            info!("");

            // 3. Files but no geometry
            info!("=== Items With Files But No Geometry ({}) ===", files_no_geometry.len());
            if files_no_geometry.is_empty() {
                info!("  (none - all items with files have geometry)");
            } else {
                for code in &files_no_geometry {
                    warn!("  - {}", code);
                }
            }
            info!("");

            // 4. Folders but no archive
            info!("=== Items With Folders But No Archive ({}) ===", folders_no_archive.len());
            if folders_no_archive.is_empty() {
                info!("  (none - all items with folders have archives)");
            } else {
                for code in &folders_no_archive {
                    warn!("  - {}", code);
                }
            }
            info!("");

            // Items without geometry (for reference)
            info!("=== Items Missing Geometry ({}) ===", items_without_geometry.len());
            if items_without_geometry.is_empty() {
                info!("  (none - all items have geometry)");
            } else {
                for code in &items_without_geometry {
                    warn!("  - {}", code);
                }
            }
            info!("");

            info!("=== Validation Issues ({}) ===", issues.len());
            if issues.is_empty() {
                info!("  (no validation issues)");
            } else {
                for issue in &issues {
                    match issue.severity.as_str() {
                        "error" => error!("  [{}] {}", issue.item_id, issue.message),
                        "warning" => warn!("  [{}] {}", issue.item_id, issue.message),
                        _ => info!("  [{}] {}", issue.item_id, issue.message),
                    }
                }
            }
            info!("");

            // Applied overrides section
            info!("=== Applied Overrides ({}) ===", applied_overrides.len());
            if applied_overrides.is_empty() {
                info!("  (no overrides applied - all files had valid CRS metadata)");
            } else {
                for (code, source, reason) in &applied_overrides {
                    info!("  [{}] {} - {}", code, source, reason);
                }
            }
            info!("");

            // S3 validation (optional)
            if validate_s3 {
                info!("S3 validation not yet implemented");
                // TODO: Implement S3 HEAD requests to verify files exist
                info!("");
            }

            // === SUMMARY (at the very end) ===
            info!("╔══════════════════════════════════════════════════════════════╗");
            info!("║                         SUMMARY                              ║");
            info!("╚══════════════════════════════════════════════════════════════╝");
            info!("");
            info!("=== Overview ===");
            info!("  Total Excel items:     {}", items.len());
            info!("  Total data folders:    {}", scanned_folders_for_report.len());
            info!("  Collections generated: {}", num_collections);
            info!("  Items:                 {}", num_items);
            info!("  Total assets:          {}", num_assets);
            info!("");
            info!("=== Geometry Coverage ===");
            info!("  Items WITH geometry:    {} ({:.1}%)",
                items_with_geometry.len(),
                if items.is_empty() { 0.0 } else { items_with_geometry.len() as f64 / items.len() as f64 * 100.0 }
            );
            info!("  Items WITHOUT geometry: {} ({:.1}%)",
                items_without_geometry.len(),
                if items.is_empty() { 0.0 } else { items_without_geometry.len() as f64 / items.len() as f64 * 100.0 }
            );
            info!("");
            info!("=== Issue Counts ===");
            info!("  Excel items without folders: {}", excel_without_folders.len());
            info!("  Orphan folders:              {}", orphan_folders.len());
            info!("  Files but no geometry:       {}", files_no_geometry.len());
            info!("  Folders but no archive:      {}", folders_no_archive.len());
            info!("  Validation issues:           {}", issues.len());
            info!("  Applied overrides:           {}", applied_overrides.len());

            // Run validation after generation if requested
            if validate {
                info!("");
                info!("=== Running Post-Generation Validation ===");
                let validation_issues = validate_catalog(&output)?;

                let has_errors = validation_issues.iter().any(|i| i.severity == "error");
                info!("Validation issues: {}", validation_issues.len());

                if !validation_issues.is_empty() {
                    for issue in &validation_issues {
                        match issue.severity.as_str() {
                            "error" => error!("  [{}] {}", issue.item_id, issue.message),
                            "warning" => warn!("  [{}] {}", issue.item_id, issue.message),
                            _ => info!("  [{}] {}", issue.item_id, issue.message),
                        }
                    }
                }

                if has_errors {
                    error!("Validation failed with errors");
                    std::process::exit(1);
                } else {
                    info!("Validation passed");
                }
            }
        }

        Commands::Validate { catalog_dir } => {
            info!("=== STAC Validator (v{}) ===", STAC_VERSION);
            info!("Validating {:?}...", catalog_dir);

            let issues = validate_catalog(&catalog_dir)?;

            info!("");
            info!("=== Validation Results ===");
            info!("Issues found: {}", issues.len());

            if !issues.is_empty() {
                for issue in &issues {
                    match issue.severity.as_str() {
                        "error" => error!("  [{}] {}", issue.item_id, issue.message),
                        "warning" => warn!("  [{}] {}", issue.item_id, issue.message),
                        _ => info!("  [{}] {}", issue.item_id, issue.message),
                    }
                }
            }

            std::process::exit(if issues.iter().any(|i| i.severity == "error") { 1 } else { 0 });
        }

        Commands::Extract { input, output, provider, overwrite, dry_run } => {
            info!("=== Archive Extractor ===");
            info!("Input:    {:?}", input);
            info!("Output:   {:?}", output);
            info!("Provider: {}", provider);
            if dry_run {
                info!("Mode:     DRY RUN");
            }

            // Validate provider name
            let valid_providers = ["Terradata", "Geopraevent", "DNAGE"];
            if !valid_providers.contains(&provider.as_str()) {
                error!("Invalid provider '{}'. Must be one of: {:?}", provider, valid_providers);
                std::process::exit(1);
            }

            // Create output directory structure
            let provider_output = output.join(&provider);
            if !dry_run {
                fs::create_dir_all(&provider_output)?;
            }

            // Find all archive files
            let archives: Vec<_> = WalkDir::new(&input)
                .min_depth(1)
                .max_depth(1)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let ext = e.path().extension().and_then(|s| s.to_str()).unwrap_or("");
                    ext == "zip" || ext == "7z"
                })
                .collect();

            info!("Found {} archives", archives.len());

            let mut extracted = 0;
            let mut skipped = 0;
            let mut failed = 0;

            for entry in archives {
                let archive_path = entry.path();
                let filename = archive_path.file_stem().unwrap_or_default().to_string_lossy();

                // Extract code from filename (e.g., "06Ha00_Radar..." -> "06Ha00")
                let code = extract_code_from_folder(&filename, &HashMap::new());

                if let Some(code) = code {
                    let target_dir = provider_output.join(&code);

                    if target_dir.exists() && !overwrite {
                        info!("  SKIP: {} -> {} (already exists)", filename, code);
                        skipped += 1;
                        continue;
                    }

                    if dry_run {
                        info!("  WOULD EXTRACT: {} -> {}", filename, code);
                        extracted += 1;
                    } else {
                        info!("  Extracting: {} -> {}", filename, code);
                        match extract_archive(archive_path, &target_dir) {
                            Ok(_) => {
                                extracted += 1;
                            }
                            Err(e) => {
                                error!("  FAILED: {} - {}", filename, e);
                                failed += 1;
                            }
                        }
                    }
                } else {
                    warn!("  SKIP: {} (could not extract code from filename)", filename);
                    skipped += 1;
                }
            }

            info!("");
            info!("=== Summary ===");
            info!("  Extracted: {}", extracted);
            info!("  Skipped:   {}", skipped);
            info!("  Failed:    {}", failed);
        }

        Commands::Package { data, output, xlsx, overwrite, dry_run } => {
            info!("=== Archive Packager ===");
            info!("Data:   {:?}", data);
            info!("Output: {:?}", output);
            if let Some(ref x) = xlsx {
                info!("XLSX:   {:?}", x);
            }
            if dry_run {
                info!("Mode:   DRY RUN");
            }

            // Load Excel codes if provided (for validation)
            let valid_codes: Option<std::collections::HashSet<String>> = if let Some(ref xlsx_path) = xlsx {
                let items = parse_xlsx(xlsx_path)?;
                Some(items.iter().map(|i| i.code.clone()).collect())
            } else {
                None
            };

            // Create output directory
            if !dry_run {
                fs::create_dir_all(&output)?;
            }

            // Scan all provider folders
            let mut packaged = 0;
            let mut skipped = 0;
            let mut failed = 0;

            for provider in &["Terradata", "Geopraevent", "DNAGE"] {
                let provider_path = data.join(provider);
                if !provider_path.exists() {
                    continue;
                }

                for entry in WalkDir::new(&provider_path)
                    .min_depth(1)
                    .max_depth(1)
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_type().is_dir())
                {
                    let folder_name = entry.file_name().to_string_lossy().to_string();
                    let code = extract_code_from_folder(&folder_name, &HashMap::new());

                    if let Some(code) = code {
                        // Check against Excel codes if provided
                        if let Some(ref codes) = valid_codes {
                            if !codes.contains(&code) {
                                warn!("  SKIP: {} (not in Excel metadata)", code);
                                skipped += 1;
                                continue;
                            }
                        }

                        let archive_path = output.join(format!("{}.zip", code));

                        if archive_path.exists() && !overwrite {
                            info!("  SKIP: {} (archive already exists)", code);
                            skipped += 1;
                            continue;
                        }

                        if dry_run {
                            // Count files and size
                            let (file_count, total_size) = count_folder_contents(entry.path());
                            info!("  WOULD PACKAGE: {} ({} files, {})", code, file_count, format_size(total_size));
                            packaged += 1;
                        } else {
                            info!("  Packaging: {} ...", code);
                            match create_archive(entry.path(), &archive_path) {
                                Ok(size) => {
                                    info!("    Created: {} ({})", archive_path.display(), format_size(size));
                                    packaged += 1;
                                }
                                Err(e) => {
                                    error!("  FAILED: {} - {}", code, e);
                                    failed += 1;
                                }
                            }
                        }
                    } else {
                        warn!("  SKIP: {} (could not extract code)", folder_name);
                        skipped += 1;
                    }
                }
            }

            info!("");
            info!("=== Summary ===");
            info!("  Packaged: {}", packaged);
            info!("  Skipped:  {}", skipped);
            info!("  Failed:   {}", failed);
        }
    }

    Ok(())
}

// =============================================================================
// Archive Operations
// =============================================================================

/// Extract a zip archive to a target directory
fn extract_archive(archive_path: &Path, target_dir: &Path) -> Result<()> {
    let file = File::open(archive_path)?;
    let mut archive = zip::ZipArchive::new(file)?;

    // Create target directory
    fs::create_dir_all(target_dir)?;

    for i in 0..archive.len() {
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

        // Set permissions on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Some(mode) = file.unix_mode() {
                fs::set_permissions(&outpath, fs::Permissions::from_mode(mode))?;
            }
        }
    }

    Ok(())
}

/// Create a zip archive from a directory
fn create_archive(source_dir: &Path, archive_path: &Path) -> Result<u64> {
    use zip::write::SimpleFileOptions;

    let file = File::create(archive_path)?;
    let mut zip = zip::ZipWriter::new(file);
    let options = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .compression_level(Some(6));

    // Walk the directory and add all files
    for entry in WalkDir::new(source_dir).into_iter().filter_map(|e| e.ok()) {
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
// Folder Scanning
// =============================================================================

/// Scan FINAL_Data directory for provider folders and map to item codes
fn scan_data_folders(
    data: &Path,
    dnage_mappings_path: Option<&Path>,
) -> Result<HashMap<String, ScannedFolder>> {
    let mut folders = HashMap::new();

    // Load DNAGE mappings if provided
    let dnage_mappings: HashMap<String, serde_json::Value> = if let Some(path) = dnage_mappings_path {
        let content = fs::read_to_string(path)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;
        data.get("mappings")
            .and_then(|m| m.as_object())
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default()
    } else {
        HashMap::new()
    };

    // Scan each provider directory
    for provider in &["DNAGE", "Geopraevent", "Terradata"] {
        let provider_path = data.join(provider);
        if !provider_path.exists() {
            continue;
        }

        for entry in WalkDir::new(&provider_path)
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_dir() {
                continue;
            }

            let folder_name = entry.file_name().to_string_lossy().to_string();

            // Extract code from folder name (e.g., "13Da01_Orthophoto_..." -> "13Da01")
            let code = extract_code_from_folder(&folder_name, &dnage_mappings);

            if let Some(code) = code {
                // Collect files in this folder
                let files: Vec<PathBuf> = WalkDir::new(entry.path())
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_type().is_file())
                    .map(|e| e.path().to_path_buf())
                    .collect();

                folders.insert(code.clone(), ScannedFolder {
                    code,
                    path: entry.path().to_path_buf(),
                    provider: provider.to_string(),
                    files,
                    archive_path: None,
                    archive_size: None,
                });
            }
        }
    }

    // Apply DNAGE mappings for shared folders
    for (code, mapping) in &dnage_mappings {
        if folders.contains_key(code) {
            continue;
        }

        if let Some(folder_rel) = mapping.get("folder").and_then(|f| f.as_str()) {
            let folder_path = data.join(folder_rel);
            if folder_path.exists() {
                let files: Vec<PathBuf> = WalkDir::new(&folder_path)
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_type().is_file())
                    .map(|e| e.path().to_path_buf())
                    .collect();

                folders.insert(code.clone(), ScannedFolder {
                    code: code.clone(),
                    path: folder_path,
                    provider: "DNAGE".to_string(),
                    files,
                    archive_path: None,
                    archive_size: None,
                });
            }
        }
    }

    Ok(folders)
}

/// Extract item code from folder name
fn extract_code_from_folder(folder_name: &str, dnage_mappings: &HashMap<String, serde_json::Value>) -> Option<String> {
    // Pattern: code at start followed by underscore (e.g., "13Da01_...")
    let code_re = regex::Regex::new(r"^(\d{2}[A-Z][a-z]\d{2})").ok()?;

    if let Some(caps) = code_re.captures(folder_name) {
        return Some(caps.get(1)?.as_str().to_string());
    }

    // Check DNAGE mappings for non-standard folder names
    for (code, mapping) in dnage_mappings {
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
    data: Option<&Path>,
    sensors: &HashMap<String, SensorLocation>,
    scanned_folders: &HashMap<String, ScannedFolder>,
    tmpdir: &Path,
) -> ExtractionResult {
    let code = &item.code;
    let data_format = item.format.as_deref().unwrap_or("").to_uppercase();

    // Check if format has extractable geometry
    let is_geospatial = matches!(data_format.as_str(), "TIF" | "TIFF" | "LAZ" | "LAS");

    // Try to extract from scanned folder files first
    if is_geospatial {
        if let Some(folder) = scanned_folders.get(code) {
            // Find a geospatial file
            for file in &folder.files {
                let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                if ext == "tif" || ext == "tiff" {
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

    // Try archive extraction if data provided
    if let Some(data_path) = data {
        if let Some(ref archive_name) = item.archive_file {
            let archive_path = data_path.join(archive_name);
            if archive_path.exists() && archive_path.extension().map(|e| e == "zip").unwrap_or(false) {
                if let Some(geo) = extract_from_zip(&archive_path, tmpdir) {
                    return ExtractionResult::Extracted {
                        source: geo.source,
                        bbox: geo.bbox,
                        geometry: geo.geometry,
                        override_reason: None, // TODO: could track overrides in zip extraction too
                    };
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
