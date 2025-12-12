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
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

const STAC_VERSION: &str = "1.1.0";

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

        /// Base URL for STAC links (use ${STAC_BASE_URL} for placeholder)
        #[arg(short, long, default_value = "${STAC_BASE_URL}")]
        base_url: String,

        /// Save intermediate metadata JSON
        #[arg(long)]
        save_metadata: Option<PathBuf>,

        /// Data directory containing archives for geometry extraction
        #[arg(short, long)]
        data_dir: Option<PathBuf>,

        /// Sensor locations JSON for manual coordinates
        #[arg(long)]
        sensors_file: Option<PathBuf>,
    },
    /// Validate an existing STAC catalog
    Validate {
        /// Catalog directory
        #[arg(short, long, default_value = "stac")]
        catalog_dir: PathBuf,
    },
}

// =============================================================================
// Data Structures
// =============================================================================

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
        ("P", CollectionDef {
            id: "hydrology",
            title: "Hydrology Data",
            description: "Lake level and volume measurements",
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

/// Sensor location with manual coordinates
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SensorLocation {
    name: Option<String>,
    lon: Option<f64>,
    lat: Option<f64>,
    elevation_m: Option<f64>,
    items: Vec<String>,
}

/// Load sensor locations from JSON file
fn load_sensor_locations(path: &Path) -> Result<HashMap<String, SensorLocation>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }

    let content = fs::read_to_string(path)?;
    let data: serde_json::Value = serde_json::from_str(&content)?;

    let mut lookup = HashMap::new();
    if let Some(sensors) = data.get("sensors").and_then(|s| s.as_object()) {
        for (sensor_id, sensor_data) in sensors {
            let sensor: SensorLocation = serde_json::from_value(sensor_data.clone())?;
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
    bbox: Vec<f64>,
    geometry: serde_json::Value,
    source: String,
}

/// Transform coordinates from a source EPSG to WGS84 using GDAL
fn transform_to_wgs84(minx: f64, miny: f64, maxx: f64, maxy: f64, epsg: i32) -> Option<Vec<f64>> {
    let source_srs = SpatialRef::from_epsg(epsg as u32).ok()?;
    let target_srs = SpatialRef::from_epsg(4326).ok()?;
    let transform = CoordTransform::new(&source_srs, &target_srs).ok()?;

    // Transform all four corners for accuracy
    let mut xs = [minx, maxx, maxx, minx];
    let mut ys = [miny, miny, maxy, maxy];
    let mut zs = [0.0; 4];

    transform.transform_coords(&mut xs, &mut ys, &mut zs).ok()?;

    Some(vec![
        xs.iter().cloned().fold(f64::INFINITY, f64::min),
        ys.iter().cloned().fold(f64::INFINITY, f64::min),
        xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
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
fn extract_geotiff_geometry(path: &Path) -> Option<ExtractedGeometry> {
    let dataset = Dataset::open(path).ok()?;

    // Get geotransform: [origin_x, pixel_width, rotation_x, origin_y, rotation_y, pixel_height]
    let gt = dataset.geo_transform().ok()?;
    let (width, height) = dataset.raster_size();

    // Calculate corners from geotransform
    let minx = gt[0];
    let maxy = gt[3];
    let maxx = minx + (width as f64 * gt[1]);
    let miny = maxy + (height as f64 * gt[5]); // gt[5] is typically negative

    // Ensure correct ordering
    let (minx, maxx) = (minx.min(maxx), minx.max(maxx));
    let (miny, maxy) = (miny.min(maxy), miny.max(maxy));

    // Get CRS and transform to WGS84
    let spatial_ref = dataset.spatial_ref().ok()?;
    let target_srs = SpatialRef::from_epsg(4326).ok()?;

    let bbox = if let Ok(transform) = CoordTransform::new(&spatial_ref, &target_srs) {
        // Transform all four corners
        let mut xs = [minx, maxx, maxx, minx];
        let mut ys = [miny, miny, maxy, maxy];
        let mut zs = [0.0; 4];

        if transform.transform_coords(&mut xs, &mut ys, &mut zs).is_ok() {
            vec![
                xs.iter().cloned().fold(f64::INFINITY, f64::min),
                ys.iter().cloned().fold(f64::INFINITY, f64::min),
                xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            ]
        } else {
            // Fallback: try to extract EPSG and use our transform function
            let wkt = spatial_ref.to_wkt().ok()?;
            let epsg_str = extract_epsg_from_crs(&wkt)?;
            let epsg: i32 = epsg_str.strip_prefix("EPSG:")?.parse().ok()?;
            transform_to_wgs84(minx, miny, maxx, maxy, epsg)?
        }
    } else {
        // Try extracting EPSG from WKT
        let wkt = spatial_ref.to_wkt().ok()?;
        let epsg_str = extract_epsg_from_crs(&wkt)?;
        let epsg: i32 = epsg_str.strip_prefix("EPSG:")?.parse().ok()?;

        if epsg == 4326 {
            vec![minx, miny, maxx, maxy]
        } else {
            transform_to_wgs84(minx, miny, maxx, maxy, epsg)?
        }
    };

    let geometry = bbox_to_polygon(&bbox);

    Some(ExtractedGeometry {
        bbox,
        geometry,
        source: format!("GeoTIFF: {}", path.file_name()?.to_string_lossy()),
    })
}

/// Extract geometry from a LAZ/LAS file
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

    let bbox = if epsg == 4326 {
        vec![minx, miny, maxx, maxy]
    } else {
        transform_to_wgs84(minx, miny, maxx, maxy, epsg)?
    };

    let geometry = bbox_to_polygon(&bbox);

    Some(ExtractedGeometry {
        bbox,
        geometry,
        source: format!("LAZ/LAS: {} (EPSG:{})", path.file_name()?.to_string_lossy(), epsg),
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
        extract_geotiff_geometry(&outpath)
    } else if name_lower.ends_with(".laz") || name_lower.ends_with(".las") {
        extract_las_geometry(&outpath)
    } else {
        None
    }
}

/// Extract geometry for an item
fn extract_geometry_for_item(
    item: &mut ItemMetadata,
    data_dir: &Path,
    sensors: &HashMap<String, SensorLocation>,
    tmpdir: &Path,
) -> Result<Option<String>> {
    let code = &item.code;
    let data_format = item.format.as_deref().unwrap_or("").to_uppercase();

    // Check if format has extractable geometry
    let is_geospatial = matches!(data_format.as_str(), "TIF" | "TIFF" | "LAZ" | "LAS");

    // Get archive file
    let archive_name = match &item.archive_file {
        Some(name) => name.clone(),
        None => return Ok(None),
    };

    let archive_path = data_dir.join(&archive_name);
    if !archive_path.exists() {
        // Try dummy folder
        let dummy_path = data_dir.join("dummy").join(&archive_name);
        if !dummy_path.exists() {
            return Ok(Some(format!("Archive not found: {}", archive_name)));
        }
    }

    let archive_path = if archive_path.exists() {
        archive_path
    } else {
        data_dir.join("dummy").join(&archive_name)
    };

    if is_geospatial && archive_path.extension().map(|e| e == "zip").unwrap_or(false) {
        // Try extracting from zip
        if let Some(geo) = extract_from_zip(&archive_path, tmpdir) {
            item.bbox = Some(geo.bbox);
            item.geometry = Some(geo.geometry);
            return Ok(Some(format!("Extracted: {}", geo.source)));
        }
    }

    // Fall back to manual coordinates
    if let Some(sensor) = sensors.get(code) {
        if let (Some(lon), Some(lat)) = (sensor.lon, sensor.lat) {
            let (bbox, geometry) = point_to_geometry(lon, lat, sensor.elevation_m);
            item.bbox = Some(bbox);
            item.geometry = Some(geometry);
            return Ok(Some(format!(
                "Manual: {} at [{}, {}]",
                sensor.name.as_deref().unwrap_or(code),
                lon,
                lat
            )));
        }
    }

    Ok(None)
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

/// Parse the Test_Data sheet from XLSX
fn parse_xlsx(path: &PathBuf) -> Result<Vec<ItemMetadata>> {
    let mut workbook: Xlsx<_> = open_workbook(path)
        .with_context(|| format!("Failed to open XLSX: {:?}", path))?;

    let range = workbook
        .worksheet_range("Test_Data")
        .with_context(|| "Sheet 'Test_Data' not found")?;

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

/// Create a STAC Item from metadata
fn create_stac_item(item: &ItemMetadata, base_url: &str) -> serde_json::Value {
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

    // Build title
    let title = format!(
        "{} - {}",
        item.sensor.as_deref().unwrap_or(""),
        item.dataset.as_deref().unwrap_or("")
    )
    .trim_matches(|c| c == ' ' || c == '-')
    .to_string();
    let title = if title.is_empty() { item.code.clone() } else { title };

    // Build properties
    let mut properties = serde_json::json!({
        "title": title,
        "datetime": datetime,
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

    // Custom properties
    properties["blatten:code"] = serde_json::json!(item.code);
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

    // Build assets
    let mut assets = serde_json::Map::new();
    if let Some(ref archive) = item.archive_file {
        let media_type = if archive.ends_with(".zip") {
            "application/zip"
        } else if archive.ends_with(".7z") {
            "application/x-7z-compressed"
        } else {
            "application/octet-stream"
        };

        let mut asset = serde_json::json!({
            "href": format!("{}/s3/data/{}", base_url, archive),
            "type": media_type,
            "title": format!("{} Data Archive", item.code),
            "roles": ["data"]
        });

        if let Some(mb) = item.storage_mb {
            asset["file:size"] = serde_json::json!((mb * 1024.0 * 1024.0) as i64);
        }

        assets.insert("data".to_string(), asset);
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

    serde_json::json!({
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "stac_extensions": [
            "https://stac-extensions.github.io/timestamps/v1.2.0/schema.json"
        ],
        "id": item.code,
        "geometry": item.geometry,
        "bbox": item.bbox,
        "properties": properties,
        "links": links,
        "assets": assets,
        "collection": collection_id
    })
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
fn generate_catalog(
    items: &[ItemMetadata],
    output_dir: &PathBuf,
    base_url: &str,
) -> Result<(usize, usize, Vec<ValidationIssue>)> {
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

    // Generate STAC items and collections
    let mut all_stac_items = Vec::new();
    let mut stac_collections = Vec::new();

    for (coll_id, coll_items) in &items_by_collection {
        // Find collection definition
        let def = collections_defs
            .values()
            .find(|d| d.id == coll_id)
            .expect("Unknown collection");

        // Create STAC items
        let mut stac_items = Vec::new();
        for item in coll_items {
            let stac_item = create_stac_item(item, base_url);
            stac_items.push(stac_item.clone());
            all_stac_items.push(stac_item);

            // Validate
            if item.geometry.is_none() {
                issues.push(ValidationIssue {
                    item_id: item.code.clone(),
                    severity: "warning".to_string(),
                    message: "Missing geometry".to_string(),
                });
            }
            if item.archive_file.is_none() {
                issues.push(ValidationIssue {
                    item_id: item.code.clone(),
                    severity: "warning".to_string(),
                    message: "No archive file mapped".to_string(),
                });
            }
        }

        // Create collection
        let collection = create_stac_collection(def, coll_items, base_url);
        stac_collections.push(collection.clone());

        // Write collection file
        let coll_path = output_dir.join("collections").join(format!("{}.json", coll_id));
        fs::write(&coll_path, serde_json::to_string_pretty(&collection)?)?;

        // Write collection items
        let items_path = output_dir.join("collections").join(format!("{}_items.json", coll_id));
        let items_fc = serde_json::json!({
            "type": "FeatureCollection",
            "features": stac_items,
            "numberMatched": stac_items.len(),
            "numberReturned": stac_items.len()
        });
        fs::write(&items_path, serde_json::to_string_pretty(&items_fc)?)?;
    }

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

    // Write all items
    let all_items_fc = serde_json::json!({
        "type": "FeatureCollection",
        "features": all_stac_items,
        "numberMatched": all_stac_items.len(),
        "numberReturned": all_stac_items.len()
    });
    fs::write(
        output_dir.join("all_items.json"),
        serde_json::to_string_pretty(&all_items_fc)?,
    )?;

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

    // Write validation report
    let report = serde_json::json!({
        "generated_at": Utc::now().to_rfc3339(),
        "stac_version": STAC_VERSION,
        "total_collections": stac_collections.len(),
        "total_items": all_stac_items.len(),
        "items_with_geometry": all_stac_items.iter().filter(|i| !i["geometry"].is_null()).count(),
        "items_missing_geometry": all_stac_items.iter().filter(|i| i["geometry"].is_null()).count(),
        "issues": issues,
        "validation_passed": issues.iter().all(|i| i.severity != "error")
    });
    fs::write(
        output_dir.join("validation_report.json"),
        serde_json::to_string_pretty(&report)?,
    )?;

    Ok((stac_collections.len(), all_stac_items.len(), issues))
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

    // Check all_items.json
    let all_items_path = catalog_dir.join("all_items.json");
    if all_items_path.exists() {
        let data: serde_json::Value = serde_json::from_str(&fs::read_to_string(&all_items_path)?)?;
        if let Some(features) = data.get("features").and_then(|f| f.as_array()) {
            println!("Found {} items", features.len());

            for item in features {
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
        }
    }

    Ok(issues)
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            xlsx,
            output,
            base_url,
            save_metadata,
            data_dir,
            sensors_file,
        } => {
            println!("=== STAC Generator (v{}) ===\n", STAC_VERSION);

            println!("1. Parsing XLSX metadata...");
            let mut items = parse_xlsx(&xlsx)?;
            println!("   Parsed {} items\n", items.len());

            // Extract geometry if data_dir is provided
            if let Some(ref data_path) = data_dir {
                println!("2. Extracting geometry from data files...");

                // Load sensor locations
                let sensors = if let Some(ref sensors_path) = sensors_file {
                    load_sensor_locations(sensors_path)?
                } else {
                    // Try default location
                    let default_path = data_path.join("sensor_locations.json");
                    if default_path.exists() {
                        load_sensor_locations(&default_path)?
                    } else {
                        HashMap::new()
                    }
                };

                // Create temp directory for extraction
                let tmpdir = std::env::temp_dir().join("stac-gen-extract");
                fs::create_dir_all(&tmpdir)?;

                let mut extracted = 0;
                let mut manual = 0;
                let mut failed = 0;

                for item in &mut items {
                    match extract_geometry_for_item(item, data_path, &sensors, &tmpdir) {
                        Ok(Some(msg)) => {
                            if msg.starts_with("Extracted:") {
                                extracted += 1;
                                println!("   [{}] {}", item.code, msg);
                            } else if msg.starts_with("Manual:") {
                                manual += 1;
                                println!("   [{}] {}", item.code, msg);
                            } else {
                                failed += 1;
                                println!("   [{}] Failed: {}", item.code, msg);
                            }
                        }
                        Ok(None) => {
                            // No geometry extracted
                        }
                        Err(e) => {
                            failed += 1;
                            println!("   [{}] Error: {}", item.code, e);
                        }
                    }
                }

                println!("\n   Geometry extraction: {} extracted, {} manual, {} failed\n", extracted, manual, failed);

                // Cleanup
                let _ = fs::remove_dir_all(&tmpdir);
            }

            // Save intermediate metadata if requested
            if let Some(ref meta_path) = save_metadata {
                println!("3. Saving metadata to {:?}...", meta_path);
                if let Some(parent) = meta_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let metadata = serde_json::json!({
                    "generated_at": Utc::now().to_rfc3339(),
                    "stac_version": STAC_VERSION,
                    "items": items
                });
                fs::write(meta_path, serde_json::to_string_pretty(&metadata)?)?;
                println!("   Saved {} items\n", items.len());
            }

            let step = if data_dir.is_some() { "4" } else { "2" };
            println!("{}. Generating STAC catalog to {:?}...", step, output);
            let (num_collections, num_items, issues) = generate_catalog(&items, &output, &base_url)?;

            println!("\n=== Summary ===");
            println!("Collections: {}", num_collections);
            println!("Items: {}", num_items);
            println!("Items with geometry: {}", items.iter().filter(|i| i.geometry.is_some()).count());
            println!("Issues: {}", issues.len());

            if !issues.is_empty() {
                println!("\nIssues:");
                for issue in issues.iter().take(20) {
                    println!("  [{}] {}: {}", issue.severity, issue.item_id, issue.message);
                }
                if issues.len() > 20 {
                    println!("  ... and {} more", issues.len() - 20);
                }
            }
        }

        Commands::Validate { catalog_dir } => {
            println!("=== STAC Validator (v{}) ===\n", STAC_VERSION);
            println!("Validating {:?}...", catalog_dir);

            let issues = validate_catalog(&catalog_dir)?;

            println!("\n=== Validation Results ===");
            println!("Issues found: {}", issues.len());

            if !issues.is_empty() {
                for issue in issues.iter().take(30) {
                    println!("  - [{}] {}: {}", issue.severity, issue.item_id, issue.message);
                }
                if issues.len() > 30 {
                    println!("  ... and {} more", issues.len() - 30);
                }
            }

            std::process::exit(if issues.iter().any(|i| i.severity == "error") { 1 } else { 0 });
        }
    }

    Ok(())
}
