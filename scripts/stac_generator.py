#!/usr/bin/env python3
"""
STAC Catalog Generator for Blatten Data API

This consolidated script replaces the previous multi-script pipeline:
- parse_xlsx_to_json.py
- extract_geometry.py
- generate_stac_catalog.py
- scan_assets.py (removed - was redundant)

Usage:
    # Full pipeline from Excel
    python stac_generator.py generate --xlsx metadata.xlsx --data-dir ./data/

    # Just regenerate STAC from existing metadata
    python stac_generator.py regenerate --metadata data/metadata_with_geometry.json

    # Validate existing catalog
    python stac_generator.py validate --catalog-dir ./stac
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Only import pandas when needed (for xlsx parsing)
pd = None

# =============================================================================
# Constants
# =============================================================================

STAC_VERSION = "1.1.0"

# STAC Extensions
EXTENSIONS = [
    "https://stac-extensions.github.io/scientific/v1.0.0/schema.json",
    "https://stac-extensions.github.io/timestamps/v1.2.0/schema.json",
    "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
]

# Product type code -> Collection definition
COLLECTIONS = {
    "A": {
        "id": "webcam-image",
        "title": "Webcam Images",
        "description": "Time-series webcam imagery from monitoring cameras",
    },
    "B": {
        "id": "deformation-analysis",
        "title": "Deformation Analysis",
        "description": "DEFOX deformation analysis imagery",
    },
    "D": {
        "id": "orthophoto",
        "title": "Orthophotos",
        "description": "Orthorectified aerial and drone imagery",
    },
    "H": {
        "id": "radar-velocity",
        "title": "Radar Velocity Data",
        "description": "Interferometric radar velocity measurements per ROI",
    },
    "I": {
        "id": "dsm",
        "title": "Digital Surface Models",
        "description": "Digital Surface Models (DSM) from drone and heliborne surveys",
    },
    "K": {
        "id": "point-cloud",
        "title": "Point Clouds",
        "description": "LiDAR and photogrammetric point cloud data",
    },
    "L": {
        "id": "3d-model",
        "title": "3D Models",
        "description": "3D visualization models for Sketchfab and similar platforms",
    },
    "M": {
        "id": "gnss-data",
        "title": "GNSS Position Data",
        "description": "High-precision GNSS/GPS position time series",
    },
    "N": {
        "id": "thermal-image",
        "title": "Thermal Imagery",
        "description": "Heliborne thermal infrared imagery",
    },
    "P": {
        "id": "hydrology",
        "title": "Hydrology Data",
        "description": "Lake level and volume measurements",
    },
}

# Collection-level archive files
COLLECTION_ARCHIVES = {
    "deformation-analysis": "04Ba00_DEFOX_all.zip",
    "radar-velocity": "06Ha00_Radar_Velocities_ROI.zip",
    "point-cloud": "14Ka04.zip",
    "3d-model": "14La02.zip",
    "gnss-data": "10M_11M_GNSS.zip",
    "thermal-image": "14Na01.zip",
    "hydrology": "17_LakeLevel_Geoazimut.zip",
}

# Item code -> Archive filename mapping
CODE_TO_FILE = {
    "02Ah00": "02Ah00_FlexCam_Birchgletscher_BirchbachChannel_SAMPLE.zip",
    "04Ba00": "04Ba00_DEFOX_all.zip",
    "04Ba01": "04Ba01_DEFOX_2to3_per_d.zip",
    "04Ba02": "04Ba02_DEFOX_1_per_d.zip",
    "06Ha00": "06Ha00_Radar_Velocities_ROI.zip",
    "08Aa00": "08Aa00_Webcam_Lonza_all.7z",
    "08Aa01": "08Aa01_Webcam_Lonza_1h.zip",
    "08Aa02": "08Aa02_Webcam_Lonza_30min.7z",
    "10Ma00": "10M_11M_GNSS.zip",
    "11Ma00": "10M_11M_GNSS.zip",
    "11Mb00": "10M_11M_GNSS.zip",
    "11Mc00": "10M_11M_GNSS.zip",
    "11Md00": "10M_11M_GNSS.zip",
    "11Me00": "10M_11M_GNSS.zip",
    "11Mf00": "10M_11M_GNSS.zip",
    "11Mg00": "10M_11M_GNSS.zip",
    "11Mh00": "10M_11M_GNSS.zip",
    "13Db06": "13Db06.7z",
    "13Ib06": "13Ib06.zip",
    "14Ia04": "14Ia04.zip",
    "14Ka04": "14Ka04.zip",
    "14La02": "14La02.zip",
    "14Na01": "14Na01.zip",
    "15Da01": "15Da01.zip",
    "17Pa00": "17_LakeLevel_Geoazimut.zip",
    "17Pb00": "17_LakeLevel_Geoazimut.zip",
}

# Default providers
DEFAULT_PROVIDERS = [
    {
        "name": "Canton du Valais",
        "roles": ["licensor", "host"],
        "url": "https://www.vs.ch/",
    },
    {
        "name": "EPFL ALPOLE",
        "roles": ["processor", "host"],
        "url": "https://www.epfl.ch/research/domains/alpole/",
    },
]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ItemMetadata:
    """Parsed metadata for a single item."""

    code: str
    product_id: str | None = None
    sensor_id: str | None = None
    dataset_id: str | None = None
    bundle_id: str | None = None
    sensor: str | None = None
    product_type: str | None = None
    dataset: str | None = None
    bundle: str | None = None
    description: str | None = None
    format: str | None = None
    technical_info: str | None = None
    processing_level: int | None = None
    phase: str | None = None
    date_first: str | None = None
    date_last: str | None = None
    continued: bool = False
    frequency: str | None = None
    location: str | None = None
    source: str | None = None
    additional_remarks: str | None = None
    storage_mb: float | None = None
    internal_commentary: str | None = None
    # Derived
    collection_id: str | None = None
    geometry: dict | None = None
    bbox: list[float] | None = None
    geo_metadata: dict = field(default_factory=dict)
    archive_file: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in self.__dict__.items() if v is not None or k in ("geometry", "bbox")}


@dataclass
class ValidationIssue:
    """A validation issue found in the catalog."""

    item_id: str
    severity: str  # "error", "warning", "info"
    message: str


# =============================================================================
# Utility Functions
# =============================================================================


def clean_string(val: Any) -> str | None:
    """Clean string value, returning None for NaN/empty."""
    if val is None:
        return None
    if pd is not None and pd.isna(val):
        return None
    s = str(val).strip()
    return s if s and s.lower() != "nan" else None


def parse_date(val: Any) -> str | None:
    """Convert various date formats to ISO 8601 date string."""
    if val is None:
        return None
    if pd is not None and pd.isna(val):
        return None
    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, str):
        return val.strip() if val.strip() else None
    return str(val)


def parse_float(val: Any) -> float | None:
    """Parse float value, returning None for NaN."""
    if val is None:
        return None
    if pd is not None and pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_int(val: Any) -> int | None:
    """Parse integer value, returning None for NaN."""
    if val is None:
        return None
    if pd is not None and pd.isna(val):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def parse_bool(val: Any) -> bool:
    """Parse boolean value from 'x' or similar."""
    if val is None:
        return False
    if pd is not None and pd.isna(val):
        return False
    return str(val).strip().lower() in ("x", "true", "yes", "1")


def bbox_to_polygon(bbox: list[float]) -> dict:
    """Convert bbox [west, south, east, north] to GeoJSON Polygon."""
    west, south, east, north = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
            ]
        ],
    }


def run_command(cmd: list[str], input_text: str | None = None, timeout: int = 60) -> tuple[bool, str]:
    """Run a command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, input=input_text
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


# =============================================================================
# Geometry Extraction
# =============================================================================


def transform_coords_to_wgs84(
    minx: float, miny: float, maxx: float, maxy: float, epsg: int = 2056
) -> list[float] | None:
    """Transform bbox from EPSG code to WGS84 using gdaltransform."""
    success, output = run_command(
        ["gdaltransform", "-s_srs", f"EPSG:{epsg}", "-t_srs", "EPSG:4326"],
        input_text=f"{minx} {miny}\n{maxx} {maxy}\n",
        timeout=30,
    )
    if not success:
        return None

    lines = output.strip().split("\n")
    if len(lines) < 2:
        return None

    try:
        p1 = [float(x) for x in lines[0].split()[:2]]
        p2 = [float(x) for x in lines[1].split()[:2]]
        return [min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1])]
    except (ValueError, IndexError):
        return None


def extract_geotiff_bounds(filepath: Path) -> dict | None:
    """Extract bounds from a GeoTIFF using gdalinfo."""
    success, output = run_command(["gdalinfo", "-json", str(filepath)])
    if not success:
        return None

    try:
        info = json.loads(output)
    except json.JSONDecodeError:
        return None

    result = {}

    # Get WGS84 bounds if available
    if "wgs84Extent" in info:
        extent = info["wgs84Extent"]
        coords = extent["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        result["bbox"] = [min(lons), min(lats), max(lons), max(lats)]
        result["geometry"] = extent
    elif "cornerCoordinates" in info:
        # Try to transform from native CRS
        corners = info["cornerCoordinates"]
        ul = corners.get("upperLeft", [0, 0])
        lr = corners.get("lowerRight", [0, 0])

        crs = info.get("coordinateSystem", {}).get("wkt", "")
        import re

        match = re.search(r"EPSG[\",:]+(\d+)", str(crs))
        if match:
            epsg = int(match.group(1))
            bbox = transform_coords_to_wgs84(ul[0], lr[1], lr[0], ul[1], epsg)
            if bbox:
                result["bbox"] = bbox
                result["geometry"] = bbox_to_polygon(bbox)

    if "bbox" not in result:
        return None

    # Add metadata
    result["crs"] = info.get("coordinateSystem", {}).get("wkt", "Unknown")
    size = info.get("size", [0, 0])
    result["size"] = [size[0], size[1]]
    result["bands"] = len(info.get("bands", []))

    return result


def extract_pointcloud_bounds(filepath: Path) -> dict | None:
    """Extract bounds from a LAZ/LAS file using pdal info."""
    success, output = run_command(["pdal", "info", "--summary", str(filepath)], timeout=120)
    if not success:
        return None

    try:
        info = json.loads(output)
    except json.JSONDecodeError:
        return None

    summary = info.get("summary", {})
    bounds = summary.get("bounds", {})

    if not bounds:
        return None

    minx = bounds.get("minx", 0)
    miny = bounds.get("miny", 0)
    maxx = bounds.get("maxx", 0)
    maxy = bounds.get("maxy", 0)

    # Assume Swiss LV95 (EPSG:2056) if coordinates look Swiss
    if minx > 2000000 and miny > 1000000:
        bbox = transform_coords_to_wgs84(minx, miny, maxx, maxy, 2056)
    else:
        bbox = [minx, miny, maxx, maxy]

    if not bbox:
        return None

    return {
        "bbox": bbox,
        "geometry": bbox_to_polygon(bbox),
        "point_count": summary.get("num_points", 0),
        "crs": summary.get("srs", {}).get("wkt", "Assumed EPSG:2056"),
    }


def extract_from_archive(archive_path: Path, temp_dir: Path) -> dict | None:
    """Extract geometry from files within an archive."""
    if archive_path.suffix == ".zip":
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                geo_files = [
                    f
                    for f in zf.namelist()
                    if f.upper().endswith((".TIF", ".TIFF", ".LAZ", ".LAS"))
                    and not f.startswith("__MACOSX")
                ]
                if not geo_files:
                    return None

                target = geo_files[0]
                zf.extract(target, temp_dir)
                extracted = temp_dir / target

                if target.upper().endswith((".TIF", ".TIFF")):
                    return extract_geotiff_bounds(extracted)
                else:
                    return extract_pointcloud_bounds(extracted)
        except zipfile.BadZipFile:
            return None

    elif archive_path.suffix == ".7z":
        # List contents
        success, output = run_command(["7z", "l", str(archive_path)])
        if not success:
            return None

        geo_files = []
        for line in output.split("\n"):
            for ext in [".tif", ".tiff", ".laz", ".las"]:
                if ext in line.lower():
                    parts = line.split()
                    if parts:
                        geo_files.append(parts[-1])
                    break

        if not geo_files:
            return None

        target = geo_files[0]
        success, _ = run_command(
            ["7z", "e", str(archive_path), target, f"-o{temp_dir}", "-y"], timeout=300
        )
        if not success:
            return None

        extracted = temp_dir / Path(target).name
        if target.lower().endswith((".tif", ".tiff")):
            return extract_geotiff_bounds(extracted)
        else:
            return extract_pointcloud_bounds(extracted)

    return None


# =============================================================================
# Excel Parsing
# =============================================================================


def parse_xlsx(xlsx_path: Path) -> list[ItemMetadata]:
    """Parse the Excel metadata file."""
    global pd
    import pandas

    pd = pandas

    xlsx = pd.read_excel(xlsx_path, sheet_name=None)

    if "Test_Data" not in xlsx:
        raise ValueError("'Test_Data' sheet not found in XLSX")

    df = xlsx["Test_Data"]
    items = []

    for idx, row in df.iterrows():
        if idx < 4:  # Skip header rows
            continue

        code = clean_string(row.iloc[1])
        if not code:
            continue

        product_id = clean_string(row.iloc[2])

        item = ItemMetadata(
            code=code,
            product_id=product_id,
            sensor_id=clean_string(row.iloc[3]),
            dataset_id=clean_string(row.iloc[4]),
            bundle_id=clean_string(row.iloc[5]),
            sensor=clean_string(row.iloc[6]),
            product_type=clean_string(row.iloc[7]),
            dataset=clean_string(row.iloc[8]),
            bundle=clean_string(row.iloc[9]),
            description=clean_string(row.iloc[10]),
            format=clean_string(row.iloc[11]),
            technical_info=clean_string(row.iloc[12]),
            processing_level=parse_int(row.iloc[13]),
            phase=clean_string(row.iloc[14]),
            date_first=parse_date(row.iloc[15]),
            date_last=parse_date(row.iloc[16]),
            continued=parse_bool(row.iloc[17]),
            frequency=clean_string(row.iloc[18]),
            location=clean_string(row.iloc[19]),
            source=clean_string(row.iloc[20]),
            additional_remarks=clean_string(row.iloc[21]),
            storage_mb=parse_float(row.iloc[22]),
            internal_commentary=clean_string(row.iloc[23]),
        )

        # Map product_id to collection
        if product_id and product_id in COLLECTIONS:
            item.collection_id = COLLECTIONS[product_id]["id"]

        items.append(item)

    return items


# =============================================================================
# Geometry Extraction Pipeline
# =============================================================================


def extract_geometries(
    items: list[ItemMetadata],
    data_dir: Path,
    sensor_locations: dict | None = None,
) -> dict:
    """Extract geometries for all items."""
    results = {"extracted": [], "manual": [], "failed": [], "no_file": []}
    sensor_locations = sensor_locations or {}

    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)

        for item in items:
            code = item.code
            data_format = (item.format or "").upper()
            archive_name = CODE_TO_FILE.get(code)

            if not archive_name:
                results["no_file"].append(code)
                continue

            archive_path = data_dir / archive_name
            if not archive_path.exists():
                archive_path = data_dir / "dummy" / archive_name
                if not archive_path.exists():
                    results["no_file"].append(code)
                    continue

            item.archive_file = archive_name

            # Try extracting from geospatial files
            if data_format in ("TIF", "LAZ", "LAS"):
                geo_info = extract_from_archive(archive_path, temp_dir)
                if geo_info:
                    item.bbox = geo_info["bbox"]
                    item.geometry = geo_info["geometry"]
                    item.geo_metadata = {
                        k: v for k, v in geo_info.items() if k not in ("bbox", "geometry")
                    }
                    results["extracted"].append(code)
                    continue

            # Try manual coordinates
            if code in sensor_locations:
                sensor = sensor_locations[code]
                if sensor.get("lon") is not None and sensor.get("lat") is not None:
                    lon, lat = sensor["lon"], sensor["lat"]
                    buffer = 0.001  # ~100m
                    item.bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]
                    item.geometry = {"type": "Point", "coordinates": [lon, lat]}
                    if sensor.get("elevation_m"):
                        item.geometry["coordinates"].append(sensor["elevation_m"])
                    results["manual"].append(code)
                    continue

            results["failed"].append(code)

    return results


# =============================================================================
# STAC Generation
# =============================================================================


def create_stac_collection(
    collection_def: dict, items: list[ItemMetadata], base_url: str
) -> dict:
    """Create a STAC collection."""
    collection_id = collection_def["id"]

    # Compute spatial extent
    bboxes = [item.bbox for item in items if item.bbox]
    if bboxes:
        spatial_bbox = [
            [
                min(b[0] for b in bboxes),
                min(b[1] for b in bboxes),
                max(b[2] for b in bboxes),
                max(b[3] for b in bboxes),
            ]
        ]
    else:
        spatial_bbox = [[None, None, None, None]]

    # Compute temporal extent
    dates = []
    for item in items:
        if item.date_first:
            dates.append(item.date_first)
        if item.date_last:
            dates.append(item.date_last)

    if dates:
        dates.sort()
        temporal_interval = [[f"{dates[0]}T00:00:00Z", f"{dates[-1]}T23:59:59Z"]]
    else:
        temporal_interval = [[None, None]]

    collection = {
        "type": "Collection",
        "id": collection_id,
        "stac_version": STAC_VERSION,
        "stac_extensions": EXTENSIONS,
        "title": collection_def["title"],
        "description": collection_def["description"],
        "license": "CC-BY-NC-SA-4.0",
        "keywords": [
            "Birch Glacier",
            "Blatten",
            "Switzerland",
            "glacier collapse",
            "landslide",
            collection_def["title"].lower(),
        ],
        "providers": DEFAULT_PROVIDERS,
        "extent": {
            "spatial": {"bbox": spatial_bbox},
            "temporal": {"interval": temporal_interval},
        },
        "summaries": {
            "processing_level": list(
                set(item.processing_level for item in items if item.processing_level is not None)
            ),
            "source": list(set(item.source for item in items if item.source)),
        },
        "links": [
            {
                "rel": "self",
                "href": f"{base_url}/stac/collections/{collection_id}",
                "type": "application/json",
            },
            {
                "rel": "root",
                "href": f"{base_url}/stac/catalog.json",
                "type": "application/json",
            },
            {
                "rel": "parent",
                "href": f"{base_url}/stac/catalog.json",
                "type": "application/json",
            },
            {
                "rel": "items",
                "href": f"{base_url}/stac/collections/{collection_id}/items",
                "type": "application/geo+json",
            },
        ],
        "assets": {},
    }

    # Add collection-level archive
    archive = COLLECTION_ARCHIVES.get(collection_id)
    if archive:
        collection["assets"]["archive"] = {
            "href": f"{base_url}/s3/data/{archive}",
            "type": "application/zip",
            "title": f"Full {collection_def['title']} Archive",
            "roles": ["data"],
        }

    return collection


def create_stac_item(item: ItemMetadata, base_url: str) -> dict:
    """Create a STAC item."""
    code = item.code
    collection_id = item.collection_id or "unknown"

    # Build datetime
    if item.date_first and item.date_last:
        datetime_str = None
        start_datetime = f"{item.date_first}T00:00:00Z"
        end_datetime = f"{item.date_last}T23:59:59Z"
    elif item.date_first:
        datetime_str = f"{item.date_first}T00:00:00Z"
        start_datetime = None
        end_datetime = None
    else:
        datetime_str = None
        start_datetime = None
        end_datetime = None

    # Build properties
    title = f"{item.sensor or ''} - {item.dataset or ''}".strip(" -") or code
    properties = {
        "title": title,
        "description": item.description,
        "datetime": datetime_str,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "blatten:code": code,
        "blatten:sensor": item.sensor,
        "blatten:product_type": item.product_type,
        "blatten:dataset": item.dataset,
        "blatten:bundle": item.bundle,
        "blatten:source": item.source,
        "blatten:processing_level": item.processing_level,
        "blatten:phase": item.phase,
        "blatten:frequency": item.frequency,
        "blatten:continued": item.continued,
        "blatten:format": item.format,
        "blatten:storage_mb": item.storage_mb,
    }

    # Add projection info
    if item.geo_metadata.get("crs"):
        properties["proj:epsg"] = 2056
        if item.geo_metadata.get("size"):
            properties["proj:shape"] = item.geo_metadata["size"]

    # Remove None values
    properties = {k: v for k, v in properties.items() if v is not None}

    # Build assets
    assets = {}
    if item.archive_file:
        if item.archive_file.endswith(".zip"):
            media_type = "application/zip"
        elif item.archive_file.endswith(".7z"):
            media_type = "application/x-7z-compressed"
        else:
            media_type = "application/octet-stream"

        assets["data"] = {
            "href": f"{base_url}/s3/data/{item.archive_file}",
            "type": media_type,
            "title": f"{code} Data Archive",
            "roles": ["data"],
            "file:size": int((item.storage_mb or 0) * 1024 * 1024),
        }

    # Build links
    links = [
        {
            "rel": "self",
            "href": f"{base_url}/stac/collections/{collection_id}/items/{code}",
            "type": "application/geo+json",
        },
        {
            "rel": "parent",
            "href": f"{base_url}/stac/collections/{collection_id}",
            "type": "application/json",
        },
        {
            "rel": "collection",
            "href": f"{base_url}/stac/collections/{collection_id}",
            "type": "application/json",
        },
        {
            "rel": "root",
            "href": f"{base_url}/stac/catalog.json",
            "type": "application/json",
        },
    ]

    return {
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "stac_extensions": EXTENSIONS,
        "id": code,
        "geometry": item.geometry,
        "bbox": item.bbox,
        "properties": properties,
        "links": links,
        "assets": assets,
        "collection": collection_id,
    }


def generate_stac_catalog(
    items: list[ItemMetadata], output_dir: Path, base_url: str = "${STAC_BASE_URL}"
) -> tuple[int, int, list[ValidationIssue]]:
    """Generate complete STAC catalog."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "collections").mkdir(exist_ok=True)

    # Group items by collection
    items_by_collection: dict[str, list[ItemMetadata]] = {}
    for item in items:
        if item.collection_id:
            items_by_collection.setdefault(item.collection_id, []).append(item)

    collections = []
    all_items = []
    issues = []

    for product_code, coll_def in COLLECTIONS.items():
        coll_id = coll_def["id"]
        coll_items = items_by_collection.get(coll_id, [])

        if not coll_items:
            continue

        # Create STAC items
        stac_items = []
        for item_meta in coll_items:
            stac_item = create_stac_item(item_meta, base_url)
            stac_items.append(stac_item)
            all_items.append(stac_item)

            # Validate
            if stac_item.get("geometry") is None:
                issues.append(
                    ValidationIssue(item_meta.code, "warning", "Missing geometry")
                )
            if not stac_item.get("assets"):
                issues.append(
                    ValidationIssue(item_meta.code, "warning", "No assets defined")
                )

        # Create collection
        collection = create_stac_collection(coll_def, coll_items, base_url)
        collections.append(collection)

        # Write collection file
        with open(output_dir / "collections" / f"{coll_id}.json", "w") as f:
            json.dump(collection, f, indent=2)

        # Write collection items
        with open(output_dir / "collections" / f"{coll_id}_items.json", "w") as f:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "features": stac_items,
                    "numberMatched": len(stac_items),
                    "numberReturned": len(stac_items),
                },
                f,
                indent=2,
            )

    # Create root catalog
    catalog = {
        "type": "Catalog",
        "id": "birch-glacier-collapse",
        "stac_version": STAC_VERSION,
        "title": "Birch Glacier Collapse and Landslide Dataset",
        "description": "Dataset collected during the 2025 Birch glacier collapse and landslide at Blatten, CH-VS",
        "links": [
            {"rel": "self", "href": f"{base_url}/stac/catalog.json", "type": "application/json"},
            {"rel": "root", "href": f"{base_url}/stac/catalog.json", "type": "application/json"},
        ]
        + [
            {
                "rel": "child",
                "href": f"{base_url}/stac/collections/{c['id']}",
                "type": "application/json",
                "title": c["title"],
            }
            for c in collections
        ],
        "conformsTo": [
            "https://api.stacspec.org/v1.0.0/core",
            "https://api.stacspec.org/v1.0.0/collections",
            "https://api.stacspec.org/v1.0.0/item-search",
        ],
    }

    # Write catalog
    with open(output_dir / "catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)

    # Write all items
    with open(output_dir / "all_items.json", "w") as f:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": all_items,
                "numberMatched": len(all_items),
                "numberReturned": len(all_items),
            },
            f,
            indent=2,
        )

    # Write collections list
    with open(output_dir / "collections.json", "w") as f:
        json.dump(
            {
                "collections": collections,
                "links": [
                    {"rel": "self", "href": f"{base_url}/stac/collections"},
                    {"rel": "root", "href": f"{base_url}/stac/catalog.json"},
                ],
            },
            f,
            indent=2,
        )

    # Write validation report
    with open(output_dir / "validation_report.json", "w") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "stac_version": STAC_VERSION,
                "total_collections": len(collections),
                "total_items": len(all_items),
                "items_with_geometry": sum(1 for i in all_items if i.get("geometry")),
                "items_missing_geometry": sum(1 for i in all_items if not i.get("geometry")),
                "issues": [{"item": i.item_id, "severity": i.severity, "message": i.message} for i in issues],
                "validation_passed": not any(i.severity == "error" for i in issues),
            },
            f,
            indent=2,
        )

    return len(collections), len(all_items), issues


# =============================================================================
# CLI Commands
# =============================================================================


def cmd_generate(args):
    """Full pipeline: XLSX -> metadata -> geometry -> STAC."""
    print(f"=== STAC Generator (v{STAC_VERSION}) ===\n")

    # Parse XLSX
    print(f"1. Parsing {args.xlsx}...")
    items = parse_xlsx(args.xlsx)
    print(f"   Parsed {len(items)} items\n")

    # Load sensor locations if available
    sensor_locations = {}
    if args.sensors and args.sensors.exists():
        print(f"2. Loading sensor locations from {args.sensors}...")
        with open(args.sensors) as f:
            data = json.load(f)
        for sensor_id, sensor in data.get("sensors", {}).items():
            for item_code in sensor.get("items", []):
                sensor_locations[item_code] = sensor
        print(f"   Loaded {len(sensor_locations)} sensor mappings\n")
    else:
        print("2. No sensor locations file, skipping manual coordinates\n")

    # Extract geometries
    if args.data_dir:
        print(f"3. Extracting geometries from {args.data_dir}...")
        results = extract_geometries(items, args.data_dir, sensor_locations)
        print(f"   Extracted: {len(results['extracted'])}")
        print(f"   Manual: {len(results['manual'])}")
        print(f"   Failed: {len(results['failed'])}")
        print(f"   No file: {len(results['no_file'])}\n")
    else:
        print("3. No data directory, skipping geometry extraction\n")

    # Save intermediate metadata
    if args.save_metadata:
        print(f"4. Saving metadata to {args.save_metadata}...")
        args.save_metadata.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_metadata, "w") as f:
            json.dump(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source_file": str(args.xlsx),
                    "stac_version": STAC_VERSION,
                    "items": [item.to_dict() for item in items],
                    "collections": list(COLLECTIONS.values()),
                },
                f,
                indent=2,
            )
        print(f"   Saved {len(items)} items\n")
    else:
        print("4. Skipping metadata save\n")

    # Generate STAC
    print(f"5. Generating STAC catalog to {args.output}...")
    num_collections, num_items, issues = generate_stac_catalog(
        items, args.output, args.base_url
    )

    print(f"\n=== Summary ===")
    print(f"Collections: {num_collections}")
    print(f"Items: {num_items}")
    print(f"Issues: {len(issues)}")

    if issues:
        print("\nIssues:")
        for issue in issues[:20]:
            print(f"  [{issue.severity}] {issue.item_id}: {issue.message}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")


def cmd_regenerate(args):
    """Regenerate STAC from existing metadata JSON."""
    print(f"=== STAC Regenerator (v{STAC_VERSION}) ===\n")

    print(f"Loading metadata from {args.metadata}...")
    with open(args.metadata) as f:
        metadata = json.load(f)

    items = []
    for item_dict in metadata.get("items", []):
        item = ItemMetadata(code=item_dict["code"])
        for key, value in item_dict.items():
            if hasattr(item, key):
                setattr(item, key, value)
        items.append(item)

    print(f"Loaded {len(items)} items\n")

    print(f"Generating STAC catalog to {args.output}...")
    num_collections, num_items, issues = generate_stac_catalog(
        items, args.output, args.base_url
    )

    print(f"\n=== Summary ===")
    print(f"Collections: {num_collections}")
    print(f"Items: {num_items}")
    print(f"Issues: {len(issues)}")


def cmd_validate(args):
    """Validate an existing STAC catalog."""
    print(f"=== STAC Validator (v{STAC_VERSION}) ===\n")

    catalog_path = args.catalog_dir / "catalog.json"
    if not catalog_path.exists():
        print(f"Error: {catalog_path} not found")
        sys.exit(1)

    print(f"Validating {args.catalog_dir}...")

    issues = []

    # Check catalog
    with open(catalog_path) as f:
        catalog = json.load(f)

    if catalog.get("stac_version") != STAC_VERSION:
        issues.append(f"Catalog version {catalog.get('stac_version')} != {STAC_VERSION}")

    # Check all_items
    all_items_path = args.catalog_dir / "all_items.json"
    if all_items_path.exists():
        with open(all_items_path) as f:
            data = json.load(f)
        items = data.get("features", [])
        print(f"Found {len(items)} items")

        for item in items:
            item_id = item.get("id", "unknown")
            if item.get("geometry") is None:
                issues.append(f"{item_id}: Missing geometry")
            if not item.get("assets"):
                issues.append(f"{item_id}: No assets")
            if item.get("stac_version") != STAC_VERSION:
                issues.append(f"{item_id}: Wrong STAC version")

    print(f"\n=== Validation Results ===")
    print(f"Issues found: {len(issues)}")

    if issues:
        for issue in issues[:30]:
            print(f"  - {issue}")
        if len(issues) > 30:
            print(f"  ... and {len(issues) - 30} more")

    sys.exit(0 if not issues else 1)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="STAC Catalog Generator for Blatten Data API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate command
    gen = subparsers.add_parser("generate", help="Full pipeline from XLSX")
    gen.add_argument("--xlsx", type=Path, required=True, help="Input XLSX file")
    gen.add_argument("--data-dir", type=Path, help="Directory with data archives")
    gen.add_argument("--sensors", type=Path, default=Path("data/sensor_locations.json"))
    gen.add_argument("--save-metadata", type=Path, help="Save intermediate metadata JSON")
    gen.add_argument("--output", type=Path, default=Path("stac"), help="Output directory")
    gen.add_argument("--base-url", default="${STAC_BASE_URL}", help="Base URL for links")
    gen.set_defaults(func=cmd_generate)

    # Regenerate command
    regen = subparsers.add_parser("regenerate", help="Regenerate from metadata JSON")
    regen.add_argument("--metadata", type=Path, required=True, help="Input metadata JSON")
    regen.add_argument("--output", type=Path, default=Path("stac"), help="Output directory")
    regen.add_argument("--base-url", default="${STAC_BASE_URL}", help="Base URL for links")
    regen.set_defaults(func=cmd_regenerate)

    # Validate command
    val = subparsers.add_parser("validate", help="Validate existing catalog")
    val.add_argument("--catalog-dir", type=Path, default=Path("stac"), help="Catalog directory")
    val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
