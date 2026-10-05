# Blatten Data API

STAC API server for Blatten4Science: Observational Research Data of the 2025 Nesthorn – Birchgletscher Process Cascade.

## Overview

This API provides [STAC (SpatioTemporal Asset Catalog)](https://stacspec.org/) access to the 2025 Birch Glacier collapse dataset collected at Blatten, CH-VS. It serves metadata for webcam imagery, deformation analysis, orthophotos, DEMs, point clouds, GNSS data, and hydrological measurements.

## Quick Start

### Prerequisites

- Rust 1.75+
- For the catalog generator (`stac-gen`) only: GDAL and clang (for bindgen)
  - Arch: `sudo pacman -S gdal clang`
  - Debian/Ubuntu: `sudo apt install libgdal-dev libclang-dev`

### Running Locally

```bash
cargo build --release
STAC_BASE_URL=http://localhost:3000 ./target/release/blatten-api
```

The server reads the catalog in `stac/` at startup.

### Using Docker

```bash
docker compose up
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /stac` | Landing page |
| `GET /stac/conformance` | Conformance classes |
| `GET /stac/collections` | List all collections |
| `GET /stac/collections/{id}` | Get a specific collection |
| `GET /stac/collections/{id}/items` | List items in a collection |
| `GET /stac/collections/{id}/items/{item_id}` | Get a specific item |
| `GET /stac/search` | Search items (supports bbox, datetime, collections) |
| `POST /stac/search` | Search items (JSON body) |
| `GET /health` | Health check |

### Search Parameters

- `bbox`: Bounding box filter `west,south,east,north`
- `datetime`: Temporal filter (RFC 3339 interval)
- `collections`: Filter by collection IDs (comma-separated)
- `limit`: Maximum results (default: 100)
- `offset`: Pagination offset
- `source`: Filter by data provider (e.g., "Geoprevent", "Terradata")
- `processing_level`: Filter by processing level (1-4)

### Example Queries

```bash
# Get all collections
curl http://localhost:3000/stac/collections

# Get orthophoto items
curl http://localhost:3000/stac/collections/orthophoto/items

# Search by bounding box
curl "http://localhost:3000/stac/search?bbox=7.78,46.39,7.85,46.43"

# Search by provider
curl "http://localhost:3000/stac/search?source=Terradata"
```

## Data Pipeline

`stac-gen` builds the catalog in `stac/` and the files published on S3 from the data
delivered on SharePoint. It runs from a working folder that holds links to everything it
needs, under fixed names, so it takes no arguments:

```
working/
├── stac-gen              → blatten-data-api/target/release/stac-gen
├── config.yaml           → blatten-data-api/config.yaml
├── dataset_overview.csv  → SharePoint: dataset overview CSV (one row per item)
├── input/                → SharePoint: FINAL_Data/ (Terradata/, Geopraevent/, DNAGE/)
├── Documentation/        → SharePoint: Documentation/ (PDFs, Factsheets/)
├── stac/                 → blatten-data-api/stac/
└── output/               → staging folder uploaded to S3
```

| Name | Role |
|------|------|
| `dataset_overview.csv` | Item list and metadata (semicolon-separated, UTF-8 or Latin-1, with or without BOM) |
| `input/` | Data files, matched to item codes by folder name |
| `Documentation/` | `StationFactsheet_<NN>_*.pdf` in subfolders are added to every item of sensor `NN`; PDFs at the top level are published to `docs/` |
| `config.yaml` | Our corrections: excluded items, sensor coordinates, folder mappings, CRS overrides |
| `stac/` | Catalog written by the run, built into the API image |
| `output/` | `assets/<code>/`, `archives/<code>.zip` and `docs/`, mirroring the S3 prefix |

### Build

```bash
cargo build --release --features generator
```

After a system GDAL upgrade the build keeps linking the old library version, because the
GDAL bindings only rebuild when GDAL environment variables change. Clear them once:

```bash
cargo clean -p gdal-sys --release
cargo build --release --features generator
```

### Run

```bash
cd working
./stac-gen -y
```

`-y` skips the confirmation prompt. A run:

1. reads `dataset_overview.csv` and drops the items listed in `exclude_items`
2. matches folders in `input/` to item codes and extracts geometry
3. stages files into `output/assets/<code>/`, replacing staged files whose source has changed
4. adds the station factsheets to each item
5. hashes the files and rebuilds `output/archives/<code>.zip` for items whose content changed
6. copies `dataset_overview.csv` and the top-level PDFs of `Documentation/` to `output/docs/`
7. writes the catalog to `stac/` and a report to `stac/validation_report.json`

Staged files are converted from links to real copies at the end (skip with `--no-materialize`).

### Upload and deploy

`output/` is uploaded to the `dev/` prefix of the S3 bucket (`S3_BUCKET` in
`blatten-data-ui/.env`) with the `Blatten S3` rclone remote. `sync` also deletes files
under the prefix that are no longer in `output/`, so preview first:

```bash
rclone sync output/ "Blatten S3:<bucket>/dev/" --dry-run
rclone sync output/ "Blatten S3:<bucket>/dev/" --progress
```

Commit the regenerated `stac/` and push to `main`: the GitHub workflow builds the API
image with the catalog included. Semver tags (`*.*.*`) build release images.

## Collections

| Collection ID | Title |
|---------------|-------|
| `3d-model` | 3D Model |
| `deformation-analysis` | Deformation Analysis |
| `digital-surface-model-dsm` | Digital Surface Model |
| `digital-terrain-model-dtm` | Digital Terrain Model |
| `gps-data` | GNSS Data |
| `hydrology` | Hydrology |
| `orthophoto` | Orthophoto |
| `point-cloud` | Point Cloud |
| `radar` | Radar |
| `thermal-image` | Thermal Image |
| `webcam-image` | Webcam Image |

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `STAC_BASE_URL` | `http://localhost:3000` | Base URL for STAC links |
| `STAC_CATALOG_DIR` | `stac` | Directory containing STAC JSON |
| `PORT` | `3000` | Server port |
| `RUST_LOG` | `info` | Log level |

## Adding Manual Coordinates

Items without extractable geometry (webcams, GNSS stations, hydrology stations) take their
location from the `sensors:` section of `config.yaml`, in LV95 (EPSG:2056). `stac-gen`
converts them to WGS84:

```yaml
sensors:
  02-flexcam-birchbach:
    name: FlexCam Birchgletscher
    x: 2627304.17
    y: 1141251.56
    elevation_m: 2102
    items: [02Aa00, 02Aa01]
```

Then re-run `./stac-gen -y` from the working folder.

## Validation

Each run writes `stac/validation_report.json`, listing issues by severity (error, warning,
info) and statistics per collection. With `--validate`, `stac-gen` exits with status 1 when
there are errors.

## License

Dataset: CC-BY-NC-SA-4.0
Code: MIT
