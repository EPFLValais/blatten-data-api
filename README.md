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
| `GET /stac/queryables` | Properties a `filter` may name, as a JSON Schema |
| `GET /stac/collections/{id}/queryables` | The same, for one collection |
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
- `filter`: Basic CQL2 filter on the queryables (`=`, `<>`, `<`, `<=`, `>`, `>=`, `IS NULL`, `AND`, `OR`, `NOT`), on search and collection items
- `filter-lang`: `cql2-text` (the GET default) or `cql2-json` (the POST default)

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

# Search with a CQL2 filter
curl "http://localhost:3000/stac/search" --get --data-urlencode "filter=blatten:source='Terradata' AND blatten:processing_level >= 3"
```

### Queryables and Filter

`GET /stac/queryables` (and `/stac/collections/{id}/queryables`) returns, as a JSON Schema,
the properties a `filter` may name. A filter naming anything else is answered 400.

| Queryable | Type |
|-----------|------|
| `blatten:series` | string: the series a day item belongs to, absent on every other item |
| `blatten:sensor` | string |
| `blatten:code` | string: the dataset code, also carried by its day items |
| `blatten:source` | string |
| `blatten:processing_level` | integer |
| `datetime`, `start_datetime`, `end_datetime` | date-time |

`filter` works on `/stac/search` and on `/stac/collections/{id}/items`, and the `next` and
`prev` links keep it. Pass it URL-encoded:

```bash
# The days of one series, a page at a time, in date order
curl "http://localhost:3000/stac/collections/webcam-image/items" --get \
  --data-urlencode "filter=blatten:series='07Aa00'"

# Every item that is not a day: one entry per dataset, as the collection was before the split
curl "http://localhost:3000/stac/collections/webcam-image/items" --get \
  --data-urlencode "filter=blatten:series IS NULL"

# The days of a series within a date range
curl "http://localhost:3000/stac/search" --get \
  --data-urlencode "filter=blatten:series='07Aa00' AND start_datetime >= TIMESTAMP('2025-06-01T00:00:00Z') AND end_datetime < TIMESTAMP('2025-06-03T00:00:00Z')"

# The same filter as cql2-json in a POST body
curl -X POST "http://localhost:3000/stac/search" -H "Content-Type: application/json" \
  -d '{"filter": {"op": "=", "args": [{"property": "blatten:series"}, "07Aa00"]}}'
```

### Series and Day Items

A dataset whose item would exceed 100 KB of JSON (long webcam and radar series, thousands of
files) is written as a series item and its day items:

- The **series item** keeps the dataset code as its id (e.g. `07Aa00`), its collection,
  geometry, properties and full date range. `blatten:file_count` is still the number of all
  its files. Its assets are the `archive`, the `manifest` and the files with no date in their
  name (such as station factsheets). A link `rel: "items"`, titled `Days`, lists its days:
  `/stac/collections/{collection}/items?filter=blatten:series='{code}'`.
- A **day item** holds the files of one date: id `{code}_{YYYY-MM-DD}` (e.g.
  `07Aa00_2025-06-01`), `datetime` null, `start_datetime` and `end_datetime` the first and
  last file time of the day, `blatten:series` the series id, `blatten:file_count` and
  `blatten:total_size` the number and total `file:size` of its files, and the series' geometry,
  `blatten:code`, sensor, source, processing level, format and product type. Its assets are
  the files of that day, with the same keys, hrefs, sizes and checksums they had on the whole
  item. A link `rel: "related"`, titled `Series`, points back to the series item. A day that
  alone exceeds the size is cut, in file-time order, into parts `{code}_{YYYY-MM-DD}_{n}`.

Items are listed in id order, so a series' days come after it, in date order. A `datetime`
search matches a day by its own time range, and a series by its whole range. Items under the size limit are unchanged and
have no `blatten:series`.

### Manifest

Each series item has a `manifest` asset (`text/csv`, href `/s3/archives/{code}.manifest.csv`,
beside the archive) listing every file of the dataset, its days' files included, one row per
file sorted by path:

| Column | Content |
|--------|---------|
| `path` | The file's S3 key under the `/s3` prefix, e.g. `assets/07Aa00/2025-06-01 0000.jpg`; percent-encode it to build the URL |
| `datetime` | The file's time, read from its name as on its asset, or empty |
| `size` | Size in bytes, as the asset's `file:size` |
| `checksum` | SHA-256 multihash in hex, as the asset's `file:checksum` |

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
| `STAC_ASSET_CONCURRENCY` | `4` | Search, collection items and item requests answered at once; more wait for a slot |
| `STAC_RESPONSE_DEADLINE` | `60` | Seconds one of those requests may hold its slot, building and sending together; a response not sent by then is cut off |
| `STAC_SLOT_WAIT` | `30` | Seconds a request may wait for a slot before it is answered 503 with `Retry-After` |
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

The run checks the catalog it writes:

- the catalog, every collection and every item against the STAC 1.1.0 core schemas and the
  schema of each extension they declare (an error per failure, with the object id and JSON
  path). The schemas are vendored under `schemas/`, so the check needs no network;
- an empty collection description (error) or item description (warning);
- a CSV Format that matches none of an item's data file extensions (warning);
- a CSV Phase that is not a number or a range like `1-3` (warning). A spreadsheet that
  turned the range into a date (`01. Mär`) is reported with the range it stood for;
- items whose data files have identical checksums (warning). A bundle `00` sharing the files
  of its own siblings, tile schemes and a lone boundary frame do not count.

`validation_passed` is false when any check reports an error.

## License

Dataset: CC-BY-NC-SA-4.0
Code: MIT
