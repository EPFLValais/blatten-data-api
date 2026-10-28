# Changelog

## 1.0.0

### Changed

- Items whose JSON would exceed 100 KB are now **series items**. A series item keeps its id,
  collection, geometry, properties and date range, but its assets are only the `archive`, the
  new `manifest` and the files with no date in their name. Its dated files moved to **day
  items**, one per date (`{code}_{YYYY-MM-DD}`, or `{code}_{YYYY-MM-DD}_{n}` when one day is
  cut into parts), each with `blatten:series` set to the series id. See the README's "Series
  and Day Items".
- File hrefs and archive paths are unchanged: every file keeps its asset key, href, size and
  checksum, and `archives/{code}.zip` is where it was.

### Added

- STAC API Filter: `filter` (Basic CQL2, `cql2-text` or `cql2-json`) on `/stac/search` and
  `/stac/collections/{id}/items`, over the properties listed at `/stac/queryables`.
- A `manifest` asset on each series item, `archives/{code}.manifest.csv`, listing every file
  of the dataset with its `path`, `datetime`, `size` and `checksum`.
- `blatten:file_count` and `blatten:total_size` on each day item: the number and total
  `file:size` of its own files.

### Migrating scripts

A script that reads every file of a dataset from its item's `assets` now gets, for a series,
only the archive, the manifest and the undated files. Read the manifest instead, or page the
series' `items` link.

Read the manifest:

```python
import csv, io, urllib.parse, requests

item = requests.get("https://blatten-data.epfl.ch/stac/collections/webcam-image/items/07Aa00").json()
manifest = requests.get(item["assets"]["manifest"]["href"]).text
for row in csv.DictReader(io.StringIO(manifest)):
    url = "https://blatten-data.epfl.ch/s3/" + urllib.parse.quote(row["path"])
    print(url, row["datetime"], row["size"])
```

Page the days:

```python
import requests

item = requests.get("https://blatten-data.epfl.ch/stac/collections/webcam-image/items/07Aa00").json()
url = next(link["href"] for link in item["links"] if link["rel"] == "items")
while url:
    page = requests.get(url).json()
    for day in page["features"]:
        for asset in day["assets"].values():
            print(asset["href"])
    url = next((link["href"] for link in page["links"] if link["rel"] == "next"), None)
```

A script that lists a collection's items and wants one entry per dataset adds
`filter=blatten:series IS NULL`.
