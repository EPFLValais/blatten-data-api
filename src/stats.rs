//! Download statistics: request and byte totals per file and day, read from the file proxy's
//! Prometheus counters. This module is the pure logic; `stats_store` keeps the totals in Postgres.
//!
//! The proxy's counters are cumulative and reset when it restarts, so each scrape is turned into an
//! increment against the last value saved for the same path. A request is a successful GET
//! (status 200 or 206); HEAD requests and error responses are not counted.
//!
//! The first scrape into a store that has never taken one is a baseline: its readings are saved and
//! nothing is counted, so the proxy's uptime before the store existed is not booked to that day.
//! Counting starts with the next scrape. A proxy process first seen after that still counts from zero.

use std::collections::{BTreeMap, HashMap};

// ============================================================================
// Reading the proxy's counters
// ============================================================================

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct Totals {
    pub requests: u64,
    pub bytes: u64,
    /// Bytes sent divided by the file's size, in thousandths: ten ranged pieces that together
    /// make up one file are one whole file, and a transfer cut off halfway is half of one.
    pub equiv: u64,
}

impl Totals {
    pub fn add(&mut self, other: Totals) {
        self.requests = self.requests.saturating_add(other.requests);
        self.bytes = self.bytes.saturating_add(other.bytes);
        self.equiv = self.equiv.saturating_add(other.equiv);
    }

    pub fn whole_files(&self) -> f64 {
        (self.equiv as f64 / 100.0).round() / 10.0
    }
}

/// The longest path the database stores; a longer one would fail the whole scrape.
pub const MAX_PATH_CHARS: usize = 2048;

/// The longest item code the database stores.
pub const MAX_CODE_CHARS: usize = 64;

/// The largest counter the database stores (a Postgres `bigint`).
pub const MAX_COUNTER: u64 = i64::MAX as u64;

/// A counter value from the metrics text, or `None` when it is not a number the database can store:
/// negative, not finite or above `MAX_COUNTER`.
fn counter(text: &str) -> Option<u64> {
    let v = text.trim().parse::<f64>().ok()?;
    (v.is_finite() && v >= 0.0 && v <= MAX_COUNTER as f64).then(|| (v as u64).min(MAX_COUNTER))
}

/// One scrape of the proxy: totals per object path since the proxy started.
#[derive(Debug, Default, PartialEq)]
pub struct Scrape {
    /// `process_start_time_seconds`, which tells one proxy process from another.
    pub start_time: Option<u64>,
    pub paths: BTreeMap<String, Totals>,
}

/// Parse the proxy's metrics text. Series are summed over host and status: only GET requests with
/// status 200 or 206 on the business server count.
pub fn parse_scrape(text: &str) -> Scrape {
    let mut scrape = Scrape::default();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("process_start_time_seconds ") {
            scrape.start_time = counter(rest);
            continue;
        }
        let (is_requests, rest) = if let Some(r) = line.strip_prefix("http_requests_total{") {
            (true, r)
        } else if let Some(r) = line.strip_prefix("http_response_size_bytes_sum{") {
            (false, r)
        } else {
            continue;
        };
        let Some((labels, value)) = split_labels(rest) else {
            continue;
        };
        let get = |name: &str| labels.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str());
        if get("method") != Some("GET")
            || get("server") != Some("business")
            || !matches!(get("status_code"), Some("200") | Some("206"))
        {
            continue;
        }
        let Some(path) = get("path") else { continue };
        // A directory listing is not a download, and a path the database cannot store is skipped so
        // that one series never stops the rest from being counted.
        if path.is_empty() || path.ends_with('/') || path.chars().count() > MAX_PATH_CHARS {
            continue;
        }
        let Some(value) = counter(value) else {
            continue;
        };
        let totals = scrape.paths.entry(path.to_string()).or_default();
        let sum = if is_requests { &mut totals.requests } else { &mut totals.bytes };
        *sum = sum.saturating_add(value).min(MAX_COUNTER);
    }
    scrape
}

/// Split `a="1",b="2"} 42` into its labels and the value text. Label values may contain escaped
/// quotes and backslashes.
fn split_labels(s: &str) -> Option<(Vec<(String, String)>, &str)> {
    let mut labels = Vec::new();
    let mut chars = s.char_indices().peekable();
    loop {
        let mut name = String::new();
        while let Some(&(_, c)) = chars.peek() {
            if c == '=' || c == '}' {
                break;
            }
            if c != ',' {
                name.push(c);
            }
            chars.next();
        }
        match chars.next()? {
            (i, '}') => return Some((labels, &s[i + 1..])),
            _ => {}
        }
        if chars.next()?.1 != '"' {
            return None;
        }
        let mut value = String::new();
        loop {
            match chars.next()?.1 {
                '\\' => match chars.next()?.1 {
                    'n' => value.push('\n'),
                    c => value.push(c),
                },
                '"' => break,
                c => value.push(c),
            }
        }
        labels.push((name, value));
    }
}

/// What to add to the stored total when a counter of one proxy process reads `now` and that
/// process's last saved reading was `last`. A process's counters never go down, so a lower reading
/// is an older one posted late and adds nothing; a process with no saved reading adds it whole.
pub fn increment(last: Option<u64>, now: u64) -> u64 {
    match last {
        Some(last) => now.saturating_sub(last),
        None => now,
    }
}

/// What one scrape adds to the stored totals, and the readings to remember for the next one.
#[derive(Debug, Default, PartialEq)]
pub struct Changes {
    /// Requests, bytes and whole-file thousandths to add, per path; only paths that add something.
    pub added: BTreeMap<String, Totals>,
    /// The (requests, bytes) reading to save per path; only paths whose reading changed.
    pub save: BTreeMap<String, (u64, u64)>,
}

/// Turn a scrape into increments against one proxy process's last saved (requests, bytes) reading
/// per path. A reading no higher than the saved one in both counters is that process's older
/// reading posted late: it adds nothing and is not saved. A `baseline` scrape adds nothing and saves
/// every reading.
pub fn changes(
    scrape: &Scrape,
    saved: &HashMap<String, (u64, u64)>,
    sizes: &HashMap<String, u64>,
    baseline: bool,
) -> Changes {
    let mut out = Changes::default();
    for (path, now) in &scrape.paths {
        let last = saved.get(path).copied();
        if last.is_some_and(|(r, b)| now.requests <= r && now.bytes <= b) {
            continue;
        }
        let requests = increment(last.map(|l| l.0), now.requests);
        let bytes = increment(last.map(|l| l.1), now.bytes);
        out.save.insert(path.clone(), (now.requests, now.bytes));
        if !baseline && (requests > 0 || bytes > 0) {
            let equiv = whole_file_thousandths(bytes, sizes.get(path.as_str()).copied());
            out.added.insert(path.clone(), Totals { requests, bytes, equiv });
        }
    }
    out
}

// ============================================================================
// Mapping a path to an item
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    File,
    Archive,
    Docs,
    Other,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::File => "file",
            Kind::Archive => "archive",
            Kind::Docs => "docs",
            Kind::Other => "other",
        }
    }
}

/// Classify an object path as the proxy sees it (the ingress has already removed `/s3`):
/// `/assets/<code>/<file>`, `/archives/<code>.zip` or `/docs/<file>`.
pub fn classify(path: &str) -> (Kind, Option<&str>) {
    let path = path.trim_start_matches('/');
    let mut parts = path.splitn(3, '/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some("assets"), Some(code), Some(_)) => (Kind::File, Some(code)),
        (Some("archives"), Some(file), None) => match file.strip_suffix(".zip") {
            Some(code) => (Kind::Archive, Some(code)),
            None => (Kind::Other, None),
        },
        (Some("docs"), Some(_), _) => (Kind::Docs, None),
        _ => (Kind::Other, None),
    }
}

/// Bytes sent as thousandths of the file's size; zero when the size is not known.
pub fn whole_file_thousandths(bytes: u64, size: Option<u64>) -> u64 {
    match size {
        Some(size) if size > 0 => (bytes as u128 * 1000 / size as u128) as u64,
        _ => 0,
    }
}

/// The key a path is summed under per day: the item code for a catalog file or archive, `~docs` for a
/// document and `~other` for anything else.
pub fn code_key(path: &str) -> String {
    match classify(path) {
        (Kind::File | Kind::Archive, Some(code)) if code.chars().count() <= MAX_CODE_CHARS => code.to_string(),
        (Kind::Docs, _) => "~docs".to_string(),
        _ => "~other".to_string(),
    }
}

// ============================================================================
// Rows
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DailyRow {
    pub day: String,
    pub path: String,
    pub totals: Totals,
}

// ============================================================================
// Summary and CSV
// ============================================================================

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SummaryRow {
    pub day: String,
    pub collection: String,
    pub item: String,
    pub kind: &'static str,
    pub requests: u64,
    pub bytes: u64,
    pub whole_files: f64,
}

/// Group daily rows by day, collection, item and kind. `collections` maps an item code to its
/// collection; a code not in it is counted under `unknown`, so nothing taken from a request path
/// reaches the output except a known code.
pub fn summarize(rows: &[DailyRow], collections: &HashMap<String, String>) -> Vec<SummaryRow> {
    let mut grouped: BTreeMap<(String, String, String, Kind), Totals> = BTreeMap::new();
    for row in rows {
        let (kind, code) = classify(&row.path);
        let (collection, item) = match code.and_then(|c| collections.get(c).map(|col| (col, c))) {
            Some((col, c)) => (col.clone(), c.to_string()),
            None if kind == Kind::File || kind == Kind::Archive => ("unknown".into(), "unknown".into()),
            None => ("n/a".into(), "n/a".into()),
        };
        grouped.entry((row.day.clone(), collection, item, kind)).or_default().add(row.totals);
    }
    grouped
        .into_iter()
        .map(|((day, collection, item, kind), t)| SummaryRow {
            day,
            collection,
            item,
            kind: kind.as_str(),
            requests: t.requests,
            bytes: t.bytes,
            whole_files: t.whole_files(),
        })
        .collect()
}

/// The rows of the summary, the same for its JSON and its CSV: daily rows that pass `filters`,
/// grouped by day, collection, item and archive or file, with collections taken from the catalog.
pub fn summary_rows(mut rows: Vec<DailyRow>, items: &[ItemInfo], filters: &Filters) -> Vec<SummaryRow> {
    let by_code: HashMap<&str, &ItemInfo> = items.iter().map(|i| (i.code.as_str(), i)).collect();
    rows.retain(|r| code_passes(&code_key(&r.path), &by_code, filters).is_some());
    let collections: HashMap<String, String> =
        items.iter().map(|i| (i.code.clone(), i.facets.collection.clone())).collect();
    summarize(&rows, &collections)
}

// ============================================================================
// Per-item view
// ============================================================================

/// The attributes items are filtered and grouped by.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct Facets {
    pub collection: String,
    /// Who supplied the data (`blatten:source`).
    pub provider: String,
    /// The kind of product (`blatten:product_type`).
    pub data_type: String,
    pub dataset: String,
    pub sensor: String,
    pub format: String,
    /// Processing level.
    pub level: String,
}

/// How items are filtered and grouped on the dashboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim {
    Total,
    /// One group per catalog item; not an attribute, so it cannot be filtered on.
    Item,
    Collection,
    Provider,
    DataType,
    Dataset,
    Sensor,
    Format,
    Level,
}

impl Dim {
    /// Every attribute an item can be filtered or grouped by.
    pub const ATTRIBUTES: [Dim; 7] = [
        Dim::Collection,
        Dim::Provider,
        Dim::DataType,
        Dim::Dataset,
        Dim::Sensor,
        Dim::Format,
        Dim::Level,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Dim::Total => "total",
            Dim::Item => "item",
            Dim::Collection => "collection",
            Dim::Provider => "provider",
            Dim::DataType => "data_type",
            Dim::Dataset => "dataset",
            Dim::Sensor => "sensor",
            Dim::Format => "format",
            Dim::Level => "level",
        }
    }

    pub fn parse(s: &str) -> Option<Dim> {
        [Dim::Total, Dim::Item]
            .into_iter()
            .chain(Dim::ATTRIBUTES)
            .find(|d| d.key() == s)
    }

    pub fn of(self, f: &Facets) -> &str {
        match self {
            Dim::Total => "All",
            Dim::Item => "",
            Dim::Collection => &f.collection,
            Dim::Provider => &f.provider,
            Dim::DataType => &f.data_type,
            Dim::Dataset => &f.dataset,
            Dim::Sensor => &f.sensor,
            Dim::Format => &f.format,
            Dim::Level => &f.level,
        }
    }
}

/// Items kept by exact values of their attributes; an attribute may allow several values.
#[derive(Debug, Clone, Default)]
pub struct Filters(Vec<(Dim, Vec<String>)>);

impl Filters {
    /// Read `get(key)` for every attribute; a value may list alternatives separated by `|`.
    pub fn from_lookup<'a>(get: impl Fn(&str) -> Option<&'a str>) -> Filters {
        Filters(
            Dim::ATTRIBUTES
                .iter()
                .filter_map(|d| {
                    let values: Vec<String> = get(d.key())?
                        .split('|')
                        .filter(|v| !v.is_empty())
                        .map(str::to_string)
                        .collect();
                    (!values.is_empty()).then_some((*d, values))
                })
                .collect(),
        )
    }

    pub fn is_active(&self) -> bool {
        !self.0.is_empty()
    }

    pub fn matches(&self, f: &Facets) -> bool {
        self.0.iter().all(|(dim, values)| values.iter().any(|v| v == dim.of(f)))
    }
}

/// What the catalog says about an item: used to list items that were never downloaded and to
/// compare downloaded files with the files that exist.
#[derive(Debug, Clone)]
pub struct ItemInfo {
    pub code: String,
    pub title: String,
    /// Number of files, not counting the archive.
    pub files: usize,
    pub facets: Facets,
}

/// Every value of every attribute with the number of items that have it, for the filter menus.
pub fn facet_values(items: &[ItemInfo]) -> BTreeMap<&'static str, Vec<(String, usize)>> {
    Dim::ATTRIBUTES
        .iter()
        .map(|d| {
            let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
            for i in items {
                *counts.entry(d.of(&i.facets)).or_default() += 1;
            }
            (d.key(), counts.into_iter().map(|(v, n)| (v.to_string(), n)).collect())
        })
        .collect()
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct ItemTotals {
    pub item: String,
    pub title: String,
    #[serde(flatten)]
    pub facets: Facets,
    pub requests: u64,
    pub bytes: u64,
    /// Bytes sent as a number of whole files (see `Totals::equiv`).
    pub whole_files: f64,
    pub archive_requests: u64,
    /// Distinct files requested at least once.
    pub files_downloaded: usize,
    pub files_total: usize,
}

#[derive(Debug, Default, PartialEq)]
pub struct Breakdown {
    /// Every catalog item, including those with no requests.
    pub items: Vec<ItemTotals>,
    pub docs: Totals,
    /// Paths that are not a known item or document.
    pub unknown: Totals,
}

/// Fold per-path totals into one row per catalog item.
pub fn breakdown(paths: &[(String, Totals)], items: &[ItemInfo]) -> Breakdown {
    let mut out = Breakdown {
        items: items
            .iter()
            .map(|i| ItemTotals {
                item: i.code.clone(),
                title: i.title.clone(),
                facets: i.facets.clone(),
                files_total: i.files,
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };
    let index: HashMap<&str, usize> =
        items.iter().enumerate().map(|(n, i)| (i.code.as_str(), n)).collect();
    for (path, totals) in paths {
        let (kind, code) = classify(path);
        let known = code.and_then(|c| index.get(c).copied());
        match (kind, known) {
            (Kind::File, Some(n)) => {
                let it = &mut out.items[n];
                it.requests = it.requests.saturating_add(totals.requests);
                it.bytes = it.bytes.saturating_add(totals.bytes);
                it.whole_files += totals.equiv as f64 / 1000.0;
                it.files_downloaded += 1;
            }
            (Kind::Archive, Some(n)) => {
                let it = &mut out.items[n];
                it.requests = it.requests.saturating_add(totals.requests);
                it.bytes = it.bytes.saturating_add(totals.bytes);
                it.whole_files += totals.equiv as f64 / 1000.0;
                it.archive_requests = it.archive_requests.saturating_add(totals.requests);
            }
            (Kind::Docs, _) => out.docs.add(*totals),
            _ => out.unknown.add(*totals),
        }
    }
    out
}

impl Breakdown {
    /// Keep the items that pass the filters. With a filter active, documents and unknown paths
    /// belong to no item and are dropped from the totals.
    pub fn filtered(mut self, filters: &Filters) -> Breakdown {
        if filters.is_active() {
            self.items.retain(|i| filters.matches(&i.facets));
            self.docs = Totals::default();
            self.unknown = Totals::default();
        }
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct GroupTotals {
    pub name: String,
    pub items: usize,
    pub items_with_downloads: usize,
    pub requests: u64,
    pub bytes: u64,
    pub whole_files: f64,
    pub archive_requests: u64,
    pub files_downloaded: usize,
    pub files_total: usize,
}

/// Sum item rows by an attribute, largest by requests first.
pub fn group_totals(items: &[ItemTotals], dim: Dim) -> Vec<GroupTotals> {
    let mut groups: BTreeMap<&str, GroupTotals> = BTreeMap::new();
    for row in items {
        let g = groups.entry(dim.of(&row.facets)).or_default();
        g.items += 1;
        g.items_with_downloads += (row.requests > 0) as usize;
        g.requests = g.requests.saturating_add(row.requests);
        g.bytes = g.bytes.saturating_add(row.bytes);
        g.whole_files += row.whole_files;
        g.archive_requests = g.archive_requests.saturating_add(row.archive_requests);
        g.files_downloaded += row.files_downloaded;
        g.files_total += row.files_total;
    }
    let mut out: Vec<_> = groups
        .into_iter()
        .map(|(name, mut g)| {
            g.name = name.to_string();
            g
        })
        .collect();
    out.sort_by(|a, b| b.requests.cmp(&a.requests).then(a.name.cmp(&b.name)));
    out
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SeriesGroup {
    pub name: String,
    pub requests: Vec<u64>,
    pub bytes: Vec<u64>,
    pub whole_files: Vec<f64>,
}

/// The width of one bar in a series: totals are kept per hour and per day, and weeks are summed
/// from days (a week starts on Monday, UTC).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Resolution {
    Hour,
    #[default]
    Day,
    Week,
}

/// Windows up to this many days are shown by hour, up to `DAILY_UP_TO_DAYS` by day, longer by week.
const HOURLY_UP_TO_DAYS: i64 = 14;
const DAILY_UP_TO_DAYS: i64 = 366;

/// The most bars one series may have; a resolution asked for that would give more is refused.
pub const MAX_BUCKETS: i64 = 2_500;

impl Resolution {
    pub fn parse(s: &str) -> Option<Resolution> {
        match s {
            "hour" => Some(Resolution::Hour),
            "day" => Some(Resolution::Day),
            "week" => Some(Resolution::Week),
            _ => None,
        }
    }

    /// The resolution for a window of whole days, `first..=last`.
    pub fn for_window(first: chrono::NaiveDate, last: chrono::NaiveDate) -> Resolution {
        let days = (last - first).num_days() + 1;
        if days <= HOURLY_UP_TO_DAYS {
            Resolution::Hour
        } else if days <= DAILY_UP_TO_DAYS {
            Resolution::Day
        } else {
            Resolution::Week
        }
    }

    fn step(self) -> chrono::Duration {
        match self {
            Resolution::Hour => chrono::Duration::hours(1),
            Resolution::Day => chrono::Duration::days(1),
            Resolution::Week => chrono::Duration::weeks(1),
        }
    }

    /// The start of the bucket holding `t`.
    pub fn align(self, t: chrono::NaiveDateTime) -> chrono::NaiveDateTime {
        use chrono::{Datelike, Timelike};
        match self {
            Resolution::Hour => t.date().and_hms_opt(t.hour(), 0, 0).unwrap_or(t),
            Resolution::Day => t.date().and_time(chrono::NaiveTime::MIN),
            Resolution::Week => {
                let monday = t.date() - chrono::Duration::days(t.weekday().num_days_from_monday() as i64);
                monday.and_time(chrono::NaiveTime::MIN)
            }
        }
    }

    fn label(self, start: chrono::NaiveDateTime) -> String {
        match self {
            Resolution::Hour => start.format("%Y-%m-%dT%H:00Z").to_string(),
            Resolution::Day | Resolution::Week => start.format("%Y-%m-%d").to_string(),
        }
    }
}

/// The buckets a series covers: from `first` to `last`, both bucket starts (UTC), inclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub resolution: Resolution,
    pub first: chrono::NaiveDateTime,
    pub last: chrono::NaiveDateTime,
}

impl Window {
    fn starts(&self) -> Vec<chrono::NaiveDateTime> {
        let mut out = Vec::new();
        let mut t = self.first;
        while t <= self.last {
            out.push(t);
            t += self.resolution.step();
        }
        out
    }
}

/// The window of a series over the days `from..=to`. Without `from` it opens on `first_data`, the
/// first day with downloads; without `to` it closes now, and never later than the current bucket.
/// Without a resolution one is picked from the width of the window. `None` when the resolution
/// asked for would give more than `MAX_BUCKETS` bars.
pub fn series_window(
    resolution: Option<Resolution>,
    from: Option<chrono::NaiveDate>,
    to: Option<chrono::NaiveDate>,
    first_data: Option<chrono::NaiveDate>,
    now: chrono::NaiveDateTime,
) -> Option<Window> {
    let today = now.date();
    let first_day = from.or(first_data).unwrap_or(today);
    let last_day = to.unwrap_or(today);
    let resolution = resolution.unwrap_or_else(|| Resolution::for_window(first_day, last_day));
    let end_of_last_day = last_day.and_hms_opt(23, 59, 59)?;
    let window = Window {
        resolution,
        first: resolution.align(first_day.and_time(chrono::NaiveTime::MIN)),
        last: resolution.align(end_of_last_day.min(now)),
    };
    let buckets = (window.last - window.first).num_seconds() / resolution.step().num_seconds() + 1;
    (buckets <= MAX_BUCKETS).then_some(window)
}

#[derive(Debug, Default, PartialEq, serde::Serialize)]
pub struct Series {
    pub resolution: Resolution,
    /// The start of each bucket: `YYYY-MM-DDTHH:00Z` by hour, `YYYY-MM-DD` by day and week.
    pub buckets: Vec<String>,
    pub groups: Vec<SeriesGroup>,
}

/// Whether the daily total kept under `code` belongs in a view with these filters. Without a
/// filter everything counts; with one, only catalog items that match.
pub fn code_passes<'a>(
    code: &str,
    by_code: &HashMap<&'a str, &'a ItemInfo>,
    filters: &Filters,
) -> Option<Option<&'a ItemInfo>> {
    match by_code.get(code) {
        Some(info) if filters.matches(&info.facets) => Some(Some(info)),
        Some(_) => None,
        None if filters.is_active() => None,
        None => Some(None),
    }
}

/// Totals per group and bucket of `window`, from rows of (bucket start, code, totals). The largest
/// `top` groups by requests are kept, the rest are summed as "Smaller groups", and documents and
/// unknown paths are "Documents and other".
pub fn series(
    rows: &[(chrono::NaiveDateTime, String, Totals)],
    infos: &[ItemInfo],
    dim: Dim,
    filters: &Filters,
    top: usize,
    window: &Window,
) -> Series {
    let by_code: HashMap<&str, &ItemInfo> = infos.iter().map(|i| (i.code.as_str(), i)).collect();
    let starts = window.starts();
    let span = starts.len();
    let slot: HashMap<chrono::NaiveDateTime, usize> = starts.iter().enumerate().map(|(n, t)| (*t, n)).collect();

    let mut sums: BTreeMap<String, Vec<Totals>> = BTreeMap::new();
    for (start, code, t) in rows {
        let Some(&n) = slot.get(&window.resolution.align(*start)) else { continue };
        let Some(info) = code_passes(code, &by_code, filters) else { continue };
        let name = match info {
            Some(info) if dim == Dim::Item => info.code.clone(),
            Some(info) => dim.of(&info.facets).to_string(),
            None => "Documents and other".to_string(),
        };
        sums.entry(name).or_insert_with(|| vec![Totals::default(); span])[n].add(*t);
    }

    let mut named: Vec<(String, Vec<Totals>)> = sums.into_iter().collect();
    let total_requests = |v: &Vec<Totals>| v.iter().map(|t| t.requests).sum::<u64>();
    named.sort_by(|a, b| total_requests(&b.1).cmp(&total_requests(&a.1)).then(a.0.cmp(&b.0)));
    let mut kept: Vec<(String, Vec<Totals>)> = Vec::new();
    let mut rest = vec![Totals::default(); span];
    let mut rest_any = false;
    for (name, v) in named {
        let is_other = name == "Documents and other";
        if is_other || kept.iter().filter(|(n, _)| n != "Documents and other").count() < top {
            kept.push((name, v));
        } else {
            rest_any = true;
            for (n, t) in v.into_iter().enumerate() {
                rest[n].add(t);
            }
        }
    }
    if rest_any {
        kept.push(("Smaller groups".to_string(), rest));
    }
    Series {
        resolution: window.resolution,
        buckets: starts.iter().map(|t| window.resolution.label(*t)).collect(),
        groups: kept
            .into_iter()
            .map(|(name, v)| SeriesGroup {
                name,
                requests: v.iter().map(|t| t.requests).collect(),
                bytes: v.iter().map(|t| t.bytes).collect(),
                whole_files: v.iter().map(|t| t.whole_files()).collect(),
            })
            .collect(),
    }
}

/// The path the file proxy reports for a catalog asset link: everything from `/assets/` or
/// `/archives/`, with percent-escapes decoded as the proxy does.
pub fn proxy_path(href: &str) -> Option<String> {
    let start = ["/assets/", "/archives/"].iter().filter_map(|p| href.find(p)).min()?;
    let raw = href[start..].split(['?', '#']).next().unwrap_or("");
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |b: u8| (b as char).to_digit(16);
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).ok()
}

/// The path of a file below its item, for display: `/assets/<code>/a/b.tif` becomes `a/b.tif`.
pub fn file_name_in_item<'a>(path: &'a str, code: &str) -> &'a str {
    let prefix = format!("/assets/{code}/");
    path.strip_prefix(prefix.as_str()).unwrap_or(path)
}

fn csv_cell(value: &str) -> String {
    // A cell that starts with one of these is read as a formula by spreadsheet programs.
    let value = if value.starts_with(['=', '+', '-', '@', '\t', '\r']) {
        format!("'{value}")
    } else {
        value.to_string()
    };
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value
    }
}

pub fn to_csv(rows: &[SummaryRow]) -> String {
    let mut out = String::from("day,collection,item,kind,file_requests,bytes_sent,whole_file_equivalents\n");
    for r in rows {
        out.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            csv_cell(&r.day),
            csv_cell(&r.collection),
            csv_cell(&r.item),
            r.kind,
            r.requests,
            r.bytes,
            r.whole_files
        ));
    }
    out
}

/// One row per file, as stored: `path,requests,bytes`.
pub fn file_csv(rows: &[(String, Totals)]) -> String {
    let mut out = String::from("path,file_requests,bytes_sent,whole_file_equivalents\n");
    for (path, t) in rows {
        out.push_str(&format!("{},{},{},{}\n", csv_cell(path), t.requests, t.bytes, t.whole_files()));
    }
    out
}

/// A setting read from the environment, without surrounding whitespace (a secret file often ends
/// in a newline); an empty one is absent.
pub fn setting(value: Option<String>) -> Option<String> {
    value.map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(code: &str, collection: &str, provider: &str, data_type: &str, files: usize) -> ItemInfo {
        ItemInfo {
            code: code.into(),
            title: code.into(),
            files,
            facets: Facets {
                collection: collection.into(),
                provider: provider.into(),
                data_type: data_type.into(),
                dataset: "Set".into(),
                sensor: "Sensor".into(),
                format: "TIF".into(),
                level: "3".into(),
            },
        }
    }

    const METRICS: &str = r#"# HELP http_requests_total requests
process_start_time_seconds 1.7e+09
http_requests_total{host="a",method="GET",path="/assets/10Aa00/x.tif",server="business",status_code="200"} 4
http_requests_total{host="b",method="GET",path="/assets/10Aa00/x.tif",server="business",status_code="206"} 3
http_requests_total{host="a",method="GET",path="/assets/10Aa00/x.tif",server="business",status_code="404"} 9
http_requests_total{host="a",method="HEAD",path="/assets/10Aa00/x.tif",server="business",status_code="200"} 5
http_requests_total{host="a",method="GET",path="/health",server="internal",status_code="200"} 50
http_response_size_bytes_sum{host="a",method="GET",path="/assets/10Aa00/x.tif",server="business",status_code="200"} 4000
http_response_size_bytes_sum{host="b",method="GET",path="/assets/10Aa00/x.tif",server="business",status_code="206"} 300
http_requests_total{host="a",method="GET",path="/assets/10Aa00/",server="business",status_code="200"} 6
http_requests_total{host="a",method="GET",path="/odd\"path,x",server="business",status_code="200"} 1
http_request_duration_seconds_count{method="GET"} 2
"#;

    #[test]
    fn parse_counts_only_successful_gets() {
        let s = parse_scrape(METRICS);
        assert_eq!(s.start_time, Some(1_700_000_000));
        assert_eq!(
            s.paths["/assets/10Aa00/x.tif"],
            Totals { requests: 7, bytes: 4300, ..Default::default() }
        );
        assert_eq!(s.paths["/odd\"path,x"].requests, 1);
        assert_eq!(s.paths.len(), 2);
    }

    #[test]
    fn increment_handles_growth_and_first_reading() {
        assert_eq!(increment(None, 5), 5);
        assert_eq!(increment(Some(5), 8), 3);
        assert_eq!(increment(Some(5), 5), 0);
    }

    #[test]
    fn a_lower_reading_from_the_same_process_adds_nothing() {
        assert_eq!(increment(Some(100), 40), 0);
    }

    fn scrape(paths: &[(&str, u64, u64)]) -> Scrape {
        Scrape {
            start_time: Some(1_700_000_000),
            paths: paths
                .iter()
                .map(|&(p, requests, bytes)| (p.to_string(), Totals { requests, bytes, ..Default::default() }))
                .collect(),
        }
    }

    fn added(c: &Changes) -> Vec<(&str, u64, u64)> {
        c.added.iter().map(|(p, t)| (p.as_str(), t.requests, t.bytes)).collect()
    }

    #[test]
    fn a_path_with_no_saved_reading_counts_whole_and_is_saved() {
        let c = changes(&parse_scrape(METRICS), &HashMap::new(), &HashMap::new(), false);
        assert_eq!(added(&c), vec![("/assets/10Aa00/x.tif", 7, 4300), ("/odd\"path,x", 1, 0)]);
        assert_eq!(c.save["/assets/10Aa00/x.tif"], (7, 4300));
    }

    #[test]
    fn a_baseline_scrape_saves_every_reading_and_adds_nothing() {
        let now = scrape(&[("/assets/30Cc00/kiln.tif", 4, 40), ("/archives/30Cc00.zip", 1, 900)]);
        let c = changes(&now, &HashMap::new(), &HashMap::new(), true);
        assert!(c.added.is_empty());
        assert_eq!(c.save["/assets/30Cc00/kiln.tif"], (4, 40));
        assert_eq!(c.save["/archives/30Cc00.zip"], (1, 900));
        let c = changes(&now, &HashMap::new(), &HashMap::new(), false);
        assert_eq!(added(&c), vec![("/archives/30Cc00.zip", 1, 900), ("/assets/30Cc00/kiln.tif", 4, 40)]);
    }

    #[test]
    fn growth_adds_the_difference_and_an_unchanged_reading_adds_nothing() {
        let saved = HashMap::from([
            ("/assets/20Bb00/a.tif".to_string(), (3, 300)),
            ("/assets/20Bb00/b.tif".to_string(), (2, 50)),
        ]);
        let now = scrape(&[("/assets/20Bb00/a.tif", 5, 700), ("/assets/20Bb00/b.tif", 2, 50)]);
        let c = changes(&now, &saved, &HashMap::new(), false);
        assert_eq!(added(&c), vec![("/assets/20Bb00/a.tif", 2, 400)]);
        assert_eq!(c.save.len(), 1);
        assert_eq!(c.save["/assets/20Bb00/a.tif"], (5, 700));
    }

    #[test]
    fn a_lower_reading_of_the_same_process_adds_and_saves_nothing() {
        let saved = HashMap::from([("/archives/20Bb00.zip".to_string(), (40, 4000))]);
        let now = scrape(&[("/archives/20Bb00.zip", 3, 90)]);
        let c = changes(&now, &saved, &HashMap::new(), false);
        assert_eq!(c, Changes::default());
    }

    #[test]
    fn a_series_that_appears_later_counts_from_zero() {
        let saved = HashMap::from([("/assets/20Bb00/a.tif".to_string(), (1, 10))]);
        let now = scrape(&[("/assets/20Bb00/a.tif", 1, 10), ("/assets/20Bb00/new.tif", 2, 20)]);
        let c = changes(&now, &saved, &HashMap::new(), false);
        assert_eq!(added(&c), vec![("/assets/20Bb00/new.tif", 2, 20)]);
    }

    #[test]
    fn bytes_grow_without_requests_and_are_carried_apart() {
        // A long transfer is counted in bytes before its request completes.
        let saved = HashMap::from([("/assets/20Bb00/a.tif".to_string(), (1, 100))]);
        let now = scrape(&[("/assets/20Bb00/a.tif", 1, 350)]);
        let sizes = HashMap::from([("/assets/20Bb00/a.tif".to_string(), 500)]);
        let c = changes(&now, &saved, &sizes, false);
        assert_eq!(
            c.added["/assets/20Bb00/a.tif"],
            Totals { requests: 0, bytes: 250, equiv: 500 }
        );
    }

    #[test]
    fn series_the_database_would_refuse_are_skipped() {
        let long = format!("/assets/10Aa00/{}.tif", "a".repeat(3_000));
        let line = |path: &str, value: &str| {
            format!("http_requests_total{{method=\"GET\",path=\"{path}\",server=\"business\",status_code=\"200\"}} {value}\n")
        };
        let text = [
            line(&long, "1"),
            line("/assets/10Aa00/inf.tif", "+Inf"),
            line("/assets/10Aa00/nan.tif", "NaN"),
            line("/assets/10Aa00/huge.tif", "1e19"),
            line("/assets/10Aa00/neg.tif", "-3"),
            line("", "1"),
            line("/assets/10Aa00/ok.tif", "2"),
        ]
        .concat();
        let s = parse_scrape(&text);
        assert_eq!(s.paths.keys().collect::<Vec<_>>(), vec!["/assets/10Aa00/ok.tif"]);
    }

    #[test]
    fn series_summed_over_hosts_saturate() {
        let near = (i64::MAX as u64).to_string();
        let text = format!(
            "http_requests_total{{host=\"a\",method=\"GET\",path=\"/assets/10Aa00/x.tif\",server=\"business\",status_code=\"200\"}} {near}\n\
             http_requests_total{{host=\"b\",method=\"GET\",path=\"/assets/10Aa00/x.tif\",server=\"business\",status_code=\"206\"}} {near}\n"
        );
        assert_eq!(parse_scrape(&text).paths["/assets/10Aa00/x.tif"].requests, i64::MAX as u64);
        let mut t = Totals { requests: u64::MAX, bytes: 1, equiv: 0 };
        t.add(Totals { requests: 1, bytes: 1, equiv: 1 });
        assert_eq!((t.requests, t.bytes, t.equiv), (u64::MAX, 2, 1));
    }

    #[test]
    fn a_code_too_long_to_store_is_summed_as_other() {
        let long = format!("/assets/{}/x.tif", "C".repeat(65));
        assert_eq!(code_key(&long), "~other");
        assert_eq!(code_key(&format!("/archives/{}.zip", "C".repeat(65))), "~other");
        assert_eq!(code_key("/assets/10Aa00/x.tif"), "10Aa00");
    }

    #[test]
    fn a_setting_is_trimmed_and_an_empty_one_is_absent() {
        assert_eq!(setting(Some("secret-key\n".to_string())).as_deref(), Some("secret-key"));
        assert_eq!(setting(Some(" \t\n".to_string())), None);
        assert_eq!(setting(None), None);
    }

    #[test]
    fn classify_known_shapes() {
        assert_eq!(classify("/assets/13Da01/sub/f.tif"), (Kind::File, Some("13Da01")));
        assert_eq!(classify("/archives/13Da01.zip"), (Kind::Archive, Some("13Da01")));
        assert_eq!(classify("/docs/dataset_overview.csv"), (Kind::Docs, None));
        assert_eq!(classify("/favicon.ico"), (Kind::Other, None));
        assert_eq!(classify("/archives/13Da01.tar"), (Kind::Other, None));
    }

    #[test]
    fn summary_groups_by_item_and_never_echoes_unknown_codes() {
        let rows = vec![
            DailyRow { day: "2026-10-22".into(), path: "/assets/10Aa00/a.tif".into(), totals: Totals { requests: 2, bytes: 20, ..Default::default() } },
            DailyRow { day: "2026-10-22".into(), path: "/assets/10Aa00/b.tif".into(), totals: Totals { requests: 1, bytes: 10, ..Default::default() } },
            DailyRow { day: "2026-10-22".into(), path: "/assets/=cmd/c.tif".into(), totals: Totals { requests: 1, bytes: 1, ..Default::default() } },
            DailyRow { day: "2026-10-22".into(), path: "/docs/x.pdf".into(), totals: Totals { requests: 4, bytes: 40, ..Default::default() } },
        ];
        let collections = HashMap::from([("10Aa00".to_string(), "webcam-image".to_string())]);
        let out = summarize(&rows, &collections);
        let known = out.iter().find(|r| r.item == "10Aa00").unwrap();
        assert_eq!((known.requests, known.bytes, known.collection.as_str()), (3, 30, "webcam-image"));
        assert!(out.iter().any(|r| r.item == "unknown" && r.requests == 1));
        assert!(!to_csv(&out).contains("=cmd"));
    }

    fn daily(path: &str, requests: u64, bytes: u64) -> DailyRow {
        DailyRow { day: "2026-03-14".into(), path: path.into(), totals: Totals { requests, bytes, equiv: 0 } }
    }

    #[test]
    fn stored_paths_map_to_item_collection_and_archive_or_file() {
        let items = vec![item("05Kb02", "kite-photo", "Breeze", "Kite", 4)];
        let rows = vec![
            daily("/archives/05Kb02.zip", 2, 200),
            daily("/assets/05Kb02/tail.png", 3, 30),
            daily("/assets/99Zz99/spool.png", 1, 5),
            daily("/lanterns/paper.txt", 1, 1),
        ];
        let out = summary_rows(rows, &items, &Filters::default());
        let found: Vec<_> = out.iter().map(|r| (r.collection.as_str(), r.item.as_str(), r.kind, r.requests)).collect();
        assert_eq!(
            found,
            vec![
                ("kite-photo", "05Kb02", "file", 3),
                ("kite-photo", "05Kb02", "archive", 2),
                ("n/a", "n/a", "other", 1),
                ("unknown", "unknown", "file", 1),
            ]
        );
    }

    #[test]
    fn csv_rows_equal_json_rows_for_the_same_filter() {
        let items = vec![
            item("05Kb02", "kite-photo", "Breeze", "Kite", 4),
            item("06Kc01", "kite-track", "Gust", "Track", 2),
        ];
        let rows = || {
            vec![
                daily("/archives/05Kb02.zip", 2, 200),
                daily("/assets/05Kb02/tail.png", 3, 30),
                daily("/assets/06Kc01/line.csv", 7, 70),
                daily("/docs/manual.pdf", 1, 9),
            ]
        };
        let filters = [Filters::default(), Filters::from_lookup(|k| (k == "provider").then_some("Gust"))];
        for f in &filters {
            let out = summary_rows(rows(), &items, f);
            let json = serde_json::to_value(&out).unwrap();
            let from_json: Vec<String> = json
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    ["day", "collection", "item", "kind", "requests", "bytes", "whole_files"]
                        .iter()
                        .map(|k| match &r[k] {
                            serde_json::Value::String(s) => s.clone(),
                            v => v.as_f64().unwrap().to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .collect();
            let csv = to_csv(&out);
            let from_csv: Vec<&str> = csv.lines().skip(1).collect();
            assert_eq!(from_csv, from_json);
        }
        let gust = summary_rows(rows(), &items, &filters[1]);
        assert_eq!(gust.len(), 1);
        assert_eq!(gust[0].item, "06Kc01");
    }

    #[test]
    fn summary_sums_whole_file_equivalents_into_the_csv() {
        let rows = vec![
            DailyRow { day: "2026-10-22".into(), path: "/assets/10Aa00/a.tif".into(), totals: Totals { requests: 3, bytes: 30, equiv: 1_500 } },
            DailyRow { day: "2026-10-22".into(), path: "/assets/10Aa00/b.tif".into(), totals: Totals { requests: 1, bytes: 10, equiv: 500 } },
        ];
        let collections = HashMap::from([("10Aa00".to_string(), "webcam-image".to_string())]);
        let out = summarize(&rows, &collections);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].whole_files, 2.0);
        assert!(to_csv(&out).ends_with(",4,40,2\n"));
    }

    #[test]
    fn breakdown_lists_every_item_and_counts_distinct_files() {
        let items = vec![
            item("10Aa00", "webcam-image", "A", "W", 3),
            item("11Bb00", "dem", "B", "M", 1),
        ];
        let t = |requests, bytes| Totals { requests, bytes, ..Default::default() };
        let paths = vec![
            ("/assets/10Aa00/a.jpg".to_string(), t(5, 50)),
            ("/assets/10Aa00/b.jpg".to_string(), t(1, 10)),
            ("/archives/10Aa00.zip".to_string(), t(2, 200)),
            ("/docs/x.pdf".to_string(), t(7, 70)),
            ("/assets/99Zz99/a.jpg".to_string(), t(3, 30)),
            ("/favicon.ico".to_string(), t(1, 1)),
        ];
        let b = breakdown(&paths, &items);
        assert_eq!(b.items.len(), 2);
        let cam = &b.items[0];
        assert_eq!((cam.requests, cam.bytes, cam.archive_requests), (8, 260, 2));
        assert_eq!((cam.files_downloaded, cam.files_total), (2, 3));
        assert_eq!(b.items[1].requests, 0);
        assert_eq!(b.docs, t(7, 70));
        assert_eq!(b.unknown, t(4, 31));
        assert_eq!(file_name_in_item("/assets/10Aa00/sub/a.jpg", "10Aa00"), "sub/a.jpg");
    }

    fn infos() -> Vec<ItemInfo> {
        vec![
            item("10Aa00", "webcam-image", "Alpha", "Webcam", 2),
            item("11Bb00", "dem", "Beta", "Model", 2),
            item("12Cc00", "dem", "Alpha", "Model", 2),
        ]
    }

    fn filters(pairs: &[(&str, &str)]) -> Filters {
        Filters::from_lookup(|k| pairs.iter().find(|(key, _)| *key == k).map(|(_, v)| *v))
    }

    #[test]
    fn groups_sum_items_by_attribute() {
        let infos = infos();
        let t = |requests, bytes| Totals { requests, bytes, ..Default::default() };
        let paths = vec![
            ("/assets/10Aa00/a.jpg".to_string(), t(5, 50)),
            ("/assets/11Bb00/a.tif".to_string(), t(2, 20)),
            ("/assets/12Cc00/a.tif".to_string(), t(4, 40)),
        ];
        let b = breakdown(&paths, &infos);
        let by_provider = group_totals(&b.items, Dim::Provider);
        assert_eq!(by_provider[0].name, "Alpha");
        assert_eq!((by_provider[0].requests, by_provider[0].items, by_provider[0].items_with_downloads), (9, 2, 2));
        assert_eq!(by_provider[1].name, "Beta");
        let by_collection = group_totals(&b.items, Dim::Collection);
        assert_eq!(
            by_collection.iter().map(|g| (g.name.as_str(), g.requests)).collect::<Vec<_>>(),
            vec![("dem", 6), ("webcam-image", 5)]
        );
    }

    #[test]
    fn series_can_group_by_item() {
        let infos = infos();
        let t = |requests, bytes| Totals { requests, bytes, ..Default::default() };
        let rows = vec![
            (at("2026-10-22 00:00"), "10Aa00".to_string(), t(5, 50)),
            (at("2026-10-22 00:00"), "11Bb00".to_string(), t(2, 20)),
        ];
        let s = series(&rows, &infos, Dim::Item, &Filters::default(), 7, &days("2026-10-22", "2026-10-22"));
        let got: Vec<_> = s.groups.iter().map(|g| (g.name.as_str(), g.requests[0])).collect();
        assert_eq!(got, vec![("10Aa00", 5), ("11Bb00", 2)]);
        assert!(!Dim::ATTRIBUTES.contains(&Dim::Item));
    }

    #[test]
    fn filters_combine_attributes_and_alternatives() {
        let infos = infos();
        let f = filters(&[("provider", "Alpha"), ("data_type", "Model")]);
        assert!(f.is_active());
        let kept: Vec<_> = infos.iter().filter(|i| f.matches(&i.facets)).map(|i| i.code.as_str()).collect();
        assert_eq!(kept, vec!["12Cc00"]);
        let either = filters(&[("collection", "dem|webcam-image")]);
        assert_eq!(infos.iter().filter(|i| either.matches(&i.facets)).count(), 3);
        assert!(!filters(&[("provider", "")]).is_active());
        assert!(!filters(&[("unknown_key", "x")]).is_active());
    }

    #[test]
    fn filtered_breakdown_drops_documents_and_other_items() {
        let infos = infos();
        let t = |requests, bytes| Totals { requests, bytes, ..Default::default() };
        let paths = vec![
            ("/assets/10Aa00/a.jpg".to_string(), t(5, 50)),
            ("/assets/12Cc00/a.tif".to_string(), t(4, 40)),
            ("/docs/x.pdf".to_string(), t(9, 9)),
        ];
        let b = breakdown(&paths, &infos).filtered(&filters(&[("provider", "Alpha")]));
        assert_eq!(b.items.len(), 2);
        assert_eq!(b.docs, Totals::default());
        assert_eq!(b.unknown, Totals::default());
        let all = breakdown(&paths, &infos).filtered(&Filters::default());
        assert_eq!(all.items.len(), 3);
        assert_eq!(all.docs, t(9, 9));
    }

    #[test]
    fn facet_menus_count_items() {
        let menus = facet_values(&infos());
        assert_eq!(menus["provider"], vec![("Alpha".to_string(), 2), ("Beta".to_string(), 1)]);
        assert_eq!(menus["level"], vec![("3".to_string(), 3)]);
    }

    #[test]
    fn series_fills_days_and_merges_small_groups() {
        let infos = infos();
        let t = |requests, bytes| Totals { requests, bytes, ..Default::default() };
        let rows = vec![
            (at("2026-10-20 00:00"), "10Aa00".to_string(), t(5, 50)),
            (at("2026-10-20 00:00"), "~docs".to_string(), t(1, 1)),
            (at("2026-10-22 00:00"), "11Bb00".to_string(), t(2, 20)),
            (at("2026-10-22 00:00"), "12Cc00".to_string(), t(1, 10)),
        ];
        let none = Filters::default();
        let s = series(&rows, &infos, Dim::Collection, &none, 1, &days("2026-10-20", "2026-10-22"));
        assert_eq!(s.buckets, vec!["2026-10-20", "2026-10-21", "2026-10-22"]);
        let names: Vec<_> = s.groups.iter().map(|g| g.name.as_str()).collect();
        assert_eq!(names, vec!["webcam-image", "Documents and other", "Smaller groups"]);
        assert_eq!(s.groups[0].requests, vec![5, 0, 0]);
        assert_eq!(s.groups[2].requests, vec![0, 0, 3]);
        let total = series(&rows, &infos, Dim::Total, &none, 5, &days("2026-10-21", "2026-10-22"));
        assert_eq!(total.buckets.len(), 2);
        assert_eq!(total.groups.iter().map(|g| g.requests.iter().sum::<u64>()).sum::<u64>(), 3);
    }

    #[test]
    fn series_can_be_grouped_by_one_attribute_and_filtered_by_another() {
        let infos = infos();
        let t = |requests, bytes| Totals { requests, bytes, ..Default::default() };
        let rows = vec![
            (at("2026-10-22 00:00"), "10Aa00".to_string(), t(5, 50)),
            (at("2026-10-22 00:00"), "11Bb00".to_string(), t(2, 20)),
            (at("2026-10-22 00:00"), "12Cc00".to_string(), t(4, 40)),
            (at("2026-10-22 00:00"), "~docs".to_string(), t(7, 7)),
        ];
        // Models only, compared by provider; documents are not part of a filtered view.
        let s = series(&rows, &infos, Dim::Provider, &filters(&[("data_type", "Model")]), 7, &days("2026-10-22", "2026-10-22"));
        let got: Vec<_> = s.groups.iter().map(|g| (g.name.as_str(), g.requests[0])).collect();
        assert_eq!(got, vec![("Alpha", 4), ("Beta", 2)]);
    }

    fn at(s: &str) -> chrono::NaiveDateTime {
        chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M").unwrap()
    }

    fn date(s: &str) -> chrono::NaiveDate {
        chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
    }

    fn days(from: &str, to: &str) -> Window {
        series_window(Some(Resolution::Day), Some(date(from)), Some(date(to)), None, at("2030-01-01 00:00")).unwrap()
    }

    #[test]
    fn resolution_follows_the_width_of_the_window() {
        let pick = |from: &str, days: i64| Resolution::for_window(date(from), date(from) + chrono::Duration::days(days - 1));
        assert_eq!(pick("2026-10-01", 10), Resolution::Hour);
        assert_eq!(pick("2026-10-01", 14), Resolution::Hour);
        assert_eq!(pick("2026-10-01", 15), Resolution::Day);
        assert_eq!(pick("2026-01-01", 100), Resolution::Day);
        assert_eq!(pick("2026-01-01", 366), Resolution::Day);
        assert_eq!(pick("2026-01-01", 367), Resolution::Week);
        assert_eq!(pick("2025-01-01", 730), Resolution::Week);
        assert_eq!(Resolution::parse("hour"), Some(Resolution::Hour));
        assert_eq!(Resolution::parse("month"), None);
    }

    #[test]
    fn a_window_without_a_resolution_picks_one_and_stops_at_the_current_hour() {
        let now = at("2026-10-22 13:40");
        let w = series_window(None, Some(date("2026-10-20")), None, None, now).unwrap();
        assert_eq!((w.resolution, w.first, w.last), (Resolution::Hour, at("2026-10-20 00:00"), at("2026-10-22 13:00")));
        // Without a start, the window opens on the first day with data.
        let w = series_window(None, None, None, Some(date("2025-06-01")), now).unwrap();
        assert_eq!(w.resolution, Resolution::Week);
        assert_eq!(w.first, at("2025-05-26 00:00"), "weeks start on Monday");
        assert_eq!(w.last, at("2026-10-19 00:00"));
        let w = series_window(Some(Resolution::Day), Some(date("2026-10-01")), Some(date("2026-10-05")), None, now).unwrap();
        assert_eq!((w.first, w.last), (at("2026-10-01 00:00"), at("2026-10-05 00:00")));
        // An hourly series over years is refused rather than cut short.
        assert!(series_window(Some(Resolution::Hour), Some(date("2020-01-01")), None, None, now).is_none());
    }

    #[test]
    fn series_by_hour_fills_the_empty_hours() {
        let infos = infos();
        let t = |requests| Totals { requests, ..Default::default() };
        let rows = vec![
            (at("2026-10-22 09:00"), "10Aa00".to_string(), t(3)),
            (at("2026-10-22 12:00"), "10Aa00".to_string(), t(4)),
        ];
        let window = series_window(None, Some(date("2026-10-22")), None, None, at("2026-10-22 12:30")).unwrap();
        let s = series(&rows, &infos, Dim::Total, &Filters::default(), 7, &window);
        assert_eq!(s.resolution, Resolution::Hour);
        assert_eq!(s.buckets.len(), 13);
        assert_eq!(s.buckets[9], "2026-10-22T09:00Z");
        let mut want = vec![0; 13];
        want[9] = 3;
        want[12] = 4;
        assert_eq!(s.groups[0].requests, want);
    }

    #[test]
    fn series_by_week_puts_each_row_in_its_week() {
        let infos = infos();
        let t = |requests| Totals { requests, ..Default::default() };
        let rows = vec![
            (at("2026-09-28 00:00"), "10Aa00".to_string(), t(2)),
            (at("2026-10-12 00:00"), "10Aa00".to_string(), t(5)),
        ];
        let window = series_window(Some(Resolution::Week), Some(date("2026-10-01")), Some(date("2026-10-14")), None, at("2026-10-22 12:00")).unwrap();
        let s = series(&rows, &infos, Dim::Total, &Filters::default(), 7, &window);
        assert_eq!(s.buckets, vec!["2026-09-28", "2026-10-05", "2026-10-12"]);
        assert_eq!(s.groups[0].requests, vec![2, 0, 5]);
    }

    #[test]
    fn asset_links_map_to_proxy_paths() {
        assert_eq!(proxy_path("http://localhost:8888/s3/assets/01Aa00/a.gif").as_deref(), Some("/assets/01Aa00/a.gif"));
        assert_eq!(proxy_path("/s3/archives/01Aa00.zip").as_deref(), Some("/archives/01Aa00.zip"));
        assert_eq!(proxy_path("https://h/s3/assets/01Aa00/a%20b%2Bc.tif").as_deref(), Some("/assets/01Aa00/a b+c.tif"));
        assert_eq!(proxy_path("https://h/stac/collections/x"), None);
    }

    #[test]
    fn whole_files_follow_bytes_over_size() {
        assert_eq!(whole_file_thousandths(500, Some(1000)), 500);
        // Ten ranged pieces that make up one file are one whole file.
        assert_eq!((0..10).map(|_| whole_file_thousandths(100, Some(1000))).sum::<u64>(), 1000);
        assert_eq!(whole_file_thousandths(5, None), 0);
        assert_eq!(whole_file_thousandths(5, Some(0)), 0);
        assert_eq!(Totals { equiv: 2_500, ..Default::default() }.whole_files(), 2.5);
    }

    #[test]
    fn csv_cells_resist_formulas_and_quote_commas() {
        assert_eq!(csv_cell("=1+1"), "'=1+1");
        assert_eq!(csv_cell("a,b"), "\"a,b\"");
        assert_eq!(csv_cell("plain"), "plain");
    }
}
