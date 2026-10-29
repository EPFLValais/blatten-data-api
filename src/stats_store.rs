//! Postgres storage for the download statistics.
//!
//! Every query is parameterised: no value that comes from a request or from the proxy's metrics is
//! ever written into SQL text. The tables are created by the migrations in `migrations/`, which are
//! embedded in the binary and applied once the database can be reached.

use crate::stats::{changes, code_key, DailyRow, FileIndex, Resolution, Scrape, Totals, MAX_COUNTER};
use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use sqlx::{
    postgres::{PgConnectOptions, PgPoolOptions},
    PgPool, Row,
};
use std::{
    collections::{BTreeMap, HashMap},
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tracing::{info, warn};

/// How long per-scrape activity is kept for the live view.
const RECENT_RETENTION_SECONDS: i64 = 48 * 3600;

/// How long the last readings of a proxy process are kept after it was last scraped.
const PROCESS_RETENTION_SECONDS: i64 = 48 * 3600;

/// Held for the length of an ingest so that two scrapes arriving together are applied one after
/// the other, each against the readings the previous one saved.
const INGEST_LOCK: i64 = 0x626c_6174_7374_6174;

/// A query that runs longer than this is cancelled by the database.
const STATEMENT_TIMEOUT_MS: &str = "15000";

/// Parse the statistics database URL without connecting.
pub fn database_options(url: &str) -> Result<PgConnectOptions> {
    PgConnectOptions::from_str(url)
        .map_err(|e| anyhow::anyhow!("STATS_DATABASE_URL is not a valid Postgres URL: {e}"))
}

#[derive(Clone)]
pub struct Store {
    pool: PgPool,
    ready: Arc<AtomicBool>,
    /// The schema a test store lives in, dropped with the last clone of the store.
    #[cfg(test)]
    schema: Option<Arc<tests::DropSchema>>,
}

fn as_u64(v: i64) -> u64 {
    v.max(0) as u64
}

/// A counter as a `bigint`, held at the largest one rather than wrapping negative.
fn as_i64(v: u64) -> i64 {
    v.min(MAX_COUNTER) as i64
}


impl Store {
    /// Prepare a connection pool from a `postgres://` URL. Nothing is contacted yet, so the API
    /// starts whether or not the database is up.
    #[cfg(test)]
    pub fn lazy(url: &str) -> Result<Self> {
        Ok(Self::lazy_with(database_options(url)?))
    }

    pub fn lazy_with(options: PgConnectOptions) -> Self {
        let options = options.options([
            ("statement_timeout", STATEMENT_TIMEOUT_MS),
            ("idle_in_transaction_session_timeout", "30000"),
        ]);
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .acquire_timeout(Duration::from_secs(5))
            .connect_lazy_with(options);
        Self {
            pool,
            ready: Arc::new(AtomicBool::new(false)),
            #[cfg(test)]
            schema: None,
        }
    }

    /// Apply the migrations, retrying until the database accepts them. Run it in a background task.
    pub async fn migrate_until_ready(&self, files: &FileIndex) {
        loop {
            match self.migrate(files).await {
                Ok(()) => {
                    info!("Statistics database is ready");
                    return;
                }
                Err(e) => {
                    warn!("Statistics database is not ready: {e:#}");
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
            }
        }
    }

    /// Apply the migrations, then label the totals they left without a file format, reading the
    /// formats from the catalog's `files`.
    pub async fn migrate(&self, files: &FileIndex) -> Result<()> {
        sqlx::migrate!("./migrations")
            .run(&self.pool)
            .await
            .context("apply statistics migrations")?;
        self.label_formats(files).await.context("label the formats of earlier totals")?;
        self.ready.store(true, Ordering::Release);
        Ok(())
    }

    /// Give a file format to the day and hour totals summed before totals were kept per format,
    /// which the migration leaves at ''. The days are summed again from `daily` per path, under the
    /// keys `code_key` and the formats `files` give; a day with no paths to sum keeps its totals as
    /// `Unknown`. An hour takes the format of its day's totals of the code when they are all of one
    /// format, else `Unknown`. Once every row has a format this does nothing.
    pub async fn label_formats(&self, files: &FileIndex) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        sqlx::query("SELECT pg_advisory_xact_lock($1)").bind(INGEST_LOCK).execute(&mut *tx).await?;
        let pending: bool = sqlx::query_scalar(
            "SELECT EXISTS (SELECT 1 FROM daily_code WHERE format = '') OR EXISTS (SELECT 1 FROM hourly_code WHERE format = '')",
        )
        .fetch_one(&mut *tx)
        .await?;
        if !pending {
            return Ok(());
        }

        let paths: Vec<String> = sqlx::query_scalar(
            "SELECT DISTINCT path FROM daily WHERE day IN (SELECT day FROM daily_code WHERE format = '')",
        )
        .fetch_all(&mut *tx)
        .await?;
        let codes: Vec<String> = paths.iter().map(|p| code_key(p)).collect();
        let formats: Vec<String> = paths.iter().map(|p| files.format(p)).collect();
        sqlx::query("CREATE TEMP TABLE path_label (path text PRIMARY KEY, code text NOT NULL, format text NOT NULL) ON COMMIT DROP")
            .execute(&mut *tx)
            .await?;
        sqlx::query("INSERT INTO path_label (path, code, format) SELECT * FROM UNNEST($1::text[], $2::text[], $3::text[])")
            .bind(&paths)
            .bind(&codes)
            .bind(&formats)
            .execute(&mut *tx)
            .await?;
        sqlx::query(
            "CREATE TEMP TABLE day_label ON COMMIT DROP AS
             SELECT d.day, l.code, l.format,
                    LEAST(SUM(d.requests), 9223372036854775807)::bigint AS requests,
                    LEAST(SUM(d.bytes), 9223372036854775807)::bigint AS bytes,
                    LEAST(SUM(d.equiv), 9223372036854775807)::bigint AS equiv
             FROM daily d JOIN path_label l USING (path)
             WHERE d.day IN (SELECT day FROM daily_code WHERE format = '')
             GROUP BY d.day, l.code, l.format",
        )
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "DELETE FROM daily_code c WHERE c.format = ''
                 AND EXISTS (SELECT 1 FROM day_label l WHERE l.day = c.day AND l.code = c.code)",
        )
        .execute(&mut *tx)
        .await?;
        // A day is summed again only while none of its rows has a format, so nothing is counted twice:
        // a row already there is a conflict, which fails the labelling rather than adding to it.
        sqlx::query(
            "INSERT INTO daily_code (day, code, format, requests, bytes, equiv)
             SELECT day, code, format, requests, bytes, equiv FROM day_label",
        )
        .execute(&mut *tx)
        .await?;
        sqlx::query("UPDATE daily_code SET format = 'Unknown' WHERE format = ''").execute(&mut *tx).await?;
        sqlx::query(
            "UPDATE hourly_code h SET format = s.format
             FROM (SELECT day, code, MIN(format) AS format FROM daily_code GROUP BY day, code HAVING COUNT(*) = 1) s
             WHERE h.format = '' AND s.day = (h.hour AT TIME ZONE 'UTC')::date AND s.code = h.code",
        )
        .execute(&mut *tx)
        .await?;
        sqlx::query("UPDATE hourly_code SET format = 'Unknown' WHERE format = ''").execute(&mut *tx).await?;
        tx.commit().await?;
        info!("Labelled the formats of {} earlier file paths", paths.len());
        Ok(())
    }

    /// Whether the migrations have been applied, so queries can run.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Add one scrape to the totals of `day` (UTC) read at unix time `now`. Posting the same scrape
    /// twice adds nothing the second time. The first scrape into a store that has never ingested
    /// (no `last_ingest` in `meta`) is a baseline: its readings are saved and nothing is added.
    pub async fn ingest(
        &self,
        day: NaiveDate,
        now: i64,
        scrape: &Scrape,
        files: &FileIndex,
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        sqlx::query("SELECT pg_advisory_xact_lock($1)").bind(INGEST_LOCK).execute(&mut *tx).await?;
        let baseline = sqlx::query("SELECT 1 FROM meta WHERE key = 'last_ingest'")
            .fetch_optional(&mut *tx)
            .await?
            .is_none();

        // A scrape without a start time is filed under process 0.
        let process = scrape.start_time.map(as_i64).unwrap_or(0);
        sqlx::query(
            "INSERT INTO proxy_process (start_time, ts) VALUES ($1, $2)
             ON CONFLICT (start_time) DO UPDATE SET ts = GREATEST(proxy_process.ts, EXCLUDED.ts)",
        )
        .bind(process)
        .bind(now)
        .execute(&mut *tx)
        .await?;

        let saved: HashMap<String, (u64, u64)> =
            sqlx::query("SELECT path, requests, bytes FROM series_last WHERE start_time = $1")
                .bind(process)
                .fetch_all(&mut *tx)
                .await?
                .iter()
                .map(|r| (r.get::<String, _>(0), (as_u64(r.get(1)), as_u64(r.get(2)))))
                .collect();

        // Only paths whose reading changed are written.
        let changed = changes(scrape, &saved, &files.sizes, baseline);

        if !changed.added.is_empty() {
            let paths: Vec<String> = changed.added.keys().cloned().collect();
            let requests: Vec<i64> = changed.added.values().map(|t| as_i64(t.requests)).collect();
            let bytes: Vec<i64> = changed.added.values().map(|t| as_i64(t.bytes)).collect();
            let equiv: Vec<i64> = changed.added.values().map(|t| as_i64(t.equiv)).collect();
            sqlx::query(
                "INSERT INTO recent (ts, path, requests, bytes)
                 SELECT $1, p, r, b FROM UNNEST($2::text[], $3::bigint[], $4::bigint[]) AS u(p, r, b)
                 ON CONFLICT (ts, path) DO UPDATE SET
                     requests = LEAST(recent.requests::numeric + EXCLUDED.requests, 9223372036854775807)::bigint,
                     bytes = LEAST(recent.bytes::numeric + EXCLUDED.bytes, 9223372036854775807)::bigint",
            )
            .bind(now)
            .bind(&paths)
            .bind(&requests)
            .bind(&bytes)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "INSERT INTO daily (day, path, requests, bytes, equiv)
                 SELECT $1, p, r, b, e FROM UNNEST($2::text[], $3::bigint[], $4::bigint[], $5::bigint[]) AS u(p, r, b, e)
                 ON CONFLICT (day, path) DO UPDATE SET
                     requests = LEAST(daily.requests::numeric + EXCLUDED.requests, 9223372036854775807)::bigint,
                     bytes = LEAST(daily.bytes::numeric + EXCLUDED.bytes, 9223372036854775807)::bigint,
                     equiv = LEAST(daily.equiv::numeric + EXCLUDED.equiv, 9223372036854775807)::bigint",
            )
            .bind(day)
            .bind(&paths)
            .bind(&requests)
            .bind(&bytes)
            .bind(&equiv)
            .execute(&mut *tx)
            .await?;

            // One row per code and format, so no key appears twice in the one statement.
            let mut by_code: BTreeMap<(String, String), Totals> = BTreeMap::new();
            for (path, totals) in &changed.added {
                by_code.entry((code_key(path), files.format(path))).or_default().add(*totals);
            }
            let codes: Vec<String> = by_code.keys().map(|k| k.0.clone()).collect();
            let formats: Vec<String> = by_code.keys().map(|k| k.1.clone()).collect();
            let c_requests: Vec<i64> = by_code.values().map(|t| as_i64(t.requests)).collect();
            let c_bytes: Vec<i64> = by_code.values().map(|t| as_i64(t.bytes)).collect();
            let c_equiv: Vec<i64> = by_code.values().map(|t| as_i64(t.equiv)).collect();
            sqlx::query(
                "INSERT INTO daily_code (day, code, format, requests, bytes, equiv)
                 SELECT $1, c, f, r, b, e FROM UNNEST($2::text[], $3::text[], $4::bigint[], $5::bigint[], $6::bigint[]) AS u(c, f, r, b, e)
                 ON CONFLICT (day, code, format) DO UPDATE SET
                     requests = LEAST(daily_code.requests::numeric + EXCLUDED.requests, 9223372036854775807)::bigint,
                     bytes = LEAST(daily_code.bytes::numeric + EXCLUDED.bytes, 9223372036854775807)::bigint,
                     equiv = LEAST(daily_code.equiv::numeric + EXCLUDED.equiv, 9223372036854775807)::bigint",
            )
            .bind(day)
            .bind(&codes)
            .bind(&formats)
            .bind(&c_requests)
            .bind(&c_bytes)
            .bind(&c_equiv)
            .execute(&mut *tx)
            .await?;
            let hour = DateTime::<Utc>::from_timestamp(now - now.rem_euclid(3600), 0)
                .context("scrape time out of range")?;
            sqlx::query(
                "INSERT INTO hourly_code (hour, code, format, requests, bytes, equiv)
                 SELECT $1, c, f, r, b, e FROM UNNEST($2::text[], $3::text[], $4::bigint[], $5::bigint[], $6::bigint[]) AS u(c, f, r, b, e)
                 ON CONFLICT (hour, code, format) DO UPDATE SET
                     requests = LEAST(hourly_code.requests::numeric + EXCLUDED.requests, 9223372036854775807)::bigint,
                     bytes = LEAST(hourly_code.bytes::numeric + EXCLUDED.bytes, 9223372036854775807)::bigint,
                     equiv = LEAST(hourly_code.equiv::numeric + EXCLUDED.equiv, 9223372036854775807)::bigint",
            )
            .bind(hour)
            .bind(&codes)
            .bind(&formats)
            .bind(&c_requests)
            .bind(&c_bytes)
            .bind(&c_equiv)
            .execute(&mut *tx)
            .await?;
        }

        if !changed.save.is_empty() {
            let paths: Vec<String> = changed.save.keys().cloned().collect();
            let requests: Vec<i64> = changed.save.values().map(|s| as_i64(s.0)).collect();
            let bytes: Vec<i64> = changed.save.values().map(|s| as_i64(s.1)).collect();
            sqlx::query(
                "INSERT INTO series_last (start_time, path, requests, bytes)
                 SELECT $1, p, r, b FROM UNNEST($2::text[], $3::bigint[], $4::bigint[]) AS u(p, r, b)
                 ON CONFLICT (start_time, path) DO UPDATE SET
                     requests = GREATEST(series_last.requests, EXCLUDED.requests),
                     bytes = GREATEST(series_last.bytes, EXCLUDED.bytes)",
            )
            .bind(process)
            .bind(&paths)
            .bind(&requests)
            .bind(&bytes)
            .execute(&mut *tx)
            .await?;
        }

        sqlx::query("DELETE FROM recent WHERE ts < $1")
            .bind(now - RECENT_RETENTION_SECONDS)
            .execute(&mut *tx)
            .await?;
        sqlx::query(
            "DELETE FROM series_last WHERE start_time IN
                 (SELECT start_time FROM proxy_process WHERE ts < $1)",
        )
        .bind(now - PROCESS_RETENTION_SECONDS)
        .execute(&mut *tx)
        .await?;
        sqlx::query("DELETE FROM proxy_process WHERE ts < $1")
            .bind(now - PROCESS_RETENTION_SECONDS)
            .execute(&mut *tx)
            .await?;
        set_meta(&mut tx, "last_ingest", now).await?;
        tx.commit().await?;
        Ok(())
    }

    /// Daily rows with `from <= day <= to` (either bound optional).
    pub async fn daily(&self, from: Option<NaiveDate>, to: Option<NaiveDate>) -> Result<Vec<DailyRow>> {
        let rows = sqlx::query(
            "SELECT day, path, requests, bytes, equiv FROM daily
             WHERE ($1::date IS NULL OR day >= $1) AND ($2::date IS NULL OR day <= $2)
             ORDER BY day, path",
        )
        .bind(from)
        .bind(to)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| DailyRow {
                day: r.get::<NaiveDate, _>(0).to_string(),
                path: r.get(1),
                totals: totals(r, 2),
            })
            .collect())
    }

    /// Totals per object path over the days `from..=to`.
    pub async fn path_totals(
        &self,
        from: Option<NaiveDate>,
        to: Option<NaiveDate>,
    ) -> Result<Vec<(String, Totals)>> {
        let rows = sqlx::query(
            "SELECT path,
                    LEAST(SUM(requests), 9223372036854775807)::bigint,
                    LEAST(SUM(bytes), 9223372036854775807)::bigint,
                    LEAST(SUM(equiv), 9223372036854775807)::bigint
             FROM daily
             WHERE ($1::date IS NULL OR day >= $1) AND ($2::date IS NULL OR day <= $2)
             GROUP BY path",
        )
        .bind(from)
        .bind(to)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(|r| (r.get(0), totals(r, 1))).collect())
    }

    /// Totals per day, item code (or `~docs`, `~other`) and file format over the days `from..=to`.
    pub async fn code_days(
        &self,
        from: Option<NaiveDate>,
        to: Option<NaiveDate>,
    ) -> Result<Vec<(String, String, String, Totals)>> {
        let rows = sqlx::query(
            "SELECT day, code, format, requests, bytes, equiv FROM daily_code
             WHERE ($1::date IS NULL OR day >= $1) AND ($2::date IS NULL OR day <= $2)
             ORDER BY day, code, format",
        )
        .bind(from)
        .bind(to)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| (r.get::<NaiveDate, _>(0).to_string(), r.get(1), r.get(2), totals(r, 3)))
            .collect())
    }

    /// Totals per bucket start (UTC), item code and file format over the days `from..=to`, by hour,
    /// day or week. Weeks start on Monday and are summed from the daily totals.
    pub async fn code_buckets(
        &self,
        resolution: Resolution,
        from: Option<NaiveDate>,
        to: Option<NaiveDate>,
    ) -> Result<Vec<(NaiveDateTime, String, String, Totals)>> {
        let start_of = |d: NaiveDate| d.and_time(chrono::NaiveTime::MIN).and_utc();
        let rows = match resolution {
            Resolution::Hour => {
                sqlx::query(
                    "SELECT hour, code, format, requests, bytes, equiv FROM hourly_code
                     WHERE ($1::timestamptz IS NULL OR hour >= $1) AND ($2::timestamptz IS NULL OR hour < $2)
                     ORDER BY hour, code, format",
                )
                .bind(from.map(start_of))
                .bind(to.and_then(|d| d.succ_opt()).map(start_of))
                .fetch_all(&self.pool)
                .await?
            }
            Resolution::Day | Resolution::Week => {
                let bucket = if resolution == Resolution::Week { "date_trunc('week', day::timestamp)::date" } else { "day" };
                // `bucket` is one of the two constant expressions above, never outside input.
                sqlx::query(&format!(
                    "SELECT {bucket} AS b, code, format, LEAST(SUM(requests), 9223372036854775807)::bigint, LEAST(SUM(bytes), 9223372036854775807)::bigint, LEAST(SUM(equiv), 9223372036854775807)::bigint
                     FROM daily_code
                     WHERE ($1::date IS NULL OR day >= $1) AND ($2::date IS NULL OR day <= $2)
                     GROUP BY b, code, format ORDER BY b, code, format"
                ))
                .bind(from)
                .bind(to)
                .fetch_all(&self.pool)
                .await?
            }
        };
        Ok(rows
            .iter()
            .map(|r| {
                let start = match resolution {
                    Resolution::Hour => r.get::<DateTime<Utc>, _>(0).naive_utc(),
                    _ => r.get::<NaiveDate, _>(0).and_time(chrono::NaiveTime::MIN),
                };
                (start, r.get(1), r.get(2), totals(r, 3))
            })
            .collect())
    }

    /// The first day with any download.
    pub async fn first_day(&self) -> Result<Option<NaiveDate>> {
        Ok(sqlx::query_scalar("SELECT MIN(day) FROM daily_code").fetch_one(&self.pool).await?)
    }

    /// Per-scrape activity since unix time `since`: (time, path, totals), oldest first.
    pub async fn recent_rows(&self, since: i64) -> Result<Vec<(i64, String, Totals)>> {
        let rows = sqlx::query(
            "SELECT ts, path, requests, bytes FROM recent WHERE ts >= $1 ORDER BY ts",
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| {
                (
                    r.get(0),
                    r.get(1),
                    Totals { requests: as_u64(r.get(2)), bytes: as_u64(r.get(3)), ..Default::default() },
                )
            })
            .collect())
    }

    /// Unix time of the last accepted scrape.
    pub async fn last_ingest(&self) -> Result<Option<i64>> {
        Ok(sqlx::query_scalar("SELECT value FROM meta WHERE key = 'last_ingest'")
            .fetch_optional(&self.pool)
            .await?)
    }
}

fn totals(r: &sqlx::postgres::PgRow, first: usize) -> Totals {
    Totals {
        requests: as_u64(r.get(first)),
        bytes: as_u64(r.get(first + 1)),
        equiv: as_u64(r.get(first + 2)),
    }
}

async fn set_meta(tx: &mut sqlx::Transaction<'_, sqlx::Postgres>, key: &str, value: i64) -> Result<()> {
    sqlx::query(
        "INSERT INTO meta (key, value) VALUES ($1, $2)
         ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
    )
    .bind(key)
    .bind(value)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

// These tests need a Postgres to run against. Each test creates its own schema in it, so they can
// run side by side, and drops it when the test ends. Run them against the test database, never the
// development database `stats`, which they refuse:
//   STATS_TEST_DATABASE_URL=postgres://stats:stats-local-development@localhost:5433/stats_test cargo test -- --include-ignored
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    /// The options of the test database at `url`, which is never the development database.
    fn test_database_options(url: &str) -> PgConnectOptions {
        let options = PgConnectOptions::from_str(url).expect("STATS_TEST_DATABASE_URL is a Postgres URL");
        assert_ne!(
            options.get_database(),
            Some("stats"),
            "STATS_TEST_DATABASE_URL names the development database stats; point it at stats_test"
        );
        options
    }

    /// Drops a test schema when the last store using it goes, whether the test passed or panicked.
    pub(crate) struct DropSchema {
        options: PgConnectOptions,
        schema: String,
    }

    impl Drop for DropSchema {
        fn drop(&mut self) {
            // Drop may run inside the test's runtime, which cannot be blocked on, so the schema is
            // dropped from a thread with a runtime of its own.
            let options = self.options.clone();
            let sql = format!("DROP SCHEMA IF EXISTS {} CASCADE", self.schema);
            let dropped = std::thread::spawn(move || {
                tokio::runtime::Builder::new_current_thread().enable_all().build()?.block_on(async {
                    use sqlx::{Connection, PgConnection};
                    let mut conn = PgConnection::connect_with(&options.options([("lock_timeout", "10000")])).await?;
                    sqlx::query(&sql).execute(&mut conn).await?;
                    conn.close().await?;
                    anyhow::Ok(())
                })
            })
            .join();
            // A panic here, during a test's unwinding, would abort every test left to run.
            if !matches!(dropped, Ok(Ok(()))) {
                eprintln!("test schema {} was not dropped", self.schema);
            }
        }
    }

    /// A migrated store in a schema of its own.
    pub(crate) async fn test_store() -> Store {
        let store = empty_store().await;
        store.migrate(&FileIndex::default()).await.expect("migrate");
        store
    }

    /// A store in a schema of its own with only the migrations before `version` applied, to seed
    /// rows the way earlier versions wrote them.
    async fn store_before(version: i64) -> Store {
        let store = empty_store().await;
        let mut migrator = sqlx::migrate!("./migrations");
        migrator.migrations = migrator.migrations.iter().filter(|m| m.version < version).cloned().collect::<Vec<_>>().into();
        migrator.run(&store.pool).await.expect("migrate");
        store
    }

    /// A store in a schema of its own, with no table in it.
    async fn empty_store() -> Store {
        let url = std::env::var("STATS_TEST_DATABASE_URL")
            .expect("STATS_TEST_DATABASE_URL must point at a Postgres for these tests");
        let database = test_database_options(&url);
        let schema = format!("t_{}_{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed));
        let admin = PgPool::connect_with(database.clone()).await.expect("connect");
        // The schema name is built here from a process id and a counter, never from outside input.
        sqlx::query(&format!("CREATE SCHEMA {schema}")).execute(&admin).await.expect("schema");
        admin.close().await;
        let guard = Arc::new(DropSchema { options: database.clone(), schema: schema.clone() });
        let mut store = Store::lazy_with(database.options([("search_path", schema.as_str())]));
        store.schema = Some(guard);
        store
    }

    /// A migrated store that has already taken its baseline scrape, so the next scrape counts.
    pub(crate) async fn baselined_store() -> Store {
        let store = test_store().await;
        sqlx::query("INSERT INTO meta (key, value) VALUES ('last_ingest', 0)")
            .execute(&store.pool)
            .await
            .expect("baseline");
        store
    }

    fn day(s: &str) -> NaiveDate {
        NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
    }

    fn scrape(start: u64, requests: u64, bytes: u64) -> Scrape {
        let mut s = Scrape { start_time: Some(start), ..Default::default() };
        s.paths.insert(
            "/assets/10Aa00/x.tif".into(),
            Totals { requests, bytes, ..Default::default() },
        );
        s
    }

    fn none() -> FileIndex {
        FileIndex::default()
    }

    #[test]
    #[should_panic(expected = "stats_test")]
    fn tests_refuse_the_development_database() {
        test_database_options("postgres://stats:x@localhost:5433/stats");
    }

    #[test]
    fn tests_accept_the_test_database() {
        let options = test_database_options("postgres://stats:x@localhost:5433/stats_test");
        assert_eq!(options.get_database(), Some("stats_test"));
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn a_store_drops_its_schema_when_it_is_dropped() {
        let store = test_store().await;
        let schema: String = sqlx::query_scalar("SELECT current_schema()").fetch_one(&store.pool).await.unwrap();
        let admin = PgPool::connect(&std::env::var("STATS_TEST_DATABASE_URL").unwrap()).await.unwrap();
        let exists = || async {
            sqlx::query_scalar::<_, bool>("SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = $1)")
                .bind(&schema)
                .fetch_one(&admin)
                .await
                .unwrap()
        };
        assert!(exists().await);
        let copy = store.clone();
        drop(store);
        assert!(exists().await, "a clone still holds the schema");
        drop(copy);
        assert!(!exists().await, "{schema} is left behind");
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn the_first_scrape_into_an_empty_store_is_a_baseline() {
        let store = test_store().await;
        let d = day("2026-10-22");
        store.ingest(d, 1, &scrape(100, 5, 50), &none()).await.unwrap();
        assert!(store.daily(None, None).await.unwrap().is_empty());
        assert!(store.path_totals(None, None).await.unwrap().is_empty());
        assert!(store.code_days(None, None).await.unwrap().is_empty());
        assert!(store.code_buckets(Resolution::Hour, None, None).await.unwrap().is_empty());
        assert!(store.recent_rows(0).await.unwrap().is_empty());
        assert_eq!(store.last_ingest().await.unwrap(), Some(1));
        let held = || async {
            let t = store.path_totals(None, None).await.unwrap()[0].1;
            (t.requests, t.bytes)
        };
        store.ingest(d, 2, &scrape(100, 8, 80), &none()).await.unwrap();
        assert_eq!(held().await, (3, 30));
        // A restart after the baseline counts the new process from zero.
        store.ingest(d, 3, &scrape(200, 2, 20), &none()).await.unwrap();
        assert_eq!(held().await, (5, 50));
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn ingest_adds_increments_once_and_survives_restart() {
        let store = baselined_store().await;
        store.ingest(day("2026-10-22"), 1, &scrape(100, 5, 50), &none()).await.unwrap();
        store.ingest(day("2026-10-22"), 2, &scrape(100, 5, 50), &none()).await.unwrap();
        store.ingest(day("2026-10-22"), 3, &scrape(100, 8, 80), &none()).await.unwrap();
        // The proxy restarted: its counters start again and read higher than the last saved value.
        store.ingest(day("2026-10-23"), 4, &scrape(200, 20, 200), &none()).await.unwrap();
        let rows = store.daily(None, None).await.unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!((rows[0].totals.requests, rows[0].totals.bytes), (8, 80));
        assert_eq!((rows[1].totals.requests, rows[1].totals.bytes), (20, 200));
        assert_eq!(store.last_ingest().await.unwrap(), Some(4));
        assert_eq!(store.daily(Some(day("2026-10-23")), None).await.unwrap().len(), 1);
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn two_proxy_processes_scraped_in_turn_count_each_reading_once() {
        let store = baselined_store().await;
        let d = day("2026-10-22");
        store.ingest(d, 1, &scrape(100, 10, 100), &none()).await.unwrap();
        store.ingest(d, 2, &scrape(200, 1, 10), &none()).await.unwrap();
        store.ingest(d, 3, &scrape(100, 12, 120), &none()).await.unwrap();
        store.ingest(d, 4, &scrape(200, 3, 30), &none()).await.unwrap();
        let rows = store.daily(None, None).await.unwrap();
        assert_eq!((rows[0].totals.requests, rows[0].totals.bytes), (12 + 3, 120 + 30));
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn an_older_reading_posted_late_is_skipped() {
        let store = baselined_store().await;
        let d = day("2026-10-22");
        store.ingest(d, 1, &scrape(100, 5, 50), &none()).await.unwrap();
        store.ingest(d, 2, &scrape(100, 9, 90), &none()).await.unwrap();
        store.ingest(d, 3, &scrape(100, 7, 70), &none()).await.unwrap();
        store.ingest(d, 4, &scrape(100, 10, 100), &none()).await.unwrap();
        let rows = store.daily(None, None).await.unwrap();
        assert_eq!((rows[0].totals.requests, rows[0].totals.bytes), (10, 100));
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn readings_of_a_process_unseen_for_two_days_are_pruned() {
        let store = baselined_store().await;
        let d = day("2026-10-22");
        store.ingest(d, 1_000_000, &scrape(100, 5, 50), &none()).await.unwrap();
        let mut idle = scrape(300, 1, 1);
        idle.paths.clear();
        store.ingest(d, 1_000_000 + 40 * 3600, &idle, &none()).await.unwrap();
        store.ingest(d, 1_000_000 + 47 * 3600, &scrape(200, 1, 10), &none()).await.unwrap();
        store.ingest(d, 1_000_000 + 49 * 3600, &scrape(300, 0, 0), &none()).await.unwrap();
        let processes: Vec<i64> =
            sqlx::query_scalar("SELECT DISTINCT start_time FROM series_last ORDER BY 1")
                .fetch_all(&store.pool)
                .await
                .unwrap();
        assert_eq!(processes, vec![200, 300]);
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn recent_activity_is_kept_and_pruned() {
        let store = baselined_store().await;
        let d = day("2026-10-22");
        store.ingest(d, 1_000_000, &scrape(100, 5, 50), &none()).await.unwrap();
        store.ingest(d, 1_000_060, &scrape(100, 9, 90), &none()).await.unwrap();
        let line = store.recent_rows(0).await.unwrap();
        assert_eq!(
            line.iter().map(|(ts, _, t)| (*ts, t.requests)).collect::<Vec<_>>(),
            vec![(1_000_000, 5), (1_000_060, 4)]
        );
        assert_eq!(store.recent_rows(1_000_030).await.unwrap().len(), 1);
        store.ingest(d, 1_000_000 + 49 * 3600, &scrape(100, 10, 100), &none()).await.unwrap();
        assert_eq!(store.recent_rows(0).await.unwrap().len(), 1);
        assert_eq!(store.path_totals(None, None).await.unwrap()[0].1.requests, 10);
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn whole_files_and_item_totals_are_stored() {
        let store = baselined_store().await;
        let sizes = FileIndex { sizes: HashMap::from([("/assets/10Aa00/x.tif".to_string(), 1_000u64)]), ..Default::default() };
        let mut s = scrape(100, 4, 2_500);
        s.paths.insert("/docs/a.pdf".into(), Totals { requests: 1, bytes: 10, ..Default::default() });
        store.ingest(day("2026-10-22"), 1, &s, &sizes).await.unwrap();
        let by_path = store.path_totals(None, None).await.unwrap();
        let tif = by_path.iter().find(|p| p.0.ends_with("x.tif")).unwrap();
        assert_eq!(tif.1.whole_files(), 2.5);
        let codes = store.code_days(None, None).await.unwrap();
        let find = |c: &str| codes.iter().find(|r| r.1 == c).map(|r| r.3.requests);
        assert_eq!((find("10Aa00"), find("~docs/a.pdf")), (Some(4), Some(1)));
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn scrapes_in_one_hour_sum_into_one_hourly_bucket() {
        let store = baselined_store().await;
        let d = day("2026-10-22");
        let ten = 1_792_663_200; // 2026-10-22 10:00 UTC
        store.ingest(d, ten + 60, &scrape(100, 5, 50), &none()).await.unwrap();
        store.ingest(d, ten + 1_800, &scrape(100, 8, 80), &none()).await.unwrap();
        store.ingest(d, ten + 3_600 + 5, &scrape(100, 9, 90), &none()).await.unwrap();
        let rows = store.code_buckets(Resolution::Hour, Some(d), Some(d)).await.unwrap();
        let got: Vec<_> = rows.iter().map(|(h, c, _, t)| (h.format("%H:%M").to_string(), c.as_str(), t.requests, t.bytes)).collect();
        assert_eq!(got, vec![("10:00".into(), "10Aa00", 8, 80), ("11:00".into(), "10Aa00", 1, 10)]);
        // Hourly totals outlive the per-scrape detail.
        store.ingest(d, ten + 72 * 3600, &scrape(100, 9, 90), &none()).await.unwrap();
        assert!(store.recent_rows(0).await.unwrap().is_empty());
        assert_eq!(store.code_buckets(Resolution::Hour, None, None).await.unwrap().len(), 2);
        assert!(store.code_buckets(Resolution::Hour, Some(day("2026-10-23")), None).await.unwrap().is_empty());
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn daily_totals_sum_into_weeks_starting_monday() {
        let store = baselined_store().await;
        // 2026-10-18 is a Sunday, 2026-10-19 and 2026-10-25 a Monday and the Sunday after it.
        store.ingest(day("2026-10-18"), 1, &scrape(100, 2, 20), &none()).await.unwrap();
        store.ingest(day("2026-10-19"), 2, &scrape(100, 5, 50), &none()).await.unwrap();
        store.ingest(day("2026-10-25"), 3, &scrape(100, 6, 60), &none()).await.unwrap();
        let rows = store.code_buckets(Resolution::Week, None, None).await.unwrap();
        let got: Vec<_> = rows.iter().map(|(w, _, _, t)| (w.format("%Y-%m-%d").to_string(), t.requests)).collect();
        assert_eq!(got, vec![("2026-10-12".into(), 2), ("2026-10-19".into(), 4)]);
        let days = store.code_buckets(Resolution::Day, Some(day("2026-10-19")), None).await.unwrap();
        assert_eq!(days.len(), 2);
        assert_eq!(store.first_day().await.unwrap(), Some(day("2026-10-18")));
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn totals_at_the_largest_counter_hold_and_later_scrapes_still_count() {
        let store = baselined_store().await;
        let d = day("2026-10-22");
        let max = MAX_COUNTER;
        let mut s = scrape(100, max, max);
        s.paths.insert("/assets/10Aa00/y.tif".into(), Totals { requests: max, bytes: max, ..Default::default() });
        store.ingest(d, 1, &s, &none()).await.unwrap();
        // The proxy restarted: every reading is new and is added to totals already at the largest value.
        store.ingest(d, 2, &scrape(200, 5, 50), &none()).await.unwrap();
        store.ingest(d, 3, &scrape(200, 7, 70), &none()).await.unwrap();
        let x = store.daily(None, None).await.unwrap().into_iter().find(|r| r.path.ends_with("x.tif")).unwrap();
        assert_eq!((x.totals.requests, x.totals.bytes), (max, max));
        store.ingest(day("2026-10-23"), 4, &scrape(200, 9, 90), &none()).await.unwrap();
        let totals = store.path_totals(None, None).await.unwrap();
        assert_eq!(totals.iter().find(|p| p.0.ends_with("x.tif")).unwrap().1.requests, max);
        let codes = store.code_days(None, None).await.unwrap();
        assert!(codes.iter().all(|c| c.3.requests <= max));
        assert_eq!(store.last_ingest().await.unwrap(), Some(4));
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn values_from_the_metrics_are_data_never_sql() {
        let store = baselined_store().await;
        let hostile = "/assets/10Aa00/'; DROP TABLE daily; --\" \\ %s $1 ☃.tif";
        let mut s = Scrape { start_time: Some(1), ..Default::default() };
        s.paths.insert(hostile.into(), Totals { requests: 3, bytes: 30, ..Default::default() });
        store.ingest(day("2026-10-22"), 1, &s, &none()).await.unwrap();
        let rows = store.daily(None, None).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].path, hostile);
        assert_eq!(rows[0].totals.requests, 3);
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn two_scrapes_arriving_together_count_once() {
        let store = baselined_store().await;
        let s = scrape(100, 7, 70);
        let sizes = none();
        let (a, b) = tokio::join!(
            store.ingest(day("2026-10-22"), 1, &s, &sizes),
            store.ingest(day("2026-10-22"), 1, &s, &sizes)
        );
        a.unwrap();
        b.unwrap();
        assert_eq!(store.daily(None, None).await.unwrap()[0].totals.requests, 7);
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn the_schema_holds_no_identifying_columns() {
        let store = test_store().await;
        let columns: Vec<(String, String)> = sqlx::query_as(
            "SELECT table_name::text, column_name::text FROM information_schema.columns
             WHERE table_schema = current_schema() AND table_name <> '_sqlx_migrations'
             ORDER BY table_name, ordinal_position",
        )
        .fetch_all(&store.pool)
        .await
        .unwrap();
        let allowed = [
            "day", "hour", "path", "requests", "bytes", "equiv", "code", "format", "ts", "key", "value", "start_time",
        ];
        for (table, column) in &columns {
            assert!(allowed.contains(&column.as_str()), "{table}.{column} is not an allowed column");
        }
        let mut tables: Vec<&str> = columns.iter().map(|(t, _)| t.as_str()).collect();
        tables.dedup();
        assert_eq!(
            tables,
            vec!["daily", "daily_code", "hourly_code", "meta", "proxy_process", "recent", "series_last"]
        );
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn every_connection_has_a_statement_timeout() {
        let store = test_store().await;
        let timeout: String = sqlx::query_scalar("SHOW statement_timeout").fetch_one(&store.pool).await.unwrap();
        assert_eq!(timeout, "15s");
        let slow = sqlx::query("SELECT pg_sleep(20)").execute(&store.pool).await;
        assert!(slow.is_err(), "a 20 second query must be cancelled");
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn the_database_refuses_negative_counts() {
        let store = test_store().await;
        let bad = sqlx::query("INSERT INTO daily (day, path, requests, bytes) VALUES ('2026-10-22', '/x', -1, 0)")
            .execute(&store.pool)
            .await;
        assert!(bad.is_err());
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn the_migration_splits_the_documents_of_each_day_by_file() {
        let store = store_before(20261029090841).await;
        let long = format!("/docs/{}.pdf", "k".repeat(70));
        let paths = [
            ("2026-10-21", "/docs/lantern_manual.pdf", 2),
            ("2026-10-22", "/docs/lantern_manual.pdf", 3),
            ("2026-10-22", "/docs/kite_guide.csv", 5),
            ("2026-10-22", long.as_str(), 1),
            ("2026-10-22", "/assets/10Aa00/x.tif", 7),
        ];
        for (d, path, n) in paths {
            sqlx::query("INSERT INTO daily (day, path, requests, bytes, equiv) VALUES ($1, $2, $3, $3 * 10, 0)")
                .bind(day(d))
                .bind(path)
                .bind(n)
                .execute(&store.pool)
                .await
                .unwrap();
        }
        let before = [("2026-10-21", "~docs", 2), ("2026-10-22", "~docs", 9), ("2026-10-22", "10Aa00", 7), ("2026-10-23", "~docs", 4)];
        for (d, code, n) in before {
            sqlx::query("INSERT INTO daily_code (day, code, requests, bytes, equiv) VALUES ($1, $2, $3, $3 * 10, 0)")
                .bind(day(d))
                .bind(code)
                .bind(n)
                .execute(&store.pool)
                .await
                .unwrap();
        }
        store.migrate(&FileIndex::default()).await.unwrap();
        let mut got: Vec<_> = store
            .code_days(None, None)
            .await
            .unwrap()
            .into_iter()
            .map(|(d, c, _, t)| (d, c, t.requests, t.bytes))
            .collect();
        got.sort();
        let row = |d: &str, c: &str, n: u64| (d.to_string(), c.to_string(), n, n * 10);
        assert_eq!(
            got,
            vec![
                row("2026-10-21", "~docs/lantern_manual.pdf", 2),
                row("2026-10-22", "10Aa00", 7),
                row("2026-10-22", "~docs", 1),
                row("2026-10-22", "~docs/kite_guide.csv", 5),
                row("2026-10-22", "~docs/lantern_manual.pdf", 3),
                // A day with no rows in `daily` to split it by keeps its total.
                row("2026-10-23", "~docs", 4),
            ]
        );
        let keys: Vec<_> = paths.iter().map(|(d, p, _)| (d.to_string(), code_key(p))).collect();
        assert!(keys.iter().all(|k| got.iter().any(|g| (&g.0, &g.1) == (&k.0, &k.1))), "the keys are those of code_key");
    }

    fn kite_files() -> FileIndex {
        FileIndex {
            media_types: HashMap::from([
                ("/assets/kites-a/tail.laz".to_string(), "application/vnd.laszip".to_string()),
                ("/assets/kites-a/flight_notes.pdf".to_string(), "application/pdf".to_string()),
                ("/assets/kites-b/sail.jpeg".to_string(), "image/jpeg".to_string()),
            ]),
            ..Default::default()
        }
    }

    /// One kite item whose catalog format is LAZ, holding a laz and a pdf.
    fn kite_items() -> Vec<crate::stats::ItemInfo> {
        vec![crate::stats::ItemInfo {
            code: "kites-a".into(),
            title: "Kites".into(),
            files_by_format: BTreeMap::from([("LAZ".to_string(), 1), ("PDF".to_string(), 1)]),
            facets: crate::stats::Facets { format: "LAZ".into(), ..Default::default() },
        }]
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn hour_and_day_series_count_each_file_under_its_own_format() {
        use crate::stats::{series, series_window, Dim, Filters};
        let store = baselined_store().await;
        let d = day("2026-10-22");
        let ten = 1_792_663_200; // 2026-10-22 10:00 UTC
        let mut s = Scrape { start_time: Some(1), ..Default::default() };
        for path in ["/assets/kites-a/tail.laz", "/assets/kites-a/flight_notes.pdf"] {
            s.paths.insert(path.into(), Totals { requests: 1, bytes: 10, ..Default::default() });
        }
        store.ingest(d, ten + 60, &s, &kite_files()).await.unwrap();

        let now = d.and_hms_opt(23, 0, 0).unwrap();
        let items = kite_items();
        let by_format = |rows: &[_], resolution, filters: &Filters| {
            let window = series_window(Some(resolution), Some(d), Some(d), None, now).unwrap();
            let s = series(rows, &items, Dim::Format, filters, 7, &window);
            s.groups.into_iter().map(|g| (g.name, g.requests.iter().sum::<u64>())).collect::<Vec<_>>()
        };
        let both = vec![("LAZ".to_string(), 1), ("PDF".to_string(), 1)];
        let pdf = Filters::from_lookup(|k| (k == "format").then_some("PDF"));
        for resolution in [Resolution::Hour, Resolution::Day, Resolution::Week] {
            let rows = store.code_buckets(resolution, Some(d), Some(d)).await.unwrap();
            assert_eq!(by_format(&rows, resolution, &Filters::default()), both, "{resolution:?}");
            assert_eq!(by_format(&rows, resolution, &pdf), vec![("PDF".to_string(), 1)], "{resolution:?}");
        }
        let today: Vec<_> = store.code_days(None, None).await.unwrap().into_iter().map(|r| (r.1, r.2, r.3.requests)).collect();
        assert_eq!(today, vec![("kites-a".into(), "LAZ".into(), 1), ("kites-a".into(), "PDF".into(), 1)]);
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn the_migration_splits_days_by_file_format_and_reads_a_mixed_hour_as_unknown() {
        let store = store_before(20261029130002).await;
        let paths = [
            ("2026-10-22", "/assets/kites-a/tail.laz", 2),
            ("2026-10-22", "/assets/kites-a/flight_notes.pdf", 1),
            ("2026-10-22", "/archives/kites-b.zip", 3),
            ("2026-10-22", "/assets/kites-b/sail.jpeg", 4),
            ("2026-10-22", "/docs/lantern_manual.pdf", 5),
        ];
        for (d, path, n) in paths {
            sqlx::query("INSERT INTO daily (day, path, requests, bytes, equiv) VALUES ($1, $2, $3, $3 * 10, 0)")
                .bind(day(d))
                .bind(path)
                .bind(n)
                .execute(&store.pool)
                .await
                .unwrap();
        }
        let before = [
            ("2026-10-22", "kites-a", 3),
            ("2026-10-22", "kites-b", 7),
            ("2026-10-22", "~docs/lantern_manual.pdf", 5),
            ("2026-10-23", "kites-a", 6),
        ];
        for (d, code, n) in before {
            sqlx::query("INSERT INTO daily_code (day, code, requests, bytes, equiv) VALUES ($1, $2, $3, $3 * 10, 0)")
                .bind(day(d))
                .bind(code)
                .bind(n)
                .execute(&store.pool)
                .await
                .unwrap();
        }
        let hours = [
            ("2026-10-22 09:00:00+00", "kites-a", 3),
            ("2026-10-22 09:00:00+00", "kites-b", 7),
            ("2026-10-22 23:00:00+00", "~docs/lantern_manual.pdf", 5),
            ("2026-10-23 00:00:00+00", "kites-a", 6),
        ];
        for (h, code, n) in hours {
            sqlx::query("INSERT INTO hourly_code (hour, code, requests, bytes, equiv) VALUES ($1::timestamptz, $2, $3, $3 * 10, 0)")
                .bind(h)
                .bind(code)
                .bind(n)
                .execute(&store.pool)
                .await
                .unwrap();
        }

        store.migrate(&kite_files()).await.unwrap();
        let row = |d: &str, c: &str, f: &str, n: u64| (d.to_string(), c.to_string(), f.to_string(), n, n * 10);
        let days: Vec<_> = store
            .code_days(None, None)
            .await
            .unwrap()
            .into_iter()
            .map(|(d, c, f, t)| (d, c, f, t.requests, t.bytes))
            .collect();
        assert_eq!(
            days,
            vec![
                row("2026-10-22", "kites-a", "LAZ", 2),
                row("2026-10-22", "kites-a", "PDF", 1),
                // A jpeg is named by its media type from the catalog, as the API names it.
                row("2026-10-22", "kites-b", "JPG", 4),
                row("2026-10-22", "kites-b", "ZIP", 3),
                row("2026-10-22", "~docs/lantern_manual.pdf", "PDF", 5),
                // A day with no rows in `daily` to split it by keeps its total.
                row("2026-10-23", "kites-a", "Unknown", 6),
            ]
        );
        let hours: Vec<_> = store
            .code_buckets(Resolution::Hour, None, None)
            .await
            .unwrap()
            .into_iter()
            .map(|(h, c, f, t)| (h.format("%d %H").to_string(), c, f, t.requests, t.bytes))
            .collect();
        assert_eq!(
            hours,
            vec![
                row("22 09", "kites-a", "Unknown", 3),
                row("22 09", "kites-b", "Unknown", 7),
                row("22 23", "~docs/lantern_manual.pdf", "PDF", 5),
                row("23 00", "kites-a", "Unknown", 6),
            ]
        );

        // Labelling again finds nothing left to label and changes nothing.
        store.label_formats(&FileIndex::default()).await.unwrap();
        assert_eq!(store.code_days(None, None).await.unwrap().len(), 6);
    }

    const SEED: &str = include_str!("../scripts/seed_stats.sql");

    /// Run the development seed into the store's schema, allowing the test database by name.
    async fn seed(store: &Store) {
        let db: String = sqlx::query_scalar("SELECT current_database()").fetch_one(&store.pool).await.unwrap();
        let mut conn = store.pool.acquire().await.unwrap();
        sqlx::query("SELECT set_config('seed.database', $1, false)").bind(&db).execute(&mut *conn).await.unwrap();
        sqlx::raw_sql(SEED).execute(&mut *conn).await.expect("seed");
        sqlx::query("RESET seed.database").execute(&mut *conn).await.unwrap();
    }

    fn sum_by<K: Ord>(rows: impl IntoIterator<Item = (K, Totals)>) -> BTreeMap<K, (u64, u64, u64)> {
        let mut out = BTreeMap::new();
        for (k, t) in rows {
            let e: &mut (u64, u64, u64) = out.entry(k).or_default();
            *e = (e.0 + t.requests, e.1 + t.bytes, e.2 + t.equiv);
        }
        out
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn the_development_seed_fills_every_resolution_consistently_and_repeatably() {
        let store = baselined_store().await;
        sqlx::query("INSERT INTO series_last (start_time, path, requests, bytes) VALUES (7, '/assets/10Aa00/x.tif', 3, 30)")
            .execute(&store.pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO proxy_process (start_time, ts) VALUES (7, 1)").execute(&store.pool).await.unwrap();
        seed(&store).await;

        let weeks = store.code_buckets(Resolution::Week, None, None).await.unwrap();
        let days = store.code_buckets(Resolution::Day, None, None).await.unwrap();
        let hours = store.code_buckets(Resolution::Hour, None, None).await.unwrap();
        let recent = store.recent_rows(0).await.unwrap();
        assert!(!weeks.is_empty() && !days.is_empty() && !hours.is_empty() && !recent.is_empty());
        let first = store.first_day().await.unwrap().unwrap();
        let today = Utc::now().date_naive();
        assert!((today - first).num_days() >= 360, "about a year of history, from {first}");
        assert!(days.iter().map(|d| d.0.date()).max().unwrap() >= today.pred_opt().unwrap());
        assert!(days.iter().all(|d| d.3.equiv <= d.3.requests * 1000), "bytes never above whole files requested");

        let per_path = sum_by(store.daily(None, None).await.unwrap().into_iter().map(|r| (r.day, r.totals)));
        let per_code = sum_by(store.code_days(None, None).await.unwrap().into_iter().map(|r| (r.0, r.3)));
        assert_eq!(per_path, per_code, "daily_code is the sum of daily, day by day");
        let files = FileIndex::default();
        let per_key = sum_by(
            store.daily(None, None).await.unwrap().into_iter().map(|r| ((r.day, code_key(&r.path), files.format(&r.path)), r.totals)),
        );
        let stored = sum_by(store.code_days(None, None).await.unwrap().into_iter().map(|r| ((r.0, r.1, r.2), r.3)));
        assert_eq!(per_key, stored, "daily_code is summed under the keys code_key and the formats file_format give");
        assert!(stored.keys().any(|(_, c, _)| c.starts_with("~docs/")), "documents are seeded under their own keys");
        let by_day = sum_by(days.iter().map(|d| ((d.0.date(), d.1.clone(), d.2.clone()), d.3)));
        let by_hour = sum_by(hours.iter().map(|h| ((h.0.date(), h.1.clone(), h.2.clone()), h.3)));
        assert_eq!(by_hour, by_day, "hourly_code sums to daily_code per day, code and format over the whole year");
        let months_back = today - chrono::Days::new(100);
        let window = store.code_buckets(Resolution::Hour, Some(months_back), Some(months_back + chrono::Days::new(4))).await.unwrap();
        assert!(!window.is_empty(), "a short window months back has hours to draw");
        let in_weeks = sum_by(weeks.iter().map(|w| ((), w.3)));
        assert_eq!(in_weeks, sum_by(days.iter().map(|d| ((), d.3))), "weeks hold every day");

        let paths = store.path_totals(None, None).await.unwrap();
        assert!(paths.iter().any(|p| p.0.starts_with("/archives/") && p.0.ends_with(".zip")));
        assert!(paths.iter().any(|p| p.0.starts_with("/assets/")));

        seed(&store).await;
        assert_eq!(store.code_buckets(Resolution::Day, None, None).await.unwrap(), days);
        assert_eq!(store.code_buckets(Resolution::Hour, None, None).await.unwrap(), hours);
        assert_eq!(store.recent_rows(0).await.unwrap(), recent);

        let kept: (i64, i64, i64) = sqlx::query_as(
            "SELECT (SELECT COUNT(*) FROM series_last), (SELECT requests FROM series_last), (SELECT COUNT(*) FROM proxy_process)",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(kept, (1, 3, 1), "the seed leaves the scrape baseline alone");
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn the_development_seed_refuses_a_database_not_named_stats() {
        let store = test_store().await;
        let mut conn = store.pool.acquire().await.unwrap();
        assert!(sqlx::raw_sql(SEED).execute(&mut *conn).await.is_err());
        sqlx::query("ROLLBACK").execute(&mut *conn).await.unwrap();
        let rows: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM daily").fetch_one(&mut *conn).await.unwrap();
        assert_eq!(rows, 0);
    }

    #[tokio::test]
    #[ignore = "needs STATS_TEST_DATABASE_URL"]
    async fn a_large_scrape_is_applied_in_one_pass() {
        let store = baselined_store().await;
        let mut s = Scrape { start_time: Some(1), ..Default::default() };
        for i in 0..20_000 {
            s.paths.insert(
                format!("/assets/{:02}Aa00/file_{i}.jpg", i % 90),
                Totals { requests: 1 + (i % 3) as u64, bytes: 1000, ..Default::default() },
            );
        }
        let started = std::time::Instant::now();
        store.ingest(day("2026-10-22"), 1, &s, &none()).await.unwrap();
        assert!(started.elapsed() < Duration::from_secs(20));
        assert_eq!(store.path_totals(None, None).await.unwrap().len(), 20_000);
    }
}
