-- Synthetic download history for the local statistics database, so the charts have something to
-- show in development. Never run this against production.
--
--   docker exec -i blatten-stats-db psql -U stats -d stats -v ON_ERROR_STOP=1 < scripts/seed_stats.sql
--
-- It refuses any database not named `stats`; another name is allowed only by setting
-- `seed.database` to it first, as the store's tests do for `stats_test`.
--
-- The history is drawn per quarter-hour scrape for a year ending now, then summed into the tables
-- the way the scraper fills them: `daily`, `daily_code` and `hourly_code` for the whole year, so
-- every resolution of any window agrees, and `recent` for the last 24 hours. The seed owns those
-- windows: every row in them is replaced, so running it again gives the same history.
-- `series_last`, `proxy_process` and `meta` are never touched, so the next real scrape still
-- counts from its own baseline.
--
-- The values are drawn from a hash of the path and the time, not from random(), so they repeat.

BEGIN;

DO $$
DECLARE
    allowed text := COALESCE(NULLIF(current_setting('seed.database', true), ''), 'stats');
BEGIN
    IF current_database() <> allowed THEN
        RAISE EXCEPTION 'seed_stats.sql only runs against the local database %, not %', allowed, current_database();
    END IF;
END
$$;

-- Paths and sizes from the local catalog (stac/items.ndjson), with requests per day at the
-- busiest hours.
-- `format` is the extension upper-cased, which is what `file_format` gives these paths.
CREATE TEMP TABLE seed_path (
    path    text PRIMARY KEY,
    code    text NOT NULL,
    size    bigint NOT NULL,
    per_day float8 NOT NULL,
    format  text GENERATED ALWAYS AS (upper(substring(path FROM '\.([^./]+)$'))) STORED
)
ON COMMIT DROP;
INSERT INTO seed_path (path, code, size, per_day) VALUES
    ('/archives/01Aa02.zip', '01Aa02', 40383194, 3),
    ('/assets/01Aa02/20250515T163515Z.jpg', '01Aa02', 1569915, 14),
    ('/assets/01Aa02/20250516T063515Z.jpg', '01Aa02', 1687649, 9),
    ('/archives/02Aa00.zip', '02Aa00', 588233777, 1.5),
    ('/assets/02Aa00/Blatten_Birchbach-FlexCam-20250528_1515_1530.mp4', '02Aa00', 13731921, 22),
    ('/archives/04Ab01.zip', '04Ab01', 1470452055, 0.6),
    ('/assets/04Ab01/20250527T130018Z.png', '04Ab01', 65913240, 4),
    ('/archives/06Ea02.zip', '06Ea02', 86891172, 1),
    ('/assets/06Ea02/GP-AIM-24_20250531-1000_43_918_0.07-DTM.jpg', '06Ea02', 424600, 6),
    ('/archives/13Da01.zip', '13Da01', 939512482, 2.5),
    ('/assets/13Da01/orthophoto_20250530-190300_100mmPP-0-0.tif', '13Da01', 5726798, 8),
    ('/assets/13Da01/orthophoto_20250530-190300_100mmPP-0-1.tif', '13Da01', 225427998, 5),
    ('/archives/13Ia17.zip', '13Ia17', 1025014510, 1.2),
    ('/assets/13Ia17/DEM_20250615-044500_100mmPP-0-1.tif', '13Ia17', 205620062, 3),
    ('/archives/14Da01.zip', '14Da01', 4292064880, 0.8),
    ('/assets/14Da01/orthophoto_20250523-102700_50mmPP-0-0.tif', '14Da01', 98927482, 3),
    ('/archives/14Ka01.zip', '14Ka01', 3483306622, 0.5),
    ('/assets/14Ka01/LIDAR_250523_LV95_NF02_2624000_1137500.laz', '14Ka01', 429885, 2),
    ('/archives/14La02.zip', '14La02', 617603582, 0.7),
    ('/assets/14La02/klnh_2.jpg', '14La02', 16819284, 5),
    ('/archives/17Pa00.zip', '17Pa00', 3352357, 2),
    ('/assets/17Pa00/17_LakeLevel_Geoazimut.csv', '17Pa00', 2699024, 7),
    ('/archives/19Oa00.zip', '19Oa00', 922800, 1.5),
    ('/assets/19Oa00/19_LonzaRiverDischarge_Blatten.csv', '19Oa00', 260870, 6),
    ('/docs/A50039_Blatten4Science_TechnicalDocumentation.pdf', '~docs/A50039_Blatten4Science_TechnicalDocumentation.pdf', 4000000, 10),
    ('/docs/dataset_overview.csv', '~docs/dataset_overview.csv', 60000, 12);

-- UTC times without a zone throughout; `now_slot` is the last quarter hour that has passed.
CREATE TEMP TABLE seed_window ON COMMIT DROP AS
SELECT now_slot,
       now_slot::date - 364 AS first_day,
       now_slot - interval '24 hours' AS first_recent
FROM (SELECT date_trunc('hour', timezone('UTC', now()))
             + floor(extract(minute FROM timezone('UTC', now())) / 15) * interval '15 minutes' AS now_slot) n;

-- One row per scrape and path with something downloaded. The rate follows the hour of day, is
-- lower at weekends, grows over the year, and has a few busy days and one busy afternoon.
CREATE TEMP TABLE seed_scrape ON COMMIT DROP AS
WITH slot AS (
    SELECT t
    FROM seed_window w, generate_series(w.first_day::timestamp, w.now_slot, interval '15 minutes') t
),
rate AS (
    SELECT s.t, p.path, p.size,
           p.per_day / 96.0
           * (0.15 + 1.6 * greatest(0, cos((extract(hour FROM s.t) - 11) * pi() / 12)))
           * CASE WHEN extract(isodow FROM s.t) >= 6 THEN 0.45 ELSE 1 END
           * (0.5 + 0.5 * (s.t::date - w.first_day) / 364.0)
           * CASE WHEN (s.t::date - date '2000-01-01') % 47 = 5 THEN 6 ELSE 1 END
           * CASE WHEN s.t::date = w.now_slot::date - 3 AND extract(hour FROM s.t) BETWEEN 14 AND 15 THEN 12 ELSE 1 END
           AS expected,
           (abs(hashtextextended(p.path || '@' || s.t, 0)) % 1000000) / 1000000.0 AS u,
           (abs(hashtextextended(p.path || '#' || s.t, 0)) % 1000000) / 1000000.0 AS part
    FROM slot s CROSS JOIN seed_path p CROSS JOIN seed_window w
),
counted AS (
    SELECT t, path, size, floor(expected + u)::bigint AS requests, part FROM rate
)
SELECT t, path, requests,
       floor(requests * size * (0.4 + 0.6 * part))::bigint AS bytes,
       floor(floor(requests * size * (0.4 + 0.6 * part)) * 1000 / size)::bigint AS equiv
FROM counted
WHERE requests > 0;

DELETE FROM daily WHERE day >= (SELECT first_day FROM seed_window);
DELETE FROM daily_code WHERE day >= (SELECT first_day FROM seed_window);
DELETE FROM hourly_code WHERE hour >= (SELECT first_day::timestamp AT TIME ZONE 'UTC' FROM seed_window);
DELETE FROM recent WHERE ts >= (SELECT extract(epoch FROM first_recent AT TIME ZONE 'UTC')::bigint FROM seed_window);

INSERT INTO daily (day, path, requests, bytes, equiv)
SELECT t::date, path, SUM(requests), SUM(bytes), SUM(equiv)
FROM seed_scrape
GROUP BY 1, 2;

INSERT INTO daily_code (day, code, format, requests, bytes, equiv)
SELECT s.t::date, p.code, p.format, SUM(s.requests), SUM(s.bytes), SUM(s.equiv)
FROM seed_scrape s JOIN seed_path p USING (path)
GROUP BY 1, 2, 3;

INSERT INTO hourly_code (hour, code, format, requests, bytes, equiv)
SELECT date_trunc('hour', s.t) AT TIME ZONE 'UTC', p.code, p.format, SUM(s.requests), SUM(s.bytes), SUM(s.equiv)
FROM seed_scrape s JOIN seed_path p USING (path)
GROUP BY 1, 2, 3;

INSERT INTO recent (ts, path, requests, bytes)
SELECT extract(epoch FROM s.t AT TIME ZONE 'UTC')::bigint, s.path, s.requests, s.bytes
FROM seed_scrape s, seed_window w
WHERE s.t >= w.first_recent;

COMMIT;
