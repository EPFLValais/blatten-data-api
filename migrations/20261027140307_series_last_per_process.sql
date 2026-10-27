-- The proxy runs as more than one process at a time (several replicas, or old and new side by side
-- during a rolling update), and each counts from zero. The last reading is therefore kept per
-- process, named by its `process_start_time_seconds`, so an increment is always taken against the
-- same process's previous reading.
ALTER TABLE series_last ADD COLUMN start_time bigint NOT NULL DEFAULT 0;
UPDATE series_last SET start_time = COALESCE((SELECT value FROM meta WHERE key = 'start_time'), 0);
ALTER TABLE series_last ALTER COLUMN start_time DROP DEFAULT;
ALTER TABLE series_last DROP CONSTRAINT series_last_pkey;
ALTER TABLE series_last ADD PRIMARY KEY (start_time, path);

-- When each process was last scraped, as unix seconds. The readings of a process not scraped for
-- 48 hours are removed.
CREATE TABLE proxy_process (
    start_time bigint PRIMARY KEY,
    ts         bigint NOT NULL
);
INSERT INTO proxy_process (start_time, ts)
SELECT s.value, COALESCE((SELECT value FROM meta WHERE key = 'last_ingest'), 0)
FROM meta s WHERE s.key = 'start_time';
DELETE FROM meta WHERE key = 'start_time';
