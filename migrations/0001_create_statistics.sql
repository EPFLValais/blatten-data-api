-- Download statistics for the file proxy.
--
-- Only counts per file path and day are stored. There is no column for a client address, a user
-- agent, a session or anything else that says who made a request, and none may be added without
-- changing the privacy position of the service.

-- Requests and bytes per file and UTC day. `equiv` is the bytes sent divided by the file's size,
-- in thousandths: a file fetched in pieces adds up to one whole file.
CREATE TABLE daily (
    day      date   NOT NULL,
    path     text   NOT NULL CHECK (char_length(path) BETWEEN 1 AND 2048),
    requests bigint NOT NULL CHECK (requests >= 0),
    bytes    bigint NOT NULL CHECK (bytes >= 0),
    equiv    bigint NOT NULL DEFAULT 0 CHECK (equiv >= 0),
    PRIMARY KEY (day, path)
);

-- The same totals summed per catalog item code (`~docs` for documents, `~other` for anything else),
-- so charts over long periods do not read every file row.
CREATE TABLE daily_code (
    day      date   NOT NULL,
    code     text   NOT NULL CHECK (char_length(code) BETWEEN 1 AND 64),
    requests bigint NOT NULL CHECK (requests >= 0),
    bytes    bigint NOT NULL CHECK (bytes >= 0),
    equiv    bigint NOT NULL DEFAULT 0 CHECK (equiv >= 0),
    PRIMARY KEY (day, code)
);

-- The last reading of the proxy's cumulative counters, per path. The proxy restarts from zero, so
-- each new reading is turned into an increment against this one.
CREATE TABLE series_last (
    path     text   PRIMARY KEY CHECK (char_length(path) BETWEEN 1 AND 2048),
    requests bigint NOT NULL CHECK (requests >= 0),
    bytes    bigint NOT NULL CHECK (bytes >= 0)
);

-- Increments per scrape for the live view; rows older than 48 hours are removed on every scrape.
-- `ts` is unix seconds.
CREATE TABLE recent (
    ts       bigint NOT NULL,
    path     text   NOT NULL CHECK (char_length(path) BETWEEN 1 AND 2048),
    requests bigint NOT NULL CHECK (requests >= 0),
    bytes    bigint NOT NULL CHECK (bytes >= 0),
    PRIMARY KEY (ts, path)
);

-- Scalars: `start_time` of the proxy process last read, `last_ingest` unix time of the last scrape.
CREATE TABLE meta (
    key   text   PRIMARY KEY,
    value bigint NOT NULL
);
