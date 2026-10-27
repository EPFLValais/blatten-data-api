-- Requests and bytes per catalog item code and UTC hour, filled with `daily_code` by every scrape
-- and kept for good, so short bursts stay visible after the per-scrape detail in `recent` is gone.
-- `hour` is the start of the hour. Hours before this table existed have no rows.
CREATE TABLE hourly_code (
    hour     timestamptz NOT NULL,
    code     text        NOT NULL CHECK (char_length(code) BETWEEN 1 AND 64),
    requests bigint      NOT NULL CHECK (requests >= 0),
    bytes    bigint      NOT NULL CHECK (bytes >= 0),
    equiv    bigint      NOT NULL DEFAULT 0 CHECK (equiv >= 0),
    PRIMARY KEY (hour, code)
);
