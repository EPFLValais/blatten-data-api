-- Documents were summed per day under one key, `~docs`. Each now has a key of its own,
-- `~docs/<name>` with the name below `/docs/`, or `~docs` when that is longer than 64 characters,
-- as `code_key` gives them. The days already summed are split again from `daily`, which holds the
-- same totals per path. `hourly_code` has no per-path detail to split by, so its `~docs` rows stay
-- and read as documents together.
CREATE TEMP TABLE document_day ON COMMIT DROP AS
SELECT day,
       CASE WHEN name <> '' AND char_length(name) <= 58 THEN '~docs/' || name ELSE '~docs' END AS code,
       requests, bytes, equiv
FROM (SELECT day, regexp_replace(path, '^/*docs/', '') AS name, requests, bytes, equiv
      FROM daily
      WHERE path ~ '^/*docs/') d;

DELETE FROM daily_code WHERE code = '~docs' AND day IN (SELECT day FROM document_day);

INSERT INTO daily_code (day, code, requests, bytes, equiv)
SELECT day, code,
       LEAST(SUM(requests), 9223372036854775807)::bigint,
       LEAST(SUM(bytes), 9223372036854775807)::bigint,
       LEAST(SUM(equiv), 9223372036854775807)::bigint
FROM document_day
GROUP BY day, code;
