-- Day and hour totals were summed per item code only, so a chart grouped or filtered by format put
-- every file of an item under the item's format. They are now kept per code and file format, the
-- label `file_format` gives the path from the catalog's media type or its extension.
--
-- Rows summed before have the format '' until the API labels them on start (`label_formats` in
-- `src/stats_store.rs`): the days are summed again from `daily`, which holds the same totals per
-- path, and an hour takes the format of its day's downloads of the code when they are all of one
-- format, or `Unknown`. The label is not written here because it needs the catalog's media types.
ALTER TABLE daily_code ADD COLUMN format text NOT NULL DEFAULT '' CHECK (char_length(format) <= 16);
ALTER TABLE daily_code ALTER COLUMN format DROP DEFAULT;
ALTER TABLE daily_code DROP CONSTRAINT daily_code_pkey, ADD PRIMARY KEY (day, code, format);

ALTER TABLE hourly_code ADD COLUMN format text NOT NULL DEFAULT '' CHECK (char_length(format) <= 16);
ALTER TABLE hourly_code ALTER COLUMN format DROP DEFAULT;
ALTER TABLE hourly_code DROP CONSTRAINT hourly_code_pkey, ADD PRIMARY KEY (hour, code, format);
