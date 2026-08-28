CREATE SCHEMA IF NOT EXISTS raw_dev;
CREATE TABLE IF NOT EXISTS raw_dev.geoboundaries_admin0_africa (
  country_iso2 TEXT,
  country_iso3 TEXT,
  country_name TEXT,
  geog GEOGRAPHY,
  min_lat DOUBLE PRECISION,
  max_lat DOUBLE PRECISION,
  min_lng DOUBLE PRECISION,
  max_lng DOUBLE PRECISION,
  source TEXT,
  loaded_at TIMESTAMPTZ
);
