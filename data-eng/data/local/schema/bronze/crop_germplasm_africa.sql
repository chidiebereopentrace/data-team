CREATE SCHEMA IF NOT EXISTS raw_dev;
CREATE TABLE IF NOT EXISTS raw_dev.crop_germplasm_africa (
  id BIGINT,
  taxon TEXT,
  objectid BIGINT,
  geography GEOGRAPHY
);
