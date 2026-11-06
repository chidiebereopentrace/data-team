# WFP country food prices — local CSV upload

Companion notebook: [`wfp_hbx_46_data_upload.ipynb`](./wfp_hbx_46_data_upload.ipynb)

The CSVs are the 46 Humdata country files already saved under `data-eng/ingestion/data_csv`. The notebook reads those files. It does not download them.

## Grain

One table per file. The table name is `hdx_wfp_food_prices_<iso3>`, from `wfp_food_prices_<iso3>.csv`.

One row is one market price, keyed by `date`, `market_id`, `commodity_id`, `pricetype`, and `unit`.

The publisher file has no country column. `raw_dev` adds `country_iso3` and `country_name` from the filename.

## Columns

Landing (`landing_dev`) keeps the publisher columns:

`date`, `admin1`, `admin2`, `market`, `market_id`, `latitude`, `longitude`, `category`, `commodity`, `commodity_id`, `unit`, `priceflag`, `pricetype`, `currency`, `price`, `usdprice`

plus `_source`, `_run_id`, `_ingested_at`, `_ingest_date`, `_source_url`, `_record_hash`.

Raw (`raw_dev`) is that table after the `hdx_wfp_prices` transform (typed date, ids, price, lat/lon) plus `country_iso3` and `country_name`.

A second CSV line that starts with `#` is the HXL tag row. The transform drops it before `read_csv`.

## Run

Put the files in `data-eng/ingestion/data_csv`. A file must be named `wfp_food_prices_<iso3>.csv` and its first line must start with `date,`. Any other file is skipped.

Open the notebook and run all cells. A missing folder, or a folder with no matching CSV, prints a message and loads nothing.

Each load uses `replace_partition` on `landing_dev` and `raw_dev`. Production datasets `landing` and `raw` are not written.
