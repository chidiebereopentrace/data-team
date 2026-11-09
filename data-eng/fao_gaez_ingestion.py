import datetime as dt
import pandas as pd
import requests
from google.cloud import bigquery

# Configuration
PROJECT_ID = "opentrace-prod-5ga4"
DATASET_ID = "landing"
TABLE_ID = "fao_gaez_agroinformatics_layers"
# Use the direct data/CSV file link if available, or fetch text/HTML safely
SOURCE_URL = "https://www.fao.org/gaez/en"

def extract() -> pd.DataFrame:
    # 1. Fetch content from source
    resp = requests.get(SOURCE_URL, timeout=120)
    resp.raise_for_status()
    
    # Check if the URL returns JSON or HTML/CSV
    try:
        data = resp.json()
        df = pd.json_normalize(data)
    except Exception:
        # If it's a webpage or table, wrap it into a basic dataframe record
        df = pd.DataFrame([{
            "page_content": resp.text[:5000],  # Save snippet or adjust for CSV
            "status_code": resp.status_code
        }])
    
    # 2. Add required OpenTrace metadata columns
    df["source_url"] = SOURCE_URL
    df["_ingested_at"] = dt.datetime.now(dt.timezone.utc)
    return df

def load(df: pd.DataFrame) -> None:
    client = bigquery.Client.from_service_account_json("data/local/keys/opentrace-bq-key.json", project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )
    
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    
    n = client.get_table(table_ref).num_rows
    print(f"Loaded {len(df)} rows; table now has {n} rows -> {table_ref}")

if __name__ == "__main__":
    load(extract())