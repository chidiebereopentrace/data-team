"""Run dbt in Composer — point bash_command or KubernetesPodOperator at your dbt project path."""

from datetime import datetime
from pathlib import Path

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

DASHBOARD = Path(__file__).resolve().parents[3] / "Continental OTA Dashboard Codebase"

with DAG(
    dag_id="dbt_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["transformations", "dbt"],
):
    dbt_run = EmptyOperator(task_id="dbt_run")
    dbt_test = EmptyOperator(task_id="dbt_test")
    ota_dashboard = BashOperator(
        task_id="ota_dashboard",
        bash_command=f'bash "{DASHBOARD / "build.sh"}"',
    )
    _ = dbt_run >> dbt_test >> ota_dashboard
