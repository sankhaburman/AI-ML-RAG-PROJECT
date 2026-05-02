from airflow.decorators import dag, task
from datetime import datetime, timedelta
import os
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
import requests

# -----------------------------
# DAG Definition
# -----------------------------
@dag(
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mf-advisor-etl-pipeline"],
)
def mf_advisor_etl_dag():

    # Step 1: Ingest dataset from MFAPI
    @task(retries=3, retry_delay=timedelta(minutes=5))
    def ingest():
        try:
            os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
            response = requests.get(DATASET_URL, timeout=30)
            response.raise_for_status()  # Raise error if HTTP request failed
            with open(RAW_PATH, "wb") as f:
                f.write(response.content)
            print(f"✅ Dataset successfully saved to {RAW_PATH}")
            return RAW_PATH
        except Exception as e:
            print(f"❌ Failed to ingest dataset: {str(e)}")
            # Raising ensures Airflow marks this task as failed
            raise RuntimeError(f"Dataset ingestion failed: {e}") from e
    # Step 1: Initialize datastore
    initialize_datastore = SparkSubmitOperator(
        task_id="init_db",
        conn_id="my_spark_conn",
        application="include/scripts/mf_db_setup/mf_advisor_db_setup.py",
        execution_timeout=timedelta(minutes=30),
        retries=2,
        retry_delay=timedelta(minutes=10),
        verbose=True,
    )

    # Step 2: Dummy task (runs after init_db)
    dummy_task = EmptyOperator(
        task_id="post_init_dummy_task"
    )

    # ✅ Set dependency
    initialize_datastore >> dummy_task

    return initialize_datastore, dummy_task

# Register DAG
dag = mf_advisor_etl_dag()