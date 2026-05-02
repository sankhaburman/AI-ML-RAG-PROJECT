"""
Airflow DAG for Customer Churn Prediction Pipeline
--------------------------------------------------

This DAG orchestrates the ETL + ML pipeline for the Telco Customer Churn dataset.  
Steps included:
    1. Initialize datastore (DB setup with Spark job).
    2. Ingest dataset from GitHub into the raw zone.
    3. Transform & persist dataset into processed zone + Postgres.
    4. Perform feature engineering with Spark.
    5. Train ML model on processed data.

Schedule: Daily at midnight
"""

from airflow.decorators import dag, task
from datetime import datetime, timedelta
import os
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
import requests



# -----------------------------
# Configurations & Constants
# -----------------------------

# Remote dataset location (CSV from GitHub repo)
DATASET_URL = (
    "https://github.com/sankhaburman/datascience_project/raw/"
    "be4ddb0f22282af0e9e29151eb35f4112a9e8df7/telco-customer-churn.csv"
)

# Business date used for partitioning
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")

# Local file system paths (raw zone & processed zone)
RAW_PATH = f"include/raw-zone/{CURRENT_DATE}/telco-customer-churn.csv"
PROCESSED_PATH = f"include/processed-zone/{CURRENT_DATE}/"

# Metadata
DATA_SOURCE = "api"

# Database tables
RAW_DATA_DB_TABLE_NAME = "telco_churn_data_raw"
PROCESSED_DATA_DB_TABLE_NAME = "telco_customer_churn_processed"


def make_spark_path(path: str) -> str:
    """
    Ensures Spark can read/write using the correct file path format.
    If path is local, it prefixes with 'file://'.

    Args:
        path (str): Local or distributed path.

    Returns:
        str: Spark-compatible path.
    """
    if not path.startswith(("s3a://", "hdfs://", "file://")):
        abs_path = os.path.abspath(path)
        return f"file://{abs_path}"
    return path


# -----------------------------
# DAG Definition
# -----------------------------
@dag(
    start_date=datetime(2025, 1, 1),             # DAG will not run before this date
    schedule="@daily",                           # Runs every day at midnight
    catchup=False,                               # Skip backfilling missed runs
    tags=["customer_churn", "etl-pipeline"],     # Tags for Airflow UI filtering
)
def customer_churn_dag():
    """
    Main DAG for orchestrating the Telco Customer Churn pipeline.
    Orchestrates DB initialization, ingestion, transformation,
    feature engineering, and model training.
    """

    # Step 1: Initialize datastore (DB schema/tables for raw + processed data)
    initialize_datastore = SparkSubmitOperator(
        task_id="init_db",
        conn_id="my_spark_conn",
        application="include/scripts/prepare/db_init_job.py",
        application_args=[
            make_spark_path(RAW_PATH),   # Raw dataset path
            CURRENT_DATE,                # Business date
        ],
        execution_timeout=timedelta(minutes=30),
        retries=2,
        retry_delay=timedelta(minutes=10),
        verbose=True,
    )

    # Step 2: Ingest dataset from GitHub into raw zone
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
        
    # Step 3: Transform & persist data into processed zone + Postgres
    transform_and_persist = SparkSubmitOperator(
        task_id="enrichment",
        conn_id="my_spark_conn",
        application="include/scripts/enrich/transform_and_load_job.py",
        application_args=[
            make_spark_path(RAW_PATH),      # Input raw dataset
            make_spark_path(PROCESSED_PATH),# Output processed zone
            CURRENT_DATE,                   # Business date
            DATA_SOURCE,                    # Metadata column
        ],
        execution_timeout=timedelta(minutes=45),
        retries=2,
        retry_delay=timedelta(minutes=10),
        verbose=True,
    )

    # Step 4: Feature engineering (Spark job with JDBC to Postgres)
    feature_engineering = SparkSubmitOperator(
        task_id="feature_engineering",
        conn_id="my_spark_conn",
        application="/usr/local/airflow/include/scripts/feature_engineering/feature_engineer_job.py",
        packages="org.postgresql:postgresql:42.7.3",   # Auto-fetch JDBC driver
        application_args=[
            RAW_DATA_DB_TABLE_NAME,
        ],
        execution_timeout=timedelta(minutes=30),
        verbose=True,
    )

    # Step 5: Train ML model (Spark MLlib / sklearn job)
    train_model = SparkSubmitOperator(
        task_id="train_model",
        conn_id="my_spark_conn",
        application="/usr/local/airflow/include/scripts/train_model/train_model_job.py",
        packages="org.postgresql:postgresql:42.7.3",
        application_args=[
            PROCESSED_DATA_DB_TABLE_NAME,
        ],
        execution_timeout=timedelta(minutes=30),
        verbose=True,
    )

    # -----------------------------
    # Task Dependencies
    # -----------------------------
    ingest_task = ingest()
    initialize_datastore >> ingest_task >> transform_and_persist >> feature_engineering >> train_model


# Register DAG with Airflow
customer_churn_dag()
