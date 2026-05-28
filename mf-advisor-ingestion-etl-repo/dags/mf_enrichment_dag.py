from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from datetime import datetime

from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from pyspark.sql import SparkSession
import logging

logger = logging.getLogger(__name__)

# -----------------------------
# Spark + Connection Helpers
# -----------------------------
def get_spark_session(**context):
    spark = (
        SparkSession.builder
        .appName("Airflow-Spark-Postgres-Window-DAG")
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3")
        .getOrCreate()
    )

    context["ti"].xcom_push(key="spark_created", value=True)
    return spark


def get_jdbc_config(**context):
    conn = BaseHook.get_connection("postgres_default")

    jdbc_url = f"jdbc:postgresql://{conn.host}:{conn.port or 5432}/{conn.schema}"

    context["ti"].xcom_push(key="jdbc_url", value=jdbc_url)
    context["ti"].xcom_push(key="user", value=conn.login)
    context["ti"].xcom_push(key="password", value=conn.password)

    logger.info(f"JDBC URL: {jdbc_url}")


# -----------------------------
# Load Data from Postgres
# -----------------------------
def load_nav_data(**context):
    spark = SparkSession.getActiveSession()

    ti = context["ti"]
    jdbc_url = ti.xcom_pull(key="jdbc_url")
    user = ti.xcom_pull(key="user")
    password = ti.xcom_pull(key="password")

    df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "mf_raw_nav")
        .option("user", user)
        .option("password", password)
        .option("driver", "org.postgresql.Driver")
        .load()
    )

    df.createOrReplaceTempView("mf_raw_nav")
    logger.info("Data loaded into Spark temp view")


# -----------------------------
# Run Spark SQL Window Function
# -----------------------------
def run_window_query(**context):
    spark = SparkSession.getActiveSession()

    result_df = spark.sql("""
        WITH nav_returns AS (
            SELECT
                scheme_code,
                nav_date,
                nav,

                LAG(nav) OVER (
                    PARTITION BY scheme_code
                    ORDER BY nav_date
                ) AS prev_nav

            FROM mf_raw_nav
        )

        SELECT
            scheme_code,
            nav_date,
            nav,
            prev_nav,

            CASE
                WHEN prev_nav IS NULL OR prev_nav = 0 THEN NULL
                ELSE ROUND(((nav - prev_nav) / prev_nav) * 100, 6)
            END AS daily_return_pct

        FROM nav_returns
        WHERE nav_date >= CURRENT_DATE - INTERVAL 5 YEARS
    """)

    context["ti"].xcom_push(key="result_count", value=result_df.count())

    # store DF in global session for next task
    result_df.createOrReplaceTempView("mf_daily_returns")
    logger.info("Window SQL executed successfully")


# -----------------------------
# Write Result Back to Postgres
# -----------------------------
def write_to_postgres(**context):
    spark = SparkSession.getActiveSession()
    ti = context["ti"]

    jdbc_url = ti.xcom_pull(key="jdbc_url")
    user = ti.xcom_pull(key="user")
    password = ti.xcom_pull(key="password")

    df = spark.sql("SELECT * FROM mf_daily_returns")

    df.write \
        .mode("overwrite") \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", "mf_daily_returns_5y") \
        .option("user", user) \
        .option("password", password) \
        .option("driver", "org.postgresql.Driver") \
        .save()

    logger.info("Results written to Postgres table mf_daily_returns_5y")


# -----------------------------
# Final Task
# -----------------------------
def end_task(**context):
    logger.info("Pipeline completed successfully")


# -----------------------------
# DAG Definition
# -----------------------------
default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
}

with DAG(
        dag_id="spark_postgres_daily_return_dag",
        default_args=default_args,
        schedule_interval=None,
        catchup=False,
        description="Spark SQL Window function pipeline for MF NAV returns",
) as dag:

    #task_get_spark = PythonOperator(
     #   task_id="create_spark_session",
      #  python_callable=get_spark_session,
       # provide_context=True,
    #)

    daily_return_task = SparkSubmitOperator(task_id="daily_return_task",
                                    application="./include/scripts/mf_daily_return_enricher/calculate_daily_return.py",
                                    conn_id="my_spark_conn",
                                    verbose=True,
                                    packages="org.postgresql:postgresql:42.7.3",
                                    jars="/opt/spark/jars/postgresql-42.7.3.jar",)

    # -----------------------------
    # Task Dependencies (Sequential)
    # -----------------------------
    #task_get_spark >> task_get_conn >> task_load >> task_transform >> task_write >> task_end
    daily_return_task