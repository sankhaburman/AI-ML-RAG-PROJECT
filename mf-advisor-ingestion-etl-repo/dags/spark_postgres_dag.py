from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from datetime import datetime
from pyspark.sql import SparkSession
import logging

#########################################################
# LOGGER CONFIGURATION
#########################################################

logger = logging.getLogger(__name__)
def run_spark_postgres_job():
    # -----------------------------
    # 1. Get Airflow Connection
    # -----------------------------
    conn = BaseHook.get_connection("postgres_default")

    host = conn.host
    port = conn.port or 5432
    db = conn.schema
    user = conn.login
    password = conn.password

    jdbc_url = f"jdbc:postgresql://{host}:{port}/{db}"
    logger.info(f"JDBC URL: {jdbc_url}")
    # -----------------------------
    # 2. Start Spark Session
    # -----------------------------
    spark = (
        SparkSession.builder
        .appName("Airflow-Spark-Postgres-Job")
        .config(
            "spark.jars.packages",
            "org.postgresql:postgresql:42.7.3"
        )
        .getOrCreate()
    )

    # -----------------------------
    # 3. Read from Postgres using Spark JDBC
    # -----------------------------
    df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "public.weather_data")   # change table
        .option("user", user)
        .option("password", password)
        .option("driver", "org.postgresql.Driver")
        .load()
    )

    df.createOrReplaceTempView("source_table")

    # -----------------------------
    # 4. Run Spark SQL
    # -----------------------------
    result_df = spark.sql("""
        SELECT
            COUNT(*) AS total_rows
        FROM source_table
    """)

    result_df.show()

    # Optional: write back to Postgres
    #(
     #   result_df.write.format("jdbc")
      #  .option("url", jdbc_url)
       # .option("dbtable", "public.spark_result_table")
        #.option("user", user)
        #.option("password", password)
        #.option("driver", "org.postgresql.Driver")
        #.mode("overwrite")
        #.save()
    #)

    spark.stop()


# -----------------------------
# 5. Airflow DAG Definition
# -----------------------------
with DAG(
        dag_id="spark_postgres_jdbc_dag",
        start_date=datetime(2024, 1, 1),
        schedule=None,
        catchup=False,
        tags=["spark", "postgres", "astro"]
) as dag:

    spark_task = PythonOperator(
        task_id="run_spark_postgres_job",
        python_callable=run_spark_postgres_job
    )

    spark_task