# read.py
from include.scripts.utils import _get_db_connection, enrich_data_with_additional_columns, persist_to_postgres
from pyspark.sql import SparkSession
import sys
from pyspark.sql.functions import lit
from airflow.providers.postgres.hooks.postgres import PostgresHook
import traceback


def main():
    if len(sys.argv) < 5:
        raise ValueError(
            "Usage: spark-submit read.py <input_path> <output_path> <current_date> <data_source>"
        )

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    current_date = sys.argv[3]
    data_source = sys.argv[4]

    spark = None
    conn = None
    cursor = None

    try:
        spark = SparkSession.builder.appName("TelcoChurnProcessing").getOrCreate()
        # Get DB connection and hook
        db_conn, pg_hook = _get_db_connection()

        cursor = db_conn.cursor()

        # Read raw data
        df = spark.read.csv(input_path, header=True, inferSchema=True)

        enriched_df = enrich_data_with_additional_columns(df, current_date, data_source)
        persist_to_postgres(enriched_df, db_conn, cursor, "telco_churn_data_raw")

        db_conn.commit()
        print("✅ Data inserted successfully into PostgreSQL table.")

    except Exception as e:
        print("Error occurred in ETL process")
        traceback.print_exc()
        raise RuntimeError(f"ETL failed: {str(e)}") from e

    finally:
        if cursor:
            cursor.close()
            print("Closed Postgres cursor")
        if db_conn:
            db_conn.close()
            print("Closed Postgres connection")
        if spark:
            spark.stop()
            print("Stopped Spark session")



if __name__ == "__main__":
    main()
