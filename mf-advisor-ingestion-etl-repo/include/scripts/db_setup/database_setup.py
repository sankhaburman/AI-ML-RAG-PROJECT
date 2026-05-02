import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
import traceback


def main():
    spark = None
    conn = None
    cursor = None
    try:
        # Start Spark session
        spark = SparkSession.builder.appName("DataBase_Initializer_job").getOrCreate()


        print("✅ Table created or already exists in PostgreSQL.")

    except Exception as e:
        print("Error occurred during database initialization job")
        traceback.print_exc()

        if db_conn:
            db_conn.rollback()
            print("Rolled back DB transaction")

        # re-raise so Airflow/Spark marks task as failed
        raise RuntimeError(f"Database initialization job failed: {str(e)}") from e

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