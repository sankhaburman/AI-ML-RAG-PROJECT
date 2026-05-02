import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
import traceback
import traceback
from airflow.providers.postgres.hooks.postgres import PostgresHook
from pyspark.sql.functions import lit


POSTGRES_CONN_ID = 'postgres_default'

def get_db_connection():
    """
    Helper function to get a database connection using Airflow's PostgresHook.
    Raises RuntimeError if connection cannot be established.
    """
    try:
        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        conn = pg_hook.get_conn()
        print("Successfully established Postgres connection")
        return conn, pg_hook
    except Exception as e:
        print("Failed to establish Postgres connection")
        traceback.print_exc()
        raise RuntimeError(f"Database connection failed: {str(e)}") from e

def create_table_for_mf_data(cursor):
    """Create raw data table if it doesn't exist."""
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mf_advisor_scheme_raw (
                scheme_code INT PRIMARY KEY,
                scheme_name VARCHAR(255),
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("mf_advisor_scheme_raw created or already exists in PostgreSQL.")
    except Exception as e:
        raise RuntimeError(f"Failed to create table: {e}")



def main():
    spark = None
    conn = None
    cursor = None

    try:
        # Start Spark session
        spark = SparkSession.builder.appName("DataBase_Initializer_job").getOrCreate()

        # Get DB connection and hook
        db_conn, pg_hook = get_db_connection()

        cursor = db_conn.cursor()
        create_table_for_mf_data(cursor)
        #create_table_for_processed_data(cursor)
        db_conn.commit()

        print("Table created or already exists in PostgreSQL.")

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