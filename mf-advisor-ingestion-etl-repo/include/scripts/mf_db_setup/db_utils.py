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
        print("✅ Successfully established Postgres connection")
        return conn, pg_hook
    except Exception as e:
        print("Failed to establish Postgres connection")
        traceback.print_exc()
        raise RuntimeError(f"Database connection failed: {str(e)}") from e