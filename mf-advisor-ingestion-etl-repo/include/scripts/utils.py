import traceback
from airflow.providers.postgres.hooks.postgres import PostgresHook
from pyspark.sql.functions import lit

POSTGRES_CONN_ID = 'postgres_default'


def _get_db_connection():
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


def _get_jdbc_props(pg_hook):
    """
    Helper function to extract JDBC properties from an Airflow PostgresHook.
    This is necessary to allow Spark to connect to the database.
    """
    conn = pg_hook.get_connection(POSTGRES_CONN_ID)
    jdbc_url = f"jdbc:postgresql://{conn.host}:{conn.port}/{conn.schema}"
    connection_properties = {
        "user": conn.login,
        "password": conn.password,
        "driver": "org.postgresql.Driver"
    }
    return jdbc_url, connection_properties


def create_table_for_processed_data(cursor):
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telco_customer_churn_processed (
                age BIGINT,
                avg_monthly_gb_download BIGINT,
                avg_monthly_long_distance_charges DOUBLE PRECISION,
                churn_category VARCHAR,
                churn_reason VARCHAR,
                churn_score BIGINT,
                city VARCHAR,
                cltv BIGINT,
                contract VARCHAR,
                country VARCHAR,
                customer_status VARCHAR,
                dependents BIGINT,
                device_protection_plan BIGINT,
                gender VARCHAR,
                internet_service BIGINT,
                internet_type VARCHAR,
                lat_long VARCHAR,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                married BIGINT,
                monthly_charge DOUBLE PRECISION,
                multiple_lines BIGINT,
                number_of_dependents BIGINT,
                number_of_referrals BIGINT,
                offer VARCHAR,
                online_backup BIGINT,
                online_security BIGINT,
                paperless_billing BIGINT,
                partner BIGINT,
                payment_method VARCHAR,
                phone_service BIGINT,
                population BIGINT,
                premium_tech_support BIGINT,
                quarter VARCHAR,
                referred_a_friend BIGINT,
                satisfaction_score BIGINT,
                senior_citizen BIGINT,
                state VARCHAR,
                streaming_movies BIGINT,
                streaming_music BIGINT,
                streaming_tv BIGINT,
                tenure_in_months DOUBLE PRECISION,
                total_charges DOUBLE PRECISION,
                total_extra_data_charges BIGINT,
                total_long_distance_charges DOUBLE PRECISION,
                total_refunds DOUBLE PRECISION,
                total_revenue DOUBLE PRECISION,
                under_30 BIGINT,
                unlimited_data BIGINT,
                zip_code BIGINT,
                churn BIGINT,
                ingested_date DATE,
                data_source VARCHAR
            );
        """)
        print("✅ telco_customer_churn_processed  table created or already exists in PostgreSQL.")
    except Exception as e:
        raise RuntimeError(f"Failed to create table: {e}")


def create_table_for_raw_data(cursor):
    """Create raw data table if it doesn't exist."""
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telco_churn_data_raw (
                age INT,
                avg_monthly_gb_download INT,
                avg_monthly_long_distance_charges DOUBLE PRECISION,
                churn_category VARCHAR(255),
                churn_reason VARCHAR(255),
                churn_score INT,
                city VARCHAR(255),
                cltv INT,
                contract VARCHAR(255),
                country VARCHAR(255),
                customer_id VARCHAR(255),
                customer_status VARCHAR(255),
                dependents INT,
                device_protection_plan INT,
                gender VARCHAR(255),
                internet_service INT,
                internet_type VARCHAR(255),
                lat_long VARCHAR(255),
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                married INT,
                monthly_charge DOUBLE PRECISION,
                multiple_lines INT,
                number_of_dependents INT,
                number_of_referrals INT,
                offer VARCHAR(255),
                online_backup INT,
                online_security INT,
                paperless_billing INT,
                partner INT,
                payment_method VARCHAR(255),
                phone_service INT,
                population INT,
                premium_tech_support INT,
                quarter VARCHAR(255),
                referred_a_friend INT,
                satisfaction_score INT,
                senior_citizen INT,
                state VARCHAR(255),
                streaming_movies INT,
                streaming_music INT,
                streaming_tv INT,
                tenure_in_months INT,
                total_charges DOUBLE PRECISION,
                total_extra_data_charges INT,
                total_long_distance_charges DOUBLE PRECISION,
                total_refunds DOUBLE PRECISION,
                total_revenue DOUBLE PRECISION,
                under_30 INT,
                unlimited_data INT,
                zip_code INT,
                churn INT,
                ingested_date DATE,
                data_source VARCHAR(255)
            );
        """)
        print("✅ telco_churn_data_raw table created or already exists in PostgreSQL.")
    except Exception as e:
        raise RuntimeError(f"Failed to create table: {e}")


def enrich_data_with_additional_columns(df, current_date, data_source):
    """
    Add 'ingested_date' and 'data_source' columns to the DataFrame.
    """
    try:
        df_cleaned = df.toDF(*[c.lower().replace(" ", "_") for c in df.columns])
        df_cleaned = (
            df_cleaned.withColumn("ingested_date", lit(current_date))
                      .withColumn("data_source", lit(data_source))
        )
        df_cleaned.printSchema()
        df_cleaned.show(10)
        return df_cleaned
    except Exception as e:
        print("Error during data enrichment")
        traceback.print_exc()
        raise


def persist_to_postgres(enriched_df, conn, cursor, table_name):
    """
    Persist a Spark DataFrame into Postgres.
    """
    try:
        rows = [tuple(row) for row in enriched_df.collect()]
        if rows:
            columns = [col for col in enriched_df.columns]
            quoted_columns = [f'"{col}"' for col in columns]
            placeholders = ','.join(['%s'] * len(columns))
            insert_query = f"""
                INSERT INTO {table_name} ({','.join(quoted_columns)})
                VALUES ({placeholders});
            """
            cursor.executemany(insert_query, rows)
            conn.commit()
            print(f"✅ Inserted {len(rows)} rows into telco_churn_data_raw")
    except Exception as e:
        print("Error inserting data into Postgres")
        traceback.print_exc()
        raise
