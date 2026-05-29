from airflow.hooks.base import BaseHook
from pyspark.sql import SparkSession
import logging

logger = logging.getLogger(__name__)

# =========================================================
# CONFIGURATION
# =========================================================

APP_NAME = "COMBINE-ENRICHED-NAV-JOB"

POSTGRES_DRIVER = "org.postgresql.Driver"

FETCH_SIZE = "1000"

OUTPUT_TABLE = "mf_final_nav_enriched"

# =========================================================
# SPARK SESSION
# =========================================================

def create_spark_session():

    return (
        SparkSession.builder
        .appName(APP_NAME)

        .config(
            "spark.jars.packages",
            "org.postgresql:postgresql:42.7.3"
        )

        .config("spark.executor.memory", "5g")

        .config("spark.driver.memory", "5g")

        .config("spark.executor.cores", "4")

        .config("spark.default.parallelism", "2")

        .config("spark.sql.shuffle.partitions", "20")

        .getOrCreate()
    )

# =========================================================
# POSTGRES CONNECTION
# =========================================================

def get_postgres_connection():

    conn = BaseHook.get_connection("postgres_default")

    jdbc_url = (
        f"jdbc:postgresql://"
        f"{conn.host}:{conn.port or 5432}/{conn.schema}"
    )

    return {
        "jdbc_url": jdbc_url,
        "user": conn.login,
        "password": conn.password
    }

# =========================================================
# BUILD QUERY
# =========================================================

def build_query():

    logger.info(
        "************* Building Query *************"
    )

    return """
    (

        SELECT
        d.scheme_code,
        d.nav_date,
        d.nav,
        d.daily_return_pct,
        d.weekly_return_pct,
        d.monthly_return_pct,
        d.rolling_return_30d_pct,
        d.rolling_return_90d_pct,
        d.moving_avg_7d,
        d.moving_avg_30d,
        d.moving_avg_90d,
        d.moving_avg_200d,
        a.cagr_percent,
        a.sharpe_ratio,
        a.daily_volatility,
        a.annualized_volatility
        FROM mf_daily_returns d
            LEFT JOIN mf_aggregated_scheme_metrics a
            ON d.scheme_code = a.scheme_code
        ) final_table
    """

# =========================================================
# LOAD DATAFRAME
# =========================================================

def load_dataframe(
        spark,
        connection,
        query
):

    logger.info(
        "Loading joined enriched dataframe..."
    )

    return (
        spark.read.format("jdbc")

        .option("url", connection["jdbc_url"])

        .option("dbtable", query)

        .option("user", connection["user"])

        .option("password", connection["password"])

        .option("driver", POSTGRES_DRIVER)

        .option("fetchsize", FETCH_SIZE)

        .option("partitionColumn", "scheme_code")

        .option("lowerBound", "1")

        .option("upperBound", "200000")

        .option("numPartitions", "4")

        .load()
    )

# =========================================================
# VALIDATE DATAFRAME
# =========================================================

def validate_dataframe(df):

    logger.info(
        "Displaying sample enriched records..."
    )

    df.show(20, truncate=False)
# =========================================================
# WRITE TO POSTGRES
# =========================================================

def write_to_postgres(
        df,
        connection
):

    logger.info(
        f"Writing dataframe to table: {OUTPUT_TABLE}"
    )

    (
        df.write
        .mode("overwrite")
        .format("jdbc")

        .option("url", connection["jdbc_url"])

        .option("dbtable", OUTPUT_TABLE)

        .option("user", connection["user"])

        .option("password", connection["password"])

        .option("driver", POSTGRES_DRIVER)

        .option("batchsize", "5000")

        .save()
    )

    logger.info(
        "Data write completed successfully."
    )

# =========================================================
# MAIN PIPELINE
# =========================================================

def combine_enriched_data():

    logger.info(
        "Starting enriched NAV combination job..."
    )

    # -----------------------------------------
    # Spark Session
    # -----------------------------------------

    spark = create_spark_session()

    # -----------------------------------------
    # PostgreSQL Connection
    # -----------------------------------------

    connection = get_postgres_connection()

    # -----------------------------------------
    # Build Query
    # -----------------------------------------

    query = build_query()

    logger.info(
        "Generated SQL query successfully."
    )

    # -----------------------------------------
    # Load Dataframe
    # -----------------------------------------

    final_df = load_dataframe(
        spark=spark,
        connection=connection,
        query=query
    )

    # -----------------------------------------
    # Validate
    # -----------------------------------------

    validate_dataframe(final_df)

    # -----------------------------------------
    # Write Output
    # -----------------------------------------

    write_to_postgres(
        df=final_df,
        connection=connection
    )

    # -----------------------------------------
    # Stop Spark
    # -----------------------------------------

    spark.stop()

    logger.info(
        "COMBINE-ENRICHED-NAV-JOB completed successfully."
    )

# =========================================================
# ENTRY POINT
# =========================================================

def main():

    combine_enriched_data()

if __name__ == "__main__":
    main()