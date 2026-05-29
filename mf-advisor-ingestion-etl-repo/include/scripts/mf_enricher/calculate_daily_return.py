from airflow.hooks.base import BaseHook
from pyspark.sql import SparkSession
import logging
from pyspark.sql.functions import col

logger = logging.getLogger(__name__)

# =========================================================
# CONFIGURATION
# =========================================================

APP_NAME = "CALCULATE-DAILY-RETURN-JOB"
POSTGRES_DRIVER = "org.postgresql.Driver"
FETCH_SIZE = "1000"
LOOKBACK_PERIOD = "3 years"
OUTPUT_TABLE = "mf_daily_returns"
#=======================================================
# DELTA DATAFRAME
#=======================================================
def get_delta_dataframe(spark,new_df,connection):
    logger.info("Loading existing table keys...")

    existing_df = (
        spark.read.format("jdbc")
        .option("url", connection["jdbc_url"])
        .option("dbtable", OUTPUT_TABLE)
        .option("user", connection["user"])
        .option("password", connection["password"])
        .option("driver", POSTGRES_DRIVER)
        .load()
        .select(
            "scheme_code",
            "nav_date"
        )
    )

    logger.info("Calculating delta records...")
    delta_df = (
        new_df.alias("new")
        .join(
            existing_df.alias("existing"),
            on=[
                col("new.scheme_code") == col("existing.scheme_code"),
                col("new.nav_date") == col("existing.nav_date")
            ],
            how="left_anti"
        )
    )
    return delta_df
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
# QUERY PARTS
# =========================================================
def build_nav_base_cte():
    return f"""
    nav_base AS (
        SELECT
            scheme_code,
            nav_date,
            nav
        FROM public.mf_raw_nav
        WHERE nav_date >= CURRENT_DATE - INTERVAL '{LOOKBACK_PERIOD}'
    )
    """

def build_lag_features_cte():
    return """
    lag_features AS (
        SELECT
            scheme_code,
            nav_date,
            nav,
            -- =====================================
            -- LAG FEATURES
            -- =====================================
            LAG(nav, 1) OVER (
                PARTITION BY scheme_code
                ORDER BY nav_date
            ) AS prev_day_nav,

            LAG(nav, 7) OVER (
                PARTITION BY scheme_code
                ORDER BY nav_date
            ) AS prev_week_nav,

            LAG(nav, 30) OVER (
                PARTITION BY scheme_code
                ORDER BY nav_date
            ) AS prev_month_nav,

            LAG(nav, 30) OVER (
                PARTITION BY scheme_code
                ORDER BY nav_date
            ) AS nav_30d_ago,

            LAG(nav, 90) OVER (
                PARTITION BY scheme_code
                ORDER BY nav_date
            ) AS nav_90d_ago

        FROM nav_base
    )
    """
# =========================================================
# MOVING AVERAGE FEATURES
# =========================================================

def build_moving_average_cte():
    return """
    moving_average_features AS (
        SELECT
            *,
            -- =====================================
            -- MOVING AVERAGE 7 DAYS
            -- =====================================
            ROUND(
                AVG(nav) OVER (
                    PARTITION BY scheme_code
                    ORDER BY nav_date
                    ROWS BETWEEN 6 PRECEDING
                    AND CURRENT ROW
                ),
                6
            ) AS moving_avg_7d,
            -- =====================================
            -- MOVING AVERAGE 30 DAYS
            -- =====================================
            ROUND(
                AVG(nav) OVER (
                    PARTITION BY scheme_code
                    ORDER BY nav_date
                    ROWS BETWEEN 29 PRECEDING
                    AND CURRENT ROW
                ),
                6
            ) AS moving_avg_30d,

            -- =====================================
            -- MOVING AVERAGE 90 DAYS
            -- =====================================
            ROUND(
                AVG(nav) OVER (
                    PARTITION BY scheme_code
                    ORDER BY nav_date
                    ROWS BETWEEN 89 PRECEDING
                    AND CURRENT ROW
                ),
                6
            ) AS moving_avg_90d,
            -- =====================================
            -- MOVING AVERAGE 200 DAYS
            -- =====================================
            ROUND(
                AVG(nav) OVER (
                    PARTITION BY scheme_code
                    ORDER BY nav_date
                    ROWS BETWEEN 199 PRECEDING
                    AND CURRENT ROW
                ),
                6
            ) AS moving_avg_200d

        FROM lag_features

    )
    """
# =========================================================
# RETURN FEATURE SQL BUILDERS
# =========================================================
def build_daily_return_sql():
    return """
    CASE
        WHEN prev_day_nav IS NULL
             OR prev_day_nav = 0
        THEN NULL
        ELSE ROUND(
            (
                (nav - prev_day_nav)
                / prev_day_nav
            ) * 100,
            6
        )
    END AS daily_return_pct
    """

def build_weekly_return_sql():
    return """
    CASE
        WHEN prev_week_nav IS NULL
             OR prev_week_nav = 0
        THEN NULL
        ELSE ROUND(
            (
                (nav - prev_week_nav)
                / prev_week_nav
            ) * 100,
            6
        )
    END AS weekly_return_pct
    """

def build_monthly_return_sql():
    return """
    CASE
        WHEN prev_month_nav IS NULL
             OR prev_month_nav = 0
        THEN NULL

        ELSE ROUND(
            (
                (nav - prev_month_nav)
                / prev_month_nav
            ) * 100,
            6
        )
    END AS monthly_return_pct
    """

def build_rolling_30d_return_sql():
    return """
    CASE
        WHEN nav_30d_ago IS NULL
             OR nav_30d_ago = 0
        THEN NULL

        ELSE ROUND(
            (
                (nav - nav_30d_ago)
                / nav_30d_ago
            ) * 100,
            6
        )
    END AS rolling_return_30d_pct
    """


def build_rolling_90d_return_sql():
    return """
    CASE
        WHEN nav_90d_ago IS NULL
             OR nav_90d_ago = 0
        THEN NULL
        ELSE ROUND(
            (
                (nav - nav_90d_ago)
                / nav_90d_ago
            ) * 100,
            6
        )
    END AS rolling_return_90d_pct
    """
# =========================================================
# RETURN FEATURES CTE
# =========================================================

def build_return_features_cte():
    daily_return_sql = build_daily_return_sql()
    weekly_return_sql = build_weekly_return_sql()
    monthly_return_sql = build_monthly_return_sql()
    rolling_30d_sql = build_rolling_30d_return_sql()
    rolling_90d_sql = build_rolling_90d_return_sql()

    return f"""
    return_features AS (
        SELECT
            scheme_code,
            nav_date,
            nav,
            prev_day_nav,
            prev_week_nav,
            prev_month_nav,
            nav_30d_ago,
            nav_90d_ago,
            -- =====================================
            -- MOVING AVERAGES
            -- =====================================
            moving_avg_7d,
            moving_avg_30d,
            moving_avg_90d,
            moving_avg_200d,
            -- =====================================
            -- DAILY RETURN
            -- =====================================
            {daily_return_sql},
            -- =====================================
            -- WEEKLY RETURN
            -- =====================================
            {weekly_return_sql},
            -- =====================================
            -- MONTHLY RETURN
            -- =====================================
            {monthly_return_sql},
            -- =====================================
            -- ROLLING 30 DAY RETURN
            -- =====================================
            {rolling_30d_sql},
            -- =====================================
            -- ROLLING 90 DAY RETURN
            -- =====================================
            {rolling_90d_sql}

        FROM moving_average_features

    )
    """
# =========================================================
# FINAL QUERY BUILDER
# =========================================================
def build_feature_engineering_query():
    nav_base_cte = build_nav_base_cte()
    lag_features_cte = build_lag_features_cte()
    moving_average_cte = build_moving_average_cte()
    return_features_cte = build_return_features_cte()
    query = f"""
       (
        WITH
        {nav_base_cte},
        {lag_features_cte},
        {moving_average_cte},
        {return_features_cte}
        SELECT *
        FROM return_features
    ) final_table
    """
    return query
# =========================================================
# LOAD DATAFRAME
# =========================================================
def load_feature_dataframe(
        spark,
        connection,
        query
):

    logger.info("Loading feature engineered dataset...")
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
        .option("numPartitions", "5")
        .load()
    )
# =========================================================
# VALIDATION
# =========================================================
def validate_dataframe(df):
    logger.info("Displaying sample records...")
    #df.show(100, truncate=False)

# =========================================================
# WRITE OUTPUT
# =========================================================
def write_to_postgres(df,connection):
    logger.info(f"Writing data to PostgreSQL table: {OUTPUT_TABLE}")
    (
        df.write
        .mode("append")
        .format("jdbc")
        .option("url", connection["jdbc_url"])
        .option("dbtable", OUTPUT_TABLE)
        .option("user", connection["user"])
        .option("password", connection["password"])
        .option("driver", POSTGRES_DRIVER)
        .option("batchsize", "5000")
        .save()
    )
    logger.info("Data write completed.")
# =========================================================
# MAIN PIPELINE
# =========================================================
def calculate_daily_returns():
    logger.info("Starting mutual fund feature engineering pipeline...")
    spark = create_spark_session()
    connection = get_postgres_connection()
    query = build_feature_engineering_query()
    logger.info("Generated SQL query successfully.")
    # -----------------------------------------
    # Load Data
    # -----------------------------------------
    result_df = load_feature_dataframe(
        spark=spark,
        connection=connection,
        query=query
    )
    # -----------------------------------------
    # Validate Data
    # -----------------------------------------
    delta_df = get_delta_dataframe(
        spark=spark,
        new_df=result_df,
        connection=connection
    )
    validate_dataframe(delta_df)
    # -----------------------------------------
    # Write Output
    # -----------------------------------------
    logger.info(f"About to write : calculated daily returns into mf_daily_returns Postgres Table")
    write_to_postgres(df=delta_df, connection=connection)

    # -----------------------------------------
    # Stop Spark
    # -----------------------------------------
    spark.stop()
    logger.info(
        "MF Daily Return Enrichment pipeline completed successfully."
    )
# =========================================================
# ENTRY POINT
# =========================================================
def main():
    calculate_daily_returns()

if __name__ == "__main__":
    main()