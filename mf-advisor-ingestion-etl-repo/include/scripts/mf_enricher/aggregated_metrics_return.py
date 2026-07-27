import logging

from airflow.hooks.base import BaseHook
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

# =========================================================
# CONFIGURATION
# =========================================================

APP_NAME = "CALCULATE-ROLLING-METRICS-JOB"

POSTGRES_DRIVER = "org.postgresql.Driver"

FETCH_SIZE = "1000"

OUTPUT_TABLE = "mf_rolling_scheme_metrics"

RISK_FREE_RATE = 6.0

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
        .config("spark.executor.cores", "2")
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
# ROLLING METRICS QUERY
# =========================================================

def build_rolling_metrics_query():

    logger.info(
        "****** Building Rolling Metrics SQL Query ******"
    )

    return f"""
    (

        WITH base_data AS (

            SELECT

                scheme_code,

                nav_date,

                nav,

                daily_return_pct,

                AVG(daily_return_pct)
                OVER (
                    PARTITION BY scheme_code
                    ORDER BY nav_date
                    ROWS BETWEEN 89 PRECEDING
                    AND CURRENT ROW
                ) AS avg_return_90d,

                STDDEV(daily_return_pct)
                OVER (
                    PARTITION BY scheme_code
                    ORDER BY nav_date
                    ROWS BETWEEN 89 PRECEDING
                    AND CURRENT ROW
                ) AS volatility_90d,

                AVG(daily_return_pct)
                OVER (
                    PARTITION BY scheme_code
                    ORDER BY nav_date
                    ROWS BETWEEN 179 PRECEDING
                    AND CURRENT ROW
                ) AS avg_return_180d,

                STDDEV(daily_return_pct)
                OVER (
                    PARTITION BY scheme_code
                    ORDER BY nav_date
                    ROWS BETWEEN 179 PRECEDING
                    AND CURRENT ROW
                ) AS volatility_180d,

                LAG(nav, 365)
                OVER (
                    PARTITION BY scheme_code
                    ORDER BY nav_date
                ) AS nav_365d_ago

            FROM mf_daily_returns

            WHERE nav IS NOT NULL

        )

        SELECT

            scheme_code,

            nav_date,

            ROUND(
                CAST(
                    (
                        SQRT(252)
                        *
                        (
                            avg_return_90d
                            -
                            ({RISK_FREE_RATE} / 252.0)
                        )
                        /
                        NULLIF(volatility_90d, 0)
                    )
                    AS NUMERIC
                ),
                6
            ) AS rolling_sharpe_90d,

            ROUND(
                CAST(
                    (
                        SQRT(252)
                        *
                        (
                            avg_return_180d
                            -
                            ({RISK_FREE_RATE} / 252.0)
                        )
                        /
                        NULLIF(volatility_180d, 0)
                    )
                    AS NUMERIC
                ),
                6
            ) AS rolling_sharpe_180d,

            ROUND(
                CAST(
                    (
                        volatility_90d
                        * SQRT(252)
                    )
                    AS NUMERIC
                ),
                6
            ) AS rolling_volatility_90d,

            ROUND(
                CAST(
                    (
                        volatility_180d
                        * SQRT(252)
                    )
                    AS NUMERIC
                ),
                6
            ) AS rolling_volatility_180d,

            ROUND(
                CAST(
                    (
                        (
                            nav
                            /
                            NULLIF(nav_365d_ago, 0)
                        )
                        - 1
                    ) * 100
                    AS NUMERIC
                ),
                6
            ) AS rolling_cagr_365d

        FROM base_data

    ) rolling_metrics_table
    """


# =========================================================
# LOAD DATAFRAME
# =========================================================

def load_dataframe(
        spark,
        connection,
        query):

    logger.info("Loading dataframe")

    return (
        spark.read
        .format("jdbc")
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
# VALIDATION
# =========================================================

def validate_dataframe(df, title):

    logger.info(
        f"Displaying sample records for {title}"
    )

    logger.info(
        f"Record count = {df.count()}"
    )

    df.show(
        20,
        truncate=False
    )


# =========================================================
# DATA CLEANING
# =========================================================

def clean_metrics(df):

    logger.info(
        "Cleaning rolling metrics dataframe"
    )

    df = df.dropDuplicates(
        [
            "scheme_code",
            "nav_date"
        ]
    )

    return df


# =========================================================
# WRITE OUTPUT
# =========================================================

def write_to_postgres(
        df,
        connection):

    logger.info(
        f"Writing output table: {OUTPUT_TABLE}"
    )

    (
        df.write
        .mode("overwrite")
        .format("jdbc")
        .option(
            "url",
            connection["jdbc_url"]
        )
        .option(
            "dbtable",
            OUTPUT_TABLE
        )
        .option(
            "user",
            connection["user"]
        )
        .option(
            "password",
            connection["password"]
        )
        .option(
            "driver",
            POSTGRES_DRIVER
        )
        .option(
            "batchsize",
            "5000"
        )
        .save()
    )

    logger.info(
        "Rolling metrics write completed"
    )


# =========================================================
# MAIN PIPELINE
# =========================================================

def calculate_rolling_metrics():

    logger.info(
        "Starting rolling metrics pipeline"
    )

    spark = create_spark_session()

    connection = get_postgres_connection()

    rolling_query = (
        build_rolling_metrics_query()
    )

    rolling_df = load_dataframe(
        spark=spark,
        connection=connection,
        query=rolling_query
    )

    validate_dataframe(
        rolling_df,
        "Raw Rolling Metrics"
    )

    rolling_df = clean_metrics(
        rolling_df
    )

    validate_dataframe(
        rolling_df,
        "Clean Rolling Metrics"
    )

    write_to_postgres(
        df=rolling_df,
        connection=connection
    )

    spark.stop()

    logger.info(
        "Rolling metrics pipeline completed successfully"
    )


# =========================================================
# ENTRY POINT
# =========================================================

def main():

    calculate_rolling_metrics()


if __name__ == "__main__":

    main()