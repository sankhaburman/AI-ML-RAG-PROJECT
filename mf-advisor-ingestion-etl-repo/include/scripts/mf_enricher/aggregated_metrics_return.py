from airflow.hooks.base import BaseHook
from pyspark.sql import SparkSession
import logging

logger = logging.getLogger(__name__)

# =========================================================
# CONFIGURATION
# =========================================================

APP_NAME = "CALCULATE-NAV-AGGREGATED-METRICS-JOB"

POSTGRES_DRIVER = "org.postgresql.Driver"

FETCH_SIZE = "1000"

LOOKBACK_PERIOD = "3 years"

OUTPUT_TABLE = "mf_aggregated_scheme_metrics"

RISK_FREE_RATE = 6.0

MIN_RECORDS_REQUIRED = 30

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
# CAGR QUERY
# =========================================================

def build_cagr_query():

    logger.info("****** Building CAGR SQL Query ******")

    return f"""
    (

        WITH filtered_nav AS (

            SELECT
                scheme_code,
                nav_date,
                nav

            FROM mf_daily_returns

            WHERE nav_date >=
                  CURRENT_DATE - INTERVAL '{LOOKBACK_PERIOD}'

              AND nav IS NOT NULL

        ),

        valid_schemes AS (

            SELECT
                scheme_code,
                COUNT(*) AS total_records

            FROM filtered_nav

            GROUP BY scheme_code

            HAVING COUNT(*) >= {MIN_RECORDS_REQUIRED}
        ),

        ranked_nav AS (

            SELECT
                f.scheme_code,
                f.nav_date,
                f.nav,

                ROW_NUMBER() OVER (
                    PARTITION BY f.scheme_code
                    ORDER BY f.nav_date ASC
                ) AS rn_start,

                ROW_NUMBER() OVER (
                    PARTITION BY f.scheme_code
                    ORDER BY f.nav_date DESC
                ) AS rn_end

            FROM filtered_nav f

            INNER JOIN valid_schemes v
                ON f.scheme_code = v.scheme_code
        ),

        start_nav AS (

            SELECT

                scheme_code,

                nav AS initial_nav,

                nav_date AS start_date

            FROM ranked_nav

            WHERE rn_start = 1
        ),

        end_nav AS (

            SELECT

                scheme_code,

                nav AS final_nav,

                nav_date AS end_date

            FROM ranked_nav

            WHERE rn_end = 1
        ),

        cagr_base AS (

            SELECT

                s.scheme_code,

                (
                    e.end_date - s.start_date
                ) / 365.25 AS years,

                s.initial_nav,

                e.final_nav

            FROM start_nav s

            INNER JOIN end_nav e
                ON s.scheme_code = e.scheme_code
        )

        SELECT

            scheme_code,

            ROUND(
                CAST(
                    (
                        POWER(
                            final_nav / initial_nav,
                            1 / NULLIF(years, 0)
                        ) - 1
                    ) * 100
                    AS NUMERIC
                ),
                6
            ) AS cagr_percent

        FROM cagr_base

        WHERE years > 0

          AND initial_nav > 0

          AND final_nav > 0

    ) cagr_table
    """

# =========================================================
# SHARPE RATIO QUERY
# =========================================================

def build_sharpe_ratio_query():

    logger.info(
        "****** Building Sharpe Ratio SQL Query ******"
    )

    return f"""
    (

        WITH return_stats AS (

            SELECT

                scheme_code,

                COUNT(*) AS total_records,

                AVG(daily_return_pct) AS avg_daily_return,

                STDDEV(daily_return_pct) AS volatility

            FROM mf_daily_returns

            WHERE daily_return_pct IS NOT NULL

              AND nav_date >=
                  CURRENT_DATE - INTERVAL '{LOOKBACK_PERIOD}'

            GROUP BY scheme_code

            HAVING COUNT(*) >= {MIN_RECORDS_REQUIRED}
        )

        SELECT

            scheme_code,

            ROUND(
                CAST(
                    SQRT(252)

                    *

                    (
                        (
                            avg_daily_return
                            - ({RISK_FREE_RATE} / 252)
                        )

                        / NULLIF(volatility, 0)
                    )

                    AS NUMERIC
                ),
                6
            ) AS sharpe_ratio

        FROM return_stats

        WHERE volatility > 0

    ) sharpe_ratio_table
    """

# =========================================================
# DAILY VOLATILITY QUERY
# =========================================================

def build_daily_volatility_query():

    logger.info(
        "****** Building Daily Volatility SQL Query ******"
    )

    return f"""
    (

        SELECT

            scheme_code,

            ROUND(
                CAST(
                    STDDEV(daily_return_pct)
                    AS NUMERIC
                ),
                6
            ) AS daily_volatility

        FROM mf_daily_returns

        WHERE daily_return_pct IS NOT NULL

          AND nav_date >=
              CURRENT_DATE - INTERVAL '{LOOKBACK_PERIOD}'

        GROUP BY scheme_code

        HAVING COUNT(*) >= {MIN_RECORDS_REQUIRED}

    ) daily_volatility_table
    """

# =========================================================
# ANNUALIZED VOLATILITY QUERY
# =========================================================

def build_annualized_volatility_query():

    logger.info(
        "****** Building Annualized Volatility SQL Query ******"
    )

    return f"""
    (

        SELECT

            scheme_code,

            ROUND(
                CAST(
                    STDDEV(daily_return_pct)
                    * SQRT(252)
                    AS NUMERIC
                ),
                6
            ) AS annualized_volatility

        FROM mf_daily_returns

        WHERE daily_return_pct IS NOT NULL

          AND nav_date >=
              CURRENT_DATE - INTERVAL '{LOOKBACK_PERIOD}'

        GROUP BY scheme_code

        HAVING COUNT(*) >= {MIN_RECORDS_REQUIRED}

    ) annualized_volatility_table
    """

# =========================================================
# LOAD DATAFRAME
# =========================================================

def load_dataframe(
        spark,
        connection,
        query
):

    logger.info("Loading dataframe...")

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

        .option("numPartitions", "2")

        .load()
    )

# =========================================================
# VALIDATION
# =========================================================

def validate_dataframe(df, title):

    logger.info(
        f"Displaying sample records for: {title}"
    )

    df.show(20, truncate=False)

# =========================================================
# JOIN ALL METRICS
# =========================================================

def join_metrics(
        cagr_df,
        sharpe_df,
        daily_volatility_df,
        annualized_volatility_df
):

    logger.info("Joining all metrics dataframes...")

    final_df = (

        cagr_df.alias("c")

        .join(
            sharpe_df.alias("s"),
            on="scheme_code",
            how="left"
        )

        .join(
            daily_volatility_df.alias("d"),
            on="scheme_code",
            how="left"
        )

        .join(
            annualized_volatility_df.alias("a"),
            on="scheme_code",
            how="left"
        )
    )

    return final_df

# =========================================================
# WRITE OUTPUT
# =========================================================

def write_to_postgres(
        df,
        connection
):

    logger.info(
        f"Writing metrics to table: {OUTPUT_TABLE}"
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

        .option("batchsize", "1000")

        .save()
    )

    logger.info("Metrics write completed.")

# =========================================================
# MAIN PIPELINE
# =========================================================

def calculate_aggregate_metrics():

    logger.info(
        "Starting aggregate metrics pipeline..."
    )

    # -----------------------------------------
    # Spark Session
    # -----------------------------------------

    spark = create_spark_session()

    # -----------------------------------------
    # PostgreSQL Connection
    # -----------------------------------------

    connection = get_postgres_connection()

    # =====================================================
    # CAGR
    # =====================================================

    cagr_query = build_cagr_query()

    cagr_df = load_dataframe(
        spark=spark,
        connection=connection,
        query=cagr_query
    )

    validate_dataframe(
        cagr_df,
        "CAGR"
    )

    # =====================================================
    # SHARPE RATIO
    # =====================================================

    sharpe_query = build_sharpe_ratio_query()

    sharpe_df = load_dataframe(
        spark=spark,
        connection=connection,
        query=sharpe_query
    )

    validate_dataframe(
        sharpe_df,
        "Sharpe Ratio"
    )

    # =====================================================
    # DAILY VOLATILITY
    # =====================================================

    daily_volatility_query = (
        build_daily_volatility_query()
    )

    daily_volatility_df = load_dataframe(
        spark=spark,
        connection=connection,
        query=daily_volatility_query
    )

    validate_dataframe(
        daily_volatility_df,
        "Daily Volatility"
    )

    # =====================================================
    # ANNUALIZED VOLATILITY
    # =====================================================

    annualized_volatility_query = (
        build_annualized_volatility_query()
    )

    annualized_volatility_df = load_dataframe(
        spark=spark,
        connection=connection,
        query=annualized_volatility_query
    )

    validate_dataframe(
        annualized_volatility_df,
        "Annualized Volatility"
    )

    # =====================================================
    # JOIN METRICS
    # =====================================================

    final_df = join_metrics(
        cagr_df=cagr_df,
        sharpe_df=sharpe_df,
        daily_volatility_df=daily_volatility_df,
        annualized_volatility_df=annualized_volatility_df
    )

    validate_dataframe(
        final_df,
        "Final Metrics"
    )

    # =====================================================
    # WRITE OUTPUT
    # =====================================================

    write_to_postgres(
        df=final_df,
        connection=connection
    )

    # =====================================================
    # STOP SPARK
    # =====================================================

    spark.stop()

    logger.info(
        "Aggregate metrics pipeline completed successfully."
    )

# =========================================================
# ENTRY POINT
# =========================================================

def main():

    calculate_aggregate_metrics()

if __name__ == "__main__":
    main()