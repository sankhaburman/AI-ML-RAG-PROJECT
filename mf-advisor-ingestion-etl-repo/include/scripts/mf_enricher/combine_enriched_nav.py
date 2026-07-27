import logging

from airflow.hooks.base import BaseHook

from pyspark.sql import SparkSession

from pyspark.sql.functions import (
    col,
    when,
    lit
)

logger = logging.getLogger(__name__)

# =====================================================
# CONFIGURATION
# =====================================================

APP_NAME = "MF-FINAL-NAV-ENRICHMENT"

POSTGRES_DRIVER = "org.postgresql.Driver"

OUTPUT_TABLE = "mf_final_nav_enriched"

FETCH_SIZE = "1000"

# =====================================================
# SPARK SESSION
# =====================================================

def create_spark_session():

    return (
        SparkSession.builder
        .appName(APP_NAME)
        .config(
            "spark.jars.packages",
            "org.postgresql:postgresql:42.7.3"
        )
        .config(
            "spark.executor.memory",
            "5g"
        )
        .config(
            "spark.driver.memory",
            "5g"
        )
        .config(
            "spark.executor.cores",
            "2"
        )
        .config(
            "spark.default.parallelism",
            "2"
        )
        .config(
            "spark.sql.shuffle.partitions",
            "20"
        )
        .getOrCreate()
    )

# =====================================================
# POSTGRES CONNECTION
# =====================================================

def get_postgres_connection():

    conn = BaseHook.get_connection(
        "postgres_default"
    )

    jdbc_url = (
        f"jdbc:postgresql://"
        f"{conn.host}:"
        f"{conn.port or 5432}/"
        f"{conn.schema}"
    )

    return {

        "jdbc_url": jdbc_url,

        "user": conn.login,

        "password": conn.password
    }

# =====================================================
# BUILD ENRICHED DATAFRAME
# =====================================================

def build_enriched_dataframe(
        nav_df,
        rolling_df):

    logger.info(
        "Joining NAV and rolling metrics"
    )

    enriched_df = (

        nav_df.alias("n")

        .join(

            rolling_df.alias("r"),

            (
                    (
                            col("n.scheme_code")
                            ==
                            col("r.scheme_code")
                    )
                    &
                    (
                            col("n.nav_date")
                            ==
                            col("r.nav_date")
                    )
            ),

            "left"

        )
    )

    logger.info(
        "Creating engineered features"
    )

    # ==========================================
    # FEATURE 1
    # Momentum
    # ==========================================

    enriched_df = enriched_df.withColumn(

        "momentum_90d",

        col("n.rolling_return_90d_pct")
    )

    # ==========================================
    # FEATURE 2
    # Trend Strength
    # ==========================================

    enriched_df = enriched_df.withColumn(

        "trend_strength",

        (
                col("n.moving_avg_30d")
                -
                col("n.moving_avg_90d")
        )
    )

    # ==========================================
    # FEATURE 3
    # MA Crossover
    # ==========================================

    enriched_df = enriched_df.withColumn(

        "ma_crossover",

        when(

            col("n.moving_avg_30d")
            >
            col("n.moving_avg_90d"),

            lit(1)

        ).otherwise(

            lit(0)
        )
    )

    # ==========================================
    # FEATURE 4
    # Short-Term Trend
    # ==========================================

    enriched_df = enriched_df.withColumn(

        "short_term_trend",

        (
                col("n.moving_avg_7d")
                -
                col("n.moving_avg_30d")
        )
    )

    # ==========================================
    # FEATURE 5
    # Long-Term Trend
    # ==========================================

    enriched_df = enriched_df.withColumn(

        "long_term_trend",

        (
                col("n.moving_avg_30d")
                -
                col("n.moving_avg_200d")
        )
    )

    logger.info(
        "Selecting final columns"
    )

    enriched_df = enriched_df.select(

        col("n.scheme_code"),

        col("n.nav_date"),

        col("n.nav"),

        col("n.daily_return_pct"),

        col("n.weekly_return_pct"),

        col("n.monthly_return_pct"),

        col("n.rolling_return_30d_pct"),

        col("n.rolling_return_90d_pct"),

        col("n.moving_avg_7d"),

        col("n.moving_avg_30d"),

        col("n.moving_avg_90d"),

        col("n.moving_avg_200d"),

        col("momentum_90d"),

        col("trend_strength"),

        col("ma_crossover"),

        col("short_term_trend"),

        col("long_term_trend"),

        col("r.rolling_sharpe_90d"),

        col("r.rolling_sharpe_180d"),

        col("r.rolling_volatility_90d"),

        col("r.rolling_volatility_180d"),

        col("r.rolling_cagr_365d")

    )

    return enriched_df

# =====================================================
# BUILD ENRICHED DATAFRAME
# =====================================================

def build_enriched_dataframe(
        nav_df,
        rolling_df):

    logger.info(
        "Joining NAV and rolling metrics"
    )

    joined_df = (

        nav_df.alias("n")

        .join(

            rolling_df.alias("r"),

            (
                    (
                            col("n.scheme_code")
                            ==
                            col("r.scheme_code")
                    )
                    &
                    (
                            col("n.nav_date")
                            ==
                            col("r.nav_date")
                    )
            ),

            "left"

        )
    )

    logger.info(
        "Creating derived features"
    )

    enriched_df = (

        joined_df

        .withColumn(

            "momentum_90d",

            col(
                "n.rolling_return_90d_pct"
            )

        )

        .withColumn(

            "trend_strength",

            (
                    col("n.moving_avg_30d")
                    -
                    col("n.moving_avg_90d")
            )

        )

        .withColumn(

            "ma_crossover",

            when(

                col("n.moving_avg_30d")
                >
                col("n.moving_avg_90d"),

                lit(1)

            ).otherwise(

                lit(0)
            )

        )

        .withColumn(

            "short_term_trend",

            (
                    col("n.moving_avg_7d")
                    -
                    col("n.moving_avg_30d")
            )

        )

        .withColumn(

            "long_term_trend",

            (
                    col("n.moving_avg_30d")
                    -
                    col("n.moving_avg_200d")
            )

        )

    )

    logger.info(
        "Selecting final output columns"
    )

    return (

        enriched_df

        .select(

            # ====================================
            # IDENTIFIERS
            # ====================================

            col("n.scheme_code"),

            col("n.nav_date"),

            col("n.nav"),

            # ====================================
            # RETURNS
            # ====================================

            col("n.daily_return_pct"),

            col("n.weekly_return_pct"),

            col("n.monthly_return_pct"),

            col("n.rolling_return_30d_pct"),

            col("n.rolling_return_90d_pct"),

            # ====================================
            # MOVING AVERAGES
            # ====================================

            col("n.moving_avg_7d"),

            col("n.moving_avg_30d"),

            col("n.moving_avg_90d"),

            col("n.moving_avg_200d"),

            # ====================================
            # ENGINEERED FEATURES
            # ====================================

            col("momentum_90d"),

            col("trend_strength"),

            col("ma_crossover"),

            col("short_term_trend"),

            col("long_term_trend"),

            # ====================================
            # ROLLING METRICS
            # ====================================

            col("r.rolling_sharpe_90d"),

            col("r.rolling_sharpe_180d"),

            col("r.rolling_volatility_90d"),

            col("r.rolling_volatility_180d"),

            col("r.rolling_cagr_365d")

        )
    )

# =====================================================
# CLEAN DATAFRAME
# =====================================================

def clean_dataframe(df):

    logger.info(
        "Cleaning dataframe"
    )

    initial_count = df.count()

    logger.info(
        f"Initial Count: {initial_count}"
    )

    df = df.dropDuplicates(
        [
            "scheme_code",
            "nav_date"
        ]
    )

    cleaned_count = df.count()

    logger.info(
        f"Final Count: {cleaned_count}"
    )

    logger.info(
        f"Duplicates Removed: "
        f"{initial_count - cleaned_count}"
    )

    return df


# =====================================================
# DATA QUALITY CHECKS
# =====================================================

def validate_final_dataframe(df):

    logger.info(
        "Running final validation"
    )

    logger.info(
        f"Final Record Count = "
        f"{df.count()}"
    )

    null_summary = (

        df.select(

            *[
                col(c).isNull()
                .cast("int")
                .alias(c)

                for c in df.columns
            ]

        )
    )

    logger.info(
        "Validation completed"
    )

    return df


# =====================================================
# WRITE TO POSTGRES
# =====================================================

def write_output(
        df,
        connection):

    logger.info(
        f"Writing table: "
        f"{OUTPUT_TABLE}"
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

        .option(
            "truncate",
            "true"
        )

        .save()
    )

    logger.info(
        f"{OUTPUT_TABLE} write completed"
    )