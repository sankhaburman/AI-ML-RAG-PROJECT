from airflow.hooks.base import BaseHook
from pyspark.sql import SparkSession
import logging

logger = logging.getLogger(__name__)


def run_optimized_daily_return():

    # -----------------------------
    # 1. PostgreSQL Connection
    # -----------------------------
    conn = BaseHook.get_connection("postgres_default")

    jdbc_url = f"jdbc:postgresql://{conn.host}:{conn.port or 5432}/{conn.schema}"
    user = conn.login
    password = conn.password

    # -----------------------------
    # 2. Spark Session
    # -----------------------------
    spark = (
        SparkSession.builder
        .appName("NAV-Daily-Return-Optimized")
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )

    # -----------------------------
    # 3. Push computation to PostgreSQL (IMPORTANT)
    # -----------------------------
    query = """
    (
        SELECT
            scheme_code,
            nav_date,
            nav,
            prev_nav,
            CASE
                WHEN prev_nav IS NULL OR prev_nav = 0 THEN NULL
                ELSE ROUND(((nav - prev_nav) / prev_nav) * 100, 6)
            END AS daily_return_pct
        FROM (
            SELECT
                scheme_code,
                nav_date,
                nav,
                LAG(nav) OVER (
                    PARTITION BY scheme_code
                    ORDER BY nav_date
                ) AS prev_nav
            FROM public.mf_raw_nav
            WHERE nav_date >= CURRENT_DATE - INTERVAL '5 years'
        ) t
    ) final_table
    """

    # -----------------------------
    # 4. Read from PostgreSQL via JDBC
    # -----------------------------
    result_df = spark.read.format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", query) \
        .option("user", user) \
        .option("password", password) \
        .option("driver", "org.postgresql.Driver") \
        .option("fetchsize", "10000") \
        .load()

    # -----------------------------
    # 5. Basic validation
    # -----------------------------
    logger.info("Sample output:")
    result_df.show(10, truncate=False)

    logger.info(f"Total rows: {result_df.count()}")

    # -----------------------------
    # 6. Write back to PostgreSQL (optional)
    # -----------------------------
    # result_df.write \
    #     .mode("append") \
    #     .format("jdbc") \
    #     .option("url", jdbc_url) \
    #     .option("dbtable", "mf_daily_returns_5y") \
    #     .option("user", user) \
    #     .option("password", password) \
    #     .option("driver", "org.postgresql.Driver") \
    #     .option("batchsize", "5000") \
    #     .save()

    spark.stop()


def main():
    run_optimized_daily_return()


if __name__ == "__main__":
    main()