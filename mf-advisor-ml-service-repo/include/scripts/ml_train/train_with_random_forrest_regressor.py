from airflow.hooks.base import BaseHook
from pyspark.sql import SparkSession
import logging
from pyspark.sql.functions import col
from airflow.hooks.base import BaseHook
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lead

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

import logging
logger = logging.getLogger(__name__)

# =========================================================
# CONFIGURATION
# =========================================================

APP_NAME = "RANDOM-FORREST-ML-JOB"
POSTGRES_DRIVER = "org.postgresql.Driver"
FETCH_SIZE = "1000"
LOOKBACK_PERIOD = "3 years"
OUTPUT_TABLE = "mf_daily_returns"

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

# ==========================================================
# LOAD DATA
# ==========================================================

def load_training_dataframe(spark,connection):

    query = """
    (
        SELECT
            scheme_code,
            nav_date,
            nav,
            daily_return_pct,
            weekly_return_pct,
            monthly_return_pct,
            rolling_return_30d_pct,
            rolling_return_90d_pct,
            moving_avg_7d,
            moving_avg_30d,
            moving_avg_90d,
            moving_avg_200d,
            cagr_percent,
            sharpe_ratio,
            annualized_volatility
        FROM mf_final_nav_enriched
        WHERE nav_date >= CURRENT_DATE - INTERVAL '1 years'
    ) training_data
    """

    return (
        spark.read
        .format("jdbc")
        .option("url", connection["jdbc_url"])
        .option("dbtable", query)
        .option("user", connection["user"])
        .option("password", connection["password"])
        .option("driver", POSTGRES_DRIVER)
        .option("partitionColumn", "scheme_code")
        .option("lowerBound", "1")
        .option("upperBound", "200000")
        .option("numPartitions", "8")
        .option("fetchsize", "1000")
        .load()
    )
# ================================================================
def train_with_random_forrest():
    logger.info("Starting ML Training using Random Forrest...")
    spark = create_spark_session()
    connection = get_postgres_connection()
    raw_df = load_training_dataframe(spark, connection)
    logger.info(raw_df.show(5, truncate=False))



# =========================================================
# ENTRY POINT
# =========================================================
def main():
    train_with_random_forrest()

if __name__ == "__main__":
    main()