from airflow.hooks.base import BaseHook
from pyspark.sql import SparkSession
import logging
from pyspark.sql.functions import col

logger = logging.getLogger(__name__)

# =========================================================
# CONFIGURATION
# =========================================================

APP_NAME = "LINEAR-REGRESSION-ML-JOB"
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

# ================================================================
def train_with_linear_regression():
    logger.info("Starting ML Training using Linear Regression...")
    spark = create_spark_session()
    connection = get_postgres_connection()




# =========================================================
# ENTRY POINT
# =========================================================
def main():
    train_with_linear_regression()

if __name__ == "__main__":
    main()