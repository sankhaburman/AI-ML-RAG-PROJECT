import logging
from airflow.hooks.base import BaseHook
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, when, lead
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor

logger = logging.getLogger(__name__)
APP_NAME = "RANDOM-FOREST-ML-JOB"
POSTGRES_DRIVER = "org.postgresql.Driver"

MODEL_BASE_PATH = "/tmp/models"
MODEL_PATH = "/tmp/models/random_forrest/latest"

# SPARK SESSION
#-------------------------------------------------
def create_spark_session():
    return (
        SparkSession.builder
        .appName(APP_NAME)
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3")
        .config("spark.executor.memory", "6g")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.cores", "4")
        .config("spark.default.parallelism", "4")
        .config("spark.sql.shuffle.partitions", "32")
        .getOrCreate()
    )

# POSTGRES CONNECTION
def get_postgres_connection():
    conn = BaseHook.get_connection("postgres_default")
    jdbc_url = (
        f"jdbc:postgresql://{conn.host}:{conn.port or 5432}/{conn.schema}"
    )
    return {
        "jdbc_url": jdbc_url,
        "user": conn.login,
        "password": conn.password
    }

# LOAD DATA
#--------------------------------------------
def load_training_dataframe(spark, connection):
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
        spark.read.format("jdbc")
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

# TARGET CREATION
# =========================================================
def create_target_column(df):
    window_spec = (
        Window.partitionBy("scheme_code")
        .orderBy("nav_date")
    )

    df = df.withColumn(
        "future_nav",
        lead("nav", 30).over(window_spec)
    )

    df = df.withColumn(
        "target_30d_return",
        when(
            (col("nav").isNull()) |
            (col("nav") <= 0) |
            (col("future_nav").isNull()),
            None
        ).otherwise(
            ((col("future_nav") - col("nav")) / col("nav")) * 100
        )
    )
    return df.filter(col("target_30d_return").isNotNull())

# FEATURES
# =========================================================
def get_feature_columns():
    return [
        "daily_return_pct",
        "weekly_return_pct",
        "monthly_return_pct",
        "rolling_return_30d_pct",
        "rolling_return_90d_pct",
        "moving_avg_7d",
        "moving_avg_30d",
        "moving_avg_90d",
        "moving_avg_200d",
        "cagr_percent",
        "sharpe_ratio",
        "annualized_volatility"
    ]

# DATA PREP
# =========================================================
def prepare_dataset(df):
    feature_cols = get_feature_columns()
    df = df.na.drop(subset=feature_cols)
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )
    return assembler.transform(df)

# TRAIN MODEL
# =========================================================

def train_model(dataset):
    train_df, test_df = dataset.randomSplit([0.8, 0.2], seed=42)
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="target_30d_return",
        predictionCol="prediction",
        numTrees=100,
        maxDepth=8,
        seed=42
    )
    model = rf.fit(train_df)
    return model

# SAFE MODEL SAVE
# =========================================================
def save_model(model):
    logger.info(f"Saving model to {MODEL_PATH}")
    model.write().overwrite().save(MODEL_PATH)
    logger.info(f"Model saved successfully at {MODEL_PATH}")

# TRAIN PIPELINE
# =========================================================
def train_with_random_forest():
    logger.info("Starting ML Training...")
    spark = create_spark_session()
    connection = get_postgres_connection()
    raw_df = load_training_dataframe(spark, connection)
    raw_df.show(5, truncate=False)
    target_df = create_target_column(raw_df)
    dataset = prepare_dataset(target_df)
    model = train_model(dataset)
    save_model(model)
    spark.stop()

# ENTRY POINT
# =========================================================
def main():
    train_with_random_forest()

if __name__ == "__main__":
    main()