import logging
import os
import pickle

from airflow.hooks.base import BaseHook
from pyspark.sql import SparkSession

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

APP_NAME = "RANDOM-FOREST-ML-JOB"
POSTGRES_DRIVER = "org.postgresql.Driver"

MODEL_BASE_PATH = "/tmp/models"
MODEL_PATH = "/tmp/models/random_forest.pkl"

# =========================================================
# CONFIG (4 MONTH MODEL)
# =========================================================
TRAINING_WINDOW_DAYS = 120
PREDICTION_HORIZON_DAYS = 120

TOTAL_LOOKBACK_DAYS = TRAINING_WINDOW_DAYS + PREDICTION_HORIZON_DAYS  # 240 days

# =========================================================
# SPARK SESSION (LIGHT OPTIMIZED)
# =========================================================
def create_spark_session():
    return (
        SparkSession.builder
        .appName(APP_NAME)
        .config(
            "spark.jars.packages",
            "org.postgresql:postgresql:42.7.3"
        )
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "100")
        .getOrCreate()
    )

# =========================================================
# POSTGRES CONNECTION
# =========================================================
def get_postgres_connection():
    conn = BaseHook.get_connection("postgres_default")

    jdbc_url = (
        f"jdbc:postgresql://{conn.host}:"
        f"{conn.port or 5432}/"
        f"{conn.schema}"
    )

    return {
        "jdbc_url": jdbc_url,
        "user": conn.login,
        "password": conn.password
    }

# =========================================================
# OPTIMIZED SQL QUERY
# =========================================================
def load_training_dataframe(spark, connection):

    query = f"""
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

        WHERE nav_date >= CURRENT_DATE - INTERVAL '{TOTAL_LOOKBACK_DAYS} days'

        AND nav IS NOT NULL
        AND nav > 0

        AND daily_return_pct IS NOT NULL
        AND weekly_return_pct IS NOT NULL
        AND monthly_return_pct IS NOT NULL

        -- reduce useless rows early
        AND moving_avg_30d IS NOT NULL
        AND moving_avg_90d IS NOT NULL
    ) training_data
    """

    return (
        spark.read.format("jdbc")
        .option("url", connection["jdbc_url"])
        .option("dbtable", query)
        .option("user", connection["user"])
        .option("password", connection["password"])
        .option("driver", POSTGRES_DRIVER)
        .option("fetchsize", "2000")
        .load()
    )

# =========================================================
# FEATURE COLUMNS
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

# =========================================================
# TARGET (4 MONTH FUTURE NAV)
# =========================================================
def create_target_column(pdf):

    pdf = pdf.sort_values(["scheme_code", "nav_date"]).copy()

    pdf["future_nav"] = (
        pdf.groupby("scheme_code")["nav"]
        .shift(-PREDICTION_HORIZON_DAYS)
    )

    pdf["target_future_nav"] = pdf["future_nav"]

    pdf = pdf[
        (pdf["nav"] > 0) &
        (pdf["target_future_nav"].notna())
        ]

    return pdf

# =========================================================
# DATASET PREP
# =========================================================
def prepare_dataset(pdf):

    feature_cols = get_feature_columns()

    pdf = pdf.dropna(subset=feature_cols + ["target_future_nav"])

    X = pdf[feature_cols]
    y = pdf["target_future_nav"]

    return X, y

# =========================================================
# MODEL TRAINING
# =========================================================
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    logger.info(f"Train R² Score: {model.score(X_train, y_train):.4f}")
    logger.info(f"Test R² Score: {model.score(X_test, y_test):.4f}")

    return model

# =========================================================
# SAVE MODEL
# =========================================================
def save_model(model):

    os.makedirs(MODEL_BASE_PATH, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model saved at {MODEL_PATH}")

# =========================================================
# PIPELINE
# =========================================================
def train_with_random_forest():

    logger.info("Starting 4-month NAV prediction model...")

    spark = create_spark_session()

    try:
        connection = get_postgres_connection()

        spark_df = load_training_dataframe(spark, connection)

        logger.info(f"Rows loaded: {spark_df.count()}")

        pdf = spark_df.toPandas()

        pdf["nav_date"] = pd.to_datetime(pdf["nav_date"])

        pdf = create_target_column(pdf)

        logger.info(f"Rows after target creation: {len(pdf)}")

        X, y = prepare_dataset(pdf)

        logger.info(f"Training shape: {X.shape}")

        model = train_model(X, y)

        save_model(model)

        logger.info("Training completed successfully.")

    finally:
        spark.stop()

# =========================================================
# ENTRY POINT
# =========================================================
def main():
    train_with_random_forest()

if __name__ == "__main__":
    main()