import logging
import os
import pickle
import json

from airflow.hooks.base import BaseHook
from pyspark.sql import SparkSession

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import joblib

logger = logging.getLogger(__name__)

APP_NAME = "RANDOM-FOREST-ML-JOB"
POSTGRES_DRIVER = "org.postgresql.Driver"

MODEL_BASE_PATH = "/tmp/models"
MODEL_PATH = "/tmp/models/random_forest.pkl"
SCALER_PATH = "/tmp/models/scaler.pkl"
METADATA_PATH = "/tmp/models/model_metadata.json"

# =========================================================
# CONFIG (4 MONTH MODEL)
# =========================================================
TRAINING_WINDOW_DAYS = 365
PREDICTION_HORIZON_DAYS = 30

TOTAL_LOOKBACK_DAYS = TRAINING_WINDOW_DAYS + PREDICTION_HORIZON_DAYS  # 485 days

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
        .config("spark.driver.memory", "6g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
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
        AND rolling_return_30d_pct IS NOT NULL
        AND rolling_return_90d_pct IS NOT NULL
        AND moving_avg_7d IS NOT NULL
        AND moving_avg_30d IS NOT NULL
        AND sharpe_ratio IS NOT NULL
        AND annualized_volatility IS NOT NULL

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
        .option("partitionColumn", "scheme_code")
        .option("lowerBound", 1)
        .option("upperBound", 200000)
        .option("numPartitions", 8)
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
    pdf["future_return_pct"] = (
                                       (
                                               pdf["future_nav"] - pdf["nav"]
                                       ) / pdf["nav"]
                               ) * 100
    pdf = pdf[
        (pdf["nav"] > 0)
        &
        (pdf["future_return_pct"].notna())
        ]
    return pdf

# =========================================================
# DATASET PREP
# =========================================================
def prepare_dataset(pdf):
    feature_cols = get_feature_columns()
    pdf = pdf.dropna(
        subset=feature_cols + ["future_return_pct"]
    )
    return pdf

# =========================================================
# MODEL TRAINING
# =========================================================
def train_model(pdf):

    feature_cols = get_feature_columns()

    # =====================================================
    # TIME-BASED SPLIT
    # =====================================================

    pdf = pdf.sort_values("nav_date")

    train_cutoff = pdf["nav_date"].quantile(0.60)
    val_cutoff = pdf["nav_date"].quantile(0.80)

    train_df = pdf[
        pdf["nav_date"] <= train_cutoff
        ]

    val_df = pdf[
        (pdf["nav_date"] > train_cutoff)
        &
        (pdf["nav_date"] <= val_cutoff)
        ]

    test_df = pdf[
        pdf["nav_date"] > val_cutoff
        ]

    logger.info("=" * 60)
    logger.info("TIME-BASED DATA SPLIT")
    logger.info("=" * 60)

    logger.info(f"Train Rows: {len(train_df)}")
    logger.info(f"Validation Rows: {len(val_df)}")
    logger.info(f"Test Rows: {len(test_df)}")

    logger.info(
        f"Train Date Range: "
        f"{train_df['nav_date'].min()} "
        f"to "
        f"{train_df['nav_date'].max()}"
    )

    logger.info(
        f"Validation Date Range: "
        f"{val_df['nav_date'].min()} "
        f"to "
        f"{val_df['nav_date'].max()}"
    )

    logger.info(
        f"Test Date Range: "
        f"{test_df['nav_date'].min()} "
        f"to "
        f"{test_df['nav_date'].max()}"
    )

    # =====================================================
    # DATASETS
    # =====================================================

    X_train = train_df[feature_cols]
    y_train = train_df["future_return_pct"]

    X_val = val_df[feature_cols]
    y_val = val_df["future_return_pct"]

    X_test = test_df[feature_cols]
    y_test = test_df["future_return_pct"]

    logger.info(f"Training shape: {X_train.shape}")
    logger.info(f"Validation shape: {X_val.shape}")
    logger.info(f"Test shape: {X_test.shape}")

    # =====================================================
    # FEATURE SCALING
    # =====================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = scaler.transform(X_val)

    X_test_scaled = scaler.transform(X_test)

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=30,
        min_samples_leaf=15,
        max_features='sqrt',
        max_samples=0.6,
        random_state=42,
        n_jobs=-1
    )

    logger.info("Training Random Forest model...")

    model.fit(
        X_train_scaled,
        y_train
    )

    # =====================================================
    # PREDICTIONS
    # =====================================================

    y_pred_train = model.predict(X_train_scaled)

    y_pred_val = model.predict(X_val_scaled)

    y_pred_test = model.predict(X_test_scaled)

    # =====================================================
    # METRICS
    # =====================================================

    train_r2 = r2_score(
        y_train,
        y_pred_train
    )

    val_r2 = r2_score(
        y_val,
        y_pred_val
    )

    test_r2 = r2_score(
        y_test,
        y_pred_test
    )

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            y_pred_train
        )
    )

    val_rmse = np.sqrt(
        mean_squared_error(
            y_val,
            y_pred_val
        )
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred_test
        )
    )

    train_mae = mean_absolute_error(
        y_train,
        y_pred_train
    )

    val_mae = mean_absolute_error(
        y_val,
        y_pred_val
    )

    test_mae = mean_absolute_error(
        y_test,
        y_pred_test
    )

    # =====================================================
    # OVERFITTING ANALYSIS
    # =====================================================

    rmse_gap = test_rmse - train_rmse

    r2_gap = train_r2 - test_r2

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING METRICS")
    logger.info("=" * 60)

    logger.info(
        f"Train R² Score: {train_r2:.4f}"
    )

    logger.info(
        f"Validation R² Score: {val_r2:.4f}"
    )

    logger.info(
        f"Test R² Score: {test_r2:.4f}"
    )

    logger.info(
        f"\nTrain RMSE: {train_rmse:.4f}"
    )

    logger.info(
        f"Validation RMSE: {val_rmse:.4f}"
    )

    logger.info(
        f"Test RMSE: {test_rmse:.4f}"
    )

    logger.info(
        f"\nTrain MAE: {train_mae:.4f}"
    )

    logger.info(
        f"Validation MAE: {val_mae:.4f}"
    )

    logger.info(
        f"Test MAE: {test_mae:.4f}"
    )

    logger.info("\n" + "=" * 60)
    logger.info("OVERFITTING ANALYSIS")
    logger.info("=" * 60)

    logger.info(
        f"RMSE Gap (Test - Train): {rmse_gap:.4f}"
    )

    logger.info(
        f"R² Gap (Train - Test): {r2_gap:.6f}"
    )

    if rmse_gap > 10:
        logger.warning(
            "⚠️ HIGH OVERFITTING DETECTED: Large RMSE gap!"
        )
    elif r2_gap > 0.01:
        logger.warning(
            "⚠️ MODERATE OVERFITTING: R² gap significant"
        )
    else:
        logger.info(
            "✓ Good generalization: Train/test metrics close"
        )

    logger.info("=" * 60 + "\n")

    # =====================================================
    # CROSS VALIDATION
    # =====================================================

    cv_scores = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv=3,
        scoring="r2",
        n_jobs=-1
    )

    logger.info(
        f"Cross-validation R² Scores: {cv_scores}"
    )

    logger.info(
        f"Cross-validation Mean R²: "
        f"{cv_scores.mean():.4f} "
        f"(+/- {cv_scores.std():.4f})"
    )

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    feature_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    logger.info(
        f"\nTop 5 Important Features:\n"
        f"{feature_importance.head()}"
    )
    top_feature = feature_importance.iloc[0]

    logger.info(
        f"Most Important Feature: "
        f"{top_feature['feature']} "
        f"({top_feature['importance']:.4f})"
    )

    if top_feature["importance"] > 0.50:
        logger.warning(
            "Potential feature leakage detected. "
            "One feature contributes more than 50% "
            "of model importance."
        )

    metrics = {
        "train_r2": float(train_r2),
        "val_r2": float(val_r2),
        "test_r2": float(test_r2),
        "train_rmse": float(train_rmse),
        "val_rmse": float(val_rmse),
        "test_rmse": float(test_rmse),
        "train_mae": float(train_mae),
        "val_mae": float(val_mae),
        "test_mae": float(test_mae),
        "rmse_gap": float(rmse_gap),
        "r2_gap": float(r2_gap),
        "cv_mean_r2": float(cv_scores.mean()),
        "cv_std_r2": float(cv_scores.std())
    }

    return (
        model,
        scaler,
        metrics,
        feature_importance
    )
# =========================================================
# SAVE MODEL
# =========================================================
def save_model(model, scaler, metrics, feature_importance):

    os.makedirs(MODEL_BASE_PATH, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    joblib.dump(scaler, SCALER_PATH)

    metadata = {
        "model_type": "RandomForestRegressor",
        "target": "future_return_pct",
        "metrics": metrics,
        "feature_importance": feature_importance.to_dict(orient='records'),
        "features": get_feature_columns(),
        "prediction_horizon_days": PREDICTION_HORIZON_DAYS,
        "training_window_days": TRAINING_WINDOW_DAYS
    }

    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model saved at {MODEL_PATH}")
    logger.info(f"Scaler saved at {SCALER_PATH}")
    logger.info(f"Metadata saved at {METADATA_PATH}")

# =========================================================
# PIPELINE
# =========================================================
def train_with_random_forest():

    logger.info("Starting Random Forest training...")

    spark = create_spark_session()

    try:

        connection = get_postgres_connection()

        spark_df = load_training_dataframe(
            spark,
            connection
        )

        logger.info(
            f"Source rows: {spark_df.count()}"
        )

        # ==================================================
        # LIMIT DATA BEFORE PANDAS
        # ==================================================

        # Random sampling preserves scheme history much better
        sample_fraction = 0.20

        spark_df = spark_df.sample(
            withReplacement=False,
            fraction=sample_fraction,
            seed=42
        )
        # Reduce partitions before collect
        spark_df = spark_df.coalesce(4)
        pdf = spark_df.toPandas()
        logger.info(
            f"Pandas dataframe shape: {pdf.shape}"
        )

        # Reduce partitions before collect
        spark_df = spark_df.coalesce(4)
        # ==================================================
        # CONVERT TO PANDAS
        # ==================================================

        pdf = spark_df.toPandas()

        logger.info(
            f"Pandas dataframe shape: "
            f"{pdf.shape}"
        )

        pdf["nav_date"] = pd.to_datetime(
            pdf["nav_date"]
        )

        pdf = create_target_column(pdf)
        logger.info(
            f"Target statistics:\n"
            f"{pdf['future_return_pct'].describe()}"
        )

        logger.info(
            f"95th percentile: "
            f"{pdf['future_return_pct'].quantile(0.95)}"
        )

        logger.info(
            f"99th percentile: "
            f"{pdf['future_return_pct'].quantile(0.99)}"
        )

        logger.info(
            f"Maximum target: "
            f"{pdf['future_return_pct'].max()}"
        )

        logger.info(
            f"Rows after target creation: "
            f"{len(pdf)}"
        )

        pdf = prepare_dataset(pdf)

        logger.info(
            f"Rows after preparation: "
            f"{len(pdf)}"
        )

        model, scaler, metrics, feature_importance = (
            train_model(pdf)
        )

        save_model(
            model,
            scaler,
            metrics,
            feature_importance
        )

        logger.info(
            "Training completed successfully."
        )

    finally:

        spark.stop()
# =========================================================
# ENTRY POINT
# =========================================================
def main():
    train_with_random_forest()

if __name__ == "__main__":
    main()