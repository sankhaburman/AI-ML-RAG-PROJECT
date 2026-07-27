import logging
import os
import pickle
import json

from airflow.hooks.base import BaseHook
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import joblib

logger = logging.getLogger(__name__)

# =========================================================
# CONFIGURATION
# =========================================================

APP_NAME = "RANDOM-FOREST-ML-JOB"
POSTGRES_DRIVER = "org.postgresql.Driver"

MODEL_BASE_PATH = "/tmp/models"
MODEL_PATH = "/tmp/models/random_forest.pkl"
SCALER_PATH = "/tmp/models/scaler.pkl"
METADATA_PATH = "/tmp/models/model_metadata.json"

# =========================================================
# TRAINING CONFIG
# =========================================================

TRAINING_WINDOW_DAYS = 365
PREDICTION_HORIZON_DAYS = 30

TOTAL_LOOKBACK_DAYS = (
        TRAINING_WINDOW_DAYS +
        PREDICTION_HORIZON_DAYS
)

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

# =========================================================
# LOAD TRAINING DATA
# =========================================================

def load_training_dataframe(
        spark,
        connection
):

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

        WHERE nav_date >= CURRENT_DATE
              - INTERVAL '{TOTAL_LOOKBACK_DAYS} days'

        AND nav IS NOT NULL
        AND nav > 0

        AND daily_return_pct IS NOT NULL
        AND weekly_return_pct IS NOT NULL
        AND monthly_return_pct IS NOT NULL

        AND rolling_return_30d_pct IS NOT NULL
        AND rolling_return_90d_pct IS NOT NULL

        AND moving_avg_7d IS NOT NULL
        AND moving_avg_30d IS NOT NULL
        AND moving_avg_90d IS NOT NULL
        AND moving_avg_200d IS NOT NULL

        AND cagr_percent IS NOT NULL
        AND sharpe_ratio IS NOT NULL
        AND annualized_volatility IS NOT NULL

    ) training_data
    """

    spark_df = (
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

    # =====================================================
    # FIX DECIMAL ISSUE AT SOURCE
    # =====================================================

    numeric_cols = [
        "nav",
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

    for column_name in numeric_cols:

        spark_df = spark_df.withColumn(
            column_name,
            col(column_name).cast("double")
        )

    return spark_df

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
# NUMERIC CONVERSION HELPER
# =========================================================

def convert_numeric_columns(pdf):

    numeric_cols = [
        "nav",
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

    for column_name in numeric_cols:

        if column_name in pdf.columns:

            pdf[column_name] = pd.to_numeric(
                pdf[column_name],
                errors="coerce"
            ).astype("float64")

    return pdf

# =========================================================
# TARGET CREATION
# =========================================================

def create_target_column(pdf):

    pdf = pdf.sort_values(
        ["scheme_code", "nav_date"]
    ).copy()

    pdf["nav"] = pd.to_numeric(
        pdf["nav"],
        errors="coerce"
    ).astype("float64")

    pdf["future_nav"] = (
        pdf.groupby("scheme_code")["nav"]
        .shift(-PREDICTION_HORIZON_DAYS)
    )

    pdf["future_nav"] = pd.to_numeric(
        pdf["future_nav"],
        errors="coerce"
    ).astype("float64")

    pdf["future_return_pct"] = (
                                       (
                                               pdf["future_nav"] -
                                               pdf["nav"]
                                       )
                                       /
                                       pdf["nav"]
                               ) * 100.0

    pdf["future_return_pct"] = pd.to_numeric(
        pdf["future_return_pct"],
        errors="coerce"
    ).astype("float64")

    pdf = pdf[
        (pdf["nav"] > 0)
        &
        (pdf["future_return_pct"].notna())
        ]

    logger.info(
        f"future_return_pct dtype: "
        f"{pdf['future_return_pct'].dtype}"
    )

    return pdf


# =========================================================
# DATASET PREPARATION
# =========================================================

def prepare_dataset(pdf):

    feature_cols = get_feature_columns()

    pdf = pdf.dropna(
        subset=feature_cols +
               ["future_return_pct"]
    )

    if len(pdf) == 0:
        raise ValueError(
            "No rows remaining after dataset preparation"
        )

    return pdf


# =========================================================
# MODEL TRAINING
# =========================================================

def train_model(pdf):

    feature_cols = get_feature_columns()

    logger.info(
        f"Training rows received: {len(pdf)}"
    )

    pdf = pdf.sort_values(
        "nav_date"
    ).copy()

    # =====================================================
    # TIME BASED SPLIT
    # =====================================================

    train_cutoff = pdf["nav_date"].quantile(
        0.60
    )

    val_cutoff = pdf["nav_date"].quantile(
        0.80
    )

    train_df = pdf[
        pdf["nav_date"] <= train_cutoff
        ].copy()

    val_df = pdf[
        (
                pdf["nav_date"] > train_cutoff
        )
        &
        (
                pdf["nav_date"] <= val_cutoff
        )
        ].copy()

    test_df = pdf[
        pdf["nav_date"] > val_cutoff
        ].copy()

    # =====================================================
    # VALIDATION
    # =====================================================

    if len(train_df) == 0:
        raise ValueError(
            "Training dataset is empty"
        )

    if len(val_df) == 0:
        raise ValueError(
            "Validation dataset is empty"
        )

    if len(test_df) == 0:
        raise ValueError(
            "Test dataset is empty"
        )

    logger.info("=" * 60)
    logger.info("TIME-BASED SPLIT")
    logger.info("=" * 60)

    logger.info(
        f"Train Rows: {len(train_df)}"
    )

    logger.info(
        f"Validation Rows: {len(val_df)}"
    )

    logger.info(
        f"Test Rows: {len(test_df)}"
    )

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
    # ENSURE NUMERIC FEATURES
    # =====================================================

    for feature in feature_cols:

        train_df[feature] = pd.to_numeric(
            train_df[feature],
            errors="coerce"
        )

        val_df[feature] = pd.to_numeric(
            val_df[feature],
            errors="coerce"
        )

        test_df[feature] = pd.to_numeric(
            test_df[feature],
            errors="coerce"
        )

    train_df = train_df.dropna(
        subset=feature_cols +
               ["future_return_pct"]
    )

    val_df = val_df.dropna(
        subset=feature_cols +
               ["future_return_pct"]
    )

    test_df = test_df.dropna(
        subset=feature_cols +
               ["future_return_pct"]
    )

    # =====================================================
    # FEATURES / TARGET
    # =====================================================

    X_train = train_df[
        feature_cols
    ]

    y_train = train_df[
        "future_return_pct"
    ].astype(float)

    X_val = val_df[
        feature_cols
    ]

    y_val = val_df[
        "future_return_pct"
    ].astype(float)

    X_test = test_df[
        feature_cols
    ]

    y_test = test_df[
        "future_return_pct"
    ].astype(float)

    logger.info(
        f"Training Shape: "
        f"{X_train.shape}"
    )

    logger.info(
        f"Validation Shape: "
        f"{X_val.shape}"
    )

    logger.info(
        f"Test Shape: "
        f"{X_test.shape}"
    )

    # =====================================================
    # STANDARD SCALER
    # =====================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    logger.info(
        "Feature scaling completed"
    )

    # =====================================================
    # RANDOM FOREST MODEL
    # =====================================================

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=30,
        min_samples_leaf=15,
        max_features="sqrt",
        max_samples=0.6,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    logger.info(
        "Training Random Forest model..."
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    logger.info(
        "Model training completed"
    )

    # =====================================================
    # PREDICTIONS
    # =====================================================

    y_pred_train = model.predict(
        X_train_scaled
    )

    y_pred_val = model.predict(
        X_val_scaled
    )

    y_pred_test = model.predict(
        X_test_scaled
    )

    # =====================================================
    # R2 SCORES
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

    # =====================================================
    # RMSE
    # =====================================================

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

    # =====================================================
    # MAE
    # =====================================================

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

    rmse_gap = (
            test_rmse -
            train_rmse
    )

    r2_gap = (
            train_r2 -
            test_r2
    )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING METRICS")
    logger.info("=" * 60)

    logger.info(
        f"Train R² Score: "
        f"{train_r2:.6f}"
    )

    logger.info(
        f"Validation R² Score: "
        f"{val_r2:.6f}"
    )

    logger.info(
        f"Test R² Score: "
        f"{test_r2:.6f}"
    )

    logger.info(
        f"Train RMSE: "
        f"{train_rmse:.6f}"
    )

    logger.info(
        f"Validation RMSE: "
        f"{val_rmse:.6f}"
    )

    logger.info(
        f"Test RMSE: "
        f"{test_rmse:.6f}"
    )

    logger.info(
        f"Train MAE: "
        f"{train_mae:.6f}"
    )

    logger.info(
        f"Validation MAE: "
        f"{val_mae:.6f}"
    )

    logger.info(
        f"Test MAE: "
        f"{test_mae:.6f}"
    )

    logger.info("\n" + "=" * 60)
    logger.info("OVERFITTING ANALYSIS")
    logger.info("=" * 60)

    logger.info(
        f"RMSE Gap: "
        f"{rmse_gap:.6f}"
    )

    logger.info(
        f"R² Gap: "
        f"{r2_gap:.6f}"
    )

    if rmse_gap > 10:
        logger.warning(
            "HIGH OVERFITTING DETECTED"
        )

    elif r2_gap > 0.01:
        logger.warning(
            "MODERATE OVERFITTING DETECTED"
        )

    else:
        logger.info(
            "Good generalization observed"
        )

    logger.info("=" * 60)

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
        f"CV Scores: {cv_scores}"
    )

    logger.info(
        f"CV Mean R²: "
        f"{cv_scores.mean():.6f}"
    )

    logger.info(
        f"CV Std R²: "
        f"{cv_scores.std():.6f}"
    )

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    feature_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance":
                model.feature_importances_
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            by="importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    logger.info(
        "\nTop 10 Features:\n%s",
        feature_importance.head(10)
    )

    top_feature = (
        feature_importance.iloc[0]
    )

    logger.info(
        f"Most Important Feature: "
        f"{top_feature['feature']} "
        f"({top_feature['importance']:.6f})"
    )

    if top_feature["importance"] > 0.50:

        logger.warning(
            "Potential feature leakage. "
            "Top feature exceeds 50%% importance."
        )

    # =====================================================
    # METADATA METRICS
    # =====================================================

    metrics = {

        "train_r2":
            float(train_r2),

        "val_r2":
            float(val_r2),

        "test_r2":
            float(test_r2),

        "train_rmse":
            float(train_rmse),

        "val_rmse":
            float(val_rmse),

        "test_rmse":
            float(test_rmse),

        "train_mae":
            float(train_mae),

        "val_mae":
            float(val_mae),

        "test_mae":
            float(test_mae),

        "rmse_gap":
            float(rmse_gap),

        "r2_gap":
            float(r2_gap),

        "cv_mean_r2":
            float(
                cv_scores.mean()
            ),

        "cv_std_r2":
            float(
                cv_scores.std()
            )
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

def save_model(
        model,
        scaler,
        metrics,
        feature_importance
):

    os.makedirs(
        MODEL_BASE_PATH,
        exist_ok=True
    )

    with open(
            MODEL_PATH,
            "wb"
    ) as f:

        pickle.dump(
            model,
            f
        )

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    metadata = {

        "model_type":
            "RandomForestRegressor",

        "target":
            "future_return_pct",

        "metrics":
            metrics,

        "feature_importance":
            feature_importance.to_dict(
                orient="records"
            ),

        "features":
            get_feature_columns(),

        "prediction_horizon_days":
            PREDICTION_HORIZON_DAYS,

        "training_window_days":
            TRAINING_WINDOW_DAYS
    }

    with open(
            METADATA_PATH,
            "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    logger.info(
        f"Model saved at {MODEL_PATH}"
    )

    logger.info(
        f"Scaler saved at {SCALER_PATH}"
    )

    logger.info(
        f"Metadata saved at {METADATA_PATH}"
    )


# =========================================================
# TRAINING PIPELINE
# =========================================================

def train_with_random_forest():

    logger.info(
        "Starting Random Forest training..."
    )

    spark = create_spark_session()

    try:

        connection = (
            get_postgres_connection()
        )

        spark_df = (
            load_training_dataframe(
                spark,
                connection
            )
        )

        source_rows = spark_df.count()

        logger.info(
            f"Source rows: "
            f"{source_rows}"
        )

        if source_rows == 0:

            raise ValueError(
                "No rows returned "
                "from source table"
            )

        # =====================================
        # OPTIONAL SAMPLING
        # =====================================

        sample_fraction = 0.20

        spark_df = spark_df.sample(
            withReplacement=False,
            fraction=sample_fraction,
            seed=42
        )

        logger.info(
            f"Rows after sampling: "
            f"{spark_df.count()}"
        )

        # =====================================
        # REDUCE PARTITIONS
        # =====================================

        spark_df = spark_df.coalesce(4)

        # =====================================
        # SINGLE PANDAS CONVERSION
        # =====================================

        pdf = spark_df.toPandas()

        logger.info(
            f"Pandas dataframe shape: "
            f"{pdf.shape}"
        )

        if len(pdf) == 0:

            raise ValueError(
                "Pandas dataframe empty"
            )

        # =====================================
        # DATE CONVERSION
        # =====================================

        pdf["nav_date"] = pd.to_datetime(
            pdf["nav_date"],
            errors="coerce"
        )

        # =====================================
        # NUMERIC CONVERSION
        # =====================================

        pdf = convert_numeric_columns(
            pdf
        )

        logger.info(
            "Numeric conversion completed"
        )

        # =====================================
        # TARGET CREATION
        # =====================================

        pdf = create_target_column(
            pdf
        )

        if len(pdf) == 0:

            raise ValueError(
                "No rows remaining after "
                "target creation"
            )

        logger.info(
            f"Rows after target creation: "
            f"{len(pdf)}"
        )

        logger.info(
            f"Target dtype: "
            f"{pdf['future_return_pct'].dtype}"
        )

        logger.info(
            f"Null target count: "
            f"{pdf['future_return_pct'].isna().sum()}"
        )

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

        # =====================================
        # DATASET PREPARATION
        # =====================================

        pdf = prepare_dataset(
            pdf
        )

        logger.info(
            f"Rows after preparation: "
            f"{len(pdf)}"
        )

        if len(pdf) < 1000:

            logger.warning(
                "Very small training dataset: "
                f"{len(pdf)} rows"
            )

        # =====================================
        # TRAIN MODEL
        # =====================================

        (
            model,
            scaler,
            metrics,
            feature_importance
        ) = train_model(pdf)

        # =====================================
        # SAVE MODEL
        # =====================================

        save_model(
            model,
            scaler,
            metrics,
            feature_importance
        )

        logger.info(
            "Training completed successfully."
        )

        logger.info(
            f"Final Test R²: "
            f"{metrics['test_r2']:.6f}"
        )

        logger.info(
            f"Final Test RMSE: "
            f"{metrics['test_rmse']:.6f}"
        )

    except Exception as ex:

        logger.exception(
            "Training failed"
        )

        raise ex

    finally:

        spark.stop()

        logger.info(
            "Spark session stopped"
        )


# =========================================================
# ENTRY POINT
# =========================================================

def main():

    train_with_random_forest()


if __name__ == "__main__":

    main()
