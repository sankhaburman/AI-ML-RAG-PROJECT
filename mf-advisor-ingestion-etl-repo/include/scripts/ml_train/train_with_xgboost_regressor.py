import logging
import os
import json
import pickle
import joblib

from airflow.hooks.base import BaseHook

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.model_selection import (
    TimeSeriesSplit,
    cross_val_score
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

logger = logging.getLogger(__name__)

# =========================================================
# CONFIGURATION
# =========================================================
APP_NAME = "XGBOOST-MF-ADVISOR-TRAINING"
POSTGRES_DRIVER = "org.postgresql.Driver"
MODEL_BASE_PATH = "/tmp/models"
MODEL_PATH = (
    f"{MODEL_BASE_PATH}/xgboost_model.pkl"
)
METADATA_PATH = (
    f"{MODEL_BASE_PATH}/model_metadata.json"
)
FEATURE_IMPORTANCE_PATH = (
    f"{MODEL_BASE_PATH}/feature_importance.csv"
)
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
# MEMORY CONTROL
# =========================================================
MAX_TRAINING_ROWS = 500000
SAMPLE_FRACTION = 0.20

# =========================================================
# SPARK CONFIG
# =========================================================
SPARK_EXECUTOR_MEMORY = "4g"
SPARK_DRIVER_MEMORY = "6g"
SPARK_SHUFFLE_PARTITIONS = "32"

# =========================================================
# XGBOOST CONFIG
# =========================================================
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 40,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "gamma": 0.5,
    "reg_alpha": 5.0,
    "reg_lambda": 10.0,
    "random_state": 42,
    "n_jobs": -1
}
# =========================================================
# SPARK SESSION
# =========================================================
def create_spark_session():
    logger.info(
        "Creating Spark Session..."
    )
    return (
        SparkSession.builder
        .appName(APP_NAME)
        .config(
            "spark.jars.packages",
            "org.postgresql:postgresql:42.7.3"
        )
        .config(
            "spark.executor.memory",
            SPARK_EXECUTOR_MEMORY
        )
        .config(
            "spark.driver.memory",
            SPARK_DRIVER_MEMORY
        )
        .config(
            "spark.driver.maxResultSize",
            "2g"
        )
        .config(
            "spark.sql.shuffle.partitions",
            SPARK_SHUFFLE_PARTITIONS
        )
        .config(
            "spark.sql.execution.arrow.pyspark.enabled",
            "true"
        )
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
    logger.info(
        "Postgres connection loaded"
    )
    return {
        "jdbc_url":
            jdbc_url,
        "user":
            conn.login,
        "password":
            conn.password
    }

# =========================================================
# LOAD TRAINING DATA
# =========================================================
def load_training_dataframe(
        spark,
        connection
):

    logger.info("Loading training dataset...")
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
        spark.read
        .format("jdbc")
        .option(
            "url",
            connection["jdbc_url"]
        )
        .option(
            "dbtable",
            query
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
            "fetchsize",
            "5000"
        )
        .option(
            "partitionColumn",
            "scheme_code"
        )
        .option(
            "lowerBound",
            1
        )
        .option(
            "upperBound",
            200000
        )
        .option(
            "numPartitions",
            8
        )
        .load()
    )
    logger.info(
        f"Rows Loaded : {spark_df.count()}"
    )
    # ==========================================
    # FIX DECIMAL TYPES
    # ==========================================
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
# FEATURE LIST
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
# NUMERIC CONVERSION
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
# ADVANCED FEATURE ENGINEERING
# =========================================================
def create_advanced_features(pdf):

    logger.info(
        "Creating advanced XGBoost features..."
    )

    # ==========================================
    # MOMENTUM
    # ==========================================

    pdf["momentum_90d"] = (
                                  (
                                          pdf["nav"] -
                                          pdf["moving_avg_90d"]
                                  )
                                  /
                                  pdf["moving_avg_90d"]
                          ) * 100

    pdf["momentum_30d"] = (
                                  (
                                          pdf["nav"] -
                                          pdf["moving_avg_30d"]
                                  )
                                  /
                                  pdf["moving_avg_30d"]
                          ) * 100

    pdf["momentum_200d"] = (
                                   (
                                           pdf["nav"] -
                                           pdf["moving_avg_200d"]
                                   )
                                   /
                                   pdf["moving_avg_200d"]
                           ) * 100

    # ==========================================
    # RISK ADJUSTED RETURN
    # ==========================================

    pdf["risk_adjusted_return"] = (
            pdf["rolling_return_90d_pct"]
            /
            pdf["annualized_volatility"]
            .replace(0, np.nan)
    )

    # ==========================================
    # QUALITY SCORE
    # ==========================================

    pdf["quality_score"] = (
            pdf["sharpe_ratio"]
            *
            np.log1p(
                np.maximum(
                    pdf["cagr_percent"],
                    0
                )
            )
    )

    logger.info(
        "Advanced feature engineering completed"
    )

    return pdf

# =========================================================
# FEATURE LIST FOR XGBOOST
# =========================================================

def get_xgboost_features():

    return [

        # ======================================
        # RETURNS
        # ======================================

        "daily_return_pct",
        "weekly_return_pct",
        "monthly_return_pct",

        "rolling_return_30d_pct",
        "rolling_return_90d_pct",

        # ======================================
        # MOVING AVERAGES
        # ======================================

        "moving_avg_7d",
        "moving_avg_30d",
        "moving_avg_90d",
        "moving_avg_200d",

        # ======================================
        # RISK METRICS
        # ======================================

        "sharpe_ratio",
        "annualized_volatility",

        # ======================================
        # ENGINEERED FEATURES
        # ======================================

        "momentum_30d",
        "momentum_90d",
        "momentum_200d",

        "risk_adjusted_return",

        "quality_score"
    ]
# =========================================================
# TARGET CREATION
# =========================================================

def create_target_column(pdf):

    logger.info(
        "Creating future return target..."
    )

    pdf = pdf.sort_values(
        [
            "scheme_code",
            "nav_date"
        ]
    ).copy()

    pdf["future_nav"] = (

        pdf.groupby(
            "scheme_code"
        )["nav"]

        .shift(
            -PREDICTION_HORIZON_DAYS
        )
    )

    pdf["future_return_pct"] = (

                                   (
                                           (
                                                   pdf["future_nav"]
                                                   -
                                                   pdf["nav"]
                                           )

                                           /

                                           pdf["nav"]
                                   )

                               ) * 100.0

    pdf = pdf[

        (
            pdf["future_return_pct"]
            .notna()
        )

        &

        (
                pdf["nav"] > 0
        )
        ]

    logger.info(
        f"Rows after target creation: {len(pdf)}"
    )

    return pdf


# =========================================================
# REMOVE EXTREME OUTLIERS
# =========================================================

def remove_target_outliers(pdf):

    logger.info("Removing extreme target outliers...")

    q01 = pdf["future_return_pct"].quantile(0.01)
    q99 = pdf["future_return_pct"].quantile(0.99)

    pdf["future_return_pct"] = (
    pdf["future_return_pct"]
    .clip(q01, q99)
)
    logger.info(f"Rows after outlier removal: {len(pdf)}")
    return pdf


# =========================================================
# DATASET PREPARATION
# =========================================================

def prepare_dataset(pdf):

    feature_cols = (
        get_xgboost_features()
    )

    required_cols = (

            feature_cols

            +

            [
                "future_return_pct"
            ]
    )

    pdf = pdf.dropna(
        subset=required_cols
    )

    pdf = pdf.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    pdf = pdf.dropna(
        subset=required_cols
    )

    if len(pdf) == 0:

        raise ValueError(
            "No rows left after preparation"
        )

    logger.info(
        f"Prepared dataset rows: {len(pdf)}"
    )

    return pdf


# =========================================================
# TIME SERIES SPLIT
# =========================================================

def create_train_val_test_split(pdf):

    logger.info(
        "Creating time-based split..."
    )

    pdf = pdf.sort_values(
        "nav_date"
    )

    train_cutoff = (
        pdf["nav_date"]
        .quantile(0.60)
    )

    val_cutoff = (
        pdf["nav_date"]
        .quantile(0.80)
    )

    train_df = pdf[
        pdf["nav_date"]
        <= train_cutoff
        ].copy()

    val_df = pdf[

        (
                pdf["nav_date"]
                > train_cutoff
        )

        &

        (
                pdf["nav_date"]
                <= val_cutoff
        )

        ].copy()

    test_df = pdf[
        pdf["nav_date"]
        > val_cutoff
        ].copy()

    logger.info(
        f"Train Rows: {len(train_df)}"
    )

    logger.info(
        f"Validation Rows: {len(val_df)}"
    )

    logger.info(
        f"Test Rows: {len(test_df)}"
    )

    return (
        train_df,
        val_df,
        test_df
    )


# =========================================================
# LOAD AND PREPARE PANDAS DATASET
# =========================================================

def load_and_prepare_dataset(spark_df):

    logger.info(
        "Converting Spark dataframe to Pandas..."
    )

    spark_df = spark_df.coalesce(4)

    pdf = spark_df.toPandas()

    logger.info(
        f"Pandas shape: {pdf.shape}"
    )

    if len(pdf) > MAX_TRAINING_ROWS:

        logger.warning(
            f"Sampling dataset from "
            f"{len(pdf)} rows "
            f"to {MAX_TRAINING_ROWS}"
        )

        pdf = pdf.sample(
            n=MAX_TRAINING_ROWS,
            random_state=42
        )

    pdf["nav_date"] = pd.to_datetime(
        pdf["nav_date"],
        errors="coerce"
    )

    pdf = convert_numeric_columns(
        pdf
    )

    pdf = create_advanced_features(
        pdf
    )

    pdf = create_target_column(
        pdf
    )

    pdf = remove_target_outliers(
        pdf
    )

    pdf = prepare_dataset(
        pdf
    )

    logger.info(
        f"Final Training Rows: {len(pdf)}"
    )

    return pdf

# =========================================================
# TRAIN XGBOOST MODEL
# =========================================================

def train_xgboost_model(pdf):

    logger.info("=" * 80)
    logger.info("STARTING XGBOOST TRAINING")
    logger.info("=" * 80)

    feature_cols = get_xgboost_features()

    (
        train_df,
        val_df,
        test_df
    ) = create_train_val_test_split(pdf)

    # =====================================================
    # FEATURES / TARGET
    # =====================================================

    X_train = train_df[feature_cols]

    y_train = train_df[
        "future_return_pct"
    ].astype(float)

    X_val = val_df[feature_cols]

    y_val = val_df[
        "future_return_pct"
    ].astype(float)

    X_test = test_df[feature_cols]

    y_test = test_df[
        "future_return_pct"
    ].astype(float)

    logger.info(
        f"Training Shape : {X_train.shape}"
    )

    logger.info(
        f"Validation Shape : {X_val.shape}"
    )

    logger.info(
        f"Test Shape : {X_test.shape}"
    )

    # =====================================================
    # XGBOOST MODEL
    # =====================================================

    model = XGBRegressor(
        **XGB_PARAMS,
        early_stopping_rounds=50
    )

    logger.info(
        "Training XGBoost model..."
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_val, y_val)
        ],
        #early_stopping_rounds=50,
        verbose=50
    )

    logger.info(
        "XGBoost training completed"
    )

    # =====================================================
    # PREDICTIONS
    # =====================================================

    y_pred_train = model.predict(
        X_train
    )

    y_pred_val = model.predict(
        X_val
    )

    y_pred_test = model.predict(
        X_test
    )

    # =====================================================
    # R²
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

    r2_gap = (
            train_r2 -
            test_r2
    )

    rmse_gap = (
            test_rmse -
            train_rmse
    )

    logger.info("=" * 80)
    logger.info("MODEL PERFORMANCE")
    logger.info("=" * 80)

    logger.info(
        f"Train R² : {train_r2:.6f}"
    )

    logger.info(
        f"Validation R² : {val_r2:.6f}"
    )

    logger.info(
        f"Test R² : {test_r2:.6f}"
    )

    logger.info(
        f"Train RMSE : {train_rmse:.6f}"
    )

    logger.info(
        f"Validation RMSE : {val_rmse:.6f}"
    )

    logger.info(
        f"Test RMSE : {test_rmse:.6f}"
    )

    logger.info(
        f"Train MAE : {train_mae:.6f}"
    )

    logger.info(
        f"Validation MAE : {val_mae:.6f}"
    )

    logger.info(
        f"Test MAE : {test_mae:.6f}"
    )

    logger.info(
        f"R² Gap : {r2_gap:.6f}"
    )

    logger.info(
        f"RMSE Gap : {rmse_gap:.6f}"
    )

    if r2_gap > 0.03:

        logger.warning(
            "Potential overfitting detected"
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

        .reset_index(
            drop=True
        )
    )

    logger.info(
        "\nTop Features:\n%s",
        feature_importance.head(20)
    )

    # =====================================================
    # TOP FEATURE
    # =====================================================

    top_feature = (
        feature_importance.iloc[0]
    )

    logger.info(
        f"Most Important Feature: "
        f"{top_feature['feature']} "
        f"({top_feature['importance']:.6f})"
    )

    # =====================================================
    # TIME SERIES CROSS VALIDATION
    # =====================================================

    logger.info("=" * 80)
    logger.info("TIME SERIES CV")
    logger.info("=" * 80)

    tscv = TimeSeriesSplit(
        n_splits=5
    )

    cv_scores = []

    fold_number = 1

    for train_idx, test_idx in tscv.split(X_train):

        X_fold_train = (
            X_train.iloc[train_idx]
        )

        y_fold_train = (
            y_train.iloc[train_idx]
        )

        X_fold_test = (
            X_train.iloc[test_idx]
        )

        y_fold_test = (
            y_train.iloc[test_idx]
        )

        fold_model = XGBRegressor(
            **XGB_PARAMS
        )

        fold_model.fit(
            X_fold_train,
            y_fold_train,
            verbose=False
        )

        fold_prediction = (
            fold_model.predict(
                X_fold_test
            )
        )

        fold_r2 = r2_score(
            y_fold_test,
            fold_prediction
        )

        cv_scores.append(
            fold_r2
        )

        logger.info(
            f"Fold {fold_number} "
            f"R² = {fold_r2:.6f}"
        )

        fold_number += 1

    cv_mean = np.mean(
        cv_scores
    )

    cv_std = np.std(
        cv_scores
    )

    logger.info(
        f"CV Mean R² = {cv_mean:.6f}"
    )

    logger.info(
        f"CV Std R² = {cv_std:.6f}"
    )

    # =====================================================
    # SHAP EXPLAINABILITY
    # =====================================================

    try:

        import shap

        logger.info(
            "Calculating SHAP values..."
        )

        sample_rows = min(
            5000,
            len(X_test)
        )

        X_shap = (
            X_test
            .sample(
                sample_rows,
                random_state=42
            )
        )

        explainer = (
            shap.TreeExplainer(
                model
            )
        )

        shap_values = (
            explainer.shap_values(
                X_shap
            )
        )

        shap_importance = pd.DataFrame(
            {
                "feature":
                    feature_cols,

                "mean_abs_shap":
                    np.abs(
                        shap_values
                    ).mean(axis=0)
            }
        )

        shap_importance = (

            shap_importance

            .sort_values(
                by="mean_abs_shap",
                ascending=False
            )

            .reset_index(
                drop=True
            )
        )

        logger.info(
            "\nTop SHAP Features:\n%s",
            shap_importance.head(20)
        )

    except Exception as ex:

        logger.warning(
            f"SHAP skipped : {str(ex)}"
        )

        shap_importance = pd.DataFrame()

    # =====================================================
    # TOP FUND RANKING EVALUATION
    # =====================================================

    logger.info("=" * 80)
    logger.info("TOP FUND RANKING TEST")
    logger.info("=" * 80)

    ranking_df = test_df.copy()

    ranking_df[
        "predicted_return"
    ] = y_pred_test

    ranking_df = (
        ranking_df
        .sort_values(
            by="predicted_return",
            ascending=False
        )
    )

    top10 = ranking_df.head(10)

    logger.info(
        "\nTop 10 Recommended Funds:\n%s",
        top10[
            [
                "scheme_code",
                "predicted_return",
                "future_return_pct",
                "cagr_percent",
                "sharpe_ratio"
            ]
        ]
    )

    # =====================================================
    # METRICS
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

        "r2_gap":
            float(r2_gap),

        "rmse_gap":
            float(rmse_gap),

        "cv_mean_r2":
            float(cv_mean),

        "cv_std_r2":
            float(cv_std)
    }

    return (

        model,

        metrics,

        feature_importance,

        shap_importance
    )

# =========================================================
# SAVE MODEL
# =========================================================

def save_model(
        model,
        metrics,
        feature_importance,
        shap_importance
):

    logger.info("=" * 80)
    logger.info("SAVING MODEL ARTIFACTS")
    logger.info("=" * 80)

    os.makedirs(
        MODEL_BASE_PATH,
        exist_ok=True
    )

    # ==========================================
    # SAVE MODEL
    # ==========================================

    with open(
            MODEL_PATH,
            "wb"
    ) as model_file:

        pickle.dump(
            model,
            model_file
        )

    logger.info(
        f"Model saved : {MODEL_PATH}"
    )

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False
    )

    logger.info(
        f"Feature importance saved : "
        f"{FEATURE_IMPORTANCE_PATH}"
    )

    # ==========================================
    # SHAP IMPORTANCE
    # ==========================================

    shap_path = (
        f"{MODEL_BASE_PATH}/"
        f"shap_importance.csv"
    )

    if not shap_importance.empty:

        shap_importance.to_csv(
            shap_path,
            index=False
        )

        logger.info(
            f"SHAP importance saved : "
            f"{shap_path}"
        )

    # ==========================================
    # METADATA
    # ==========================================

    metadata = {

        "model_type":
            "XGBRegressor",

        "training_date":
            pd.Timestamp.now()
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "prediction_horizon_days":
            PREDICTION_HORIZON_DAYS,

        "training_window_days":
            TRAINING_WINDOW_DAYS,

        "features":
            get_xgboost_features(),

        "metrics":
            metrics
    }

    with open(
            METADATA_PATH,
            "w"
    ) as metadata_file:

        json.dump(
            metadata,
            metadata_file,
            indent=4
        )

    logger.info(
        f"Metadata saved : "
        f"{METADATA_PATH}"
    )


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_training_summary(
        metrics,
        feature_importance
):

    logger.info("=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)

    logger.info(
        f"Train R² : "
        f"{metrics['train_r2']:.6f}"
    )

    logger.info(
        f"Validation R² : "
        f"{metrics['val_r2']:.6f}"
    )

    logger.info(
        f"Test R² : "
        f"{metrics['test_r2']:.6f}"
    )

    logger.info(
        f"CV Mean R² : "
        f"{metrics['cv_mean_r2']:.6f}"
    )

    logger.info(
        f"CV Std R² : "
        f"{metrics['cv_std_r2']:.6f}"
    )

    logger.info(
        f"Test RMSE : "
        f"{metrics['test_rmse']:.6f}"
    )

    logger.info(
        f"Test MAE : "
        f"{metrics['test_mae']:.6f}"
    )

    logger.info(
        f"R² Gap : "
        f"{metrics['r2_gap']:.6f}"
    )

    logger.info(
        "\nTop 10 Features\n%s",
        feature_importance.head(10)
    )


# =========================================================
# DATA QUALITY CHECK
# =========================================================

def perform_data_quality_checks(
        pdf
):

    logger.info("=" * 80)
    logger.info("DATA QUALITY CHECKS")
    logger.info("=" * 80)

    logger.info(
        f"Rows : {len(pdf)}"
    )

    logger.info(
        f"Columns : {len(pdf.columns)}"
    )

    missing_df = (
        pdf.isnull()
        .sum()
        .reset_index()
    )

    missing_df.columns = [
        "column",
        "missing_count"
    ]

    missing_df = missing_df[
        missing_df["missing_count"] > 0
        ]

    if len(missing_df) > 0:

        logger.warning(
            "\nMissing Values\n%s",
            missing_df
        )

    else:

        logger.info(
            "No missing values found"
        )

    target_stats = pdf[
        "future_return_pct"
    ].describe()

    logger.info(
        "\nTarget Statistics\n%s",
        target_stats
    )


# =========================================================
# TRAIN PIPELINE
# =========================================================

def train_with_xgboost():

    logger.info("=" * 80)
    logger.info(
        "STARTING MF ADVISOR XGBOOST TRAINING"
    )
    logger.info("=" * 80)

    spark = None

    try:

        # ======================================
        # SPARK
        # ======================================

        spark = create_spark_session()

        # ======================================
        # CONNECTION
        # ======================================

        connection = (
            get_postgres_connection()
        )

        # ======================================
        # LOAD DATA
        # ======================================

        spark_df = (
            load_training_dataframe(
                spark=spark,
                connection=connection
            )
        )

        # ======================================
        # PREPARE DATASET
        # ======================================

        pdf = (
            load_and_prepare_dataset(
                spark_df
            )
        )

        # ======================================
        # QUALITY CHECK
        # ======================================

        perform_data_quality_checks(
            pdf
        )

        # ======================================
        # TRAIN MODEL
        # ======================================

        (
            model,
            metrics,
            feature_importance,
            shap_importance

        ) = train_xgboost_model(
            pdf
        )

        # ======================================
        # SAVE MODEL
        # ======================================

        save_model(
            model=model,
            metrics=metrics,
            feature_importance=
            feature_importance,
            shap_importance=
            shap_importance
        )

        # ======================================
        # SUMMARY
        # ======================================

        print_training_summary(
            metrics,
            feature_importance
        )

        logger.info("=" * 80)
        logger.info(
            "TRAINING COMPLETED SUCCESSFULLY"
        )
        logger.info("=" * 80)

    except Exception as ex:

        logger.exception(
            f"Training Failed : "
            f"{str(ex)}"
        )

        raise

    finally:

        if spark:

            spark.stop()

            logger.info(
                "Spark session stopped"
            )


# =========================================================
# MAIN
# =========================================================

def main():

    train_with_xgboost()


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s "
            "%(message)s"
        )
    )

    main()