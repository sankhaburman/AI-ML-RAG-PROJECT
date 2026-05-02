"""
TrainModel & Model Training Job for Telco Customer Churn Dataset

This script performs the following:
    1. Reads a processed PostgreSQL table into Spark.
    2. Converts the Spark DataFrame to Pandas for modeling.
    3. Filters only safe features to avoid data leakage.
    4. Encodes categorical variables and handles numeric scaling.
    5. Trains and evaluates two models:
        - Decision Tree Classifier
        - Logistic Regression
    6. Prints evaluation metrics (confusion matrix, classification report, accuracy).

Usage:
    spark-submit read.py <PROCESSED_DATA_DB_TABLE_NAME>
"""

import sys
import traceback
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pyspark.sql import SparkSession
from include.scripts.utils import (
    _get_db_connection,
    _get_jdbc_props,
    persist_to_postgres,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# -----------------------------------------------------
# Safe feature list: Avoids leakage and ensures model fairness
# -----------------------------------------------------
SAFE_FEATURES = [
    "gender", "senior_citizen", "partner", "dependents",
    "tenure_in_months", "contract", "paperless_billing", "payment_method",
    "monthly_charges", "total_charges",
    "phone_service", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
    "churn"   # Target column
]


# -----------------------------------------------------
# Utility Functions
# -----------------------------------------------------

def init_spark(app_name: str = "FeatureEngineering_job") -> SparkSession:
    """
    Initialize and return a SparkSession.

    Args:
        app_name (str): Name of the Spark application.

    Returns:
        SparkSession: Active Spark session.
    """
    return SparkSession.builder.appName(app_name).getOrCreate()


def read_from_postgres(spark: SparkSession, raw_table: str):
    """
    Read table from PostgreSQL into Spark DataFrame.

    Args:
        spark (SparkSession): Active Spark session.
        raw_table (str): Name of the PostgreSQL table to read.

    Returns:
        (DataFrame, connection, cursor):
            - Spark DataFrame containing the table.
            - PostgreSQL connection object.
            - PostgreSQL cursor object.
    """
    db_conn, pg_hook = _get_db_connection()
    cursor = db_conn.cursor()
    jdbc_url, connection_props = _get_jdbc_props(pg_hook)

    print(f"✅ Reading data from PostgreSQL table: {raw_table}")

    spark_df = spark.read.jdbc(
        url=jdbc_url,
        table=raw_table,
        properties=connection_props
    )
    return spark_df, db_conn, cursor


def _filter_safe_features(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only safe Telco churn features to prevent data leakage.

    Args:
        pdf (pd.DataFrame): Input Pandas DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame with only safe features.
    """
    available = [col for col in SAFE_FEATURES if col in pdf.columns]
    return pdf[available]


def _print_evaluation(y_test, y_pred, model_name: str):
    """
    Print evaluation metrics (confusion matrix, classification report, accuracy).

    Args:
        y_test (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        model_name (str): Name of the trained model.
    """
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_No", "Actual_Yes"],
        columns=["Pred_No", "Pred_Yes"]
    )

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    print(f"\n ✅================ {model_name} Evaluation ================\n")
    print("✅Confusion Matrix:")
    print(cm_df)
    print("\n ✅Classification Report:")
    print(report_df.round(3))
    print(f"\n ✅Accuracy Score: {accuracy_score(y_test, y_pred):.3f}")
    print("=========================================================\n")


# -----------------------------------------------------
# Model Training Functions
# -----------------------------------------------------

def train_decision_tree(pdf: pd.DataFrame):
    """
    Train and evaluate a Decision Tree model.

    Args:
        pdf (pd.DataFrame): Input Pandas DataFrame.
    """
    pdf = _filter_safe_features(pdf)

    if "customer_id" in pdf.columns:
        pdf = pdf.drop(["customer_id"], axis=1)

    if "churn" not in pdf.columns:
        raise ValueError("Target column 'churn' not found in dataset")

    # Encode categorical variables
    for col in pdf.select_dtypes(include=["object"]).columns:
        if col == "churn":
            continue
        pdf[col] = LabelEncoder().fit_transform(pdf[col].astype(str))

    pdf["churn"] = LabelEncoder().fit_transform(pdf["churn"])

    # Features and target
    X = pdf.drop("churn", axis=1)
    y = pdf["churn"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    _print_evaluation(y_test, y_pred, "Decision Tree")


def train_logistic_regression(pdf: pd.DataFrame):
    """
    Train and evaluate a Logistic Regression model.

    Args:
        pdf (pd.DataFrame): Input Pandas DataFrame.
    """
    pdf = _filter_safe_features(pdf)

    if "customer_id" in pdf.columns:
        pdf = pdf.drop(["customer_id"], axis=1)

    if "churn" not in pdf.columns:
        raise ValueError("Target column 'churn' not found in dataset")

    # Encode categorical variables
    for col in pdf.select_dtypes(include=["object"]).columns:
        if col == "churn":
            continue
        pdf[col] = LabelEncoder().fit_transform(pdf[col].astype(str))

    pdf["churn"] = LabelEncoder().fit_transform(pdf["churn"])

    # Features and target
    X = pdf.drop("churn", axis=1)
    y = pdf["churn"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    _print_evaluation(y_test, y_pred, "Logistic Regression")


# -----------------------------------------------------
# Main Orchestration
# -----------------------------------------------------

def main():
    """
    Orchestrates the TrainModel and model training workflow.
    """
    if len(sys.argv) < 2:
        raise ValueError("Usage: spark-submit read.py <PROCESSED_DATA_DB_TABLE_NAME>")

    processed_zone_table = sys.argv[1]
    print(f"✅Processed zone table name: {processed_zone_table}")

    spark, db_conn, cursor = None, None, None

    try:
        # Step 1: Initialize Spark
        spark = init_spark()

        # Step 2: Read processed data from PostgreSQL
        processed_df, db_conn, cursor = read_from_postgres(spark, processed_zone_table)

        # Step 3: Convert Spark DataFrame to Pandas
        pdf = processed_df.toPandas()

        # Step 4: Train models
        train_decision_tree(pdf)
        train_logistic_regression(pdf)

    except Exception as e:
        print("Error occurred during TrainModel job")
        traceback.print_exc()

        if db_conn:
            db_conn.rollback()
            print("Rolled back DB transaction")

        raise RuntimeError(f"TrainModel job failed: {str(e)}") from e

    finally:
        if cursor:
            cursor.close()
            print("Closed Postgres cursor")
        if db_conn:
            db_conn.close()
            print("Closed Postgres connection")
        if spark:
            spark.stop()
            print("Stopped Spark session")


if __name__ == "__main__":
    main()
