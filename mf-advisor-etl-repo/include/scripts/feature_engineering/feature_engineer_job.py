"""
Feature Engineering Job for Telco Customer Churn Dataset

This script reads a raw PostgreSQL table containing customer data, performs
feature engineering, handles missing values, encodes categorical variables,
and creates additional features to support predictive modeling.

Usage:
    spark-submit read.py <RAW_DATA_DB_TABLE_NAME>
"""

import sys
import traceback
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pyspark.sql import SparkSession
from include.scripts.utils import _get_db_connection, _get_jdbc_props, persist_to_postgres


# ---------------- Utility Functions ---------------- #

def init_spark(app_name="FeatureEngineering_job"):
    """Initialize and return a SparkSession."""
    return SparkSession.builder.appName(app_name).getOrCreate()


def read_from_postgres(spark, raw_table):
    """Read raw table from PostgreSQL into Spark DataFrame."""
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


def preprocess_data(pdf):
    """Handle missing values, drop unnecessary columns, convert datatypes."""
    print("✅ Initial dataset shape:", pdf.shape)

    # Dataset summary
    print("\n ✅Statistical Summary:")
    print(pdf.describe())

    # Missing values
    missing_values = pdf.isnull().sum()
    print("***** ✅Missing values in each column: ********")
    print(missing_values[missing_values > 0])

    # Convert total_charges to numeric and drop missing
    pdf['total_charges'] = pd.to_numeric(pdf['total_charges'], errors='coerce')
    pdf.dropna(subset=['total_charges'], inplace=True)

    # Drop unique identifier column
    if 'customer_id' in pdf.columns:
        pdf.drop(columns=['customer_id'], inplace=True)

    return pdf


def feature_engineering(pdf):
    """Encode categorical variables and scale numerical features."""
    # Encode binary categorical columns
    binary_cols = ['partner', 'dependents', 'phone_service', 'paperless_billing', 'churn']
    le = LabelEncoder()
    for col in binary_cols:
        if col in pdf.columns:
            pdf[col] = le.fit_transform(pdf[col])

    # Scale numerical columns
    scaler = StandardScaler()
    scaled_cols = ['tenure_in_months', 'monthly_charge', 'total_charges']
    pdf[scaled_cols] = scaler.fit_transform(pdf[scaled_cols])

    return pdf


def save_to_postgres(spark_df, db_conn, cursor, table_name="telco_customer_churn_processed"):
    """Persist processed Spark DataFrame to PostgreSQL."""
    persist_to_postgres(spark_df, db_conn, cursor, table_name)
    db_conn.commit()
    print(f"✅Data persisted successfully into table: {table_name}")


# ---------------- Main Orchestration ---------------- #

def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: spark-submit read.py <RAW_DATA_DB_TABLE_NAME>")

    raw_zone_table = sys.argv[1]
    print(f"✅Raw zone table name: {raw_zone_table}")

    spark, db_conn, cursor = None, None, None

    try:
        # Step 1: Initialize Spark
        spark = init_spark()

        # Step 2: Read raw data from PostgreSQL
        raw_df, db_conn, cursor = read_from_postgres(spark, raw_zone_table)

        # Step 3: Convert to Pandas for easier manipulation
        pdf = raw_df.toPandas()

        # Step 4: Preprocess data
        pdf = preprocess_data(pdf)

        # Step 5: Apply feature engineering
        pdf = feature_engineering(pdf)

        # Step 6: Convert Pandas back to Spark
        spark_df = spark.createDataFrame(pdf)
        print("\n ✅Schema of Spark DataFrame after feature engineering:")
        spark_df.printSchema()

        # Step 7: Persist processed data into Postgres
        save_to_postgres(spark_df, db_conn, cursor)

    except Exception as e:
        print("Error occurred during feature engineering job")
        traceback.print_exc()

        if db_conn:
            db_conn.rollback()
            print("Rolled back DB transaction")

        raise RuntimeError(f"Feature Engineering job failed: {str(e)}") from e

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
