from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.decorators import task
from airflow.utils.dates import days_ago

import logging
import pickle
import pandas as pd

#########################################################
# LOGGER CONFIGURATION
#########################################################

logger = logging.getLogger(__name__)

POSTGRES_CONN_ID = "postgres_default"
API_CONN_ID = "mf_api"

BATCH_SIZE = 1000
INSERT_BATCH_SIZE = 7000

MODEL_PATH = "/tmp/models/random_forest.pkl"

default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
}

#########################################################
# DAG DEFINITION
#########################################################

with DAG(
        dag_id="mf_advisor_ml_pipeline",
        default_args=default_args,
        schedule="@weekly",
        catchup=False,
        tags=["mf-advisor"],
) as dag:

    #########################################################
    # TASK 1 - LINEAR REGRESSION
    #########################################################

    @task(task_id="train_ml_with_linear_regression")
    def train_ml_with_linear_regression():

        logger.info("=======================================================")
        logger.info("Starting ML Training with Linear Regressor")
        logger.info("=======================================================")

        spark_task = SparkSubmitOperator(
            task_id="train_with_linear_regressor_task_internal",
            application="./include/scripts/ml_train/train_with_linear_regressor.py",
            conn_id="my_spark_conn",
            verbose=True,
            packages="org.postgresql:postgresql:42.7.3",
            jars="/opt/spark/jars/postgresql-42.7.3.jar",
        )

        spark_task.execute(context={})

        logger.info("=======================================================")
        logger.info("Linear Regressor Training Completed")
        logger.info("=======================================================")

        return "linear_regression_completed"

    #########################################################
    # TASK 2 - RANDOM FOREST
    #########################################################

    @task(task_id="train_ml_with_random_forest")
    def train_ml_with_random_forest():

        logger.info("=======================================================")
        logger.info("Starting ML Training with Random Forest Regressor")
        logger.info("=======================================================")

        spark_task = SparkSubmitOperator(
            task_id="train_with_random_forest_regressor_task_internal",
            application="./include/scripts/ml_train/train_with_random_forrest_regressor.py",
            conn_id="my_spark_conn",
            verbose=True,
            packages="org.postgresql:postgresql:42.7.3",
            jars="/opt/spark/jars/postgresql-42.7.3.jar",
            conf={
                "spark.driver.memory": "6g",
                "spark.executor.memory": "4g",
                "spark.driver.maxResultSize": "2g"
            }
        )

        spark_task.execute(context={})

        logger.info("=======================================================")
        logger.info("Random Forest Regressor Training Completed")
        logger.info("=======================================================")

        return "random_forest_completed"

    #########################################################
    # TASK 3 - XGBOOST
    #########################################################
    @task(task_id="train_ml_with_xgboost")
    def train_ml_with_xgboost():
        logger.info("=======================================================")
        logger.info("Starting ML Training with XGBoost Regressor")
        logger.info("=======================================================")

        spark_task = SparkSubmitOperator(
            task_id="train_with_xgboost_regressor_task_internal",
            application="./include/scripts/ml_train/train_with_xgboost_regressor.py",
            conn_id="my_spark_conn",
            verbose=True,
            packages="org.postgresql:postgresql:42.7.3",
            jars="/opt/spark/jars/postgresql-42.7.3.jar",
            conf={
                "spark.driver.memory": "6g",
                "spark.executor.memory": "4g",
                "spark.driver.maxResultSize": "2g"
            }
        )

        spark_task.execute(context={})

        logger.info("=======================================================")
        logger.info("XGBoost Regressor Training Completed")
        logger.info("=======================================================")

        return "xg_boost_training_completed"

    #########################################################
    # TASK 4 - VERIFY RF MODEL
    #########################################################

    @task(task_id="verify_random_forest_model")
    def verify_random_forest_model():

        logger.info("=======================================================")
        logger.info("Starting Random Forest Model Verification")
        logger.info("=======================================================")

        logger.info(f"Loading model from {MODEL_PATH}")

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        logger.info(
            f"Model loaded successfully: {type(model)}"
        )

        logger.info(
            f"Number of trees: {len(model.estimators_)}"
        )

        if hasattr(model, "feature_names_in_"):
            logger.info(
                f"Feature Names: {list(model.feature_names_in_)}"
            )

        sample_data = pd.DataFrame([
            {
                "daily_return_pct": 0.25,
                "weekly_return_pct": 1.50,
                "monthly_return_pct": 4.20,
                "rolling_return_30d_pct": 5.80,
                "rolling_return_90d_pct": 12.50,
                "moving_avg_7d": 102.40,
                "moving_avg_30d": 101.70,
                "moving_avg_90d": 99.20,
                "moving_avg_200d": 95.60,
                "cagr_percent": 14.30,
                "sharpe_ratio": 1.40,
                "annualized_volatility": 12.10,
            }
        ])

        if hasattr(model, "feature_names_in_"):
            sample_data = sample_data[
                model.feature_names_in_
            ]

        prediction = model.predict(sample_data)

        logger.info(
            f"Predicted 30-day return: {prediction[0]:.4f}%"
        )

        logger.info("=======================================================")
        logger.info("Model Verification Completed Successfully")
        logger.info("=======================================================")

        return float(prediction[0])

    #########################################################
    # DAG FLOW
    #########################################################

    linear_task = train_ml_with_linear_regression()
    random_forest_task = train_ml_with_random_forest()
    xgboost_task  = train_ml_with_xgboost()
    verify_rf_model_task = verify_random_forest_model()
    linear_task >> random_forest_task >> verify_rf_model_task >> xgboost_task