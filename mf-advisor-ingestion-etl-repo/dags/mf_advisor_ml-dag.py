from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.decorators import task
from airflow.utils.dates import days_ago
import logging

#########################################################
# LOGGER CONFIGURATION
#########################################################

logger = logging.getLogger(__name__)

POSTGRES_CONN_ID = "postgres_default"
API_CONN_ID = "mf_api"

BATCH_SIZE = 1000
INSERT_BATCH_SIZE = 7000

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
        tags=["mfadvisor-model-training"],
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
        )

        spark_task.execute(context={})

        logger.info("=======================================================")
        logger.info("Random Forest Regressor Training Completed")
        logger.info("=======================================================")

        return "random_forest_completed"

    #########################################################
    # DAG FLOW
    #########################################################

    linear_task = train_ml_with_linear_regression()
    random_forest_task = train_ml_with_random_forest()

    linear_task >> random_forest_task