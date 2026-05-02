# my_dag.py
from airflow.decorators import dag, task
from datetime import datetime

from pyspark import SparkContext
from pyspark.sql import SparkSession
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
import pandas as pd

@dag(
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
)
def my_dag():
    read_data = SparkSubmitOperator(task_id="read_data",
                                    application="./include/scripts/read.py",
                                    conn_id="my_spark_conn",
                                    verbose=True)

    read_data
my_dag()


