from airflow import DAG
from airflow.providers.http.hooks.http import HttpHook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.decorators import task
from airflow.utils.dates import days_ago
import requests
import json

POSTGRES_CONN_ID='postgres_default'
API_CONN_ID='mf_api'

default_args={
    'owner':'airflow',
    'start_date':days_ago(1)
}

## DAG
with DAG(dag_id='mf_advisor_ingestion_pipeline',
         default_args=default_args,
         schedule_interval='@weekly',
         catchup=False) as dags:

    @task
    def fetch_all_mutual_fund():
        """Extract weather data from Open-Meteo API using Airflow Connection."""

        # Use HTTP Hook to get connection details from Airflow connection

        http_hook=HttpHook(http_conn_id=API_CONN_ID,method='GET')

        ## Build the API endpoint
        ## https://api.open-meteo.com/v1/forecast?latitude=51.5074&longitude=-0.1278&current_weather=true
        endpoint=f'/mf'

        ## Make the request via the HTTP Hook
        response=http_hook.run(endpoint)

        if response.status_code == 200:
            resp =  response.json()
            transformed_obj = transform_scheme_data(resp)
            load_scheme_data(transformed_obj)
        else:
            raise Exception(f"Failed to fetch weather data: {response.status_code}")


    def transform_scheme_data(raw_json):
        """
        Convert JSON list → list of tuples
        """
        transformed = []

        for item in raw_json:
            transformed.append((
            item.get("schemeCode"),
            item.get("schemeName"),
            item.get("isinGrowth"),
            item.get("isinDivReinvestment")
        ))
        return transformed


    def load_scheme_data(schemes):
        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

        create_table_query = """
        CREATE TABLE IF NOT EXISTS mf_schemes (
            scheme_code BIGINT PRIMARY KEY,
            scheme_name TEXT,
            isin_growth TEXT,
            isin_div_reinvestment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        insert_query = """
        INSERT INTO mf_schemes (
            scheme_code,
            scheme_name,
            isin_growth,
            isin_div_reinvestment
        )
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (scheme_code) DO NOTHING;
        """

        pg_hook.run(create_table_query)

        pg_hook.insert_rows(
            table="mf_schemes",
            rows=schemes,
            target_fields=[
                "scheme_code",
                "scheme_name",
                "isin_growth",
                "isin_div_reinvestment"
            ]
        )

    ## DAG Worflow- ETL Pipeline
    weather_data= fetch_all_mutual_fund()
    #transformed_data=transform_weather_data(weather_data)
    #load_weather_data(transformed_data)
