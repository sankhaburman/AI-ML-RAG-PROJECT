from datetime import datetime, date
import logging

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.http.hooks.http import HttpHook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.decorators import task
from airflow.utils.dates import days_ago
from psycopg2.extras import execute_batch

logger = logging.getLogger(__name__)

POSTGRES_CONN_ID = 'postgres_default'
API_CONN_ID = 'mf_api'

BATCH_SIZE = 1000
INSERT_BATCH_SIZE = 7000

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1)
}


# =========================================================
# DAG DEFINITION
# =========================================================
with DAG(
        dag_id='mf_advisor_ingestion_pipeline',
        default_args=default_args,
        schedule_interval='@weekly',
        catchup=False,
        tags=["mf-advisor"]
) as dag:


    # =========================================================
    # TABLE CREATION (UNCHANGED)
    # =========================================================
    @task
    def create_tables():
        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

        pg_hook.run("""
        CREATE TABLE IF NOT EXISTS mf_schemes (
            scheme_code BIGINT PRIMARY KEY,
            scheme_name TEXT,
            isin_growth TEXT,
            isin_div_reinvestment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        pg_hook.run("""
        CREATE TABLE IF NOT EXISTS mf_raw_nav (
            scheme_code BIGINT,
            nav_date DATE,
            nav NUMERIC(18,6),
            fund_house TEXT,
            scheme_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (scheme_code, nav_date)
        );
        """)

        pg_hook.run("""
        CREATE INDEX IF NOT EXISTS idx_mf_raw_nav_scheme_date
        ON mf_raw_nav (scheme_code, nav_date);
        """)

        logger.info("Tables ready")


    # =========================================================
    # SAFE DATE PARSER (CRITICAL FIX)
    # =========================================================
    def to_date(value):
        if value is None:
            return None

        if isinstance(value, date):
            return value

        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, str):
            value = value.strip()

            # ISO format
            try:
                return datetime.fromisoformat(value).date()
            except Exception:
                pass

            # DD-MM-YYYY (FIXED ISSUE)
            try:
                return datetime.strptime(value, "%d-%m-%Y").date()
            except Exception:
                pass

            # YYYY-MM-DD fallback
            try:
                return datetime.strptime(value, "%Y-%m-%d").date()
            except Exception:
                pass

        raise ValueError(f"Invalid date format: {value}")


    # =========================================================
    # NAV INSERT BATCH
    # =========================================================
    def insert_nav_data_batch(nav_rows):
        if not nav_rows:
            return

        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        conn = pg_hook.get_conn()
        cursor = conn.cursor()

        cursor.executemany("""
            INSERT INTO mf_raw_nav (
                scheme_code,
                nav_date,
                nav,
                fund_house,
                scheme_name
            )
            VALUES (%s, TO_DATE(%s, 'DD-MM-YYYY'), %s, %s, %s)
            ON CONFLICT (scheme_code, nav_date) DO NOTHING;
        """, nav_rows)

        conn.commit()
        cursor.close()
        conn.close()


    # =========================================================
    # FETCH NAV DATA (FIXED + STABLE)
    # =========================================================
    @task
    def fetch_nav_data():

        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        http_hook = HttpHook(http_conn_id=API_CONN_ID, method='GET')

        # -----------------------------
        # Schemes
        # -----------------------------
        scheme_codes = [
            row[0] for row in pg_hook.get_records("""
                SELECT scheme_code FROM mf_schemes ORDER BY scheme_code;
            """)
        ]

        # -----------------------------
        # Latest NAV per scheme (FAST WINDOW)
        # -----------------------------
        existing_records = pg_hook.get_records("""
            SELECT scheme_code, MAX(nav_date)
            FROM mf_raw_nav
            GROUP BY scheme_code;
        """)

        latest_nav_map = {
            row[0]: to_date(row[1]) for row in existing_records
        }

        insert_buffer = []

        # -----------------------------
        # API processing
        # -----------------------------
        for batch_start in range(0, len(scheme_codes), BATCH_SIZE):

            batch = scheme_codes[batch_start:batch_start + BATCH_SIZE]

            for scheme_code in batch:

                try:
                    response = http_hook.run(f"/mf/{scheme_code}")

                    if response.status_code != 200:
                        continue

                    data = response.json()
                    meta = data.get("meta", {})
                    nav_list = data.get("data", [])

                    fund_house = meta.get("fund_house")
                    scheme_name = meta.get("scheme_name")

                    last_nav_date = latest_nav_map.get(scheme_code)

                    for nav_item in nav_list:

                        raw_date = nav_item.get("date")
                        nav_value = nav_item.get("nav")

                        try:
                            nav_date = to_date(raw_date)
                        except Exception:
                            logger.error(f"Skipping bad date {raw_date} for {scheme_code}")
                            continue

                        if last_nav_date and nav_date and nav_date <= last_nav_date:
                            continue

                        insert_buffer.append(
                            (
                                scheme_code,
                                raw_date,
                                nav_value,
                                fund_house,
                                scheme_name
                            )
                        )

                    # update latest date safely
                    if nav_list:
                        try:
                            max_date = max(
                                to_date(x.get("date"))
                                for x in nav_list
                                if x.get("date")
                            )

                            latest_nav_map[scheme_code] = max(
                                latest_nav_map.get(scheme_code, max_date),
                                max_date
                            )

                        except Exception as e:
                            logger.error(f"Max date error {scheme_code}: {str(e)}")

                except Exception as e:
                    logger.error(f"Error processing {scheme_code}: {str(e)}")

            # batch insert
            if insert_buffer:
                insert_nav_data_batch(insert_buffer)
                insert_buffer.clear()

        if insert_buffer:
            insert_nav_data_batch(insert_buffer)

        logger.info("NAV ingestion completed")


    # =========================================================
    # SPARK TASKS (UNCHANGED)
    # =========================================================
    @task
    def enrich_scheme_with_daily_returns():
        SparkSubmitOperator(
            task_id="daily_return_task",
            application="./include/scripts/mf_enricher/calculate_daily_return.py",
            conn_id="my_spark_conn",
            verbose=True,
            packages="org.postgresql:postgresql:42.7.3",
            jars="/opt/spark/jars/postgresql-42.7.3.jar"
        ).execute(context={})


    @task
    def enrich_scheme_with_aggregated_metrics():
        SparkSubmitOperator(
            task_id="aggregated_metrics_task",
            application="./include/scripts/mf_enricher/aggregated_metrics_return.py",
            conn_id="my_spark_conn",
            verbose=True,
            packages="org.postgresql:postgresql:42.7.3",
            jars="/opt/spark/jars/postgresql-42.7.3.jar"
        ).execute(context={})


    @task
    def combine_nav_enriched_data():
        SparkSubmitOperator(
            task_id="final_nav_enrich_task",
            application="./include/scripts/mf_enricher/combine_enriched_nav.py",
            conn_id="my_spark_conn",
            verbose=True,
            packages="org.postgresql:postgresql:42.7.3",
            jars="/opt/spark/jars/postgresql-42.7.3.jar"
        ).execute(context={})


    # =========================================================
    # DAG FLOW
    # =========================================================
    create_tables_task = create_tables()
    nav_task = fetch_nav_data()

    enrich_daily = enrich_scheme_with_daily_returns()
    enrich_agg = enrich_scheme_with_aggregated_metrics()
    combine = combine_nav_enriched_data()

    create_tables_task >> nav_task >> enrich_daily >> enrich_agg >> combine