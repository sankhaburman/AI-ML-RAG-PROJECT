from airflow import DAG
from airflow.providers.http.hooks.http import HttpHook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.decorators import task
from airflow.utils.dates import days_ago
from psycopg2.extras import execute_batch
import logging

#########################################################
# LOGGER CONFIGURATION
#########################################################

logger = logging.getLogger(__name__)

#########################################################
# CONSTANTS
#########################################################

POSTGRES_CONN_ID = 'postgres_default'
API_CONN_ID = 'mf_api'

BATCH_SIZE = 1000
INSERT_BATCH_SIZE = 7000

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1)
}

#########################################################
# DAG DEFINITION
#########################################################

with DAG(
        dag_id='mf_advisor_ingestion_pipeline',
        default_args=default_args,
        schedule_interval='@weekly',
        catchup=False
) as dag:

    #########################################################
    # CREATE TABLES
    #########################################################

    def create_tables():

        logger.info("Creating tables if not exists")

        pg_hook = PostgresHook(
            postgres_conn_id=POSTGRES_CONN_ID
        )

        create_scheme_table = """
        CREATE TABLE IF NOT EXISTS mf_schemes (
            scheme_code BIGINT PRIMARY KEY,
            scheme_name TEXT,
            isin_growth TEXT,
            isin_div_reinvestment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        create_nav_table = """
        CREATE TABLE IF NOT EXISTS mf_raw_nav (
            scheme_code BIGINT,
            nav_date DATE,
            nav NUMERIC(18,6),
            fund_house TEXT,
            scheme_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (scheme_code, nav_date)
        );
        """

        pg_hook.run(create_scheme_table)
        pg_hook.run(create_nav_table)

        logger.info("Tables ready")

    #########################################################
    # TRANSFORM SCHEME DATA
    #########################################################

    def transform_scheme_data(raw_json):

        logger.info("Transforming scheme data")

        transformed = []

        for item in raw_json:

            transformed.append((
                item.get("schemeCode"),
                item.get("schemeName"),
                item.get("isinGrowth"),
                item.get("isinDivReinvestment")
            ))

        logger.info(
            f"Total transformed schemes: "
            f"{len(transformed)}"
        )

        return transformed

    #########################################################
    # FILTER EXISTING SCHEMES
    #########################################################

    def filter_existing_schemes(schemes):

        logger.info("Checking existing schemes")

        pg_hook = PostgresHook(
            postgres_conn_id=POSTGRES_CONN_ID
        )

        existing_query = """
        SELECT scheme_code
        FROM mf_schemes;
        """

        existing_records = pg_hook.get_records(
            existing_query
        )

        existing_codes = {
            row[0] for row in existing_records
        }

        new_schemes = [
            scheme for scheme in schemes
            if scheme[0] not in existing_codes
        ]

        logger.info(
            f"New schemes identified: "
            f"{len(new_schemes)}"
        )

        return new_schemes

    #########################################################
    # BATCH INSERT SCHEME DATA
    #########################################################

    def load_scheme_data(schemes):

        if not schemes:
            logger.info("No schemes to insert")
            return

        logger.info(
            f"Inserting {len(schemes)} schemes"
        )

        pg_hook = PostgresHook(
            postgres_conn_id=POSTGRES_CONN_ID
        )

        conn = pg_hook.get_conn()
        cursor = conn.cursor()

        insert_query = """
        INSERT INTO mf_schemes (
            scheme_code,
            scheme_name,
            isin_growth,
            isin_div_reinvestment
        )
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (scheme_code)
        DO NOTHING;
        """

        execute_batch(
            cursor,
            insert_query,
            schemes,
            page_size=1000
        )

        conn.commit()

        cursor.close()
        conn.close()

        logger.info("Scheme insertion completed")

    #########################################################
    # DATE CONVERTER
    #########################################################

    def convert_date_format(date_str):

        day, month, year = date_str.split("-")

        return f"{year}-{month}-{day}"

    #########################################################
    # INSERT NAV DATA IN BATCHES
    #########################################################

    def insert_nav_data_batch(nav_rows):

        if not nav_rows:
            logger.info("No NAV rows to insert")
            return

        logger.info(
            f"Inserting batch of "
            f"{len(nav_rows)} NAV rows"
        )

        pg_hook = PostgresHook(
            postgres_conn_id=POSTGRES_CONN_ID
        )

        conn = pg_hook.get_conn()
        cursor = conn.cursor()

        insert_query = """
        INSERT INTO mf_raw_nav (
            scheme_code,
            nav_date,
            nav,
            fund_house,
            scheme_name
        )
        VALUES (
            %s,
            TO_DATE(%s, 'DD-MM-YYYY'),
            %s,
            %s,
            %s
        )
        ON CONFLICT (scheme_code, nav_date)
        DO NOTHING;
        """

        execute_batch(
            cursor,
            insert_query,
            nav_rows,
            page_size=1000
        )

        conn.commit()

        cursor.close()
        conn.close()

        logger.info(
            "NAV batch insertion completed"
        )

    #########################################################
    # FETCH ALL MF SCHEMES
    #########################################################

    @task
    def fetch_all_mutual_fund():

        logger.info(
            "Starting mutual fund ingestion"
        )

        http_hook = HttpHook(
            http_conn_id=API_CONN_ID,
            method='GET'
        )

        endpoint = '/mf'

        response = http_hook.run(endpoint)

        logger.info(
            f"MF API status: "
            f"{response.status_code}"
        )

        if response.status_code != 200:

            logger.error("MF API failed")

            raise Exception(
                f"API failed with "
                f"{response.status_code}"
            )

        raw_data = response.json()

        logger.info(
            f"Total schemes received: "
            f"{len(raw_data)}"
        )

        transformed_data = transform_scheme_data(
            raw_data
        )

        validated_data = filter_existing_schemes(
            transformed_data
        )

        load_scheme_data(validated_data)

    #########################################################
    # FETCH NAV DATA IN BATCHES
    #########################################################

    @task
    def fetch_nav_data():

        logger.info(
            "Starting NAV ingestion"
        )

        pg_hook = PostgresHook(
            postgres_conn_id=POSTGRES_CONN_ID
        )

        scheme_query = """
        SELECT scheme_code
        FROM mf_schemes
        ORDER BY scheme_code;
        """

        scheme_records = pg_hook.get_records(
            scheme_query
        )

        logger.info(
            f"Total schemes to process: "
            f"{len(scheme_records)}"
        )

        scheme_codes = [
            row[0] for row in scheme_records
        ]

        http_hook = HttpHook(
            http_conn_id=API_CONN_ID,
            method='GET'
        )

        #####################################################
        # PROCESS API CALLS IN BATCHES
        #####################################################

        for batch_start in range(
                0,
                len(scheme_codes),
                BATCH_SIZE
        ):

            batch_end = batch_start + BATCH_SIZE

            batch_scheme_codes = scheme_codes[
                batch_start:batch_end
            ]

            logger.info(
                f"Processing batch "
                f"{batch_start} to {batch_end}"
            )

            batch_nav_rows = []

            #################################################
            # FETCH NAV DATA
            #################################################

            for scheme_code in batch_scheme_codes:

                try:

                    endpoint = f'/mf/{scheme_code}'

                    response = http_hook.run(
                        endpoint
                    )

                    if response.status_code != 200:

                        logger.warning(
                            f"Skipping "
                            f"{scheme_code} "
                            f"due to API failure"
                        )

                        continue

                    nav_json = response.json()

                    meta = nav_json.get(
                        "meta",
                        {}
                    )

                    nav_data = nav_json.get(
                        "data",
                        []
                    )

                    fund_house = meta.get(
                        "fund_house"
                    )

                    scheme_name = meta.get(
                        "scheme_name"
                    )

                    #################################################
                    # FETCH EXISTING NAV DATES
                    #################################################

                    existing_query = """
                    SELECT nav_date
                    FROM mf_raw_nav
                    WHERE scheme_code = %s;
                    """

                    existing_records = (
                        pg_hook.get_records(
                            existing_query,
                            parameters=(
                                scheme_code,
                            )
                        )
                    )

                    existing_dates = {
                        str(row[0])
                        for row in existing_records
                    }

                    #################################################
                    # FILTER NEW NAV DATA
                    #################################################

                    new_count = 0

                    for nav_item in nav_data:

                        nav_date = nav_item.get(
                            "date"
                        )

                        nav_value = nav_item.get(
                            "nav"
                        )

                        formatted_date = (
                            convert_date_format(
                                nav_date
                            )
                        )

                        if (
                                formatted_date
                                in existing_dates
                        ):
                            continue

                        batch_nav_rows.append((
                            scheme_code,
                            nav_date,
                            nav_value,
                            fund_house,
                            scheme_name
                        ))

                        new_count += 1

                    logger.info(
                        f"{scheme_code} -> "
                        f"New NAV rows: "
                        f"{new_count}"
                    )

                except Exception as e:

                    logger.error(
                        f"Error processing "
                        f"{scheme_code}: "
                        f"{str(e)}"
                    )

            #################################################
            # INSERT NAV DATA IN BATCHES
            #################################################

            logger.info(
                f"Batch NAV rows ready: "
                f"{len(batch_nav_rows)}"
            )

            for insert_start in range(
                    0,
                    len(batch_nav_rows),
                    INSERT_BATCH_SIZE
            ):

                insert_end = (
                        insert_start
                        + INSERT_BATCH_SIZE
                )

                insert_batch = batch_nav_rows[
                    insert_start:insert_end
                ]

                insert_nav_data_batch(
                    insert_batch
                )

        logger.info(
            "NAV ingestion completed"
        )

    #########################################################
    # INITIALIZE TABLES
    #########################################################

    create_tables()

    #########################################################
    # DAG FLOW
    #########################################################

    schemes_task = fetch_all_mutual_fund()

    nav_task = fetch_nav_data()

    schemes_task >> nav_task