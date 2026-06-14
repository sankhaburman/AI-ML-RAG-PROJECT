MF Advisor — Ingestion & Enrichment ETL
======================================

Project summary
---------------
This Airflow (Astronomer) project ingests mutual fund NAV data, stores raw NAVs in Postgres, and runs Spark enrichment jobs to compute daily and aggregated returns for downstream ML and UI components.

Key features
------------
- Orchestrated DAGs: `mf_advisor_ingestor.py` handles table creation, API ingestion, deduplication, and batched inserts.
- Spark enrichment: `SparkSubmitOperator` runs scripts in `include/scripts/mf_enricher/` to compute returns and combine outputs.
- Utilities: resilient date parsing, batched DB writes, Postgres indexing for fast lookups.

Repository layout
-----------------
- `dags/` — Airflow DAGs (`mf_advisor_ingestor.py`, `etl_weather.py`, `spark_postgres_dag.py`, etc.)
- `include/scripts/mf_enricher/` — Spark enrichment Python scripts
- `requirements.txt`, `packages.txt`, `Dockerfile`, and `airflow_settings.yaml` for local Astro dev

Run locally
-----------
Start the local Astro/Airflow development environment:

```bash
astro dev start
```

Open the Airflow UI at http://localhost:8080. Postgres is available at `localhost:5432/postgres` (default credentials: `postgres`/`postgres`).

Notes & next steps
------------------
- Add CI tests and DAG integrity checks.
- Add monitoring and alerting for failed ingestion runs.
- Ensure idempotent validations for production deploys.

Contact
-------
For issues or questions, see the DAGs in `dags/` and the enrichment scripts in `include/scripts/mf_enricher/`.
