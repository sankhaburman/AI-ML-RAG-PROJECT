import logging
import psycopg2

from mf_advisor.service.abstract_base_service import (
    AbstractBaseService
)

logging.basicConfig(level=logging.INFO)

POSTGRES_DRIVER = "org.postgresql.Driver"

class PortfolioRebalanceService(AbstractBaseService):

    def __init__(self):
        super().__init__()

    def rebalance_portfolio(self, portfolio_req):
        try:
            logging.info(f"Portfolio request: {portfolio_req}")
            fund_names = self.extract_fund_names(portfolio_req)
            logging.info(f"Fund Names-------: {fund_names}")
            scheme_codes = self.fetch_scheme_codes(fund_names)
            if not scheme_codes:
                return {
                    "risk_profile": portfolio_req.get(
                        "risk_profile"
                    ),
                    "fund_metrics": []
                }
            latest_metrics_df = (self.load_latest_fund_metrics(scheme_codes))
            return {
                "risk_profile": portfolio_req.get(
                    "risk_profile"
                ),
                "scheme_codes": scheme_codes,
                "latest_metrics": [
                    row.asDict()
                    for row in latest_metrics_df.collect()
                ]
            }

        except Exception:
            logging.exception("Error while processing portfolio request")
            raise

    def extract_fund_names(self,portfolio_req):
        return [
            fund.get("name")
            for fund in portfolio_req.get(
                "funds",
                []
            )
            if fund.get("name")
        ]

    def fetch_scheme_codes(self,fund_names):
        if not fund_names:
            return []
        placeholders = ",".join(["%s"] * len(fund_names))
        query = f"""
            SELECT
                scheme_code
            FROM mf_schemes
            WHERE scheme_name IN ({placeholders})
        """
        conn = psycopg2.connect(
            host=self.connection["host"],
            port=self.connection["port"],
            database=self.connection["database"],
            user=self.connection["user"],
            password=self.connection["password"]
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(query,tuple(fund_names))
                rows = cursor.fetchall()
                return [
                    int(row[0])
                    for row in rows
                ]
        finally:
            conn.close()

    def load_latest_fund_metrics(self,scheme_codes):
        scheme_list = ",".join(map(str, scheme_codes))
        query = f"""
        (
            SELECT *
            FROM
            (
                SELECT
                    scheme_code,
                    nav_date,
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
                    annualized_volatility,
                    ROW_NUMBER() OVER
                    (
                        PARTITION BY scheme_code
                        ORDER BY nav_date DESC
                    ) rn
                FROM mf_final_nav_enriched
                WHERE scheme_code IN ({scheme_list})
            ) t
            WHERE rn = 1
        ) latest_data
        """
        jdbc_url = (
            f"jdbc:postgresql://"
            f"{self.connection['host']}:"
            f"{self.connection['port']}/"
            f"{self.connection['database']}"
        )
        return (
            self.spark.read
            .format("jdbc")
            .option("url", jdbc_url)
            .option("dbtable", query)
            .option("user",self.connection["user"])
            .option("password",self.connection["password"])
            .option("driver",POSTGRES_DRIVER)
            .load()
        )