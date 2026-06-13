import json
import logging

from mf_advisor.service.abstract_base_service import AbstractBaseService

logging.basicConfig(level=logging.INFO)
POSTGRES_DRIVER = "org.postgresql.Driver"

class PortfolioRebalanceService(AbstractBaseService):

    def __init__(self):
        super().__init__()

    def rebalance_portfolio(self, portfolio_req):
        try:
            logging.info(f"Portfolio request in PortfolioRebalanceService: {portfolio_req}")
            fund_names = self.extract_fund_names(portfolio_req)
            scheme_code_csv = self.fetch_scheme_code_csv(fund_names)
            #latest_nav_for_funds = self.fetch_recent_records_for_fund(scheme_code_csv)
            return {
                "risk_profile": portfolio_req.get("risk_profile"),
                #"latest_nav": latest_nav_for_funds
            }
        except Exception:
            logging.exception(
                "Error while processing portfolio request"
            )
            raise

    def extract_fund_names(self, portfolio_req):
        funds = portfolio_req.get("funds", [])
        return [
            fund.get("name")
            for fund in funds
            if fund.get("name")
        ]

    def fetch_scheme_code_csv(self, fund_names):
        placeholders = ",".join(["%s"] * len(fund_names))
        query = f"""
            SELECT
                scheme_code
            FROM mf_schemes
            WHERE scheme_name IN ({placeholders})
        """

        results = self.execute_query(query,tuple(fund_names))
        scheme_codes = [
            str(row["scheme_code"])
            for row in results
        ]
        return ",".join(scheme_codes)


    def load_latest_fund_metrics(spark, connection,scheme_codes):

        scheme_list = ",".join(
            map(str, scheme_codes)
        )

        query = f"""
        (
            SELECT *
            FROM (
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
                    ROW_NUMBER() OVER (
                        PARTITION BY scheme_code
                        ORDER BY nav_date DESC
                    ) rn
                FROM mf_final_nav_enriched
                WHERE scheme_code IN ({scheme_list})
            ) t
            WHERE rn = 1
        ) latest_data
        """

        return (
            spark.read.format("jdbc")
            .option("url", connection["jdbc_url"])
            .option("dbtable", query)
            .option("user", connection["user"])
            .option("password", connection["password"])
            .option("driver", POSTGRES_DRIVER)
            .load()
        )



