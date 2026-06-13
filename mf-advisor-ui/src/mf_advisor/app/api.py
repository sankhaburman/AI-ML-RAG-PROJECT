from fastapi import FastAPI
from pydantic import BaseModel
import logging

from mf_advisor.service.engine import analyze_portfolio, answer_question
from mf_advisor.service.llm_intent_service import LLMIntentService
from mf_advisor.service.portfolio_rebalance_service import PortfolioRebalanceService

logging.basicConfig(level=logging.INFO)

app = FastAPI()
intent_service = LLMIntentService()
rebalance_service = PortfolioRebalanceService()
class InputRequest(BaseModel):
    portfolio: dict | None = None
    question: str | None = None



@app.post("/process")
def process(req: InputRequest):
    portfolio = req.portfolio
    question = req.question

    logging.info(f"Portfolio request: {req.portfolio}")
    logging.info(f"Question asked: {req.question}")
    # SAFE GUARD (prevents crashes)
    if portfolio is None and question is None:
        return {"error": "Both portfolio and question are empty"}



    try:
        user_intent = intent_service.find_user_intent(question)
        if user_intent == 'REBALANCE_PORTFOLIO':
           return rebalance_service.rebalance_portfolio(req.portfolio)



    except Exception as e:
        return {"error": f"processing failed: {str(e)}"}
