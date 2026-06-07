from fastapi import FastAPI
from pydantic import BaseModel
import logging

from mf_advisor.service.engine import analyze_portfolio, answer_question
from mf_advisor.service.llm_intent_service import LLMIntentService
from mf_advisor.service.router import route_request

logging.basicConfig(level=logging.INFO)

app = FastAPI()
intent_service = LLMIntentService()

class InputRequest(BaseModel):
    portfolio: dict | None = None
    question: str | None = None



@app.post("/process")
def process(req: InputRequest):
    logging.info(f"REQUEST: {req.dict()}")
    portfolio = req.portfolio
    question = req.question

    # SAFE GUARD (prevents crashes)
    if portfolio is None and question is None:
        return {"error": "Both portfolio and question are empty"}

    try:
        route = route_request(portfolio, question)
    except Exception as e:
        return {"error": f"route_request failed: {str(e)}"}

    user_intent = intent_service.find_user_intent(question)
    try:
        if route == "portfolio":
            if not portfolio:
                return {"error": "Portfolio missing for portfolio route"}
            return analyze_portfolio(portfolio)

        if route == "qa":
            if not question:
                return {"error": "Question missing for QA route"}
            return answer_question(question)

        if route == "hybrid":
            return {
                "portfolio": analyze_portfolio(portfolio) if portfolio else None,
                "qa": answer_question(question) if question else None
            }

        return {"message": "No valid route detected"}

    except Exception as e:
        return {"error": f"processing failed: {str(e)}"}
