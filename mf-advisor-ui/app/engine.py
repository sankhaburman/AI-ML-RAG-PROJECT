def analyze_portfolio(portfolio: dict):
    funds = portfolio.get("funds", [])
    total = sum(f["amount"] for f in funds)

    return {
        "total_investment": total,
        "risk": "High" if total > 100000 else "Low",
        "suggestion": "Add index funds for stability"
    }


def answer_question(question: str):
    return {
        "answer": f"You asked: {question}",
        "confidence": 0.8
    }
