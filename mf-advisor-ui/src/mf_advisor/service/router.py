def route_request(portfolio, question):

    has_portfolio = portfolio is not None and len(portfolio.get("funds", [])) > 0
    has_question = question is not None and question.strip() != ""

    if has_portfolio and has_question:
        return "hybrid"
    if has_portfolio:
        return "portfolio"
    if has_question:
        return "qa"

    return "none"