import json
import logging

from mf_advisor.service.abstract_base_service import (
    AbstractBaseService
)

logging.basicConfig(level=logging.INFO)

class LLMIntentService(AbstractBaseService):

    def __init__(self):

        super().__init__()

        self.intent_prompt = """
You are an intent classification engine for a Mutual Fund Advisor application.

Classify the user query into ONE of the following intents:

- REBALANCE_PORTFOLIO
- PORTFOLIO_ANALYSIS
- RISK_ASSESSMENT
- PERFORMANCE_ANALYSIS
- FUND_RECOMMENDATION
- SIP_ADVICE
- TAX_PLANNING
- GENERAL_FINANCE_QUERY
- GREETING
- UNKNOWN

Rules:
1. Return ONLY valid JSON.
2. Do not provide explanations.

Output format:

{
    "intent": "<INTENT_NAME>",
    "confidence": 0.95
}
"""

    def find_user_intent(self,question: str) -> str:
        try:
            logging.info(f"User question: {question}")
            content = self.invoke_llm(system_prompt=self.intent_prompt,
                user_prompt=question,
                temperature=0,
                json_response=True
            )
            logging.info(f"LLM response: {content}")
            result = json.loads(content)
            intent = result.get("intent","UNKNOWN")

            confidence = result.get(
                "confidence",
                0
            )

            logging.info(
                f"Intent={intent}, Confidence={confidence}"
            )

            return intent

        except Exception as ex:

            logging.exception(
                f"Intent detection failed: {str(ex)}"
            )

            return "UNKNOWN"