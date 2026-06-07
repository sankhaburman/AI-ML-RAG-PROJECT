import json
import logging
import os

from dotenv import load_dotenv
from groq import Groq

logging.basicConfig(level=logging.INFO)


class LLMIntentService:

    def __init__(self):
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Check your .env file."
            )

        self.client = Groq(api_key=api_key)

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

    def find_user_intent(self, question: str) -> str:
        try:
            logging.info(f"User question: {question}")
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": self.intent_prompt
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            )
            content = response.choices[0].message.content
            logging.info(f"LLM response: {content}")
            result = json.loads(content)
            intent = result.get("intent", "UNKNOWN")
            logging.info(f"Detected intent: {intent}")
            return intent

        except Exception:
            logging.exception("Intent detection failed")
            return "UNKNOWN"