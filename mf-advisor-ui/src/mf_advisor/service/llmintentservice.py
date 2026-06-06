import logging

logging.basicConfig(level=logging.INFO)

class LLMIntentService:
    """
    Service responsible for detecting user intent from natural language query.
    """

    def __init__(self):
        # In real world, you can initialize LLM client here (OpenAI, Azure, etc.)
        pass

    def find_user_intent(self, question: str) -> str:
        logging.info(f"User question: {question}")
        return "DUMMY"