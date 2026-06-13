import os
from abc import ABC

import psycopg2
from dotenv import load_dotenv
from groq import Groq
from pyspark.sql import SparkSession


class AbstractBaseService(ABC):

    _spark_session = None

    def __init__(self):
        load_dotenv()
        self._initialize_groq()
        self._initialize_database()

    def _initialize_groq(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in .env"
            )
        self.client = Groq(api_key=api_key)

    def _initialize_database(self):
        self.connection = psycopg2.connect(
            host=os.getenv("AURORA_HOST"),
            port=os.getenv("AURORA_PORT", "5432"),
            dbname=os.getenv("AURORA_DATABASE"),
            user=os.getenv("AURORA_USERNAME"),
            password=os.getenv("AURORA_PASSWORD")
        )

    @property
    def spark(self):
        if AbstractBaseService._spark_session is None:
            AbstractBaseService._spark_session = (
                SparkSession.builder
                .appName("MF Advisor")
                .config(
                    "spark.jars.packages",
                    "org.postgresql:postgresql:42.7.3"
                )
                .config(
                    "spark.executor.memory",
                    "4g"
                )
                .config(
                    "spark.driver.memory",
                    "4g"
                )
                .getOrCreate()
            )

        return AbstractBaseService._spark_session

    def execute_query(self,query,params=None):
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, params)
            if cursor.description:
                columns = [
                    desc[0]
                    for desc in cursor.description
                ]
                rows = cursor.fetchall()
                return [
                    dict(zip(columns, row))
                    for row in rows
                ]
            self.connection.commit()
            return []
        finally:
            cursor.close()


    def close(self):
        if self.connection:
            self.connection.close()


    def invoke_llm(self, system_prompt: str, user_prompt: str, model: str = "llama-3.3-70b-versatile",temperature: float = 0,json_response: bool = False):
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        }

        if json_response:
            payload["response_format"] = {
                "type": "json_object"
            }

        response = self.client.chat.completions.create(
            **payload
        )
        return response.choices[0].message.content