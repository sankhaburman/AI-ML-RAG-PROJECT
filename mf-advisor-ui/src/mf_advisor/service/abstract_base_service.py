import logging
from abc import ABC
import os

from dotenv import load_dotenv
from groq import Groq
from pyspark.sql import SparkSession

load_dotenv()
APP_NAME = "MF-ADVISOR"
logging.basicConfig(level=logging.INFO)

class AbstractBaseService(ABC):

    _spark_session = None

    def __init__(self):
        self._initialize_database()
        self._initialize_groq()

    def _initialize_database(self):
        self._connection = {
            "host": os.getenv("POSTGRES_HOST"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB"),
            "user": os.getenv("POSTGRES_USER"),
            "password": os.getenv("POSTGRES_PASSWORD")
        }

    def _initialize_groq(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not configured"
            )
        self._groq_client = Groq(
            api_key=api_key
        )

    @property
    def connection(self):
        return self._connection

    @property
    def groq_client(self):
        return self._groq_client

    def invoke_llm(self, system_prompt,user_prompt,temperature=0,json_response=False):
        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            response_format=(
                {"type": "json_object"}
                if json_response
                else None
            ),
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
        return response.choices[0].message.content

    @classmethod
    def get_spark_session(cls):
        os.environ["HADOOP_HOME"] = r"D:\software\hadoop"
        os.environ["hadoop.home.dir"] = r"D:\software\hadoop"
        if cls._spark_session is None:
            cls._spark_session = (
                SparkSession.builder
                .appName(APP_NAME)
                .config(
                    "spark.jars.packages",
                    "org.postgresql:postgresql:42.7.3"
                )
                .getOrCreate()
            )
        return cls._spark_session

    @property
    def spark(self):
        return self.get_spark_session()