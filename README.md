# USE CASE FLOW
Use Case Flow:
	User Access
The user opens the application (Stream lit UI) and lands on the dashboard, which provides options to input portfolio details or ask queries.
	Portfolio Input
    The user enters:
        o	  Mutual fund scheme codes / names
        o	Allocation percentages or can be calculated
        o	Risk appetite (low, moderate, high)
        o	This forms the primary input for analysis.
	Query (Optional)
    The user may:
        o	Ask a question (e.g., “Should I rebalance my portfolio?”)
        o	Or proceed without a query
        o	If no query is provided, the system assumes a default intent to analyze the portfolio.

	Data Retrieval
    The system fetches:
        o	Historical NAV data (e.g., last 2 years)
        o	Fund details from external APIs

	Data Processing & Analysis
    The system computes:
        o	Returns (CAGR)
        o	Volatility
        o	Risk metrics
        o	Diversification

	Prediction
    The LSTM model:
        o	Analyzes time-series data
        o	Predicts future performance trends
        o	Flags potential underperformers

	Portfolio Evaluation
    The system evaluates:
        o	Risk alignment with user profile
        o	Allocation balance
        o	Fund performance
        o	Generates insights like:
            	High equity exposure
            	Low diversification
            	Underperforming fund

	Decision Generation
        Based on analysis, the system provides a recommendation (hold or rebalance).

	Explanation (RAG)
    Relevant financial knowledge is retrieved to generate a clear and explainable justification.

	Result Display
    The UI displays:
        o	Recommendation (hold/rebalance)
        o	Key insights
        o	Suggested actions
        o	Explanation

	User Action
    The user reviews the output and decides whether to adjust the portfolio.

##FLOW SUMMARY:
User Query → NLP → Intent → Planning → Data → Analytics → ML → Evaluation → Decision → RAG → Response


# AI-ML-RAG-PROJECT
AI-ML Mutual Fund Advisor - AgenticAI , ML, RAG, Vector DB

AI-ML-RAG-PROJECT/
│
├── mf-advisor-api-gateway-repo/                  # Entry point (FastAPI)
│   ├── app/
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── advisor.py
│   │   │   ├── portfolio.py
│   │   │   └── health.py
│   │   ├── clients/
│   │   │   ├── ingestion_client.py
│   │   │   ├── analytics_client.py
│   │   │   ├── ml_client.py
│   │   │   ├── rag_client.py
│   │   │   └── portfolio_client.py
│   │   ├── schemas/
│   │   │   ├── request.py
│   │   │   └── response.py
│   │   └── config.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── mf-advisor-llm-orchestor-agent-repo/                # LLM Orchestrator (Agentic AI)
│   ├── app/
│   │   ├── main.py
│   │   ├── agent/
│   │   │   ├── agent.py         # Core reasoning logic
│   │   │   ├── tools.py         # Tool definitions (APIs)
│   │   │   └── prompts.py       # Prompt templates
│   │   ├── services/
│   │   │   ├── ingestion_client.py
│   │   │   ├── analytics_client.py
│   │   │   ├── ml_client.py
│   │   │   └── rag_client.py
│   │   └── config.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── mf-advisor-ingestion-etl-repo/            # MFAPI Data Fetch
│   ├── app/
│   │   ├── main.py
│   │   ├── services/
│   │   │   ├── mfapi_client.py
│   │   │   └── fetch_nav.py
│   │   ├── utils/
│   │   │   └── async_fetch.py
│   │   └── config.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── mf-advisor-analytics-repo/            # Financial Metrics Engine
│   ├── app/
│   │   ├── main.py
│   │   ├── services/
│   │   │   ├── returns.py
│   │   │   ├── volatility.py
│   │   │   ├── sharpe.py
│   │   │   ├── drawdown.py
│   │   │   └── portfolio_metrics.py
│   │   └── config.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── mf-advisor-ml-service-repo/                   # LSTM Inference Service
│   ├── app/
│   │   ├── main.py
│   │   ├── services/
│   │   │   ├── model_loader.py
│   │   │   ├── predictor.py
│   │   │   └── preprocessing.py
│   │   ├── models/
│   │   │   └── lstm_model.pkl
│   │   └── config.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── training-pipeline/            # Offline Model Training
│   ├── dags/                    # Airflow DAGs
│   │   └── train_lstm.py
│   ├── src/
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── save_model.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── mf-advisor-rag-service-repo/                  # RAG + Vector DB
│   ├── app/
│   │   ├── main.py
│   │   ├── services/
│   │   │   ├── retriever.py
│   │   │   ├── embedder.py
│   │   │   ├── vector_store.py
│   │   │   └── ingestion.py     # PDF/Doc ingestion
│   │   ├── data/
│   │   │   └── raw_docs/
│   │   └── config.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── mf-advisor-portfolio-service-repo/            # Decision Engine
│   ├── app/
│   │   ├── main.py
│   │   ├── services/
│   │   │   ├── evaluator.py     # Flags (risk/diversification)
│   │   │   ├── decision.py      # HOLD / REBALANCE
│   │   │   └── optimizer.py     # Suggested allocation
│   │   └── config.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── mf-advisor-notification-service-repo/         # Alerts (Optional)
│   ├── app/
│   │   ├── main.py
│   │   ├── services/
│   │   │   ├── email.py
│   │   │   └── alerts.py
│   │   └── config.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── mf-advisor-ui-repo/                  # Streamlit UI
│   ├── app.py
│   ├── pages/
│   │   ├── portfolio.py
│   │   ├── advisor.py
│   │   └── analytics.py
│   ├── components/
│   │   ├── charts.py
│   │   └── inputs.py
│   ├── assets/
│   └── config.py
│   ├── requirements.txt
│
├── shared-lib/                   # Common utilities
│   ├── schemas/
│   │   ├── request.py
│   │   ├── response.py
│   │   └── portfolio.py
│   ├── utils/
│   │   ├── logger.py
│   │   └── constants.py
│   └── config.py
│
├── data/                         # Optional centralized storage
│   ├── raw/
│   ├── processed/
│   └── models/
│
├── infra/                        # Deployment & Orchestration
│   ├── docker-compose.yml
│   ├── k8s/
│   │   ├── api-gateway.yaml
│   │   ├── ingestion.yaml
│   │   ├── analytics.yaml
│   │   ├── ml.yaml
│   │   ├── rag.yaml
│   │   └── advisor.yaml
│   ├── airflow/
│   │   └── dags/
│   └── env/
│       └── .env
│
└── README.md