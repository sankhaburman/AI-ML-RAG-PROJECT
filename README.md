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