import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/process"

st.title("📊 Mutual Fund AI Advisor")
st.subheader("📌 Portfolio + Query Based Advisor")

# -----------------------------
# PREDEFINED QUESTIONS
# -----------------------------
PREDEFINED_QUESTIONS = [
    "Should I rebalance my portfolio?",
    "Am I overexposed to a single fund?",
    "Is my portfolio aligned with my risk profile?",
    "How can I improve returns?",
    "Should I switch to a safer allocation?",
    "Custom question"
]

# -----------------------------
# USER QUERY (DROPDOWN + CUSTOM)
# -----------------------------
selected_question = st.selectbox(
    "Select a question",
    PREDEFINED_QUESTIONS,
    index=0
)

if selected_question == "Custom question":
    question = st.text_input("Enter your question")
else:
    question = selected_question

st.markdown("---")

# -----------------------------
# PORTFOLIO INPUT
# -----------------------------
st.subheader("💼 Portfolio Details")

fund_a = st.number_input(
    "Fund A - Invested Amount",
    min_value=0,
    value=100000,
    key="fund_a"
)

fund_b = st.number_input(
    "Fund B - Invested Amount",
    min_value=0,
    value=100000,
    key="fund_b"
)

fund_c = st.number_input(
    "Fund C - Invested Amount",
    min_value=0,
    value=100000,
    key="fund_c"
)

risk_profile = st.selectbox(
    "Risk Profile",
    ["Low", "Moderate", "High"],
    index=1
)

st.markdown("---")

# -----------------------------
# BUILD PAYLOAD
# -----------------------------
portfolio = {
    "funds": [
        {"name": "Fund A", "amount": fund_a},
        {"name": "Fund B", "amount": fund_b},
        {"name": "Fund C", "amount": fund_c}
    ],
    "risk_profile": risk_profile
}

# -----------------------------
# SUBMIT BUTTON
# -----------------------------
if st.button("Analyze Portfolio"):

    if not question:
        st.error("Please enter a question before submitting.")
        st.stop()

    payload = {
        "portfolio": portfolio,
        "question": question
    }

    with st.spinner("Analyzing your portfolio..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            st.success("Analysis Complete")

            st.subheader("📊 Result")
            st.json(result)

        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")