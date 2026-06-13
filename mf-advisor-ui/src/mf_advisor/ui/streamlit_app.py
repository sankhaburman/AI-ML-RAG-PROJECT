import streamlit as st
import requests

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/process"

st.set_page_config(
    page_title="Mutual Fund AI Advisor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Mutual Fund AI Advisor")
st.subheader("📌 Portfolio + Query Based Advisor")

# --------------------------------------------------
# PREDEFINED QUESTIONS
# --------------------------------------------------
PREDEFINED_QUESTIONS = [
    "Which holdings should be reduced or increased?"
]

# --------------------------------------------------
# USER QUERY
# --------------------------------------------------
selected_question = st.selectbox(
    "Select a question",
    PREDEFINED_QUESTIONS,
    index=0
)

# Assign selected value to question
question = selected_question

st.markdown("---")

# --------------------------------------------------
# PORTFOLIO INPUT
# --------------------------------------------------
st.subheader("💼 Portfolio Details")

fund_a = st.number_input(
    "Nippon India Multi Cap Fund - Direct Plan Growth Plan - Growth Option - Invested Amount",
    min_value=0,
    value=100000,
    step=5000,
    key="fund_a"
)

fund_b = st.number_input(
    "Parag Parikh Flexi Cap Fund - Direct Plan - IDCW - Invested Amount",
    min_value=0,
    value=500000,
    step=5000,
    key="fund_b"
)

fund_c = st.number_input(
    "Quantum Multi Asset Active FOF - Direct Plan Growth Option - Invested Amount",
    min_value=0,
    value=1000000,
    step=5000,
    key="fund_c"
)

fund_d = st.number_input(
    "Invesco India Large & Mid Cap Fund - Direct Plan - Growth - Invested Amount",
    min_value=0,
    value=700000,
    step=5000,
    key="fund_d"
)

fund_e = st.number_input(
    "Bandhan Short Duration Fund - Direct Plan - Growth - Invested Amount",
    min_value=0,
    value=200000,
    step=5000,
    key="fund_e"
)

risk_profile = st.selectbox(
    "Risk Profile",
    ["Low", "Moderate", "High"],
    index=1
)

st.markdown("---")

# --------------------------------------------------
# BUILD PORTFOLIO OBJECT
# --------------------------------------------------
portfolio = {
    "funds": [
        {
            "name": "Nippon India Multi Cap Fund - Direct Plan Growth Plan - Growth Option",
            "amount": float(fund_a)
        },
        {
            "name": "Parag Parikh Flexi Cap Fund - Direct Plan - IDCW",
            "amount": float(fund_b)
        },
        {
            "name": "Quantum Multi Asset Active FOF - Direct Plan Growth Option",
            "amount": float(fund_c)
        },
        {
            "name": "Invesco India Large & Mid Cap Fund - Direct Plan - Growth",
            "amount": float(fund_d)
        },
        {
            "name": "Bandhan Short Duration Fund - Direct Plan - Growth",
            "amount": float(fund_e)
        }
    ],
    "risk_profile": risk_profile
}

# --------------------------------------------------
# DISPLAY PORTFOLIO SUMMARY
# --------------------------------------------------
total_investment = fund_a + fund_b + fund_c + fund_d + fund_e

st.metric(
    label="Total Portfolio Value",
    value=f"₹{total_investment:,.0f}"
)

# --------------------------------------------------
# SUBMIT BUTTON
# --------------------------------------------------
if st.button("🚀 Analyze Portfolio", type="primary"):

    if not question:
        st.error("Please select a question before submitting.")
        st.stop()

    payload = {
        "portfolio": portfolio,
        "question": question
    }

    with st.spinner("Analyzing your portfolio..."):
        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            st.success("✅ Analysis Complete")
            st.subheader("📊 Analysis Result")
            if isinstance(result, dict):
                st.json(result)
            else:
                st.write(result)
        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Could not connect to the API. "
                "Make sure your FastAPI server is running on port 8000."
            )

        except requests.exceptions.Timeout:
            st.error(
                "❌ Request timed out. "
                "The backend is taking too long to respond."
            )

        except requests.exceptions.HTTPError as e:
            st.error(f"❌ HTTP Error: {e}")

            try:
                st.json(response.json())
            except Exception:
                st.text(response.text)

        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")