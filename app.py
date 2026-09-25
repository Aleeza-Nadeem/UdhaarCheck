import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
MODEL_PATH = "kiryana_default_model.pkl"
FINAL_THRESHOLD = 0.35  # used only to set the visual "risk tier" cue below —
                         # the actual decision is left to the user via the
                         # displayed probability, not a hard flag (see Result section)

st.set_page_config(page_title="Kiryana Credit Risk Checker", page_icon="🧾", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🧾 Kiryana Credit Risk Checker")
st.caption(
    "Estimates the probability a customer will default on informal credit, "
    "based on 6 months of billing and payment history. Random Forest, "
    "ROC-AUC 0.76 / PR-AUC 0.54 on held-out test data."
)

st.divider()

# ---------------------------------------------------------
# Inputs
# ---------------------------------------------------------
with st.form("customer_form"):
    st.subheader("Customer basics")
    c1, c2, c3 = st.columns(3)
    with c1:
        limit_bal = st.number_input("Credit limit", min_value=1000, value=50000, step=1000)
    with c2:
        age = st.number_input("Age", min_value=18, max_value=90, value=30)
    with c3:
        sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")

    c4, c5 = st.columns(2)
    with c4:
        education = st.selectbox(
            "Education", options=[1, 2, 3, 4],
            format_func=lambda x: {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}[x]
        )
    with c5:
        marriage = st.selectbox(
            "Marital status", options=[1, 2, 3],
            format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x]
        )

    st.subheader("Repayment status (last 6 months)")
    st.caption("-1 = paid on time, 0 = revolving credit, 1-9 = months delayed")
    pay_cols = st.columns(6)
    pay_status = []
    labels = ["Most recent", "Month -2", "Month -3", "Month -4", "Month -5", "Month -6"]
    for i, col in enumerate(pay_cols):
        with col:
            pay_status.append(st.number_input(labels[i], min_value=-2, max_value=9, value=0, key=f"pay_{i}"))

    st.subheader("Bill amount per month")
    bill_cols = st.columns(6)
    bill_amts = []
    for i, col in enumerate(bill_cols):
        with col:
            bill_amts.append(st.number_input(labels[i], min_value=0, value=5000, step=500, key=f"bill_{i}"))

    st.subheader("Amount actually paid per month")
    pay_amt_cols = st.columns(6)
    pay_amts = []
    for i, col in enumerate(pay_amt_cols):
        with col:
            pay_amts.append(st.number_input(labels[i], min_value=0, value=1000, step=500, key=f"payamt_{i}"))

    submitted = st.form_submit_button("Check risk", use_container_width=True)

# ---------------------------------------------------------
# Feature engineering (must match training exactly)
# ---------------------------------------------------------
def build_features(limit_bal, sex, education, marriage, age, pay_status, bill_amts, pay_amts):
    row = {
        "ID": 0,  # the training data included ID as a column (unintentionally) —
                  # kept here only so the model's expected columns line up;
                  # this value has no real meaning and doesn't affect the prediction logic
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        "PAY_0": pay_status[0], "PAY_2": pay_status[1], "PAY_3": pay_status[2],
        "PAY_4": pay_status[3], "PAY_5": pay_status[4], "PAY_6": pay_status[5],
        "BILL_AMT1": bill_amts[0], "BILL_AMT2": bill_amts[1], "BILL_AMT3": bill_amts[2],
        "BILL_AMT4": bill_amts[3], "BILL_AMT5": bill_amts[4], "BILL_AMT6": bill_amts[5],
        "PAY_AMT1": pay_amts[0], "PAY_AMT2": pay_amts[1], "PAY_AMT3": pay_amts[2],
        "PAY_AMT4": pay_amts[3], "PAY_AMT5": pay_amts[4], "PAY_AMT6": pay_amts[5],
    }
    df = pd.DataFrame([row])

    # debt_trend: change in balance from month 1 to month 6
    df["debt_trend"] = df["BILL_AMT6"] - df["BILL_AMT1"]

    # avg_payment_ratio: mean of PAY_AMT/BILL_AMT across the 6 months, 0 when bill was 0
    bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    pcols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
    ratios = []
    for b, p in zip(bill_cols, pcols):
        ratios.append(df[p] / df[b].replace(0, pd.NA))
    ratio_df = pd.concat(ratios, axis=1).fillna(0)
    df["avg_payment_ratio"] = ratio_df.mean(axis=1)

    # utilization_rate
    df["utilization_rate"] = (df["BILL_AMT1"] / df["LIMIT_BAL"].replace(0, pd.NA)).fillna(0)

    # net_unpaid_balance
    df["net_unpaid_balance"] = df["BILL_AMT1"] - df["PAY_AMT1"]

    return df

# ---------------------------------------------------------
# Prediction + display
# ---------------------------------------------------------
if submitted:
    X = build_features(limit_bal, sex, education, marriage, age, pay_status, bill_amts, pay_amts)

    # align column order to what the model expects, if available
    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]

    proba = model.predict_proba(X)[0, 1]

    st.divider()
    st.subheader("Result")

    # Probability is the primary output — the model reports its confidence,
    # the user makes the actual call. No binary "flagged / not flagged" verdict:
    # at ~0.44 precision, presenting a hard flag would overstate certainty and
    # risks eroding trust in the tool the first time a flag turns out wrong.
    st.metric("Default probability", f"{proba:.1%}")
    st.progress(min(proba, 1.0))

    if proba < FINAL_THRESHOLD:
        tier, msg = "Low", "Model sees limited risk signals for this customer."
    elif proba < 0.65:
        tier, msg = "Medium", "Some risk signals present — weigh this alongside what you know about this customer."
    else:
        tier, msg = "High", "Strong risk signals — worth extra caution, but this isn't a final verdict."

    st.write(f"**Risk tier: {tier}**")
    st.caption(msg)

    st.info(
        "This is a probability estimate, not a decision. It reflects payment "
        "patterns only — it doesn't know your history with this customer. "
        "Use it as one input alongside your own judgment."
    )

    with st.expander("Why this matters / model limitations"):
        st.write(
            "- This model reaches ROC-AUC 0.78 and PR-AUC 0.56 on held-out data — "
            "meaningfully better than a guess, but not a guarantee for any individual customer.\n"
            "- Six months of payment history can't capture external shocks (job loss, medical "
            "costs, etc.) that also drive defaults — this tool works best combined with what "
            "you already know about the customer, not as a replacement for it.\n"
            "- The risk tier above is a visual aid only. The probability percentage is the "
            "real signal — use your own judgment on where your comfort line sits."
        )

st.divider()
st.caption("Built as a portfolio project exploring ML for informal merchant credit risk.")
