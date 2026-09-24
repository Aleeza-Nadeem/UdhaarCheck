# UdhaarCheck — Credit Default Risk Detector

A machine learning tool that estimates the probability a customer will default on informal credit ("udhaar"), built for the context of small retail (kiryana) shopkeepers who extend credit based on relationship and trust rather than formal underwriting.

## The problem

Kiryana store owners routinely extend informal credit to regular customers, tracked in paper ledgers, with no structured way to assess who's a growing risk before it becomes a loss. This project explores whether payment-behavior patterns — the kind of signal a shopkeeper half-remembers but can't systematically track — can be captured in a model and surfaced as a decision aid.

## Dataset

UCI "Default of Credit Card Clients" dataset (30,000 rows, ~22% default rate) — used as a proxy for informal merchant credit, since no public dataset exists for udhaar/kiryana lending specifically. Each row has 6 months of billing amount, payment amount, and repayment status per customer, plus demographics (age, education, marital status, credit limit).

**Known proxy limitation:** this is formal credit-card behavior, not informal shopkeeper credit — the patterns likely transfer partially, not perfectly. Documented here rather than glossed over.

## Engineered features

Raw monthly columns capture a single snapshot; these were added to capture *behavior over time*, which turned out to matter more than any single month's numbers:

| Feature | Formula | What it captures |
|---|---|---|
| `debt_trend` | `BILL_AMT6 - BILL_AMT1` | Whether total debt is growing or shrinking over the 6-month window |
| `avg_payment_ratio` | mean(`PAY_AMT / BILL_AMT`) across 6 months | How much of what's owed is actually being paid back, on average |
| `utilization_rate` | `BILL_AMT1 / LIMIT_BAL` | How close the customer is to their credit ceiling |
| `net_unpaid_balance` | `BILL_AMT1 - PAY_AMT1` | Raw debt rolling over after the most recent payment |

**Tested and dropped:** an `is_single` flag based on marital status. A chi-square test confirmed the association between marital status and default was statistically significant (p ≈ 0.0000008) — but the effect size was small (~3 percentage points) and redundant with the behavior-based features above, so it was left out in favor of a leaner feature set.

## Modeling journey

This project deliberately documents the full diagnostic process, not just the final number — the path here is as much the point as the result.

| Attempt | Result (class 1 / default) | Outcome |
|---|---|---|
| Random Forest, threshold 0.5, `class_weight='balanced'` | Precision 0.53, Recall 0.53, F1 0.53 | Baseline — imbalance still visible despite balancing |
| Threshold lowered to 0.35 | Precision 0.38, Recall 0.72, F1 0.50 | Recall improved, precision dropped — expected tradeoff |
| XGBoost | Worse than RF baseline | Discarded — likely missing `scale_pos_weight` for imbalance |
| SMOTE oversampling | Precision 0.49, Recall 0.55, F1 0.52 | No real improvement over threshold tuning alone |
| **New features + Random Forest, threshold 0.3** | **Precision 0.51, Recall 0.54, F1 0.53** | Final model — see below |

**Why SMOTE and threshold-tuning both plateaued around the same point:** neither changes what information the model has access to — SMOTE resamples within the existing feature space, and threshold tuning just moves the cut point on the same predicted probabilities. Real improvement required new *information* (the engineered behavior features), not more resampling of the old information.

## Final model performance

Random Forest Classifier, `class_weight='balanced'`, decision threshold = **0.3** (deliberately recall-favoring — a missed defaulter costs more than a false alarm on a reliable customer).

```
              precision    recall  f1-score   support

           0       0.87      0.85      0.86      4673
           1       0.51      0.54      0.53      1327

    accuracy                           0.78      6000
```

- **ROC-AUC: 0.761** — in line with published benchmarks on this dataset
- **PR-AUC: 0.540** — over 2x the random-guess baseline (~0.22, the positive class rate)

**Why the ceiling exists, not just what it is:** six months of payment history can't capture external shocks — job loss, medical costs, other debts — that also drive real-world defaults. This ceiling was confirmed, not assumed, by testing multiple independent approaches (resampling, algorithm swap, feature engineering) and finding they all converged around the same performance band.

## SHAP feature importance

![SHAP summary plot](shap_summary.png)

Top drivers of the model's predictions: `PAY_0` and `PAY_2` (most recent repayment status) dominate, followed by `utilization_rate`, `LIMIT_BAL`, and `PAY_4`/`PAY_3`. The engineered features (`debt_trend`, `avg_payment_ratio`, `net_unpaid_balance`) contribute but rank below the raw repayment-status columns — recent delinquency history is the strongest single signal, with the engineered trend features adding secondary, complementary signal.

## Limitations

- Proxy dataset (formal credit card data), not real udhaar/kiryana transaction history
- 6-month window can't capture external life events driving default
- No feedback loop yet — real-world deployment would need to log outcomes and retrain over time
- Model is a decision aid, not an autonomous approval/denial system — designed to sit alongside a shopkeeper's own judgment, not replace it

## App

`app.py` is a Streamlit interface: enter a customer's credit limit, demographics, and 6 months of repayment status / bill / payment amounts, and get back a default probability, a Low/Medium/High risk tier, and a plain-language explanation of what the score means and doesn't mean.

### Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires `kiryana_default_model.pkl` (the trained model) in the same directory as `app.py`.

## Future work

- **Exposure-amount regression** — a second model predicting *how much* is likely to go unpaid, not just *who's* likely to default, reusing the same engineered features
- **Feedback loop** — log predictions against real outcomes to retrain and improve beyond what any static dataset can offer
- **Human-in-the-loop scoring** — combine the model's score with relationship signals a shopkeeper already has (how long they've known the customer, informal reputation) that aren't captured in transaction data at all

## Tech stack

Python, pandas, scikit-learn, SHAP, Streamlit, joblib
