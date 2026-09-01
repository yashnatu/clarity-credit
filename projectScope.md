# 🚗 Auto Loan Default Predictor

> I'm buying a car. I wanted to understand how lenders actually decide who gets approved — and at what rate. So I built the model myself.

This project predicts the likelihood that a borrower will default on an auto loan, using real lending data, interpretable ML, and SHAP-based explanations that surface *why* any individual applicant is flagged as high-risk. It mirrors the kind of credit risk decisioning used by lenders like Capital One's auto finance division.

---

## Motivation

When you apply for an auto loan, a black box decides your fate — approved, denied, or approved at a punishing interest rate. The factors that go into that decision (your debt-to-income ratio, credit utilization, loan-to-income ratio) are rarely explained.

I wanted to open that black box. As someone currently in the car buying process, I built this to understand what lenders actually look for — and to build the kind of explainability layer that fair lending regulations increasingly require but consumer-facing products almost never surface.

---

## What This Project Does

- **Predicts default probability** for a loan applicant given their financial profile
- **Explains individual predictions** — not just "denied," but "denied primarily because your DTI is in the 87th percentile and your credit utilization exceeds 72%"
- **Compares modeling approaches** with honest metric selection (PR-AUC, F1 on minority class — not accuracy, which is meaningless on imbalanced credit data)
- **Deploys as an interactive app** where you can input a real loan profile and see your risk score + explanation in real time

---

## Dataset

**Primary:** [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) — ~2M loan records with borrower financials, loan terms, and repayment outcomes.

> While Lending Club issues personal loans rather than auto loans specifically, the credit risk structure is identical: same underwriting signals (DTI, credit history, payment-to-income), same class imbalance (~10–15% default rate), same regulatory context. The pipeline maps directly to auto loan decisioning.

---

## Feature Engineering

Raw financial data doesn't go straight into a model. The interesting work is in how you construct features:

| Feature | Construction | Why It Matters |
|---|---|---|
| `debt_to_income_ratio` | Total debt / gross income | Core underwriting signal |
| `payment_to_income` | Monthly installment / monthly income | Affordability check |
| `loan_to_income` | Loan amount / annual income | Critical for auto specifically |
| `credit_utilization` | Revolving balance / credit limit | Leading indicator of distress |
| `credit_age_months` | Derived from earliest credit line date | Stability signal |
| `delinquency_flag` | Binary from delinquency history | Hard negative signal |

Employment length is ordinally encoded (it has natural order). Loan purpose is target-encoded. State is converted to a default-rate risk score — one-hot encoding 50 states is noise, not signal.

---

## Handling Class Imbalance

Default rates sit around 10–15%. A naive model achieves ~90% accuracy by predicting "no default" every time — which is useless. This project addresses imbalance directly:

- **Primary approach:** `class_weight='balanced'` in tree models, `scale_pos_weight` in XGBoost
- **Augmentation:** SMOTE on training data to synthesize minority class examples
- **Threshold tuning:** Shift the decision boundary away from 0.5 based on the cost asymmetry — missing a default (false negative) is more costly than a false positive in a lending context
- **Evaluation:** PR-AUC and F1 on the minority class. Accuracy is not reported as a primary metric.

---

## Models

Three models trained and compared:

```
Logistic Regression     → Baseline, interpretable, fast
Random Forest           → Captures nonlinear interactions
XGBoost                 → Best performer; handles imbalance natively
```

The initial baseline uses imbalance-aware model weights and validation-period
threshold tuning. XGBoost currently wins on validation PR-AUC; Optuna
hyperparameter tuning remains a planned enhancement.

**Measured baseline results (first 200,000 source rows; December 2015 test period):**

| Model | PR-AUC | F1 (Default Class) |
|---|---|---|
| Logistic Regression | 0.4216 | 0.4485 |
| Random Forest | 0.4215 | 0.4470 |
| XGBoost | 0.4307 | 0.4520 |

---

## Explainability — SHAP

Financial ML models in the U.S. operate under ECOA and fair lending regulations that require adverse action notices — you can't just say "denied," you have to say why. SHAP (SHapley Additive exPlanations) makes this possible.

**Three levels of explanation:**

**1. Global — What does the model care about overall?**

SHAP summary plot ranked by mean absolute value across all predictions. DTI, credit utilization, and payment-to-income consistently rank highest.

**2. Local — Why was this specific applicant flagged?**

Force plot for any individual prediction. Shows exactly which features pushed the risk score up or down from the baseline.

```
Base rate: 12.4% → Your score: 31.7%
↑ DTI of 0.48 (+9.2%)
↑ Credit utilization 71% (+6.8%)
↓ Credit age 94 months (-3.1%)
```

**3. Interaction — How do features interact?**

Dependence plot for DTI × interest rate. High DTI borrowers at high interest rates default at dramatically higher rates — the interaction matters, not just the individual signals.

---

## Interactive App

Built with Streamlit. Input a loan profile, get a default probability and a SHAP force plot explaining the prediction.

Deployed on Hugging Face Spaces: `[link]`

**Try it with your own profile.** If you're in the car buying process, plug in your actual DTI, loan amount, and credit utilization and see where you land — and what's driving it.

---

## Stack

```
pandas / numpy          Data wrangling
scikit-learn            Logistic regression, random forest, SMOTE, preprocessing
XGBoost                 Primary model
SHAP                    Explainability
Optuna                  Hyperparameter tuning
Streamlit               Demo app
Hugging Face Spaces     Deployment
```

---

## Project Structure

```
auto-loan-default/
├── data/
│   └── .gitkeep                  # Download instructions in README
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_explainability.ipynb
├── src/
│   ├── features.py
│   ├── train.py
│   └── explain.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Broader Context

This project sits at the intersection of two things I care about: understanding financial systems that affect real decisions in my life, and building ML that's legible — not just accurate.

The explainability layer here isn't an afterthought. In regulated industries, a model that can't explain itself isn't deployable. Building with SHAP from the start, and framing outputs as adverse action notices rather than probability scores, reflects how this actually gets built in production at lenders like Capital One, Ally, or any bank with an auto finance division.

---

## What's Next

- [ ] Swap in auto-specific dataset if one becomes available (HMDA auto subset, dealer-level data)
- [ ] Add fairness audit — check for disparate impact across demographic proxies
- [ ] Build a "what would change my outcome?" counterfactual explainer (DiCE library)
- [ ] Connect to CarIQ for a consumer-facing transparency layer
