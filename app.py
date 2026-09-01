"""Interactive credit default risk scoring and local SHAP explanations."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from src.explain import adverse_action_reasons, explain_prediction
from src.features import engineer_features


MODEL_PATH = Path(__file__).parent / "artifacts" / "best_model.joblib"


@st.cache_resource
def load_model_bundle() -> dict:
    """Load the persisted model once per Streamlit process."""

    return joblib.load(MODEL_PATH)


def calculate_installment(amount: float, annual_rate: float, months: int) -> float:
    """Calculate the fixed monthly payment for an amortizing loan."""

    monthly_rate = annual_rate / 100 / 12
    if monthly_rate == 0:
        return amount / months
    factor = (1 + monthly_rate) ** months
    return amount * monthly_rate * factor / (factor - 1)


def employment_label(years: int) -> str:
    if years == 0:
        return "< 1 year"
    if years == 1:
        return "1 year"
    if years >= 10:
        return "10+ years"
    return f"{years} years"


def build_application(
    *,
    loan_amount: float,
    term_months: int,
    interest_rate: float,
    annual_income: float,
    dti_percent: float,
    utilization_percent: float,
    credit_age_years: int,
    delinquency_count: int,
    employment_years: int,
    open_accounts: int,
    public_records: int,
    total_accounts: int,
    mortgage_accounts: int,
    bankruptcy_records: int,
    state: str,
    purpose: str,
) -> pd.DataFrame:
    """Build one raw Lending Club-compatible row from application inputs."""

    issue_date = pd.Timestamp.today().normalize().replace(day=1)
    credit_age_months = credit_age_years * 12
    earliest_credit = issue_date - pd.DateOffset(months=credit_age_months)
    revolving_limit = 10_000.0

    return pd.DataFrame(
        [
            {
                "loan_amnt": loan_amount,
                "term": f"{term_months} months",
                "int_rate": f"{interest_rate}%",
                "installment": calculate_installment(
                    loan_amount, interest_rate, term_months
                ),
                "annual_inc": annual_income,
                "issue_d": issue_date.strftime("%b-%Y"),
                "purpose": purpose,
                "addr_state": state,
                "dti": dti_percent,
                "delinq_2yrs": delinquency_count,
                "earliest_cr_line": earliest_credit.strftime("%b-%Y"),
                "open_acc": open_accounts,
                "pub_rec": public_records,
                "revol_bal": revolving_limit * utilization_percent / 100,
                "revol_util": f"{utilization_percent}%",
                "total_acc": total_accounts,
                "mort_acc": mortgage_accounts,
                "pub_rec_bankruptcies": bankruptcy_records,
                "total_rev_hi_lim": revolving_limit,
                "emp_length": employment_label(employment_years),
            }
        ]
    )


def main() -> None:
    st.set_page_config(page_title="Clarity Credit Risk", page_icon="🔎")
    st.title("Clarity")
    st.caption("Interpretable credit default prediction with SHAP")

    if not MODEL_PATH.exists():
        st.error("No trained model found. Run `python -m src.train` first.")
        st.stop()

    bundle = load_model_bundle()
    encodings = bundle["target_encodings"]
    state_options = sorted(encodings.state_default_rate)
    purpose_options = sorted(encodings.purpose_default_rate)

    with st.form("application"):
        st.subheader("Loan and income")
        loan_left, loan_right = st.columns(2)
        with loan_left:
            loan_amount = st.number_input(
                "Loan amount ($)", 1_000, 100_000, 25_000, 500
            )
            term_months = st.selectbox("Term", [36, 60], index=0)
            interest_rate = st.number_input(
                "Annual interest rate (%)", 0.0, 40.0, 10.0, 0.25
            )
            annual_income = st.number_input(
                "Annual income ($)", 1_000, 2_000_000, 75_000, 1_000
            )
        with loan_right:
            dti_percent = st.number_input(
                "Debt-to-income ratio (%)", 0.0, 150.0, 25.0, 1.0
            )
            utilization_percent = st.number_input(
                "Credit utilization (%)", 0.0, 200.0, 40.0, 1.0
            )
            purpose = st.selectbox(
                "Loan purpose",
                purpose_options,
                index=purpose_options.index("car") if "car" in purpose_options else 0,
                format_func=lambda value: value.replace("_", " ").title(),
            )
            state = st.selectbox(
                "State",
                state_options,
                index=state_options.index("CA") if "CA" in state_options else 0,
            )

        st.subheader("Credit history")
        credit_left, credit_right = st.columns(2)
        with credit_left:
            credit_age_years = st.number_input(
                "Credit history (years)", 1, 70, 10
            )
            employment_years = st.number_input(
                "Employment length (years)", 0, 10, 5
            )
            delinquency_count = st.number_input(
                "Delinquencies in last 2 years", 0, 20, 0
            )
            open_accounts = st.number_input("Open accounts", 1, 100, 10)
        with credit_right:
            total_accounts = st.number_input("Total accounts", 1, 200, 20)
            mortgage_accounts = st.number_input("Mortgage accounts", 0, 50, 1)
            public_records = st.number_input("Public records", 0, 50, 0)
            bankruptcy_records = st.number_input("Bankruptcy records", 0, 20, 0)

        submitted = st.form_submit_button("Assess risk", type="primary")

    if not submitted:
        st.info("Enter an application and select **Assess risk**.")
        return

    raw_application = build_application(
        loan_amount=loan_amount,
        term_months=term_months,
        interest_rate=interest_rate,
        annual_income=annual_income,
        dti_percent=dti_percent,
        utilization_percent=utilization_percent,
        credit_age_years=credit_age_years,
        delinquency_count=delinquency_count,
        employment_years=employment_years,
        open_accounts=open_accounts,
        public_records=public_records,
        total_accounts=total_accounts,
        mortgage_accounts=mortgage_accounts,
        bankruptcy_records=bankruptcy_records,
        state=state,
        purpose=purpose,
    )
    features = engineer_features(raw_application, encodings)
    model = bundle["model"]
    score = float(model.predict_proba(features)[:, 1][0])
    threshold = float(bundle["threshold"])
    high_risk = score >= threshold

    st.divider()
    result_left, result_right = st.columns(2)
    result_left.metric("Model risk score", f"{score:.1%}")
    result_right.metric("Decision threshold", f"{threshold:.1%}")
    if high_risk:
        st.error("Elevated modeled default risk")
    else:
        st.success("Lower modeled default risk")

    with st.spinner("Calculating SHAP contributions..."):
        local = explain_prediction(model, features)

    st.subheader("Primary factors increasing modeled risk")
    reasons = adverse_action_reasons(local)
    if reasons:
        for reason in reasons:
            st.markdown(f"- {reason}")
    else:
        st.write("No feature increased modeled risk relative to the model baseline.")

    st.subheader("Feature contributions")
    display = local.values.head(10)[
        ["label", "display_value", "direction", "shap_value"]
    ].rename(
        columns={
            "label": "Feature",
            "display_value": "Applicant value",
            "direction": "Direction",
            "shap_value": "SHAP impact (log-odds)",
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")

    shap.plots.waterfall(local.shap_explanation, max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

    st.warning(
        "This is an educational model trained on Lending Club personal-loan "
        "data, not a lending decision. Its weighted-model output is a risk "
        "score and has not yet been probability-calibrated."
    )


if __name__ == "__main__":
    main()
