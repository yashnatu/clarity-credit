"""Local SHAP explanations and plain-language credit-risk reasons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


FEATURE_LABELS = {
    "loan_amount": "Loan amount",
    "term_months": "Loan term",
    "interest_rate": "Interest rate",
    "installment": "Monthly payment",
    "annual_income": "Annual income",
    "debt_to_income_ratio": "Debt-to-income ratio",
    "payment_to_income": "Payment-to-income ratio",
    "loan_to_income": "Loan-to-income ratio",
    "credit_utilization": "Credit utilization",
    "credit_age_months": "Credit history length",
    "delinquency_flag": "Recent delinquency",
    "emp_length_years": "Employment length",
    "open_accounts": "Open accounts",
    "public_records": "Public records",
    "total_accounts": "Total accounts",
    "mortgage_accounts": "Mortgage accounts",
    "bankruptcy_records": "Bankruptcy records",
    "state_default_rate": "State historical default rate",
    "purpose_default_rate": "Loan-purpose historical default rate",
}

PERCENT_FEATURES = {
    "interest_rate",
    "debt_to_income_ratio",
    "payment_to_income",
    "loan_to_income",
    "credit_utilization",
    "state_default_rate",
    "purpose_default_rate",
}

CURRENCY_FEATURES = {
    "loan_amount",
    "installment",
    "annual_income",
}


@dataclass(frozen=True)
class LocalExplanation:
    """A prediction explanation with additive log-odds contributions."""

    values: pd.DataFrame
    base_value: float
    shap_explanation: shap.Explanation


def explain_prediction(model: Pipeline, features: pd.DataFrame) -> LocalExplanation:
    """Calculate TreeSHAP values for one row from a fitted XGBoost pipeline."""

    if len(features) != 1:
        raise ValueError("Local explanations require exactly one feature row.")
    if "imputer" not in model.named_steps or "model" not in model.named_steps:
        raise ValueError("Expected a pipeline with imputer and model steps.")

    transformed = model.named_steps["imputer"].transform(features)
    estimator = model.named_steps["model"]
    explainer = shap.TreeExplainer(estimator)
    raw_explanation = explainer(transformed, check_additivity=False)[0]
    explanation = shap.Explanation(
        values=raw_explanation.values,
        base_values=raw_explanation.base_values,
        data=transformed[0],
        feature_names=list(features.columns),
    )

    shap_values = np.asarray(explanation.values, dtype=float)
    values = pd.DataFrame(
        {
            "feature": features.columns,
            "label": [FEATURE_LABELS.get(name, name) for name in features.columns],
            "value": features.iloc[0].to_numpy(dtype=float),
            "shap_value": shap_values,
        }
    )
    values["direction"] = np.where(
        values["shap_value"] >= 0, "Higher modeled risk", "Lower modeled risk"
    )
    values["absolute_impact"] = values["shap_value"].abs()
    values["display_value"] = [
        format_feature_value(feature, value)
        for feature, value in zip(values["feature"], values["value"])
    ]
    values = values.sort_values("absolute_impact", ascending=False).reset_index(drop=True)

    return LocalExplanation(
        values=values,
        base_value=float(np.asarray(explanation.base_values).reshape(-1)[0]),
        shap_explanation=explanation,
    )


def adverse_action_reasons(
    explanation: LocalExplanation,
    *,
    limit: int = 4,
) -> list[str]:
    """Return the strongest features pushing this applicant toward default."""

    higher_risk = explanation.values[explanation.values["shap_value"] > 0].head(limit)
    return [
        f"{row.label}: {row.display_value}"
        for row in higher_risk.itertuples(index=False)
    ]


def format_feature_value(feature: str, value: float) -> str:
    """Format engineered values for consumer-facing display."""

    if np.isnan(value):
        return "not reported"
    if feature in PERCENT_FEATURES:
        return f"{value:.1%}"
    if feature in CURRENCY_FEATURES:
        return f"${value:,.0f}"
    if feature == "credit_age_months":
        return f"{value / 12:.1f} years"
    if feature == "term_months":
        return f"{value:.0f} months"
    if feature == "delinquency_flag":
        return "Yes" if value >= 0.5 else "No"
    return f"{value:,.1f}"
