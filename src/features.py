"""Feature engineering for the auto loan default predictor.

The raw Lending Club dataset is personal-loan data, but the engineered
features mirror common auto underwriting signals: affordability, leverage,
credit utilization, credit history length, and delinquency history.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}

NON_DEFAULT_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}

RAW_COLUMNS = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "total_rev_hi_lim",
]

FEATURE_COLUMNS = [
    "loan_amount",
    "term_months",
    "interest_rate",
    "installment",
    "annual_income",
    "debt_to_income_ratio",
    "payment_to_income",
    "loan_to_income",
    "credit_utilization",
    "credit_age_months",
    "delinquency_flag",
    "emp_length_years",
    "open_accounts",
    "public_records",
    "total_accounts",
    "mortgage_accounts",
    "bankruptcy_records",
    "state_default_rate",
    "purpose_default_rate",
]


@dataclass(frozen=True)
class TargetEncodings:
    """Target encodings fitted from a training sample."""

    state_default_rate: dict[str, float]
    purpose_default_rate: dict[str, float]
    global_default_rate: float


def load_lending_club_data(
    path: str | Path,
    *,
    nrows: int | None = None,
    usecols: Iterable[str] | None = RAW_COLUMNS,
) -> pd.DataFrame:
    """Load raw Lending Club accepted loan data."""

    return pd.read_csv(path, nrows=nrows, usecols=usecols, low_memory=False)


def prepare_training_table(
    raw: pd.DataFrame,
    *,
    encodings: TargetEncodings | None = None,
) -> tuple[pd.DataFrame, pd.Series, TargetEncodings]:
    """Create model features and the binary default target.

    If `encodings` is omitted, state and purpose target encodings are fitted
    from `raw`. For final modeling, fit encodings only on the training split
    and pass them into validation/test transformations.
    """

    loans = prepare_completed_loans(raw)

    if encodings is None:
        encodings = fit_target_encodings(loans)

    features = engineer_features(loans, encodings)
    target = loans.loc[features.index, "default"].rename("default")
    return features, target, encodings


def prepare_completed_loans(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep loans with resolved outcomes and attach target and issue month."""

    loans = raw.copy()
    loans["default"] = build_default_target(loans["loan_status"])
    loans = loans[loans["default"].notna()].copy()
    loans["default"] = loans["default"].astype(int)
    loans["issue_month"] = pd.to_datetime(
        loans["issue_d"], format="%b-%Y", errors="coerce"
    )
    return loans[loans["issue_month"].notna()].copy()


def build_default_target(status: pd.Series) -> pd.Series:
    """Map resolved Lending Club statuses to a binary default target."""

    normalized = status.astype("string").str.strip()
    target = pd.Series(np.nan, index=status.index, dtype="float")
    target.loc[normalized.isin(DEFAULT_STATUSES)] = 1
    target.loc[normalized.isin(NON_DEFAULT_STATUSES)] = 0
    return target


def fit_target_encodings(loans: pd.DataFrame) -> TargetEncodings:
    """Fit simple default-rate encodings for high-cardinality categoricals."""

    default_rate = float(loans["default"].mean())

    state_rates = (
        loans.groupby("addr_state", dropna=True)["default"].mean().astype(float).to_dict()
    )
    purpose_rates = (
        loans.groupby("purpose", dropna=True)["default"].mean().astype(float).to_dict()
    )

    return TargetEncodings(
        state_default_rate=state_rates,
        purpose_default_rate=purpose_rates,
        global_default_rate=default_rate,
    )


def engineer_features(loans: pd.DataFrame, encodings: TargetEncodings) -> pd.DataFrame:
    """Convert raw Lending Club rows into numeric underwriting features."""

    features = pd.DataFrame(index=loans.index)

    features["loan_amount"] = pd.to_numeric(loans["loan_amnt"], errors="coerce")
    features["term_months"] = _clean_term_months(loans["term"])
    features["interest_rate"] = _clean_percent(loans["int_rate"])
    features["installment"] = pd.to_numeric(loans["installment"], errors="coerce")
    features["annual_income"] = pd.to_numeric(loans["annual_inc"], errors="coerce")

    dti = pd.to_numeric(loans["dti"], errors="coerce")
    features["debt_to_income_ratio"] = np.where(dti > 1, dti / 100, dti)

    monthly_income = features["annual_income"] / 12
    features["payment_to_income"] = _safe_divide(features["installment"], monthly_income)
    features["loan_to_income"] = _safe_divide(
        features["loan_amount"], features["annual_income"]
    )
    features["credit_utilization"] = _build_credit_utilization(loans)
    features["credit_age_months"] = _build_credit_age_months(loans)
    features["delinquency_flag"] = (
        pd.to_numeric(loans["delinq_2yrs"], errors="coerce").fillna(0) > 0
    ).astype(int)
    features["emp_length_years"] = _clean_emp_length(loans["emp_length"])
    features["open_accounts"] = pd.to_numeric(loans["open_acc"], errors="coerce")
    features["public_records"] = pd.to_numeric(loans["pub_rec"], errors="coerce")
    features["total_accounts"] = pd.to_numeric(loans["total_acc"], errors="coerce")
    features["mortgage_accounts"] = pd.to_numeric(loans["mort_acc"], errors="coerce")
    features["bankruptcy_records"] = pd.to_numeric(
        loans["pub_rec_bankruptcies"], errors="coerce"
    )
    features["state_default_rate"] = loans["addr_state"].map(
        encodings.state_default_rate
    )
    features["purpose_default_rate"] = loans["purpose"].map(
        encodings.purpose_default_rate
    )

    fallback_columns = ["state_default_rate", "purpose_default_rate"]
    features[fallback_columns] = features[fallback_columns].fillna(
        encodings.global_default_rate
    )

    return features.replace([np.inf, -np.inf], np.nan)[FEATURE_COLUMNS]


def _clean_percent(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace("%", "", regex=False).str.strip()
    values = pd.to_numeric(cleaned, errors="coerce").astype("float64")
    scaled = np.where(values > 1, values / 100, values)
    return pd.Series(scaled, index=series.index, dtype="float64")


def _clean_term_months(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def _clean_emp_length(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip().str.lower()
    values = cleaned.replace(
        {
            "< 1 year": "0",
            "1 year": "1",
            "10+ years": "10",
            "n/a": np.nan,
        }
    )
    values = values.str.replace(" years", "", regex=False)
    return pd.to_numeric(values, errors="coerce").astype("float64")


def _build_credit_utilization(loans: pd.DataFrame) -> pd.Series:
    revol_util = pd.Series(_clean_percent(loans["revol_util"]), index=loans.index)
    revol_bal = pd.to_numeric(loans["revol_bal"], errors="coerce")
    credit_limit = pd.to_numeric(loans["total_rev_hi_lim"], errors="coerce")
    computed_util = _safe_divide(revol_bal, credit_limit)
    return revol_util.fillna(computed_util)


def _build_credit_age_months(loans: pd.DataFrame) -> pd.Series:
    earliest_credit = pd.to_datetime(
        loans["earliest_cr_line"], format="%b-%Y", errors="coerce"
    )
    issue_date = pd.to_datetime(loans["issue_d"], format="%b-%Y", errors="coerce")

    return (issue_date.dt.year - earliest_credit.dt.year) * 12 + (
        issue_date.dt.month - earliest_credit.dt.month
    )


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator
