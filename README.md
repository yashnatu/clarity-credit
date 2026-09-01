# Clarity

**Interpretable credit default prediction with SHAP**

This project predicts borrower default risk from lending data and explains each prediction with SHAP-based drivers.

The first implementation uses Lending Club loan data as a proxy for auto loan credit risk because it contains similar underwriting signals: income, installment burden, credit utilization, delinquency history, and repayment outcome.

## Run Locally

Run these commands from the project root.

### 1. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost also requires the OpenMP runtime:

```bash
brew install libomp
```

### 2. Add the data

Download the Lending Club loan data from Kaggle:

https://www.kaggle.com/datasets/wordsforthewise/lending-club

Place the raw accepted loans CSV in `data/`, for example:

```text
data/accepted_2007_to_2018Q4.csv.gz
```

Raw data files are intentionally not tracked in git.

### 3. Train the models

The default command reads 200,000 rows and uses whole issue months for
train/validation/test splits:

```bash
python -m src.train
```

Use the complete dataset after validating the sample workflow:

```bash
python -m src.train --rows 0
```

State and purpose target encodings, missing-value imputation, and scaling are
fit on training data only. The classification threshold is selected on the
validation period using a configurable 5:1 false-negative-to-false-positive
cost. The untouched test period is used for final reporting.

The first 200,000-row baseline selected XGBoost with 0.4307 test PR-AUC, 0.4520
default-class F1, and 75.38% default recall. Generated metrics are saved to
`artifacts/metrics.json`; the ignored model binary is written to
`artifacts/best_model.joblib`.

### 4. Start the application

```bash
streamlit run app.py
```

Open http://localhost:8501 in a browser. Enter a loan and borrower profile,
then select **Assess risk** to view the model risk score, decision threshold,
plain-language risk factors, and SHAP waterfall plot.

Stop the server with `Ctrl+C`.

The app converts an applicant profile into the same engineered features used
during training, applies the saved decision threshold, and presents local
TreeSHAP contributions as plain-language risk reasons. The current weighted
model output is labeled as a risk score rather than a calibrated probability.

## Current Build Status

1. Feature engineering pipeline — complete
2. Exploratory data analysis — complete
3. Baseline model comparison — complete
4. SHAP explanations — complete
5. Interactive Streamlit application — complete
