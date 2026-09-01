# Clarity — Crash Course

A guided tour of what this project does, why the technical choices matter, and what every jargon term actually means. Written so a student can learn the ideas, an engineer can audit the pipeline, and a recruiter can evaluate the work in about 15 minutes.

---

## 1. The one-paragraph version

Clarity estimates **how likely a borrower is to default** on a loan, using public Lending Club data as a stand-in for auto-loan underwriting. Raw fields (income, installment, revolving balance, delinquencies) are turned into the same kinds of ratios a lender uses — **debt-to-income**, **payment-to-income**, **loan-to-income**, **credit utilization**. Three classifiers are trained with **class-imbalance** handling, compared on **PR-AUC** (not accuracy), and given a **cost-sensitive decision threshold**. The Streamlit application uses **TreeSHAP** to show which inputs pushed an applicant's score up or down.

Lending Club issues personal loans, not auto loans. The mapping is intentional: the credit-risk *structure* is the same (affordability, leverage, utilization, history, ~10–20% default rate), even though the product is different.

---

## What Clarity is and what still needs to be done

Clarity's strongest positioning is:

> **A consumer-facing auto-loan risk simulator that explains the score and identifies the smallest realistic changes needed to improve the outcome.**

The current application delivers the first half of that promise: it scores an
applicant, applies a cost-sensitive decision threshold, and presents local
TreeSHAP contributions as plain-language risk factors.

To make the full distinction credible, Clarity still needs:

1. **Actionable counterfactual recommendations** — identify the smallest
   realistic changes that would move an applicant below the risk threshold.
2. **Probability calibration and uncertainty** — turn the weighted model's
   risk score into a reliable probability and communicate uncertainty.
3. **Fairness and explanation-stability audits** — measure group-level
   performance disparities and verify that reason rankings remain stable.
4. **Loan-term and interest-rate scenario comparison** — let users compare
   realistic financing structures side by side.
5. **Reproducible decision records and model monitoring** — retain model,
   input, threshold, explanation, and timestamp metadata while tracking drift.

These are product and model-governance goals, not claims about the current
implementation. Clarity presently provides adverse-action-style explanations,
not legally validated adverse-action notices.

---

## 2. Who this is for, in 30 seconds

| Audience | What to take away |
|---|---|
| **Student** | Credit default is a rare-event classification problem. Accuracy lies. Features are constructed, not just selected. Time-based splits prevent cheating. |
| **Engineer** | The real work is leakage control, target definition, encoding fitted on train only, and metric/threshold design. Read `src/features.py` then `src/train.py`. |
| **Recruiter** | This is a credit-risk ML project: imbalanced binary classification, underwriting feature engineering, honest metrics, cost-sensitive thresholds, and local TreeSHAP explanations. |

---

## 3. The problem, without ML jargon

When someone applies for a car loan, a lender must answer two questions:

1. **Will this person likely repay?** (risk)
2. **If we say no, or charge a high rate, what is the reason?** (explainability)

A model that always predicts “will repay” looks ~80–90% accurate because most borrowers *do* repay. That model is useless: it never catches the defaults that actually cost money. This project is built around that fact.

**Default** here means the loan ended badly (charged off / defaulted), not that a payment is currently late. Loans still in progress (`Current`, late, in grace) are **dropped**, not labeled as “good.” Labeling an unfinished loan as a non-default would pretend the story is over.

On the first 200,000 accepted-loan rows (issued mid–late 2015):

- 176,083 loans have a resolved outcome and can be modeled
- **19.93% defaulted** — so the default class is the **minority class**, but not an ultra-rare 1% event

---

## 4. What the pipeline actually does

```
raw Lending Club CSV
        │
        ▼
keep completed loans only  →  binary target: default = 1 / paid = 0
        │
        ▼
split by issue month (older → train, newer → val/test)
        │
        ▼
fit state/purpose default-rate encodings on TRAIN only
        │
        ▼
engineer 19 numeric underwriting features
        │
        ▼
impute medians (train-fitted) → train LR / Random Forest / XGBoost
        │
        ▼
pick a probability threshold on validation using FN:FP cost = 5:1
        │
        ▼
report PR-AUC, precision, recall, F1 on untouched test months
        │
        ▼
local TreeSHAP explanations → Streamlit app
```

### Current status (honest)

| Piece | Status |
|---|---|
| Feature table (`src/features.py`) | Implemented |
| EDA notebook + findings | Implemented on a 200k-row sample |
| Training, time split, cost threshold, metrics (`src/train.py`) | Implemented |
| Saved test metrics / `artifacts/` | Implemented; model binary remains ignored |
| SMOTE oversampling | In the project scope and `imbalanced-learn` dependency; **not used in `train.py`** |
| Optuna hyperparameter search | In requirements; **not wired into training yet** |
| SHAP explanations (`src/explain.py`) | Implemented for local XGBoost predictions |
| Streamlit app (`app.py`) | Implemented and connected to the saved model |

The measured 200,000-row baseline selected XGBoost with **0.4307 test
PR-AUC**, **0.4520 default-class F1**, and **75.38% default recall**. These are
baseline results, not production validation.

---

## 5. Data and the target

**Source:** [Lending Club loan data on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (~2M accepted loans). Place `data/accepted_2007_to_2018Q4.csv.gz` locally; raw files are not in git.

**Target construction** (`build_default_target`):

| `loan_status` | Label |
|---|---|
| Charged Off, Default, and the “does not meet credit policy” charged-off variant | `1` (default) |
| Fully Paid, and the paid “does not meet credit policy” variant | `0` (non-default) |
| Current, late, grace period, anything unresolved | excluded (`NaN`) |

That exclusion is a modeling decision, not a detail. Including current loans as 0 would mix “has not defaulted *yet*” with “repaid.”

---

## 6. Feature engineering — the actual underwriting signals

Raw columns are not dumped into the model. `engineer_features` builds a 19-column numeric table that looks like a credit file, not a spreadsheet dump.

| Feature | How it is built | Why a lender cares |
|---|---|---|
| `debt_to_income_ratio` | Lending Club `dti`, scaled to a fraction if stored as percent | Share of income already claimed by debt. High DTI means little room for a new payment. |
| `payment_to_income` | monthly installment / (annual income / 12) | Can this specific payment fit the budget? |
| `loan_to_income` | loan amount / annual income | Leverage. Especially relevant for auto (car price vs income). |
| `credit_utilization` | revolving utilization, or balance / credit limit if util is missing | High revolving use is a classic distress signal. Values **above 100%** can be real (borrower over the reported limit). |
| `credit_age_months` | months from earliest credit line to issue date | Longer history is a stability signal. |
| `delinquency_flag` | 1 if any delinquency in the last 2 years | Hard negative. |
| `emp_length_years` | ordinal: `< 1 year` → 0, `10+ years` → 10 | Employment tenure has a natural order, so it is **not** one-hot encoded. |
| `state_default_rate` | historical default rate of the borrower’s state, **from training data only** | Geography as a risk score. 50 one-hot state dummies would mostly be noise. |
| `purpose_default_rate` | same idea for loan purpose | Some purposes (e.g. small business) historically default more than others. |
| plus | loan amount, term, interest rate, installment, income, open/total/mortgage accounts, public records, bankruptcies | Size, price, and bureau-style depth. |

**Zero income** makes the ratios undefined; those ratios become missing, then median-imputed from **training data only**.

**Interest rate** is a feature here, but it is also a *decision the lender already made*. In a strict origination model you might exclude it (it encodes the lender’s own risk view). It is kept as a strong correlate of risk; be ready to discuss that tradeoff.

---

## 7. The two sins this project is designed to avoid

### 7.1 Leakage

**Leakage** means the model saw information it would not have at decision time, so test scores look great and production scores collapse.

Controls in this repo:

- **No post-origination fields** (payments made, recoveries, months since last payment) enter the feature table.
- **Time split by whole issue months**, ~70/15/15. Older months train; later months validate and test. Random row shuffling would let “December knowledge” leak into a “June” loan.
- **Target encodings fit on train only.** If you compute “NV default rate” using the test set’s own defaults, you have partially handed the model the answer.
- **Threshold chosen on validation.** Test is for final reporting only.

The EDA notebook *does* fit encodings on the full sample for exploration, and it says so. Training does not.

### 7.2 The accuracy trap

If 80% of loans are good, predicting “good” for everyone is 80% **accurate** and 0% useful.

This project therefore:

- Does **not** treat accuracy as the headline metric
- Ranks models by **PR-AUC**
- Reports **precision / recall / F1 of the default class**
- Moves the yes/no cutoff away from 0.5 using explicit costs (default: missing a default costs **5×** a false alarm)

---

## 8. Class imbalance — three different tools, three different jobs

**Class imbalance:** one label is much rarer than the other. Here, defaults are the **minority class** (~20% in the EDA sample; often cited as ~10–15% for this domain).

Imbalance hurts training *and* evaluation. A model can ignore defaults and still look fine on accuracy or even ROC-AUC. The project’s answers:

### A. Reweighting (what `train.py` actually does)

The loss is told “a missed default is more important than a missed non-default,” without inventing new rows.

- **Logistic regression:** `class_weight="balanced"` — sklearn sets each class’s weight to `n_samples / (n_classes × n_samples_in_class)`, so defaults pull as hard as non-defaults in aggregate.
- **Random forest:** `class_weight="balanced_subsample"` — the same idea, recomputed inside each bootstrap sample.
- **XGBoost:** `scale_pos_weight = n_negative / n_positive` on the training split. If there are 4 paid loans per default, each default is treated as 4× as important in the gradient.

**`scale_pos_weight`** is not a probability calibration trick and not SMOTE. It is a **loss multiplier** for the positive class (default = 1). Formula used here:

```text
scale_pos_weight = (count of 0s in train) / (count of 1s in train)
```

### B. SMOTE (scoped, not in the trainer)

**SMOTE** = Synthetic Minority Over-sampling Technique. It builds *new* minority-class training rows by interpolating between a real default and one of its nearest minority neighbors in feature space.

- It is **oversampling**, not reweighting.
- It must run **after the split**, on **train only**. Running SMOTE on the full dataset before splitting is a classic leak (synthetic cousins of test defaults leak into train).
- It can distort probability calibration and is less natural for mixed numeric/categorical credit data than people assume.
- `projectScope.md` lists SMOTE as augmentation. `src/train.py` does **not** call it. `imbalanced-learn` is in `requirements.txt` for that path. Do not say “we got 0.81 via SMOTE”; that collapses two separate ideas (a target PR-AUC, and an unused oversampler).

### C. Threshold tuning (also in `train.py`)

Even a well-trained model outputs a **score** (probability-like number from 0 to 1). Turning that into approve/deny requires a **threshold**.

Default sklearn threshold is 0.5. In lending, **false negatives** (approve someone who defaults) typically cost more than **false positives** (decline or price-up someone who would have paid). The trainer searches validation scores and picks the cutoff that minimizes:

```text
cost = 5 × (false negatives) + 1 × (false positives)
```

Those 5 and 1 are CLI flags (`--false-negative-cost`, `--false-positive-cost`), not learned parameters. Changing them changes who gets flagged, **without retraining**.

---

## 9. Models, and what “winning” means

| Model | Role | Imbalance knob |
|---|---|---|
| **Logistic regression** | Linear baseline; coefficients are signed and readable | `class_weight="balanced"` + StandardScaler |
| **Random forest** | Nonlinear interactions; bagging | `balanced_subsample` |
| **XGBoost** | Gradient-boosted trees; expected strongest ranker | `scale_pos_weight` |

**Why trees often beat logistic regression here:** default risk is interactive. High DTI *and* high utilization is worse than either alone. A linear model adds effects; trees can split on combinations.

**How a winner is chosen:** highest **validation PR-AUC**. Thresholds are tuned per model on validation; test metrics are reported but do not pick the model.

**Optuna** (in requirements, not yet used) is a hyperparameter optimizer. It would search things like tree depth and learning rate to maximize validation PR-AUC. Current XGBoost settings are fixed in code (`n_estimators=400`, `max_depth=5`, `learning_rate=0.05`, …).

The measured XGBoost test PR-AUC is **0.4307** on the 200,000-row baseline.
PR-AUC's random-ranking baseline is approximately the default prevalence
(~0.20 here), so this is meaningful lift but leaves substantial room for
feature, calibration, and tuning work.

---

## 10. Metrics glossary — what to say in an interview

### Confusion matrix (after a threshold)

For the default class as “positive”:

|  | Predicted pay (0) | Predicted default (1) |
|---|---|---|
| **Actually paid** | True negative (TN) | False positive (FP) — false alarm |
| **Actually defaulted** | False negative (FN) — missed default | True positive (TP) |

- **Precision (default class)** = TP / (TP + FP). Of those we flagged, how many really defaulted?
- **Recall (default class)** = TP / (TP + FN). Of all actual defaults, how many did we catch? Also called **sensitivity** or **true positive rate**.
- **F1 (default class)** = harmonic mean of precision and recall. One number when you need both. “F1 on the minority class” means F1 computed for default=1, not a macro-average that hides poor default detection.

### ROC-AUC vs PR-AUC

**ROC curve:** true positive rate vs false positive rate, as you sweep the threshold. **ROC-AUC** is the area under that curve. 0.5 = random, 1.0 = perfect ranking.

**Precision–recall curve:** precision vs recall as you sweep the threshold. **PR-AUC** (also called **average precision**, which is what sklearn’s `average_precision_score` estimates) is the area under *that* curve.

**Why PR-AUC is the headline here:** ROC-AUC can look strong when negatives dominate, because it is easy to have a low false-positive *rate* when there are huge numbers of true negatives. PR-AUC focuses on the rare class: it asks, as you try to catch more defaults, how contaminated is the flagged set? Random ranking yields PR-AUC ≈ prevalence (here ~0.20), so 0.81 would be a large lift.

**Accuracy** = (TP+TN) / n. Not a primary metric in this project.

---

## 11. SHAP explainability — what it is, and why credit cares

**SHAP** = SHapley Additive exPlanations. It borrows **Shapley values** from cooperative game theory: each feature gets a payout equal to its average contribution to the prediction across all coalitions of other features.

Properties that matter in production:

- **Additivity:** `prediction ≈ baseline + sum of SHAP values`. You can point at a score and decompose it.
- **Local:** explanation for *this* applicant, not only global importance.
- **Consistent ranking:** if a feature contributes more, its SHAP magnitude is not arbitrarily smaller than a less important feature’s.

The current and planned explanation views are:

1. **Local / waterfall plot — implemented:** this applicant's score relative
   to the model baseline, with ranked risk-increasing and risk-reducing inputs.
2. **Global — planned:** mean |SHAP| across applicants.
3. **Dependence / interaction — planned:** e.g. DTI's effect changing with
   interest rate.

**Why this is not optional in U.S. lending:** **ECOA** (Equal Credit Opportunity Act) and related fair-lending rules require **adverse action notices** — if credit is denied or priced worse, the applicant is owed principal reasons. A score without reasons is hard to deploy. SHAP is one engineering answer; it is **not** by itself a fairness audit (that is listed under “what’s next”).

**Do not overclaim:** SHAP explains *the model*, not ground-truth causality. If the model relies on a proxy, SHAP will faithfully surface the proxy.

---

## 12. Other terms you will hit in this repo

**Underwriting** — the lender’s process of deciding whether to extend credit and on what terms.

**Origination** — the moment the loan is booked. Features must be knowable at origination.

**Charge-off** — accounting write-off of a loan the lender does not expect to collect. Treated as default here.

**DTI (debt-to-income)** — debt burden vs income. In the sample, the 99th percentile is ~39%, with a 999% sentinel-style extreme. Trees can split on extremes; linear models are more sensitive, which is why logistic regression is scaled.

**Credit utilization** — revolving balance / credit limit. Over 100% is allowed in this dataset.

**Ordinal encoding** — map ordered categories to numbers (`emp_length`). Contrast **one-hot** (a dummy column per category) and **target encoding** (replace a category with its historical default rate).

**Target encoding** — `state_default_rate` and `purpose_default_rate`. Powerful and leak-prone. Train-only fit is mandatory.

**Imputation** — fill missing values. Here, **median imputation fitted on train**. Trees could theoretically leave missings, but the pipelines impute for a uniform matrix.

**StandardScaler** — subtract mean, divide by standard deviation. Used for logistic regression so coefficients are not dominated by raw magnitude (income in tens of thousands vs DTI ~0.2).

**Pipeline (sklearn)** — imputer (+ scaler) + model saved as one object, so the same transforms apply at train and serve time.

**Temporal / out-of-time validation** — train on the past, test on the future. The right default for credit; random CV overstates performance when the world drifts.

**Cost-sensitive threshold** — convert probabilities to actions using business costs, not 0.5.

**Calibration** — whether a predicted 0.30 means “about 30% of similar people default.” Ranking metrics (PR-AUC) can be strong while probabilities are miscalibrated. Reweighting and SMOTE often hurt calibration; a separate calibration step would be a production follow-on.

**Proxy dataset** — Lending Club stands in for auto loans. Same *family* of signals, different product (no LTV against a vehicle, no dealer, no used-car residual). The README is explicit about this.

**Disparate impact** — a model can harm a protected group even without using race/gender. Listed as future work, not implemented.

**Counterfactual explanation** — “what is the smallest change that would flip the decision?” (e.g. DiCE). Also future work.

**Adverse action notice** — legally required statement of principal reasons for a negative credit decision.

---

## 13. How to read the code efficiently

Start here, in order:

1. **`src/features.py`** — target definition, encodings, the 19 features. This is the domain logic.
2. **`docs/eda_findings.md`** then **`notebooks/01_eda.ipynb`** — imbalance, missingness, outliers, leakage notes on a 200k sample.
3. **`src/train.py`** — month split, models, cost threshold, PR-AUC selection, artifact dump.
4. **`src/explain.py`** and **`app.py`** — local SHAP reasons and the interactive product surface.
5. **`projectScope.md`** — narrative and roadmap. Cross-check against the code; scope remains ahead of implementation for SMOTE, Optuna, fairness, and deployment.

Train (after the CSV is in `data/`):

```bash
python -m src.train --data data/accepted_2007_to_2018Q4.csv.gz --rows 200000
```

`--rows 0` reads the full file. Output: `artifacts/best_model.joblib` and `artifacts/metrics.json`.

---

## 14. How to talk about this project without overselling

**Accurate:** “Binary credit-default model on Lending Club as an
auto-underwriting proxy. Time-based splits, train-only target encoding,
imbalance via class weights and `scale_pos_weight`, model selection by PR-AUC,
a cost-sensitive threshold, and local TreeSHAP explanations in Streamlit.”

**Misleading:** “Clarity is an auto-loan approval system with compliant adverse
action notices.” The training data is a personal-loan proxy, the app is
educational, and its SHAP reasons have not undergone legal or fairness
validation.

The interesting engineering is not “we used XGBoost.” It is **defining default only on completed loans**, **refusing accuracy as a vanity metric**, **not leaking the future into the past**, and **separating ranking quality (PR-AUC) from the business decision (threshold + costs)**.
