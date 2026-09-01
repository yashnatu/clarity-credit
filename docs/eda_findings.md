# EDA findings

These findings use the first 200,000 rows of the accepted-loans file. That
sample contains loans issued from July through December 2015.

## Target

- 176,083 loans have resolved outcomes and are eligible for modeling.
- 19.93% of eligible loans defaulted.
- Unresolved statuses (`Current`, late, and grace-period loans) are excluded
  rather than labeled as non-defaults.
- Accuracy is not a primary metric. Model comparison uses PR-AUC and
  default-class precision, recall, and F1.

## Data quality

- Employment length has approximately 6.4% missingness.
- Credit utilization has approximately 0.03% missingness.
- Other engineered fields are effectively complete in this sample.
- Two modeled rows report zero annual income; resulting affordability ratios
  are converted to missing values.
- DTI has a 99th percentile of 38.64% but includes a 999% extreme value.
- Revolving utilization can exceed 100%, which can legitimately represent
  borrowers over their reported credit limit.

Missing numeric values are median-imputed using training data only. Logistic
regression also uses train-fitted scaling to reduce sensitivity to magnitude.
Tree models retain the observed extreme values because they split by rank and
can learn whether those values represent elevated risk.

## Leakage controls

- Splits use whole issue months: older loans train the model and newer loans
  form validation and test sets.
- State and purpose default-rate encodings are fitted on training loans only.
- No post-origination repayment fields are loaded into the feature table.
- Decision thresholds are selected on validation predictions; test data is
  used only for final reporting.
