"""Train and compare default-risk models without temporal or encoding leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import (
    FEATURE_COLUMNS,
    TargetEncodings,
    engineer_features,
    fit_target_encodings,
    load_lending_club_data,
    prepare_completed_loans,
)


RANDOM_SEED = 42


def split_by_issue_month(
    loans: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Split complete calendar months into 70/15/15 train/validation/test sets."""

    months = sorted(pd.Timestamp(month) for month in loans["issue_month"].unique())
    if len(months) < 3:
        raise ValueError("At least three distinct issue months are required.")

    train_count = max(1, int(len(months) * 0.70))
    validation_count = max(1, int(len(months) * 0.15))
    if train_count + validation_count >= len(months):
        train_count = len(months) - 2
        validation_count = 1

    train_months = months[:train_count]
    validation_months = months[train_count : train_count + validation_count]
    test_months = months[train_count + validation_count :]

    splits = (
        loans[loans["issue_month"].isin(train_months)].copy(),
        loans[loans["issue_month"].isin(validation_months)].copy(),
        loans[loans["issue_month"].isin(test_months)].copy(),
    )
    for name, split in zip(("train", "validation", "test"), splits):
        if split.empty or split["default"].nunique() < 2:
            raise ValueError(f"The {name} split must contain both target classes.")

    month_labels = {
        "train": [month.strftime("%Y-%m") for month in train_months],
        "validation": [month.strftime("%Y-%m") for month in validation_months],
        "test": [month.strftime("%Y-%m") for month in test_months],
    }
    return *splits, month_labels


def build_feature_splits(
    train_loans: pd.DataFrame,
    validation_loans: pd.DataFrame,
    test_loans: pd.DataFrame,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.Series],
    TargetEncodings,
]:
    """Fit categorical encodings on training outcomes and transform each split."""

    encodings = fit_target_encodings(train_loans)
    loan_splits = {
        "train": train_loans,
        "validation": validation_loans,
        "test": test_loans,
    }
    features = {
        name: engineer_features(split, encodings)
        for name, split in loan_splits.items()
    }
    targets = {
        name: split.loc[features[name].index, "default"].astype(int)
        for name, split in loan_splits.items()
    }
    return features, targets, encodings


def build_models(positive_weight: float, seed: int) -> dict[str, Pipeline]:
    """Create imbalance-aware candidate pipelines."""

    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=18,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=5,
                        min_child_weight=5,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        scale_pos_weight=positive_weight,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def select_cost_threshold(
    target: pd.Series,
    scores: np.ndarray,
    *,
    false_negative_cost: float,
    false_positive_cost: float,
) -> float:
    """Choose the validation threshold with minimum empirical misclassification cost."""

    y = np.asarray(target, dtype=int)
    order = np.argsort(scores)[::-1]
    sorted_scores = np.asarray(scores)[order]
    sorted_target = y[order]

    true_positives = np.cumsum(sorted_target)
    false_positives = np.cumsum(1 - sorted_target)
    false_negatives = true_positives[-1] - true_positives
    costs = (
        false_negative_cost * false_negatives
        + false_positive_cost * false_positives
    )

    # Only evaluate boundaries that produce a distinct classification.
    boundary = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    candidate_indices = np.flatnonzero(boundary)
    best_index = candidate_indices[np.argmin(costs[candidate_indices])]
    return float(sorted_scores[best_index])


def evaluate(
    target: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int | list[list[int]]]:
    """Calculate ranking and threshold-dependent minority-class metrics."""

    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    return {
        "pr_auc": float(average_precision_score(target, scores)),
        "roc_auc": float(roc_auc_score(target, scores)),
        "threshold": float(threshold),
        "precision_default": float(
            precision_score(target, predictions, zero_division=0)
        ),
        "recall_default": float(recall_score(target, predictions, zero_division=0)),
        "f1_default": float(f1_score(target, predictions, zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "rows": int(len(target)),
        "default_rate": float(np.mean(target)),
    }


def train_models(
    features: dict[str, pd.DataFrame],
    targets: dict[str, pd.Series],
    *,
    seed: int,
    false_negative_cost: float,
    false_positive_cost: float,
) -> tuple[dict[str, Pipeline], dict[str, dict[str, Any]], str]:
    """Fit candidates, tune thresholds on validation, and report untouched test metrics."""

    negatives = int((targets["train"] == 0).sum())
    positives = int((targets["train"] == 1).sum())
    models = build_models(negatives / positives, seed)
    metrics: dict[str, dict[str, Any]] = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(features["train"], targets["train"])

        validation_scores = model.predict_proba(features["validation"])[:, 1]
        threshold = select_cost_threshold(
            targets["validation"],
            validation_scores,
            false_negative_cost=false_negative_cost,
            false_positive_cost=false_positive_cost,
        )
        test_scores = model.predict_proba(features["test"])[:, 1]
        metrics[name] = {
            "validation": evaluate(
                targets["validation"], validation_scores, threshold
            ),
            "test": evaluate(targets["test"], test_scores, threshold),
        }

    best_name = max(
        models,
        key=lambda name: metrics[name]["validation"]["pr_auc"],
    )
    return models, metrics, best_name


def print_summary(metrics: dict[str, dict[str, Any]], best_name: str) -> None:
    """Print a compact comparison table."""

    header = (
        f"{'model':<22} {'val PR-AUC':>10} {'test PR-AUC':>12} "
        f"{'test F1':>9} {'test recall':>12} {'threshold':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for name, results in metrics.items():
        validation = results["validation"]
        test = results["test"]
        print(
            f"{name:<22} {validation['pr_auc']:>10.4f} "
            f"{test['pr_auc']:>12.4f} {test['f1_default']:>9.4f} "
            f"{test['recall_default']:>12.4f} {test['threshold']:>10.4f}"
        )
    print(f"\nSelected by validation PR-AUC: {best_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/accepted_2007_to_2018Q4.csv.gz"),
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=200_000,
        help="Rows to read; use 0 for the complete dataset.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--false-negative-cost", type=float, default=5.0)
    parser.add_argument("--false-positive-cost", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.false_negative_cost <= 0 or args.false_positive_cost <= 0:
        raise ValueError("Misclassification costs must be positive.")

    row_limit = None if args.rows == 0 else args.rows
    print(f"Loading {args.data} ({row_limit or 'all'} rows)...")
    raw = load_lending_club_data(args.data, nrows=row_limit)
    loans = prepare_completed_loans(raw)
    train_loans, validation_loans, test_loans, months = split_by_issue_month(loans)
    features, targets, encodings = build_feature_splits(
        train_loans, validation_loans, test_loans
    )

    split_summary = {
        name: {
            "rows": int(len(target)),
            "default_rate": float(target.mean()),
            "months": months[name],
        }
        for name, target in targets.items()
    }
    print(json.dumps(split_summary, indent=2))

    models, metrics, best_name = train_models(
        features,
        targets,
        seed=args.seed,
        false_negative_cost=args.false_negative_cost,
        false_positive_cost=args.false_positive_cost,
    )
    print_summary(metrics, best_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold = metrics[best_name]["validation"]["threshold"]
    bundle = {
        "model_name": best_name,
        "model": models[best_name],
        "threshold": threshold,
        "target_encodings": encodings,
        "feature_columns": FEATURE_COLUMNS,
        "split_months": months,
        "false_negative_cost": args.false_negative_cost,
        "false_positive_cost": args.false_positive_cost,
    }
    joblib.dump(bundle, args.output_dir / "best_model.joblib")

    report = {
        "selected_model": best_name,
        "selection_metric": "validation_pr_auc",
        "split_summary": split_summary,
        "costs": {
            "false_negative": args.false_negative_cost,
            "false_positive": args.false_positive_cost,
        },
        "models": metrics,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"Saved model and metrics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
