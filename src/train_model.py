from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "processed" / "synthetic_orders.csv"
MODEL_PATH = ROOT_DIR / "models" / "cod_risk_model.joblib"
THRESHOLD_PATH = ROOT_DIR / "models" / "decision_threshold.json"

TARGET = "is_simulated_abuse"
GROUP = "campaign_id"

NUMERIC_FEATURES = [
    "order_value_inr",
    "orders_last_15m",
    "failed_confirmations",
    "distinct_accounts",
]

CATEGORICAL_FEATURES = [
    "confirmation_status",
    "payment_method",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------
# 2. BUILD THE MODEL PIPELINE
# --------------------------------------------------

def build_model():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    return model


# --------------------------------------------------
# 3. HELPER: GROUP-BASED SPLIT
# --------------------------------------------------

def split_by_campaign(X, y, groups, test_size, random_state):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, holdout_idx = next(
        splitter.split(X, y, groups=groups)
    )

    train_groups = set(groups.iloc[train_idx])
    holdout_groups = set(groups.iloc[holdout_idx])

    if train_groups.intersection(holdout_groups):
        raise ValueError("Campaign leakage detected!")

    if y.iloc[train_idx].nunique() < 2:
        raise ValueError("Training split must contain both classes.")

    if y.iloc[holdout_idx].nunique() < 2:
        raise ValueError("Holdout split must contain both classes.")

    return train_idx, holdout_idx


# --------------------------------------------------
# 4. LOAD DATA
# --------------------------------------------------

def main():
    print("\nCOD SHIELD AI - MODEL TRAINING")
    print("=" * 55)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run src/data_generator.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET, GROUP]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    groups = df[GROUP].copy()

    if y.nunique() != 2:
        raise ValueError("The target must contain both classes.")

    print(f"Total orders: {len(df)}")
    print(f"Total campaigns: {groups.nunique()}")
    print(f"Simulated abuse rate: {y.mean():.2%}")

    # --------------------------------------------------
    # 5. CREATE OUTER TRAINING AND TEST SETS
    # --------------------------------------------------

    train_idx, test_idx = split_by_campaign(
        X,
        y,
        groups,
        test_size=0.20,
        random_state=42,
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)

    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    print("\nOUTER DATA SPLIT")
    print("-" * 55)
    print(f"Training orders: {len(X_train)}")
    print(f"Test orders: {len(X_test)}")
    print(f"Training campaigns: {groups_train.nunique()}")
    print(f"Test campaigns: {groups.iloc[test_idx].nunique()}")

    # --------------------------------------------------
    # 6. CREATE INNER FIT AND VALIDATION SETS
    # --------------------------------------------------

    fit_idx, validation_idx = split_by_campaign(
        X_train,
        y_train,
        groups_train,
        test_size=0.20,
        random_state=21,
    )

    X_fit = X_train.iloc[fit_idx]
    y_fit = y_train.iloc[fit_idx]

    X_validation = X_train.iloc[validation_idx]
    y_validation = y_train.iloc[validation_idx]

    print("\nTHRESHOLD VALIDATION SPLIT")
    print("-" * 55)
    print(f"Model-fitting orders: {len(X_fit)}")
    print(f"Validation orders: {len(X_validation)}")
    print(f"Validation abuse cases: {int(y_validation.sum())}")

    # --------------------------------------------------
    # 7. TRAIN TEMPORARY MODEL
    # --------------------------------------------------

    validation_model = build_model()
    validation_model.fit(X_fit, y_fit)

    validation_probabilities = validation_model.predict_proba(
        X_validation
    )[:, 1]

    # --------------------------------------------------
    # 8. COMPARE DECISION THRESHOLDS
    # --------------------------------------------------

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    results = []

    print("\nTHRESHOLD COMPARISON")
    print("-" * 75)
    print(
        f"{'Threshold':<12}"
        f"{'Precision':<13}"
        f"{'Recall':<13}"
        f"{'F1-score':<13}"
        f"{'False alarms':<15}"
    )

    for threshold in thresholds:
        predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        ).ravel()

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_alarms": int(fp),
                "missed_abuse": int(fn),
                "true_positives": int(tp),
                "true_negatives": int(tn),
            }
        )

        print(
            f"{threshold:<12.2f}"
            f"{precision:<13.3f}"
            f"{recall:<13.3f}"
            f"{f1:<13.3f}"
            f"{fp:<15}"
        )

    # Select the threshold with the highest validation F1.
    # If multiple thresholds tie, prefer the higher threshold
    # to avoid unnecessary false alarms.
    best_result = max(
        results,
        key=lambda row: (row["f1"], row["threshold"]),
    )

    selected_threshold = float(best_result["threshold"])

    print("\nSELECTED THRESHOLD")
    print("-" * 55)
    print(f"Threshold: {selected_threshold:.2f}")
    print(f"Validation precision: {best_result['precision']:.3f}")
    print(f"Validation recall: {best_result['recall']:.3f}")
    print(f"Validation F1-score: {best_result['f1']:.3f}")

    # --------------------------------------------------
    # 9. RETRAIN FINAL MODEL ON THE COMPLETE OUTER TRAIN SET
    # --------------------------------------------------

    final_model = build_model()
    final_model.fit(X_train, y_train)

    # --------------------------------------------------
    # 10. EVALUATE ON THE UNTOUCHED TEST SET
    # --------------------------------------------------

    test_probabilities = final_model.predict_proba(X_test)[:, 1]

    test_predictions = (
        test_probabilities >= selected_threshold
    ).astype(int)

    print("\nFINAL TEST RESULTS")
    print("=" * 55)
    print(f"Selected threshold: {selected_threshold:.2f}")
    print(f"Accuracy: {accuracy_score(y_test, test_predictions):.3f}")
    print(
        "Precision: "
        f"{precision_score(y_test, test_predictions, zero_division=0):.3f}"
    )
    print(
        "Recall: "
        f"{recall_score(y_test, test_predictions, zero_division=0):.3f}"
    )
    print(
        "F1-score: "
        f"{f1_score(y_test, test_predictions, zero_division=0):.3f}"
    )
    print(
        "Average precision: "
        f"{average_precision_score(y_test, test_probabilities):.3f}"
    )
    print(
        "ROC-AUC: "
        f"{roc_auc_score(y_test, test_probabilities):.3f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["Legitimate", "Simulated abuse"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            test_predictions,
            labels=[0, 1],
        )
    )

    # --------------------------------------------------
    # 11. SAVE MODEL AND THRESHOLD
    # --------------------------------------------------

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, MODEL_PATH)

    threshold_config = {
        "decision_threshold": selected_threshold,
        "selection_metric": "validation_f1",
        "validation_precision": float(best_result["precision"]),
        "validation_recall": float(best_result["recall"]),
        "validation_f1": float(best_result["f1"]),
        "data_type": "synthetic",
        "campaign_grouped_split": True,
    }

    with open(THRESHOLD_PATH, "w", encoding="utf-8") as file:
        json.dump(threshold_config, file, indent=4)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Threshold saved to: {THRESHOLD_PATH}")

    print("\nIMPORTANT:")
    print(
        "These results use synthetic data. They do not establish "
        "real-world fraud-detection performance."
    )


if __name__ == "__main__":
    main()