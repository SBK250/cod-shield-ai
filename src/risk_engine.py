"""Reusable COD Shield AI risk scoring interface."""

from pathlib import Path
import json

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = PROJECT_ROOT / "models" / "cod_risk_model.joblib"

THRESHOLD_FILE = PROJECT_ROOT / "models" / "decision_threshold.json"


def assess_order(
    order_value_inr: float,
    orders_last_15m: int,
    failed_confirmations: int,
    distinct_accounts: int,
    confirmation_status: str,
    payment_method: str,
) -> dict:
    """Score a fictional COD order using the trained model."""

    # Check model and threshold files
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Trained model not found. Run python src/train_model.py first."
        )

    if not THRESHOLD_FILE.exists():
        raise FileNotFoundError(
            "Decision threshold not found. Run python src/train_model.py first."
        )

    # Validate inputs
    if order_value_inr <= 0:
        raise ValueError("Order value must be positive.")

    if orders_last_15m < 0:
        raise ValueError("Order count cannot be negative.")

    if failed_confirmations < 0:
        raise ValueError(
            "Failed confirmation count cannot be negative."
        )

    if distinct_accounts < 1:
        raise ValueError(
            "Distinct account count must be at least one."
        )

    if confirmation_status not in {"confirmed", "unconfirmed"}:
        raise ValueError("Invalid confirmation status.")

    if payment_method not in {"COD", "PREPAID"}:
        raise ValueError("Invalid payment method.")

    # Load model and selected threshold
    model = joblib.load(MODEL_FILE)

    with open(THRESHOLD_FILE, "r", encoding="utf-8") as file:
        threshold_config = json.load(file)

    decision_threshold = float(
        threshold_config["decision_threshold"]
    )

    if not 0 < decision_threshold < 1:
        raise ValueError(
            "Decision threshold must be between 0 and 1."
        )

    # Prepare input
    order = pd.DataFrame(
        [
            {
                "order_value_inr": order_value_inr,
                "orders_last_15m": orders_last_15m,
                "failed_confirmations": failed_confirmations,
                "distinct_accounts": distinct_accounts,
                "confirmation_status": confirmation_status,
                "payment_method": payment_method,
            }
        ]
    )

    # Predict score
    probability = float(model.predict_proba(order)[0, 1])

    # Assign risk category
    if probability >= 0.70:
        category = "High"
        recommendation = (
            "Request additional verification before dispatch."
        )

    elif probability >= decision_threshold:
        category = "Medium"
        recommendation = (
            "Request customer confirmation and review the order."
        )

    else:
        category = "Low"
        recommendation = (
            "Continue normal processing; retain ordinary safeguards."
        )

    return {
        "risk_probability": round(probability, 4),
        "risk_category": category,
        "recommendation": recommendation,
        "decision_threshold": decision_threshold,
        "model_type": "Logistic Regression",
        "data_scope": "Synthetic demonstration only",
    }