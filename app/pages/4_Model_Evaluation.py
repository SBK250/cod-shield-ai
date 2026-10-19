"""COD Shield AI — Model Evaluation."""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import GroupShuffleSplit


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic_orders.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "cod_risk_model.joblib"
)

THRESHOLD_FILE = (
    PROJECT_ROOT
    / "models"
    / "decision_threshold.json"
)

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
# PAGE HEADER
# --------------------------------------------------

st.title("📈 COD Shield AI — Model Evaluation")

st.caption(
    "Evaluation of the saved Logistic Regression model "
    "using held-out synthetic campaigns."
)

st.warning(
    "Research demonstration only. All labels are synthetic. "
    "These metrics do not establish real-world fraud-detection "
    "performance or calibrated fraud probabilities."
)


# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

@st.cache_data
def load_data(path, modified_time):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)


@st.cache_data
def load_threshold(path, modified_time):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# CAMPAIGN-BASED TEST SPLIT
# --------------------------------------------------

@st.cache_data
def create_test_indices(data):
    """
    Reproduce the outer campaign-based split used during training.
    """

    groups = data[GROUP]
    y = data[TARGET].astype(int)
    X = data[FEATURES]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    train_campaigns = set(groups.iloc[train_indices])
    test_campaigns = set(groups.iloc[test_indices])

    if train_campaigns.intersection(test_campaigns):
        raise ValueError(
            "Campaign overlap detected between training and test sets."
        )

    if y.iloc[test_indices].nunique() < 2:
        raise ValueError(
            "The test set must contain both target classes."
        )

    return test_indices


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

@st.cache_data
def calculate_metrics(y_test, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    return {
        "predictions": predictions,
        "matrix": matrix,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "average_precision": average_precision_score(
            y_test,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }


# --------------------------------------------------
# CHECK REQUIRED FILES
# --------------------------------------------------

missing_files = [
    path
    for path in [DATA_FILE, MODEL_FILE, THRESHOLD_FILE]
    if not path.exists()
]

if missing_files:
    st.error(
        "Required project files are missing:\n\n"
        + "\n".join(str(path) for path in missing_files)
    )
    st.stop()


try:
    df = load_data(
        str(DATA_FILE),
        DATA_FILE.stat().st_mtime,
    )

    required_columns = FEATURES + [TARGET, GROUP]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    model = load_model(
        str(MODEL_FILE),
        MODEL_FILE.stat().st_mtime,
    )

    threshold_config = load_threshold(
        str(THRESHOLD_FILE),
        THRESHOLD_FILE.stat().st_mtime,
    )

    decision_threshold = float(
        threshold_config["decision_threshold"]
    )

    if not 0 < decision_threshold < 1:
        raise ValueError(
            "Saved decision threshold must be between 0 and 1."
        )

    test_indices = create_test_indices(df)

    test_df = df.iloc[test_indices].copy()

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(int)

    probabilities = model.predict_proba(X_test)[:, 1]

    results = calculate_metrics(
        y_test,
        probabilities,
        decision_threshold,
    )

except (
    ValueError,
    KeyError,
    FileNotFoundError,
    json.JSONDecodeError,
) as exc:
    st.error(f"Evaluation could not be completed: {exc}")
    st.stop()


# --------------------------------------------------
# DATASET SUMMARY
# --------------------------------------------------

st.subheader("Test Dataset")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total synthetic orders",
    f"{len(df):,}",
)

col2.metric(
    "Test orders",
    f"{len(test_df):,}",
)

col3.metric(
    "Test campaigns",
    f"{test_df[GROUP].nunique():,}",
)

col4.metric(
    "Selected threshold",
    f"{decision_threshold:.0%}",
)

st.caption(
    f"Test-set simulated abuse rate: {y_test.mean():.2%}"
)

st.divider()


# --------------------------------------------------
# CLASSIFICATION METRICS
# --------------------------------------------------

st.subheader("Classification Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{results['accuracy']:.3f}")
c2.metric("Abuse precision", f"{results['precision']:.3f}")
c3.metric("Abuse recall", f"{results['recall']:.3f}")
c4.metric("Abuse F1-score", f"{results['f1']:.3f}")

c5, c6 = st.columns(2)

c5.metric(
    "Average precision",
    f"{results['average_precision']:.3f}",
)

c6.metric(
    "ROC-AUC",
    f"{results['roc_auc']:.3f}",
)

st.caption(
    "Precision measures how many flagged orders are labelled "
    "abuse in the synthetic test set. Recall measures how many "
    "simulated-abuse orders were detected. F1 balances precision "
    "and recall."
)

st.info(
    f"Predictions use the saved decision threshold of "
    f"{decision_threshold:.2f}. The ROC and Precision–Recall "
    "curves use continuous model scores rather than this cutoff."
)

st.divider()


# --------------------------------------------------
# CONFUSION MATRIX AND ROC CURVE
# --------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")

    tn, fp, fn, tp = results["matrix"].ravel()

    matrix_df = pd.DataFrame(
        results["matrix"],
        index=["Actual Legitimate", "Actual Abuse"],
        columns=["Predicted Legitimate", "Predicted Abuse"],
    )

    fig = px.imshow(
        matrix_df,
        text_auto=True,
        aspect="auto",
        title="Actual vs Predicted",
        labels={
            "x": "Predicted class",
            "y": "Actual class",
            "color": "Number of orders",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.write(f"True negatives: {tn}")
    st.write(f"False positives: {fp}")
    st.write(f"False negatives: {fn}")
    st.write(f"True positives: {tp}")


with right:
    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )

    roc_df = pd.DataFrame(
        {
            "False Positive Rate": fpr,
            "True Positive Rate": tpr,
        }
    )

    fig = px.line(
        roc_df,
        x="False Positive Rate",
        y="True Positive Rate",
        title=f"ROC-AUC: {results['roc_auc']:.3f}",
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line={"dash": "dash"},
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# --------------------------------------------------
# PRECISION-RECALL CURVE
# --------------------------------------------------

st.divider()

st.subheader("Precision–Recall Curve")

precision_values, recall_values, _ = precision_recall_curve(
    y_test,
    probabilities,
)

pr_df = pd.DataFrame(
    {
        "Recall": recall_values,
        "Precision": precision_values,
    }
)

fig = px.line(
    pr_df,
    x="Recall",
    y="Precision",
    title=(
        "Precision–Recall Curve | "
        f"Average Precision: {results['average_precision']:.3f}"
    ),
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

st.divider()

st.subheader("Classification Report")

report = classification_report(
    y_test,
    results["predictions"],
    labels=[0, 1],
    target_names=["Legitimate", "Simulated Abuse"],
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(3),
    use_container_width=True,
)


# --------------------------------------------------
# RESEARCH LIMITATIONS
# --------------------------------------------------

st.divider()

st.subheader("Research Limitations")

st.markdown(
    """
    - The dataset is artificially generated.
    - The labels are assigned by the synthetic data generator.
    - The generator creates patterns that may be easier to
      distinguish than real-world abuse.
    - Campaign-based splitting prevents campaign overlap between
      the training and test sets, but does not establish
      generalization to real-world campaigns.
    - The saved model was trained on the outer training split;
      the evaluation page uses the matching held-out test split.
    - The decision threshold was selected using a separate
      validation split within the training data.
    - Model scores are not calibrated real-world fraud probabilities.
    - No real customer data or platform integrations are used.
    """
)

st.caption(
    "COD Shield AI is a synthetic-data portfolio research prototype. "
    "Its results should not be interpreted as validated "
    "real-world fraud-detection performance."
)