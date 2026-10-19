"""Interactive risk checker for fictional COD orders."""

import sys
from pathlib import Path

import streamlit as st


# --------------------------------------------------
# PROJECT SETUP
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cloud_setup import prepare_cloud_files
from src.risk_engine import assess_order

try:
    prepare_cloud_files()
except Exception as exc:
    st.error("Unable to prepare the model for risk assessment.")
    st.exception(exc)
    st.stop()


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.title("🛡️ COD Shield AI — Risk Checker")

st.caption(
    "Machine-learning risk assessment for fictional "
    "Cash-on-Delivery orders."
)

st.warning(
    "Research demonstration only. This model uses synthetic data "
    "and has not been validated for real-world fraud detection. "
    "Do not use its scores to make actual order or customer decisions."
)


# --------------------------------------------------
# SCENARIO SELECTION
# --------------------------------------------------

st.subheader("1. Select a scenario")

scenario = st.selectbox(
    "Fictional scenario",
    ["Normal order", "Order burst"],
)

if scenario == "Normal order":
    defaults = {
        "value": 1500.0,
        "orders": 1,
        "failures": 0,
        "accounts": 1,
        "confirmation": "confirmed",
    }
else:
    defaults = {
        "value": 2500.0,
        "orders": 8,
        "failures": 3,
        "accounts": 5,
        "confirmation": "unconfirmed",
    }

scenario_key = scenario.lower().replace(" ", "_")


# --------------------------------------------------
# ORDER INPUT FORM
# --------------------------------------------------

st.subheader("2. Enter order characteristics")

with st.form("risk_checker_form"):

    left, right = st.columns(2)

    with left:
        order_value = st.number_input(
            "Order value (INR)",
            min_value=1.0,
            max_value=1_000_000.0,
            value=defaults["value"],
            step=100.0,
            key=f"{scenario_key}_value",
        )

        orders_last_15m = st.number_input(
            "Orders in the last 15 minutes",
            min_value=0,
            max_value=1000,
            value=defaults["orders"],
            step=1,
            key=f"{scenario_key}_orders",
        )

        failed_confirmations = st.number_input(
            "Failed confirmations",
            min_value=0,
            max_value=1000,
            value=defaults["failures"],
            step=1,
            key=f"{scenario_key}_failures",
        )

    with right:
        distinct_accounts = st.number_input(
            "Distinct accounts",
            min_value=1,
            max_value=1000,
            value=defaults["accounts"],
            step=1,
            key=f"{scenario_key}_accounts",
        )

        confirmation_status = st.selectbox(
            "Confirmation status",
            ["confirmed", "unconfirmed"],
            index=["confirmed", "unconfirmed"].index(
                defaults["confirmation"]
            ),
            key=f"{scenario_key}_confirmation",
        )

        payment_method = st.selectbox(
            "Payment method",
            ["COD", "PREPAID"],
            key=f"{scenario_key}_payment",
        )

    submitted = st.form_submit_button(
        "Assess Risk",
        type="primary",
        width="stretch",
    )


# --------------------------------------------------
# RISK ASSESSMENT
# --------------------------------------------------

if submitted:

    try:
        result = assess_order(
            order_value_inr=order_value,
            orders_last_15m=int(orders_last_15m),
            failed_confirmations=int(failed_confirmations),
            distinct_accounts=int(distinct_accounts),
            confirmation_status=confirmation_status,
            payment_method=payment_method,
        )

    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))

    else:

        st.divider()

        st.subheader("3. Risk assessment results")

        # Main metrics
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Risk category",
            result["risk_category"],
        )

        col2.metric(
            "Model score",
            f"{result['risk_probability']:.1%}",
        )

        col3.metric(
            "Decision threshold",
            f"{result['decision_threshold']:.0%}",
        )

        # Risk category indicator
        category = result["risk_category"]

        if category == "High":
            st.error(
                f"🔴 {category} Risk — Additional verification "
                "is recommended in this demonstration."
            )

        elif category == "Medium":
            st.warning(
                f"🟡 {category} Risk — Customer confirmation "
                "and review are suggested in this demonstration."
            )

        else:
            st.success(
                f"🟢 {category} Risk — The model assigned a lower "
                "score. Ordinary safeguards would still apply."
            )

        # Score visualization
        st.write("### Model score visualization")

        st.progress(
            min(
                max(float(result["risk_probability"]), 0.0),
                1.0,
            )
        )

        st.caption(
            "The bar visualizes the model's output score. "
            "It is not a validated real-world fraud probability."
        )

        # Recommendation
        st.info(
            result["recommendation"],
            icon="💡",
        )

        # Technical details
        with st.expander("Technical details"):

            st.write(
                f"**Model:** {result['model_type']}"
            )

            st.write(
                f"**Selected decision threshold:** "
                f"{result['decision_threshold']:.2f}"
            )

            st.write(
                f"**Data scope:** {result['data_scope']}"
            )

            st.write(
                f"**Payment method:** {payment_method}"
            )

            st.write(
                f"**Confirmation status:** {confirmation_status}"
            )

        st.caption(
            "COD Shield AI is a portfolio research prototype. "
            "Its synthetic training labels, model scores, and "
            "recommendations do not establish real-world fraud risk."
        )

else:
    st.info(
        "Choose a fictional scenario, review the inputs, "
        "and select Assess Risk to run the model."
    )