"""COD Shield AI — Fraud Analytics Dashboard."""

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


# Make the project root available for imports.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.risk_engine import assess_order


DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic_orders.csv"
)


st.set_page_config(
    page_title="COD Shield AI",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data
def load_orders():
    """Load the synthetic dataset."""
    return pd.read_csv(DATA_FILE)


st.title("🛡️ COD Shield AI")
st.subheader(
    "AI-powered Cash-on-Delivery abuse detection"
)

st.caption(
    "Portfolio prototype | Synthetic data only | "
    "No connection to real e-commerce platforms"
)


# Load the dataset.
if not DATA_FILE.exists():
    st.error(
        "Dataset not found. Run "
        "'python src/data_generator.py' first."
    )
    st.stop()

df = load_orders()


# Sidebar navigation.
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a module",
    [
        "Analytics Dashboard",
        "AI Risk Checker",
    ],
)


# -----------------------------------------
# PAGE 1: ANALYTICS DASHBOARD
# -----------------------------------------

if page == "Analytics Dashboard":

    st.header("Order Monitoring")

    total_orders = len(df)

    abuse_count = int(
        df["is_simulated_abuse"].sum()
    )

    legitimate_count = (
        total_orders - abuse_count
    )

    abuse_rate = (
        abuse_count / total_orders * 100
        if total_orders
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Orders",
        f"{total_orders:,}",
    )

    col2.metric(
        "Simulated Legitimate Orders",
        f"{legitimate_count:,}",
    )

    col3.metric(
        "Simulated Abuse Orders",
        f"{abuse_count:,}",
    )

    col4.metric(
        "Simulated Abuse Rate",
        f"{abuse_rate:.2f}%",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Order Distribution")

        distribution = (
            df["is_simulated_abuse"]
            .value_counts()
            .rename(index={
                0: "Legitimate",
                1: "Simulated Abuse",
            })
            .reset_index()
        )

        distribution.columns = [
            "Category",
            "Orders",
        ]

        fig = px.pie(
            distribution,
            names="Category",
            values="Orders",
            hole=0.45,
            title="Synthetic Order Categories",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        st.subheader("Order Burst Analysis")

        fig = px.histogram(
            df,
            x="orders_last_15m",
            color="is_simulated_abuse",
            barmode="overlay",
            opacity=0.75,
            labels={
                "orders_last_15m": (
                    "Orders in the Last 15 Minutes"
                ),
                "is_simulated_abuse": (
                    "Simulated Abuse Label"
                ),
            },
            title="Order Frequency Distribution",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader("Recent Synthetic Orders")

    display_columns = [
        "order_id",
        "order_value_inr",
        "orders_last_15m",
        "failed_confirmations",
        "confirmation_status",
        "payment_method",
        "is_simulated_abuse",
    ]

    st.dataframe(
        df[display_columns].head(25),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "These visualizations describe our synthetic "
        "dataset. They do not represent actual platform "
        "activity or verified real-world fraud."
    )


# -----------------------------------------
# PAGE 2: AI RISK CHECKER
# -----------------------------------------

elif page == "AI Risk Checker":

    st.header("AI Order Risk Checker")

    st.write(
        "Enter the details of a fictional order to "
        "receive a model-generated risk assessment."
    )

    with st.form("risk_checker_form"):

        col1, col2 = st.columns(2)

        with col1:
            order_value = st.number_input(
                "Order value (INR)",
                min_value=1.0,
                max_value=1000000.0,
                value=1500.0,
                step=100.0,
            )

            orders_last_15m = st.number_input(
                "Orders in the last 15 minutes",
                min_value=0,
                max_value=1000,
                value=1,
                step=1,
            )

            failed_confirmations = st.number_input(
                "Failed confirmations",
                min_value=0,
                max_value=1000,
                value=0,
                step=1,
            )

        with col2:
            distinct_accounts = st.number_input(
                "Distinct accounts",
                min_value=1,
                max_value=1000,
                value=1,
                step=1,
            )

            confirmation_status = st.selectbox(
                "Confirmation status",
                ["confirmed", "unconfirmed"],
            )

            payment_method = st.selectbox(
                "Payment method",
                ["COD", "PREPAID"],
            )

        submitted = st.form_submit_button(
            "Assess Order Risk"
        )

    if submitted:

        try:
            result = assess_order(
                order_value_inr=order_value,
                orders_last_15m=int(
                    orders_last_15m
                ),
                failed_confirmations=int(
                    failed_confirmations
                ),
                distinct_accounts=int(
                    distinct_accounts
                ),
                confirmation_status=(
                    confirmation_status
                ),
                payment_method=payment_method,
            )

            probability = (
                result["risk_probability"] * 100
            )

            st.divider()

            st.subheader("Assessment Result")

            col1, col2 = st.columns(2)

            col1.metric(
                "Model Abuse Probability",
                f"{probability:.2f}%",
            )

            col2.metric(
                "Risk Category",
                result["risk_category"],
            )

            st.write(
                "**Recommended next step:**",
                result["recommendation"],
            )

            st.progress(
                result["risk_probability"]
            )

            st.caption(
                "This probability reflects the model's "
                "training on synthetic data. It is not a "
                "validated real-world fraud probability."
            )

            st.warning(
                "Do not use this demonstration score "
                "to accuse customers, deny purchases, "
                "or automatically block deliveries."
            )

        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
