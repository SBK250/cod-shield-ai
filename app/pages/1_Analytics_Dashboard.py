"""Analytics dashboard for the synthetic COD order dataset."""

from pathlib import Path

import pandas as pd
import streamlit as st
from src.cloud_setup import prepare_cloud_files


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "synthetic_orders.csv"


@st.cache_data
def load_orders() -> pd.DataFrame:
	return pd.read_csv(DATA_FILE)


st.title("Analytics Dashboard")
st.caption("Synthetic demonstration data only")

try:
    prepare_cloud_files()
except Exception as exc:
    st.error("Unable to prepare the application data and model.")
    st.exception(exc)
    st.stop()

orders = load_orders()
required_columns = {
	"order_id",
	"order_value_inr",
	"orders_last_15m",
	"failed_confirmations",
	"distinct_accounts",
	"confirmation_status",
	"payment_method",
	"campaign_id",
	"is_simulated_abuse",
}
missing_columns = required_columns.difference(orders.columns)
if missing_columns:
	st.error(f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}")
	st.stop()

order_count = len(orders)
abuse_count = int(orders["is_simulated_abuse"].sum())
legitimate_count = order_count - abuse_count
abuse_rate = abuse_count / order_count if order_count else 0

with st.container(horizontal=True):
	st.metric("Orders loaded", f"{order_count:,}")
	st.metric("Simulated legitimate", f"{legitimate_count:,}")
	st.metric("Simulated abuse", f"{abuse_count:,}")
	st.metric("Simulated abuse rate", f"{abuse_rate:.2%}")

left, right = st.columns(2)

with left:
	st.subheader("Order labels")
	label_counts = (
		orders["is_simulated_abuse"]
		.map({0: "Legitimate", 1: "Simulated abuse"})
		.value_counts()
		.rename_axis("Order label")
		.reset_index(name="Orders")
	)
	st.bar_chart(
		label_counts,
		x="Order label",
		y="Orders",
		alt="Counts of legitimate and simulated abuse orders",
	)

with right:
	st.subheader("Order bursts")
	burst_counts = (
		pd.crosstab(
			orders["orders_last_15m"],
			orders["is_simulated_abuse"],
		)
		.rename(columns={0: "Legitimate", 1: "Simulated abuse"})
		.rename_axis("Orders in last 15 minutes")
		.reset_index()
	)
	st.bar_chart(
		burst_counts,
		x="Orders in last 15 minutes",
		y=["Legitimate", "Simulated abuse"],
		stack=False,
		alt="Order frequency split by simulated abuse label",
	)

st.subheader(f"Synthetic orders ({order_count:,})")
orders_table = orders.assign(
	simulated_label=orders["is_simulated_abuse"].map(
		{0: "Legitimate", 1: "Simulated abuse"}
	)
).drop(columns="is_simulated_abuse")
st.dataframe(
	orders_table,
	hide_index=True,
	height=520,
	alt="All synthetic orders with features and simulated labels",
)
