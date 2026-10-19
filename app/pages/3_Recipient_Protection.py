"""COD Shield AI — Recipient Protection Portal."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import (
    create_incident,
    get_incidents,
    initialize_database,
)


st.set_page_config(
    page_title="Recipient Protection | COD Shield AI",
    page_icon="🛡️",
    layout="wide",
)

initialize_database()

st.title("🛡️ Recipient Protection Portal")

st.write(
    "Report unwanted Cash-on-Delivery orders "
    "and review submitted incident reports."
)

st.warning(
    "Demonstration only. Use fictional information. "
    "This portal does not contact shopping platforms "
    "or cancel orders."
)

st.divider()

tab_report, tab_history = st.tabs(
    ["Report an Incident", "Incident History"]
)


with tab_report:
    st.subheader("Submit an Incident Report")

    with st.form("incident_form"):

        order_reference = st.text_input(
            "Order reference",
            max_chars=50,
            help="Use a fictional order reference.",
        )

        platform = st.selectbox(
            "Platform",
            [
                "Amazon",
                "Flipkart",
                "Zomato",
                "Swiggy",
                "Blinkit",
                "Other",
            ],
        )

        incident_type = st.selectbox(
            "Incident type",
            [
                "Unwanted COD order",
                "Repeated unwanted orders",
                "Order not placed by recipient",
                "Other suspicious activity",
            ],
        )

        description = st.text_area(
            "What happened?",
            max_chars=1000,
            help=(
                "Do not enter phone numbers, home addresses, "
                "passwords, payment information, or other "
                "sensitive personal details."
            ),
        )

        submitted = st.form_submit_button(
            "Submit Incident"
        )

    if submitted:

        if not order_reference.strip():
            st.error("Enter a fictional order reference.")

        elif not description.strip():
            st.error("Please describe the incident.")

        else:
            incident_id = create_incident(
                order_reference=order_reference.strip(),
                platform=platform,
                incident_type=incident_type,
                description=description.strip(),
            )

            st.success(
                f"Incident submitted successfully. "
                f"Reference: INC-{incident_id:05d}"
            )


with tab_history:
    st.subheader("Submitted Incidents")

    incidents = get_incidents()

    if incidents:
        df = pd.DataFrame(incidents)

        df["incident_reference"] = df["id"].apply(
            lambda value: f"INC-{value:05d}"
        )

        display_columns = [
            "incident_reference",
            "order_reference",
            "platform",
            "incident_type",
            "reported_at",
            "status",
        ]

        st.dataframe(
            df[display_columns],
            width="stretch",
            hide_index=True,
        )

        selected_id = st.selectbox(
            "View incident details",
            options=df["id"].tolist(),
            format_func=lambda value: (
                f"INC-{value:05d}"
            ),
        )

        selected = next(
            item
            for item in incidents
            if item["id"] == selected_id
        )

        st.write(
            "**Description:**",
            selected["description"],
        )

        st.write(
            "**Current status:**",
            selected["status"],
        )

    else:
        st.info(
            "No incidents submitted yet. "
            "Create a fictional report to test the portal."
        )