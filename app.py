import streamlit as st

st.set_page_config(
    page_title="COD Shield AI",
    page_icon="🛡️",
    layout="wide",
)

pages = [
    st.Page(
        "app/pages/1_Analytics_Dashboard.py",
        title="Analytics Dashboard",
        icon="📊",
        default=True,
    ),
    st.Page(
        "app/pages/2_AI_Risk_Checker.py",
        title="AI Risk Checker",
        icon="🤖",
    ),
    st.Page(
        "app/pages/3_Recipient_Protection.py",
        title="Recipient Protection",
        icon="🛡️",
    ),
    st.Page(
    "app/pages/4_Model_Evaluation.py",
    title="Model Evaluation",
    icon="📈",
)
]

navigation = st.navigation(pages)
navigation.run()