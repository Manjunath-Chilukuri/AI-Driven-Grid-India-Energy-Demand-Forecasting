import streamlit as st
from PIL import Image

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI-Driven Grid India Level Energy Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#F8F9FA;
}

h1{
    color:#0B3C5D;
}

h2{
    color:#0B3C5D;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:15px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚡ Navigation")

st.sidebar.success(
    "AI-Driven Grid India Level\nEnergy Demand Forecasting"
)

st.sidebar.info(
"""
Modules

• Data Collection

• Data Preprocessing

• Feature Engineering

• Exploratory Data Analysis

• Machine Learning

• Model Comparison

• Demand Forecasting

Use the Pages menu to navigate.
"""
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("⚡ AI-Driven Grid India Level Energy Demand Forecasting")

st.markdown(
"""
Artificial Intelligence Based Electricity Demand Forecasting
using Historical Grid India Dataset.
"""
)

st.divider()

# --------------------------------------------------
# PROJECT OVERVIEW
# --------------------------------------------------

col1,col2,col3,col4=st.columns(4)

col1.metric(
    "Dataset",
    "Grid India"
)

col2.metric(
    "Interval",
    "15 Minutes"
)

col3.metric(
    "Study Period",
    "2024-2026"
)

col4.metric(
    "Target",
    "Demand (MW)"
)

st.divider()

# --------------------------------------------------
# SYSTEM WORKFLOW
# --------------------------------------------------

st.subheader("System Workflow")

st.markdown("""

""")

st.divider()

# --------------------------------------------------
# MODULES
# --------------------------------------------------

st.subheader("Project Modules")

c1,c2,c3=st.columns(3)

with c1:

    st.info("""
    📂 Data Collection

    Load Grid India Dataset

    View Dataset

    Dataset Statistics
    """)

with c2:

    st.success("""
    ⚙ Feature Engineering

    Generate

    • Year

    • Month

    • Day

    • Hour

    • Weekday

    • Quarter
    """)

with c3:

    st.warning("""
    📈 Machine Learning

    Linear Regression

    Random Forest

    XGBoost

    ANN

    LSTM

    GRU

    Bi-LSTM
    """)

st.divider()

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.caption(
    "M.Tech Dissertation | AI-Driven Grid India Level Energy Demand Forecasting"
)