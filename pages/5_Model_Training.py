"""
=========================================================
Model Training Dashboard
AI-Driven Grid India Level Energy Demand Forecasting
=========================================================
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Model Training",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning & Deep Learning Models")

st.markdown("""
This dashboard presents the trained Machine Learning and
Deep Learning models used for electricity demand forecasting.
""")

st.divider()

# ----------------------------------------------------
# Models Used
# ----------------------------------------------------

ml_models = [
    "Linear Regression",
    "Random Forest",
    "XGBoost"
]

dl_models = [
    "ANN",
    "LSTM",
    "GRU",
    "Bi-LSTM"
]

col1, col2 = st.columns(2)

with col1:

    st.subheader("Machine Learning")

    for model in ml_models:
        st.success(model)

with col2:

    st.subheader("Deep Learning")

    for model in dl_models:
        st.info(model)

st.divider()

# ----------------------------------------------------
# Performance Report
# ----------------------------------------------------

report_path = "outputs/reports/Model_Performance.csv"

if os.path.exists(report_path):

    report = pd.read_csv(report_path)

    st.subheader("Model Performance")

    st.dataframe(
        report,
        use_container_width=True
    )

    st.divider()

    # ----------------------------------------------

    st.subheader("R² Score Comparison")

    fig = px.bar(
        report,
        x="Model",
        y="R²",
        text="R²",
        color="R²",
        title="Model Comparison (R²)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ----------------------------------------------

    st.subheader("RMSE Comparison")

    fig = px.bar(
        report,
        x="Model",
        y="RMSE",
        color="RMSE",
        title="RMSE Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ----------------------------------------------

    st.subheader("MAE Comparison")

    fig = px.bar(
        report,
        x="Model",
        y="MAE",
        color="MAE",
        title="MAE Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ----------------------------------------------

    best = report.sort_values(
        "R²",
        ascending=False
    ).iloc[0]

    st.success(f"""
### Best Performing Model

**Model:** {best['Model']}

**R² Score:** {best['R²']:.4f}

**RMSE:** {best['RMSE']:.2f}

**MAE:** {best['MAE']:.2f}
""")

    st.divider()

    csv = report.to_csv(index=False).encode()

    st.download_button(

        "⬇ Download Performance Report",

        csv,

        "Model_Performance.csv",

        "text/csv"

    )

else:

    st.warning(
        "Model_Performance.csv not found."
    )

st.divider()

# ----------------------------------------------------
# Model Details
# ----------------------------------------------------

st.subheader("Model Summary")

summary = pd.DataFrame({

    "Model":[
        "Linear Regression",
        "Random Forest",
        "XGBoost",
        "ANN",
        "LSTM",
        "GRU",
        "Bi-LSTM"
    ],

    "Category":[
        "Machine Learning",
        "Machine Learning",
        "Machine Learning",
        "Deep Learning",
        "Deep Learning",
        "Deep Learning",
        "Deep Learning"
    ],

    "Purpose":[
        "Baseline Regression",
        "Ensemble Regression",
        "Boosting Regression",
        "Feedforward Neural Network",
        "Sequence Learning",
        "Sequence Learning",
        "Bidirectional Sequence Learning"
    ]

})

st.dataframe(
    summary,
    use_container_width=True
)

st.divider()

st.info("""
All models were trained using the processed Grid India dataset
and evaluated using:

• MAE

• RMSE

• R² Score

The trained models are saved inside the **models/**
directory and are used directly during forecasting.
""")