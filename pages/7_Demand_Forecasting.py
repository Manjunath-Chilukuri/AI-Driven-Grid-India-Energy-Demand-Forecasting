"""
=========================================================
Demand Forecasting
AI-Driven Grid India Level Energy Demand Forecasting
=========================================================
"""

import os
import streamlit as st
import pandas as pd

from utils.prediction_engine import (
    predict_linear_regression,
    predict_random_forest,
    predict_xgboost,
    predict_ann,
    predict_lstm,
    predict_gru,
    predict_bilstm
)

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Electricity Demand Forecasting")

st.markdown("""
Forecast future electricity demand using the trained
Machine Learning and Deep Learning models.
""")

st.divider()

# -------------------------------------------------------
# Model Selection
# -------------------------------------------------------

model = st.selectbox(
    "Select Forecasting Model",
    [
        "Linear Regression",
        "Random Forest",
        "XGBoost",
        "ANN",
        "LSTM",
        "GRU",
        "BiLSTM"
    ]
)

st.divider()

# -------------------------------------------------------
# Input
# -------------------------------------------------------

c1, c2 = st.columns(2)

with c1:

    year = st.number_input(
        "Year",
        min_value=2024,
        max_value=2035,
        value=2026
    )

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=6
    )

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=18
    )

with c2:

    hour = st.number_input(
        "Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    minute = st.selectbox(
        "Minute",
        [0, 15, 30, 45]
    )

    timeblock = st.number_input(
        "Time Block",
        min_value=1,
        max_value=96,
        value=1
    )

st.divider()

# -------------------------------------------------------
# Prediction
# -------------------------------------------------------

from datetime import datetime

if st.button("Forecast Demand", use_container_width=True):

    try:

        forecast_date = datetime(
            int(year),
            int(month),
            int(day)
        )

        if model == "Linear Regression":

            prediction = predict_linear_regression(
                forecast_date,
                int(timeblock)
            )

        elif model == "Random Forest":

            prediction = predict_random_forest(
                forecast_date,
                int(timeblock)
            )

        elif model == "XGBoost":

            prediction = predict_xgboost(
                forecast_date,
                int(timeblock)
            )

        elif model == "ANN":

            prediction = predict_ann(
                forecast_date,
                int(timeblock)
            )

        elif model == "LSTM":

            prediction = predict_lstm()

        elif model == "GRU":

            prediction = predict_gru()

        else:

            prediction = predict_bilstm()

        st.success("Prediction Completed Successfully")

        st.metric(
            "Predicted Electricity Demand (MW)",
            f"{float(prediction):,.2f} MW"
        )

        st.info(f"""
Forecast Date : {forecast_date.strftime("%d-%m-%Y")}

Time Block : {timeblock}
""")

    except Exception as e:

        st.error(f"Prediction Failed\n\n{e}")
st.subheader("Model Information")

info = {

    "Linear Regression":
        "Baseline regression model.",

    "Random Forest":
        "Ensemble learning model using multiple decision trees.",

    "XGBoost":
        "Gradient Boosting model with high forecasting accuracy.",

    "ANN":
        "Artificial Neural Network for nonlinear demand prediction.",

    "LSTM":
        "Long Short-Term Memory network for time-series forecasting.",

    "GRU":
        "Gated Recurrent Unit network with efficient sequence learning.",

    "BiLSTM":
        "Bidirectional LSTM capturing both forward and backward temporal dependencies."

}

st.info(info[model])