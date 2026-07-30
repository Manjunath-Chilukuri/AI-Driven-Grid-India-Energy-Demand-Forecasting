"""
=========================================================
Feature Engineering
AI-Driven Grid India Level Energy Demand Forecasting
=========================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_dataset
from utils.preprocessing import preprocessing_pipeline
from utils.feature_engineering import (
    feature_engineering_pipeline,
    feature_summary
)

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Feature Engineering",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Feature Engineering")

st.markdown("""
This module extracts meaningful temporal features from the
historical Grid India dataset for Machine Learning and
Deep Learning based forecasting.
""")

st.divider()

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

df = load_dataset()

df, _ = preprocessing_pipeline(df)

df = feature_engineering_pipeline(df)

# -------------------------------------------------------
# Metrics
# -------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", len(df))

col2.metric("Original Features", 19)

col3.metric("Engineered Features", 15)

col4.metric("Total Features", len(df.columns))

st.divider()

# -------------------------------------------------------
# Engineered Features
# -------------------------------------------------------

st.subheader("Generated Features")

st.dataframe(
    feature_summary(df),
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# Season Distribution
# -------------------------------------------------------

st.subheader("Season Distribution")

season = df["SEASON"].value_counts().reset_index()
season.columns = ["Season", "Count"]

fig = px.pie(
    season,
    names="Season",
    values="Count",
    title="Season Distribution"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# -------------------------------------------------------
# Weekday Distribution
# -------------------------------------------------------

st.subheader("Weekday Distribution")

weekday = (
    df["WEEKDAY"]
    .value_counts()
    .reset_index()
)

weekday.columns = ["Weekday", "Count"]

order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekday["Weekday"] = pd.Categorical(
    weekday["Weekday"],
    categories=order,
    ordered=True
)

weekday = weekday.sort_values("Weekday")

fig = px.bar(
    weekday,
    x="Weekday",
    y="Count",
    title="Records by Weekday"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# -------------------------------------------------------
# Quarter Distribution
# -------------------------------------------------------

st.subheader("Quarter Distribution")

quarter = (
    df["QUARTER"]
    .value_counts()
    .sort_index()
    .reset_index()
)

quarter.columns = ["Quarter", "Count"]

fig = px.bar(
    quarter,
    x="Quarter",
    y="Count",
    title="Quarter-wise Records"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# -------------------------------------------------------
# Peak Hour Analysis
# -------------------------------------------------------

st.subheader("Peak Hour Analysis")

peak = (
    df["PEAK_HOUR"]
    .value_counts()
    .reset_index()
)

peak.columns = ["Peak Hour", "Count"]

peak["Peak Hour"] = peak["Peak Hour"].replace({
    0: "Non-Peak",
    1: "Peak"
})

fig = px.pie(
    peak,
    names="Peak Hour",
    values="Count",
    title="Peak vs Non-Peak Hours"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# -------------------------------------------------------
# Weekend Analysis
# -------------------------------------------------------

st.subheader("Weekend Analysis")

weekend = (
    df["IS_WEEKEND"]
    .value_counts()
    .reset_index()
)

weekend.columns = ["Weekend", "Count"]

weekend["Weekend"] = weekend["Weekend"].replace({
    0: "Weekday",
    1: "Weekend"
})

fig = px.bar(
    weekend,
    x="Weekend",
    y="Count",
    title="Weekend vs Weekday"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# -------------------------------------------------------
# Time Block Distribution
# -------------------------------------------------------

st.subheader("Time Block Distribution")

tb = (
    df["TIME_BLOCK"]
    .value_counts()
    .sort_index()
    .reset_index()
)

tb.columns = ["Time Block", "Count"]

fig = px.line(
    tb,
    x="Time Block",
    y="Count",
    markers=True,
    title="15-Minute Time Block Distribution"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# -------------------------------------------------------
# Engineered Dataset Preview
# -------------------------------------------------------

st.subheader("Engineered Dataset Preview")

rows = st.slider(
    "Number of Rows",
    5,
    50,
    10
)

st.dataframe(
    df.head(rows),
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# Download
# -------------------------------------------------------

st.subheader("Download Engineered Dataset")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download Engineered Dataset",
    data=csv,
    file_name="Engineered_GridIndia_Dataset.csv",
    mime="text/csv"
)

st.divider()

# -------------------------------------------------------
# Summary
# -------------------------------------------------------

st.success("""
Feature Engineering Completed Successfully.

Generated Features:

• Year

• Month

• Day

• Hour

• Minute

• Weekday

• Day of Week

• Day of Year

• Week Number

• Quarter

• Weekend Indicator

• Time Block

• Peak Hour

• Season

• Weekday Encoding

Dataset is now ready for Machine Learning and Deep Learning model training.
""")