"""
=========================================================
Data Collection
AI-Driven Grid India Level Energy Demand Forecasting
=========================================================
"""

import streamlit as st
import pandas as pd

from utils.data_loader import (
    load_dataset,
    dataset_information,
    statistics,
    preview,
    missing_summary,
    duplicate_summary,
    convert_datetime,
    sort_dataset
)

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Data Collection",
    page_icon="📂",
    layout="wide"
)

st.title("📂 Data Collection")

st.markdown(
"""
This module loads the historical Grid India electricity
demand dataset used throughout the forecasting system.
"""
)

st.divider()

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

df = load_dataset()

df = convert_datetime(df)

df = sort_dataset(df)

# -------------------------------------------------------
# Dataset Metrics
# -------------------------------------------------------

info = dataset_information(df)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Records",
    f"{info['Rows']:,}"
)

c2.metric(
    "Total Columns",
    info["Columns"]
)

c3.metric(
    "Missing Values",
    info["Missing Values"]
)

c4.metric(
    "Duplicate Records",
    info["Duplicate Records"]
)

st.divider()

# -------------------------------------------------------
# Dataset Preview
# -------------------------------------------------------

st.subheader("Dataset Preview")

rows = st.slider(
    "Number of Rows",
    min_value=5,
    max_value=100,
    value=10
)

st.dataframe(
    preview(df, rows),
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# Dataset Information
# -------------------------------------------------------

st.subheader("Dataset Information")

dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(
    dtype_df,
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# Missing Value Summary
# -------------------------------------------------------

st.subheader("Missing Value Summary")

missing = missing_summary(df)

if len(missing) == 0:

    st.success("No Missing Values Found.")

else:

    st.dataframe(
        missing,
        use_container_width=True
    )

st.divider()

# -------------------------------------------------------
# Duplicate Summary
# -------------------------------------------------------

st.subheader("Duplicate Records")

duplicates = duplicate_summary(df)

if duplicates == 0:

    st.success("No Duplicate Records Found.")

else:

    st.warning(
        f"{duplicates} Duplicate Records Found."
    )

st.divider()

# -------------------------------------------------------
# Dataset Statistics
# -------------------------------------------------------

st.subheader("Descriptive Statistics")

st.dataframe(
    statistics(df),
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# Download Dataset
# -------------------------------------------------------

st.subheader("Download Dataset")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(

    label="⬇ Download Dataset",

    data=csv,

    file_name="GridIndia_Dataset.csv",

    mime="text/csv"

)

st.divider()

# -------------------------------------------------------
# Dataset Summary
# -------------------------------------------------------

st.subheader("Dataset Summary")

st.info(
"""
The Grid India dataset contains historical electricity demand
records collected at 15-minute intervals.

It includes:

• Demand Met (MW)

• Thermal Generation

• Hydro Generation

• Solar Generation

• Wind Generation

• Nuclear Generation

• Gas Generation

• Net Demand

• Total Generation

• Frequency

• Time Information

These records are used for Machine Learning and Deep Learning
based electricity demand forecasting.
"""
)
