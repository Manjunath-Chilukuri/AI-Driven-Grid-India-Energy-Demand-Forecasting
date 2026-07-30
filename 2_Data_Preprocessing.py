"""
=========================================================
Data Preprocessing
AI-Driven Grid India Level Energy Demand Forecasting
=========================================================
"""

import streamlit as st
import pandas as pd

from utils.data_loader import load_dataset
from utils.preprocessing import (
    preprocessing_pipeline,
    data_quality,
    check_data_types,
    missing_value_report,
    duplicate_report,
    negative_values,
    infinite_values,
    detect_outliers
)

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Data Preprocessing",
    page_icon="⚙",
    layout="wide"
)

st.title("⚙ Data Preprocessing")

st.markdown("""
This module performs data cleaning, validation,
duplicate removal, missing value handling,
and quality assessment before feature engineering.
""")

st.divider()

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

df = load_dataset()

original_rows = len(df)

# -------------------------------------------------------
# Execute Pipeline
# -------------------------------------------------------

clean_df, summary = preprocessing_pipeline(df)

quality = data_quality(clean_df)

# -------------------------------------------------------
# Dashboard Metrics
# -------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Rows",
    quality["Rows"]
)

c2.metric(
    "Columns",
    quality["Columns"]
)

c3.metric(
    "Missing Values",
    quality["Missing"]
)

c4.metric(
    "Duplicates",
    quality["Duplicates"]
)

st.divider()

# -------------------------------------------------------
# Pipeline Summary
# -------------------------------------------------------

st.subheader("Preprocessing Summary")

summary_df = pd.DataFrame({

    "Step":[

        "Missing Values Before",

        "Missing Values After",

        "Duplicates Removed"

    ],

    "Value":[

        summary["Missing Before"],

        summary["Missing After"],

        summary["Duplicates Removed"]

    ]

})

st.dataframe(
    summary_df,
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# Data Quality Report
# -------------------------------------------------------

st.subheader("Data Quality Report")

quality_df = pd.DataFrame({

    "Metric":quality.keys(),

    "Value":quality.values()

})

st.dataframe(
    quality_df,
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# Data Types
# -------------------------------------------------------

st.subheader("Column Data Types")

st.dataframe(

    check_data_types(clean_df),

    use_container_width=True

)

st.divider()

# -------------------------------------------------------
# Missing Value Report
# -------------------------------------------------------

st.subheader("Missing Value Analysis")

st.dataframe(

    missing_value_report(clean_df),

    use_container_width=True

)

st.divider()

# -------------------------------------------------------
# Duplicate Records
# -------------------------------------------------------

st.subheader("Duplicate Records")

duplicates = duplicate_report(clean_df)

if len(duplicates)==0:

    st.success("No duplicate records found.")

else:

    st.dataframe(

        duplicates,

        use_container_width=True

    )

st.divider()

# -------------------------------------------------------
# Negative Values
# -------------------------------------------------------

st.subheader("Negative Value Analysis")

st.dataframe(

    negative_values(clean_df),

    use_container_width=True

)

st.divider()

# -------------------------------------------------------
# Infinite Values
# -------------------------------------------------------

st.subheader("Infinite Value Analysis")

count = infinite_values(clean_df)

if count==0:

    st.success("No infinite values found.")

else:

    st.error(f"{count} Infinite Values Found")

st.divider()

# -------------------------------------------------------
# Outlier Detection
# -------------------------------------------------------

st.subheader("Outlier Detection (IQR Method)")

outliers = detect_outliers(clean_df)

st.dataframe(

    outliers,

    use_container_width=True

)

st.divider()

# -------------------------------------------------------
# Preview Clean Dataset
# -------------------------------------------------------

st.subheader("Clean Dataset Preview")

rows = st.slider(

    "Rows",

    5,

    50,

    10

)

st.dataframe(

    clean_df.head(rows),

    use_container_width=True

)

st.divider()

# -------------------------------------------------------
# Download
# -------------------------------------------------------

st.subheader("Download Clean Dataset")

csv = clean_df.to_csv(index=False).encode("utf-8")

st.download_button(

    "⬇ Download Clean Dataset",

    csv,

    "Clean_GridIndia_Dataset.csv",

    "text/csv"

)

st.divider()

# -------------------------------------------------------
# Conclusion
# -------------------------------------------------------

st.success("""

✔ Missing values handled

✔ Duplicate records removed

✔ Timestamp converted

✔ Dataset sorted chronologically

✔ Dataset validated successfully

Ready for Feature Engineering.

""")