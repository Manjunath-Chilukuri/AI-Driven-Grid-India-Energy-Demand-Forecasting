"""
=============================================
Data Loader Utility
AI-Driven Grid India Level Energy Demand Forecasting
=============================================
"""

import os
import pandas as pd
import streamlit as st


# -------------------------------------------------------
# Dataset Path
# -------------------------------------------------------

DATA_PATH = "data.xlsx"


# -------------------------------------------------------
# Original Column Names
# -------------------------------------------------------

COLUMN_NAMES = [

    "START_TIME",
    "TIME",
    "TIMEBLOCK",
    "FREQUENCY",
    "DEMAND_MET_MW",
    "NUCLEAR_MW",
    "WIND_MW",
    "SOLAR_MW",
    "HYDRO_MW",
    "GAS_MW",
    "THERMAL_MW",
    "OTHERS_MW",
    "NET_DEMAND_MW",
    "TOTAL_GENERATION_MW",
    "DAY",
    "MONTH",
    "YEAR",
    "END_TIME",
    "NET_EXCHANGE"

]


# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

@st.cache_data(show_spinner=True)

def load_dataset():

    """
    Load Grid India dataset.
    """

    if not os.path.exists(DATA_PATH):

        st.error("Dataset not found.")

        st.stop()

    try:

        df = pd.read_excel(

            DATA_PATH,

            skiprows=1

        )

        df.columns = COLUMN_NAMES

        return df

    except Exception as e:

        st.error(f"Unable to load dataset.\n\n{e}")

        st.stop()


# -------------------------------------------------------
# Dataset Information
# -------------------------------------------------------

def dataset_information(df):

    info = {

        "Rows": df.shape[0],

        "Columns": df.shape[1],

        "Missing Values": df.isnull().sum().sum(),

        "Duplicate Records": df.duplicated().sum()

    }

    return info


# -------------------------------------------------------
# Numeric Columns
# -------------------------------------------------------

def numeric_columns(df):

    return df.select_dtypes(

        include="number"

    ).columns.tolist()


# -------------------------------------------------------
# Date Conversion
# -------------------------------------------------------

def convert_datetime(df):

    df["START_TIME"] = pd.to_datetime(

        df["START_TIME"]

    )

    df["END_TIME"] = pd.to_datetime(

        df["END_TIME"]

    )

    return df


# -------------------------------------------------------
# Sort Dataset
# -------------------------------------------------------

def sort_dataset(df):

    return df.sort_values(

        by="START_TIME"

    ).reset_index(drop=True)


# -------------------------------------------------------
# Preview Dataset
# -------------------------------------------------------

def preview(df, rows=10):

    return df.head(rows)


# -------------------------------------------------------
# Missing Value Summary
# -------------------------------------------------------

def missing_summary(df):

    missing = df.isnull().sum()

    return missing[missing > 0]


# -------------------------------------------------------
# Duplicate Summary
# -------------------------------------------------------

def duplicate_summary(df):

    return df.duplicated().sum()


# -------------------------------------------------------
# Dataset Statistics
# -------------------------------------------------------

def statistics(df):

    return df.describe().T
