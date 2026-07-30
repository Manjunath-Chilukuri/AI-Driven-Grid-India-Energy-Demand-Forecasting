"""
=====================================================
Data Preprocessing Utility
AI-Driven Grid India Level Energy Demand Forecasting
=====================================================
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------
# Missing Value Handling
# ---------------------------------------------------

def handle_missing_values(df):

    """
    Handle missing values.
    """

    before = df.isnull().sum().sum()

    numeric_columns = df.select_dtypes(include=np.number).columns

    for col in numeric_columns:

        df[col] = df[col].fillna(df[col].median())

    object_columns = df.select_dtypes(include="object").columns

    for col in object_columns:

        df[col] = df[col].fillna(df[col].mode()[0])

    after = df.isnull().sum().sum()

    return df, before, after


# ---------------------------------------------------
# Remove Duplicate Records
# ---------------------------------------------------

def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    removed = before - after

    return df.reset_index(drop=True), removed


# ---------------------------------------------------
# Convert Timestamp
# ---------------------------------------------------

def convert_timestamp(df):

    df["START_TIME"] = pd.to_datetime(df["START_TIME"])

    df["END_TIME"] = pd.to_datetime(df["END_TIME"])

    return df


# ---------------------------------------------------
# Sort Chronologically
# ---------------------------------------------------

def sort_by_time(df):

    df = df.sort_values("START_TIME")

    df = df.reset_index(drop=True)

    return df


# ---------------------------------------------------
# Data Types
# ---------------------------------------------------

def check_data_types(df):

    return pd.DataFrame({

        "Column": df.columns,

        "Data Type": df.dtypes.astype(str)

    })


# ---------------------------------------------------
# Missing Value Report
# ---------------------------------------------------

def missing_value_report(df):

    report = pd.DataFrame({

        "Column": df.columns,

        "Missing Values": df.isnull().sum(),

        "Percentage": round(

            df.isnull().sum()/len(df)*100,

            2

        )

    })

    return report


# ---------------------------------------------------
# Duplicate Report
# ---------------------------------------------------

def duplicate_report(df):

    duplicates = df[df.duplicated()]

    return duplicates


# ---------------------------------------------------
# Numerical Validation
# ---------------------------------------------------

def numerical_summary(df):

    return df.describe().T


# ---------------------------------------------------
# Negative Value Check
# ---------------------------------------------------

def negative_values(df):

    numeric = df.select_dtypes(include=np.number)

    result = {}

    for col in numeric.columns:

        result[col] = int((numeric[col] < 0).sum())

    return pd.DataFrame({

        "Column": result.keys(),

        "Negative Values": result.values()

    })


# ---------------------------------------------------
# Infinite Value Check
# ---------------------------------------------------

def infinite_values(df):

    numeric = df.select_dtypes(include=np.number)

    count = np.isinf(numeric).sum().sum()

    return count


# ---------------------------------------------------
# Outlier Detection
# ---------------------------------------------------

def detect_outliers(df):

    numeric = df.select_dtypes(include=np.number)

    outliers = {}

    for col in numeric.columns:

        Q1 = numeric[col].quantile(0.25)

        Q3 = numeric[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR

        upper = Q3 + 1.5 * IQR

        count = ((numeric[col] < lower) |

                 (numeric[col] > upper)).sum()

        outliers[col] = count

    return pd.DataFrame({

        "Column": outliers.keys(),

        "Outliers": outliers.values()

    })


# ---------------------------------------------------
# Data Quality Report
# ---------------------------------------------------

def data_quality(df):

    quality = {

        "Rows": len(df),

        "Columns": len(df.columns),

        "Missing": df.isnull().sum().sum(),

        "Duplicates": df.duplicated().sum(),

        "Infinite": np.isinf(

            df.select_dtypes(include=np.number)

        ).sum().sum()

    }

    return quality


# ---------------------------------------------------
# Complete Pipeline
# ---------------------------------------------------

def preprocessing_pipeline(df):

    df, before_missing, after_missing = handle_missing_values(df)

    df, duplicates_removed = remove_duplicates(df)

    df = convert_timestamp(df)

    df = sort_by_time(df)

    summary = {

        "Missing Before": before_missing,

        "Missing After": after_missing,

        "Duplicates Removed": duplicates_removed

    }

    return df, summary