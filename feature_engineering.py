"""
=========================================================
Feature Engineering Utility
AI-Driven Grid India Level Energy Demand Forecasting
=========================================================
"""

import pandas as pd
import numpy as np


# -------------------------------------------------------
# Extract Date-Time Features
# -------------------------------------------------------

def extract_datetime_features(df):

    """
    Extract useful temporal features from START_TIME.
    """

    df = df.copy()

    df["YEAR"] = df["START_TIME"].dt.year

    df["MONTH"] = df["START_TIME"].dt.month

    df["DAY"] = df["START_TIME"].dt.day

    df["HOUR"] = df["START_TIME"].dt.hour

    df["MINUTE"] = df["START_TIME"].dt.minute

    df["WEEKDAY"] = df["START_TIME"].dt.day_name()

    df["DAY_OF_WEEK"] = df["START_TIME"].dt.dayofweek

    df["DAY_OF_YEAR"] = df["START_TIME"].dt.dayofyear

    df["WEEK"] = df["START_TIME"].dt.isocalendar().week.astype(int)

    df["QUARTER"] = df["START_TIME"].dt.quarter

    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([5, 6]).astype(int)

    return df


# -------------------------------------------------------
# Create Time Block Feature
# -------------------------------------------------------

def create_time_block(df):

    """
    Create Time Block (1-96) from Hour and Minute.
    """

    df = df.copy()

    df["TIME_BLOCK"] = (

        df["HOUR"] * 4

        +

        (df["MINUTE"] // 15)

        +

        1

    )

    return df


# -------------------------------------------------------
# Peak Hour Feature
# -------------------------------------------------------

def create_peak_hour(df):

    """
    Peak Hours:
    06:00 - 10:00
    18:00 - 22:00
    """

    df = df.copy()

    df["PEAK_HOUR"] = np.where(

        (

            (df["HOUR"] >= 6)

            &

            (df["HOUR"] <= 10)

        )

        |

        (

            (df["HOUR"] >= 18)

            &

            (df["HOUR"] <= 22)

        ),

        1,

        0

    )

    return df


# -------------------------------------------------------
# Season Feature
# -------------------------------------------------------

def create_season(df):

    """
    Create Season Column
    """

    df = df.copy()

    season_map = {

        12: "Winter",
        1: "Winter",
        2: "Winter",

        3: "Summer",
        4: "Summer",
        5: "Summer",

        6: "Monsoon",
        7: "Monsoon",
        8: "Monsoon",
        9: "Monsoon",

        10: "Post-Monsoon",
        11: "Post-Monsoon"

    }

    df["SEASON"] = df["MONTH"].map(season_map)

    return df


# -------------------------------------------------------
# Encode Weekday
# -------------------------------------------------------

def encode_weekday(df):

    """
    Convert weekday names to numerical labels.
    """

    df = df.copy()

    mapping = {

        "Monday":0,

        "Tuesday":1,

        "Wednesday":2,

        "Thursday":3,

        "Friday":4,

        "Saturday":5,

        "Sunday":6

    }

    df["WEEKDAY_CODE"] = df["WEEKDAY"].map(mapping)

    return df


# -------------------------------------------------------
# Feature Summary
# -------------------------------------------------------

def feature_summary(df):

    features = [

        "YEAR",

        "MONTH",

        "DAY",

        "HOUR",

        "MINUTE",

        "WEEKDAY",

        "DAY_OF_WEEK",

        "DAY_OF_YEAR",

        "WEEK",

        "QUARTER",

        "IS_WEEKEND",

        "TIME_BLOCK",

        "PEAK_HOUR",

        "SEASON",

        "WEEKDAY_CODE"

    ]

    return df[features].head()


# -------------------------------------------------------
# Final Feature Pipeline
# -------------------------------------------------------

def feature_engineering_pipeline(df):

    df = extract_datetime_features(df)

    df = create_time_block(df)

    df = create_peak_hour(df)

    df = create_season(df)

    df = encode_weekday(df)

    return df