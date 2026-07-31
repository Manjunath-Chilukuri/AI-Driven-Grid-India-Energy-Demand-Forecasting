"""
===========================================================
Prediction Engine
AI-Driven Grid India Level Energy Demand Forecasting
===========================================================

Supports

1. Linear Regression
2. Random Forest
3. XGBoost
4. Artificial Neural Network
5. LSTM
6. GRU
7. BiLSTM

Author : M.Tech Project
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime
warnings.filterwarnings("ignore")


# ==========================================================
# Project Paths
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

MODEL_DIR = os.path.join(PROJECT_DIR, "models")
DATA_DIR = os.path.join(PROJECT_DIR, "data")


# ==========================================================
# Dataset Path
# ==========================================================

DATASET_PATH = os.path.join(DATA_DIR, "data.xlsx")


# ==========================================================
# Classical ML Models
# ==========================================================

LINEAR_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "LinearRegression.pkl"
)

RF_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "RandomForest.pkl"
)

XGB_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "XGBoost.pkl"
)


# ==========================================================
# Deep Learning Models
# ==========================================================
ANN_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ANN_Model.h5"
)

LSTM_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "Best_LSTM_Model.h5"
)

GRU_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "Best_GRU_Model.h5"
)

BILSTM_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "Best_BiLSTM_Model.h5"
)
# ==========================================================
# Scaler
# ==========================================================

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)


# ==========================================================
# Feature Lists
# ==========================================================

ML_FEATURES = [

    "YEAR",
    "MONTH",
    "DAY",
    "Hour",
    "Minute",
    "Weekday",
    "TIMEBLOCK",
    "Lag_1",
    "Lag_2",
    "Lag_4",
    "Lag_96",
    "Rolling_4",
    "Rolling_96"

]

ANN_FEATURES = [

    "YEAR",
    "MONTH",
    "DAY",
    "TIMEBLOCK",
    "HOUR",
    "MINUTE",
    "WEEKDAY"

]


SEQUENCE_LENGTH = 96


# ==========================================================
# Model Loading Functions
# ==========================================================

def load_pickle_model(path):

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    return joblib.load(path)


def load_dl_model(path):

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    return load_model(path, compile=False)
# ==========================================================
# Load All Models
# ==========================================================

linear_model = load_pickle_model(LINEAR_MODEL_PATH)

random_forest_model = load_pickle_model(RF_MODEL_PATH)

xgboost_model = load_pickle_model(XGB_MODEL_PATH)

ann_model = load_dl_model(ANN_MODEL_PATH)

lstm_model = load_dl_model(LSTM_MODEL_PATH)

gru_model = load_dl_model(GRU_MODEL_PATH)

bilstm_model = load_dl_model(BILSTM_MODEL_PATH)

scaler = load_pickle_model(SCALER_PATH)


# ==========================================================
# Read Dataset
# ==========================================================
def load_dataset():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(DATASET_PATH)

    df = pd.read_excel(
        DATASET_PATH,
        skiprows=1
    )

    print("Columns BEFORE rename:")
    print(df.columns.tolist())

    print("Shape:", df.shape)

    df.columns = [
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

    print("Columns AFTER rename:")
    print(df.columns.tolist())

    return df
# ==========================================================
# Basic Time Utilities
# ==========================================================

def get_hour_minute(timeblock):

    """
    Converts

    TimeBlock 1

    →

    00:00

    TimeBlock 96

    →

    23:45
    """

    minutes = (timeblock - 1) * 15

    hour = minutes // 60

    minute = minutes % 60

    return hour, minute


def get_weekday(date_value):

    """
    Monday = 0
    Sunday = 6
    """

    return pd.Timestamp(date_value).weekday()


def validate_timeblock(timeblock):

    if timeblock < 1 or timeblock > 96:

        raise ValueError(
            "TIMEBLOCK should be between 1 and 96"
        )


def validate_date(date_value):

    try:

        pd.Timestamp(date_value)

    except Exception:

        raise ValueError(
            "Invalid Date Selected"
        )
# ==========================================================
# DATA PREPARATION
# ==========================================================

def prepare_dataset():
    """
    Prepare historical dataset and create lag features
    required by ML models.
    """

    df = load_dataset().copy()

    # -----------------------------
    # Date & Time
    # -----------------------------

    df["START_TIME"] = pd.to_datetime(df["START_TIME"])

    df["YEAR"] = df["START_TIME"].dt.year
    df["MONTH"] = df["START_TIME"].dt.month
    df["DAY"] = df["START_TIME"].dt.day

    df["Hour"] = df["START_TIME"].dt.hour
    df["Minute"] = df["START_TIME"].dt.minute

    df["Weekday"] = df["START_TIME"].dt.weekday

    # -----------------------------
    # Lag Features
    # -----------------------------

    df["Lag_1"] = df["DEMAND_MET_MW"].shift(1)

    df["Lag_2"] = df["DEMAND_MET_MW"].shift(2)

    df["Lag_4"] = df["DEMAND_MET_MW"].shift(4)

    df["Lag_96"] = df["DEMAND_MET_MW"].shift(96)

    # -----------------------------
    # Rolling Features
    # -----------------------------

    df["Rolling_4"] = (
        df["DEMAND_MET_MW"]
        .rolling(4)
        .mean()
    )

    df["Rolling_96"] = (
        df["DEMAND_MET_MW"]
        .rolling(96)
        .mean()
    )

    df = df.dropna().reset_index(drop=True)

    return df


# ==========================================================
# Historical Dataset
# ==========================================================

history_df = prepare_dataset()
# ==========================================================
# Latest Historical Values
# ==========================================================

def latest_history():

    """
    Returns latest processed row.
    """

    return history_df.iloc[-1]


def previous_demand(n=1):

    """
    Previous demand values.
    """

    return history_df["DEMAND_MET_MW"].iloc[-n]


def rolling_average(window):

    """
    Rolling average from historical data.
    """

    return history_df["DEMAND_MET_MW"].tail(window).mean()

# ==========================================================
# Feature Vector for ML Models
# ==========================================================

def create_ml_features(date_value, timeblock):

    """
    Creates the exact 13 features used
    during ML model training.
    """

    validate_date(date_value)
    validate_timeblock(timeblock)

    hour, minute = get_hour_minute(timeblock)

    feature_dict = {

        "YEAR": pd.Timestamp(date_value).year,

        "MONTH": pd.Timestamp(date_value).month,

        "DAY": pd.Timestamp(date_value).day,

        "Hour": hour,

        "Minute": minute,

        "Weekday": get_weekday(date_value),

        "TIMEBLOCK": timeblock,

        "Lag_1": previous_demand(1),

        "Lag_2": previous_demand(2),

        "Lag_4": previous_demand(4),

        "Lag_96": previous_demand(96),

        "Rolling_4": rolling_average(4),

        "Rolling_96": rolling_average(96)

    }

    feature_df = pd.DataFrame([feature_dict])

    feature_df = feature_df[ML_FEATURES]

    return feature_df

# ==========================================================
# ANN Features
# ==========================================================

def create_ann_features(date_value, timeblock):

    """
    Creates ANN input.
    """

    validate_date(date_value)

    validate_timeblock(timeblock)

    hour, minute = get_hour_minute(timeblock)

    features = pd.DataFrame({

        "YEAR":[pd.Timestamp(date_value).year],

        "MONTH":[pd.Timestamp(date_value).month],

        "DAY":[pd.Timestamp(date_value).day],

        "TIMEBLOCK":[timeblock],

        "HOUR":[hour],

        "MINUTE":[minute],

        "WEEKDAY":[get_weekday(date_value)]

    })

    return features
# ==========================================================
# Sequence Generation
# ==========================================================

def create_sequence():

    """
    Returns latest 96 demand values.

    Shape

    (1,96,1)
    """

    sequence = history_df["DEMAND_MET_MW"].tail(
        SEQUENCE_LENGTH
    ).values.reshape(-1,1)

    sequence = scaler.transform(sequence)

    sequence = sequence.reshape(
        1,
        SEQUENCE_LENGTH,
        1
    )

    return sequence

# ==========================================================
# LINEAR REGRESSION PREDICTION
# ==========================================================

def predict_linear_regression(date_value, timeblock):
    """
    Predict demand using Linear Regression.
    """

    features = create_ml_features(date_value, timeblock)

    prediction = linear_model.predict(features)

    return float(prediction[0])


# ==========================================================
# RANDOM FOREST PREDICTION
# ==========================================================

def predict_random_forest(date_value, timeblock):
    """
    Predict demand using Random Forest.
    """

    features = create_ml_features(date_value, timeblock)

    prediction = random_forest_model.predict(features)

    return float(prediction[0])


# ==========================================================
# XGBOOST PREDICTION
# ==========================================================

def predict_xgboost(date_value, timeblock):
    """
    Predict demand using XGBoost.
    """

    features = create_ml_features(date_value, timeblock)

    prediction = xgboost_model.predict(features)

    return float(prediction[0])

# ==========================================================
# GENERIC ML PREDICTION
# ==========================================================

def predict_ml(model_name, date_value, timeblock):
    """
    Generic prediction function
    for all classical ML models.
    """

    model_name = model_name.lower()

    if model_name == "linear regression":
        return predict_linear_regression(date_value, timeblock)

    elif model_name == "random forest":
        return predict_random_forest(date_value, timeblock)

    elif model_name == "xgboost":
        return predict_xgboost(date_value, timeblock)

    else:
        raise ValueError(
            f"Unknown ML Model : {model_name}"
        )

# ==========================================================
# DAILY FORECAST USING ML
# ==========================================================

def forecast_ml_day(model_name, date_value):
    """
    Forecast all 96 time blocks
    using a selected ML model.
    """

    predictions = []

    for block in range(1, 97):

        value = predict_ml(
            model_name=model_name,
            date_value=date_value,
            timeblock=block
        )

        hour, minute = get_hour_minute(block)

        predictions.append({

            "Date": pd.Timestamp(date_value).date(),

            "Time Block": block,

            "Time": f"{hour:02d}:{minute:02d}",

            "Predicted Demand (MW)": round(value, 2)

        })

    return pd.DataFrame(predictions)

# ==========================================================
# FORECAST SUMMARY
# ==========================================================

def forecast_summary(df):
    """
    Summary statistics for
    predicted demand.
    """

    return {

        "Maximum Demand (MW)":
            round(df["Predicted Demand (MW)"].max(), 2),

        "Minimum Demand (MW)":
            round(df["Predicted Demand (MW)"].min(), 2),

        "Average Demand (MW)":
            round(df["Predicted Demand (MW)"].mean(), 2),

        "Total Energy":
            round(df["Predicted Demand (MW)"].sum(), 2)

    }

# ==========================================================
# AVAILABLE MODELS
# ==========================================================

AVAILABLE_MODELS = [

    "Linear Regression",

    "Random Forest",

    "XGBoost",

    "ANN",

    "LSTM",

    "GRU",

    "BiLSTM"

]

# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    today = datetime.today()

    result = predict_linear_regression(today,1)

    print("Linear Regression Prediction")

    print(result)
# ==========================================================
# ANN PREDICTION
# ==========================================================

def predict_ann(date_value, timeblock):
    """
    Predict demand using Artificial Neural Network.
    """

    features = create_ann_features(date_value, timeblock)

    prediction = ann_model.predict(
        features.values,
        verbose=0
    )

    return float(prediction[0][0])


# ==========================================================
# LSTM PREDICTION
# ==========================================================

def predict_lstm():

    """
    Predict demand using LSTM.
    """

    sequence = create_sequence()

    prediction = lstm_model.predict(
        sequence,
        verbose=0
    )

    prediction = scaler.inverse_transform(
        prediction
    )

    return float(prediction[0][0])


# ==========================================================
# GRU PREDICTION
# ==========================================================

def predict_gru():

    """
    Predict demand using GRU.
    """

    sequence = create_sequence()

    prediction = gru_model.predict(
        sequence,
        verbose=0
    )

    prediction = scaler.inverse_transform(
        prediction
    )

    return float(prediction[0][0])


# ==========================================================
# BiLSTM PREDICTION
# ==========================================================

def predict_bilstm():

    """
    Predict demand using Bidirectional LSTM.
    """

    sequence = create_sequence()

    prediction = bilstm_model.predict(
        sequence,
        verbose=0
    )

    prediction = scaler.inverse_transform(
        prediction
    )

    return float(prediction[0][0])

# ==========================================================
# DEEP LEARNING ROUTER
# ==========================================================

def predict_dl(model_name,
               date_value,
               timeblock):
    """
    Common interface for all
    Deep Learning models.
    """

    model_name = model_name.lower()

    if model_name == "ann":

        return predict_ann(
            date_value,
            timeblock
        )

    elif model_name == "lstm":

        return predict_lstm()

    elif model_name == "gru":

        return predict_gru()

    elif model_name == "bilstm":

        return predict_bilstm()

    else:

        raise ValueError(
            "Invalid Deep Learning Model"
        )
    
# ==========================================================
# FORECAST USING DEEP LEARNING
# ==========================================================

def forecast_dl_day(model_name,
                    date_value):
    """
    Forecast all 96 blocks using
    Deep Learning models.
    """

    predictions = []

    for block in range(1,97):

        hour, minute = get_hour_minute(block)

        value = predict_dl(
            model_name,
            date_value,
            block
        )

        predictions.append({

            "Date":
            pd.Timestamp(date_value).date(),

            "Time Block":
            block,

            "Time":
            f"{hour:02d}:{minute:02d}",

            "Predicted Demand (MW)":
            round(value,2)

        })

    return pd.DataFrame(predictions)

# ==========================================================
# MODEL CATEGORY
# ==========================================================

def is_ml_model(model_name):

    return model_name.lower() in [

        "linear regression",

        "random forest",

        "xgboost"

    ]


def is_dl_model(model_name):

    return model_name.lower() in [

        "ann",

        "lstm",

        "gru",

        "bilstm"

    ]

# ==========================================================
# SINGLE PREDICTION
# ==========================================================

def predict(model_name,
            date_value,
            timeblock):
    """
    Single prediction interface.
    """

    if is_ml_model(model_name):

        return predict_ml(
            model_name,
            date_value,
            timeblock
        )

    elif is_dl_model(model_name):

        return predict_dl(
            model_name,
            date_value,
            timeblock
        )

    else:

        raise ValueError(
            "Unknown Model Selected"
        )

    
# ==========================================================
# DAILY FORECAST API
# ==========================================================

def forecast_day(model_name,
                 date_value):
    """
    Returns forecast DataFrame
    for selected model.
    """

    if is_ml_model(model_name):

        return forecast_ml_day(
            model_name,
            date_value
        )

    elif is_dl_model(model_name):

        return forecast_dl_day(
            model_name,
            date_value
        )

    else:

        raise ValueError(
            "Unknown Model"
        )

