"""
=========================================================
Training Engine
AI-Driven Grid India Level Energy Demand Forecasting
=========================================================

This module trains all Machine Learning and
Deep Learning models used in the project.

Models

1. Linear Regression
2. Random Forest
3. XGBoost
4. Artificial Neural Network
5. LSTM
6. GRU
7. Bi-LSTM
"""

# ==========================================================
# Imports
# ==========================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LSTM,
    GRU,
    Bidirectional
)

from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

# ==========================================================
# Project Directories
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================================
# Dataset Path
# ==========================================================

DATA_PATH = os.path.join(
    DATA_DIR,
    "data.xlsx"
)

# ==========================================================
# Output Report
# ==========================================================

REPORT_PATH = os.path.join(
    REPORT_DIR,
    "Model_Performance.csv"
)

# ==========================================================
# Model Paths
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

ANN_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ANN.keras"
)

LSTM_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "LSTM.keras"
)

GRU_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "GRU.keras"
)

BILSTM_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "BiLSTM.keras"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)
# ==========================================================
# ML Features
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

# ==========================================================
# ANN Features
# ==========================================================

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
# Load Dataset
# ==========================================================

def load_dataset():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_excel(
        DATA_PATH,
        skiprows=1
    )

    return df

# ==========================================================
# Prepare Dataset
# ==========================================================

def prepare_dataset():

    """
    Prepare complete dataset
    for ML and DL models.
    """

    df = load_dataset().copy()

    # ------------------------------------------
    # Rename Columns
    # ------------------------------------------

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

    # ------------------------------------------
    # Datetime
    # ------------------------------------------

    df["START_TIME"] = pd.to_datetime(
        df["START_TIME"]
    )

    # ------------------------------------------
    # Time Features
    # ------------------------------------------

    df["YEAR"] = df["START_TIME"].dt.year

    df["MONTH"] = df["START_TIME"].dt.month

    df["DAY"] = df["START_TIME"].dt.day

    df["Hour"] = df["START_TIME"].dt.hour

    df["Minute"] = df["START_TIME"].dt.minute

    df["Weekday"] = df["START_TIME"].dt.weekday

    # ANN naming

    df["HOUR"] = df["Hour"]

    df["MINUTE"] = df["Minute"]

    df["WEEKDAY"] = df["Weekday"]

    # ------------------------------------------
    # Lag Features
    # ------------------------------------------

    df["Lag_1"] = df["DEMAND_MET_MW"].shift(1)

    df["Lag_2"] = df["DEMAND_MET_MW"].shift(2)

    df["Lag_4"] = df["DEMAND_MET_MW"].shift(4)

    df["Lag_96"] = df["DEMAND_MET_MW"].shift(96)

    # ------------------------------------------
    # Rolling Features
    # ------------------------------------------

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

    df = df.dropna()

    df = df.reset_index(drop=True)

    return df

# ==========================================================
# ML Dataset
# ==========================================================

def create_ml_dataset(df):

    X = df[ML_FEATURES]

    y = df["DEMAND_MET_MW"]

    return train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        shuffle=False

    )

# ==========================================================
# ANN Dataset
# ==========================================================

def create_ann_dataset(df):

    X = df[ANN_FEATURES].values

    y = df["DEMAND_MET_MW"].values

    scaler = MinMaxScaler()

    y_scaled = scaler.fit_transform(

        y.reshape(-1,1)

    )

    joblib.dump(

        scaler,

        SCALER_PATH

    )

    X_train,X_test,y_train,y_test = train_test_split(

        X,

        y_scaled,

        test_size=0.20,

        shuffle=False

    )

    return (

        X_train,

        X_test,

        y_train,

        y_test,

        scaler

    )

# ==========================================================
# Sequence Dataset
# ==========================================================

def create_sequence_dataset(df):

    values = df["DEMAND_MET_MW"].values.reshape(-1,1)

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(values)

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    X = []

    y = []

    for i in range(

        SEQUENCE_LENGTH,

        len(scaled)

    ):

        X.append(

            scaled[
                i-SEQUENCE_LENGTH:i
            ]
        )

        y.append(

            scaled[i]
        )

    X = np.array(X)

    y = np.array(y)

    split = int(

        len(X)*0.80

    )

    X_train = X[:split]

    X_test = X[split:]

    y_train = y[:split]

    y_test = y[split:]

    return (

        X_train,

        X_test,

        y_train,

        y_test,

        scaler

    )

# ==========================================================
# Evaluation Metrics
# ==========================================================

def evaluate_model(

    y_true,

    y_pred

):

    mae = mean_absolute_error(

        y_true,

        y_pred

    )

    rmse = np.sqrt(

        mean_squared_error(

            y_true,

            y_pred

        )

    )

    r2 = r2_score(

        y_true,

        y_pred

    )

    return mae, rmse, r2

# ==========================================================
# Results Container
# ==========================================================

performance_results = []

# ==========================================================
# TRAIN LINEAR REGRESSION
# ==========================================================

def train_linear_regression():

    print("\nTraining Linear Regression...")

    X_train, X_test, y_train, y_test = create_ml_dataset(
        prepare_dataset()
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae, rmse, r2 = evaluate_model(
        y_test,
        predictions
    )

    joblib.dump(
        model,
        LINEAR_MODEL_PATH
    )

    performance_results.append({

        "Model": "Linear Regression",

        "MAE": round(mae, 4),

        "RMSE": round(rmse, 4),

        "R²": round(r2, 4)

    })

    print("Linear Regression Completed")

    return model


# ==========================================================
# TRAIN RANDOM FOREST
# ==========================================================

def train_random_forest():

    print("\nTraining Random Forest...")

    X_train, X_test, y_train, y_test = create_ml_dataset(
        prepare_dataset()
    )

    model = RandomForestRegressor(

        n_estimators=200,

        max_depth=20,

        min_samples_split=2,

        min_samples_leaf=1,

        random_state=42,

        n_jobs=-1

    )

    model.fit(

        X_train,

        y_train

    )

    predictions = model.predict(

        X_test

    )

    mae, rmse, r2 = evaluate_model(

        y_test,

        predictions

    )

    joblib.dump(

        model,

        RF_MODEL_PATH

    )

    performance_results.append({

        "Model": "Random Forest",

        "MAE": round(mae, 4),

        "RMSE": round(rmse, 4),

        "R²": round(r2, 4)

    })

    print("Random Forest Completed")

    return model


# ==========================================================
# TRAIN XGBOOST
# ==========================================================

def train_xgboost():

    print("\nTraining XGBoost...")

    X_train, X_test, y_train, y_test = create_ml_dataset(
        prepare_dataset()
    )

    model = XGBRegressor(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=8,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        random_state=42

    )

    model.fit(

        X_train,

        y_train

    )

    predictions = model.predict(

        X_test

    )

    mae, rmse, r2 = evaluate_model(

        y_test,

        predictions

    )

    joblib.dump(

        model,

        XGB_MODEL_PATH

    )

    performance_results.append({

        "Model": "XGBoost",

        "MAE": round(mae, 4),

        "RMSE": round(rmse, 4),

        "R²": round(r2, 4)

    })

    print("XGBoost Completed")

    return model


# ==========================================================
# TRAIN ALL MACHINE LEARNING MODELS
# ==========================================================

def train_ml_models():

    """
    Train all classical Machine Learning models.
    """

    print("\n======================================")
    print(" MACHINE LEARNING MODEL TRAINING")
    print("======================================")

    linear_model = train_linear_regression()

    random_forest_model = train_random_forest()

    xgboost_model = train_xgboost()

    print("\nMachine Learning Training Completed.")

    return {

        "Linear Regression": linear_model,

        "Random Forest": random_forest_model,

        "XGBoost": xgboost_model

    }

# ==========================================================
# ARTIFICIAL NEURAL NETWORK (ANN)
# ==========================================================

def train_ann():

    print("=" * 60)
    print("Training Artificial Neural Network...")
    print("=" * 60)

    df = prepare_dataset()

    X_train, X_test, y_train, y_test, scaler = create_ann_dataset(df)

    ann = Sequential()

    ann.add(Dense(
        128,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ))

    ann.add(Dense(
        64,
        activation="relu"
    ))

    ann.add(Dense(
        32,
        activation="relu"
    ))

    ann.add(Dense(
        1,
        activation="linear"
    ))

    ann.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = ann.fit(

        X_train,
        y_train,

        validation_split=0.20,

        epochs=100,

        batch_size=32,

        callbacks=[early_stop],

        verbose=1

    )

    predictions_scaled = ann.predict(X_test)

    predictions = scaler.inverse_transform(predictions_scaled)

    actual = scaler.inverse_transform(y_test.reshape(-1,1))

    mae, rmse, r2 = evaluate_model(actual, predictions)

    ann.save(ANN_MODEL_PATH)

    performance_results.append({

        "Model":"ANN",

        "MAE":round(mae,4),

        "RMSE":round(rmse,4),

        "R²":round(r2,4)

    })

    print()

    print("ANN Training Completed")

    print(f"MAE  : {mae:.4f}")

    print(f"RMSE : {rmse:.4f}")

    print(f"R²   : {r2:.4f}")

    print()

    return ann

# ==========================================================
# LSTM MODEL
# ==========================================================

def train_lstm():

    print("=" * 60)
    print("Training LSTM Model...")
    print("=" * 60)

    df = prepare_dataset()

    X_train, X_test, y_train, y_test, scaler = create_sequence_dataset(df)

    model = Sequential()

    model.add(
        LSTM(
            128,
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )

    model.add(Dropout(0.2))

    model.add(
        LSTM(
            64,
            return_sequences=False
        )
    )

    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    pred_scaled = model.predict(X_test)

    prediction = scaler.inverse_transform(pred_scaled)

    actual = scaler.inverse_transform(y_test.reshape(-1,1))

    mae, rmse, r2 = evaluate_model(actual, prediction)

    model.save(LSTM_MODEL_PATH)

    performance_results.append({

        "Model":"LSTM",

        "MAE":round(mae,4),

        "RMSE":round(rmse),

        "R²":round(r2,4)

    })

    print("LSTM Completed")

    print("MAE :", mae)

    print("RMSE:", rmse)

    print("R²  :", r2)

    return model

# ==========================================================
# GRU MODEL
# ==========================================================

def train_gru():

    print("=" * 60)
    print("Training GRU Model...")
    print("=" * 60)

    df = prepare_dataset()

    X_train, X_test, y_train, y_test, scaler = create_sequence_dataset(df)

    model = Sequential()

    model.add(
        GRU(
            128,
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )

    model.add(Dropout(0.2))

    model.add(
        GRU(
            64,
            return_sequences=False
        )
    )

    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    pred_scaled = model.predict(X_test)

    prediction = scaler.inverse_transform(pred_scaled)

    actual = scaler.inverse_transform(y_test.reshape(-1,1))

    mae, rmse, r2 = evaluate_model(actual, prediction)

    model.save(GRU_MODEL_PATH)

    performance_results.append({

        "Model":"GRU",

        "MAE":round(mae,4),

        "RMSE":round(rmse),

        "R²":round(r2,4)

    })

    print("GRU Completed")

    print("MAE :", mae)

    print("RMSE:", rmse)

    print("R²  :", r2)

    return model

# ==========================================================
# Bi-LSTM MODEL
# ==========================================================

def train_bilstm():

    print("=" * 60)
    print("Training Bidirectional LSTM...")
    print("=" * 60)

    df = prepare_dataset()

    X_train, X_test, y_train, y_test, scaler = create_sequence_dataset(df)

    model = Sequential()

    model.add(

        Bidirectional(

            LSTM(

                128,

                return_sequences=True

            ),

            input_shape=(X_train.shape[1], X_train.shape[2])

        )

    )

    model.add(Dropout(0.2))

    model.add(

        Bidirectional(

            LSTM(

                64,

                return_sequences=False

            )

        )

    )

    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))

    model.add(Dense(1))

    model.compile(

        optimizer="adam",

        loss="mse",

        metrics=["mae"]

    )

    early_stop = EarlyStopping(

        monitor="val_loss",

        patience=10,

        restore_best_weights=True

    )

    model.fit(

        X_train,

        y_train,

        validation_split=0.20,

        epochs=100,

        batch_size=32,

        callbacks=[early_stop],

        verbose=1

    )

    pred_scaled = model.predict(X_test)

    prediction = scaler.inverse_transform(pred_scaled)

    actual = scaler.inverse_transform(y_test.reshape(-1,1))

    mae, rmse, r2 = evaluate_model(actual, prediction)

    model.save(BILSTM_MODEL_PATH)

    performance_results.append({

        "Model":"BiLSTM",

        "MAE":round(mae,4),

        "RMSE":round(rmse,4),

        "R²":round(r2,4)

    })

    print("BiLSTM Completed")

    print("MAE :", mae)

    print("RMSE:", rmse)

    print("R²  :", r2)

    return model

# ==========================================================
# TRAIN ALL DL MODELS
# ==========================================================

def train_dl_models():

    ann = train_ann()

    lstm = train_lstm()

    gru = train_gru()

    bilstm = train_bilstm()

    return {

        "ANN": ann,

        "LSTM": lstm,

        "GRU": gru,

        "BiLSTM": bilstm

    }

# ==========================================================
# GENERATE MODEL PERFORMANCE REPORT
# ==========================================================

def save_performance_report():

    print("=" * 60)
    print("Generating Model Performance Report...")
    print("=" * 60)

    report = pd.DataFrame(performance_results)

    report = report.sort_values(
        by="R²",
        ascending=False
    ).reset_index(drop=True)

    report_path = os.path.join(
        REPORT_DIR,
        "Model_Performance.csv"
    )

    report.to_csv(
        report_path,
        index=False
    )

    print()

    print("Performance Report Saved")

    print(report_path)

    print()

    print(report)

    return report

# ==========================================================
# BEST MODEL
# ==========================================================

def display_best_model(report):

    best = report.iloc[0]

    print("=" * 60)

    print("BEST MODEL")

    print("=" * 60)

    print(f"Model : {best['Model']}")

    print(f"R²    : {best['R²']:.4f}")

    print(f"RMSE  : {best['RMSE']:.4f}")

    print(f"MAE   : {best['MAE']:.4f}")

    print("=" * 60)

    return best

# ==========================================================
# TRAIN ALL MODELS
# ==========================================================

def train_all_models():

    print()

    print("=" * 70)

    print("AI DRIVEN GRID INDIA ENERGY DEMAND FORECASTING")

    print("=" * 70)

    print()

    performance_results.clear()

    print("STEP 1 : MACHINE LEARNING MODELS")

    train_ml_models()

    print()

    print("STEP 2 : DEEP LEARNING MODELS")

    train_dl_models()

    print()

    print("STEP 3 : GENERATING REPORT")

    report = save_performance_report()

    print()

    print("STEP 4 : BEST MODEL")

    best = display_best_model(report)

    print()

    print("=" * 70)

    print("ALL MODELS TRAINED SUCCESSFULLY")

    print("=" * 70)

    return report


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    train_all_models()

