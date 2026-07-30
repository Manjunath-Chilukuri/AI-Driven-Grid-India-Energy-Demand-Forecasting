"""
=========================================================
Model Comparison Dashboard
AI-Driven Grid India Level Energy Demand Forecasting
=========================================================
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Model Comparison",
    page_icon="🏆",
    layout="wide"
)

st.title("🏆 Model Comparison Dashboard")

st.markdown("""
Compare the performance of all Machine Learning and
Deep Learning models using standard regression metrics.
""")

st.divider()

# ----------------------------------------------------
# Load Report
# ----------------------------------------------------

report_path = "outputs/reports/Model_Performance.csv"

if not os.path.exists(report_path):

    st.error("Model_Performance.csv not found.")

    st.stop()

report = pd.read_csv(report_path)

# ----------------------------------------------------
# Ranking
# ----------------------------------------------------

ranking = report.sort_values(
    "R²",
    ascending=False
).reset_index(drop=True)

ranking.index = ranking.index + 1

st.subheader("🏅 Model Ranking")

st.dataframe(
    ranking,
    use_container_width=True
)

st.divider()

# ----------------------------------------------------
# Best Model
# ----------------------------------------------------

best = ranking.iloc[0]

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Best Model",
    best["Model"]
)

c2.metric(
    "R²",
    f"{best['R²']:.4f}"
)

c3.metric(
    "RMSE",
    f"{best['RMSE']:.2f}"
)

c4.metric(
    "MAE",
    f"{best['MAE']:.2f}"
)

st.divider()

# ----------------------------------------------------
# R² Comparison
# ----------------------------------------------------

st.subheader("R² Score Comparison")

fig = px.bar(
    ranking,
    x="Model",
    y="R²",
    color="R²",
    text="R²"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ----------------------------------------------------
# RMSE Comparison
# ----------------------------------------------------

st.subheader("RMSE Comparison")

fig = px.bar(
    ranking,
    x="Model",
    y="RMSE",
    color="RMSE"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ----------------------------------------------------
# MAE Comparison
# ----------------------------------------------------

st.subheader("MAE Comparison")

fig = px.bar(
    ranking,
    x="Model",
    y="MAE",
    color="MAE"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ----------------------------------------------------
# Radar Chart
# ----------------------------------------------------

st.subheader("Performance Radar Chart")

radar = report.copy()

radar["R²"] = radar["R²"] / radar["R²"].max()

radar["RMSE"] = 1 - radar["RMSE"] / radar["RMSE"].max()

radar["MAE"] = 1 - radar["MAE"] / radar["MAE"].max()

fig = go.Figure()

for _, row in radar.iterrows():

    fig.add_trace(

        go.Scatterpolar(

            r=[
                row["R²"],
                row["RMSE"],
                row["MAE"]
            ],

            theta=[
                "R²",
                "RMSE",
                "MAE"
            ],

            fill="toself",

            name=row["Model"]

        )

    )

fig.update_layout(

    polar=dict(

        radialaxis=dict(

            visible=True,

            range=[0,1]

        )

    ),

    showlegend=True

)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ----------------------------------------------------
# Performance Table
# ----------------------------------------------------

st.subheader("Detailed Performance")

st.dataframe(
    ranking,
    use_container_width=True
)

st.divider()

# ----------------------------------------------------
# Strengths
# ----------------------------------------------------

strength = {

"Linear Regression":"Simple baseline model.",

"Random Forest":"Captures nonlinear relationships.",

"XGBoost":"Excellent boosting performance.",

"ANN":"Learns complex nonlinear mappings.",

"LSTM":"Captures long-term temporal dependencies.",

"GRU":"Efficient sequential learning.",

"BiLSTM":"Learns both past and future temporal context."

}

rows=[]

for _,row in ranking.iterrows():

    rows.append({

        "Model":row["Model"],

        "Strength":strength[row["Model"]]

    })

st.subheader("Model Strengths")

st.dataframe(

    pd.DataFrame(rows),

    use_container_width=True

)

st.divider()

# ----------------------------------------------------
# Conclusion
# ----------------------------------------------------

st.success(f"""

### Best Model

🏆 {best['Model']}

This model achieved the highest prediction accuracy among all
implemented Machine Learning and Deep Learning algorithms
and is recommended for electricity demand forecasting.

""")