"""
=========================================================
Visualization Utility
AI-Driven Grid India Level Energy Demand Forecasting
=========================================================
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# --------------------------------------------------------
# Demand Trend
# --------------------------------------------------------

def demand_trend(df):

    fig = px.line(
        df,
        x="START_TIME",
        y="DEMAND_MET_MW",
        title="Electricity Demand Trend",
        labels={
            "START_TIME": "Time",
            "DEMAND_MET_MW": "Demand (MW)"
        }
    )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    return fig


# --------------------------------------------------------
# Monthly Demand
# --------------------------------------------------------

def monthly_demand(df):

    monthly = df.groupby("MONTH")["DEMAND_MET_MW"].mean().reset_index()

    fig = px.bar(
        monthly,
        x="MONTH",
        y="DEMAND_MET_MW",
        title="Average Monthly Demand"
    )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    return fig


# --------------------------------------------------------
# Hourly Demand
# --------------------------------------------------------

def hourly_demand(df):

    hourly = df.groupby("HOUR")["DEMAND_MET_MW"].mean().reset_index()

    fig = px.line(
        hourly,
        x="HOUR",
        y="DEMAND_MET_MW",
        markers=True,
        title="Average Hourly Demand"
    )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    return fig


# --------------------------------------------------------
# Weekday Demand
# --------------------------------------------------------

def weekday_demand(df):

    weekday = df.groupby("WEEKDAY")["DEMAND_MET_MW"].mean().reset_index()

    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekday["WEEKDAY"] = pd.Categorical(
        weekday["WEEKDAY"],
        categories=order,
        ordered=True
    )

    weekday = weekday.sort_values("WEEKDAY")

    fig = px.bar(
        weekday,
        x="WEEKDAY",
        y="DEMAND_MET_MW",
        title="Average Weekday Demand"
    )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    return fig


# --------------------------------------------------------
# Demand Distribution
# --------------------------------------------------------

def demand_distribution(df):

    fig = px.histogram(
        df,
        x="DEMAND_MET_MW",
        nbins=50,
        title="Demand Distribution"
    )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    return fig


# --------------------------------------------------------
# Correlation Heatmap
# --------------------------------------------------------

def correlation_heatmap(df):

    numeric = df.select_dtypes(include="number")

    corr = numeric.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )

    fig.update_layout(
        height=700
    )

    return fig


# --------------------------------------------------------
# Generation Mix
# --------------------------------------------------------

def generation_mix(df):

    generation = {

        "Thermal": df["THERMAL_MW"].mean(),

        "Hydro": df["HYDRO_MW"].mean(),

        "Solar": df["SOLAR_MW"].mean(),

        "Wind": df["WIND_MW"].mean(),

        "Nuclear": df["NUCLEAR_MW"].mean(),

        "Gas": df["GAS_MW"].mean(),

        "Others": df["OTHERS_MW"].mean()

    }

    fig = px.pie(

        names=list(generation.keys()),

        values=list(generation.values()),

        title="Average Generation Mix"

    )

    fig.update_layout(height=550)

    return fig


# --------------------------------------------------------
# Peak Demand
# --------------------------------------------------------

def peak_demand(df):

    top = df.nlargest(20, "DEMAND_MET_MW")

    fig = px.bar(

        top,

        x="START_TIME",

        y="DEMAND_MET_MW",

        title="Top 20 Peak Demand Records"

    )

    fig.update_layout(

        template="plotly_white",

        height=500

    )

    return fig


# --------------------------------------------------------
# Daily Demand
# --------------------------------------------------------

def daily_demand(df):

    daily = df.groupby("DAY")["DEMAND_MET_MW"].mean().reset_index()

    fig = px.line(

        daily,

        x="DAY",

        y="DEMAND_MET_MW",

        markers=True,

        title="Average Daily Demand"

    )

    fig.update_layout(height=500)

    return fig


# --------------------------------------------------------
# Quarterly Demand
# --------------------------------------------------------

def quarterly_demand(df):

    quarter = df.groupby("QUARTER")["DEMAND_MET_MW"].mean().reset_index()

    fig = px.bar(

        quarter,

        x="QUARTER",

        y="DEMAND_MET_MW",

        title="Quarter-wise Demand"

    )

    return fig


# --------------------------------------------------------
# Weekend vs Weekday
# --------------------------------------------------------

def weekend_analysis(df):

    weekend = df.groupby("IS_WEEKEND")["DEMAND_MET_MW"].mean().reset_index()

    weekend["IS_WEEKEND"] = weekend["IS_WEEKEND"].replace({

        0: "Weekday",

        1: "Weekend"

    })

    fig = px.bar(

        weekend,

        x="IS_WEEKEND",

        y="DEMAND_MET_MW",

        title="Weekend vs Weekday Demand"

    )

    return fig


# --------------------------------------------------------
# Time Block Demand
# --------------------------------------------------------

def timeblock_demand(df):

    tb = df.groupby("TIME_BLOCK")["DEMAND_MET_MW"].mean().reset_index()

    fig = px.line(

        tb,

        x="TIME_BLOCK",

        y="DEMAND_MET_MW",

        title="Time Block Demand Pattern"

    )

    return fig


# --------------------------------------------------------
# Feature Importance Placeholder
# --------------------------------------------------------

def empty_chart(title):

    fig = go.Figure()

    fig.update_layout(

        title=title,

        height=500,

        template="plotly_white"

    )

    return fig