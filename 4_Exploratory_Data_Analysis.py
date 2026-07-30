"""
=========================================================
Exploratory Data Analysis
AI-Driven Grid India Level Energy Demand Forecasting
=========================================================
"""

import streamlit as st

from utils.data_loader import load_dataset
from utils.preprocessing import preprocessing_pipeline
from utils.feature_engineering import feature_engineering_pipeline

from utils.visualization import (
    demand_trend,
    monthly_demand,
    hourly_demand,
    daily_demand,
    weekday_demand,
    demand_distribution,
    correlation_heatmap,
    generation_mix,
    quarterly_demand,
    weekend_analysis,
    timeblock_demand,
    peak_demand
)

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Exploratory Data Analysis")

st.markdown("""
This module presents interactive visualizations to analyze
historical electricity demand patterns, generation trends,
and feature relationships.
""")

st.divider()

# -------------------------------------------------------
# Load Data
# -------------------------------------------------------

df = load_dataset()

df, _ = preprocessing_pipeline(df)

df = feature_engineering_pipeline(df)

# -------------------------------------------------------
# Dashboard Metrics
# -------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Records",
    f"{len(df):,}"
)

c2.metric(
    "Features",
    len(df.columns)
)

c3.metric(
    "Maximum Demand (MW)",
    f"{df['DEMAND_MET_MW'].max():,.0f}"
)

c4.metric(
    "Average Demand (MW)",
    f"{df['DEMAND_MET_MW'].mean():,.0f}"
)

st.divider()

# -------------------------------------------------------
# Demand Trend
# -------------------------------------------------------

st.subheader("1. Electricity Demand Trend")

st.plotly_chart(
    demand_trend(df),
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# Monthly & Hourly Demand
# -------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("2. Monthly Demand")

    st.plotly_chart(
        monthly_demand(df),
        use_container_width=True
    )

with col2:

    st.subheader("3. Hourly Demand")

    st.plotly_chart(
        hourly_demand(df),
        use_container_width=True
    )

st.divider()

# -------------------------------------------------------
# Daily & Weekday Demand
# -------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("4. Daily Demand")

    st.plotly_chart(
        daily_demand(df),
        use_container_width=True
    )

with col2:

    st.subheader("5. Weekday Demand")

    st.plotly_chart(
        weekday_demand(df),
        use_container_width=True
    )

st.divider()

# -------------------------------------------------------
# Distribution
# -------------------------------------------------------

st.subheader("6. Demand Distribution")

st.plotly_chart(
    demand_distribution(df),
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# Correlation
# -------------------------------------------------------

st.subheader("7. Correlation Heatmap")

st.plotly_chart(
    correlation_heatmap(df),
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# Generation Mix
# -------------------------------------------------------

st.subheader("8. Generation Mix")

st.plotly_chart(
    generation_mix(df),
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# Quarter & Weekend
# -------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("9. Quarter-wise Demand")

    st.plotly_chart(
        quarterly_demand(df),
        use_container_width=True
    )

with col2:

    st.subheader("10. Weekend vs Weekday")

    st.plotly_chart(
        weekend_analysis(df),
        use_container_width=True
    )

st.divider()

# -------------------------------------------------------
# Time Block Pattern
# -------------------------------------------------------

st.subheader("11. Time Block Demand Pattern")

st.plotly_chart(
    timeblock_demand(df),
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# Peak Demand
# -------------------------------------------------------

st.subheader("12. Top 20 Peak Demand Records")

st.plotly_chart(
    peak_demand(df),
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# Statistical Summary
# -------------------------------------------------------

st.subheader("Statistical Summary")

st.dataframe(
    df.describe(),
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# Key Insights
# -------------------------------------------------------

st.success("""
### Key Insights

- Electricity demand follows clear seasonal and daily patterns.
- Peak demand occurs during specific high-load time blocks.
- Thermal generation contributes the largest share of total generation.
- Strong correlations exist between demand and generation variables.
- Time-based features significantly improve forecasting performance.
- The processed dataset is suitable for both Machine Learning and Deep Learning models.
""")