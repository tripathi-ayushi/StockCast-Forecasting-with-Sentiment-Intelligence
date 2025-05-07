import streamlit as st
import pandas as pd
import plotly.express as px

# Set Streamlit to light theme and wide layout
st.set_page_config(
    page_title="StockCast Dashboard",
    page_icon="📈",
    layout="wide"
)

# ----------------------------- Title Section -----------------------------
st.title("📈 StockCast: Sentiment-Driven Stock Forecasting")
st.caption("Blend market data with Twitter sentiment to forecast the future — interactively.")

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("### About This Project")
    st.markdown("""
**StockCast** is a forecasting dashboard built using:

- Facebook Prophet
- LightGBM
- RoBERTa Sentiment
- Plotly & Streamlit

Explore how sentiment affects stock trends. View predictions, compare models, and analyze changes over time.

🔗 [GitHub Repo](https://github.com/tripathi-ayushi/StockCast-Forecasting-with-Sentiment-Intelligence)
    """)

# ----------------------------- Data Loading -----------------------------
@st.cache_data
def load_forecast(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

vanilla_df = load_forecast("outputs/forecasts/vanilla_prophet_forecast.csv")
roberta_df = load_forecast("outputs/forecasts/roberta_prophet_forecast.csv")
lgbm_adv_df = load_forecast("outputs/forecasts/lightgbm_advanced_preds.csv")
lgbm_comp_df = load_forecast("outputs/forecasts/lightgbm_sentiment_preds.csv")

# ----------------------------- Prophet Forecast Section -----------------------------
st.header("🧙 Prophet Forecasting")
st.markdown("""
Prophet is a time-series forecasting model developed by Facebook. 
We compare vanilla Prophet (price-only) with sentiment-enriched forecasts.
""")

tab1, tab2 = st.tabs(["Vanilla Prophet", "Prophet + Sentiment"])

with tab1:
    if {"ds", "yhat"}.issubset(vanilla_df.columns):
        fig = px.line(vanilla_df, x="ds", y="yhat", title="Vanilla Prophet Forecast")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing required columns in Vanilla Prophet forecast.")

with tab2:
    if {"ds", "yhat"}.issubset(roberta_df.columns):
        fig = px.line(roberta_df, x="ds", y="yhat", title="Prophet + RoBERTa Sentiment Forecast")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing required columns in RoBERTa Prophet forecast.")

# ----------------------------- LightGBM Forecast Section -----------------------------
st.header("⚡ LightGBM Forecasting")
st.markdown("""
LightGBM enables feature-rich forecasting including lag features and sentiment averages.
""")

if {"date", "y_true", "y_pred"}.issubset(lgbm_adv_df.columns):
    fig = px.line(lgbm_adv_df, x="date", y=["y_true", "y_pred"], title="LightGBM Advanced Forecast vs Actual")
    st.plotly_chart(fig, use_container_width=True)
elif {"y_true", "y_pred"}.issubset(lgbm_adv_df.columns):
    fig = px.line(lgbm_adv_df.reset_index(), x="index", y=["y_true", "y_pred"], title="LightGBM Advanced Forecast vs Actual")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Missing columns in LightGBM advanced forecast.")

# ----------------------------- Model Comparison Section -----------------------------
st.header("📊 Model Comparison")
st.markdown("Compare how different LightGBM variants perform against actual market behavior.")

if {"date", "y_true", "y_pred"}.issubset(lgbm_comp_df.columns):
    fig = px.line(lgbm_comp_df, x="date", y=["y_true", "y_pred"], title="LightGBM Sentiment Forecast vs Actual")
    st.plotly_chart(fig, use_container_width=True)
elif {"y_true", "y_pred"}.issubset(lgbm_comp_df.columns):
    fig = px.line(lgbm_comp_df.reset_index(), x="index", y=["y_true", "y_pred"], title="LightGBM Sentiment Forecast vs Actual")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Missing columns in LightGBM sentiment forecast.")

# ----------------------------- Footer -----------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | [GitHub](https://github.com/tripathi-ayushi/StockCast-Forecasting-with-Sentiment-Intelligence)")
