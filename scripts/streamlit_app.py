import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="StockCast Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title and Caption
st.title("📈 StockCast: Sentiment-Driven Stock Forecasting")
st.caption("Blend market data with Twitter sentiment to forecast the future — interactively.")

# Sidebar - Project Info
with st.sidebar:
    st.markdown("### About This Project")
    st.markdown("""
    **StockCast** is a forecasting dashboard built using:

    - Facebook Prophet
    - LightGBM
    - RoBERTa Sentiment
    - Plotly & Streamlit

    Explore how sentiment affects stock trends. View predictions, compare models, and analyze changes over time.
    
    [GitHub Repo](https://github.com/tripathi-ayushi/StockCast-Forecasting-with-Sentiment-Intelligence)
    """)

# Load precomputed forecast CSVs
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

# Prophet Forecasting Section
st.header("🧙 Prophet Forecasting")
st.markdown("""
Prophet is a time-series forecasting model developed by Facebook. 
We compare vanilla Prophet (price-only) with sentiment-enriched forecasts.
""")

tab1, tab2 = st.tabs(["Vanilla Prophet", "Prophet + Sentiment"])

with tab1:
    if "ds" in vanilla_df.columns and "yhat" in vanilla_df.columns:
        fig1 = px.line(vanilla_df, x="ds", y="yhat", title="Vanilla Prophet Forecast")
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("Vanilla Prophet data is missing required columns.")

with tab2:
    if "ds" in roberta_df.columns and "yhat" in roberta_df.columns:
        fig2 = px.line(roberta_df, x="ds", y="yhat", title="Prophet + RoBERTa Sentiment Forecast")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("RoBERTa Prophet data is missing required columns.")

# LightGBM Section
st.header("⚡ LightGBM Forecasting")
st.markdown("""
LightGBM enables feature-rich forecasting including lag features and sentiment averages.
""")

with st.container():
    if "date" in lgbm_adv_df.columns and {"y_pred", "y_true"}.issubset(lgbm_adv_df.columns):
        fig3 = px.line(lgbm_adv_df, x="date", y=["y_pred", "y_true"], title="LightGBM Advanced Forecast vs Actual")
        st.plotly_chart(fig3, use_container_width=True)
    elif {"y_pred", "y_true"}.issubset(lgbm_adv_df.columns):
        fig3 = px.line(lgbm_adv_df.reset_index(), x="index", y=["y_pred", "y_true"], title="LightGBM Advanced Forecast vs Actual")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("LightGBM advanced data is missing required columns.")

# Comparison Section
st.header("📊 Model Comparison")
st.markdown("Compare how different models track against actual stock price movements.")

if "date" in lgbm_comp_df.columns and {"y_pred", "y_true"}.issubset(lgbm_comp_df.columns):
    fig4 = px.line(lgbm_comp_df, x="date", y=["y_pred", "y_true"], title="LightGBM Sentiment Forecast vs Actual")
    st.plotly_chart(fig4, use_container_width=True)
elif {"y_pred", "y_true"}.issubset(lgbm_comp_df.columns):
    fig4 = px.line(lgbm_comp_df.reset_index(), x="index", y=["y_pred", "y_true"], title="LightGBM Sentiment Forecast vs Actual")
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.warning("LightGBM comparison data is missing required columns.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | [GitHub Repo](https://github.com/tripathi-ayushi/StockCast-Forecasting-with-Sentiment-Intelligence)")
