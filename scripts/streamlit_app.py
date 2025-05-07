import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Page config
st.set_page_config(page_title="StockCast Dashboard", layout="wide")

# Sidebar: Forecast model selection
st.sidebar.title("Forecast Options")
model_choice = st.sidebar.selectbox(
    "Choose a model to view predictions:",
    (
        "LightGBM (Advanced)",
        "LightGBM (No Sentiment)",
        "LightGBM (With Sentiment)",
        "Prophet (Vanilla)",
        "Prophet + RoBERTa"
    )
)

# Sidebar: Toggle sentiment overlay
show_sentiment = st.sidebar.checkbox("Overlay RoBERTa Sentiment Trend", value=False)

# About this project (sidebar footer)
st.sidebar.markdown("---")
st.sidebar.markdown("**About This Project**")
st.sidebar.markdown("Built by Shivani using:")
st.sidebar.markdown("""
- [Facebook Prophet](https://facebook.github.io/prophet/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [RoBERTa Sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- [Streamlit](https://streamlit.io)
""")
st.sidebar.markdown(" [GitHub Repo](https://github.com/tripathi-ayushi/StockCast-Forecasting-with-Sentiment-Intelligence)")

# Map model choices to file paths
model_map = {
    "LightGBM (Advanced)": "outputs/forecasts/lightgbm_advanced_preds.csv",
    "LightGBM (No Sentiment)": "outputs/forecasts/lightgbm_nosentiment_preds.csv",
    "LightGBM (With Sentiment)": "outputs/forecasts/lightgbm_sentiment_preds.csv",
    "Prophet (Vanilla)": "outputs/forecasts/vanilla_prophet_forecast.csv",
    "Prophet + RoBERTa": "outputs/forecasts/roberta_prophet_forecast.csv"
}

# Load predictions
df = pd.read_csv(model_map[model_choice])
y_true = df['y_true'] if 'y_true' in df.columns else df['y']
y_pred = df['y_pred'] if 'y_pred' in df.columns else df['yhat']

# Load sentiment data
sentiment_df = pd.read_csv("data/processed/merged_sentiment_stock_avg.csv")
sentiment_series = sentiment_df["Sentiment_norm"][-len(y_true):].reset_index(drop=True)

# Page Title
st.title("StockCast Forecasting Dashboard")
st.markdown(f"###  Viewing: **{model_choice}**")

# Metrics expander
with st.expander("Forecast Evaluation (Approximate)"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    st.write(f"**MAE**: `{mae:.4f}`")
    st.write(f"**RMSE**: `{rmse:.4f}`")

# Main plot
st.subheader("Actual vs. Predicted (with optional sentiment overlay)")
fig, ax1 = plt.subplots(figsize=(12, 5))

# Plot price prediction
ax1.plot(y_true, label='Actual', color='black', linewidth=2)
ax1.plot(y_pred, label='Predicted', color='orange', linestyle='--')
ax1.set_xlabel("Time Index")
ax1.set_ylabel("Normalized Price")
ax1.set_title(f"{model_choice} Forecast")

# Overlay sentiment
if show_sentiment:
    ax2 = ax1.twinx()
    ax2.plot(sentiment_series, label='RoBERTa Sentiment', color='blue', alpha=0.3)
    ax2.set_ylabel("Sentiment", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

# Merge legends
lines, labels = ax1.get_legend_handles_labels()
if show_sentiment:
    l2, l2_labels = ax2.get_legend_handles_labels()
    lines += l2
    labels += l2_labels
ax1.legend(lines, labels, loc="upper left")

st.pyplot(fig)
