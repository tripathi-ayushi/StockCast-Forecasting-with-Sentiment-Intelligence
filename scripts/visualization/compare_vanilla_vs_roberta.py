import pandas as pd
import matplotlib.pyplot as plt
import os

# File paths
vanilla_path = 'outputs/forecasts/vanilla_prophet_forecast.csv'
roberta_path = 'outputs/forecasts/roberta_prophet_forecast.csv'
true_data_path = 'data/processed/merged_sentiment_stock.csv'
output_plot = 'outputs/charts/vanilla_vs_roberta_forecast.png'

print("Loading forecast and true data...")
vanilla = pd.read_csv(vanilla_path)
roberta = pd.read_csv(roberta_path)
truth = pd.read_csv(true_data_path)

# Align dates
vanilla['ds'] = pd.to_datetime(vanilla['ds'])
roberta['ds'] = pd.to_datetime(roberta['ds'])
truth['Date'] = pd.to_datetime(truth['Date'])

# Merge to get true close_norm values for baseline
merged = pd.merge(truth[['Date', 'Close_norm']], vanilla, left_on='Date', right_on='ds', how='inner')
merged_roberta = pd.merge(truth[['Date', 'Close_norm']], roberta, left_on='Date', right_on='ds', how='inner')

print("Plotting predictions...")
plt.figure(figsize=(12, 6))
plt.plot(merged['Date'], merged['Close_norm'], label='Actual', color='black')
plt.plot(vanilla['ds'], vanilla['yhat'], label='Vanilla Prophet', color='green', linestyle='--')
plt.plot(roberta['ds'], roberta['yhat'], label='RoBERTa + Prophet', color='orange')

plt.fill_between(vanilla['ds'], vanilla['yhat_lower'], vanilla['yhat_upper'], color='green', alpha=0.2)
plt.fill_between(roberta['ds'], roberta['yhat_lower'], roberta['yhat_upper'], color='orange', alpha=0.2)

plt.title("Stock Forecast: Vanilla Prophet vs RoBERTa-Enhanced")
plt.xlabel("Date")
plt.ylabel("Normalized Price")
plt.legend()
plt.tight_layout()

# Save
os.makedirs(os.path.dirname(output_plot), exist_ok=True)
plt.savefig(output_plot)
plt.close()

print(f" Comparison plot saved to {output_plot}")
