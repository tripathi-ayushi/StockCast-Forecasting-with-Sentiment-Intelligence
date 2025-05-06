import pandas as pd
import matplotlib.pyplot as plt
import os

forecast_file = 'outputs/forecasts/roberta_prophet_forecast.csv'
truth_file = 'data/processed/merged_sentiment_stock.csv'
output_file = 'outputs/charts/actual_vs_predicted_roberta.png'

print("Loading data...")
forecast = pd.read_csv(forecast_file)
truth = pd.read_csv(truth_file)

df = pd.merge(forecast, truth, left_on='ds', right_on='Date')

# Plot
print("Plotting...")
plt.figure(figsize=(10, 5))
plt.plot(df['ds'], df['Close_norm'], label='Actual', color='blue')
plt.plot(df['ds'], df['yhat'], label='Predicted', color='orange')
plt.fill_between(df['ds'], df['yhat_lower'], df['yhat_upper'], color='orange', alpha=0.3)
plt.title("Actual vs Predicted (RoBERTa + Prophet)")
plt.xlabel("Date")
plt.ylabel("Normalized Price")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(output_file), exist_ok=True)
plt.savefig(output_file)
plt.close()
print(f" Plot saved to {output_file}")
