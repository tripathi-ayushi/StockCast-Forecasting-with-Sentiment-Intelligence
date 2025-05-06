import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

forecast_path = 'outputs/forecasts/roberta_prophet_forecast.csv'
true_data_path = 'data/processed/merged_sentiment_stock.csv'

print("Loading data...")
forecast = pd.read_csv(forecast_path)
truth = pd.read_csv(true_data_path)

# Match overlapping dates
merged = pd.merge(forecast, truth, left_on='ds', right_on='Date')
y_true = merged['Close_norm']
y_pred = merged['yhat']

print("Evaluating forecast...")
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f" MAE:  {mae:.4f}")
print(f" RMSE: {rmse:.4f}")
