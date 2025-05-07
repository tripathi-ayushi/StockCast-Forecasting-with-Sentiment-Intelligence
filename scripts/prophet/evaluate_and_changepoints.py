import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt
import os

# File paths
INPUT_FILE = 'data/processed/prophet_ready.csv'
FORECAST_FILE = 'outputs/forecasts/vanilla_prophet_forecast.csv'
PLOT_FILE = 'outputs/charts/vanilla_prophet_changepoints.png'

# Load data
df = pd.read_csv(INPUT_FILE)
df['ds'] = pd.to_datetime(df['ds'])

# Train Prophet again (to access changepoints)
print("Retraining Prophet for changepoint visualization...")
model = Prophet()
model.fit(df)

# Forecast again to get forecast object
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Evaluation (on known range only)
merged = pd.merge(forecast, df, on='ds')
mae = mean_absolute_error(merged['y'], merged['yhat'])
rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))

print(f" MAE:  {mae:.4f}")
print(f" RMSE: {rmse:.4f}")

# Plot changepoints
print("Plotting changepoints...")
fig = model.plot(forecast)
add_changepoints_to_plot(fig.gca(), model, forecast)

os.makedirs(os.path.dirname(PLOT_FILE), exist_ok=True)
fig.savefig(PLOT_FILE)
plt.close()

print(f" Changepoint plot saved to {PLOT_FILE}")
