import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# File paths
INPUT_FILE = 'data/processed/prophet_ready.csv'
FORECAST_OUTPUT = 'outputs/forecasts/vanilla_prophet_forecast.csv'
PLOT_OUTPUT = 'outputs/charts/vanilla_prophet_forecast_plot.png'

print("Loading data...")
df = pd.read_csv(INPUT_FILE)
df['ds'] = pd.to_datetime(df['ds'])  # Ensure datetime format

# Initialize Prophet
print("Training vanilla Prophet model...")
model = Prophet()
model.fit(df)

# Create future dataframe (30-day forecast)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Save forecast
os.makedirs(os.path.dirname(FORECAST_OUTPUT), exist_ok=True)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(FORECAST_OUTPUT, index=False)

# Plot
print("Saving plot...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['ds'], df['y'], label='Actual', color='blue')
ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='green')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='green', alpha=0.2)
ax.set_title("Vanilla Prophet Forecast (No Sentiment)")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(PLOT_OUTPUT), exist_ok=True)
plt.savefig(PLOT_OUTPUT)
plt.close()

print(f" Forecast saved to {FORECAST_OUTPUT}")
print(f" Plot saved to {PLOT_OUTPUT}")
