import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# File paths
INPUT_FILE = 'data/processed/merged_sentiment_stock.csv'
FORECAST_OUTPUT = 'outputs/forecasts/roberta_prophet_forecast.csv'
PLOT_OUTPUT = 'outputs/charts/roberta_prophet_forecast_plot.png'

print("Loading merged data...")
df = pd.read_csv(INPUT_FILE)
df['ds'] = pd.to_datetime(df['Date'])
df['y'] = df['Close_norm']  # Target variable

# Prophet setup
print("Initializing Prophet model with sentiment regressor...")
model = Prophet()
model.add_regressor('Sentiment_norm')

# Fit model
print("Training model...")
model.fit(df[['ds', 'y', 'Sentiment_norm']])

# Create future DataFrame
print("Creating future dataframe...")
future = model.make_future_dataframe(periods=30)
future['Sentiment_norm'] = df['Sentiment_norm'].iloc[-1]  # Fill with last known value

# Forecast
print("Generating forecast...")
forecast = model.predict(future)

# Save forecast
os.makedirs(os.path.dirname(FORECAST_OUTPUT), exist_ok=True)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(FORECAST_OUTPUT, index=False)

# Plot using matplotlib
print("Plotting and saving with Matplotlib...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['ds'], df['y'], label='Actual', color='blue')
ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='orange')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.3)
ax.set_title("Prophet Forecast with RoBERTa Sentiment")
ax.set_xlabel("Date")
ax.set_ylabel("Normalized Price")
ax.legend()

os.makedirs(os.path.dirname(PLOT_OUTPUT), exist_ok=True)
plt.tight_layout()
plt.savefig(PLOT_OUTPUT)
plt.close()

print(f" Forecast saved to {FORECAST_OUTPUT}")
print(f" Plot saved to {PLOT_OUTPUT}")
