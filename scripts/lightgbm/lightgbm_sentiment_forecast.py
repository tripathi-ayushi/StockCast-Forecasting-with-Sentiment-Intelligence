import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import numpy as np

# File paths
INPUT_FILE = 'data/processed/lgbm_ready.csv'
PRED_OUTPUT = 'outputs/forecasts/lightgbm_sentiment_preds.csv'
PLOT_OUTPUT = 'outputs/charts/lightgbm_sentiment_forecast_plot.png'

# Load data
print("Loading LightGBM-ready data...")
df = pd.read_csv(INPUT_FILE)

# Features and target
features = ['lag_close_1', 'lag_sentiment_1', 'roll_mean_3', 'roll_std_3']
X = df[features]
y = df['target']

# Train-test split
print("Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train LightGBM
print("Training LightGBM regressor...")
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# Predict
print("Predicting...")
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f" MAE:  {mae:.4f}")
print(f" RMSE: {rmse:.4f}")

# Save predictions
pred_df = pd.DataFrame({
    'y_true': y_test.values,
    'y_pred': y_pred
})
os.makedirs(os.path.dirname(PRED_OUTPUT), exist_ok=True)
pred_df.to_csv(PRED_OUTPUT, index=False)

# Plot
print("Plotting...")
plt.figure(figsize=(10, 5))
plt.plot(pred_df['y_true'].values, label='Actual', color='blue')
plt.plot(pred_df['y_pred'].values, label='Predicted', color='orange')
plt.title("LightGBM Forecast (Sentiment + Rolling Features)")
plt.xlabel("Test Sample Index")
plt.ylabel("Normalized Close Price")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(PLOT_OUTPUT), exist_ok=True)
plt.savefig(PLOT_OUTPUT)
plt.close()

print(f" Predictions saved to {PRED_OUTPUT}")
print(f" Plot saved to {PLOT_OUTPUT}")
