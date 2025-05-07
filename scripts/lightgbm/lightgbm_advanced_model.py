import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths
INPUT_FILE = 'data/processed/lgbm_advanced_ready.csv'
PRED_OUTPUT = 'outputs/forecasts/lightgbm_advanced_preds.csv'
PLOT_OUTPUT = 'outputs/charts/lightgbm_advanced_forecast_plot.png'

# Load data
print("Loading advanced feature dataset...")
df = pd.read_csv(INPUT_FILE)

# Features (exclude Date and target)
features = [col for col in df.columns if col not in ['Date', 'target', 'Close_norm', 'Sentiment_norm']]
X = df[features]
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train
print("Training LightGBM on advanced features...")
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f" MAE:  {mae:.4f}")
print(f" RMSE: {rmse:.4f}")

# Save predictions
pred_df = pd.DataFrame({'y_true': y_test.values, 'y_pred': y_pred})
os.makedirs(os.path.dirname(PRED_OUTPUT), exist_ok=True)
pred_df.to_csv(PRED_OUTPUT, index=False)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(pred_df['y_true'], label='Actual', color='blue')
plt.plot(pred_df['y_pred'], label='Predicted (Advanced)', color='red')
plt.title("LightGBM Forecast with Advanced Features")
plt.xlabel("Test Sample Index")
plt.ylabel("Normalized Price")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_OUTPUT)
plt.close()

print(f" Predictions saved to {PRED_OUTPUT}")
print(f" Plot saved to {PLOT_OUTPUT}")
