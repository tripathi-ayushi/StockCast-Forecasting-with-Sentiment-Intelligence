import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import os

# File paths
INPUT_FILE = 'data/processed/lgbm_ready.csv'
PRED_OUTPUT = 'outputs/forecasts/lightgbm_final_preds.csv'
PLOT_OUTPUT = 'outputs/charts/lightgbm_final_forecast_plot.png'

# Load data
print("Loading processed data...")
df = pd.read_csv(INPUT_FILE)

# Features (with sentiment)
features = ['lag_close_1', 'lag_sentiment_1', 'roll_mean_3', 'roll_std_3']
X = df[features]
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Final tuned model
print("Training LightGBM with best hyperparameters...")
model = lgb.LGBMRegressor(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=150,
    num_leaves=15
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f" Final MAE:  {mae:.4f}")
print(f" Final RMSE: {rmse:.4f}")

# Save predictions
pred_df = pd.DataFrame({'y_true': y_test.values, 'y_pred': y_pred})
os.makedirs(os.path.dirname(PRED_OUTPUT), exist_ok=True)
pred_df.to_csv(PRED_OUTPUT, index=False)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(pred_df['y_true'], label='Actual', color='black')
plt.plot(pred_df['y_pred'], label='Predicted (Final LightGBM)', color='darkorange')
plt.title("Final LightGBM Forecast (Tuned)")
plt.xlabel("Test Sample Index")
plt.ylabel("Normalized Price")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_OUTPUT)
plt.close()

print(f" Predictions saved to {PRED_OUTPUT}")
print(f" Plot saved to {PLOT_OUTPUT}")
