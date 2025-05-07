import pandas as pd
import matplotlib.pyplot as plt
import os

# Load each model's predictions
print("Loading predictions...")
base = pd.read_csv('outputs/forecasts/lightgbm_sentiment_preds.csv')
nosent = pd.read_csv('outputs/forecasts/lightgbm_nosentiment_preds.csv')
advanced = pd.read_csv('outputs/forecasts/lightgbm_advanced_preds.csv')

# All have the same y_true
y_true = base['y_true'].values
y_base = base['y_pred'].values
y_nosent = nosent['y_pred'].values
y_adv = advanced['y_pred'].values

# Plot
print("Plotting comparisons...")
plt.figure(figsize=(12, 6))
plt.plot(y_true, label='Actual', color='black')
plt.plot(y_base, label='Predicted: LightGBM + Sentiment', linestyle='--')
plt.plot(y_nosent, label='Predicted: LightGBM (No Sentiment)', linestyle='--')
plt.plot(y_adv, label='Predicted: Advanced Features', linestyle='-')
plt.title("Model Comparison: LightGBM Variants")
plt.xlabel("Test Sample Index")
plt.ylabel("Normalized Close Price")
plt.legend()
plt.tight_layout()

# Save
out_path = 'outputs/charts/lightgbm_model_comparison.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path)
plt.close()

print(f" Comparison chart saved to {out_path}")
