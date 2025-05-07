import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, make_scorer
import os

# File path
INPUT_FILE = 'data/processed/lgbm_ready.csv'
RESULTS_FILE = 'outputs/tuning/lightgbm_tuning_results.csv'

print("Loading data...")
df = pd.read_csv(INPUT_FILE)

# Use features WITH sentiment for now
features = ['lag_close_1', 'lag_sentiment_1', 'roll_mean_3', 'roll_std_3']
X = df[features]
y = df['target']

# Define model and parameter grid
model = lgb.LGBMRegressor()

param_grid = {
    'num_leaves': [15, 31, 50],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 150]
}

# TimeSeriesSplit preserves temporal order
tscv = TimeSeriesSplit(n_splits=5)
scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# Grid search
print("Running GridSearchCV...")
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    scoring=scorer,
                    cv=tscv,
                    verbose=1,
                    n_jobs=-1)

grid.fit(X, y)

# Save results
results = pd.DataFrame(grid.cv_results_)
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
results.to_csv(RESULTS_FILE, index=False)

print("Tuning complete.")
print(f"Best params: {grid.best_params_}")
print(f"Best MAE (neg): {grid.best_score_:.4f}")
print(f" Results saved to {RESULTS_FILE}")
