
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#load the dataset
df = pd.read_csv("regression_model_data.csv")

features = [
    "home_win_rate_before",
    "away_win_rate_before",
    "home_avg_goal_difference_last3",
    "away_avg_goal_difference_last3",
    "home_avg_goals_for_before",
    "away_avg_goals_for_before",
    "fifa_rank_advantage",
    "stage_number",
]

target = "goal_difference"

#validate the data
assert len(df) == 104, f"Expected 104 rows, found {len(df)}"
assert len(features) == 8, "Expected exactly 8 explanatory variables"
assert df[features + [target]].isna().sum().sum() == 0, "Missing values found"

#split first 80% for training, final 20% for testing
split_index = int(len(df) * 0.80)

X_train = df[features].iloc[:split_index]
X_test = df[features].iloc[split_index:]
y_train = df[target].iloc[:split_index]
y_test = df[target].iloc[split_index:]

#train the model
model = LinearRegression()
model.fit(X_train, y_train)

#predict the test matches
predictions = model.predict(X_test)

#evaluate against actual goal differences
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

#always predict the training-set mean
baseline_predictions = np.full(len(y_test), y_train.mean())
baseline_mae = mean_absolute_error(y_test, baseline_predictions)

print("LINEAR REGRESSION RESULTS")
print("-" * 40)
print("Total matches:", len(df))
print("Explanatory variables:", len(features))
print("Training matches:", len(X_train))
print("Testing matches:", len(X_test))
print(f"Training target mean: {y_train.mean():.3f}")
print()
print(f"MAE: {mae:.3f} goals")
print(f"RMSE: {rmse:.3f} goals")
print(f"R-squared: {r2:.3f}")
print(f"Baseline MAE: {baseline_mae:.3f} goals")

print("\nMODEL COEFFICIENTS")
for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: {coefficient:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

#save actual and predicted goal differences
results = pd.DataFrame({
    "actual_goal_difference": y_test.to_numpy(),
    "predicted_goal_difference": predictions,
    "rounded_prediction": np.rint(predictions).astype(int),
})

results.to_csv("regression_predictions.csv", index=False)

#save metrics for reporting
metrics = pd.DataFrame([{
    "training_matches": len(X_train),
    "testing_matches": len(X_test),
    "MAE": mae,
    "RMSE": rmse,
    "R_squared": r2,
    "baseline_MAE": baseline_mae,
}])
metrics.to_csv("regression_metrics.csv", index=False)

print("\nSaved regression_predictions.csv")
print("Saved regression_metrics.csv")
print("\nTEST SET PREDICTIONS")
print(results.to_string(index=False))
