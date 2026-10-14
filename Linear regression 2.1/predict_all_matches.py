import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("regression_dataset.csv")

# Selecting eight explanatory variables
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

# Checking the dataset size and required columns.
assert len(df) == 104, f"Expected 104 matches, found {len(df)}"
assert len(features) == 8, "Expected exactly 8 explanatory variables"
assert all(col in df.columns for col in features + [target]), (
    "One or more required columns are missing."
)
assert df[features + [target]].isna().sum().sum() == 0, (
    "Missing values found in model data."
)

# Separating input variables from target
X = df[features]
y = df[target]

# Training Linear Regression model using all 104 matches.
# These predictions are in-sample, not an independent evaluation.
model = LinearRegression()
model.fit(X, y)

# Generate predictions for all matches
df["predicted_goal_difference"] = model.predict(X)

#Rounding the prediction to nearest whole number
df["rounded_prediction"] = (
    df["predicted_goal_difference"].round().astype(int)
)

# Save predictions
output_columns = [
    "match_date",
    "home_team",
    "away_team",
    "goal_difference",
    "predicted_goal_difference",
    "rounded_prediction",
]

df[output_columns].to_csv("all_match_predictions.csv", index=False)

# Display results summary
print("Predictions generated:", len(df))
print("Explanatory variables:", len(features))
print("Missing model values:", df[features + [target]].isna().sum().sum())

# Showing first 10 predictions
print("\nFirst 10 predictions:")
print(df[output_columns].head(10).to_string(index=False))

print("\nSaved all_match_predictions.csv")
print(
    "Note: these are in-sample predictions. "
    "Use regression_metrics.csv for held-out test performance."
)
