import re
import pandas as pd

# first load fixtures and remove blank separator rows to make program simper
df = pd.read_csv("fixture.csv")
df = df.dropna(subset=["Date", "Home", "Away", "Score"]).copy()

#to parse dates in the original CSV format month/day/year
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

# Convert labels
# to consistent team names.
def clean_team_name(value):
    parts = str(value).strip().split()

    if parts and re.fullmatch(r"[a-z]{2,3}", parts[0]):
        parts = parts[1:]
    if parts and re.fullmatch(r"[a-z]{2,3}", parts[-1]):
        parts = parts[:-1]

    return " ".join(parts)


df["Home"] = df["Home"].apply(clean_team_name)
df["Away"] = df["Away"].apply(clean_team_name)

# Load team rankings recorded before the tournament.
ranking_df = pd.read_csv("fifa_rankings.csv")
fifa_rankings = dict(zip(ranking_df["team"], ranking_df["fifa_rank"]))

all_teams = set(df["Home"]) | set(df["Away"])
missing_teams = all_teams - set(fifa_rankings)
if missing_teams:
    raise ValueError(f"Missing FIFA rankings for: {sorted(missing_teams)}")

# Extract the actual match score, excluding penalty shootout totals.
def parse_score(value):
    match = re.search(r"(\d+)\s*[–-]\s*(\d+)", str(value))
    if not match:
        raise ValueError(f"Cannot parse score: {value}")
    return int(match.group(1)), int(match.group(2))


scores = df["Score"].apply(parse_score)
df["home_goals"] = scores.apply(lambda x: x[0])
df["away_goals"] = scores.apply(lambda x: x[1])
df["goal_difference"] = df["home_goals"] - df["away_goals"]

# Match stage is known before kickoff.
stage_numbers = {
    "Group stage": 1,
    "Round of 32": 2,
    "Round of 16": 3,
    "Quarter-finals": 4,
    "Semi-finals": 5,
    "Third-place match": 6,
    "Final": 7,
}
df["stage_number"] = df["Round"].map(stage_numbers)

if df["stage_number"].isna().any():
    raise ValueError("Unrecognised tournament stage found.")

#Team history contains only earlier matches.
df = df.sort_values(["Date", "Time"], kind="stable").reset_index(drop=True)
history = {}
rows = []

def get_history(team):
    return history.get(team, [])

def get_features(team, match_date):
    games = get_history(team)

    win_rate = (
        sum(g["win"] for g in games) / len(games)
        if games else 0.0
    )

    recent = games[-3:]
    recent_avg_gd = (
        sum(g["gd"] for g in recent) / len(recent)
        if recent else 0.0
    )

    avg_goals_for = (
        sum(g["gf"] for g in games) / len(games)
        if games else 0.0
    )


    last_date = games[-1]["date"] if games else match_date - pd.Timedelta(days=7)
    rest_days = (match_date - last_date).days

    return win_rate, recent_avg_gd, avg_goals_for, rest_days


for _, match in df.iterrows():
    home = match["Home"]
    away = match["Away"]
    date = match["Date"]

    home_stats = get_features(home, date)
    away_stats = get_features(away, date)

    # Exactly eight explanatory variables as requirement
    rows.append({
        "match_date": date.strftime("%Y-%m-%d"),
        "home_team": home,
        "away_team": away,

        "home_win_rate_before": home_stats[0],
        "away_win_rate_before": away_stats[0],

        "home_avg_goal_difference_last3": home_stats[1],
        "away_avg_goal_difference_last3": away_stats[1],

        "home_avg_goals_for_before": home_stats[2],
        "away_avg_goals_for_before": away_stats[2],

        "fifa_rank_advantage": fifa_rankings[away] - fifa_rankings[home],
        "stage_number": int(match["stage_number"]),

        "goal_difference": int(match["goal_difference"]),
    })

    # Update histories only after creating this match's predictors.
    home_goals = int(match["home_goals"])
    away_goals = int(match["away_goals"])

    history.setdefault(home, []).append({
        "date": date,
        "gf": home_goals,
        "ga": away_goals,
        "gd": home_goals - away_goals,
        "win": int(home_goals > away_goals),
    })

    history.setdefault(away, []).append({
        "date": date,
        "gf": away_goals,
        "ga": home_goals,
        "gd": away_goals - home_goals,
        "win": int(away_goals > home_goals),
    })


result = pd.DataFrame(rows)

feature_columns = [
    "home_win_rate_before",
    "away_win_rate_before",
    "home_avg_goal_difference_last3",
    "away_avg_goal_difference_last3",
    "home_avg_goals_for_before",
    "away_avg_goals_for_before",
    "fifa_rank_advantage",
    "stage_number",
]

print("Match rows:", len(result))
print("Explanatory variables:", len(feature_columns))
print("Missing values in model columns:",
      result[feature_columns + ["goal_difference"]].isna().sum().sum())
print("Duplicate fixtures:",
      result.duplicated(["match_date", "home_team", "away_team"]).sum())
print("\nFirst five rows:")
print(result.head().to_string(index=False))

# Save the complete dataset, including metadata for identifying fixtures.
result.to_csv("regression_dataset.csv", index=False)

# Save the modelling data with exactly eight predictors and one target.
result[feature_columns + ["goal_difference"]].to_csv(
    "regression_model_data.csv", index=False
)

print("\nSaved regression_dataset.csv")
print("Saved regression_model_data.csv")