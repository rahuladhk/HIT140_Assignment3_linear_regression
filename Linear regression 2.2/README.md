# Assessment 3 — Linear Regression 2.2

## Project purpose

This project builds and evaluates a multiple linear regression model to predict the number of goals scored by one team in one FIFA World
Cup 2026 match.

## Dataset structure

- 104 match-level observations.
- 208 team-match observations.
- Target variable: goals_scored.
- Exactly eight pre-match explanatory variables.

Each match contributes two team-match observations.

## Data source

Original source website: FBREF

Exact source URL: https://fbref.com/en/comps/1/schedule/World-Cup-Scores-and-Fixtures 

Date accessed: 05/10/2026

The original CSV is retained in data/raw. All cleaning, feature engineering, modelling, evaluation, and visualisation are performed
in Python.

## Explanatory variables

1. team_avg_goals_scored_before
2. team_avg_goals_conceded_before
3. team_goal_difference_before
4. team_win_rate_before
5. opponent_avg_goals_scored_before
6. opponent_avg_goals_conceded_before
7. opponent_goal_difference_before
8. opponent_win_rate_before

Historical predictors exclude the current match.

Opponent predictors are joined from the opponent's own pre-match record for the same fixture.

Opening-match predictors use a fixed zero-history convention.
These zeros indicate no earlier tournament history, not zero ability.

## Score handling

Penalty-shootout tallies are excluded from match goals.
Recorded match scores include extra-time goals where applicable.
A match level after extra time is treated as a draw when calculating the historical scoring win rate.


## Evaluation design

The chronological split uses 82 training matches and 22 test matches, corresponding to 164 and 44 team-match observations.

The test period begins on 2 July 2026. Both observations from a match remain in the same partition.

Model coefficients remain fixed during testing. Historical predictors for later test matches may use results of earlier test matches because those results are available before the later match.

This represents sequential pre-match prediction, not prediction of all remaining matches at one fixed point in time.

## Outputs

Processed datasets and model evaluation tables: data/processed/ Charts: report/
The report folder contains figures.

## Limitations

- Opening-match form predictors do not distinguish team strength.
- Several historical-form predictors are correlated.
- Linear regression can produce negative, non-integer goal predictions.
- The chronological test period differs in tournament-stage composition
  from the training period.
- Results are based on one tournament and one chronological holdout.
- The two observations from a match may be dependent.

## Contributions

Maintained collaboration evidence in the lecturer-provided Microsoft Teams space.