import sys
sys.path.insert(0, '.')
from features.build_features import load_matches, add_elo_features
from features.elo import EloRating
from models.poisson_model import PoissonGoalModel, build_team_rolling_stats
from config import DATA_RAW, CONFIRMED_R32_2026
import pandas as pd
import numpy as np

# Load and prepare data
df = load_matches(DATA_RAW / "results.csv")
df, elo = add_elo_features(df)
df = build_team_rolling_stats(df, window=10)

# Fit Poisson model
features = ["strength_gap", "avg_goals_scored_10_gap", "form_score", "is_knockout"]
avail = [f for f in features if f in df.columns]
poisson = PoissonGoalModel(features=avail)
poisson.fit(df)

# Predict scores for all R32 matches
print(f"\n{'═'*55}")
print(f"  PREDICTED SCORES — ROUND OF 32")
print(f"{'═'*55}")

results = []
for team_a, team_b in CONFIRMED_R32_2026:
    ra = elo.get_rating(team_a)
    rb = elo.get_rating(team_b)
    gap = ra - rb

    # Build feature row
    row = {
        "strength_gap": gap,
        "avg_goals_scored_10_gap": 0.0,
        "form_score": 1.5,
        "is_knockout": 1,
    }

    lam_a, lam_b = poisson.predict_expected_goals(row, row)
    goals_a = round(lam_a, 1)
    goals_b = round(lam_b, 1)

    # Most likely exact scoreline
    top = poisson.predict_scoreline_probs(row, row).head(1)
    likely_a = int(top["home_goals"].values[0])
    likely_b = int(top["away_goals"].values[0])

    pw, pd_, pl = elo.predict_win_prob(team_a, team_b)
    fav = team_a if pw > pl else team_b
    conf = max(pw, pl)

    print(f"{team_a:<22} vs {team_b:<22}")
    print(f"  Expected goals:    {goals_a:.1f} — {goals_b:.1f}")
    print(f"  Most likely score: {likely_a} — {likely_b}")
    print(f"  Favourite:         {fav} ({conf:.0%})")
    print()

    results.append({
        "match": f"{team_a} vs {team_b}",
        "expected_goals_a": goals_a,
        "expected_goals_b": goals_b,
        "likely_score": f"{likely_a}-{likely_b}",
        "favourite": fav,
        "confidence": f"{conf:.0%}"
    })

pd.DataFrame(results).to_csv("results/r32_predicted_scores.csv", index=False)
print("Saved to results/r32_predicted_scores.csv")
