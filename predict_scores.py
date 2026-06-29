import sys
sys.path.insert(0, '.')
from simulator.monte_carlo import CONFIRMED_R32_2026
from features.build_features import load_matches, add_elo_features
from config import DATA_RAW
import pandas as pd

df = load_matches(DATA_RAW / 'results.csv')
df, elo = add_elo_features(df)

print()
print('PREDICTED SCORES — ROUND OF 32')
print('='*55)

results = []
for team_a, team_b in CONFIRMED_R32_2026:
    pw, pd_, pl = elo.predict_win_prob(team_a, team_b)
    fav = team_a if pw > pl else team_b
    exp_a = round(pw * 2.5 + pd_ * 1.2, 1)
    exp_b = round(pl * 2.5 + pd_ * 1.2, 1)
    likely_a = int(round(exp_a))
    likely_b = int(round(exp_b))
    print(f'{team_a} vs {team_b}')
    print(f'  Predicted score: {likely_a}-{likely_b}  |  Favourite: {fav} ({max(pw,pl):.0%})')
    print()
    results.append({
        'match': f'{team_a} vs {team_b}',
        'predicted_score': f'{likely_a}-{likely_b}',
        'favourite': fav,
        'confidence': f'{max(pw,pl):.0%}',
        'p_win': f'{pw:.0%}',
        'p_draw': f'{pd_:.0%}',
        'p_loss': f'{pl:.0%}'
    })

pd.DataFrame(results).to_csv('results/r32_predicted_scores.csv', index=False)
print('Saved to results/r32_predicted_scores.csv')
