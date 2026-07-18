import sys
sys.path.insert(0, '.')
from config import FINAL_2026, THIRD_PLACE_2026, FORM_ADJUSTMENT_FINAL
from config import LATE_GOALS_2026, SET_PIECE_GOALS_2026, SHOTS_ON_TARGET_2026
from config import GOALS_CONCEDED_2026, STAR_PLAYER_FORM, PENALTY_RECORD_FINAL
from config import TOURNAMENT_GOALS_SF_2026, DATA_RAW
from features.build_features import load_matches, add_elo_features
import pandas as pd

df = load_matches(DATA_RAW / 'results.csv')
df, elo = add_elo_features(df)

matches = [
    ("3RD PLACE - France vs England | July 18 Miami", THIRD_PLACE_2026),
    ("FINAL - Spain vs Argentina | July 19 New Jersey", FINAL_2026),
]

results = []
for label, (team_a, team_b) in matches:
    pw, pd_, pl = elo.predict_win_prob(team_a, team_b)

    pw += (TOURNAMENT_GOALS_SF_2026.get(team_a,5) - TOURNAMENT_GOALS_SF_2026.get(team_b,5)) * 0.015
    pw += FORM_ADJUSTMENT_FINAL.get(team_a, 0)
    pl += FORM_ADJUSTMENT_FINAL.get(team_b, 0)
    pw += (GOALS_CONCEDED_2026.get(team_b,5) - GOALS_CONCEDED_2026.get(team_a,5)) * 0.01
    pw += (LATE_GOALS_2026.get(team_a,3) - LATE_GOALS_2026.get(team_b,3)) * 0.01
    pw += (SET_PIECE_GOALS_2026.get(team_a,2) - SET_PIECE_GOALS_2026.get(team_b,2)) * 0.01

    pw = max(0.10, min(0.85, pw))
    pl = max(0.10, min(0.85, 1.0 - pw - pd_))
    pd_ = max(0.05, 1.0 - pw - pl)
    total = pw + pd_ + pl
    pw, pd_, pl = pw/total, pd_/total, pl/total

    pen_a = PENALTY_RECORD_FINAL.get(team_a,(1,3))
    pen_b = PENALTY_RECORD_FINAL.get(team_b,(1,3))
    pra = pen_a[0]/max(pen_a[1],1)
    prb = pen_b[0]/max(pen_b[1],1)

    fav = team_a if pw > pl else team_b
    la = int(round(pw*2.0 + pd_*0.9))
    lb = int(round(pl*2.0 + pd_*0.9))

    print()
    print('='*55)
    print(label)
    print('='*55)
    print(f'Predicted score:     {la}-{lb}')
    print(f'Favourite:           {fav} ({max(pw,pl):.0%})')
    print(f'P(Win/Draw/Loss):    {pw:.0%} / {pd_:.0%} / {pl:.0%}')
    print(f'Penalty strength:    {team_a} {pra:.0%} vs {team_b} {prb:.0%}')
    print(f'Shots on target/gm: {SHOTS_ON_TARGET_2026.get(team_a,0)} vs {SHOTS_ON_TARGET_2026.get(team_b,0)}')
    print(f'Goals conceded:      {GOALS_CONCEDED_2026.get(team_a,0)} vs {GOALS_CONCEDED_2026.get(team_b,0)}')
    print(f'Set piece goals:     {SET_PIECE_GOALS_2026.get(team_a,0)} vs {SET_PIECE_GOALS_2026.get(team_b,0)}')
    print(f'Late goals 75min+:   {LATE_GOALS_2026.get(team_a,0)} vs {LATE_GOALS_2026.get(team_b,0)}')
    sa = STAR_PLAYER_FORM.get(team_a,{})
    sb = STAR_PLAYER_FORM.get(team_b,{})
    if sa: print(f'Star: {team_a} - {sa["player"]} {sa["note"]}')
    if sb: print(f'Star: {team_b} - {sb["player"]} {sb["note"]}')

    results.append({'match':f'{team_a} vs {team_b}','predicted_score':f'{la}-{lb}','favourite':fav,'confidence':f'{max(pw,pl):.0%}','p_win':f'{pw:.0%}','p_draw':f'{pd_:.0%}','p_loss':f'{pl:.0%}'})

pd.DataFrame(results).to_csv('results/final_predictions.csv', index=False)
print()
print('Saved to results/final_predictions.csv')
