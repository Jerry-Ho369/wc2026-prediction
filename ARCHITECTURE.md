# World Cup 2026 ML Prediction System — Architecture

## 1. Overview & Philosophy

This system takes **two philosophies and merges them**:

| Source | Approach |
|---|---|
| Joachim Klement (Panmure Liberum) | Econometric socioeconomic model: GDP/capita, population, temperature, host advantage, FIFA ranking |
| Ari Joury (Towards Data Science) | Multi-model ensemble: Elo, Poisson, Random Forest, XGBoost, Neural Net, betting odds |
| This system | Both + unconventional signal layer: travel fatigue, media pressure, referee bias, climate |

**Core principle**: No single model gives a single number and calls it truth. We run N models through the same tournament simulator, benchmark against 2014/2018/2022, then predict 2026 group→knockout→final.

---

## 2. Data Sources (All Free)

### Primary Match Data
| Dataset | URL | Content |
|---|---|---|
| Kaggle: International results 1872–2026 | kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017 | 49,000+ match scores |
| openfootball/worldcup | github.com/openfootball/worldcup | WC fixtures 2014–2026 |
| Fjelstul WC Database | github.com/jfjelstul/worldcup | All WC 1930–2022, goals, cards, referees |
| Kaggle: FIFA WC 1930–2022 | kaggle.com/datasets/piterfm/fifa-football-world-cup | Match results + rankings |
| Kaggle: WC 2026 Live Dataset | kaggle.com/datasets/mominullptr/fifa-world-cup-2026-dataset | 2026 squads, xG, events |
| Kaggle: WC 2026 Team Dataset | kaggle.com/datasets/harrachimustapha/fifa-world-cup-team-dataset | Pre-tournament features |

### Player & Squad Data
| Dataset | URL | Content |
|---|---|---|
| FBref (soccerdata Python lib) | fbref.com | xG, progressive passes, pressures, per-90 stats |
| Kaggle: WC 2026 Player Data | github.com/openfootball/awesome-football | 1363 players, per-90 stats |
| Transfermarkt (soccerdata lib) | transfermarkt.com | Market value, squad age, injuries |

### Unconventional / External Features
| Feature | Source | Notes |
|---|---|---|
| GDP/capita | World Bank API (free) | worldbank.org/indicator/NY.GDP.PCAP.CD |
| Temperature at venue | Open-Meteo API (free) | open-meteo.com — historical + forecast by lat/lon/date |
| Travel distance | geopy + venue coordinates | Great-circle km from team's country capital to match city |
| Media pressure index | GDELT Project (free) | gdeltproject.org — news volume per country per day |
| Referee nationality | Fjelstul WC Database | referee country ≠ either team → bias proxy |
| FIFA ranking points | FIFA / Kaggle | Ranking points at tournament start |
| Squad market value | Transfermarkt scrape | Total squad value in EUR |
| Average player age | Transfermarkt / Kaggle | Squad age profile |
| Days rest between matches | Computed from fixture schedule | Recovery proxy |
| Host nation flag | Known from draw | USA, Canada, Mexico = 1 |
| Altitude of venue | Wikipedia / manual CSV | Affects stamina |
| Match importance (stage) | Derived from round | Group=1, R32=2, R16=3, QF=4, SF=5, F=6 |

---

## 3. Feature Engineering

### Team-Level Features (per match, per team)
```
elo_rating              — Elo rating at match date
colley_rating           — Colley linear system rating
pagerank_rating         — PageRank from results graph
fifa_ranking_points     — Official FIFA points
avg_goals_scored_10     — Rolling avg goals scored, last 10 matches
avg_goals_conceded_10   — Rolling avg goals conceded, last 10 matches
xg_for_10               — Rolling xG for, last 10 matches
xg_against_10           — Rolling xG against, last 10 matches
win_pct_5               — Win% last 5 matches
form_score              — Weighted W/D/L over last 6: W=3,D=1,L=0
squad_market_value_log  — Log of total market value (EUR)
avg_squad_age           — Mean age of starting XI
days_rest               — Days since last competitive match
travel_distance_km      — Great-circle km to venue
temperature_celsius     — Historical/forecast temp at venue on match date
altitude_m              — Venue altitude in metres
gdp_per_capita_log      — Log GDP/capita (Klement factor)
population_log          — Log population (Klement factor)
is_host_nation          — 1 if USA/Canada/Mexico
media_pressure_index    — GDELT news volume 7-day rolling avg
referee_same_conf       — 1 if referee from same confederation
stage_importance        — 1–6 ordinal match weight
strength_gap            — Δ Elo between teams
combined_strength       — Sum of Elo both teams
is_knockout             — 0=group, 1=knockout
```

### Match-Level Target
```
result      — W/D/L (3-class for classifiers)
goal_diff   — signed goal difference (for regression)
home_goals  — for Poisson GLM
away_goals  — for Poisson GLM
```

---

## 4. Model Suite (7 Models)

### Rating Systems (produce team strength → win probability)
| # | Model | Library | Notes |
|---|---|---|---|
| 1 | Elo | Custom Python | K=30, init=1500, updated match-by-match |
| 2 | Colley | NumPy (linear system) | No free params, closed-form |
| 3 | PageRank | NetworkX | d=0.85, power iteration |

### Probabilistic Goal Models
| # | Model | Library | Notes |
|---|---|---|---|
| 4 | Poisson GLM | statsmodels | λ = f(strength_gap, features) |
| 5 | Negative Binomial GLM | statsmodels | Relaxes mean=variance |

### ML Classifiers (predict W/D/L from full feature vector)
| # | Model | Library | Notes |
|---|---|---|---|
| 6 | XGBoost | xgboost | Gradient boosted trees; handles feature interactions |
| 7 | Neural Network (MLP) | PyTorch | 64→32→16→3, softmax, dropout 0.3 |

### Ensemble Output
- All 7 models expose `predict_proba(team_a, team_b) → (p_win, p_draw, p_loss)`
- Tournament simulator runs 50,000 Monte Carlo iterations per model
- Final output: weighted ensemble (weights from historical backtesting accuracy)

---

## 5. Tournament Simulator

```
For each Monte Carlo iteration:
  1. Group stage:
     - Simulate all 36 group matches using model probabilities
     - Apply FIFA 2026 tiebreaker rules (H2H points first, then GD)
     - Identify top-2 per group + 8 best 3rd-place teams
  2. Round of 32 → Round of 16 → QF → SF → Final:
     - Knockout: P(win in 90min), P(extra time), P(penalties)
     - Extra time: reduce goal rate by 20%
     - Penalty: 50/50 (no reliable predictors)
  3. Record champion, finalist, SF participants
After 50,000 iterations:
  - P(win tournament) = count(team won) / 50,000
  - P(reach round X) = count(team reached round X) / 50,000
```

---

## 6. Benchmarking Strategy

### Phase 1: Historical Validation
Train on all international matches **before** each tournament start date.
Simulate 2014, 2018, 2022 tournaments.
Score metrics:
- Brier Score (probability calibration)
- Ranked Probability Score (RPS) — the gold standard for multi-outcome forecasts
- Top-4 accuracy (did we predict the correct semi-finalists?)
- Champion accuracy (binary)

### Phase 2: 2026 Group Stage Benchmark (already played)
Group stage is now COMPLETE (June 11–27, 2026).
We can:
1. Run our model as if it's June 10 (pre-tournament), predict all group results
2. Compare to actual results from search (all 72 group matches)
3. Compute RPS and accuracy vs. actual outcomes
4. Then retrain/update model with group stage data → improve knockout predictions

### Phase 3: Live Knockout Predictions
Round of 32 began June 28. Some R32 matches may already be played.
Update Elo after each completed match, re-run simulator for remaining bracket.

---

## 7. Module Structure

```
wc2026/
├── ARCHITECTURE.md           ← this file
├── requirements.txt
├── config.py                 ← paths, hyperparams, feature lists
├── data/
│   ├── download_data.py      ← pulls all free datasets
│   ├── raw/                  ← downloaded CSVs (gitignored)
│   └── processed/            ← cleaned, feature-engineered parquets
├── features/
│   ├── elo.py                ← Elo rating system
│   ├── colley.py             ← Colley matrix solver
│   ├── pagerank.py           ← PageRank from match graph
│   ├── poisson_features.py   ← rolling goal rates, xG
│   ├── unconventional.py     ← GDP, temperature, travel, GDELT, referee
│   └── build_features.py     ← orchestrates all feature modules
├── models/
│   ├── base_model.py         ← abstract base: predict_proba interface
│   ├── rating_models.py      ← Elo, Colley, PageRank wrappers
│   ├── poisson_model.py      ← Poisson + NegBinom GLMs
│   ├── xgboost_model.py      ← XGBoost classifier
│   ├── neural_net.py         ← PyTorch MLP
│   └── ensemble.py           ← weighted ensemble of all models
├── simulator/
│   ├── tournament.py         ← 2026 bracket, tiebreaker rules
│   ├── monte_carlo.py        ← 50K simulation engine (vectorised)
│   └── knockout_rules.py     ← ET, penalties, suspension rules
├── benchmark/
│   ├── historical_backtest.py ← run on 2014/2018/2022
│   ├── group_stage_2026.py   ← benchmark vs actual 2026 group results
│   └── metrics.py            ← Brier, RPS, accuracy
├── results/
│   ├── group_stage_predictions.csv
│   ├── knockout_predictions.csv
│   └── champion_probabilities.csv
└── main.py                   ← full pipeline runner
```

---

## 8. Known Limitations & Mitigations

| Risk | Mitigation |
|---|---|
| Small training set (358 WC matches) | Augment with all international matches (49K+) with WC-weighted loss |
| XGBoost overfits small data | Cross-validate; use early stopping; compare against simpler models |
| Unconventional features are noisy | Ablation study: compare model with/without each feature group |
| Penalty shootout is near-random | Model as coin flip; penalise models that over-predict favorite in ET |
| GDELT media pressure is a proxy | Use log-transform + clip outliers; treat as ordinal signal not cardinal |
| Referee bias is hard to quantify | Binary flag: referee from same continent as one team |
| 2026 uses 3 host nations | Split host effect: Mexico (traditional), USA/Canada (diluted) |

---

## 9. What We Can Already Benchmark (TODAY, June 29 2026)

The group stage is **done**. Round of 32 has started. Here is what we know:

**Group winners confirmed:**
- A: Mexico | B: Switzerland | C: Brazil | D: USA | E: Germany | F: Netherlands
- G: Belgium | H: Spain | I: France | J: Argentina | K: Portugal | L: England

**Klement model accuracy vs actual:**
- Germany top of Group E ✓ | Ecuador 2nd ✗ (Ecuador lost to Germany on GD; Ivory Coast advanced)
- Japan upset prediction: Japan drew Netherlands 2-2, beat Tunisia 4-0 ✓ (advanced)
- Brazil upset: Brazil vs Japan in R32 — TBD (match June 29)
- France winning Group I ✓ | Netherlands advancing ✓ | Spain winning H ✓
- Argentina winning J ✓

**Key divergence from Klement:** South Africa advanced from Group A (not Czechia). Turkey eliminated. USA survived (despite Klement's prediction they'd come 3rd — they came 1st in Group D).
