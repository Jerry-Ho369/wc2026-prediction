FIFA World Cup 2026: ML Match Prediction System

A Python pipeline that predicts FIFA World Cup 2026 match outcomes and tournament progression, built and updated live, round by round, while the tournament was being played.

It combines classical rating systems, goal-count models and machine-learning classifiers, runs Monte Carlo simulations of the knockout bracket, and adds tournament-specific adjustments (form, fatigue, penalty-shootout record) for the later rounds.

Outcome: Spain won the tournament, beating Argentina 1-0 after extra time (Ferran Torres, 106').

Inspired by Joachim Klement's econometric World Cup forecast (Panmure Liberum, April 2026) and Ari Joury's multi-model ensemble approach ("I Built 11 Models to Predict the 2026 World Cup", Towards Data Science).

Results
Pre-knockout Monte Carlo (50,000 simulations, run before the Round of 32)
Team	P(win tournament)
Argentina	16.5%
Spain	12.2%
France	11.4%
Brazil	7.3%
England	5.9%

The model's top two title probabilities, Argentina and Spain, were the two finalists. The Netherlands, Klement's pick, was given 4.1% and lost on penalties to Morocco in the Round of 32. Treat this as one data point, not proof of skill: a single tournament cannot separate skill from luck.

Knockout-round accuracy (did the model's favourite advance?)
Round	Correct	Notes
Round of 32	13 / 16	Misses: Germany (lost to Paraguay on penalties), USA (beat Bosnia, model favoured Bosnia), Australia (lost to Egypt on penalties)
Round of 16	6 / 8	Misses: Brazil (lost to Norway), Colombia (lost to Switzerland on penalties)
Quarter-finals	4 / 4	France, Spain, England, Argentina
Semi-finals	not verified	The semi-final prediction output was not saved; see Limitations
Final	see below	Model: 1-1 after 90, Argentina slight favourite in 90 minutes, Spain on penalties. Actual: 0-0 after 90, Spain won 1-0 in extra time

Round of 32 through quarter-finals: 23 / 28 (82%).

Important context: "favourite advances" is a binary task and a naive "higher Elo wins" rule would score similarly, because the predictions are largely Elo-driven. No claim is made that the extra adjustments added accuracy.

Group stage benchmark (52 recorded matches, models trained on data before 11 June 2026)
Model	Win/draw/loss accuracy	Mean RPS (lower is better)
Elo	59.6%	0.167
Colley	59.6%	0.185
PageRank	61.5%	0.189

Only the three rating models were benchmarked on the group stage.

How it works
Raw data  ->  Feature matrix  ->  Models  ->  Ensemble  ->  Monte Carlo bracket  ->  Round-by-round update
Data. About 49,000 international results (1872 to 2026), Fjelstul World Cup database (matches, goals, referees), plus Kaggle 2026 datasets. Training uses matches from 1990 onwards.
Features. Rolling Elo (updated match by match, pre-match values stored), rolling goals scored/conceded, form score, FIFA ranking points, stage importance, and the Elo gap between teams.
Models. Seven models, each exposing a win/draw/loss probability:
Model	Type	Idea
Elo	Rating system	Match-by-match strength updates, weighted by competition importance
Colley	Rating system	Closed-form linear system, no free parameters
PageRank	Rating system	Strength from the quality of opponents beaten (results graph)
Poisson GLM	Goal model	Expected goals per side, scoreline probability matrix
Negative Binomial GLM	Goal model	Poisson with overdispersion
XGBoost	ML classifier	Non-linear feature interactions
PyTorch MLP	ML classifier	64-32-16 network with dropout and batch norm
Ensemble. Weighted average of model probabilities.
Monte Carlo simulator. Simulates the 2026 bracket (group stage with the 8 best third-placed teams, then Round of 32 to Final) with extra time and penalty shootouts.
Live updates. After each real round, actual results are added to config.py and the next round is predicted.
Round-specific adjustments (Round of 16 onward). Predictions came from Elo probabilities plus transparent, hand-set adjustments: in-tournament goals, a form adjustment, a fatigue penalty for teams that played extra time, a suspension penalty, and a historical penalty-shootout record. Later rounds also used late goals, set-piece goals, goals conceded and shots on target.

Matches that could end level are handled with a knockout rule: draw after 90, then extra time with lower scoring, then penalties.

Quick start (macOS)
bash
git clone https://github.com/Jerry-Ho369/wc2026-prediction.git
cd wc2026-prediction

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# XGBoost on macOS needs the OpenMP runtime
brew install libomp

# Data (needs a Kaggle API token in ~/.kaggle/kaggle.json)
python3 data/download_data.py

# Fast run: skips external-API features
python3 main.py --mode=knockout_only --no-unconventional

Results are written to results/. Python 3.11+ recommended. A GPU is not required; PyTorch falls back to CPU.

Note: with the current code the --sims flag does not override the simulation count; runs use 50,000 simulations.

Project structure
wc2026-prediction/
├── main.py                  # full pipeline runner
├── config.py                # groups, results, brackets, per-round adjustment data
├── data/download_data.py    # pulls the free datasets
├── features/
│   ├── elo.py               # Elo, Colley, PageRank
│   ├── build_features.py    # feature matrix builder
│   └── unconventional.py    # GDP, weather, travel, rest, referee (see Limitations)
├── models/
│   ├── poisson_model.py     # Poisson and Negative Binomial GLMs
│   ├── ml_models.py         # XGBoost and PyTorch MLP
│   └── ensemble.py          # weighted ensemble
├── simulator/monte_carlo.py # tournament simulation
├── benchmark/metrics.py     # RPS, Brier, accuracy, backtest and group benchmark
├── predict_scores.py        # score-line predictions (Elo-based heuristic)
├── final_predict.py         # Final and third-place prediction
└── results/                 # CSV outputs per round
Limitations (please read)

This was built in a day, live, and several parts are weaker than the headline numbers suggest.

Backtest is not clean out-of-sample. The 2014, 2018 and 2022 backtest reused models trained on data up to June 2026, so those tournaments were in the training data. It also scored matches, not tournament winners. Do not read it as "the model predicted the last three champions."
XGBoost and neural-net backtest rows are placeholders. They reuse the Elo prediction function, so their scores equal Elo's, and the inverse-RPS ensemble weights came out nearly equal as a result. The ensemble is effectively an equal-weight average.
Unconventional features were not tested. GDP, venue temperature, travel distance, rest days and referee features exist in features/unconventional.py, but every run used --no-unconventional, and the media-pressure feature is an empty placeholder. No ablation study was done, so there is no evidence on whether they help.
Knockout predictions from the Round of 16 onward are not the full ensemble. They are Elo probabilities plus hand-set adjustments. The adjustment sizes are judgment calls, not fitted values, and were never validated.
Several inputs are approximate. Tournament goals, late goals, set-piece goals, shots on target and penalty records in config.py were compiled from news coverage and estimates, not an official data feed. Some group-stage scores in config.py were placeholders.
Score predictions are heuristics. The Poisson fit failed on missing values, so predicted scorelines come from an Elo-based expected-goals formula. Scores are indicative only.
Semi-final accuracy was not recorded. The prediction output was not saved, so it is excluded from the tally.
Small samples. 28 knockout matches cannot distinguish skill from luck.
Suggested next steps
Re-run the backtest with models trained strictly before each tournament.
Wire XGBoost and the neural net into the backtest and the ensemble properly.
Run an ablation of conventional vs. unconventional features.
Replace hand-set adjustments with fitted coefficients and add time-decayed Elo.
Use real xG and shot data.
Data sources
International football results, Mart Jürisoo (Kaggle / GitHub)
Fjelstul World Cup Database (GitHub)
Kaggle FIFA World Cup 2026 datasets
Tournament results compiled from public sports coverage
References
Klement, J. (2026). FIFA World Cup Predictions 2026. Panmure Liberum.
Joury, A. I Built 11 Models to Predict the 2026 World Cup. Towards Data Science.
Hoffmann, Ging & Ramasamy (2002). The socio-economic determinants of international soccer performance.
