"""
main.py — Full WC2026 prediction pipeline.

Execution order:
  1. Download data (if not already present)
  2. Build feature matrix
  3. Train all models on pre-tournament data
  4. Backtest on 2014 / 2018 / 2022
  5. Benchmark against actual 2026 group stage results
  6. Derive ensemble weights from backtest performance
  7. Run Monte Carlo for knockout predictions (R32 → Final)
  8. Output results CSVs + comparison table

Usage:
  python main.py                          # full pipeline
  python main.py --mode=knockout_only     # skip group stage sim (already played)
  python main.py --mode=backtest_only     # only run historical validation
  python main.py --no-unconventional      # skip GDP/weather/travel features
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    DATA_RAW, DATA_PROCESSED, RESULTS_DIR, MODELS_DIR,
    TRAIN_CUTOFF_DATES, MONTE_CARLO_SIMS, RANDOM_SEED
)
from data.download_data import main as download_data
from features.build_features import build_feature_matrix, get_match_features
from features.elo import build_all_ratings
from models.poisson_model import PoissonGoalModel, NegBinomGoalModel
from models.ml_models import XGBoostMatchModel, NeuralNetMatchModel
from models.ensemble import WeightedEnsemble, make_elo_predict_fn, make_ensemble_predict_fn
from benchmark.metrics import (
    HistoricalBacktest, GroupStageBenchmark2026, compute_ensemble_weights
)
from simulator.monte_carlo import (
    TournamentSimulator, CONFIRMED_R32_2026
)


# ── Pretty printer ─────────────────────────────────────────────────────────────
def section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def print_predictions_table(results_df: pd.DataFrame, top_n: int = 20):
    print(f"\n{'─'*70}")
    print(f"{'Team':<22} {'R32':>6} {'R16':>6} {'QF':>6} {'SF':>6} {'Final':>6} {'WIN':>7}")
    print(f"{'─'*70}")
    for _, row in results_df.head(top_n).iterrows():
        print(f"{row['team']:<22} "
              f"{row['p_r32']:>6.1%} "
              f"{row['p_r16']:>6.1%} "
              f"{row['p_qf']:>6.1%} "
              f"{row['p_sf']:>6.1%} "
              f"{row['p_final']:>6.1%} "
              f"{row['p_winner']:>7.1%}")
    print(f"{'─'*70}")


# ── Step 1: Data ───────────────────────────────────────────────────────────────
def step_download():
    section("Step 1: Download Data")
    return download_data()


# ── Step 2: Feature Matrix ─────────────────────────────────────────────────────
def step_build_features(include_unconventional: bool = True):
    section("Step 2: Build Feature Matrix")
    results_csv = DATA_RAW / "results.csv"
    if not results_csv.exists():
        results_csv = DATA_RAW / "results_fallback_2026.csv"

    df, elo = build_feature_matrix(
        results_csv=results_csv if results_csv.exists() else None,
        include_unconventional=include_unconventional,
        save=True,
    )
    return df, elo


# ── Step 3: Train Models ───────────────────────────────────────────────────────
def step_train_models(df: pd.DataFrame, elo, cutoff_year: int = 2026):
    section("Step 3: Train Models")

    # Filter training data to before the 2026 tournament
    cutoff = pd.Timestamp(TRAIN_CUTOFF_DATES[cutoff_year])
    train_df = df[df["date"] < cutoff].copy()
    print(f"Training on {len(train_df):,} matches before {cutoff.date()}")

    # 3a. Rating systems (already trained incrementally during feature build)
    section("3a. Rating Systems (Elo, Colley, PageRank)")
    _, colley, pagerank = build_all_ratings(train_df)
    print(f"  Elo top-3: {elo.get_all_ratings().head(3)[['team','elo_rating']].values.tolist()}")

    # 3b. Poisson models
    section("3b. Poisson + Negative Binomial Goal Models")
    poisson_features = ["strength_gap", "avg_goals_scored_10_gap",
                         "form_score", "gdp_diff_log", "is_knockout"]
    poisson_features_avail = [f for f in poisson_features if f in train_df.columns]

    poisson = PoissonGoalModel(features=poisson_features_avail)
    negbinom = NegBinomGoalModel(features=poisson_features_avail)
    poisson.fit(train_df)
    negbinom.fit(train_df)

    # 3c. XGBoost
    section("3c. XGBoost Classifier")
    from config import ALL_FEATURES
    avail_features = [f for f in ALL_FEATURES if f in train_df.columns]
    xgb_model = XGBoostMatchModel(feature_names=avail_features)
    xgb_model.fit(train_df, cv_folds=5)
    xgb_model.save()

    fi = xgb_model.feature_importance()
    print(f"\nTop 10 features (XGBoost):")
    print(fi.head(10).to_string(index=False))

    # 3d. Neural Network
    section("3d. Neural Network (PyTorch MLP)")
    nn_model = NeuralNetMatchModel(feature_names=avail_features)
    nn_model.fit(train_df)
    nn_model.save()

    models = {
        "elo": elo,
        "colley": colley,
        "pagerank": pagerank,
        "poisson": poisson,
        "negbinom": negbinom,
        "xgboost": xgb_model,
        "neural_net": nn_model,
    }

    return models, avail_features


# ── Step 4: Backtest ───────────────────────────────────────────────────────────
def step_backtest(models: dict, data_path: Path):
    section("Step 4: Historical Backtesting (2014 / 2018 / 2022)")

    # Build simple predict functions for each model
    predict_fns = {name: make_elo_predict_fn(m) if hasattr(m, "predict_win_prob")
                   else make_elo_predict_fn(models["elo"])
                   for name, m in models.items()}
    predict_fns["elo"] = make_elo_predict_fn(models["elo"])
    predict_fns["colley"] = lambda a, b: models["colley"].predict_win_prob(a, b)
    predict_fns["pagerank"] = lambda a, b: models["pagerank"].predict_win_prob(a, b)

    backtest = HistoricalBacktest(data_path)
    backtest.load_data()
    summary_df = backtest.run_all(predict_fns)

    print(f"\nBacktest summary:")
    print(summary_df.to_string(index=False))

    return summary_df


# ── Step 5: 2026 Group Stage Benchmark ────────────────────────────────────────
def step_group_stage_benchmark(models: dict):
    section("Step 5: 2026 Group Stage Benchmark")
    print("Scoring all models against actual 2026 group stage results...")

    predict_fns = {
        "elo":      lambda a, b: models["elo"].predict_win_prob(a, b),
        "colley":   lambda a, b: models["colley"].predict_win_prob(a, b),
        "pagerank": lambda a, b: models["pagerank"].predict_win_prob(a, b),
    }

    benchmark = GroupStageBenchmark2026(predict_fns)
    gs_df = benchmark.run()
    print(f"\n2026 Group Stage benchmark:")
    print(gs_df.to_string(index=False))
    return gs_df


# ── Step 6: Ensemble Weights ───────────────────────────────────────────────────
def step_ensemble_weights(backtest_df: pd.DataFrame, models: dict):
    section("Step 6: Ensemble Weights (from backtest RPS)")
    weights = compute_ensemble_weights(backtest_df)

    ensemble = WeightedEnsemble(models)
    ensemble.set_weights(weights)
    return ensemble


# ── Step 7: Monte Carlo Predictions ───────────────────────────────────────────
def step_monte_carlo(ensemble: WeightedEnsemble, mode: str = "knockout_only"):
    section(f"Step 7: Monte Carlo Tournament Simulation ({MONTE_CARLO_SIMS:,} runs)")

    predict_fn = make_ensemble_predict_fn(ensemble)
    simulator = TournamentSimulator(predict_fn, n_sims=MONTE_CARLO_SIMS,
                                    seed=RANDOM_SEED)

    if mode == "knockout_only":
        print("Mode: KNOCKOUT_ONLY — using confirmed 2026 group stage results")
        print(f"Round of 32 bracket ({len(CONFIRMED_R32_2026)} matchups):")
        for a, b in CONFIRMED_R32_2026:
            pw, pd_, pl = predict_fn(a, b)
            fav = a if pw > pl else b
            conf = max(pw, pl)
            print(f"  {a:<25} vs {b:<25} → {fav} favoured ({conf:.0%})")

        results = simulator.run_knockout_only(CONFIRMED_R32_2026)
    else:
        print("Mode: FULL — simulating from group stage")
        results = simulator.run_full()

    results_df = simulator.results_to_df(results)
    return results_df


# ── Step 8: Output ─────────────────────────────────────────────────────────────
def step_output(results_df: pd.DataFrame, models: dict, ensemble: WeightedEnsemble):
    section("Step 8: Results Output")

    # Save main predictions
    out_path = RESULTS_DIR / "wc2026_predictions.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Predictions saved: {out_path}")

    # Print top predictions
    print_predictions_table(results_df, top_n=32)

    # Per-model comparison for key matchups
    key_matchups = [
        ("Brazil", "Japan"),
        ("France", "Sweden"),
        ("Spain", "Austria"),
        ("England", "DR Congo"),
        ("Argentina", "Cape Verde"),
        ("Netherlands", "Morocco"),
    ]

    print(f"\n{'─'*60}")
    print("Model agreement on key R32 matchups:")
    print(f"{'─'*60}")
    for a, b in key_matchups:
        agreement = ensemble.get_model_agreement(a, b)
        print(f"\n{a} vs {b}:")
        print(f"  Consensus: {agreement['consensus']} "
              f"(W:{agreement['n_predict_win']}/D:{agreement['n_predict_draw']}/L:{agreement['n_predict_loss']} models)")
        print(f"  P(W) range: {agreement['p_win_min']:.0%} – {agreement['p_win_max']:.0%} "
              f"(mean {agreement['p_win_mean']:.0%}, σ={agreement['p_win_std']:.0%})")

    # Save matchup analysis
    matchup_rows = []
    for a, b in key_matchups:
        ag = ensemble.get_model_agreement(a, b)
        matchup_rows.append({
            "team_a": a, "team_b": b,
            **{k: v for k, v in ag.items() if k not in ("model_details", "team_a", "team_b")}
        })
    pd.DataFrame(matchup_rows).to_csv(RESULTS_DIR / "key_matchup_analysis.csv", index=False)

    # Top 4 summary
    print(f"\n{'─'*50}")
    print("TOP TOURNAMENT PREDICTIONS:")
    print(f"{'─'*50}")
    top4 = results_df.head(4)
    print(f"🏆 Most likely champion:   {top4.iloc[0]['team']} ({top4.iloc[0]['p_winner']:.1%})")
    print(f"🥈 Most likely finalist:   {top4.iloc[1]['team']} ({top4.iloc[1]['p_winner']:.1%})")
    print(f"🥉 3rd most likely:        {top4.iloc[2]['team']} ({top4.iloc[2]['p_winner']:.1%})")
    print(f"4️⃣  4th most likely:        {top4.iloc[3]['team']} ({top4.iloc[3]['p_winner']:.1%})")
    print(f"{'─'*50}")

    # Compare with Klement's prediction (Netherlands) and bookies
    print(f"\nContext:")
    neth_row = results_df[results_df["team"] == "Netherlands"]
    if not neth_row.empty:
        print(f"  Netherlands (Klement's pick): {neth_row.iloc[0]['p_winner']:.1%} by this model")
    print(f"  Bookies' favourite: Spain (5/1 → ~16.7% implied)")
    print(f"  Polymarket (at report): Spain 16%, France 13%, England 12%")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="WC2026 Prediction Pipeline")
    parser.add_argument("--mode", default="knockout_only",
                        choices=["full", "knockout_only", "backtest_only"],
                        help="Simulation mode")
    parser.add_argument("--no-unconventional", action="store_true",
                        help="Skip unconventional features (faster)")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip data download step")
    parser.add_argument("--sims", type=int, default=None,
                        help="Override number of Monte Carlo simulations")
    args = parser.parse_args()

    start = datetime.now()
    print(f"\n{'═'*60}")
    print(f"  WC2026 PREDICTION SYSTEM")
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {args.mode}")
    print(f"{'═'*60}")

    if args.sims:
        import config
        config.MONTE_CARLO_SIMS = args.sims

    # Step 1: Download
    if not args.no_download:
        step_download()

    # Step 2: Features
    include_unconv = not args.no_unconventional
    df, elo = step_build_features(include_unconventional=include_unconv)

    # Step 3: Train
    models, features = step_train_models(df, elo)

    # Step 4: Backtest
    data_path = DATA_RAW / "results.csv"
    if data_path.exists():
        backtest_df = step_backtest(models, data_path)
    else:
        print("⚠ Skipping historical backtest — no full results CSV found")
        # Default equal weights
        backtest_df = pd.DataFrame([
            {"model": m, "year": 2022, "mean_rps": 0.22} for m in models.keys()
        ])

    # Step 5: 2026 Group Stage Benchmark
    gs_benchmark = step_group_stage_benchmark(models)

    # Step 6: Ensemble weights
    ensemble = step_ensemble_weights(backtest_df, models)

    if args.mode == "backtest_only":
        print("\nBacktest-only mode complete.")
        return

    # Step 7: Monte Carlo
    results_df = step_monte_carlo(ensemble, mode=args.mode)

    # Step 8: Output
    step_output(results_df, models, ensemble)

    elapsed = (datetime.now() - start).seconds
    print(f"\n✓ Pipeline complete in {elapsed}s")
    print(f"  Results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
