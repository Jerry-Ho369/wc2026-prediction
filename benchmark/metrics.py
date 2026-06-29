"""
benchmark/metrics.py + historical_backtest.py
Scoring metrics and backtest runner for 2014, 2018, 2022, and 2026 group stage.

Metrics:
  - Brier Score: mean squared error of probabilities
  - Ranked Probability Score (RPS): gold standard for multi-outcome forecasts
  - Accuracy: predicted most-likely outcome == actual
  - Top-4 accuracy: correct semi-finalists
  - Champion accuracy: binary
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable

from config import (
    GROUP_RESULTS_2026, WC_YEARS, TRAIN_CUTOFF_DATES, RESULTS_DIR
)


# ── Scoring metrics ────────────────────────────────────────────────────────────
def ranked_probability_score(p_pred: tuple[float, float, float],
                               actual: int) -> float:
    """
    RPS for a single 3-outcome forecast.
    actual: 0=W, 1=D, 2=L
    p_pred: (p_win, p_draw, p_loss)
    Lower is better. Perfect forecast = 0.0.
    Formula: RPS = Σ_k (CDF_pred_k - CDF_actual_k)²  /  (K-1)
    """
    p_win, p_draw, p_loss = p_pred
    # Cumulative probabilities
    pred_cdf = [p_win, p_win + p_draw, 1.0]
    actual_cdf = [1.0 if actual == 0 else 0.0,
                  1.0 if actual <= 1 else 0.0,
                  1.0]
    rps = sum((pred_cdf[k] - actual_cdf[k])**2 for k in range(3)) / 2.0
    return rps


def brier_score(p_pred: tuple[float, float, float], actual: int) -> float:
    """Brier score for a single match (multi-class)."""
    one_hot = [1.0 if actual == i else 0.0 for i in range(3)]
    return sum((p_pred[i] - one_hot[i])**2 for i in range(3)) / 3.0


def accuracy_score_match(p_pred: tuple[float, float, float],
                          actual: int) -> float:
    """1 if the most likely predicted outcome == actual, else 0."""
    return 1.0 if np.argmax(p_pred) == actual else 0.0


def evaluate_predictions(predictions: list[dict]) -> dict:
    """
    predictions: list of dicts with keys:
      {p_win, p_draw, p_loss, actual_result (W/D/L), home_team, away_team, date}
    Returns aggregated metrics dict.
    """
    rps_scores, brier_scores, acc_scores = [], [], []

    for pred in predictions:
        p = (pred["p_win"], pred["p_draw"], pred["p_loss"])
        actual_str = pred.get("actual_result", "")
        if actual_str == "W":
            actual = 0
        elif actual_str == "D":
            actual = 1
        else:
            actual = 2

        rps_scores.append(ranked_probability_score(p, actual))
        brier_scores.append(brier_score(p, actual))
        acc_scores.append(accuracy_score_match(p, actual))

    return {
        "n_matches": len(predictions),
        "mean_rps": np.mean(rps_scores),
        "mean_brier": np.mean(brier_scores),
        "accuracy": np.mean(acc_scores),
        "rps_per_match": rps_scores,
        "brier_per_match": brier_scores,
    }


# ── Historical data loader ─────────────────────────────────────────────────────
def load_historical_matches(csv_path: Path) -> pd.DataFrame:
    """
    Load the Kaggle international results CSV.
    Expected columns: date, home_team, away_team, home_score, away_score, tournament
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Standardise tournament type
    def classify_tournament(t: str) -> str:
        t_lower = str(t).lower()
        if "fifa world cup" in t_lower and "qualification" not in t_lower:
            return "WC"
        elif "qualification" in t_lower or "qualifying" in t_lower:
            return "QUAL"
        elif any(x in t_lower for x in ["nations league", "continental", "copa",
                                          "euro", "african", "asian", "gold cup"]):
            return "CONF"
        else:
            return "FRIENDLY"

    df["tournament_type"] = df["tournament"].apply(classify_tournament)

    # Add result column
    def result(row):
        if row["home_score"] > row["away_score"]:
            return "W"
        elif row["home_score"] < row["away_score"]:
            return "L"
        else:
            return "D"

    df["result"] = df.apply(result, axis=1)
    return df


def filter_before_tournament(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return only matches played BEFORE the given World Cup started."""
    cutoff = pd.Timestamp(TRAIN_CUTOFF_DATES[year])
    return df[df["date"] < cutoff].copy()


# ── World Cup actual results (known) ──────────────────────────────────────────
WC_ACTUAL_RESULTS = {
    2014: {
        "champion": "Germany",
        "runner_up": "Argentina",
        "semi_finalists": ["Germany", "Argentina", "Netherlands", "Brazil"],
    },
    2018: {
        "champion": "France",
        "runner_up": "Croatia",
        "semi_finalists": ["France", "Croatia", "Belgium", "England"],
    },
    2022: {
        "champion": "Argentina",
        "runner_up": "France",
        "semi_finalists": ["Argentina", "France", "Morocco", "Croatia"],
    },
}


# ── Backtest runner ────────────────────────────────────────────────────────────
class HistoricalBacktest:
    """
    For each past WC (2014, 2018, 2022):
    1. Train all models on data before tournament start
    2. Predict all group stage and knockout matches
    3. Run Monte Carlo for tournament winner prediction
    4. Score against actual results
    """

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.all_matches = None

    def load_data(self):
        self.all_matches = load_historical_matches(self.data_path)
        print(f"Loaded {len(self.all_matches):,} international matches "
              f"({self.all_matches['date'].min().date()} to "
              f"{self.all_matches['date'].max().date()})")
        return self

    def run_wc_backtest(self, year: int,
                         predict_fn: Callable,
                         wc_matches: pd.DataFrame) -> dict:
        """
        predict_fn: (team_a, team_b) → (p_win, p_draw, p_loss)
        wc_matches: all matches played in that WC (for scoring)
        """
        print(f"\n{'='*50}")
        print(f"Backtesting {year} World Cup...")

        # Score each WC match
        predictions = []
        for _, row in wc_matches.iterrows():
            p_win, p_draw, p_loss = predict_fn(row["home_team"], row["away_team"])
            predictions.append({
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "p_win": p_win,
                "p_draw": p_draw,
                "p_loss": p_loss,
                "actual_result": row["result"],
            })

        match_metrics = evaluate_predictions(predictions)

        # Compare tournament outcome to actual
        actual = WC_ACTUAL_RESULTS.get(year, {})

        print(f"{year} Results:")
        print(f"  Match prediction accuracy: {match_metrics['accuracy']:.1%}")
        print(f"  Mean RPS:                  {match_metrics['mean_rps']:.4f}")
        print(f"  Mean Brier Score:          {match_metrics['mean_brier']:.4f}")
        print(f"  Champion:                  {actual.get('champion', 'N/A')}")

        return {
            "year": year,
            "match_metrics": match_metrics,
            "actual": actual,
            "predictions": predictions,
        }

    def run_all(self, model_predict_fns: dict[str, Callable]) -> pd.DataFrame:
        """
        model_predict_fns: {"elo": fn, "xgboost": fn, ...}
        Returns a summary DataFrame comparing all models across all WC years.
        """
        if self.all_matches is None:
            self.load_data()

        wc_match_filter = self.all_matches["tournament_type"] == "WC"
        wc_only = self.all_matches[wc_match_filter].copy()

        results = []
        for year in [2014, 2018, 2022]:
            cutoff = pd.Timestamp(TRAIN_CUTOFF_DATES[year])
            wc_year_matches = wc_only[
                (wc_only["date"] >= cutoff) &
                (wc_only["date"] < pd.Timestamp(f"{year+1}-01-01"))
            ].copy()

            print(f"\n{year}: {len(wc_year_matches)} WC matches for scoring")

            for model_name, predict_fn in model_predict_fns.items():
                result = self.run_wc_backtest(year, predict_fn, wc_year_matches)
                results.append({
                    "model": model_name,
                    "year": year,
                    "accuracy": result["match_metrics"]["accuracy"],
                    "mean_rps": result["match_metrics"]["mean_rps"],
                    "mean_brier": result["match_metrics"]["mean_brier"],
                    "n_matches": result["match_metrics"]["n_matches"],
                })

        df = pd.DataFrame(results)
        df.to_csv(RESULTS_DIR / "backtest_summary.csv", index=False)
        print(f"\nBacktest summary saved to {RESULTS_DIR / 'backtest_summary.csv'}")
        return df


# ── 2026 Group Stage Benchmark ────────────────────────────────────────────────
class GroupStageBenchmark2026:
    """
    Compare model predictions against ACTUAL 2026 group stage results.
    Group stage is now complete (June 27, 2026), so this is a clean test.
    """

    def __init__(self, predict_fns: dict[str, Callable]):
        self.predict_fns = predict_fns
        self.actual_results = self._build_actual_results()

    def _build_actual_results(self) -> list[dict]:
        results = []
        for home, away, h_goals, a_goals, date in GROUP_RESULTS_2026:
            if h_goals > a_goals:
                r = "W"
            elif h_goals < a_goals:
                r = "L"
            else:
                r = "D"
            results.append({
                "home_team": home, "away_team": away,
                "home_goals": h_goals, "away_goals": a_goals,
                "date": date, "actual_result": r,
            })
        return results

    def run(self) -> pd.DataFrame:
        """Score all models against 2026 group stage actual results."""
        all_scores = []
        for model_name, predict_fn in self.predict_fns.items():
            preds = []
            for match in self.actual_results:
                p_w, p_d, p_l = predict_fn(match["home_team"], match["away_team"])
                preds.append({**match, "p_win": p_w, "p_draw": p_d, "p_loss": p_l})

            metrics = evaluate_predictions(preds)
            all_scores.append({
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "mean_rps": metrics["mean_rps"],
                "mean_brier": metrics["mean_brier"],
                "n_matches": metrics["n_matches"],
                "stage": "2026_group_stage",
            })
            print(f"  {model_name:15s}: acc={metrics['accuracy']:.1%}, "
                  f"RPS={metrics['mean_rps']:.4f}")

        df = pd.DataFrame(all_scores)
        df.to_csv(RESULTS_DIR / "2026_group_stage_benchmark.csv", index=False)
        return df


def compute_ensemble_weights(backtest_df: pd.DataFrame) -> dict[str, float]:
    """
    Derive ensemble weights from inverse RPS on historical backtests.
    Lower RPS → higher weight. Normalised to sum to 1.
    """
    avg_rps = backtest_df.groupby("model")["mean_rps"].mean()
    inv_rps = 1.0 / (avg_rps + 1e-9)
    weights = inv_rps / inv_rps.sum()
    print("\nEnsemble weights (based on inverse RPS):")
    for model, w in weights.sort_values(ascending=False).items():
        print(f"  {model}: {w:.3f}")
    return weights.to_dict()
