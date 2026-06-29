"""
models/poisson_model.py
Poisson GLM and Negative Binomial GLM for goal prediction.
Both expose the standard interface: predict_win_prob(team_a, team_b).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson, nbinom
from itertools import product

from config import ALL_FEATURES, CONVENTIONAL_FEATURES


class PoissonGoalModel:
    """
    Poisson GLM: goals ~ exp(β₀ + β₁*strength_diff + β₂*xg_for + ...)
    Fit two models: one for home goals, one for away goals.
    Win probability recovered from outer product of Poisson distributions.
    Max goals considered for probability table: 10.
    """

    MAX_GOALS = 10
    MODEL_TYPE = "poisson"

    def __init__(self, features: list[str] = None):
        self.features = features or ["strength_gap", "xg_for_10", "xg_against_10",
                                      "form_score", "gdp_diff_log"]
        self.model_home = None
        self.model_away = None
        self._is_fit = False

    def _make_X(self, df: pd.DataFrame, is_home: bool) -> np.ndarray:
        """Build design matrix. For away team, flip directional features."""
        X_rows = []
        for _, row in df.iterrows():
            feats = []
            for f in self.features:
                val = row.get(f, 0.0)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0.0
                # For away perspective, flip signed gap features
                if not is_home and f in ("strength_gap", "gdp_diff_log",
                                          "travel_advantage", "rest_advantage"):
                    val = -val
                feats.append(float(val))
            X_rows.append(feats)
        X = np.array(X_rows, dtype=float)
        return sm.add_constant(X, has_constant="add")

    def fit(self, matches: pd.DataFrame) -> "PoissonGoalModel":
        """
        matches must have columns:
            home_score, away_score + all features in self.features
        """
        X_home = self._make_X(matches, is_home=True)
        X_away = self._make_X(matches, is_home=False)

        y_home = matches["home_score"].values.astype(float)
        y_away = matches["away_score"].values.astype(float)

        family = sm.families.Poisson()
        self.model_home = sm.GLM(y_home, X_home, family=family).fit(disp=0)
        self.model_away = sm.GLM(y_away, X_away, family=family).fit(disp=0)
        self._is_fit = True

        print(f"Poisson model fit. Home AIC: {self.model_home.aic:.1f}, "
              f"Away AIC: {self.model_away.aic:.1f}")
        return self

    def _predict_lambda(self, team_a_row: dict, team_b_row: dict):
        """
        Given a dict of features for a match (team_a = home perspective),
        predict λ_home and λ_away.
        """
        # Build single-row DataFrames
        row_home = pd.DataFrame([team_a_row])
        row_away = pd.DataFrame([team_b_row])

        X_h = self._make_X(row_home, is_home=True)
        X_a = self._make_X(row_away, is_home=False)

        lambda_home = float(self.model_home.predict(X_h)[0])
        lambda_away = float(self.model_away.predict(X_a)[0])
        return max(lambda_home, 0.1), max(lambda_away, 0.1)

    def _probs_from_lambdas(self, lam_h: float, lam_a: float):
        """
        Compute P(W/D/L) from a 10x10 scoreline probability matrix.
        """
        g = range(self.MAX_GOALS + 1)
        # Vectorised outer product
        p_home = np.array([poisson.pmf(i, lam_h) for i in g])
        p_away = np.array([poisson.pmf(i, lam_a) for i in g])
        matrix = np.outer(p_home, p_away)  # P(home scores i, away scores j)

        p_win = float(np.tril(matrix, -1).sum())   # i > j
        p_draw = float(np.trace(matrix))
        p_loss = float(np.triu(matrix, 1).sum())   # i < j

        # Normalise (should sum to ~1 already)
        total = p_win + p_draw + p_loss
        return p_win / total, p_draw / total, p_loss / total

    def predict_win_prob(self, team_a_features: dict, team_b_features: dict):
        """Returns (p_win, p_draw, p_loss) for team_a vs team_b."""
        if not self._is_fit:
            raise RuntimeError("Model not fitted. Call fit() first.")
        lam_h, lam_a = self._predict_lambda(team_a_features, team_b_features)
        return self._probs_from_lambdas(lam_h, lam_a)

    def predict_expected_goals(self, team_a_features: dict, team_b_features: dict):
        """Returns (expected_goals_a, expected_goals_b)"""
        lam_h, lam_a = self._predict_lambda(team_a_features, team_b_features)
        return lam_h, lam_a

    def predict_scoreline_probs(self, team_a_features: dict, team_b_features: dict):
        """Returns DataFrame of P(score) for every scoreline up to MAX_GOALS x MAX_GOALS."""
        lam_h, lam_a = self._predict_lambda(team_a_features, team_b_features)
        g = range(self.MAX_GOALS + 1)
        records = []
        for i, j in product(g, g):
            p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            records.append({"home_goals": i, "away_goals": j, "probability": p})
        return pd.DataFrame(records).sort_values("probability", ascending=False)


class NegBinomGoalModel(PoissonGoalModel):
    """
    Negative Binomial GLM — relaxes Poisson's mean=variance assumption.
    Uses statsmodels NegativeBinomial family.
    Interface identical to PoissonGoalModel.
    """
    MODEL_TYPE = "negbinom"

    def fit(self, matches: pd.DataFrame) -> "NegBinomGoalModel":
        X_home = self._make_X(matches, is_home=True)
        X_away = self._make_X(matches, is_home=False)
        y_home = matches["home_score"].values.astype(float)
        y_away = matches["away_score"].values.astype(float)

        # NegativeBinomial requires initial Poisson fit as start_params
        family = sm.families.NegativeBinomial()
        self.model_home = sm.GLM(y_home, X_home, family=family).fit(disp=0)
        self.model_away = sm.GLM(y_away, X_away, family=family).fit(disp=0)
        self._is_fit = True

        alpha_h = self.model_home.scale
        alpha_a = self.model_away.scale
        print(f"NegBinom model fit. Dispersion α_home={alpha_h:.4f}, "
              f"α_away={alpha_a:.4f} (0 → Poisson)")
        return self

    def _probs_from_lambdas(self, lam_h: float, lam_a: float):
        """
        For NegBinom: use nbinom PMF instead of poisson PMF.
        alpha ≈ dispersion parameter from fitted model.
        """
        alpha_h = getattr(self.model_home, "scale", 0.01)
        alpha_a = getattr(self.model_away, "scale", 0.01)

        def nb_pmf(k, mu, alpha):
            if alpha < 1e-6:
                return poisson.pmf(k, mu)
            n = 1.0 / alpha
            p = n / (n + mu)
            return nbinom.pmf(k, n, p)

        g = range(self.MAX_GOALS + 1)
        p_home_vec = np.array([nb_pmf(i, lam_h, alpha_h) for i in g])
        p_away_vec = np.array([nb_pmf(i, lam_a, alpha_a) for i in g])

        matrix = np.outer(p_home_vec, p_away_vec)
        p_win = float(np.tril(matrix, -1).sum())
        p_draw = float(np.trace(matrix))
        p_loss = float(np.triu(matrix, 1).sum())

        total = p_win + p_draw + p_loss + 1e-9
        return p_win / total, p_draw / total, p_loss / total


# ── Poisson feature builder from match history ─────────────────────────────────
def build_team_rolling_stats(matches: pd.DataFrame,
                              window: int = 10) -> pd.DataFrame:
    """
    For each match, compute rolling stats (last N matches) for each team.
    Returns matches with added columns:
        home_avg_goals_scored, home_avg_goals_conceded,
        away_avg_goals_scored, away_avg_goals_conceded,
        home_form_score, away_form_score
    """
    matches = matches.sort_values("date").copy()

    # Build long format: one row per team per match
    home_rows = matches[["date", "home_team", "home_score", "away_score"]].copy()
    home_rows.columns = ["date", "team", "goals_for", "goals_against"]
    home_rows["result"] = np.sign(home_rows["goals_for"] - home_rows["goals_against"]).map(
        {1: 3, 0: 1, -1: 0}
    )

    away_rows = matches[["date", "away_team", "away_score", "home_score"]].copy()
    away_rows.columns = ["date", "team", "goals_for", "goals_against"]
    away_rows["result"] = np.sign(away_rows["goals_for"] - away_rows["goals_against"]).map(
        {1: 3, 0: 1, -1: 0}
    )

    long = pd.concat([home_rows, away_rows]).sort_values(["team", "date"])

    # Rolling stats per team
    rolling_stats = {}
    for team, grp in long.groupby("team"):
        grp = grp.sort_values("date").reset_index(drop=True)
        grp["avg_scored"] = grp["goals_for"].shift(1).rolling(window, min_periods=1).mean()
        grp["avg_conceded"] = grp["goals_against"].shift(1).rolling(window, min_periods=1).mean()
        grp["form_score"] = grp["result"].shift(1).rolling(6, min_periods=1).mean()
        rolling_stats[team] = grp.set_index("date")[["avg_scored", "avg_conceded", "form_score"]]

    # Merge back
    def get_stat(team, date, col, default=0.0):
        if team not in rolling_stats:
            return default
        df_t = rolling_stats[team]
        prior = df_t[df_t.index <= str(date)]
        if prior.empty:
            return default
        return float(prior.iloc[-1][col])

    for side, prefix in [("home_team", "home"), ("away_team", "away")]:
        matches[f"{prefix}_avg_scored"] = [
            get_stat(row[side], row["date"], "avg_scored") for _, row in matches.iterrows()
        ]
        matches[f"{prefix}_avg_conceded"] = [
            get_stat(row[side], row["date"], "avg_conceded") for _, row in matches.iterrows()
        ]
        matches[f"{prefix}_form_score"] = [
            get_stat(row[side], row["date"], "form_score") for _, row in matches.iterrows()
        ]

    return matches
