"""
models/ensemble.py
Weighted ensemble of all 7 models.
Weights derived from inverse RPS on historical backtests.
Exposes the same predict_proba interface as individual models.
"""

import numpy as np
import pandas as pd
from typing import Callable


class WeightedEnsemble:
    """
    Combines N models via weighted average of (p_win, p_draw, p_loss).
    Default weights are equal; call set_weights() after backtesting
    to use performance-derived weights.
    """

    def __init__(self, models: dict[str, object],
                 predict_fns: dict[str, Callable] = None):
        """
        models: {"elo": EloRating, "xgboost": XGBoostMatchModel, ...}
        predict_fns: override predict functions if models use different signatures
        """
        self.models = models
        self.model_names = list(models.keys())
        self.weights = {name: 1.0 / len(models) for name in self.model_names}
        self._predict_fns = predict_fns or {}

    def set_weights(self, weights: dict[str, float]):
        """
        Set ensemble weights. Will be normalised to sum to 1.
        weights: {"elo": 0.3, "xgboost": 0.2, ...}
        """
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
        print("Ensemble weights set:")
        for name, w in sorted(self.weights.items(), key=lambda x: -x[1]):
            print(f"  {name:20s}: {w:.3f}")

    def predict_proba(self, team_a: str, team_b: str,
                       match_features: dict = None) -> tuple[float, float, float]:
        """
        Weighted ensemble prediction for team_a vs team_b.
        Returns (p_win, p_draw, p_loss) from team_a perspective.
        """
        p_wins, p_draws, p_losses = [], [], []
        model_weights = []

        for name, model in self.models.items():
            try:
                w = self.weights.get(name, 0.0)
                if w == 0:
                    continue

                # Different model types have different call signatures
                if name in ("elo", "colley", "pagerank"):
                    pw, pd_, pl = model.predict_win_prob(team_a, team_b)
                elif name in ("poisson", "negbinom"):
                    if match_features:
                        pw, pd_, pl = model.predict_win_prob(match_features, match_features)
                    else:
                        # Fall back to elo-based proxy
                        elo_model = self.models.get("elo")
                        if elo_model:
                            pw, pd_, pl = elo_model.predict_win_prob(team_a, team_b)
                        else:
                            continue
                elif name in ("xgboost", "neural_net"):
                    if match_features:
                        pw, pd_, pl = model.predict_proba(match_features)
                    else:
                        # Fall back to elo
                        elo_model = self.models.get("elo")
                        if elo_model:
                            pw, pd_, pl = elo_model.predict_win_prob(team_a, team_b)
                        else:
                            continue
                else:
                    continue

                p_wins.append(pw * w)
                p_draws.append(pd_ * w)
                p_losses.append(pl * w)
                model_weights.append(w)

            except Exception as e:
                print(f"  Warning: {name} failed for {team_a} vs {team_b}: {e}")
                continue

        if not model_weights:
            return 0.4, 0.25, 0.35  # uniform fallback

        total_w = sum(model_weights)
        p_win = sum(p_wins) / total_w
        p_draw = sum(p_draws) / total_w
        p_loss = sum(p_losses) / total_w

        # Normalise
        total = p_win + p_draw + p_loss
        return p_win / total, p_draw / total, p_loss / total

    def predict_all_models(self, team_a: str, team_b: str,
                            match_features: dict = None) -> pd.DataFrame:
        """
        Returns a DataFrame showing each model's prediction for this match.
        Useful for inspecting disagreement between models.
        """
        rows = []
        for name, model in self.models.items():
            try:
                if name in ("elo", "colley", "pagerank"):
                    pw, pd_, pl = model.predict_win_prob(team_a, team_b)
                elif name in ("xgboost", "neural_net") and match_features:
                    pw, pd_, pl = model.predict_proba(match_features)
                elif name in ("poisson", "negbinom") and match_features:
                    pw, pd_, pl = model.predict_win_prob(match_features, match_features)
                else:
                    continue

                rows.append({
                    "model": name,
                    "p_win": pw,
                    "p_draw": pd_,
                    "p_loss": pl,
                    "weight": self.weights.get(name, 0),
                    "predicted_outcome": ["W", "D", "L"][np.argmax([pw, pd_, pl])],
                })
            except Exception as e:
                rows.append({"model": name, "p_win": np.nan, "error": str(e)})

        df = pd.DataFrame(rows)
        df["team_a"] = team_a
        df["team_b"] = team_b
        return df

    def get_model_agreement(self, team_a: str, team_b: str,
                             match_features: dict = None) -> dict:
        """
        Returns a summary of model agreement/disagreement.
        High std = models disagree; low std = consensus.
        """
        df = self.predict_all_models(team_a, team_b, match_features)
        df = df.dropna(subset=["p_win"])

        return {
            "team_a": team_a,
            "team_b": team_b,
            "p_win_mean": df["p_win"].mean(),
            "p_win_std": df["p_win"].std(),
            "p_win_min": df["p_win"].min(),
            "p_win_max": df["p_win"].max(),
            "p_draw_mean": df["p_draw"].mean(),
            "p_loss_mean": df["p_loss"].mean(),
            "consensus": df["predicted_outcome"].mode()[0],
            "n_predict_win": (df["predicted_outcome"] == "W").sum(),
            "n_predict_draw": (df["predicted_outcome"] == "D").sum(),
            "n_predict_loss": (df["predicted_outcome"] == "L").sum(),
            "model_details": df.to_dict("records"),
        }


# ── Simple Elo-only predict function (fast, no feature matrix needed) ──────────
def make_elo_predict_fn(elo_model) -> Callable:
    """Returns a simple (team_a, team_b) → (p_win, p_draw, p_loss) function."""
    def predict(team_a: str, team_b: str):
        return elo_model.predict_win_prob(team_a, team_b)
    return predict


def make_ensemble_predict_fn(ensemble: WeightedEnsemble,
                               feature_lookup: dict = None) -> Callable:
    """
    Returns a (team_a, team_b) → (p_win, p_draw, p_loss) function
    backed by the full ensemble.
    feature_lookup: precomputed match features per team pair
    """
    def predict(team_a: str, team_b: str):
        key = (team_a, team_b)
        feats = (feature_lookup or {}).get(key, None)
        return ensemble.predict_proba(team_a, team_b, feats)
    return predict
