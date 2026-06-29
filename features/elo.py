"""
features/elo.py — Elo, Colley, and PageRank rating systems.

All three expose the same interface:
    fit(matches_df) -> self
    get_rating(team) -> float
    predict_win_prob(team_a, team_b) -> (p_win, p_draw, p_loss)
"""

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid
import networkx as nx
from config import ELO_CONFIG, CONFEDERATION


# ── Shared draw-probability calibration ────────────────────────────────────────
def _draw_prob_from_gap(abs_strength_gap: float, intercept: float = 0.25,
                        slope: float = -0.002) -> float:
    """
    P(draw) decreases as strength gap grows.
    Fit these parameters on historical WC data via logistic regression
    (see benchmark/metrics.py). Defaults are reasonable priors.
    """
    p_draw = intercept + slope * abs_strength_gap
    return float(np.clip(p_draw, 0.05, 0.35))


def _split_probs(win_expectancy: float, abs_gap: float):
    """Convert binary win expectancy + gap → (p_win, p_draw, p_loss)."""
    p_draw = _draw_prob_from_gap(abs_gap)
    remainder = 1.0 - p_draw
    p_win = win_expectancy * remainder
    p_loss = (1.0 - win_expectancy) * remainder
    return float(p_win), float(p_draw), float(p_loss)


# ── 1. Elo ─────────────────────────────────────────────────────────────────────
class EloRating:
    """
    Elo rating adapted for international football.
    - K-factor weighted by match importance (WC > qualifier > friendly)
    - No home advantage (all WC games on neutral ground)
    - Ratings initialised at 1500; updated chronologically
    """

    def __init__(self, config: dict = None):
        self.config = config or ELO_CONFIG
        self.ratings: dict[str, float] = {}
        self.history: list[dict] = []

    def _k(self, tournament_type: str) -> float:
        weight = self.config["confederation_weight"].get(tournament_type, 1.0)
        return self.config["k_factor"] * weight

    def _win_expectancy(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _get(self, team: str) -> float:
        return self.ratings.get(team, self.config["init_rating"])

    def fit(self, matches: pd.DataFrame) -> "EloRating":
        """
        matches must have columns:
            home_team, away_team, home_score, away_score, date, tournament
        Sorted chronologically (oldest first).
        """
        matches = matches.sort_values("date").reset_index(drop=True)

        for _, row in matches.iterrows():
            a, b = row["home_team"], row["away_team"]
            ra, rb = self._get(a), self._get(b)

            if row["home_score"] > row["away_score"]:
                sa, sb = 1.0, 0.0
            elif row["home_score"] < row["away_score"]:
                sa, sb = 0.0, 1.0
            else:
                sa, sb = 0.5, 0.5

            ea = self._win_expectancy(ra, rb)
            k = self._k(row.get("tournament_type", "QUAL"))

            self.ratings[a] = ra + k * (sa - ea)
            self.ratings[b] = rb + k * (sb - (1.0 - ea))

            self.history.append({
                "date": row["date"], "home": a, "away": b,
                "rating_home_after": self.ratings[a],
                "rating_away_after": self.ratings[b],
            })

        return self

    def get_rating(self, team: str) -> float:
        return self._get(team)

    def predict_win_prob(self, team_a: str, team_b: str):
        ra, rb = self._get(team_a), self._get(team_b)
        we = self._win_expectancy(ra, rb)
        gap = abs(ra - rb)
        return _split_probs(we, gap)

    def get_all_ratings(self) -> pd.DataFrame:
        return pd.DataFrame(
            [(t, r) for t, r in self.ratings.items()],
            columns=["team", "elo_rating"]
        ).sort_values("elo_rating", ascending=False).reset_index(drop=True)


# ── 2. Colley Matrix ───────────────────────────────────────────────────────────
class ColleyRating:
    """
    Colley (2004) rating — closed-form linear system, no free parameters.
    C r = b, where C_ii = 2 + games_i, C_ij = -games_ij, b_i = 1 + (W-L)/2.
    All ratings live in [0, 1]. 0.5 is the prior.
    """

    def __init__(self):
        self.ratings: dict[str, float] = {}
        self._teams: list[str] = []

    def fit(self, matches: pd.DataFrame) -> "ColleyRating":
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        C = np.zeros((n, n))
        b = np.ones(n)  # 1 + (W - L) / 2 — initialise b_i = 1

        for _, row in matches.iterrows():
            a, b_team = row["home_team"], row["away_team"]
            i, j = idx[a], idx[b_team]
            C[i, i] += 1
            C[j, j] += 1
            C[i, j] -= 1
            C[j, i] -= 1

            hs, as_ = row["home_score"], row["away_score"]
            if hs > as_:        # home win
                b[i] += 0.5; b[j] -= 0.5
            elif hs < as_:      # away win
                b[i] -= 0.5; b[j] += 0.5
            # draw: no change to b

        # Add Laplace prior: C_ii += 2
        C += 2 * np.eye(n)

        ratings_vec = np.linalg.solve(C, b)
        self.ratings = {teams[i]: float(ratings_vec[i]) for i in range(n)}
        self._teams = teams
        return self

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, 0.5)

    def predict_win_prob(self, team_a: str, team_b: str):
        ra, rb = self.get_rating(team_a), self.get_rating(team_b)
        # Map to win expectancy via logistic on normalised gap
        gap_norm = (ra - rb) * 400  # scale to Elo-like units
        we = 1.0 / (1.0 + 10 ** (-gap_norm / 400.0))
        return _split_probs(we, abs(gap_norm))

    def get_all_ratings(self) -> pd.DataFrame:
        return pd.DataFrame(
            [(t, r) for t, r in self.ratings.items()],
            columns=["team", "colley_rating"]
        ).sort_values("colley_rating", ascending=False).reset_index(drop=True)


# ── 3. PageRank ────────────────────────────────────────────────────────────────
class PageRankRating:
    """
    Treat match results as a directed graph: loser → winner.
    A team ranked highly if other strong teams lost to it.
    Uses NetworkX's built-in PageRank (power iteration, d=0.85).
    Draws split the edge weight both ways.
    """

    def __init__(self, damping: float = 0.85):
        self.damping = damping
        self.ratings: dict[str, float] = {}

    def fit(self, matches: pd.DataFrame) -> "PageRankRating":
        G = nx.DiGraph()

        for _, row in matches.iterrows():
            a, b = row["home_team"], row["away_team"]
            hs, as_ = row["home_score"], row["away_score"]

            if not G.has_node(a): G.add_node(a)
            if not G.has_node(b): G.add_node(b)

            if hs > as_:
                # b lost → b points to a with weight 1
                if G.has_edge(b, a):
                    G[b][a]["weight"] += 1
                else:
                    G.add_edge(b, a, weight=1)
            elif hs < as_:
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)
            else:
                # draw: 0.5 weight each way
                for src, dst in [(a, b), (b, a)]:
                    if G.has_edge(src, dst):
                        G[src][dst]["weight"] += 0.5
                    else:
                        G.add_edge(src, dst, weight=0.5)

        pr = nx.pagerank(G, alpha=self.damping, weight="weight")
        self.ratings = pr
        return self

    def get_rating(self, team: str) -> float:
        if not self.ratings:
            return 1.0 / 100  # prior
        mean = np.mean(list(self.ratings.values()))
        return self.ratings.get(team, mean)

    def predict_win_prob(self, team_a: str, team_b: str):
        ra, rb = self.get_rating(team_a), self.get_rating(team_b)
        total = ra + rb
        if total == 0:
            we = 0.5
        else:
            we = ra / total
        # Scale gap to Elo-like units for draw calibration
        gap_elo = abs(ra - rb) / (np.mean(list(self.ratings.values())) + 1e-9) * 200
        return _split_probs(we, gap_elo)

    def get_all_ratings(self) -> pd.DataFrame:
        return pd.DataFrame(
            [(t, r) for t, r in self.ratings.items()],
            columns=["team", "pagerank_rating"]
        ).sort_values("pagerank_rating", ascending=False).reset_index(drop=True)


# ── Combined rating builder ─────────────────────────────────────────────────────
def build_all_ratings(matches: pd.DataFrame) -> tuple[EloRating, ColleyRating, PageRankRating]:
    """
    Fit all three rating systems on the same match history.
    Returns (elo, colley, pagerank) instances.
    """
    print("Fitting Elo ratings...")
    elo = EloRating().fit(matches)

    print("Fitting Colley matrix...")
    colley = ColleyRating().fit(matches)

    print("Fitting PageRank...")
    pr = PageRankRating().fit(matches)

    return elo, colley, pr


def get_combined_rating_df(elo: EloRating, colley: ColleyRating,
                            pr: PageRankRating) -> pd.DataFrame:
    """Merge all three ratings into one DataFrame, z-scored for comparison."""
    df_elo = elo.get_all_ratings()
    df_col = colley.get_all_ratings()
    df_pr = pr.get_all_ratings()

    df = df_elo.merge(df_col, on="team", how="outer").merge(df_pr, on="team", how="outer")

    for col in ["elo_rating", "colley_rating", "pagerank_rating"]:
        mu, sigma = df[col].mean(), df[col].std()
        df[f"{col}_z"] = (df[col] - mu) / (sigma + 1e-9)

    return df.sort_values("elo_rating", ascending=False).reset_index(drop=True)
