"""
features/build_features.py
Orchestrates all feature modules and builds the unified match feature matrix.
Output: data/processed/match_features.parquet
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_RAW, DATA_PROCESSED, CONFEDERATION, ALL_FEATURES,
    CONVENTIONAL_FEATURES, UNCONVENTIONAL_FEATURES, GROUP_RESULTS_2026
)
from features.elo import build_all_ratings, EloRating
from features.unconventional import build_unconventional_features
from models.poisson_model import build_team_rolling_stats


# ── Load raw data ──────────────────────────────────────────────────────────────
def load_matches(results_csv: Path = None) -> pd.DataFrame:
    """
    Load and clean the international results CSV.
    Falls back to 2026 group stage data if file not found.
    """
    if results_csv is None:
        candidates = [
            DATA_RAW / "results.csv",
            DATA_RAW / "results_fallback_2026.csv",
        ]
        for c in candidates:
            if c.exists():
                results_csv = c
                break

    if results_csv is None or not results_csv.exists():
        print("⚠ No results CSV found. Using 2026 group stage only.")
        return _make_2026_group_df()

    df = pd.read_csv(results_csv, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Normalise column names
    rename = {
        "home_team": "home_team",
        "away_team": "away_team",
        "home_score": "home_score",
        "away_score": "away_score",
    }
    for old, new in rename.items():
        if old in df.columns and old != new:
            df.rename(columns={old: new}, inplace=True)

    # Filter to reasonable date range (1990 onward — older data has less signal)
    df = df[df["date"] >= "1990-01-01"].copy()

    # Classify tournament type
    def classify(t: str) -> str:
        t = str(t).lower()
        if "fifa world cup" in t and "qualif" not in t:
            return "WC"
        elif "qualif" in t:
            return "QUAL"
        elif any(x in t for x in ["nations league", "copa", "euro", "african cup",
                                    "asian cup", "gold cup", "confederations"]):
            return "CONF"
        else:
            return "FRIENDLY"

    df["tournament_type"] = df.get("tournament", pd.Series(["FRIENDLY"]*len(df))).apply(classify)

    # Result column
    df["result"] = df.apply(
        lambda r: "W" if r["home_score"] > r["away_score"]
                  else ("D" if r["home_score"] == r["away_score"] else "L"),
        axis=1
    )

    print(f"Loaded {len(df):,} matches ({df['date'].min().date()} → "
          f"{df['date'].max().date()})")
    return df


def _make_2026_group_df() -> pd.DataFrame:
    records = []
    for home, away, hg, ag, date in GROUP_RESULTS_2026:
        records.append({
            "date": pd.Timestamp(date), "home_team": home, "away_team": away,
            "home_score": hg, "away_score": ag, "tournament": "FIFA World Cup",
            "tournament_type": "WC",
            "result": "W" if hg > ag else ("D" if hg == ag else "L"),
        })
    return pd.DataFrame(records)


# ── Build Elo ratings for each match row ───────────────────────────────────────
def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, record the Elo ratings of both teams BEFORE the match.
    Also adds strength_gap and combined_strength.
    """
    elo = EloRating()
    df = df.sort_values("date").copy()

    home_elos, away_elos = [], []
    cfg = elo.config

    for _, row in df.iterrows():
        a, b = row["home_team"], row["away_team"]
        ra = elo._get(a)
        rb = elo._get(b)
        home_elos.append(ra)
        away_elos.append(rb)

        # Update Elo after recording pre-match ratings
        if row["home_score"] > row["away_score"]:
            sa, sb = 1.0, 0.0
        elif row["home_score"] < row["away_score"]:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        k = elo._k(row.get("tournament_type", "QUAL"))
        elo.ratings[a] = ra + k * (sa - ea)
        elo.ratings[b] = rb + k * (sb - (1 - ea))

    df["elo_home"] = home_elos
    df["elo_away"] = away_elos
    df["strength_gap"] = df["elo_home"] - df["elo_away"]
    df["combined_strength"] = df["elo_home"] + df["elo_away"]
    df["elo_rating"] = df["elo_home"]  # from home perspective

    return df, elo  # return trained elo for future use


# ── Add stage importance ───────────────────────────────────────────────────────
STAGE_IMPORTANCE = {
    "group stage": 1, "group": 1,
    "round of 32": 2, "round of 16": 3,
    "quarter-final": 4, "quarter final": 4,
    "semi-final": 5, "semi final": 5,
    "final": 6, "third place": 5,
}


def add_stage_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def stage_score(t: str) -> int:
        t_lower = str(t).lower()
        for key, val in STAGE_IMPORTANCE.items():
            if key in t_lower:
                return val
        return 1  # default = group stage

    df["stage_importance"] = df.get("stage", df.get("tournament", "")).apply(stage_score)
    df["is_knockout"] = (df["stage_importance"] > 1).astype(int)
    return df


# ── Add confederation features ─────────────────────────────────────────────────
def add_confederation_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["conf_home"] = df["home_team"].map(CONFEDERATION).fillna("UNKNOWN")
    df["conf_away"] = df["away_team"].map(CONFEDERATION).fillna("UNKNOWN")
    df["same_confederation"] = (df["conf_home"] == df["conf_away"]).astype(int)
    return df


# ── FIFA ranking points (static lookup, updated before each tournament) ────────
def load_fifa_rankings(year: int = 2026) -> dict[str, float]:
    """
    Load FIFA ranking points from file if available.
    Falls back to Elo-proxy if not.
    """
    ranking_file = DATA_RAW / f"fifa_rankings_{year}.csv"
    if ranking_file.exists():
        df = pd.read_csv(ranking_file)
        if "team" in df.columns and "points" in df.columns:
            return dict(zip(df["team"], df["points"]))
    # Fallback: approximate ranking from known FIFA order (June 2026)
    # Based on pre-tournament FIFA rankings
    return {
        "France": 1849, "Spain": 1843, "England": 1794, "Brazil": 1779,
        "Argentina": 1775, "Belgium": 1771, "Portugal": 1764, "Netherlands": 1740,
        "Germany": 1720, "Italy": 1713, "Colombia": 1695, "USA": 1685,
        "Mexico": 1680, "Croatia": 1675, "Uruguay": 1665, "Morocco": 1650,
        "Senegal": 1640, "Japan": 1630, "Switzerland": 1625, "Ecuador": 1600,
        "Australia": 1590, "South Korea": 1585, "Austria": 1580, "Norway": 1560,
        "Turkey": 1555, "Sweden": 1545, "Denmark": 1540, "Poland": 1535,
        "Tunisia": 1520, "Egypt": 1515, "Algeria": 1510, "Iran": 1505,
        "Paraguay": 1490, "Canada": 1485, "Czechia": 1480, "Ghana": 1470,
        "Qatar": 1440, "Saudi Arabia": 1435, "Iraq": 1420, "South Africa": 1410,
        "Bolivia": 1400, "Ivory Coast": 1395, "Cape Verde": 1380, "Uzbekistan": 1365,
        "Jordan": 1340, "Croatia": 1675, "DR Congo": 1330, "New Zealand": 1320,
        "Colombia": 1695, "Panama": 1315, "Haiti": 1300, "Curacao": 1280,
        "Bosnia and Herzegovina": 1460, "Scotland": 1500,
    }


# ── Master feature builder ─────────────────────────────────────────────────────
def build_feature_matrix(results_csv: Path = None,
                          include_unconventional: bool = True,
                          save: bool = True) -> tuple[pd.DataFrame, EloRating]:
    """
    Full pipeline:
    1. Load matches
    2. Add Elo features (computed rolling, pre-match)
    3. Add rolling stats (goals, form)
    4. Add stage/confederation features
    5. Add FIFA rankings
    6. Optionally add unconventional features (GDP, temp, travel, etc.)
    7. Save processed parquet

    Returns: (feature_df, trained_elo)
    """
    print("\n" + "="*60)
    print("Building feature matrix...")
    print("="*60)

    # 1. Load
    df = load_matches(results_csv)

    # 2. Elo (rolling, pre-match)
    print("\n[1/6] Computing Elo ratings...")
    df, elo = add_elo_features(df)

    # 3. Rolling goal stats
    print("[2/6] Computing rolling goal stats...")
    df = build_team_rolling_stats(df, window=10)

    # 4. Stage + confederation
    print("[3/6] Adding stage and confederation features...")
    df = add_stage_features(df)
    df = add_confederation_features(df)

    # 5. FIFA rankings
    print("[4/6] Adding FIFA rankings...")
    rankings = load_fifa_rankings(2026)
    df["fifa_ranking_home"] = df["home_team"].map(rankings).fillna(1400)
    df["fifa_ranking_away"] = df["away_team"].map(rankings).fillna(1400)
    df["fifa_ranking_gap"] = df["fifa_ranking_home"] - df["fifa_ranking_away"]

    # 6. Unconventional features
    if include_unconventional:
        print("[5/6] Adding unconventional features (GDP, temp, travel)...")
        if "venue_city" not in df.columns:
            df["venue_city"] = ""  # no venue data for historical matches
        df = build_unconventional_features(df)
    else:
        print("[5/6] Skipping unconventional features (include_unconventional=False)")
        for feat in UNCONVENTIONAL_FEATURES:
            if feat not in df.columns:
                df[feat] = np.nan

    # 7. Build perspective-normalised feature rows
    # All features from home-team perspective (for ML models)
    print("[6/6] Normalising features...")
    df = _build_perspective_features(df)

    if save:
        out_path = DATA_PROCESSED / "match_features.parquet"
        df.to_parquet(out_path, index=False)
        print(f"\nSaved feature matrix: {out_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Features: {[c for c in df.columns if c in ALL_FEATURES]}")

    return df, elo


def _build_perspective_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename rolling stats to perspective-neutral names for ML.
    Adds signed 'gap' features (home - away).
    """
    df = df.copy()

    # Map rolling stats to feature names
    pairs = [
        ("home_avg_scored", "away_avg_scored", "avg_goals_scored_10", "avg_goals_allowed_10"),
        ("home_avg_conceded", "away_avg_conceded", "avg_goals_conceded_10", "avg_goals_conceded_opp_10"),
        ("home_form_score", "away_form_score", "form_home", "form_away"),
    ]
    for h_col, a_col, feat_h, feat_a in pairs:
        if h_col in df.columns:
            df[feat_h] = df[h_col]
            df[feat_a] = df[a_col]
            df[f"{feat_h}_gap"] = df[h_col] - df[a_col]

    # xG proxy: use avg_scored as xG approximation when real xG unavailable
    df["xg_for_10"] = df.get("home_avg_scored", df.get("avg_goals_scored_10", 1.5))
    df["xg_against_10"] = df.get("home_avg_conceded", df.get("avg_goals_conceded_10", 1.2))

    # Form score (0-3 scale)
    df["form_score"] = df.get("home_form_score", 1.5)
    df["win_pct_5"] = df.get("home_form_score", 1.5) / 3.0

    # Squad features (NaN if not available — filled in later from Kaggle data)
    for col in ["squad_market_value_log", "avg_squad_age"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


# ── Per-match feature dict builder (for live predictions) ─────────────────────
def get_match_features(team_a: str, team_b: str,
                        elo: EloRating,
                        feature_lookup: dict = None,
                        venue_city: str = "",
                        match_date: str = "2026-07-01",
                        stage: str = "knockout") -> dict:
    """
    Build a feature dict for a single match between team_a and team_b.
    Used at inference time (knockout predictions).

    feature_lookup: dict of precomputed per-team stats
                    {team: {avg_goals_scored_10: ..., form_score: ..., ...}}
    """
    ra = elo.get_rating(team_a)
    rb = elo.get_rating(team_b)
    rankings = load_fifa_rankings(2026)

    feats_a = (feature_lookup or {}).get(team_a, {})
    feats_b = (feature_lookup or {}).get(team_b, {})

    stage_map = {"group": 1, "knockout": 2, "r16": 3, "qf": 4, "sf": 5, "final": 6}
    stage_val = stage_map.get(stage.lower(), 2)

    row = {
        "elo_rating": ra,
        "strength_gap": ra - rb,
        "combined_strength": ra + rb,
        "fifa_ranking_gap": rankings.get(team_a, 1500) - rankings.get(team_b, 1500),
        "avg_goals_scored_10": feats_a.get("avg_scored", 1.5),
        "avg_goals_conceded_10": feats_a.get("avg_conceded", 1.2),
        "avg_goals_scored_10_gap": feats_a.get("avg_scored", 1.5) - feats_b.get("avg_scored", 1.5),
        "xg_for_10": feats_a.get("avg_scored", 1.5),
        "xg_against_10": feats_a.get("avg_conceded", 1.2),
        "form_score": feats_a.get("form_score", 1.5),
        "win_pct_5": feats_a.get("win_pct", 0.5),
        "stage_importance": stage_val,
        "is_knockout": 1 if stage_val > 1 else 0,
        "is_host_home": 1 if team_a in {"USA", "Canada", "Mexico"} else 0,
        "altitude_m": {}.get(venue_city, {}).get("alt_m", 0),
        "gdp_diff_log": 0.0,   # filled by unconventional module
        "pop_diff_log": 0.0,
        "travel_advantage": 0.0,
        "days_rest_home": 4,
        "days_rest_away": 4,
        "rest_advantage": 0,
        "temperature_celsius": 25.0,  # default summer USA temp
        "referee_same_conf": 0,
        "media_pressure_index": 0.5,
        "squad_market_value_log": np.nan,
        "avg_squad_age": 27.0,
    }

    return row


if __name__ == "__main__":
    df, elo = build_feature_matrix(include_unconventional=False)
    print(f"\nFeature matrix built successfully: {df.shape}")
    print(f"Sample columns: {list(df.columns[:15])}")
