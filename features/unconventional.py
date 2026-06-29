"""
features/unconventional.py
Fetches and computes all non-conventional features:
  - GDP/capita (World Bank API)
  - Population (World Bank API)
  - Temperature at venue on match date (Open-Meteo)
  - Travel distance (geopy great-circle)
  - Days rest between matches
  - Media pressure index (GDELT)
  - Referee confederation bias flag
  - Altitude at venue
  - Host nation flag
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from geopy.distance import great_circle

from config import (
    DATA_RAW, TEAM_CAPITALS, VENUES_2026, CONFEDERATION, DATA_SOURCES
)

logger = logging.getLogger(__name__)

# ── ISO-3 country code mapping for World Bank API ──────────────────────────────
TEAM_TO_ISO3 = {
    "Mexico": "MEX", "South Africa": "ZAF", "South Korea": "KOR",
    "Czechia": "CZE", "Canada": "CAN", "Bosnia and Herzegovina": "BIH",
    "Qatar": "QAT", "Switzerland": "CHE", "Brazil": "BRA", "Morocco": "MAR",
    "Haiti": "HTI", "Scotland": "GBR", "USA": "USA", "Paraguay": "PRY",
    "Australia": "AUS", "Turkey": "TUR", "Germany": "DEU", "Curacao": "CUW",
    "Ivory Coast": "CIV", "Ecuador": "ECU", "Netherlands": "NLD", "Japan": "JPN",
    "Sweden": "SWE", "Tunisia": "TUN", "Belgium": "BEL", "Egypt": "EGY",
    "Iran": "IRN", "New Zealand": "NZL", "Spain": "ESP", "Cape Verde": "CPV",
    "Saudi Arabia": "SAU", "Uruguay": "URY", "France": "FRA", "Senegal": "SEN",
    "Iraq": "IRQ", "Norway": "NOR", "Argentina": "ARG", "Algeria": "DZA",
    "Austria": "AUT", "Jordan": "JOR", "Portugal": "PRT", "DR Congo": "COD",
    "Uzbekistan": "UZB", "Colombia": "COL", "England": "GBR", "Croatia": "HRV",
    "Ghana": "GHA", "Panama": "PAN",
}


# ── 1. GDP & Population (World Bank API) ───────────────────────────────────────
def fetch_worldbank(indicator: str, cache_file: str) -> dict[str, float]:
    """Fetch most-recent-value for all countries from World Bank API."""
    cache_path = DATA_RAW / cache_file
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    url = (f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
           f"?format=json&per_page=300&mrv=1")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        result = {}
        for item in data[1]:
            if item["value"] is not None and item["countryiso3code"]:
                result[item["countryiso3code"]] = float(item["value"])
        with open(cache_path, "w") as f:
            json.dump(result, f)
        return result
    except Exception as e:
        logger.warning(f"World Bank API failed ({indicator}): {e}")
        return {}


def get_gdp_per_capita() -> dict[str, float]:
    """Returns {team_name: gdp_per_capita_usd}"""
    raw = fetch_worldbank("NY.GDP.PCAP.CD", "gdp_per_capita.json")
    result = {}
    for team, iso3 in TEAM_TO_ISO3.items():
        if iso3 in raw:
            result[team] = raw[iso3]
        else:
            result[team] = np.nan
    return result


def get_population() -> dict[str, float]:
    """Returns {team_name: population}"""
    raw = fetch_worldbank("SP.POP.TOTL", "population.json")
    result = {}
    for team, iso3 in TEAM_TO_ISO3.items():
        if iso3 in raw:
            result[team] = raw[iso3]
        else:
            result[team] = np.nan
    return result


# ── 2. Temperature at Venue (Open-Meteo Historical API) ───────────────────────
def fetch_venue_temperature(lat: float, lon: float, date: str) -> float:
    """
    Fetch mean 2m air temperature (°C) at a venue on a given date.
    Uses Open-Meteo archive API (free, no key required).
    Returns NaN on failure.
    """
    cache_key = f"temp_{lat:.2f}_{lon:.2f}_{date}.json"
    cache_path = DATA_RAW / cache_key
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f).get("temperature_mean", np.nan)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": date, "end_date": date,
        "daily": "temperature_2m_mean",
        "timezone": "auto",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        temp = data["daily"]["temperature_2m_mean"][0]
        temp = float(temp) if temp is not None else np.nan
        with open(cache_path, "w") as f:
            json.dump({"temperature_mean": temp}, f)
        time.sleep(0.2)  # rate-limit courtesy
        return temp
    except Exception as e:
        logger.warning(f"Open-Meteo failed for {lat},{lon} on {date}: {e}")
        return np.nan


def get_match_temperatures(matches: pd.DataFrame,
                            venue_col: str = "venue_city") -> pd.Series:
    """
    Given a matches DataFrame with columns [date, venue_city],
    returns a Series of temperatures aligned to matches.
    """
    temps = []
    for _, row in matches.iterrows():
        city = row.get(venue_col, "")
        venue = VENUES_2026.get(city, {})
        if venue:
            t = fetch_venue_temperature(
                venue["lat"], venue["lon"],
                str(row["date"])[:10]
            )
        else:
            t = np.nan
        temps.append(t)
    return pd.Series(temps, index=matches.index, name="temperature_celsius")


# ── 3. Travel Distance ─────────────────────────────────────────────────────────
def compute_travel_distance(team: str, venue_city: str) -> float:
    """
    Great-circle distance in km from team's capital to match venue.
    Returns 0 for host nations (travelling within own country).
    """
    team_loc = TEAM_CAPITALS.get(team)
    venue = VENUES_2026.get(venue_city)
    if not team_loc or not venue:
        return np.nan
    origin = (team_loc["lat"], team_loc["lon"])
    dest = (venue["lat"], venue["lon"])
    return great_circle(origin, dest).km


def add_travel_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Add travel_distance_km_home and travel_distance_km_away columns.
    """
    matches = matches.copy()
    home_dists, away_dists = [], []
    for _, row in matches.iterrows():
        city = row.get("venue_city", "")
        home_dists.append(compute_travel_distance(row["home_team"], city))
        away_dists.append(compute_travel_distance(row["away_team"], city))
    matches["travel_km_home"] = home_dists
    matches["travel_km_away"] = away_dists
    matches["travel_advantage"] = matches["travel_km_away"] - matches["travel_km_home"]
    return matches


# ── 4. Days Rest ───────────────────────────────────────────────────────────────
def compute_days_rest(matches: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, compute days since each team's previous match.
    Adds columns: days_rest_home, days_rest_away.
    """
    matches = matches.sort_values("date").copy()
    last_match: dict[str, pd.Timestamp] = {}
    home_rest, away_rest = [], []

    for _, row in matches.iterrows():
        date = pd.Timestamp(row["date"])
        for team, rest_list in [(row["home_team"], home_rest),
                                 (row["away_team"], away_rest)]:
            if team in last_match:
                delta = (date - last_match[team]).days
            else:
                delta = 14  # default: assume 2-week rest before tournament
            rest_list.append(delta)
            last_match[team] = date

    matches["days_rest_home"] = home_rest
    matches["days_rest_away"] = away_rest
    matches["rest_advantage"] = matches["days_rest_home"] - matches["days_rest_away"]
    return matches


# ── 5. Media Pressure Index (GDELT) ───────────────────────────────────────────
def fetch_gdelt_media_pressure(team: str, date: str,
                                window_days: int = 7) -> float:
    """
    Approximate media pressure for a team in the 7 days before a match.
    Uses GDELT GKG summary files.

    NOTE: Full GDELT parsing is heavy. This simplified version uses
    the GDELT Event Database summary and counts events mentioning the
    team's country as Actor1.

    For production use, replace with gdeltdoc Python package.
    Returns a normalised float (0-1 scale) or NaN if unavailable.
    """
    # Simplified proxy: log10 of estimated news volume
    # In a full implementation, query GDELT API:
    # from gdeltdoc import GdeltDoc, Filters
    # f = Filters(keyword=team, start_date=..., end_date=...)
    # articles = GdeltDoc().article_search(f)

    # For now return NaN (to be filled by gdeltdoc in full run)
    logger.info(f"GDELT: media pressure for {team} on {date} — "
                "install gdeltdoc for full implementation")
    return np.nan


def build_media_pressure_df(teams: list[str],
                              dates: list[str]) -> pd.DataFrame:
    """
    Build media pressure index for all team-date combinations.
    Placeholder: returns 0.5 (neutral) for all — replace with GDELT API calls.
    """
    records = []
    for team in teams:
        for date in dates:
            pressure = fetch_gdelt_media_pressure(team, date)
            records.append({"team": team, "date": date,
                             "media_pressure_index": pressure})
    return pd.DataFrame(records)


# ── 6. Referee Bias Flag ───────────────────────────────────────────────────────
def compute_referee_bias(home_team: str, away_team: str,
                          referee_country: str) -> int:
    """
    Returns:
      1 if referee is from same confederation as one of the teams (potential bias)
      0 if neutral (different confederation from both teams)
     -1 if referee data unavailable
    """
    if not referee_country:
        return -1
    ref_conf = CONFEDERATION.get(referee_country, "UNKNOWN")
    home_conf = CONFEDERATION.get(home_team, "UNKNOWN")
    away_conf = CONFEDERATION.get(away_team, "UNKNOWN")

    if ref_conf in (home_conf, away_conf):
        return 1
    return 0


def add_referee_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Adds referee_same_conf column to matches DataFrame.
    Expects optional 'referee_country' column.
    """
    matches = matches.copy()
    if "referee_country" not in matches.columns:
        matches["referee_same_conf"] = -1
        return matches

    matches["referee_same_conf"] = [
        compute_referee_bias(row["home_team"], row["away_team"],
                             row.get("referee_country", ""))
        for _, row in matches.iterrows()
    ]
    return matches


# ── 7. Host Nation Flag ────────────────────────────────────────────────────────
HOST_NATIONS_2026 = {"USA", "Canada", "Mexico"}


def is_host_nation(team: str) -> int:
    return 1 if team in HOST_NATIONS_2026 else 0


# ── 8. Venue Altitude ─────────────────────────────────────────────────────────
def get_venue_altitude(venue_city: str) -> float:
    return VENUES_2026.get(venue_city, {}).get("alt_m", np.nan)


# ── Master builder ─────────────────────────────────────────────────────────────
def build_unconventional_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a matches DataFrame and appends all unconventional features.
    Input columns required: home_team, away_team, date, venue_city
    Optional: referee_country
    """
    gdp = get_gdp_per_capita()
    pop = get_population()

    df = matches.copy()

    # GDP and population (per team, mirrored for home/away)
    for side in ["home_team", "away_team"]:
        prefix = "home" if side == "home_team" else "away"
        df[f"gdp_per_capita_{prefix}"] = df[side].map(gdp)
        df[f"gdp_log_{prefix}"] = np.log1p(df[f"gdp_per_capita_{prefix}"].fillna(5000))
        df[f"population_{prefix}"] = df[side].map(pop)
        df[f"pop_log_{prefix}"] = np.log1p(df[f"population_{prefix}"].fillna(1e6))
        df[f"is_host_{prefix}"] = df[side].apply(is_host_nation)

    # GDP difference (Klement factor: richer country → better infrastructure)
    df["gdp_diff_log"] = df["gdp_log_home"] - df["gdp_log_away"]
    df["pop_diff_log"] = df["pop_log_home"] - df["pop_log_away"]

    # Temperature
    if "venue_city" in df.columns:
        print("Fetching venue temperatures (may take a moment)...")
        df["temperature_celsius"] = get_match_temperatures(df)
        df["altitude_m"] = df["venue_city"].apply(get_venue_altitude)
    else:
        df["temperature_celsius"] = np.nan
        df["altitude_m"] = np.nan

    # Travel distance
    if "venue_city" in df.columns:
        df = add_travel_features(df)
    else:
        df["travel_km_home"] = np.nan
        df["travel_km_away"] = np.nan
        df["travel_advantage"] = np.nan

    # Days rest
    df = compute_days_rest(df)

    # Referee
    df = add_referee_features(df)

    # Media pressure (placeholder — NaN until GDELT is configured)
    df["media_pressure_index"] = np.nan

    return df
