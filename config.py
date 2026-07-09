"""
config.py — Central configuration for WC2026 prediction system.
All paths, hyperparameters, feature lists, and tournament data live here.
"""

from pathlib import Path

# ── Directories ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "saved_models"

for d in [DATA_RAW, DATA_PROCESSED, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data Source URLs ────────────────────────────────────────────────────────────
DATA_SOURCES = {
    # Primary: 49K+ international results (download manually from Kaggle or use API)
    "international_results": "https://raw.githubusercontent.com/martj42/international-results/master/results.csv",
    # openfootball WC fixtures (parsed separately per year)
    "openfootball_wc": "https://raw.githubusercontent.com/openfootball/worldcup/master/2022/worldcup.json",
    # World Bank GDP/capita (JSON API)
    "worldbank_gdp": "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.CD?format=json&per_page=300&mrv=1",
    # World Bank Population
    "worldbank_pop": "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&per_page=300&mrv=1",
    # Open-Meteo historical weather API (called per venue per date)
    "open_meteo": "https://archive-api.open-meteo.com/v1/archive",
    # GDELT (requires separate handling — see features/unconventional.py)
    "gdelt_base": "http://data.gdeltproject.org/events/",
}

# ── Tournament Constants ────────────────────────────────────────────────────────
WC_YEARS = [2014, 2018, 2022, 2026]
TRAIN_CUTOFF_DATES = {
    2014: "2014-06-11",
    2018: "2018-06-14",
    2022: "2022-11-20",
    2026: "2026-06-11",  # tournament start; group stage now done (2026-06-27)
}
GROUP_STAGE_END_2026 = "2026-06-27"
ROUND32_START_2026 = "2026-06-28"

# ── 2026 Tournament Structure ───────────────────────────────────────────────────
GROUPS_2026 = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Actual 2026 group stage results (for benchmarking — now complete)
# Format: (home, away, home_goals, away_goals, date)
GROUP_RESULTS_2026 = [
    # Group A
    ("Mexico", "South Africa", 2, 0, "2026-06-11"),
    ("South Korea", "Czechia", 2, 1, "2026-06-11"),
    ("Czechia", "South Africa", 1, 1, "2026-06-18"),
    ("Mexico", "South Korea", 1, 0, "2026-06-18"),
    ("Mexico", "Czechia", 3, 0, "2026-06-24"),
    ("South Africa", "South Korea", 1, 0, "2026-06-24"),
    # Group B
    ("Canada", "Qatar", 1, 1, "2026-06-12"),  # placeholder — verify
    ("Switzerland", "Bosnia and Herzegovina", 2, 0, "2026-06-12"),  # placeholder
    # Group C
    ("Brazil", "Morocco", 2, 0, "2026-06-13"),  # placeholder
    ("Scotland", "Haiti", 2, 0, "2026-06-13"),  # placeholder
    # Group D
    ("USA", "Australia", 2, 0, "2026-06-12"),
    ("Paraguay", "Turkey", 1, 0, "2026-06-12"),
    ("USA", "Turkey", 3, 2, "2026-06-20"),  # approx — check actual
    ("Australia", "Paraguay", 0, 0, "2026-06-20"),
    # Group E
    ("Germany", "Curacao", 7, 1, "2026-06-14"),
    ("Ivory Coast", "Ecuador", 1, 0, "2026-06-14"),
    ("Germany", "Ivory Coast", 2, 1, "2026-06-20"),
    ("Ecuador", "Curacao", 0, 0, "2026-06-20"),
    ("Curacao", "Ivory Coast", 0, 2, "2026-06-25"),
    ("Ecuador", "Germany", 2, 1, "2026-06-25"),
    # Group F
    ("Netherlands", "Japan", 2, 2, "2026-06-14"),
    ("Sweden", "Tunisia", 5, 1, "2026-06-14"),
    ("Netherlands", "Sweden", 5, 1, "2026-06-20"),
    ("Japan", "Tunisia", 4, 0, "2026-06-20"),
    ("Japan", "Sweden", 1, 1, "2026-06-25"),
    ("Netherlands", "Tunisia", 3, 1, "2026-06-25"),
    # Group G
    ("Belgium", "Egypt", 1, 1, "2026-06-15"),
    ("Iran", "New Zealand", 2, 2, "2026-06-15"),
    ("Belgium", "Iran", 0, 0, "2026-06-21"),
    ("Egypt", "New Zealand", 3, 1, "2026-06-21"),
    ("Egypt", "Iran", 1, 1, "2026-06-26"),
    ("Belgium", "New Zealand", 5, 1, "2026-06-26"),
    # Group H
    ("Spain", "Cape Verde", 0, 0, "2026-06-15"),
    ("Saudi Arabia", "Uruguay", 1, 1, "2026-06-15"),
    ("Spain", "Saudi Arabia", 4, 0, "2026-06-21"),
    ("Uruguay", "Cape Verde", 2, 2, "2026-06-21"),
    ("Cape Verde", "Saudi Arabia", 0, 0, "2026-06-26"),
    ("Spain", "Uruguay", 1, 0, "2026-06-26"),
    # Group I
    ("France", "Senegal", 3, 1, "2026-06-16"),  # Olise led
    ("Iraq", "Norway", 1, 4, "2026-06-16"),      # Haaland 2 goals
    ("France", "Senegal", 3, 2, "2026-06-23"),   # placeholder
    ("Norway", "Iraq", 3, 0, "2026-06-23"),      # placeholder
    ("France", "Norway", 4, 1, "2026-06-26"),
    ("Senegal", "Iraq", 5, 0, "2026-06-26"),
    # Group J
    ("Argentina", "Algeria", 3, 0, "2026-06-15"),  # Messi hat-trick
    ("Austria", "Jordan", 2, 0, "2026-06-15"),
    # Group K
    ("Portugal", "Uzbekistan", 5, 0, "2026-06-17"),  # Ronaldo 2 goals
    ("Colombia", "DR Congo", 1, 0, "2026-06-17"),
    # Group L
    ("England", "Ghana", 0, 0, "2026-06-17"),
    ("Croatia", "Panama", 1, 0, "2026-06-17"),
    ("England", "Panama", 2, 0, "2026-06-23"),  # Kane + Bellingham
    ("Croatia", "Ghana", 2, 1, "2026-06-23"),
]

# Confirmed 2026 group standings (final)
GROUP_QUALIFIERS_2026 = {
    "A": {"1st": "Mexico", "2nd": "South Africa", "3rd": "South Korea"},
    "B": {"1st": "Switzerland", "2nd": "Canada", "3rd": "Qatar"},
    "C": {"1st": "Brazil", "2nd": "Morocco", "3rd": "Scotland"},
    "D": {"1st": "USA", "2nd": "Australia", "3rd": "Paraguay"},
    "E": {"1st": "Germany", "2nd": "Ivory Coast", "3rd": "Ecuador"},
    "F": {"1st": "Netherlands", "2nd": "Japan", "3rd": "Sweden"},
    "G": {"1st": "Belgium", "2nd": "Egypt", "3rd": "Senegal"},
    "H": {"1st": "Spain", "2nd": "Uruguay", "3rd": "Cape Verde"},
    "I": {"1st": "France", "2nd": "Norway", "3rd": "Senegal"},  # Senegal as best 3rd
    "J": {"1st": "Argentina", "2nd": "Austria", "3rd": "Algeria"},
    "K": {"1st": "Portugal", "2nd": "Colombia", "3rd": None},
    "L": {"1st": "England", "2nd": "Croatia", "3rd": None},
}

# Round of 32 bracket (as confirmed — some already played as of June 29)
ROUND32_2026 = [
    # June 28
    ("South Africa", "Canada"),         # Group A 2nd vs Group B 2nd
    # June 29
    ("Brazil", "Japan"),                # Group C 1st vs Group F 2nd
    ("Germany", "Paraguay"),            # Group E 1st vs Group D 3rd/best-3rd
    ("Netherlands", "Morocco"),         # Group F 1st vs Group C 2nd
    # June 30
    ("Ivory Coast", "Norway"),          # Group E 2nd vs Group I 2nd
    ("France", "Sweden"),               # Group I 1st vs Group F 3rd
    ("Mexico", "Ecuador"),              # Group A 1st vs Group E 3rd
    # July 1
    ("England", "DR Congo"),            # Group L 1st vs Group K 2nd (placeholder)
    ("Belgium", "Senegal"),             # Group G 1st vs best 3rd
    ("USA", "Bosnia and Herzegovina"),  # Group D 1st vs Group B 3rd
    # July 2
    ("Spain", "Austria"),               # Group H 1st vs Group J 2nd
    ("Argentina", "Cape Verde"),        # Group J 1st vs Group H 3rd
    # July 3
    ("Switzerland", "Algeria"),         # Group B 1st vs Group J 3rd
    ("Portugal", "Croatia"),            # Group K 1st vs Group L 2nd
    ("Australia", "Egypt"),             # Group D 2nd vs Group G 2nd
    ("Colombia", "Scotland"),           # Group K... placeholder
]

# ── Venue Data ──────────────────────────────────────────────────────────────────
VENUES_2026 = {
    "Mexico City":       {"lat": 19.43, "lon": -99.13, "alt_m": 2240, "country": "Mexico"},
    "Guadalajara":       {"lat": 20.66, "lon": -103.35, "alt_m": 1566, "country": "Mexico"},
    "Monterrey":         {"lat": 25.67, "lon": -100.31, "alt_m": 538,  "country": "Mexico"},
    "Los Angeles":       {"lat": 34.01, "lon": -118.29, "alt_m": 88,   "country": "USA"},
    "San Francisco":     {"lat": 37.77, "lon": -122.42, "alt_m": 16,   "country": "USA"},
    "Dallas":            {"lat": 32.78, "lon": -96.80,  "alt_m": 130,  "country": "USA"},
    "Houston":           {"lat": 29.76, "lon": -95.37,  "alt_m": 15,   "country": "USA"},
    "Kansas City":       {"lat": 39.10, "lon": -94.58,  "alt_m": 265,  "country": "USA"},
    "Atlanta":           {"lat": 33.75, "lon": -84.39,  "alt_m": 320,  "country": "USA"},
    "Miami":             {"lat": 25.77, "lon": -80.19,  "alt_m": 2,    "country": "USA"},
    "Philadelphia":      {"lat": 39.95, "lon": -75.17,  "alt_m": 12,   "country": "USA"},
    "Boston":            {"lat": 42.36, "lon": -71.06,  "alt_m": 9,    "country": "USA"},
    "New York":          {"lat": 40.71, "lon": -74.01,  "alt_m": 10,   "country": "USA"},
    "Seattle":           {"lat": 47.61, "lon": -122.33, "alt_m": 56,   "country": "USA"},
    "Toronto":           {"lat": 43.65, "lon": -79.38,  "alt_m": 76,   "country": "Canada"},
    "Vancouver":         {"lat": 49.25, "lon": -123.12, "alt_m": 70,   "country": "Canada"},
}

# ── Team Metadata ───────────────────────────────────────────────────────────────
# Country capitals for travel distance calculation
TEAM_CAPITALS = {
    "Mexico":         {"lat": 19.43, "lon": -99.13},
    "South Africa":   {"lat": -25.75, "lon": 28.19},
    "South Korea":    {"lat": 37.57, "lon": 126.98},
    "Czechia":        {"lat": 50.08, "lon": 14.44},
    "Canada":         {"lat": 45.42, "lon": -75.69},
    "Bosnia and Herzegovina": {"lat": 43.85, "lon": 18.39},
    "Qatar":          {"lat": 25.29, "lon": 51.53},
    "Switzerland":    {"lat": 46.95, "lon": 7.45},
    "Brazil":         {"lat": -15.78, "lon": -47.93},
    "Morocco":        {"lat": 34.02, "lon": -6.85},
    "Haiti":          {"lat": 18.54, "lon": -72.34},
    "Scotland":       {"lat": 55.95, "lon": -3.19},
    "USA":            {"lat": 38.91, "lon": -77.04},
    "Paraguay":       {"lat": -25.29, "lon": -57.65},
    "Australia":      {"lat": -35.28, "lon": 149.13},
    "Turkey":         {"lat": 39.93, "lon": 32.86},
    "Germany":        {"lat": 52.52, "lon": 13.41},
    "Curacao":        {"lat": 12.11, "lon": -68.93},
    "Ivory Coast":    {"lat": 5.35, "lon": -4.01},
    "Ecuador":        {"lat": -0.22, "lon": -78.51},
    "Netherlands":    {"lat": 52.37, "lon": 4.90},
    "Japan":          {"lat": 35.68, "lon": 139.69},
    "Sweden":         {"lat": 59.33, "lon": 18.07},
    "Tunisia":        {"lat": 36.82, "lon": 10.18},
    "Belgium":        {"lat": 50.85, "lon": 4.35},
    "Egypt":          {"lat": 30.06, "lon": 31.25},
    "Iran":           {"lat": 35.69, "lon": 51.39},
    "New Zealand":    {"lat": -41.29, "lon": 174.78},
    "Spain":          {"lat": 40.42, "lon": -3.70},
    "Cape Verde":     {"lat": 14.93, "lon": -23.51},
    "Saudi Arabia":   {"lat": 24.69, "lon": 46.72},
    "Uruguay":        {"lat": -34.90, "lon": -56.19},
    "France":         {"lat": 48.85, "lon": 2.35},
    "Senegal":        {"lat": 14.72, "lon": -17.47},
    "Iraq":           {"lat": 33.34, "lon": 44.40},
    "Norway":         {"lat": 59.91, "lon": 10.75},
    "Argentina":      {"lat": -34.61, "lon": -58.38},
    "Algeria":        {"lat": 36.74, "lon": 3.06},
    "Austria":        {"lat": 48.21, "lon": 16.37},
    "Jordan":         {"lat": 31.96, "lon": 35.95},
    "Portugal":       {"lat": 38.72, "lon": -9.14},
    "DR Congo":       {"lat": -4.32, "lon": 15.32},
    "Uzbekistan":     {"lat": 41.30, "lon": 69.27},
    "Colombia":       {"lat": 4.71, "lon": -74.07},
    "England":        {"lat": 51.51, "lon": -0.13},
    "Croatia":        {"lat": 45.81, "lon": 15.98},
    "Ghana":          {"lat": 5.56, "lon": -0.20},
    "Panama":         {"lat": 8.99, "lon": -79.52},
}

# ── Model Hyperparameters ───────────────────────────────────────────────────────
ELO_CONFIG = {
    "k_factor": 30,
    "init_rating": 1500,
    "home_advantage": 0,  # neutral ground for WC
    "tournament_weight": 1.5,  # WC matches count more than friendlies
    "confederation_weight": {
        "WC": 1.5,
        "CONF": 1.2,
        "QUAL": 1.1,
        "FRIENDLY": 0.8,
    },
}

XGBOOST_CONFIG = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "early_stopping_rounds": 50,
    "objective": "multi:softprob",
    "num_class": 3,
    "random_state": 42,
}

NEURAL_NET_CONFIG = {
    "hidden_dims": [64, 32, 16],
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 200,
    "batch_size": 64,
    "patience": 20,  # early stopping
}

MONTE_CARLO_SIMS = 50_000
RANDOM_SEED = 42

# ── Feature Groups (for ablation study) ────────────────────────────────────────
CONVENTIONAL_FEATURES = [
    "elo_rating", "colley_rating", "pagerank_rating", "fifa_ranking_points",
    "avg_goals_scored_10", "avg_goals_conceded_10", "xg_for_10", "xg_against_10",
    "win_pct_5", "form_score", "squad_market_value_log", "avg_squad_age",
    "strength_gap", "combined_strength", "is_knockout", "stage_importance",
]

UNCONVENTIONAL_FEATURES = [
    "gdp_per_capita_log", "population_log", "is_host_nation",
    "temperature_celsius", "altitude_m", "travel_distance_km",
    "days_rest", "media_pressure_index", "referee_same_conf",
]

ALL_FEATURES = CONVENTIONAL_FEATURES + UNCONVENTIONAL_FEATURES

# ── Confederation Mapping ───────────────────────────────────────────────────────
CONFEDERATION = {
    # UEFA
    "Germany": "UEFA", "France": "UEFA", "Spain": "UEFA", "England": "UEFA",
    "Netherlands": "UEFA", "Portugal": "UEFA", "Belgium": "UEFA", "Croatia": "UEFA",
    "Switzerland": "UEFA", "Sweden": "UEFA", "Norway": "UEFA", "Austria": "UEFA",
    "Czechia": "UEFA", "Scotland": "UEFA", "Turkey": "UEFA",
    "Bosnia and Herzegovina": "UEFA",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    # CONCACAF
    "Mexico": "CONCACAF", "USA": "CONCACAF", "Canada": "CONCACAF",
    "Panama": "CONCACAF", "Haiti": "CONCACAF", "Curacao": "CONCACAF",
    "Costa Rica": "CONCACAF",
    # CAF
    "Morocco": "CAF", "Senegal": "CAF", "Egypt": "CAF", "Ghana": "CAF",
    "South Africa": "CAF", "Tunisia": "CAF", "Algeria": "CAF", "Cape Verde": "CAF",
    "Ivory Coast": "CAF", "DR Congo": "CAF",
    # AFC
    "Japan": "AFC", "South Korea": "AFC", "Australia": "AFC", "Iran": "AFC",
    "Saudi Arabia": "AFC", "Qatar": "AFC", "Iraq": "AFC", "Jordan": "AFC",
    "Uzbekistan": "AFC",
    # OFC
    "New Zealand": "OFC",
}
# Confirmed R32 actual results
ROUND32_RESULTS_2026 = [
    ("Canada",      "South Africa",           1, 0, "regular"),
    ("Brazil",      "Japan",                  2, 1, "regular"),
    ("Paraguay",    "Germany",                1, 1, "penalties"),
    ("Morocco",     "Netherlands",            1, 1, "penalties"),
    ("Norway",      "Ivory Coast",            2, 1, "regular"),
    ("France",      "Sweden",                 3, 0, "regular"),
    ("Mexico",      "Ecuador",                2, 0, "regular"),
    ("England",     "DR Congo",               2, 1, "regular"),
    ("Belgium",     "Senegal",                3, 2, "aet"),
    ("USA",         "Bosnia and Herzegovina", 2, 0, "regular"),
    ("Spain",       "Austria",                3, 0, "regular"),
    ("Portugal",    "Croatia",                2, 1, "aet"),
    ("Switzerland", "Algeria",                2, 0, "regular"),
    ("Egypt",       "Australia",              1, 1, "penalties"),
    ("Argentina",   "Cape Verde",             3, 2, "aet"),
    ("Colombia",    "Ghana",                  1, 0, "regular"),
]

# Confirmed Round of 16 bracket
ROUND16_2026 = [
    ("Canada",      "Morocco"),
    ("France",      "Paraguay"),
    ("Brazil",      "Norway"),
    ("Mexico",      "England"),
    ("Portugal",    "Spain"),
    ("USA",         "Belgium"),
    ("Argentina",   "Egypt"),
    ("Switzerland", "Colombia"),
]

# In-tournament goals scored (boosts recent form signal)
TOURNAMENT_GOALS_2026 = {
    "France":       10,
    "Spain":         8,
    "Argentina":    11,
    "Brazil":        9,
    "England":       9,
    "Norway":       10,
    "Colombia":      5,
    "Morocco":       2,
    "Canada":        8,
    "Mexico":        7,
    "USA":           5,
    "Portugal":      9,
    "Switzerland":   7,
    "Belgium":       9,
    "Egypt":         3,
    "Paraguay":      2,
}

# Penalty shootout historical record (won, played)
PENALTY_RECORD = {
    "Argentina":    (5, 7),
    "Spain":        (4, 5),
    "France":       (3, 4),
    "Portugal":     (4, 6),
    "Brazil":       (3, 6),
    "England":      (2, 8),
    "Germany":      (4, 6),
    "Netherlands":  (2, 5),
    "Morocco":      (2, 3),
    "Egypt":        (3, 4),
    "USA":          (1, 3),
    "Mexico":       (1, 5),
    "Colombia":     (2, 3),
    "Belgium":      (2, 3),
    "Norway":       (1, 2),
    "Switzerland":  (1, 3),
    "Canada":       (0, 1),
    "Paraguay":     (2, 3),
}
# Extra time fatigue — teams that played 120 mins in R32
# Reduces effective strength by ~3% in R16
EXTRA_TIME_R32 = {
    "Belgium",      # 3-2 Senegal AET
    "Portugal",     # 2-1 Croatia AET
    "Argentina",    # 3-2 Cape Verde AET
    "Egypt",        # 1-1 Australia (pens)
    "Paraguay",     # 1-1 Germany (pens)
    "Morocco",      # 1-1 Netherlands (pens)
}

# Key player suspensions / injuries for R16
# Format: {team: [suspended/injured players, reason]}
PLAYER_FLAGS_R16 = {
    "USA": {
        "suspended": ["Folarin Balogun"],
        "reason": "Red card vs Bosnia — wrongly given but cannot appeal"
    },
    "Argentina": {
        "fitness_concern": ["Lionel Messi"],
        "reason": "Played 120 mins vs Cape Verde, age 39, fatigue risk"
    },
    "Portugal": {
        "fitness_concern": ["Cristiano Ronaldo"],
        "reason": "Played 120 mins vs Croatia, age 41, fatigue risk"
    },
    "Belgium": {
        "fitness_concern": ["Romelu Lukaku"],
        "reason": "Played 120 mins vs Senegal, scored in 86th min"
    },
    "Morocco": {
        "fitness_concern": ["Achraf Hakimi"],
        "reason": "Played 120 mins + penalties vs Netherlands"
    },
}

# Strength adjustment factors for R16
# Based on current tournament performance vs pre-tournament Elo
# Positive = performing above Elo expectation
# Negative = underperforming
FORM_ADJUSTMENT_R16 = {
    "France":       +0.08,   # Mbappé brace, Dembélé hat-trick, 10 goals — dominant
    "Morocco":      +0.06,   # Beat Netherlands on pens — massively outperforming
    "Norway":       +0.05,   # Haaland 4 goals, beating expectations
    "Paraguay":     +0.05,   # Beat Germany on pens — giant killer
    "Egypt":        +0.04,   # Penalty win vs Australia — organised
    "Canada":       +0.03,   # Historic wins, home crowd
    "Mexico":       +0.03,   # Home crowd boost in Mexico City
    "Colombia":     +0.02,   # Solid but not spectacular
    "Spain":        +0.01,   # Won but not convincing until Austria
    "Argentina":   -0.02,    # Struggled vs Cape Verde, Messi fatigue
    "USA":         -0.03,    # Balogun suspended, unconvincing vs Bosnia
    "England":     -0.02,    # Unconvincing, relying on individual moments
    "Belgium":     -0.02,    # Needed AET to beat Senegal
    "Portugal":    -0.01,    # Needed AET vs Croatia
    "Brazil":       0.00,    # Beat Japan but not convincing
    "Switzerland":  0.00,    # Solid 2-0 vs Algeria, reliable
}# Confirmed R16 actual results
ROUND16_RESULTS_2026 = [
    ("Morocco",     "Canada",     3, 0, "regular"),
    ("France",      "Paraguay",   1, 0, "regular"),
    ("Norway",      "Brazil",     2, 1, "regular"),
    ("England",     "Mexico",     3, 2, "regular"),
    ("Spain",       "Portugal",   1, 0, "regular"),
    ("Belgium",     "USA",        4, 1, "regular"),
    ("Argentina",   "Egypt",      3, 2, "regular"),
    ("Switzerland", "Colombia",   0, 0, "penalties"),  # SUI win 4-3
]

# Confirmed Quarterfinals bracket
QF_2026 = [
    ("France",    "Morocco"),
    ("Spain",     "Belgium"),
    ("Norway",    "England"),
    ("Argentina", "Switzerland"),
]

# Updated tournament goals including R16
TOURNAMENT_GOALS_R16_2026 = {
    "France":       11,   # +1 vs Paraguay (Mbappé pen)
    "Morocco":      10,   # +3 vs Canada — Ounahi brace
    "Norway":       12,   # +2 vs Brazil — Haaland dominant
    "England":      12,   # +3 vs Mexico — Kane brace
    "Spain":        9,    # +1 vs Portugal — late winner
    "Belgium":      13,   # +4 vs USA — De Ketelaere brace
    "Argentina":    14,   # +3 vs Egypt — Messi + Di Maria
    "Switzerland":  7,    # 0 vs Colombia — defensive
}

# Extra time fatigue for QF
EXTRA_TIME_QF = {
    "Switzerland",   # 0-0 vs Colombia, 120 mins + pens
    "Argentina",     # 3-2 vs Egypt, came from behind twice
}

# QF form adjustments
FORM_ADJUSTMENT_QF = {
    "Morocco":      +0.10,  # 3-0 vs Canada — best performance of tournament
    "Norway":       +0.10,  # Beat Brazil — Haaland unstoppable
    "France":       +0.06,  # Grinding wins — efficient
    "Belgium":      +0.05,  # 4-1 vs USA — De Bruyne back
    "England":      +0.04,  # 3-2 comeback vs Mexico — Kane brace
    "Spain":        +0.03,  # Won but unconvincing vs Portugal
    "Argentina":   -0.03,   # Struggled vs Egypt — Messi fatigue
    "Switzerland": -0.02,   # 0-0 vs Colombia — poor attack
}

# Penalty record updated with tournament data
PENALTY_RECORD_QF = {
    "France":       (3, 4),
    "Morocco":      (2, 3),
    "Spain":        (4, 5),
    "Belgium":      (2, 3),
    "Norway":       (1, 2),
    "England":      (2, 8),   # still historically poor
    "Argentina":    (5, 7),
    "Switzerland":  (2, 4),   # just won vs Colombia
}