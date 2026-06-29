"""
data/download_data.py
Downloads all free datasets needed for the WC2026 prediction system.
Run this first before anything else.

Datasets downloaded:
  1. Kaggle: International football results (martj42) — 49K+ matches
  2. Kaggle: FIFA WC 2026 live dataset (squads, xG, events)
  3. Kaggle: FIFA WC team dataset (pre-tournament features)
  4. World Bank: GDP/capita + Population (via API — handled in unconventional.py)
  5. Open-Meteo: Temperature (fetched on-demand per match)

For Kaggle datasets you need:
  - kaggle CLI: pip install kaggle
  - API key at ~/.kaggle/kaggle.json
  OR download manually and place in data/raw/
"""

import os
import sys
import subprocess
import json
import zipfile
import requests
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_PROCESSED


def run(cmd: str, desc: str = ""):
    print(f"\n{'─'*50}")
    print(f"Running: {desc or cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARN: {result.stderr.strip()[:200]}")
    else:
        print(f"  OK")
    return result.returncode == 0


def kaggle_download(dataset: str, filename: str = None) -> bool:
    """Download a Kaggle dataset to DATA_RAW."""
    cmd = f"kaggle datasets download -d {dataset} -p {DATA_RAW} --unzip"
    success = run(cmd, f"Kaggle: {dataset}")
    if not success:
        print(f"  → Manual download: https://www.kaggle.com/datasets/{dataset}")
        print(f"    Place files in: {DATA_RAW}/")
    return success


def download_github_csv(url: str, out_name: str) -> bool:
    """Download a raw CSV from GitHub."""
    out_path = DATA_RAW / out_name
    if out_path.exists():
        print(f"  Already exists: {out_name}")
        return True
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        print(f"  Downloaded: {out_name} ({len(resp.content)//1024} KB)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def check_kaggle_auth() -> bool:
    """Check if kaggle CLI is configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n⚠  Kaggle API key not found at ~/.kaggle/kaggle.json")
        print("   Steps to set up:")
        print("   1. Go to https://www.kaggle.com/settings → API → Create New Token")
        print("   2. Save kaggle.json to ~/.kaggle/")
        print("   3. chmod 600 ~/.kaggle/kaggle.json")
        print("   OR: Download datasets manually and place in data/raw/\n")
        return False
    return True


def create_manual_fallback_csvs():
    """
    Create minimal placeholder CSVs so the pipeline can run
    even if Kaggle download fails. These contain the 2026 group stage
    results we already know from search results.
    """
    import pandas as pd
    from config import GROUP_RESULTS_2026

    # Minimal international results fallback (just 2026 group stage)
    records = []
    for home, away, h_goals, a_goals, date in GROUP_RESULTS_2026:
        records.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_score": h_goals,
            "away_score": a_goals,
            "tournament": "FIFA World Cup",
            "neutral": True,
        })

    df = pd.DataFrame(records)
    fallback_path = DATA_RAW / "results_fallback_2026.csv"
    df.to_csv(fallback_path, index=False)
    print(f"  Created fallback CSV: {fallback_path} ({len(df)} rows)")
    return fallback_path


def find_results_csv() -> Path:
    """Find the international results CSV in data/raw/."""
    candidates = [
        DATA_RAW / "results.csv",
        DATA_RAW / "international_football_results.csv",
        DATA_RAW / "matches_1930_2022.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    print("=" * 60)
    print("WC2026 Prediction System — Data Download")
    print("=" * 60)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    kaggle_ok = check_kaggle_auth()

    # ── 1. International results (primary training data) ───────────────────────
    print("\n[1/4] International football results (49K+ matches)...")
    results_path = find_results_csv()

    if results_path:
        print(f"  Found existing: {results_path.name}")
    elif kaggle_ok:
        kaggle_download("martj42/international-football-results-from-1872-to-2017",
                        "results.csv")
        results_path = find_results_csv()
    else:
        # Try direct GitHub download
        print("  Trying direct GitHub download...")
        success = download_github_csv(
            "https://raw.githubusercontent.com/martj42/international-results/master/results.csv",
            "results.csv"
        )
        if success:
            results_path = DATA_RAW / "results.csv"

    if not results_path or not results_path.exists():
        print("  ⚠ Creating fallback with 2026 group stage data only")
        results_path = create_manual_fallback_csvs()

    # ── 2. FIFA WC 2026 live dataset ───────────────────────────────────────────
    print("\n[2/4] FIFA WC 2026 live dataset (squads, xG, events)...")
    wc2026_path = DATA_RAW / "fifa-world-cup-2026-dataset"
    if wc2026_path.exists():
        print(f"  Found existing directory")
    elif kaggle_ok:
        kaggle_download("mominullptr/fifa-world-cup-2026-dataset")
    else:
        print("  ⚠ Skipping — download manually from:")
        print("    https://www.kaggle.com/datasets/mominullptr/fifa-world-cup-2026-dataset")

    # ── 3. WC team-level dataset (pre-tournament features) ────────────────────
    print("\n[3/4] FIFA WC team dataset (market values, rankings, ages)...")
    team_dataset_path = DATA_RAW / "fifa-world-cup-team-dataset"
    if team_dataset_path.exists():
        print(f"  Found existing directory")
    elif kaggle_ok:
        kaggle_download("harrachimustapha/fifa-world-cup-team-dataset")
    else:
        print("  ⚠ Skipping — download manually from:")
        print("    https://www.kaggle.com/datasets/harrachimustapha/fifa-world-cup-team-dataset")

    # ── 4. WC historical match data (Fjelstul — includes referees) ────────────
    print("\n[4/4] Historical WC data with referee info...")
    fjelstul_path = DATA_RAW / "worldcup_fjelstul"
    if fjelstul_path.exists():
        print("  Found existing")
    else:
        fjelstul_path.mkdir(exist_ok=True)
        files = [
            ("matches", "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/matches.csv"),
            ("goals", "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/goals.csv"),
            ("referees", "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/referees.csv"),
        ]
        for name, url in files:
            download_github_csv(url, f"worldcup_fjelstul/{name}.csv")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Download complete. Files in data/raw/:")
    for f in sorted(DATA_RAW.rglob("*.csv")):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.relative_to(DATA_RAW)}: {size_kb} KB")

    print("\nNext steps:")
    print("  python features/build_features.py   # build feature matrix")
    print("  python main.py                       # run full pipeline")
    print("=" * 60)

    return results_path


if __name__ == "__main__":
    main()
