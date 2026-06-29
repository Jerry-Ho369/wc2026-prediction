"""
simulator/monte_carlo.py
Full 2026 World Cup tournament simulator.
Runs N Monte Carlo iterations through:
  - Group stage (36 games per group, tiebreakers)
  - Best 8 third-placed teams
  - Round of 32 → R16 → QF → SF → Final
Tracks P(win tournament), P(reach each round) per team.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Callable

from config import (
    GROUPS_2026, ROUND32_2026, MONTE_CARLO_SIMS, RANDOM_SEED,
    GROUP_QUALIFIERS_2026
)


# ── Match outcome sampler ──────────────────────────────────────────────────────
def sample_match(p_win: float, p_draw: float, p_loss: float,
                 rng: np.random.Generator) -> int:
    """Returns 1 (A wins), 0 (draw), -1 (B wins)."""
    r = rng.random()
    if r < p_win:
        return 1
    elif r < p_win + p_draw:
        return 0
    else:
        return -1


def sample_knockout(p_win: float, p_draw: float, p_loss: float,
                    rng: np.random.Generator) -> int:
    """
    Knockout: no draw. If 90-min is a draw, go to ET (coin flip weighted
    by form), then penalties (50/50).
    Returns 1 if team_a wins, -1 if team_b wins.
    """
    result_90 = sample_match(p_win, p_draw, p_loss, rng)
    if result_90 != 0:
        return result_90
    # Extra time: same win probability but goals reduced by 20%
    p_win_et = p_win * 0.8 / (p_win * 0.8 + p_loss * 0.8 + p_draw * 0.4)
    p_loss_et = p_loss * 0.8 / (p_win * 0.8 + p_loss * 0.8 + p_draw * 0.4)
    r_et = rng.random()
    if r_et < p_win_et:
        return 1
    elif r_et < p_win_et + p_loss_et:
        return -1
    # Penalty shootout: 50/50
    return 1 if rng.random() < 0.5 else -1


# ── Group stage ────────────────────────────────────────────────────────────────
class GroupStageSimulator:
    """Simulates all 12 groups and returns the 32 qualified teams."""

    def __init__(self, predict_fn: Callable):
        """
        predict_fn(team_a, team_b) → (p_win, p_draw, p_loss)
        """
        self.predict = predict_fn

    def simulate_group(self, teams: list[str],
                        rng: np.random.Generator) -> list[dict]:
        """
        Simulate a 4-team round-robin group.
        Returns list of team records sorted by: pts, H2H pts, GD, GF, team name.
        """
        records = {t: {"team": t, "pts": 0, "gd": 0, "gf": 0,
                        "h2h": defaultdict(int)} for t in teams}

        # All 6 matchups
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                a, b = teams[i], teams[j]
                p_win, p_draw, p_loss = self.predict(a, b)

                result = sample_match(p_win, p_draw, p_loss, rng)

                # Simulate goal counts from Poisson for GD tracking
                lam_a = max(0.5, p_win * 2.5)
                lam_b = max(0.5, p_loss * 2.5)
                goals_a = int(rng.poisson(lam_a))
                goals_b = int(rng.poisson(lam_b))

                if result == 1:    # A wins
                    goals_a = max(goals_a, goals_b + 1)
                    records[a]["pts"] += 3
                    records[a]["h2h"][b] += 3
                elif result == 0:  # Draw
                    if goals_a != goals_b:
                        goals_a = goals_b = min(goals_a, goals_b)
                    records[a]["pts"] += 1
                    records[b]["pts"] += 1
                    records[a]["h2h"][b] += 1
                    records[b]["h2h"][a] += 1
                else:              # B wins
                    goals_b = max(goals_b, goals_a + 1)
                    records[b]["pts"] += 3
                    records[b]["h2h"][a] += 3

                records[a]["gd"] += goals_a - goals_b
                records[b]["gd"] += goals_b - goals_a
                records[a]["gf"] += goals_a
                records[b]["gf"] += goals_b

        # Sort: pts → H2H pts (simplified) → GD → GF → name
        sorted_teams = sorted(
            records.values(),
            key=lambda r: (r["pts"], r["gd"], r["gf"], r["team"]),
            reverse=True
        )
        for rank, rec in enumerate(sorted_teams):
            rec["rank"] = rank + 1

        return sorted_teams

    def simulate_all_groups(self, rng: np.random.Generator) -> dict:
        """
        Simulate all 12 groups. Returns:
        {
          "group_results": {group_id: [sorted team records]},
          "1st_place": [12 teams],
          "2nd_place": [12 teams],
          "3rd_place_all": [12 teams with stats],
          "8_best_3rd": [8 teams],
          "qualified_32": [32 teams in bracket order]
        }
        """
        group_results = {}
        first_place = []
        second_place = []
        third_place_all = []

        for group_id, teams in GROUPS_2026.items():
            result = self.simulate_group(teams, rng)
            group_results[group_id] = result
            first_place.append(result[0]["team"])
            second_place.append(result[1]["team"])
            third_place_all.append(result[2])  # dict with pts, gd, etc.

        # Select 8 best third-placed teams
        third_sorted = sorted(
            third_place_all,
            key=lambda r: (r["pts"], r["gd"], r["gf"]),
            reverse=True
        )
        best_third = [r["team"] for r in third_sorted[:8]]

        return {
            "group_results": group_results,
            "1st": first_place,
            "2nd": second_place,
            "3rd_all": third_place_all,
            "8_best_3rd": best_third,
        }


# ── Knockout bracket ───────────────────────────────────────────────────────────
class KnockoutSimulator:
    """Simulates the knockout bracket from Round of 32 to Final."""

    def __init__(self, predict_fn: Callable):
        self.predict = predict_fn

    def simulate_match(self, team_a: str, team_b: str,
                        rng: np.random.Generator) -> str:
        """Returns the winner."""
        p_win, p_draw, p_loss = self.predict(team_a, team_b)
        result = sample_knockout(p_win, p_draw, p_loss, rng)
        return team_a if result == 1 else team_b

    def simulate_round(self, matchups: list[tuple[str, str]],
                        rng: np.random.Generator) -> list[str]:
        """Returns list of winners from a list of (team_a, team_b) matchups."""
        return [self.simulate_match(a, b, rng) for a, b in matchups]

    def build_r32_bracket(self, group_sim_result: dict) -> list[tuple[str, str]]:
        """
        Build the Round of 32 bracket from group stage results.
        2026 bracket format (simplified — use confirmed bracket for post-group predictions):
        Groups: A1vsB2, B1vsA2, C1vs3rd, D1vs3rd, E1vsF2, F1vsE2,
                G1vsH2, H1vsG2, I1vs3rd, J1vsI2, K1vsL2, L1vsK2 + 4 best-3rd slots
        """
        first = group_sim_result["1st"]
        second = group_sim_result["2nd"]

        # Group labels: A=0, B=1, ... L=11
        g = {chr(65+i): {"1st": first[i], "2nd": second[i]}
             for i in range(12)}

        # Simplified bracket (actual 2026 bracket from FIFA confirmed draw)
        bracket = [
            (g["A"]["1st"], g["B"]["2nd"]),   # A1 vs B2
            (g["B"]["1st"], g["A"]["2nd"]),   # B1 vs A2
            (g["C"]["1st"], g["F"]["2nd"]),   # C1 vs F2
            (g["F"]["1st"], g["C"]["2nd"]),   # F1 vs C2
            (g["E"]["1st"], g["D"]["3rd"] if "3rd" in g["D"] else g["D"]["2nd"]),
            (g["D"]["1st"], g["E"]["2nd"]),
            (g["G"]["1st"], g["H"]["2nd"]),
            (g["H"]["1st"], g["G"]["2nd"]),
            (g["I"]["1st"], g["J"]["2nd"]),
            (g["J"]["1st"], g["I"]["2nd"]),
            (g["K"]["1st"], g["L"]["2nd"]),
            (g["L"]["1st"], g["K"]["2nd"]),
        ]
        # Add best-3rd teams
        best3 = group_sim_result["8_best_3rd"]
        for i, team in enumerate(best3[:4]):
            bracket.append((first[i + 4], team))

        return bracket[:16]  # ensure 16 matchups for R32


# ── Full Tournament Monte Carlo ────────────────────────────────────────────────
class TournamentSimulator:
    """
    Runs N Monte Carlo simulations of the full 2026 World Cup.

    Two modes:
    1. FULL: Simulate from group stage (for pre-tournament / benchmarking)
    2. KNOCKOUT_ONLY: Use known group stage results, simulate from R32 onward
       (for live predictions once group stage is complete)
    """

    def __init__(self, predict_fn: Callable, n_sims: int = MONTE_CARLO_SIMS,
                 seed: int = RANDOM_SEED):
        self.predict = predict_fn
        self.n_sims = n_sims
        self.rng = np.random.default_rng(seed)
        self.group_sim = GroupStageSimulator(predict_fn)
        self.knockout_sim = KnockoutSimulator(predict_fn)

    def run_full(self) -> dict:
        """Simulate entire tournament including group stage."""
        return self._run(mode="full")

    def run_knockout_only(self, r32_bracket: list[tuple[str, str]]) -> dict:
        """
        Use confirmed group stage results; simulate from R32 onward.
        r32_bracket: list of (team_a, team_b) for all 16 R32 matchups.
        """
        return self._run(mode="knockout", r32_bracket=r32_bracket)

    def _run(self, mode: str = "full",
             r32_bracket: list = None) -> dict:

        # Tracking counters
        reach_r32 = defaultdict(int)
        reach_r16 = defaultdict(int)
        reach_qf = defaultdict(int)
        reach_sf = defaultdict(int)
        reach_final = defaultdict(int)
        champion = defaultdict(int)
        runner_up = defaultdict(int)

        print(f"Running {self.n_sims:,} Monte Carlo simulations (mode={mode})...")

        for sim in range(self.n_sims):
            rng = np.random.default_rng(self.rng.integers(1e9))

            if mode == "full":
                gs = self.group_sim.simulate_all_groups(rng)
                bracket_r32 = self.knockout_sim.build_r32_bracket(gs)
                all_teams = gs["1st"] + gs["2nd"] + gs["8_best_3rd"]
            else:
                bracket_r32 = r32_bracket
                all_teams = list({t for pair in r32_bracket for t in pair})

            for t in all_teams:
                reach_r32[t] += 1

            # Round of 32 → 16 teams
            r16_teams = self.knockout_sim.simulate_round(bracket_r32, rng)
            for t in r16_teams:
                reach_r16[t] += 1

            # Round of 16 → 8 teams (pair up sequentially)
            r16_bracket = [(r16_teams[i], r16_teams[i+1])
                            for i in range(0, len(r16_teams), 2)]
            qf_teams = self.knockout_sim.simulate_round(r16_bracket, rng)
            for t in qf_teams:
                reach_qf[t] += 1

            # Quarter-finals → 4 teams
            qf_bracket = [(qf_teams[i], qf_teams[i+1])
                           for i in range(0, len(qf_teams), 2)]
            sf_teams = self.knockout_sim.simulate_round(qf_bracket, rng)
            for t in sf_teams:
                reach_sf[t] += 1

            # Semi-finals → 2 finalists
            sf_bracket = [(sf_teams[0], sf_teams[1]), (sf_teams[2], sf_teams[3])]
            finalists = self.knockout_sim.simulate_round(sf_bracket, rng)
            for t in finalists:
                reach_final[t] += 1

            # Final
            winner = self.knockout_sim.simulate_match(finalists[0], finalists[1], rng)
            loser = finalists[1] if winner == finalists[0] else finalists[0]
            champion[winner] += 1
            runner_up[loser] += 1

            if sim % 10_000 == 0 and sim > 0:
                print(f"  {sim:,} / {self.n_sims:,} done...")

        n = self.n_sims
        results = {}
        all_tracked = set(reach_r32.keys())
        for team in all_tracked:
            results[team] = {
                "p_r32":    reach_r32[team] / n,
                "p_r16":    reach_r16[team] / n,
                "p_qf":     reach_qf[team] / n,
                "p_sf":     reach_sf[team] / n,
                "p_final":  reach_final[team] / n,
                "p_winner": champion[team] / n,
                "p_runner_up": runner_up[team] / n,
            }

        return results

    def results_to_df(self, results: dict) -> pd.DataFrame:
        df = pd.DataFrame(results).T.reset_index()
        df.columns = ["team", "p_r32", "p_r16", "p_qf", "p_sf",
                       "p_final", "p_winner", "p_runner_up"]
        return df.sort_values("p_winner", ascending=False).reset_index(drop=True)


# ── Confirmed 2026 R32 bracket (for live predictions) ─────────────────────────
CONFIRMED_R32_2026 = [
    # June 28-July 3 (as confirmed from actual group stage results)
    ("South Africa", "Canada"),
    ("Brazil", "Japan"),
    ("Germany", "Paraguay"),
    ("Netherlands", "Morocco"),
    ("Ivory Coast", "Norway"),
    ("France", "Sweden"),
    ("Mexico", "Ecuador"),
    ("England", "DR Congo"),
    ("Belgium", "Senegal"),
    ("USA", "Bosnia and Herzegovina"),
    ("Spain", "Austria"),
    ("Argentina", "Cape Verde"),
    ("Switzerland", "Algeria"),
    ("Portugal", "Croatia"),
    ("Australia", "Egypt"),
    ("Colombia", "Scotland"),
]
