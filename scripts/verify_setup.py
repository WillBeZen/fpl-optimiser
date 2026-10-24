"""
verify_setup.py — one-off health check for your environment and the live data
sources. Run from the repo root after installing requirements:

    python scripts/verify_setup.py

Checks config, the FPL API (xG + CBIT fields), the opponent-stats adapter (which
picks a working advanced source — FBref by default, Understat as fallback), and
FPL<->source team-name matching.
"""
from __future__ import annotations

import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

OK = "\u2713"
NO = "\u2717"


def step(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    failures = 0

    step("1. config")
    try:
        import config
        config.validate()
        print(f"{OK} config valid | train={config.SEASON_TRAIN} live={config.SEASON_LIVE} "
              f"| advanced source = {config.ADVANCED_STATS_SOURCE}")
    except Exception:
        traceback.print_exc()
        print(f"{NO} config failed — fix this first")
        return 1

    from fpl_optimiser.ingest import fpl_api, opponent_stats, team_map

    # 2 — FPL players
    step("2. FPL players")
    fpl_team_names: list[str] = []
    try:
        players = fpl_api.players()
        print(f"{OK} pulled {len(players)} players")
        for col in ["expected_goals", "expected_assists",
                    "clearances_blocks_interceptions", "defensive_contribution"]:
            n = int(players[col].notna().sum()) if col in players.columns else 0
            mark = OK if n > 0 else NO
            if n == 0:
                failures += 1
            print(f"   {mark} {col}: {n} non-null")
        top = players.sort_values("total_points", ascending=False).head(3)
        print("   top scorers:",
              ", ".join(f"{r.web_name} ({r.position}, £{r.price}m)" for r in top.itertuples()))
        fpl_team_names = sorted(players["team_name"].dropna().unique())
        print(f"   {len(fpl_team_names)} teams found")
    except Exception:
        failures += 1
        traceback.print_exc()
        print(f"{NO} FPL players failed")

    # 3 — FPL fixtures + gameweek
    step("3. FPL fixtures + gameweek")
    try:
        fixtures = fpl_api.fixtures()
        gw = fpl_api.current_gameweek()
        print(f"{OK} {len(fixtures)} fixtures | current/next gameweek = {gw}")
    except Exception:
        failures += 1
        traceback.print_exc()
        print(f"{NO} FPL fixtures failed")

    # 4 — opponent stats via the adapter (auto-selects a working source)
    step("4. opponent stats (advanced; auto-selects source)")
    opp_team_names: list[str] = []
    try:
        opp = opponent_stats.opponent_stats(config.SEASON_TRAIN)
        src = opp.attrs.get("source", "?")
        print(f"{OK} via '{src}' | {len(opp)} teams | columns: {list(opp.columns)}")
        have_xga = "xga_per_game" in opp.columns and opp["xga_per_game"].notna().any()
        sort_col = "xga_per_game" if have_xga else "goals_conceded_per_game"
        label = "xGA/game" if have_xga else "goals conceded/game"
        cols = [c for c in ["team", "matches", "xga_per_game",
                            "goals_conceded_per_game", "ppda_mean"] if c in opp.columns]
        print(f"   leakiest 3 by {label}:")
        print(opp.sort_values(sort_col, ascending=False).head(3)[cols].to_string(index=False))
        opp_team_names = sorted(opp["team"].dropna().unique())
        print(f"   {len(opp_team_names)} teams")
    except Exception:
        failures += 1
        traceback.print_exc()
        print(f"{NO} opponent stats failed — paste the column list above if it's a KeyError")

    # 5 — reconciliation FPL <-> opponent source
    step("5. team-name reconciliation FPL <-> opponent source")
    if fpl_team_names and opp_team_names:
        rec = team_map.reconcile(fpl_team_names, opp_team_names)
        print(f"{OK} matched {len(rec['matched'])}/{len(fpl_team_names)}")
        print("   unmatched (FPL):   ", rec["fpl_only"])
        print("   unmatched (source):", rec["fbref_only"])
        if rec["fpl_only"] or rec["fbref_only"]:
            failures += 1
            print("   -> add aliases in fpl_optimiser/ingest/team_map.py for these")
    else:
        print("   (skipped — needed both team lists from steps 2 and 4)")

    step("summary")
    if failures == 0:
        print(f"{OK} all good — paste steps 4-5 back and we'll start Section 2.")
    else:
        print(f"{NO} {failures} check(s) need attention above.")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())