"""
verify_setup.py — one-off health check for your environment and the live data
sources. Run from the repo root after installing requirements:

    python scripts/verify_setup.py

Checks: config, the FPL API (xG + CBIT fields), Understat (the primary advanced
source — browserless), and FPL<->Understat team-name matching. FBref is probed
last and is optional (it needs Google Chrome); a skip there is fine.
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

    from fpl_optimiser.ingest import fpl_api, understat, fbref, team_map

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

    # 4 — Understat (primary advanced source)
    step("4. Understat advanced team stats (browserless)")
    us_team_names: list[str] = []
    try:
        tm = understat.team_matches(config.SEASON_TRAIN)
        print(f"{OK} team-match rows: {len(tm)} | columns: {list(tm.columns)}")
        ts = understat.team_season(config.SEASON_TRAIN)
        print(f"   season aggregates: {len(ts)} teams")
        print("   leakiest 3 by mean xGA:")
        print(ts.sort_values("xga_mean", ascending=False)
                .head(3)[["team", "matches", "xga_mean", "ppda_mean",
                          "deep_allowed_mean", "goals_conceded"]].to_string(index=False))
        us_team_names = sorted(tm["team"].dropna().unique())
        print(f"   {len(us_team_names)} teams")
    except Exception:
        failures += 1
        traceback.print_exc()
        print(f"{NO} Understat failed")

    # 5 — reconciliation FPL <-> Understat
    step("5. team-name reconciliation FPL <-> Understat")
    if fpl_team_names and us_team_names:
        rec = team_map.reconcile(fpl_team_names, us_team_names)
        print(f"{OK} matched {len(rec['matched'])}/{len(fpl_team_names)}")
        print("   unmatched (FPL):      ", rec["fpl_only"])
        print("   unmatched (Understat):", rec["fbref_only"])
        if rec["fpl_only"] or rec["fbref_only"]:
            failures += 1
            print("   -> add aliases in fpl_optimiser/ingest/team_map.py for these")
    else:
        print("   (skipped — needed both team lists from steps 2 and 4)")

    # 6 — FBref (optional)
    step("6. FBref (optional — needs Google Chrome)")
    try:
        sched = fbref.schedule(config.SEASON_TRAIN)
        print(f"{OK} FBref reachable: schedule rows {len(sched)}")
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else type(e).__name__
        print(f"   - skipped (this is fine; Understat is primary): {first}")

    step("summary")
    if failures == 0:
        print(f"{OK} all good — paste steps 4-5 back and we'll start Section 2.")
    else:
        print(f"{NO} {failures} check(s) need attention above.")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())