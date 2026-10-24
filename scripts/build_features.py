"""
build_features.py — build (and cache) the Section 2 training table from live data.

    python scripts/build_features.py --limit 50    # quick smoke (top-50 players)
    python scripts/build_features.py               # full season (all players; slow first run)

Steps: pull players/fixtures/teams + opponent stats, build defensive archetypes,
fetch each player's match history (cached after first run), assemble the
leakage-safe per-(player, gameweek) table, and save it to
data/processed/training_table.csv for Section 3.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from fpl_optimiser.ingest import fpl_api, opponent_stats
from fpl_optimiser.features import opponent, build


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only the top-N players by points (quick test)")
    ap.add_argument("--season", default=config.SEASON_TRAIN)
    args = ap.parse_args()

    print("1/5 players, fixtures, teams ...")
    players = fpl_api.players()
    fixtures = fpl_api.fixtures()
    teams = fpl_api.teams()

    print("2/5 opponent stats + defensive archetypes ...")
    opp = opponent_stats.opponent_stats(args.season)
    profiles = opponent.assign_archetypes(opponent.build_opponent_profiles(opp, teams))
    print(opponent.archetype_summary(profiles).to_string(index=False))

    ids = players.sort_values("total_points", ascending=False)["id"].tolist()
    if args.limit:
        ids = ids[: args.limit]

    print(f"3/5 fetching match history for {len(ids)} players (cached after first run) ...")
    frames = []
    for i, pid in enumerate(ids, 1):
        h = fpl_api.player_history(int(pid))
        if not h.empty:
            frames.append(h.assign(id=int(pid)))
        if i % 100 == 0:
            print(f"   {i}/{len(ids)}")
    history_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    print("4/5 assembling training table ...")
    tbl = build.build_training_table(history_all, fixtures, players, profiles)

    print("5/5 saving ...")
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED_DIR / "training_table.csv"
    tbl.to_csv(out, index=False)
    print(f"saved {len(tbl)} rows x {tbl.shape[1]} cols -> {out}")
    print(tbl.head(8).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())