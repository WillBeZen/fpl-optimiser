"""
pull_fpl.py — file the live FPL season into data/raw/fpl/<season>/.

    python scripts/pull_fpl.py               # every player (~700 requests, a few minutes)
    python scripts/pull_fpl.py --limit 50    # quick: top-50 players by points

Re-running is cheap: player files younger than --ttl hours are kept. Any older
season still unfrozen is frozen afterwards — the live API never serves it again,
so it must never be overwritten.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fpl_optimiser.ingest import season_store


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only the top-N players by points (quick test)")
    ap.add_argument("--ttl", type=float, default=12.0,
                    help="keep player files younger than this many hours")
    args = ap.parse_args()

    print("filing the live FPL season ...")
    season = season_store.pull_live(limit=args.limit, ttl_hours=args.ttl)

    for older in season_store.available():
        if older < season and not season_store.is_frozen(older):
            season_store.freeze(older, note="Auto-frozen: the live API no longer serves this season.\n")
            print(f"froze {older} (the live API no longer serves it)")

    s = season_store.load(season)
    print(f"filed {season}: {len(s.players)} players, {s.history['id'].nunique()} with match "
          f"history, {s.finished_gameweeks} finished gameweeks -> {season_store.season_dir(season)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
