"""
build_features.py — build the Section 2 training table from the season store.

    python scripts/pull_fpl.py                         # first: file the live season
    python scripts/build_features.py                   # every filed season
    python scripts/build_features.py --seasons 2026-27

Each season is built separately (ids are per season), with the previous season as
a prior when it is filed, then stacked with `season`, `code` (the player's permanent
id) and `sample_weight` (an older season counts config.PREV_SEASON_WEIGHT as much
per season back). Saved to data/processed/training_table.csv for Section 3.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config
from fpl_optimiser.features import multi_season
from fpl_optimiser.ingest import season_store


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", default=None,
                    help="seasons to build, e.g. 2025-26 2026-27 (default: every filed season)")
    args = ap.parse_args()

    seasons = args.seasons or season_store.available()
    if not seasons:
        print("nothing filed yet: run scripts/pull_fpl.py first")
        return 1

    print(f"building {', '.join(seasons)} ...")
    tbl = multi_season.build_all(seasons)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED_DIR / "training_table.csv"
    tbl.to_csv(out, index=False)

    summary = tbl.groupby("season").agg(rows=("id", "size"), players=("code", "nunique"),
                                        with_last_season=("has_prev_season", "mean"),
                                        sample_weight=("sample_weight", "first"))
    summary["with_last_season"] = (summary["with_last_season"] * 100).round().astype(int).astype(str) + "%"
    print(summary.to_string())
    print(f"saved {len(tbl)} rows x {tbl.shape[1]} cols -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
