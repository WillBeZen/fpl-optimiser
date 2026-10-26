"""
pull_archive.py — download an earlier season's match records from the public archive
(ingest/archive.py) into data/raw/archive/<season>/, then check them against FPL's
own filed data.

    python scripts/pull_archive.py               # every season listed in archive.SOURCES
    python scripts/pull_archive.py 2025-26

One file per season (2025/26: merged_gw.csv, 5.4 MB), pinned to an archive commit
and kept only if it is exactly that file. Already downloaded: nothing is fetched.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fpl_optimiser.ingest import archive, season_store


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", nargs="*", help="e.g. 2025-26 (default: every season in archive.SOURCES)")
    args = ap.parse_args()

    failed = 0
    for season in args.seasons or sorted(archive.SOURCES):
        if season not in archive.SOURCES:
            print(f"{season}: no archive file listed in archive.SOURCES")
            failed += 1
            continue
        src = archive.SOURCES[season]
        print(f"{season}: {src['url']} ({src['bytes'] / 1e6:.1f} MB) ...")
        print(f"   -> {archive.pull(season)}")
        if season not in season_store.available():
            print(f"   {season} isn't filed from FPL, so there is nothing to check it against")
            continue
        for c in archive.season_checks(season):
            print(f"   {'ok  ' if c['ok'] else 'FAIL'} {c['check']}: {c['detail']}")
            failed += not c["ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
