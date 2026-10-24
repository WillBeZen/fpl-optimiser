"""
make_report.py — write the HTML report (one self-contained file; works offline).

    python scripts/pull_fpl.py            # optional: refresh the live season first
    python scripts/make_report.py         # -> reports/fpl_report.html
    python scripts/make_report.py --open  # ...and open it in your browser
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import webbrowser

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fpl_optimiser.app import report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=None, help="season to report on (default: the latest filed)")
    ap.add_argument("--open", action="store_true", help="open the report in your browser")
    args = ap.parse_args()

    print("building report ...")
    out = report.build_report(args.season).write()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if args.open:
        webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
