"""
study_last_season.py — choose the recency half-life and the blend weight on last
season, check them on this season, and file the study for the dashboard
(model/tuning.py).

    python scripts/study_last_season.py                  # about 25 minutes; no API spend
    python scripts/study_last_season.py --before 8 0.5   # measured against other settings

Needs last season's full match records (python scripts/pull_archive.py). The weekly
run only reads the filed study; run this once a season, or after changing the model.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config
from fpl_optimiser.ingest import archive, season_store
from fpl_optimiser.model import evaluation, tuning
from fpl_optimiser.optimise import decisions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", nargs=2, type=float, metavar=("HALF_LIFE", "WEIGHT"),
                    help="the settings the choice is measured against (default: config.py's)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    this = season_store.available()[-1]
    last = season_store.previous_season(this)
    if last not in season_store.available():
        print(f"{last} isn't filed: there is no last season to study")
        return 1
    checks = archive.season_checks(last)
    if not checks:
        print(f"{last} has FPL's match records for its top players only: run python scripts/pull_archive.py first")
        return 1
    for c in checks:
        print(f"    {'ok  ' if c['ok'] else 'FAIL'} {c['check']}: {c['detail']}")
    if not all(c["ok"] for c in checks):
        print("last season's records don't agree with FPL's: not studying them")
        return 1

    print(f"1/3 {last}: replaying every deadline at each half-life ({', '.join(f'{h:g}' for h in tuning.HALF_LIVES)}) ...")
    study, ev = tuning.run(last, before=args.before)
    study["checks"] = checks

    print(f"2/3 {last}: the decision test ...")
    dec = decisions.run(ev, season_store.load(last))
    study["decisions"] = {"picks": dec["picks"], "season": decisions.season_summary(dec["seasons"]),
                          "weeks": {m: s.weeks for m, s in dec["seasons"].items()}}

    print(f"3/3 {this}: the chosen settings against the ones before ...")
    study["this_season"] = tuning.check_this_season(this, study)
    tuning.save(study)

    h, w = study["chosen"]["half_life"], study["chosen"]["weight"]
    s = study["methods"]["all"]["summary"].set_index("method")
    t = study["this_season"]
    print()
    b = study["before"]
    print(f"{last} GW{study['rounds'][0]}-{study['rounds'][-1]}, every player: blend RMSE "
          f"{s.at[evaluation.REFERENCE, 'rmse']:.3f}, position/price x P(plays) {s.at['pos_price_p', 'rmse']:.3f}")
    print("half-life RMSE: " + ", ".join(f"{k:g}: {v:.4f}" for k, v in sorted(study["rmse"].items())))
    print(f"chosen: half-life {h:g} (before {b['half_life']:g}), blend weight {w:g} (before {b['weight']:g})")
    for m, label in (("rmse", "RMSE"), ("top11", "top 11")):
        x = t[m]
        print(f"{this} GW{t['rounds'][0]}-{t['rounds'][-1]} {label}: chosen {x['new']:.3f} vs before {x['old']:.3f} "
              f"(difference {x['mean_diff']:+.3f}, 90% interval {x['low']:+.3f} to {x['high']:+.3f})")
    print(f"-> {tuning.path(last)}")
    if (h, w) != (float(config.RECENCY_HALF_LIFE_GW), float(config.EVENT_MODEL_WEIGHT)):
        print("config.py differs from the chosen settings: update RECENCY_HALF_LIFE_GW / EVENT_MODEL_WEIGHT "
              "to use them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
