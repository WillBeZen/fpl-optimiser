"""
run_week.py — the weekly run, in one command:

    fresh FPL data -> expected points -> your transfer plan -> Claude's news check -> report

    python scripts/run_week.py --open             # everything, then open the report
    python scripts/run_week.py --no-pull          # skip refreshing FPL data
    python scripts/run_week.py --team-id 1234567  # or put FPL_TEAM_ID=1234567 in .env
    python scripts/run_week.py --free-transfers 2 # override the estimate from your history
    python scripts/run_week.py --no-advisor       # skip the Claude news check
    python scripts/run_week.py --advisor-effort low --max-searches 2   # a cheaper news check
    python scripts/run_week.py --backtest         # also re-check the model (~1 minute)

Your team id is the number in your team's URL on the FPL site
(fantasy.premierleague.com/entry/<id>/event/<gw>). The news check needs
ANTHROPIC_API_KEY in .env.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import webbrowser

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config
from fpl_optimiser.app import report
from fpl_optimiser.claude import advisor
from fpl_optimiser.features import multi_season
from fpl_optimiser.ingest import my_team, rules_check, season_store
from fpl_optimiser.model import backtest, predict
from fpl_optimiser.optimise import planner, squad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pull", action="store_true", help="don't refresh FPL data first")
    ap.add_argument("--team-id", type=int, default=None, help="your FPL team id (else FPL_TEAM_ID)")
    ap.add_argument("--free-transfers", type=int, default=None, help="override the estimate")
    ap.add_argument("--horizon", type=int, default=config.PLANNING_HORIZON_GW)
    ap.add_argument("--no-advisor", action="store_true", help="skip the Claude news check")
    ap.add_argument("--advisor-effort", choices=["low", "medium", "high", "xhigh", "max"],
                    default=config.ADVISOR_EFFORT, help="how hard Claude thinks (lower = cheaper)")
    ap.add_argument("--max-searches", type=int, default=config.ADVISOR_MAX_SEARCHES,
                    help="most web searches Claude may run ($0.01 each)")
    ap.add_argument("--backtest", action="store_true", help="also re-run the model backtest")
    ap.add_argument("--open", action="store_true", help="open the report when done")
    args = ap.parse_args()

    try:                                   # .env holds FPL_TEAM_ID and ANTHROPIC_API_KEY
        from dotenv import load_dotenv
        load_dotenv(config.ROOT / ".env")
    except ImportError:
        pass

    if not args.no_pull:
        print("1/6 refreshing FPL data ...")
        season = season_store.pull_live(log=lambda *_: None)
        for problem in rules_check.check_rules(season_store.load(season).bootstrap):
            print(f"    WARNING rules: {problem}  -> update config.py")
    else:
        print("1/6 using the FPL data already filed")

    print("2/6 predicting expected points ...")
    table = multi_season.build_all()
    fc = predict.forecast(table=table, horizon=args.horizon)
    players = fc.players
    nxt = fc.xp[fc.xp["gw"] == fc.next_gw]
    print(f"    GW{fc.next_gw}-{fc.gameweeks[-1]}, deadline {fc.deadline or '?'} UTC")

    print("3/6 best squad from scratch ...")
    wildcard = squad.pick_squad(nxt)

    team_id = args.team_id or (int(os.environ["FPL_TEAM_ID"]) if os.environ.get("FPL_TEAM_ID", "").strip().isdigit() else None)
    plan = baseline = mine = None
    if team_id:
        print(f"4/6 planning transfers for team {team_id} ...")
        mine = my_team.load(team_id, season_store.load(fc.season).bootstrap, fc.next_gw)
        if args.free_transfers is not None:
            mine.free_transfers = args.free_transfers
        plan = planner.plan(fc.xp, players, mine.squad_ids, mine.bank, mine.free_transfers, mine.sell_prices)
        baseline = planner.plan(fc.xp, players, mine.squad_ids, mine.bank, mine.free_transfers,
                                mine.sell_prices, max_transfers_per_gw=0)
    else:
        print("4/6 no FPL_TEAM_ID set: skipping your transfer plan (best squad from scratch only)")

    advice = None
    if not args.no_advisor:
        print("5/6 Claude news check ...")
        if plan is not None:
            first = plan.weeks[0]
            text = advisor.brief(first, players, fc.deadline, baseline_xp=baseline.xp,
                                 plan_xp=plan.xp, hits_cost=plan.hit_cost, note=mine.note)
        else:                              # no team: check the from-scratch squad instead
            week = planner.Week(gw=fc.next_gw, sold=nxt.iloc[0:0], bought=nxt.iloc[0:0],
                                free_transfers=0, hits=0, bank_after=0.0, team=wildcard.players,
                                xp=wildcard.xp, formation=wildcard.formation)
            text = advisor.brief(week, players, fc.deadline, from_scratch=True)
        advice = advisor.advise(text, effort=args.advisor_effort, max_searches=args.max_searches)
        if advice.ok:
            cost = f", about ${advice.cost:.2f}" if advice.cost is not None else ""
            print(f"    {advice.action_score}/100, {advice.verdict} ({advice.searches} searches, "
                  f"{advice.input_tokens:,} tokens in, {advice.output_tokens:,} out{cost})")
        else:
            print(f"    unavailable: {advice.error}")
    else:
        print("5/6 skipping the Claude news check")

    bt = None
    if args.backtest:
        print("    re-checking the model (walk-forward backtest) ...")
        bt = backtest.summarise(backtest.walk_forward(table, fc.season))

    print("6/6 writing the report ...")
    week = report.WeekResults(forecast=fc, wildcard=wildcard, plan=plan, baseline=baseline,
                              team=mine, advice=advice, backtest=bt)
    out = report.build_report(fc.season, week=week).write()

    print()
    if plan is not None:
        first = plan.weeks[0]
        moves = (", ".join(f"{o} -> {i}" for o, i in zip(first.sold["web_name"], first.bought["web_name"]))
                 or "no transfers (bank it)")
        print(f"GW{fc.next_gw}: {moves}; hits {first.hits}; captain {first.captain}")
        print(f"Plan vs no transfers over GW{fc.next_gw}-{fc.gameweeks[-1]}: "
              f"{plan.net - baseline.xp:+.1f} expected points (after hits)")
    else:
        print(f"Best GW{fc.next_gw} squad from scratch: {wildcard.formation}, £{wildcard.cost:.1f}m, "
              f"{wildcard.xp:.1f} expected pts, captain {wildcard.captain}")
    print(f"Report: {out}")
    if args.open:
        webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
