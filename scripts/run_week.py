"""
run_week.py — the weekly run, in one command:

    fresh FPL data -> expected points -> your transfer plan -> Claude's news check -> report

    python scripts/run_week.py --open             # everything, then open the report
    python scripts/run_week.py --no-pull          # skip refreshing FPL data
    python scripts/run_week.py --team-id 1234567  # or put FPL_TEAM_ID=1234567 in .env
    python scripts/run_week.py --free-transfers 2 # override the estimate from your history
    python scripts/run_week.py --no-advisor       # no new Claude news check (reuses this
                                                  # gameweek's last one, if saved)
    python scripts/run_week.py --advisor-effort low --max-searches 2   # a cheaper news check
    python scripts/run_week.py --quick            # skip the backtest and the planner's what-ifs

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
from fpl_optimiser.model import accuracy, evaluation, forecast_log, predict
from fpl_optimiser.optimise import planner, squad, whatif

ADVICE_FILE = config.ROOT / "reports" / "last_advice.json"


def _accuracy(fc, table, team_id):
    """Replay every finished gameweek as it stood (model/evaluation.py), the half-life
    check and the code-path check; print the verdict and the data checks."""
    season = season_store.load(fc.season)
    done = sorted(int(g) for g in season.gameweeks.loc[season.gameweeks["finished"] == True, "id"])  # noqa: E712
    if len(done) < 2:
        return None, None, None
    squads = my_team.squads_before(team_id, done[1:]) if team_id else {}
    ev = evaluation.run(fc.season, table, squads=squads)
    h = config.RECENCY_HALF_LIFE_GW
    print(f"    accuracy: GW{ev.rounds[0]}-{ev.rounds[-1]} replayed as each deadline stood; "
          f"half-lives {h / 2:g} and {h * 2:g} for comparison ...")
    half_lives = accuracy.half_life_check(ev, (h / 2, h, h * 2))
    path_check = accuracy.forecast_path_check(table, ev)
    pgw = evaluation.player_gameweeks(ev, "pool")
    s = evaluation.summary(evaluation.by_gameweek(pgw), pgw).set_index("method")
    ref, base = s.loc[evaluation.REFERENCE], s.loc["pos_price_p"]
    print(f"    the planner's pool: blend RMSE {ref['rmse']:.3f}, top 11 {ref['top11']:.1f} a gameweek; "
          f"simplest strong baseline (position/price x P(plays)) {base['rmse']:.3f}, {base['top11']:.1f}")
    checks = accuracy.reconcile(ev, season) + [path_check]
    bad = [c for c in checks if c["ok"] is not True]
    print(f"    data checks: {len(checks) - len(bad)} of {len(checks)} pass")
    for c in bad:
        print(f"      {'FAIL' if c['ok'] is False else 'info'}: {c['check']} - {c['detail']}")
    return ev, half_lives, path_check


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
    ap.add_argument("--quick", action="store_true",
                    help="skip the backtest and the planner's what-ifs (about 2 minutes faster)")
    ap.add_argument("--backtest", action="store_true", help=argparse.SUPPRESS)   # now the default
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
    logged = forecast_log.save(fc)                 # kept for the Accuracy tab once GW is played
    print(f"    GW{fc.next_gw} forecast logged ({logged.relative_to(config.ROOT)})")

    print("3/6 best squad from scratch ...")
    wildcard = squad.pick_squad(nxt)

    team_id = args.team_id or (int(os.environ["FPL_TEAM_ID"]) if os.environ.get("FPL_TEAM_ID", "").strip().isdigit() else None)
    plan = baseline = mine = ladder = budget = chip_grid = None
    if team_id:
        print(f"4/6 planning transfers for team {team_id} ...")
        mine = my_team.load(team_id, season_store.load(fc.season).bootstrap, fc.next_gw)
        if args.free_transfers is not None:
            mine.free_transfers = args.free_transfers
        chips_left = "; ".join(f"GW{a}-{b}: " + (", ".join(planner.CHIP_NAMES[k] for k in names) or "none")
                               for (a, b), names in mine.chips_available.items())
        print(f"    chips left: {chips_left or 'none'}")
        plan = planner.plan(fc.xp, players, mine.squad_ids, mine.bank, mine.free_transfers, mine.sell_prices,
                            chips=mine.chips_available)
        baseline = planner.plan(fc.xp, players, mine.squad_ids, mine.bank, mine.free_transfers,
                                mine.sell_prices, max_transfers_per_gw=0)
        proven = "" if plan.stats["proven_optimal"] else f", best found in the {planner.CHIP_SOLVE_SECONDS}s chip limit"
        print(f"    solved in {plan.stats['solve_seconds']:.0f}s ({plan.stats['candidates']} of "
              f"{plan.stats['players']} players considered{proven})")
        if not args.quick:
            print("    what-ifs: 0..n transfers this week, more money, each chip in each gameweek ...")
            inputs = whatif.plan_inputs(fc.xp, players, mine.squad_ids, mine.bank, mine.free_transfers,
                                        mine.sell_prices, chips=mine.chips_available)
            held = whatif.hold_chips(inputs, plan)          # transfers and money, the plan's chips kept
            ladder = whatif.transfer_ladder(held)
            budget = whatif.budget_value(held, base=plan)
            chip_grid = whatif.chip_weeks(inputs, chosen=plan)
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
        advice.gameweek = fc.next_gw
        if advice.ok:
            advisor.save(advice, ADVICE_FILE)
            cost = f", about ${advice.cost:.2f}" if advice.cost is not None else ""
            print(f"    {advice.action_score}/100, {advice.verdict} ({advice.searches} searches, "
                  f"{advice.input_tokens:,} tokens in, {advice.output_tokens:,} out{cost})")
        else:
            print(f"    unavailable: {advice.error}")
    else:
        advice = advisor.load(ADVICE_FILE, fc.next_gw)
        print(f"5/6 no new news check; reusing GW{fc.next_gw}'s from {advice.checked_at} UTC" if advice
              else "5/6 skipping the Claude news check")

    bt = ev = half_lives = path_check = None
    if not args.quick:
        print("    re-checking the models: every past deadline replayed as it stood ...")
        ev, half_lives, path_check = _accuracy(fc, table, team_id)
        bt = evaluation.backtest_table(ev) if ev is not None else None

    print("6/6 writing the report ...")
    week = report.WeekResults(forecast=fc, wildcard=wildcard, plan=plan, baseline=baseline,
                              team=mine, advice=advice, backtest=bt, table=table, ladder=ladder,
                              budget=budget, chips=chip_grid, evaluation=ev, half_lives=half_lives,
                              forecast_check=path_check)
    out = report.build_report(fc.season, week=week).write()

    print()
    if plan is not None:
        first = plan.weeks[0]
        moves = (", ".join(f"{o} -> {i}" for o, i in zip(first.sold["web_name"], first.bought["web_name"]))
                 or "no transfers (bank it)")
        chip = f"; chip: {planner.CHIP_NAMES[first.chip]}" if first.chip else ""
        print(f"GW{fc.next_gw}: {moves}; hits {first.hits}; captain {first.captain}{chip}")
        print("Chips in the plan: " + (", ".join(f"{planner.CHIP_NAMES[k]} GW{g}" for k, g in plan.chips.items())
                                       or f"none in GW{fc.next_gw}-{fc.gameweeks[-1]}"))
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
