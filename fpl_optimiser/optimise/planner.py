"""Multi-week transfer planner — one mixed-integer program over the whole horizon.

From your current squad, bank and free transfers, decide for each of the next few
gameweeks who to buy and sell, the XI and the captain, and when to play a chip,
maximising expected points over the horizon. Because every week is planned together,
it can see that banking a transfer now buys two next week, that a -4 hit pays off
over five weeks, or that a bench boost is worth more in a double gameweek.

Each gameweek t has the single-gameweek squad rules (optimise/squad.py), plus:
  squad[t] = squad[t-1] + bought[t] - sold[t]     (squad[0] = your squad today)
  bank[t]  = bank[t-1] + sales - purchases >= 0   (your players sell at the prices
                                                   passed in: ingest/my_team.py)
  free transfers: hits[t] >= transfers[t] - free[t];
                  free[t+1] <= free[t] - transfers[t] + hits[t] + 1, and <= the cap

Chips (`chips`: those still available per set, my_team.chips_available): at most one a
gameweek, each once per set, and only within its set's gameweeks:
  wildcard        that week's transfers are free (no hits); the new squad stays
  free hit        a different squad for that week only; your squad makes no transfers
                  and is back the week after. With prices held fixed, transfers never
                  change bank + squad value, so each week's best free-hit squad is
                  picked up front (squad.pick_squad) and the plan decides when to play it
  bench boost     the bench scores in full that week
  triple captain  the captain counts config.TRIPLE_CAPTAIN_MULTIPLIER times, not 2
Free transfers you had are kept through a wildcard or free hit (plus the weekly one).

Objective, each gameweek weighted config.HORIZON_DECAY ** (weeks ahead):
  XI expected points + captain again + bench at config.BENCH_WEIGHT
  (on a bench boost the bench in full; on a triple captain the captain once more)
  - config.HIT_THRESHOLD_POINTS per hit   (a -4 is taken only if it gains more than
                                           this: the margin for prediction error)
  - config.CHIP_THRESHOLD_POINTS[chip] per chip played, decayed like that week's points
    (its opportunity value; not charged when its set expires within the horizon)
  + config.FREE_TRANSFER_VALUE per transfer still banked after the last gameweek

Speed. Before solving, players who can't be optimal are dropped (optimise/pool.py), and
so are chip options that can't beat their threshold (a bench boost can't add more than
the week's best bench, a triple captain more than its best player). With chips in, CBC
takes minutes to find even the plan without chips, so it is handed a good plan to
start from (_solve_with_chips): the plan without chips, then the chips worth playing on
that plan's squads, then the best plan with those chips. Proving that nothing beats it
can still take minutes, so the full chip problem stops after CHIP_SOLVE_SECONDS with
the best plan found (`Plan.stats["proven_optimal"]` says which). `Plan.stats` also
records the problem's size and solve time.
"""
from __future__ import annotations

import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd
import pulp

import config
from fpl_optimiser.optimise import pool
from fpl_optimiser.optimise.squad import _VICE_TIEBREAK, _present, pick_squad, solve

_FT_TIEBREAK = 1e-4
CHIP_NAMES = {"wildcard": "wildcard", "free_hit": "free hit", "bench_boost": "bench boost",
              "triple_captain": "triple captain"}     # config.CHIPS, as people say them
# The full chip problem (step 4 of _solve_with_chips) stops here if it hasn't proven its
# plan optimal: it then returns the best plan found, never worse than its starting plan,
# which has usually been the answer already. Proving it can take CBC minutes.
CHIP_SOLVE_SECONDS = 60


@dataclass
class Week:
    gw: int
    sold: pd.DataFrame          # players out (with the price you'd get)
    bought: pd.DataFrame        # players in
    free_transfers: int         # available before this gameweek's transfers
    hits: int                   # paid transfers (each costs config.TRANSFER_HIT_COST)
    bank_after: float
    team: pd.DataFrame          # the 15, XI/bench, captain, ordered like a team sheet
    xp: float                   # XI + captain's double (+ what the chip adds), this gameweek
    formation: str
    free_after: int = 0         # free transfers carried into the next gameweek
    chip: str | None = None     # the chip played this gameweek (a config.CHIPS name)
    chip_gain: float = 0.0      # what a bench boost (bench) or triple captain adds to xp

    @property
    def captain(self) -> str:
        return self.team.loc[self.team["captain"], "web_name"].iloc[0]


@dataclass
class Plan:
    weeks: list[Week] = field(default_factory=list)
    xp: float = 0.0             # expected points over the horizon (undiscounted, before hits)
    hit_cost: int = 0           # points spent on hits
    status: str = ""
    free_end: int = 0           # free transfers still banked after the last gameweek
    objective: float = 0.0      # the solver's objective (decayed, with penalties and bonus)
    stats: dict = field(default_factory=dict)   # problem size and solve time
    chip_charge: float = 0.0    # chip thresholds charged in the objective

    @property
    def net(self) -> float:
        return self.xp - self.hit_cost

    @property
    def chips(self) -> dict[str, int]:
        """The chips the plan plays: {chip: gameweek}."""
        return {w.chip: w.gw for w in self.weeks if w.chip}


def plan(xp: pd.DataFrame, players: pd.DataFrame, squad_ids: list[int], bank: float,
         free_transfers: int, sell_prices: dict[int, float] | None = None,
         max_transfers_per_gw: int | None = None, moves_now: int | None = None,
         decay: float = config.HORIZON_DECAY, hit_penalty: float = config.HIT_THRESHOLD_POINTS,
         ft_value: float = config.FREE_TRANSFER_VALUE,
         bench_weight: float = config.BENCH_WEIGHT, prune: bool = True,
         chips: dict[tuple[int, int], Iterable[str]] | None = None,
         chip_threshold: dict[str, float] = config.CHIP_THRESHOLD_POINTS,
         play_chips: dict[str, int] | None = None) -> Plan:
    """The best transfer plan over the gameweeks in `xp`.

    xp            one row per (player id, gw) with column `xp` (predict.Forecast.xp)
    players       id, web_name, team_name, position, price (today's price)
    squad_ids     your 15 player ids now; bank in £m; free_transfers available now
    sell_prices   what each of your players sells for (default: today's price)
    max_transfers_per_gw  0 = "no transfers" (the baseline to compare a plan with)
    moves_now     exactly this many transfers in the first gameweek (for what-ifs)
    prune         drop players who can't be optimal first (pool.py); same answer, faster
    chips         chips still available per set, {(first gw, last gw): chip names}
                  (my_team.chips_available); None = no chips
    chip_threshold  the gain each chip must beat (config.CHIP_THRESHOLD_POINTS)
    play_chips    {chip: gw}: play these chips in those gameweeks (for what-ifs)
    """
    started = time.perf_counter()
    gws = sorted(xp["gw"].unique())
    n_players = xp["id"].nunique()
    if prune:
        xp = xp[xp["id"].isin(pool.prune(xp, players, keep=squad_ids))]
    pts = xp.pivot_table(index="id", columns="gw", values="xp", aggfunc="sum").reindex(columns=gws)
    c = players.set_index("id").loc[lambda d: d.index.isin(pts.index) | d.index.isin(squad_ids)]
    pts = pts.reindex(c.index).fillna(0.0)
    ids = list(c.index)
    owned = {i: int(i in set(squad_ids)) for i in ids}
    buy_price = c["price"].to_dict()
    sell_price = {i: (sell_prices or {}).get(i, buy_price[i]) if owned[i] else buy_price[i] for i in ids}
    T = range(len(gws))
    key = [(i, t) for i in ids for t in T]

    # chip options (k, t): chip k can be played in gameweek t; `where` names its set
    sets = {(k, w) for w, names in (chips or {}).items() for k in names}
    if {k for k, _ in sets} - set(config.CHIPS):
        raise ValueError(f"unknown chips {sorted({k for k, _ in sets} - set(config.CHIPS))}: use {config.CHIPS}")
    where = {(k, t): w for k, w in sets for t in T if w[0] <= gws[t] <= w[1]}
    forced = set()
    for k, gw in (play_chips or {}).items():
        if gw not in gws or (k, gws.index(gw)) not in where:
            raise ValueError(f"{k} can't be played in GW{gw} (not available then)")
        forced.add((k, gws.index(gw)))
    charge = {(k, t): chip_charge(k, t, w, gws, decay, chip_threshold) for (k, t), w in where.items()}
    ceiling = _chip_ceilings(pts, c["position"], gws, decay, bench_weight)
    options = sorted(kt for kt in where if kt in forced or ceiling.get(kt, float("inf")) > charge[kt])
    budget = bank + sum(sell_price[i] for i in ids if owned[i])     # the same after any transfers
    fh_squads = {t: _free_hit_squad(xp[xp["gw"] == gws[t]], c, sell_price, budget, squad_ids, bench_weight)
                 for k, t in options if k == "free_hit"}
    extra_cap = config.TRIPLE_CAPTAIN_MULTIPLIER - config.CAPTAIN_MULTIPLIER
    timing = {"build": time.perf_counter() - started}       # preparing the data counts as building

    def build(options: list[tuple[str, int]], forced: set[tuple[str, int]] = forced):
        """The integer program with these chip options (`forced` ones played); returns
        (problem, variables)."""
        begun = time.perf_counter()
        prob = pulp.LpProblem("plan", pulp.LpMaximize)
        sq = prob.add_variable_dicts("squad", key, cat="Binary")
        st = prob.add_variable_dicts("start", key, cat="Binary")
        cp = prob.add_variable_dicts("captain", key, cat="Binary")
        vc = prob.add_variable_dicts("vice", key, cat="Binary")
        buy = prob.add_variable_dicts("buy", key, cat="Binary")
        sell = prob.add_variable_dicts("sell", key, cat="Binary")
        ft = prob.add_variable_dicts("free", list(range(len(gws) + 1)), lowBound=0,
                                     upBound=config.MAX_BANKED_TRANSFERS, cat="Integer")
        hits = prob.add_variable_dicts("hits", list(T), lowBound=0, cat="Integer")
        money = prob.add_variable_dicts("bank", list(T), lowBound=0)
        chip = prob.add_variable_dicts("chip", options, cat="Binary")
        on = lambda k, t: chip.get((k, t), 0)                   # noqa: E731
        # Chip effects, linked per player (a fractional chip then only helps fractionally):
        #   plays[i,t]   he scores in full: the XI, or all 15 on a bench boost
        #   triple[i,t]  his points count once more as captain (triple captain)
        #   idle[t]      on a free hit your own squad doesn't play: this takes back the
        #                bench points it would otherwise still be credited
        plays = prob.add_variable_dicts("plays", [(i, t) for i, t in key if ("bench_boost", t) in chip],
                                        lowBound=0, upBound=1)
        triple = prob.add_variable_dicts("triple", [(i, t) for i, t in key if ("triple_captain", t) in chip],
                                         lowBound=0, upBound=1)
        fh_weeks = [t for t in T if ("free_hit", t) in chip]
        idle = prob.add_variable_dicts("idle", fh_weeks, lowBound=0)
        full = {(i, t): plays.get((i, t), st[i, t]) for i, t in key}

        prob += (pulp.lpSum(decay ** t * pts.at[i, gws[t]] * (full[i, t] + cp[i, t] + _VICE_TIEBREAK * vc[i, t]
                                                              + bench_weight * (sq[i, t] - full[i, t])
                                                              + extra_cap * triple.get((i, t), 0))
                            for i, t in key)
                 + pulp.lpSum(decay ** t * (fh_squads[t][1] * chip["free_hit", t] - idle[t]) for t in fh_weeks)
                 - pulp.lpSum(charge[kt] * chip[kt] for kt in chip)
                 - hit_penalty * pulp.lpSum(hits[t] for t in T)
                 + ft_value * ft[len(gws)]
                 # tie-break only: report the free transfers you'd really have each week
                 # (otherwise any value up to the true count is equally "optimal")
                 + _FT_TIEBREAK * pulp.lpSum(ft[t] for t in range(1, len(gws))))

        prob += ft[0] == min(max(int(free_transfers), 0), config.MAX_BANKED_TRANSFERS)
        for k, w in sets:
            in_set = [chip[k, t] for t in T if (k, t) in chip and where[k, t] == w]
            if in_set:
                prob += pulp.lpSum(in_set) <= 1
        for kt in forced:
            prob += chip[kt] == 1
        for t in T:
            n_moves = pulp.lpSum(buy[i, t] for i in ids)
            wildcard, playing = on("wildcard", t), 1 - on("free_hit", t)   # your squad plays unless free hit
            if any((k, t) in chip for k in config.CHIPS):
                prob += pulp.lpSum(on(k, t) for k in config.CHIPS) <= 1
                # a wildcard or free hit keeps the free transfers you had, plus the weekly one
                prob += ft[t + 1] <= ft[t] + config.FREE_TRANSFERS_PER_GW
            prob += n_moves == pulp.lpSum(sell[i, t] for i in ids)
            prob += hits[t] >= n_moves - ft[t] - config.SQUAD_SIZE * wildcard        # wildcard: all free
            prob += ft[t + 1] <= (ft[t] - n_moves + hits[t] + config.FREE_TRANSFERS_PER_GW
                                  + config.SQUAD_SIZE * wildcard)
            if max_transfers_per_gw is not None:
                prob += n_moves <= max_transfers_per_gw
            if moves_now is not None and t == 0:
                prob += n_moves == moves_now
            prob += money[t] == ((bank if t == 0 else money[t - 1])
                                 + pulp.lpSum(sell_price[i] * sell[i, t] - buy_price[i] * buy[i, t] for i in ids))
            for i in ids:
                prev = owned[i] if t == 0 else sq[i, t - 1]
                prob += sq[i, t] == prev + buy[i, t] - sell[i, t]
                prob += buy[i, t] + sell[i, t] <= 1
                prob += st[i, t] <= sq[i, t]
                prob += cp[i, t] <= st[i, t]
                prob += vc[i, t] <= st[i, t]
                prob += cp[i, t] + vc[i, t] <= 1
                if (i, t) in plays:
                    prob += plays[i, t] >= st[i, t]
                    prob += plays[i, t] <= sq[i, t]
                if (i, t) in triple:
                    prob += triple[i, t] <= cp[i, t]
            prob += pulp.lpSum(sq[i, t] for i in ids) == config.SQUAD_SIZE
            prob += pulp.lpSum(st[i, t] for i in ids) == config.XI_SIZE * playing
            prob += pulp.lpSum(cp[i, t] for i in ids) == playing
            prob += pulp.lpSum(vc[i, t] for i in ids) == playing
            for pos, quota in config.POSITION_QUOTA.items():
                in_pos = [i for i in ids if c.at[i, "position"] == pos]
                lo, hi = config.FORMATION_BOUNDS[pos]
                prob += pulp.lpSum(sq[i, t] for i in in_pos) == quota
                prob += pulp.lpSum(st[i, t] for i in in_pos) >= lo * playing
                prob += pulp.lpSum(st[i, t] for i in in_pos) <= hi * playing
            for club, members in c.groupby("team_name").groups.items():
                prob += pulp.lpSum(sq[i, t] for i in members) <= config.MAX_PER_CLUB
            if ("bench_boost", t) in chip:           # 11 score in full, or all 15 on a bench boost
                prob += pulp.lpSum(plays[i, t] for i in ids) == (
                    config.XI_SIZE * playing + (config.SQUAD_SIZE - config.XI_SIZE) * on("bench_boost", t))
            if ("triple_captain", t) in chip:
                prob += pulp.lpSum(triple[i, t] for i in ids) <= on("triple_captain", t)
            if t in fh_weeks:                        # a free hit: your squad rests, unchanged
                prob += n_moves <= config.SQUAD_SIZE * playing
                most = bench_weight * pts[gws[t]].clip(lower=0.0).nlargest(config.SQUAD_SIZE).sum()
                prob += idle[t] >= (bench_weight * pulp.lpSum(pts.at[i, gws[t]] * sq[i, t] for i in ids)
                                    - most * playing)
        timing["build"] += time.perf_counter() - begun
        return prob, dict(sq=sq, st=st, cp=cp, vc=vc, buy=buy, sell=sell, ft=ft, hits=hits,
                          money=money, chip=chip)

    if options and set(options) != forced:
        prob, v, status, proven = _solve_with_chips(build, options, forced)
    else:                                  # no chip decisions to make: one ordinary solve
        prob, v = build(options)
        status, proven = solve(prob), True
    if status != "Optimal":
        raise ValueError(f"no legal plan found (solver: {status}) — check the squad and bank")
    sq, st, cp, vc, buy, sell, ft, hits, money, chip = (v[k] for k in (
        "sq", "st", "cp", "vc", "buy", "sell", "ft", "hits", "money", "chip"))

    out = Plan(status=status, free_end=int(round(ft[len(gws)].value())),
               objective=float(pulp.value(prob.objective)),
               stats={"players": n_players, "candidates": len(ids), "gameweeks": len(gws),
                      "variables": len(prob.variables()), "constraints": prob.numConstraints(),
                      "binary": sum(var.cat == pulp.LpInteger and var.upBound == 1 for var in prob.variables()),
                      "chip_options": len(options), "chip_options_dropped": len(where) - len(options),
                      "proven_optimal": proven,
                      "build_seconds": timing["build"],
                      "solve_seconds": time.perf_counter() - started - timing["build"]})
    info = c.reset_index()[["id", "web_name", "team_name", "position", "price"]]
    for t, gw in enumerate(gws):
        played = next((k for k in config.CHIPS if (k, t) in chip and chip[k, t].value() > 0.5), None)
        if played == "free_hit":           # the week's team is the free hit's; yours is back after
            sheet, _, bank_after = fh_squads[t]
            yours = {i for i in ids if sq[i, t].value() > 0.5}
            hit_team = set(sheet.players["id"])
            moved_out = [i for i in ids if i in yours - hit_team]
            moved_in = [i for i in ids if i in hit_team - yours]
        else:
            chosen = [i for i in ids if sq[i, t].value() > 0.5]
            team = info.set_index("id").loc[chosen].reset_index()
            team["xp"] = [pts.at[i, gw] for i in chosen]
            team["role"] = ["XI" if st[i, t].value() > 0.5 else "bench" for i in chosen]
            team["captain"] = [cp[i, t].value() > 0.5 for i in chosen]
            team["vice"] = [vc[i, t].value() > 0.5 for i in chosen]
            sheet = _present(team, "xp")
            moved_out = [i for i in ids if sell[i, t].value() > 0.5]
            moved_in = [i for i in ids if buy[i, t].value() > 0.5]
            bank_after = money[t].value()
        team = sheet.players
        gain = 0.0
        if played == "bench_boost":
            gain = float(team.loc[team["role"] == "bench", "xp"].sum())
        elif played == "triple_captain":
            gain = extra_cap * float(team.loc[team["captain"], "xp"].sum())
        out.weeks.append(Week(
            gw=int(gw),
            sold=info.set_index("id").loc[moved_out].reset_index().assign(
                sell_price=[sell_price[i] for i in moved_out], xp=[pts.at[i, gw] for i in moved_out]),
            bought=info.set_index("id").loc[moved_in].reset_index().assign(
                xp=[pts.at[i, gw] for i in moved_in]),
            free_transfers=int(round(ft[t].value())), hits=int(round(hits[t].value())),
            bank_after=round(bank_after, 1), team=team, xp=sheet.xp + gain,
            formation=sheet.formation, free_after=int(round(ft[t + 1].value())),
            chip=played, chip_gain=gain))
        if played:
            out.chip_charge += charge[played, t]
    out.xp = sum(w.xp for w in out.weeks)
    out.hit_cost = config.TRANSFER_HIT_COST * sum(w.hits for w in out.weeks)
    return out


def chip_charge(chip: str, t: int, window: tuple[int, int], gws: list[int],
                decay: float = config.HORIZON_DECAY,
                threshold: dict[str, float] = config.CHIP_THRESHOLD_POINTS) -> float:
    """What playing `chip` in the t-th gameweek of the horizon must gain in the objective:
    its threshold in that week's (decayed) points, or nothing if its set (`window`, first
    and last gameweek) ends within the horizon: a chip about to expire is worth nothing
    saved, so it is used in its best remaining week however small the gain."""
    return decay ** t * threshold.get(chip, 0.0) if window[1] > gws[-1] else 0.0


def _solve_with_chips(build, options: list[tuple[str, int]], forced: set[tuple[str, int]]):
    """The chip problem, solved from a good starting plan, because CBC on its own takes
    minutes to find one (even the plan without chips):
      1. the plan without chip decisions: the ordinary, fast problem;
      2. keeping that plan's squads, which chips are worth playing (a small problem);
      3. if any, the best plan with those chips played (again an ordinary problem);
      4. the full problem with every chip option, starting from the best of these.
    Returns (problem, variables, status, proven optimal?)."""
    prob, v = build(sorted(forced), forced)
    if solve(prob) != "Optimal":
        return prob, v, pulp.LpStatus[prob.status], False
    best = (prob, v)
    trial, tv = build(options, forced)
    values = {var.name: var.varValue for var in prob.variables()}
    for name in ("sq", "buy", "sell"):
        for var in tv[name].values():
            var.lowBound = var.upBound = round(values[var.name])
    if solve(trial) == "Optimal":
        picked = {kt for kt, var in tv["chip"].items() if var.value() > 0.5}
        if picked != forced:
            prob, v = build(sorted(picked), picked)
            if solve(prob) == "Optimal":
                best = (prob, v)
    start = {var.name: var.varValue for var in best[0].variables()}
    prob, v = build(options, forced)
    status, proven = _solve_from(prob, start)
    if status != "Optimal" or (not proven and pulp.value(prob.objective) < pulp.value(best[0].objective)):
        return *best, "Optimal", False     # stopped by the time cap without beating the start
    return prob, v, status, proven


def _solve_from(prob: pulp.LpProblem, start: dict[str, float],
                seconds: float | None = None) -> tuple[str, bool]:
    """Solve starting from a known plan (`start`: variable name -> value, from a smaller
    version of the same problem) -> (status, proven optimal?). Variables it didn't have
    start at 0, except plays (= start: the XI scores in full). CBC 2.10 misreads a starting solution in a
    maximisation, so it is solved as the minimisation of minus the objective; its own
    heuristics and cuts only slow it down once it has a good plan to start from."""
    for var in prob.variables():
        name = var.name.replace("plays_", "start_", 1) if var.name.startswith("plays_") else var.name
        value = start.get(name) or 0.0
        var.setInitialValue(round(value) if var.cat == pulp.LpInteger else value, check=False)
    prob.sense, prob.objective = pulp.LpMinimize, -prob.objective
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PULP_CBC_CMD is deprecated", category=DeprecationWarning)
            prob.solve(pulp.PULP_CBC_CMD(msg=False, warmStart=True,
                                         timeLimit=CHIP_SOLVE_SECONDS if seconds is None else seconds,
                                         options=["heuristicsOnOff off", "cutsOnOff off"]))
    finally:
        prob.sense, prob.objective = pulp.LpMaximize, -prob.objective
    return pulp.LpStatus[prob.status], prob.sol_status == pulp.LpSolutionOptimal


def _chip_ceilings(pts: pd.DataFrame, position: pd.Series, gws: list, decay: float,
                   bench_weight: float) -> dict[tuple[str, int], float]:
    """The most a bench boost or triple captain could add in each gameweek (decayed, as in
    the objective): the week's best keeper and three best outfielders on the bench in
    full, or the week's best player captained once more."""
    extra_cap = config.TRIPLE_CAPTAIN_MULTIPLIER - config.CAPTAIN_MULTIPLIER
    gk = position.reindex(pts.index) == "GK"
    out = {}
    for t, gw in enumerate(gws):
        week = pts[gw].clip(lower=0.0)
        best_bench = week[gk].max() + week[~gk].nlargest(config.SQUAD_SIZE - config.XI_SIZE - 1).sum()
        out["bench_boost", t] = decay ** t * (1 - bench_weight) * float(best_bench)
        out["triple_captain", t] = decay ** t * extra_cap * float(week.max())
    return out


def _free_hit_squad(xp_week: pd.DataFrame, c: pd.DataFrame, sell_price: dict[int, float],
                    budget: float, keep: list[int], bench_weight: float):
    """The best one-week squad for a free hit -> (Squad, what it adds to the objective, bank
    left). Your own players cost what they'd sell for; `budget` is bank + their selling
    prices. Players dominated in that week alone are left out first (pool.py)."""
    ids = sorted(pool.prune(xp_week, c.reset_index(), keep=keep) & set(c.index))
    cand = c.loc[ids, ["web_name", "team_name", "position"]].reset_index()
    cand["price"] = cand["id"].map(sell_price)
    cand["xp"] = cand["id"].map(xp_week.groupby("id")["xp"].sum()).fillna(0.0)
    s = pick_squad(cand, budget=budget + 1e-6, bench_weight=bench_weight)
    team = s.players
    value = (s.xp + bench_weight * team.loc[team["role"] == "bench", "xp"].sum()
             + _VICE_TIEBREAK * team.loc[team["vice"], "xp"].sum())
    bank_left = budget - team["price"].sum()
    team["price"] = team["id"].map(c["price"])           # show today's prices, as other weeks do
    return s, float(value), bank_left
